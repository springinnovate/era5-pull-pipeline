"""Verify staged sources and retain reusable, validated conversion checkpoints."""

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
import shutil
from typing import Any

from era5_pull_pipeline.cds import DATASET
from era5_pull_pipeline.conversion import convert_batch


def verify_source(batch: Path, metadata: dict[str, Any]) -> None:
    """Verify a staged download's request identity and complete GRIB bytes.

    Args:
        batch: Completed download directory.
        metadata: Recorded download provenance.

    Raises:
        ValueError: The dataset, request identity, size, or checksum differs.
        OSError: The source file cannot be read.
    """
    identity = json.dumps(
        {"dataset": metadata["dataset"], "request": metadata["request"]},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    if metadata["dataset"] != DATASET or hashlib.sha256(identity).hexdigest() != metadata["request_sha256"]:
        raise ValueError("The staged source request does not match its recorded identity.")
    grib = batch / "data.grib"
    with grib.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if grib.stat().st_size != metadata["size_bytes"] or checksum != metadata["sha256"]:
        raise ValueError("The staged GRIB does not match its recorded size and SHA-256 checksum.")


def find_downloads(
    month_staging: Path, requests: Sequence[dict[str, Any]]
) -> list[tuple[Path, dict[str, Any]]]:
    """Find verified, disjoint downloads contained in the missing monthly selection.

    Args:
        month_staging: Month directory containing request directories and attempts.
        requests: Available missing selections for this calendar month.

    Returns:
        Completed batch paths and provenance. Their variable/hour combinations
        are disjoint subsets of the requested selections, even if availability
        or the requested batch size changed since they were downloaded.

    Raises:
        ValueError: A reusable completed source fails integrity verification.
        Exception: Metadata and file reading errors propagate.
    """
    if not requests:
        return []
    remaining = {
        (variable, day, hour)
        for request in requests
        for variable in request["variable"]
        for day in request["day"]
        for hour in request["time"]
    }
    selection = {
        key: value for key, value in requests[0].items()
        if key not in ("variable", "day", "time")
    }
    found = []
    for path in sorted(month_staging.glob("*/*/metadata.json")):
        if path.parent.suffix == ".partial" or not (path.parent / "data.grib").is_file():
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        request = metadata["request"]
        if {
            key: value for key, value in request.items()
            if key not in ("variable", "day", "time")
        } != selection:
            continue
        fields = {
            (variable, day, hour)
            for variable in request["variable"]
            for day in request["day"]
            for hour in request["time"]
        }
        if fields <= remaining:
            verify_source(path.parent, metadata)
            found.append((path.parent, metadata))
            remaining.difference_update(fields)
    return found


def conversion_checkpoint(
    batch: Path, metadata: dict[str, Any], spatial_chunks: tuple[int, int]
) -> tuple[Path, dict[str, Any], bool]:
    """Reuse a verified conversion or rebuild it from an already verified GRIB.

    A manifest is written only after conversion's round-trip validation and
    flushing the store. Incomplete or changed checkpoints are rebuilt from the
    retained source; the caller verifies that source before calling.

    Args:
        batch: Completed, verified download directory owned by range staging.
        metadata: Verified source provenance.
        spatial_chunks: Latitude and longitude cells per read chunk.

    Returns:
        Checkpoint path, validation report, and whether conversion was reused.

    Raises:
        ValueError: A checkpoint resolves outside its owning batch directory.
        Exception: Conversion, metadata, and filesystem errors propagate.
    """
    output = batch / "converted.zarr"
    manifest = batch / "checkpoint.json"
    identity = {
        "format_version": 1,
        "request_sha256": metadata["request_sha256"],
        "source_sha256": metadata["sha256"],
        "spatial_chunks": list(spatial_chunks),
    }
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved["identity"] == identity:
            report_digest = hashlib.sha256(json.dumps(
                saved["report"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest()
            if saved.get("report_sha256") == report_digest:
                checksums = {}
                for path in sorted(output.rglob("*")):
                    if path.is_file():
                        with path.open("rb") as stream:
                            checksums[path.relative_to(output).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
                if checksums == saved["files"]:
                    return output, saved["report"], True

    # A failed decode/write has no valid manifest; rebuild only this owned store.
    manifest.unlink(missing_ok=True)
    owner = batch.resolve()
    for directory in [output, *batch.glob(".decoded-*")]:
        target = directory.resolve()
        if target == owner or not target.is_relative_to(owner):
            raise ValueError(f"Conversion scratch is outside its owning batch: {directory}.")
        if directory.exists():
            shutil.rmtree(directory)
    report = convert_batch(batch / "data.grib", metadata, output, spatial_chunks)
    checksums = {}
    for directory, _, files in os.walk(output, topdown=False):
        for filename in files:
            path = Path(directory) / filename
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
                checksums[path.relative_to(output).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
        if os.name != "nt":
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    temporary = manifest.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        report_digest = hashlib.sha256(json.dumps(
            report, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        json.dump({
            "identity": identity, "files": checksums,
            "report": report, "report_sha256": report_digest,
        }, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(manifest)
    if os.name != "nt":
        descriptor = os.open(batch, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return output, report, False
