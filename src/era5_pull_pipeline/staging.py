"""Resume staged ERA5 requests using cataloged paths and completion state."""

from collections.abc import Sequence
import json
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any
from uuid import uuid4

from era5_pull_pipeline.conversion import convert_batch
from era5_pull_pipeline.planning import subtract_requests


def prepare_batches(
    db: sqlite3.Connection, archive: Path, requests: Sequence[dict[str, Any]]
) -> list[sqlite3.Row]:
    """Reuse cataloged selections and register new batch paths before downloading.

    Args:
        db: Catalog connection protected by the archive writer lock.
        archive: Archive owning the staged batches.
        requests: Available missing selections for one calendar month.

    Returns:
        Catalog rows covering the selection exactly once. Existing requests can
        be pending, downloaded, or converted; no filesystem discovery is needed.

    Raises:
        sqlite3.Error: Reading or recording staged requests fails.
    """
    if not requests:
        return []
    year, month = int(requests[0]["year"][0]), int(requests[0]["month"][0])
    remaining = {
        (variable, day, hour)
        for request in requests
        for variable in request["variable"]
        for day in request["day"]
        for hour in request["time"]
    }
    batches = []
    selected = []
    for batch in db.execute(
        "SELECT * FROM staged_batches WHERE year = ? AND month = ? ORDER BY rowid",
        (year, month),
    ):
        request = json.loads(batch["request_json"])
        fields = {
            (variable, day, hour)
            for variable in request["variable"]
            for day in request["day"]
            for hour in request["time"]
        }
        if fields <= remaining:
            batches.append(batch)
            selected.append(request)
            remaining.difference_update(fields)
    with db:
        for request in subtract_requests(requests, selected):
            identifier = uuid4().hex
            path = archive / "staging" / "batches" / f"{year:04d}-{month:02d}" / identifier
            db.execute(
                "INSERT INTO staged_batches (id, year, month, request_json, batch_path) VALUES (?, ?, ?, ?, ?)",
                (identifier, year, month, json.dumps(request), str(path.resolve())),
            )
            batches.append(db.execute(
                "SELECT * FROM staged_batches WHERE id = ?", (identifier,)
            ).fetchone())
    return batches


def conversion_checkpoint(
    db: sqlite3.Connection,
    batch: sqlite3.Row,
    metadata: dict[str, Any],
    spatial_chunks: tuple[int, int],
) -> tuple[Path, dict[str, Any], bool]:
    """Reuse a completed conversion or rebuild an interrupted write.

    The catalog report is committed after conversion and flushing the store.
    Completed data are immutable; a restart trusts the recorded completion.

    Args:
        db: Catalog connection protected by the archive writer lock.
        batch: Cataloged batch with its path and optional conversion report.
        metadata: Completed source download provenance.
        spatial_chunks: Latitude and longitude cells per read chunk.

    Returns:
        Checkpoint path, conversion report, and whether conversion was reused.

    Raises:
        ValueError: A scratch path resolves outside its owning batch directory.
        Exception: Conversion, filesystem, and catalog errors propagate.
    """
    path = Path(batch["batch_path"])
    output = path / "converted.zarr"
    if batch["report_json"] is not None:
        report = json.loads(batch["report_json"])
        if report["layout"]["spatial_chunks"] == list(spatial_chunks):
            return output, report, True
        with db:
            db.execute("UPDATE staged_batches SET report_json = NULL WHERE id = ?", (batch["id"],))
    # No completed report means a previous write may have stopped partway through.
    owner = path.resolve()
    for directory in [output, *path.glob(".decoded-*")]:
        target = directory.resolve()
        if target == owner or not target.is_relative_to(owner):
            raise ValueError(f"Conversion scratch is outside its owning batch: {directory}.")
        if directory.exists():
            shutil.rmtree(directory)
    report = convert_batch(path / "data.grib", metadata, output, spatial_chunks)
    for directory, _, files in os.walk(output, topdown=False):
        for filename in files:
            with (Path(directory) / filename).open("r+b") as stream:
                os.fsync(stream.fileno())
        if os.name != "nt":
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    with db:
        db.execute(
            "UPDATE staged_batches SET report_json = ? WHERE id = ?",
            (json.dumps(report), batch["id"]),
        )
    return output, report, False
