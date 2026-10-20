"""Coordinate one global ERA5 batch through download and archive publication."""

from collections.abc import Sequence
from datetime import date
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from era5_pull_pipeline.catalog import (
    archive_session,
    existing_month,
    prepare_revision,
    publish_revision,
)
from era5_pull_pipeline.cds import DATASET, DEFAULT_MAX_DAYS, build_batch_request, download_batch
from era5_pull_pipeline.conversion import convert_batch
from era5_pull_pipeline.credentials import load_credentials


def pull_batch(
    start: date,
    end: date,
    variables: Sequence[str],
    archive: Path,
    spatial_chunks: tuple[int, int] = (32, 32),
    max_days: int = DEFAULT_MAX_DAYS,
) -> Path:
    """Download and publish one batch, skipping an identical published request.

    Args:
        start: First UTC date, inclusive.
        end: Last UTC date, inclusive and in the same calendar month.
        variables: CDS single-level variable names.
        archive: Archive directory containing monthly stores and catalog.sqlite.
        spatial_chunks: Latitude and longitude grid cells per read chunk.
        max_days: Maximum days allowed in this one request.

    Returns:
        The published YYYY/MM/era5.zarr directory.

    Raises:
        ValueError: The selection is invalid or the month contains a different
            selection. Adding to an existing month is a separate operation.
        Exception: Credential, service, conversion, and publication errors
            propagate. Unpublished GRIB files remain available for ingestion.
    """
    request = build_batch_request(start, end, variables, max_days)
    identity = json.dumps(
        {"dataset": DATASET, "request": request}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    request_sha256 = hashlib.sha256(identity).hexdigest()
    with archive_session(archive) as db:
        existing = existing_month(db, archive, start.year, start.month, request_sha256)
        if existing is not None:
            return existing
        batch = download_batch(
            load_credentials(), start, end, variables, archive / "staging", max_days
        )
        metadata = json.loads((batch / "metadata.json").read_text(encoding="utf-8"))
        return _publish_batch(db, archive, batch, metadata, spatial_chunks)


def ingest_batch(
    batch: Path,
    archive: Path,
    spatial_chunks: tuple[int, int] = (32, 32),
) -> Path:
    """Publish a completed staged batch without contacting CDS.

    Args:
        batch: Completed download directory with metadata.json and data.grib.
        archive: Archive directory containing monthly stores and catalog.sqlite.
        spatial_chunks: Latitude and longitude grid cells per read chunk.

    Returns:
        The published YYYY/MM/era5.zarr directory. Repeating an already published
        batch also succeeds after its GRIB has been removed.

    Raises:
        ValueError: The download is incomplete, its checksum or selection is
            inconsistent, or the month already contains a different selection.
        Exception: File, conversion, and publication errors propagate.
    """
    if batch.suffix == ".partial":
        raise ValueError("The download is incomplete. Select a completed batch directory.")
    metadata = json.loads((batch / "metadata.json").read_text(encoding="utf-8"))
    with archive_session(archive) as db:
        return _publish_batch(db, archive, batch, metadata, spatial_chunks)


def _publish_batch(
    db: sqlite3.Connection,
    archive: Path,
    batch: Path,
    metadata: dict[str, Any],
    spatial_chunks: tuple[int, int],
) -> Path:
    """Verify the downloaded source, convert it, and commit its publication.

    Args:
        db: Catalog connection protected by the archive writer lock.
        archive: Archive directory.
        batch: Completed staging directory.
        metadata: Downloader provenance from metadata.json.
        spatial_chunks: Latitude and longitude grid cells per read chunk.

    Returns:
        The published monthly directory.

    Raises:
        ValueError: Source provenance, checksum, or archive selection conflicts.
        Exception: Conversion and publication failures propagate, retaining GRIB.
    """
    request = metadata["request"]
    identity = json.dumps(
        {"dataset": metadata["dataset"], "request": request},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if metadata["dataset"] != DATASET or hashlib.sha256(identity).hexdigest() != metadata["request_sha256"]:
        raise ValueError("The staged source request does not match its recorded identity.")
    year, month = int(request["year"][0]), int(request["month"][0])
    existing = existing_month(db, archive, year, month, metadata["request_sha256"])
    if existing is not None:
        return existing
    grib = batch / "data.grib"
    with grib.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if grib.stat().st_size != metadata["size_bytes"] or checksum != metadata["sha256"]:
        raise ValueError("The staged GRIB does not match its recorded size and SHA-256 checksum.")
    revision, staged = prepare_revision(db, archive, batch, metadata)
    report = convert_batch(grib, metadata, staged, spatial_chunks)
    return publish_revision(db, archive, revision, report)
