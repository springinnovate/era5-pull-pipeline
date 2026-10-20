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
    current_month,
    prepare_revision,
    publish_revision,
)
from era5_pull_pipeline.cds import DATASET, DEFAULT_MAX_DAYS, build_batch_request, download_batch
from era5_pull_pipeline.conversion import convert_batch
from era5_pull_pipeline.credentials import load_credentials
from era5_pull_pipeline.planning import missing_requests


def pull_batch(
    start: date,
    end: date,
    variables: Sequence[str],
    archive: Path,
    spatial_chunks: tuple[int, int] = (32, 32),
    max_days: int = DEFAULT_MAX_DAYS,
) -> Path:
    """Fill missing variables and hours within one bounded monthly interval.

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
        ValueError: The selection or its archive metadata is inconsistent.
        Exception: Credential, service, conversion, and publication errors
            propagate. Unpublished GRIB files remain available for ingestion.
    """
    request = build_batch_request(start, end, variables, max_days)
    with archive_session(archive) as db:
        selections = missing_requests(db, request)
        if not selections:
            return current_month(db, archive, start.year, start.month)[0]
        credentials = load_credentials()
        for selection in selections:
            batch = download_batch(credentials, selection, archive / "staging")
            metadata = json.loads((batch / "metadata.json").read_text(encoding="utf-8"))
            published = _publish_batch(db, archive, batch, metadata, spatial_chunks)
        return published


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
        ValueError: The download is incomplete, its checksum is inconsistent,
            or its grid, layout, or variable metadata conflicts with the archive.
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
    previous = current_month(db, archive, year, month)
    if not missing_requests(db, request):
        return previous[0]
    grib = batch / "data.grib"
    with grib.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if grib.stat().st_size != metadata["size_bytes"] or checksum != metadata["sha256"]:
        raise ValueError("The staged GRIB does not match its recorded size and SHA-256 checksum.")
    revision, staged = prepare_revision(db, archive, batch, metadata)
    report = convert_batch(
        grib, metadata, staged, spatial_chunks,
        previous=previous[0] if previous else None,
        previous_report=previous[1] if previous else None,
    )
    return publish_revision(db, archive, revision, report)
