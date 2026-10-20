"""Pull global ERA5 coverage through bounded batches and monthly publication."""

from calendar import monthrange
from collections.abc import Sequence
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from era5_pull_pipeline.catalog import (
    archive_session,
    current_month,
    prepare_revision,
    publish_revision,
)
from era5_pull_pipeline.cds import DATASET, DEFAULT_MAX_DAYS, build_batch_request, download_batch, load_availability
from era5_pull_pipeline.conversion import assemble_month, convert_batch
from era5_pull_pipeline.credentials import load_credentials
from era5_pull_pipeline.planning import missing_requests, month_requests
from era5_pull_pipeline.staging import conversion_checkpoint, prepare_batches


def pull_range(
    start: date,
    end: date,
    variables: Sequence[str],
    archive: Path,
    spatial_chunks: tuple[int, int] = (32, 32),
    batch_days: int = DEFAULT_MAX_DAYS,
    *,
    refresh: bool = False,
) -> dict[str, int]:
    """Fill or refresh a date range, resuming batches and publishing each month once.

    Counts describe variable/hour pairs: completed fields were published by this
    run, skipped fields were already stored, unavailable fields were absent from
    the source's availability snapshot, and pending fields remain unattempted or
    staged. A failed batch or monthly assembly is reported before re-raising its
    original error. Availability and credentials are loaded only when needed.

    Args:
        start: First UTC date, inclusive.
        end: Last UTC date, inclusive; may span months or years.
        variables: CDS single-level variable names.
        archive: Archive directory containing monthly stores and catalog.sqlite.
        spatial_chunks: Latitude and longitude cells per read chunk.
        batch_days: Maximum calendar days per batch, also subject to CDS limits.
        refresh: Retrieve all selected fields and replace stored overlaps. Each
            month's full selection must be available before its refresh starts.

    Returns:
        Counts for completed, skipped, unavailable, failed, and pending fields.

    Raises:
        ValueError: The selection or staged/archive integrity is inconsistent,
            or selected refresh fields are unavailable from CDS.
        Exception: Source, conversion, and publication errors propagate after
            reporting progress. Unpublished downloads and checkpoints remain.
    """
    if end < start:
        raise ValueError("The end date must be on or after the start date.")
    if batch_days < 1:
        raise ValueError("batch_days must be at least 1.")
    if not variables or any(not variable for variable in variables):
        raise ValueError("Select at least one nonempty CDS variable name.")
    variables = sorted(set(variables))
    summary = {
        "completed": 0, "skipped": 0, "unavailable": 0, "failed": 0,
        "pending": ((end - start).days + 1) * 24 * len(variables),
    }
    availability = None
    credentials = None
    active_fields = 0
    activity = "opening the archive"
    try:
        with archive_session(archive) as db:
            cursor = start
            while cursor <= end:
                last = min(end, date(cursor.year, cursor.month, monthrange(cursor.year, cursor.month)[1]))
                label = f"{cursor.year:04d}-{cursor.month:02d}"
                activity = f"planning {label}"
                requests, skipped, _ = month_requests(
                    db, cursor, last, variables, batch_days, refresh=refresh,
                )
                summary["skipped"] += skipped
                summary["pending"] -= skipped
                if requests:
                    if availability is None:
                        print("Reading CDS field availability.", flush=True)
                        availability = load_availability()
                    requests, _, unavailable = month_requests(
                        db, cursor, last, variables, batch_days, availability, refresh=refresh,
                    )
                    summary["unavailable"] += unavailable
                    summary["pending"] -= unavailable
                    if refresh and unavailable:
                        raise ValueError(
                            f"Cannot refresh {label}: {unavailable} selected variable-hours "
                            "are unavailable from CDS. This month is unchanged; "
                            "select an available interval or retry later."
                        )
                else:
                    unavailable = 0
                work = prepare_batches(db, archive, requests, refresh=refresh)
                print(
                    f"{label}: {len(work)} batches; {skipped} stored and "
                    f"{unavailable} unavailable variable-hours.", flush=True,
                )
                batches = []
                sources = []
                checkpoints = []
                month_fields = 0
                for index, record in enumerate(work, 1):
                    request = json.loads(record["request_json"])
                    batch = Path(record["batch_path"])
                    active_fields = len(request["variable"]) * len(request["day"]) * len(request["time"])
                    activity = f"downloading {label} batch {index}/{len(work)}"
                    if record["metadata_json"] is None:
                        # The final rename can precede an interrupted catalog commit.
                        if not batch.exists():
                            if credentials is None:
                                credentials = load_credentials()
                            print(f"{activity.capitalize()}; CDS may queue the request.", flush=True)
                            download_batch(credentials, request, batch)
                        metadata = json.loads((batch / "metadata.json").read_text(encoding="utf-8"))
                        with db:
                            db.execute(
                                "UPDATE staged_batches SET metadata_json = ? WHERE id = ?",
                                (json.dumps(metadata), record["id"]),
                            )
                    else:
                        metadata = json.loads(record["metadata_json"])
                        print(f"{label} batch {index}/{len(work)}: reusing completed GRIB.", flush=True)
                    activity = f"converting {label} batch {index}/{len(work)}"
                    checkpoint, report, reused = conversion_checkpoint(db, record, metadata, spatial_chunks)
                    print(
                        f"{label} batch {index}/{len(work)}: "
                        f"{'reused' if reused else 'saved'} validated conversion.", flush=True,
                    )
                    batches.append(batch)
                    sources.append(metadata)
                    checkpoints.append((checkpoint, report))
                    month_fields += active_fields
                    active_fields = 0
                if batches:
                    active_fields = month_fields
                    activity = f"assembling and publishing {label}"
                    print(f"{activity.capitalize()}.", flush=True)
                    identity = json.dumps(
                        {"dataset": DATASET, "sources": sources}, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                    metadata = {
                        "dataset": DATASET,
                        "request_sha256": hashlib.sha256(identity).hexdigest(),
                        "sources": sources,
                    }
                    previous = current_month(db, archive, cursor.year, cursor.month)
                    revision, staged = prepare_revision(db, archive, batches, metadata)
                    report = assemble_month(
                        checkpoints, metadata, staged, spatial_chunks,
                        previous=previous[0] if previous else None,
                        previous_report=previous[1] if previous else None,
                        refresh=refresh,
                    )
                    published = publish_revision(db, archive, revision, report)
                    summary["completed"] += month_fields
                    summary["pending"] -= month_fields
                    active_fields = 0
                    print(f"Published {published}.", flush=True)
                cursor = last + timedelta(days=1)
    except Exception:
        summary["failed"] += active_fields
        summary["pending"] -= active_fields
        print(f"Failed while {activity}; staged work is retained.", flush=True)
        raise
    finally:
        print("Summary (variable-hours): " + ", ".join(f"{key}={value}" for key, value in summary.items()) + ".", flush=True)
    return summary


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
            batch = download_batch(credentials, selection, archive / "staging" / uuid4().hex)
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
        ValueError: The download is incomplete, or its grid, layout, or variable
            metadata conflicts with the archive.
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
    """Convert a completed source and commit its publication.

    Args:
        db: Catalog connection protected by the archive writer lock.
        archive: Archive directory.
        batch: Completed staging directory.
        metadata: Downloader provenance from metadata.json.
        spatial_chunks: Latitude and longitude grid cells per read chunk.

    Returns:
        The published monthly directory.

    Raises:
        ValueError: Source coverage or archive selection conflicts.
        Exception: Conversion and publication failures propagate, retaining GRIB.
    """
    request = metadata["request"]
    year, month = int(request["year"][0]), int(request["month"][0])
    previous = current_month(db, archive, year, month)
    if not missing_requests(db, request):
        return previous[0]
    grib = batch / "data.grib"
    revision, staged = prepare_revision(db, archive, [batch], metadata)
    report = convert_batch(
        grib, metadata, staged, spatial_chunks,
        previous=previous[0] if previous else None,
        previous_report=previous[1] if previous else None,
    )
    return publish_revision(db, archive, revision, report)
