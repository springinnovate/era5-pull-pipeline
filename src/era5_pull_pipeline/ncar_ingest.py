"""Ingest downloaded NCAR NetCDF data into monthly Zarr stores."""

from calendar import monthrange
from collections import defaultdict
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
import hashlib
import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from era5_pull_pipeline.catalog import (
    archive_session, current_month, delete_ingested_netcdf_files, group_hour_intervals,
    prepare_revision, publish_revision, remove_published_source_hours,
)
from era5_pull_pipeline.conversion import write_monthly_zarr
from era5_pull_pipeline.ncar_conversion import read_ncar_netcdf_arrays


def ingest_ncar(
    archive: Path,
    year: int,
    month: int,
    spatial_chunks: tuple[int, int] = (32, 32),
    *,
    refresh: bool = False,
) -> Path:
    """Ingest downloaded NCAR NetCDF data into the monthly Zarr archive.

    Read file paths and selected hours from the catalog, convert the data, and
    publish the month. Preserve existing fields unless refresh is requested.
    Delete each NetCDF file after all its selected hours have been published.
    A failed conversion or publication retains the files for another attempt.

    Args:
        archive: Archive containing a download-ncar catalog and staged sources.
        year: Calendar year of the selected valid UTC hours.
        month: Calendar month to ingest.
        spatial_chunks: Latitude and longitude cells per Zarr read chunk.
        refresh: Replace selected stored fields using the recorded downloads.

    Returns:
        Path of the published monthly Zarr store.

    Raises:
        FileNotFoundError: The archive catalog or a required download is absent.
        ValueError: There are no selected sources or source coverage is incomplete.
        Exception: Decoder, storage, and catalog failures propagate.
    """
    if not (archive / "catalog.sqlite").is_file():
        raise FileNotFoundError(f"No catalog in {archive}. Run download-ncar first.")
    with archive_session(archive) as db:
        return ingest_ncar_month(db, archive, year, month, spatial_chunks, refresh=refresh)


def ingest_ncar_month(
    db: sqlite3.Connection,
    archive: Path,
    year: int,
    month: int,
    spatial_chunks: tuple[int, int] = (32, 32),
    *,
    refresh: bool = False,
    selected_hours: set[tuple[str, int]] | None = None,
    source_ids: set[str] | None = None,
) -> Path:
    """Ingest selected NCAR hours while the caller holds the archive writer lock.

    Use the caller's catalog connection so bulk pulls can download and publish
    several months under one writer lock. Standalone ingestion includes every
    pending selection in the month; a bulk pull supplies its exact selection.

    Args:
        db: Catalog connection protected by archive_session.
        archive: Archive containing the staged NetCDF files.
        year: Year of the valid UTC hours.
        month: Month to publish.
        spatial_chunks: Latitude and longitude cells per read chunk.
        refresh: Replace selected stored fields.
        selected_hours: Optional source variable and Unix-hour pairs to ingest.
        source_ids: Optional source IDs whose files supply the values. Publication
            also completes pending selections for the same hours from older files.

    Returns:
        Path of the published monthly Zarr store.

    Raises:
        FileNotFoundError: A required download is incomplete.
        ValueError: Required source hours are absent or publication is invalid.
        Exception: NetCDF, Zarr, and catalog errors propagate.
    """
    start = datetime(year, month, 1, tzinfo=UTC)
    end = start + timedelta(days=monthrange(year, month)[1], hours=-1)
    previous = current_month(db, archive, year, month)
    selections = []
    sources = {}
    owners: dict[tuple[str, int], str] = {}
    # A newer provider object wins if a correction was downloaded before
    # its older pending selection was ingested.
    records = db.execute(
        """SELECT f.*, s.start_time, s.end_time FROM source_files f
           JOIN source_selections s ON s.source_id = f.id
           WHERE s.start_time <= ? AND s.end_time >= ?
           ORDER BY f.last_modified DESC, f.created_at DESC""",
        (end.isoformat(), start.isoformat()),
    ).fetchall()
    for record in records:
        first = max(start, datetime.fromisoformat(record["start_time"]))
        last = min(end, datetime.fromisoformat(record["end_time"]))
        hours = [
            hour for hour in range(int(first.timestamp()), int(last.timestamp()) + 1, 3600)
            if selected_hours is None or (record["variable"], hour) in selected_hours
        ]
        for first_hour, last_hour in group_hour_intervals(hours):
            selections.append({
                "source_id": record["id"],
                "start_time": (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=first_hour)).isoformat(),
                "end_time": (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=last_hour)).isoformat(),
            })
        # Publishing these variable-hours also completes pending selections
        # from superseded object versions. Only the chosen versions supply data.
        if source_ids is not None and record["id"] not in source_ids:
            continue
        sources[record["id"]] = record
        for hour in hours:
            owners.setdefault((record["variable"], hour), record["id"])
    covered = {
        (field["source_variable"], field["valid_time"])
        for field in previous[1]["fields"] if "source_variable" in field
    } if previous else set()
    selected: dict[str, list[int]] = defaultdict(list)
    for pair, source_id in owners.items():
        if refresh or pair not in covered:
            selected[source_id].append(pair[1])
    if not selected:
        if previous is None:
            raise ValueError(f"No downloaded selections for {year:04d}-{month:02d}. Run download-ncar first.")
        with db:
            remove_published_source_hours(db, selections)
        delete_ingested_netcdf_files(db)
        print(f"{year:04d}-{month:02d}: selected hours are already published.", flush=True)
        return previous[0]

    provenance: list[dict[str, Any]] = []
    for source_id, hours in selected.items():
        record = sources[source_id]
        if record["downloaded_at"] is None:
            raise FileNotFoundError(f"Incomplete source {record['object_key']}. Run download-ncar again.")
        provenance.append({
            key: record[key] for key in (
                "id", "object_key", "variable", "size", "etag", "last_modified",
                "valid_start", "valid_end", "downloaded_at",
            )
        } | {"selected_hours": sorted(hours)})
    metadata = {
        "dataset": "nsf-ncar-era5",
        "sources": provenance,
        "source_selections": selections,
    }
    identity = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    metadata["request_sha256"] = hashlib.sha256(identity).hexdigest()
    revision, staged = prepare_revision(db, archive, metadata, year=year, month=month)
    with TemporaryDirectory(prefix=".decoded-", dir=staged.parent) as scratch:
        with ExitStack() as resources:
            incoming = []
            for source in provenance:
                record = sources[source["id"]]
                print(f"Decoding {record['variable']}: {len(source['selected_hours'])} selected UTC hours.", flush=True)
                source_identity = hashlib.sha256(
                    json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                arrays, report = read_ncar_netcdf_arrays(
                    Path(record["path"]), record["id"], record["variable"],
                    source["selected_hours"], Path(scratch), resources,
                    source_identity=source_identity,
                )
                source["attributes"] = report["source_attributes"]
                incoming.append((arrays, report))
            with db:
                db.execute(
                    "UPDATE revisions SET provenance_json = ? WHERE id = ?",
                    (json.dumps(metadata), revision),
                )
            report = write_monthly_zarr(
                incoming, metadata, staged, spatial_chunks,
                previous=previous[0] if previous else None,
                previous_report=previous[1] if previous else None,
                refresh=refresh,
            )
    return publish_revision(db, archive, revision, report)
