"""Ingest downloaded NCAR NetCDF data into hourly COG observations."""

from calendar import monthrange
from collections import defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
import json
import sqlite3
from pathlib import Path

from era5_pull_pipeline.catalog import (
    archive_session, catalog_transaction, delete_ingested_netcdf_files, get_published_hours, get_staging_dir, group_hour_intervals,
    prepare_observation, publish_observation, remove_published_source_hours,
)
from era5_pull_pipeline.conversion import write_observation_cog
from era5_pull_pipeline.ncar_conversion import read_ncar_netcdf_arrays
from era5_pull_pipeline.staging import copy_local_cogs, get_source_path


def ingest_ncar(
    archive: Path,
    year: int,
    month: int,
    *,
    refresh: bool = False,
) -> Path:
    """Publish downloaded NCAR NetCDF hours as COG observations.

    Read file paths and selected hours from the catalog, convert the data, and
    publish individual observations. Preserve existing fields unless refresh is requested.
    Delete each NetCDF file after all its selected hours have been published.
    A failed conversion or publication retains the files for another attempt.

    Args:
        archive: Archive containing a download-ncar catalog and staged sources.
        year: Calendar year of the selected valid UTC hours.
        month: Calendar month to ingest.
        refresh: Replace selected stored fields using the recorded downloads.

    Returns:
        Directory containing the month's published observation files.

    Raises:
        FileNotFoundError: The archive catalog or a required download is absent.
        ValueError: There are no selected sources or source coverage is incomplete.
        Exception: Decoder, storage, and catalog failures propagate.
    """
    if not (archive / "catalog.sqlite").is_file():
        raise FileNotFoundError(f"No catalog in {archive}. Run download-ncar first.")
    with archive_session(archive) as db:
        return ingest_ncar_month(db, archive, year, month, refresh=refresh)


def ingest_ncar_month(
    db: sqlite3.Connection,
    archive: Path,
    year: int,
    month: int,
    *,
    refresh: bool = False,
    selected_hours: set[tuple[str, int]] | None = None,
    source_ids: set[str] | None = None,
    on_published: Callable[[], None] | None = None,
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
        refresh: Replace selected stored fields.
        selected_hours: Optional source variable and Unix-hour pairs to ingest.
        source_ids: Optional source IDs whose files supply the values. Publication
            also completes pending selections for the same hours from older files.
        on_published: Optional progress callback after each cataloged observation.

    Returns:
        Directory containing the month's published observation files.

    Raises:
        FileNotFoundError: A required download is incomplete.
        ValueError: Required source hours are absent or publication is invalid.
        Exception: NetCDF, COG, and catalog errors propagate.
    """
    sources, selected, selections = select_ncar_source_hours(
        db, year, month, refresh=refresh, selected_hours=selected_hours, source_ids=source_ids,
    )
    destination = archive / f"{year:04d}" / f"{month:02d}"
    if not selected:
        if not get_published_hours(db, year, month):
            raise ValueError(f"No downloaded selections for {year:04d}-{month:02d}. Run download-ncar first.")
        with catalog_transaction(db):
            remove_published_source_hours(db, selections)
        delete_ingested_netcdf_files(db, archive)
        print(f"{year:04d}-{month:02d}: selected hours are already published.", flush=True)
        return destination

    # A fixed scratch file bounds abandoned decoded storage after a hard stop.
    # The next source truncates it; the ExitStack closes mappings before unlink.
    local_root = get_staging_dir(db)
    scratch = (local_root or archive) / "staging" / "decoded"
    scratch.mkdir(parents=True, exist_ok=True)
    for source_id, hours in selected.items():
        record = sources[source_id]
        if record["downloaded_at"] is None:
            raise FileNotFoundError(f"Incomplete source {record['object_key']}. Run download-ncar again.")
        print(f"Decoding {record['variable']}: {len(hours)} selected UTC hours.", flush=True)
        try:
            with ExitStack() as resources:
                arrays, report = read_ncar_netcdf_arrays(
                    get_source_path(archive, local_root, record), source_id, record["variable"],
                    hours, scratch, resources,
                )
                with catalog_transaction(db):
                    db.execute(
                        "UPDATE source_files SET attributes_json = ? WHERE id = ?",
                        (json.dumps(report["source_attributes"]), source_id),
                    )
                for index, field in enumerate(report["fields"]):
                    attributes = report["variables"][field["variable"]]
                    publication, staged = prepare_observation(db, archive, field, attributes)
                    raster = write_observation_cog(
                        arrays[field["variable"]][index], field, attributes, report["grid"], staged,
                    )
                    if local_root is not None:
                        copy_local_cogs(local_root, archive, [staged])
                    publish_observation(db, archive, publication, raster)
                    if on_published is not None:
                        on_published()
                    if (index + 1) % 24 == 0 or index + 1 == len(hours):
                        print(f"Published {record['variable']}: {index + 1}/{len(hours)} observations.", flush=True)
        finally:
            (scratch / "values.bin").unlink(missing_ok=True)

    # Source selections survive interruptions. A repeat skips published COGs,
    # finishes remaining hours, and then releases the matching source inputs.
    with catalog_transaction(db):
        remove_published_source_hours(db, selections)
    delete_ingested_netcdf_files(db, archive)
    return destination


def select_ncar_source_hours(
    db: sqlite3.Connection, year: int, month: int, *, refresh: bool = False,
    selected_hours: set[tuple[str, int]] | None = None,
    source_ids: set[str] | None = None,
) -> tuple[dict[str, sqlite3.Row], dict[str, list[int]], list[dict[str, str]]]:
    """Select the source and pending hours for each monthly observation.

    Both offline ingestion and overlapping pulls use this ownership decision.
    Include superseded source selections for retirement after publication, while
    only the chosen sources supply decoded data.

    Args:
        db: Catalog connection owned by the archive coordinator.
        year: Selected UTC year.
        month: Selected UTC month.
        refresh: Include observations that are already published.
        selected_hours: Restrict work to these source variable and Unix-hour pairs.
        source_ids: Restrict data providers to these registered source IDs.

    Returns:
        Source rows, selected hours grouped by source ID, and source selections
        to retire after the complete monthly selection is published.
    """
    start = datetime(year, month, 1, tzinfo=UTC)
    end = start + timedelta(days=monthrange(year, month)[1], hours=-1)
    covered = get_published_hours(db, year, month)
    selections = []
    sources = {}
    owners: dict[tuple[str, int], str] = {}
    # A newer provider object wins if a correction was downloaded before
    # its older pending selection was ingested.
    records = db.execute(
        """SELECT f.*, s.start_time, s.end_time, l.path AS local_path FROM source_files f
           JOIN source_selections s ON s.source_id = f.id
           LEFT JOIN local_files l ON l.kind='source' AND l.id=f.id
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
    selected: dict[str, list[int]] = defaultdict(list)
    for pair, source_id in owners.items():
        if refresh or pair not in covered:
            selected[source_id].append(pair[1])
    return sources, selected, selections
