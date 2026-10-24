"""Download and publish requested global ERA5 hours from NCAR."""

from calendar import monthrange
from collections import Counter, deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from datetime import UTC, date, datetime, time, timedelta
import json
from multiprocessing import get_context
from pathlib import Path
import signal
import sqlite3
from typing import Any

import requests

from era5_pull_pipeline.catalog import (
    archive_session, delete_ingested_netcdf_files, get_published_hours,
    prepare_observation, publish_observation, register_sources, remove_published_source_hours,
)
from era5_pull_pipeline.conversion import write_decoded_observation_cog
from era5_pull_pipeline.ncar import SourceFile, discover_files
from era5_pull_pipeline.ncar_conversion import decode_ncar_source
from era5_pull_pipeline.ncar_download import NcarDownloadPool
from era5_pull_pipeline.ncar_ingest import ingest_ncar_month, select_ncar_source_hours


def _ignore_worker_interrupt() -> None:
    """Let the coordinator stop scheduling and wait for active worker writes."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def download_and_ingest_month(
    db: sqlite3.Connection, archive: Path, year: int, month: int,
    sources: Sequence[SourceFile], rows: Sequence[sqlite3.Row],
    executor: ProcessPoolExecutor, *, download_workers: int, processing_workers: int,
    selected_hours: set[tuple[str, int]], refresh: bool,
    on_published: Callable[[], None],
) -> Path:
    """Download monthly sources while workers decode and encode COG observations.

    The coordinator alone writes SQLite and publishes files. Decode one source
    at a time and submit at most processing_workers hourly COGs from its scratch
    array. Keep that array until every reader exits, including after a failure.
    Downloads refill independently whenever the coordinator handles completions.

    Args:
        db: Catalog connection protected by the archive writer lock.
        archive: Root of source, scratch, staging, and published paths.
        year: Selected UTC year.
        month: Selected UTC month.
        sources: Budgeted provider objects to download or reuse.
        rows: Registered source rows in the same order as sources.
        executor: Reusable spawned process pool owned by the pull invocation.
        download_workers: Maximum simultaneous transfers.
        processing_workers: Maximum in-flight observation jobs.
        selected_hours: Exact source variable and Unix-hour selection.
        refresh: Replace selected published observations.
        on_published: Count each successfully cataloged observation.

    Returns:
        Directory containing the published month.

    Raises:
        BaseException: Failures propagate after transfers and processing stop;
            completed observations and retained sources remain resumable.
    """
    records, selected, selections = select_ncar_source_hours(
        db, year, month, refresh=refresh, selected_hours=selected_hours,
        source_ids={row["id"] for row in rows},
    )
    scratch = archive / "staging" / "decoded"
    scratch.mkdir(parents=True, exist_ok=True)
    decoding: Future[dict[str, Any]] | None = None
    active: dict[Future[dict[str, Any]], str] = {}
    current: str | None = None
    report: dict[str, Any] | None = None
    next_index = 0
    published = 0
    try:
        with NcarDownloadPool(db, sources, rows, download_workers, archive) as downloads:
            ready = deque(source_id for source_id in downloads.ready if source_id in selected)
            while downloads.active or ready or current is not None:
                ready.extend(source_id for source_id in downloads.record_completed() if source_id in selected)
                if current is None and ready:
                    current = ready.popleft()
                    record = records[current]
                    print(f"Decoding {record['variable']}: {len(selected[current])} selected UTC hours.", flush=True)
                    decoding = executor.submit(
                        decode_ncar_source, archive / record["path"], current,
                        record["variable"], selected[current], scratch,
                    )
                if decoding is not None and decoding.done():
                    report = decoding.result()
                    decoding = None
                    with db:
                        db.execute(
                            "UPDATE source_files SET attributes_json = ? WHERE id = ?",
                            (json.dumps(report["source_attributes"]), current),
                        )
                    next_index = published = 0
                for future in list(active):
                    if future.done():
                        raster = future.result()
                        publish_observation(db, archive, active[future], raster)
                        del active[future]
                        on_published()
                        published += 1
                        if published % 24 == 0 or published == len(report["fields"]):
                            print(f"Published {records[current]['variable']}: {published}/{len(report['fields'])} observations.", flush=True)
                if report is not None:
                    while next_index < len(report["fields"]) and len(active) < processing_workers:
                        field = report["fields"][next_index]
                        attributes = report["variables"][field["variable"]]
                        publication, staged = prepare_observation(db, archive, field, attributes)
                        future = executor.submit(
                            write_decoded_observation_cog, scratch / "values.bin",
                            next_index, field, attributes, report["grid"], staged,
                        )
                        active[future] = publication
                        next_index += 1
                    if not active:
                        (scratch / "values.bin").unlink()
                        current = report = None
                        continue
                pending = [*downloads.active, *active]
                if decoding is not None:
                    pending.append(decoding)
                if pending:
                    wait(pending, return_when=FIRST_COMPLETED)
    except BaseException:
        # Cancel queued work and wait for running jobs before releasing scratch.
        # The outer invocation also owns this executor and exits on this error.
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        (scratch / "values.bin").unlink(missing_ok=True)
    with db:
        remove_published_source_hours(db, selections)
    delete_ingested_netcdf_files(db, archive)
    return archive / f"{year:04d}" / f"{month:02d}"


def pull_range(
    start: date,
    end: date,
    variables: Sequence[str],
    archive: Path,
    *,
    download_workers: int = 2,
    processing_workers: int = 4,
    max_source_gb: float = 16,
    refresh: bool = False,
) -> dict[str, int]:
    """Download selected NCAR hours and publish hourly COG observations.

    Schedule one month at a time. Reuse cataloged downloads, skip published
    hours, and keep forecast files needed by the next month's selection.
    The source-file budget includes pending downloads and retained inputs;
    decoded arrays and staged COG files require additional space.

    Args:
        start: First requested UTC date, inclusive.
        end: Last requested UTC date, inclusive.
        variables: NCAR short variable names from the source inventory.
        archive: Directory containing the catalog and observation files.
        download_workers: Maximum simultaneous file transfers.
        processing_workers: Maximum worker processes and in-flight COG observations.
        max_source_gb: Maximum reserved NetCDF source bytes in decimal GB.
        refresh: Retrieve current sources and replace selected stored fields.

    Returns:
        Counts of completed, skipped, unavailable, failed, and pending
        variable-hours for this invocation.

    Raises:
        ValueError: Dates or limits are invalid, the source budget is too small,
            or required refresh hours are unavailable.
        Exception: Transfer, conversion, and publication failures propagate.
            Previously published observations and staged downloads remain recoverable.
    """
    if end < start:
        raise ValueError("The end date must be on or after the start date.")
    if download_workers < 1 or processing_workers < 1 or max_source_gb <= 0:
        raise ValueError("Download workers, processing workers, and the source-file budget must be positive.")
    variables = sorted(set(variables))
    summary = dict(completed=0, skipped=0, unavailable=0, failed=0,
                   pending=((end - start).days + 1) * 24 * len(variables))
    active_hours = 0
    activity = "opening the archive"

    def record_completed_observation() -> None:
        """Count each committed observation even if later work is interrupted."""
        nonlocal active_hours
        summary["completed"] += 1
        summary["pending"] -= 1
        active_hours -= 1

    try:
        with archive_session(archive) as db, requests.Session() as session, ProcessPoolExecutor(
            max_workers=processing_workers, mp_context=get_context("spawn"),
            initializer=_ignore_worker_interrupt,
        ) as executor:
            cursor = start
            while cursor <= end:
                last = min(end, date(cursor.year, cursor.month, monthrange(cursor.year, cursor.month)[1]))
                first_hour = int(datetime.combine(cursor, time.min, UTC).timestamp())
                last_hour = int(datetime.combine(last, time(23), UTC).timestamp())
                requested = {
                    (variable, hour) for variable in variables
                    for hour in range(first_hour, last_hour + 1, 3600)
                }
                published = get_published_hours(db, cursor.year, cursor.month)
                needed = requested if refresh else requested - published
                skipped = len(requested) - len(needed)
                summary["skipped"] += skipped
                summary["pending"] -= skipped
                label = f"{cursor.year:04d}-{cursor.month:02d}"
                if skipped:
                    # Standalone downloads may also select hours already stored.
                    # Retire those selections before reserving more source bytes.
                    pending = db.execute(
                        """SELECT f.variable FROM source_selections s JOIN source_files f
                           ON f.id = s.source_id WHERE s.start_time <= ? AND s.end_time >= ?""",
                        (datetime.combine(last, time(23), UTC).isoformat(),
                         datetime.combine(cursor, time.min, UTC).isoformat()),
                    )
                    if any(row["variable"] in variables for row in pending):
                        ingest_ncar_month(
                            db, archive, cursor.year, cursor.month,
                            selected_hours=requested & published, source_ids=set(),
                        )
                if not needed:
                    print(f"{label}: skipped {skipped} published variable-hours.", flush=True)
                    cursor = last + timedelta(days=1)
                    continue
                activity = f"planning {label}"
                active_hours = len(needed)
                sources: dict[tuple[str, str], SourceFile] = {}
                owners: dict[tuple[str, int], tuple[str, str]] = {}
                if not refresh:
                    # Recorded pending files can finish a stopped run offline.
                    rows = db.execute(
                        """SELECT f.* FROM source_files f
                           WHERE valid_start <= ? AND valid_end >= ? AND
                           (downloaded_at IS NOT NULL OR EXISTS (
                               SELECT 1 FROM source_selections s WHERE s.source_id=f.id
                           )) ORDER BY last_modified DESC, created_at DESC""",
                        (datetime.combine(last, time(23), UTC).isoformat(),
                         datetime.combine(cursor, time.min, UTC).isoformat()),
                    ).fetchall()
                    for row in rows:
                        source = SourceFile(
                            key=row["object_key"], variable=row["variable"], size=row["size"],
                            etag=row["etag"], last_modified=row["last_modified"],
                            valid_start=datetime.fromisoformat(row["valid_start"]),
                            valid_end=datetime.fromisoformat(row["valid_end"]),
                        )
                        key = source.key, source.etag
                        sources[key] = source
                        for hour in range(max(first_hour, int(source.valid_start.timestamp())),
                                          min(last_hour, int(source.valid_end.timestamp())) + 1, 3600):
                            pair = source.variable, hour
                            if pair in needed:
                                owners.setdefault(pair, key)
                undiscovered = needed - owners.keys()
                if undiscovered:
                    names = sorted({variable for variable, _ in undiscovered})
                    print(f"{label}: reading NCAR availability for {', '.join(names)}.", flush=True)
                    discovered = discover_files(cursor, last, names, session)
                    for source in sorted(discovered, key=lambda item: item.last_modified, reverse=True):
                        key = source.key, source.etag
                        sources[key] = source
                        for hour in range(max(first_hour, int(source.valid_start.timestamp())),
                                          min(last_hour, int(source.valid_end.timestamp())) + 1, 3600):
                            pair = source.variable, hour
                            if pair in needed:
                                owners.setdefault(pair, key)
                unavailable = needed - owners.keys()
                summary["unavailable"] += len(unavailable)
                summary["pending"] -= len(unavailable)
                active_hours -= len(unavailable)
                if unavailable:
                    counts = Counter(variable for variable, _ in unavailable)
                    print(f"{label}: unavailable variable-hours: " + ", ".join(
                        f"{variable}={counts[variable]}" for variable in sorted(counts)
                    ), flush=True)
                    if refresh:
                        raise ValueError(f"{label}: refresh requires every selected hour; NCAR is missing {len(unavailable)}.")
                if not owners:
                    active_hours = 0
                    cursor = last + timedelta(days=1)
                    continue
                required_sources = [sources[key] for key in dict.fromkeys(owners.values())]
                reserved = {
                    (row["object_key"], row["etag"]): row["size"] for row in db.execute(
                        """SELECT f.object_key, f.etag, f.size FROM source_files f
                           WHERE removed_at IS NULL"""
                    )
                }
                reserved.update({(source.key, source.etag): source.size for source in required_sources})
                required_bytes = sum(reserved.values())
                if required_bytes > max_source_gb * 1_000_000_000:
                    raise ValueError(
                        f"{label}: staged and required NetCDF files need {required_bytes / 1_000_000_000:.3f} GB; "
                        f"--max-source-gb is {max_source_gb:g}. Increase it or select fewer variables."
                    )
                # Reserve only requested missing future hours in a shared file,
                # so publication of this month cannot delete next month's input.
                registered_hours = set(owners)
                future_coverage: dict[tuple[int, int], set[tuple[str, int]]] = {}
                for source in required_sources:
                    hour_time = max(source.valid_start, datetime.combine(last + timedelta(days=1), time.min, UTC))
                    stop_time = min(source.valid_end, datetime.combine(end, time(23), UTC))
                    while hour_time <= stop_time:
                        month = hour_time.year, hour_time.month
                        if month not in future_coverage:
                            future_coverage[month] = get_published_hours(db, *month)
                        pair = source.variable, int(hour_time.timestamp())
                        if refresh or pair not in future_coverage[month]:
                            registered_hours.add(pair)
                        hour_time += timedelta(hours=1)
                rows = register_sources(
                    db, archive, required_sources, cursor, end, selected_hours=registered_hours,
                )
                print(f"{label}: {len(owners)} variable-hours to publish; {skipped} skipped; "
                      f"{len(required_sources)} source files; {required_bytes / 1_000_000_000:.3f} GB reserved.", flush=True)
                activity = f"downloading, converting and publishing {label}"
                print(f"{label}: overlapping downloads and processing ({processing_workers} processing workers).", flush=True)
                result = download_and_ingest_month(
                    db, archive, cursor.year, cursor.month, required_sources, rows, executor,
                    download_workers=download_workers, processing_workers=processing_workers,
                    selected_hours=set(owners), refresh=refresh,
                    on_published=record_completed_observation,
                )
                active_hours = 0
                print(f"Published {result}.", flush=True)
                cursor = last + timedelta(days=1)
    except BaseException:
        summary["failed"] += active_hours
        summary["pending"] -= active_hours
        print(f"Stopped while {activity}. Run the same command to resume; "
              "published observations and unfinished downloads remain in the archive.", flush=True)
        raise
    finally:
        print("Summary: " + ", ".join(f"{key}={value}" for key, value in summary.items()), flush=True)
    return summary
