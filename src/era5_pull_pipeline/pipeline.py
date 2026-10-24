"""Download and publish requested global ERA5 hours from NCAR."""

from calendar import monthrange
from collections import Counter, deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
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
from era5_pull_pipeline.ncar import SourceFile, discover_files
from era5_pull_pipeline.ncar_conversion import NcarChunk, plan_ncar_chunk_jobs, write_ncar_chunk_cogs
from era5_pull_pipeline.ncar_download import NcarDownloadPool
from era5_pull_pipeline.ncar_ingest import ingest_ncar_month, select_ncar_source_hours


def _ignore_worker_interrupt() -> None:
    """Let the coordinator stop scheduling and wait for active worker writes."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def download_and_ingest_month(
    db: sqlite3.Connection, archive: Path, year: int, month: int,
    sources: Sequence[SourceFile], rows: Sequence[sqlite3.Row],
    executor: ProcessPoolExecutor, *, downloads: NcarDownloadPool,
    processing_workers: int, max_decoded_gb: float,
    selected_hours: set[tuple[str, int]], refresh: bool,
    on_published: Callable[[], None],
    prefetch: Future[list[SourceFile]] | None = None,
    on_inventory: Callable[[list[SourceFile]], Future[list[SourceFile]] | None] | None = None,
) -> tuple[Path, Future[list[SourceFile]] | None]:
    """Overlap source transfers with parallel native-chunk decoding and encoding.

    Workers read independent temporal chunks and encode their selected COGs in
    memory. The coordinator reserves decoded bytes before submission, keeps at
    most processing_workers batches outstanding, and publishes observations one
    at a time while continuing to service transfer and inventory completions.

    Args:
        db: Catalog connection protected by the archive writer lock.
        archive: Root of source, staging, and published paths.
        year: Selected UTC year.
        month: Selected UTC month.
        sources: Budgeted provider objects to download or reuse.
        rows: Registered source rows in the same order as sources.
        executor: Spawned process pool owned by this invocation.
        downloads: Transfer pool retained across monthly processing boundaries.
        processing_workers: Maximum outstanding metadata or chunk jobs.
        max_decoded_gb: Maximum reserved decoded chunk bytes in decimal GB.
        selected_hours: Exact source variable and Unix-hour selection.
        refresh: Replace selected published observations.
        on_published: Count each successfully cataloged observation.
        prefetch: Optional future-month inventory lookup running independently.
        on_inventory: Reserve sources and return the next budgeted lookahead lookup.

    Returns:
        Published month directory and any inventory lookup still in flight.

    Raises:
        ValueError: A native chunk cannot fit the decoded-byte budget.
        BaseException: Failures propagate after processing workers stop; source
            downloads and completed observations remain resumable.
    """
    records, selected, selections = select_ncar_source_hours(
        db, year, month, refresh=refresh, selected_hours=selected_hours,
        source_ids={row["id"] for row in rows},
    )
    downloads.add_sources(sources, rows)
    waiting = set(selected)
    ready: deque[str] = deque()
    planning: dict[Future[Any], str] = {}
    chunks: deque[tuple[str, NcarChunk]] = deque()
    reports: dict[str, dict[str, Any]] = {}
    active: dict[Future[Any], tuple[str, NcarChunk, list[str]]] = {}
    completed: deque[tuple[Future[Any], deque[tuple[str, dict[str, Any]]]]] = deque()
    finished: set[Future[Any]] = set()
    reserved = 0
    limit = int(max_decoded_gb * 1_000_000_000)
    published: Counter[str] = Counter()
    try:
        while waiting or ready or planning or chunks or active:
            available = downloads.ready + downloads.record_completed()
            downloads.ready.clear()
            for source_id in available:
                if source_id in waiting:
                    waiting.remove(source_id)
                    ready.append(source_id)
            if prefetch is not None and prefetch.done():
                prefetch = on_inventory(prefetch.result())
            for future, source_id in list(planning.items()):
                if future.done():
                    report, jobs = future.result()
                    for job in jobs:
                        if job.decoded_bytes > limit:
                            raise ValueError(
                                f"A native {records[source_id]['variable']} chunk needs "
                                f"{job.decoded_bytes / 1e9:.3f} GB of decoded arrays; "
                                "increase --max-decoded-gb."
                            )
                    reports[source_id] = report
                    with db:
                        db.execute("UPDATE source_files SET attributes_json = ? WHERE id = ?",
                                   (json.dumps(report["source_attributes"]), source_id))
                    chunks.extend((source_id, job) for job in jobs)
                    del planning[future]
                    print(f"Processing {records[source_id]['variable']}: {len(jobs)} independent native chunks.", flush=True)
            for future, (_, _, publications) in active.items():
                if future.done() and future not in finished:
                    completed.append((future, deque(zip(publications, future.result(), strict=True))))
                    finished.add(future)
            if completed:
                future, observations = completed[0]
                publication, raster = observations.popleft()
                publish_observation(db, archive, publication, raster)
                on_published()
                source_id, chunk, _ = active[future]
                published[source_id] += 1
                if published[source_id] % 24 == 0 or published[source_id] == len(selected[source_id]):
                    print(f"Published {records[source_id]['variable']}: {published[source_id]}/{len(selected[source_id])} observations.", flush=True)
                if not observations:
                    completed.popleft()
                    del active[future]
                    finished.remove(future)
                    reserved -= chunk.decoded_bytes
            # Inspect newly available sources without blocking transfer scheduling.
            while ready and len(planning) + len(active) < processing_workers:
                source_id = ready.popleft()
                record = records[source_id]
                future = executor.submit(plan_ncar_chunk_jobs, archive / record["path"], source_id,
                                         record["variable"], selected[source_id])
                planning[future] = source_id
            submitted = False
            for _ in range(len(chunks)):
                if len(planning) + len(active) >= processing_workers:
                    break
                source_id, chunk = chunks.popleft()
                if reserved + chunk.decoded_bytes > limit:
                    chunks.append((source_id, chunk))
                    continue
                report = reports[source_id]
                attributes = report["variables"][chunk.fields[0]["variable"]]
                preparations = [prepare_observation(db, archive, field, attributes) for field in chunk.fields]
                future = executor.submit(
                    write_ncar_chunk_cogs, archive / records[source_id]["path"], chunk,
                    attributes, report["grid"], [path for _, path in preparations],
                )
                active[future] = source_id, chunk, [identifier for identifier, _ in preparations]
                reserved += chunk.decoded_bytes
                submitted = True
                break  # Service transfers and inventory between batch preparations.
            if not (waiting or ready or planning or chunks or active):
                break
            if not completed and not submitted:
                pending = [*downloads.active, *planning, *(future for future in active if future not in finished)]
                if prefetch is not None:
                    pending.append(prefetch)
                if pending:
                    wait(pending, return_when=FIRST_COMPLETED)
    except BaseException:
        downloads.stop_event.set()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    with db:
        remove_published_source_hours(db, selections)
    delete_ingested_netcdf_files(db, archive)
    return archive / f"{year:04d}" / f"{month:02d}", prefetch


def pull_range(
    start: date,
    end: date,
    variables: Sequence[str],
    archive: Path,
    *,
    download_workers: int = 2,
    processing_workers: int = 4,
    max_source_gb: float = 16,
    max_decoded_gb: float = 2,
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
        processing_workers: Maximum worker processes and outstanding native-chunk jobs.
        max_source_gb: Maximum reserved NetCDF source bytes in decimal GB.
        max_decoded_gb: Maximum decoded chunk-array bytes in decimal GB.
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
    if download_workers < 1 or processing_workers < 1 or max_source_gb <= 0 or max_decoded_gb <= 0:
        raise ValueError("Download workers, processing workers, source and decoded budgets must be positive.")
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
        ) as executor, NcarDownloadPool(db, [], [], download_workers, archive) as downloads, ThreadPoolExecutor(max_workers=1) as inventory_pool:
            inventory_cache: dict[tuple[int, int], list[SourceFile]] = {}
            prefetch_cursor = start
            prefetch: Future[list[SourceFile]] | None = None
            next_first = next_last = start
            next_needed: set[tuple[str, int]] = set()

            def start_prefetch() -> Future[list[SourceFile]] | None:
                """Look up one future month while processing continues.

                Returns:
                    An independent lookup, or None when lookahead is finished.
                """
                nonlocal prefetch_cursor, next_first, next_last, next_needed
                while prefetch_cursor <= end:
                    next_first = prefetch_cursor
                    next_last = min(end, date(next_first.year, next_first.month, monthrange(next_first.year, next_first.month)[1]))
                    next_hours = {(name, hour) for name in variables for hour in range(
                        int(datetime.combine(next_first, time.min, UTC).timestamp()),
                        int(datetime.combine(next_last, time(23), UTC).timestamp()) + 1, 3600)}
                    next_needed = next_hours if refresh else next_hours - get_published_hours(db, next_first.year, next_first.month)
                    if next_needed:
                        break
                    prefetch_cursor = next_last + timedelta(days=1)
                else:
                    return None
                _, known = _recorded_source_owners(db, next_first, next_last, next_needed, refresh)
                names = sorted({name for name, hour in next_needed - known.keys()})
                cached = inventory_cache.get((next_first.year, next_first.month))
                if names and cached is None:
                    return inventory_pool.submit(_read_month_inventory, next_first, next_last, names)
                future: Future[list[SourceFile]] = Future()
                future.set_result(cached or [])
                return future

            def queue_prefetched_sources(found: list[SourceFile]) -> Future[list[SourceFile]] | None:
                """Fill spare source capacity with consecutive future months.

                Args:
                    found: Completed inventory for the current lookahead month.

                Returns:
                    The following lookup if capacity remains, otherwise None.
                """
                nonlocal prefetch_cursor
                inventory_cache[next_first.year, next_first.month] = found
                complete = _prefetch_month_sources(db, archive, downloads, next_first, next_last,
                                                   next_needed, found, refresh, max_source_gb)
                if complete:
                    prefetch_cursor = next_last + timedelta(days=1)
                    return start_prefetch()
                return None

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
                cached_inventory = inventory_cache.pop((cursor.year, cursor.month), None)
                activity = f"planning {label}"
                active_hours = len(needed)
                sources, owners = _recorded_source_owners(db, cursor, last, needed, refresh)
                undiscovered = needed - owners.keys()
                if undiscovered:
                    if prefetch is not None and (next_first.year, next_first.month) == (cursor.year, cursor.month):
                        cached_inventory = prefetch.result()
                        prefetch = None
                        prefetch_cursor = next_last + timedelta(days=1)
                    names = sorted({variable for variable, _ in undiscovered})
                    print(f"{label}: reading NCAR availability for {', '.join(names)}.", flush=True)
                    discovered = cached_inventory if cached_inventory is not None else discover_files(cursor, last, names, session)
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
                if prefetch is None:
                    prefetch_cursor = max(prefetch_cursor, last + timedelta(days=1))
                    prefetch = start_prefetch()

                result, prefetch = download_and_ingest_month(
                    db, archive, cursor.year, cursor.month, required_sources, rows, executor,
                    downloads=downloads, processing_workers=processing_workers, max_decoded_gb=max_decoded_gb,
                    selected_hours=set(owners), refresh=refresh,
                    on_published=record_completed_observation, prefetch=prefetch,
                    on_inventory=queue_prefetched_sources,
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


def _read_month_inventory(first: date, last: date, names: Sequence[str]) -> list[SourceFile]:
    """Fetch future-month availability without blocking processing completions.

    Args:
        first: First requested UTC date in a future month.
        last: Last requested UTC date in that month.
        names: Variables whose pending hours have no recorded source.

    Returns:
        Current provider objects for the requested selection.
    """
    with requests.Session() as session:
        return discover_files(first, last, names, session)


def _recorded_source_owners(
    db: sqlite3.Connection, first: date, last: date,
    needed: set[tuple[str, int]], refresh: bool,
) -> tuple[dict[tuple[str, str], SourceFile], dict[tuple[str, int], tuple[str, str]]]:
    """Find recorded providers for pending hours before querying NCAR.

    Args:
        db: Catalog owned by the coordinator.
        first: First selected UTC date.
        last: Last selected UTC date.
        needed: Variable-hours still requested.
        refresh: Require current provider identities instead of recorded inputs.

    Returns:
        Recorded source identities and their owned variable-hours.
    """
    first_hour = int(datetime.combine(first, time.min, UTC).timestamp())
    last_hour = int(datetime.combine(last, time(23), UTC).timestamp())
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
             datetime.combine(first, time.min, UTC).isoformat()),
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
    return sources, owners


def _prefetch_month_sources(
    db: sqlite3.Connection, archive: Path, downloads: NcarDownloadPool,
    first: date, last: date, needed: set[tuple[str, int]], found: Sequence[SourceFile],
    refresh: bool, max_source_gb: float,
) -> bool:
    """Queue future source files only while their full sizes fit the budget.

    Args:
        db: Catalog connection owned by the coordinator.
        archive: Root of registered source paths.
        downloads: Ongoing bounded transfer pool.
        first: First selected UTC date in a future month.
        last: Last selected UTC date in that month.
        needed: Future variable-hours still requested.
        found: Current provider inventory for missing source selections.
        refresh: Select current objects for replacement observations.
        max_source_gb: Budget including current, retained, partial, and prefetched files.

    Returns:
        Whether available sources fit and there is capacity for another lookup.
        Stop lookahead at an unavailable month or when a source cannot fit.
    """
    sources, owners = _recorded_source_owners(db, first, last, needed, refresh)
    first_hour = int(datetime.combine(first, time.min, UTC).timestamp())
    last_hour = int(datetime.combine(last, time(23), UTC).timestamp())
    for source in sorted(found, key=lambda item: item.last_modified, reverse=True):
        key = source.key, source.etag
        sources[key] = source
        for hour in range(max(first_hour, int(source.valid_start.timestamp())),
                          min(last_hour, int(source.valid_end.timestamp())) + 1, 3600):
            pair = source.variable, hour
            if pair in needed:
                owners.setdefault(pair, key)
    if refresh and needed - owners.keys():
        return False
    reserved = {(row["object_key"], row["etag"]): row["size"] for row in db.execute(
        "SELECT object_key, etag, size FROM source_files WHERE removed_at IS NULL")}
    total = sum(reserved.values())
    complete = bool(owners)
    for key in dict.fromkeys(owners.values()):
        source = sources[key]
        additional = 0 if key in reserved else source.size
        if total + additional > max_source_gb * 1_000_000_000:
            complete = False
            continue
        rows = register_sources(db, archive, [source], first, last,
                                selected_hours={pair for pair, owner in owners.items() if owner == key})
        downloads.add_sources([source], rows)
        total += additional
        reserved[key] = source.size
    return complete and total < max_source_gb * 1_000_000_000
