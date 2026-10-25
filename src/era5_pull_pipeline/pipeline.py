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
    catalog_transaction, delete_ingested_netcdf_files, get_published_hours, get_staging_dir,
    prepare_observations, publish_observations, register_sources, remove_published_source_hours,
)
from era5_pull_pipeline.ncar import SourceFile, discover_files
from era5_pull_pipeline.ncar_conversion import NcarChunk, plan_ncar_chunk_jobs, write_ncar_chunk_cogs
from era5_pull_pipeline.ncar_download import NcarDownloadPool, record_download_completions
from era5_pull_pipeline.publication import PublicationWorker
from era5_pull_pipeline.ncar_ingest import ingest_ncar_month, select_ncar_source_hours
from era5_pull_pipeline.staging import copy_local_cogs, get_source_path


def _ignore_worker_interrupt() -> None:
    """Let the coordinator stop scheduling and wait for active worker writes."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _prepare_source_publications(
    db: sqlite3.Connection, archive: Path, source_id: str,
    report: dict[str, Any], jobs: Sequence[NcarChunk],
) -> dict[int, tuple[str, Path]]:
    """Record source metadata and prepare its selected observations together.

    Args:
        db: Connection owned by the publication worker.
        archive: Archive receiving staged COGs.
        source_id: Registered input identity.
        report: Validated source metadata and grid.
        jobs: Planned native chunks containing the selected observations.

    Returns:
        Publication identities and paths indexed by valid UTC time.
    """
    fields = [field for job in jobs for field in job.fields]
    with catalog_transaction(db):
        db.execute("UPDATE source_files SET attributes_json = ? WHERE id = ?",
                   (json.dumps(report["source_attributes"]), source_id))
        prepared = prepare_observations(db, archive, fields,
                                       report["variables"][fields[0]["variable"]])
    return {field["valid_time"]: preparation for field, preparation in zip(fields, prepared, strict=True)}


def _finish_month_sources(
    db: sqlite3.Connection, archive: Path, selections: Sequence[dict[str, str]],
) -> None:
    """Retire published selections and remove inputs no longer needed.

    Args:
        db: Connection owned by the publication worker.
        archive: Root of recorded source paths.
        selections: Current month's selected source intervals.
    """
    with catalog_transaction(db):
        remove_published_source_hours(db, selections)
    delete_ingested_netcdf_files(db, archive)


def download_and_ingest_month(
    publisher: PublicationWorker, archive: Path, year: int, month: int,
    sources: Sequence[SourceFile], rows: Sequence[sqlite3.Row],
    executor: ProcessPoolExecutor, *, downloads: NcarDownloadPool,
    uploads: ThreadPoolExecutor, upload_workers: int,
    processing_workers: int, max_decoded_gb: float,
    selected_hours: set[tuple[str, int]], refresh: bool,
    on_published: Callable[[], None], poll_prefetch: Callable[[], Future[Any] | None],
) -> Path:
    """Overlap downloads, native-chunk conversion, uploads, and batch publication.

    Processing reservations end when encoding finishes. A separate allowance of
    twice the process count bounds native batches from submission through durable
    publication acknowledgement, including files still being encoded.

    Args:
        publisher: Sole catalog and final-file writer with a bounded request queue.
        archive: Root of source, staging, and published paths.
        year: Selected UTC year.
        month: Selected UTC month.
        sources: Budgeted provider objects to download or reuse.
        rows: Registered source rows in the same order as sources.
        executor: Spawned process pool owned by this invocation.
        downloads: Transfer pool retained across monthly processing boundaries.
        uploads: Separate pool copying validated COGs to the archive.
        upload_workers: Maximum concurrent native-batch uploads.
        processing_workers: Maximum outstanding metadata or chunk jobs.
        max_decoded_gb: Maximum reserved decoded chunk bytes in decimal GB.
        selected_hours: Exact source variable and Unix-hour selection.
        refresh: Replace selected published observations.
        on_published: Count each durably cataloged observation.
        poll_prefetch: Service independent lookahead without blocking scheduling.

    Returns:
        Published month directory.

    Raises:
        ValueError: A native chunk cannot fit the decoded-byte budget.
        BaseException: Failures propagate after workers stop; inputs and durable
            publications remain available for resume.
    """
    records, selected, selections = publisher.call(
        select_ncar_source_hours, year, month, refresh=refresh, selected_hours=selected_hours,
        source_ids={row["id"] for row in rows},
    )
    downloads.add_sources(sources, rows)
    waiting = set(selected)
    ready: deque[str] = deque()
    planning: dict[Future[Any], str] = {}
    preparing: dict[Future[Any], tuple[str, list[NcarChunk]]] = {}
    unprepared: deque[tuple[str, list[NcarChunk]]] = deque()
    preparations: dict[str, dict[int, tuple[str, Path]]] = {}
    chunks: deque[tuple[str, NcarChunk]] = deque()
    reports: dict[str, dict[str, Any]] = {}
    active: dict[Future[Any], tuple[str, NcarChunk, list[str]]] = {}
    encoded: deque[tuple[str, NcarChunk, list[tuple[str, dict[str, Any]]]]] = deque()
    uploading: dict[Future[Any], tuple[str, NcarChunk, list[tuple[str, dict[str, Any]]]]] = {}
    uploaded: deque[tuple[str, NcarChunk, list[tuple[str, dict[str, Any]]]]] = deque()
    publishing: dict[Future[Any], tuple[str, NcarChunk]] = {}
    reserved = 0
    limit = int(max_decoded_gb * 1_000_000_000)
    published: Counter[str] = Counter()

    def acknowledge_publications() -> None:
        """Release staging capacity and count batches after their durable commits."""
        for future, (source_id, chunk) in list(publishing.items()):
            if not future.done() or future.cancelled():
                continue
            future.result()
            del publishing[future]
            for _ in chunk.fields:
                on_published()
            published[source_id] += len(chunk.fields)
            print(f"Published {records[source_id]['variable']}: {published[source_id]}/{len(selected[source_id])} observations.", flush=True)

    try:
        while waiting or ready or planning or unprepared or preparing or chunks or active or encoded or uploading or uploaded or publishing:
            available = downloads.ready + downloads.record_completed()
            downloads.ready.clear()
            for source_id in available:
                if source_id in waiting:
                    waiting.remove(source_id)
                    ready.append(source_id)
            prefetch = poll_prefetch()
            acknowledge_publications()
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
                    unprepared.append((source_id, jobs))
                    del planning[future]
                    print(f"Processing {records[source_id]['variable']}: {len(jobs)} independent native chunks.", flush=True)
            for future, (source_id, jobs) in list(preparing.items()):
                if future.done():
                    preparations[source_id] = future.result()
                    chunks.extend((source_id, job) for job in jobs)
                    del preparing[future]
            for future, (source_id, chunk, identifiers) in list(active.items()):
                if future.done():
                    encoded.append((source_id, chunk, list(zip(identifiers, future.result(), strict=True))))
                    reserved -= chunk.decoded_bytes
                    del active[future]
            # Prepare each source once, before its chunks compete for publication.
            while unprepared:
                source_id, jobs = unprepared[0]
                future = publisher.try_submit(_prepare_source_publications, archive, source_id, reports[source_id], jobs)
                if future is None:
                    break
                unprepared.popleft()
                preparing[future] = source_id, jobs
            for future, batch in list(uploading.items()):
                if future.done():
                    future.result()
                    uploaded.append(batch)
                    del uploading[future]
            while encoded and len(uploading) < upload_workers:
                source_id, chunk, observations = encoded.popleft()
                if downloads.staging_dir is None:
                    uploaded.append((source_id, chunk, observations))
                else:
                    paths = [preparations[source_id][field["valid_time"]][1] for field in chunk.fields]
                    future = uploads.submit(copy_local_cogs, downloads.staging_dir, archive, paths,
                                            stop_event=downloads.stop_event)
                    uploading[future] = source_id, chunk, observations
            while uploaded:
                source_id, chunk, observations = uploaded[0]
                if downloads.staging_dir is None:
                    future = publisher.try_submit(publish_observations, archive, observations)
                else:
                    future = publisher.try_submit(publish_observations, archive, observations, files_flushed=True)
                if future is None:
                    break
                uploaded.popleft()
                publishing[future] = source_id, chunk
            while ready and len(planning) + len(active) < processing_workers:
                source_id = ready.popleft()
                record = records[source_id]
                future = executor.submit(plan_ncar_chunk_jobs, get_source_path(archive, downloads.staging_dir, record), source_id,
                                         record["variable"], selected[source_id])
                planning[future] = source_id
            for _ in range(len(chunks)):
                if (len(planning) + len(active) >= processing_workers
                        or len(active) + len(encoded) + len(uploading) + len(uploaded) + len(publishing) >= 2 * processing_workers):
                    break
                source_id, chunk = chunks.popleft()
                if reserved + chunk.decoded_bytes > limit:
                    chunks.append((source_id, chunk))
                    continue
                report = reports[source_id]
                prepared = [preparations[source_id][field["valid_time"]] for field in chunk.fields]
                future = executor.submit(
                    write_ncar_chunk_cogs, get_source_path(archive, downloads.staging_dir, records[source_id]), chunk,
                    report["variables"][chunk.fields[0]["variable"]], report["grid"],
                    [path for _, path in prepared],
                )
                active[future] = source_id, chunk, [identifier for identifier, _ in prepared]
                reserved += chunk.decoded_bytes
            pending = [*downloads.pending, *planning, *preparing, *active, *uploading, *publishing]
            if prefetch is not None:
                pending.append(prefetch)
            # A full publication queue must not stop transfer completion handling.
            if pending:
                wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
    except BaseException:
        downloads.stop_event.set()
        publisher.cancel_pending()
        executor.shutdown(wait=True, cancel_futures=True)
        uploads.shutdown(wait=True, cancel_futures=True)
        publisher.drain()
        # A publication already executing can commit during shutdown. Count it
        # without replacing the original failure with another failed request.
        for future, (_, chunk) in publishing.items():
            if not future.cancelled() and future.exception() is None:
                for _ in chunk.fields:
                    on_published()
        raise
    publisher.call(_finish_month_sources, archive, selections)
    return archive / f"{year:04d}" / f"{month:02d}"


def pull_range(
    start: date,
    end: date,
    variables: Sequence[str],
    archive: Path,
    *,
    download_workers: int = 2,
    processing_workers: int = 4,
    staging_dir: Path | None = None,
    upload_workers: int = 2,
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
        staging_dir: Local temporary root to remember; omit to reuse configured storage.
        upload_workers: Maximum simultaneous native-batch uploads from local staging.
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
    if download_workers < 1 or processing_workers < 1 or upload_workers < 1 or max_source_gb <= 0 or max_decoded_gb <= 0:
        raise ValueError("Download workers, processing workers, upload workers, source and decoded budgets must be positive.")
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
        with PublicationWorker(archive, processing_workers, staging_dir=staging_dir) as publisher, requests.Session() as session, ProcessPoolExecutor(
            max_workers=processing_workers, mp_context=get_context("spawn"),
            initializer=_ignore_worker_interrupt,
        ) as executor, NcarDownloadPool(
            lambda identifiers: publisher.try_submit(record_download_completions, identifiers),
            [], [], download_workers, archive, staging_dir=publisher.call(get_staging_dir),
        ) as downloads, ThreadPoolExecutor(max_workers=upload_workers) as uploads, ThreadPoolExecutor(max_workers=1) as inventory_pool:
            inventory_cache: dict[tuple[int, int], list[SourceFile]] = {}
            prefetch_cursor = start
            prefetch: Future[Any] | None = None
            prefetch_paused = False

            def prefetch_month(first: date) -> tuple[Any, ...] | None:
                """Discover and reserve a future month off the scheduling thread.

                Args:
                    first: Earliest date still eligible for lookahead.

                Returns:
                    Selection, inventory, admitted sources and remaining capacity,
                    or None when every future selection is already published.
                """
                selection = publisher.call(_select_prefetch_month, first, end, variables, refresh)
                if selection is None:
                    return None
                first, last, needed, names = selection
                found = inventory_cache.get((first.year, first.month))
                if found is None:
                    found = _read_month_inventory(first, last, names) if names else []
                admitted, rows, complete = publisher.call(
                    _prefetch_month_sources, archive, first, last, needed, found, refresh, max_source_gb,
                )
                return first, last, found, admitted, rows, complete

            def poll_prefetch() -> Future[Any] | None:
                """Admit completed lookahead transfers without waiting on the writer.

                Returns:
                    Outstanding lookahead work, if any.
                """
                nonlocal prefetch, prefetch_cursor, prefetch_paused
                if prefetch is not None and prefetch.done():
                    result = prefetch.result()
                    prefetch = None
                    if result is None:
                        prefetch_cursor = end + timedelta(days=1)
                    else:
                        first, last, found, admitted, rows, complete = result
                        inventory_cache[first.year, first.month] = found
                        downloads.add_sources(admitted, rows)
                        prefetch_cursor = last + timedelta(days=1) if complete else first
                        prefetch_paused = not complete
                if prefetch is None and not prefetch_paused and prefetch_cursor <= end:
                    prefetch = inventory_pool.submit(prefetch_month, prefetch_cursor)
                return prefetch

            cursor = start
            while cursor <= end:
                last = min(end, date(cursor.year, cursor.month, monthrange(cursor.year, cursor.month)[1]))
                first_hour = int(datetime.combine(cursor, time.min, UTC).timestamp())
                last_hour = int(datetime.combine(last, time(23), UTC).timestamp())
                requested = {
                    (variable, hour) for variable in variables
                    for hour in range(first_hour, last_hour + 1, 3600)
                }
                published = publisher.call(get_published_hours, cursor.year, cursor.month)
                needed = requested if refresh else requested - published
                skipped = len(requested) - len(needed)
                summary["skipped"] += skipped
                summary["pending"] -= skipped
                label = f"{cursor.year:04d}-{cursor.month:02d}"
                if skipped:
                    publisher.call(
                        ingest_ncar_month, archive, cursor.year, cursor.month,
                        selected_hours=requested & published, source_ids=set(),
                    )
                if not needed:
                    print(f"{label}: skipped {skipped} published variable-hours.", flush=True)
                    cursor = last + timedelta(days=1)
                    continue
                # Wait only for lookahead that can own this month's selection.
                # A later month's inventory must never delay current processing.
                if prefetch is not None and prefetch_cursor <= cursor:
                    prefetch.result()
                    poll_prefetch()
                cached_inventory = inventory_cache.pop((cursor.year, cursor.month), None)
                activity = f"planning {label}"
                active_hours = len(needed)
                sources, owners = publisher.call(_recorded_source_owners, cursor, last, needed, refresh)
                undiscovered = needed - owners.keys()
                if undiscovered:
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
                rows, required_bytes = publisher.call(
                    _register_month_sources, archive, required_sources, cursor, last,
                    end, set(owners), refresh, max_source_gb,
                )
                print(f"{label}: {len(owners)} variable-hours to publish; {skipped} skipped; "
                      f"{len(required_sources)} source files; {required_bytes / 1_000_000_000:.3f} GB reserved.", flush=True)
                activity = f"downloading, converting and publishing {label}"
                print(f"{label}: overlapping downloads and processing ({processing_workers} processing workers).", flush=True)
                if prefetch is None:
                    prefetch_cursor = max(prefetch_cursor, last + timedelta(days=1))
                    prefetch_paused = False
                    poll_prefetch()
                result = download_and_ingest_month(
                    publisher, archive, cursor.year, cursor.month, required_sources, rows, executor,
                    downloads=downloads, uploads=uploads, upload_workers=upload_workers, processing_workers=processing_workers, max_decoded_gb=max_decoded_gb,
                    selected_hours=set(owners), refresh=refresh,
                    on_published=record_completed_observation, poll_prefetch=poll_prefetch,
                )
                active_hours = 0
                print(f"Published {result}.", flush=True)
                cursor = last + timedelta(days=1)
    except BaseException:
        summary["failed"] += active_hours
        summary["pending"] -= active_hours
        print(f"Stopped while {activity}. Run the same command to resume; "
              "published observations and retained inputs remain at their cataloged paths.", flush=True)
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
        db: Catalog owned by the publication worker.
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
    db: sqlite3.Connection, archive: Path,
    first: date, last: date, needed: set[tuple[str, int]], found: Sequence[SourceFile],
    refresh: bool, max_source_gb: float,
) -> tuple[list[SourceFile], list[sqlite3.Row], bool]:
    """Reserve future source files only while their full sizes fit the budget.

    Args:
        db: Catalog connection owned by the publication worker.
        archive: Root of registered source paths.
        first: First selected UTC date in a future month.
        last: Last selected UTC date in that month.
        needed: Future variable-hours still requested.
        found: Current provider inventory for missing source selections.
        refresh: Select current objects for replacement observations.
        max_source_gb: Budget including current, retained, partial, and prefetched files.

    Returns:
        Admitted sources, their rows, and whether another lookup can fit.
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
        return [], [], False
    reserved = {(row["object_key"], row["etag"]): row["size"] for row in db.execute(
        "SELECT object_key, etag, size FROM source_files WHERE removed_at IS NULL")}
    total = sum(reserved.values())
    complete = bool(owners)
    admitted: list[SourceFile] = []
    admitted_rows: list[sqlite3.Row] = []
    for key in dict.fromkeys(owners.values()):
        source = sources[key]
        additional = 0 if key in reserved else source.size
        if total + additional > max_source_gb * 1_000_000_000:
            complete = False
            continue
        rows = register_sources(db, archive, [source], first, last,
                                selected_hours={pair for pair, owner in owners.items() if owner == key})
        admitted.append(source)
        admitted_rows.extend(rows)
        total += additional
        reserved[key] = source.size
    return admitted, admitted_rows, complete and total < max_source_gb * 1_000_000_000



def _register_month_sources(
    db: sqlite3.Connection, archive: Path, required_sources: Sequence[SourceFile],
    first: date, last: date, end: date, owners: set[tuple[str, int]],
    refresh: bool, max_source_gb: float,
) -> tuple[list[sqlite3.Row], int]:
    """Reserve current inputs and the future selected hours in shared source files.

    Args:
        db: Connection owned by the publication worker.
        archive: Root of source paths.
        required_sources: Files owning current month's required observations.
        first: First selected date this month.
        last: Last selected date this month.
        end: Last requested date across all months.
        owners: Required variable-hour pairs.
        refresh: Replace selected observations even when already published.
        max_source_gb: Full-size budget for all retained and reserved inputs.

    Returns:
        Registered source rows and total reserved bytes.

    Raises:
        ValueError: Current and retained inputs exceed the source budget.
    """
    label = f"{first.year:04d}-{first.month:02d}"
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
        db, archive, required_sources, first, end, selected_hours=registered_hours,
    )
    return rows, required_bytes


def _select_prefetch_month(
    db: sqlite3.Connection, first: date, end: date,
    variables: Sequence[str], refresh: bool,
) -> tuple[date, date, set[tuple[str, int]], list[str]] | None:
    """Select the next incomplete month and variables needing provider inventory.

    Args:
        db: Connection owned by the publication worker.
        first: Earliest eligible date.
        end: Last requested date.
        variables: Requested provider variable names.
        refresh: Include already published observations for replacement.

    Returns:
        Month interval, missing hours and undiscovered variables, or None.
    """
    while first <= end:
        last = min(end, date(first.year, first.month, monthrange(first.year, first.month)[1]))
        hours = {(name, hour) for name in variables for hour in range(
            int(datetime.combine(first, time.min, UTC).timestamp()),
            int(datetime.combine(last, time(23), UTC).timestamp()) + 1, 3600)}
        needed = hours if refresh else hours - get_published_hours(db, first.year, first.month)
        if needed:
            _, known = _recorded_source_owners(db, first, last, needed, refresh)
            return first, last, needed, sorted({name for name, _ in needed - known.keys()})
        first = last + timedelta(days=1)
    return None
