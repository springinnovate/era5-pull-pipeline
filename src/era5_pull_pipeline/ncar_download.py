"""Download NCAR source files with catalog-owned selection and completion state."""

from collections import deque
from collections.abc import Sequence
from concurrent.futures import CancelledError, FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, date, datetime
import os
from pathlib import Path
import sqlite3
from threading import Event
import time
from types import TracebackType

import requests
from tqdm import tqdm

from era5_pull_pipeline.catalog import archive_session, register_sources
from era5_pull_pipeline.ncar import BUCKET_URL, SourceFile, discover_files

TRANSFER_ATTEMPTS = 3
CHUNK_BYTES = 1024 * 1024
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def download_file(
    source: SourceFile, target: Path, *, stop_event: Event | None = None,
    position: int = 0,
) -> None:
    """Download an NCAR NetCDF file, resuming any bytes already on disk.

    The source ETag pins every request to the discovered object. Pending files
    remain at their cataloged paths after interruption; only catalog completion
    makes a file available for ingestion.

    Args:
        source: Object identity and expected size from NCAR's inventory.
        target: Cataloged path for this source, possibly partially downloaded.
        stop_event: Signal from the scheduler to stop and retain partial bytes.
        position: Terminal progress-bar row assigned to this transfer.

    Raises:
        ValueError: A resumed response does not cover the requested byte range.
        requests.RequestException: A permanent HTTP error occurs or transient
            transfer failures exhaust the bounded retry attempts.
        OSError: Creating, writing, or flushing the source file fails.
        CancelledError: The scheduler stops this transfer before completion.
    """
    stop_event = stop_event or Event()
    target.parent.mkdir(parents=True, exist_ok=True)
    initial = target.stat().st_size if target.exists() else 0
    started = time.monotonic()
    with requests.Session() as session, tqdm(
        total=source.size, initial=initial, desc=source.variable,
        unit="B", unit_scale=True, unit_divisor=1024, position=position,
    ) as progress:
        for attempt in range(TRANSFER_ATTEMPTS):
            if stop_event.is_set():
                raise CancelledError()
            offset = target.stat().st_size if target.exists() else 0
            # A process can stop after receiving every byte but before the
            # catalog commit. Finish flushing that pending file without a GET.
            if offset == source.size:
                with target.open("r+b") as stream:
                    os.fsync(stream.fileno())
                break
            headers = {"If-Match": source.etag, "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
                print(f"Resuming {target.name} from byte {offset:,}.", flush=True)
            try:
                with session.get(
                    f"{BUCKET_URL}/{source.key}", headers=headers,
                    stream=True, timeout=(10, 30),
                ) as response:
                    response.raise_for_status()
                    if offset and (
                        response.status_code != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {offset}-{source.size - 1}/{source.size}"
                    ):
                        raise ValueError(
                            f"NCAR did not return the requested byte range for {source.key}; "
                            "the partial file was retained."
                        )
                    with target.open("ab" if offset else "wb") as stream:
                        for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                            if stop_event.is_set():
                                raise CancelledError()
                            stream.write(chunk)
                            progress.update(len(chunk))
                        stream.flush()
                        os.fsync(stream.fileno())
                received = target.stat().st_size
                if received != source.size:
                    raise requests.exceptions.ChunkedEncodingError(
                        f"NCAR transfer ended at {received:,} of {source.size:,} bytes."
                    )
                break
            except (
                requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError, requests.HTTPError,
            ) as exc:
                if isinstance(exc, requests.exceptions.SSLError):
                    raise
                if isinstance(exc, requests.HTTPError) and exc.response.status_code not in RETRY_STATUS:
                    raise
                if attempt + 1 == TRANSFER_ATTEMPTS:
                    raise
                delay = 2 ** (attempt + 1)
                print(
                    f"Transfer interrupted ({exc}); retrying in {delay}s "
                    f"({attempt + 2}/{TRANSFER_ATTEMPTS}).",
                    flush=True,
                )
                if stop_event.wait(delay):
                    raise CancelledError() from exc
    elapsed = time.monotonic() - started
    print(
        f"Downloaded {target.name}: {source.size:,} bytes total; "
        f"{source.size - initial:,} new bytes in {elapsed:.1f}s.",
        flush=True,
    )


class NcarDownloadPool:
    """Keep bounded transfers running while the coordinator processes completions."""

    def __init__(
        self, db: sqlite3.Connection, sources: Sequence[SourceFile],
        rows: Sequence[sqlite3.Row], download_workers: int, archive: Path,
    ) -> None:
        """Prepare transfers for registered sources and reuse completed downloads.

        Args:
            db: Catalog connection owned exclusively by the calling thread.
            sources: Provider objects in the same order as their catalog rows.
            rows: Registered paths and completion state.
            download_workers: Maximum simultaneous transfers.
            archive: Root of the cataloged paths.
        """
        self.db = db
        self.archive = archive
        self.workers = download_workers
        self.ready: list[str] = []
        self.remaining: deque[tuple[SourceFile, sqlite3.Row]] = deque()
        self.queued: set[str] = set()
        self.started = False
        self.stop_event = Event()
        self.executor = ThreadPoolExecutor(max_workers=download_workers)
        self.active: dict[Future[None], tuple[sqlite3.Row, int]] = {}
        self.add_sources(sources, rows)

    def add_sources(self, sources: Sequence[SourceFile], rows: Sequence[sqlite3.Row]) -> None:
        """Queue newly budgeted sources without duplicating in-progress transfers.

        Args:
            sources: Provider objects already reserved by the coordinator.
            rows: Catalog rows in the same order as sources.
        """
        for source, row in zip(sources, rows, strict=True):
            if row["downloaded_at"] is not None:
                print(f"Reusing completed source: {row['path']}", flush=True)
                self.ready.append(row["id"])
            elif row["id"] not in self.queued:
                self.remaining.append((source, row))
                self.queued.add(row["id"])
        if self.started:
            occupied = {position for _, position in self.active.values()}
            for position in range(self.workers):
                if position not in occupied:
                    self._submit_next(position)

    def _submit_next(self, position: int) -> None:
        """Fill a free transfer slot with the next registered source.

        Args:
            position: Progress display slot freed by a completed transfer.
        """
        if self.remaining:
            source, row = self.remaining.popleft()
            future = self.executor.submit(
                download_file, source, self.archive / row["path"],
                stop_event=self.stop_event, position=position,
            )
            self.active[future] = (row, position)

    def __enter__(self) -> "NcarDownloadPool":
        """Start the bounded batch and return its coordinator interface.

        Returns:
            This batch with its initial transfer futures and reused source IDs.

        Raises:
            Exception: Submission failures propagate after active transfers stop.
        """
        self.started = True
        try:
            for position in range(self.workers):
                self._submit_next(position)
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise
        return self

    def record_completed(self) -> list[str]:
        """Record finished downloads and refill free transfer slots.

        Returns:
            Source IDs whose complete bytes are now recorded in SQLite.

        Raises:
            Exception: Transfer or catalog failures propagate to the coordinator.
        """
        completed = []
        positions = []
        for future, (row, position) in list(self.active.items()):
            if not future.done():
                continue
            future.result()
            retrieved_at = datetime.now(UTC).isoformat()
            with self.db:
                self.db.execute(
                    "UPDATE source_files SET downloaded_at = ?, retrieved_at = ? WHERE id = ?",
                    (retrieved_at, retrieved_at, row["id"]),
                )
            del self.active[future]
            self.queued.remove(row["id"])
            completed.append(row["id"])
            positions.append(position)
        for position in positions:
            self._submit_next(position)
        return completed

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop transfers and preserve successful concurrent downloads on failure.

        Args:
            exc_type: Exception type leaving the coordinator, if any.
            exc: Original failure to annotate if catalog recording also fails.
            traceback: Original coordinator traceback.

        Raises:
            sqlite3.Error: Completion recording fails without an original error.
        """
        self.stop_event.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
        retrieved_at = datetime.now(UTC).isoformat()
        completed = [
            (retrieved_at, retrieved_at, row["id"])
            for future, (row, _) in self.active.items()
            if not future.cancelled() and future.exception() is None
        ]
        if completed:
            try:
                with self.db:
                    self.db.executemany(
                        "UPDATE source_files SET downloaded_at = ?, retrieved_at = ? WHERE id = ?",
                        completed,
                    )
            except sqlite3.Error as catalog_error:
                if exc is None:
                    raise
                exc.add_note(f"Could not record completed downloads: {catalog_error}")


def download_registered_sources(
    db: sqlite3.Connection, sources: Sequence[SourceFile],
    rows: Sequence[sqlite3.Row], download_workers: int = 2, *, archive: Path,
) -> None:
    """Download registered NCAR files concurrently and record their completion.

    Args:
        db: Catalog connection owned exclusively by the calling thread.
        sources: Provider objects in the same order as their catalog rows.
        rows: Registered paths and completion state.
        download_workers: Maximum simultaneous file downloads.
        archive: Root of the cataloged relative paths.

    Raises:
        Exception: Transfer and catalog errors propagate after workers stop.
        KeyboardInterrupt: An interrupted run stops its active transfers.
    """
    with NcarDownloadPool(db, sources, rows, download_workers, archive) as downloads:
        while downloads.active:
            wait(downloads.active, return_when=FIRST_COMPLETED)
            downloads.record_completed()


def download_ncar(
    start: date, end: date, variables: Sequence[str], archive: Path,
    *, dry_run: bool = False, download_workers: int = 2,
) -> list[Path]:
    """Discover and stage whole NCAR files for an inclusive UTC date selection.

    Source files can contain extra hours; SQLite records the requested overlap
    separately for ingestion. Completed downloads are reused directly and are
    retained on disk. This command does not publish COG observations.

    Args:
        start: First selected UTC date.
        end: Last selected UTC date.
        variables: NCAR short names read from its file inventory.
        archive: Directory owning catalog.sqlite and source staging paths.
        dry_run: List source files and sizes without creating local state.
        download_workers: Maximum number of simultaneous file downloads.

    Returns:
        Completed local source paths, or an empty list for a dry run.

    Raises:
        Exception: Inventory, transfer, filesystem, and catalog errors propagate.
            Incomplete downloads remain available for another invocation.
    """
    print("Reading NCAR's public file inventory.", flush=True)
    with requests.Session() as session:
        sources = discover_files(start, end, variables, session)
        for variable in sorted(set(variables) - {source.variable for source in sources}):
            print(f"No files found for {variable} in the selected date range.", flush=True)
        total = sum(source.size for source in sources)
        print(f"Selected {len(sources)} source files ({total:,} bytes).", flush=True)
        for source in sources:
            print(f"  {source.size:>14,} bytes  {source.key}", flush=True)
        if dry_run or not sources:
            return []
        with archive_session(archive) as db:
            rows = register_sources(db, archive, sources, start, end)
            download_registered_sources(db, sources, rows, download_workers, archive=archive)
            completed = [archive / row["path"] for row in rows]
        print(f"Retained {len(completed)} NetCDF files; catalog: {archive / 'catalog.sqlite'}", flush=True)
        return completed
