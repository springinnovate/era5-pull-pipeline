"""Download NCAR source files with catalog-owned selection and completion state."""

from collections.abc import Sequence
from datetime import UTC, date, datetime
import os
from pathlib import Path
import time

import requests
from tqdm import tqdm

from era5_pull_pipeline.catalog import archive_session, register_sources
from era5_pull_pipeline.ncar import BUCKET_URL, SourceFile, discover_files

TRANSFER_ATTEMPTS = 3
CHUNK_BYTES = 1024 * 1024
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def download_file(source: SourceFile, target: Path, session: requests.Session) -> None:
    """Resume an incomplete source file and flush it before recording completion.

    The source ETag pins every request to the discovered object. Pending files
    remain at their cataloged paths after interruption; only catalog completion
    makes a file available for ingestion.

    Args:
        source: Object identity and expected size from NCAR's inventory.
        target: Cataloged path for this source, possibly partially downloaded.
        session: HTTP session for anonymous source access.

    Raises:
        ValueError: A resumed response does not cover the requested byte range.
        requests.RequestException: A permanent HTTP error occurs or transient
            transfer failures exhaust the bounded retry attempts.
        OSError: Creating, writing, or flushing the source file fails.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    initial = target.stat().st_size if target.exists() else 0
    started = time.monotonic()
    with tqdm(
        total=source.size, initial=initial, desc=source.variable,
        unit="B", unit_scale=True, unit_divisor=1024,
    ) as progress:
        for attempt in range(TRANSFER_ATTEMPTS):
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
                time.sleep(delay)
    elapsed = time.monotonic() - started
    print(
        f"Downloaded {target.name}: {source.size:,} bytes total; "
        f"{source.size - initial:,} new bytes in {elapsed:.1f}s.",
        flush=True,
    )


def download_ncar(
    start: date, end: date, variables: Sequence[str], archive: Path,
    *, dry_run: bool = False,
) -> list[Path]:
    """Discover and stage whole NCAR files for an inclusive UTC date selection.

    Source files can contain extra hours; SQLite records the requested overlap
    separately for ingestion. Completed downloads are reused directly and are
    retained on disk. This command does not publish Zarr coverage.

    Args:
        start: First selected UTC date.
        end: Last selected UTC date.
        variables: NCAR short names read from its file inventory.
        archive: Directory owning catalog.sqlite and source staging paths.
        dry_run: List source files and sizes without creating local state.

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
        completed = []
        with archive_session(archive) as db:
            rows = register_sources(db, archive, sources, start, end)
            for source, row in zip(sources, rows, strict=True):
                path = Path(row["path"])
                if row["downloaded_at"] is not None:
                    print(f"Reusing completed source: {path}", flush=True)
                else:
                    download_file(source, path, session)
                    with db:
                        db.execute(
                            "UPDATE source_files SET downloaded_at = ? WHERE id = ?",
                            (datetime.now(UTC).isoformat(), row["id"]),
                        )
                completed.append(path)
        print(f"Retained {len(completed)} NetCDF files; catalog: {archive / 'catalog.sqlite'}", flush=True)
        return completed
