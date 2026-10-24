"""Locate retained sources and copy validated local COGs to archive staging."""

from collections.abc import Sequence
from concurrent.futures import CancelledError
import os
from pathlib import Path
import sqlite3
from threading import Event


def get_source_path(archive: Path, staging_dir: Path | None, record: sqlite3.Row) -> Path:
    """Locate a source using its cataloged storage root and relative path.

    Args:
        archive: Root of sources retained in the archive.
        staging_dir: Configured local root, when present.
        record: Source row including its nullable local_path from the catalog.

    Returns:
        The recorded local or archive-side NetCDF path.
    """
    if record["local_path"] is not None:
        return staging_dir / record["local_path"]
    return archive / record["path"]


def copy_local_cogs(
    staging_dir: Path, archive: Path, paths: Sequence[Path], *, stop_event: Event | None = None,
) -> None:
    """Copy and flush validated COGs to their archive-side journal paths.

    The relative local path is also the catalog's archive staging path. A partial
    copy stays unpublished; source inputs and local COGs remain available until
    the archive writer commits ready state and coverage. A retry overwrites the
    uncommitted destination rather than appending bytes from a partial upload.

    Args:
        staging_dir: Configured root of the prepared local COG paths.
        archive: Root of journaled upload destinations.
        paths: Validated paths returned by catalog preparation, in batch order.
        stop_event: Stop between copy blocks after interruption.

    Raises:
        CancelledError: The scheduler requests cancellation.
        OSError: Reading, copying, or flushing an output fails.
    """
    for path in paths:
        destination = archive / path.relative_to(staging_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as source, destination.open("wb") as target:
            while True:
                if stop_event is not None and stop_event.is_set():
                    raise CancelledError()
                block = source.read(8 * 1024 * 1024)
                if not block:
                    break
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
