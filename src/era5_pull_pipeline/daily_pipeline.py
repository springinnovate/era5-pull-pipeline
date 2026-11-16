"""Pull complete UTC days through the existing hourly workflow and daily reducer."""

from calendar import monthrange
from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import portalocker

from era5_pull_pipeline.catalog import archive_session
from era5_pull_pipeline.daily import (
    INTERVAL_VARIABLES, aggregate_daily_session, completed_daily_days,
    validate_daily_selection,
)
from era5_pull_pipeline.pipeline import pull_range


def _open_hourly_scratch(work_dir: Path, archive: Path, first: date, last: date,
                         variable: str, *, create: bool) -> Path:
    """Identify an owned hourly scratch directory for one monthly selection.

    Keep its ownership record beside the directory, so interrupted creation or
    partial cleanup remains recognizable without a separate progress record.

    Args:
        work_dir: Resolved private-work root protected by the daily-pull lock.
        archive: Resolved daily output archive that owns this scratch selection.
        first: First requested date in the month.
        last: Last requested date in the month.
        variable: Validated NCAR short variable name.
        create: Create absent scratch when work is needed; otherwise only inspect it.

    Returns:
        Confined scratch path, which may be absent when create is false.

    Raises:
        ValueError: An existing path is a link, escapes the work root, or lacks
            the ownership record for this archive and selection.
        OSError: Creating or reading the ownership record fails.
    """
    scratch = work_dir / f"daily-hourly-{first:%Y-%m-%d}-{last.day:02d}-{variable}"
    if scratch.is_symlink() or not scratch.resolve().is_relative_to(work_dir):
        raise ValueError("Daily hourly work must stay inside its private work directory.")
    owner = {"archive": str(archive), "start": first.isoformat(),
             "end": last.isoformat(), "variable": variable}
    marker = work_dir / f".{scratch.name}.owner.json"
    if marker.exists() or marker.is_symlink():
        if marker.is_symlink() or not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != owner:
            raise ValueError("The daily hourly work directory belongs to another selection; choose a different --work-dir.")
    elif scratch.exists():
        raise ValueError("The daily hourly work directory belongs to another selection; choose a different --work-dir.")
    elif create:
        # Publish ownership before allocating scratch. A partial cleanup cannot
        # remove this record, and interrupted mkdir can reuse it on the next run.
        record = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=work_dir,
                                              prefix=f".{scratch.name}.", suffix=".tmp", delete=False)
        temporary = Path(record.name)
        try:
            with record:
                json.dump(owner, record)
                record.flush()
                os.fsync(record.fileno())
            temporary.replace(marker)
        finally:
            temporary.unlink(missing_ok=True)
    if create and not scratch.exists():
        scratch.mkdir()
    return scratch


def pull_daily(
    start: date,
    end: date,
    variables: Sequence[str],
    archive: Path,
    *,
    work_dir: Path,
    download_workers: int = 2,
    processing_workers: int = 4,
    upload_workers: int = 2,
    max_source_gb: float = 16,
    max_decoded_gb: float = 2,
    refresh: bool = False,
) -> dict[str, int]:
    """Download hourly inputs and publish complete UTC daily means and totals.

    Process one variable and calendar-month selection at a time using the
    existing NCAR download/conversion pipeline. Temporary hourly COGs stay
    outside the daily archive. Keep incomplete or failed selections for resume;
    stop at the first incomplete selection instead of accumulating unavailable
    future months. Remove only this invocation's owned monthly inputs after all
    requested daily observations are durably published. Repeating the same
    completed selection makes no provider requests.

    Args:
        start: First requested UTC date, inclusive.
        end: Last requested UTC date, inclusive.
        variables: Supported NCAR short variable names for daily means or totals.
        archive: Separate daily output archive, held by one writer throughout.
        work_dir: Private hourly work root disjoint from the daily output archive.
        download_workers: Maximum simultaneous NCAR file transfers.
        processing_workers: Maximum simultaneous hourly conversion processes.
        upload_workers: Maximum simultaneous hourly COG uploads.
        max_source_gb: Maximum retained NetCDF source bytes in decimal GB.
        max_decoded_gb: Maximum decoded hourly chunk bytes in decimal GB.
        refresh: Check current provider identities and replace selected daily data.

    Returns:
        Completed, skipped, incomplete, and remaining pending variable-day counts.

    Raises:
        ValueError: Selection, concurrency, budgets, paths, or scratch ownership
            are invalid, or required refresh inputs are unavailable.
        portalocker.exceptions.LockException: Another writer owns the daily archive
            or private work directory.
        Exception: Download, conversion, reduction, or publication fails. Owned
            hourly inputs remain available to resume the same selection.
    """
    names = validate_daily_selection(start, end, variables)
    if end == date.max - timedelta(days=1) and set(names) & INTERVAL_VARIABLES:
        raise ValueError("Daily totals must end before 9999-12-30 so the hourly workflow can include following midnight.")
    workers = (download_workers, processing_workers, upload_workers)
    budgets = (max_source_gb, max_decoded_gb)
    if (any(not isinstance(value, int) or value < 1 for value in workers)
            or any(not math.isfinite(value) or value <= 0 for value in budgets)):
        raise ValueError("Worker counts must be positive integers and source/decoded budgets must be finite and positive.")
    archive = archive.expanduser().resolve()
    work_dir = work_dir.expanduser().resolve()
    if archive.is_relative_to(work_dir) or work_dir.is_relative_to(archive):
        raise ValueError("Choose --work-dir outside and separate from the daily archive.")
    summary = dict(completed=0, skipped=0, incomplete=0,
                   pending=((end - start).days + 1) * len(names))
    work_dir.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(work_dir / ".daily-pull.lock", "a", timeout=0), archive_session(
        archive, resolution="daily",
    ) as output_db:
        cursor = start
        while cursor <= end:
            last = min(end, date(cursor.year, cursor.month, monthrange(cursor.year, cursor.month)[1]))
            days = [cursor + timedelta(days=offset) for offset in range((last - cursor).days + 1)]
            for variable in names:
                published = completed_daily_days(output_db, cursor, last, (variable,))
                needed = days if refresh else [day for day in days if (variable, day) not in published]
                scratch = _open_hourly_scratch(work_dir, archive, cursor, last, variable,
                                               create=bool(needed))
                if not needed:
                    summary["skipped"] += len(days)
                    summary["pending"] -= len(days)
                else:
                    hour_offset = 1 if variable in INTERVAL_VARIABLES else 0
                    selected_hours = {
                        (variable, int(datetime.combine(day, time.min, UTC).timestamp()) + hour * 3600)
                        for day in needed for hour in range(hour_offset, hour_offset + 24)
                    }
                    hourly_end = needed[-1] + timedelta(days=hour_offset)
                    pull_range(
                        needed[0], hourly_end, (variable,), scratch,
                        download_workers=download_workers, processing_workers=processing_workers,
                        upload_workers=upload_workers, max_source_gb=max_source_gb,
                        max_decoded_gb=max_decoded_gb, refresh=refresh,
                        selected_hours=selected_hours,
                    )
                    # Keep another hourly writer out while the reducer reads this
                    # private source. Publication/recovery has settled beforehand.
                    with archive_session(scratch):
                        result = aggregate_daily_session(
                            output_db, cursor, last, (variable,), scratch, archive,
                            refresh=refresh,
                        )
                    for key in ("completed", "skipped", "incomplete"):
                        summary[key] += result[key]
                        summary["pending"] -= result[key]
                    if result["incomplete"]:
                        print("Daily summary: " + ", ".join(f"{key}={value}" for key, value in summary.items()), flush=True)
                        return summary
                if scratch.exists():
                    # Recheck the exact owned, resolved target immediately before
                    # recursive cleanup; never remove the user-supplied work root.
                    scratch = _open_hourly_scratch(work_dir, archive, cursor, last,
                                                   variable, create=False)
                    shutil.rmtree(scratch)
                (work_dir / f".{scratch.name}.owner.json").unlink(missing_ok=True)
            cursor = last + timedelta(days=1)
    print("Daily summary: " + ", ".join(f"{key}={value}" for key, value in summary.items()), flush=True)
    return summary
