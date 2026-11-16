"""Publish complete UTC daily means and totals from hourly observation COGs."""

from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, date, datetime, time, timedelta
import json
from pathlib import Path, PurePath
import sqlite3
from typing import Any

import numpy as np
from numpy.typing import NDArray
import rasterio
from rasterio.transform import Affine

from era5_pull_pipeline.catalog import archive_session, prepare_observation, publish_observation
from era5_pull_pipeline.conversion import write_observation_cog

INTERVAL_VARIABLES = frozenset({"mtpr", "ssrd", "strd", "ssr", "str"})
MEAN_VARIABLES = frozenset({"2t", "2d", "sp", "10u", "10v"})


def validate_daily_selection(start: date, end: date, variables: Sequence[str]) -> tuple[str, ...]:
    """Validate inclusive UTC dates and return supported, unique NCAR names.

    Args:
        start: First UTC day to aggregate.
        end: Last UTC day to aggregate.
        variables: NCAR short names for daily means or totals.

    Returns:
        Sorted variable names with duplicates removed.

    Raises:
        ValueError: Dates are reversed, no variables are selected, or a name
            does not have a supported daily reduction.
    """
    if end < start:
        raise ValueError("The end date must be on or after the start date.")
    if end == date.max:
        raise ValueError("The last daily date must allow a following UTC midnight (before 9999-12-31).")
    selected = tuple(sorted(set(variables)))
    if not selected:
        raise ValueError("Choose at least one variable for daily aggregation.")
    unknown = set(selected) - MEAN_VARIABLES - INTERVAL_VARIABLES
    if unknown:
        raise ValueError(f"Daily aggregation does not support: {', '.join(sorted(unknown))}.")
    return selected


def completed_daily_days(
    db: sqlite3.Connection, start: date, end: date, variables: Sequence[str],
) -> set[tuple[str, date]]:
    """Return published variable-days in a daily archive for resume planning.

    Args:
        db: Connection held by a daily archive writer.
        start: First UTC day, inclusive.
        end: Last UTC day, inclusive.
        variables: Selected NCAR short names.

    Returns:
        Pairs of source short name and UTC interval-start date. Ready journal
        entries are excluded until their publication completes.
    """
    first = int(datetime.combine(start, time.min, UTC).timestamp()) + 86400
    last = int(datetime.combine(end, time.min, UTC).timestamp()) + 86400
    selected = set(variables)
    return {
        (row["source_variable"], (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=row["interval_start"])).date())
        for row in db.execute(
            "SELECT source_variable, interval_start FROM observations WHERE valid_time BETWEEN ? AND ?",
            (first, last),
        )
        if row["source_variable"] in selected
    }


def _get_hourly_source_path(archive: Path, relative: str) -> Path:
    """Confine an external catalog's relative observation path to its archive.

    Args:
        archive: Resolved source archive root.
        relative: Catalog path to an hourly COG.

    Returns:
        Resolved source path inside the archive.

    Raises:
        ValueError: An absolute path, parent traversal, or symlink escapes it.
    """
    path = Path(relative)
    resolved = (archive / path).resolve()
    if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(archive):
        raise ValueError(f"The hourly catalog contains a path outside its archive: {relative}")
    return resolved


def _read_only_catalog_uri(path: PurePath) -> str:
    """Encode a catalog path for SQLite, including Windows network shares.

    Args:
        path: Absolute catalog path already confined to its source archive.

    Returns:
        SQLite URI in read-only mode. UNC hosts remain part of the path rather
        than an authority that SQLite rejects.
    """
    uri = path.as_uri()
    if path.drive.startswith("\\\\"):
        uri = uri.replace("file://", "file:////", 1)
    return uri + "?mode=ro"


def _reduce_day(
    rows: list[sqlite3.Row], short_name: str, archive: Path,
) -> tuple[NDArray[np.float64], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reduce one complete day and describe its grid, units and provenance.

    Args:
        rows: Exactly 24 hourly public observations in UTC order.
        short_name: Supported NCAR variable name.
        archive: Resolved, read-only hourly archive root.

    Returns:
        Float64 values, daily field, variable attributes, and COG writer grid.

    Raises:
        ValueError: Hourly units, interval semantics, or grids are inconsistent
            or cannot be represented by the existing latitude/longitude writer.
        Exception: Raster and source-path errors propagate.
    """
    first = rows[0]
    interval = short_name in INTERVAL_VARIABLES
    expected_type = "mean" if short_name == "mtpr" else "accumulation" if interval else "instant"
    transform = Affine.from_gdal(*json.loads(first["transform"]))
    if (rasterio.crs.CRS.from_user_input(first["crs"]) != rasterio.crs.CRS.from_epsg(4326)
            or transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0):
        raise ValueError("Daily aggregation requires a regular north-up latitude/longitude grid.")
    unit = first["units"]
    if short_name == "mtpr" and unit not in {"kg m**-2 s**-1", "kg m-2 s-1", "kg m^-2 s^-1"}:
        raise ValueError(f"Hourly precipitation must use kg m-2 s-1, found {unit!r}.")
    if interval and short_name != "mtpr" and unit not in {
        "W m**-2 s", "W m-2 s", "J m**-2", "J m-2", "J m^-2",
    }:
        raise ValueError(f"Hourly radiation must use accumulated energy per area, found {unit!r}.")
    values = np.zeros((first["height"], first["width"]), dtype=np.float64)
    providers: dict[str, dict[str, Any]] = {}
    inputs = []
    for row in rows:
        valid = row["valid_time"]
        if (row["temporal_type"] != expected_type or row["interval_end"] != valid
                or row["interval_start"] != valid - (3600 if interval else 0)):
            raise ValueError(f"{short_name} at {valid} does not describe the required hourly input interval.")
        if any(row[key] != first[key] for key in ("variable", "name", "units", "crs", "transform", "height", "width")):
            raise ValueError(f"{short_name} hourly observations have inconsistent units, identity or grids.")
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open(_get_hourly_source_path(archive, row["path"])) as source:
            hourly = source.read(row["band"], out_dtype="float64", masked=True).filled(np.nan)
            if hourly.shape != values.shape:
                raise ValueError(f"{row['path']}: hourly raster shape differs from its catalog grid.")
            values += hourly
        inputs.append({key: row[key] for key in (
            "variable", "valid_time", "path", "source_file_id", "reference_time", "interval_start", "interval_end",
        )})
        providers[row["source_file_id"]] = {key: row[key] for key in (
            "source_variable", "object_key", "etag", "last_modified", "retrieved_at",
        )}
    if not interval:
        values /= 24
    elif short_name == "mtpr":
        values *= 3600
    start = first["valid_time"] - (3600 if interval else 0)
    variable = "total_precipitation" if short_name == "mtpr" else first["variable"]
    field = {
        "variable": variable, "source_variable": short_name, "source_file_id": None,
        "valid_time": start + 86400, "reference_time": None,
        "interval_start": start, "interval_end": start + 86400,
        "start_step": None, "end_step": None, "step_units": "s",
        "step_type": "accum" if interval else "avg",
        "derivation": {
            "method": "utc_daily_precipitation_integral" if short_name == "mtpr" else "utc_daily_sum" if interval else "utc_daily_mean",
            "providers": providers, "observations": inputs,
        },
    }
    attributes = {
        "name": "Daily total precipitation" if short_name == "mtpr" else f"Daily {'total' if interval else 'mean'} {first['name']}",
        "units": "mm" if short_name == "mtpr" else "J m**-2" if interval else unit,
        "source_variable": short_name,
        "step_type": field["step_type"], "step_units": "s",
    }
    grid = {
        "resolution": [-transform.e, transform.a],
        "latitude_first": transform.f + transform.e / 2,
        "longitude_first": transform.c + transform.a / 2,
    }
    return values, field, attributes, grid


def aggregate_daily_session(
    db: sqlite3.Connection, start: date, end: date, variables: Sequence[str],
    source_archive: Path, archive: Path, *, refresh: bool = False,
) -> dict[str, int]:
    """Publish requested UTC days while the caller holds the daily writer lock.

    Read at most one month of metadata for one variable and reduce one day at a
    time. Missing hours count as incomplete; missing pixels propagate as NaN.
    Hourly inputs remain unchanged. Their catalog must stay unchanged during
    this operation, following the archive update/refresh/calculate agreement.

    Args:
        db: Connection from archive_session(archive, resolution="daily").
        start: First UTC date, inclusive.
        end: Last UTC date, inclusive.
        variables: Supported NCAR short names.
        source_archive: Hourly archive containing a public observations view.
        archive: Separate destination archive held by the caller.
        refresh: Replace selected completed days using current hourly inputs.

    Returns:
        Counts of completed, skipped, and incomplete variable-days.

    Raises:
        ValueError: Selection, source paths, or hourly metadata are invalid, or
            source and output directories overlap.
        Exception: Catalog, raster, or recoverable publication errors propagate.
    """
    selected = validate_daily_selection(start, end, variables)
    source_archive, archive = source_archive.resolve(), archive.resolve()
    if source_archive.is_relative_to(archive) or archive.is_relative_to(source_archive):
        raise ValueError("Choose separate, non-overlapping hourly and daily archive directories.")
    catalog_path = _get_hourly_source_path(source_archive, "catalog.sqlite")
    completed = completed_daily_days(db, start, end, selected)
    summary = {"completed": 0, "skipped": 0, "incomplete": 0}
    with closing(sqlite3.connect(_read_only_catalog_uri(catalog_path), uri=True)) as source:
        source.row_factory = sqlite3.Row
        if source.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='archive_config'").fetchone():
            configured = source.execute("SELECT resolution FROM archive_config WHERE id=1").fetchone()
            if configured is not None and configured["resolution"] != "hourly":
                raise ValueError("Choose an hourly source archive for daily aggregation.")
        month = start
        while month <= end:
            stop = end + timedelta(days=1) if (month.year, month.month) == (end.year, end.month) else (
                month.replace(day=28) + timedelta(days=4)
            ).replace(day=1)
            for variable in selected:
                days = [month + timedelta(days=index) for index in range((stop - month).days)]
                pending = [day for day in days if refresh or (variable, day) not in completed]
                if not pending:
                    summary["skipped"] += len(days)
                    continue
                print(f"Aggregating {variable}: {month} through {stop - timedelta(days=1)} ({len(pending)} UTC days).", flush=True)
                offset = 3600 if variable in INTERVAL_VARIABLES else 0
                first = int(datetime.combine(month, time.min, UTC).timestamp()) + offset
                last = int(datetime.combine(stop, time.min, UTC).timestamp()) + offset
                rows = source.execute(
                    "SELECT * FROM observations WHERE source_variable=? AND valid_time>=? AND valid_time<? ORDER BY valid_time",
                    (variable, first, last),
                ).fetchall()
                by_time = {row["valid_time"]: row for row in rows}
                if len(by_time) != len(rows) or any(timestamp % 3600 for timestamp in by_time):
                    raise ValueError(f"The hourly catalog has duplicate or off-hour observations for {variable}.")
                day = month
                incomplete = []
                while day < stop:
                    if not refresh and (variable, day) in completed:
                        summary["skipped"] += 1
                    else:
                        midnight = int(datetime.combine(day, time.min, UTC).timestamp())
                        hours = range(midnight + offset, midnight + offset + 86400, 3600)
                        if any(hour not in by_time for hour in hours):
                            summary["incomplete"] += 1
                            incomplete.append(day.isoformat())
                        else:
                            values, field, attributes, grid = _reduce_day([by_time[hour] for hour in hours], variable, source_archive)
                            identifier, staged = prepare_observation(db, archive, field, attributes)
                            report = write_observation_cog(values, field, attributes, grid, staged)
                            publish_observation(db, archive, identifier, report)
                            summary["completed"] += 1
                    day += timedelta(days=1)
                if incomplete:
                    print(f"Incomplete {variable} UTC days (24 hourly inputs required): {', '.join(incomplete)}", flush=True)
            month = stop
    return summary


def aggregate_daily(
    start: date, end: date, variables: Sequence[str], source_archive: Path,
    archive: Path, *, refresh: bool = False,
) -> dict[str, int]:
    """Create or update a separate daily archive from complete hourly UTC days.

    Args:
        start: First UTC day, inclusive.
        end: Last UTC day, inclusive.
        variables: NCAR short names for daily means and totals.
        source_archive: Existing read-only hourly COG archive.
        archive: Separate daily output directory.
        refresh: Recompute selected completed variable-days.

    Returns:
        Counts of completed, skipped, and incomplete variable-days.

    Raises:
        ValueError: Selection or archive metadata is incompatible.
        Exception: Locking, source reading, or recoverable publication fails.
    """
    validate_daily_selection(start, end, variables)
    source_root, output_root = source_archive.resolve(), archive.resolve()
    if source_root.is_relative_to(output_root) or output_root.is_relative_to(source_root):
        raise ValueError("Choose separate, non-overlapping hourly and daily archive directories.")
    with archive_session(archive, resolution="daily") as db:
        summary = aggregate_daily_session(db, start, end, variables, source_archive, archive, refresh=refresh)
    print("Daily summary: " + ", ".join(f"{key}={value}" for key, value in summary.items()), flush=True)
    return summary
