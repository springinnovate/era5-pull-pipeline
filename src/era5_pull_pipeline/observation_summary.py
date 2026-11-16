"""Maintain compact metadata groups and time bounds for published observations.

The public ``observation_summary`` view contains one row for each distinct
combination of raster grid, band, numeric type, NoData and temporal type within
a variable. Names and units come from the current variable record. Different
valid descriptions remain separate groups so consumers can compare normalized
metadata and reject heterogeneous grids or variable descriptions.

Each row contains its positive group observation count and repeats the variable's
total observation count and time bounds. Group counts sum to that total. Time
bounds use integer UTC seconds since 1970-01-01; gaps and partial coverage are
valid. The summary certifies that every represented observation has a unique
variable/time identity, integer times and nullable integer reference time.
Instantaneous intervals equal their valid time; other intervals have positive
duration and end at their valid time. Thus the maximum interval end equals the
maximum valid time. These guarantees cover metadata, not another read of TIFFs;
the ordinary COG publication pipeline owns file validation.

The view is empty while summaries are disabled or uninitialized, and while any
ready publication is awaiting coverage recording. Consumers use ``observations``
in those states. Successful rebuilds establish complete group counts and bounds;
subsequent publications maintain both in the coverage transaction. The backing
tables are private implementation state, not additional consumer interfaces.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
import json
import math
import sqlite3
from typing import Any

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

SUMMARY_SCHEMA = """
CREATE TABLE IF NOT EXISTS observation_summary_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1))
);
CREATE TABLE IF NOT EXISTS observation_metadata_groups (
    variable TEXT NOT NULL REFERENCES variables(name),
    metadata_json TEXT NOT NULL,
    observation_count INTEGER NOT NULL CHECK (observation_count >= 0),
    PRIMARY KEY (variable, metadata_json)
);
CREATE TABLE IF NOT EXISTS observation_bounds (
    variable TEXT PRIMARY KEY REFERENCES variables(name),
    observation_count INTEGER NOT NULL CHECK (observation_count > 0),
    valid_time_min INTEGER NOT NULL,
    valid_time_max INTEGER NOT NULL,
    interval_start_min INTEGER NOT NULL,
    interval_end_max INTEGER NOT NULL
);
CREATE VIEW IF NOT EXISTS observation_summary AS
SELECT g.variable,
       json_extract(v.attributes_json, '$.name') AS name,
       json_extract(v.attributes_json, '$.units') AS units,
       json_extract(g.metadata_json, '$.band') AS band,
       json_extract(g.metadata_json, '$.dtype') AS dtype,
       json_extract(g.metadata_json, '$.nodata') AS nodata,
       json_extract(g.metadata_json, '$.temporal_type') AS temporal_type,
       json_extract(g.metadata_json, '$.crs') AS crs,
       json_extract(g.metadata_json, '$.transform') AS transform,
       json_extract(g.metadata_json, '$.height') AS height,
       json_extract(g.metadata_json, '$.width') AS width,
       g.observation_count AS group_observation_count,
       b.observation_count, b.valid_time_min, b.valid_time_max,
       b.interval_start_min, b.interval_end_max
FROM observation_metadata_groups g
JOIN observation_bounds b ON b.variable=g.variable
JOIN variables v ON v.name=g.variable
WHERE (SELECT enabled FROM observation_summary_state WHERE id=1)=1
  AND NOT EXISTS (SELECT 1 FROM publications WHERE status='ready');
"""

_GROUP_FIELDS = ("band", "dtype", "nodata", "temporal_type", "crs", "transform", "height", "width")
_TEMPORAL_TYPES = {"instant": "instant", "avg": "mean", "accum": "accumulation"}


def initialize_observation_summary(db: sqlite3.Connection) -> None:
    """Enable summaries for an empty archive, leaving existing history opt-in.

    Args:
        db: Writer connection inside the catalog initialization transaction.

    Raises:
        sqlite3.Error: Summary state or its interval index cannot be created.
    """
    db.execute(
        "INSERT OR IGNORE INTO observation_summary_state VALUES (1, NOT EXISTS (SELECT 1 FROM coverage LIMIT 1))"
    )
    if db.execute("SELECT enabled FROM observation_summary_state WHERE id=1").fetchone()[0]:
        db.execute("CREATE INDEX IF NOT EXISTS coverage_variable_interval_start ON coverage(variable, interval_start)")


def _encode_metadata_group(row: Mapping[str, Any] | sqlite3.Row) -> str:
    """Encode the observation fields whose distinct values require comparison.

    Args:
        row: Public observation metadata with a JSON-encoded GDAL transform.

    Returns:
        Stable JSON containing raster and time-support metadata. Variable names
        and units come from the catalog's current variable row instead.
    """
    metadata = {key: row[key] for key in _GROUP_FIELDS}
    metadata["transform"] = json.loads(metadata["transform"])
    return json.dumps(metadata, separators=(",", ":"))


@lru_cache(maxsize=128)
def _validate_metadata_group(metadata_json: str) -> None:
    """Validate one distinct raster description without reading its TIFF.

    Args:
        metadata_json: Encoded metadata group from the public observation fields.

    Raises:
        ValueError: Grid, numeric type, band, or time support is unsupported.
        Exception: CRS, transform, dtype, or NoData parsing fails.
    """
    metadata = json.loads(metadata_json)
    CRS.from_user_input(metadata["crs"])
    transform = Affine.from_gdal(*metadata["transform"])
    if (metadata["band"] != 1 or not isinstance(metadata["height"], int) or metadata["height"] <= 0
            or not isinstance(metadata["width"], int) or metadata["width"] <= 0
            or not all(math.isfinite(value) for value in transform)
            or transform.a <= 0 or transform.e >= 0 or transform.b != 0 or transform.d != 0):
        raise ValueError("Unsupported observation raster grid")
    if np.dtype(metadata["dtype"]).kind not in "iuf" or metadata["temporal_type"] not in _TEMPORAL_TYPES.values():
        raise ValueError("Unsupported observation variable metadata")
    if metadata["nodata"] is not None:
        float(metadata["nodata"])


def _validate_observation(row: Mapping[str, Any] | sqlite3.Row) -> str:
    """Validate one observation's metadata and time fields for summary admission.

    Args:
        row: Public observation fields, including variable description and times.

    Returns:
        Its validated metadata group key. Different valid groups are retained
        so consumers can reject heterogeneous grids or variable descriptions.

    Raises:
        ValueError: A name, unit, time or interval violates the public contract.
        Exception: Raster metadata parsing fails.
    """
    if any(not isinstance(row[key], str) or not row[key].strip() for key in ("variable", "name", "units")):
        raise ValueError(f"Unsupported variable metadata: {row['variable']}")
    metadata = _encode_metadata_group(row)
    _validate_metadata_group(metadata)
    valid, start, end, reference = (row[key] for key in ("valid_time", "interval_start", "interval_end", "reference_time"))
    if (not all(isinstance(value, int) for value in (valid, start, end))
            or (reference is not None and not isinstance(reference, int))
            or (row["temporal_type"] == "instant" and (start != valid or end != valid))
            or (row["temporal_type"] != "instant" and (start >= end or end != valid))):
        raise ValueError(f"Invalid time or interval for {row['variable']} at {valid}")
    return metadata


def _read_publication_metadata(publication: sqlite3.Row) -> dict[str, Any]:
    """Describe a validated publication using the public observation fields.

    Args:
        publication: Ready journal row containing field, variable and raster JSON.

    Returns:
        Metadata suitable for the same validation used during a full rebuild.
    """
    field = json.loads(publication["field_json"])
    attributes = json.loads(publication["attributes_json"])
    raster = json.loads(publication["raster_json"])
    return {
        **field, "name": attributes["name"], "units": attributes["units"],
        "band": raster["band"], "dtype": raster["dtype"], "nodata": raster["nodata"],
        "crs": raster["crs"], "transform": json.dumps(raster["transform"]),
        "height": raster["shape"][0], "width": raster["shape"][1],
        "temporal_type": _TEMPORAL_TYPES.get(field["step_type"]),
    }


def prepare_observation_summary_changes(
    db: sqlite3.Connection, publications: Sequence[sqlite3.Row],
) -> Counter[tuple[str, str]] | None:
    """Count metadata additions and replacements before coverage is changed.

    Args:
        db: Writer connection in the coverage publication transaction.
        publications: Ready journal rows about to replace their coverage keys.

    Returns:
        Count changes keyed by variable and metadata JSON, or None when disabled.
        Zero deltas retain the affected variable for time-bound replacement.

    Raises:
        Exception: Publication metadata validation or indexed catalog reads fail.
    """
    if not db.execute("SELECT enabled FROM observation_summary_state WHERE id=1").fetchone()[0]:
        return None
    rows = [_read_publication_metadata(publication) for publication in publications]
    selected: dict[str, list[int]] = {}
    for row in rows:
        selected.setdefault(row["variable"], []).append(row["valid_time"])
    previous: dict[tuple[str, int], sqlite3.Row] = {}
    for variable, times in selected.items():
        placeholders = ",".join("?" for _ in times)
        previous.update({
            (variable, old["valid_time"]): old
            for old in db.execute(
                f"SELECT valid_time, step_type, raster_json FROM coverage WHERE variable=? AND valid_time IN ({placeholders})",
                [variable, *times],
            )
        })
    changes: Counter[tuple[str, str]] = Counter()
    for row in rows:
        metadata = _validate_observation(row)
        changes[(row["variable"], metadata)] += 1
        old_row = previous.get((row["variable"], row["valid_time"]))
        if old_row is not None:
            raster = json.loads(old_row["raster_json"])
            old = {
                **raster, "transform": json.dumps(raster["transform"]),
                "height": raster["shape"][0], "width": raster["shape"][1],
                "temporal_type": _TEMPORAL_TYPES[old_row["step_type"]],
            }
            changes[(row["variable"], _encode_metadata_group(old))] -= 1
    return changes


def _apply_group_counts(db: sqlite3.Connection, changes: Counter[tuple[str, str]]) -> None:
    """Apply a batch of group-count changes in the caller's transaction.

    Args:
        db: Writer connection holding the publication or rebuild transaction.
        changes: Signed counts keyed by variable and metadata JSON.

    Raises:
        sqlite3.Error: Updating counts fails or would make a count negative.
    """
    for (variable, metadata), count in changes.items():
        if count > 0:
            db.execute(
                "INSERT INTO observation_metadata_groups VALUES (?, ?, ?) "
                "ON CONFLICT(variable,metadata_json) DO UPDATE SET observation_count=observation_count+excluded.observation_count",
                (variable, metadata, count),
            )
        elif count < 0:
            db.execute(
                "UPDATE observation_metadata_groups SET observation_count=observation_count+? WHERE variable=? AND metadata_json=?",
                (count, variable, metadata),
            )
            db.execute(
                "DELETE FROM observation_metadata_groups WHERE variable=? AND metadata_json=? AND observation_count=0",
                (variable, metadata),
            )


def apply_observation_summary_changes(
    db: sqlite3.Connection, changes: Counter[tuple[str, str]] | None,
) -> None:
    """Update compact groups and indexed time extrema after coverage changes.

    Args:
        db: Writer connection in the same transaction as the coverage upsert.
        changes: Prepared group deltas, or None for a disabled summary.

    Raises:
        sqlite3.Error: Updating groups, indexed bounds, or summary rows fails.
    """
    if changes is None:
        return
    _apply_group_counts(db, changes)
    count_changes: Counter[str] = Counter()
    for (variable, _), count in changes.items():
        count_changes[variable] += count
    for variable, count_change in count_changes.items():
        first = db.execute("SELECT valid_time FROM coverage WHERE variable=? ORDER BY valid_time LIMIT 1", (variable,)).fetchone()[0]
        last = db.execute("SELECT valid_time FROM coverage WHERE variable=? ORDER BY valid_time DESC LIMIT 1", (variable,)).fetchone()[0]
        support = db.execute("SELECT interval_start FROM coverage WHERE variable=? ORDER BY interval_start LIMIT 1", (variable,)).fetchone()[0]
        previous = db.execute("SELECT observation_count FROM observation_bounds WHERE variable=?", (variable,)).fetchone()
        count = (previous[0] if previous is not None else 0) + count_change
        db.execute(
            "INSERT INTO observation_bounds VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(variable) DO UPDATE SET "
            "observation_count=excluded.observation_count, valid_time_min=excluded.valid_time_min, "
            "valid_time_max=excluded.valid_time_max, interval_start_min=excluded.interval_start_min, interval_end_max=excluded.interval_end_max",
            (variable, count, first, last, support, last),
        )


def rebuild_observation_summary_rows(db: sqlite3.Connection) -> dict[str, int]:
    """Validate published history once and replace its compact metadata summary.

    Stream metadata in bounded batches without opening rasters. The caller
    disables the old summary before beginning this transaction, so interruption
    cannot leave an incomplete or invalid summary visible to consumers.

    Args:
        db: Writer connection inside a rebuild transaction, after ready recovery.

    Returns:
        Counts of summarized observations, variables, and distinct metadata groups.

    Raises:
        Exception: Metadata validation, index creation or catalog updates fail;
            the caller rolls back and leaves the summary disabled.
    """
    db.execute("DELETE FROM observation_metadata_groups")
    db.execute("DELETE FROM observation_bounds")
    db.execute("CREATE INDEX IF NOT EXISTS coverage_variable_interval_start ON coverage(variable, interval_start)")
    rows = db.execute("SELECT * FROM observations")
    bounds: dict[str, list[int]] = {}
    while batch := rows.fetchmany(512):
        groups: Counter[tuple[str, str]] = Counter()
        for row in batch:
            metadata = _validate_observation(row)
            variable = row["variable"]
            groups[(variable, metadata)] += 1
            if variable not in bounds:
                bounds[variable] = [0, row["valid_time"], row["valid_time"], row["interval_start"], row["interval_end"]]
            current = bounds[variable]
            current[0] += 1
            current[1] = min(current[1], row["valid_time"])
            current[2] = max(current[2], row["valid_time"])
            current[3] = min(current[3], row["interval_start"])
            current[4] = max(current[4], row["interval_end"])
        _apply_group_counts(db, groups)
    db.executemany("INSERT INTO observation_bounds VALUES (?, ?, ?, ?, ?, ?)", [(variable, *values) for variable, values in bounds.items()])
    db.execute("UPDATE observation_summary_state SET enabled=1 WHERE id=1")
    return {
        "observations": sum(values[0] for values in bounds.values()),
        "variables": len(bounds),
        "groups": db.execute("SELECT COUNT(*) FROM observation_metadata_groups").fetchone()[0],
    }
