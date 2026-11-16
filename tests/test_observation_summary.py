"""Verify compact metadata against real publication, refresh and recovery."""

from collections import Counter
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import pytest

from era5_pull_pipeline import catalog, daily
from era5_pull_pipeline.conversion import write_observation_cog
from era5_pull_pipeline.ncar import SourceFile
from test_daily import publish_hours


def prepare_observation(
    db: sqlite3.Connection, archive: Path, valid_time: int, *,
    variable: str = "temperature", duration: int = 0, dtype: str = "float32",
    longitude_first: float = -180,
) -> tuple[str, dict[str, Any]]:
    """Stage a real COG with controllable metadata for summary boundary tests.

    Args:
        db: Archive writer connection.
        archive: Archive receiving the staged observation.
        valid_time: UTC Unix observation label, including negative timestamps.
        variable: Catalog variable name and fixture provider identity.
        duration: Preceding mean interval in seconds, or zero for an instant.
        dtype: Numeric type encoded in the actual COG.
        longitude_first: First column's longitude center.

    Returns:
        Publication ID and validated raster report for publish_observation.
    """
    timestamp = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=valid_time)
    source = SourceFile(
        key=f"fixtures/{variable}.nc", variable=variable, size=1,
        etag='"fixture"', last_modified="2026-01-01T00:00:00+00:00",
        valid_start=timestamp, valid_end=timestamp,
    )
    registered = catalog.register_sources(db, archive, [source], timestamp.date(), timestamp.date())[0]
    field = {
        "variable": variable, "source_variable": variable, "source_file_id": registered["id"],
        "valid_time": valid_time, "reference_time": None, "interval_start": valid_time - duration,
        "interval_end": valid_time, "start_step": None, "end_step": None,
        "step_units": "s", "step_type": "avg" if duration else "instant",
    }
    attributes = {"name": variable.title(), "units": "K"}
    identifier, staged = catalog.prepare_observation(db, archive, field, attributes)
    report = write_observation_cog(np.ones((3, 4), dtype=dtype), field, attributes, {
        "resolution": [90, 90], "latitude_first": 90, "longitude_first": longitude_first,
    }, staged)
    return identifier, report


def assert_summary_matches_observations(db: sqlite3.Connection) -> None:
    """Compare compact groups, counts and extrema with the complete public rows.

    Args:
        db: Connection to a quiescent archive whose summary is enabled.
    """
    keys = ("variable", "name", "units", "band", "dtype", "nodata", "temporal_type", "crs", "transform", "height", "width")

    def metadata(row: sqlite3.Row) -> tuple[Any, ...]:
        """Normalize JSON spacing when comparing the public metadata fields.

        Args:
            row: Full observation or compact summary row.

        Returns:
            Comparable public values with the transform decoded into a tuple.
        """
        return tuple(tuple(json.loads(row[key])) if key == "transform" else row[key] for key in keys)

    observations = db.execute("SELECT * FROM observations").fetchall()
    summary = db.execute("SELECT * FROM observation_summary").fetchall()
    assert Counter(metadata(row) for row in observations) == Counter({metadata(row): row["group_observation_count"] for row in summary})
    for row in summary:
        selected = [observation for observation in observations if observation["variable"] == row["variable"]]
        assert row["observation_count"] == len(selected)
        assert row["valid_time_min"] == min(item["valid_time"] for item in selected)
        assert row["valid_time_max"] == max(item["valid_time"] for item in selected)
        assert row["interval_start_min"] == min(item["interval_start"] for item in selected)
        assert row["interval_end_max"] == max(item["interval_end"] for item in selected)


def test_new_hourly_archive_summarizes_partial_pre1970_coverage(tmp_path: Path) -> None:
    """Automatically summarize additions with gaps and more than one variable."""
    first = int(datetime(1950, 1, 1, tzinfo=UTC).timestamp())
    with catalog.archive_session(tmp_path) as db:
        assert db.execute("SELECT enabled FROM observation_summary_state").fetchone()[0] == 1
        for variable, timestamp in [("temperature", first), ("temperature", first + 7200), ("pressure", first)]:
            catalog.publish_observation(db, tmp_path, *prepare_observation(db, tmp_path, timestamp, variable=variable))
        assert db.execute("SELECT COUNT(*) FROM observation_summary").fetchone()[0] == 2
        assert_summary_matches_observations(db)


def test_rebuild_existing_archive_is_explicit_repeatable_and_preserves_rasters(tmp_path: Path) -> None:
    """Leave older history unsummarized until requested and preserve all COGs."""
    with catalog.archive_session(tmp_path) as db:
        for timestamp in (0, 7200):
            catalog.publish_observation(db, tmp_path, *prepare_observation(db, tmp_path, timestamp))
        files = {tmp_path / row[0] for row in db.execute("SELECT path FROM observations")}
        before = {path: path.read_bytes() for path in files}
        with catalog.catalog_transaction(db):
            db.execute("DROP VIEW observation_summary")
            db.execute("DROP TABLE observation_summary_state")
            db.execute("DROP TABLE observation_metadata_groups")
            db.execute("DROP TABLE observation_bounds")
            db.execute("DROP INDEX coverage_variable_interval_start")
    with catalog.archive_session(tmp_path) as db:
        assert db.execute("SELECT enabled FROM observation_summary_state").fetchone()[0] == 0
        assert db.execute("SELECT * FROM observation_summary").fetchall() == []
        assert db.execute("SELECT 1 FROM sqlite_master WHERE name='coverage_variable_interval_start'").fetchone() is None
        catalog.publish_observation(db, tmp_path, *prepare_observation(db, tmp_path, 10800))
        assert db.execute("SELECT * FROM observation_summary").fetchall() == []
    for _ in range(2):
        assert catalog.rebuild_observation_summary(tmp_path) == {"observations": 3, "variables": 1, "groups": 1}
        with catalog.archive_session(tmp_path) as db:
            assert_summary_matches_observations(db)
    assert all(path.read_bytes() == value for path, value in before.items())
    with catalog.archive_session(tmp_path) as db:
        catalog.publish_observation(db, tmp_path, *prepare_observation(db, tmp_path, 14400))
        assert_summary_matches_observations(db)


def test_partial_grid_and_dtype_refresh_keeps_all_metadata_groups(tmp_path: Path) -> None:
    """Expose inconsistent intermediate metadata and remove exhausted old groups."""
    with catalog.archive_session(tmp_path) as db:
        for timestamp in (0, 3600):
            catalog.publish_observation(db, tmp_path, *prepare_observation(db, tmp_path, timestamp))
        catalog.publish_observation(db, tmp_path, *prepare_observation(db, tmp_path, 0, dtype="float64", longitude_first=-179.5))
        summary = db.execute("SELECT * FROM observation_summary").fetchall()
        assert len(summary) == 2
        assert {row["dtype"] for row in summary} == {"float32", "float64"}
        assert {row["group_observation_count"] for row in summary} == {1}
        assert {row["observation_count"] for row in summary} == {2}
        assert_summary_matches_observations(db)
    assert catalog.rebuild_observation_summary(tmp_path)["groups"] == 2
    with catalog.archive_session(tmp_path) as db:
        catalog.publish_observation(db, tmp_path, *prepare_observation(db, tmp_path, 3600, dtype="float64", longitude_first=-179.5))
        assert db.execute("SELECT COUNT(*) FROM observation_metadata_groups").fetchone()[0] == 1
        assert_summary_matches_observations(db)


def test_replacement_can_shrink_interval_support_without_changing_counts(tmp_path: Path) -> None:
    """Recompute the earliest physical interval after replacing its old minimum."""
    with catalog.archive_session(tmp_path) as db:
        for timestamp, duration in [(0, 10800), (3600, 3600)]:
            catalog.publish_observation(db, tmp_path, *prepare_observation(db, tmp_path, timestamp, duration=duration))
        assert db.execute("SELECT interval_start_min FROM observation_summary").fetchone()[0] == -10800
        catalog.publish_observation(db, tmp_path, *prepare_observation(db, tmp_path, 0, duration=3600))
        row = db.execute("SELECT * FROM observation_summary").fetchone()
        assert row["interval_start_min"] == -3600 and row["observation_count"] == 2
        assert_summary_matches_observations(db)
        for query in (
            "SELECT valid_time FROM coverage WHERE variable=? ORDER BY valid_time LIMIT 1",
            "SELECT valid_time FROM coverage WHERE variable=? ORDER BY valid_time DESC LIMIT 1",
            "SELECT interval_start FROM coverage WHERE variable=? ORDER BY interval_start LIMIT 1",
        ):
            plan = [row[3] for row in db.execute("EXPLAIN QUERY PLAN " + query, ("temperature",))]
            assert all("SCAN" not in step and "TEMP B-TREE" not in step for step in plan)
            assert any("SEARCH coverage USING COVERING INDEX" in step for step in plan)


@pytest.mark.parametrize("boundary", ["install", "coverage", "summary"])
def test_ready_publication_hides_summary_until_atomic_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    """Keep every summary hidden after failure, then recover exact current groups."""
    with catalog.archive_session(tmp_path) as db:
        for timestamp in (0, 3600):
            catalog.publish_observation(db, tmp_path, *prepare_observation(db, tmp_path, timestamp))
        identifier, report = prepare_observation(db, tmp_path, 0, dtype="float64")
        if boundary != "install":
            table = "coverage" if boundary == "coverage" else "observation_bounds"
            event = "INSERT" if boundary == "coverage" else "UPDATE"
            db.execute(f"CREATE TEMP TRIGGER fail_summary_test BEFORE {event} ON {table} BEGIN SELECT RAISE(ABORT,'publication interrupted'); END")

        def fail_install(source: Path, destination: Path) -> None:
            """Interrupt after the ready journal has committed."""
            raise OSError("publication interrupted")

        with monkeypatch.context() as patch:
            if boundary == "install":
                patch.setattr(catalog, "_install_observation", fail_install)
            with pytest.raises((OSError, sqlite3.IntegrityError), match="publication interrupted"):
                catalog.publish_observation(db, tmp_path, identifier, report)
        assert db.execute("SELECT * FROM observation_summary").fetchall() == []
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
        assert db.execute("SELECT SUM(observation_count) FROM observation_metadata_groups").fetchone()[0] == 2
    with catalog.archive_session(tmp_path) as db:
        assert db.execute("SELECT COUNT(*) FROM observation_summary").fetchone()[0] == 2
        assert_summary_matches_observations(db)


@pytest.mark.parametrize("invalid", ["reference_time", "interval", "grid", "dtype", "name"])
def test_failed_rebuild_leaves_summary_disabled(tmp_path: Path, invalid: str) -> None:
    """Reject invalid persisted metadata without advertising the previous summary."""
    with catalog.archive_session(tmp_path) as db:
        catalog.publish_observation(db, tmp_path, *prepare_observation(db, tmp_path, 0))
        with catalog.catalog_transaction(db):
            if invalid == "reference_time":
                db.execute("UPDATE coverage SET reference_time='not a timestamp'")
            elif invalid == "interval":
                db.execute("UPDATE coverage SET interval_start=-1")
            elif invalid == "grid":
                db.execute("UPDATE coverage SET raster_json=json_set(raster_json, '$.shape[0]', 0)")
            elif invalid == "dtype":
                db.execute("UPDATE coverage SET raster_json=json_set(raster_json, '$.dtype', 'bool')")
            else:
                db.execute("UPDATE variables SET attributes_json=json_set(attributes_json, '$.name', '')")
    with pytest.raises(ValueError):
        catalog.rebuild_observation_summary(tmp_path)
    with catalog.archive_session(tmp_path) as db:
        assert db.execute("SELECT enabled FROM observation_summary_state").fetchone()[0] == 0
        assert db.execute("SELECT * FROM observation_summary").fetchall() == []
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1


def test_daily_archive_summary_and_rebuild_preserve_its_identity(tmp_path: Path) -> None:
    """Use the same compact contract for derived daily means with nullable sources."""
    source, output = tmp_path / "hourly", tmp_path / "daily"
    first = datetime(1950, 1, 1, tzinfo=UTC)
    publish_hours(source, first, 24)
    daily.aggregate_daily(first.date(), first.date(), ["2t"], source, output)
    descriptor = (output / "dataset.yaml").read_bytes()
    assert catalog.rebuild_observation_summary(output) == {"observations": 1, "variables": 1, "groups": 1}
    with catalog.archive_session(output, resolution="daily") as db:
        assert_summary_matches_observations(db)
        row = db.execute("SELECT * FROM observation_summary").fetchone()
        assert row["temporal_type"] == "mean" and row["dtype"] == "float64"
        assert row["valid_time_min"] - row["interval_start_min"] == 86400
    assert (output / "dataset.yaml").read_bytes() == descriptor


def test_empty_archive_rebuild_and_missing_archive(tmp_path: Path) -> None:
    """Handle empty metadata without creating a new archive for a bad CLI path."""
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        catalog.rebuild_observation_summary(missing)
    assert not missing.exists()
    with catalog.archive_session(tmp_path):
        pass
    assert catalog.rebuild_observation_summary(tmp_path) == {"observations": 0, "variables": 0, "groups": 0}
