"""Exercise per-observation COG publication, recovery and writer ownership."""

from datetime import UTC, datetime, timedelta
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any

import numpy as np
import portalocker
import pytest
import rasterio
from rasterio.transform import from_origin

from era5_pull_pipeline import catalog
from era5_pull_pipeline.ncar import SourceFile
from era5_pull_pipeline.staging import get_source_path


def _stage(
    db: sqlite3.Connection,
    archive: Path,
    *,
    variable: str = "2_metre_temperature",
    hour: int = 0,
    value: float = 280,
    step_type: str = "instant",
) -> tuple[str, Path, Path, dict[str, Any]]:
    """Stage a real COG and source identity for publication boundary tests."""
    timestamp = datetime(1950, 1, 1, tzinfo=UTC) + timedelta(hours=hour)
    short_name = {"2_metre_temperature": "2t", "precipitation": "mtpr"}[variable]
    source = SourceFile(
        key=f"e5.oper.an.sfc/195001/{short_name}.nc", variable=short_name,
        size=12, etag='"test-source"', last_modified="2026-01-01T00:00:00+00:00",
        valid_start=datetime(1950, 1, 1, tzinfo=UTC),
        valid_end=datetime(1950, 1, 31, 23, tzinfo=UTC),
    )
    row = catalog.register_sources(db, archive, [source], timestamp.date(), timestamp.date())[0]
    source_path = get_source_path(archive, catalog.get_staging_dir(db), row)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"source bytes")
    with db:
        db.execute("UPDATE source_files SET downloaded_at = ? WHERE id = ?",
                   (datetime.now(UTC).isoformat(), row["id"]))
    second = int(timestamp.timestamp())
    field = {
        "variable": variable, "valid_time": second,
        "reference_time": None if step_type == "instant" else second - 3600,
        "interval_start": second if step_type == "instant" else second - 3600,
        "interval_end": second, "start_step": None if step_type == "instant" else 0,
        "end_step": None if step_type == "instant" else 3600,
        "step_units": "s", "step_type": step_type,
        "source_file_id": row["id"], "source_variable": short_name,
    }
    attributes = {"name": variable.replace("_", " "), "units": "K" if short_name == "2t" else "kg m-2 s-1"}
    publication, staged = catalog.prepare_observation(db, archive, field, attributes)
    transform = from_origin(-180.125, 90.125, 0.25, 0.25)
    with rasterio.open(
        staged, "w", driver="COG", width=2, height=2, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform,
        nodata=np.nan, compress="DEFLATE", blocksize=512,
    ) as raster:
        raster.write(np.full((2, 2), value, dtype=np.float32), 1)
    report = {
        "band": 1, "shape": [2, 2], "dtype": "float32", "nodata": "NaN",
        "crs": "EPSG:4326", "transform": list(transform.to_gdal()),
    }
    return publication, staged, source_path, report


def test_publication_exposes_generic_observation_metadata(tmp_path: Path) -> None:
    """Expose a relative COG path and UTC semantics only after publication."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        publication, staged, source, report = _stage(db, archive)
        assert catalog.get_published_hours(db, 1950, 1) == set()
        final = catalog.publish_observation(db, archive, publication, report)
        assert final == archive / "1950/01/2_metre_temperature/2_metre_temperature_19500101T000000Z.tif"
        assert not staged.exists() and source.exists()
        observation = db.execute("SELECT * FROM observations").fetchone()
        assert observation["path"] == final.relative_to(archive).as_posix()
        assert observation["units"] == "K"
        assert observation["temporal_type"] == "instant"
        assert observation["reference_time"] is None
        assert observation["interval_start"] == observation["interval_end"] == observation["valid_time"]
        assert observation["dtype"] == "float32" and observation["nodata"] == "NaN"
        assert observation["band"] == 1 and observation["crs"] == "EPSG:4326"
        assert json.loads(observation["transform"]) == report["transform"]
        assert catalog.get_published_hours(db, 1950, 1) == {("2t", observation["valid_time"])}
        assert catalog.get_published_hours(db, 1950, 2) == set()
        assert db.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 0
    assert (archive / "dataset.yaml").read_text() == catalog.DATASET_YAML
    with rasterio.open(final) as raster:
        np.testing.assert_array_equal(raster.read(1), 280)


@pytest.mark.parametrize(("step_type", "temporal_type"), [("avg", "mean"), ("accum", "accumulation")])
def test_interval_semantics_are_explicit(tmp_path: Path, step_type: str, temporal_type: str) -> None:
    """Describe mean and accumulated observations without interpreting filenames."""
    with catalog.archive_session(tmp_path) as db:
        publication, _, _, report = _stage(db, tmp_path, variable="precipitation", step_type=step_type)
        catalog.publish_observation(db, tmp_path, publication, report)
        observation = db.execute("SELECT * FROM observations").fetchone()
        assert observation["temporal_type"] == temporal_type
        assert observation["interval_end"] - observation["interval_start"] == 3600
        assert observation["reference_time"] == observation["interval_start"]


def test_refresh_changes_only_selected_file(tmp_path: Path) -> None:
    """Refresh one observation while preserving other hours and variables."""
    with catalog.archive_session(tmp_path) as db:
        first, _, _, report = _stage(db, tmp_path)
        first_path = catalog.publish_observation(db, tmp_path, first, report)
        later, _, _, report = _stage(db, tmp_path, hour=1)
        later_path = catalog.publish_observation(db, tmp_path, later, report)
        other, _, _, report = _stage(db, tmp_path, variable="precipitation", step_type="avg")
        other_path = catalog.publish_observation(db, tmp_path, other, report)
        untouched = {path: path.read_bytes() for path in (later_path, other_path)}
        replacement, _, _, report = _stage(db, tmp_path, value=281)
        assert catalog.publish_observation(db, tmp_path, replacement, report) == first_path
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 3
        assert all(path.read_bytes() == contents for path, contents in untouched.items())
        with rasterio.open(first_path) as raster:
            np.testing.assert_array_equal(raster.read(1), 281)


def test_incomplete_conversion_can_restart_without_hiding_previous_file(tmp_path: Path) -> None:
    """Discard the abandoned staged file for the same observation on retry."""
    with catalog.archive_session(tmp_path) as db:
        first, _, _, report = _stage(db, tmp_path)
        final = catalog.publish_observation(db, tmp_path, first, report)
        abandoned, old_stage, source, _ = _stage(db, tmp_path, value=281)
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
    with catalog.archive_session(tmp_path) as db:
        replacement, stage, _, report = _stage(db, tmp_path, value=282)
        assert not old_stage.exists() and stage.exists() and source.exists()
        assert replacement != abandoned
        assert db.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 1
        catalog.publish_observation(db, tmp_path, replacement, report)
        with rasterio.open(final) as raster:
            np.testing.assert_array_equal(raster.read(1), 282)


@pytest.mark.parametrize("after_install", [False, True])
def test_interrupted_publication_rolls_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_install: bool,
) -> None:
    """Recover both sides of the file-install-to-catalog-commit boundary."""
    install = catalog._install_observation

    def interrupt(source: Path, destination: Path) -> None:
        """Interrupt before or after replacing the complete file."""
        if after_install:
            install(source, destination)
        raise KeyboardInterrupt

    with catalog.archive_session(tmp_path) as db:
        first, _, _, report = _stage(db, tmp_path)
        final = catalog.publish_observation(db, tmp_path, first, report)
        replacement, staged, source, report = _stage(db, tmp_path, value=281)
        with monkeypatch.context() as patch:
            patch.setattr(catalog, "_install_observation", interrupt)
            with pytest.raises(KeyboardInterrupt):
                catalog.publish_observation(db, tmp_path, replacement, report)
        assert staged.exists() != after_install
        assert source.exists()
        assert db.execute("SELECT status FROM publications").fetchone()[0] == "ready"
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
    with catalog.archive_session(tmp_path) as db:
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 0
        with rasterio.open(final) as raster:
            np.testing.assert_array_equal(raster.read(1), 281)


def test_catalog_failure_retains_ready_journal_and_source(tmp_path: Path) -> None:
    """Recover an installed observation when its coverage transaction fails."""
    with catalog.archive_session(tmp_path) as db:
        publication, stage, source, report = _stage(db, tmp_path)
        db.execute("CREATE TEMP TRIGGER fail_coverage BEFORE INSERT ON coverage "
                   "BEGIN SELECT RAISE(ABORT, 'publication failed'); END")
        with pytest.raises(sqlite3.IntegrityError, match="publication failed"):
            catalog.publish_observation(db, tmp_path, publication, report)
        assert not stage.exists() and source.exists()
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
        assert db.execute("SELECT status FROM publications").fetchone()[0] == "ready"
        assert db.execute("SELECT COUNT(*) FROM source_selections").fetchone()[0] == 1
    with catalog.archive_session(tmp_path) as db:
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
        assert source.exists()


def test_relocation_preserves_relative_catalog_paths(tmp_path: Path) -> None:
    """Move the whole archive without rewriting output or staged source paths."""
    original = tmp_path / "original"
    with catalog.archive_session(original) as db:
        publication, _, source, report = _stage(db, original)
        final = catalog.publish_observation(db, original, publication, report)
        cog_relative = final.relative_to(original)
        source_relative = source.relative_to(original)
    relocated = tmp_path / "relocated"
    original.rename(relocated)
    with catalog.archive_session(relocated) as db:
        observation = db.execute("SELECT * FROM observations").fetchone()
        source_row = db.execute("SELECT * FROM source_files").fetchone()
        assert relocated / observation["path"] == relocated / cog_relative
        assert relocated / source_row["path"] == relocated / source_relative
        assert (relocated / observation["path"]).is_file()
        assert (relocated / source_row["path"]).is_file()


def test_only_one_writer_holds_archive(tmp_path: Path) -> None:
    """Reject a second writer while permitting reads of the catalog."""
    with catalog.archive_session(tmp_path) as db:
        with pytest.raises(portalocker.exceptions.LockException):
            with catalog.archive_session(tmp_path):
                pytest.fail("A second writer must not acquire the archive.")
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0


def test_incompatible_catalog_rejection_leaves_existing_archive_unchanged(tmp_path: Path) -> None:
    """Reject an incompatible catalog before changing schema, source files or metadata."""
    database = tmp_path / "catalog.sqlite"
    source = tmp_path / "pending.nc"
    source.write_bytes(b"retained NetCDF download")
    with closing(sqlite3.connect(database)) as db:
        db.executescript(
            "CREATE TABLE coverage (revision_id TEXT, variable TEXT, valid_time INTEGER);"
            "CREATE TABLE source_files (id TEXT, path TEXT);"
            "INSERT INTO coverage VALUES ('old-revision', '2_metre_temperature', 946684800);"
        )
        with db:
            db.execute("INSERT INTO source_files VALUES (?, ?)", ("source", str(source)))
    before = database.read_bytes()
    with pytest.raises(ValueError, match="not a COG catalog.*Use an existing COG archive or choose a new archive directory"):
        with catalog.archive_session(tmp_path):
            pytest.fail("The old archive must not be opened as a COG archive.")
    assert database.read_bytes() == before
    assert source.read_bytes() == b"retained NetCDF download"
    assert not (tmp_path / "dataset.yaml").exists()


def test_writer_lock_is_released_when_process_exits(tmp_path: Path) -> None:
    """Acquire writer ownership after a process exits without context cleanup."""
    code = (
        "import os, portalocker, sys; "
        "lock = portalocker.Lock(sys.argv[1], 'a', timeout=0); "
        "lock.acquire(); os._exit(0)"
    )
    subprocess.run([sys.executable, "-c", code, str(tmp_path / ".writer.lock")], check=True, timeout=20)
    with catalog.archive_session(tmp_path) as db:
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0


def test_register_sources_records_only_missing_variable_hours(tmp_path: Path) -> None:
    """Keep holes, source boundaries and other variables out of pending selections."""
    start = datetime(1950, 1, 1, tzinfo=UTC)
    source = SourceFile(
        key="e5.oper.an.sfc/195001/temperature.nc", variable="2t", size=12,
        etag='"version"', last_modified="2026-01-01T00:00:00+00:00",
        valid_start=start, valid_end=start + timedelta(hours=5),
    )
    midnight = int(start.timestamp())
    selected = {("2t", midnight + hour * 3600) for hour in (-1, 0, 1, 3, 5, 6)}
    selected.add(("10u", midnight + 4 * 3600))
    with catalog.archive_session(tmp_path) as db:
        catalog.register_sources(db, tmp_path, [source], start.date(), start.date(), selected_hours=selected)
        intervals = [tuple(row) for row in db.execute(
            "SELECT start_time, end_time FROM source_selections ORDER BY start_time"
        )]
        assert intervals == [
            ((start + timedelta(hours=first)).isoformat(), (start + timedelta(hours=last)).isoformat())
            for first, last in [(0, 1), (3, 3), (5, 5)]
        ]


@pytest.mark.parametrize("boundary", ["flush", "ready", "install", "coverage"])
def test_batch_publication_recovers_each_durability_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    """Retain all source inputs and recover mixed installed/staged batch files."""
    with catalog.archive_session(tmp_path) as db:
        batch = [_stage(db, tmp_path, hour=hour) for hour in range(3)]
        install = catalog._install_observation

        def interrupted_install(source: Path, destination: Path) -> None:
            """Leave one ready file staged while other parallel installs finish."""
            if source == batch[1][1]:
                raise OSError("batch interrupted")
            install(source, destination)

        def failed_flush(descriptor: int) -> None:
            """Simulate durable file flushing failing before any ready commit."""
            raise OSError("batch interrupted")

        with monkeypatch.context() as patch:
            if boundary == "install":
                patch.setattr(catalog, "_install_observation", interrupted_install)
            elif boundary == "flush":
                patch.setattr(catalog.os, "fsync", failed_flush)
            elif boundary == "ready":
                db.execute("CREATE TEMP TRIGGER fail_ready BEFORE UPDATE ON publications "
                           "WHEN NEW.valid_time = -631148400 "
                           "BEGIN SELECT RAISE(ABORT, 'batch interrupted'); END")
            else:
                db.execute("CREATE TEMP TRIGGER fail_coverage BEFORE INSERT ON coverage "
                           "WHEN NEW.valid_time = -631148400 "
                           "BEGIN SELECT RAISE(ABORT, 'batch interrupted'); END")
            with pytest.raises((OSError, sqlite3.IntegrityError), match="batch interrupted"):
                catalog.publish_observations(db, tmp_path, [(identifier, report) for identifier, _, _, report in batch])
        assert all(source.exists() for _, _, source, _ in batch)
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
        expected = "preparing" if boundary in {"flush", "ready"} else "ready"
        assert [row[0] for row in db.execute("SELECT status FROM publications")] == [expected] * 3
        if boundary == "install":
            assert 0 < sum(stage.exists() for _, stage, _, _ in batch) < len(batch)
    with catalog.archive_session(tmp_path) as db:
        if boundary in {"flush", "ready"}:
            catalog.publish_observations(db, tmp_path, [(identifier, report) for identifier, _, _, report in batch])
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 3
        assert db.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 0
        assert all(source.exists() for _, _, source, _ in batch)


def test_source_preparation_and_batch_publication_use_three_commits(tmp_path: Path) -> None:
    """Keep the durable transaction count independent of observations in a batch."""
    with catalog.archive_session(tmp_path) as db:
        original = [_stage(db, tmp_path, hour=hour) for hour in range(6)]
        rows = db.execute("SELECT field_json,attributes_json FROM publications ORDER BY valid_time").fetchall()
        fields = [json.loads(row["field_json"]) for row in rows]
        attributes = json.loads(rows[0]["attributes_json"])
        payloads = [staged.read_bytes() for _, staged, _, _ in original]
        statements: list[str] = []
        db.set_trace_callback(statements.append)
        prepared = catalog.prepare_observations(db, tmp_path, fields, attributes)
        assert statements.count("COMMIT") == 1
        for (_, staged), payload in zip(prepared, payloads, strict=True):
            staged.write_bytes(payload)
        catalog.publish_observations(db, tmp_path, [
            (identifier, original[index][3]) for index, (identifier, _) in enumerate(prepared)
        ])
        assert statements.count("COMMIT") == 3
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 6


@pytest.mark.parametrize("failed_commit", [1, 2])
def test_batch_commit_failure_keeps_durable_recovery_state(tmp_path: Path, failed_commit: int) -> None:
    """Roll back a denied ready or coverage commit without losing source inputs."""
    with catalog.archive_session(tmp_path) as db:
        batch = [_stage(db, tmp_path, hour=hour) for hour in range(3)]
        commits = 0

        def deny_commit(action: int, first: str | None, second: str | None,
                        database: str | None, trigger: str | None) -> int:
            """Reject the selected real SQLite COMMIT at the authorization boundary."""
            nonlocal commits
            if action == sqlite3.SQLITE_TRANSACTION and first == "COMMIT":
                commits += 1
                if commits == failed_commit:
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        db.set_authorizer(deny_commit)
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            catalog.publish_observations(db, tmp_path, [(identifier, report) for identifier, _, _, report in batch])
        db.set_authorizer(None)
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
        expected = "preparing" if failed_commit == 1 else "ready"
        assert [row[0] for row in db.execute("SELECT status FROM publications")] == [expected] * 3
        assert all(source.exists() for _, _, source, _ in batch)
    with catalog.archive_session(tmp_path) as db:
        if failed_commit == 1:
            catalog.publish_observations(db, tmp_path, [(identifier, report) for identifier, _, _, report in batch])
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 3


def test_later_batch_does_not_acknowledge_an_interrupted_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave failed ready entries for restart while acknowledging queued successes."""
    with catalog.archive_session(tmp_path) as db:
        first = _stage(db, tmp_path, hour=0)
        second = _stage(db, tmp_path, hour=1)

        def interrupted_install(source: Path, destination: Path) -> None:
            """Leave the first batch ready but uninstalled."""
            raise OSError("installation interrupted")

        with monkeypatch.context() as patch:
            patch.setattr(catalog, "_install_observation", interrupted_install)
            with pytest.raises(OSError, match="installation interrupted"):
                catalog.publish_observations(db, tmp_path, [(first[0], first[3])])
        catalog.publish_observations(db, tmp_path, [(second[0], second[3])])
        assert first[1].exists()
        assert db.execute("SELECT id FROM publications").fetchone()[0] == first[0]
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
    with catalog.archive_session(tmp_path) as db:
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 2
