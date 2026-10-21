"""Exercise NCAR publication recovery and locking with real SQLite and Zarr."""

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any
from uuid import uuid4
from unittest.mock import Mock

import numpy as np
import portalocker
import pytest
import xarray as xr
import zarr

from era5_pull_pipeline import catalog
from era5_pull_pipeline.ncar import SourceFile


def _stage(
    db: sqlite3.Connection,
    archive: Path,
    month: int = 1,
    variable: str = "2_metre_temperature",
    hours: tuple[int, ...] = (0,),
    year: int = 2000,
) -> tuple[str, Path, Path, dict[str, Any]]:
    """Stage a small monthly store and registered NCAR input for publication tests.

    Args:
        db: Open writer catalog.
        archive: Archive receiving the store.
        month: Calendar month for the fixture.
        variable: Variable to add.
        hours: Hours since the beginning of the month.
        year: Calendar year for the fixture.

    Returns:
        Revision identifier, staged store, source NetCDF path, and field report.
    """
    previous = catalog.current_month(db, archive, year, month)
    short_name = {"2_metre_temperature": "2t", "10_metre_u_wind_component": "10u"}[variable]
    dates = [datetime(year, month, 1, tzinfo=UTC) + timedelta(hours=hour) for hour in hours]
    source = SourceFile(
        key=f"e5.oper.an.sfc/{year:04d}{month:02d}/{short_name}-{uuid4().hex}.nc",
        variable=short_name, size=12, etag='"test-version"',
        last_modified="2026-01-01T00:00:00+00:00",
        valid_start=dates[0], valid_end=dates[-1],
    )
    row = catalog.register_sources(db, archive, [source], dates[0].date(), dates[-1].date())[0]
    source_path = Path(row["path"])
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"source bytes")
    with db:
        db.execute("UPDATE source_files SET downloaded_at = ? WHERE id = ?",
                   (datetime.now(UTC).isoformat(), row["id"]))
    metadata = {
        "dataset": "nsf-ncar-era5",
        "request_sha256": f"selection-{year}-{month}-{variable}-{hours}",
        "sources": [{"id": row["id"], "object_key": source.key, "etag": source.etag}],
        "source_selections": [{
            "source_id": row["id"], "start_time": dates[0].isoformat(), "end_time": dates[-1].isoformat(),
        }],
    }
    revision, stage = catalog.prepare_revision(db, archive, metadata, year=year, month=month)
    timestamps = [int(value.timestamp()) for value in dates]
    report = {
        "year": year, "month": month,
        "grid": {"latitude_size": 2, "longitude_size": 2},
        "layout": {"spatial_chunks": [2, 2], "time_chunk": "calendar_month", "shard_spatial_multiple": 2},
        "variables": {variable: {"short_name": short_name,
            "units": "K" if short_name == "2t" else "m s**-1", "step_type": "instant"}},
        "fields": [{
            "variable": variable, "valid_time": timestamp,
            "reference_time": timestamp, "interval_start": timestamp, "interval_end": timestamp,
            "start_step": 0, "end_step": 0, "step_units": "s", "step_type": "instant",
            "source_request_sha256": metadata["request_sha256"],
            "source_file_id": row["id"], "source_variable": short_name,
        } for timestamp in timestamps],
        "validation": {"fields": len(hours), "roundtrip": True},
    }
    dataset = xr.Dataset(
        {variable: (("time", "latitude", "longitude"),
            np.full((len(hours), 2, 2), 280.0 if short_name == "2t" else 3.0, dtype=np.float32))},
        coords={"time": np.array(timestamps, dtype="datetime64[s]"),
                "latitude": [90.0, 89.75], "longitude": [0.0, 0.25]},
    )
    if previous is not None:
        with xr.open_zarr(previous[0], consolidated=False, chunks=None) as old:
            dataset = dataset.combine_first(old.load())
        report["variables"] = {**previous[1]["variables"], **report["variables"]}
        report["fields"] = list({
            (field["variable"], field["valid_time"]): field
            for field in report["fields"] + previous[1]["fields"]
        }.values())
    dataset.attrs = {"provenance": metadata}
    dataset.to_zarr(stage, zarr_format=3, consolidated=False)
    return revision, stage, source_path, report


def test_publication_records_actual_coverage_and_removes_source(tmp_path: Path) -> None:
    """Record fields and source identities before removing the ingested file."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, stage, source, report = _stage(db, archive, hours=(0, 1))
        assert catalog.get_published_hours(db, 2000, 1) == set()
        final = catalog.publish_revision(db, archive, revision, report)
        assert final == archive / "2000" / "01" / "era5.zarr"
        assert not stage.exists() and not source.exists()
        expected = {("2t", field["valid_time"]) for field in report["fields"]}
        assert catalog.get_published_hours(db, 2000, 1) == expected
        assert catalog.current_month(db, archive, 2000, 1)[1]["fields"] == [
            {**field, "source_revision_id": revision} for field in report["fields"]
        ]
        assert db.execute("SELECT COUNT(*) FROM source_selections").fetchone()[0] == 0
    with catalog.open_month(archive, 2000, 1) as dataset:
        np.testing.assert_array_equal(dataset["2_metre_temperature"].values, 280.0)


def test_replacement_preserves_history_and_independent_coverage(tmp_path: Path) -> None:
    """Keep original field origins while adding another variable and hour."""
    archive = tmp_path / "archive"
    start = int(datetime(1950, 1, 1, tzinfo=UTC).timestamp())
    with catalog.archive_session(archive) as db:
        first, _, _, report = _stage(db, archive, year=1950, hours=(0, 1))
        catalog.publish_revision(db, archive, first, report)
        second, _, source, report = _stage(
            db, archive, year=1950, variable="10_metre_u_wind_component", hours=(1,),
        )
        catalog.publish_revision(db, archive, second, report)
        assert not source.exists()
        assert catalog.get_published_hours(db, 1950, 1) == {
            ("2t", start), ("2t", start + 3600), ("10u", start + 3600),
        }
        assert catalog.get_published_hours(db, 1950, 2) == set()
        current = catalog.current_month(db, archive, 1950, 1)[1]
        assert current["revision_id"] == second
        assert {f["source_revision_id"] for f in current["fields"] if f["source_variable"] == "2t"} == {first}
        assert db.execute("SELECT COUNT(*) FROM revisions WHERE status = 'published'").fetchone()[0] == 2


def test_preparing_revision_is_never_readable(tmp_path: Path) -> None:
    """Leave incomplete conversion attempts without published coverage."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, stage, source, report = _stage(db, archive)
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (revision,)).fetchone()[0] == "preparing"
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0
        assert catalog.current_month(db, archive, 2000, 1) is None
        timestamp = report["fields"][0]["valid_time"]
        assert catalog.get_published_hours(db, 2000, 1) == set()
    with pytest.raises(FileNotFoundError, match="No published"):
        with catalog.open_month(archive, 2000, 1):
            pytest.fail("Preparing data must not be exposed.")
    assert stage.exists()
    assert source.exists()


def test_restarting_conversion_removes_only_abandoned_same_month_store(tmp_path: Path) -> None:
    """Discard interrupted decoded arrays while retaining sources and other months."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        first, _, _, report = _stage(db, archive)
        catalog.publish_revision(db, archive, first, report)
        abandoned, old_stage, source, _ = _stage(db, archive, variable="10_metre_u_wind_component")
        scratch = old_stage.parent / ".decoded-interrupted"
        scratch.mkdir()
        (scratch / "values.bin").write_bytes(b"unfinished decoded source")
        other, other_stage, _, _ = _stage(db, archive, month=2)
        metadata = json.loads(db.execute(
            "SELECT provenance_json FROM revisions WHERE id = ?", (abandoned,),
        ).fetchone()[0])
        revision, stage = catalog.prepare_revision(db, archive, metadata, year=2000, month=1)
        assert revision != abandoned
        assert not old_stage.exists() and not scratch.exists()
        assert not stage.exists()
        assert db.execute("SELECT id FROM revisions WHERE id = ?", (abandoned,)).fetchone() is None
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (other,)).fetchone()[0] == "preparing"
        assert other_stage.exists() and source.exists()
        assert catalog.current_month(db, archive, 2000, 1)[1]["revision_id"] == first


@pytest.mark.parametrize("failure_at", [1, 2])
def test_rename_failure_restores_old_month(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: int
) -> None:
    """Keep the previous month readable after either publication rename fails."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        first, _, _, report = _stage(db, archive)
        catalog.publish_revision(db, archive, first, report)
        second, stage, source, report = _stage(db, archive, variable="10_metre_u_wind_component")
        rename = Path.rename
        calls = 0

        def fail_once(source: Path, destination: Path) -> Path:
            """Fail one rename while allowing the recovery moves."""
            nonlocal calls
            calls += 1
            if calls == failure_at:
                raise OSError("rename failed")
            return rename(source, destination)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "rename", fail_once)
            with pytest.raises(OSError, match="rename failed"):
                catalog.publish_revision(db, archive, second, report)
        assert stage.exists()
        assert source.exists()
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (second,)).fetchone()[0] == "preparing"
        assert db.execute("SELECT COUNT(*) FROM coverage WHERE revision_id = ?", (second,)).fetchone()[0] == 0
        assert catalog.current_month(db, archive, 2000, 1)[1]["revision_id"] == first
        timestamp = report["fields"][0]["valid_time"]
        assert catalog.get_published_hours(db, 2000, 1) == {("2t", timestamp)}
        with catalog.open_month(archive, 2000, 1) as dataset:
            assert set(dataset.data_vars) == {"2_metre_temperature"}
            np.testing.assert_array_equal(dataset["2_metre_temperature"].values, 280.0)
        catalog.publish_revision(db, archive, second, report)
        assert catalog.current_month(db, archive, 2000, 1)[1]["revision_id"] == second
    assert not source.exists()


def test_catalog_failure_restores_old_month(tmp_path: Path) -> None:
    """Roll back renamed stores when the catalog transaction cannot commit."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        first, _, _, report = _stage(db, archive)
        catalog.publish_revision(db, archive, first, report)
        second, stage, source, report = _stage(db, archive, variable="10_metre_u_wind_component")
        db.execute(
            """CREATE TRIGGER fail_month BEFORE UPDATE ON months
               BEGIN SELECT RAISE(ABORT, 'catalog failure'); END"""
        )
        db.commit()
        with pytest.raises(sqlite3.IntegrityError, match="catalog failure"):
            catalog.publish_revision(db, archive, second, report)
        assert stage.exists()
        assert source.exists()
        assert db.execute("SELECT COUNT(*) FROM coverage WHERE revision_id = ?", (second,)).fetchone()[0] == 0
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (second,)).fetchone()[0] == "preparing"
        with catalog.open_month(archive, 2000, 1) as dataset:
            assert set(dataset.data_vars) == {"2_metre_temperature"}
        assert catalog.current_month(db, archive, 2000, 1)[1]["revision_id"] == first
        db.execute("DROP TRIGGER fail_month")
        db.commit()
        catalog.publish_revision(db, archive, second, report)
    assert not source.exists()


@pytest.mark.parametrize("completed_renames", [0, 1, 2])
def test_interrupted_replacement_rolls_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, completed_renames: int
) -> None:
    """Recover before, between, and after the two filesystem rename boundaries."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        first, _, _, report = _stage(db, archive)
        final = catalog.publish_revision(db, archive, first, report)
        second, stage, source, report = _stage(db, archive, variable="10_metre_u_wind_component")
        rename = Path.rename
        calls = 0

        def interrupt(source: Path, destination: Path) -> Path:
            """Exit after the selected durable journal boundary."""
            nonlocal calls
            if completed_renames == 0:
                raise SystemExit("interrupted")
            result = rename(source, destination)
            calls += 1
            if calls == completed_renames:
                raise SystemExit("interrupted")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(Path, "rename", interrupt)
            with pytest.raises(SystemExit, match="interrupted"):
                catalog.publish_revision(db, archive, second, report)
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (second,)).fetchone()[0] == "ready"
        assert db.execute("SELECT current_revision_id FROM months").fetchone()[0] == first
        assert db.execute("SELECT COUNT(*) FROM coverage WHERE revision_id = ?", (second,)).fetchone()[0] == 0
        timestamp = report["fields"][0]["valid_time"]
        assert catalog.get_published_hours(db, 2000, 1) == {("2t", timestamp)}
        assert source.exists()
        assert stage.exists() == (completed_renames < 2)
        assert final.exists() == (completed_renames != 1)
    with catalog.open_month(archive, 2000, 1) as dataset:
        np.testing.assert_array_equal(dataset["2_metre_temperature"].values, 280.0)
        np.testing.assert_array_equal(dataset["10_metre_u_wind_component"].values, 3.0)
    with catalog.archive_session(archive) as db:
        assert catalog.current_month(db, archive, 2000, 1)[1]["revision_id"] == second
        assert db.execute("SELECT COUNT(*) FROM coverage WHERE revision_id = ?", (second,)).fetchone()[0] == 2
        assert db.execute("SELECT backup_path FROM revisions WHERE id = ?", (second,)).fetchone()[0] is None
        assert catalog.get_published_hours(db, 2000, 1) == {("2t", timestamp), ("10u", timestamp)}
    assert not source.exists()


def test_failed_restoration_remains_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain the ready journal if restoring the previous directory also fails."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        first, _, _, report = _stage(db, archive)
        catalog.publish_revision(db, archive, first, report)
        second, stage, source, report = _stage(db, archive, variable="10_metre_u_wind_component")
        rename = Path.rename
        calls = 0

        def fail_moves(source: Path, destination: Path) -> Path:
            """Fail the new-store move and the immediate old-store restoration."""
            nonlocal calls
            calls += 1
            if calls in (2, 3):
                raise OSError("filesystem unavailable")
            return rename(source, destination)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "rename", fail_moves)
            with pytest.raises(OSError, match="filesystem unavailable"):
                catalog.publish_revision(db, archive, second, report)
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (second,)).fetchone()[0] == "ready"
        backup = archive / db.execute("SELECT backup_path FROM revisions WHERE id = ?", (second,)).fetchone()[0]
        assert backup.is_dir()
        assert stage.is_dir()
        assert source.exists()
    with catalog.open_month(archive, 2000, 1) as dataset:
        assert set(dataset.data_vars) == {"2_metre_temperature", "10_metre_u_wind_component"}
    assert not source.exists()


def test_cleanup_failures_do_not_hide_published_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry retained backups and NetCDF sources without hiding a committed replacement."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        first, _, _, report = _stage(db, archive)
        catalog.publish_revision(db, archive, first, report)
        second, _, source, report = _stage(db, archive, variable="10_metre_u_wind_component")
        with monkeypatch.context() as patch:
            patch.setattr(catalog.shutil, "rmtree", Mock(side_effect=PermissionError("backup is open")))
            patch.setattr(Path, "unlink", Mock(side_effect=PermissionError("source is open")))
            with pytest.warns(RuntimeWarning, match="cleanup remains pending") as warnings:
                catalog.publish_revision(db, archive, second, report)
            assert len(warnings) == 2
            row = db.execute("SELECT * FROM revisions WHERE id = ?", (second,)).fetchone()
            backup = archive / row["backup_path"]
            assert backup.is_dir()
            assert source.exists()
            with pytest.warns(RuntimeWarning, match="cleanup remains pending"):
                with catalog.open_month(archive, 2000, 1) as dataset:
                    np.testing.assert_array_equal(dataset["10_metre_u_wind_component"].values, 3.0)
    with catalog.archive_session(archive) as db:
        row = db.execute("SELECT * FROM revisions WHERE id = ?", (second,)).fetchone()
        assert row["backup_path"] is None
    assert not backup.exists()
    assert not source.exists()


def test_recovery_rejects_unrelated_final_store(tmp_path: Path) -> None:
    """Never claim a final directory whose revision marker is unrelated."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, stage, source, report = _stage(db, archive)
        final = archive / "2000" / "01" / "era5.zarr"
        final.parent.mkdir(parents=True)
        stage.rename(final)
        zarr.open_group(final, mode="r+").attrs["revision_id"] = "unrelated"
        with db:
            db.execute(
                "UPDATE revisions SET status = 'ready', report_json = ? WHERE id = ?",
                (json.dumps(report), revision),
            )
    with pytest.raises(ValueError, match="different revision"):
        with catalog.archive_session(archive):
            pytest.fail("An unrelated store must not be published.")
    with sqlite3.connect(archive / "catalog.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0
    assert source.exists()


def test_layout_change_is_rejected_before_ready(tmp_path: Path) -> None:
    """Keep archive layout consistent without blocking the existing month."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, _, _, report = _stage(db, archive)
        catalog.publish_revision(db, archive, revision, report)
        revision, stage, source, report = _stage(db, archive, month=2)
        report["layout"]["spatial_chunks"] = [1, 1]
        with pytest.raises(ValueError, match="layout"):
            catalog.publish_revision(db, archive, revision, report)
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (revision,)).fetchone()[0] == "preparing"
        assert stage.exists()
        assert source.exists()
        with catalog.open_month(archive, 2000, 1) as dataset:
            np.testing.assert_array_equal(dataset["2_metre_temperature"].values, 280.0)


def test_reader_lock_allows_conversion_and_blocks_publication(tmp_path: Path) -> None:
    """Allow staged replacement while a reader holds the unchanged old month."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, _, _, report = _stage(db, archive)
        catalog.publish_revision(db, archive, revision, report)
        with pytest.raises(portalocker.exceptions.LockException):
            with catalog.archive_session(archive):
                pytest.fail("A second writer must not acquire the archive.")
        with catalog.open_month(archive, 2000, 1) as dataset:
            revision, stage, _, report = _stage(db, archive, variable="10_metre_u_wind_component")
            assert stage.is_dir()
            assert set(dataset.data_vars) == {"2_metre_temperature"}
            np.testing.assert_array_equal(dataset["2_metre_temperature"].values, 280.0)
            with pytest.raises(portalocker.exceptions.LockException):
                with portalocker.Lock(archive / ".publication.lock", "a", timeout=0):
                    pytest.fail("The reader must retain the publication lock.")
        catalog.publish_revision(db, archive, revision, report)
    with catalog.open_month(archive, 2000, 1) as dataset:
        assert set(dataset.data_vars) == {"2_metre_temperature", "10_metre_u_wind_component"}


def test_writer_lock_is_released_when_process_exits(tmp_path: Path) -> None:
    """Recover writer ownership after a process exits without context cleanup."""
    archive = tmp_path / "archive"
    archive.mkdir()
    code = (
        "import os, portalocker, sys; "
        "lock = portalocker.Lock(sys.argv[1], 'a', timeout=0); "
        "lock.acquire(); os._exit(0)"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(archive / ".writer.lock")],
        check=True,
        timeout=20,
    )
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0


def test_register_sources_records_only_missing_variable_hours(tmp_path: Path) -> None:
    """Keep holes, source boundaries, and other variables out of pending selections."""
    archive = tmp_path / "archive"
    start = datetime(1950, 1, 1, tzinfo=UTC)
    source = SourceFile(
        key="e5.oper.an.sfc/195001/temperature.nc", variable="2t", size=12,
        etag='"version"', last_modified="2026-01-01T00:00:00+00:00",
        valid_start=start, valid_end=start + timedelta(hours=5),
    )
    midnight = int(start.timestamp())
    selected = {("2t", midnight + hour * 3600) for hour in (-1, 0, 1, 3, 5, 6)}
    selected.add(("10u", midnight + 4 * 3600))
    with catalog.archive_session(archive) as db:
        rows = catalog.register_sources(
            db, archive, [source], start.date(), start.date(), selected_hours=selected,
        )
        intervals = [tuple(row) for row in db.execute(
            "SELECT start_time, end_time FROM source_selections ORDER BY start_time"
        )]
        assert len(rows) == 1
        assert intervals == [
            ((start + timedelta(hours=first)).isoformat(), (start + timedelta(hours=last)).isoformat())
            for first, last in [(0, 1), (3, 3), (5, 5)]
        ]
