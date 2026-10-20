"""Exercise publication recovery and locking with real SQLite and Zarr stores."""

from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any
from unittest.mock import Mock

import numpy as np
import portalocker
import pytest
import xarray as xr
import zarr

from era5_pull_pipeline import catalog


def _stage(
    db: sqlite3.Connection, archive: Path, month: int = 1
) -> tuple[str, Path, Path, dict[str, Any]]:
    """Build a small validated-store stand-in for catalog boundary tests.

    Args:
        db: Open writer catalog.
        archive: Archive receiving the store.
        month: Calendar month for the fixture.

    Returns:
        Revision identifier, staged store, source batch, and converter report.
    """
    batch = archive.parent / f"batch-{month}"
    batch.mkdir(exist_ok=True)
    (batch / "data.grib").write_bytes(b"source kept until publication")
    metadata = {
        "dataset": "reanalysis-era5-single-levels",
        "request": {"year": ["2000"], "month": [f"{month:02d}"]},
        "request_sha256": f"request-{month}",
        "sha256": "source-checksum",
    }
    revision, stage = catalog.prepare_revision(db, archive, batch, metadata)
    timestamp = int(datetime(2000, month, 1, tzinfo=UTC).timestamp())
    report = {
        "year": 2000,
        "month": month,
        "grid": {"latitude_size": 2, "longitude_size": 2},
        "layout": {
            "spatial_chunks": [2, 2],
            "time_chunk": "calendar_month",
            "shard_spatial_multiple": 2,
        },
        "variables": {
            "param_167": {"paramId": 167, "units": "K", "step_type": "instant"}
        },
        "fields": [{
            "variable": "param_167", "valid_time": timestamp,
            "reference_time": timestamp, "interval_start": timestamp,
            "interval_end": timestamp, "start_step": 0, "end_step": 0,
            "step_units": "s", "step_type": "instant",
        }],
        "validation": {"fields": 1, "roundtrip": True},
    }
    dataset = xr.Dataset(
        {"param_167": (("time", "latitude", "longitude"), np.full((1, 2, 2), 280.0))},
        coords={
            "time": np.array([f"2000-{month:02d}-01"], dtype="datetime64[ns]"),
            "latitude": [90.0, 89.75],
            "longitude": [0.0, 0.25],
        },
        attrs={"provenance": metadata},
    )
    dataset.to_zarr(stage, zarr_format=3, consolidated=False)
    return revision, stage, batch, report


def test_publication_records_coverage_then_removes_source(tmp_path: Path) -> None:
    """Publish exact field metadata and retain provenance after GRIB cleanup."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, stage, batch, report = _stage(db, archive)
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0
        assert (batch / "data.grib").exists()
        final = catalog.publish_revision(db, archive, revision, report)
        assert final == archive / "2000" / "01" / "era5.zarr"
        assert not stage.exists()
        assert not (batch / "data.grib").exists()
        row = db.execute("SELECT * FROM revisions WHERE id = ?", (revision,)).fetchone()
        assert row["status"] == "published"
        assert row["grib_removed"] == 1
        assert json.loads(row["provenance_json"])["sha256"] == "source-checksum"
        assert json.loads(row["report_json"]) == report
        field = dict(db.execute("SELECT * FROM coverage").fetchone())
        assert field == {"revision_id": revision, **report["fields"][0]}
        assert db.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert catalog.existing_month(db, archive, 2000, 1, "request-1") == final
        with pytest.raises(ValueError, match="different request"):
            catalog.existing_month(db, archive, 2000, 1, "another-request")
        with catalog.open_month(archive, 2000, 1) as dataset:
            np.testing.assert_array_equal(dataset["param_167"].values, 280.0)


def test_preparing_revision_is_never_readable(tmp_path: Path) -> None:
    """Leave incomplete conversion attempts without published coverage."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, stage, batch, _ = _stage(db, archive)
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (revision,)).fetchone()[0] == "preparing"
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0
        assert catalog.existing_month(db, archive, 2000, 1, "request-1") is None
    with pytest.raises(FileNotFoundError, match="No published"):
        with catalog.open_month(archive, 2000, 1):
            pytest.fail("Preparing data must not be exposed.")
    assert stage.exists()
    assert (batch / "data.grib").exists()


def test_ready_revision_recovers_after_rename_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry a ready publication whose filesystem rename did not complete."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, stage, batch, report = _stage(db, archive)
        with monkeypatch.context() as patch:
            patch.setattr(Path, "rename", Mock(side_effect=OSError("rename failed")))
            with pytest.raises(OSError, match="rename failed"):
                catalog.publish_revision(db, archive, revision, report)
        assert stage.exists()
        assert (batch / "data.grib").exists()
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (revision,)).fetchone()[0] == "ready"
    with catalog.archive_session(archive) as db:
        assert catalog.existing_month(db, archive, 2000, 1, "request-1").is_dir()
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 1
    assert not (batch / "data.grib").exists()


def test_ready_revision_recovers_after_catalog_failure(tmp_path: Path) -> None:
    """Recover a renamed store when the coverage transaction was rolled back."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, stage, batch, report = _stage(db, archive)
        db.execute(
            """CREATE TRIGGER fail_coverage BEFORE INSERT ON coverage
               BEGIN SELECT RAISE(ABORT, 'catalog failure'); END"""
        )
        db.commit()
        with pytest.raises(sqlite3.IntegrityError, match="catalog failure"):
            catalog.publish_revision(db, archive, revision, report)
        assert not stage.exists()
        assert (archive / "2000" / "01" / "era5.zarr").is_dir()
        assert (batch / "data.grib").exists()
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (revision,)).fetchone()[0] == "ready"
        db.execute("DROP TRIGGER fail_coverage")
        db.commit()
    # A reader also reconciles a ready store before exposing its arrays.
    with catalog.open_month(archive, 2000, 1) as dataset:
        np.testing.assert_array_equal(dataset["param_167"].values, 280.0)
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 1
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (revision,)).fetchone()[0] == "published"
    assert not (batch / "data.grib").exists()


def test_cleanup_failure_does_not_hide_published_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warn about retained GRIBs and retry cleanup without blocking readers."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, _, batch, report = _stage(db, archive)

        def deny_unlink(path: Path, missing_ok: bool = False) -> None:
            """Simulate a source file temporarily held by another process."""
            raise PermissionError("source is open")

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", deny_unlink)
            with pytest.warns(RuntimeWarning, match="cleanup remains pending"):
                catalog.publish_revision(db, archive, revision, report)
            assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 1
            assert db.execute("SELECT grib_removed FROM revisions").fetchone()[0] == 0
            with pytest.warns(RuntimeWarning, match="cleanup remains pending"):
                with catalog.open_month(archive, 2000, 1) as dataset:
                    np.testing.assert_array_equal(dataset["param_167"].values, 280.0)
        assert (batch / "data.grib").exists()
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT grib_removed FROM revisions").fetchone()[0] == 1
    assert not (batch / "data.grib").exists()


def test_recovery_rejects_unrelated_final_store(tmp_path: Path) -> None:
    """Never claim an existing final directory from a different source request."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, stage, batch, report = _stage(db, archive)
        final = archive / "2000" / "01" / "era5.zarr"
        final.parent.mkdir(parents=True)
        stage.rename(final)
        zarr.open_group(final, mode="r+").attrs["provenance"] = {
            "request_sha256": "unrelated"
        }
        with db:
            db.execute(
                "UPDATE revisions SET status = 'ready', report_json = ? WHERE id = ?",
                (json.dumps(report), revision),
            )
    with pytest.raises(ValueError, match="different request"):
        with catalog.archive_session(archive):
            pytest.fail("An unrelated store must not be published.")
    with sqlite3.connect(archive / "catalog.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0
    assert (batch / "data.grib").exists()


def test_layout_change_is_rejected_before_ready(tmp_path: Path) -> None:
    """Keep archive layout consistent without blocking the existing month."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, _, _, report = _stage(db, archive)
        catalog.publish_revision(db, archive, revision, report)
        revision, stage, batch, report = _stage(db, archive, month=2)
        report["layout"]["spatial_chunks"] = [1, 1]
        with pytest.raises(ValueError, match="layout"):
            catalog.publish_revision(db, archive, revision, report)
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (revision,)).fetchone()[0] == "preparing"
        assert stage.exists()
        assert (batch / "data.grib").exists()
        with catalog.open_month(archive, 2000, 1) as dataset:
            np.testing.assert_array_equal(dataset["param_167"].values, 280.0)


def test_reader_lock_allows_conversion_and_blocks_publication(tmp_path: Path) -> None:
    """Separate writer ownership from the lock held throughout lazy reading."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, _, _, report = _stage(db, archive)
        catalog.publish_revision(db, archive, revision, report)
        with pytest.raises(portalocker.exceptions.LockException):
            with catalog.archive_session(archive):
                pytest.fail("A second writer must not acquire the archive.")
        with catalog.open_month(archive, 2000, 1) as dataset:
            revision, stage, _, report = _stage(db, archive, month=2)
            assert stage.is_dir()
            np.testing.assert_array_equal(dataset["param_167"].values, 280.0)
            with pytest.raises(portalocker.exceptions.LockException):
                with portalocker.Lock(archive / ".publication.lock", "a", timeout=0):
                    pytest.fail("The reader must retain the publication lock.")
        assert catalog.publish_revision(db, archive, revision, report).is_dir()


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
