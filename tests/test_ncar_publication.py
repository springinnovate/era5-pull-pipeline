"""Exercise NCAR selection consumption and source retention after publication."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3

import pytest

from era5_pull_pipeline import catalog
from era5_pull_pipeline.ncar import SourceFile


def _download(
    db: sqlite3.Connection, archive: Path, start: datetime, end: datetime,
) -> tuple[SourceFile, sqlite3.Row]:
    """Record a downloaded source with its inclusive selected hours."""
    source = SourceFile(
        key=f"e5.oper.fc.sfc.meanflux/{start:%Y%m}/test-mtpr.nc",
        variable="mtpr", size=12, etag='"source-version"',
        last_modified="2026-01-01T00:00:00+00:00", valid_start=start, valid_end=end,
    )
    row = catalog.register_sources(db, archive, [source], start.date(), end.date())[0]
    path = archive / row["path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"source bytes")
    retrieved_at = datetime.now(UTC).isoformat()
    with db:
        db.execute("UPDATE source_files SET downloaded_at = ?, retrieved_at = ? WHERE id = ?",
                   (retrieved_at, retrieved_at, row["id"]))
    return source, db.execute("SELECT * FROM source_files WHERE id = ?", (row["id"],)).fetchone()


def test_shared_forecast_source_survives_until_both_months_consumed(tmp_path: Path) -> None:
    """Keep a forecast file across a year boundary until its last selected hour."""
    start = datetime(1999, 12, 31, 23, tzinfo=UTC)
    end = datetime(2000, 1, 1, 1, tzinfo=UTC)
    with catalog.archive_session(tmp_path) as db:
        source, row = _download(db, tmp_path, start, end)
        path = tmp_path / row["path"]
        with db:
            catalog.remove_published_source_hours(db, [{
                "source_id": row["id"], "start_time": start.isoformat(), "end_time": start.isoformat(),
            }])
        catalog.delete_ingested_netcdf_files(db, tmp_path)
        assert path.exists()
        assert [tuple(value) for value in db.execute(
            "SELECT start_time, end_time FROM source_selections"
        )] == [("2000-01-01T00:00:00+00:00", end.isoformat())]
        with db:
            catalog.remove_published_source_hours(db, [{
                "source_id": row["id"], "start_time": (start + timedelta(hours=1)).isoformat(),
                "end_time": end.isoformat(),
            }])
        catalog.delete_ingested_netcdf_files(db, tmp_path)
        assert not path.exists()
        assert db.execute("SELECT COUNT(*) FROM source_selections").fetchone()[0] == 0
        assert db.execute("SELECT removed_at FROM source_files").fetchone()[0] is not None
        assert db.execute("SELECT retrieved_at FROM source_files").fetchone()[0] == row["retrieved_at"]
        repeated = catalog.register_sources(db, tmp_path, [source], start.date(), end.date())[0]
        assert repeated["id"] == row["id"]
        assert repeated["downloaded_at"] is None and repeated["removed_at"] is None


def test_consuming_middle_selection_retains_both_unpublished_sides(tmp_path: Path) -> None:
    """Complete one day while leaving earlier and later source selections pending."""
    start = datetime(2000, 1, 1, tzinfo=UTC)
    with catalog.archive_session(tmp_path) as db:
        _, row = _download(db, tmp_path, start, start + timedelta(hours=71))
        with db:
            catalog.remove_published_source_hours(db, [{
                "source_id": row["id"], "start_time": (start + timedelta(hours=24)).isoformat(),
                "end_time": (start + timedelta(hours=47)).isoformat(),
            }])
        assert [tuple(value) for value in db.execute(
            "SELECT start_time, end_time FROM source_selections ORDER BY start_time"
        )] == [
            (start.isoformat(), (start + timedelta(hours=23)).isoformat()),
            ((start + timedelta(hours=48)).isoformat(), (start + timedelta(hours=71)).isoformat()),
        ]
        catalog.delete_ingested_netcdf_files(db, tmp_path)
        assert (tmp_path / row["path"]).exists()


def test_cleanup_failure_retries_on_next_writer_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep source state after a busy-file error and retry deletion next run."""
    start = datetime(2000, 1, 1, tzinfo=UTC)
    unlink = Path.unlink
    with catalog.archive_session(tmp_path) as db:
        _, row = _download(db, tmp_path, start, start)
        path = tmp_path / row["path"]
        with db:
            catalog.remove_published_source_hours(db, [{
                "source_id": row["id"], "start_time": start.isoformat(), "end_time": start.isoformat(),
            }])

        def fail_source_unlink(target: Path, missing_ok: bool = False) -> None:
            """Simulate a temporarily busy NetCDF file after publication."""
            if target == path:
                raise PermissionError("source busy")
            unlink(target, missing_ok=missing_ok)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", fail_source_unlink)
            with pytest.warns(RuntimeWarning, match="source cleanup remains pending"):
                catalog.delete_ingested_netcdf_files(db, tmp_path)
        assert path.exists()
        assert db.execute("SELECT downloaded_at FROM source_files").fetchone()[0] is not None
    with catalog.archive_session(tmp_path) as db:
        assert not path.exists()
        assert db.execute("SELECT downloaded_at FROM source_files").fetchone()[0] is None
        assert db.execute("SELECT removed_at FROM source_files").fetchone()[0] is not None
        assert db.execute("SELECT retrieved_at FROM source_files").fetchone()[0] == row["retrieved_at"]
