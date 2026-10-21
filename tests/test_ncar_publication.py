"""Exercise NCAR field origins and source lifetime at the publication boundary."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import pytest
import zarr

from era5_pull_pipeline import catalog
from era5_pull_pipeline.ncar import SourceFile


def _stage_downloaded_source(
    db: sqlite3.Connection, archive: Path, start: datetime, end: datetime
) -> tuple[SourceFile, sqlite3.Row]:
    """Create a placeholder NetCDF download and record its selected hours.

    Args:
        db: Open writer catalog.
        archive: Archive receiving the source.
        start: First selected and available UTC hour.
        end: Last selected and available UTC hour.

    Returns:
        Provider identity and its completed catalog row.
    """
    source = SourceFile(
        key=f"e5.oper.fc.sfc.meanflux/{start:%Y%m}/test-mtpr.nc",
        variable="mtpr", size=12, etag='"source-version"',
        last_modified="2026-01-01T00:00:00+00:00",
        valid_start=start, valid_end=end,
    )
    row = catalog.register_sources(db, archive, [source], start.date(), end.date())[0]
    path = Path(row["path"])
    path.parent.mkdir(parents=True)
    path.write_bytes(b"source bytes")
    with db:
        db.execute(
            "UPDATE source_files SET downloaded_at = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), row["id"]),
        )
    return source, db.execute("SELECT * FROM source_files WHERE id = ?", (row["id"],)).fetchone()


def _stage_month(
    db: sqlite3.Connection, archive: Path, source: sqlite3.Row,
    start: datetime, end: datetime,
) -> tuple[str, Path, dict[str, Any]]:
    """Create a tiny real store and report for one selected monthly interval.

    Args:
        db: Open writer catalog.
        archive: Archive receiving the monthly publication.
        source: Completed source row supplying each selected field.
        start: First selected valid UTC hour.
        end: Last selected valid UTC hour in the same calendar month.

    Returns:
        Preparing revision identifier, staged path, and field report.
    """
    variable = "mean_total_precipitation_rate"
    timestamps = list(range(int(start.timestamp()), int(end.timestamp()) + 1, 3600))
    metadata = {
        "dataset": "NCAR d633000",
        "request_sha256": f"selection-{start.isoformat()}-{end.isoformat()}",
        "sources": [{"id": source["id"], "object_key": source["object_key"], "etag": source["etag"]}],
        "source_selections": [{
            "source_id": source["id"], "start_time": start.isoformat(), "end_time": end.isoformat(),
        }],
    }
    revision, stage = catalog.prepare_revision(
        db, archive, [], metadata, year=start.year, month=start.month,
    )
    group = zarr.open_group(stage, mode="w", zarr_format=3)
    group.attrs["provenance"] = metadata
    group.create_array(variable, data=np.ones((len(timestamps), 1, 1), dtype=np.float32))
    report = {
        "year": start.year, "month": start.month,
        "grid": {"shape": [1, 1]}, "layout": {"spatial_chunks": [1, 1]},
        "variables": {variable: {"name": "Mean total precipitation rate", "units": "kg m-2 s-1"}},
        "fields": [{
            "variable": variable, "valid_time": timestamp,
            "reference_time": timestamp - 3600,
            "interval_start": timestamp - 3600, "interval_end": timestamp,
            "start_step": 0, "end_step": 3600, "step_units": "s", "step_type": "avg",
            "source_request_sha256": "source-snapshot-sha256",
            "source_file_id": source["id"], "source_variable": source["variable"],
        } for timestamp in timestamps],
    }
    return revision, stage, report


def test_shared_forecast_source_survives_until_both_months_publish(tmp_path: Path) -> None:
    """Keep one forecast file across a year boundary, then retain its provenance."""
    archive = tmp_path / "archive"
    start = datetime(1999, 12, 31, 23, tzinfo=UTC)
    end = datetime(2000, 1, 1, 1, tzinfo=UTC)
    with catalog.archive_session(archive) as db:
        source, row = _stage_downloaded_source(db, archive, start, end)
        path = Path(row["path"])
        first, _, report = _stage_month(db, archive, row, start, start)
        catalog.publish_revision(db, archive, first, report)
        assert path.exists()
        assert [tuple(value) for value in db.execute(
            "SELECT start_time, end_time FROM source_selections"
        )] == [("2000-01-01T00:00:00+00:00", end.isoformat())]
        field = catalog.current_month(db, archive, 1999, 12)[1]["fields"][0]
        assert field["source_file_id"] == row["id"]
        assert field["source_variable"] == "mtpr"
        second, _, report = _stage_month(db, archive, row, start + timedelta(hours=1), end)
        catalog.publish_revision(db, archive, second, report)
        assert not path.exists()
        assert db.execute("SELECT COUNT(*) FROM source_selections").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM field_sources").fetchone()[0] == 3
        repeated = catalog.register_sources(db, archive, [source], start.date(), end.date())[0]
        assert repeated["id"] == row["id"]
        assert repeated["downloaded_at"] is None


def test_publishing_middle_selection_retains_both_unpublished_sides(tmp_path: Path) -> None:
    """Publishing one day leaves earlier and later selected hours pending."""
    archive = tmp_path / "archive"
    start = datetime(2000, 1, 1, tzinfo=UTC)
    with catalog.archive_session(archive) as db:
        _, row = _stage_downloaded_source(db, archive, start, start + timedelta(hours=71))
        revision, _, report = _stage_month(
            db, archive, row, start + timedelta(hours=24), start + timedelta(hours=47),
        )
        catalog.publish_revision(db, archive, revision, report)
        assert [tuple(value) for value in db.execute(
            "SELECT start_time, end_time FROM source_selections ORDER BY start_time"
        )] == [
            (start.isoformat(), (start + timedelta(hours=23)).isoformat()),
            ((start + timedelta(hours=48)).isoformat(), (start + timedelta(hours=71)).isoformat()),
        ]
        assert Path(row["path"]).exists()
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 24


def test_failed_publication_does_not_consume_selections(tmp_path: Path) -> None:
    """Roll source consumption back with coverage when catalog publication fails."""
    archive = tmp_path / "archive"
    start = datetime(2000, 1, 1, tzinfo=UTC)
    with catalog.archive_session(archive) as db:
        _, row = _stage_downloaded_source(db, archive, start, start)
        revision, stage, report = _stage_month(db, archive, row, start, start)
        db.execute(
            "CREATE TEMP TRIGGER fail_month BEFORE INSERT ON months "
            "BEGIN SELECT RAISE(ABORT, 'publication failed'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="publication failed"):
            catalog.publish_revision(db, archive, revision, report)
        assert stage.exists()
        assert Path(row["path"]).exists()
        assert db.execute("SELECT COUNT(*) FROM source_selections").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM field_sources").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0
        db.execute("DROP TRIGGER fail_month")
        catalog.publish_revision(db, archive, revision, report)
        assert not Path(row["path"]).exists()


def test_interrupted_publication_recovers_source_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover the rename-to-catalog gap before removing the last source input."""
    archive = tmp_path / "archive"
    start = datetime(2000, 1, 1, tzinfo=UTC)
    rename = catalog._rename_store

    def interrupt_after_rename(source: Path, destination: Path) -> None:
        """Interrupt immediately after the new monthly directory is visible."""
        rename(source, destination)
        raise KeyboardInterrupt

    with catalog.archive_session(archive) as db:
        _, row = _stage_downloaded_source(db, archive, start, start)
        revision, _, report = _stage_month(db, archive, row, start, start)
        with monkeypatch.context() as patch:
            patch.setattr(catalog, "_rename_store", interrupt_after_rename)
            with pytest.raises(KeyboardInterrupt):
                catalog.publish_revision(db, archive, revision, report)
        assert Path(row["path"]).exists()
        assert db.execute("SELECT COUNT(*) FROM source_selections").fetchone()[0] == 1
    with catalog.archive_session(archive) as db:
        assert catalog.current_month(db, archive, 2000, 1)[1]["fields"][0]["source_file_id"] == row["id"]
        assert not Path(row["path"]).exists()
        assert db.execute("SELECT COUNT(*) FROM source_selections").fetchone()[0] == 0


def test_source_cleanup_failure_retries_without_losing_published_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep completed download state until an interrupted source cleanup succeeds."""
    archive = tmp_path / "archive"
    start = datetime(2000, 1, 1, tzinfo=UTC)
    unlink = Path.unlink
    with catalog.archive_session(archive) as db:
        _, row = _stage_downloaded_source(db, archive, start, start)
        path = Path(row["path"])
        revision, _, report = _stage_month(db, archive, row, start, start)

        def fail_source_unlink(target: Path, missing_ok: bool = False) -> None:
            """Simulate a temporarily busy NetCDF file after successful publication."""
            if target == path:
                raise PermissionError("source busy")
            unlink(target, missing_ok=missing_ok)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", fail_source_unlink)
            with pytest.warns(RuntimeWarning, match="source cleanup remains pending"):
                catalog.publish_revision(db, archive, revision, report)
        assert path.exists()
        assert catalog.current_month(db, archive, 2000, 1) is not None
        assert db.execute("SELECT downloaded_at FROM source_files").fetchone()[0] is not None
    with catalog.archive_session(archive) as db:
        assert not path.exists()
        assert db.execute("SELECT downloaded_at FROM source_files").fetchone()[0] is None
