"""Exercise NCAR source identity and requested-hour tracking in SQLite."""

from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from era5_pull_pipeline import catalog
from era5_pull_pipeline.ncar import SourceFile


@pytest.fixture
def source() -> SourceFile:
    """Return an analysis file containing every hour of January 2000."""
    return SourceFile(
        key=(
            "e5.oper.an.sfc/200001/"
            "e5.oper.an.sfc.128_167_2t.ll025sc.2000010100_2000013123.nc"
        ),
        variable="2t",
        size=3072000,
        etag='"object-version"',
        last_modified="2026-01-02T03:04:05+00:00",
        valid_start=datetime(2000, 1, 1, tzinfo=UTC),
        valid_end=datetime(2000, 1, 31, 23, tzinfo=UTC),
    )


def test_register_source_before_download(tmp_path: Path, source: SourceFile) -> None:
    """Persist source identity and the selected day without claiming publication."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        row = catalog.register_sources(
            db, archive, [source], date(2000, 1, 15), date(2000, 1, 15)
        )[0]
        assert row["object_key"] == source.key
        assert row["variable"] == source.variable
        assert row["size"] == source.size
        assert row["etag"] == source.etag
        assert row["last_modified"] == source.last_modified
        assert row["valid_start"] == "2000-01-01T00:00:00+00:00"
        assert row["valid_end"] == "2000-01-31T23:00:00+00:00"
        assert row["downloaded_at"] is None
        assert Path(row["path"]) == (
            Path("staging") / "ncar" / row["id"] / Path(source.key).name
        )
        assert not (archive / row["path"]).exists()
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0

    with catalog.archive_session(archive) as db:
        saved = db.execute("SELECT * FROM source_files").fetchone()
        assert dict(saved) == dict(row)
        selection = db.execute("SELECT * FROM source_selections").fetchone()
        assert dict(selection) == {
            "source_id": row["id"],
            "start_time": "2000-01-15T00:00:00+00:00",
            "end_time": "2000-01-15T23:00:00+00:00",
        }


def test_repeated_selections_reuse_source_and_keep_intersections(
    tmp_path: Path, source: SourceFile
) -> None:
    """Record selected hours across runs while downloading a source only once."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        first = catalog.register_sources(
            db, archive, [source], date(1999, 12, 31), date(2000, 1, 2)
        )[0]
        repeated = catalog.register_sources(
            db, archive, [source], date(1999, 12, 31), date(2000, 1, 2)
        )[0]
        later = catalog.register_sources(
            db, archive, [source], date(2000, 1, 31), date(2000, 2, 2)
        )[0]
        assert dict(first) == dict(repeated) == dict(later)
        assert db.execute("SELECT COUNT(*) FROM source_files").fetchone()[0] == 1
        selections = db.execute(
            "SELECT start_time, end_time FROM source_selections ORDER BY start_time"
        ).fetchall()
        assert [tuple(row) for row in selections] == [
            ("2000-01-01T00:00:00+00:00", "2000-01-02T23:00:00+00:00"),
            ("2000-01-31T00:00:00+00:00", "2000-01-31T23:00:00+00:00"),
        ]


def test_completed_source_registration_uses_recorded_state(
    tmp_path: Path, source: SourceFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse a completed download without reopening or inspecting its file."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        row = catalog.register_sources(
            db, archive, [source], date(2000, 1, 1), date(2000, 1, 1)
        )[0]
        path = archive / row["path"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"completed source")
        completed_at = datetime.now(UTC).isoformat()
        with db:
            db.execute(
                "UPDATE source_files SET downloaded_at = ? WHERE id = ?",
                (completed_at, row["id"]),
            )
        with monkeypatch.context() as patch:
            patch.setattr(Path, "stat", Mock(side_effect=AssertionError("unexpected stat")))
            patch.setattr(Path, "open", Mock(side_effect=AssertionError("unexpected open")))
            reused = catalog.register_sources(
                db, archive, [source], date(2000, 1, 2), date(2000, 1, 2)
            )[0]
        assert reused["id"] == row["id"]
        assert reused["path"] == row["path"]
        assert reused["downloaded_at"] == completed_at
        assert path.read_bytes() == b"completed source"
