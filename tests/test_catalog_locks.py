"""Exercise reader contention at SQLite's durable publication boundaries."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pytest

from era5_pull_pipeline import catalog
from era5_pull_pipeline.staging import copy_local_cogs
from test_catalog import _stage


@pytest.mark.parametrize("boundary", ["ready", "coverage"])
@pytest.mark.parametrize("local_staging", [False, True])
def test_reader_releases_blocked_publication_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str, local_staging: bool,
) -> None:
    """Publish once after an independent reader releases either durable commit."""
    archive = tmp_path / "archive"
    local = tmp_path / "local" if local_staging else None
    delays: list[float] = []
    installed: list[Path] = []
    with catalog.archive_session(archive, staging_dir=local) as db, closing(
        sqlite3.connect(archive / "catalog.sqlite", timeout=.001)
    ) as reader:
        db.execute("PRAGMA busy_timeout=1")
        batch = [_stage(db, archive, hour=hour) for hour in range(3)]
        if local is not None:
            copy_local_cogs(local, archive, [stage for _, stage, _, _ in batch])
        install = catalog._install_observation

        def hold_reader() -> None:
            """Retain a real shared lock after finishing the reader's SELECT."""
            reader.execute("BEGIN")
            reader.execute("SELECT status FROM publications").fetchall()

        def record_install(source: Path, destination: Path) -> None:
            """Record completed file work without sharing a SQLite connection."""
            install(source, destination)
            installed.append(destination)

        def release_reader(delay: float) -> None:
            """Release contention only after COMMIT has actually returned busy."""
            delays.append(delay)
            assert db.in_transaction and reader.in_transaction
            assert all(source.exists() for _, _, source, _ in batch)
            assert reader.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
            if boundary == "ready":
                assert not installed
                assert all(stage.exists() for _, stage, _, _ in batch)
            else:
                assert len(installed) == len(batch)
                assert all(path.exists() for path in installed)
                if local is not None:
                    assert all(not stage.exists() for _, stage, _, _ in batch)
            reader.rollback()

        monkeypatch.setattr(catalog, "_install_observation", record_install)
        monkeypatch.setattr(catalog, "sleep", release_reader)
        if boundary == "ready":
            hold_reader()
        statements: list[str] = []

        def trace_transaction(sql: str) -> None:
            """Acquire the independent reader on its owner thread before coverage."""
            statements.append(sql)
            if boundary == "coverage" and sql.startswith("BEGIN") and len(installed) == len(batch):
                hold_reader()

        db.set_trace_callback(trace_transaction)
        result = catalog.publish_observations(
            db, archive, [(identifier, report) for identifier, _, _, report in batch],
            files_flushed=local is not None,
        )
        assert delays == [1]
        assert set(result) == set(installed) and len(installed) == len(batch)
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 3
        assert db.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM local_files WHERE kind='cog'").fetchone()[0] == 0
        assert all(source.exists() for _, _, source, _ in batch)
        assert not db.in_transaction
        assert sum("UPDATE publications SET status='ready'" in sql for sql in statements) == 3
        assert statements.count("COMMIT") == 3  # Two successful boundaries, one blocked attempt.
        assert "ROLLBACK" not in statements


@pytest.mark.parametrize("boundary", ["ready", "coverage"])
def test_reader_outlasts_retries_and_publication_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    """Keep journaled local outputs and inputs recoverable after bounded retries."""
    archive, local = tmp_path / "archive", tmp_path / "local"
    delays: list[float] = []
    installed: list[Path] = []
    with catalog.archive_session(archive, staging_dir=local) as db, closing(
        sqlite3.connect(archive / "catalog.sqlite", timeout=.001)
    ) as reader:
        db.execute("PRAGMA busy_timeout=1")
        batch = [_stage(db, archive, hour=hour) for hour in range(3)]
        payloads = [stage.read_bytes() for _, stage, _, _ in batch]
        fields = [json.loads(row[0]) for row in db.execute(
            "SELECT field_json FROM publications ORDER BY valid_time"
        )]
        attributes = json.loads(db.execute("SELECT attributes_json FROM publications LIMIT 1").fetchone()[0])
        copy_local_cogs(local, archive, [stage for _, stage, _, _ in batch])
        install = catalog._install_observation

        def hold_reader() -> None:
            """Keep a separate read transaction open through the retry budget."""
            reader.execute("BEGIN")
            reader.execute("SELECT status FROM publications").fetchall()

        def record_install(source: Path, destination: Path) -> None:
            """Record file installation while leaving SQLite on its owner thread."""
            install(source, destination)
            installed.append(destination)

        def trace_transaction(sql: str) -> None:
            """Block coverage after the ready journal and files are durable."""
            if boundary == "coverage" and sql.startswith("BEGIN") and len(installed) == len(batch):
                hold_reader()

        def keep_reader(delay: float) -> None:
            """Observe bounded backoff while keeping the real reader lock held."""
            delays.append(delay)
            assert db.in_transaction and reader.in_transaction
            assert all(source.exists() for _, _, source, _ in batch)

        with monkeypatch.context() as patch:
            patch.setattr(catalog, "_install_observation", record_install)
            patch.setattr(catalog, "sleep", keep_reader)
            if boundary == "ready":
                hold_reader()
            db.set_trace_callback(trace_transaction)
            with pytest.raises(sqlite3.OperationalError, match="database is locked") as caught:
                catalog.publish_observations(
                    db, archive, [(identifier, report) for identifier, _, _, report in batch],
                    files_flushed=True,
                )
        assert caught.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
        assert delays == [1, 2, 4] and not db.in_transaction
        reader.rollback()
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
        expected = "preparing" if boundary == "ready" else "ready"
        assert [row[0] for row in db.execute("SELECT status FROM publications")] == [expected] * 3
        assert all(source.exists() for _, _, source, _ in batch)
        assert len(installed) == (0 if boundary == "ready" else 3)

    with catalog.archive_session(archive) as db:
        if boundary == "ready":
            # Startup discards uncommitted local COGs; rebuild selected outputs.
            assert all(not stage.exists() for _, stage, _, _ in batch)
            prepared = catalog.prepare_observations(db, archive, fields, attributes)
            for (_, stage), payload in zip(prepared, payloads, strict=True):
                stage.write_bytes(payload)
            copy_local_cogs(local, archive, [stage for _, stage in prepared])
            catalog.publish_observations(db, archive, [
                (identifier, original[3]) for (identifier, _), original in zip(prepared, batch, strict=True)
            ], files_flushed=True)
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 3
        assert db.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM local_files WHERE kind='cog'").fetchone()[0] == 0
        assert all(source.exists() for _, _, source, _ in batch)


@pytest.mark.parametrize("boundary", ["body", "commit"])
def test_non_lock_errors_roll_back_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    """Propagate constraint or commit authorization errors without waiting."""
    with closing(sqlite3.connect(tmp_path / "catalog.sqlite")) as db:
        db.execute("CREATE TABLE samples (value INTEGER UNIQUE)")

        def unexpected_delay(delay: float) -> None:
            """Reject retrying errors unrelated to temporary lock contention."""
            pytest.fail("a non-lock error was retried")

        def deny_commit(action: int, first: str | None, second: str | None,
                        database: str | None, trigger: str | None) -> int:
            """Reject the real COMMIT while permitting rollback and inspection."""
            return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_TRANSACTION and first == "COMMIT" else sqlite3.SQLITE_OK

        monkeypatch.setattr(catalog, "sleep", unexpected_delay)
        if boundary == "commit":
            db.set_authorizer(deny_commit)
        with pytest.raises(sqlite3.DatabaseError) as caught:
            with catalog.catalog_transaction(db):
                db.execute("INSERT INTO samples VALUES (1)")
                if boundary == "body":
                    db.execute("INSERT INTO samples VALUES (1)")
        db.set_authorizer(None)
        assert caught.value.sqlite_errorcode == (sqlite3.SQLITE_CONSTRAINT_UNIQUE if boundary == "body" else sqlite3.SQLITE_AUTH)
        assert not db.in_transaction
        assert db.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0


def test_interruption_during_commit_backoff_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release the writer transaction when cancellation interrupts lock backoff."""
    path = tmp_path / "catalog.sqlite"
    failure = KeyboardInterrupt("cancelled during commit backoff")
    with closing(sqlite3.connect(path, timeout=.001)) as db, closing(sqlite3.connect(path)) as reader:
        db.execute("CREATE TABLE samples (value INTEGER)")
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM samples").fetchall()

        def interrupt(delay: float) -> None:
            """Interrupt only after an actual blocked COMMIT retains its changes."""
            assert db.in_transaction
            raise failure

        monkeypatch.setattr(catalog, "sleep", interrupt)
        with pytest.raises(KeyboardInterrupt) as caught:
            with catalog.catalog_transaction(db):
                db.execute("INSERT INTO samples VALUES (1)")
        assert caught.value is failure and not db.in_transaction
        reader.rollback()
        assert db.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0
