"""Exercise bounded catalog ownership, acknowledgement and shutdown."""

from concurrent.futures import Future
from pathlib import Path
import sqlite3
from threading import Event, get_ident

import portalocker
import pytest

from era5_pull_pipeline.catalog import archive_session
from era5_pull_pipeline.publication import PublicationWorker


def test_writer_bounds_requests_and_drains_before_releasing_lock(tmp_path: Path) -> None:
    """Own SQLite off the coordinator and retain the archive lock through work."""
    started = Event()
    release = Event()
    owner = get_ident()
    threads: set[int] = set()

    def blocked(db: sqlite3.Connection) -> int:
        """Hold the active request while its bounded queue fills."""
        threads.add(get_ident())
        started.set()
        assert release.wait(10)
        return db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]

    with PublicationWorker(tmp_path, 1) as publisher:
        running = publisher.submit(blocked)
        assert started.wait(10)
        queued = publisher.try_submit(lambda db: get_ident())
        assert isinstance(queued, Future)
        assert publisher.try_submit(lambda db: 0) is None
        assert not running.done() and not queued.done()
        with pytest.raises(portalocker.exceptions.LockException):
            with archive_session(tmp_path):
                pytest.fail("writer lock released before acknowledgement")
        release.set()
    assert running.result() == 0 and queued.result() in threads and owner not in threads
    with archive_session(tmp_path):
        pass


def test_writer_cancels_queued_requests_and_preserves_original_exception(tmp_path: Path) -> None:
    """Cancel waiting work while finishing the active request during interruption."""
    started = Event()
    release = Event()
    executed = Event()

    def blocked(db: sqlite3.Connection) -> None:
        """Keep one request active while cancellation removes pending work."""
        started.set()
        assert release.wait(10)
        db.execute("SELECT COUNT(*) FROM observations").fetchone()

    with pytest.raises(KeyboardInterrupt, match="interrupted"):
        with PublicationWorker(tmp_path, 1) as publisher:
            active = publisher.submit(blocked)
            assert started.wait(10)
            queued = publisher.submit(lambda db: executed.set())
            publisher.cancel_pending()
            release.set()
            raise KeyboardInterrupt("interrupted")
    assert active.done() and queued.cancelled() and not executed.is_set()


def test_writer_startup_and_request_failures_propagate(tmp_path: Path) -> None:
    """Report lock and operation failures without leaking a writer thread."""
    with archive_session(tmp_path):
        with pytest.raises(portalocker.exceptions.LockException):
            with PublicationWorker(tmp_path, 1):
                pytest.fail("second writer started")

    def fail(db: sqlite3.Connection) -> None:
        """Raise the actual catalog error at the request boundary."""
        db.execute("SELECT * FROM missing_table")

    with PublicationWorker(tmp_path, 1) as publisher:
        with pytest.raises(sqlite3.OperationalError, match="missing_table"):
            publisher.call(fail)
        assert publisher.call(lambda db: db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]) == 0
