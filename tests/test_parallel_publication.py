"""Exercise concurrent final-file installation under one catalog owner."""

from contextlib import closing
from pathlib import Path
import sqlite3
from threading import Barrier, Event, Lock, Thread, get_ident

import pytest

from era5_pull_pipeline import catalog
from era5_pull_pipeline.staging import copy_local_cogs
from test_catalog import _stage


@pytest.mark.parametrize("local_staging", [False, True])
def test_parallel_installation_is_bounded_and_catalog_stays_on_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_staging: bool,
) -> None:
    """Install files concurrently only after ready commits, then acknowledge coverage."""
    archive = tmp_path / "archive"
    local = tmp_path / "local" if local_staging else None
    owner = get_ident()
    worker_threads: set[int] = set()
    sql_threads: set[int] = set()
    lock = Lock()
    first_wave = Barrier(8, timeout=10)
    active = peak = started = 0
    with catalog.archive_session(archive, staging_dir=local) as db:
        batch = [_stage(db, archive, hour=hour) for hour in range(24)]
        if local is not None:
            copy_local_cogs(local, archive, [stage for _, stage, _, _ in batch])
        install = catalog._install_observation

        def record_sql(sql: str) -> None:
            """Record the actual thread executing this writer connection's SQL."""
            sql_threads.add(get_ident())

        def held_install(source: Path, destination: Path) -> None:
            """Hold the first eight installs together at the durable file boundary."""
            nonlocal active, peak, started
            with lock:
                active += 1
                started += 1
                ordinal = started
                peak = max(peak, active)
                worker_threads.add(get_ident())
            try:
                if ordinal == 1:
                    # This independent connection observes committed ready rows,
                    # while the catalog writer connection remains on its owner.
                    with closing(sqlite3.connect(archive / "catalog.sqlite")) as reader:
                        assert reader.execute("SELECT COUNT(*) FROM publications WHERE status='ready'").fetchone()[0] == 24
                        assert reader.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
                if ordinal <= 8:
                    first_wave.wait()
                install(source, destination)
            finally:
                with lock:
                    active -= 1

        db.set_trace_callback(record_sql)
        monkeypatch.setattr(catalog, "_install_observation", held_install)
        paths = catalog.publish_observations(
            db, archive, [(identifier, report) for identifier, _, _, report in batch],
            files_flushed=local is not None,
        )
        assert peak == 8 and active == 0 and len(worker_threads) == 8
        assert owner not in worker_threads and sql_threads == {owner}
        assert len(paths) == 24 and all(path.exists() for path in paths)
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 24
        assert db.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM local_files WHERE kind='cog'").fetchone()[0] == 0
        assert all(source.exists() for _, _, source, _ in batch)


@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
def test_failed_parallel_install_settles_workers_before_recovering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_type: type[BaseException],
) -> None:
    """Retain mixed staged/installed files and settle concurrent work before raising."""
    archive, local = tmp_path / "archive", tmp_path / "local"
    failure = failure_type("parallel installation interrupted")
    first_wave = Barrier(8, timeout=10)
    failed, release, returned = Event(), Event(), Event()
    lock = Lock()
    started = active = 0
    installed: list[Path] = []
    failed_paths: list[Path] = []
    observer_errors: list[BaseException] = []
    with catalog.archive_session(archive, staging_dir=local) as db:
        batch = [_stage(db, archive, hour=hour) for hour in range(24)]
        copy_local_cogs(local, archive, [stage for _, stage, _, _ in batch])
        install = catalog._install_observation

        def interrupted_install(source: Path, destination: Path) -> None:
            """Fail one active install while other active files await release."""
            nonlocal started, active
            with lock:
                active += 1
                started += 1
                ordinal = started
            try:
                if ordinal <= 8:
                    first_wave.wait()
                if ordinal == 1:
                    failed_paths.append(source)
                    failed.set()
                    raise failure
                assert release.wait(10)
                install(source, destination)
                installed.append(destination)
            finally:
                with lock:
                    active -= 1

        def release_remaining_workers() -> None:
            """Check that publication cannot return while file work remains active."""
            try:
                assert failed.wait(10)
                assert not returned.wait(.1)
                with lock:
                    assert active > 0
            except BaseException as exc:
                observer_errors.append(exc)
            finally:
                release.set()

        observer = Thread(target=release_remaining_workers)
        observer.start()
        with monkeypatch.context() as patch:
            patch.setattr(catalog, "_install_observation", interrupted_install)
            try:
                with pytest.raises(failure_type) as caught:
                    catalog.publish_observations(
                        db, archive, [(identifier, report) for identifier, _, _, report in batch],
                        files_flushed=True,
                    )
            finally:
                returned.set()
                observer.join(10)
        assert not observer.is_alive() and not observer_errors
        assert caught.value is failure and active == 0
        assert installed and all(path.exists() for path in installed)
        assert failed_paths and all(path.exists() for path in failed_paths)
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM publications WHERE status='ready'").fetchone()[0] == 24
        assert all(stage.exists() and source.exists() for _, stage, source, _ in batch)
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 24
        assert db.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM local_files WHERE kind='cog'").fetchone()[0] == 0
        assert all(not stage.exists() and source.exists() for _, stage, source, _ in batch)
