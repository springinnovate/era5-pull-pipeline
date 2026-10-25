"""Bound numerical-library pools before spawned processing imports begin."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
from typing import Any, Iterator

import pytest

from era5_pull_pipeline import pipeline
from test_ncar_pipeline import NcarFiles, files


@pytest.fixture
def parent_thread_settings(monkeypatch: pytest.MonkeyPatch) -> dict[str, str | None]:
    """Set explicit and absent parent controls to check exact restoration."""
    values = {"OPENBLAS_NUM_THREADS": "7", "OMP_NUM_THREADS": None,
              "MKL_NUM_THREADS": "3", "NUMEXPR_NUM_THREADS": None}
    for name, value in values.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    return values


@pytest.mark.parametrize("fail_body", [False, True])
def test_spawned_main_import_observes_bounded_library_threads(
    tmp_path: Path, parent_thread_settings: dict[str, str | None], fail_body: bool,
) -> None:
    """Apply limits before Windows-style spawn imports the worker's main module."""
    script = tmp_path / "worker_import_probe.py"
    script.write_text('''import os
NAMES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
IMPORTED_VALUES = {name: os.environ.get(name) for name in NAMES}
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy
from era5_pull_pipeline.pipeline import _limit_worker_library_threads


def imported_environment():
    return IMPORTED_VALUES


if __name__ == "__main__":
    before = {name: os.environ.get(name) for name in NAMES}
    observed = None
    try:
        with _limit_worker_library_threads(), ProcessPoolExecutor(
                max_workers=1, mp_context=get_context("spawn")) as executor:
            observed = executor.submit(imported_environment).result(timeout=30)
            if sys.argv[1] == "fail":
                raise RuntimeError("body failed")
    except RuntimeError as exc:
        assert str(exc) == "body failed"
    after = {name: os.environ.get(name) for name in NAMES}
    print(json.dumps({"before": before, "worker_import": observed, "after": after}))
''')
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(Path(pipeline.__file__).parents[1]), environment.get("PYTHONPATH"),
    )))
    result = subprocess.run(
        [sys.executable, str(script), "fail" if fail_body else "success"],
        capture_output=True, text=True, env=environment, check=True, timeout=60,
    )
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed["worker_import"] == {name: "1" for name in parent_thread_settings}
    assert observed["before"] == observed["after"] == parent_thread_settings
    assert {name: os.environ.get(name) for name in parent_thread_settings} == parent_thread_settings


def test_overlapping_pool_scopes_restore_only_after_last_pool_exits(
    parent_thread_settings: dict[str, str | None],
) -> None:
    """Let concurrent callers overlap without restoring settings beneath a live pool."""
    entered, release = Event(), Event()
    errors: list[BaseException] = []
    expected = {name: "1" for name in parent_thread_settings}

    def second_pool() -> None:
        """Hold a second caller's environment after the first caller exits."""
        try:
            with pipeline._limit_worker_library_threads():
                assert {name: os.environ.get(name) for name in expected} == expected
                entered.set()
                assert release.wait(10)
                assert {name: os.environ.get(name) for name in expected} == expected
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=second_pool)
    try:
        with pipeline._limit_worker_library_threads():
            thread.start()
            assert entered.wait(10), "independent processing scopes were serialized"
        assert {name: os.environ.get(name) for name in expected} == expected
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and not errors
    assert {name: os.environ.get(name) for name in expected} == parent_thread_settings


@pytest.mark.parametrize("fail_body", [False, True])
def test_pull_limits_surround_pool_creation_and_shutdown(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
    parent_thread_settings: dict[str, str | None], fail_body: bool,
) -> None:
    """Keep controls set from pool creation through shutdown on workflow success or failure."""
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    pool_factory = pipeline.ProcessPoolExecutor
    expected = {name: "1" for name in parent_thread_settings}
    lifecycle: list[str] = []

    @contextmanager
    def observed_pool(**kwargs: Any) -> Iterator[ThreadPoolExecutor]:
        """Observe configuration at the actual pull coordinator's pool boundary."""
        assert {name: os.environ.get(name) for name in expected} == expected
        lifecycle.append("created")
        try:
            with pool_factory(**kwargs) as executor:
                yield executor
        finally:
            assert {name: os.environ.get(name) for name in expected} == expected
            lifecycle.append("stopped")

    def fail_conversion(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """Fail inside active processing so the ordinary shutdown path runs."""
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", observed_pool)
    if fail_body:
        monkeypatch.setattr(pipeline, "write_ncar_chunk_cogs", fail_conversion)
        with pytest.raises(RuntimeError, match="conversion failed"):
            pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], tmp_path / "archive")
    else:
        result = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], tmp_path / "archive")
        assert result["completed"] == 24
    assert lifecycle == ["created", "stopped"]
    assert {name: os.environ.get(name) for name in parent_thread_settings} == parent_thread_settings
