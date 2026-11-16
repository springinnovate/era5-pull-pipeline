"""Verify daily pulls through real hourly NetCDF conversion and COG publication."""

from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path
import sqlite3

import numpy as np
from numpy.typing import NDArray
import portalocker
import pytest
import rasterio

from era5_pull_pipeline import catalog, daily_pipeline, pipeline
from test_ncar_pipeline import NcarFiles, files


def read_daily_values(archive: Path) -> dict[tuple[str, int], NDArray[np.float64]]:
    """Read every published daily raster using its public catalog path.

    Args:
        archive: Daily archive containing the public observations view.

    Returns:
        Full-resolution arrays keyed by source short name and interval start.
    """
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        rows = db.execute("SELECT source_variable, interval_start, path FROM observations").fetchall()
    result = {}
    for variable, start, relative in rows:
        with rasterio.open(archive / relative) as dataset:
            result[variable, start] = dataset.read(1)
    return result


def test_daily_pull_handles_following_midnight_and_completed_repeat(
    tmp_path: Path, files: NcarFiles,
) -> None:
    """Publish complete January totals with only following midnight available.

    Args:
        tmp_path: Isolated provider, daily archive, and private work roots.
        files: Real NetCDF fixture provider replacing only network boundaries.
    """
    archive, work = tmp_path / "daily", tmp_path / "work"
    files.add_file("2t", "2000-01-31T00", "2000-01-31T23", value=280)
    files.add_file("mtpr", "2000-01-31T00", "2000-01-31T12", value=0.1)
    day = date(2000, 1, 31)
    result = daily_pipeline.pull_daily(day, day, ["2t", "mtpr"], archive, work_dir=work)
    assert result == {"completed": 2, "skipped": 0, "incomplete": 0, "pending": 0}
    start = int(datetime(2000, 1, 31, tzinfo=UTC).timestamp())
    values = read_daily_values(archive)
    assert set(values) == {("2t", start), ("mtpr", start)}
    np.testing.assert_array_equal(values["2t", start], 280)
    np.testing.assert_allclose(values["mtpr", start], np.float64(np.float32(.1)) * 86400)
    assert all(value.dtype == np.float64 for value in values.values())
    assert not list(work.glob("daily-hourly-*"))
    inventory_calls, downloads = len(files.inventory_calls), len(files.downloaded)
    repeated = daily_pipeline.pull_daily(day, day, ["2t", "mtpr"], archive, work_dir=work)
    assert repeated == {"completed": 0, "skipped": 2, "incomplete": 0, "pending": 0}
    assert (len(files.inventory_calls), len(files.downloaded)) == (inventory_calls, downloads)
    files.add_file("mtpr", "2000-01-31T00", "2000-01-31T12", version=2, value=.2)
    refreshed = daily_pipeline.pull_daily(day, day, ["mtpr"], archive,
                                          work_dir=work, refresh=True)
    assert refreshed == {"completed": 1, "skipped": 0, "incomplete": 0, "pending": 0}
    after = read_daily_values(archive)
    np.testing.assert_allclose(after["mtpr", start], np.float64(np.float32(.2)) * 86400)
    np.testing.assert_array_equal(after["2t", start], values["2t", start])
    assert not list(work.glob("daily-hourly-*"))


def test_exact_hourly_selection_preserves_next_month_source_and_refresh(
    tmp_path: Path, files: NcarFiles,
) -> None:
    """Use exact hours for prefetch, source retention, and refresh availability.

    Args:
        tmp_path: Isolated hourly archive.
        files: Real forecast source ending at the required next-month midnight.
    """
    archive = tmp_path / "hourly"
    source = files.add_file("mtpr", "1999-12-31T00", "1999-12-31T12", value=.1)
    first = int(datetime(1999, 12, 31, 1, tzinfo=UTC).timestamp())
    hours = {("mtpr", first + offset * 3600) for offset in range(24)}
    result = pipeline.pull_range(date(1999, 12, 31), date(2000, 1, 1), ["mtpr"],
                                 archive, selected_hours=hours)
    assert result == {"completed": 24, "skipped": 0, "unavailable": 0, "failed": 0, "pending": 0}
    assert files.downloaded == [(source.key, source.etag)]
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert set(db.execute("SELECT source_variable, valid_time FROM observations")) == hours
        assert db.execute("SELECT COUNT(*) FROM source_selections").fetchone()[0] == 0
    files.add_file("mtpr", "1999-12-31T00", "1999-12-31T12", version=2, value=.2)
    refreshed = pipeline.pull_range(date(1999, 12, 31), date(2000, 1, 1), ["mtpr"],
                                    archive, selected_hours=hours, refresh=True)
    assert refreshed == {"completed": 24, "skipped": 0, "unavailable": 0, "failed": 0, "pending": 0}


def test_daily_refresh_preserves_unselected_days_and_variables(
    tmp_path: Path, files: NcarFiles,
) -> None:
    """Replace one selected day from current provider data and retain other daily data.

    Args:
        tmp_path: Isolated output and work roots.
        files: Mutable real NetCDF provider versions.
    """
    archive, work = tmp_path / "daily", tmp_path / "work"
    files.add_file("2t", "2000-01-01T00", "2000-01-02T23", value=280)
    files.add_file("10u", "2000-01-01T00", "2000-01-02T23", value=5)
    daily_pipeline.pull_daily(date(2000, 1, 1), date(2000, 1, 2), ["2t", "10u"],
                               archive, work_dir=work)
    before = read_daily_values(archive)
    files.add_file("2t", "2000-01-01T00", "2000-01-02T23", version=2, value=290)
    result = daily_pipeline.pull_daily(date(2000, 1, 1), date(2000, 1, 1), ["2t"],
                                       archive, work_dir=work, refresh=True)
    assert result == {"completed": 1, "skipped": 0, "incomplete": 0, "pending": 0}
    after = read_daily_values(archive)
    selected = "2t", int(datetime(2000, 1, 1, tzinfo=UTC).timestamp())
    np.testing.assert_array_equal(after[selected], 290)
    for key in before.keys() - {selected}:
        np.testing.assert_array_equal(after[key], before[key])


def test_incomplete_month_retains_inputs_and_stops_before_later_months(
    tmp_path: Path, files: NcarFiles,
) -> None:
    """Resume a missing final hour without accumulating future unavailable work.

    Args:
        tmp_path: Isolated output and work roots.
        files: Real source with one initially missing hour.
    """
    archive, work = tmp_path / "daily", tmp_path / "work"
    files.add_file("2t", "2000-01-31T00", "2000-01-31T22")
    files.add_file("2t", "2000-02-01T00", "2000-02-01T23")
    result = daily_pipeline.pull_daily(date(2000, 1, 31), date(2000, 2, 1), ["2t"],
                                       archive, work_dir=work)
    assert result == {"completed": 0, "skipped": 0, "incomplete": 1, "pending": 1}
    assert len(list(work.glob("daily-hourly-*"))) == 1
    assert len(list(work.rglob("*.tif"))) == 23
    assert all(first.month == 1 for first, _, _ in files.inventory_calls)
    files.add_file("2t", "2000-01-31T18", "2000-01-31T23")
    resumed = daily_pipeline.pull_daily(date(2000, 1, 31), date(2000, 2, 1), ["2t"],
                                        archive, work_dir=work)
    assert resumed == {"completed": 2, "skipped": 0, "incomplete": 0, "pending": 0}
    assert len(files.downloaded) == 3
    assert not list(work.glob("daily-hourly-*"))


def test_failed_daily_publication_retains_hourly_inputs_until_recovery(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover a durable ready publication before removing its owned hourly source.

    Args:
        tmp_path: Isolated output and work roots.
        files: Real complete temperature source.
        monkeypatch: Inject a daily file-installation failure at publication.
    """
    archive, work = tmp_path / "daily", tmp_path / "work"
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    install = catalog._install_observation

    def fail_daily_install(source: Path, destination: Path) -> None:
        """Fail daily file installation while allowing hourly publication.

        Args:
            source: Validated staged raster.
            destination: Authoritative publication path.

        Raises:
            OSError: The injected daily archive storage failure.
        """
        if destination.is_relative_to(archive):
            raise OSError("daily storage unavailable")
        install(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(catalog, "_install_observation", fail_daily_install)
        with pytest.raises(OSError, match="daily storage unavailable"):
            daily_pipeline.pull_daily(date(2000, 1, 1), date(2000, 1, 1), ["2t"],
                                        archive, work_dir=work)
    assert len(list(work.rglob("*.tif"))) == 24
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert db.execute("SELECT status FROM publications").fetchall() == [("ready",)]
    calls = len(files.inventory_calls), len(files.downloaded)
    result = daily_pipeline.pull_daily(date(2000, 1, 1), date(2000, 1, 1), ["2t"],
                                       archive, work_dir=work)
    assert result == {"completed": 0, "skipped": 1, "incomplete": 0, "pending": 0}
    assert (len(files.inventory_calls), len(files.downloaded)) == calls
    assert not list(work.glob("daily-hourly-*"))
    assert len(read_daily_values(archive)) == 1


@pytest.mark.parametrize("owner", ["archive", "work"])
def test_daily_pull_rejects_competing_writer_before_network(
    tmp_path: Path, files: NcarFiles, owner: str,
) -> None:
    """Keep both daily output and private scratch exclusive for the whole invocation.

    Args:
        tmp_path: Isolated output and work roots.
        files: Provider whose access remains observable.
        owner: Lock already held by another writer.
    """
    archive, work = tmp_path / "daily", tmp_path / "work"
    work.mkdir()
    lock = (catalog.archive_session(archive, resolution="daily") if owner == "archive"
            else portalocker.Lock(work / ".daily-pull.lock", "a", timeout=0))
    with lock, pytest.raises(portalocker.exceptions.LockException):
        daily_pipeline.pull_daily(date(2000, 1, 1), date(2000, 1, 1), ["2t"],
                                   archive, work_dir=work)
    assert not files.inventory_calls and not files.downloaded


@pytest.mark.parametrize("start,end,variables", [
    (date(2000, 1, 2), date(2000, 1, 1), ["2t"]),
    (date(2000, 1, 1), date(2000, 1, 1), []),
    (date(2000, 1, 1), date(2000, 1, 1), ["unsupported"]),
    (date(9999, 12, 30), date(9999, 12, 30), ["mtpr"]),
    (date.max, date.max, ["2t"]),
])
def test_daily_pull_validates_selection_before_transfer(
    tmp_path: Path, files: NcarFiles, start: date, end: date, variables: list[str],
) -> None:
    """Reject unsupported selections before creating archives or querying NCAR.

    Args:
        tmp_path: Isolated output and work roots.
        files: Provider whose access remains observable.
        start: First requested UTC day.
        end: Last requested UTC day.
        variables: Requested provider names.
    """
    archive, work = tmp_path / "daily", tmp_path / "work"
    with pytest.raises(ValueError):
        daily_pipeline.pull_daily(start, end, variables, archive, work_dir=work)
    assert not archive.exists() and not work.exists()
    assert not files.inventory_calls and not files.downloaded


@pytest.mark.parametrize("location", ["same", "inside", "parent"])
def test_daily_pull_rejects_overlapping_work_paths(
    tmp_path: Path, files: NcarFiles, location: str,
) -> None:
    """Reject path overlap before creating scratch or contacting NCAR.

    Args:
        tmp_path: Isolated output root.
        files: Provider whose access remains observable.
        location: Relative location of the conflicting private work directory.
    """
    archive = tmp_path / "daily"
    work = {"same": archive, "inside": archive / "work", "parent": tmp_path}[location]
    with pytest.raises(ValueError, match="outside and separate"):
        daily_pipeline.pull_daily(date(2000, 1, 1), date(2000, 1, 1), ["2t"],
                                   archive, work_dir=work)
    assert not archive.exists()
    assert not files.inventory_calls and not files.downloaded


def test_daily_pull_preserves_unowned_work_directory(
    tmp_path: Path, files: NcarFiles,
) -> None:
    """Refuse a colliding user directory instead of adopting or deleting its files.

    Args:
        tmp_path: Isolated output and preexisting work roots.
        files: Provider whose access remains observable.
    """
    archive, work = tmp_path / "daily", tmp_path / "work"
    unrelated = work / "daily-hourly-2000-01-01-01-2t"
    unrelated.mkdir(parents=True)
    note = unrelated / "keep.txt"
    note.write_text("user data", encoding="utf-8")
    with pytest.raises(ValueError, match="belongs to another selection"):
        daily_pipeline.pull_daily(date(2000, 1, 1), date(2000, 1, 1), ["2t"],
                                   archive, work_dir=work)
    assert note.read_text(encoding="utf-8") == "user data"
    assert not files.inventory_calls and not files.downloaded


def test_daily_pull_recovers_interrupted_scratch_initialization(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave the canonical scratch name usable after initialization fails.

    Args:
        tmp_path: Isolated output and work roots.
        files: Real complete temperature source.
        monkeypatch: Fail scratch mkdir after its ownership record is durable.
    """
    archive, work = tmp_path / "daily", tmp_path / "work"
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    mkdir = Path.mkdir

    def fail_initialization(path: Path, mode: int = 0o777,
                            parents: bool = False, exist_ok: bool = False) -> None:
        """Raise at scratch creation while preserving other directory creation.

        Args:
            path: Directory being created.
            mode: Requested directory permissions.
            parents: Whether to create absent parent directories.
            exist_ok: Whether an existing directory is acceptable.

        Raises:
            OSError: The injected scratch initialization failure.
        """
        if path.name.startswith("daily-hourly-"):
            raise OSError("scratch initialization interrupted")
        mkdir(path, mode, parents, exist_ok)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "mkdir", fail_initialization)
        with pytest.raises(OSError, match="scratch initialization interrupted"):
            daily_pipeline.pull_daily(date(2000, 1, 1), date(2000, 1, 1), ["2t"],
                                       archive, work_dir=work)
    assert not list(work.glob("daily-hourly-*"))
    assert len(list(work.glob(".daily-hourly-*.owner.json"))) == 1
    assert not files.inventory_calls and not files.downloaded
    result = daily_pipeline.pull_daily(date(2000, 1, 1), date(2000, 1, 1), ["2t"],
                                       archive, work_dir=work)
    assert result == {"completed": 1, "skipped": 0, "incomplete": 0, "pending": 0}
    assert not list(work.glob(".daily-hourly-*.owner.json"))


def test_daily_pull_recovers_partial_scratch_cleanup(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finish interrupted cleanup without downloading already published days.

    Args:
        tmp_path: Isolated output and work roots.
        files: Real complete temperature source.
        monkeypatch: Remove most scratch children before raising a cleanup error.
    """
    archive, work = tmp_path / "daily", tmp_path / "work"
    scratch = work / "daily-hourly-2000-01-01-01-2t"
    marker = work / f".{scratch.name}.owner.json"
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    remove_tree = daily_pipeline.shutil.rmtree

    def fail_cleanup(path: Path) -> None:
        """Remove scratch children except its catalog before failing cleanup.

        Args:
            path: Validated owned directory selected for recursive removal.

        Raises:
            OSError: The injected partial cleanup failure.
        """
        assert path == scratch
        for child in path.iterdir():
            if child.name == "catalog.sqlite":
                continue
            if child.is_dir():
                remove_tree(child)
            else:
                child.unlink()
        raise OSError("scratch cleanup interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(daily_pipeline.shutil, "rmtree", fail_cleanup)
        with pytest.raises(OSError, match="scratch cleanup interrupted"):
            daily_pipeline.pull_daily(date(2000, 1, 1), date(2000, 1, 1), ["2t"],
                                       archive, work_dir=work)
    assert marker.is_file()
    assert [child.name for child in scratch.iterdir()] == ["catalog.sqlite"]
    assert len(read_daily_values(archive)) == 1
    calls = len(files.inventory_calls), len(files.downloaded)
    result = daily_pipeline.pull_daily(date(2000, 1, 1), date(2000, 1, 1), ["2t"],
                                       archive, work_dir=work)
    assert result == {"completed": 0, "skipped": 1, "incomplete": 0, "pending": 0}
    assert (len(files.inventory_calls), len(files.downloaded)) == calls
    assert not scratch.exists() and not marker.exists()


def test_daily_pull_preserves_unowned_record_without_scratch(
    tmp_path: Path, files: NcarFiles,
) -> None:
    """Reject an ownership record for another archive even when scratch is absent.

    Args:
        tmp_path: Isolated output and work roots.
        files: Provider whose access remains observable.
    """
    archive, work = tmp_path / "daily", tmp_path / "work"
    work.mkdir()
    marker = work / ".daily-hourly-2000-01-01-01-2t.owner.json"
    original = '{"archive": "another archive"}'
    marker.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="belongs to another selection"):
        daily_pipeline.pull_daily(date(2000, 1, 1), date(2000, 1, 1), ["2t"],
                                   archive, work_dir=work)
    assert marker.read_text(encoding="utf-8") == original
    assert not list(work.glob("daily-hourly-*"))
    assert not files.inventory_calls and not files.downloaded


@pytest.mark.parametrize("options", [
    {"download_workers": 0}, {"processing_workers": 1.5},
    {"upload_workers": -1}, {"max_source_gb": float("nan")},
    {"max_decoded_gb": float("inf")}, {"max_source_gb": 0},
])
def test_daily_pull_validates_limits_before_transfer(
    tmp_path: Path, files: NcarFiles, options: dict[str, float],
) -> None:
    """Reject invalid resource settings without creating archive or scratch work.

    Args:
        tmp_path: Isolated output and work roots.
        files: Provider whose access remains observable.
        options: Invalid concurrency or memory option.
    """
    archive, work = tmp_path / "daily", tmp_path / "work"
    with pytest.raises(ValueError, match="positive"):
        daily_pipeline.pull_daily(date(2000, 1, 1), date(2000, 1, 1), ["2t"],
                                   archive, work_dir=work, **options)
    assert not archive.exists() and not work.exists()
    assert not files.inventory_calls and not files.downloaded
