"""Exercise the disposable converter against real Zarr, SQLite, and COG files."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
from threading import Barrier, Event, Lock, get_ident
from typing import Any

import numpy as np
import portalocker
import pytest
import rasterio
import zarr


SPEC = importlib.util.spec_from_file_location(
    "convert_zarr_archive", Path(__file__).parents[1] / "scripts" / "convert_zarr_archive.py",
)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


@pytest.fixture
def controlled_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use threads for fault coordination while retaining real raster I/O.

    The successful conversion tests exercise spawned processes. These tests
    need shared events and monkeypatches to stop work at publication boundaries.
    """
    def executor(*, max_workers: int, mp_context: Any) -> ThreadPoolExecutor:
        """Run the same bounded Future workflow with controllable worker threads."""
        return ThreadPoolExecutor(max_workers=max_workers)

    monkeypatch.setattr(converter, "ProcessPoolExecutor", executor)


@pytest.fixture
def old_archive(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Build a stopped archive with current coverage, fill slots, and downloads."""
    source = tmp_path / "old archive #1"
    source.mkdir()
    (source / ".writer.lock").touch()
    missing = -(2**63)
    times = np.array([946677600, 946681200], dtype=np.int64)  # 1999-12-31 22:00 and 23:00 UTC.
    temperature = np.arange(16, dtype=np.float32).reshape(2, 2, 4)
    temperature[0, 0, 0] = np.nan
    temperature[0, 0, 1] = 0
    temperature[1] = np.nan  # Covered all-NoData observation must remain an observation.
    precipitation = np.full((2, 2, 4), np.nan, dtype=np.float64)
    precipitation[1] = np.float64(0.000000123456789012345)
    variables = {
        "2_metre_temperature": {"name": "2 metre temperature", "units": "K", "source_variable": "2t", "source_dtype": "float32", "step_type": "instant", "step_units": "s"},
        "mean_total_precipitation_rate": {"name": "Mean total precipitation rate", "units": "kg m**-2 s**-1", "source_variable": "mtpr", "source_dtype": "float64", "step_type": "avg", "step_units": "s"},
    }
    grid = {"type": "regular_ll", "shape": [2, 4], "latitude_first": 90, "latitude_last": 89.75, "longitude_first": 0, "longitude_last": 0.75, "resolution": [0.25, 0.25], "scanning_mode": 0}
    report = {"year": 1999, "month": 12, "grid": grid, "variables": variables}
    path = source / "1999" / "12" / "era5.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=3)
    group.create_array("time", data=times)
    for variable, values in (("2_metre_temperature", temperature), ("mean_total_precipitation_rate", precipitation)):
        group.create_array(variable, data=values, chunks=(2, 1, 2), shards=(2, 2, 4), fill_value=np.nan)
    source_attributes = {"global": {"history": "NCAR test data"}, "variable": {"units": "K"}, "encoding": {"temperature": {"_FillValue": 9.999e20}}}
    with closing(sqlite3.connect(source / "catalog.sqlite")) as db:
        db.executescript("""
            CREATE TABLE source_files (
                id TEXT PRIMARY KEY, object_key TEXT, variable TEXT, size INTEGER,
                etag TEXT, last_modified TEXT, valid_start TEXT, valid_end TEXT,
                path TEXT, created_at TEXT, downloaded_at TEXT, removed_at TEXT
            );
            CREATE TABLE source_selections (source_id TEXT, start_time TEXT, end_time TEXT);
            CREATE TABLE revisions (
                id TEXT PRIMARY KEY, year INTEGER, month INTEGER, status TEXT,
                final_path TEXT, report_json TEXT, provenance_json TEXT
            );
            CREATE TABLE months (year INTEGER, month INTEGER, current_revision_id TEXT);
            CREATE TABLE coverage (
                revision_id TEXT, variable TEXT, valid_time INTEGER, reference_time INTEGER,
                interval_start INTEGER, interval_end INTEGER, start_step INTEGER, end_step INTEGER,
                step_units TEXT, step_type TEXT, source_revision_id TEXT, source_request_sha256 TEXT
            );
            CREATE TABLE field_sources (
                revision_id TEXT, variable TEXT, valid_time INTEGER, source_file_id TEXT, source_variable TEXT
            );
        """)
        for identifier, variable, removed, downloaded in (
            ("temperature", "2t", "2026-01-01T00:00:00+00:00", None),
            ("partial", "mtpr", None, None),
            ("complete", "2d", None, "2026-01-01T00:00:00+00:00"),
        ):
            db.execute(
                "INSERT INTO source_files VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (identifier, f"e5.oper.an.sfc/199912/{identifier}.nc", variable,
                 len(b"completed netcdf data") if identifier == "complete" else 20,
                 f"etag-{identifier}", "2026-01-01T00:00:00+00:00", "1999-12-01T00:00:00+00:00",
                 "2000-01-02T23:00:00+00:00", f"C:/other-machine/staging/ncar/{identifier}/{identifier}.nc",
                 "2026-01-01T00:00:00+00:00", downloaded, removed),
            )
        for identifier in ("partial", "complete"):
            db.execute("INSERT INTO source_selections VALUES (?, ?, ?)", (identifier, "2000-01-01T00:00:00+00:00", "2000-01-01T01:00:00+00:00"))
            download = source / "staging" / "ncar" / identifier / f"{identifier}.nc"
            download.parent.mkdir(parents=True)
            download.write_bytes(b"partial bytes" if identifier == "partial" else b"completed netcdf data")
        provenance = {"sources": [
            {"id": "temperature", "attributes": source_attributes, "downloaded_at": "2025-12-31T23:00:00+00:00"},
            {"id": "partial", "attributes": {"variable": {"units": "kg m**-2 s**-1"}}, "downloaded_at": "2025-12-31T23:30:00+00:00"},
        ]}
        db.execute("INSERT INTO revisions VALUES (?, ?, ?, ?, ?, ?, ?)", ("current", 1999, 12, "published", "1999/12/era5.zarr", json.dumps(report), json.dumps(provenance)))
        db.execute("INSERT INTO months VALUES (1999, 12, 'current')")
        for index, valid in enumerate(times):
            db.execute("INSERT INTO coverage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("current", "2_metre_temperature", int(valid), missing, int(valid), int(valid), missing, missing, "s", "instant", "current", "old-temperature-identity"))
            db.execute("INSERT INTO field_sources VALUES (?, ?, ?, ?, ?)", ("current", "2_metre_temperature", int(valid), "temperature", "2t"))
        db.execute("INSERT INTO coverage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("current", "mean_total_precipitation_rate", int(times[1]), int(times[0]), int(times[0]), int(times[1]), 0, 3600, "s", "avg", "current", "old-precipitation-identity"))
        db.execute("INSERT INTO field_sources VALUES (?, ?, ?, ?, ?)", ("current", "mean_total_precipitation_rate", int(times[1]), "partial", "mtpr"))
        # Historical coverage must not be advertised or read by the converter.
        db.execute("INSERT INTO revisions VALUES (?, ?, ?, ?, ?, ?, ?)", ("old", 1999, 12, "published", "absent/old.zarr", json.dumps(report), json.dumps(provenance)))
        db.execute("INSERT INTO coverage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("old", "2_metre_temperature", int(times[0]) - 3600, missing, int(times[0]) - 3600, int(times[0]) - 3600, missing, missing, "s", "instant", "old", "old-identity"))
        db.execute("INSERT INTO field_sources VALUES (?, ?, ?, ?, ?)", ("old", "2_metre_temperature", int(times[0]) - 3600, "temperature", "2t"))
        db.commit()
    return source, {"temperature": temperature, "precipitation": precipitation, "times": times, "attributes": source_attributes}


@pytest.mark.parametrize("workers", [1, 3])
def test_conversion_preserves_coverage_values_and_pending_sources(old_archive: tuple[Path, dict[str, Any]], tmp_path: Path, workers: int) -> None:
    """Publish only current covered observations and preserve both download states."""
    source, expected = old_archive
    archive = tmp_path / "cogs"
    original_bytes = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    assert converter.convert_zarr_archive(source, archive, workers=workers) == 3
    assert {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()} == original_bytes
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        db.row_factory = sqlite3.Row
        observations = db.execute("SELECT * FROM observations ORDER BY variable, valid_time").fetchall()
        assert len(observations) == 3
        for index, observation in enumerate(observations[:2]):
            assert observation["temporal_type"] == "instant"
            assert observation["reference_time"] is None
            assert observation["units"] == "K"
            assert observation["retrieved_at"] == "2025-12-31T23:00:00+00:00"
            with rasterio.open(archive / observation["path"]) as raster:
                np.testing.assert_array_equal(raster.read(1), expected["temperature"][index])
                assert raster.dtypes == ("float32",)
        observation = observations[2]
        assert observation["temporal_type"] == "mean"
        assert observation["interval_start"] == expected["times"][0]
        assert observation["interval_end"] == expected["times"][1]
        with rasterio.open(archive / observation["path"]) as raster:
            np.testing.assert_array_equal(raster.read(1), expected["precipitation"][1])
            assert raster.dtypes == ("float64",)
        assert db.execute("SELECT COUNT(*) FROM source_selections").fetchone()[0] == 2
        sources = {row["id"]: row for row in db.execute("SELECT * FROM source_files")}
        assert json.loads(sources["temperature"]["attributes_json"]) == expected["attributes"]
        assert sources["complete"]["downloaded_at"] is not None
        assert sources["partial"]["downloaded_at"] is None
        assert sources["temperature"]["retrieved_at"] == "2025-12-31T23:00:00+00:00"
        assert sources["partial"]["retrieved_at"] == "2025-12-31T23:30:00+00:00"
        assert sources["complete"]["retrieved_at"] == sources["complete"]["downloaded_at"]
        for identifier in ("partial", "complete"):
            path = sources[identifier]["path"]
            assert path == f"staging/ncar/{identifier}/{identifier}.nc"
            assert (archive / path).read_bytes() == (source / path).read_bytes()
    assert "format: temporal_cog" in (archive / "dataset.yaml").read_text()
    assert not (archive / "staging" / "conversion" / "values.bin").exists()


def test_interrupted_conversion_resumes_without_rewriting_published_cogs(old_archive: tuple[Path, dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, controlled_workers: None) -> None:
    """A failed COG write leaves source bytes and prior publications reusable."""
    source, _ = old_archive
    archive = tmp_path / "cogs"
    write = converter.write_observation_cog
    calls = []

    def interrupt_second(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Interrupt one conversion after its first complete publication."""
        calls.append(args[1]["valid_time"])
        if len(calls) == 2:
            raise OSError("disk write interrupted")
        return write(*args, **kwargs)

    monkeypatch.setattr(converter, "write_observation_cog", interrupt_second)
    with pytest.raises(OSError, match="interrupted"):
        converter.convert_zarr_archive(source, archive, workers=1)
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        row = db.execute("SELECT path FROM observations").fetchone()
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
    published = archive / row[0]
    timestamp = published.stat().st_mtime_ns
    monkeypatch.setattr(converter, "write_observation_cog", write)
    assert converter.convert_zarr_archive(source, archive, workers=1) == 2
    assert published.stat().st_mtime_ns == timestamp
    assert converter.convert_zarr_archive(source, archive, workers=1) == 0


def test_interrupted_retained_source_copy_resumes_before_marking_download_complete(old_archive: tuple[Path, dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Interrupted copies keep their prefix and stay incomplete in the catalog."""
    source, _ = old_archive
    archive = tmp_path / "cogs"
    copy = converter.shutil.copyfileobj

    def interrupt_copy(incoming: Any, outgoing: Any, length: int) -> None:
        """Write a partial destination before simulating an interruption."""
        outgoing.write(incoming.read(3))
        raise OSError("copy interrupted")

    monkeypatch.setattr(converter.shutil, "copyfileobj", interrupt_copy)
    with pytest.raises(OSError, match="copy interrupted"):
        converter.convert_zarr_archive(source, archive)
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert db.execute("SELECT downloaded_at FROM source_files WHERE id = 'complete'").fetchone()[0] is None
    copied = archive / "staging" / "ncar" / "complete" / "complete.nc"
    assert copied.read_bytes() == b"com"
    monkeypatch.setattr(converter.shutil, "copyfileobj", copy)
    assert converter.convert_zarr_archive(source, archive) == 3
    assert copied.read_bytes() == (source / copied.relative_to(archive)).read_bytes()


def test_converter_requires_stopped_source_writer(old_archive: tuple[Path, dict[str, Any]], tmp_path: Path) -> None:
    """An active old pipeline prevents a partial snapshot during conversion."""
    source, _ = old_archive
    with portalocker.Lock(source / ".writer.lock", "a", timeout=0):
        with pytest.raises(portalocker.exceptions.LockException):
            converter.convert_zarr_archive(source, tmp_path / "cogs")


def test_converter_requires_original_publication_recovery(old_archive: tuple[Path, dict[str, Any]], tmp_path: Path) -> None:
    """Ready Zarr revisions must be recovered before choosing current coverage."""
    source, _ = old_archive
    with closing(sqlite3.connect(source / "catalog.sqlite")) as db:
        db.execute("UPDATE revisions SET status = 'ready' WHERE id = 'old'")
        db.commit()
    with pytest.raises(ValueError, match="old pipeline"):
        converter.convert_zarr_archive(source, tmp_path / "cogs")
    assert not (tmp_path / "cogs").exists()


def test_parallel_writers_overlap_with_one_catalog_owner(
    old_archive: tuple[Path, dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    controlled_workers: None,
) -> None:
    """Write separate rasters concurrently while the caller owns publication."""
    source, _ = old_archive
    archive = tmp_path / "parallel"
    write = converter.write_observation_cog
    publish = converter.publish_observation
    owner = get_ident()
    barrier = Barrier(2, timeout=10)
    lock = Lock()
    active = peak = 0

    def overlapping_write(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Hold both temperature writes together at the real raster boundary."""
        nonlocal active, peak
        assert get_ident() != owner
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if args[1]["variable"] == "2_metre_temperature":
                barrier.wait()
            return write(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    def publish_from_owner(*args: Any, **kwargs: Any) -> Path:
        """Check catalog ownership while exercising actual file publication."""
        assert get_ident() == owner
        return publish(*args, **kwargs)

    monkeypatch.setattr(converter, "write_observation_cog", overlapping_write)
    monkeypatch.setattr(converter, "publish_observation", publish_from_owner)
    assert converter.convert_zarr_archive(source, archive, workers=2) == 3
    assert peak == 2 and active == 0
    monkeypatch.setattr(converter, "write_observation_cog", lambda *args, **kwargs: pytest.fail("Repeat encoded a stored observation"))
    assert converter.convert_zarr_archive(source, archive, workers=2) == 0


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_parallel_failure_keeps_committed_rasters_and_resumes(
    old_archive: tuple[Path, dict[str, Any]], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, failure: type[BaseException], controlled_workers: None,
) -> None:
    """Retain the committed first raster when a concurrent writer fails."""
    source, expected = old_archive
    archive = tmp_path / "interrupted"
    write = converter.write_observation_cog
    publish = converter.publish_observation
    first_published = Event()

    def fail_after_first_commit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Fail the second temperature write after the first becomes durable."""
        field = args[1]
        if field["variable"] == "2_metre_temperature" and field["valid_time"] == int(expected["times"][1]):
            assert first_published.wait(10)
            raise failure("parallel conversion interrupted")
        return write(*args, **kwargs)

    def record_first_commit(*args: Any, **kwargs: Any) -> Path:
        """Release the failing writer once an actual publication completes."""
        path = publish(*args, **kwargs)
        first_published.set()
        return path

    with monkeypatch.context() as patch:
        patch.setattr(converter, "write_observation_cog", fail_after_first_commit)
        patch.setattr(converter, "publish_observation", record_first_commit)
        with pytest.raises(failure, match="parallel conversion interrupted"):
            converter.convert_zarr_archive(source, archive, workers=2)
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        rows = db.execute("SELECT path FROM observations").fetchall()
        assert len(rows) == 1
        assert db.execute("SELECT COUNT(*) FROM source_selections").fetchone()[0] == 2
    first = archive / rows[0][0]
    modified = first.stat().st_mtime_ns
    assert not (archive / "staging/conversion/values.bin").exists()
    assert converter.convert_zarr_archive(source, archive, workers=2) == 2
    assert first.stat().st_mtime_ns == modified


def test_interrupt_waits_for_active_writers_before_releasing_decoded_arrays(
    old_archive: tuple[Path, dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    controlled_workers: None,
) -> None:
    """Finish active raster I/O before closing its shared memory mapping."""
    source, _ = old_archive
    archive = tmp_path / "stopped"
    started = Event()
    release = Event()
    finished = Event()
    write = converter.write_observation_cog

    def held_write(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Keep a real TIFF write in flight when the coordinator is stopped."""
        started.set()
        assert release.wait(10)
        result = write(*args, **kwargs)
        finished.set()
        return result

    def interrupt_wait(*args: Any, **kwargs: Any) -> Any:
        """Interrupt the coordinator once a writer has borrowed the array."""
        assert started.wait(10)
        with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
            assert db.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 1
        release.set()
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(converter, "write_observation_cog", held_write)
        patch.setattr(converter, "wait", interrupt_wait)
        with pytest.raises(KeyboardInterrupt):
            converter.convert_zarr_archive(source, archive, workers=1)
    assert finished.is_set()
    assert not (archive / "staging/conversion/values.bin").exists()
    assert converter.convert_zarr_archive(source, archive, workers=2) == 3
