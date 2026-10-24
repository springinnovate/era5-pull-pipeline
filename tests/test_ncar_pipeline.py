"""Exercise bulk NCAR scheduling through real NetCDF, COG, and SQLite files."""

from collections.abc import Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
import shutil
import sqlite3
from threading import Barrier, Event, Lock, get_ident
from typing import Any

import numpy as np
import pytest
import requests
import xarray as xr

from archive_assertions import read_raster_values

from era5_pull_pipeline import catalog, conversion, ncar_download, pipeline
from era5_pull_pipeline.ncar import SourceFile
from era5_pull_pipeline.staging import get_source_path


@dataclass
class NcarFiles:
    """Provide small real NetCDF files behind a replaceable NCAR inventory."""

    directory: Path
    sources: list[SourceFile] = field(default_factory=list)
    payloads: dict[tuple[str, str], Path] = field(default_factory=dict)
    downloaded: list[tuple[str, str]] = field(default_factory=list)
    inventory_calls: list[tuple[date, date, tuple[str, ...]]] = field(default_factory=list)

    def add_file(
        self, variable: str, first: str, last: str, *,
        version: int = 1, value: float = 280.0,
    ) -> SourceFile:
        """Write an analysis or forecast source and expose its current version.

        Args:
            variable: Temperature, wind, or mean precipitation short name.
            first: First valid analysis hour or forecast initialization hour.
            last: Last valid analysis hour or forecast initialization hour.
            version: Provider revision represented by a distinct ETag.
            value: Constant source value used to compare published revisions.

        Returns:
            Source identity and its actual NetCDF byte size.
        """
        first_time = datetime.fromisoformat(first).replace(tzinfo=UTC)
        last_time = datetime.fromisoformat(last).replace(tzinfo=UTC)
        coordinates: dict[str, Any] = {
            "latitude": conversion.GLOBAL_LATITUDES,
            "longitude": conversion.GLOBAL_LONGITUDES,
        }
        if variable == "mtpr":
            collection = "e5.oper.fc.sfc.meanflux"
            dimensions = ("forecast_initial_time", "forecast_hour", "latitude", "longitude")
            references = np.arange(
                np.datetime64(first), np.datetime64(last) + np.timedelta64(12, "h"),
                np.timedelta64(12, "h"),
            )
            coordinates["forecast_initial_time"] = references
            coordinates["forecast_hour"] = ("forecast_hour", np.arange(1, 13), {"units": "hour"})
            shape, chunks = (len(references), 12, 3, 4), (1, 4, 2, 3)
            name, units = "Mean total precipitation rate", "kg m**-2 s**-1"
            valid_start = first_time + timedelta(hours=1)
            valid_end = last_time + timedelta(hours=12)
        else:
            collection = "e5.oper.an.sfc"
            dimensions = ("time", "latitude", "longitude")
            coordinates["time"] = np.arange(
                np.datetime64(first), np.datetime64(last) + np.timedelta64(1, "h"),
                np.timedelta64(1, "h"),
            )
            shape, chunks = (len(coordinates["time"]), 3, 4), (6, 2, 3)
            name, units = {
                "2t": ("2 metre temperature", "K"),
                "10u": ("10 metre U wind component", "m s**-1"),
            }[variable]
            valid_start, valid_end = first_time, last_time
        filename = f"{collection}.{variable}.ll025sc.{first_time:%Y%m%d%H}_{valid_end:%Y%m%d%H}.nc"
        key = f"{collection}/{first_time:%Y%m}/{filename}"
        path = self.directory / str(version) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset = xr.Dataset(
            {variable.upper(): (dimensions, np.full(shape, value, dtype=np.float32), {
                "long_name": name, "short_name": variable, "units": units,
            })},
            coords=coordinates,
        )
        dataset.to_netcdf(path, engine="h5netcdf", encoding={
            variable.upper(): {
                "chunksizes": chunks, "zlib": True, "_FillValue": np.float32(9.999e20),
            },
            "latitude": {"_FillValue": None}, "longitude": {"_FillValue": None},
        })
        source = SourceFile(
            key, variable, path.stat().st_size, f'"version-{version}"',
            f"2026-01-{version:02d}T00:00:00+00:00", valid_start, valid_end,
        )
        self.sources = [existing for existing in self.sources if existing.key != key]
        self.sources.append(source)
        self.payloads[(key, source.etag)] = path
        return source

    def discover(
        self, start: date, end: date, variables: Sequence[str], session: requests.Session,
    ) -> list[SourceFile]:
        """Return current fixture files overlapping the requested UTC selection.

        Args:
            start: First selected date.
            end: Last selected date.
            variables: Requested source short names.
            session: Unused HTTP session supplied by the scheduler.

        Returns:
            Available current provider files in chronological order.
        """
        self.inventory_calls.append((start, end, tuple(variables)))
        first = datetime.combine(start, time.min, tzinfo=UTC)
        last = datetime.combine(end, time(23), tzinfo=UTC)
        return sorted([
            source for source in self.sources
            if source.variable in variables and source.valid_start <= last and source.valid_end >= first
        ], key=lambda source: (source.valid_start, source.variable))

    def transfer(self, source: SourceFile, target: Path, **kwargs: Any) -> None:
        """Copy one fixture input at the transfer boundary.

        Args:
            source: Provider object selected for transfer.
            target: Registered destination path.
            **kwargs: Transfer cancellation and progress parameters.
        """
        identity = source.key, source.etag
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.payloads[identity], target)
        self.downloaded.append(identity)

    def download(
        self, db: sqlite3.Connection, sources: Sequence[SourceFile],
        rows: Sequence[sqlite3.Row], download_workers: int = 2, *, archive: Path,
    ) -> None:
        """Copy fixture bytes and record completion at the download boundary.

        Args:
            db: Writer-owned catalog connection.
            sources: Provider objects selected by the scheduler.
            rows: Registered source paths and completion state.
            archive: Root of the cataloged relative paths.
            download_workers: Concurrency setting, tested by the downloader suite.
        """
        for source, row in zip(sources, rows, strict=True):
            if row["downloaded_at"] is not None:
                continue
            identity = (source.key, source.etag)
            target = get_source_path(archive, catalog.get_staging_dir(db), row)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.payloads[identity], target)
            with db:
                db.execute(
                    "UPDATE source_files SET downloaded_at = ? WHERE id = ?",
                    (datetime.now(UTC).isoformat(), row["id"]),
                )
            self.downloaded.append(identity)


@pytest.fixture
def files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> NcarFiles:
    """Replace provider I/O while retaining real ingestion and publication.

    Args:
        tmp_path: Directory for external fixture source files.
        monkeypatch: Fixture provider and small global-grid replacements.

    Returns:
        Mutable provider inventory and transfer observations.
    """
    monkeypatch.setattr(conversion, "GLOBAL_LATITUDES", np.array([90.0, 89.75, 89.5]))
    monkeypatch.setattr(conversion, "GLOBAL_LONGITUDES", np.array([0.0, 0.25, 0.5, 0.75]))
    fixture = NcarFiles(tmp_path / "provider")
    monkeypatch.setattr(pipeline, "discover_files", fixture.discover)
    monkeypatch.setattr(ncar_download, "download_file", fixture.transfer)

    def controlled_pool(*, max_workers: int, **kwargs: Any) -> ThreadPoolExecutor:
        """Retain small-grid fixtures and fault hooks across worker calls."""
        return ThreadPoolExecutor(max_workers=max_workers)

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", controlled_pool)
    return fixture


def test_year_boundary_reuses_forecast_file_and_completed_repeat_is_offline(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain the shared forecast file through both months and then remove it."""
    archive = tmp_path / "archive"
    shared = files.add_file("mtpr", "1999-12-30T18", "1999-12-31T18", value=0.1)
    following = files.add_file("mtpr", "2000-01-01T06", "2000-01-01T18", value=0.2)
    ingest = pipeline.download_and_ingest_month
    source_after_december: list[Path] = []

    def inspect_published_month(
        db: pipeline.PublicationWorker, destination: Path, year: int, month: int,
        *args: Any, **kwargs: Any,
    ) -> Path:
        """Check the shared source immediately after December publication."""
        result = ingest(db, destination, year, month, *args, **kwargs)
        if (year, month) == (1999, 12):
            source = db.call(lambda connection: connection.execute("SELECT path FROM source_files WHERE object_key = ?", (shared.key,)).fetchone())
            source_after_december.append((destination / source["path"]))
            assert source_after_december[0].exists()
        return result

    monkeypatch.setattr(pipeline, "download_and_ingest_month", inspect_published_month)
    result = pipeline.pull_range(date(1999, 12, 31), date(2000, 1, 1), ["mtpr"], archive)
    assert result == {"completed": 48, "skipped": 0, "unavailable": 0, "failed": 0, "pending": 0}
    assert files.downloaded == [(shared.key, shared.etag), (following.key, following.etag)]
    assert not source_after_december[0].exists()
    dataset = read_raster_values(archive, 1999, 12)
    assert dataset["mean_total_precipitation_rate"].shape == (24, 3, 4)
    np.testing.assert_array_equal(dataset["mean_total_precipitation_rate"], np.float32(0.1))
    dataset = read_raster_values(archive, 2000, 1)
    values = dataset["mean_total_precipitation_rate"]
    np.testing.assert_array_equal(values[:7], np.float32(0.1))
    np.testing.assert_array_equal(values[7:], np.float32(0.2))
    inventory_count, transfer_count = len(files.inventory_calls), len(files.downloaded)
    repeated = pipeline.pull_range(date(1999, 12, 31), date(2000, 1, 1), ["mtpr"], archive)
    assert repeated == {"completed": 0, "skipped": 48, "unavailable": 0, "failed": 0, "pending": 0}
    assert (len(files.inventory_calls), len(files.downloaded)) == (inventory_count, transfer_count)


def test_availability_is_per_variable_and_missing_hours_are_reconsidered(
    tmp_path: Path, files: NcarFiles,
) -> None:
    """A later inventory fills previously unavailable hours without fetching stored fields."""
    archive = tmp_path / "archive"
    files.add_file("2t", "2000-01-01T00", "2000-01-02T23")
    files.add_file("10u", "2000-01-01T00", "2000-01-01T23", value=5)
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 2), ["2t", "10u"], archive)
    assert result == {"completed": 72, "skipped": 0, "unavailable": 24, "failed": 0, "pending": 0}
    newly_available = files.add_file("10u", "2000-01-02T00", "2000-01-02T23", value=6)
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 2), ["2t", "10u"], archive)
    assert result == {"completed": 24, "skipped": 72, "unavailable": 0, "failed": 0, "pending": 0}
    assert len(files.downloaded) == 3
    assert files.downloaded[-1] == (newly_available.key, newly_available.etag)
    dataset = read_raster_values(archive, 2000, 1)
    np.testing.assert_array_equal(dataset["2_metre_temperature"], np.float32(280))
    np.testing.assert_array_equal(dataset["10_metre_u_wind_component"][:24], np.float32(5))
    np.testing.assert_array_equal(dataset["10_metre_u_wind_component"][24:], np.float32(6))


@pytest.mark.parametrize("local_staging", [False, True])
def test_add_variable_then_refresh_across_months_preserves_unselected_pending_work(
    tmp_path: Path, files: NcarFiles, local_staging: bool,
) -> None:
    """Refresh temperature while preserving published and separately staged wind."""
    archive = tmp_path / "archive"
    for day in ("1999-12-31", "2000-01-01"):
        files.add_file("2t", f"{day}T00", f"{day}T23")
        files.add_file("10u", f"{day}T00", f"{day}T23", value=5)
    first, last = date(1999, 12, 31), date(2000, 1, 1)
    pipeline.pull_range(first, last, ["2t"], archive, staging_dir=tmp_path / "ssd" if local_staging else None)
    added = pipeline.pull_range(first, last, ["10u"], archive)
    assert added["completed"] == 48
    for day in ("1999-12-31", "2000-01-01"):
        files.add_file("2t", f"{day}T00", f"{day}T23", version=2, value=290)
    pending_wind = files.add_file("10u", "2000-01-01T00", "2000-01-01T23", version=2, value=10)
    with catalog.archive_session(archive) as db:
        rows = catalog.register_sources(db, archive, [pending_wind], last, last)
        files.download(db, [pending_wind], rows, archive=archive)
        wind_id, wind_path = rows[0]["id"], ((catalog.get_staging_dir(db) or archive) / rows[0]["path"])
    refreshed = pipeline.pull_range(first, last, ["2t"], archive, refresh=True)
    assert refreshed == {"completed": 48, "skipped": 0, "unavailable": 0, "failed": 0, "pending": 0}
    for year, month in ((1999, 12), (2000, 1)):
        dataset = read_raster_values(archive, year, month)
        np.testing.assert_array_equal(dataset["2_metre_temperature"], np.float32(290))
        np.testing.assert_array_equal(dataset["10_metre_u_wind_component"], np.float32(5))
    assert wind_path.exists()
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM source_selections WHERE source_id = ?", (wind_id,)).fetchone()[0] == 1


def test_refresh_requires_complete_selected_availability_before_downloading(
    tmp_path: Path, files: NcarFiles,
) -> None:
    """Keep the published month unchanged when one selected variable is unavailable."""
    archive = tmp_path / "archive"
    files.add_file("2t", "2000-01-01T00", "2000-01-02T23")
    first, last = date(2000, 1, 1), date(2000, 1, 2)
    pipeline.pull_range(first, last, ["2t"], archive)
    files.add_file("2t", "2000-01-01T00", "2000-01-02T23", version=2, value=290)
    files.add_file("10u", "2000-01-01T00", "2000-01-01T23", value=5)
    with pytest.raises(ValueError, match="refresh requires every selected hour"):
        pipeline.pull_range(first, last, ["2t", "10u"], archive, refresh=True)
    assert len(files.downloaded) == 1
    dataset = read_raster_values(archive, 2000, 1)
    np.testing.assert_array_equal(dataset["2_metre_temperature"], np.float32(280))
    assert "10_metre_u_wind_component" not in dataset
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 48


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
@pytest.mark.parametrize("completed_january_hours", [0, 6])
def test_failed_second_month_resumes_recorded_download_without_inventory(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], failure: type[BaseException], completed_january_hours: int,
) -> None:
    """Keep the committed first month and reuse the second month's source on retry."""
    archive = tmp_path / "archive"
    files.add_file("2t", "1999-12-31T00", "1999-12-31T23")
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    publish = pipeline.publish_observations

    def fail_january(
        db: sqlite3.Connection, destination: Path, observations: list[tuple[str, dict[str, Any]]],
    ) -> list[Path]:
        """Fail at January's publication boundary after decoding has completed."""
        valid_time = db.execute("SELECT valid_time FROM publications WHERE id = ?", (observations[0][0],)).fetchone()[0]
        completed = db.execute("SELECT COUNT(*) FROM observations WHERE valid_time >= 946684800").fetchone()[0]
        if valid_time >= 946684800 and completed == completed_january_hours:
            raise failure("January publication interrupted")
        return publish(db, destination, observations)

    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "publish_observations", fail_january)
        with pytest.raises(failure, match="January publication interrupted"):
            pipeline.pull_range(date(1999, 12, 31), date(2000, 1, 1), ["2t"], archive)
    output = capsys.readouterr().out
    assert f"completed={24 + completed_january_hours}" in output
    assert f"failed={24 - completed_january_hours}" in output
    dataset = read_raster_values(archive, 1999, 12)
    assert dataset["2_metre_temperature"].shape == (24, 3, 4)
    inventory_count = len(files.inventory_calls)
    result = pipeline.pull_range(date(1999, 12, 31), date(2000, 1, 1), ["2t"], archive)
    assert result == {"completed": 24 - completed_january_hours, "skipped": 24 + completed_january_hours, "unavailable": 0, "failed": 0, "pending": 0}
    assert len(files.downloaded) == 2
    assert len(files.inventory_calls) == inventory_count


def test_source_limit_counts_retained_files_before_starting_transfers(
    tmp_path: Path, files: NcarFiles,
) -> None:
    """Reject a source budget that cannot hold pending files and the next month."""
    archive = tmp_path / "archive"
    temperature = files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    wind = files.add_file("10u", "2000-02-01T00", "2000-02-01T23", value=5)
    with catalog.archive_session(archive) as db:
        rows = catalog.register_sources(db, archive, [wind], date(2000, 2, 1), date(2000, 2, 1))
        files.download(db, [wind], rows, archive=archive)
        wind_path = (archive / rows[0]["path"])
    limit = (temperature.size + wind.size - 1) / 10**9
    with pytest.raises(ValueError, match="source|staging|bytes"):
        pipeline.pull_range(
            date(2000, 1, 1), date(2000, 1, 1), ["2t"], archive, max_source_gb=limit,
        )
    assert files.downloaded == [(wind.key, wind.etag)]
    assert wind_path.exists()
    assert not (archive / "2000" / "01").exists()


@pytest.mark.parametrize("published_days,selected_days", [(1, 2), (1, 3)])
def test_pull_consumes_already_published_pending_hours_within_its_selection(
    tmp_path: Path, files: NcarFiles, published_days: int, selected_days: int,
) -> None:
    """Consume repeated download selections while retaining unrequested later hours."""
    archive = tmp_path / "archive"
    source = files.add_file("2t", "2000-01-01T00", "2000-01-03T23")
    first, last = date(2000, 1, 1), date(2000, 1, 2)
    pipeline.pull_range(first, date(2000, 1, published_days), ["2t"], archive)
    with catalog.archive_session(archive) as db:
        rows = catalog.register_sources(db, archive, [source], first, date(2000, 1, selected_days))
        files.download(db, [source], rows, archive=archive)
        source_path = (archive / rows[0]["path"])
    inventory_count, transfer_count = len(files.inventory_calls), len(files.downloaded)
    result = pipeline.pull_range(first, last, ["2t"], archive)
    assert result == {
        "completed": (2 - published_days) * 24, "skipped": published_days * 24,
        "unavailable": 0, "failed": 0, "pending": 0,
    }
    assert (len(files.inventory_calls), len(files.downloaded)) == (inventory_count, transfer_count)
    with catalog.archive_session(archive) as db:
        pending = [tuple(row) for row in db.execute(
            "SELECT start_time, end_time FROM source_selections ORDER BY start_time"
        )]
    if selected_days == 3:
        assert pending == [("2000-01-03T00:00:00+00:00", "2000-01-03T23:00:00+00:00")]
        assert source_path.exists()
    else:
        assert pending == []
        assert not source_path.exists()


def test_shared_source_is_not_retained_for_already_published_next_month(
    tmp_path: Path, files: NcarFiles,
) -> None:
    """Catch up an older month without retaining inputs for already stored later hours."""
    archive = tmp_path / "archive"
    shared = files.add_file("mtpr", "1999-12-30T18", "1999-12-31T18", value=0.1)
    files.add_file("mtpr", "2000-01-01T06", "2000-01-01T18", value=0.2)
    pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["mtpr"], archive)
    result = pipeline.pull_range(date(1999, 12, 31), date(2000, 1, 1), ["mtpr"], archive)
    assert result["completed"] == result["skipped"] == 24
    with catalog.archive_session(archive) as db:
        row = db.execute("SELECT path, downloaded_at FROM source_files WHERE object_key = ?", (shared.key,)).fetchone()
        assert row["downloaded_at"] is None
        assert not (archive / row["path"]).exists()
        assert db.execute("SELECT COUNT(*) FROM source_selections").fetchone()[0] == 0


@pytest.mark.parametrize("workers", [1, 3])
def test_publication_overlaps_downloads_with_bounded_workers(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch, workers: int,
) -> None:
    """Publish while another transfer is blocked and bound concurrent COG work."""
    archive = tmp_path / "archive"
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    files.add_file("10u", "2000-01-01T00", "2000-01-01T23")
    first_publication = Event()
    slow_download = Event()
    worker_barrier = Barrier(workers)
    lock = Lock()
    active = peak = 0
    transfer_count = 0
    encode_count = 0
    owner = get_ident()
    writer_threads: set[int] = set()
    catalog_threads: set[int] = set()
    write = pipeline.write_ncar_chunk_cogs
    publish = pipeline.publish_observations
    prepare = pipeline.prepare_observations

    def transfer(source: SourceFile, target: Path, **kwargs: Any) -> None:
        """Hold the second transfer until the first source has a published COG."""
        nonlocal transfer_count
        transfer_count += 1
        if transfer_count == 2:
            slow_download.set()
            assert first_publication.wait(10), "processing waited for all downloads"
        files.transfer(source, target, **kwargs)

    def encode(*args: Any) -> list[dict[str, Any]]:
        """Synchronize the first jobs so concurrency and its limit are observable."""
        nonlocal active, peak, encode_count
        with lock:
            active += 1
            peak = max(peak, active)
            writer_threads.add(get_ident())
            encode_count += 1
            ordinal = encode_count
        try:
            if ordinal <= workers:
                worker_barrier.wait(timeout=10)
            return write(*args)
        finally:
            with lock:
                active -= 1

    def prepare_on_owner(db: sqlite3.Connection, *args: Any) -> list[tuple[str, Path]]:
        """Prepare source journals on the dedicated catalog owner."""
        assert get_ident() != owner
        catalog_threads.add(get_ident())
        return prepare(db, *args)

    def publish_on_owner(db: sqlite3.Connection, *args: Any) -> list[Path]:
        """Require a real durable observation before releasing the pending transfer."""
        assert get_ident() != owner
        catalog_threads.add(get_ident())
        assert slow_download.wait(10)
        assert len(list((archive / "staging" / "observations").glob("*.tif"))) <= 2 * workers * 6
        result = publish(db, *args)
        first_publication.set()
        return result

    monkeypatch.setattr(ncar_download, "download_file", transfer)
    monkeypatch.setattr(pipeline, "write_ncar_chunk_cogs", encode)
    monkeypatch.setattr(pipeline, "prepare_observations", prepare_on_owner)
    monkeypatch.setattr(pipeline, "publish_observations", publish_on_owner)
    result = pipeline.pull_range(
        date(2000, 1, 1), date(2000, 1, 1), ["2t", "10u"], archive,
        download_workers=1, processing_workers=workers,
        max_source_gb=sum(source.size for source in files.sources) / 10**9,
    )
    assert result["completed"] == 48
    assert peak == workers and len(writer_threads) == workers
    assert len(catalog_threads) == 1 and not catalog_threads & writer_threads
    assert not (archive / "staging" / "decoded" / "values.bin").exists()


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_processing_failure_stops_downloads_and_waits_for_chunk_readers(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
) -> None:
    """Retain sources and wait for active chunk readers before returning."""
    archive = tmp_path / "archive"
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    files.add_file("10u", "2000-01-01T00", "2000-01-01T23")
    reader_started = Event()
    download_stopped = Event()
    reader_finished = Event()
    write = pipeline.write_ncar_chunk_cogs
    calls: list[SourceFile] = []

    def transfer(source: SourceFile, target: Path, *, stop_event: Event, **kwargs: Any) -> None:
        """Keep a partial source downloading until publication fails."""
        calls.append(source)
        if len(calls) == 1:
            files.transfer(source, target, **kwargs)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(files.payloads[source.key, source.etag].read_bytes()[:17])
            assert stop_event.wait(10)
            download_stopped.set()
            raise CancelledError()

    def encode(*args: Any) -> list[dict[str, Any]]:
        """Keep one worker reading the source during the coordinator's shutdown."""
        if args[1].fields[0]["valid_time"] == 946684800:
            assert reader_started.wait(10)
            return write(*args)
        reader_started.set()
        assert download_stopped.wait(10)
        assert args[0].exists(), "source deleted while a worker still needed it"
        result = write(*args)
        reader_finished.set()
        return result

    def fail_publication(*args: Any) -> list[Path]:
        """Interrupt installation of a validated output."""
        raise failure("publication interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(ncar_download, "download_file", transfer)
        patch.setattr(pipeline, "write_ncar_chunk_cogs", encode)
        patch.setattr(pipeline, "publish_observations", fail_publication)
        with pytest.raises(failure, match="publication interrupted"):
            pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t", "10u"], archive,
                                download_workers=1, processing_workers=2)
    assert download_stopped.is_set() and reader_finished.is_set()
    assert not (archive / "staging" / "decoded" / "values.bin").exists()
    with catalog.archive_session(archive) as db:
        rows = db.execute("SELECT * FROM source_files ORDER BY downloaded_at DESC").fetchall()
        assert rows[0]["downloaded_at"] is not None
        assert rows[1]["downloaded_at"] is None
        assert (archive / rows[1]["path"]).stat().st_size == 17
        assert all((archive / row["path"]).exists() for row in rows)
    resumed = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t", "10u"], archive,
                                  processing_workers=2)
    assert resumed["completed"] == 48
    assert len(files.downloaded) == 2  # Completed source reused, partial source retried.


@pytest.mark.parametrize("workers", [0, -1])
def test_processing_worker_limit_rejected_before_archive_changes(tmp_path: Path, workers: int) -> None:
    """Reject an unusable pool size before starting any work."""
    archive = tmp_path / "archive"
    with pytest.raises(ValueError, match="processing workers"):
        pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], archive,
                            processing_workers=workers)
    assert not archive.exists()


def test_download_slots_refill_during_netcdf_decoding(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep transferring new inputs while a worker is busy decoding a source."""
    files.add_file("10u", "2000-01-01T00", "2000-01-01T23")
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    files.add_file("2t", "2000-01-02T00", "2000-01-02T23")
    decoding = Event()
    last_download = Event()
    decode = pipeline.write_ncar_chunk_cogs
    calls = 0

    def transfer(source: SourceFile, target: Path, **kwargs: Any) -> None:
        """Require the first decode to be active before finishing later transfers."""
        nonlocal calls
        calls += 1
        if calls > 1:
            assert decoding.wait(10)
        files.transfer(source, target, **kwargs)
        if calls == 3:
            last_download.set()

    def decode_after_downloads(*args: Any) -> list[dict[str, Any]]:
        """Wait for a refilled download slot before releasing the decoder."""
        decoding.set()
        assert last_download.wait(10), "downloads were not refilled during decoding"
        return decode(*args)

    monkeypatch.setattr(ncar_download, "download_file", transfer)
    monkeypatch.setattr(pipeline, "write_ncar_chunk_cogs", decode_after_downloads)
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 2), ["2t", "10u"],
                                 tmp_path / "archive", download_workers=1, processing_workers=2)
    assert result["completed"] == 72 and calls == 3


def test_cog_worker_failure_keeps_committed_hours_and_source_for_resume(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume remaining observations after a worker fails to write its COG."""
    archive = tmp_path / "archive"
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    write = pipeline.write_ncar_chunk_cogs
    committed = Event()
    publish = pipeline.publish_observations

    def publish_first(*args: Any) -> list[Path]:
        """Acknowledge the first durable batch before injecting a later failure."""
        result = publish(*args)
        committed.set()
        return result


    def fail_second_chunk(*args: Any) -> list[dict[str, Any]]:
        """Fail an output write after one durably published native chunk."""
        if args[1].fields[0]["valid_time"] == 946684800 + 6 * 3600:
            assert committed.wait(10)
            raise OSError("COG storage unavailable")
        return write(*args)

    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "publish_observations", publish_first)
        patch.setattr(pipeline, "write_ncar_chunk_cogs", fail_second_chunk)
        with pytest.raises(OSError, match="COG storage unavailable"):
            pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], archive,
                                processing_workers=1)
    before = {path: path.stat().st_mtime_ns for path in archive.glob("2000/01/*/*.tif")}
    assert len(before) == 6
    with catalog.archive_session(archive) as db:
        source = db.execute("SELECT * FROM source_files").fetchone()
        assert source["downloaded_at"] is not None and (archive / source["path"]).exists()
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], archive,
                                 processing_workers=2)
    assert result["completed"] == 18 and result["skipped"] == 6
    assert len(files.downloaded) == 1
    assert before == {path: path.stat().st_mtime_ns for path in before}


@pytest.mark.parametrize("local_staging", [False, True])
@pytest.mark.parametrize("fits", [True, False])
def test_next_month_prefetch_overlaps_processing_only_within_source_budget(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch, fits: bool, local_staging: bool,
) -> None:
    """Prefetch while January processes, or wait for its source bytes to be freed."""
    archive = tmp_path / "archive"
    january = files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    february = files.add_file("2t", "2000-02-01T00", "2000-02-01T23")
    with catalog.archive_session(archive) as db:
        rows = catalog.register_sources(db, archive, [january], date(2000, 1, 1), date(2000, 1, 1))
        files.download(db, [january], rows, archive=archive)
    processing = Event()
    prefetched = Event()
    january_done = Event()
    write = pipeline.write_ncar_chunk_cogs
    ingest = pipeline.download_and_ingest_month
    budget = january.size + february.size - (0 if fits else 1)

    def transfer(source: SourceFile, target: Path, **kwargs: Any) -> None:
        """Require overlap when budgeted, and defer transfer when it would overflow."""
        assert source.key == february.key
        if fits:
            assert processing.wait(10)
            assert not january_done.is_set()
        else:
            assert january_done.is_set()
        with sqlite3.connect(archive / "catalog.sqlite") as db:
            reserved = db.execute("SELECT SUM(size) FROM source_files WHERE removed_at IS NULL").fetchone()[0]
            assert reserved <= budget
        files.transfer(source, target, **kwargs)
        prefetched.set()

    def encode(*args: Any) -> list[dict[str, Any]]:
        """Keep January decoding alive until the budgeted next-month transfer finishes."""
        if args[1].fields[0]["valid_time"] < 949363200:
            processing.set()
            if fits:
                assert prefetched.wait(10), "next-month downloads waited for processing"
        return write(*args)

    def month_finished(db: pipeline.PublicationWorker, target: Path, year: int, month: int, *args: Any, **kwargs: Any) -> Path:
        """Observe the durable month boundary before the next month starts."""
        result = ingest(db, target, year, month, *args, **kwargs)
        if month == 1:
            january_done.set()
        return result

    monkeypatch.setattr(ncar_download, "download_file", transfer)
    monkeypatch.setattr(pipeline, "write_ncar_chunk_cogs", encode)
    monkeypatch.setattr(pipeline, "download_and_ingest_month", month_finished)
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 2, 1), ["2t"], archive,
                                 download_workers=1, processing_workers=2, max_source_gb=budget / 1e9,
                                 staging_dir=tmp_path / "ssd" if local_staging else None)
    assert result["completed"] == 48
    assert prefetched.is_set()
    assert files.downloaded.count((february.key, february.etag)) == 1


def test_decoded_byte_budget_limits_concurrent_native_chunks(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Limit decoded arrays by bytes even when more process slots are available."""
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    write = pipeline.write_ncar_chunk_cogs
    barrier = Barrier(2)
    lock = Lock()
    active = peak = calls = 0
    byte_budget = 2 * 6 * 3 * 4 * 4  # Two six-hour float32 fixture chunks.

    def encode(*args: Any) -> list[dict[str, Any]]:
        """Measure the actual live decoded reservations at the worker boundary."""
        nonlocal active, peak, calls
        with lock:
            active += args[1].decoded_bytes
            peak = max(peak, active)
            calls += 1
            ordinal = calls
        try:
            assert active <= byte_budget
            if ordinal <= 2:
                barrier.wait(timeout=10)
            return write(*args)
        finally:
            with lock:
                active -= args[1].decoded_bytes

    monkeypatch.setattr(pipeline, "write_ncar_chunk_cogs", encode)
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], tmp_path / "archive",
                                 processing_workers=4, max_decoded_gb=byte_budget / 1e9)
    assert result["completed"] == 24 and calls == 4 and peak == byte_budget


def test_chunk_larger_than_decoded_budget_retains_source_for_retry(tmp_path: Path, files: NcarFiles) -> None:
    """Reject an undersized decoding budget before staging observations."""
    archive = tmp_path / "archive"
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    with pytest.raises(ValueError, match="increase --max-decoded-gb"):
        pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], archive,
                            max_decoded_gb=(6 * 3 * 4 * 4 - 1) / 1e9)
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 0
        row = db.execute("SELECT path,downloaded_at FROM source_files").fetchone()
        assert row["downloaded_at"] and (archive / row["path"]).exists()
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], archive)
    assert result["completed"] == 24 and len(files.downloaded) == 1


def test_prefetched_sources_survive_interrupted_current_month(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse prefetched next-month bytes after current-month publication fails."""
    archive = tmp_path / "archive"
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    february = files.add_file("2t", "2000-02-01T00", "2000-02-01T23")
    prefetched = Event()
    publish = pipeline.publish_observations
    count = 0

    def transfer(source: SourceFile, target: Path, **kwargs: Any) -> None:
        """Signal completion of next month's retained input."""
        files.transfer(source, target, **kwargs)
        if source.key == february.key:
            prefetched.set()

    def interrupt(db: sqlite3.Connection, *args: Any) -> list[Path]:
        """Fail after one committed chunk and a completed prefetch."""
        nonlocal count
        count += 1
        if count >= 2:
            assert prefetched.wait(10)
            raise KeyboardInterrupt("stop current month")
        return publish(db, *args)

    with monkeypatch.context() as patch:
        patch.setattr(ncar_download, "download_file", transfer)
        patch.setattr(pipeline, "publish_observations", interrupt)
        with pytest.raises(KeyboardInterrupt, match="stop current month"):
            pipeline.pull_range(date(2000, 1, 1), date(2000, 2, 1), ["2t"], archive)
    with catalog.archive_session(archive) as db:
        row = db.execute("SELECT path,downloaded_at FROM source_files WHERE object_key=?", (february.key,)).fetchone()
        assert row["downloaded_at"] is not None and (archive / row["path"]).exists()
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 2, 1), ["2t"], archive)
    assert result["completed"] == 42 and result["skipped"] == 6
    assert files.downloaded.count((february.key, february.etag)) == 1


@pytest.mark.parametrize("february_complete", [False, True])
def test_lookahead_fills_spare_budget_beyond_one_future_month(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch, february_complete: bool,
) -> None:
    """Download March during January processing, including past a completed month."""
    archive = tmp_path / "archive"
    for month in (1, 2, 3):
        last_day = 29 if month == 2 and february_complete else 1
        files.add_file("2t", f"2000-{month:02d}-01T00", f"2000-{month:02d}-{last_day:02d}T23")
    if february_complete:
        pipeline.pull_range(date(2000, 2, 1), date(2000, 2, 29), ["2t"], archive)
    march_ready = Event()
    write = pipeline.write_ncar_chunk_cogs

    def transfer(source: SourceFile, target: Path, **kwargs: Any) -> None:
        """Signal the second future month's successful transfer."""
        files.transfer(source, target, **kwargs)
        if source.valid_start.month == 3:
            march_ready.set()

    def encode(*args: Any) -> list[dict[str, Any]]:
        """Keep current processing active while lookahead advances twice."""
        if args[1].fields[0]["valid_time"] < 949363200:
            assert march_ready.wait(10), "lookahead stopped at a fixed one-month barrier"
        return write(*args)

    monkeypatch.setattr(ncar_download, "download_file", transfer)
    monkeypatch.setattr(pipeline, "write_ncar_chunk_cogs", encode)
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 3, 1), ["2t"], archive,
                                 processing_workers=2, max_source_gb=sum(source.size for source in files.sources) / 1e9)
    assert march_ready.is_set() and result["completed"] == (48 if february_complete else 72)
    assert len(files.downloaded) == 3


def test_prefetch_transfer_continues_across_publication_month_boundary(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Advance the processing month without stopping or waiting for its download."""
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    february = files.add_file("2t", "2000-02-01T00", "2000-02-01T23")
    january_done = Event()
    transfer_started = Event()
    ingest = pipeline.download_and_ingest_month

    def transfer(source: SourceFile, target: Path, *, stop_event: Event, **kwargs: Any) -> None:
        """Hold February's active transfer through January's completion."""
        if source.key == february.key:
            transfer_started.set()
            assert january_done.wait(10), "current month waited for a future transfer"
            assert not stop_event.is_set(), "transfer was stopped at the month boundary"
        files.transfer(source, target, **kwargs)

    def finished(db: pipeline.PublicationWorker, archive: Path, year: int, month: int, *args: Any, **kwargs: Any) -> Path:
        """Signal the published month without requiring future transfers to finish."""
        result = ingest(db, archive, year, month, *args, **kwargs)
        if month == 1:
            assert transfer_started.wait(10)
            january_done.set()
        return result

    monkeypatch.setattr(ncar_download, "download_file", transfer)
    monkeypatch.setattr(pipeline, "download_and_ingest_month", finished)
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 2, 1), ["2t"], tmp_path / "archive",
                                 processing_workers=2)
    assert result["completed"] == 48
    assert files.downloaded.count((february.key, february.etag)) == 1


def test_future_inventory_does_not_hold_up_next_month_processing(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process February while March's inventory remains blocked in another thread."""
    for month in (1, 2, 3):
        files.add_file("2t", f"2000-{month:02d}-01T00", f"2000-{month:02d}-01T23")
    march_lookup = Event()
    february_processing = Event()
    write = pipeline.write_ncar_chunk_cogs

    def discover(first: date, last: date, names: Sequence[str], session: requests.Session) -> list[SourceFile]:
        """Hold the future lookup until the next month has actually started work."""
        if first.month == 3:
            march_lookup.set()
            assert february_processing.wait(10), "future inventory blocked processing"
        return files.discover(first, last, names, session)

    def encode(*args: Any) -> list[dict[str, Any]]:
        """Observe February decoding before permitting March availability to return."""
        month = datetime.fromtimestamp(args[1].fields[0]["valid_time"], UTC).month
        if month == 1:
            assert march_lookup.wait(10)
        elif month == 2:
            february_processing.set()
        return write(*args)

    monkeypatch.setattr(pipeline, "discover_files", discover)
    monkeypatch.setattr(pipeline, "write_ncar_chunk_cogs", encode)
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 3, 1), ["2t"], tmp_path / "archive",
                                 processing_workers=2)
    assert result["completed"] == 72 and february_processing.is_set()
    assert len(files.downloaded) == 3


def test_refresh_does_not_prefetch_an_incomplete_future_selection(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve refresh's complete-month availability check during lookahead."""
    january = files.add_file("2t", "2000-01-31T00", "2000-01-31T23")
    files.add_file("2t", "2000-02-01T00", "2000-02-01T22")
    prefetch_checked = Event()
    prefetch = pipeline._prefetch_month_sources
    write = pipeline.write_ncar_chunk_cogs

    def inspect_prefetch(*args: Any, **kwargs: Any) -> tuple[list[SourceFile], list[sqlite3.Row], bool]:
        """Allow current work to finish after future source admission is evaluated."""
        result = prefetch(*args, **kwargs)
        prefetch_checked.set()
        return result

    def encode(*args: Any) -> list[dict[str, Any]]:
        """Keep processing open until next-month prefetch has been considered."""
        assert prefetch_checked.wait(10)
        return write(*args)

    monkeypatch.setattr(pipeline, "_prefetch_month_sources", inspect_prefetch)
    monkeypatch.setattr(pipeline, "write_ncar_chunk_cogs", encode)
    with pytest.raises(ValueError, match="refresh requires every selected hour"):
        pipeline.pull_range(date(2000, 1, 31), date(2000, 2, 1), ["2t"], tmp_path / "archive",
                            refresh=True, processing_workers=2)
    assert files.downloaded == [(january.key, january.etag)]


def test_blocked_publication_keeps_decoding_and_transfer_refills_bounded(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continue conversion and transfers while the writer fills its bounded queue."""
    archive = tmp_path / "archive"
    files.add_file("2t", "2000-01-01T00", "2000-01-02T23")
    files.add_file("10u", "2000-01-01T00", "2000-01-01T23")
    files.add_file("10u", "2000-01-02T00", "2000-01-02T23")
    publication_started = Event()
    four_encoded = Event()
    last_download = Event()
    checked = Event()
    lock = Lock()
    encoded = transfers = 0
    encode = pipeline.write_ncar_chunk_cogs
    publish = pipeline.publish_observations

    def transfer(source: SourceFile, target: Path, **kwargs: Any) -> None:
        """Require both later transfers to refill while publication is blocked."""
        nonlocal transfers
        transfers += 1
        if transfers > 1:
            assert publication_started.wait(10)
        files.transfer(source, target, **kwargs)
        if transfers == 3:
            last_download.set()

    def write(*args: Any) -> list[dict[str, Any]]:
        """Signal that four batches encoded with only one chunk's decoded memory."""
        nonlocal encoded
        result = encode(*args)
        with lock:
            encoded += 1
            if encoded == 4:
                four_encoded.set()
        return result

    def blocked_publish(db: sqlite3.Connection, *args: Any) -> list[Path]:
        """Hold final installation until independent stages progress to their bounds."""
        if not checked.is_set():
            publication_started.set()
            assert four_encoded.wait(10), "decoding slots waited for publication"
            assert last_download.wait(10), "download slots waited for catalog commits"
            Event().wait(0.15)  # Give scheduling an opportunity to exceed the bound.
            assert encoded == 4
            assert len(list((archive / "staging/observations").glob("*.tif"))) == 24
            assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
            assert db.execute("SELECT SUM(size) FROM source_files WHERE removed_at IS NULL").fetchone()[0] <= sum(s.size for s in files.sources)
            checked.set()
        return publish(db, *args)

    monkeypatch.setattr(ncar_download, "download_file", transfer)
    monkeypatch.setattr(pipeline, "write_ncar_chunk_cogs", write)
    monkeypatch.setattr(pipeline, "publish_observations", blocked_publish)
    result = pipeline.pull_range(
        date(2000, 1, 1), date(2000, 1, 2), ["2t", "10u"], archive,
        download_workers=1, processing_workers=2, max_decoded_gb=6 * 3 * 4 * 4 / 1e9,
        max_source_gb=sum(source.size for source in files.sources) / 1e9,
    )
    assert checked.is_set() and result["completed"] == 96
