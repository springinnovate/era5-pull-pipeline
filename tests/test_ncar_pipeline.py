"""Exercise bulk NCAR scheduling through real NetCDF, COG, and SQLite files."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
import shutil
import sqlite3
from typing import Any

import numpy as np
import pytest
import requests
import xarray as xr

from archive_assertions import read_raster_values

from era5_pull_pipeline import catalog, conversion, ncar_ingest, pipeline
from era5_pull_pipeline.ncar import SourceFile


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
            target = (archive / row["path"])
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
    monkeypatch.setattr(pipeline, "download_registered_sources", fixture.download)
    return fixture


def test_year_boundary_reuses_forecast_file_and_completed_repeat_is_offline(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain the shared forecast file through both months and then remove it."""
    archive = tmp_path / "archive"
    shared = files.add_file("mtpr", "1999-12-30T18", "1999-12-31T18", value=0.1)
    following = files.add_file("mtpr", "2000-01-01T06", "2000-01-01T18", value=0.2)
    ingest = pipeline.ingest_ncar_month
    source_after_december: list[Path] = []

    def inspect_published_month(
        db: sqlite3.Connection, destination: Path, year: int, month: int,
        **kwargs: Any,
    ) -> Path:
        """Check the shared source immediately after December publication."""
        result = ingest(db, destination, year, month, **kwargs)
        if (year, month) == (1999, 12):
            source = db.execute("SELECT path FROM source_files WHERE object_key = ?", (shared.key,)).fetchone()
            source_after_december.append((destination / source["path"]))
            assert source_after_december[0].exists()
        return result

    monkeypatch.setattr(pipeline, "ingest_ncar_month", inspect_published_month)
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


def test_add_variable_then_refresh_across_months_preserves_unselected_pending_work(
    tmp_path: Path, files: NcarFiles,
) -> None:
    """Refresh temperature while preserving published and separately staged wind."""
    archive = tmp_path / "archive"
    for day in ("1999-12-31", "2000-01-01"):
        files.add_file("2t", f"{day}T00", f"{day}T23")
        files.add_file("10u", f"{day}T00", f"{day}T23", value=5)
    first, last = date(1999, 12, 31), date(2000, 1, 1)
    pipeline.pull_range(first, last, ["2t"], archive)
    added = pipeline.pull_range(first, last, ["10u"], archive)
    assert added["completed"] == 48
    for day in ("1999-12-31", "2000-01-01"):
        files.add_file("2t", f"{day}T00", f"{day}T23", version=2, value=290)
    pending_wind = files.add_file("10u", "2000-01-01T00", "2000-01-01T23", version=2, value=10)
    with catalog.archive_session(archive) as db:
        rows = catalog.register_sources(db, archive, [pending_wind], last, last)
        files.download(db, [pending_wind], rows, archive=archive)
        wind_id, wind_path = rows[0]["id"], (archive / rows[0]["path"])
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
@pytest.mark.parametrize("completed_january_hours", [0, 2])
def test_failed_second_month_resumes_recorded_download_without_inventory(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], failure: type[BaseException], completed_january_hours: int,
) -> None:
    """Keep the committed first month and reuse the second month's source on retry."""
    archive = tmp_path / "archive"
    files.add_file("2t", "1999-12-31T00", "1999-12-31T23")
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    publish = ncar_ingest.publish_observation

    def fail_january(
        db: sqlite3.Connection, destination: Path, revision: str, report: dict[str, Any],
    ) -> Path:
        """Fail at January's publication boundary after decoding has completed."""
        if db.execute("SELECT valid_time FROM publications WHERE id = ?", (revision,)).fetchone()[0] == 946684800 + completed_january_hours * 3600:
            raise failure("January publication interrupted")
        return publish(db, destination, revision, report)

    with monkeypatch.context() as patch:
        patch.setattr(ncar_ingest, "publish_observation", fail_january)
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
