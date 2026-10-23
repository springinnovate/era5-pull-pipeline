"""Exercise NCAR ingestion with real NetCDF, COG, and SQLite files."""

from contextlib import nullcontext
from datetime import UTC, date, datetime
from pathlib import Path
import sqlite3
from unittest.mock import Mock
from typing import Any

from archive_assertions import read_raster_values

import numpy as np
from numpy.typing import NDArray
import pytest
import xarray as xr

from era5_pull_pipeline import catalog, conversion, ncar_ingest
from era5_pull_pipeline.ncar import SourceFile


@pytest.fixture(autouse=True)
def small_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep actual storage tests small while retaining coordinate validation."""
    monkeypatch.setattr(conversion, "GLOBAL_LATITUDES", np.array([90.0, 89.75, 89.5]))
    monkeypatch.setattr(conversion, "GLOBAL_LONGITUDES", np.array([0.0, 0.25, 0.5, 0.75]))


def _stage_analysis(
    archive: Path, variable: str = "2t", *,
    first_day: int = 1, last_day: int = 1, version: int = 1, year: int = 2000,
    offset: float = 0.0, omit_hour: int | None = None, complete: bool = True,
) -> tuple[sqlite3.Row, NDArray[np.float32]]:
    """Create a completed two-day provider file with separately selected dates.

    Args:
        archive: Archive receiving the provider file and catalog selection.
        variable: Temperature or wind source name.
        first_day: First selected January day.
        last_day: Last selected January day.
        version: Distinct provider revision for refresh fixtures.
        year: Calendar year for the January source and selected dates.
        offset: Value added before the source is recorded as complete.
        omit_hour: Optional missing external source hour for coverage failure.
        complete: False leaves half the file on disk as an interrupted download.

    Returns:
        Completed source catalog row and the original two days of values.
    """
    names = {
        "2t": ("2 metre temperature", "K"),
        "10u": ("10 metre U wind component", "m s**-1"),
    }
    source = SourceFile(
        key=f"e5.oper.an.sfc/{year:04d}01/e5.oper.an.sfc.{variable}.nc",
        variable=variable, size=1, etag=f'"version-{version}"',
        last_modified=f"2026-01-{version:02d}T00:00:00+00:00",
        valid_start=datetime(year, 1, 1, tzinfo=UTC),
        valid_end=datetime(year, 1, 2, 23, tzinfo=UTC),
    )
    with catalog.archive_session(archive) as db:
        row = catalog.register_sources(
            db, archive, [source], date(year, 1, first_day), date(year, 1, last_day),
        )[0]
        path = archive / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.arange(48 * 3 * 4, dtype=np.float32).reshape(48, 3, 4)
        values = values * np.float32(0.00123) + np.float32(offset)
        values[0, 0, 0] = np.nan
        dataset = xr.Dataset(
            {variable.upper(): (("time", "latitude", "longitude"), values, {
                "long_name": names[variable][0], "short_name": variable,
                "units": names[variable][1], "number_of_significant_digits": np.int32(7),
            })},
            coords={
                "time": np.arange(f"{year:04d}-01-01T00", f"{year:04d}-01-03T00", dtype="datetime64[h]"),
                "latitude": conversion.GLOBAL_LATITUDES,
                "longitude": conversion.GLOBAL_LONGITUDES,
            },
            attrs={"history": "NCAR fixture"},
        )
        if omit_hour is not None:
            dataset = dataset.isel(time=[hour for hour in range(48) if hour != omit_hour])
        dataset.to_netcdf(path, engine="h5netcdf", encoding={
            variable.upper(): {
                "chunksizes": (6, 2, 3), "zlib": True, "_FillValue": np.float32(9.999e20),
            },
            "latitude": {"_FillValue": None},
            "longitude": {"_FillValue": None},
        })
        source_size = path.stat().st_size
        if not complete:
            with path.open("r+b") as stream:
                stream.truncate(source_size // 2)
        with db:
            db.execute(
                "UPDATE source_files SET size = ?, downloaded_at = ? WHERE id = ?",
                (source_size, datetime.now(UTC).isoformat() if complete else None, row["id"]),
            )
        return db.execute("SELECT * FROM source_files WHERE id = ?", (row["id"],)).fetchone(), values


def test_selected_dates_then_added_variable_and_hours(tmp_path: Path) -> None:
    """Publish only selected observations and preserve them when adding data."""
    archive = tmp_path / "archive"
    temperature, expected = _stage_analysis(archive)
    assert ncar_ingest.ingest_ncar(archive, 2000, 1) == archive / "2000" / "01"
    assert not (archive / temperature["path"]).exists()
    data = read_raster_values(archive, 2000, 1)
    assert data["2_metre_temperature"].dtype == np.dtype("float32")
    np.testing.assert_array_equal(data["2_metre_temperature"], expected[:24])
    wind, expected_wind = _stage_analysis(archive, "10u", last_day=2)
    ncar_ingest.ingest_ncar(archive, 2000, 1)
    data = read_raster_values(archive, 2000, 1)
    np.testing.assert_array_equal(data["2_metre_temperature"], expected[:24])
    np.testing.assert_array_equal(data["10_metre_u_wind_component"], expected_wind)
    with catalog.archive_session(archive) as db:
        rows = db.execute("SELECT * FROM observations").fetchall()
        assert len(rows) == 72
        for row in rows:
            assert row["source_file_id"] == (temperature["id"] if row["source_variable"] == "2t" else wind["id"])
            assert row["reference_time"] is None
            assert row["temporal_type"] == "instant"
    _stage_analysis(archive, first_day=2, last_day=2)
    ncar_ingest.ingest_ncar(archive, 2000, 1)
    np.testing.assert_array_equal(read_raster_values(archive, 2000, 1)["2_metre_temperature"], expected)


def test_repeat_uses_catalog_without_decoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retire repeated source selections without decoding or replacing TIFFs."""
    archive = tmp_path / "archive"
    _stage_analysis(archive)
    ncar_ingest.ingest_ncar(archive, 2000, 1)
    before = {path: path.stat().st_mtime_ns for path in archive.glob("2000/01/*/*.tif")}
    repeated, _ = _stage_analysis(archive)
    monkeypatch.setattr(ncar_ingest, "read_ncar_netcdf_arrays", Mock(side_effect=AssertionError("repeat decoded")))
    ncar_ingest.ingest_ncar(archive, 2000, 1)
    ncar_ingest.ingest_ncar(archive, 2000, 1)
    assert not (archive / repeated["path"]).exists()
    assert before == {path: path.stat().st_mtime_ns for path in before}
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 24
        assert db.execute("SELECT COUNT(*) FROM source_selections").fetchone()[0] == 0


def test_refresh_preserves_unselected_files_and_origins(tmp_path: Path) -> None:
    """Replace one day's temperature without rewriting other hours or variables."""
    archive = tmp_path / "archive"
    original, expected = _stage_analysis(archive, last_day=2)
    wind, expected_wind = _stage_analysis(archive, "10u", last_day=2)
    ncar_ingest.ingest_ncar(archive, 2000, 1)
    untouched = {path: path.stat().st_mtime_ns for path in archive.glob("2000/01/*/*.tif")
                 if "wind" in str(path) or "20000102" in path.name}
    refreshed, corrected = _stage_analysis(archive, version=2, offset=1000)
    ncar_ingest.ingest_ncar(archive, 2000, 1, refresh=True)
    data = read_raster_values(archive, 2000, 1)
    np.testing.assert_array_equal(data["2_metre_temperature"][:24], corrected[:24])
    np.testing.assert_array_equal(data["2_metre_temperature"][24:], expected[24:])
    np.testing.assert_array_equal(data["10_metre_u_wind_component"], expected_wind)
    assert untouched == {path: path.stat().st_mtime_ns for path in untouched}
    with catalog.archive_session(archive) as db:
        fields = db.execute("SELECT * FROM observations").fetchall()
        for field in fields:
            expected_id = wind["id"] if field["source_variable"] == "10u" else (
                refreshed["id"] if field["valid_time"] < 946771200 else original["id"])
            assert field["source_file_id"] == expected_id
        assert len(fields) == 96


def test_missing_source_hour_prevents_publication(tmp_path: Path) -> None:
    """An incomplete external NetCDF selection preserves existing observations."""
    archive = tmp_path / "archive"
    _, original = _stage_analysis(archive)
    ncar_ingest.ingest_ncar(archive, 2000, 1)
    incomplete, _ = _stage_analysis(archive, version=2, offset=1000, omit_hour=23)
    with pytest.raises(ValueError, match="each selected UTC hour"):
        ncar_ingest.ingest_ncar(archive, 2000, 1, refresh=True)
    assert (archive / incomplete["path"]).exists()
    np.testing.assert_array_equal(read_raster_values(archive, 2000, 1)["2_metre_temperature"], original[:24])


def test_interrupted_ingestion_resumes_remaining_observations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep completed COGs and source input when the next file cannot be written."""
    archive = tmp_path / "archive"
    source, expected = _stage_analysis(archive)
    write = ncar_ingest.write_observation_cog
    count = 0

    def fail_third(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Interrupt a real output write after two completed observations."""
        nonlocal count
        count += 1
        if count == 3:
            raise OSError("disk unavailable")
        return write(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(ncar_ingest, "write_observation_cog", fail_third)
        with pytest.raises(OSError, match="disk unavailable"):
            ncar_ingest.ingest_ncar(archive, 2000, 1)
    assert (archive / source["path"]).exists()
    assert read_raster_values(archive, 2000, 1)["2_metre_temperature"].shape[0] == 2
    before = {path: path.stat().st_mtime_ns for path in archive.glob("2000/01/*/*.tif")}
    ncar_ingest.ingest_ncar(archive, 2000, 1)
    assert before == {path: path.stat().st_mtime_ns for path in before}
    assert not (archive / source["path"]).exists()
    np.testing.assert_array_equal(read_raster_values(archive, 2000, 1)["2_metre_temperature"], expected[:24])


def test_pre_1970_source_publishes_exact_utc_hours(tmp_path: Path) -> None:
    """Preserve January 1950 timestamps through the NetCDF and COG boundaries."""
    archive = tmp_path / "archive"
    source, expected = _stage_analysis(archive, year=1950)
    start = int(datetime(1950, 1, 1, tzinfo=UTC).timestamp())
    hours = {("2t", start + hour * 3600) for hour in range(24)}
    with catalog.archive_session(archive) as db:
        ncar_ingest.ingest_ncar_month(db, archive, 1950, 1, selected_hours=hours, source_ids={source["id"]})
        assert catalog.get_published_hours(db, 1950, 1) == hours
    np.testing.assert_array_equal(read_raster_values(archive, 1950, 1)["2_metre_temperature"], expected[:24])


def test_corrected_source_consumes_only_selected_pending_hours(tmp_path: Path) -> None:
    """Release superseded source selections while retaining unrelated later hours."""
    archive = tmp_path / "archive"
    older, original = _stage_analysis(archive, last_day=2)
    newer, corrected = _stage_analysis(archive, version=2, offset=1000)
    hours = {("2t", 946684800 + hour * 3600) for hour in range(24)}
    with catalog.archive_session(archive) as db:
        ncar_ingest.ingest_ncar_month(db, archive, 2000, 1, refresh=True, selected_hours=hours, source_ids={newer["id"]})
        assert [tuple(row) for row in db.execute("SELECT source_id, start_time, end_time FROM source_selections")] == [
            (older["id"], "2000-01-02T00:00:00+00:00", "2000-01-02T23:00:00+00:00")]
    assert (archive / older["path"]).exists()
    assert not (archive / newer["path"]).exists()
    ncar_ingest.ingest_ncar(archive, 2000, 1)
    data = read_raster_values(archive, 2000, 1)["2_metre_temperature"]
    np.testing.assert_array_equal(data[:24], corrected[:24])
    np.testing.assert_array_equal(data[24:], original[24:])


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_obsolete_partial_source_cleanup_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool) -> None:
    """Retain failed cleanup in the catalog and retry it on the next run."""
    archive = tmp_path / "archive"
    older, _ = _stage_analysis(archive, complete=False)
    newer, _ = _stage_analysis(archive, version=2, offset=1000)
    partial = archive / older["path"]
    unlink = Path.unlink

    def reject_partial_cleanup(path: Path, missing_ok: bool = False) -> None:
        """Simulate a source still open on a shared filesystem."""
        if path == partial:
            raise PermissionError("partial source is open")
        unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        if cleanup_fails:
            patch.setattr(Path, "unlink", reject_partial_cleanup)
        expectation = pytest.warns(RuntimeWarning, match="source cleanup remains pending") if cleanup_fails else nullcontext()
        with expectation:
            ncar_ingest.ingest_ncar(archive, 2000, 1, refresh=True)
    assert partial.exists() == cleanup_fails
    with catalog.archive_session(archive) as db:
        catalog.delete_ingested_netcdf_files(db, archive)
        assert db.execute("SELECT removed_at FROM source_files WHERE id = ?", (older["id"],)).fetchone()[0] is not None
    assert not partial.exists()
