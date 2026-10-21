"""Exercise monthly NCAR ingestion with real NetCDF, Zarr, and SQLite files."""

from datetime import UTC, date, datetime
from pathlib import Path
import sqlite3
from unittest.mock import Mock

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
    first_day: int = 1, last_day: int = 1, version: int = 1,
    offset: float = 0.0, omit_hour: int | None = None,
) -> tuple[sqlite3.Row, NDArray[np.float32]]:
    """Create a completed two-day provider file with separately selected dates.

    Args:
        archive: Archive receiving the provider file and catalog selection.
        variable: Temperature or wind source name.
        first_day: First selected January day.
        last_day: Last selected January day.
        version: Distinct provider revision for refresh fixtures.
        offset: Value added before the source is recorded as complete.
        omit_hour: Optional missing external source hour for coverage failure.

    Returns:
        Completed source catalog row and the original two days of values.
    """
    names = {
        "2t": ("2 metre temperature", "K"),
        "10u": ("10 metre U wind component", "m s**-1"),
    }
    source = SourceFile(
        key=f"e5.oper.an.sfc/200001/e5.oper.an.sfc.{variable}.nc",
        variable=variable, size=1, etag=f'"version-{version}"',
        last_modified=f"2026-01-{version:02d}T00:00:00+00:00",
        valid_start=datetime(2000, 1, 1, tzinfo=UTC),
        valid_end=datetime(2000, 1, 2, 23, tzinfo=UTC),
    )
    with catalog.archive_session(archive) as db:
        row = catalog.register_sources(
            db, archive, [source], date(2000, 1, first_day), date(2000, 1, last_day),
        )[0]
        path = Path(row["path"])
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
                "time": np.arange("2000-01-01T00", "2000-01-03T00", dtype="datetime64[h]"),
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
        with db:
            db.execute(
                "UPDATE source_files SET size = ?, downloaded_at = ? WHERE id = ?",
                (path.stat().st_size, datetime.now(UTC).isoformat(), row["id"]),
            )
        return db.execute("SELECT * FROM source_files WHERE id = ?", (row["id"],)).fetchone(), values


def test_selected_dates_publish_actual_fields_then_add_wind(tmp_path: Path) -> None:
    """Extra packaged hours stay absent until a selected variable supplies them."""
    archive = tmp_path / "archive"
    temperature, expected_temperature = _stage_analysis(archive)
    published = ncar_ingest.ingest_ncar(archive, 2000, 1, (2, 2))
    assert published == archive / "2000" / "01" / "era5.zarr"
    assert not Path(temperature["path"]).exists()
    with catalog.open_month(archive, 2000, 1) as dataset:
        assert dataset["2_metre_temperature"].dtype == np.dtype("float32")
        assert dataset["2_metre_temperature"].shape == (24, 3, 4)
        np.testing.assert_array_equal(dataset["2_metre_temperature"].values, expected_temperature[:24])
        assert dataset["2_metre_temperature"].attrs["units"] == "K"
    wind, expected_wind = _stage_analysis(archive, "10u", last_day=2)
    ncar_ingest.ingest_ncar(archive, 2000, 1, (2, 2))
    with catalog.open_month(archive, 2000, 1) as dataset:
        np.testing.assert_array_equal(dataset["2_metre_temperature"].values[:24], expected_temperature[:24])
        assert np.isnan(dataset["2_metre_temperature"].values[24:]).all()
        np.testing.assert_array_equal(dataset["10_metre_u_wind_component"].values, expected_wind)
        assert dataset["2_metre_temperature_coverage"].values.sum() == 24
        assert dataset["10_metre_u_wind_component_coverage"].values.sum() == 48
    with catalog.archive_session(archive) as db:
        report = catalog.current_month(db, archive, 2000, 1)[1]
        assert len(report["fields"]) == 72
        assert {
            field["source_file_id"] for field in report["fields"] if field["source_variable"] == "2t"
        } == {temperature["id"]}
        assert {
            field["source_file_id"] for field in report["fields"] if field["source_variable"] == "10u"
        } == {wind["id"]}
    _stage_analysis(archive, first_day=2, last_day=2)
    ncar_ingest.ingest_ncar(archive, 2000, 1, (2, 2))
    with catalog.open_month(archive, 2000, 1) as dataset:
        np.testing.assert_array_equal(dataset["2_metre_temperature"].values, expected_temperature)
        assert dataset["2_metre_temperature_coverage"].values.all()


def test_repeat_uses_published_coverage_without_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated sources are consumed without decoding or creating a revision."""
    archive = tmp_path / "archive"
    _stage_analysis(archive)
    published = ncar_ingest.ingest_ncar(archive, 2000, 1, (2, 2))
    repeated, _ = _stage_analysis(archive)
    monkeypatch.setattr(ncar_ingest, "read_ncar_netcdf_arrays", Mock(side_effect=AssertionError("repeat decoded")))
    assert ncar_ingest.ingest_ncar(archive, 2000, 1, (2, 2)) == published
    assert not Path(repeated["path"]).exists()
    assert ncar_ingest.ingest_ncar(archive, 2000, 1, (2, 2)) == published
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM source_selections").fetchone()[0] == 0


def test_refresh_replaces_selected_day_and_retains_other_fields(tmp_path: Path) -> None:
    """A corrected day's temperature preserves other days, variables, and origins."""
    archive = tmp_path / "archive"
    original, expected_temperature = _stage_analysis(archive, last_day=2)
    wind, expected_wind = _stage_analysis(archive, "10u", last_day=2)
    ncar_ingest.ingest_ncar(archive, 2000, 1, (2, 2))
    refreshed, corrected = _stage_analysis(archive, version=2, offset=1000)
    ncar_ingest.ingest_ncar(archive, 2000, 1, (2, 2), refresh=True)
    with catalog.open_month(archive, 2000, 1) as dataset:
        np.testing.assert_array_equal(dataset["2_metre_temperature"].values[:24], corrected[:24])
        np.testing.assert_array_equal(dataset["2_metre_temperature"].values[24:], expected_temperature[24:])
        np.testing.assert_array_equal(dataset["10_metre_u_wind_component"].values, expected_wind)
    with catalog.archive_session(archive) as db:
        fields = catalog.current_month(db, archive, 2000, 1)[1]["fields"]
        cutoff = int(datetime(2000, 1, 2, tzinfo=UTC).timestamp())
        for field in fields:
            expected = wind["id"] if field["source_variable"] == "10u" else (
                refreshed["id"] if field["valid_time"] < cutoff else original["id"]
            )
            assert field["source_file_id"] == expected
        assert len(fields) == 96


def test_incomplete_refresh_retains_previous_month_and_sources(tmp_path: Path) -> None:
    """An externally missing selected hour prevents any partial refresh publication."""
    archive = tmp_path / "archive"
    _, original = _stage_analysis(archive)
    ncar_ingest.ingest_ncar(archive, 2000, 1, (2, 2))
    incomplete, _ = _stage_analysis(archive, version=2, offset=1000, omit_hour=23)
    with pytest.raises(ValueError, match="each selected UTC hour"):
        ncar_ingest.ingest_ncar(archive, 2000, 1, (2, 2), refresh=True)
    assert Path(incomplete["path"]).exists()
    with catalog.open_month(archive, 2000, 1) as dataset:
        np.testing.assert_array_equal(dataset["2_metre_temperature"].values, original[:24])
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM source_selections").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM revisions WHERE status = 'published'").fetchone()[0] == 1


def test_failed_replacement_retains_source_and_previous_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A publication rename failure restores the previous store and permits retry."""
    archive = tmp_path / "archive"
    _, original = _stage_analysis(archive)
    ncar_ingest.ingest_ncar(archive, 2000, 1, (2, 2))
    refreshed, corrected = _stage_analysis(archive, version=2, offset=1000)
    rename = catalog._rename_store
    rejected = False

    def reject_new_store(source: Path, destination: Path) -> None:
        """Fail the new store rename while allowing restoration of its parent."""
        nonlocal rejected
        if destination == archive / "2000" / "01" / "era5.zarr" and not rejected:
            rejected = True
            raise OSError("publication destination unavailable")
        rename(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(catalog, "_rename_store", reject_new_store)
        with pytest.raises(OSError, match="publication destination unavailable"):
            ncar_ingest.ingest_ncar(archive, 2000, 1, (2, 2), refresh=True)
    assert Path(refreshed["path"]).exists()
    with catalog.open_month(archive, 2000, 1) as dataset:
        np.testing.assert_array_equal(dataset["2_metre_temperature"].values, original[:24])
    ncar_ingest.ingest_ncar(archive, 2000, 1, (2, 2), refresh=True)
    assert not Path(refreshed["path"]).exists()
    with catalog.open_month(archive, 2000, 1) as dataset:
        np.testing.assert_array_equal(dataset["2_metre_temperature"].values, corrected[:24])
