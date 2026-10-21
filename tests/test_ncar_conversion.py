"""Exercise actual NetCDF decoding for analysis and hourly forecast fields."""

from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest
import xarray as xr

from era5_pull_pipeline import conversion, ncar_conversion


@pytest.fixture(autouse=True)
def small_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a small regular grid while retaining the external grid check."""
    monkeypatch.setattr(conversion, "GLOBAL_LATITUDES", np.array([90.0, 89.75, 89.5]))
    monkeypatch.setattr(conversion, "GLOBAL_LONGITUDES", np.array([0.0, 0.25, 0.5, 0.75]))


def write_source(path: Path, variable: str = "2t") -> NDArray[np.float32]:
    """Write a compressed NetCDF fixture with NCAR's time and field conventions.

    Args:
        path: Destination source file.
        variable: Analysis temperature or forecast radiation/precipitation name.

    Returns:
        Original float32 values, with a missing cell for the decoder to preserve.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    coordinates: dict[str, Any] = {
        "latitude": conversion.GLOBAL_LATITUDES,
        "longitude": conversion.GLOBAL_LONGITUDES,
    }
    if variable == "2t":
        dimensions = ("time", "latitude", "longitude")
        coordinates["time"] = np.arange("2000-01-01T00", "2000-01-01T04", dtype="datetime64[h]")
        shape, chunks = (4, 3, 4), (2, 2, 3)
        name, units = "2 metre temperature", "K"
    else:
        dimensions = ("forecast_initial_time", "forecast_hour", "latitude", "longitude")
        coordinates["forecast_initial_time"] = np.array(["1999-12-31T18"], dtype="datetime64[h]")
        coordinates["forecast_hour"] = ("forecast_hour", np.arange(1, 13, dtype=np.int32), {
            "units": "hour", "standard_name": "forecast_period",
            "units_qualifier": "hours since forecast_initial_time",
        })
        shape, chunks = (1, 12, 3, 4), (1, 4, 2, 3)
        name, units = (
            ("Surface solar radiation downwards", "W m**-2 s") if variable == "ssrd"
            else ("Mean total precipitation rate", "kg m**-2 s**-1")
        )
    values = (np.arange(np.prod(shape), dtype=np.float32) * np.float32(0.000123)).reshape(shape)
    values.reshape(-1)[0] = np.nan
    data = xr.Dataset(
        {variable.upper(): (dimensions, values, {
            "long_name": name, "short_name": variable, "units": units,
            "number_of_significant_digits": np.int32(7),
            "minimum_value": np.float32(-1),
        })},
        coords=coordinates,
        attrs={"NETCDF_COMPRESSION": "Source precision metadata", "history": "NCAR fixture"},
    )
    data.to_netcdf(path, engine="h5netcdf", encoding={
        variable.upper(): {"chunksizes": chunks, "zlib": True, "_FillValue": np.float32(9.999e20)},
        "latitude": {"_FillValue": None},
        "longitude": {"_FillValue": None},
    })
    return values


def test_analysis_preserves_selected_float32_values_and_metadata(tmp_path: Path) -> None:
    """Selected noncontiguous hours retain precision and analysis interval meaning."""
    path = tmp_path / "temperature.nc"
    expected = write_source(path)
    selected = np.array(["2000-01-01T00", "2000-01-01T03"], dtype="datetime64[s]").astype(np.int64)
    with ExitStack() as resources:
        arrays, report = ncar_conversion.decode_source(
            path, "analysis-source", "2t", selected, tmp_path, resources,
            source_identity="analysis-sha256",
        )
        assert arrays["2_metre_temperature"].dtype == np.dtype("float32")
        np.testing.assert_array_equal(arrays["2_metre_temperature"], expected[[0, 3]])
        attributes = report["variables"]["2_metre_temperature"]
        assert attributes["units"] == "K"
        assert attributes["number_of_significant_digits"] == 7
        assert "minimum_value" not in attributes
        assert report["source_attributes"]["variable"]["minimum_value"] == -1
        assert report["source_attributes"]["global"]["history"] == "NCAR fixture"
        assert (report["year"], report["month"]) == (2000, 1)
        for field, valid in zip(report["fields"], selected, strict=True):
            assert field["source_file_id"] == "analysis-source"
            assert field["source_variable"] == "2t"
            assert field["source_request_sha256"] == "analysis-sha256"
            assert field["interval_start"] == field["interval_end"] == valid
            assert field["reference_time"] == conversion.MISSING_INTEGER
            assert field["start_step"] == field["end_step"] == conversion.MISSING_INTEGER
            assert field["step_type"] == "instant"


@pytest.mark.parametrize("variable,collection,step_type", [
    ("ssrd", "accumu", "accum"), ("mtpr", "meanflux", "avg"),
])
def test_forecasts_keep_preceding_hour_values_across_month_boundary(
    tmp_path: Path, variable: str, collection: str, step_type: str,
) -> None:
    """Radiation and precipitation retain values rather than being differenced."""
    path = tmp_path / f"e5.oper.fc.sfc.{collection}.nc"
    expected = write_source(path, variable)
    selected = np.array(["2000-01-01T00", "2000-01-01T01"], dtype="datetime64[s]").astype(np.int64)
    with ExitStack() as resources:
        arrays, report = ncar_conversion.decode_source(
            path, "forecast-source", variable, selected, tmp_path, resources,
            source_identity="forecast-sha256",
        )
        actual = next(iter(arrays.values()))
        assert actual.dtype == np.dtype("float32")
        np.testing.assert_array_equal(actual, expected[0, [5, 6]])
        assert (report["year"], report["month"]) == (2000, 1)
        for field, valid, lead in zip(report["fields"], selected, [6, 7], strict=True):
            assert field["reference_time"] == int(np.datetime64("1999-12-31T18", "s").astype(np.int64))
            assert field["interval_start"] == valid - 3600
            assert field["interval_end"] == valid
            assert field["start_step"] == (lead - 1) * 3600
            assert field["end_step"] == lead * 3600
            assert field["step_type"] == step_type
        units = next(iter(report["variables"].values()))["units"]
        assert units == ("W m**-2 s" if variable == "ssrd" else "kg m**-2 s**-1")


def test_missing_selected_hour_rejects_incomplete_source(tmp_path: Path) -> None:
    """A source with less time coverage than the catalog promises cannot publish."""
    path = tmp_path / "temperature.nc"
    write_source(path)
    selected = [int(np.datetime64("2000-01-01T05", "s").astype(np.int64))]
    with ExitStack() as resources, pytest.raises(ValueError, match="each selected UTC hour"):
        ncar_conversion.decode_source(
            path, "missing-hour", "2t", selected, tmp_path, resources,
            source_identity="source-sha256",
        )


def test_regional_source_is_not_accepted_as_global(tmp_path: Path) -> None:
    """Unexpected provider grid extents fail before creating archive coverage."""
    path = tmp_path / "temperature.nc"
    write_source(path)
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        region = dataset.isel(latitude=slice(0, 2)).load()
    region.to_netcdf(path, engine="h5netcdf")
    selected = [int(np.datetime64("2000-01-01T00", "s").astype(np.int64))]
    with ExitStack() as resources, pytest.raises(ValueError, match="full global"):
        ncar_conversion.decode_source(
            path, "regional-source", "2t", selected, tmp_path, resources,
            source_identity="source-sha256",
        )
