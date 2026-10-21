"""Verify monthly Zarr conversion against source arrays and retained coverage."""

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest
import xarray as xr
import zarr
from zarr.codecs import ZstdCodec

from era5_pull_pipeline import conversion

START = int(datetime(1950, 1, 1, tzinfo=UTC).timestamp())
METADATA = {"dataset": "nsf-ncar-era5", "request_sha256": "selection", "sources": [{"id": "source"}]}


@pytest.fixture(autouse=True)
def small_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a small regular grid to exercise several spatial shards quickly."""
    monkeypatch.setattr(conversion, "GLOBAL_LATITUDES", np.array([90.0, 89.75, 89.5]))
    monkeypatch.setattr(conversion, "GLOBAL_LONGITUDES", np.array([0.0, 0.25, 0.5, 0.75]))


def source_arrays(
    variable: str, hours: tuple[int, ...], values: tuple[float, ...],
    dtype: type[np.floating[Any]] = np.float32, *, source_id: str = "source",
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """Create source arrays and the field report supplied by the NetCDF reader.

    Args:
        variable: Descriptive output array name.
        hours: Selected UTC hours since the beginning of January 1950.
        values: Constant grid value for each hour.
        dtype: Decoded numerical precision.
        source_id: Source catalog identity recorded on every field.

    Returns:
        Arrays and their monthly field metadata.
    """
    names = {
        "2_metre_temperature": ("2 metre temperature", "2t", "K", "instant"),
        "mean_total_precipitation_rate": ("Mean total precipitation rate", "mtpr", "kg m-2 s-1", "avg"),
        "10_metre_u_wind_component": ("10 metre U wind component", "10u", "m s-1", "instant"),
    }
    name, short_name, units, step_type = names[variable]
    forecast = step_type == "avg"
    times = [START + hour * 3600 for hour in hours]
    fields = [{
        "variable": variable, "valid_time": valid,
        "reference_time": valid - 3600 if forecast else int(conversion.MISSING_INTEGER),
        "interval_start": valid - 3600 if forecast else valid,
        "interval_end": valid,
        "start_step": 0 if forecast else int(conversion.MISSING_INTEGER),
        "end_step": 3600 if forecast else int(conversion.MISSING_INTEGER),
        "step_units": "s", "step_type": step_type,
        "source_file_id": source_id, "source_variable": short_name,
        "source_request_sha256": source_id,
    } for valid in times]
    arrays = {variable: np.stack([np.full((3, 4), value, dtype=dtype) for value in values])}
    report = {
        "year": 1950, "month": 1,
        "grid": {"type": "regular_ll", "shape": [3, 4], "resolution": [0.25, 0.25]},
        "variables": {variable: {"name": name, "short_name": short_name,
                                 "units": units, "step_type": step_type}},
        "fields": fields,
    }
    return arrays, report


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_monthly_zarr_preserves_dtype_values_coordinates_and_intervals(
    tmp_path: Path, dtype: type[np.floating[Any]],
) -> None:
    """Round-trip source precision, NaN cells, UTC intervals, and lossless compression."""
    temperature = source_arrays("2_metre_temperature", (0, 1), (280.123456789, np.nan), dtype)
    precipitation = source_arrays("mean_total_precipitation_rate", (0, 1), (0.00000123456789, 0.0), dtype)
    output = tmp_path / "month.zarr"
    report = conversion.write_monthly_zarr(
        [temperature, precipitation], METADATA, output, (1, 1), previous=None, previous_report=None,
    )
    stored = zarr.open_group(output, mode="r")
    for arrays, source_report in (temperature, precipitation):
        variable, expected = next(iter(arrays.items()))
        actual = stored[variable]
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual[:], expected)
        assert actual.chunks == (744, 1, 1)
        assert actual.shards == (744, 2, 2)
        assert actual.compressors == (ZstdCodec(level=3, checksum=True),)
        assert actual.metadata.dimension_names == ("time", "latitude", "longitude")
        assert actual.attrs["units"] == source_report["variables"][variable]["units"]
    np.testing.assert_array_equal(stored["latitude"][:], conversion.GLOBAL_LATITUDES)
    np.testing.assert_array_equal(stored["longitude"][:], conversion.GLOBAL_LONGITUDES)
    np.testing.assert_array_equal(stored["time"][:], [START, START + 3600])
    assert stored["2_metre_temperature_coverage"][:].tolist() == [True, True]
    assert stored["mean_total_precipitation_rate_interval_start"][:].tolist() == [START - 3600, START]
    assert report["validation"]["field_count"] == 4
    assert report["validation"]["numerical_roundtrip"] == "exact_source_dtype_equal_nan"
    with xr.open_zarr(output, chunks=None, consolidated=False) as dataset:
        assert dataset["2_metre_temperature"].dtype == dtype
        assert dataset["time"].values[0] == np.datetime64("1950-01-01T00:00:00")


def test_add_variables_and_hours_preserves_prior_fields(tmp_path: Path) -> None:
    """Merge independent selections while retaining prior values and source origins."""
    initial = source_arrays("2_metre_temperature", (0, 1), (np.nan, 281.0))
    previous = tmp_path / "previous.zarr"
    previous_report = conversion.write_monthly_zarr(
        [initial], METADATA, previous, (1, 1), previous=None, previous_report=None,
    )
    for field in previous_report["fields"]:
        field["source_revision_id"] = "published-revision"
    later = source_arrays("2_metre_temperature", (1, 2), (999.0, 282.0), source_id="later")
    rain = source_arrays("mean_total_precipitation_rate", (2, 3), (0.002, 0.003), source_id="rain")
    output = tmp_path / "replacement.zarr"
    report = conversion.write_monthly_zarr(
        [later, rain], {**METADATA, "sources": [{"id": "later"}, {"id": "rain"}]}, output, (1, 1),
        previous=previous, previous_report=previous_report,
    )
    stored = zarr.open_group(output, mode="r")
    assert stored["2_metre_temperature"].shape == (4, 3, 4)
    np.testing.assert_array_equal(stored["2_metre_temperature"][:2], initial[0]["2_metre_temperature"])
    np.testing.assert_array_equal(stored["2_metre_temperature"][2], 282.0)
    assert np.isnan(stored["2_metre_temperature"][3]).all()
    assert stored["2_metre_temperature_coverage"][:].tolist() == [True, True, True, False]
    assert stored["mean_total_precipitation_rate_coverage"][:].tolist() == [False, False, True, True]
    np.testing.assert_array_equal(stored["mean_total_precipitation_rate"][2:], rain[0]["mean_total_precipitation_rate"])
    for field in previous_report["fields"]:
        assert field in report["fields"]
    assert len(report["fields"]) == 5
    assert stored.attrs["source_provenance"] == [{"id": "source"}, {"id": "later"}, {"id": "rain"}]
    unchanged = zarr.open_group(previous, mode="r")
    np.testing.assert_array_equal(unchanged["2_metre_temperature"][:], initial[0]["2_metre_temperature"])


@pytest.mark.parametrize("refresh", [False, True])
def test_refresh_replaces_selected_values_and_preserves_other_fields(tmp_path: Path, refresh: bool) -> None:
    """Refresh only selected hours, including their provenance and interval metadata."""
    initial = source_arrays("2_metre_temperature", (0, 1), (280.0, 281.0))
    rain = source_arrays("mean_total_precipitation_rate", (0, 1), (0.001, 0.002))
    wind = source_arrays("10_metre_u_wind_component", (0, 1), (2.0, 3.0))
    previous = tmp_path / "previous.zarr"
    previous_report = conversion.write_monthly_zarr(
        [initial, rain, wind], METADATA, previous, (2, 2), previous=None, previous_report=None,
    )
    for field in previous_report["fields"]:
        field["source_revision_id"] = "published-revision"
    changed = source_arrays("2_metre_temperature", (1,), (np.nan,), source_id="corrected")
    changed_rain = source_arrays("mean_total_precipitation_rate", (0,), (0.01,), source_id="corrected")
    changed_rain[1]["fields"][0]["reference_time"] -= 3600
    changed_rain[1]["fields"][0]["start_step"] = 3600
    changed_rain[1]["fields"][0]["end_step"] = 7200
    output = tmp_path / "replacement.zarr"
    report = conversion.write_monthly_zarr(
        [changed, changed_rain], METADATA, output, (2, 2),
        previous=previous, previous_report=previous_report, refresh=refresh,
    )
    stored = zarr.open_group(output, mode="r")
    np.testing.assert_array_equal(stored["2_metre_temperature"][0], 280.0)
    np.testing.assert_array_equal(stored["2_metre_temperature"][1], np.nan if refresh else 281.0)
    np.testing.assert_array_equal(stored["mean_total_precipitation_rate"][0], np.float32(0.01 if refresh else 0.001))
    np.testing.assert_array_equal(stored["10_metre_u_wind_component"][:], wind[0]["10_metre_u_wind_component"])
    assert stored["mean_total_precipitation_rate_start_step"][0] == (3600 if refresh else 0)
    replacements = {(f["variable"], f["valid_time"]): f for _, r in (changed, changed_rain) for f in r["fields"]}
    fields = {(f["variable"], f["valid_time"]): f for f in report["fields"]}
    for field in previous_report["fields"]:
        pair = field["variable"], field["valid_time"]
        assert fields[pair] == (replacements[pair] if refresh and pair in replacements else field)
    unchanged = zarr.open_group(previous, mode="r")
    np.testing.assert_array_equal(unchanged["2_metre_temperature"][:], initial[0]["2_metre_temperature"])


@pytest.mark.parametrize("conflict", ["overlap", "month"])
def test_conflicting_source_selections_are_not_written(tmp_path: Path, conflict: str) -> None:
    """Reject overlapping source fields or different months before creating a store."""
    first = source_arrays("2_metre_temperature", (0,), (280.0,))
    second = deepcopy(first)
    if conflict == "month":
        second[1]["month"] = 2
    output = tmp_path / "month.zarr"
    with pytest.raises(ValueError, match="duplicate" if conflict == "overlap" else "same calendar month"):
        conversion.write_monthly_zarr([first, second], METADATA, output, (2, 2), previous=None, previous_report=None)
    assert not output.exists()
