"""Exercise real ecCodes decoding and Zarr storage on small fixture grids."""

from datetime import UTC, datetime
from copy import deepcopy
from pathlib import Path
from typing import Any

import eccodes
import numpy as np
from numpy.typing import NDArray
import pytest
import xarray as xr
import zarr

from era5_pull_pipeline import conversion


@pytest.fixture(autouse=True)
def small_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a small regular grid while exercising the converter's exact checks."""
    monkeypatch.setattr(
        conversion, "GLOBAL_LATITUDES", np.array([90.0, 89.75, 89.5])
    )
    monkeypatch.setattr(
        conversion, "GLOBAL_LONGITUDES", np.array([0.0, 0.25, 0.5, 0.75])
    )


def write_grib(
    path: Path, messages: list[dict[str, Any]]
) -> list[NDArray[np.float64]]:
    """Encode real fields, returning values after ecCodes' packing round trip.

    Args:
        path: Destination GRIB file.
        messages: GRIB keys for each message; ``values`` supplies its data.

    Returns:
        Decoded float64 fields, including bitmap missing values as NaN.
    """
    decoded = []
    with path.open("wb") as stream:
        for message in messages:
            handle = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            try:
                settings = {
                    "Ni": 4,
                    "Nj": 3,
                    "latitudeOfFirstGridPointInDegrees": 90.0,
                    "latitudeOfLastGridPointInDegrees": 89.5,
                    "longitudeOfFirstGridPointInDegrees": 0.0,
                    "longitudeOfLastGridPointInDegrees": 0.75,
                    "iDirectionIncrementInDegrees": 0.25,
                    "jDirectionIncrementInDegrees": 0.25,
                    "scanningMode": 0,
                    "paramId": 167,
                    "dataDate": 20000101,
                    "dataTime": 0,
                    "stepType": "instant",
                    "bitsPerValue": 24,
                }
                settings.update({key: value for key, value in message.items() if key != "values"})
                for key, value in settings.items():
                    eccodes.codes_set(handle, key, value)
                values = np.array(
                    message.get("values", np.linspace(280.01, 281.02, 12)),
                    dtype=np.float64,
                )
                if np.isnan(values).any():
                    eccodes.codes_set(handle, "bitmapPresent", 1)
                    eccodes.codes_set(handle, "missingValue", -9999.0)
                    values = np.where(np.isnan(values), -9999.0, values)
                eccodes.codes_set_values(handle, values)
                eccodes.codes_write(handle, stream)
                expected = eccodes.codes_get_values(handle)
                if eccodes.codes_get(handle, "bitmapPresent"):
                    expected[eccodes.codes_get_array(handle, "bitmap") == 0] = np.nan
                decoded.append(expected.reshape(3, 4))
            finally:
                eccodes.codes_release(handle)
    return decoded


@pytest.fixture
def batch(tmp_path: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Describe two UTC hours of temperature and hourly accumulated precipitation."""
    metadata = {
        "dataset": "reanalysis-era5-single-levels",
        "request": {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature", "total_precipitation"],
            "year": ["2000"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00", "01:00"],
            "grid": [0.25, 0.25],
            "data_format": "grib",
            "download_format": "unarchived",
        },
        "cds_request_id": "offline-conversion",
    }
    messages = [
        {"dataTime": 0},
        {
            "paramId": 228, "dataDate": 19991231, "dataTime": 1800,
            "stepType": "accum", "startStep": 5, "endStep": 6,
            "values": np.linspace(0, 0.001, 12),
        },
        {"dataTime": 100},
        {
            "paramId": 228, "dataDate": 19991231, "dataTime": 1800,
            "stepType": "accum", "startStep": 6, "endStep": 7,
            "values": np.linspace(0, 0.002, 12),
        },
    ]
    return tmp_path / "batch.grib", metadata, messages


def test_temperature_precipitation_roundtrip_and_layout(
    tmp_path: Path,
    batch: tuple[Path, dict[str, Any], list[dict[str, Any]]],
) -> None:
    """Retain float64 precision and previous-month accumulation intervals."""
    grib, metadata, messages = batch
    expected = write_grib(grib, messages)
    output = tmp_path / "month.zarr"
    report = conversion.convert_batch(grib, metadata, output, spatial_chunks=(2, 2))
    assert grib.exists()
    assert not list(tmp_path.glob(".decoded-*"))
    assert report["year"] == 2000
    assert report["month"] == 1
    assert report["layout"] == {
        "spatial_chunks": [2, 2],
        "time_chunk": "calendar_month",
        "shard_spatial_multiple": 2,
    }
    assert report["validation"]["field_count"] == 4
    assert report["validation"]["metadata_roundtrip"] is True
    assert report["variables"]["param_167"]["units"] == "K"
    assert report["variables"]["param_228"]["units"] == "m"
    midnight = int(datetime(2000, 1, 1, tzinfo=UTC).timestamp())
    rain = next(field for field in report["fields"] if field["variable"] == "param_228")
    assert rain == {
        "variable": "param_228",
        "valid_time": midnight,
        "reference_time": midnight - 6 * 3600,
        "interval_start": midnight - 3600,
        "interval_end": midnight,
        "start_step": 5 * 3600,
        "end_step": 6 * 3600,
        "step_units": "s",
        "step_type": "accum",
    }
    stored = zarr.open_group(output, mode="r")
    assert stored.metadata.zarr_format == 3
    assert stored["param_167"].dtype == np.dtype(np.float64)
    assert stored["param_167"].chunks == (744, 2, 2)
    assert stored["param_167"].shards == (744, 4, 4)
    assert stored["param_167"].compressors == (zarr.codecs.ZstdCodec(level=3, checksum=True),)
    assert np.array_equal(stored["param_167"][:], np.stack([expected[0], expected[2]]))
    assert np.array_equal(stored["param_228"][:], np.stack([expected[1], expected[3]]))
    assert np.any(expected[0] != expected[0].astype(np.float32).astype(np.float64))
    with xr.open_zarr(output, consolidated=False, chunks=None) as dataset:
        assert dataset["param_167"].dims == ("time", "latitude", "longitude")
        assert dataset["time"].values[0] == np.datetime64("2000-01-01T00:00:00")
        assert dataset["param_228_interval_start"].values[0] == np.datetime64("1999-12-31T23:00:00")
        assert dataset["param_228"].attrs["units"] == "m"


def test_bitmap_missing_values_do_not_remove_field_coverage(
    tmp_path: Path,
    batch: tuple[Path, dict[str, Any], list[dict[str, Any]]],
) -> None:
    """Retain partial and wholly missing spatial fields as present GRIB messages."""
    grib, metadata, messages = batch
    messages[0]["values"] = np.array([np.nan, 9999.0, *range(10)], dtype=np.float64)
    messages[1]["values"] = np.full(12, np.nan)
    expected = write_grib(grib, messages)
    output = tmp_path / "month.zarr"
    report = conversion.convert_batch(grib, metadata, output, spatial_chunks=(2, 2))
    assert len(report["fields"]) == 4
    stored = zarr.open_group(output, mode="r")
    assert np.array_equal(stored["param_167"][0], expected[0], equal_nan=True)
    assert stored["param_167"][0, 0, 1] == 9999.0
    assert np.isnan(stored["param_228"][0]).all()
    assert stored["param_228_coverage"][:].tolist() == [True, True]


@pytest.mark.parametrize("mutation", ["grid", "time", "duplicate", "missing", "parameter"])
def test_inconsistent_input_never_produces_validated_output(
    tmp_path: Path,
    batch: tuple[Path, dict[str, Any], list[dict[str, Any]]],
    mutation: str,
) -> None:
    """Reject mismatched geometry, time coverage, duplicate fields, and parameters."""
    grib, metadata, messages = batch
    if mutation == "grid":
        messages[-1]["longitudeOfFirstGridPointInDegrees"] = 10
    elif mutation == "time":
        messages[0]["dataTime"] = 200
    elif mutation == "duplicate":
        messages.append(messages[0])
    elif mutation == "missing":
        messages.pop()
    else:
        messages[0]["paramId"] = 168
        messages[2]["paramId"] = 168
    write_grib(grib, messages)
    output = tmp_path / "month.zarr"
    with pytest.raises(ValueError):
        conversion.convert_batch(grib, metadata, output, spatial_chunks=(2, 2))
    assert grib.exists()
    assert not output.exists()
    assert not list(tmp_path.glob(".decoded-*"))


def test_add_variable_with_independent_coverage_then_fill_its_gaps(
    tmp_path: Path,
    batch: tuple[Path, dict[str, Any], list[dict[str, Any]]],
) -> None:
    """Preserve a full temperature month while extending sparse wind coverage."""
    grib, metadata, _ = batch
    metadata["request"].update({
        "month": ["02"],
        "day": [f"{day:02d}" for day in range(1, 30)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "variable": ["2m_temperature"],
    })
    messages = [
        {"dataDate": 20000200 + day, "dataTime": hour * 100,
         "values": np.full(12, 270 + day + hour / 24)}
        for day in range(1, 30) for hour in range(24)
    ]
    expected_temperature = np.stack(write_grib(grib, messages))
    first = tmp_path / "temperature.zarr"
    first_report = conversion.convert_batch(grib, metadata, first, (2, 2))
    for field in first_report["fields"]:
        field["source_revision_id"] = "temperature-source"

    wind_metadata = deepcopy(metadata)
    wind_metadata["request"].update({
        "variable": ["10m_u_component_of_wind"], "day": ["01", "03"],
    })
    wind_grib = tmp_path / "wind.grib"
    wind_values = np.stack(write_grib(wind_grib, [
        {"paramId": 165, "dataDate": 20000200 + day, "dataTime": hour * 100,
         "values": np.full(12, day + hour / 24)}
        for day in (1, 3) for hour in range(24)
    ]))
    second = tmp_path / "sparse-wind.zarr"
    second_report = conversion.convert_batch(
        wind_grib, wind_metadata, second, (2, 2),
        previous=first, previous_report=first_report,
    )
    stored = zarr.open_group(second, mode="r")
    assert np.array_equal(stored["param_167"][:], expected_temperature)
    assert stored["param_167_coverage"][:].all()
    wind_rows = np.r_[0:24, 48:72]
    assert np.array_equal(stored["param_165"].oindex[wind_rows, :, :], wind_values)
    assert np.flatnonzero(stored["param_165_coverage"][:]).tolist() == wind_rows.tolist()
    assert np.isnan(stored["param_165"][24:48]).all()
    assert (stored["param_165_reference_time"][24:48] == conversion.MISSING_INTEGER).all()
    assert [field for field in second_report["fields"] if field["variable"] == "param_167"] == first_report["fields"]
    with xr.open_zarr(second, consolidated=False, chunks=None) as dataset:
        assert dataset["param_165_coverage"].dtype == np.dtype(np.bool_)
        assert not dataset["param_165_coverage"].values[24]
        assert np.isnat(dataset["param_165_reference_time"].values[24])
        assert np.isnan(dataset["param_165_start_step"].values[24])

    fill_metadata = deepcopy(wind_metadata)
    fill_metadata["request"]["day"] = ["02", "04"]
    fill_grib = tmp_path / "fill.grib"
    filled_values = np.stack(write_grib(fill_grib, [
        {"paramId": 165, "dataDate": 20000200 + day, "dataTime": hour * 100,
         "values": np.full(12, day + hour / 24)}
        for day in (2, 4) for hour in range(24)
    ]))
    third = tmp_path / "filled-wind.zarr"
    report = conversion.convert_batch(
        fill_grib, fill_metadata, third, (2, 2),
        previous=second, previous_report=second_report,
    )
    merged = zarr.open_group(third, mode="r")
    assert np.array_equal(merged["param_167"][:], expected_temperature)
    assert np.array_equal(merged["param_165"].oindex[wind_rows, :, :], wind_values)
    assert np.array_equal(merged["param_165"].oindex[np.r_[24:48, 72:96], :, :], filled_values)
    assert merged["param_165_coverage"][:96].all()
    assert not merged["param_165_coverage"][96:].any()
    assert report["validation"]["field_count"] == 29 * 24 + 4 * 24
    assert merged.attrs["source_provenance"] == [metadata, wind_metadata, fill_metadata]
    assert np.array_equal(zarr.open_group(first, mode="r")["param_167"][:], expected_temperature)


def test_union_times_preserves_overlaps_and_all_nan_source_fields(
    tmp_path: Path,
    batch: tuple[Path, dict[str, Any], list[dict[str, Any]]],
) -> None:
    """Fill an interior gap and extend the end without refreshing stored fields."""
    grib, metadata, _ = batch
    metadata["request"].update({
        "variable": ["2m_temperature"], "time": ["00:00", "02:00"],
    })
    original = write_grib(grib, [
        {"dataTime": 0, "values": np.full(12, np.nan)},
        {"dataTime": 200, "values": np.full(12, 282.0)},
    ])
    previous = tmp_path / "previous.zarr"
    previous_report = conversion.convert_batch(grib, metadata, previous, (1, 1))
    for field in previous_report["fields"]:
        field["source_revision_id"] = "original-source"
    incoming_metadata = deepcopy(metadata)
    incoming_metadata["request"]["time"] = [f"{hour:02d}:00" for hour in range(4)]
    incoming = tmp_path / "incoming.grib"
    expected = write_grib(incoming, [
        {"dataTime": 0, "values": np.full(12, 999.0)},
        {"dataTime": 100, "values": np.full(12, 281.0)},
        {"dataTime": 100, "startStep": 1, "endStep": 1, "values": np.full(12, 999.0)},
        {"dataTime": 300, "values": np.full(12, 283.0)},
    ])
    output = tmp_path / "merged.zarr"
    report = conversion.convert_batch(
        incoming, incoming_metadata, output, (1, 1),
        previous=previous, previous_report=previous_report,
    )
    stored = zarr.open_group(output, mode="r")
    assert np.array_equal(
        stored["param_167"][:], np.stack([original[0], expected[1], original[1], expected[3]]),
        equal_nan=True,
    )
    assert stored["param_167_coverage"][:].tolist() == [True] * 4
    assert report["fields"][0] == previous_report["fields"][0]
    assert report["fields"][2] == previous_report["fields"][1]
    assert "source_revision_id" not in report["fields"][1]
    assert stored["param_167_start_step"][:].tolist() == [0] * 4
    assert stored.attrs["provenance"] == incoming_metadata
    assert grib.exists() and incoming.exists()
    assert not list(tmp_path.glob(".decoded-*"))


def test_incompatible_previous_metadata_keeps_published_store(
    tmp_path: Path,
    batch: tuple[Path, dict[str, Any], list[dict[str, Any]]],
) -> None:
    """Reject a unit conflict before creating the replacement store."""
    grib, metadata, messages = batch
    write_grib(grib, messages)
    previous = tmp_path / "previous.zarr"
    previous_report = conversion.convert_batch(grib, metadata, previous, (2, 2))
    original = zarr.open_group(previous, mode="r")["param_167"][:]
    previous_report["variables"]["param_167"]["units"] = "degC"
    output = tmp_path / "merged.zarr"
    with pytest.raises(ValueError, match="different metadata"):
        conversion.convert_batch(
            grib, metadata, output, (2, 2),
            previous=previous, previous_report=previous_report,
        )
    assert not output.exists()
    assert np.array_equal(zarr.open_group(previous, mode="r")["param_167"][:], original)
    assert grib.exists()


def test_truncated_grib_is_not_published(
    tmp_path: Path,
    batch: tuple[Path, dict[str, Any], list[dict[str, Any]]],
) -> None:
    """Let ecCodes reject a truncated message while retaining the source file."""
    grib, metadata, messages = batch
    write_grib(grib, messages)
    grib.write_bytes(grib.read_bytes()[:-10])
    with pytest.raises((eccodes.CodesInternalError, ValueError)):
        conversion.convert_batch(grib, metadata, tmp_path / "month.zarr", (2, 2))
    assert grib.exists()
    assert not list(tmp_path.glob(".decoded-*"))


@pytest.mark.parametrize("corruption", ["values", "attributes", "coordinate"])
def test_reopened_zarr_corruption_fails_validation(
    tmp_path: Path,
    batch: tuple[Path, dict[str, Any], list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    """Detect changes made to the real store before it is reopened for validation."""
    grib, metadata, messages = batch
    write_grib(grib, messages)
    output = tmp_path / "month.zarr"
    original_open = zarr.open_group

    def corrupt_before_open(store: Path, *, mode: str, **kwargs: Any) -> zarr.Group:
        """Modify real stored data at the reader boundary."""
        if mode == "r":
            writable = original_open(store, mode="r+")
            if corruption == "values":
                writable["param_167"][0, 0, 0] = -999.0
            elif corruption == "attributes":
                writable["param_167"].attrs["units"] = "wrong units"
            else:
                writable["time"][0] = 0
        return original_open(store, mode=mode, **kwargs)

    monkeypatch.setattr(zarr, "open_group", corrupt_before_open)
    with pytest.raises(ValueError, match="round trip"):
        conversion.convert_batch(grib, metadata, output, spatial_chunks=(2, 2))
    assert grib.exists()
    assert not list(tmp_path.glob(".decoded-*"))
