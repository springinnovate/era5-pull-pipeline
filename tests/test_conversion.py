"""Verify exact pixels, native sample centers, and COG display overviews."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from era5_pull_pipeline.conversion import write_observation_cog

START = int(datetime(1950, 1, 1, tzinfo=UTC).timestamp())
GRID = {
    "shape": [721, 1440], "resolution": [0.25, 0.25],
    "latitude_first": 90.0, "longitude_first": 0.0,
}


def observation(step_type: str = "instant") -> tuple[dict[str, Any], dict[str, Any]]:
    """Return source-reader metadata for an instantaneous or interval field.

    Args:
        step_type: Source temporal meaning: instant, avg, or accum.

    Returns:
        One field's time/source metadata and variable attributes.
    """
    interval = step_type != "instant"
    field = {
        "variable": "sample_variable", "valid_time": START,
        "reference_time": START - 21600 if interval else None,
        "interval_start": START - 3600 if interval else START,
        "interval_end": START, "start_step": 18000 if interval else None,
        "end_step": 21600 if interval else None, "step_units": "s",
        "step_type": step_type, "source_variable": "sample",
        "source_file_id": "provider-object-revision",
    }
    attributes = {"name": "Sample variable", "units": "K"}
    return field, attributes


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_global_pixels_precision_and_sample_centers(tmp_path: Path, dtype: Any) -> None:
    """Preserve all base pixels and NaN locations across the seam and polar rows."""
    values = np.linspace(-3.141592653589793, 12345.123456789, 721 * 1440, dtype=dtype).reshape(721, 1440)
    values[0, 720] = np.nan
    values[-1, -1] = np.nan
    values[123, 456] = -0.0
    field, attributes = observation()
    path = tmp_path / "global.tif"
    report = write_observation_cog(values, field, attributes, GRID, path)

    with rasterio.open(path) as raster:
        expected = np.concatenate((values[:, 720:], values[:, :720]), axis=1)
        np.testing.assert_array_equal(raster.read(1), expected)
        np.testing.assert_array_equal(raster.read_masks(1) == 0, np.isnan(expected))
        assert raster.dtypes == (np.dtype(dtype).name,)
        assert raster.transform == Affine(0.25, 0, -180.125, 0, -0.25, 90.125)
        assert raster.xy(0, 0) == (-180.0, 90.0)
        assert raster.xy(720, 1439) == (179.75, -90.0)
        assert tuple(raster.bounds) == (-180.125, -90.125, 179.875, 90.125)
        assert raster.crs.to_epsg() == 4326
        assert raster.tags()["AREA_OR_POINT"] == "Point"
        assert raster.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
        assert raster.tags(ns="IMAGE_STRUCTURE")["COMPRESSION"] == "DEFLATE"
        assert raster.tags(ns="IMAGE_STRUCTURE")["PREDICTOR"] == "3"
        assert raster.block_shapes == [(512, 512)]
        assert raster.overviews(1) == [2, 4]
        assert raster.units == ("K",)
        assert raster.tags()["valid_time"] == str(START)
        assert "reference_time" not in raster.tags()
    assert report["transform"] == [-180.125, 0.25, 0, 90.125, 0, -0.25]
    assert report["shape"] == [721, 1440]
    assert report["nodata"] == "NaN"


def test_overviews_use_nearest_samples_without_modifying_base_pixels(tmp_path: Path) -> None:
    """Display overviews select source samples instead of averaging the field."""
    values = np.arange(721 * 1440, dtype=np.float32).reshape(721, 1440)
    field, attributes = observation()
    path = tmp_path / "overview.tif"
    write_observation_cog(values, field, attributes, GRID, path)
    expected = np.concatenate((values[:, 720:], values[:, :720]), axis=1)
    with rasterio.open(path, overview_level=0) as overview:
        rows = np.floor(np.arange(overview.height) * 721 / overview.height + 0.5).astype(int)
        columns = np.floor(np.arange(overview.width) * 1440 / overview.width + 0.5).astype(int)
        np.testing.assert_array_equal(overview.read(1), expected[np.ix_(rows, columns)])
    with rasterio.open(path) as raster:
        np.testing.assert_array_equal(raster.read(1), expected)


@pytest.mark.parametrize("step_type,temporal_type,units", [
    ("avg", "mean", "kg m**-2 s**-1"),
    ("accum", "accumulation", "W m**-2 s"),
])
def test_all_nodata_observation_preserves_interval_and_units(
    tmp_path: Path, step_type: str, temporal_type: str, units: str,
) -> None:
    """Retain a covered all-NoData observation and its source time semantics."""
    values = np.full((3, 4), np.nan, dtype=np.float32)
    field, attributes = observation(step_type)
    attributes["units"] = units
    path = tmp_path / "missing-values.tif"
    write_observation_cog(values, field, attributes, GRID, path)
    with rasterio.open(path) as raster:
        assert np.isnan(raster.read(1)).all()
        assert not raster.read_masks(1).any()
        assert raster.units == (units,)
        tags = raster.tags()
        assert tags["temporal_type"] == temporal_type
        assert tags["interval_start"] == str(START - 3600)
        assert tags["interval_end"] == str(START)
        assert tags["reference_time"] == str(START - 21600)
        assert tags["start_step"] == "18000"
        assert tags["end_step"] == "21600"
        assert tags["source_file_id"] == "provider-object-revision"
