"""Write validated COG rasters from decoded observation arrays."""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import rasterio
from rasterio.transform import Affine

GLOBAL_LATITUDES = np.arange(90.0, -90.25, -0.25, dtype=np.float64)
GLOBAL_LONGITUDES = np.arange(0.0, 360.0, 0.25, dtype=np.float64)


def write_observation_cog(
    values: NDArray[Any],
    field: dict[str, Any],
    attributes: dict[str, Any],
    grid: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    """Write and validate one observation as a losslessly compressed COG.

    Reorder longitude columns into -180 through 180 degrees without changing
    values or sample centers. Internal overviews use nearest-neighbor sampling;
    the full-resolution band retains the decoded source precision and NoData.
    The caller publishes the file only after this function returns successfully.

    Args:
        values: Decoded two-dimensional array in latitude, longitude order.
        field: Observation variable, UTC Unix times, interval, and source identity.
        attributes: Variable description and native units from the source reader.
        grid: Regular latitude/longitude grid with first centers and resolution.
        path: Unpublished TIFF path owned by the publication journal.

    Returns:
        JSON-compatible raster metadata. Transform uses GDAL's six-value order:
        west edge, pixel width, row rotation, north edge, column rotation,
        pixel height. Floating-point NoData is represented as the string "NaN".

    Raises:
        ValueError: Written values, georeferencing, or metadata differ from input.
        Exception: Rasterio and filesystem errors propagate from their libraries.
    """
    latitude_step, longitude_step = grid["resolution"]
    longitudes = grid["longitude_first"] + np.arange(values.shape[1]) * longitude_step
    wrapped = (longitudes + 180) % 360 - 180
    order = np.argsort(wrapped)
    reordered = values[:, order]
    transform = Affine(
        longitude_step, 0, float(wrapped[order[0]]) - longitude_step / 2,
        0, -latitude_step, grid["latitude_first"] + latitude_step / 2,
    )
    floating = np.issubdtype(values.dtype, np.floating)
    nodata = np.nan if floating else None
    temporal_type = {"instant": "instant", "avg": "mean", "accum": "accumulation"}[field["step_type"]]
    tags = {
        "AREA_OR_POINT": "Point",
        "variable": field["variable"],
        "units": attributes["units"],
        "temporal_type": temporal_type,
        "time_units": "seconds since 1970-01-01 00:00:00 UTC",
        **{
            key: str(field[key])
            for key in (
                "valid_time", "reference_time", "interval_start", "interval_end",
                "start_step", "end_step", "step_units", "step_type",
                "source_file_id", "source_variable",
            )
            if field[key] is not None
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="COG", height=values.shape[0], width=values.shape[1],
        count=1, dtype=values.dtype, crs="EPSG:4326", transform=transform,
        nodata=nodata, compress="DEFLATE", blocksize=512,
        predictor="FLOATING_POINT" if floating else "STANDARD",
        overview_resampling="NEAREST",
    ) as target:
        target.write(reordered, 1)
        target.set_band_unit(1, attributes["units"])
        target.set_band_description(1, attributes["name"])
        target.update_tags(**tags)

    with rasterio.open(path) as actual:
        structure = actual.tags(ns="IMAGE_STRUCTURE")
        if (
            actual.count != 1 or actual.shape != values.shape
            or np.dtype(actual.dtypes[0]) != values.dtype
            or actual.crs != rasterio.crs.CRS.from_epsg(4326)
            or actual.transform != transform
            or actual.block_shapes != [(512, 512)]
            or structure.get("LAYOUT") != "COG"
            or structure.get("COMPRESSION") != "DEFLATE"
            or (not np.isnan(actual.nodata) if floating else actual.nodata is not None)
            or actual.units != (attributes["units"],)
            or actual.descriptions != (attributes["name"],)
            or any(actual.tags().get(key) != value for key, value in tags.items())
        ):
            raise ValueError(f"{path}: COG spatial or observation metadata round trip failed.")
        if not np.array_equal(actual.read(1), reordered, equal_nan=True):
            raise ValueError(f"{path}: COG numerical round trip failed.")
        overviews = actual.overviews(1)

    return {
        "shape": list(values.shape), "dtype": str(values.dtype),
        "nodata": "NaN" if floating else None, "crs": "EPSG:4326",
        "transform": list(transform.to_gdal()), "band": 1,
        "block_shape": [512, 512], "compression": "DEFLATE",
        "overview_resampling": "nearest", "overviews": overviews,
    }


def write_decoded_observation_cog(
    decoded_path: Path, index: int, field: dict[str, Any],
    attributes: dict[str, Any], grid: dict[str, Any], path: Path,
) -> dict[str, Any]:
    """Read one decoded hour from scratch storage and write its validated COG.

    Each process reads only its hourly raster; month-sized arrays never travel
    through the process pool's serialization queue.

    Args:
        decoded_path: Completed values.bin containing one source's selected hours.
        index: Hour's row in the decoded array and ordered metadata report.
        field: Observation timestamps and source identity.
        attributes: Variable name, units, and decoded data type.
        grid: Source grid coordinates and dimensions.
        path: Journaled staging path owned by this observation.

    Returns:
        Validated raster metadata for the coordinator's publication transaction.

    Raises:
        Exception: Scratch reads, COG writes, and validation failures propagate.
    """
    dtype = np.dtype(attributes["source_dtype"])
    count = int(np.prod(grid["shape"]))
    with decoded_path.open("rb") as stream:
        values = np.fromfile(stream, dtype=dtype, count=count, offset=index * count * dtype.itemsize)
    return write_observation_cog(values.reshape(grid["shape"]), field, attributes, grid, path)
