"""Read selected hourly data from NCAR NetCDF files into NumPy arrays."""

from collections.abc import Sequence
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from itertools import product
import mmap
from pathlib import Path
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from era5_pull_pipeline import conversion


def _attributes_to_json_values(values: dict[str, Any]) -> dict[str, Any]:
    """Convert NetCDF attribute scalars and arrays to JSON-compatible values.

    Args:
        values: Attributes returned by the NetCDF decoder.

    Returns:
        Equivalent attributes using Python scalar and list types.
    """
    return {
        key: value.tolist() if isinstance(value, (np.ndarray, np.generic)) else value
        for key, value in values.items()
    }


def read_ncar_netcdf_arrays(
    path: Path,
    source_id: str,
    source_variable: str,
    selected_times: Sequence[int],
    scratch: Path,
    resources: ExitStack,
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """Read selected hours from an NCAR NetCDF file into NumPy arrays.

    The arrays have dimensions (time, latitude, longitude) and use temporary
    memory-mapped files. Keep resources open while using the returned arrays.
    Read each needed NetCDF chunk once and preserve its decoded values and units.

    Analysis fields use their time coordinate. Forecast valid times are
    initialization time plus lead hours. Accumulations and mean rates retain
    their preceding-hour intervals. The returned report includes field coverage,
    interval endpoints, and the original file attributes for COG publication.

    Args:
        path: Downloaded NCAR NetCDF file recorded in the catalog.
        source_id: Catalog identity of this exact provider file revision.
        source_variable: NCAR short name selected by the user.
        selected_times: Requested UTC Unix seconds, all in one calendar month.
        scratch: Existing directory for one source's disk-backed decoded values.
            Its values.bin file is reused by the next call.
        resources: ExitStack that closes the temporary files and memory mappings.

    Returns:
        A dictionary mapping the variable's readable name to its NumPy array,
        and a report containing field times, grid, source identity, and attributes.

    Raises:
        ValueError: The external file lacks selected hours, duplicates valid
            hours, or does not contain the selected full global variable.
        Exception: NetCDF and filesystem failures propagate.
    """
    requested = np.array(sorted(selected_times), dtype=np.int64)
    month = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=int(requested[0]))
    with xr.open_dataset(path, engine="h5netcdf", decode_timedelta=False) as dataset:
        if not (
            np.array_equal(dataset.latitude.values, conversion.GLOBAL_LATITUDES)
            and np.array_equal(dataset.longitude.values, conversion.GLOBAL_LONGITUDES)
        ):
            raise ValueError("NCAR fields must use the full global 0.25 degree grid.")
        candidates = [
            name for name, array in dataset.data_vars.items()
            if array.attrs.get("short_name") == source_variable
            and array.dims[-2:] == ("latitude", "longitude")
        ]
        if len(candidates) != 1:
            raise ValueError(f"NCAR file does not contain one global {source_variable} variable.")
        array = dataset[candidates[0]]
        temporal_dimensions = array.dims[:-2]
        forecast = temporal_dimensions == ("forecast_initial_time", "forecast_hour")
        if forecast:
            references = dataset.forecast_initial_time.values.astype("datetime64[s]").astype(np.int64)
            # NCAR encodes lead times as integer hours with units="hour", which
            # xarray does not automatically recognize as a timedelta unit.
            leads = dataset.forecast_hour.values.astype("timedelta64[h]").astype("timedelta64[s]").astype(np.int64)
            valid = references[:, None] + leads[None, :]
            references = np.broadcast_to(references[:, None], valid.shape)
            ends = np.broadcast_to(leads[None, :], valid.shape)
            starts = ends - 3600
            step_type = "avg" if ".meanflux." in path.name else "accum"
        else:
            if temporal_dimensions != ("time",):
                raise ValueError(f"Unsupported NCAR time dimensions: {temporal_dimensions}.")
            valid = dataset.time.values.astype("datetime64[s]").astype(np.int64)
            references = valid
            starts = ends = np.zeros(valid.shape, dtype=np.int64)
            step_type = "instant"
        flat_indices = np.flatnonzero(np.isin(valid, requested))
        actual = valid.reshape(-1)[flat_indices]
        if not np.array_equal(np.sort(actual), requested):
            raise ValueError(f"NCAR {source_variable} does not contain each selected UTC hour exactly once.")
        order = np.argsort(actual)
        flat_indices = flat_indices[order]
        indices = np.column_stack(np.unravel_index(flat_indices, valid.shape))
        name = str(array.attrs["long_name"])
        variable = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        attributes = {
            "name": name,
            "short_name": source_variable,
            "units": array.attrs["units"],
            "source_variable": source_variable,
            "source_dtype": str(array.dtype),
            "step_type": step_type,
            "step_units": "s",
        }
        attributes.update(_attributes_to_json_values({
            key: value for key, value in array.attrs.items()
            if key in {"ecmwf_local_table", "ecmwf_parameter", "number_of_significant_digits"}
            or key.startswith("Quantize")
        }))
        shape = (len(requested), len(dataset.latitude), len(dataset.longitude))
        raw = resources.enter_context((scratch / "values.bin").open("w+b"))
        raw.truncate(int(np.prod(shape)) * array.dtype.itemsize)
        mapping = resources.enter_context(mmap.mmap(raw.fileno(), 0))
        decoded = np.ndarray(shape, dtype=array.dtype, buffer=mapping)

        # Read each compressed native chunk once. A single-hour loop would
        # repeatedly decompress NCAR chunks that contain multiple time steps.
        chunks = array.encoding["chunksizes"]
        temporal_chunks: dict[tuple[int, ...], list[int]] = {}
        for row, coordinate in enumerate(indices):
            chunk = tuple(int(index) // size for index, size in zip(coordinate, chunks))
            temporal_chunks.setdefault(chunk, []).append(row)
        for chunk, rows in temporal_chunks.items():
            temporal_slices = {
                dimension: slice(index * size, min((index + 1) * size, array.sizes[dimension]))
                for dimension, index, size in zip(temporal_dimensions, chunk, chunks)
            }
            local_indices = tuple(
                indices[rows, axis] - selection.start
                for axis, selection in enumerate(temporal_slices.values())
            )
            for latitude, longitude in product(
                range(0, shape[1], chunks[-2]), range(0, shape[2], chunks[-1])
            ):
                lat = slice(latitude, min(latitude + chunks[-2], shape[1]))
                lon = slice(longitude, min(longitude + chunks[-1], shape[2]))
                block = array.isel(**temporal_slices, latitude=lat, longitude=lon).values
                decoded[rows, lat, lon] = block[local_indices]

        fields = [
            {
                "variable": variable,
                "source_variable": source_variable,
                "source_file_id": source_id,
                "valid_time": int(valid.reshape(-1)[index]),
                "reference_time": (
                    int(references.reshape(-1)[index]) if forecast else None
                ),
                "interval_start": int(references.reshape(-1)[index] + starts.reshape(-1)[index]),
                "interval_end": int(references.reshape(-1)[index] + ends.reshape(-1)[index]),
                "start_step": int(starts.reshape(-1)[index]) if forecast else None,
                "end_step": int(ends.reshape(-1)[index]) if forecast else None,
                "step_units": "s",
                "step_type": step_type,
            }
            for index in flat_indices
        ]
        return {variable: decoded}, {
            "year": month.year,
            "month": month.month,
            "grid": {
                "type": "regular_ll",
                "shape": list(shape[1:]),
                "latitude_first": float(dataset.latitude.values[0]),
                "latitude_last": float(dataset.latitude.values[-1]),
                "longitude_first": float(dataset.longitude.values[0]),
                "longitude_last": float(dataset.longitude.values[-1]),
                "resolution": [0.25, 0.25],
                "scanning_mode": 0,
            },
            "variables": {variable: attributes},
            "fields": fields,
            "source_attributes": {
                "global": _attributes_to_json_values(dataset.attrs),
                "variable": _attributes_to_json_values(array.attrs),
                "coordinates": {
                    key: _attributes_to_json_values(dataset[key].attrs)
                    for key in (*temporal_dimensions, "latitude", "longitude")
                },
                "encoding": {
                    key: {
                        "dtype": str(dataset[key].encoding["dtype"]),
                        **_attributes_to_json_values({
                            name: value for name, value in dataset[key].encoding.items()
                            if name in {"units", "calendar", "_FillValue", "missing_value", "scale_factor", "add_offset"}
                        }),
                    }
                    for key in (candidates[0], *temporal_dimensions, "latitude", "longitude")
                },
            },
        }
