"""Decode one GRIB batch into a validated, unpublished monthly Zarr store."""

from calendar import monthrange
from contextlib import ExitStack
from datetime import UTC, datetime
import mmap
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import eccodes
import numpy as np
from numpy.typing import NDArray
import zarr
from zarr.codecs import ZstdCodec

GLOBAL_LATITUDES = np.arange(90.0, -90.25, -0.25, dtype=np.float64)
GLOBAL_LONGITUDES = np.arange(0.0, 360.0, 0.25, dtype=np.float64)
TIME_ATTRIBUTES = {
    "units": "seconds since 1970-01-01 00:00:00",
    "calendar": "proleptic_gregorian",
    "timezone": "UTC",
}


def _decode_grib(
    grib: Path,
    times: NDArray[np.int64],
    requested_variables: list[str],
    scratch: Path,
    resources: ExitStack,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Decode messages into disk-backed arrays and validate their field metadata.

    Args:
        grib: Source GRIB file.
        times: Expected UTC timestamps, expressed as Unix seconds.
        requested_variables: CDS names from the completed download request.
        scratch: Directory for temporary decoded arrays.
        resources: Scope that closes all temporary files and memory mappings.

    Returns:
        Decoded arrays, variable attributes, field records, and grid metadata.

    Raises:
        ValueError: Fields have inconsistent grids, metadata, or hour coverage.
        Exception: File and ecCodes failures retain their original exceptions.
    """
    expected_grid = {
        "gridType": "regular_ll",
        "Ni": len(GLOBAL_LONGITUDES),
        "Nj": len(GLOBAL_LATITUDES),
        "scanningMode": 0,
        "latitudeOfFirstGridPointInDegrees": float(GLOBAL_LATITUDES[0]),
        "latitudeOfLastGridPointInDegrees": float(GLOBAL_LATITUDES[-1]),
        "longitudeOfFirstGridPointInDegrees": float(GLOBAL_LONGITUDES[0]),
        "longitudeOfLastGridPointInDegrees": float(GLOBAL_LONGITUDES[-1]),
        "iDirectionIncrementInDegrees": 0.25,
        "jDirectionIncrementInDegrees": 0.25,
    }
    time_indices = {int(value): index for index, value in enumerate(times)}
    shape = (len(times), len(GLOBAL_LATITUDES), len(GLOBAL_LONGITUDES))
    arrays: dict[str, NDArray[np.float64]] = {}
    variables: dict[str, Any] = {}
    fields: list[dict[str, Any]] = []
    seen: dict[str, set[int]] = {}
    with grib.open("rb") as stream:
        while (handle := eccodes.codes_grib_new_from_file(stream)) is not None:
            try:
                grid = {key: eccodes.codes_get(handle, key) for key in expected_grid}
                if grid != expected_grid:
                    raise ValueError("GRIB fields must share the full global 0.25 degree grid.")
                param_id = int(eccodes.codes_get(handle, "paramId"))
                variable = f"param_{param_id}"
                attributes = {
                    "paramId": param_id,
                    "name": eccodes.codes_get(handle, "name"),
                    "shortName": eccodes.codes_get(handle, "shortName"),
                    "cfVarName": eccodes.codes_get(handle, "cfVarName"),
                    "units": eccodes.codes_get(handle, "units"),
                    "type_of_level": eccodes.codes_get(handle, "typeOfLevel"),
                    "level": eccodes.codes_get(handle, "level"),
                    "step_type": eccodes.codes_get(handle, "stepType"),
                    "step_units": "s",
                }
                reference_time = int(
                    datetime.strptime(
                        f"{eccodes.codes_get(handle, 'dataDate'):08d}"
                        f"{eccodes.codes_get(handle, 'dataTime'):04d}",
                        "%Y%m%d%H%M",
                    ).replace(tzinfo=UTC).timestamp()
                )
                valid_time = int(
                    datetime.strptime(
                        f"{eccodes.codes_get(handle, 'validityDate'):08d}"
                        f"{eccodes.codes_get(handle, 'validityTime'):04d}",
                        "%Y%m%d%H%M",
                    ).replace(tzinfo=UTC).timestamp()
                )
                # This changes the decoder's units, not the source GRIB.
                eccodes.codes_set(handle, "stepUnits", "s")
                start_step = eccodes.codes_get_long(handle, "startStep")
                end_step = eccodes.codes_get_long(handle, "endStep")
                if valid_time not in time_indices:
                    raise ValueError(f"{variable} has a valid timestamp outside the request.")
                if start_step > end_step or reference_time + end_step != valid_time:
                    raise ValueError(f"{variable} has inconsistent forecast interval timestamps.")
                if variable in variables:
                    if attributes != variables[variable]:
                        raise ValueError(f"{variable} has inconsistent variable metadata.")
                    if valid_time in seen[variable]:
                        raise ValueError(f"{variable} has a duplicate valid timestamp.")
                else:
                    variables[variable] = attributes
                    seen[variable] = set()
                    raw = resources.enter_context((scratch / f"{variable}.bin").open("w+b"))
                    raw.truncate(int(np.prod(shape)) * np.dtype(np.float64).itemsize)
                    mapping = resources.enter_context(mmap.mmap(raw.fileno(), 0))
                    arrays[variable] = np.ndarray(shape, dtype=np.float64, buffer=mapping)
                values = eccodes.codes_get_values(handle)
                if eccodes.codes_get(handle, "bitmapPresent"):
                    bitmap = eccodes.codes_get_array(handle, "bitmap")
                    values[bitmap == 0] = np.nan
                if np.isinf(values).any():
                    raise ValueError(f"{variable} contains infinite values.")
                arrays[variable][time_indices[valid_time]] = values.reshape(shape[1:])
                seen[variable].add(valid_time)
                fields.append({
                    "variable": variable,
                    "valid_time": valid_time,
                    "reference_time": reference_time,
                    "interval_start": reference_time + start_step,
                    "interval_end": reference_time + end_step,
                    "start_step": start_step,
                    "end_step": end_step,
                    "step_units": "s",
                    "step_type": attributes["step_type"],
                })
            finally:
                eccodes.codes_release(handle)

    if len(variables) != len(set(requested_variables)):
        raise ValueError("Decoded parameter count differs from the requested variable count.")
    expected_times = set(time_indices)
    if not variables or any(coverage != expected_times for coverage in seen.values()):
        raise ValueError("Every decoded parameter must contain all requested UTC hours.")
    fields.sort(key=lambda item: (item["variable"], item["valid_time"]))
    grid_metadata = {
        "type": "regular_ll",
        "shape": list(shape[1:]),
        "latitude_first": float(GLOBAL_LATITUDES[0]),
        "latitude_last": float(GLOBAL_LATITUDES[-1]),
        "longitude_first": float(GLOBAL_LONGITUDES[0]),
        "longitude_last": float(GLOBAL_LONGITUDES[-1]),
        "resolution": [0.25, 0.25],
        "scanning_mode": 0,
    }
    return arrays, variables, fields, grid_metadata


def convert_batch(
    grib: Path,
    metadata: dict[str, Any],
    output: Path,
    spatial_chunks: tuple[int, int] = (32, 32),
) -> dict[str, Any]:
    """Write and reopen one batch as unpublished monthly Zarr, retaining its GRIB.

    CDS request names remain in provenance; GRIB parameter IDs identify stored
    variables. Validation compares parameter counts and hour sets, without
    guessing a CDS-name-to-parameter association. The caller verifies the source
    checksum and owns publication, coverage recording, and GRIB cleanup.

    Args:
        grib: Completed staged GRIB file.
        metadata: Download metadata containing the exact single-month request.
        output: New directory for the unpublished Zarr store.
        spatial_chunks: Latitude and longitude cells per read chunk.

    Returns:
        A JSON-compatible report with year, month, grid, variables, validated
        fields, and numerical and metadata round-trip results.

    Raises:
        ValueError: Chunk dimensions, GRIB fields, or the Zarr round trip are invalid.
        Exception: File, decoding, and Zarr errors propagate from their libraries.
    """
    if len(spatial_chunks) != 2 or any(size < 1 for size in spatial_chunks):
        raise ValueError("Spatial chunk dimensions must be positive.")
    request = metadata["request"]
    year, month = int(request["year"][0]), int(request["month"][0])
    times = np.array(
        sorted(
            int(datetime.fromisoformat(
                f"{year:04d}-{month:02d}-{int(day):02d}T{hour}:00+00:00"
            ).timestamp())
            for day in request["day"]
            for hour in request["time"]
        ),
        dtype=np.int64,
    )
    month_hours = monthrange(year, month)[1] * 24
    chunks = (month_hours, *spatial_chunks)
    shards = (month_hours, spatial_chunks[0] * 2, spatial_chunks[1] * 2)
    compressor = ZstdCodec(level=3, checksum=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".decoded-", dir=output.parent) as scratch:
        with ExitStack() as resources:
            arrays, variables, fields, grid = _decode_grib(
                grib, times, request["variable"], Path(scratch), resources
            )
            group_attributes = {
                "Conventions": "CF-1.12",
                "source": metadata["dataset"],
                "provenance": metadata,
                "grid": grid,
                "year": year,
                "month": month,
                "spatial_chunks": list(spatial_chunks),
                "variable_identity": "ECMWF GRIB paramId; CDS names retained in provenance",
            }
            root = zarr.open_group(output, mode="w-", zarr_format=3)
            root.attrs.update(group_attributes)
            coordinate_values = {
                "time": times,
                "latitude": GLOBAL_LATITUDES,
                "longitude": GLOBAL_LONGITUDES,
            }
            coordinate_attributes = {
                "time": {**TIME_ATTRIBUTES, "standard_name": "time", "axis": "T"},
                "latitude": {"units": "degrees_north", "standard_name": "latitude", "axis": "Y"},
                "longitude": {"units": "degrees_east", "standard_name": "longitude", "axis": "X"},
            }
            expected_attributes = dict(coordinate_attributes)
            for name, data in coordinate_values.items():
                root.create_array(
                    name, data=data, chunks=(len(data),),
                    compressors=[compressor], dimension_names=(name,),
                    attributes=coordinate_attributes[name],
                )
            for variable, data in arrays.items():
                records = [field for field in fields if field["variable"] == variable]
                auxiliary_names = []
                for key in ("reference_time", "interval_start", "interval_end", "start_step", "end_step"):
                    name = f"{variable}_{key}"
                    auxiliary_names.append(name)
                    auxiliary = np.array([field[key] for field in records], dtype=np.int64)
                    attributes = {"long_name": f"{variables[variable]['name']}: {key}"}
                    if key in ("reference_time", "interval_start", "interval_end"):
                        attributes.update(TIME_ATTRIBUTES)
                    else:
                        attributes["units"] = "s"
                    coordinate_values[name] = auxiliary
                    expected_attributes[name] = attributes
                    root.create_array(
                        name, data=auxiliary, chunks=(month_hours,),
                        compressors=[compressor], dimension_names=("time",),
                        attributes=attributes,
                    )
                attributes = {
                    **variables[variable],
                    "long_name": variables[variable]["name"],
                    "ancillary_variables": " ".join(auxiliary_names),
                }
                expected_attributes[variable] = attributes
                target = root.create_array(
                    variable, shape=data.shape, dtype=np.float64, fill_value=np.nan,
                    chunks=chunks, shards=shards, compressors=[compressor],
                    dimension_names=("time", "latitude", "longitude"),
                    attributes=attributes,
                )
                for latitude in range(0, data.shape[1], shards[1]):
                    for longitude in range(0, data.shape[2], shards[2]):
                        selection = (
                            slice(None),
                            slice(latitude, latitude + shards[1]),
                            slice(longitude, longitude + shards[2]),
                        )
                        target[selection] = data[selection]

            reopened = zarr.open_group(output, mode="r")
            if dict(reopened.attrs) != group_attributes:
                raise ValueError("Zarr group metadata changed during the round trip.")
            if set(reopened.array_keys()) != set(expected_attributes):
                raise ValueError("Zarr array inventory changed during the round trip.")
            for name, attributes in expected_attributes.items():
                if dict(reopened[name].attrs) != attributes:
                    raise ValueError(f"{name}: Zarr attributes changed during the round trip.")
            for name, expected in coordinate_values.items():
                actual = reopened[name]
                dimension = name if name in ("latitude", "longitude") else "time"
                if (
                    actual.dtype != expected.dtype
                    or actual.metadata.dimension_names != (dimension,)
                    or not np.array_equal(actual[:], expected)
                ):
                    raise ValueError(f"{name}: coordinate or interval metadata round trip failed.")
            compared_shards = 0
            for variable, expected in arrays.items():
                actual = reopened[variable]
                if (
                    actual.dtype != np.dtype(np.float64)
                    or actual.shape != expected.shape
                    or actual.chunks != chunks
                    or actual.shards != shards
                    or actual.compressors != (compressor,)
                    or actual.metadata.dimension_names != ("time", "latitude", "longitude")
                ):
                    raise ValueError(f"{variable}: Zarr storage metadata round trip failed.")
                for latitude in range(0, expected.shape[1], shards[1]):
                    for longitude in range(0, expected.shape[2], shards[2]):
                        selection = (
                            slice(None),
                            slice(latitude, latitude + shards[1]),
                            slice(longitude, longitude + shards[2]),
                        )
                        if not np.array_equal(
                            actual[selection], expected[selection], equal_nan=True
                        ):
                            raise ValueError(f"{variable}: numerical Zarr round trip failed.")
                        compared_shards += 1
            return {
                "year": year,
                "month": month,
                "grid": grid,
                "layout": {
                    "spatial_chunks": list(spatial_chunks),
                    "time_chunk": "calendar_month",
                    "shard_spatial_multiple": 2,
                },
                "variables": variables,
                "fields": fields,
                "validation": {
                    "field_count": len(fields),
                    "variable_count": len(variables),
                    "compared_shards": compared_shards,
                    "numerical_roundtrip": "exact_float64_equal_nan",
                    "metadata_roundtrip": True,
                    "request_variable_mapping": "distinct_parameter_count_only",
                },
            }
