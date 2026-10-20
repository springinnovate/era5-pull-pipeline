"""Decode GRIB batches and assemble validated, unpublished monthly Zarr stores."""

from calendar import monthrange
from collections.abc import Sequence
from contextlib import ExitStack
from datetime import UTC, datetime
import mmap
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any

import eccodes
import numpy as np
from numpy.typing import NDArray
import zarr
from zarr.codecs import ZstdCodec

GLOBAL_LATITUDES = np.arange(90.0, -90.25, -0.25, dtype=np.float64)
GLOBAL_LONGITUDES = np.arange(0.0, 360.0, 0.25, dtype=np.float64)
MISSING_INTEGER = np.iinfo(np.int64).min
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
    parameter_names: dict[int, str] = {}
    with grib.open("rb") as stream:
        while (handle := eccodes.codes_grib_new_from_file(stream)) is not None:
            try:
                grid = {key: eccodes.codes_get(handle, key) for key in expected_grid}
                if grid != expected_grid:
                    raise ValueError("GRIB fields must share the full global 0.25 degree grid.")
                param_id = int(eccodes.codes_get(handle, "paramId"))
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
                variable = re.sub(r"[^a-z0-9]+", "_", attributes["name"].lower()).strip("_")
                if parameter_names.setdefault(param_id, variable) != variable:
                    raise ValueError(f"Parameter {param_id} has inconsistent descriptive names.")
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
                    if variables[variable]["paramId"] != param_id:
                        raise ValueError(f"Different parameters share the array name {variable}.")
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

    if len(parameter_names) != len(set(requested_variables)):
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


def _merged_shard(
    sources: list[
        tuple[NDArray[np.float64] | zarr.Array, NDArray[np.intp], NDArray[np.intp]]
    ],
    selection: tuple[slice, slice, slice],
    time_count: int,
) -> NDArray[np.float64]:
    """Assemble one spatial shard from explicitly covered source hours.

    Args:
        sources: Arrays with source and destination time indices. Their
            destination indices are disjoint, with existing fields preferred.
        selection: Full time axis and bounded latitude/longitude slices.
        time_count: Length of the merged monthly time axis.

    Returns:
        Float64 values with NaN only where a field or source cell is missing.
    """
    _, latitude, longitude = selection
    merged = np.full(
        (time_count, latitude.stop - latitude.start, longitude.stop - longitude.start),
        np.nan,
        dtype=np.float64,
    )
    for source, source_indices, destination_indices in sources:
        if isinstance(source, zarr.Array):
            values = source.oindex[source_indices, latitude, longitude]
        else:
            values = source[source_indices, latitude, longitude]
        merged[destination_indices] = values
    return merged


def convert_batch(
    grib: Path,
    metadata: dict[str, Any],
    output: Path,
    spatial_chunks: tuple[int, int] = (32, 32),
    *,
    previous: Path | None = None,
    previous_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decode and validate a batch checkpoint or single-batch monthly replacement.

    Descriptive ecCodes names identify arrays; parameter IDs remain in metadata.
    Validation checks distinct parameter counts and complete requested hours,
    trusting CDS to select the requested variables. The caller verifies source
    integrity and owns checkpoint persistence, publication, and GRIB cleanup.

    Args:
        grib: Completed staged GRIB file.
        metadata: Download metadata containing the exact single-month request.
        output: New directory for the unpublished Zarr store.
        spatial_chunks: Latitude and longitude cells per read chunk.
        previous: Published monthly store to preserve, when extending a month.
        previous_report: Its catalog report, including explicit field coverage.

    Returns:
        A JSON-compatible report of exact field coverage and round-trip checks.

    Raises:
        ValueError: Source metadata, field coverage, or the round trip is invalid.
        Exception: File, decoding, and Zarr errors propagate from their libraries.
    """
    request = metadata["request"]
    year, month = int(request["year"][0]), int(request["month"][0])
    times = np.array(
        sorted(
            int(datetime.fromisoformat(
                f"{year:04d}-{month:02d}-{int(day):02d}T{hour}:00+00:00"
            ).timestamp())
            for day in request["day"] for hour in request["time"]
        ),
        dtype=np.int64,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".decoded-", dir=output.parent) as scratch:
        with ExitStack() as resources:
            arrays, variables, fields, grid = _decode_grib(
                grib, times, request["variable"], Path(scratch), resources
            )
            report = {
                "year": year,
                "month": month,
                "grid": grid,
                "variables": variables,
                "fields": [
                    {**field, "source_request_sha256": metadata["request_sha256"]}
                    for field in fields
                ],
            }
            return _write_month(
                [(arrays, report)], metadata, output, spatial_chunks,
                previous=previous, previous_report=previous_report,
            )


def assemble_month(
    checkpoints: Sequence[tuple[Path, dict[str, Any]]],
    metadata: dict[str, Any],
    output: Path,
    spatial_chunks: tuple[int, int] = (32, 32),
    *,
    previous: Path | None = None,
    previous_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble completed batch checkpoints into one validated monthly replacement.

    The caller supplies completed checkpoints and their recorded reports. This
    function reads those stores by spatial shard without decoding GRIB again.
    Previously published fields take precedence over incoming data.

    Args:
        checkpoints: Checkpoint Zarr paths and their validated conversion reports.
        metadata: Monthly publication manifest with dataset, request_sha256, and
            the source download metadata in sources.
        output: New directory for the unpublished monthly store.
        spatial_chunks: Latitude and longitude cells per read chunk.
        previous: Published monthly store to preserve, when present.
        previous_report: Its catalog report, including exact field coverage.

    Returns:
        A JSON-compatible report of all retained and newly assembled fields.

    Raises:
        ValueError: Sources conflict or overlap.
        Exception: Zarr and filesystem errors propagate from their libraries.
    """
    incoming = []
    for path, report in checkpoints:
        group = zarr.open_group(path, mode="r")
        incoming.append((
            {variable: group[variable] for variable in report["variables"]},
            report,
        ))
    return _write_month(
        incoming, metadata, output, spatial_chunks,
        previous=previous, previous_report=previous_report,
    )


def _write_month(
    incoming: Sequence[tuple[dict[str, NDArray[np.float64] | zarr.Array], dict[str, Any]]],
    metadata: dict[str, Any],
    output: Path,
    spatial_chunks: tuple[int, int],
    *,
    previous: Path | None,
    previous_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge source coverage, write each spatial shard, and validate reopened data.

    Args:
        incoming: Source arrays and validated reports for one calendar month.
        metadata: Source or monthly publication provenance.
        output: New directory for the unpublished store.
        spatial_chunks: Latitude and longitude cells per read chunk.
        previous: Published store whose fields take precedence.
        previous_report: Its catalog report with field metadata and source origins.

    Returns:
        A JSON-compatible report with exact numerical and metadata validation.

    Raises:
        ValueError: Source month, grid, metadata, coverage, or round trip differs.
        Exception: Zarr and filesystem errors propagate from their libraries.
    """
    if not incoming:
        raise ValueError("Monthly assembly requires at least one completed batch.")
    if len(spatial_chunks) != 2 or any(size < 1 for size in spatial_chunks):
        raise ValueError("Spatial chunk dimensions must be positive.")
    if (previous is None) != (previous_report is None):
        raise ValueError("A previous store and its coverage report must be supplied together.")
    first_report = incoming[0][1]
    year, month, grid = first_report["year"], first_report["month"], first_report["grid"]
    all_sources = list(incoming)
    source_provenance = []
    if previous is not None and previous_report is not None:
        prior = zarr.open_group(previous, mode="r")
        if (
            previous_report["layout"]["spatial_chunks"] != list(spatial_chunks)
            or not np.array_equal(prior["latitude"][:], GLOBAL_LATITUDES)
            or not np.array_equal(prior["longitude"][:], GLOBAL_LONGITUDES)
        ):
            raise ValueError("The previous month has different coordinates or layout.")
        all_sources.insert(0, (
            {variable: prior[variable] for variable in previous_report["variables"]},
            previous_report,
        ))
        source_provenance = list(prior.attrs["source_provenance"])
    source_provenance.extend(metadata.get("sources", [metadata]))
    variables: dict[str, Any] = {}
    parameter_names: dict[int, str] = {}
    merged_fields: dict[tuple[str, int], dict[str, Any]] = {}
    incoming_fields: set[tuple[str, int]] = set()
    pending_sources = []
    for source_index, (arrays, report) in enumerate(all_sources):
        is_previous = previous is not None and source_index == 0
        if (report["year"], report["month"], report["grid"]) != (year, month, grid):
            raise ValueError("Monthly sources must share the same calendar month and grid.")
        source_times = np.array(sorted({field["valid_time"] for field in report["fields"]}), dtype=np.int64)
        for variable, attributes in report["variables"].items():
            matching = parameter_names.get(attributes["paramId"])
            if matching is not None and variables[matching] != attributes:
                raise ValueError(f"{variable} has different metadata between monthly sources.")
            if variable in variables and variables[variable]["paramId"] != attributes["paramId"]:
                raise ValueError(f"Different parameters share the array name {variable}.")
            variables[variable] = attributes
            parameter_names[attributes["paramId"]] = variable
        selected = {variable: set() for variable in arrays}
        for field in report["fields"]:
            variable = field["variable"]
            identity = variable, field["valid_time"]
            if not is_previous:
                if identity in incoming_fields:
                    raise ValueError(f"Incoming checkpoints duplicate {variable} at {field['valid_time']}.")
                incoming_fields.add(identity)
            if identity not in merged_fields:
                merged_fields[identity] = field
                selected[variable].add(field["valid_time"])
        for variable, selected_times in selected.items():
            if selected_times:
                rows = np.array([
                    index for index, value in enumerate(source_times) if int(value) in selected_times
                ], dtype=np.intp)
                pending_sources.append((variable, arrays[variable], rows, source_times[rows]))
    fields = sorted(merged_fields.values(), key=lambda field: (field["variable"], field["valid_time"]))
    times = np.array(sorted({field["valid_time"] for field in fields}), dtype=np.int64)
    time_indices = {int(value): index for index, value in enumerate(times)}
    shape = (len(times), len(GLOBAL_LATITUDES), len(GLOBAL_LONGITUDES))
    sources: dict[str, list[
        tuple[NDArray[np.float64] | zarr.Array, NDArray[np.intp], NDArray[np.intp]]
    ]] = {variable: [] for variable in variables}
    for variable, array, rows, selected_times in pending_sources:
        destinations = np.array([time_indices[int(value)] for value in selected_times], dtype=np.intp)
        sources[variable].append((array, rows, destinations))
    month_hours = monthrange(year, month)[1] * 24
    chunks = (month_hours, *spatial_chunks)
    shards = (month_hours, spatial_chunks[0] * 2, spatial_chunks[1] * 2)
    compressor = ZstdCodec(level=3, checksum=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    group_attributes = {
        "Conventions": "CF-1.12",
        "source": metadata["dataset"],
        "provenance": metadata,
        "source_provenance": source_provenance,
        "grid": grid,
        "year": year,
        "month": month,
        "spatial_chunks": list(spatial_chunks),
        "variable_identity": "Descriptive ecCodes names; paramId attributes retain parameter identity",
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
    for variable in variables:
        records = [field for field in fields if field["variable"] == variable]
        rows = np.array([
            time_indices[field["valid_time"]] for field in records
        ], dtype=np.intp)
        coverage_name = f"{variable}_coverage"
        coverage = np.zeros(len(times), dtype=np.bool_)
        coverage[rows] = True
        coverage_attributes = {
            "long_name": f"{variables[variable]['name']}: field coverage",
            "description": "True when this variable and UTC hour have a source field.",
        }
        coordinate_values[coverage_name] = coverage
        expected_attributes[coverage_name] = coverage_attributes
        root.create_array(
            coverage_name, data=coverage, chunks=(month_hours,),
            compressors=[compressor], dimension_names=("time",),
            attributes=coverage_attributes,
        )
        auxiliary_names = [coverage_name]
        for key in ("reference_time", "interval_start", "interval_end", "start_step", "end_step"):
            name = f"{variable}_{key}"
            auxiliary_names.append(name)
            auxiliary = np.full(len(times), MISSING_INTEGER, dtype=np.int64)
            auxiliary[rows] = [field[key] for field in records]
            attributes = {
                "long_name": f"{variables[variable]['name']}: {key}",
                "_FillValue": int(MISSING_INTEGER),
            }
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
            variable, shape=shape, dtype=np.float64, fill_value=np.nan,
            chunks=chunks, shards=shards, compressors=[compressor],
            dimension_names=("time", "latitude", "longitude"),
            attributes=attributes,
        )
        for latitude in range(0, shape[1], shards[1]):
            for longitude in range(0, shape[2], shards[2]):
                selection = (
                    slice(None),
                    slice(latitude, min(latitude + shards[1], shape[1])),
                    slice(longitude, min(longitude + shards[2], shape[2])),
                )
                target[selection] = _merged_shard(sources[variable], selection, len(times))

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
    for variable in variables:
        actual = reopened[variable]
        if (
            actual.dtype != np.dtype(np.float64)
            or actual.shape != shape
            or actual.chunks != chunks
            or actual.shards != shards
            or actual.compressors != (compressor,)
            or actual.metadata.dimension_names != ("time", "latitude", "longitude")
        ):
            raise ValueError(f"{variable}: Zarr storage metadata round trip failed.")
        for latitude in range(0, shape[1], shards[1]):
            for longitude in range(0, shape[2], shards[2]):
                selection = (
                    slice(None),
                    slice(latitude, min(latitude + shards[1], shape[1])),
                    slice(longitude, min(longitude + shards[2], shape[2])),
                )
                if not np.array_equal(
                    actual[selection],
                    _merged_shard(sources[variable], selection, len(times)),
                    equal_nan=True,
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
            "request_variable_validation": "distinct_parameter_count_and_complete_hours",
        },
    }
