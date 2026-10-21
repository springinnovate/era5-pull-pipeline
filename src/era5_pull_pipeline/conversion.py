"""Write validated monthly Zarr stores from decoded source arrays."""

from calendar import monthrange
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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


def _merged_shard(
    sources: list[
        tuple[NDArray[Any] | zarr.Array, NDArray[np.intp], NDArray[np.intp]]
    ],
    selection: tuple[slice, slice, slice],
    time_count: int,
) -> NDArray[Any]:
    """Assemble one spatial shard from explicitly covered source hours.

    Args:
        sources: Arrays with source and destination time indices. Their
            destination indices are disjoint after field precedence is applied.
        selection: Full time axis and bounded latitude/longitude slices.
        time_count: Length of the merged monthly time axis.

    Returns:
        Source-precision values with NaN where a field or source cell is missing.
    """
    _, latitude, longitude = selection
    merged = np.full(
        (time_count, latitude.stop - latitude.start, longitude.stop - longitude.start),
        np.nan,
        dtype=np.result_type(*(source.dtype for source, _, _ in sources)),
    )
    for source, source_indices, destination_indices in sources:
        if isinstance(source, zarr.Array):
            values = source.oindex[source_indices, latitude, longitude]
        else:
            values = source[source_indices, latitude, longitude]
        merged[destination_indices] = values
    return merged


def write_monthly_zarr(
    incoming: Sequence[tuple[dict[str, NDArray[Any] | zarr.Array], dict[str, Any]]],
    metadata: dict[str, Any],
    output: Path,
    spatial_chunks: tuple[int, int],
    *,
    previous: Path | None,
    previous_report: dict[str, Any] | None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Write and validate a monthly Zarr store from source arrays.

    Combine incoming arrays with the existing store for the same month, keeping
    existing values unless refresh is requested. Write one spatial shard at a
    time, then reopen the store to compare its values and metadata with the inputs.

    Args:
        incoming: Source arrays and validated reports for one calendar month.
        metadata: Source or monthly publication provenance.
        output: New directory for the unpublished store.
        spatial_chunks: Latitude and longitude cells per read chunk.
        previous: Published store whose unselected fields are preserved.
        previous_report: Its catalog report with field metadata and source origins.
        refresh: Prefer incoming coverage over matching previously stored fields.

    Returns:
        A JSON-compatible report with exact numerical and metadata validation.

    Raises:
        ValueError: Source month, grid, metadata, coverage, or round trip differs.
        Exception: Zarr and filesystem errors propagate from their libraries.
    """
    if not incoming:
        raise ValueError("Monthly conversion requires at least one source array.")
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
        all_sources.insert(len(incoming) if refresh else 0, (
            {variable: prior[variable] for variable in previous_report["variables"]},
            previous_report,
        ))
        source_provenance = list(prior.attrs["source_provenance"])
    source_provenance.extend(metadata["sources"])
    variables: dict[str, Any] = {}
    merged_fields: dict[tuple[str, int], dict[str, Any]] = {}
    incoming_fields: set[tuple[str, int]] = set()
    pending_sources = []
    for source_index, (arrays, report) in enumerate(all_sources):
        is_previous = previous is not None and source_index == (len(incoming) if refresh else 0)
        if (report["year"], report["month"], report["grid"]) != (year, month, grid):
            raise ValueError("Monthly sources must share the same calendar month and grid.")
        source_times = np.array(sorted({field["valid_time"] for field in report["fields"]}), dtype=np.int64)
        for variable, attributes in report["variables"].items():
            if variable in variables and variables[variable] != attributes:
                raise ValueError(f"{variable} has different metadata between monthly sources.")
            variables[variable] = attributes
        selected = {variable: set() for variable in arrays}
        for field in report["fields"]:
            variable = field["variable"]
            identity = variable, field["valid_time"]
            if not is_previous:
                if identity in incoming_fields:
                    raise ValueError(f"Incoming source arrays duplicate {variable} at {field['valid_time']}.")
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
        tuple[NDArray[Any] | zarr.Array, NDArray[np.intp], NDArray[np.intp]]
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
        "variable_identity": "Source descriptive names; original identifiers are retained in metadata",
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
    dtypes = {
        variable: np.result_type(*(source.dtype for source, _, _ in inputs))
        for variable, inputs in sources.items()
    }
    for variable in variables:
        print(f"Writing {variable}: {len(times)} UTC hours.", flush=True)
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
            variable, shape=shape, dtype=dtypes[variable], fill_value=np.nan,
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
        print(f"Validating {variable}: exact numerical round trip.", flush=True)
        actual = reopened[variable]
        if (
            actual.dtype != dtypes[variable]
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
            "numerical_roundtrip": "exact_source_dtype_equal_nan",
            "metadata_roundtrip": True,
            "request_variable_validation": "decoded_source_coverage",
        },
    }
