"""Retrieve global ERA5 fields and stage GRIB downloads."""

from collections.abc import Sequence
from datetime import UTC, date, datetime
import hashlib
import json
from pathlib import Path
from tempfile import mkdtemp

import cdsapi
import requests
import xarray as xr

from era5_pull_pipeline.credentials import CdsCredentials

DATASET = "reanalysis-era5-single-levels"
DEFAULT_MAX_DAYS = 7
# Published CDS ERA5 hourly single-level limit; CDS also applies dynamic limits.
MAX_REQUEST_FIELDS = 120_000
CLIENT_OPTIONS = {
    "verify": True,
    "quiet": True,
    # CDS debug messages include credentials.
    "debug": False,
    "progress": False,
    "timeout": 30,
    "retry_max": 2,
    "sleep_max": 2,
}
ACCESS_CHECK_REQUEST = {
    "product_type": ["reanalysis"],
    "variable": ["2m_temperature"],
    "year": ["2000"],
    "month": ["01"],
    "day": ["01"],
    "time": ["00:00"],
    "data_format": "grib",
    "download_format": "unarchived",
}


def check_access(credentials: CdsCredentials, output: Path) -> int:
    """Download and decode one global field, replacing any existing probe.

    Args:
        credentials: Connection settings from the local .cdsapirc.
        output: Destination for the access-check GRIB.

    Returns:
        The downloaded GRIB size in bytes.

    Raises:
        Exception: File, CDS, or GRIB decoding errors propagate from the library
            that raises them.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        client = cdsapi.Client(
            url=credentials.url,
            key=credentials.key,
            session=session,
            **CLIENT_OPTIONS,
        )
        client.retrieve(DATASET, ACCESS_CHECK_REQUEST, str(output))
    with xr.open_dataset(
        output, engine="cfgrib", backend_kwargs={"indexpath": "", "errors": "raise"}
    ) as dataset:
        dataset.load()
    return output.stat().st_size


def build_batch_request(
    start: date,
    end: date,
    variables: Sequence[str],
    max_days: int = DEFAULT_MAX_DAYS,
) -> dict[str, str | list[str] | list[float]]:
    """Select all hourly global fields for one bounded date interval.

    Args:
        start: First UTC date, included in the request.
        end: Last UTC date, included in the request.
        variables: CDS single-level variable names; CDS validates the names.
        max_days: Maximum number of days allowed in this batch.

    Returns:
        A CDS request for the full global grid, with sorted unique variables.

    Raises:
        ValueError: Dates cross a month, are reversed, exceed the positive day
            limit, no variables are supplied, or the request exceeds the CDS
            field limit.
    """
    if max_days < 1:
        raise ValueError("max_days must be at least 1.")
    if end < start:
        raise ValueError("The end date must be on or after the start date.")
    if (start.year, start.month) != (end.year, end.month):
        raise ValueError("A download batch must stay within one calendar month.")
    if (end - start).days + 1 > max_days:
        raise ValueError(f"The selected interval exceeds max_days={max_days}.")
    if not variables or any(not variable for variable in variables):
        raise ValueError("Select at least one nonempty CDS variable name.")
    selected_variables = sorted(set(variables))
    field_count = len(selected_variables) * ((end - start).days + 1) * 24
    if field_count > MAX_REQUEST_FIELDS:
        raise ValueError(
            f"The request has {field_count:,} fields; the CDS limit is "
            f"{MAX_REQUEST_FIELDS:,}. Select fewer days or variables."
        )

    return {
        "product_type": ["reanalysis"],
        "variable": selected_variables,
        "year": [f"{start.year:04d}"],
        "month": [f"{start.month:02d}"],
        "day": [f"{day:02d}" for day in range(start.day, end.day + 1)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "grid": [0.25, 0.25],
        "data_format": "grib",
        "download_format": "unarchived",
    }


def download_batch(
    credentials: CdsCredentials,
    request: dict[str, str | list[str] | list[float]],
    staging: Path,
) -> Path:
    """Stream a selected batch to disk and complete its staging directory.

    Args:
        credentials: Connection settings from the local .cdsapirc.
        request: Bounded single-month selection from build_batch_request, or a
            subset of its variables, days, and hours selected by the planner.
        staging: Parent directory for completed and partial batch directories.

    Returns:
        A new directory containing data.grib and metadata.json.

    Raises:
        Exception: File or CDS errors propagate. Unfinished batches retain the
            .partial directory suffix.
    """
    identity = json.dumps(
        {"dataset": DATASET, "request": request}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    period = f"{request['year'][0]}-{request['month'][0]}"
    start = f"{period}-{min(request['day'])}"
    end = f"{period}-{max(request['day'])}"
    staging.mkdir(parents=True, exist_ok=True)
    partial = Path(
        mkdtemp(prefix=f"{start}_{end}_", suffix=".partial", dir=staging)
    )
    target = partial / "data.grib"
    started_at = datetime.now(UTC).isoformat()
    with requests.Session() as session:
        client = cdsapi.Client(
            url=credentials.url,
            key=credentials.key,
            session=session,
            wait_until_complete=False,
            **CLIENT_OPTIONS,
        )
        remote = client.retrieve(DATASET, request)
        remote.download(str(target))
    retrieved_at = datetime.now(UTC).isoformat()
    with target.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "dataset": DATASET,
        "request": request,
        "request_sha256": hashlib.sha256(identity).hexdigest(),
        "cds_request_id": remote.request_id,
        "started_at": started_at,
        "retrieved_at": retrieved_at,
        "file": target.name,
        "size_bytes": target.stat().st_size,
        "sha256": checksum,
    }
    (partial / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    completed = partial.with_suffix("")
    partial.rename(completed)
    return completed
