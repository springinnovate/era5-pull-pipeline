"""Retrieve global ERA5 fields and stage GRIB downloads."""

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import time

import cdsapi
from multiurl import robust
import requests
import xarray as xr

from era5_pull_pipeline.credentials import CdsCredentials

DATASET = "reanalysis-era5-single-levels"
DEFAULT_MAX_DAYS = 7
# Published CDS ERA5 hourly single-level limit; CDS also applies dynamic limits.
MAX_REQUEST_FIELDS = 120_000
CATALOGUE_URL = f"https://cds.climate.copernicus.eu/api/catalogue/v1/collections/{DATASET}"
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = (2, 30, 2)
CLIENT_OPTIONS = {
    "verify": True,
    "quiet": False,
    # CDS debug messages include credentials.
    "debug": False,
    "progress": True,
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
    batch: Path,
) -> Path:
    """Download or resume one batch at its recorded destination.

    Args:
        credentials: Connection settings from the local .cdsapirc.
        request: Bounded single-month selection from build_batch_request, or a
            subset of its variables, days, and hours selected by the planner.
        batch: Exact completed directory allocated by the caller for this
            request. Work resumes in its .partial sibling; an interrupted
            transfer restarts its bytes using the same recorded remote job.
            Waiting messages retain the original request's elapsed time.

    Returns:
        The completed batch directory, reused directly if already present.

    Raises:
        Exception: File or CDS errors propagate. Unfinished batches retain the
            .partial directory suffix.
    """
    if batch.exists():
        return batch
    identity = json.dumps(
        {"dataset": DATASET, "request": request}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    request_sha256 = hashlib.sha256(identity).hexdigest()
    partial = batch.with_suffix(".partial")
    partial.mkdir(parents=True, exist_ok=True)
    if (partial / "request.json").exists():
        journal = json.loads((partial / "request.json").read_text(encoding="utf-8"))
    else:
        journal = {
            "dataset": DATASET,
            "request": request,
            "request_sha256": request_sha256,
            "started_at": datetime.now(UTC).isoformat(),
        }
        (partial / "request.json.tmp").write_text(json.dumps(journal) + "\n", encoding="utf-8")
        (partial / "request.json.tmp").replace(partial / "request.json")
    target = partial / "data.grib"
    with requests.Session() as session:
        client = cdsapi.Client(
            url=credentials.url,
            key=credentials.key,
            session=session,
            wait_until_complete=False,
            **CLIENT_OPTIONS,
        )
        # The native client passes this bounded exponential schedule through to
        # multiurl for submission, polling, and transfer retries.
        client.client.maximum_tries = RETRY_ATTEMPTS
        client.client.retry_after = RETRY_BACKOFF
        if "cds_request_id" in journal:
            remote = client.client.get_remote(journal["cds_request_id"])
        else:
            remote = client.retrieve(DATASET, request)
            journal["cds_request_id"] = remote.request_id
            (partial / "request.json.tmp").write_text(json.dumps(journal) + "\n", encoding="utf-8")
            (partial / "request.json.tmp").replace(partial / "request.json")
        last_notice = time.monotonic()
        poll_delay = 1.0
        while not remote.results_ready:
            now = time.monotonic()
            if now - last_notice >= 30:
                elapsed = datetime.now(UTC) - datetime.fromisoformat(journal["started_at"])
                elapsed = timedelta(seconds=int(elapsed.total_seconds()))
                status = "queued (accepted)" if remote.last_status == "accepted" else remote.last_status
                print(
                    f"CDS request {remote.request_id}: still {status}; "
                    f"{elapsed} since request started. Still checking CDS.", flush=True,
                )
                last_notice = now
            time.sleep(poll_delay)
            poll_delay = min(poll_delay * 1.5, CLIENT_OPTIONS["sleep_max"])
        remote.download(str(target))
    retrieved_at = datetime.now(UTC).isoformat()
    with target.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "dataset": DATASET,
        "request": request,
        "request_sha256": request_sha256,
        "cds_request_id": remote.request_id,
        "started_at": journal["started_at"],
        "retrieved_at": retrieved_at,
        "file": target.name,
        "size_bytes": target.stat().st_size,
        "sha256": checksum,
    }
    (partial / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    partial.rename(batch)
    return batch


def load_availability() -> list[dict[str, list[str]]]:
    """Read the provider's current ERA5 availability constraint cubes.

    Returns:
        Cartesian selections whose union describes the provider's available
        products, variables, dates, and UTC hours.

    Raises:
        Exception: HTTP or JSON errors propagate after bounded transient retries.
    """
    with requests.Session() as session:
        get = robust(session.get, maximum_tries=RETRY_ATTEMPTS, retry_after=RETRY_BACKOFF)
        response = get(CATALOGUE_URL, timeout=30)
        response.raise_for_status()
        constraints_url = next(
            link["href"] for link in response.json()["links"]
            if link["rel"] == "constraints"
        )
        response = get(constraints_url, timeout=30)
        response.raise_for_status()
        return response.json()
