"""Retrieve a single global ERA5 field to verify CDS dataset access."""

from pathlib import Path

import cdsapi
import requests
import xarray as xr

from era5_pull_pipeline.credentials import CdsCredentials

DATASET = "reanalysis-era5-single-levels"
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
