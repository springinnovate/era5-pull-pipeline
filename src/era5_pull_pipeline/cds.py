"""Retrieve a single global ERA5 field to verify CDS dataset access."""

from collections.abc import Iterator
from contextlib import contextmanager
import logging
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import warnings

import cdsapi
import requests

from era5_pull_pipeline.credentials import CdsCredentials

DATASET = "reanalysis-era5-single-levels"
DATASET_URL = f"https://cds.climate.copernicus.eu/datasets/{DATASET}"


class AccessCheckError(Exception):
    """Describe an access-check failure without exposing credentials."""


def access_check_request() -> dict[str, str | list[str]]:
    """Build the fixed one-field, full-global ERA5 access request.

    Returns:
        A request for 2 m temperature at 2000-01-01 00:00 UTC. Omitting
        an area selection requests the dataset's entire global grid.
    """
    return {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": ["2000"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
        "data_format": "grib",
        "download_format": "unarchived",
    }


def _discard_diagnostic(*args: object, **kwargs: object) -> None:
    """Discard upstream diagnostics, which may include the access token."""


class _DiscardLogs(logging.Filter):
    """Suppress downloader diagnostics that bypass CDS client callbacks."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Discard the supplied diagnostic record."""
        return False


@contextmanager
def _private_diagnostics() -> Iterator[None]:
    """Suppress raw dependency diagnostics for the duration of a CDS call.

    Yields:
        Control with downloader logging and client warnings suppressed.
    """
    loggers = [
        logging.getLogger(name)
        for name in ("multiurl.base", "multiurl.http", "multiurl.retry")
    ]
    diagnostic_filter = _DiscardLogs()
    for logger in loggers:
        logger.addFilter(diagnostic_filter)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module=r"ecmwf\.datastores\..*")
            yield
    finally:
        for logger in loggers:
            logger.removeFilter(diagnostic_filter)


def _service_error(exc: Exception) -> AccessCheckError:
    """Translate a provider failure into a fixed, credential-free message.

    Args:
        exc: The exception from client construction or retrieval.

    Returns:
        An actionable error without raw URLs, response text, or tracebacks.
    """
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        if status == 401:
            return AccessCheckError(
                "CDS rejected the credentials. Check the personal access token in .cdsapirc."
            )
        if status == 403:
            # Inspect only to classify; never display provider response content.
            detail = exc.response.text.casefold()
            if any(word in detail for word in ("licence", "license", "terms")) and any(
                word in detail for word in ("accept", "required")
            ):
                return AccessCheckError(
                    f"Accept the ERA5 dataset terms at {DATASET_URL}?tab=download, then retry."
                )
            return AccessCheckError(
                "CDS denied access. Check your token and account's access to the ERA5 dataset."
            )
        if status == 429 or status >= 500:
            return AccessCheckError("CDS is busy or unavailable. Retry the access check later.")
        if status == 404:
            return AccessCheckError(
                "CDS could not find the endpoint or dataset. Check the URL in .cdsapirc."
            )
        return AccessCheckError("CDS rejected the access-check request. Check CDS service status.")
    if isinstance(exc, requests.RequestException):
        return AccessCheckError(
            "Could not reach CDS or complete the transfer. Check your connection and retry."
        )
    return AccessCheckError(
        "CDS could not complete the access check. Check CDS service status and retry."
    )


def _retrieve_probe(credentials: CdsCredentials, target: Path) -> None:
    """Retrieve the probe using explicit credentials and private diagnostics.

    Args:
        credentials: Validated local CDS configuration.
        target: Temporary download path.

    Raises:
        AccessCheckError: CDS construction or retrieval fails.
    """
    try:
        with _private_diagnostics(), requests.Session() as session:
            client = cdsapi.Client(
                url=credentials.url,
                key=credentials.key,
                verify=True,
                quiet=True,
                debug=False,
                progress=False,
                timeout=30,
                retry_max=2,
                sleep_max=2,
                session=session,
                info_callback=_discard_diagnostic,
                warning_callback=_discard_diagnostic,
                error_callback=_discard_diagnostic,
                debug_callback=_discard_diagnostic,
            )
            client.retrieve(DATASET, access_check_request(), str(target))
    except Exception as exc:
        raise _service_error(exc) from None


def _validate_probe_grib(path: Path) -> int:
    """Check the single GRIB message's framing without decoding its array.

    Args:
        path: Completed temporary download.

    Returns:
        The size of the framed GRIB message in bytes.

    Raises:
        AccessCheckError: The response is empty, truncated, or not one GRIB message.
    """
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(16)
        if size < 20 or header[:4] != b"GRIB" or header[7] not in (1, 2):
            raise AccessCheckError("CDS returned an empty or invalid GRIB. Retry the access check.")
        length = int.from_bytes(header[4:7] if header[7] == 1 else header[8:16], "big")
        stream.seek(-4, 2)
        if length != size or stream.read(4) != b"7777":
            raise AccessCheckError("CDS returned an incomplete GRIB. Retry the access check.")
    return size


def check_access(credentials: CdsCredentials, output: Path) -> int:
    """Verify ERA5 access and retain one global field at a new output path.

    Args:
        credentials: Validated local CDS configuration.
        output: Destination for the access-check GRIB; it must not already exist.

    Returns:
        The downloaded GRIB size in bytes.

    Raises:
        AccessCheckError: Retrieval, validation, or writing the output fails.
    """
    try:
        if output.exists() or output.is_symlink():
            raise AccessCheckError("Output already exists. Choose a new --output path.")
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".era5-access-", dir=output.parent) as temporary:
            target = Path(temporary) / "probe.grib"
            _retrieve_probe(credentials, target)
            size = _validate_probe_grib(target)
            # Exclusive creation also protects a destination created during the request.
            destination = output.open("xb")
            try:
                with destination:
                    with target.open("rb") as source:
                        shutil.copyfileobj(source, destination)
            except BaseException:
                output.unlink(missing_ok=True)
                raise
            return size
    except FileExistsError:
        raise AccessCheckError("Output already exists. Choose a new --output path.") from None
    except OSError:
        raise AccessCheckError(
            "Could not read or write the access-check files. Check output permissions and disk space."
        ) from None
