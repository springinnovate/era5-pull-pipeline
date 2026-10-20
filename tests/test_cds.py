"""Exercise CDS retrieval and real GRIB decoding without network access."""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import eccodes
import pytest
import requests
import xarray as xr

from era5_pull_pipeline import cds
from era5_pull_pipeline.credentials import CdsCredentials

TOKEN = "sentinel-private-token"
CREDENTIALS = CdsCredentials("https://example.test/api", TOKEN)


def _grib_payload(edition: int, temperature: float = 280.0) -> bytes:
    """Encode a temperature field using an ecCodes sample.

    Args:
        edition: GRIB edition, either 1 or 2.
        temperature: Constant field value in kelvin.

    Returns:
        An encoded GRIB message with a real grid and data values.
    """
    handle = eccodes.codes_grib_new_from_samples(f"regular_ll_sfc_grib{edition}")
    try:
        eccodes.codes_set(handle, "paramId", 167)
        eccodes.codes_set(handle, "dataDate", 20000101)
        eccodes.codes_set(handle, "dataTime", 0)
        count = eccodes.codes_get(handle, "numberOfPoints")
        eccodes.codes_set_values(handle, [temperature] * count)
        return eccodes.codes_get_message(handle)
    finally:
        eccodes.codes_release(handle)


@pytest.fixture
def client_factory(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Replace the remote client while retaining the real GRIB reader."""
    factory = Mock()
    monkeypatch.setattr(cds.cdsapi, "Client", factory)
    return factory


@pytest.mark.parametrize("edition", [1, 2])
def test_global_request_and_real_grib_decode(
    tmp_path: Path, client_factory: Mock, edition: int
) -> None:
    """Download and decode a real GRIB through the one-field request contract."""
    payload = _grib_payload(edition)

    def retrieve(dataset: str, request: dict[str, object], target: str) -> None:
        """Supply an ecCodes-generated response at the download boundary."""
        Path(target).write_bytes(payload)

    client_factory.return_value.retrieve.side_effect = retrieve
    output = tmp_path / "downloads" / "probe.grib"
    assert cds.check_access(CREDENTIALS, output) == len(payload)
    assert output.read_bytes() == payload
    assert list(output.parent.iterdir()) == [output]
    options = dict(client_factory.call_args.kwargs)
    assert isinstance(options.pop("session"), requests.Session)
    assert options == {
        "url": CREDENTIALS.url,
        "key": TOKEN,
        **cds.CLIENT_OPTIONS,
    }
    client_factory.return_value.retrieve.assert_called_once_with(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature"],
            "year": ["2000"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
            "data_format": "grib",
            "download_format": "unarchived",
        },
        str(output),
    )
    with xr.open_dataset(
        output, engine="cfgrib", backend_kwargs={"indexpath": ""}
    ) as dataset:
        assert dataset["t2m"].values == pytest.approx(280.0)


def test_repeat_check_retrieves_and_replaces_output(
    tmp_path: Path, client_factory: Mock
) -> None:
    """Repeat the access check instead of skipping an existing download."""
    first = _grib_payload(2, 280.0)
    second = _grib_payload(2, 290.0)
    responses = iter((first, second))

    def retrieve(dataset: str, request: dict[str, object], target: str) -> None:
        """Replace the target with the next complete provider response."""
        Path(target).write_bytes(next(responses))

    client_factory.return_value.retrieve.side_effect = retrieve
    output = tmp_path / "probe.grib"
    assert cds.check_access(CREDENTIALS, output) == len(first)
    assert output.read_bytes() == first
    assert cds.check_access(CREDENTIALS, output) == len(second)
    assert output.read_bytes() == second
    assert client_factory.return_value.retrieve.call_count == 2


def test_service_error_propagates(tmp_path: Path, client_factory: Mock) -> None:
    """Retain the provider exception and its HTTP response for diagnosis."""
    response = requests.Response()
    response.status_code = 503
    error = requests.HTTPError("CDS unavailable", response=response)
    client_factory.return_value.retrieve.side_effect = error

    with pytest.raises(requests.HTTPError) as caught:
        cds.check_access(CREDENTIALS, tmp_path / "probe.grib")

    assert caught.value is error
    assert caught.value.response is response


@pytest.mark.parametrize("payload", [b"", b"not a GRIB file"])
def test_invalid_download_fails_in_grib_reader(
    tmp_path: Path, client_factory: Mock, payload: bytes
) -> None:
    """Let cfgrib reject responses containing no decodable GRIB message."""

    def retrieve(dataset: str, request: dict[str, object], target: str) -> None:
        """Write the invalid provider response under test."""
        Path(target).write_bytes(payload)

    client_factory.return_value.retrieve.side_effect = retrieve
    with pytest.raises((EOFError, eccodes.CodesInternalError)):
        cds.check_access(CREDENTIALS, tmp_path / "probe.grib")


def test_real_client_logs_without_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep ordinary client messages while excluding its debug-only token."""
    from ecmwf.datastores.legacy_client import LegacyClient

    payload = _grib_payload(2)
    seen_headers: list[dict[str, str]] = []

    def catalogue_request(
        self: requests.Session, method: str, url: str, **kwargs: Any
    ) -> requests.Response:
        """Answer the real client's constructor request using local data."""
        assert method.lower() == "get"
        assert url == "https://example.test/api/catalogue/v1/messages"
        seen_headers.append(dict(kwargs["headers"]))
        response = requests.Response()
        response.status_code = 200
        response.url = url
        response.request = requests.Request(method, url).prepare()
        response._content = (
            b'{"messages": [{"severity": "warning", '
            b'"content": "Offline catalogue notice"}]}'
        )
        return response

    def retrieve(
        self: LegacyClient, name: str, request: dict[str, object], target: str
    ) -> None:
        """Provide a real GRIB after the installed client initializes."""
        assert self.key == TOKEN
        assert name == cds.DATASET
        Path(target).write_bytes(payload)

    monkeypatch.setattr(requests.Session, "request", catalogue_request)
    monkeypatch.setattr(LegacyClient, "retrieve", retrieve)
    caplog.set_level(logging.DEBUG)
    assert cds.check_access(CREDENTIALS, tmp_path / "probe.grib") == len(payload)
    assert len(seen_headers) == 1
    assert seen_headers[0]["PRIVATE-TOKEN"] == TOKEN
    assert "Offline catalogue notice" in caplog.text
    captured = capsys.readouterr()
    assert TOKEN not in caplog.text + captured.out + captured.err
