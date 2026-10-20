"""Exercise the CDS boundary with local responses and no live credentials."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
import warnings

import pytest
import requests

from era5_pull_pipeline import cds
from era5_pull_pipeline.credentials import CdsCredentials

TOKEN = "sentinel-private-token"
CREDENTIALS = CdsCredentials("https://example.test/api", TOKEN)
# Only GRIB framing is checked here; decoding/array validation belongs to ingestion.
GRIB = b"GRIB\x00\x00\x00\x02" + (24).to_bytes(8, "big") + b"data7777"


@pytest.fixture
def client_factory(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Replace the CDS boundary with a completed single-field response."""

    def retrieve(dataset: str, request: dict[str, object], target: str) -> None:
        """Write a small framed response without contacting the service."""
        Path(target).write_bytes(GRIB)

    factory = Mock(return_value=SimpleNamespace(retrieve=Mock(side_effect=retrieve)))
    monkeypatch.setattr(cds.cdsapi, "Client", factory)
    return factory


def test_global_request_and_explicit_credentials(tmp_path: Path, client_factory: Mock) -> None:
    """Request exactly one global field and use only explicitly supplied credentials."""
    output = tmp_path / "downloads" / "probe.grib"
    assert cds.check_access(CREDENTIALS, output) == len(GRIB)
    assert output.read_bytes() == GRIB
    assert list(output.parent.iterdir()) == [output]
    options = client_factory.call_args.kwargs
    assert options["url"] == CREDENTIALS.url
    assert options["key"] == TOKEN
    assert options["verify"] is True
    assert options["debug"] is options["progress"] is False
    assert options["retry_max"] == 2
    assert options["timeout"] == 30
    dataset, request, _ = client_factory.return_value.retrieve.call_args.args
    assert dataset == "reanalysis-era5-single-levels"
    assert request == {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": ["2000"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
        "data_format": "grib",
        "download_format": "unarchived",
    }


@pytest.mark.parametrize(
    ("status", "detail", "message"),
    [
        (401, "invalid token", "rejected the credentials"),
        (403, "required licences not accepted", "Accept the ERA5 dataset terms"),
        (403, "account forbidden", "denied access"),
        (404, "not found", "endpoint or dataset"),
        (429, "slow down", "busy or unavailable"),
        (503, "unavailable", "busy or unavailable"),
        (400, "bad request", "rejected the access-check request"),
    ],
)
def test_http_errors_are_actionable_and_private(
    tmp_path: Path, client_factory: Mock, status: int, detail: str, message: str
) -> None:
    """Classify real requests exceptions without echoing provider content."""
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps({"detail": detail, "private": TOKEN}).encode()
    client_factory.return_value.retrieve.side_effect = requests.HTTPError(
        TOKEN, response=response
    )
    output = tmp_path / "probe.grib"
    with pytest.raises(cds.AccessCheckError, match=message) as caught:
        cds.check_access(CREDENTIALS, output)
    assert TOKEN not in str(caught.value)
    assert caught.value.__suppress_context__
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "error",
    [requests.Timeout(TOKEN), requests.ConnectionError(TOKEN), RuntimeError(TOKEN)],
)
def test_transfer_failure_retains_no_success_output(
    tmp_path: Path, client_factory: Mock, error: Exception
) -> None:
    """Clean temporary probe data when the provider or network fails."""

    def fail(dataset: str, request: dict[str, object], target: str) -> None:
        """Simulate a failure after part of the transfer was written."""
        Path(target).write_bytes(b"incomplete")
        raise error

    client_factory.return_value.retrieve.side_effect = fail
    with pytest.raises(cds.AccessCheckError) as caught:
        cds.check_access(CREDENTIALS, tmp_path / "probe.grib")
    assert TOKEN not in str(caught.value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("data", [b"", b"<html>error</html>", GRIB[:-1], GRIB + b"extra"])
def test_reject_empty_non_grib_and_truncated_output(
    tmp_path: Path, client_factory: Mock, data: bytes
) -> None:
    """Do not report success for incomplete or incorrectly framed responses."""

    def retrieve(dataset: str, request: dict[str, object], target: str) -> None:
        """Write the invalid provider response under test."""
        Path(target).write_bytes(data)

    client_factory.return_value.retrieve.side_effect = retrieve
    with pytest.raises(cds.AccessCheckError, match="GRIB"):
        cds.check_access(CREDENTIALS, tmp_path / "probe.grib")
    assert list(tmp_path.iterdir()) == []


def test_grib1_probe_is_supported(tmp_path: Path, client_factory: Mock) -> None:
    """Accept the GRIB edition commonly used by the source dataset."""
    data = b"GRIB" + (24).to_bytes(3, "big") + b"\x01" + b"x" * 12 + b"7777"

    def retrieve(dataset: str, request: dict[str, object], target: str) -> None:
        """Write a single GRIB1 message envelope."""
        Path(target).write_bytes(data)

    client_factory.return_value.retrieve.side_effect = retrieve
    assert cds.check_access(CREDENTIALS, tmp_path / "probe.grib") == len(data)


def test_existing_output_is_preserved_without_a_request(tmp_path: Path, client_factory: Mock) -> None:
    """Avoid overwriting an earlier successful check or an unrelated file."""
    output = tmp_path / "probe.grib"
    output.write_bytes(b"keep this")
    with pytest.raises(cds.AccessCheckError, match="already exists"):
        cds.check_access(CREDENTIALS, output)
    assert output.read_bytes() == b"keep this"
    client_factory.assert_not_called()


def test_output_created_during_request_is_preserved(tmp_path: Path, client_factory: Mock) -> None:
    """Protect a destination created by another process while CDS is running."""
    output = tmp_path / "probe.grib"

    def retrieve(dataset: str, request: dict[str, object], target: str) -> None:
        """Finish the download while another writer claims the destination."""
        Path(target).write_bytes(GRIB)
        output.write_bytes(b"other writer")

    client_factory.return_value.retrieve.side_effect = retrieve
    with pytest.raises(cds.AccessCheckError, match="already exists"):
        cds.check_access(CREDENTIALS, output)
    assert output.read_bytes() == b"other writer"


def test_disk_failure_removes_own_partial_output(
    tmp_path: Path, client_factory: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Avoid leaving a failed output copy that appears to be a completed check."""
    monkeypatch.setattr(cds.shutil, "copyfileobj", Mock(side_effect=OSError(TOKEN)))
    with pytest.raises(cds.AccessCheckError, match="disk space"):
        cds.check_access(CREDENTIALS, tmp_path / "probe.grib")
    assert list(tmp_path.iterdir()) == []


def test_close_failure_removes_own_partial_output(
    tmp_path: Path, client_factory: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clean the final file even when buffered data fail to flush on close."""
    output = tmp_path / "probe.grib"
    original_open = Path.open

    def failing_open(self: Path, *args: object, **kwargs: object) -> object:
        """Wrap only the final destination with a failing context manager."""
        stream = original_open(self, *args, **kwargs)
        if self != output or args != ("xb",):
            return stream

        def fail_close(*exception: object) -> None:
            """Close the real handle, then emulate a failed final flush."""
            stream.close()
            raise OSError("disk full")

        wrapped = MagicMock()
        wrapped.__enter__.return_value = stream
        wrapped.__exit__.side_effect = fail_close
        return wrapped

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(cds.AccessCheckError, match="disk space"):
        cds.check_access(CREDENTIALS, output)
    assert list(tmp_path.iterdir()) == []


def test_interrupt_cleans_temporary_probe(tmp_path: Path, client_factory: Mock) -> None:
    """Allow Ctrl+C to interrupt queue waiting without a successful output."""
    client_factory.return_value.retrieve.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        cds.check_access(CREDENTIALS, tmp_path / "probe.grib")
    assert list(tmp_path.iterdir()) == []


def test_constructor_diagnostics_do_not_leak_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Cover callbacks, constructor warnings, and independent downloader logging."""

    def noisy_constructor(**kwargs: object) -> None:
        """Simulate secret-bearing diagnostics from the upstream client."""
        for name in ("info_callback", "warning_callback", "error_callback", "debug_callback"):
            callback = kwargs[name]
            assert callable(callback)
            callback(TOKEN)
        warnings.warn_explicit(TOKEN, UserWarning, "client.py", 1, module="ecmwf.datastores.client")
        logging.getLogger("multiurl.base").error(TOKEN)
        logging.getLogger("multiurl.http").error(TOKEN)
        logging.getLogger("multiurl.retry").warning(TOKEN)
        raise RuntimeError(TOKEN)

    monkeypatch.setattr(cds.cdsapi, "Client", noisy_constructor)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(cds.AccessCheckError) as caught:
            cds.check_access(CREDENTIALS, tmp_path / "probe.grib")
    assert TOKEN not in str(caught.value)
    assert not caught_warnings
    assert TOKEN not in caplog.text
    logging.getLogger("multiurl.http").warning("logging restored")
    assert "logging restored" in caplog.text


def test_real_cdsapi_constructor_with_offline_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Exercise the installed PAT client constructor without a network request."""
    from ecmwf.datastores.legacy_client import LegacyClient

    seen_headers: list[dict[str, str]] = []

    def catalogue_request(
        self: requests.Session, method: str, url: str, **kwargs: object
    ) -> requests.Response:
        """Answer only the constructor's catalogue-message request locally."""
        assert method.lower() == "get"
        assert url == "https://example.test/api/catalogue/v1/messages"
        seen_headers.append(dict(kwargs["headers"]))
        response = requests.Response()
        response.status_code = 200
        response.url = url
        response._content = b'{"messages": []}'
        return response

    def retrieve(self: LegacyClient, name: str, request: dict[str, object], target: str) -> None:
        """Complete the probe after real client initialization."""
        assert self.key == TOKEN
        assert name == cds.DATASET
        assert "area" not in request
        Path(target).write_bytes(GRIB)

    monkeypatch.setattr(requests.Session, "request", catalogue_request)
    monkeypatch.setattr(LegacyClient, "retrieve", retrieve)
    caplog.set_level(logging.DEBUG)
    assert cds.check_access(CREDENTIALS, tmp_path / "probe.grib") == len(GRIB)
    assert len(seen_headers) == 1
    assert seen_headers[0]["User-Agent"].startswith("ecmwf-datastores-client/")
    assert seen_headers[0]["PRIVATE-TOKEN"] == TOKEN
    assert TOKEN not in caplog.text
