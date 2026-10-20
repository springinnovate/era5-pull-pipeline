"""Exercise the installed CDS client's retries through offline HTTP responses."""

import json
from collections.abc import Iterator
from datetime import date
from http import HTTPStatus
from pathlib import Path
from typing import Any

import multiurl.retry
import pytest
import requests

from era5_pull_pipeline import cds
from era5_pull_pipeline.credentials import CdsCredentials

API_URL = "https://example.test/api"
CREDENTIALS = CdsCredentials(API_URL, "offline-test-token")
PROCESS_URL = f"{API_URL}/retrieve/v1/processes/{cds.DATASET}"
JOB_URL = f"{API_URL}/retrieve/v1/jobs/offline-job"
FILE_URL = "https://download.example.test/batch.grib"


def _response(
    method: str, url: str, body: dict[str, Any], status: int = 200
) -> requests.Response:
    """Build a response for the installed client's HTTP transport.

    Args:
        method: HTTP method recorded on the prepared request.
        url: Requested URL.
        body: JSON response body.
        status: HTTP response status.

    Returns:
        A response with the request metadata used by the CDS client.
    """
    response = requests.Response()
    response.status_code = status
    response.reason = HTTPStatus(status).phrase
    response.url = url
    response.request = requests.Request(method, url).prepare()
    response._content = json.dumps(body).encode()
    return response


@pytest.mark.parametrize(
    ("status", "attempts", "message"),
    [(503, 2, "503 Server Error"), (403, 1, "cost limits exceeded")],
)
def test_execution_failure_retry_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    attempts: int,
    message: str,
) -> None:
    """Exhaust transient retries but preserve an immediate cost-limit rejection."""
    failures: list[requests.Response] = []
    sleeps: list[float] = []

    def request(
        self: requests.Session, method: str, url: str, **kwargs: Any
    ) -> requests.Response:
        """Provide catalogue data and reject the real client's execution call."""
        if method.lower() == "post":
            assert url == f"{PROCESS_URL}/execution"
            response = _response(
                method,
                url,
                {
                    "title": "cost limits exceeded",
                    "detail": "Your request is too large, please reduce your selection.",
                },
                status,
            )
            failures.append(response)
            return response
        assert method.lower() == "get"
        assert url in (f"{API_URL}/catalogue/v1/messages", PROCESS_URL)
        return _response(method, url, {})

    monkeypatch.setattr(requests.Session, "request", request)
    monkeypatch.setattr(multiurl.retry.time, "sleep", sleeps.append)
    with pytest.raises(requests.HTTPError, match=message) as caught:
        cds.download_batch(
            CREDENTIALS,
            date(2000, 1, 1),
            date(2000, 1, 1),
            ["2m_temperature"],
            tmp_path,
        )

    assert len(failures) == attempts
    assert caught.value.response is failures[-1]
    assert sleeps == ([2] if attempts == 2 else [])
    (partial,) = tmp_path.iterdir()
    assert partial.suffix == ".partial"
    assert list(partial.iterdir()) == []
    if status == 403:
        assert "please reduce your selection" in str(caught.value)


def test_interrupted_transfer_exhausts_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry a real streaming download once and preserve its final interruption."""
    failure = requests.exceptions.ChunkedEncodingError("Connection interrupted")
    ranges: list[str | None] = []
    sleeps: list[float] = []

    def interrupted_stream(chunk_size: int) -> Iterator[bytes]:
        """Deliver a partial body before the simulated transport disconnects."""
        yield b"part"
        raise failure

    def request(
        self: requests.Session, method: str, url: str, **kwargs: Any
    ) -> requests.Response:
        """Serve a successful job followed by interrupted HTTP file transfers."""
        if url == FILE_URL:
            response = _response(method, url, {})
            response.headers.update({"Content-Length": "12", "Accept-Ranges": "bytes"})
            if method.lower() == "get":
                assert kwargs["stream"] is True
                ranges.append(kwargs["headers"].get("range"))
                monkeypatch.setattr(response, "iter_content", interrupted_stream)
            else:
                assert method.lower() == "head"
            return response

        bodies = {
            f"{API_URL}/catalogue/v1/messages": {},
            PROCESS_URL: {},
            f"{PROCESS_URL}/execution": {
                "links": [{"rel": "monitor", "href": JOB_URL}]
            },
            JOB_URL: {"status": "successful"},
            f"{JOB_URL}/results": {
                "asset": {
                    "value": {
                        "href": FILE_URL,
                        "file:size": 12,
                        "type": "application/x-grib",
                    }
                }
            },
        }
        return _response(method, url, bodies[url])

    monkeypatch.setattr(requests.Session, "request", request)
    monkeypatch.setattr(multiurl.retry.time, "sleep", sleeps.append)
    with pytest.raises(requests.exceptions.ChunkedEncodingError) as caught:
        cds.download_batch(
            CREDENTIALS,
            date(2000, 1, 1),
            date(2000, 1, 1),
            ["2m_temperature"],
            tmp_path,
        )

    assert caught.value is failure
    assert ranges == [None, "bytes=4-"]
    assert sleeps == [2]
    (partial,) = tmp_path.iterdir()
    assert partial.suffix == ".partial"
    assert (partial / "data.grib").read_bytes() == b"partpart"
    assert not (partial / "metadata.json").exists()
