"""Exercise the installed CDS client's retries through offline HTTP responses."""

import json
import logging
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from http import HTTPStatus
from pathlib import Path
from typing import Any
from unittest.mock import Mock

from ecmwf.datastores.processing import ProcessingFailedError
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
    [(503, 3, "503 Server Error"), (403, 1, "cost limits exceeded")],
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
            cds.build_batch_request(date(2000, 1, 1), date(2000, 1, 1), ["2m_temperature"]),
            tmp_path / "batch",
        )

    assert len(failures) == attempts
    assert caught.value.response is failures[-1]
    assert sleeps == ([2, 4] if attempts == 3 else [])
    (partial,) = tmp_path.iterdir()
    assert partial.suffix == ".partial"
    assert [path.name for path in partial.iterdir()] == ["request.json"]
    assert "cds_request_id" not in json.loads((partial / "request.json").read_text())
    if status == 403:
        assert "please reduce your selection" in str(caught.value)


def test_interrupted_transfer_exhausts_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    """Retry a real streaming download with backoff and preserve its interruption."""
    failure = requests.exceptions.ChunkedEncodingError("Connection interrupted")
    ranges: list[str | None] = []
    sleeps: list[float] = []
    submissions: list[str] = []
    fail_transfer = True

    def interrupted_stream(chunk_size: int) -> Iterator[bytes]:
        """Deliver a partial body before the simulated transport disconnects."""
        if fail_transfer:
            yield b"part"
            raise failure
        yield b"fresh-source"

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
            JOB_URL: {"status": "successful", "links": [{"rel": "self", "href": JOB_URL}]},
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
        if method.lower() == "post":
            submissions.append(url)
        return _response(method, url, bodies[url])

    monkeypatch.setattr(requests.Session, "request", request)
    monkeypatch.setattr(multiurl.retry.time, "sleep", sleeps.append)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(requests.exceptions.ChunkedEncodingError) as caught:
        cds.download_batch(
            CREDENTIALS,
            cds.build_batch_request(date(2000, 1, 1), date(2000, 1, 1), ["2m_temperature"]),
            tmp_path / "batch",
        )

    assert caught.value is failure
    assert ranges == [None, "bytes=4-", "bytes=8-"]
    assert sleeps == [2, 4]
    (partial,) = tmp_path.iterdir()
    assert partial.suffix == ".partial"
    assert (partial / "data.grib").read_bytes() == b"partpartpart"
    assert not (partial / "metadata.json").exists()

    fail_transfer = False
    completed = cds.download_batch(
        CREDENTIALS,
        cds.build_batch_request(date(2000, 1, 1), date(2000, 1, 1), ["2m_temperature"]),
        tmp_path / "batch",
    )
    assert submissions == [f"{PROCESS_URL}/execution"]
    assert ranges == [None, "bytes=4-", "bytes=8-", None]
    assert completed == partial.with_suffix("")
    assert (completed / "data.grib").read_bytes() == b"fresh-source"
    captured = capsys.readouterr()
    assert "Request ID is offline-job" in captured.err
    assert "status has been updated to successful" in captured.err
    assert "B/s" in captured.err
    assert CREDENTIALS.key not in caplog.text + captured.out + captured.err


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("outcome", ["successful", "failed"])
def test_waiting_messages_follow_status_checks_and_keep_elapsed_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], resume: bool, outcome: str,
) -> None:
    """Report long waits and resumed request age through the installed CDS client."""
    origin = datetime(2026, 10, 20, 23, 30, tzinfo=UTC)
    elapsed = 0.0
    checks = []
    submissions = []
    transfers = []
    selection = cds.build_batch_request(date(2000, 1, 1), date(2000, 1, 1), ["2m_temperature"])
    batch = tmp_path / "batch"
    if resume:
        partial = batch.with_suffix(".partial")
        partial.mkdir()
        (partial / "request.json").write_text(json.dumps({
            "dataset": cds.DATASET, "request": selection,
            "started_at": (origin - timedelta(minutes=15)).isoformat(),
            "cds_request_id": "offline-job",
        }), encoding="utf-8")

    def sleep(seconds: float) -> None:
        """Advance simulated time instead of waiting for CDS polling intervals."""
        nonlocal elapsed
        elapsed += seconds

    def request(
        self: requests.Session, method: str, url: str, **kwargs: Any,
    ) -> requests.Response:
        """Serve a queued job, a running job, and its terminal result."""
        if url == FILE_URL:
            transfers.append(elapsed)
            response = _response(method, url, {})
            response.headers.update({"Content-Length": "4"})
            monkeypatch.setattr(response, "iter_content", lambda chunk_size: iter([b"data"]))
            return response
        if url == JOB_URL:
            checks.append(elapsed)
            status = "accepted" if elapsed < 65 else "running" if elapsed < 100 else outcome
            return _response(method, url, {
                "status": status, "links": [{"rel": "self", "href": JOB_URL}],
            })
        if method.lower() == "post":
            submissions.append(url)
        return _response(method, url, {
            f"{API_URL}/catalogue/v1/messages": {},
            PROCESS_URL: {},
            f"{PROCESS_URL}/execution": {"links": [{"rel": "monitor", "href": JOB_URL}]},
            f"{JOB_URL}/results": (
                {"asset": {"value": {"href": FILE_URL, "file:size": 4, "type": "application/x-grib"}}}
                if outcome == "successful" else {"title": "Processing failed", "detail": "Source unavailable"}
            ),
        }[url])

    clock = Mock(wraps=datetime)
    clock.now.side_effect = lambda tz: origin + timedelta(seconds=elapsed)
    monkeypatch.setattr(cds, "datetime", clock)
    monkeypatch.setattr(cds.time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(cds.time, "sleep", sleep)
    monkeypatch.setattr(requests.Session, "request", request)
    if outcome == "successful":
        assert cds.download_batch(CREDENTIALS, selection, batch) == batch
        assert (batch / "data.grib").read_bytes() == b"data"
        assert transfers and min(transfers) >= 100
    else:
        with pytest.raises(ProcessingFailedError, match="Source unavailable"):
            cds.download_batch(CREDENTIALS, selection, batch)
        assert not transfers
        assert (batch.with_suffix(".partial") / "request.json").exists()
        assert not batch.exists()
    output = capsys.readouterr()
    messages = [line for line in output.out.splitlines() if "Still checking CDS" in line]
    offset = timedelta(minutes=15 if resume else 0)
    assert messages == [
        f"CDS request offline-job: still {status}; "
        f"{offset + timedelta(seconds=seconds)} since request started. Still checking CDS."
        for status, seconds in [("queued (accepted)", 30), ("queued (accepted)", 60), ("running", 90)]
    ]
    assert all(any(seconds <= checked < seconds + 2 for checked in checks) for seconds in (30, 60, 90))
    assert len(submissions) == (0 if resume else 1)
    assert CREDENTIALS.key not in output.out + output.err
