"""Exercise resumable NCAR transfers against an anonymous local HTTP server."""

from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import sqlite3
from threading import Thread
from typing import Any
from unittest.mock import Mock, call

import pytest
import requests

from era5_pull_pipeline import catalog, ncar_download
from era5_pull_pipeline.ncar import SourceFile

REQUEST = requests.Session.request
START = date(2000, 1, 1)


@dataclass
class TransferServer:
    """Expose a source object and scripted HTTP failures to workflow tests."""

    payload: bytes
    source: SourceFile
    responses: list[str | int] = field(default_factory=list)
    requests: list[dict[str, str]] = field(default_factory=list)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[TransferServer]:
    """Serve source bytes locally while continuing to reject external requests.

    Args:
        monkeypatch: Isolate the public bucket URL and inventory lookup.

    Yields:
        Payload, source metadata, queued failure responses, and received headers.
    """
    payload = b"netcdf-transfer-fixture\x00" * 100000
    state = TransferServer(
        payload=payload,
        source=SourceFile(
            key="e5.oper.an.sfc/200001/e5.oper.an.sfc.128_167_2t.ll025sc.2000010100_2000013123.nc",
            variable="2t",
            size=len(payload),
            etag='"object-version"',
            last_modified="2026-01-02T03:04:05+00:00",
            valid_start=datetime(2000, 1, 1, tzinfo=UTC),
            valid_end=datetime(2000, 1, 31, 23, tzinfo=UTC),
        ),
    )

    class Handler(BaseHTTPRequestHandler):
        """Return complete, interrupted, or refused-range source transfers."""

        def do_GET(self) -> None:
            """Serve the next scripted response and record its request headers."""
            state.requests.append(dict(self.headers))
            response = state.responses.pop(0) if state.responses else "complete"
            if isinstance(response, int):
                self.send_error(response)
                return
            offset = int(self.headers.get("Range", "bytes=0-")[6:-1])
            if response == "refuse-range":
                offset = 0
            self.send_response(206 if offset else 200)
            self.send_header("Content-Length", str(len(payload) - offset))
            if offset:
                self.send_header("Content-Range", f"bytes {offset}-{len(payload) - 1}/{len(payload)}")
            self.end_headers()
            if response == "disconnect":
                self.wfile.write(payload[offset:offset + ncar_download.CHUNK_BYTES + 137])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
            else:
                try:
                    self.wfile.write(payload[offset:])
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    # Range rejection closes the response before reading its body.
                    pass

        def log_message(self, format: str, *args: Any) -> None:
            """Keep expected transfer failures out of the test runner's stderr."""

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{http.server_port}"

    def local_request(
        session: requests.Session, method: str, request_url: str, **kwargs: Any
    ) -> requests.Response:
        """Allow requests only to this fixture's local source server."""
        assert request_url.startswith(f"{url}/"), "Automated tests must not contact external servers."
        return REQUEST(session, method, request_url, **kwargs)

    monkeypatch.setattr(requests.Session, "request", local_request)
    monkeypatch.setattr(ncar_download, "BUCKET_URL", url)
    monkeypatch.setattr(ncar_download, "discover_files", Mock(return_value=[state.source]))
    monkeypatch.setattr(ncar_download.time, "sleep", Mock())
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)


def test_interrupted_transfer_resumes_and_completed_file_is_reused(
    tmp_path: Path, server: TransferServer
) -> None:
    """Resume a broken response with Range/If-Match and reuse its completed file."""
    archive = tmp_path / "archive"
    server.responses = ["disconnect"]
    paths = ncar_download.download_ncar(START, START, ["2t"], archive)
    assert len(paths) == 1
    assert paths[0].read_bytes() == server.payload
    assert len(server.requests) == 2
    assert "Range" not in server.requests[0]
    assert server.requests[1]["Range"] == f"bytes={ncar_download.CHUNK_BYTES}-"
    assert all(request["If-Match"] == server.source.etag for request in server.requests)
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert db.execute("SELECT downloaded_at FROM source_files").fetchone()[0] is not None
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0
    assert ncar_download.download_ncar(START, START, ["2t"], archive) == paths
    assert len(server.requests) == 2
    assert paths[0].read_bytes() == server.payload


def test_retry_limit_keeps_partial_source_for_next_invocation(
    tmp_path: Path, server: TransferServer
) -> None:
    """Leave exhausted work pending and resume its existing path on the next run."""
    archive = tmp_path / "archive"
    server.responses = ["disconnect", 503, 503]
    with pytest.raises(requests.HTTPError, match="503"):
        ncar_download.download_ncar(START, START, ["2t"], archive)
    assert len(server.requests) == 3
    assert ncar_download.time.sleep.call_args_list == [call(2), call(4)]
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        recorded_path, downloaded_at = db.execute(
            "SELECT path, downloaded_at FROM source_files"
        ).fetchone()
    path = Path(recorded_path)
    assert downloaded_at is None
    assert path.read_bytes() == server.payload[:ncar_download.CHUNK_BYTES]
    paths = ncar_download.download_ncar(START, START, ["2t"], archive)
    assert paths == [path]
    assert path.read_bytes() == server.payload
    assert len(server.requests) == 4
    assert server.requests[-1]["Range"] == f"bytes={ncar_download.CHUNK_BYTES}-"
    assert server.requests[-1]["If-Match"] == server.source.etag
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert db.execute("SELECT COUNT(*) FROM source_files").fetchone()[0] == 1
        assert db.execute("SELECT downloaded_at FROM source_files").fetchone()[0] is not None


def test_range_refusal_keeps_partial_bytes_unchanged(
    tmp_path: Path, server: TransferServer
) -> None:
    """Reject a complete response to a resumed request before appending bytes."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        row = catalog.register_sources(db, archive, [server.source], START, START)[0]
        path = Path(row["path"])
        path.parent.mkdir(parents=True)
        path.write_bytes(server.payload[:ncar_download.CHUNK_BYTES])
    server.responses = ["refuse-range"]
    with pytest.raises(ValueError, match="requested byte range"):
        ncar_download.download_ncar(START, START, ["2t"], archive)
    assert len(server.requests) == 1
    assert path.read_bytes() == server.payload[:ncar_download.CHUNK_BYTES]
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert db.execute("SELECT downloaded_at FROM source_files").fetchone()[0] is None


def test_permanent_http_error_is_not_retried(
    tmp_path: Path, server: TransferServer
) -> None:
    """Propagate a missing source without consuming transient retries."""
    archive = tmp_path / "archive"
    server.responses = [404]
    with pytest.raises(requests.HTTPError, match="404"):
        ncar_download.download_ncar(START, START, ["2t"], archive)
    assert len(server.requests) == 1
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert db.execute("SELECT downloaded_at FROM source_files").fetchone()[0] is None


def test_catalog_failure_after_transfer_reuses_pending_complete_bytes(
    tmp_path: Path, server: TransferServer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover the gap between receiving the full file and catalog completion."""
    archive = tmp_path / "archive"

    @contextmanager
    def failing_catalog(path: Path) -> Iterator[sqlite3.Connection]:
        """Reject the completion write while retaining the registered source."""
        with catalog.archive_session(path) as db:
            db.execute(
                "CREATE TEMP TRIGGER fail_completion BEFORE UPDATE OF downloaded_at ON source_files "
                "BEGIN SELECT RAISE(ABORT, 'completion write failed'); END"
            )
            yield db

    with monkeypatch.context() as patch:
        patch.setattr(ncar_download, "archive_session", failing_catalog)
        with pytest.raises(sqlite3.IntegrityError, match="completion write failed"):
            ncar_download.download_ncar(START, START, ["2t"], archive)
    assert len(server.requests) == 1
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert db.execute("SELECT downloaded_at FROM source_files").fetchone()[0] is None
    paths = ncar_download.download_ncar(START, START, ["2t"], archive)
    assert len(server.requests) == 1
    assert paths[0].read_bytes() == server.payload
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert db.execute("SELECT downloaded_at FROM source_files").fetchone()[0] is not None


def test_dry_run_reports_sources_without_local_state_or_transfer(
    tmp_path: Path, server: TransferServer, capsys: pytest.CaptureFixture[str]
) -> None:
    """List complete source objects and sizes without creating an archive."""
    archive = tmp_path / "archive"
    assert ncar_download.download_ncar(START, START, ["2t"], archive, dry_run=True) == []
    assert server.requests == []
    assert not archive.exists()
    output = capsys.readouterr().out
    assert server.source.key in output
    assert f"{len(server.payload):,} bytes" in output
