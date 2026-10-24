"""Exercise spawned processing workers with real HTTP, NetCDF, COG, and SQLite."""

from contextlib import closing
from datetime import UTC, date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sqlite3
from threading import Thread
from typing import Any

import numpy as np
import pytest
import rasterio
import requests
import xarray as xr

from era5_pull_pipeline import catalog, conversion, ncar_download, pipeline
from era5_pull_pipeline.ncar import SourceFile


HTTP_REQUEST = requests.Session.request


@pytest.mark.parametrize("workers,missing_hour", [(1, False), (2, False), (2, True)])
def test_spawned_pull_resumes_http_and_preserves_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int, missing_hour: bool,
) -> None:
    """Resume real source bytes and preserve exact float64 pixels across processes."""
    source_path = tmp_path / "source.nc"
    count = 1 if missing_hour else 2
    values = np.broadcast_to(
        np.arange(1440, dtype=np.float64)[None, None, :] * 0.000000123456789,
        (count, 721, 1440),
    ).copy()
    values[0, 0, 0] = np.nan
    if count == 2:
        values[1] = np.nan  # A covered all-NoData hour still needs a COG.
    dataset = xr.Dataset(
        {"T2M": (("time", "latitude", "longitude"), values, {
            "short_name": "2t", "long_name": "2 metre temperature", "units": "K",
        })},
        coords={
            "time": np.arange(np.datetime64("2000-01-01T00"), np.datetime64("2000-01-01T00") + np.timedelta64(count, "h"), np.timedelta64(1, "h")),
            "latitude": conversion.GLOBAL_LATITUDES,
            "longitude": conversion.GLOBAL_LONGITUDES,
        },
    )
    dataset.to_netcdf(source_path, engine="h5netcdf", encoding={
        "T2M": {"chunksizes": (1, 181, 360), "zlib": True},
    })
    source = SourceFile(
        "source.nc", "2t", source_path.stat().st_size, '"fixture"',
        "2026-01-01T00:00:00+00:00", datetime(2000, 1, 1, tzinfo=UTC),
        datetime(2000, 1, 1, 1, tzinfo=UTC),
    )
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        row = catalog.register_sources(db, archive, [source], date(2000, 1, 1), date(2000, 1, 1))[0]
        retained = archive / row["path"]
        retained.parent.mkdir(parents=True, exist_ok=True)
        retained.write_bytes(source_path.read_bytes()[:17])
    requests_seen: list[tuple[str | None, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        """Serve the remaining bytes of a pinned provider fixture."""

        def do_GET(self) -> None:
            """Return the byte range requested by the resume adapter."""
            requests_seen.append((self.headers.get("Range"), self.headers.get("If-Match")))
            offset = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
            self.send_response(206 if offset else 200)
            self.send_header("Content-Length", str(source.size - offset))
            self.send_header("Content-Range", f"bytes {offset}-{source.size - 1}/{source.size}")
            self.end_headers()
            self.wfile.write(source_path.read_bytes()[offset:])

        def log_message(self, format: str, *args: Any) -> None:
            """Suppress fixture request logging."""

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"

        def local_request(session: requests.Session, method: str, target: str, **kwargs: Any) -> requests.Response:
            """Allow only the test's local HTTP server through the offline guard."""
            assert target.startswith(url + "/")
            return HTTP_REQUEST(session, method, target, **kwargs)

        monkeypatch.setattr(requests.Session, "request", local_request)
        monkeypatch.setattr(ncar_download, "BUCKET_URL", url)
        monkeypatch.setattr(pipeline, "discover_files", lambda *args: [source])
        try:
            if missing_hour:
                for _ in range(2):
                    with pytest.raises(ValueError, match="each selected UTC hour"):
                        pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], archive,
                                            processing_workers=workers)
                    assert retained.exists()
                with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
                    assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
                    assert db.execute("SELECT downloaded_at FROM source_files").fetchone()[0] is not None
            else:
                result = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], archive,
                                             processing_workers=workers, max_source_gb=source.size / 10**9)
                assert result["completed"] == 2 and result["unavailable"] == 22
                with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
                    rows = db.execute("SELECT path, dtype FROM observations ORDER BY valid_time").fetchall()
                    assert len(rows) == 2
                    for index, (path, dtype) in enumerate(rows):
                        assert dtype == "float64"
                        with rasterio.open(archive / path) as raster:
                            np.testing.assert_array_equal(raster.read(1), np.roll(values[index], 720, axis=1))
                assert not retained.exists()
                repeated = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], archive,
                                               processing_workers=workers)
                assert repeated["completed"] == 0 and repeated["skipped"] == 2
            assert requests_seen == [("bytes=17-", '"fixture"')]
            assert not (archive / "staging" / "decoded" / "values.bin").exists()
        finally:
            server.shutdown()
            thread.join()
