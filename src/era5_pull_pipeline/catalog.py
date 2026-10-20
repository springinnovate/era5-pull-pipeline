"""Coordinate monthly Zarr publication, recovery, and cataloged coverage."""

from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4
import warnings

import portalocker
import xarray as xr
import zarr

SCHEMA = """
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS revisions (
    id TEXT PRIMARY KEY,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    request_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('preparing', 'ready', 'published')),
    staged_path TEXT NOT NULL,
    final_path TEXT NOT NULL,
    batch_path TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    report_json TEXT,
    created_at TEXT NOT NULL,
    published_at TEXT,
    grib_removed INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS published_month
    ON revisions (year, month) WHERE status IN ('ready', 'published');
CREATE TABLE IF NOT EXISTS coverage (
    revision_id TEXT NOT NULL REFERENCES revisions(id),
    variable TEXT NOT NULL,
    valid_time INTEGER NOT NULL,
    reference_time INTEGER NOT NULL,
    interval_start INTEGER NOT NULL,
    interval_end INTEGER NOT NULL,
    start_step INTEGER NOT NULL,
    end_step INTEGER NOT NULL,
    step_units TEXT NOT NULL,
    step_type TEXT NOT NULL,
    PRIMARY KEY (revision_id, variable, valid_time)
);
"""


def _recover_publications(db: sqlite3.Connection, archive: Path) -> None:
    """Finish validated publications while the publication lock is held.

    Args:
        db: Open catalog connection with no active transaction.
        archive: Archive containing staged and published stores.

    Raises:
        ValueError: A store does not identify the journaled source request.
        OSError: A required store cannot be published.
        sqlite3.Error: Recording published coverage fails.
    """
    for revision in db.execute(
        "SELECT * FROM revisions WHERE status = 'ready'"
    ).fetchall():
        stage = archive / revision["staged_path"]
        final = archive / revision["final_path"]
        if stage.exists() and final.exists():
            raise FileExistsError(f"Both staged and published stores exist for {final}.")
        store = stage if stage.exists() else final
        group = zarr.open_group(store, mode="r")
        if group.attrs["provenance"]["request_sha256"] != revision["request_sha256"]:
            raise ValueError(f"The store at {store} belongs to a different request.")
        if store == stage:
            final.parent.mkdir(parents=True, exist_ok=True)
            stage.rename(final)
            if os.name != "nt":
                for directory in (stage.parent, final.parent):
                    descriptor = os.open(directory, os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
        report = json.loads(revision["report_json"])
        provenance = json.loads(revision["provenance_json"])
        with db:
            db.executemany(
                """INSERT INTO coverage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        revision["id"], field["variable"], field["valid_time"],
                        field["reference_time"], field["interval_start"],
                        field["interval_end"], field["start_step"], field["end_step"],
                        field["step_units"], field["step_type"],
                    )
                    for field in report["fields"]
                ],
            )
            db.executemany(
                "INSERT OR IGNORE INTO config (key, value) VALUES (?, ?)",
                [
                    (key, json.dumps(value, sort_keys=True))
                    for key, value in {
                        "dataset": provenance["dataset"],
                        "grid": report["grid"],
                        "layout": report["layout"],
                    }.items()
                ],
            )
            db.execute(
                "UPDATE revisions SET status = 'published', published_at = ? WHERE id = ?",
                (datetime.now(UTC).isoformat(), revision["id"]),
            )

    for revision in db.execute(
        "SELECT id, batch_path FROM revisions WHERE status = 'published' AND grib_removed = 0"
    ).fetchall():
        try:
            (Path(revision["batch_path"]) / "data.grib").unlink(missing_ok=True)
        except OSError as exc:
            warnings.warn(
                f"Published data are available; source GRIB cleanup remains pending: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            with db:
                db.execute(
                    "UPDATE revisions SET grib_removed = 1 WHERE id = ?", (revision["id"],)
                )


@contextmanager
def archive_session(archive: Path) -> Iterator[sqlite3.Connection]:
    """Open the catalog with exclusive writer ownership and recover publication.

    Args:
        archive: Archive directory to create or extend.

    Yields:
        A catalog connection protected from other pipeline writers.

    Raises:
        portalocker.exceptions.LockException: Another writer holds the archive.
        Exception: Catalog or publication recovery errors propagate.
    """
    archive.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(archive / ".writer.lock", "a", timeout=0):
        with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("PRAGMA synchronous = FULL")
            with portalocker.Lock(
                archive / ".publication.lock", "a", flags=portalocker.LOCK_EX
            ):
                db.executescript(SCHEMA)
                _recover_publications(db, archive)
            yield db


def existing_month(
    db: sqlite3.Connection, archive: Path, year: int, month: int, request_sha256: str
) -> Path | None:
    """Find an identical published request or reject an occupied month.

    Args:
        db: Catalog connection from archive_session.
        archive: Archive directory.
        year: Requested calendar year.
        month: Requested calendar month.
        request_sha256: Canonical request identity.

    Returns:
        The published store for an identical request, or None for a new month.

    Raises:
        ValueError: The month contains a different request or store identity.
    """
    revision = db.execute(
        "SELECT * FROM revisions WHERE year = ? AND month = ? AND status = 'published'",
        (year, month),
    ).fetchone()
    if revision is None:
        return None
    if revision["request_sha256"] != request_sha256:
        raise ValueError(
            f"{year:04d}-{month:02d} already contains a different request; "
            "adding variables or hours to an existing month is not supported."
        )
    final = archive / revision["final_path"]
    if zarr.open_group(final, mode="r").attrs["provenance"]["request_sha256"] != request_sha256:
        raise ValueError(f"The published store at {final} belongs to a different request.")
    return final


def prepare_revision(
    db: sqlite3.Connection, archive: Path, batch: Path, metadata: dict[str, Any]
) -> tuple[str, Path]:
    """Record a new conversion attempt without claiming published coverage.

    Args:
        db: Catalog connection from archive_session.
        archive: Archive directory.
        batch: Completed source download directory.
        metadata: Download provenance and canonical request identity.

    Returns:
        Revision identifier and the Zarr path to build in staging.

    Raises:
        FileExistsError: The target month is already present.
        Exception: File and catalog errors propagate.
    """
    year = int(metadata["request"]["year"][0])
    month = int(metadata["request"]["month"][0])
    final = Path(f"{year:04d}") / f"{month:02d}" / "era5.zarr"
    if (archive / final).exists():
        raise FileExistsError(f"The target month already exists: {archive / final}")
    revision_id = uuid4().hex
    stage = Path("staging") / "revisions" / f"{revision_id}.zarr"
    (archive / stage).parent.mkdir(parents=True, exist_ok=True)
    with db:
        db.execute(
            """INSERT INTO revisions
            (id, year, month, request_sha256, status, staged_path, final_path,
             batch_path, provenance_json, created_at)
            VALUES (?, ?, ?, ?, 'preparing', ?, ?, ?, ?, ?)""",
            (
                revision_id, year, month, metadata["request_sha256"], stage.as_posix(),
                final.as_posix(), str(batch.resolve()), json.dumps(metadata),
                datetime.now(UTC).isoformat(),
            ),
        )
    return revision_id, archive / stage


def publish_revision(
    db: sqlite3.Connection, archive: Path, revision_id: str, report: dict[str, Any]
) -> Path:
    """Publish a validated staged store and its exact field coverage.

    Args:
        db: Catalog connection from archive_session.
        archive: Archive directory.
        revision_id: Preparing revision returned by prepare_revision.
        report: Converter report produced after numerical and metadata validation.

    Returns:
        The published monthly Zarr path.

    Raises:
        ValueError: The report or archive configuration does not match.
        Exception: File and catalog errors propagate for recovery on the next run.
    """
    revision = db.execute(
        "SELECT * FROM revisions WHERE id = ?", (revision_id,)
    ).fetchone()
    if (report["year"], report["month"]) != (revision["year"], revision["month"]):
        raise ValueError("The validation report does not match the revision's month.")
    provenance = json.loads(revision["provenance_json"])
    for key, value in {
        "dataset": provenance["dataset"], "grid": report["grid"], "layout": report["layout"]
    }.items():
        stored = db.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
        if stored is not None and json.loads(stored["value"]) != value:
            raise ValueError(f"The requested {key} differs from the archive configuration.")
    stage = archive / revision["staged_path"]
    if zarr.open_group(stage, mode="r").attrs["provenance"]["request_sha256"] != revision["request_sha256"]:
        raise ValueError("The staged store belongs to a different request.")
    # A writable handle is necessary for fsync on Windows.
    for directory, _, files in os.walk(stage, topdown=False):
        for filename in files:
            with (Path(directory) / filename).open("r+b") as stream:
                os.fsync(stream.fileno())
        if os.name != "nt":
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    with portalocker.Lock(
        archive / ".publication.lock", "a", flags=portalocker.LOCK_EX
    ):
        with db:
            db.execute(
                "UPDATE revisions SET status = 'ready', report_json = ? WHERE id = ?",
                (json.dumps(report), revision_id),
            )
        _recover_publications(db, archive)
    return archive / revision["final_path"]


@contextmanager
def open_month(archive: Path, year: int, month: int) -> Iterator[xr.Dataset]:
    """Read a published month while preventing replacement during lazy reads.

    Args:
        archive: Existing archive directory.
        year: Calendar year to read.
        month: Calendar month to read.

    Yields:
        An xarray dataset usable until the context exits.

    Raises:
        FileNotFoundError: The catalog or published month does not exist.
        Exception: Recovery or Zarr reader errors propagate.
    """
    if not (archive / "catalog.sqlite").is_file():
        raise FileNotFoundError(f"No catalog exists at {archive / 'catalog.sqlite'}.")
    with portalocker.Lock(
        archive / ".publication.lock", "a", flags=portalocker.LOCK_EX
    ):
        with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("PRAGMA synchronous = FULL")
            _recover_publications(db, archive)
            revision = db.execute(
                """SELECT final_path, request_sha256 FROM revisions
                   WHERE year = ? AND month = ? AND status = 'published'""",
                (year, month),
            ).fetchone()
            if revision is None:
                raise FileNotFoundError(f"No published data exist for {year:04d}-{month:02d}.")
            final = archive / revision["final_path"]
            if zarr.open_group(final, mode="r").attrs["provenance"]["request_sha256"] != revision["request_sha256"]:
                raise ValueError("The published store belongs to a different request.")
            with xr.open_zarr(final, consolidated=False, chunks=None) as dataset:
                yield dataset
