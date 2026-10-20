"""Coordinate monthly Zarr publication, recovery, and cataloged coverage."""

from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
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
    batch_paths_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    report_json TEXT,
    created_at TEXT NOT NULL,
    published_at TEXT,
    grib_removed INTEGER NOT NULL DEFAULT 0,
    parent_revision_id TEXT REFERENCES revisions(id),
    backup_path TEXT
);
CREATE TABLE IF NOT EXISTS months (
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    current_revision_id TEXT NOT NULL REFERENCES revisions(id),
    PRIMARY KEY (year, month)
);
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
    source_revision_id TEXT NOT NULL REFERENCES revisions(id),
    source_request_sha256 TEXT NOT NULL,
    PRIMARY KEY (revision_id, variable, valid_time)
);
CREATE UNIQUE INDEX IF NOT EXISTS ready_month
    ON revisions (year, month) WHERE status = 'ready';
"""


def _check_store(path: Path, revision: sqlite3.Row) -> None:
    """Match a store's revision identifier to its publication journal entry.

    Args:
        path: Existing Zarr store.
        revision: Catalog revision expected at that path.

    Raises:
        ValueError: The store identifies another revision.
        Exception: Zarr reading errors propagate.
    """
    attrs = zarr.open_group(path, mode="r").attrs
    if attrs["revision_id"] != revision["id"]:
        raise ValueError(f"The store at {path} belongs to a different revision.")


def _rename_store(source: Path, destination: Path) -> None:
    """Rename a store and persist directory entries where supported.

    Args:
        source: Existing store directory.
        destination: Unoccupied destination on the same filesystem.

    Raises:
        OSError: Renaming or flushing the directory entries fails.
    """
    source.rename(destination)
    if os.name != "nt":
        for directory in {source.parent, destination.parent}:
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _recover_publications(db: sqlite3.Connection, archive: Path) -> None:
    """Finish validated publications while the publication lock is held.

    Args:
        db: Open catalog connection with no active transaction.
        archive: Archive containing staged and published stores.

    Raises:
        ValueError: A store does not identify its journaled revision.
        OSError: A required store cannot be published.
        sqlite3.Error: Recording published coverage fails.
    """
    for revision in db.execute(
        "SELECT * FROM revisions WHERE status = 'ready'"
    ).fetchall():
        stage = archive / revision["staged_path"]
        final = archive / revision["final_path"]
        backup = archive / revision["backup_path"] if revision["backup_path"] else None
        parent = db.execute(
            "SELECT * FROM revisions WHERE id = ?", (revision["parent_revision_id"],)
        ).fetchone()
        if stage.exists():
            _check_store(stage, revision)
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                if parent is None or backup.exists():
                    raise FileExistsError(f"The publication destination is occupied: {final}.")
                _check_store(final, parent)
                _rename_store(final, backup)
            elif parent is not None and not backup.exists():
                raise FileNotFoundError(f"The previous monthly store is missing: {final}.")
            _rename_store(stage, final)
        _check_store(final, revision)
        if parent is not None:
            _check_store(backup, parent)
        report = json.loads(revision["report_json"])
        provenance = json.loads(revision["provenance_json"])
        with db:
            db.executemany(
                """INSERT INTO coverage
                   (revision_id, variable, valid_time, reference_time, interval_start,
                    interval_end, start_step, end_step, step_units, step_type,
                    source_revision_id, source_request_sha256)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        revision["id"], field["variable"], field["valid_time"],
                        field["reference_time"], field["interval_start"],
                        field["interval_end"], field["start_step"], field["end_step"],
                        field["step_units"], field["step_type"],
                        field.get("source_revision_id", revision["id"]),
                        field["source_request_sha256"],
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
            db.execute(
                """INSERT INTO months VALUES (?, ?, ?)
                   ON CONFLICT (year, month) DO UPDATE
                   SET current_revision_id = excluded.current_revision_id""",
                (revision["year"], revision["month"], revision["id"]),
            )

    for revision in db.execute(
        "SELECT id, backup_path FROM revisions WHERE status = 'published' AND backup_path IS NOT NULL"
    ).fetchall():
        backup = archive / revision["backup_path"]
        owned = (archive / "staging" / "revisions").resolve()
        target = backup.resolve()
        if target == owned or not target.is_relative_to(owned):
            raise ValueError(f"The old revision is outside the archive revision directory: {backup}.")
        try:
            if backup.exists():
                shutil.rmtree(backup)
        except OSError as exc:
            warnings.warn(
                f"Published data are available; old revision cleanup remains pending: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            with db:
                db.execute("UPDATE revisions SET backup_path = NULL WHERE id = ?", (revision["id"],))

    for revision in db.execute(
        "SELECT id, batch_paths_json FROM revisions WHERE status = 'published' AND grib_removed = 0"
    ).fetchall():
        try:
            batches = [Path(path) for path in json.loads(revision["batch_paths_json"])]
            for batch in batches:
                (batch / "data.grib").unlink(missing_ok=True)
            owned = (archive / "staging" / "batches").resolve()
            for batch in batches:
                batch = batch.resolve()
                if batch == owned or not batch.is_relative_to(owned):
                    continue
                checkpoint = batch / "converted.zarr"
                target = checkpoint.resolve()
                if target == batch or not target.is_relative_to(batch):
                    raise ValueError(f"The converted checkpoint is outside its batch directory: {checkpoint}.")
                if checkpoint.exists():
                    shutil.rmtree(target)
                (batch / "checkpoint.json").unlink(missing_ok=True)
        except OSError as exc:
            warnings.warn(
                f"Published data are available; source cleanup remains pending: {exc}",
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


def current_month(
    db: sqlite3.Connection, archive: Path, year: int, month: int
) -> tuple[Path, dict[str, Any]] | None:
    """Find the current monthly store and its cataloged field origins.

    Args:
        db: Catalog connection protected by writer or publication ownership.
        archive: Archive directory.
        year: Calendar year.
        month: Calendar month.

    Returns:
        Store path and validation report with revision and field source IDs,
        or None when the month has no published coverage.

    Raises:
        ValueError: The published store identifies a different revision.
        Exception: Catalog or Zarr reading errors propagate.
    """
    revision = db.execute(
        """SELECT r.* FROM months m JOIN revisions r ON r.id = m.current_revision_id
           WHERE m.year = ? AND m.month = ?""", (year, month)
    ).fetchone()
    if revision is None:
        return None
    final = archive / revision["final_path"]
    _check_store(final, revision)
    report = json.loads(revision["report_json"])
    report["revision_id"] = revision["id"]
    report["fields"] = [
        {key: field[key] for key in field.keys() if key != "revision_id"}
        for field in db.execute(
            "SELECT * FROM coverage WHERE revision_id = ? ORDER BY variable, valid_time",
            (revision["id"],),
        )
    ]
    return final, report


def missing_fields(
    db: sqlite3.Connection,
    year: int,
    month: int,
    variables: Sequence[str],
    valid_times: Sequence[int],
) -> dict[str, list[int]]:
    """Find CDS variable/hour pairs absent from completed source requests.

    Published monthly replacements preserve earlier coverage. Their recorded
    requests therefore identify completed selections independently of the
    decoder's array names and each retained field's original retrieval.

    Args:
        db: Catalog connection from archive_session.
        year: Calendar year.
        month: Calendar month.
        variables: Exact CDS variable names from the requested selection.
        valid_times: Requested UTC hours as integer Unix timestamps.

    Returns:
        Variables with missing hours, each sorted in ascending time order.
        An empty mapping means all requested fields are already published.

    Raises:
        sqlite3.Error: Reading catalog coverage fails.
    """
    missing = {variable: set(valid_times) for variable in variables}
    for revision in db.execute(
        """SELECT provenance_json FROM revisions
           WHERE year = ? AND month = ? AND status = 'published'""", (year, month)
    ):
        provenance = json.loads(revision["provenance_json"])
        for source in provenance.get("sources", [provenance]):
            request = source["request"]
            selected = missing.keys() & request["variable"]
            if not selected:
                continue
            covered = {
                int(datetime.fromisoformat(
                    f"{year:04d}-{month:02d}-{int(day):02d}T{hour}:00+00:00"
                ).timestamp())
                for day in request["day"] for hour in request["time"]
            }
            for variable in selected:
                missing[variable].difference_update(covered)
    return {
        variable: sorted(times) for variable, times in missing.items() if times
    }


def prepare_revision(
    db: sqlite3.Connection, archive: Path, batches: Sequence[Path], metadata: dict[str, Any]
) -> tuple[str, Path]:
    """Replace abandoned assembly attempts and record the month to publish.

    Args:
        db: Catalog connection from archive_session.
        archive: Archive directory.
        batches: Completed source download directories contributing to this revision.
        metadata: Download provenance or a monthly manifest of source requests.

    Returns:
        Revision identifier and the Zarr path to build in staging.

    Raises:
        FileExistsError: An uncataloged store occupies the monthly path.
        ValueError: An abandoned store is outside the archive revision directory.
        Exception: File and catalog errors propagate.
    """
    source = metadata.get("sources", [metadata])[0]
    year = int(source["request"]["year"][0])
    month = int(source["request"]["month"][0])
    final = Path(f"{year:04d}") / f"{month:02d}" / "era5.zarr"
    parent = db.execute(
        "SELECT current_revision_id FROM months WHERE year = ? AND month = ?", (year, month)
    ).fetchone()
    if parent is None and (archive / final).exists():
        raise FileExistsError(f"An uncataloged store occupies the target month: {archive / final}")
    owned = (archive / "staging" / "revisions").resolve()
    for attempt in db.execute(
        "SELECT id, staged_path FROM revisions WHERE year = ? AND month = ? AND status = 'preparing'",
        (year, month),
    ).fetchall():
        abandoned = (archive / attempt["staged_path"]).resolve()
        if abandoned == owned or not abandoned.is_relative_to(owned):
            raise ValueError(f"The abandoned revision is outside the archive revision directory: {abandoned}.")
        if abandoned.exists():
            shutil.rmtree(abandoned)
        with db:
            db.execute("DELETE FROM revisions WHERE id = ?", (attempt["id"],))
    revision_id = uuid4().hex
    stage = Path("staging") / "revisions" / f"{revision_id}.zarr"
    backup = stage.with_name(f"{revision_id}.previous.zarr") if parent is not None else None
    (archive / stage).parent.mkdir(parents=True, exist_ok=True)
    with db:
        db.execute(
            """INSERT INTO revisions
            (id, year, month, request_sha256, status, staged_path, final_path,
             batch_paths_json, provenance_json, created_at, parent_revision_id, backup_path)
            VALUES (?, ?, ?, ?, 'preparing', ?, ?, ?, ?, ?, ?, ?)""",
            (
                revision_id, year, month, metadata["request_sha256"], stage.as_posix(),
                final.as_posix(),
                json.dumps([str(batch.resolve()) for batch in batches]), json.dumps(metadata),
                datetime.now(UTC).isoformat(), parent[0] if parent is not None else None,
                backup.as_posix() if backup is not None else None,
            ),
        )
    return revision_id, archive / stage


def publish_revision(
    db: sqlite3.Connection, archive: Path, revision_id: str, report: dict[str, Any]
) -> Path:
    """Publish a validated replacement without discarding the previous revision.

    Args:
        db: Catalog connection from archive_session.
        archive: Archive directory.
        revision_id: Preparing revision returned by prepare_revision.
        report: Converter report containing all retained and newly added fields.

    Returns:
        The published monthly Zarr path.

    Raises:
        ValueError: The report, parent revision, or archive configuration differs.
        Exception: Publication errors propagate after restoring the old store
            where possible. Interrupted restoration remains journaled for recovery.
    """
    revision = db.execute("SELECT * FROM revisions WHERE id = ?", (revision_id,)).fetchone()
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
    group = zarr.open_group(stage, mode="r+")
    if group.attrs["provenance"]["request_sha256"] != revision["request_sha256"]:
        raise ValueError("The staged store belongs to a different request.")
    group.attrs["revision_id"] = revision_id
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
    with portalocker.Lock(archive / ".publication.lock", "a", flags=portalocker.LOCK_EX):
        parent = db.execute(
            "SELECT current_revision_id FROM months WHERE year = ? AND month = ?",
            (revision["year"], revision["month"]),
        ).fetchone()
        if (parent[0] if parent is not None else None) != revision["parent_revision_id"]:
            raise ValueError("The current month changed after this replacement was prepared.")
        with db:
            db.execute(
                "UPDATE revisions SET status = 'ready', report_json = ? WHERE id = ?",
                (json.dumps(report), revision_id),
            )
        try:
            _recover_publications(db, archive)
        except (OSError, sqlite3.Error):
            status = db.execute("SELECT status FROM revisions WHERE id = ?", (revision_id,)).fetchone()[0]
            if status == "ready":
                final = archive / revision["final_path"]
                backup = archive / revision["backup_path"] if revision["backup_path"] else None
                if not stage.exists() and final.exists():
                    _check_store(final, revision)
                    _rename_store(final, stage)
                if backup is not None and backup.exists():
                    _rename_store(backup, final)
                with db:
                    db.execute("UPDATE revisions SET status = 'preparing' WHERE id = ?", (revision_id,))
            raise
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
    with portalocker.Lock(archive / ".publication.lock", "a", flags=portalocker.LOCK_EX):
        with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("PRAGMA synchronous = FULL")
            db.executescript(SCHEMA)
            _recover_publications(db, archive)
            current = current_month(db, archive, year, month)
            if current is None:
                raise FileNotFoundError(f"No published data exist for {year:04d}-{month:02d}.")
            with xr.open_zarr(current[0], consolidated=False, chunks=None) as dataset:
                yield dataset
