"""Publish observation COGs and track NCAR sources in the SQLite catalog."""

from calendar import monthrange
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, contextmanager
from datetime import UTC, date, datetime, time, timedelta
import json
import logging
import os
from pathlib import Path
import sqlite3
from time import sleep
from typing import Any
from uuid import uuid4
import warnings

import portalocker

from era5_pull_pipeline.ncar import SourceFile

SQLITE_BUSY_TIMEOUT_SECONDS = 10
COMMIT_RETRY_DELAYS = (1, 2, 4)
INSTALL_WORKERS = 8
LOGGER = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS staging_config (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    directory TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS local_files (
    kind TEXT NOT NULL CHECK (kind IN ('source', 'cog')),
    id TEXT NOT NULL,
    path TEXT NOT NULL,
    PRIMARY KEY (kind, id)
);
CREATE TABLE IF NOT EXISTS source_files (
    id TEXT PRIMARY KEY,
    object_key TEXT NOT NULL,
    variable TEXT NOT NULL,
    size INTEGER NOT NULL,
    etag TEXT NOT NULL,
    last_modified TEXT NOT NULL,
    valid_start TEXT NOT NULL,
    valid_end TEXT NOT NULL,
    path TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    downloaded_at TEXT,
    retrieved_at TEXT,
    removed_at TEXT,
    attributes_json TEXT,
    UNIQUE (object_key, etag)
);
CREATE TABLE IF NOT EXISTS source_selections (
    source_id TEXT NOT NULL REFERENCES source_files(id),
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    PRIMARY KEY (source_id, start_time, end_time)
);
CREATE TABLE IF NOT EXISTS variables (
    name TEXT PRIMARY KEY,
    attributes_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS coverage (
    variable TEXT NOT NULL REFERENCES variables(name),
    valid_time INTEGER NOT NULL,
    path TEXT UNIQUE NOT NULL,
    reference_time INTEGER,
    interval_start INTEGER NOT NULL,
    interval_end INTEGER NOT NULL,
    start_step INTEGER,
    end_step INTEGER,
    step_units TEXT NOT NULL,
    step_type TEXT NOT NULL,
    source_file_id TEXT NOT NULL REFERENCES source_files(id),
    source_variable TEXT NOT NULL,
    raster_json TEXT NOT NULL,
    PRIMARY KEY (variable, valid_time)
);
CREATE INDEX IF NOT EXISTS coverage_time ON coverage (valid_time);
CREATE TABLE IF NOT EXISTS publications (
    id TEXT PRIMARY KEY,
    variable TEXT NOT NULL,
    valid_time INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('preparing', 'ready')),
    staged_path TEXT NOT NULL,
    final_path TEXT NOT NULL,
    field_json TEXT NOT NULL,
    attributes_json TEXT NOT NULL,
    raster_json TEXT,
    UNIQUE (variable, valid_time)
);
CREATE VIEW IF NOT EXISTS observations AS
SELECT c.variable, c.valid_time, c.path,
       json_extract(v.attributes_json, '$.name') AS name,
       json_extract(v.attributes_json, '$.units') AS units,
       json_extract(c.raster_json, '$.band') AS band,
       json_extract(c.raster_json, '$.dtype') AS dtype,
       json_extract(c.raster_json, '$.nodata') AS nodata,
       json_extract(c.raster_json, '$.crs') AS crs,
       json_extract(c.raster_json, '$.transform') AS transform,
       json_extract(c.raster_json, '$.shape[0]') AS height,
       json_extract(c.raster_json, '$.shape[1]') AS width,
       c.reference_time, c.interval_start, c.interval_end,
       CASE c.step_type WHEN 'instant' THEN 'instant'
                        WHEN 'avg' THEN 'mean'
                        WHEN 'accum' THEN 'accumulation' END AS temporal_type,
       c.start_step, c.end_step, c.step_units,
       c.source_file_id, c.source_variable, f.object_key, f.etag, f.last_modified, f.retrieved_at
FROM coverage c JOIN variables v ON v.name = c.variable
JOIN source_files f ON f.id = c.source_file_id
WHERE NOT EXISTS (
    SELECT 1 FROM publications p WHERE p.variable = c.variable
    AND p.valid_time = c.valid_time AND p.status = 'ready'
);
"""

DATASET_YAML = """id: era5-ncar
title: ERA5 hourly weather
source:
  format: temporal_cog
  catalog: catalog.sqlite
"""


@contextmanager
def catalog_transaction(db: sqlite3.Connection) -> Iterator[None]:
    """Commit catalog changes, retrying when another connection blocks COMMIT.

    A busy COMMIT leaves SQLite's transaction active. Retry that commit without
    repeating SQL statements, file installation, or source cleanup. The native
    connection timeout applies to each attempt. Roll back on exhaustion, an
    unrelated failure, or interruption, preserving durable publication journals.

    Args:
        db: Connection owned by the archive writer, using implicit transactions.

    Yields:
        Control to the caller's existing transaction body.

    Raises:
        BaseException: The original operation, commit, or interruption error
            propagates after rollback. Only an active SQLITE_BUSY commit retries.
    """
    try:
        yield
        for attempt in range(len(COMMIT_RETRY_DELAYS) + 1):
            try:
                db.commit()
                return
            except sqlite3.OperationalError as exc:
                if (getattr(exc, "sqlite_errorcode", None) != sqlite3.SQLITE_BUSY
                        or not db.in_transaction or attempt == len(COMMIT_RETRY_DELAYS)):
                    raise
                delay = COMMIT_RETRY_DELAYS[attempt]
                LOGGER.warning(
                    "Catalog commit is waiting for another database connection; retry %s/%s in %s seconds.",
                    attempt + 1, len(COMMIT_RETRY_DELAYS), delay,
                )
                sleep(delay)
    except BaseException:
        db.rollback()
        raise


def _install_observation(source: Path, destination: Path) -> None:
    """Install a ready observation or confirm its already installed recovery path.

    The publication owner creates destination directories before scheduling file
    work. Each ready journal entry may refer to a staged file or to its already
    installed destination after an interrupted publication.

    Args:
        source: Complete, validated staged COG on the archive filesystem.
        destination: Deterministic published path whose parent already exists.

    Raises:
        OSError: The journaled file is missing or installing/flushing it fails.
    """
    if not source.exists():
        if not destination.is_file():
            raise FileNotFoundError(f"The journaled observation is missing: {destination}")
        return
    os.replace(source, destination)
    if os.name != "nt":
        for directory in {source.parent, destination.parent}:
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _recover_publications(
    db: sqlite3.Connection, archive: Path, identifiers: Sequence[str] | None = None,
) -> None:
    """Install ready COGs concurrently and record their coverage in one transaction.

    At most eight file operations run together, after the ready journal commits.
    All file workers settle before coverage recording or error propagation. The
    calling publication owner performs every catalog operation and local cleanup.

    Args:
        db: Catalog connection with exclusive writer ownership.
        archive: Archive containing the journaled relative paths.
        identifiers: Completed batch to install; None recovers every ready entry.

    Raises:
        Exception: File installation or catalog errors propagate; the ready
            journal entry remains for the next writer to finish.
    """
    query = """SELECT p.*, l.path AS local_path FROM publications p
               LEFT JOIN local_files l ON l.kind='cog' AND l.id=p.id
               WHERE p.status = 'ready'"""
    if identifiers is not None:
        query += " AND p.id IN (" + ",".join("?" for _ in identifiers) + ")"
    publications = db.execute(query, identifiers or []).fetchall()
    if not publications:
        return
    staged = [archive / publication["staged_path"] for publication in publications]
    destinations = [archive / publication["final_path"] for publication in publications]
    for directory in {path.parent for path in destinations}:
        directory.mkdir(parents=True, exist_ok=True)
    if len(publications) == 1:
        _install_observation(staged[0], destinations[0])
    else:
        with ThreadPoolExecutor(max_workers=INSTALL_WORKERS, thread_name_prefix="observation-install") as executor:
            # Leaving this context waits for every started file operation, even
            # if one fails, before another writer can recover the ready journal.
            for _ in executor.map(_install_observation, staged, destinations):
                pass
    local_root = get_staging_dir(db)
    # Archive bytes and their ready journal are already durable. Remove the
    # local copy before coverage commits, so cleanup remains recoverable.
    for publication in publications:
        if publication["local_path"] is not None:
            (local_root / publication["local_path"]).unlink(missing_ok=True)
    variables = {}
    coverage = []
    for publication in publications:
        field = json.loads(publication["field_json"])
        variables[field["variable"]] = publication["attributes_json"]
        coverage.append((
            field["variable"], field["valid_time"], publication["final_path"],
            field["reference_time"], field["interval_start"], field["interval_end"],
            field["start_step"], field["end_step"], field["step_units"], field["step_type"],
            field["source_file_id"], field["source_variable"], publication["raster_json"],
        ))
    completed = [(publication["id"],) for publication in publications]
    with catalog_transaction(db):
        db.executemany(
            """INSERT INTO variables VALUES (?, ?)
               ON CONFLICT (name) DO UPDATE SET attributes_json = excluded.attributes_json""",
            variables.items(),
        )
        db.executemany(
            """INSERT INTO coverage
               (variable, valid_time, path, reference_time, interval_start,
                interval_end, start_step, end_step, step_units, step_type,
                source_file_id, source_variable, raster_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (variable, valid_time) DO UPDATE SET
                path = excluded.path, reference_time = excluded.reference_time,
                interval_start = excluded.interval_start, interval_end = excluded.interval_end,
                start_step = excluded.start_step, end_step = excluded.end_step,
                step_units = excluded.step_units, step_type = excluded.step_type,
                source_file_id = excluded.source_file_id,
                source_variable = excluded.source_variable, raster_json = excluded.raster_json""",
            coverage,
        )
        db.executemany("DELETE FROM publications WHERE id = ?", completed)
        db.executemany("DELETE FROM local_files WHERE kind='cog' AND id=?", completed)


def get_staging_dir(db: sqlite3.Connection) -> Path | None:
    """Read the archive's configured local staging directory.

    Args:
        db: Connection owned by the archive writer.

    Returns:
        The remembered absolute directory, or None for archive-side staging.
    """
    row = db.execute("SELECT directory FROM staging_config WHERE id=1").fetchone()
    return None if row is None else Path(row["directory"])


def _discard_unpublished_local_cogs(db: sqlite3.Connection, archive: Path, local_root: Path) -> None:
    """Remove uncommitted local outputs and partial uploads before rebuilding them.

    Args:
        db: Catalog after recovery of every ready publication.
        archive: Root of journaled upload destinations.
        local_root: Root of cataloged local COG paths.

    Raises:
        OSError: Cleanup fails; recorded inputs and unfinished work remain.
    """
    rows = db.execute("SELECT id,path FROM local_files WHERE kind='cog'").fetchall()
    for row in rows:
        (local_root / row["path"]).unlink(missing_ok=True)
        (archive / row["path"]).unlink(missing_ok=True)
    with catalog_transaction(db):
        db.executemany("DELETE FROM publications WHERE id=?", [(row["id"],) for row in rows])
        db.executemany("DELETE FROM local_files WHERE kind='cog' AND id=?", [(row["id"],) for row in rows])


@contextmanager
def archive_session(archive: Path, *, staging_dir: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open the catalog with one writer and finish interrupted publications.

    Consumers refresh their catalog after updates and hold source files unchanged
    during calculations. This writer lock prevents competing pipeline processes.

    Args:
        archive: Archive directory to create or extend.
        staging_dir: Optional local directory to remember for new temporary files.

    Yields:
        A catalog connection protected from other pipeline writers.

    Raises:
        portalocker.exceptions.LockException: Another writer holds the archive or local directory.
        ValueError: The catalog is incompatible, the local root overlaps the archive,
            or a different local root still has unfinished files.
        Exception: Catalog or publication recovery errors propagate.
    """
    archive.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(archive / ".writer.lock", "a", timeout=0):
        with closing(sqlite3.connect(archive / "catalog.sqlite", timeout=SQLITE_BUSY_TIMEOUT_SECONDS)) as db, ExitStack() as resources:
            db.row_factory = sqlite3.Row
            coverage_columns = {row["name"] for row in db.execute("PRAGMA table_info(coverage)")}
            if coverage_columns and "path" not in coverage_columns:
                raise ValueError(
                    f"The archive at {archive} is not a COG catalog. "
                    "Use an existing COG archive or choose a new archive directory."
                )
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("PRAGMA synchronous = FULL")
            db.executescript(SCHEMA)
            local_root = get_staging_dir(db)
            if staging_dir is not None:
                requested = staging_dir.expanduser().resolve()
                if requested.is_relative_to(archive.resolve()):
                    raise ValueError("Choose --staging-dir outside the archive directory.")
                if local_root is not None and requested != local_root and db.execute(
                    "SELECT 1 FROM local_files LIMIT 1"
                ).fetchone() is not None:
                    raise ValueError(f"Local files remain in {local_root}. Resume with that staging directory before changing it.")
                local_root = requested
            if local_root is not None:
                local_root.mkdir(parents=True, exist_ok=True)
                resources.enter_context(portalocker.Lock(local_root / ".writer.lock", "a", timeout=0))
                with catalog_transaction(db):
                    db.execute("INSERT INTO staging_config VALUES (1, ?) ON CONFLICT (id) DO UPDATE SET directory=excluded.directory",
                               (str(local_root),))
            _recover_publications(db, archive)
            if local_root is not None:
                _discard_unpublished_local_cogs(db, archive, local_root)
            delete_ingested_netcdf_files(db, archive)
            descriptor = archive / "dataset.yaml"
            if not descriptor.exists():
                staged_descriptor = archive / ".dataset.yaml.tmp"
                staged_descriptor.write_text(DATASET_YAML, encoding="utf-8")
                os.replace(staged_descriptor, descriptor)
            yield db


def prepare_observation(
    db: sqlite3.Connection, archive: Path, field: dict[str, Any], attributes: dict[str, Any],
) -> tuple[str, Path]:
    """Record an observation conversion and return its staged COG path.

    Replace an abandoned conversion for the same observation. Existing published
    coverage stays visible until the replacement file is complete and validated.

    Args:
        db: Catalog connection protected by archive_session.
        archive: Archive receiving the observation.
        field: Decoded variable, UTC timestamps, interval semantics and source IDs.
        attributes: Variable name, units and source metadata.

    Returns:
        Publication identifier and absolute or archive-rooted staged TIFF path.

    Raises:
        Exception: Filesystem or catalog errors propagate.
    """
    return prepare_observations(db, archive, [field], attributes)[0]


def prepare_observations(
    db: sqlite3.Connection, archive: Path, fields: Sequence[dict[str, Any]],
    attributes: dict[str, Any],
) -> list[tuple[str, Path]]:
    """Journal a source's selected observation conversions in one transaction.

    Each preparation has its own staged path and recovery state. Abandoned
    preparations for these observations are replaced together before encoding.

    Args:
        db: Connection owned by the archive writer.
        archive: Archive receiving the staged COGs.
        fields: Selected observations for one variable, in output order.
        attributes: Variable metadata shared by these observations.

    Returns:
        Publication identities and conversion paths in the same order as fields.
        Local paths must be uploaded before publication; their relative paths
        match the archive-side journal destinations.

    Raises:
        Exception: Staging cleanup or catalog preparation fails.
    """
    if not fields:
        return []
    previous = {row["valid_time"]: row for row in db.execute(
        "SELECT * FROM publications WHERE variable=? AND valid_time BETWEEN ? AND ?",
        (fields[0]["variable"], min(f["valid_time"] for f in fields), max(f["valid_time"] for f in fields)),
    ).fetchall()}
    local_root = get_staging_dir(db)
    output_root = local_root or archive
    (output_root / "staging" / "observations").mkdir(parents=True, exist_ok=True)
    replaced = []
    rows = []
    prepared = []
    for field in fields:
        if field["valid_time"] in previous:
            old = previous[field["valid_time"]]
            (archive / old["staged_path"]).unlink(missing_ok=True)
            if local_root is not None:
                (local_root / old["staged_path"]).unlink(missing_ok=True)
            replaced.append((old["id"],))
        timestamp = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=field["valid_time"])
        variable = field["variable"]
        final = Path(f"{timestamp:%Y/%m}") / variable / f"{variable}_{timestamp:%Y%m%dT%H%M%SZ}.tif"
        identifier = uuid4().hex
        staged = Path("staging") / "observations" / f"{identifier}.tif"
        rows.append((identifier, variable, field["valid_time"], staged.as_posix(), final.as_posix(),
                     json.dumps(field), json.dumps(attributes)))
        prepared.append((identifier, output_root / staged))
    with catalog_transaction(db):
        db.executemany("DELETE FROM publications WHERE id=?", replaced)
        db.executemany("DELETE FROM local_files WHERE kind='cog' AND id=?", replaced)
        db.executemany(
            """INSERT INTO publications
               (id, variable, valid_time, status, staged_path, final_path, field_json, attributes_json)
               VALUES (?, ?, ?, 'preparing', ?, ?, ?, ?)""", rows,
        )
        if local_root is not None:
            db.executemany("INSERT INTO local_files VALUES ('cog', ?, ?)",
                           [(row[0], row[3]) for row in rows])
    return prepared


def publish_observation(
    db: sqlite3.Connection, archive: Path, publication_id: str, raster_report: dict[str, Any],
) -> Path:
    """Install a validated COG and record its observation in the catalog.

    Flush the file before recording the ready journal. If installation or the
    coverage transaction fails, the next archive_session finishes publication.
    The caller consumes published source selections before deleting downloads.

    Args:
        db: Catalog connection protected by archive_session.
        archive: Archive receiving the COG.
        publication_id: Identifier returned by prepare_observation.
        raster_report: Validated writer report with shape, dtype, NoData and grid.

    Returns:
        The deterministic published COG path.

    Raises:
        Exception: File flushing, installation or catalog errors propagate.
    """
    return publish_observations(db, archive, [(publication_id, raster_report)])[0]


def publish_observations(
    db: sqlite3.Connection, archive: Path,
    observations: Sequence[tuple[str, dict[str, Any]]], *, files_flushed: bool = False,
) -> list[Path]:
    """Durably publish a completed COG batch with two catalog transactions.

    Flush every validated file before committing its ready journal entry.
    Install the files, then record their coverage together. An interruption
    between these steps leaves ready entries recoverable even when some files
    are already installed. Acknowledgement follows the final coverage commit.

    Args:
        db: Connection owned by the archive writer.
        archive: Archive holding journaled staged and final paths.
        observations: Publication identities and validated raster reports.
        files_flushed: Upload workers already flushed these archive-side files.

    Returns:
        Published paths in the same order as observations.

    Raises:
        Exception: Flush, installation, or catalog errors propagate with the
            source inputs and journal available for recovery.
    """
    if not observations:
        return []
    identifiers = [identifier for identifier, _ in observations]
    placeholders = ','.join('?' for _ in identifiers)
    publications = {row["id"]: row for row in db.execute(
        f"SELECT * FROM publications WHERE id IN ({placeholders})", identifiers,
    ).fetchall()}
    if not files_flushed:
        for identifier in identifiers:
            with (archive / publications[identifier]["staged_path"]).open("r+b") as stream:
                os.fsync(stream.fileno())
    with catalog_transaction(db):
        db.executemany(
            "UPDATE publications SET status='ready', raster_json=? WHERE id=?",
            [(json.dumps(report), identifier) for identifier, report in observations],
        )
    _recover_publications(db, archive, identifiers)
    return [archive / publications[identifier]["final_path"] for identifier in identifiers]


def get_published_hours(
    db: sqlite3.Connection, year: int, month: int,
) -> set[tuple[str, int]]:
    """Return published NCAR variable/hour pairs within one calendar month.

    Args:
        db: Catalog connection protected by writer ownership.
        year: Calendar year.
        month: Calendar month.

    Returns:
        Source short names and UTC hours as integer Unix timestamps.
    """
    start = datetime(year, month, 1, tzinfo=UTC)
    end = start + timedelta(days=monthrange(year, month)[1])
    return {
        (row["source_variable"], row["valid_time"])
        for row in db.execute(
            "SELECT source_variable, valid_time FROM observations WHERE valid_time >= ? AND valid_time < ?",
            (int(start.timestamp()), int(end.timestamp())),
        )
    }


def remove_published_source_hours(
    db: sqlite3.Connection, selections: Sequence[dict[str, str]]
) -> None:
    """Remove published hours from pending NCAR selections in the catalog.

    The caller commits this update in the same transaction as publication.
    Hours outside the published intervals remain pending.

    Args:
        db: Catalog connection inside the caller's publication transaction.
        selections: Published inclusive intervals with source_id, start_time,
            and end_time as UTC ISO timestamps.

    Raises:
        sqlite3.Error: Updating the pending selections fails.
    """
    for selection in selections:
        start = datetime.fromisoformat(selection["start_time"])
        end = datetime.fromisoformat(selection["end_time"])
        pending = db.execute(
            """SELECT start_time, end_time FROM source_selections
               WHERE source_id = ? AND start_time <= ? AND end_time >= ?""",
            (selection["source_id"], end.isoformat(), start.isoformat()),
        ).fetchall()
        for interval in pending:
            first = datetime.fromisoformat(interval["start_time"])
            last = datetime.fromisoformat(interval["end_time"])
            db.execute(
                """DELETE FROM source_selections
                   WHERE source_id = ? AND start_time = ? AND end_time = ?""",
                (selection["source_id"], interval["start_time"], interval["end_time"]),
            )
            remaining = []
            if first < start:
                remaining.append((first, start - timedelta(hours=1)))
            if last > end:
                remaining.append((end + timedelta(hours=1), last))
            db.executemany(
                """INSERT OR IGNORE INTO source_selections
                   (source_id, start_time, end_time) VALUES (?, ?, ?)""",
                [
                    (selection["source_id"], first.isoformat(), last.isoformat())
                    for first, last in remaining
                ],
            )


def delete_ingested_netcdf_files(db: sqlite3.Connection, archive: Path) -> None:
    """Delete NCAR NetCDF downloads after all selected hours are published.

    Delete completed or partial files once no selected hours still need them.
    Record removal and clear download completion so later selections fetch the
    source again. A failed deletion remains pending for the next archive run.

    Args:
        db: Catalog connection protected by writer ownership.
        archive: Root containing the recorded relative source paths.

    Raises:
        sqlite3.Error: Reading source state or recording removal fails.
    """
    local_root = get_staging_dir(db)
    for source in db.execute(
        """SELECT f.id, f.path, l.path AS local_path FROM source_files f
           LEFT JOIN local_files l ON l.kind='source' AND l.id=f.id
           WHERE removed_at IS NULL AND NOT EXISTS (
               SELECT 1 FROM source_selections s WHERE s.source_id = f.id
           )"""
    ).fetchall():
        try:
            target = local_root / source["local_path"] if source["local_path"] is not None else archive / source["path"]
            target.unlink(missing_ok=True)
        except OSError as exc:
            warnings.warn(
                f"Published data are available; source cleanup remains pending: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            with catalog_transaction(db):
                db.execute(
                    "UPDATE source_files SET downloaded_at = NULL, removed_at = ? WHERE id = ?",
                    (datetime.now(UTC).isoformat(), source["id"]),
                )
                db.execute("DELETE FROM local_files WHERE kind='source' AND id=?", (source["id"],))


def group_hour_intervals(hours: Iterable[int]) -> list[tuple[int, int]]:
    """Group UTC hours into contiguous inclusive timestamp intervals.

    Args:
        hours: UTC hours as integer Unix timestamps, in any order.

    Returns:
        Sorted pairs containing each interval's first and last hour.
    """
    intervals: list[tuple[int, int]] = []
    for hour in sorted(set(hours)):
        if intervals and hour == intervals[-1][1] + 3600:
            intervals[-1] = (intervals[-1][0], hour)
        else:
            intervals.append((hour, hour))
    return intervals


def register_sources(
    db: sqlite3.Connection,
    archive: Path,
    sources: Sequence[SourceFile],
    start: date,
    end: date,
    *,
    selected_hours: set[tuple[str, int]] | None = None,
) -> list[sqlite3.Row]:
    """Register source files and selected hours before starting downloads.

    The object key and ETag identify a source revision. Repeated selections
    reuse its recorded path and completion state without inspecting the file.
    Only requested hours enter the pending selections. Remove them after their
    observation publication succeeds; extra hours in the file remain unselected.

    Args:
        db: Catalog connection protected by the archive writer lock.
        archive: Archive owning the staged source files.
        sources: Discovered source files overlapping the requested dates.
        start: First selected UTC date, inclusive.
        end: Last selected UTC date, inclusive.
        selected_hours: Exact variable/hour pairs to register within these dates;
            None selects every available hour.

    Returns:
        Source catalog rows in discovery order, including download completion
        and the nullable local_path relative to the configured local root.

    Raises:
        sqlite3.Error: Reading or recording source files fails.
    """
    selected_start = datetime.combine(start, time.min, tzinfo=UTC)
    selected_end = datetime.combine(end, time(23), tzinfo=UTC)
    rows = []
    local_root = get_staging_dir(db)
    with catalog_transaction(db):
        for source in sources:
            row = db.execute(
                "SELECT * FROM source_files WHERE object_key = ? AND etag = ?",
                (source.key, source.etag),
            ).fetchone()
            if row is None:
                identifier = uuid4().hex
                path = archive / "staging" / "ncar" / identifier / Path(source.key).name
                db.execute(
                    """INSERT INTO source_files
                       (id, object_key, variable, size, etag, last_modified,
                        valid_start, valid_end, path, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        identifier, source.key, source.variable, source.size,
                        source.etag, source.last_modified,
                        source.valid_start.astimezone(UTC).isoformat(),
                        source.valid_end.astimezone(UTC).isoformat(),
                        path.relative_to(archive).as_posix(), datetime.now(UTC).isoformat(),
                    ),
                )
                if local_root is not None:
                    db.execute("INSERT INTO local_files VALUES ('source', ?, ?)",
                               (identifier, path.relative_to(archive).as_posix()))
                row = db.execute(
                    "SELECT * FROM source_files WHERE id = ?", (identifier,)
                ).fetchone()
            elif row["removed_at"] is not None:
                db.execute("UPDATE source_files SET removed_at = NULL WHERE id = ?", (row["id"],))
                if local_root is not None:
                    db.execute("INSERT INTO local_files VALUES ('source', ?, ?)", (row["id"], row["path"]))
                row = db.execute("SELECT * FROM source_files WHERE id = ?", (row["id"],)).fetchone()
            first = int(max(selected_start, source.valid_start).timestamp())
            last = int(min(selected_end, source.valid_end).timestamp())
            intervals = [(first, last)] if selected_hours is None else group_hour_intervals(
                hour for variable, hour in selected_hours
                if variable == source.variable and first <= hour <= last
            )
            db.executemany(
                """INSERT OR IGNORE INTO source_selections
                   (source_id, start_time, end_time) VALUES (?, ?, ?)""",
                [
                    (row["id"], (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=first)).isoformat(),
                     (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=last)).isoformat())
                    for first, last in intervals
                ],
            )
            rows.append(db.execute(
                "SELECT f.*, l.path AS local_path FROM source_files f LEFT JOIN local_files l "
                "ON l.kind='source' AND l.id=f.id WHERE f.id=?", (row["id"],),
            ).fetchone())
    return rows
