"""Convert an existing monthly Zarr archive into a separate hourly COG archive."""

import argparse
from contextlib import ExitStack, closing
from datetime import UTC, datetime
import json
import mmap
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
from urllib.parse import quote

import numpy as np
import portalocker
from tqdm import tqdm
import zarr

from era5_pull_pipeline.catalog import archive_session, prepare_observation, publish_observation
from era5_pull_pipeline.conversion import write_observation_cog


def copy_source_downloads(
    original: sqlite3.Connection, destination: sqlite3.Connection,
    source: Path, archive: Path,
) -> None:
    """Copy source provenance, pending selections, and retained NetCDF downloads.

    Preserve source IDs while replacing the old absolute paths with relative
    paths. Resume interrupted file copies from their existing byte count; mark
    complete downloads only after their copied bytes have been flushed to disk.

    Args:
        original: Read-only catalog of the stopped Zarr archive.
        destination: COG catalog protected by its writer lock.
        source: Zarr archive root, including its staged downloads.
        archive: Separate COG archive root.
    """
    attributes = {}
    retrieved_at = {}
    for revision in original.execute("SELECT provenance_json FROM revisions WHERE status = 'published'"):
        for item in json.loads(revision["provenance_json"])["sources"]:
            if "attributes" in item:
                attributes[item["id"]] = item["attributes"]
            if item.get("downloaded_at") is not None:
                retrieved_at[item["id"]] = max(
                    retrieved_at.get(item["id"], ""), item["downloaded_at"],
                )

    for row in original.execute("SELECT * FROM source_files ORDER BY created_at, id"):
        selections = original.execute(
            "SELECT start_time, end_time FROM source_selections WHERE source_id = ?",
            (row["id"],),
        ).fetchall()
        relative = PurePosixPath("staging", "ncar", row["id"], PurePosixPath(row["object_key"]).name)
        stored = destination.execute("SELECT * FROM source_files WHERE id = ?", (row["id"],)).fetchone()
        if stored is None:
            # Files without outstanding selections are no longer needed by the
            # new archive, even if old source cleanup had not finished.
            removed = row["removed_at"] if selections else row["removed_at"] or datetime.now(UTC).isoformat()
            with destination:
                destination.execute(
                    """INSERT INTO source_files
                       (id, object_key, variable, size, etag, last_modified,
                        valid_start, valid_end, path, created_at, downloaded_at, retrieved_at,
                        removed_at, attributes_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)""",
                    (
                        row["id"], row["object_key"], row["variable"], row["size"],
                        row["etag"], row["last_modified"], row["valid_start"],
                        row["valid_end"], relative.as_posix(), row["created_at"],
                        row["downloaded_at"] or retrieved_at.get(row["id"]),
                        removed, json.dumps(attributes[row["id"]]) if row["id"] in attributes else None,
                    ),
                )
                destination.executemany(
                    "INSERT INTO source_selections (source_id, start_time, end_time) VALUES (?, ?, ?)",
                    [(row["id"], selected["start_time"], selected["end_time"]) for selected in selections],
                )
            stored = destination.execute("SELECT * FROM source_files WHERE id = ?", (row["id"],)).fetchone()
        if not selections or stored["downloaded_at"] is not None or stored["removed_at"] is not None:
            continue
        old_path = source / relative
        if row["downloaded_at"] is None and not old_path.exists():
            continue  # Registered downloads may not have started yet.
        new_path = archive / stored["path"]
        new_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying retained source {row['object_key']}.", flush=True)
        with old_path.open("rb") as incoming, new_path.open("ab") as outgoing:
            incoming.seek(outgoing.tell())
            with tqdm.wrapattr(
                incoming, "read", total=os.fstat(incoming.fileno()).st_size,
                initial=incoming.tell(), desc=old_path.name, unit="B", unit_scale=True,
            ) as progress:
                shutil.copyfileobj(progress, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        with destination:
            destination.execute(
                "UPDATE source_files SET downloaded_at = ? WHERE id = ?",
                (row["downloaded_at"], row["id"]),
            )


def convert_zarr_archive(source: Path, archive: Path) -> int:
    """Convert published Zarr observations and copy pending download progress.

    Hold both writer locks throughout conversion. Read only current published
    coverage, skip observations already in the destination catalog, and leave
    the original archive unchanged. Decode one variable/month at a time using
    its spatial shards so month-long compressed chunks are read only once.

    Args:
        source: Existing Zarr archive whose pipeline has been stopped.
        archive: Separate destination COG archive, new or partially converted.

    Returns:
        Number of newly published GeoTIFF observations.

    Raises:
        ValueError: The roots overlap or the original needs publication recovery.
        portalocker.exceptions.LockException: Either archive has an active writer.
        Exception: Source reads, conversion, and publication errors propagate.
    """
    source = source.absolute()
    archive = archive.absolute()
    if source.resolve().is_relative_to(archive.resolve()) or archive.resolve().is_relative_to(source.resolve()):
        raise ValueError("Use separate, non-overlapping source and destination archive directories.")
    location = (source / "catalog.sqlite").as_posix()
    # Empty URI authority also supports UNC paths: file:////server/share/...
    uri = "file:" + ("//" if location.startswith("//") else "") + quote(location, safe="/:") + "?mode=ro"
    converted = 0
    with portalocker.Lock(source / ".writer.lock", "r", timeout=0):
        with closing(sqlite3.connect(uri, uri=True)) as original:
            original.row_factory = sqlite3.Row
            if original.execute("SELECT 1 FROM revisions WHERE status = 'ready' LIMIT 1").fetchone():
                raise ValueError(
                    "The Zarr archive has an interrupted publication. Run the old pipeline "
                    "once to recover it, stop that writer, and retry conversion."
                )
            with archive_session(archive) as destination:
                copy_source_downloads(original, destination, source, archive)
                for revision in original.execute(
                    """SELECT r.* FROM months m JOIN revisions r ON r.id = m.current_revision_id
                       ORDER BY m.year, m.month"""
                ):
                    report = json.loads(revision["report_json"])
                    fields = original.execute(
                        """SELECT c.*, s.source_file_id, s.source_variable FROM coverage c
                           JOIN field_sources s USING (revision_id, variable, valid_time)
                           WHERE c.revision_id = ? ORDER BY c.variable, c.valid_time""",
                        (revision["id"],),
                    ).fetchall()
                    pending = [
                        dict(field) for field in fields
                        if destination.execute(
                            "SELECT 1 FROM observations WHERE variable = ? AND valid_time = ?",
                            (field["variable"], field["valid_time"]),
                        ).fetchone() is None
                    ]
                    for field in pending:
                        for key in ("reference_time", "start_step", "end_step"):
                            if field[key] == -(2**63):
                                field[key] = None
                    print(
                        f"{revision['year']:04d}-{revision['month']:02d}: "
                        f"{len(pending)} observations to convert; {len(fields) - len(pending)} already published.",
                        flush=True,
                    )
                    if not pending:
                        continue
                    group = zarr.open_group(source / revision["final_path"], mode="r")
                    time_indices = {int(value): index for index, value in enumerate(group["time"][:])}
                    scratch_root = archive / "staging" / "conversion"
                    scratch_root.mkdir(parents=True, exist_ok=True)
                    for variable, attributes in report["variables"].items():
                        selected = [field for field in pending if field["variable"] == variable]
                        if not selected:
                            continue
                        array = group[variable]
                        rows = np.array([time_indices[field["valid_time"]] for field in selected], dtype=np.intp)
                        shape = (len(selected), *array.shape[1:])
                        spatial_shards = (array.shards or array.chunks)[-2:]
                        print(f"Reading {variable}: {len(selected)} covered UTC hours.", flush=True)
                        with ExitStack() as resources:
                            scratch = scratch_root / "values.bin"
                            resources.callback(scratch.unlink, missing_ok=True)
                            raw = resources.enter_context(scratch.open("w+b"))
                            raw.truncate(int(np.prod(shape)) * array.dtype.itemsize)
                            mapping = resources.enter_context(mmap.mmap(raw.fileno(), 0))
                            values = np.ndarray(shape, dtype=array.dtype, buffer=mapping)
                            for latitude in range(0, shape[1], spatial_shards[0]):
                                lat = slice(latitude, min(latitude + spatial_shards[0], shape[1]))
                                for longitude in range(0, shape[2], spatial_shards[1]):
                                    lon = slice(longitude, min(longitude + spatial_shards[1], shape[2]))
                                    values[:, lat, lon] = array[:, lat, lon][rows]
                            for index, field in enumerate(selected):
                                publication, staged = prepare_observation(destination, archive, field, attributes)
                                raster = write_observation_cog(values[index], field, attributes, report["grid"], staged)
                                publish_observation(destination, archive, publication, raster)
                                converted += 1
                                if (index + 1) % 24 == 0 or index + 1 == len(selected):
                                    print(f"Published {variable}: {index + 1}/{len(selected)} observations.", flush=True)
    print(f"Conversion complete: {converted} new observations in {archive}.", flush=True)
    return converted


def main() -> None:
    """Parse archive paths and run the one-time Zarr-to-COG conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Stopped source Zarr archive; remains unchanged.")
    parser.add_argument("--archive", required=True, type=Path, help="Separate new or partially converted COG archive.")
    args = parser.parse_args()
    convert_zarr_archive(args.source, args.archive)


if __name__ == "__main__":
    main()
