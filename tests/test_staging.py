"""Exercise cataloged staging and interruption recovery with real SQLite and Zarr."""

import json
from pathlib import Path
import sqlite3
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import zarr

from era5_pull_pipeline import catalog, staging
from era5_pull_pipeline.cds import DATASET


@pytest.fixture
def request_selection() -> dict[str, Any]:
    """Describe a global selection whose available hours can grow between runs."""
    return {
        "product_type": ["reanalysis"], "variable": ["2m_temperature"],
        "year": ["2000"], "month": ["01"], "day": ["01"],
        "time": ["00:00", "01:00", "02:00", "03:00"],
        "grid": [0.25, 0.25], "data_format": "grib", "download_format": "unarchived",
    }


def completed_download(db: sqlite3.Connection, batch: sqlite3.Row) -> dict[str, Any]:
    """Record a small completed source for the conversion boundary tests.

    Args:
        db: Open archive catalog.
        batch: Registered request and destination path.

    Returns:
        Source metadata recorded in SQLite and the completed batch directory.
    """
    path = Path(batch["batch_path"])
    path.mkdir(parents=True)
    (path / "data.grib").write_bytes(b"source bytes")
    metadata = {
        "dataset": DATASET, "request": json.loads(batch["request_json"]),
        "request_sha256": "source-request",
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with db:
        db.execute(
            "UPDATE staged_batches SET metadata_json = ? WHERE id = ?",
            (json.dumps(metadata), batch["id"]),
        )
    return metadata


@pytest.fixture
def writer(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Replace decoding with a small real Zarr write and observable call count."""
    def write_store(
        grib: Path, metadata: dict[str, Any], output: Path, chunks: tuple[int, int],
    ) -> dict[str, Any]:
        """Write a completed store for the separately tested GRIB decoder.

        Args:
            grib: Retained source file.
            metadata: Download provenance.
            output: Unpublished conversion destination.
            chunks: Requested spatial chunk shape.

        Returns:
            The conversion report to record in the catalog.
        """
        assert grib.read_bytes() == b"source bytes"
        group = zarr.open_group(output, mode="w-", zarr_format=3)
        group.attrs["provenance"] = metadata
        group.create_array("temperature", data=np.array([280.0, 281.0]))
        return {
            "layout": {"spatial_chunks": list(chunks)},
            "fields": [{"variable": "temperature", "valid_time": 946684800}],
        }

    convert = Mock(side_effect=write_store)
    monkeypatch.setattr(staging, "convert_batch", convert)
    return convert


def test_expanded_selection_reuses_disjoint_cataloged_pending_requests(
    tmp_path: Path, request_selection: dict[str, Any],
) -> None:
    """Retain submitted selections across regrouping before any files are created."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        first = staging.prepare_batches(db, archive, [{
            **request_selection, "time": ["00:00", "01:00"],
        }])[0]
        overlap = staging.prepare_batches(db, archive, [{
            **request_selection, "time": ["01:00", "02:00"],
        }])[0]
        later = staging.prepare_batches(db, archive, [{
            **request_selection, "time": ["03:00"],
        }])[0]
        other_month = staging.prepare_batches(db, archive, [{
            **request_selection, "month": ["02"],
        }])[0]
        planned = staging.prepare_batches(db, archive, [request_selection])
        ids = [row["id"] for row in planned]
        assert ids[:2] == [first["id"], later["id"]]
        assert overlap["id"] not in ids and other_month["id"] not in ids
        hours = [hour for row in planned for hour in json.loads(row["request_json"])["time"]]
        assert sorted(hours) == request_selection["time"]
        assert all(row["metadata_json"] is None for row in planned)
        assert all(not Path(row["batch_path"]).exists() for row in planned)
    with catalog.archive_session(archive) as db:
        regrouped = staging.prepare_batches(db, archive, [
            {**request_selection, "time": ["00:00"]},
            {**request_selection, "time": ["01:00", "02:00", "03:00"]},
        ])
        assert [row["id"] for row in regrouped] == ids


def test_unregistered_directory_is_not_discovered(
    tmp_path: Path, request_selection: dict[str, Any],
) -> None:
    """Use registered destinations even when matching metadata exists elsewhere."""
    archive = tmp_path / "archive"
    unrelated = archive / "staging" / "batches" / "2000-01" / "unregistered"
    unrelated.mkdir(parents=True)
    (unrelated / "metadata.json").write_text(
        json.dumps({"dataset": DATASET, "request": request_selection}), encoding="utf-8",
    )
    (unrelated / "data.grib").write_bytes(b"unregistered source")
    with catalog.archive_session(archive) as db:
        rows = staging.prepare_batches(db, archive, [request_selection])
        assert len(rows) == 1
        assert Path(rows[0]["batch_path"]) != unrelated
        assert rows[0]["metadata_json"] is None
        assert db.execute("SELECT COUNT(*) FROM staged_batches").fetchone()[0] == 1
    assert (unrelated / "data.grib").read_bytes() == b"unregistered source"


def test_completed_conversion_is_reused_from_catalog(
    tmp_path: Path, request_selection: dict[str, Any], writer: Mock,
) -> None:
    """Reuse a recorded conversion after reopening the archive without decoding."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        row = staging.prepare_batches(db, archive, [request_selection])[0]
        metadata = completed_download(db, row)
        output, report, reused = staging.conversion_checkpoint(db, row, metadata, (2, 2))
        assert not reused
        recorded = db.execute("SELECT report_json FROM staged_batches WHERE id = ?", (row["id"],)).fetchone()[0]
        assert json.loads(recorded) == report
    with catalog.archive_session(archive) as db:
        row = staging.prepare_batches(db, archive, [request_selection])[0]
        assert staging.conversion_checkpoint(db, row, metadata, (2, 2)) == (output, report, True)
    assert writer.call_count == 1
    np.testing.assert_array_equal(zarr.open_group(output, mode="r")["temperature"][:], [280.0, 281.0])
    assert (output.parent / "data.grib").exists()


def test_interrupted_conversion_rebuilds_only_owned_incomplete_work(
    tmp_path: Path, request_selection: dict[str, Any], writer: Mock,
) -> None:
    """Leave no completion record after interruption and retain the source for retry."""
    archive = tmp_path / "archive"
    complete_write = writer.side_effect
    with catalog.archive_session(archive) as db:
        row = staging.prepare_batches(db, archive, [request_selection])[0]
        metadata = completed_download(db, row)
        path = Path(row["batch_path"])
        scratch = path / ".decoded-interrupted"
        outside = path.parent / ".decoded-another-batch"
        outside.mkdir()

        def interrupted(
            grib: Path, provenance: dict[str, Any], output: Path, chunks: tuple[int, int],
        ) -> dict[str, Any]:
            """Stop after partial output and decoded scratch have reached disk."""
            output.mkdir()
            (output / "unfinished").write_bytes(b"partial Zarr write")
            scratch.mkdir()
            (scratch / "raw.bin").write_bytes(b"decoded scratch")
            raise KeyboardInterrupt

        writer.side_effect = interrupted
        with pytest.raises(KeyboardInterrupt):
            staging.conversion_checkpoint(db, row, metadata, (2, 2))
        assert db.execute("SELECT report_json FROM staged_batches WHERE id = ?", (row["id"],)).fetchone()[0] is None
        assert (path / "data.grib").exists()
    writer.side_effect = complete_write
    with catalog.archive_session(archive) as db:
        row = staging.prepare_batches(db, archive, [request_selection])[0]
        output, _, reused = staging.conversion_checkpoint(db, row, metadata, (2, 2))
        assert not reused
    assert writer.call_count == 2
    assert not (output / "unfinished").exists()
    assert not scratch.exists()
    assert outside.exists()
    assert (path / "data.grib").exists()


def test_changed_chunk_setting_rebuilds_recorded_conversion(
    tmp_path: Path, request_selection: dict[str, Any], writer: Mock,
) -> None:
    """Apply a changed input layout to an unpublished checkpoint."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        row = staging.prepare_batches(db, archive, [request_selection])[0]
        metadata = completed_download(db, row)
        output, _, _ = staging.conversion_checkpoint(db, row, metadata, (2, 2))
        row = staging.prepare_batches(db, archive, [request_selection])[0]
        rebuilt, report, reused = staging.conversion_checkpoint(db, row, metadata, (4, 4))
        assert rebuilt == output
        assert not reused
        assert report["layout"]["spatial_chunks"] == [4, 4]
        row = staging.prepare_batches(db, archive, [request_selection])[0]
        assert staging.conversion_checkpoint(db, row, metadata, (4, 4)) == (output, report, True)
    assert writer.call_count == 2
    assert (output.parent / "data.grib").exists()
