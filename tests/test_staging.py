"""Verify staged request reuse and conversion checkpoint integrity."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from era5_pull_pipeline import staging


def source_batch(path: Path, request: dict[str, Any]) -> dict[str, Any]:
    """Write source bytes and their exact request and content digests.

    Args:
        path: Completed batch directory to create.
        request: Canonical source selection.

    Returns:
        Download metadata matching the saved source bytes.
    """
    path.mkdir(parents=True)
    data = b"source bytes verified independently of format decoding"
    (path / "data.grib").write_bytes(data)
    identity = {"dataset": staging.DATASET, "request": request}
    metadata = {
        **identity,
        "request_sha256": hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


@pytest.fixture
def request_selection() -> dict[str, Any]:
    """Describe a small global ERA5 selection with room for expanded availability."""
    return {
        "product_type": ["reanalysis"], "variable": ["2m_temperature"],
        "year": ["2000"], "month": ["01"], "day": ["01"],
        "time": ["00:00", "01:00", "02:00", "03:00"],
        "grid": [0.25, 0.25], "data_format": "grib", "download_format": "unarchived",
    }


def test_find_downloads_reuses_disjoint_subsets_without_hashing_other_sources(
    tmp_path: Path,
    request_selection: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse earlier hours after availability expands and skip unrelated bytes."""
    month = tmp_path / "2000-01"
    early = {**request_selection, "time": ["00:00", "01:00"]}
    first = month / "00-early" / "complete"
    first_metadata = source_batch(first, early)
    overlap = month / "01-overlap" / "complete"
    source_batch(overlap, {**request_selection, "time": ["01:00", "02:00"]})
    (overlap / "data.grib").write_bytes(b"not verified because this overlaps")
    later = month / "02-later" / "complete"
    later_metadata = source_batch(later, {**request_selection, "time": ["03:00"]})
    for name, change in (
        ("variable", {"variable": ["total_precipitation"]}),
        ("month", {"month": ["02"]}),
        ("grid", {"grid": [0.5, 0.5]}),
    ):
        unrelated = month / f"03-{name}" / "complete"
        source_batch(unrelated, {**request_selection, **change})
        (unrelated / "data.grib").write_bytes(b"unrelated corrupt bytes")
    removed = month / "04-removed" / "complete"
    source_batch(removed, request_selection)
    (removed / "data.grib").unlink()
    partial = month / "05-partial" / "download.partial"
    source_batch(partial, request_selection)
    (partial / "data.grib").write_bytes(b"incomplete")
    verified = []
    original_verify = staging.verify_source

    def record_verification(batch: Path, metadata: dict[str, Any]) -> None:
        """Record actual source verification at the file boundary."""
        verified.append(batch)
        original_verify(batch, metadata)

    monkeypatch.setattr(staging, "verify_source", record_verification)
    found = staging.find_downloads(month, [request_selection])
    assert found == [(first, first_metadata), (later, later_metadata)]
    assert verified == [first, later]


@pytest.mark.parametrize("corruption", ["request", "bytes"])
def test_matching_download_requires_valid_identity_and_bytes(
    tmp_path: Path, request_selection: dict[str, Any], corruption: str,
) -> None:
    """Reject a reusable selection whose request identity or source bytes changed."""
    batch = tmp_path / "selection" / "complete"
    metadata = source_batch(batch, request_selection)
    if corruption == "request":
        metadata["request_sha256"] = "0" * 64
        (batch / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    else:
        (batch / "data.grib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity|checksum"):
        staging.find_downloads(tmp_path, [request_selection])


@pytest.mark.parametrize("corruption", ["report", "store"])
def test_checkpoint_rebuilds_changed_report_or_store_and_removes_owned_scratch(
    tmp_path: Path,
    request_selection: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    """Keep intact checkpoints reusable and rebuild corrupted progress safely."""
    batch = tmp_path / "complete"
    metadata = source_batch(batch, request_selection)
    expected_report = {"fields": [{"variable": "temperature", "valid_time": 946684800}]}
    conversions = 0

    def write_validated_store(
        grib: Path, provenance: dict[str, Any], output: Path, chunks: tuple[int, int],
    ) -> dict[str, Any]:
        """Stand in for separately tested decoding while writing real Zarr bytes."""
        nonlocal conversions
        conversions += 1
        assert not list(batch.glob(".decoded-*"))
        group = zarr.open_group(output, mode="w-", zarr_format=3)
        group.create_array("temperature", data=np.array([280.0, 281.0]))
        return deepcopy(expected_report)

    monkeypatch.setattr(staging, "convert_batch", write_validated_store)
    output, report, reused = staging.conversion_checkpoint(batch, metadata, (2, 2))
    assert not reused and report == expected_report
    assert staging.conversion_checkpoint(batch, metadata, (2, 2)) == (output, expected_report, True)
    assert conversions == 1
    manifest = batch / "checkpoint.json"
    if corruption == "report":
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        saved["report"]["fields"][0]["valid_time"] += 3600
        manifest.write_text(json.dumps(saved), encoding="utf-8")
    else:
        zarr.open_group(output, mode="r+")["temperature"][0] = 999.0
    abandoned = batch / ".decoded-killed-process"
    abandoned.mkdir()
    (abandoned / "raw.bin").write_bytes(b"abandoned decoded values")
    sibling = tmp_path / ".decoded-another-batch"
    sibling.mkdir()
    notes = batch / "notes"
    notes.mkdir()
    output, report, reused = staging.conversion_checkpoint(batch, metadata, (2, 2))
    assert not reused and report == expected_report
    assert conversions == 2
    assert not abandoned.exists()
    assert sibling.exists() and notes.exists()
    assert (batch / "data.grib").exists()
    assert zarr.open_group(output, mode="r")["temperature"][0] == 280.0
