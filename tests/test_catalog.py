"""Exercise publication recovery and locking with real SQLite and Zarr stores."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any
from uuid import uuid4
from unittest.mock import Mock

import numpy as np
import portalocker
import pytest
import xarray as xr
import zarr

from era5_pull_pipeline import catalog


def _stage(
    db: sqlite3.Connection,
    archive: Path,
    month: int = 1,
    variable: str = "2_metre_temperature",
    hours: tuple[int, ...] = (0,),
    year: int = 2000,
    requested_variables: tuple[str, ...] | None = None,
) -> tuple[str, Path, Path, dict[str, Any]]:
    """Build a small merged-store stand-in for catalog boundary tests.

    Args:
        db: Open writer catalog.
        archive: Archive receiving the store.
        month: Calendar month for the fixture.
        variable: Variable to add.
        hours: Hours since the beginning of the month.
        year: Calendar year for the fixture.
        requested_variables: Full source request, including retained overlaps.

    Returns:
        Revision identifier, staged store, source batch, and converter report.
    """
    previous = catalog.current_month(db, archive, year, month)
    batch = archive.parent / f"batch-{uuid4().hex}"
    batch.mkdir()
    (batch / "data.grib").write_bytes(b"source kept until publication")
    cds_name, param_id = {
        "2_metre_temperature": ("2m_temperature", 167),
        "10_metre_u_wind_component": ("10m_u_component_of_wind", 165),
    }[variable]
    dates = [
        datetime(year, month, 1, tzinfo=UTC) + timedelta(hours=hour)
        for hour in hours
    ]
    metadata = {
        "dataset": "reanalysis-era5-single-levels",
        "request": {
            "product_type": ["reanalysis"],
            "variable": list(requested_variables or (cds_name,)),
            "year": [str(year)], "month": [f"{month:02d}"],
            "day": sorted({value.strftime("%d") for value in dates}),
            "time": sorted({value.strftime("%H:%M") for value in dates}),
            "grid": [0.25, 0.25], "data_format": "grib",
            "download_format": "unarchived",
        },
        "request_sha256": f"request-{year}-{month}-{variable}-{hours}",
        "sha256": "source-checksum",
    }
    revision, stage = catalog.prepare_revision(db, archive, [batch], metadata)
    timestamps = [int(value.timestamp()) for value in dates]
    report = {
        "year": year,
        "month": month,
        "grid": {"latitude_size": 2, "longitude_size": 2},
        "layout": {
            "spatial_chunks": [2, 2],
            "time_chunk": "calendar_month",
            "shard_spatial_multiple": 2,
        },
        "variables": {
            variable: {
                "paramId": param_id,
                "units": "K" if variable == "2_metre_temperature" else "m s**-1",
                "step_type": "instant",
            }
        },
        "fields": [{
            "variable": variable, "valid_time": timestamp,
            "reference_time": timestamp, "interval_start": timestamp,
            "interval_end": timestamp, "start_step": 0, "end_step": 0,
            "step_units": "s", "step_type": "instant",
            "source_request_sha256": metadata["request_sha256"],
        } for timestamp in timestamps],
        "validation": {"fields": len(hours), "roundtrip": True},
    }
    dataset = xr.Dataset(
        {variable: (
            ("time", "latitude", "longitude"),
            np.full((len(hours), 2, 2), 280.0 if variable == "2_metre_temperature" else 3.0),
        )},
        coords={
            "time": np.array(timestamps, dtype="datetime64[s]"),
            "latitude": [90.0, 89.75],
            "longitude": [0.0, 0.25],
        },
    )
    if previous is not None:
        with xr.open_zarr(previous[0], consolidated=False, chunks=None) as old:
            dataset = dataset.combine_first(old.load())
        report["variables"] = {**previous[1]["variables"], **report["variables"]}
        report["fields"] = list({
            (field["variable"], field["valid_time"]): field
            for field in report["fields"] + previous[1]["fields"]
        }.values())
        report["validation"]["fields"] = len(report["fields"])
    dataset.attrs = {"provenance": metadata}
    dataset.to_zarr(stage, zarr_format=3, consolidated=False)
    return revision, stage, batch, report


def _stage_multiple(
    db: sqlite3.Connection, archive: Path
) -> tuple[str, Path, list[Path], dict[str, Any], dict[str, Any]]:
    """Stage three independent requests while retaining a published old field.

    Args:
        db: Open writer catalog.
        archive: Archive receiving the store.

    Returns:
        Revision, staged store, source batches, report, and aggregate provenance.
    """
    first, _, _, report = _stage(db, archive)
    catalog.publish_revision(db, archive, first, report)
    report = deepcopy(catalog.current_month(db, archive, 2000, 1)[1])
    report.pop("revision_id")
    template = json.loads(db.execute(
        "SELECT provenance_json FROM revisions WHERE id = ?", (first,)
    ).fetchone()[0])
    sources = []
    batches = []
    start = datetime(2000, 1, 1, tzinfo=UTC)
    for index, (variable, cds_name, param_id, hour) in enumerate([
        ("2_metre_temperature", "2m_temperature", 167, 1),
        ("10_metre_u_wind_component", "10m_u_component_of_wind", 165, 24),
        ("2_metre_temperature", "2m_temperature", 167, 50),
    ]):
        timestamp = start + timedelta(hours=hour)
        source = deepcopy(template)
        source["request"].update({
            "variable": [cds_name], "day": [timestamp.strftime("%d")],
            "time": [timestamp.strftime("%H:%M")],
        })
        source["request_sha256"] = f"batch-request-{index}"
        batch = archive / "staging" / "batches" / source["request_sha256"]
        batch.mkdir(parents=True)
        (batch / "data.grib").write_bytes(b"verified source")
        (batch / "metadata.json").write_text(json.dumps(source), encoding="utf-8")
        (batch / "converted.zarr").mkdir()
        (batch / "converted.zarr" / "zarr.json").write_text("{}", encoding="utf-8")
        batches.append(batch)
        sources.append(source)
        valid_time = int(timestamp.timestamp())
        field = {
            "variable": variable, "valid_time": valid_time,
            "reference_time": valid_time, "interval_start": valid_time,
            "interval_end": valid_time, "start_step": 0, "end_step": 0,
            "step_units": "s", "step_type": "instant",
            "source_request_sha256": source["request_sha256"],
        }
        report["fields"].append(field)
        report["variables"][variable] = {
            "paramId": param_id, "units": "K" if param_id == 167 else "m s**-1",
            "step_type": "instant",
        }
        source_report = {
            **report,
            "variables": {variable: report["variables"][variable]},
            "fields": [field],
            "validation": {"fields": 1, "roundtrip": True},
        }
        with db:
            db.execute(
                """INSERT INTO staged_batches
                   (id, year, month, request_json, batch_path, metadata_json, report_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (source["request_sha256"], 2000, 1, json.dumps(source["request"]),
                 str(batch.resolve()), json.dumps(source), json.dumps(source_report)),
            )
    metadata = {
        "dataset": template["dataset"], "request_sha256": "monthly-request",
        "sources": sources,
    }
    revision, stage = catalog.prepare_revision(db, archive, batches, metadata)
    times = sorted({field["valid_time"] for field in report["fields"]})
    values = {name: np.full((len(times), 2, 2), np.nan) for name in report["variables"]}
    for field in report["fields"]:
        values[field["variable"]][times.index(field["valid_time"])] = (
            280.0 if field["variable"] == "2_metre_temperature" else 3.0
        )
    xr.Dataset(
        {name: (("time", "latitude", "longitude"), data) for name, data in values.items()},
        coords={"time": np.array(times, dtype="datetime64[s]"),
                "latitude": [90.0, 89.75], "longitude": [0.0, 0.25]},
        attrs={"provenance": metadata},
    ).to_zarr(stage, zarr_format=3, consolidated=False)
    report["validation"]["fields"] = len(report["fields"])
    return revision, stage, batches, report, metadata


def test_publication_records_coverage_then_removes_source(tmp_path: Path) -> None:
    """Publish exact field metadata and retain provenance after GRIB cleanup."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, stage, batch, report = _stage(db, archive)
        (batch / "converted.zarr").mkdir()
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0
        assert (batch / "data.grib").exists()
        final = catalog.publish_revision(db, archive, revision, report)
        assert final == archive / "2000" / "01" / "era5.zarr"
        assert not stage.exists()
        assert not (batch / "data.grib").exists()
        assert (batch / "converted.zarr").is_dir()
        row = db.execute("SELECT * FROM revisions WHERE id = ?", (revision,)).fetchone()
        assert row["status"] == "published"
        assert row["grib_removed"] == 1
        assert json.loads(row["provenance_json"])["sha256"] == "source-checksum"
        assert json.loads(row["report_json"]) == report
        field = dict(db.execute("SELECT * FROM coverage").fetchone())
        assert field == {
            "revision_id": revision, "source_revision_id": revision,
            "source_request_sha256": row["request_sha256"], **report["fields"][0]
        }
        assert db.execute("PRAGMA synchronous").fetchone()[0] == 2
        current = catalog.current_month(db, archive, 2000, 1)
        assert current[0] == final
        assert current[1]["revision_id"] == revision
        assert current[1]["fields"][0]["source_revision_id"] == revision
        assert catalog.missing_fields(
            db, 2000, 1, ["2m_temperature"], [report["fields"][0]["valid_time"]]
        ) == {}
        with catalog.open_month(archive, 2000, 1) as dataset:
            np.testing.assert_array_equal(dataset["2_metre_temperature"].values, 280.0)


def test_replacement_keeps_history_and_field_origins(tmp_path: Path) -> None:
    """Move the current pointer while preserving old provenance and values."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        first, _, _, report = _stage(db, archive)
        final = catalog.publish_revision(db, archive, first, report)
        other, _, _, other_report = _stage(db, archive, month=2)
        catalog.publish_revision(db, archive, other, other_report)
        second, stage, batch, report = _stage(db, archive, variable="10_metre_u_wind_component")
        assert catalog.current_month(db, archive, 2000, 1)[1]["revision_id"] == first
        catalog.publish_revision(db, archive, second, report)
        assert not stage.exists()
        assert not (batch / "data.grib").exists()
        assert db.execute("SELECT COUNT(*) FROM revisions WHERE status = 'published'").fetchone()[0] == 3
        assert db.execute("SELECT parent_revision_id FROM revisions WHERE id = ?", (second,)).fetchone()[0] == first
        assert db.execute("SELECT backup_path FROM revisions WHERE id = ?", (second,)).fetchone()[0] is None
        assert db.execute("SELECT COUNT(*) FROM coverage WHERE revision_id = ?", (first,)).fetchone()[0] == 1
        current = catalog.current_month(db, archive, 2000, 1)
        assert current[0] == final
        assert current[1]["revision_id"] == second
        assert {field["variable"]: field["source_revision_id"] for field in current[1]["fields"]} == {
            "2_metre_temperature": first, "10_metre_u_wind_component": second
        }
        assert catalog.missing_fields(
            db, 2000, 1, ["2m_temperature", "10m_u_component_of_wind"],
            [report["fields"][0]["valid_time"]],
        ) == {}
        assert catalog.current_month(db, archive, 2000, 2)[1]["revision_id"] == other
    with catalog.open_month(archive, 2000, 1) as dataset:
        np.testing.assert_array_equal(dataset["2_metre_temperature"].values, 280.0)
        np.testing.assert_array_equal(dataset["10_metre_u_wind_component"].values, 3.0)


def test_multiple_sources_preserve_exact_coverage_and_origins(tmp_path: Path) -> None:
    """Publish independent source rectangles without claiming their Cartesian union."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, _, batches, report, metadata = _stage_multiple(db, archive)
        original = report["fields"][0]
        assert db.execute("SELECT COUNT(*) FROM staged_batches").fetchone()[0] == len(batches)
        catalog.publish_revision(db, archive, revision, report)
        row = db.execute("SELECT * FROM revisions WHERE id = ?", (revision,)).fetchone()
        assert json.loads(row["batch_paths_json"]) == [str(batch.resolve()) for batch in batches]
        assert json.loads(row["provenance_json"]) == metadata
        assert row["grib_removed"] == 1
        assert db.execute("SELECT COUNT(*) FROM staged_batches").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM revisions WHERE status = 'published'").fetchone()[0] == 2
        current = catalog.current_month(db, archive, 2000, 1)[1]
        fields = {(field["variable"], field["valid_time"]): field for field in current["fields"]}
        assert fields[(original["variable"], original["valid_time"])] == original
        for field in report["fields"][1:]:
            assert fields[(field["variable"], field["valid_time"])] == {
                **field, "source_revision_id": revision,
            }
        start = int(datetime(2000, 1, 1, tzinfo=UTC).timestamp())
        requested = [start + hour * 3600 for hour in (0, 1, 2, 24, 25, 26, 48, 49, 50)]
        assert catalog.missing_fields(
            db, 2000, 1, ["2m_temperature", "10m_u_component_of_wind"], requested
        ) == {
            "2m_temperature": [hour for hour in requested if hour not in (start, start + 3600, start + 50 * 3600)],
            "10m_u_component_of_wind": [hour for hour in requested if hour != start + 24 * 3600],
        }
    for batch in batches:
        assert (batch / "metadata.json").exists()
        assert not (batch / "data.grib").exists()
        assert not (batch / "converted.zarr").exists()


@pytest.mark.parametrize("failure", ["grib", "converted_store"])
def test_multiple_source_cleanup_resumes_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Published revisions own pending cleanup without leaving reusable batches."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, _, batches, report, _ = _stage_multiple(db, archive)
        assert db.execute("SELECT COUNT(*) FROM staged_batches").fetchone()[0] == len(batches)
        remove = Path.unlink if failure == "grib" else catalog.shutil.rmtree
        failed_path = batches[1] / ("data.grib" if failure == "grib" else "converted.zarr")

        def fail_second_source(path: Path, *args: Any, **kwargs: Any) -> None:
            """Leave one contributing source pending after earlier cleanup succeeds."""
            if Path(path) == failed_path:
                raise PermissionError("source is open")
            remove(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            if failure == "grib":
                patch.setattr(Path, "unlink", fail_second_source)
            else:
                patch.setattr(catalog.shutil, "rmtree", fail_second_source)
            with pytest.warns(RuntimeWarning, match="cleanup remains pending"):
                catalog.publish_revision(db, archive, revision, report)
            published = db.execute("SELECT * FROM revisions WHERE id = ?", (revision,)).fetchone()
            assert published["status"] == "published"
            assert published["grib_removed"] == 0
            assert json.loads(published["batch_paths_json"]) == [str(batch.resolve()) for batch in batches]
            assert catalog.current_month(db, archive, 2000, 1)[1]["revision_id"] == revision
            assert db.execute("SELECT COUNT(*) FROM staged_batches").fetchone()[0] == 0
            assert not (batches[0] / "data.grib").exists()
            if failure == "converted_store":
                assert all(not (batch / "data.grib").exists() for batch in batches)
                assert not (batches[0] / "converted.zarr").exists()
                assert (batches[1] / "converted.zarr").exists()
            else:
                assert (batches[1] / "data.grib").exists()
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT grib_removed FROM revisions WHERE id = ?", (revision,)).fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM staged_batches").fetchone()[0] == 0
    for batch in batches:
        assert (batch / "metadata.json").exists()
        assert not (batch / "data.grib").exists()
        assert not (batch / "converted.zarr").exists()


def test_interrupted_multiple_source_publication_retains_every_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recover the whole monthly publication before deleting any contributing source."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, _, batches, report, _ = _stage_multiple(db, archive)
        parent = db.execute("SELECT parent_revision_id FROM revisions WHERE id = ?", (revision,)).fetchone()[0]
        with monkeypatch.context() as patch:
            patch.setattr(Path, "rename", Mock(side_effect=SystemExit("interrupted")))
            with pytest.raises(SystemExit, match="interrupted"):
                catalog.publish_revision(db, archive, revision, report)
        assert db.execute("SELECT current_revision_id FROM months").fetchone()[0] == parent
        assert db.execute("SELECT COUNT(*) FROM coverage WHERE revision_id = ?", (revision,)).fetchone()[0] == 0
        assert all((batch / "data.grib").exists() for batch in batches)
        assert all((batch / "converted.zarr").exists() for batch in batches)
        assert db.execute("SELECT COUNT(*) FROM staged_batches").fetchone()[0] == len(batches)
    with catalog.archive_session(archive) as db:
        assert catalog.current_month(db, archive, 2000, 1)[1]["revision_id"] == revision
        assert db.execute("SELECT COUNT(*) FROM staged_batches").fetchone()[0] == 0
    assert all(not (batch / "data.grib").exists() for batch in batches)
    assert all(not (batch / "converted.zarr").exists() for batch in batches)


def test_missing_fields_uses_independent_coverage_and_negative_timestamps(tmp_path: Path) -> None:
    """Find interior gaps and extensions independently for each variable."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, _, _, report = _stage(db, archive, hours=(0, 2), year=1950)
        catalog.publish_revision(db, archive, revision, report)
        revision, _, _, report = _stage(
            db, archive, variable="10_metre_u_wind_component", hours=(0,), year=1950
        )
        catalog.publish_revision(db, archive, revision, report)
        start = int(datetime(1950, 1, 1, tzinfo=UTC).timestamp())
        hours = [start + 3600 * hour for hour in range(4)]
        assert catalog.missing_fields(db, 1950, 1, ["2m_temperature", "10m_u_component_of_wind"], hours) == {
            "2m_temperature": [hours[1], hours[3]],
            "10m_u_component_of_wind": hours[1:],
        }
        assert catalog.missing_fields(db, 1950, 2, ["2m_temperature"], hours) == {
            "2m_temperature": hours
        }
        assert catalog.missing_fields(db, 1950, 1, ["2m_temperature", "10m_u_component_of_wind"], hours[:1]) == {}


def test_overlapping_request_preserves_original_sources_and_completed_selection(
    tmp_path: Path,
) -> None:
    """Treat a complete request as stored even when some fields retain old sources."""
    archive = tmp_path / "archive"
    names = ["2m_temperature", "10m_u_component_of_wind"]
    with catalog.archive_session(archive) as db:
        first, _, _, report = _stage(db, archive)
        catalog.publish_revision(db, archive, first, report)
        timestamp = report["fields"][0]["valid_time"]
        second, _, _, report = _stage(
            db, archive, variable="10_metre_u_wind_component",
            requested_variables=tuple(names),
        )
        assert catalog.missing_fields(db, 2000, 1, names, [timestamp]) == {
            "10m_u_component_of_wind": [timestamp]
        }
        catalog.publish_revision(db, archive, second, report)
        current = catalog.current_month(db, archive, 2000, 1)[1]
        assert {field["variable"]: field["source_revision_id"] for field in current["fields"]} == {
            "2_metre_temperature": first, "10_metre_u_wind_component": second
        }
        assert catalog.missing_fields(db, 2000, 1, names, [timestamp]) == {}
        third, _, _, report = _stage(db, archive, hours=(0, 1))
        catalog.publish_revision(db, archive, third, report)
        assert catalog.missing_fields(db, 2000, 1, names, [timestamp, timestamp + 3600]) == {
            "10m_u_component_of_wind": [timestamp + 3600]
        }
        current = catalog.current_month(db, archive, 2000, 1)[1]
        temperature = [field for field in current["fields"] if field["variable"] == "2_metre_temperature"]
        assert [field["source_revision_id"] for field in temperature] == [first, third]


def test_preparing_revision_is_never_readable(tmp_path: Path) -> None:
    """Leave incomplete conversion attempts without published coverage."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, stage, batch, report = _stage(db, archive)
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (revision,)).fetchone()[0] == "preparing"
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0
        assert catalog.current_month(db, archive, 2000, 1) is None
        timestamp = report["fields"][0]["valid_time"]
        assert catalog.missing_fields(db, 2000, 1, ["2m_temperature"], [timestamp]) == {
            "2m_temperature": [timestamp]
        }
    with pytest.raises(FileNotFoundError, match="No published"):
        with catalog.open_month(archive, 2000, 1):
            pytest.fail("Preparing data must not be exposed.")
    assert stage.exists()
    assert (batch / "data.grib").exists()


def test_restarting_assembly_removes_only_abandoned_same_month_store(tmp_path: Path) -> None:
    """Bound failed monthly assemblies while retaining reusable source checkpoints."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        abandoned, old_stage, batches, _, metadata = _stage_multiple(db, archive)
        parent = catalog.current_month(db, archive, 2000, 1)[1]["revision_id"]
        other, other_stage, _, _ = _stage(db, archive, month=2)
        revision, stage = catalog.prepare_revision(db, archive, batches, metadata)
        assert revision != abandoned
        assert not old_stage.exists()
        assert not stage.exists()
        assert db.execute("SELECT id FROM revisions WHERE id = ?", (abandoned,)).fetchone() is None
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (revision,)).fetchone()[0] == "preparing"
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (other,)).fetchone()[0] == "preparing"
        assert other_stage.exists()
        assert catalog.current_month(db, archive, 2000, 1)[1]["revision_id"] == parent
        assert all((batch / "data.grib").exists() for batch in batches)
        assert all((batch / "converted.zarr").exists() for batch in batches)
        assert db.execute("SELECT COUNT(*) FROM staged_batches").fetchone()[0] == len(batches)


def test_abandoned_store_cleanup_rejects_path_outside_revision_directory(tmp_path: Path) -> None:
    """Never recursively remove an unowned directory recorded as an assembly."""
    archive = tmp_path / "archive"
    outside = tmp_path / "other-data"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with catalog.archive_session(archive) as db:
        revision, _, batch, _ = _stage(db, archive)
        metadata = json.loads(db.execute(
            "SELECT provenance_json FROM revisions WHERE id = ?", (revision,)
        ).fetchone()[0])
        with db:
            db.execute("UPDATE revisions SET staged_path = ? WHERE id = ?", (str(outside), revision))
        with pytest.raises(ValueError, match="outside the archive revision directory"):
            catalog.prepare_revision(db, archive, [batch], metadata)
        assert marker.read_text(encoding="utf-8") == "keep"
        assert db.execute("SELECT id FROM revisions WHERE id = ?", (revision,)).fetchone() is not None


@pytest.mark.parametrize("failure_at", [1, 2])
def test_rename_failure_restores_old_month(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: int
) -> None:
    """Keep the previous month readable after either publication rename fails."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        first, _, _, report = _stage(db, archive)
        catalog.publish_revision(db, archive, first, report)
        second, stage, batch, report = _stage(db, archive, variable="10_metre_u_wind_component")
        rename = Path.rename
        calls = 0

        def fail_once(source: Path, destination: Path) -> Path:
            """Fail one rename while allowing the recovery moves."""
            nonlocal calls
            calls += 1
            if calls == failure_at:
                raise OSError("rename failed")
            return rename(source, destination)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "rename", fail_once)
            with pytest.raises(OSError, match="rename failed"):
                catalog.publish_revision(db, archive, second, report)
        assert stage.exists()
        assert (batch / "data.grib").exists()
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (second,)).fetchone()[0] == "preparing"
        assert db.execute("SELECT COUNT(*) FROM coverage WHERE revision_id = ?", (second,)).fetchone()[0] == 0
        assert catalog.current_month(db, archive, 2000, 1)[1]["revision_id"] == first
        timestamp = report["fields"][0]["valid_time"]
        assert catalog.missing_fields(
            db, 2000, 1, ["2m_temperature", "10m_u_component_of_wind"], [timestamp]
        ) == {"10m_u_component_of_wind": [timestamp]}
        with catalog.open_month(archive, 2000, 1) as dataset:
            assert set(dataset.data_vars) == {"2_metre_temperature"}
            np.testing.assert_array_equal(dataset["2_metre_temperature"].values, 280.0)
        catalog.publish_revision(db, archive, second, report)
        assert catalog.current_month(db, archive, 2000, 1)[1]["revision_id"] == second
    assert not (batch / "data.grib").exists()


def test_catalog_failure_restores_old_month(tmp_path: Path) -> None:
    """Roll back renamed stores when the catalog transaction cannot commit."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        first, _, _, report = _stage(db, archive)
        catalog.publish_revision(db, archive, first, report)
        second, stage, batch, report = _stage(db, archive, variable="10_metre_u_wind_component")
        db.execute(
            """CREATE TRIGGER fail_month BEFORE UPDATE ON months
               BEGIN SELECT RAISE(ABORT, 'catalog failure'); END"""
        )
        db.commit()
        with pytest.raises(sqlite3.IntegrityError, match="catalog failure"):
            catalog.publish_revision(db, archive, second, report)
        assert stage.exists()
        assert (batch / "data.grib").exists()
        assert db.execute("SELECT COUNT(*) FROM coverage WHERE revision_id = ?", (second,)).fetchone()[0] == 0
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (second,)).fetchone()[0] == "preparing"
        with catalog.open_month(archive, 2000, 1) as dataset:
            assert set(dataset.data_vars) == {"2_metre_temperature"}
        assert catalog.current_month(db, archive, 2000, 1)[1]["revision_id"] == first
        db.execute("DROP TRIGGER fail_month")
        db.commit()
        catalog.publish_revision(db, archive, second, report)
    assert not (batch / "data.grib").exists()


@pytest.mark.parametrize("completed_renames", [0, 1, 2])
def test_interrupted_replacement_rolls_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, completed_renames: int
) -> None:
    """Recover before, between, and after the two filesystem rename boundaries."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        first, _, _, report = _stage(db, archive)
        final = catalog.publish_revision(db, archive, first, report)
        second, stage, batch, report = _stage(db, archive, variable="10_metre_u_wind_component")
        rename = Path.rename
        calls = 0

        def interrupt(source: Path, destination: Path) -> Path:
            """Exit after the selected durable journal boundary."""
            nonlocal calls
            if completed_renames == 0:
                raise SystemExit("interrupted")
            result = rename(source, destination)
            calls += 1
            if calls == completed_renames:
                raise SystemExit("interrupted")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(Path, "rename", interrupt)
            with pytest.raises(SystemExit, match="interrupted"):
                catalog.publish_revision(db, archive, second, report)
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (second,)).fetchone()[0] == "ready"
        assert db.execute("SELECT current_revision_id FROM months").fetchone()[0] == first
        assert db.execute("SELECT COUNT(*) FROM coverage WHERE revision_id = ?", (second,)).fetchone()[0] == 0
        timestamp = report["fields"][0]["valid_time"]
        assert catalog.missing_fields(
            db, 2000, 1, ["2m_temperature", "10m_u_component_of_wind"], [timestamp]
        ) == {"10m_u_component_of_wind": [timestamp]}
        assert (batch / "data.grib").exists()
        assert stage.exists() == (completed_renames < 2)
        assert final.exists() == (completed_renames != 1)
    with catalog.open_month(archive, 2000, 1) as dataset:
        np.testing.assert_array_equal(dataset["2_metre_temperature"].values, 280.0)
        np.testing.assert_array_equal(dataset["10_metre_u_wind_component"].values, 3.0)
    with catalog.archive_session(archive) as db:
        assert catalog.current_month(db, archive, 2000, 1)[1]["revision_id"] == second
        assert db.execute("SELECT COUNT(*) FROM coverage WHERE revision_id = ?", (second,)).fetchone()[0] == 2
        assert db.execute("SELECT backup_path FROM revisions WHERE id = ?", (second,)).fetchone()[0] is None
        assert catalog.missing_fields(
            db, 2000, 1, ["2m_temperature", "10m_u_component_of_wind"], [timestamp]
        ) == {}
    assert not (batch / "data.grib").exists()


def test_failed_restoration_remains_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain the ready journal if restoring the previous directory also fails."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        first, _, _, report = _stage(db, archive)
        catalog.publish_revision(db, archive, first, report)
        second, stage, batch, report = _stage(db, archive, variable="10_metre_u_wind_component")
        rename = Path.rename
        calls = 0

        def fail_moves(source: Path, destination: Path) -> Path:
            """Fail the new-store move and the immediate old-store restoration."""
            nonlocal calls
            calls += 1
            if calls in (2, 3):
                raise OSError("filesystem unavailable")
            return rename(source, destination)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "rename", fail_moves)
            with pytest.raises(OSError, match="filesystem unavailable"):
                catalog.publish_revision(db, archive, second, report)
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (second,)).fetchone()[0] == "ready"
        backup = archive / db.execute("SELECT backup_path FROM revisions WHERE id = ?", (second,)).fetchone()[0]
        assert backup.is_dir()
        assert stage.is_dir()
        assert (batch / "data.grib").exists()
    with catalog.open_month(archive, 2000, 1) as dataset:
        assert set(dataset.data_vars) == {"2_metre_temperature", "10_metre_u_wind_component"}
    assert not (batch / "data.grib").exists()


def test_cleanup_failures_do_not_hide_published_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry retained backups and GRIBs without hiding a committed replacement."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        first, _, _, report = _stage(db, archive)
        catalog.publish_revision(db, archive, first, report)
        second, _, batch, report = _stage(db, archive, variable="10_metre_u_wind_component")
        with monkeypatch.context() as patch:
            patch.setattr(catalog.shutil, "rmtree", Mock(side_effect=PermissionError("backup is open")))
            patch.setattr(Path, "unlink", Mock(side_effect=PermissionError("source is open")))
            with pytest.warns(RuntimeWarning, match="cleanup remains pending") as warnings:
                catalog.publish_revision(db, archive, second, report)
            assert len(warnings) == 2
            row = db.execute("SELECT * FROM revisions WHERE id = ?", (second,)).fetchone()
            backup = archive / row["backup_path"]
            assert row["grib_removed"] == 0
            assert backup.is_dir()
            assert (batch / "data.grib").exists()
            with pytest.warns(RuntimeWarning, match="cleanup remains pending"):
                with catalog.open_month(archive, 2000, 1) as dataset:
                    np.testing.assert_array_equal(dataset["10_metre_u_wind_component"].values, 3.0)
    with catalog.archive_session(archive) as db:
        row = db.execute("SELECT * FROM revisions WHERE id = ?", (second,)).fetchone()
        assert row["grib_removed"] == 1
        assert row["backup_path"] is None
    assert not backup.exists()
    assert not (batch / "data.grib").exists()


def test_recovery_rejects_unrelated_final_store(tmp_path: Path) -> None:
    """Never claim a final directory whose revision marker is unrelated."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, stage, batch, report = _stage(db, archive)
        final = archive / "2000" / "01" / "era5.zarr"
        final.parent.mkdir(parents=True)
        stage.rename(final)
        zarr.open_group(final, mode="r+").attrs["revision_id"] = "unrelated"
        with db:
            db.execute(
                "UPDATE revisions SET status = 'ready', report_json = ? WHERE id = ?",
                (json.dumps(report), revision),
            )
    with pytest.raises(ValueError, match="different revision"):
        with catalog.archive_session(archive):
            pytest.fail("An unrelated store must not be published.")
    with sqlite3.connect(archive / "catalog.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0
    assert (batch / "data.grib").exists()


def test_layout_change_is_rejected_before_ready(tmp_path: Path) -> None:
    """Keep archive layout consistent without blocking the existing month."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, _, _, report = _stage(db, archive)
        catalog.publish_revision(db, archive, revision, report)
        revision, stage, batch, report = _stage(db, archive, month=2)
        report["layout"]["spatial_chunks"] = [1, 1]
        with pytest.raises(ValueError, match="layout"):
            catalog.publish_revision(db, archive, revision, report)
        assert db.execute("SELECT status FROM revisions WHERE id = ?", (revision,)).fetchone()[0] == "preparing"
        assert stage.exists()
        assert (batch / "data.grib").exists()
        with catalog.open_month(archive, 2000, 1) as dataset:
            np.testing.assert_array_equal(dataset["2_metre_temperature"].values, 280.0)


def test_reader_lock_allows_conversion_and_blocks_publication(tmp_path: Path) -> None:
    """Allow staged replacement while a reader holds the unchanged old month."""
    archive = tmp_path / "archive"
    with catalog.archive_session(archive) as db:
        revision, _, _, report = _stage(db, archive)
        catalog.publish_revision(db, archive, revision, report)
        with pytest.raises(portalocker.exceptions.LockException):
            with catalog.archive_session(archive):
                pytest.fail("A second writer must not acquire the archive.")
        with catalog.open_month(archive, 2000, 1) as dataset:
            revision, stage, _, report = _stage(db, archive, variable="10_metre_u_wind_component")
            assert stage.is_dir()
            assert set(dataset.data_vars) == {"2_metre_temperature"}
            np.testing.assert_array_equal(dataset["2_metre_temperature"].values, 280.0)
            with pytest.raises(portalocker.exceptions.LockException):
                with portalocker.Lock(archive / ".publication.lock", "a", timeout=0):
                    pytest.fail("The reader must retain the publication lock.")
        catalog.publish_revision(db, archive, revision, report)
    with catalog.open_month(archive, 2000, 1) as dataset:
        assert set(dataset.data_vars) == {"2_metre_temperature", "10_metre_u_wind_component"}


def test_writer_lock_is_released_when_process_exits(tmp_path: Path) -> None:
    """Recover writer ownership after a process exits without context cleanup."""
    archive = tmp_path / "archive"
    archive.mkdir()
    code = (
        "import os, portalocker, sys; "
        "lock = portalocker.Lock(sys.argv[1], 'a', timeout=0); "
        "lock.acquire(); os._exit(0)"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(archive / ".writer.lock")],
        check=True,
        timeout=20,
    )
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0
