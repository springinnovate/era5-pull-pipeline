"""Verify daily reductions through real hourly COGs and durable catalogs."""

from datetime import UTC, date, datetime, timedelta
from contextlib import closing
import json
from pathlib import Path, PureWindowsPath
import sqlite3

import numpy as np
import pytest
import rasterio

from era5_pull_pipeline import catalog, daily
from era5_pull_pipeline.conversion import write_observation_cog
from era5_pull_pipeline.ncar import SourceFile


def publish_hours(
    archive: Path, first: datetime, count: int, short: str = "2t", *,
    initial: float = 0, revision: int = 1, missing_pixel: bool = False,
) -> list[Path]:
    """Publish tiny hourly COGs with distinguishable times, columns and sources.

    Args:
        archive: Hourly archive to create or extend.
        first: First UTC observation label.
        count: Number of contiguous hourly inputs.
        short: Supported fixture variable.
        initial: First hour's base value; each later hour adds one.
        revision: Provider identity used for selected refreshes.
        missing_pixel: Put NaN in the second hour's first pixel.

    Returns:
        Published hourly paths in time order.
    """
    variable, name, units = {
        "2t": ("2_metre_temperature", "2 metre temperature", "K"),
        "mtpr": ("mean_total_precipitation_rate", "Mean total precipitation rate", "kg m**-2 s**-1"),
        "ssrd": ("surface_solar_radiation_downwards", "Surface solar radiation downwards", "W m**-2 s"),
    }[short]
    interval = short in daily.INTERVAL_VARIABLES
    step = "avg" if short == "mtpr" else "accum" if interval else "instant"
    source = SourceFile(
        key=f"fixture/{short}/{first:%Y%m%d%H}.nc", variable=short, size=1,
        etag=f'"revision-{revision}"', last_modified="2026-01-01T00:00:00+00:00",
        valid_start=first, valid_end=first + timedelta(hours=count - 1),
    )
    with catalog.archive_session(archive) as db:
        registered = catalog.register_sources(db, archive, [source], first.date(), source.valid_end.date())[0]
        fields = []
        for hour in range(count):
            valid = int((first + timedelta(hours=hour)).timestamp())
            fields.append({
                "variable": variable, "source_variable": short, "source_file_id": registered["id"],
                "valid_time": valid, "reference_time": valid - 3600 if interval else None,
                "interval_start": valid - 3600 if interval else valid, "interval_end": valid,
                "start_step": 0 if interval else None, "end_step": 3600 if interval else None,
                "step_units": "s", "step_type": step,
            })
        attributes = {"name": name, "units": units}
        preparations = catalog.prepare_observations(db, archive, fields, attributes)
        reports = []
        for hour, (field, (identifier, staged)) in enumerate(zip(fields, preparations)):
            values = np.tile(np.arange(4, dtype=np.float64), (3, 1)) + initial + hour
            if missing_pixel and hour == 1:
                values[0, 0] = np.nan
            report = write_observation_cog(values, field, attributes, {
                "resolution": [90, 90], "latitude_first": 90, "longitude_first": -180,
            }, staged)
            reports.append((identifier, report))
        return catalog.publish_observations(db, archive, reports)


def observations(archive: Path) -> list[sqlite3.Row]:
    """Read published daily rows from their public consumer view.

    Args:
        archive: Daily archive with a published observation catalog.

    Returns:
        Public rows ordered by their ending timestamp and variable name.
    """
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        db.row_factory = sqlite3.Row
        return db.execute("SELECT * FROM observations ORDER BY valid_time, variable").fetchall()


def test_daily_mean_preserves_grid_missing_pixels_and_all_provider_identities(tmp_path: Path) -> None:
    """Reduce complete 1950 UTC days without altering source bytes or longitudes."""
    source, output = tmp_path / "hourly", tmp_path / "daily"
    start = datetime(1950, 1, 1, tzinfo=UTC)
    base = 100000000.125
    paths = publish_hours(source, start, 12, initial=base, missing_pixel=True)
    paths += publish_hours(source, start + timedelta(hours=12), 12, initial=base + 12)
    source_bytes = {path: path.read_bytes() for path in [source / "catalog.sqlite", *paths]}
    result = daily.aggregate_daily(start.date(), start.date(), ["2t"], source, output)
    assert result == {"completed": 1, "skipped": 0, "incomplete": 0}
    row, = observations(output)
    assert row["valid_time"] == row["interval_end"] == int((start + timedelta(days=1)).timestamp())
    assert row["interval_start"] == int(start.timestamp())
    assert row["temporal_type"] == "mean" and row["units"] == "K"
    assert row["reference_time"] is row["source_file_id"] is row["object_key"] is None
    assert row["start_step"] is row["end_step"] is None
    with rasterio.open(output / row["path"]) as raster:
        expected = np.tile(np.arange(4, dtype=np.float64) + base + 11.5, (3, 1))
        expected[0, 0] = np.nan
        np.testing.assert_equal(raster.read(1), expected)
        assert raster.dtypes == ("float64",)
        assert raster.transform.to_gdal() == (-225, 90, 0, 135, 0, -90)
        assert raster.tags()["interval_end"] == str(row["valid_time"])
    with sqlite3.connect(output / "catalog.sqlite") as db:
        provenance = json.loads(db.execute("SELECT provenance_json FROM derived_observations").fetchone()[0])
        assert provenance["method"] == "utc_daily_mean"
        assert len(provenance["observations"]) == 24 and len(provenance["providers"]) == 2
        assert {item["path"] for item in provenance["observations"]} == {path.relative_to(source).as_posix() for path in paths}
        assert all(provider["etag"] == '"revision-1"' for provider in provenance["providers"].values())
        assert db.execute("SELECT COUNT(*) FROM source_files").fetchone()[0] == 0
    assert (output / "dataset.yaml").read_text() == catalog.DAILY_DATASET_YAML
    assert all(path.read_bytes() == before for path, before in source_bytes.items())
    assert daily.aggregate_daily(start.date(), start.date(), ["2t"], source, output) == {
        "completed": 0, "skipped": 1, "incomplete": 0,
    }


@pytest.mark.parametrize(("short", "units", "factor"), [("mtpr", "mm", 3600), ("ssrd", "J m**-2", 1)])
def test_interval_totals_use_following_midnight_across_year_boundary(
    tmp_path: Path, short: str, units: str, factor: int,
) -> None:
    """Attribute 01–24 UTC precipitation and radiation inputs to December 31."""
    source, output = tmp_path / "hourly", tmp_path / "daily"
    first = datetime(1999, 12, 31, tzinfo=UTC)
    publish_hours(source, first, 25, short, initial=0.25)
    result = daily.aggregate_daily(first.date(), first.date(), [short], source, output)
    assert result["completed"] == 1
    row, = observations(output)
    assert row["units"] == units and row["temporal_type"] == "accumulation"
    assert row["interval_start"] == int(first.timestamp())
    assert row["valid_time"] == int(datetime(2000, 1, 1, tzinfo=UTC).timestamp())
    if short == "mtpr":
        assert row["variable"] == "total_precipitation" and "rate" not in row["name"].lower()
    with rasterio.open(output / row["path"]) as raster:
        expected = np.tile((306 + 24 * np.arange(4)) * factor, (3, 1))
        np.testing.assert_equal(raster.read(1), expected)


def test_incomplete_day_waits_for_midnight_and_missing_pixels_remain_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Never publish an incomplete total, then resume after the missing hour arrives."""
    source, output = tmp_path / "hourly", tmp_path / "daily"
    first = datetime(2000, 1, 31, 1, tzinfo=UTC)
    publish_hours(source, first, 23, "mtpr", initial=1, missing_pixel=True)
    day = first.date()
    assert daily.aggregate_daily(day, day, ["mtpr"], source, output) == {"completed": 0, "skipped": 0, "incomplete": 1}
    messages = capsys.readouterr().out
    assert "Incomplete mtpr UTC days (24 hourly inputs required): 2000-01-31" in messages
    assert "Daily summary: completed=0, skipped=0, incomplete=1" in messages
    assert observations(output) == []
    publish_hours(source, first + timedelta(hours=23), 1, "mtpr", initial=24)
    assert daily.aggregate_daily(day, day, ["mtpr"], source, output)["completed"] == 1
    with rasterio.open(output / observations(output)[0]["path"]) as raster:
        values = raster.read(1)
        assert np.isnan(values[0, 0])
        assert values[1, 0] == 300 * 3600


def test_refresh_replaces_only_selected_day_and_its_provenance(tmp_path: Path) -> None:
    """Keep unrelated daily bytes and replace the selected input/provider revision."""
    source, output = tmp_path / "hourly", tmp_path / "daily"
    first = datetime(2000, 1, 1, tzinfo=UTC)
    publish_hours(source, first, 48)
    daily.aggregate_daily(first.date(), date(2000, 1, 2), ["2t"], source, output)
    first_row, later = observations(output)
    later_path = output / later["path"]
    original_later = later_path.read_bytes()
    publish_hours(source, first, 24, initial=100, revision=2)
    assert daily.aggregate_daily(first.date(), first.date(), ["2t"], source, output)["skipped"] == 1
    assert daily.aggregate_daily(first.date(), first.date(), ["2t"], source, output, refresh=True)["completed"] == 1
    with rasterio.open(output / first_row["path"]) as raster:
        assert raster.read(1)[0, 0] == 111.5
    assert later_path.read_bytes() == original_later
    with sqlite3.connect(output / "catalog.sqlite") as db:
        provenance = json.loads(db.execute("SELECT provenance_json FROM derived_observations ORDER BY valid_time").fetchone()[0])
        assert {item["etag"] for item in provenance["providers"].values()} == {'"revision-2"'}


def test_ready_publication_recovers_values_and_derived_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover a complete daily file and its provenance after installation fails."""
    source, output = tmp_path / "hourly", tmp_path / "daily"
    first = datetime(2000, 1, 1, tzinfo=UTC)
    paths = publish_hours(source, first, 24)

    def fail_install(source: Path, destination: Path) -> None:
        """Stop at the durable ready-journal boundary."""
        raise OSError("installation interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(catalog, "_install_observation", fail_install)
        with pytest.raises(OSError, match="installation interrupted"):
            daily.aggregate_daily(first.date(), first.date(), ["2t"], source, output)
    assert all(path.exists() for path in paths)
    assert observations(output) == []
    assert daily.aggregate_daily(first.date(), first.date(), ["2t"], source, output)["skipped"] == 1
    with sqlite3.connect(output / "catalog.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM derived_observations").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 0


def test_resolution_guard_preserves_existing_archives_and_recovers_empty_schema(tmp_path: Path) -> None:
    """Reject mixed output before schema changes and finish interrupted creation."""
    hourly, output = tmp_path / "hourly", tmp_path / "daily"
    with catalog.archive_session(hourly):
        pass
    with sqlite3.connect(hourly / "catalog.sqlite") as db:
        db.execute("DROP TABLE archive_config")
    before = (hourly / "catalog.sqlite").read_bytes()
    with pytest.raises(ValueError, match="hourly archive"):
        with catalog.archive_session(hourly, resolution="daily"):
            pass
    assert (hourly / "catalog.sqlite").read_bytes() == before
    output.mkdir()
    with sqlite3.connect(output / "catalog.sqlite") as db:
        db.executescript(catalog.SCHEMA)
    with catalog.archive_session(output, resolution="daily"):
        pass
    with pytest.raises(ValueError, match="daily archive"):
        with catalog.archive_session(output):
            pass
    with pytest.raises(ValueError, match="hourly source"):
        daily.aggregate_daily(date(2000, 1, 1), date(2000, 1, 1), ["2t"], output, tmp_path / "another")


@pytest.mark.parametrize("failure", ["path", "units", "interval", "grid"])
def test_invalid_external_hourly_metadata_is_rejected(tmp_path: Path, failure: str) -> None:
    """Reject external path escapes and incompatible units, intervals or grids."""
    source, output = tmp_path / "hourly", tmp_path / "daily"
    first = datetime(2000, 1, 1, 1, tzinfo=UTC)
    publish_hours(source, first, 24, "mtpr")
    with sqlite3.connect(source / "catalog.sqlite") as db:
        if failure == "path":
            db.execute("UPDATE coverage SET path='../outside.tif' WHERE valid_time=?", (int(first.timestamp()),))
        elif failure == "units":
            db.execute("UPDATE variables SET attributes_json=json_set(attributes_json, '$.units', 'm')")
        elif failure == "interval":
            db.execute("UPDATE coverage SET interval_start=interval_start-3600 WHERE valid_time=?", (int(first.timestamp()),))
        else:
            db.execute("UPDATE coverage SET raster_json=json_set(raster_json, '$.transform[0]', -224) WHERE valid_time=?", (int(first.timestamp()),))
    with pytest.raises(ValueError):
        daily.aggregate_daily(first.date(), first.date(), ["mtpr"], source, output)
    assert observations(output) == []


def test_selection_and_overlapping_archives_fail_before_creation(tmp_path: Path) -> None:
    """Validate user date/variable selections and protect the original archive."""
    day = date(2000, 1, 1)
    with pytest.raises(ValueError, match="end date"):
        daily.aggregate_daily(day, date(1999, 1, 1), ["2t"], tmp_path / "source", tmp_path / "daily")
    with pytest.raises(ValueError, match="does not support"):
        daily.aggregate_daily(day, day, ["tp"], tmp_path / "source", tmp_path / "daily")
    with pytest.raises(ValueError, match="following UTC midnight"):
        daily.aggregate_daily(day, date.max, ["2t"], tmp_path / "source", tmp_path / "daily")
    with pytest.raises(ValueError, match="non-overlapping"):
        daily.aggregate_daily(day, day, ["2t"], tmp_path, tmp_path / "daily")
    assert list(tmp_path.iterdir()) == []


def test_unc_catalog_uri_uses_a_read_only_network_path() -> None:
    """Keep a UNC host in SQLite's path instead of its rejected URI authority."""
    path = PureWindowsPath("//server/share/hourly archive/catalog.sqlite")
    assert daily._read_only_catalog_uri(path) == "file:////server/share/hourly%20archive/catalog.sqlite?mode=ro"
