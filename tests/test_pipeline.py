"""Exercise missing-field planning and monthly publication without CDS access."""

from contextlib import closing
from datetime import date
from itertools import product
from pathlib import Path
import sqlite3
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import eccodes
import numpy as np
import pytest
import zarr

from era5_pull_pipeline import catalog, cds, conversion, pipeline, staging
from era5_pull_pipeline.catalog import open_month
from era5_pull_pipeline.credentials import CdsCredentials

DAY = date(2000, 1, 1)
CREDENTIALS = CdsCredentials("https://example.test/api", "pipeline-test-token")


@pytest.fixture
def grib_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Encode the requested fields on a small grid at the remote-download boundary."""
    monkeypatch.setattr(conversion, "GLOBAL_LATITUDES", np.array([90.0, 89.75, 89.5]))
    monkeypatch.setattr(conversion, "GLOBAL_LONGITUDES", np.array([0.0, 0.25, 0.5, 0.75]))
    factory = Mock()
    factory.value_offset = 0.0

    def retrieve(dataset: str, request: dict[str, Any]) -> Mock:
        """Return a provider response for precisely the submitted selection."""
        assert dataset == cds.DATASET
        assert "area" not in request
        remote = Mock()
        remote.request_id = f"pipeline-test-{factory.return_value.retrieve.call_count}"

        def download(target: str) -> None:
            """Write real GRIB messages, including deliberately changed overlap values."""
            with Path(target).open("wb") as stream:
                for variable, day, time in product(
                    request["variable"], request["day"], request["time"]
                ):
                    parameter = {
                        "2m_temperature": 167,
                        "10m_u_component_of_wind": 165,
                        "10m_v_component_of_wind": 166,
                        "provider_defined_dewpoint": 168,
                    }[variable]
                    hour = int(time[:2])
                    handle = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
                    try:
                        eccodes.codes_set_key_vals(
                            handle,
                            {
                                "Ni": 4,
                                "Nj": 3,
                                "latitudeOfFirstGridPointInDegrees": 90,
                                "latitudeOfLastGridPointInDegrees": 89.5,
                                "longitudeOfFirstGridPointInDegrees": 0,
                                "longitudeOfLastGridPointInDegrees": 0.75,
                                "iDirectionIncrementInDegrees": 0.25,
                                "jDirectionIncrementInDegrees": 0.25,
                                "scanningMode": 0,
                                "paramId": parameter,
                                "dataDate": int(request["year"][0] + request["month"][0] + day),
                                "dataTime": hour * 100,
                            },
                        )
                        values = (
                            np.arange(12) / 8 + parameter + int(day) + hour
                            + factory.value_offset
                        )
                        eccodes.codes_set_values(handle, values)
                        eccodes.codes_write(handle, stream)
                    finally:
                        eccodes.codes_release(handle)

        remote.download.side_effect = download
        return remote

    factory.return_value.retrieve.side_effect = retrieve
    monkeypatch.setattr(cds.cdsapi, "Client", factory)
    monkeypatch.setattr(pipeline, "load_credentials", Mock(return_value=CREDENTIALS))
    return factory


def requested_fields(requests: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Expand CDS Cartesian selections to compare exact variable/day/hour coverage."""
    return [
        (variable, day, hour)
        for request in requests
        for variable, day, hour in product(
            request["variable"], request["day"], request["time"]
        )
    ]


def test_pull_publishes_then_repeats_without_credentials_or_cds(
    tmp_path: Path, grib_client: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recognize published fields after cleanup even when a repeat is a subset."""
    archive = tmp_path / "archive"
    output = pipeline.pull_batch(
        DAY, DAY, ["2m_temperature"], archive, spatial_chunks=(2, 2)
    )
    expected = np.array(
        [np.arange(12).reshape(3, 4) / 8 + 168 + hour for hour in range(24)]
    )
    with open_month(archive, 2000, 1) as dataset:
        np.testing.assert_array_equal(dataset["2_metre_temperature"].values, expected)
        np.testing.assert_array_equal(dataset.longitude.values, [0, 0.25, 0.5, 0.75])
        assert dataset.time.values[0] == np.datetime64("2000-01-01T00:00:00")
        assert dataset.time.values[-1] == np.datetime64("2000-01-01T23:00:00")
    batches = [path.parent for path in (archive / "staging").glob("*/metadata.json")]
    assert len(batches) == 1
    batch = batches[0]
    assert not (batch / "data.grib").exists()
    metadata_before = (batch / "metadata.json").read_bytes()
    monkeypatch.setattr(
        pipeline, "load_credentials", Mock(side_effect=AssertionError("Repeated credentials"))
    )
    grib_client.side_effect = AssertionError("Repeated CDS request")
    assert pipeline.pull_batch(DAY, DAY, ["2m_temperature"], archive) == output
    assert pipeline.ingest_batch(batch, archive) == output
    assert (batch / "metadata.json").read_bytes() == metadata_before
    np.testing.assert_array_equal(zarr.open_group(output, mode="r")["2_metre_temperature"][:], expected)
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 24
        assert db.execute("SELECT status, grib_removed FROM revisions").fetchall() == [("published", 1)]


def test_provider_defined_request_name_publishes_and_skips_without_mapping(
    tmp_path: Path, grib_client: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trust the provider's requested name and derive array names from the GRIB."""
    archive = tmp_path / "archive"
    variables = ["provider_defined_dewpoint", "2m_temperature"]
    output = pipeline.pull_batch(DAY, DAY, variables, archive, spatial_chunks=(2, 2))
    with open_month(archive, 2000, 1) as dataset:
        assert "2_metre_dewpoint_temperature" in dataset
        assert "2_metre_temperature" in dataset
        assert np.isfinite(dataset["2_metre_dewpoint_temperature"].values).all()
    monkeypatch.setattr(
        pipeline, "load_credentials", Mock(side_effect=AssertionError("Repeated credentials"))
    )
    grib_client.side_effect = AssertionError("Repeated CDS request")
    assert pipeline.pull_batch(DAY, DAY, list(reversed(variables)), archive) == output
    assert pipeline.pull_batch(DAY, DAY, ["provider_defined_dewpoint"], archive) == output


def test_full_march_temperature_then_partial_wind_and_exact_fill(
    tmp_path: Path, grib_client: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Add and complete March 1995 wind without retrieving or changing temperature."""
    archive = tmp_path / "archive"
    start, end = date(1995, 3, 1), date(1995, 3, 31)
    temperature = "2m_temperature"
    wind = "10m_u_component_of_wind"
    output = pipeline.pull_batch(
        start, end, [temperature], archive, spatial_chunks=(2, 2), max_days=31
    )
    temperature_before = zarr.open_group(output, mode="r")["2_metre_temperature"][:]
    request = cds.build_batch_request(start, end, [wind], max_days=31)
    sparse = {**request, "day": ["01", "03"], "time": ["00:00", "02:00"]}
    batch = cds.download_batch(CREDENTIALS, sparse, archive / "staging" / uuid4().hex)
    assert pipeline.ingest_batch(batch, archive, spatial_chunks=(2, 2)) == output
    with open_month(archive, 1995, 3) as dataset:
        np.testing.assert_array_equal(dataset["2_metre_temperature"].values, temperature_before)
        assert np.isnan(dataset["10_metre_u_wind_component"].sel(time="1995-03-02T12:00:00").values).all()
        assert np.isfinite(dataset["10_metre_u_wind_component"].sel(time="1995-03-03T02:00:00").values).all()

    grib_client.reset_mock()
    pipeline.pull_batch(
        start, end, [temperature, wind], archive, spatial_chunks=(2, 2), max_days=31
    )
    submitted = [call.args[1] for call in grib_client.return_value.retrieve.call_args_list]
    actual = requested_fields(submitted)
    expected = set(requested_fields([request])) - set(requested_fields([sparse]))
    assert set(actual) == expected
    assert len(actual) == len(expected)
    assert len(submitted) == 2
    with open_month(archive, 1995, 3) as dataset:
        np.testing.assert_array_equal(dataset["2_metre_temperature"].values, temperature_before)
        assert dataset.sizes["time"] == 31 * 24
        assert np.isfinite(dataset["10_metre_u_wind_component"].values).all()
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        counts = db.execute(
            "SELECT c.variable, COUNT(*) FROM coverage c JOIN months m "
            "ON c.revision_id = m.current_revision_id GROUP BY c.variable"
        ).fetchall()
        assert dict(counts) == {"10_metre_u_wind_component": 744, "2_metre_temperature": 744}
        revisions_before = db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0]

    monkeypatch.setattr(
        pipeline, "load_credentials", Mock(side_effect=AssertionError("Repeated credentials"))
    )
    grib_client.side_effect = AssertionError("Repeated CDS request")
    assert pipeline.pull_batch(start, end, [wind, temperature], archive, max_days=31) == output
    assert pipeline.pull_batch(date(1995, 3, 2), date(1995, 3, 4), [wind], archive) == output
    assert pipeline.ingest_batch(batch, archive) == output
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == revisions_before


def test_fill_interior_hours_and_extend_time_axis(
    tmp_path: Path, grib_client: Mock
) -> None:
    """Fill interior gaps and add earlier/later hours without downloading stored cells."""
    archive = tmp_path / "archive"
    request = cds.build_batch_request(DAY, date(2000, 1, 2), ["2m_temperature"])
    sparse = {**request, "day": ["01"], "time": ["01:00", "03:00"]}
    batch = cds.download_batch(CREDENTIALS, sparse, archive / "staging" / uuid4().hex)
    pipeline.ingest_batch(batch, archive, spatial_chunks=(2, 2))
    grib_client.reset_mock()

    output = pipeline.pull_batch(
        DAY, date(2000, 1, 2), ["2m_temperature"], archive, spatial_chunks=(2, 2)
    )

    submitted = [call.args[1] for call in grib_client.return_value.retrieve.call_args_list]
    actual = requested_fields(submitted)
    expected = set(requested_fields([request])) - set(requested_fields([sparse]))
    assert set(actual) == expected
    assert len(actual) == len(expected)
    with open_month(archive, 2000, 1) as dataset:
        assert dataset.sizes["time"] == 48
        assert dataset.time.values[0] == np.datetime64("2000-01-01T00:00:00")
        assert dataset.time.values[-1] == np.datetime64("2000-01-02T23:00:00")
        assert np.isfinite(dataset["2_metre_temperature"].values).all()
    assert output == archive / "2000" / "01" / "era5.zarr"


def test_overlapping_ingest_preserves_old_values_and_other_month(
    tmp_path: Path, grib_client: Mock
) -> None:
    """Merge new hours while preserving changed overlaps and an unrelated month."""
    archive = tmp_path / "archive"
    february = pipeline.pull_batch(
        date(2000, 2, 1), date(2000, 2, 1), ["2m_temperature"], archive,
        spatial_chunks=(2, 2),
    )
    february_files = {
        path.relative_to(february): path.read_bytes()
        for path in february.rglob("*") if path.is_file()
    }
    request = cds.build_batch_request(DAY, DAY, ["2m_temperature"])
    original = {**request, "time": ["00:00", "02:00"]}
    batch = cds.download_batch(CREDENTIALS, original, archive / "staging" / uuid4().hex)
    output = pipeline.ingest_batch(batch, archive, spatial_chunks=(2, 2))
    before = zarr.open_group(output, mode="r")["2_metre_temperature"][:]
    grib_client.value_offset = 1000.0
    overlap = {**request, "time": ["00:00", "01:00", "02:00", "03:00"]}
    batch = cds.download_batch(CREDENTIALS, overlap, archive / "staging" / uuid4().hex)

    pipeline.ingest_batch(batch, archive, spatial_chunks=(2, 2))

    with open_month(archive, 2000, 1) as dataset:
        actual = dataset["2_metre_temperature"].values
        np.testing.assert_array_equal(actual[[0, 2]], before)
        np.testing.assert_array_equal(actual[1], np.arange(12).reshape(3, 4) / 8 + 1169)
        np.testing.assert_array_equal(actual[3], np.arange(12).reshape(3, 4) / 8 + 1171)
    assert not (batch / "data.grib").exists()
    assert {
        path.relative_to(february): path.read_bytes()
        for path in february.rglob("*") if path.is_file()
    } == february_files


def test_conversion_failure_keeps_old_store_catalog_and_new_grib(
    tmp_path: Path, grib_client: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave the current month readable when replacement conversion runs out of space."""
    archive = tmp_path / "archive"
    output = pipeline.pull_batch(
        DAY, DAY, ["2m_temperature"], archive, spatial_chunks=(2, 2)
    )
    original_values = zarr.open_group(output, mode="r")["2_metre_temperature"][:]
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        current_before = db.execute("SELECT * FROM months").fetchall()
        coverage_before = db.execute("SELECT * FROM coverage ORDER BY valid_time").fetchall()
    request = cds.build_batch_request(DAY, DAY, ["10m_u_component_of_wind"])
    batch = cds.download_batch(CREDENTIALS, request, archive / "staging" / uuid4().hex)
    source = batch / "data.grib"
    original_grib = source.read_bytes()
    failure = OSError("disk full")
    monkeypatch.setattr(zarr.Group, "create_array", Mock(side_effect=failure))
    with pytest.raises(OSError) as caught:
        pipeline.ingest_batch(batch, archive, spatial_chunks=(2, 2))
    assert caught.value is failure
    assert source.read_bytes() == original_grib
    with open_month(archive, 2000, 1) as dataset:
        np.testing.assert_array_equal(dataset["2_metre_temperature"].values, original_values)
        assert "10_metre_u_wind_component" not in dataset
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert db.execute("SELECT * FROM months").fetchall() == current_before
        assert db.execute("SELECT * FROM coverage ORDER BY valid_time").fetchall() == coverage_before


def test_partial_batch_cannot_be_ingested(tmp_path: Path, grib_client: Mock) -> None:
    """Respect the downloader's completion boundary even if metadata is present."""
    archive = tmp_path / "archive"
    request = cds.build_batch_request(DAY, DAY, ["2m_temperature"])
    batch = cds.download_batch(CREDENTIALS, request, archive / "staging" / uuid4().hex)
    partial = batch.with_suffix(".partial")
    batch.rename(partial)
    with pytest.raises(ValueError, match="partial|completed"):
        pipeline.ingest_batch(partial, archive, spatial_chunks=(2, 2))
    assert (partial / "data.grib").exists()
    assert not (archive / "2000" / "01" / "era5.zarr").exists()


@pytest.fixture
def availability(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Provide source availability independently of the fake GRIB service."""
    load = Mock(return_value=[{
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature", "10m_u_component_of_wind"],
        "year": ["1950", "1995", "1999", "2000"],
        "month": [f"{month:02d}" for month in range(1, 13)],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
    }])
    monkeypatch.setattr(pipeline, "load_availability", load)
    return load


def test_range_publishes_once_per_month_and_repeats_offline(
    tmp_path: Path, grib_client: Mock, availability: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross a year boundary in three batches but only two publications."""
    archive = tmp_path / "archive"
    start, end = date(1999, 12, 31), date(2000, 1, 2)
    summary = pipeline.pull_range(start, end, ["2m_temperature"], archive, (2, 2), 1)
    assert summary == {"completed": 72, "skipped": 0, "unavailable": 0, "failed": 0, "pending": 0}
    assert grib_client.return_value.retrieve.call_count == 3
    availability.assert_called_once()
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 72
        january = catalog.current_month(db, archive, 2000, 1)
        assert len({field["source_request_sha256"] for field in january[1]["fields"]}) == 2
    with open_month(archive, 2000, 1) as dataset:
        assert dataset.sizes["time"] == 48
        assert np.isfinite(dataset["2_metre_temperature"].values).all()
    assert not list((archive / "staging").rglob("data.grib"))
    assert not list((archive / "staging").rglob("converted.zarr"))
    monkeypatch.setattr(pipeline, "load_credentials", Mock(side_effect=AssertionError("credentials")))
    monkeypatch.setattr(pipeline, "load_availability", Mock(side_effect=AssertionError("availability")))
    grib_client.side_effect = AssertionError("CDS")
    summary = pipeline.pull_range(start, end, ["2m_temperature"], archive, (2, 2), 1)
    assert summary == {"completed": 0, "skipped": 72, "unavailable": 0, "failed": 0, "pending": 0}


def test_range_reuses_cataloged_progress_after_conversion_interrupt(
    tmp_path: Path, grib_client: Mock, availability: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep completed decoding, rebuild incomplete writes, and retain unpublished GRIBs."""
    archive = tmp_path / "archive"
    convert = staging.convert_batch
    calls = []

    def interrupted(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(args[0])
        if len(calls) == 2:
            args[2].mkdir()
            (args[2] / "partial-write").write_bytes(b"unfinished")
            raise KeyboardInterrupt
        return convert(*args, **kwargs)

    monkeypatch.setattr(staging, "convert_batch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        pipeline.pull_range(DAY, date(2000, 1, 2), ["2m_temperature"], archive, (2, 2), 1)
    assert len(list((archive / "staging").rglob("data.grib"))) == 2
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM staged_batches WHERE metadata_json IS NOT NULL").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM staged_batches WHERE report_json IS NOT NULL").fetchone()[0] == 1
    checked = Mock(wraps=convert)
    monkeypatch.setattr(staging, "convert_batch", checked)
    monkeypatch.setattr(pipeline, "load_credentials", Mock(side_effect=AssertionError("credentials")))
    grib_client.side_effect = AssertionError("download")
    summary = pipeline.pull_range(DAY, date(2000, 1, 2), ["2m_temperature"], archive, (2, 2), 1)
    assert summary["completed"] == 48
    assert checked.call_count == 1
    assert not list((archive / "staging").rglob("data.grib"))
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM staged_batches").fetchone()[0] == 0


def test_range_recovers_publication_without_reconverting_or_contacting_source(
    tmp_path: Path, grib_client: Mock, availability: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Roll forward an interrupted rename before planning any new requests."""
    archive = tmp_path / "archive"
    rename = catalog._rename_store

    def interrupted(source: Path, destination: Path) -> None:
        rename(source, destination)
        raise KeyboardInterrupt

    monkeypatch.setattr(catalog, "_rename_store", interrupted)
    with pytest.raises(KeyboardInterrupt):
        pipeline.pull_range(DAY, date(2000, 1, 2), ["2m_temperature"], archive, (2, 2), 1)
    assert len(list((archive / "staging").rglob("data.grib"))) == 2
    monkeypatch.setattr(catalog, "_rename_store", rename)
    monkeypatch.setattr(pipeline, "load_availability", Mock(side_effect=AssertionError("availability")))
    monkeypatch.setattr(pipeline, "assemble_month", Mock(side_effect=AssertionError("assembly")))
    grib_client.side_effect = AssertionError("download")
    summary = pipeline.pull_range(DAY, date(2000, 1, 2), ["2m_temperature"], archive, (2, 2), 1)
    assert summary["skipped"] == 48
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 48
    assert not list((archive / "staging").rglob("data.grib"))


def test_range_latest_partial_day_stays_missing_until_available(
    tmp_path: Path, grib_client: Mock, availability: Mock
) -> None:
    """Publish only available hours, then retrieve newly available hours exactly."""
    archive = tmp_path / "archive"
    complete = availability.return_value[0]
    availability.return_value = [
        {**complete, "day": ["01"]},
        {**complete, "day": ["02"], "time": [f"{hour:02d}:00" for hour in range(6)]},
    ]
    summary = pipeline.pull_range(DAY, date(2000, 1, 3), ["2m_temperature"], archive, (2, 2), 7)
    assert summary == {"completed": 30, "skipped": 0, "unavailable": 42, "failed": 0, "pending": 0}
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 30
    grib_client.reset_mock()
    availability.return_value = [complete]
    summary = pipeline.pull_range(DAY, date(2000, 1, 3), ["2m_temperature"], archive, (2, 2), 7)
    assert summary == {"completed": 42, "skipped": 30, "unavailable": 0, "failed": 0, "pending": 0}
    submitted = [call.args[1] for call in grib_client.return_value.retrieve.call_args_list]
    assert len(requested_fields(submitted)) == 42
    with open_month(archive, 2000, 1) as dataset:
        assert dataset.sizes["time"] == 72
        assert np.isfinite(dataset["2_metre_temperature"].values).all()


def test_range_failure_reports_only_the_failed_batch_and_preserves_traceback(
    tmp_path: Path, grib_client: Mock, availability: Mock,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed request leaves future work pending rather than declaring it failed."""
    failure = OSError("source transfer failed")
    monkeypatch.setattr(pipeline, "download_batch", Mock(side_effect=failure))
    with pytest.raises(OSError) as caught:
        pipeline.pull_range(DAY, date(2000, 1, 3), ["2m_temperature"], tmp_path / "archive", (2, 2), 1)
    assert caught.value is failure
    output = capsys.readouterr().out
    assert "completed=0, skipped=0, unavailable=0, failed=24, pending=48" in output


def test_range_reuses_partial_day_checkpoint_when_availability_expands(
    tmp_path: Path, grib_client: Mock, availability: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Newly available hours do not force a completed partial day to download again."""
    archive = tmp_path / "archive"
    complete = availability.return_value[0]
    availability.return_value = [{**complete, "time": [f"{hour:02d}:00" for hour in range(6)]}]
    assemble = pipeline.assemble_month
    monkeypatch.setattr(pipeline, "assemble_month", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        pipeline.pull_range(DAY, DAY, ["2m_temperature"], archive, (2, 2), 1)
    grib_client.reset_mock()
    availability.return_value = [complete]
    monkeypatch.setattr(pipeline, "assemble_month", assemble)
    convert = Mock(wraps=staging.convert_batch)
    monkeypatch.setattr(staging, "convert_batch", convert)
    summary = pipeline.pull_range(DAY, DAY, ["2m_temperature"], archive, (2, 2), 1)
    assert summary["completed"] == 24
    assert convert.call_count == 1
    request = grib_client.return_value.retrieve.call_args.args[1]
    assert request["time"] == [f"{hour:02d}:00" for hour in range(6, 24)]
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 24


def test_range_resumes_download_job_and_prior_conversion(
    tmp_path: Path, grib_client: Mock, availability: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted second transfer reuses its job and the first batch's checkpoint."""
    archive = tmp_path / "archive"
    retrieve = grib_client.return_value.retrieve.side_effect

    def interrupted(dataset: str, request: dict[str, Any]) -> Mock:
        remote = retrieve(dataset, request)
        if request["day"] == ["02"]:
            download = remote.download.side_effect

            def partial(target: str) -> None:
                Path(target).write_bytes(b"incomplete transfer")
                remote.download.side_effect = download
                raise KeyboardInterrupt

            remote.download.side_effect = partial
            grib_client.return_value.client.get_remote.return_value = remote
        return remote

    grib_client.return_value.retrieve.side_effect = interrupted
    with pytest.raises(KeyboardInterrupt):
        pipeline.pull_range(DAY, date(2000, 1, 2), ["2m_temperature"], archive, (2, 2), 1)
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM staged_batches WHERE report_json IS NOT NULL").fetchone()[0] == 1
    assert len(list((archive / "staging").rglob("*.partial"))) == 1
    convert = Mock(wraps=staging.convert_batch)
    monkeypatch.setattr(staging, "convert_batch", convert)
    summary = pipeline.pull_range(DAY, date(2000, 1, 2), ["2m_temperature"], archive, (2, 2), 1)
    assert summary["completed"] == 48
    assert convert.call_count == 1
    assert grib_client.return_value.retrieve.call_count == 2
    grib_client.return_value.client.get_remote.assert_called_once_with("pipeline-test-2")


def test_range_adopts_completed_download_after_interrupted_catalog_update(
    tmp_path: Path, grib_client: Mock, availability: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed rename is recovered at its cataloged path without another transfer."""
    archive = tmp_path / "archive"
    download = pipeline.download_batch

    def interrupted(*args: Any, **kwargs: Any) -> Path:
        download(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "download_batch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        pipeline.pull_range(DAY, DAY, ["2m_temperature"], archive, (2, 2), 1)
    with catalog.archive_session(archive) as db:
        record = db.execute("SELECT * FROM staged_batches").fetchone()
        assert record["metadata_json"] is None
        assert (Path(record["batch_path"]) / "metadata.json").exists()
    monkeypatch.setattr(pipeline, "download_batch", Mock(side_effect=AssertionError("download")))
    monkeypatch.setattr(pipeline, "load_credentials", Mock(side_effect=AssertionError("credentials")))
    summary = pipeline.pull_range(DAY, DAY, ["2m_temperature"], archive, (2, 2), 1)
    assert summary["completed"] == 24
    grib_client.return_value.retrieve.assert_called_once()
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM staged_batches").fetchone()[0] == 0
