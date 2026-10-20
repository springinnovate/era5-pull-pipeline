"""Exercise missing-field planning and monthly publication without CDS access."""

from contextlib import closing
from datetime import date
from itertools import product
from pathlib import Path
import sqlite3
from typing import Any
from unittest.mock import Mock

import eccodes
import numpy as np
import pytest
import zarr

from era5_pull_pipeline import cds, conversion, pipeline
from era5_pull_pipeline.catalog import open_month
from era5_pull_pipeline.credentials import CdsCredentials
from era5_pull_pipeline.variables import PARAMETER_IDS

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
                    parameter = PARAMETER_IDS[variable]
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
        np.testing.assert_array_equal(dataset["param_167"].values, expected)
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
    np.testing.assert_array_equal(zarr.open_group(output, mode="r")["param_167"][:], expected)
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 24
        assert db.execute("SELECT status, grib_removed FROM revisions").fetchall() == [("published", 1)]


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
    temperature_before = zarr.open_group(output, mode="r")["param_167"][:]
    request = cds.build_batch_request(start, end, [wind], max_days=31)
    sparse = {**request, "day": ["01", "03"], "time": ["00:00", "02:00"]}
    batch = cds.download_batch(CREDENTIALS, sparse, archive / "staging")
    assert pipeline.ingest_batch(batch, archive, spatial_chunks=(2, 2)) == output
    with open_month(archive, 1995, 3) as dataset:
        np.testing.assert_array_equal(dataset["param_167"].values, temperature_before)
        assert np.isnan(dataset["param_165"].sel(time="1995-03-02T12:00:00").values).all()
        assert np.isfinite(dataset["param_165"].sel(time="1995-03-03T02:00:00").values).all()

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
        np.testing.assert_array_equal(dataset["param_167"].values, temperature_before)
        assert dataset.sizes["time"] == 31 * 24
        assert np.isfinite(dataset["param_165"].values).all()
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        counts = db.execute(
            "SELECT c.variable, COUNT(*) FROM coverage c JOIN months m "
            "ON c.revision_id = m.current_revision_id GROUP BY c.variable"
        ).fetchall()
        assert dict(counts) == {"param_165": 744, "param_167": 744}
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
    batch = cds.download_batch(CREDENTIALS, sparse, archive / "staging")
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
        assert np.isfinite(dataset["param_167"].values).all()
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
    batch = cds.download_batch(CREDENTIALS, original, archive / "staging")
    output = pipeline.ingest_batch(batch, archive, spatial_chunks=(2, 2))
    before = zarr.open_group(output, mode="r")["param_167"][:]
    grib_client.value_offset = 1000.0
    overlap = {**request, "time": ["00:00", "01:00", "02:00", "03:00"]}
    batch = cds.download_batch(CREDENTIALS, overlap, archive / "staging")

    pipeline.ingest_batch(batch, archive, spatial_chunks=(2, 2))

    with open_month(archive, 2000, 1) as dataset:
        actual = dataset["param_167"].values
        np.testing.assert_array_equal(actual[[0, 2]], before)
        np.testing.assert_array_equal(actual[1], np.arange(12).reshape(3, 4) / 8 + 1169)
        np.testing.assert_array_equal(actual[3], np.arange(12).reshape(3, 4) / 8 + 1171)
    assert not (batch / "data.grib").exists()
    assert {
        path.relative_to(february): path.read_bytes()
        for path in february.rglob("*") if path.is_file()
    } == february_files


def test_checksum_failure_retains_grib_without_publication(
    tmp_path: Path, grib_client: Mock
) -> None:
    """Reject changed source bytes before any decoded fields become coverage."""
    archive = tmp_path / "archive"
    request = cds.build_batch_request(DAY, DAY, ["2m_temperature"])
    batch = cds.download_batch(CREDENTIALS, request, archive / "staging")
    source = batch / "data.grib"
    damaged = bytearray(source.read_bytes())
    damaged[-1] ^= 1
    source.write_bytes(damaged)
    with pytest.raises(ValueError, match="checksum|SHA|sha"):
        pipeline.ingest_batch(batch, archive, spatial_chunks=(2, 2))
    assert source.read_bytes() == damaged
    assert not (archive / "2000" / "01" / "era5.zarr").exists()
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0


def test_conversion_failure_keeps_old_store_catalog_and_new_grib(
    tmp_path: Path, grib_client: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave the current month readable when replacement conversion runs out of space."""
    archive = tmp_path / "archive"
    output = pipeline.pull_batch(
        DAY, DAY, ["2m_temperature"], archive, spatial_chunks=(2, 2)
    )
    original_values = zarr.open_group(output, mode="r")["param_167"][:]
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        current_before = db.execute("SELECT * FROM months").fetchall()
        coverage_before = db.execute("SELECT * FROM coverage ORDER BY valid_time").fetchall()
    request = cds.build_batch_request(DAY, DAY, ["10m_u_component_of_wind"])
    batch = cds.download_batch(CREDENTIALS, request, archive / "staging")
    source = batch / "data.grib"
    original_grib = source.read_bytes()
    failure = OSError("disk full")
    monkeypatch.setattr(zarr.Group, "create_array", Mock(side_effect=failure))
    with pytest.raises(OSError) as caught:
        pipeline.ingest_batch(batch, archive, spatial_chunks=(2, 2))
    assert caught.value is failure
    assert source.read_bytes() == original_grib
    with open_month(archive, 2000, 1) as dataset:
        np.testing.assert_array_equal(dataset["param_167"].values, original_values)
        assert "param_165" not in dataset
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert db.execute("SELECT * FROM months").fetchall() == current_before
        assert db.execute("SELECT * FROM coverage ORDER BY valid_time").fetchall() == coverage_before


def test_partial_batch_cannot_be_ingested(tmp_path: Path, grib_client: Mock) -> None:
    """Respect the downloader's completion boundary even if metadata is present."""
    archive = tmp_path / "archive"
    request = cds.build_batch_request(DAY, DAY, ["2m_temperature"])
    batch = cds.download_batch(CREDENTIALS, request, archive / "staging")
    partial = batch.with_suffix(".partial")
    batch.rename(partial)
    with pytest.raises(ValueError, match="partial|completed"):
        pipeline.ingest_batch(partial, archive, spatial_chunks=(2, 2))
    assert (partial / "data.grib").exists()
    assert not (archive / "2000" / "01" / "era5.zarr").exists()
