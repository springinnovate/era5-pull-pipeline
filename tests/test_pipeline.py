"""Exercise download, conversion, and publication together without CDS access."""

from contextlib import closing
from datetime import date
from pathlib import Path
import sqlite3
from unittest.mock import Mock

import eccodes
import numpy as np
import pytest
import zarr

from era5_pull_pipeline import cds, conversion, pipeline
from era5_pull_pipeline.catalog import open_month
from era5_pull_pipeline.credentials import CdsCredentials

DAY = date(2000, 1, 1)
CREDENTIALS = CdsCredentials("https://example.test/api", "pipeline-test-token")


@pytest.fixture
def grib_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Supply real hourly GRIB messages through a simulated CDS download."""
    monkeypatch.setattr(conversion, "GLOBAL_LATITUDES", np.array([90.0, 89.75, 89.5]))
    monkeypatch.setattr(conversion, "GLOBAL_LONGITUDES", np.array([0.0, 0.25, 0.5, 0.75]))
    payload = bytearray()
    for hour in range(24):
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
                    "paramId": 167,
                    "dataDate": 20000101,
                    "dataTime": hour * 100,
                },
            )
            eccodes.codes_set_values(handle, np.arange(12) / 8 + 280 + hour)
            payload.extend(eccodes.codes_get_message(handle))
        finally:
            eccodes.codes_release(handle)

    def download(target: str) -> None:
        """Write the encoded fields where the downloader expects its response."""
        Path(target).write_bytes(payload)

    factory = Mock()
    remote = factory.return_value.retrieve.return_value
    remote.request_id = "pipeline-test-request"
    remote.download.side_effect = download
    monkeypatch.setattr(cds.cdsapi, "Client", factory)
    return factory


def test_pull_publishes_then_repeats_without_credentials_or_cds(
    tmp_path: Path, grib_client: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publish actual GRIB data and recognize it after removing the source file."""
    credentials = Mock(return_value=CREDENTIALS)
    monkeypatch.setattr(pipeline, "load_credentials", credentials)
    archive = tmp_path / "archive"

    output = pipeline.pull_batch(
        DAY, DAY, ["2m_temperature"], archive, spatial_chunks=(2, 2)
    )

    assert output == archive / "2000" / "01" / "era5.zarr"
    root = zarr.open_group(output, mode="r")
    expected = np.array(
        [np.arange(12).reshape(3, 4) / 8 + 280 + hour for hour in range(24)]
    )
    np.testing.assert_array_equal(root["param_167"][:], expected)
    assert root["param_167"].attrs["units"] == "K"
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
    credentials.assert_called_once_with()
    grib_client.assert_called_once()
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 24
        assert db.execute(
            "SELECT status, grib_removed FROM revisions"
        ).fetchall() == [("published", 1)]

    credentials.side_effect = AssertionError("A repeat must not load credentials.")
    grib_client.side_effect = AssertionError("A repeat must not contact CDS.")
    assert pipeline.pull_batch(DAY, DAY, ["2m_temperature"], archive) == output
    assert pipeline.ingest_batch(batch, archive) == output
    with pytest.raises(ValueError):
        pipeline.pull_batch(DAY, DAY, ["10m_u_component_of_wind"], archive)
    assert (batch / "metadata.json").read_bytes() == metadata_before
    np.testing.assert_array_equal(
        zarr.open_group(output, mode="r")["param_167"][:], expected
    )
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 24
        assert db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 1


def test_checksum_failure_retains_grib_without_publication(
    tmp_path: Path, grib_client: Mock
) -> None:
    """Reject changed source bytes before any decoded fields become coverage."""
    archive = tmp_path / "archive"
    batch = cds.download_batch(
        CREDENTIALS, DAY, DAY, ["2m_temperature"], archive / "staging"
    )
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


def test_conversion_failure_retains_grib_without_publication(
    tmp_path: Path, grib_client: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain the downloaded batch when the actual Zarr writer runs out of space."""
    archive = tmp_path / "archive"
    batch = cds.download_batch(
        CREDENTIALS, DAY, DAY, ["2m_temperature"], archive / "staging"
    )
    source = batch / "data.grib"
    original = source.read_bytes()
    failure = OSError("disk full")
    monkeypatch.setattr(zarr.Group, "create_array", Mock(side_effect=failure))

    with pytest.raises(OSError) as caught:
        pipeline.ingest_batch(batch, archive, spatial_chunks=(2, 2))

    assert caught.value is failure
    assert source.read_bytes() == original
    assert not (archive / "2000" / "01" / "era5.zarr").exists()
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0


def test_partial_batch_cannot_be_ingested(
    tmp_path: Path, grib_client: Mock
) -> None:
    """Respect the downloader's completion boundary even if metadata is present."""
    archive = tmp_path / "archive"
    batch = cds.download_batch(
        CREDENTIALS, DAY, DAY, ["2m_temperature"], archive / "staging"
    )
    partial = batch.with_suffix(".partial")
    batch.rename(partial)

    with pytest.raises(ValueError, match="partial|completed"):
        pipeline.ingest_batch(partial, archive, spatial_chunks=(2, 2))

    assert (partial / "data.grib").exists()
    assert not (archive / "2000" / "01" / "era5.zarr").exists()
