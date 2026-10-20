"""Check single-batch selection and staged downloads without contacting CDS."""

from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from era5_pull_pipeline import cds
from era5_pull_pipeline.credentials import CdsCredentials

TOKEN = "batch-test-private-token"
CREDENTIALS = CdsCredentials("https://example.test/api", TOKEN)
PAYLOAD = b"downloaded bytes retained for later GRIB ingestion"


@pytest.fixture
def client_factory(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Provide a remote request whose download writes a local response."""
    factory = Mock()
    remote = factory.return_value.retrieve.return_value
    remote.request_id = "cds-test-request"

    def download(target: str) -> str:
        """Write the simulated transfer to the requested staging path."""
        Path(target).write_bytes(PAYLOAD)
        return target

    remote.download.side_effect = download
    monkeypatch.setattr(cds.cdsapi, "Client", factory)
    return factory


@pytest.mark.parametrize(
    ("start", "end", "days"),
    [
        (date(2000, 1, 1), date(2000, 1, 1), ["01"]),
        (date(2000, 1, 29), date(2000, 1, 31), ["29", "30", "31"]),
        (date(2000, 2, 28), date(2000, 2, 29), ["28", "29"]),
    ],
)
def test_build_full_global_hourly_request(
    start: date, end: date, days: list[str]
) -> None:
    """Select inclusive dates and all hours without a geographic subset."""
    request = cds.build_batch_request(
        start, end, ["total_precipitation", "2m_temperature", "total_precipitation"]
    )

    assert request == {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature", "total_precipitation"],
        "year": [str(start.year)],
        "month": [f"{start.month:02d}"],
        "day": days,
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "grid": [0.25, 0.25],
        "data_format": "grib",
        "download_format": "unarchived",
    }


def test_configured_day_limit_and_provider_variable_names() -> None:
    """Use the configured limit and leave variable-name validation to CDS."""
    request = cds.build_batch_request(
        date(2000, 1, 1), date(2000, 1, 8), ["provider_variable_name"], max_days=8
    )

    assert request["day"] == [f"{day:02d}" for day in range(1, 9)]
    assert request["variable"] == ["provider_variable_name"]


@pytest.mark.parametrize(
    ("start", "end", "variables", "max_days"),
    [
        (date(2000, 1, 2), date(2000, 1, 1), ["2m_temperature"], 7),
        (date(2000, 1, 31), date(2000, 2, 1), ["2m_temperature"], 7),
        (date(1999, 12, 31), date(2000, 1, 1), ["2m_temperature"], 7),
        (date(2000, 1, 1), date(2000, 1, 1), ["2m_temperature"], 0),
        (date(2000, 1, 1), date(2000, 1, 1), ["2m_temperature"], -1),
        (date(2000, 1, 1), date(2000, 1, 8), ["2m_temperature"], 7),
        (date(2000, 1, 1), date(2000, 1, 1), [], 7),
    ],
)
def test_invalid_batch_selection(
    start: date, end: date, variables: list[str], max_days: int
) -> None:
    """Reject selections that cannot fit one configured monthly batch."""
    with pytest.raises(ValueError):
        cds.build_batch_request(start, end, variables, max_days)


def test_field_limit_counts_unique_variables_and_all_hours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow the exact field limit and reject excess days or variables."""
    monkeypatch.setattr(cds, "MAX_REQUEST_FIELDS", 48)
    request = cds.build_batch_request(
        date(2000, 1, 1), date(2000, 1, 2), ["2m_temperature", "2m_temperature"]
    )
    assert request["variable"] == ["2m_temperature"]

    with pytest.raises(ValueError, match="fields"):
        cds.build_batch_request(
            date(2000, 1, 1), date(2000, 1, 3), ["2m_temperature"]
        )
    with pytest.raises(ValueError, match="fields"):
        cds.build_batch_request(
            date(2000, 1, 1),
            date(2000, 1, 2),
            ["2m_temperature", "10m_u_component_of_wind"],
        )


def test_completed_download_has_request_and_integrity_metadata(
    tmp_path: Path, client_factory: Mock
) -> None:
    """Publish one completed directory with reproducible identity and checksum."""
    staging = tmp_path / "staging"
    completed = cds.download_batch(
        CREDENTIALS,
        cds.build_batch_request(
            date(2000, 1, 1), date(2000, 1, 2), ["total_precipitation", "2m_temperature"]
        ),
        staging,
    )

    assert completed.parent == staging
    assert not completed.name.endswith(".partial")
    assert list(staging.iterdir()) == [completed]
    assert (completed / "data.grib").read_bytes() == PAYLOAD
    metadata_text = (completed / "metadata.json").read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)
    request = client_factory.return_value.retrieve.call_args.args[1]
    assert client_factory.return_value.retrieve.call_args.args == (cds.DATASET, request)
    assert metadata["dataset"] == cds.DATASET
    assert metadata["request"] == request
    identity = json.dumps(
        {"dataset": cds.DATASET, "request": request},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert metadata["request_sha256"] == hashlib.sha256(identity).hexdigest()
    assert metadata["cds_request_id"] == "cds-test-request"
    assert metadata["file"] == "data.grib"
    assert metadata["size_bytes"] == len(PAYLOAD)
    assert metadata["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
    started = datetime.fromisoformat(metadata["started_at"])
    retrieved = datetime.fromisoformat(metadata["retrieved_at"])
    assert started.utcoffset() == retrieved.utcoffset() == timedelta(0)
    assert started <= retrieved
    assert TOKEN not in metadata_text
    options = client_factory.call_args.kwargs
    assert options["url"] == CREDENTIALS.url
    assert options["key"] == TOKEN
    assert isinstance(options["session"], requests.Session)
    assert options["wait_until_complete"] is False
    assert options["retry_max"] == cds.CLIENT_OPTIONS["retry_max"]
    assert options["sleep_max"] == cds.CLIENT_OPTIONS["sleep_max"]


def test_repeat_selection_has_new_directory_and_same_request_identity(
    tmp_path: Path, client_factory: Mock
) -> None:
    """Keep separate retrievals while identifying equivalent variable selections."""
    first = cds.download_batch(
        CREDENTIALS,
        cds.build_batch_request(
            date(2000, 1, 1), date(2000, 1, 1), ["total_precipitation", "2m_temperature"]
        ),
        tmp_path,
    )
    second = cds.download_batch(
        CREDENTIALS,
        cds.build_batch_request(
            date(2000, 1, 1), date(2000, 1, 1),
            ["2m_temperature", "total_precipitation", "2m_temperature"],
        ),
        tmp_path,
    )

    assert first != second
    first_metadata = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
    second_metadata = json.loads((second / "metadata.json").read_text(encoding="utf-8"))
    assert first_metadata["request_sha256"] == second_metadata["request_sha256"]
    assert (first / "data.grib").read_bytes() == PAYLOAD
    assert (second / "data.grib").read_bytes() == PAYLOAD
    assert client_factory.return_value.retrieve.call_count == 2


@pytest.mark.parametrize(
    "error", [KeyboardInterrupt(), requests.ConnectionError("transfer failed")]
)
def test_interrupted_transfer_remains_partial(
    tmp_path: Path, client_factory: Mock, error: BaseException
) -> None:
    """Leave incomplete bytes clearly separated from completed staged inputs."""

    def fail_download(target: str) -> str:
        """Interrupt the transfer after some bytes reach disk."""
        Path(target).write_bytes(b"incomplete")
        raise error

    client_factory.return_value.retrieve.return_value.download.side_effect = fail_download

    with pytest.raises(type(error)) as caught:
        cds.download_batch(
            CREDENTIALS,
            cds.build_batch_request(date(2000, 1, 1), date(2000, 1, 1), ["2m_temperature"]),
            tmp_path,
        )

    assert caught.value is error
    partials = list(tmp_path.iterdir())
    assert len(partials) == 1
    assert partials[0].name.endswith(".partial")
    assert (partials[0] / "data.grib").read_bytes() == b"incomplete"
    assert not (partials[0] / "metadata.json").exists()


def test_download_preserves_selected_days_and_hours(
    tmp_path: Path, client_factory: Mock
) -> None:
    """Do not expand a planner selection back to complete days or intervals."""
    request = cds.build_batch_request(date(2000, 1, 1), date(2000, 1, 3), ["2m_temperature"])
    request["day"] = ["01", "03"]
    request["time"] = ["02:00", "16:00"]
    completed = cds.download_batch(CREDENTIALS, request, tmp_path / "staging")
    client_factory.return_value.retrieve.assert_called_once_with(cds.DATASET, request)
    metadata = json.loads((completed / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["request"] == request


def test_failed_metadata_write_keeps_directory_partial(
    tmp_path: Path, client_factory: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep a completed transfer incomplete until its metadata is also written."""
    original_write = Path.write_text
    failure = OSError("disk full")

    def fail_write(path: Path, data: str, encoding: str) -> int:
        """Simulate a metadata write that fails after writing some JSON."""
        if path.name != "metadata.json":
            return original_write(path, data, encoding=encoding)
        original_write(path, data[:10], encoding=encoding)
        raise failure

    monkeypatch.setattr(Path, "write_text", fail_write)

    with pytest.raises(OSError) as caught:
        cds.download_batch(
            CREDENTIALS,
            cds.build_batch_request(date(2000, 1, 1), date(2000, 1, 1), ["2m_temperature"]),
            tmp_path,
        )

    assert caught.value is failure
    partials = list(tmp_path.iterdir())
    assert len(partials) == 1
    assert partials[0].name.endswith(".partial")
    assert (partials[0] / "data.grib").read_bytes() == PAYLOAD
    with pytest.raises(json.JSONDecodeError):
        json.loads((partials[0] / "metadata.json").read_text(encoding="utf-8"))


def test_resume_reuses_recorded_remote_job(
    tmp_path: Path, client_factory: Mock
) -> None:
    """Reuse an interrupted job without another submission, restarting its bytes."""
    request = cds.build_batch_request(date(2000, 1, 1), date(2000, 1, 1), ["2m_temperature"])
    remote = client_factory.return_value.retrieve.return_value

    def interrupt(target: str) -> None:
        Path(target).write_bytes(b"partial bytes")
        raise KeyboardInterrupt

    remote.download.side_effect = interrupt
    with pytest.raises(KeyboardInterrupt):
        cds.download_batch(CREDENTIALS, request, tmp_path, resume=True)
    (partial,) = tmp_path.iterdir()
    journal_text = (partial / "request.json").read_text(encoding="utf-8")
    journal = json.loads(journal_text)
    assert journal["request"] == request
    assert journal["cds_request_id"] == "cds-test-request"
    assert TOKEN not in journal_text

    resumed = client_factory.return_value.client.get_remote.return_value
    resumed.request_id = remote.request_id
    resumed.download.side_effect = lambda target: Path(target).write_bytes(PAYLOAD)
    completed = cds.download_batch(CREDENTIALS, request, tmp_path, resume=True)

    assert completed == partial.with_suffix("")
    assert (completed / "data.grib").read_bytes() == PAYLOAD
    client_factory.return_value.retrieve.assert_called_once_with(cds.DATASET, request)
    client_factory.return_value.client.get_remote.assert_called_once_with("cds-test-request")
    metadata = json.loads((completed / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["started_at"] == journal["started_at"]
    assert metadata["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()


def test_resume_after_submission_failure_and_different_selection(
    tmp_path: Path, client_factory: Mock
) -> None:
    """Retry an unsubmitted selection while leaving another partial job alone."""
    request = cds.build_batch_request(date(2000, 1, 1), date(2000, 1, 1), ["2m_temperature"])
    client_factory.return_value.retrieve.side_effect = requests.ConnectionError("submit failed")
    with pytest.raises(requests.ConnectionError):
        cds.download_batch(CREDENTIALS, request, tmp_path, resume=True)
    (partial,) = tmp_path.iterdir()
    journal = json.loads((partial / "request.json").read_text(encoding="utf-8"))
    assert "cds_request_id" not in journal
    client_factory.return_value.retrieve.side_effect = None
    other = {**request, "variable": ["total_precipitation"]}
    different = cds.download_batch(CREDENTIALS, other, tmp_path, resume=True)
    assert partial.exists()
    assert different != partial.with_suffix("")
    completed = cds.download_batch(CREDENTIALS, request, tmp_path, resume=True)
    assert completed == partial.with_suffix("")
    assert (completed / "data.grib").read_bytes() == PAYLOAD
    client_factory.return_value.client.get_remote.assert_not_called()
