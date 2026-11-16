"""Check the standalone CDS helper's date selection and completed ZIP outputs."""

from collections.abc import Callable, Sequence
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


main: Callable[[Sequence[str]], int] = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "fetch_era5.py")
)["main"]


@pytest.mark.parametrize(
    ("start", "end", "months"),
    [
        ("2024-02-28", "2024-03-01", [("2024", "02", ["28", "29"]), ("2024", "03", ["01"])]),
        ("2025-12-31", "2026-01-02", [("2025", "12", ["31"]), ("2026", "01", ["01", "02"])]),
        ("2025-02-28", "2025-03-01", [("2025", "02", ["28"]), ("2025", "03", ["01"])]),
    ],
)
def test_dry_run_groups_only_selected_calendar_days(
    start: str, end: str, months: list[tuple[str, str, list[str]]],
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Preview valid month/year/leap-day selections with no credentials or writes."""
    monkeypatch.setitem(sys.modules, "cdsapi", None)
    output = tmp_path / "not-created"
    assert main(["--start-date", start, "--end-date", end, "--output-dir", str(output), "--dry-run"]) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == len(months) * 2
    for index, (year, month, days) in enumerate(months):
        for record, (variable, statistic) in zip(
            records[index * 2:index * 2 + 2],
            [("2m_temperature", "daily_mean"), ("total_precipitation", "daily_sum")],
            strict=True,
        ):
            assert record["dataset"] == "derived-era5-single-levels-daily-statistics"
            assert record["request"] == {
                "product_type": "reanalysis", "variable": [variable],
                "year": year, "month": [month], "day": days,
                "daily_statistic": statistic, "time_zone": "utc+00:00", "frequency": "1_hourly",
            }
            first = f"{year}-{month}-{days[0]}"
            last = f"{year}-{month}-{days[-1]}"
            dates = first if first == last else f"{first}_to_{last}"
            assert Path(record["output"]) == output / f"era5_global_{dates}_{variable}_{statistic}.zip"
    assert not output.exists()


def test_default_end_and_output_directory_follow_start_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Default to exactly the supplied day, including directory and filenames."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "cdsapi", None)
    assert main(["--start-date", "2026-10-02", "--dry-run"]) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 2
    assert records[0]["request"]["day"] == ["02"]
    assert Path(records[0]["output"]) == Path(
        "era5_daily_2026-10-02/era5_global_2026-10-02_2m_temperature_daily_mean.zip"
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("arguments", [
    [], ["--start-date", "2025-02-29"], ["--start-date", "not-a-date"],
    ["--start-date", "2026-10-01", "--end-date", "2026-10-32"],
    ["--start-date", "2026-10-02", "--end-date", "2026-10-01"],
])
def test_invalid_dates_fail_before_output_creation(
    arguments: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject absent, invalid, and reversed dates at the command-line boundary."""
    monkeypatch.setitem(sys.modules, "cdsapi", None)
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as caught:
        main([*arguments, "--output-dir", str(output)])
    assert caught.value.code == 2
    assert not output.exists()


def test_completed_downloads_publish_then_skip_without_loading_cds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish completed bytes atomically and avoid requests on an exact repeat."""
    output = tmp_path / "daily"
    client = Mock()

    def download(target: str) -> str:
        """Write a fake service response before the final ZIP becomes visible."""
        path = Path(target)
        assert path.parent == output
        assert path.suffix == ".part"
        assert len(list(output.glob("*.zip"))) == client.retrieve.call_count - 1
        Path(target).write_bytes(b"completed service response")
        return target

    client.retrieve.return_value.download.side_effect = download
    factory = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, "cdsapi", SimpleNamespace(Client=factory))
    arguments = ["--start-date", "2024-02-29", "--output-dir", str(output)]
    assert main(arguments) == 0
    factory.assert_called_once_with()
    assert client.retrieve.call_count == 2
    files = sorted(output.iterdir())
    assert len(files) == 2
    assert all(path.suffix == ".zip" and path.read_bytes() == b"completed service response" for path in files)
    monkeypatch.setitem(sys.modules, "cdsapi", None)
    assert main(arguments) == 0
    assert client.retrieve.call_count == 2
    assert sorted(output.iterdir()) == files


@pytest.mark.parametrize("failure", [RuntimeError("service failed"), KeyboardInterrupt()])
@pytest.mark.parametrize("failed_request", [1, 2])
def test_failed_download_has_no_completed_target_and_can_retry(
    failure: BaseException, failed_request: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Remove partial output after failure or interruption and retry missing data."""
    client = Mock()

    def interrupted_download(target: str) -> None:
        """Leave a partial response and simulate an interrupted transfer."""
        if client.retrieve.call_count == failed_request:
            Path(target).write_bytes(b"partial")
            raise failure
        Path(target).write_bytes(b"completed earlier response")

    client.retrieve.return_value.download.side_effect = interrupted_download
    monkeypatch.setitem(sys.modules, "cdsapi", SimpleNamespace(Client=Mock(return_value=client)))
    arguments = ["--start-date", "2026-10-02", "--output-dir", str(tmp_path)]
    with pytest.raises(type(failure)) as caught:
        main(arguments)
    assert caught.value is failure
    completed_before_retry = list(tmp_path.glob("*.zip"))
    assert len(completed_before_retry) == failed_request - 1
    assert not list(tmp_path.glob("*.part"))
    assert capsys.readouterr().out.count("Saved:") == failed_request - 1

    def completed_download(target: str) -> str:
        """Write a complete response for the repeated selection."""
        Path(target).write_bytes(b"complete")
        return target

    client.retrieve.return_value.download.side_effect = completed_download
    assert main(arguments) == 0
    assert len(list(tmp_path.glob("*.zip"))) == 2
    assert not list(tmp_path.glob("*.part"))
    assert client.retrieve.call_count == 3
    assert all(path.read_bytes() == b"completed earlier response" for path in completed_before_retry)


def test_failed_publication_does_not_leave_a_completed_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Propagate filesystem publication errors without claiming a saved archive."""
    client = Mock()

    def download(target: str) -> str:
        """Write a completed service response before filesystem publication."""
        Path(target).write_bytes(b"complete")
        return target

    client.retrieve.return_value.download.side_effect = download
    monkeypatch.setitem(sys.modules, "cdsapi", SimpleNamespace(Client=Mock(return_value=client)))
    failure = OSError("destination unavailable")
    monkeypatch.setattr(Path, "replace", Mock(side_effect=failure))
    with pytest.raises(OSError) as caught:
        main(["--start-date", "2026-10-02", "--output-dir", str(tmp_path)])
    assert caught.value is failure
    assert list(tmp_path.iterdir()) == []
    assert "Saved:" not in capsys.readouterr().out


def test_missing_optional_client_has_installation_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explain the optional dependency before creating an output directory."""
    monkeypatch.setitem(sys.modules, "cdsapi", None)
    with pytest.raises(SystemExit, match="pip install"):
        main(["--start-date", "2026-10-02", "--output-dir", str(tmp_path / "daily")])
    assert list(tmp_path.iterdir()) == []
