"""Verify command-line behavior without contacting CDS."""

from pathlib import Path
from datetime import date
from unittest.mock import Mock

import pytest

from era5_pull_pipeline import cli


def test_missing_credentials_raise_before_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Let the missing-file error identify the local configuration path."""
    monkeypatch.chdir(tmp_path)
    check = Mock()
    monkeypatch.setattr(cli, "check_access", check)
    with pytest.raises(FileNotFoundError, match=".cdsapirc"):
        cli.main(["check-access"])
    check.assert_not_called()


@pytest.mark.parametrize("outcome", [32, KeyboardInterrupt()])
def test_cli_reports_success_and_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: int | BaseException,
) -> None:
    """Use the local file and return useful status codes without exposing its token."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".cdsapirc").write_text(
        "url: https://example.test/api\nkey: secret-test-token\n", encoding="utf-8"
    )
    check = Mock(return_value=outcome) if isinstance(outcome, int) else Mock(side_effect=outcome)
    monkeypatch.setattr(cli, "check_access", check)
    code = cli.main(["check-access", "--output", "probe.grib"])
    output = capsys.readouterr()
    assert "secret-test-token" not in output.out + output.err
    assert check.call_args.args[0].key == "secret-test-token"
    assert check.call_args.args[1] == Path("probe.grib")
    if isinstance(outcome, int):
        assert code == 0
        assert "CDS access verified" in output.out
        assert "Saved GRIB to probe.grib" in output.out
    else:
        assert code == 130
        assert "CDS access verified" not in output.out


def test_service_errors_retain_original_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep service failures available for diagnosis without reporting success."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".cdsapirc").write_text(
        "url: https://example.test/api\nkey: test-token\n", encoding="utf-8"
    )
    failure = RuntimeError("CDS job failed with provider details")
    monkeypatch.setattr(cli, "check_access", Mock(side_effect=failure))
    with pytest.raises(RuntimeError) as caught:
        cli.main(["check-access"])
    assert caught.value is failure
    assert "CDS access verified" not in capsys.readouterr().out


def test_cli_rejects_geographic_selection() -> None:
    """Keep clipping and extent selection outside the command's inputs."""
    with pytest.raises(SystemExit) as caught:
        cli.main(["check-access", "--area", "0,0,1,1"])
    assert caught.value.code == 2


def test_download_batch_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pass the selected dates, variables, and batch settings to the downloader."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".cdsapirc").write_text(
        "url: https://example.test/api\nkey: test-token\n", encoding="utf-8"
    )
    completed = Path("staging/completed-batch")
    download = Mock(return_value=completed)
    monkeypatch.setattr(cli, "download_batch", download)

    assert cli.main([
        "download-batch",
        "--start-date", "2000-01-01",
        "--end-date", "2000-01-02",
        "--variables", "2m_temperature", "total_precipitation",
        "--max-days", "2",
        "--staging", "staging",
    ]) == 0

    credentials, *selection = download.call_args.args
    assert credentials.key == "test-token"
    assert selection == [
        cli.build_batch_request(
            date(2000, 1, 1), date(2000, 1, 2),
            ["2m_temperature", "total_precipitation"], 2,
        ),
        Path("staging"),
    ]
    assert f"Saved completed batch to {completed}" in capsys.readouterr().out


def test_archive_commands_do_not_load_credentials_before_catalog_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Allow offline ingestion and repeat requests without a local token."""
    monkeypatch.chdir(tmp_path)
    published = Path("archive/2000/01/era5.zarr")
    pull = Mock(return_value=published)
    ingest = Mock(return_value=published)
    monkeypatch.setattr(cli, "pull_batch", pull)
    monkeypatch.setattr(cli, "ingest_batch", ingest)

    assert cli.main([
        "pull-batch", "--start-date", "2000-01-01", "--end-date", "2000-01-01",
        "--variables", "2m_temperature", "--spatial-chunks", "16", "32",
    ]) == 0
    pull.assert_called_once_with(
        date(2000, 1, 1), date(2000, 1, 1), ["2m_temperature"],
        Path("archive"), (16, 32), 7,
    )
    assert cli.main(["ingest-batch", "--batch", "completed"]) == 0
    ingest.assert_called_once_with(Path("completed"), Path("archive"), (32, 32))
    assert capsys.readouterr().out.count(f"Published data available at {published}") == 2


def test_range_command_accepts_month_boundaries_and_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose resumable range processing with separate batching and storage inputs."""
    pull = Mock()
    monkeypatch.setattr(cli, "pull_range", pull)
    assert cli.main([
        "pull", "--start-date", "1999-12-31", "--end-date", "2000-01-02",
        "--variables", "2m_temperature", "total_precipitation", "--batch-days", "1",
        "--archive", "downloads/manual-archive-13", "--spatial-chunks", "16", "32",
    ]) == 0
    pull.assert_called_once_with(
        date(1999, 12, 31), date(2000, 1, 2), ["2m_temperature", "total_precipitation"],
        Path("downloads/manual-archive-13"), (16, 32), 1,
    )
