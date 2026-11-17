"""Verify the public NCAR command-line inputs and error reporting."""

from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from era5_pull_pipeline import cli


@pytest.mark.parametrize("refresh", [False, True])
def test_pull_selection_and_limits(monkeypatch: pytest.MonkeyPatch, refresh: bool) -> None:
    """Pass a cross-year selection and resource limits to the bulk workflow."""
    pull = Mock()
    monkeypatch.setattr(cli, "pull_range", pull)
    args = [
        "pull", "--start-date", "1999-12-31", "--end-date", "2000-01-02",
        "--variables", "2t", "ssrd", "mtpr", "--archive", "archive",
        "--download-workers", "1", "--processing-workers", "3", "--max-source-gb", "8",
    ]
    if refresh:
        args.append("--refresh")
    assert cli.main(args) == 0
    pull.assert_called_once_with(
        date(1999, 12, 31), date(2000, 1, 2), ["2t", "ssrd", "mtpr"], Path("archive"),
        download_workers=1, processing_workers=3, staging_dir=None, upload_workers=2, max_source_gb=8, max_decoded_gb=2, refresh=refresh,
    )


def test_full_history_ends_today(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a backfill command continue through the current UTC date."""
    pull = Mock()
    monkeypatch.setattr(cli, "pull_range", pull)
    before = datetime.now(UTC).date()
    assert cli.main(["pull", "--start-date", "1950-01-01", "--variables", "2t"]) == 0
    assert pull.call_args.args[0] == date(1950, 1, 1)
    assert before <= pull.call_args.args[1] <= datetime.now(UTC).date()


@pytest.mark.parametrize("dry_run", [False, True])
def test_source_download_command(monkeypatch: pytest.MonkeyPatch, dry_run: bool) -> None:
    """Keep standalone downloads and inventory previews available."""
    download = Mock()
    monkeypatch.setattr(cli, "download_ncar", download)
    args = [
        "download-ncar", "--start-date", "1950-01-01", "--end-date", "1950-01-31",
        "--variables", "2t", "--download-workers", "3",
    ]
    if dry_run:
        args.append("--dry-run")
    assert cli.main(args) == 0
    download.assert_called_once_with(
        date(1950, 1, 1), date(1950, 1, 31), ["2t"], Path("archive"),
        dry_run=dry_run, download_workers=3,
    )


def test_local_ingestion_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select a local month and replacement behavior without inventory access."""
    ingest = Mock(return_value=Path("archive/2000/01"))
    monkeypatch.setattr(cli, "ingest_ncar", ingest)
    assert cli.main(["ingest-ncar", "--year", "2000", "--month", "1", "--refresh"]) == 0
    ingest.assert_called_once_with(Path("archive"), 2000, 1, refresh=True)


def test_interruption_explains_resume(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Return a useful exit code after the workflow preserves interrupted work."""
    monkeypatch.setattr(cli, "pull_range", Mock(side_effect=KeyboardInterrupt))
    assert cli.main(["pull", "--start-date", "2000-01-01", "--variables", "2t"]) == 130
    assert "Run the same command again" in capsys.readouterr().err


def test_failures_keep_original_details(monkeypatch: pytest.MonkeyPatch) -> None:
    """Propagate the original failure so the traceback remains useful."""
    failure = RuntimeError("NCAR source request failed")
    monkeypatch.setattr(cli, "pull_range", Mock(side_effect=failure))
    with pytest.raises(RuntimeError) as caught:
        cli.main(["pull", "--start-date", "2000-01-01", "--variables", "2t"])
    assert caught.value is failure


def test_local_staging_and_upload_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass the local storage and bounded upload settings to pull."""
    pull = Mock()
    monkeypatch.setattr(cli, "pull_range", pull)
    assert cli.main(["pull", "--start-date", "2000-01-01", "--variables", "2t",
                     "--staging-dir", "local-cache", "--upload-workers", "3"]) == 0
    assert pull.call_args.kwargs["staging_dir"] == Path("local-cache")
    assert pull.call_args.kwargs["upload_workers"] == 3


def test_daily_pull_cli_dates_and_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward explicit daily selection and temporary-storage limits."""
    pull = Mock()
    monkeypatch.setattr(cli, "pull_daily", pull)
    assert cli.main([
        "pull-daily", "--start-date", "2024-02-29", "--end-date", "2024-03-01",
        "--variables", "2t", "mtpr", "--archive", "daily", "--work-dir", "work",
        "--download-workers", "3", "--processing-workers", "2", "--upload-workers", "1",
        "--max-source-gb", "20", "--max-decoded-gb", "4", "--refresh",
    ]) == 0
    pull.assert_called_once_with(
        date(2024, 2, 29), date(2024, 3, 1), ["2t", "mtpr"], Path("daily"),
        work_dir=Path("work"), download_workers=3, processing_workers=2,
        upload_workers=1, max_source_gb=20, max_decoded_gb=4, refresh=True,
    )


def test_daily_conversion_cli_keeps_source_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass the read-only hourly source and daily destination independently."""
    aggregate = Mock()
    monkeypatch.setattr(cli, "aggregate_daily", aggregate)
    assert cli.main([
        "aggregate-daily", "--start-date", "1999-12-31", "--end-date", "2000-01-01",
        "--variables", "2t", "ssrd", "--source-archive", "hourly", "--archive", "daily",
    ]) == 0
    aggregate.assert_called_once_with(
        date(1999, 12, 31), date(2000, 1, 1), ["2t", "ssrd"], Path("hourly"),
        Path("daily"), refresh=False,
    )


@pytest.mark.parametrize("command", ["pull-daily", "aggregate-daily"])
def test_daily_commands_require_explicit_output(command: str) -> None:
    """Avoid accidentally writing daily products into the default hourly archive."""
    with pytest.raises(SystemExit) as caught:
        cli.main([command, "--start-date", "2000-01-01", "--variables", "2t"])
    assert caught.value.code == 2


def test_build_summary_reports_existing_archive_metadata(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Build the requested archive summary and report observation/group counts."""
    rebuild = Mock(return_value={"observations": 48, "variables": 2, "groups": 2})
    monkeypatch.setattr(cli, "rebuild_observation_summary", rebuild)
    assert cli.main(["build-summary", "--archive", "existing-archive"]) == 0
    rebuild.assert_called_once_with(Path("existing-archive"))
    output = capsys.readouterr().out
    assert "observations=48" in output and "variables=2" in output and "groups=2" in output


def test_build_summary_requires_explicit_archive() -> None:
    """Require an existing archive selection before starting a summary build."""
    with pytest.raises(SystemExit) as caught:
        cli.main(["build-summary"])
    assert caught.value.code == 2
