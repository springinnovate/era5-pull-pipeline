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
        "--download-workers", "1", "--max-source-gb", "8",
    ]
    if refresh:
        args.append("--refresh")
    assert cli.main(args) == 0
    pull.assert_called_once_with(
        date(1999, 12, 31), date(2000, 1, 2), ["2t", "ssrd", "mtpr"], Path("archive"),
        download_workers=1, max_source_gb=8, refresh=refresh,
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
