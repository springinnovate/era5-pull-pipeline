"""Verify command-line behavior without contacting CDS."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from era5_pull_pipeline import cli
from era5_pull_pipeline.cds import AccessCheckError


def test_missing_credentials_are_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Report missing local credentials without attempting a request."""
    monkeypatch.chdir(tmp_path)
    check = Mock()
    monkeypatch.setattr(cli, "check_access", check)
    assert cli.main(["check-access"]) == 1
    check.assert_not_called()
    assert "Create .cdsapirc" in capsys.readouterr().err


@pytest.mark.parametrize("outcome", [32, AccessCheckError("CDS denied access."), KeyboardInterrupt()])
def test_cli_reports_success_failure_and_interruption(
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
        assert code == (130 if isinstance(outcome, KeyboardInterrupt) else 1)
        assert "CDS access verified" not in output.out


def test_cli_rejects_geographic_selection() -> None:
    """Keep clipping and extent selection outside the command's inputs."""
    with pytest.raises(SystemExit) as caught:
        cli.main(["check-access", "--area", "0,0,1,1"])
    assert caught.value.code == 2
