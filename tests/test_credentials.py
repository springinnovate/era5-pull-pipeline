"""Check local CDS credential loading without contacting the service."""

import os
from pathlib import Path
import traceback

import pytest
import yaml

from era5_pull_pipeline.credentials import load_credentials


def test_load_quoted_yaml_with_comments(tmp_path: Path) -> None:
    """Read YAML syntax, preserve supplied values, and hide the token in repr."""
    path = tmp_path / ".cdsapirc"
    path.write_text(
        '# Local connection settings\nurl: "https://example.test/api///"\n'
        "key: 'test-token-not-a-uuid' # Access token\n",
        encoding="utf-8",
    )

    credentials = load_credentials(path)

    assert credentials.url == "https://example.test/api///"
    assert credentials.key == "test-token-not-a-uuid"
    assert credentials.key not in repr(credentials)


def test_default_path_uses_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load only the current directory without changing environment settings."""
    (tmp_path / ".cdsapirc").write_text(
        "url: https://example.test/api\nkey: local-test-token\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CDSAPI_URL", "https://environment.test/api")
    monkeypatch.setenv("CDSAPI_KEY", "environment-test-token")

    credentials = load_credentials()

    assert credentials.url == "https://example.test/api"
    assert credentials.key == "local-test-token"
    assert os.environ["CDSAPI_URL"] == "https://environment.test/api"
    assert os.environ["CDSAPI_KEY"] == "environment-test-token"


def test_missing_local_file_does_not_use_home_or_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report a missing local file even if fallback credentials are available."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".cdsapirc").write_text(
        "url: https://example.test/api\nkey: home-test-token\n", encoding="utf-8"
    )
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("CDSAPI_URL", "https://example.test/api")
    monkeypatch.setenv("CDSAPI_KEY", "environment-test-token")
    monkeypatch.setenv("CDSAPI_RC", str(home / ".cdsapirc"))

    with pytest.raises(FileNotFoundError, match=".cdsapirc"):
        load_credentials()


@pytest.mark.parametrize("field", ["url", "key"])
def test_reject_missing_fields(tmp_path: Path, field: str) -> None:
    """Let a missing required field raise the normal mapping error."""
    config = {"url": "https://example.test/api", "key": "test-token"}
    config.pop(field)
    path = tmp_path / ".cdsapirc"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(KeyError, match=field):
        load_credentials(path)


def test_yaml_errors_do_not_disclose_content(tmp_path: Path) -> None:
    """Suppress parser diagnostics that would otherwise display credential lines."""
    path = tmp_path / ".cdsapirc"
    path.write_text(
        "url: https://example.test/api\nkey: [secret-test-token\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Cannot parse") as caught:
        load_credentials(path)

    assert "secret-test-token" not in "".join(
        traceback.format_exception(caught.value)
    )
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
