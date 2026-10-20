"""Check local CDS credential loading without contacting the service."""

from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import traceback

import pytest
import yaml

from era5_pull_pipeline.credentials import (
    CdsCredentials,
    CredentialError,
    load_credentials,
)


def test_load_quoted_yaml_with_comments(tmp_path: Path) -> None:
    """Read ordinary YAML syntax and normalize the API endpoint."""
    path = tmp_path / ".cdsapirc"
    path.write_text(
        '# Local connection settings\nurl: "https://example.test/api///"\n'
        "key: 'test-token-not-a-uuid' # Access token\n",
        encoding="utf-8",
    )

    credentials = load_credentials(path)

    assert credentials.url == "https://example.test/api"
    assert credentials.key == "test-token-not-a-uuid"
    assert credentials.key not in repr(credentials)


def test_credentials_are_immutable() -> None:
    """Prevent callers from replacing validated connection details."""
    credentials = CdsCredentials(url="https://example.test/api", key="test-token")

    with pytest.raises(FrozenInstanceError):
        credentials.key = "replacement"  # type: ignore[misc]


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

    with pytest.raises(CredentialError, match="not found"):
        load_credentials()


@pytest.mark.parametrize(
    "contents",
    ["", "null", "[]", "a string", "false", "42", "- url\n- key"],
)
def test_reject_non_mapping_yaml(tmp_path: Path, contents: str) -> None:
    """Require named configuration fields rather than arbitrary YAML values."""
    path = tmp_path / ".cdsapirc"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(CredentialError, match="mapping"):
        load_credentials(path)


@pytest.mark.parametrize("field", ["url", "key"])
@pytest.mark.parametrize("value", [None, "", 123, False, [], {}])
def test_reject_empty_or_non_string_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    """Require nonempty string values for both connection fields."""
    config: dict[str, object] = {
        "url": "https://example.test/api",
        "key": "test-token",
    }
    config[field] = value
    path = tmp_path / ".cdsapirc"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(CredentialError, match=field):
        load_credentials(path)


@pytest.mark.parametrize("field", ["url", "key"])
def test_reject_missing_fields(tmp_path: Path, field: str) -> None:
    """Explain which required connection field must be supplied."""
    config = {"url": "https://example.test/api", "key": "test-token"}
    config.pop(field)
    path = tmp_path / ".cdsapirc"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(CredentialError, match=field):
        load_credentials(path)


@pytest.mark.parametrize(
    "url",
    [
        " ",
        "http://example.test/api",
        "file:///api",
        "https:///api",
        "https://",
        "https://user:secret-test-token@example.test/api",
        "https://user@example.test/api",
        "https://example.test/api?key=secret-test-token",
        "https://example.test/api#secret-test-token",
        "https://example.test/api?",
        "https://example.test/api#",
        "https://example.test:invalid/api",
        "https://example.test:70000/api",
        "https://[broken/api",
        " https://example.test/api",
        "https://example.test/ api",
        "https://example.test/a\tpi",
        "https://example.test/\x00api",
    ],
)
def test_reject_unsafe_urls(tmp_path: Path, url: str) -> None:
    """Reject unsafe endpoints without echoing their potentially secret content."""
    path = tmp_path / ".cdsapirc"
    path.write_text(
        yaml.safe_dump({"url": url, "key": "secret-test-token"}), encoding="utf-8"
    )

    with pytest.raises(CredentialError, match="HTTPS") as caught:
        load_credentials(path)

    assert "secret-test-token" not in str(caught.value)
    assert "secret-test-token" not in repr(caught.value)
    assert url not in str(caught.value) or url.isspace()


@pytest.mark.parametrize(
    "key",
    [
        " ",
        " secret-test-token",
        "secret-test-token ",
        "secret test-token",
        "secret-test-token\n",
        "secret-test-token\t",
        "secret-test-token\x00",
        "secret-test-token\x7f",
        "secret-test-token\u0080",
        "12345:secret-test-token",
    ],
)
def test_reject_invalid_tokens(tmp_path: Path, key: str) -> None:
    """Reject whitespace, controls, and legacy keys without disclosing tokens."""
    path = tmp_path / ".cdsapirc"
    path.write_text(
        yaml.safe_dump({"url": "https://example.test/api", "key": key}),
        encoding="utf-8",
    )

    with pytest.raises(CredentialError, match="personal access token") as caught:
        load_credentials(path)

    assert "secret-test-token" not in str(caught.value)


def test_yaml_errors_do_not_disclose_content(tmp_path: Path) -> None:
    """Suppress parser diagnostics that would otherwise display credential lines."""
    path = tmp_path / ".cdsapirc"
    path.write_text(
        "url: https://example.test/api\nkey: [secret-test-token\n", encoding="utf-8"
    )

    with pytest.raises(CredentialError, match="Cannot parse") as caught:
        load_credentials(path)

    assert "secret-test-token" not in "".join(
        traceback.format_exception(caught.value)
    )
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


def test_missing_file(tmp_path: Path) -> None:
    """Explain how to provide a missing credential file."""
    with pytest.raises(CredentialError, match="Create .cdsapirc"):
        load_credentials(tmp_path / ".cdsapirc")


def test_unreadable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hide operating-system errors while explaining file access requirements."""

    def deny_read(self: Path, **kwargs: object) -> str:
        """Simulate a permission failure without relying on platform ACLs."""
        raise PermissionError("secret-test-token")

    monkeypatch.setattr(Path, "read_text", deny_read)

    with pytest.raises(CredentialError, match="permissions") as caught:
        load_credentials(tmp_path / ".cdsapirc")

    assert "secret-test-token" not in "".join(
        traceback.format_exception(caught.value)
    )


def test_non_utf8_file(tmp_path: Path) -> None:
    """Reject non-UTF-8 contents with a safe encoding explanation."""
    path = tmp_path / ".cdsapirc"
    path.write_bytes(b"key: secret-test-token\xff")

    with pytest.raises(CredentialError, match="UTF-8") as caught:
        load_credentials(path)

    assert "secret-test-token" not in str(caught.value)
