"""Load CDS credentials from the local configuration file."""

from dataclasses import dataclass, field
from pathlib import Path
import unicodedata
from urllib.parse import urlsplit

import yaml


class CredentialError(ValueError):
    """Indicate that local CDS credentials could not be read or validated."""


@dataclass(frozen=True)
class CdsCredentials:
    """Hold validated connection details without displaying the access token.

    Attributes:
        url: HTTPS CDS API endpoint without a trailing slash.
        key: Personal access token excluded from the object representation.
    """

    url: str
    key: str = field(repr=False)


def _has_space_or_control(value: str) -> bool:
    """Return whether a value contains whitespace or Unicode control characters.

    Args:
        value: Text to inspect.

    Returns:
        Whether any character is whitespace or a control character.
    """
    return any(
        character.isspace() or unicodedata.category(character) == "Cc"
        for character in value
    )


def _validate_url(value: str) -> str:
    """Validate and normalize a CDS API endpoint without exposing its contents.

    Args:
        value: Endpoint from the credential file.

    Returns:
        Validated endpoint without trailing slashes.

    Raises:
        CredentialError: If the endpoint is not a valid HTTPS URL.
    """
    invalid_message = (
        "Set 'url' in .cdsapirc to an HTTPS API URL with a hostname and no "
        "embedded credentials, whitespace, query, or fragment."
    )
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme == "https"
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and "?" not in value
            and "#" not in value
            and not _has_space_or_control(value)
        )
        # Accessing the port validates its syntax and range.
        _ = parsed.port
    except ValueError:
        raise CredentialError(invalid_message) from None
    if not valid:
        raise CredentialError(invalid_message)
    return value.rstrip("/")


def load_credentials(path: Path | None = None) -> CdsCredentials:
    """Read and validate credentials from a local .cdsapirc file.

    Args:
        path: Credential file to read, or None for .cdsapirc in the current
            working directory. No home-directory or environment fallback is used.

    Returns:
        Validated endpoint and personal access token.

    Raises:
        CredentialError: If the file is missing, unreadable, malformed, or
            contains invalid connection details.
    """
    try:
        credential_path = path if path is not None else Path.cwd() / ".cdsapirc"
        contents = credential_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise CredentialError(
            "Credential file not found. Create .cdsapirc in the working "
            "directory with 'url' and 'key'."
        ) from None
    except (OSError, UnicodeError):
        raise CredentialError(
            "Cannot read the credential file. Check its permissions and save "
            ".cdsapirc as UTF-8 text."
        ) from None

    try:
        config = yaml.safe_load(contents)
    except yaml.YAMLError:
        raise CredentialError(
            "Cannot parse .cdsapirc. Use a YAML mapping with 'url' and 'key'."
        ) from None

    if not isinstance(config, dict):
        raise CredentialError(
            "Use a YAML mapping with 'url' and 'key' in .cdsapirc."
        )
    url = config.get("url")
    key = config.get("key")
    if not isinstance(url, str) or not url:
        raise CredentialError("Set 'url' in .cdsapirc to a nonempty HTTPS API URL.")
    if not isinstance(key, str) or not key:
        raise CredentialError(
            "Set 'key' in .cdsapirc to a nonempty personal access token."
        )
    normalized_url = _validate_url(url)
    if ":" in key or _has_space_or_control(key):
        raise CredentialError(
            "Set 'key' in .cdsapirc to a personal access token without "
            "whitespace, control characters, or a legacy UID prefix."
        )
    return CdsCredentials(url=normalized_url, key=key)
