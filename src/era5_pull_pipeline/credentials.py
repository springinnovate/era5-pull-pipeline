"""Load CDS credentials from the local configuration file."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class CredentialError(ValueError):
    """Report malformed credential YAML without exposing its contents."""


@dataclass(frozen=True)
class CdsCredentials:
    """Hold connection details without displaying the access token.

    Attributes:
        url: CDS API endpoint from the credential file.
        key: Personal access token excluded from the object representation.
    """

    url: str
    key: str = field(repr=False)


def load_credentials(path: Path | None = None) -> CdsCredentials:
    """Read credentials from a local .cdsapirc file.

    Args:
        path: Credential file to read, or None for .cdsapirc in the current
            working directory. No home-directory or environment fallback is used.

    Returns:
        Endpoint and access token as supplied in the file.

    Raises:
        OSError: If the file cannot be read.
        UnicodeDecodeError: If the file is not UTF-8 text.
        CredentialError: If the file contains malformed YAML.
        KeyError: If a required field is missing.
        TypeError: If the YAML is not a mapping.
    """
    credential_path = path if path is not None else Path.cwd() / ".cdsapirc"
    contents = credential_path.read_text(encoding="utf-8")
    try:
        config = yaml.safe_load(contents)
    except yaml.YAMLError:
        # Parser errors include source lines, which can contain the access token.
        raise CredentialError("Cannot parse .cdsapirc as YAML.") from None
    return CdsCredentials(url=config["url"], key=config["key"])
