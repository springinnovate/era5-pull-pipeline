"""Keep the automated test suite independent of CDS and real credentials."""

import pytest
import requests


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail accidental HTTP requests unless a test installs its own transport."""

    def reject_request(*args: object, **kwargs: object) -> None:
        """Reject HTTP traffic from an offline test."""
        raise AssertionError("Automated tests must not contact CDS.")

    monkeypatch.setattr(requests.Session, "request", reject_request)
