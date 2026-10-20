"""Read live availability metadata through bounded offline HTTP retries."""

import json
from typing import Any

import multiurl.retry
import pytest
import requests

from era5_pull_pipeline import cds


@pytest.mark.parametrize("failure", [503, requests.ConnectionError("offline")])
def test_availability_follows_constraints_link_with_backoff(
    monkeypatch: pytest.MonkeyPatch, failure: int | Exception,
) -> None:
    """Retry metadata reads, following the current link instead of an extent cutoff."""
    constraints_url = "https://provider.example.test/current-constraints.json"
    cubes = [{
        "year": ["2026"], "month": ["10"], "day": ["14"], "time": ["23:00"],
        "product_type": ["reanalysis"], "variable": ["2m_temperature"],
    }]
    attempts: dict[str, int] = {}
    sleeps: list[int] = []

    def request(self: requests.Session, method: str, url: str, **kwargs: Any) -> requests.Response:
        """Serve transient failures followed by current collection metadata."""
        assert method.upper() == "GET"
        assert kwargs["timeout"] == 30
        assert self.auth is None
        attempts[url] = attempts.get(url, 0) + 1
        if url == cds.CATALOGUE_URL and attempts[url] < 3 and isinstance(failure, Exception):
            raise failure
        response = requests.Response()
        response.url = url
        response.status_code = failure if url == cds.CATALOGUE_URL and attempts[url] < 3 else 200
        response._content = json.dumps(
            {"links": [{"rel": "constraints", "href": constraints_url}],
             "extent": {"temporal": {"interval": [["1940-01-01T00:00:00Z", "2026-10-14T00:00:00Z"]]}}}
            if url == cds.CATALOGUE_URL else cubes
        ).encode()
        return response

    monkeypatch.setattr(requests.Session, "request", request)
    monkeypatch.setattr(multiurl.retry.time, "sleep", sleeps.append)
    assert cds.load_availability() == cubes
    assert attempts == {cds.CATALOGUE_URL: 3, constraints_url: 1}
    assert sleeps == [2, 4]


@pytest.mark.parametrize(("status", "attempts"), [(503, 3), (403, 1)])
def test_availability_failure_does_not_become_empty_coverage(
    monkeypatch: pytest.MonkeyPatch, status: int, attempts: int,
) -> None:
    """Keep provider outages and permanent rejections distinct from no data."""
    calls: list[str] = []

    def request(self: requests.Session, method: str, url: str, **kwargs: Any) -> requests.Response:
        calls.append(url)
        response = requests.Response()
        response.status_code = status
        response.url = url
        response._content = b"{}"
        return response

    monkeypatch.setattr(requests.Session, "request", request)
    monkeypatch.setattr(multiurl.retry.time, "sleep", lambda delay: None)
    with pytest.raises(requests.HTTPError):
        cds.load_availability()
    assert calls == [cds.CATALOGUE_URL] * attempts
