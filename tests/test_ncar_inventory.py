"""Select fixed NCAR surface files from the public S3 inventory."""

from datetime import UTC, date, datetime
from typing import Any
from xml.etree import ElementTree

import multiurl.retry
import pytest
import requests

from era5_pull_pipeline import ncar


def _inventory(keys: list[str], token: str | None = None) -> requests.Response:
    """Build an S3 ListObjectsV2 response using the provider's XML fields.

    Args:
        keys: Object keys on this inventory page.
        token: Continuation token when another page follows.

    Returns:
        An anonymous HTTP inventory response.
    """
    root = ElementTree.Element(
        "ListBucketResult", xmlns="http://s3.amazonaws.com/doc/2006-03-01/"
    )
    ElementTree.SubElement(root, "IsTruncated").text = str(token is not None).lower()
    if token is not None:
        ElementTree.SubElement(root, "NextContinuationToken").text = token
    for key in keys:
        item = ElementTree.SubElement(root, "Contents")
        for name, value in {
            "Key": key,
            "Size": "1132145857",
            "ETag": '"f2f8b4fe517442e3437df1c7f3f2ecbc-52"',
            "LastModified": "2024-06-11T18:38:47.000Z",
        }.items():
            ElementTree.SubElement(item, name).text = value
    response = requests.Response()
    response.status_code = 200
    response.url = ncar.BUCKET_URL
    response._content = ElementTree.tostring(root)
    return response


def test_inventory_includes_previous_forecast_file_in_1950(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover all required variables and valid hours crossing January 1950."""
    objects: dict[str, list[str]] = {}
    analysis = "e5.oper.an.sfc"
    objects[f"{analysis}/195001/"] = [
        f"{analysis}/195001/{analysis}.{identifier}.ll025sc.1950010100_1950013123.nc"
        for identifier in (
            "128_134_sp", "128_165_10u", "128_166_10v", "128_167_2t", "128_168_2d"
        )
    ]
    for collection, identifiers in (
        ("e5.oper.fc.sfc.accumu", ("128_169_ssrd", "128_175_strd", "128_176_ssr", "128_177_str")),
        ("e5.oper.fc.sfc.meanflux", ("235_055_mtpr",)),
    ):
        for month, intervals in (
            ("194912", ("1949120106_1949121606", "1949121606_1950010106")),
            ("195001", ("1950010106_1950011606", "1950011606_1950020106")),
        ):
            objects[f"{collection}/{month}/"] = [
                f"{collection}/{month}/{collection}.{identifier}.ll025sc.{interval}.nc"
                for identifier in identifiers for interval in intervals
            ]
    prefixes: list[str] = []

    def get(url: str, *, params: dict[str, str], **kwargs: Any) -> requests.Response:
        """Return the selected source month's listing."""
        assert url == ncar.BUCKET_URL
        prefixes.append(params["prefix"])
        return _inventory(objects.get(params["prefix"], []))

    with requests.Session() as session:
        monkeypatch.setattr(session, "get", get)
        sources = ncar.discover_files(
            date(1950, 1, 1), date(1950, 1, 31),
            ["2t", "2d", "10u", "10v", "sp", "ssrd", "strd", "ssr", "str", "mtpr"],
            session,
        )

    assert len(sources) == 20
    assert set(prefixes) == set(objects)
    assert len({source.key for source in sources}) == len(sources)
    temperature = next(source for source in sources if source.variable == "2t")
    assert temperature.valid_start == datetime(1950, 1, 1, tzinfo=UTC)
    assert temperature.valid_end == datetime(1950, 1, 31, 23, tzinfo=UTC)
    assert temperature.size == 1132145857
    assert temperature.etag == '"f2f8b4fe517442e3437df1c7f3f2ecbc-52"'
    assert temperature.last_modified == "2024-06-11T18:38:47.000Z"
    precipitation = [source for source in sources if source.variable == "mtpr"]
    assert [(source.valid_start, source.valid_end) for source in precipitation] == [
        (datetime(1949, 12, 16, 7, tzinfo=UTC), datetime(1950, 1, 1, 6, tzinfo=UTC)),
        (datetime(1950, 1, 1, 7, tzinfo=UTC), datetime(1950, 1, 16, 6, tzinfo=UTC)),
        (datetime(1950, 1, 16, 7, tzinfo=UTC), datetime(1950, 2, 1, 6, tzinfo=UTC)),
    ]


def test_inventory_follows_pagination_and_selects_only_requested_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Find a requested field on a later S3 page without selecting other fields."""
    prefix = "e5.oper.an.sfc/200002/"
    tokens: list[str | None] = []

    def get(url: str, *, params: dict[str, str], **kwargs: Any) -> requests.Response:
        """Provide two inventory pages for the analysis collection."""
        if params["prefix"] != prefix:
            return _inventory([])
        token = params.get("continuation-token")
        tokens.append(token)
        if token is None:
            return _inventory([
                prefix + "e5.oper.an.sfc.128_168_2d.ll025sc.2000020100_2000022923.nc"
            ], "next-page-token")
        return _inventory([
            prefix + "e5.oper.an.sfc.128_167_2t.ll025sc.2000020100_2000022923.nc"
        ])

    with requests.Session() as session:
        monkeypatch.setattr(session, "get", get)
        sources = ncar.discover_files(date(2000, 2, 29), date(2000, 2, 29), ["2t"], session)

    assert tokens == [None, "next-page-token"]
    assert [source.variable for source in sources] == ["2t"]
    assert sources[0].valid_end == datetime(2000, 2, 29, 23, tzinfo=UTC)


def test_inventory_retains_both_forecast_files_for_midmonth_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A day spanning the 06/07 UTC split requires both half-month files."""
    prefix = "e5.oper.fc.sfc.accumu/200001/"

    def get(url: str, *, params: dict[str, str], **kwargs: Any) -> requests.Response:
        """Return the two radiation files for January."""
        if params["prefix"] != prefix:
            return _inventory([])
        return _inventory([
            prefix + f"e5.oper.fc.sfc.accumu.128_169_ssrd.ll025sc.{interval}.nc"
            for interval in ("2000010106_2000011606", "2000011606_2000020106")
        ])

    with requests.Session() as session:
        monkeypatch.setattr(session, "get", get)
        sources = ncar.discover_files(date(2000, 1, 16), date(2000, 1, 16), ["ssrd"], session)

    assert len(sources) == 2
    assert sources[0].valid_end == datetime(2000, 1, 16, 6, tzinfo=UTC)
    assert sources[1].valid_start == datetime(2000, 1, 16, 7, tzinfo=UTC)


def test_inventory_retries_transient_http_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry a temporary S3 service failure with bounded backoff."""
    calls = 0
    sleeps: list[float] = []

    def get(url: str, *, params: dict[str, str], **kwargs: Any) -> requests.Response:
        """Fail once before providing empty listings."""
        nonlocal calls
        calls += 1
        response = _inventory([])
        if calls == 1:
            response.status_code = 503
        return response

    monkeypatch.setattr(multiurl.retry.time, "sleep", sleeps.append)
    with requests.Session() as session:
        monkeypatch.setattr(session, "get", get)
        sources = ncar.discover_files(date(2000, 1, 1), date(2000, 1, 1), ["2t"], session)

    assert sources == []
    assert calls == 6
    assert sleeps == [2]


def test_inventory_rejects_reversed_dates() -> None:
    """Reject a reversed CLI date range before contacting the provider."""
    with requests.Session() as session, pytest.raises(ValueError, match="end date"):
        ncar.discover_files(date(2000, 2, 1), date(2000, 1, 1), ["2t"], session)
