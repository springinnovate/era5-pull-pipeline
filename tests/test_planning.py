"""Check that CDS Cartesian selections contain exactly the missing fields."""

from contextlib import closing
from collections.abc import Iterator
from datetime import UTC, date, datetime
from itertools import product
import json
import sqlite3
from typing import Any
from unittest.mock import Mock

import pytest

from era5_pull_pipeline import cds, planning


@pytest.mark.parametrize("year", [1950, 1995])
@pytest.mark.parametrize(
    ("missing_hours", "request_count"),
    [
        ({"2m_temperature": [], "10m_u_component_of_wind": [], "10m_v_component_of_wind": []}, 0),
        (
            {
                "2m_temperature": [0, 1, 2, 24, 25, 26, 48, 49, 50],
                "10m_u_component_of_wind": [],
                "10m_v_component_of_wind": [0, 1, 2, 24, 25, 26, 48, 49, 50],
            },
            1,
        ),
        (
            {
                "2m_temperature": [0, 2, 25, 48, 50],
                "10m_u_component_of_wind": [0, 2, 25, 48, 50],
                "10m_v_component_of_wind": [1, 24, 26, 49],
            },
            4,
        ),
    ],
)
def test_requests_cover_only_missing_cartesian_combinations(
    monkeypatch: pytest.MonkeyPatch,
    missing_hours: dict[str, list[int]],
    request_count: int,
    year: int,
) -> None:
    """Group shared selections without inventing cross-products across different gaps."""
    start = int(datetime(year, 3, 1, tzinfo=UTC).timestamp())
    missing = {
        name: [start + offset * 3600 for offset in hours]
        for name, hours in missing_hours.items()
    }
    monkeypatch.setattr(planning, "missing_fields", Mock(return_value=missing))
    request = cds.build_batch_request(date(year, 3, 1), date(year, 3, 3), list(missing_hours))
    request["time"] = ["00:00", "01:00", "02:00"]
    with closing(sqlite3.connect(":memory:")) as db:
        selections = planning.missing_requests(db, request)
    actual = [
        (
            name,
            int(datetime.fromisoformat(f"{year}-03-{day}T{hour}:00+00:00").timestamp()),
        )
        for selection in selections
        for name, day, hour in product(
            selection["variable"], selection["day"], selection["time"]
        )
    ]
    expected = {
        (name, timestamp)
        for name, timestamps in missing.items()
        for timestamp in timestamps
    }
    assert set(actual) == expected
    assert len(actual) == len(expected)
    assert len(selections) == request_count
    assert all(selection["grid"] == [0.25, 0.25] for selection in selections)
    assert all("area" not in selection for selection in selections)


@pytest.fixture
def catalog() -> Iterator[sqlite3.Connection]:
    """Provide actual request-coverage storage without creating raster files."""
    with closing(sqlite3.connect(":memory:")) as db:
        db.row_factory = sqlite3.Row
        db.execute("CREATE TABLE revisions (year INTEGER, month INTEGER, status TEXT, provenance_json TEXT)")
        yield db


def _fields(requests: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Expand planned rectangles to compare exact variable/day/hour coverage."""
    return [
        field for request in requests
        for field in product(request["variable"], request["day"], request["time"])
    ]


@pytest.mark.parametrize("year", [1950, 2000])
@pytest.mark.parametrize("refresh", [False, True])
def test_month_plan_splits_day_windows_and_counts_stored_hours(
    catalog: sqlite3.Connection, year: int, refresh: bool,
) -> None:
    """Split a whole month and include stored coverage only for a refresh."""
    stored = cds.build_batch_request(date(year, 2, 1), date(year, 2, 2), ["2m_temperature"])
    catalog.execute("INSERT INTO revisions VALUES (?, 2, 'published', ?)", (year, json.dumps({"request": stored})))
    last_day = 29 if year == 2000 else 28
    requests, skipped, unavailable = planning.month_requests(
        catalog, date(year, 2, 1), date(year, 2, last_day), ["2m_temperature"],
        max_days=7, refresh=refresh,
    )
    fields = _fields(requests)
    first_day = 1 if refresh else 3
    assert len(fields) == len(set(fields)) == (last_day - first_day + 1) * 24
    assert skipped == (0 if refresh else 48)
    assert unavailable == 0
    assert {day for _, day, _ in fields} == {f"{day:02d}" for day in range(first_day, last_day + 1)}
    assert all(max(map(int, request["day"])) - min(map(int, request["day"])) < 7 for request in requests)
    assert all(request["grid"] == [0.25, 0.25] and "area" not in request for request in requests)


def test_availability_intersects_cubes_without_cross_products(
    catalog: sqlite3.Connection,
) -> None:
    """Keep partially available final hours distinct for each provider variable."""
    common = {"year": ["2026"], "month": ["10"], "product_type": ["reanalysis"]}
    availability = [
        {**common, "variable": ["2m_temperature"], "day": ["13"], "time": [f"{hour:02d}:00" for hour in range(24)]},
        {**common, "variable": ["2m_temperature"], "day": ["14"], "time": ["00:00", "01:00"]},
        {**common, "variable": ["total_precipitation"], "day": ["14"], "time": ["02:00"]},
    ]
    requests, skipped, unavailable = planning.month_requests(
        catalog, date(2026, 10, 13), date(2026, 10, 15),
        ["2m_temperature", "total_precipitation"], availability=availability,
    )
    expected = {
        ("2m_temperature", "13", f"{hour:02d}:00") for hour in range(24)
    } | {("2m_temperature", "14", "00:00"), ("2m_temperature", "14", "01:00"), ("total_precipitation", "14", "02:00")}
    actual = _fields(requests)
    assert set(actual) == expected
    assert len(actual) == len(expected)
    assert skipped == 0
    assert unavailable == 144 - 27


def test_refresh_selects_only_requested_dates_and_variables(catalog: sqlite3.Connection) -> None:
    """Refreshing a subset does not redownload the rest of a published month."""
    stored = cds.build_batch_request(
        date(2000, 1, 1), date(2000, 1, 3), ["2m_temperature", "total_precipitation"],
    )
    catalog.execute(
        "INSERT INTO revisions VALUES (2000, 1, 'published', ?)",
        (json.dumps({"request": stored}),),
    )
    requests, skipped, unavailable = planning.month_requests(
        catalog, date(2000, 1, 2), date(2000, 1, 2), ["2m_temperature"], refresh=True,
    )
    assert _fields(requests) == [
        ("2m_temperature", "02", f"{hour:02d}:00") for hour in range(24)
    ]
    assert (skipped, unavailable) == (0, 0)


@pytest.mark.parametrize("refresh", [False, True])
def test_provider_availability_for_stored_hours(
    catalog: sqlite3.Connection, refresh: bool,
) -> None:
    """Refresh needs provider availability even for already published fields."""
    request = cds.build_batch_request(date(1950, 1, 1), date(1950, 1, 1), ["2m_temperature"])
    catalog.execute("INSERT INTO revisions VALUES (1950, 1, 'published', ?)", (json.dumps({"request": request}),))
    availability = [{
        "year": ["1950"], "month": ["01"], "day": ["02"], "time": ["00:00"],
        "variable": ["2m_temperature"], "product_type": ["reanalysis"],
    }]
    requests, skipped, unavailable = planning.month_requests(
        catalog, date(1950, 1, 1), date(1950, 1, 2), ["2m_temperature"], availability=availability,
        refresh=refresh,
    )
    assert _fields(requests) == [("2m_temperature", "02", "00:00")]
    assert (skipped, unavailable) == ((0, 47) if refresh else (24, 23))
    assert planning.month_requests(catalog, date(1950, 1, 1), date(1950, 1, 1), ["2m_temperature"]) == ([], 24, 0)


def test_unknown_provider_variable_is_not_treated_as_unavailable(catalog: sqlite3.Connection) -> None:
    """Distinguish misspelled names from known variables outside available dates."""
    availability = [{
        "year": ["2000"], "month": ["01"], "day": ["01"], "time": ["00:00"],
        "variable": ["provider_defined_name"], "product_type": ["reanalysis"],
    }]
    with pytest.raises(ValueError, match="Unknown CDS.*typo"):
        planning.month_requests(catalog, date(2000, 1, 1), date(2000, 1, 1), ["typo"], availability=availability)
    assert planning.month_requests(
        catalog, date(2000, 2, 1), date(2000, 2, 1), ["provider_defined_name"], availability=availability,
    ) == ([], 0, 24)


@pytest.mark.parametrize("field_limit", [48, 120_000])
@pytest.mark.parametrize("refresh", [False, True])
def test_month_plan_splits_fields_before_batch_validation(
    catalog: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch, field_limit: int,
    refresh: bool,
) -> None:
    """Partition oversized variable selections instead of rejecting the month."""
    monkeypatch.setattr(cds, "MAX_REQUEST_FIELDS", field_limit)
    names = [f"provider_variable_{index}" for index in range(field_limit // 24 + 1)]
    requests, skipped, unavailable = planning.month_requests(
        catalog, date(2000, 1, 1), date(2000, 1, 2), names, max_days=2, refresh=refresh,
    )
    actual = _fields(requests)
    assert len(actual) == len(set(actual)) == len(names) * 48
    assert all(len(request["variable"]) * len(request["day"]) * len(request["time"]) <= field_limit for request in requests)
    assert (skipped, unavailable) == (0, 0)


def test_subtract_completed_partial_selection_from_expanded_availability(
    catalog: sqlite3.Connection,
) -> None:
    """Reuse earlier completed hours when newer availability changes a request."""
    requests, _, _ = planning.month_requests(
        catalog, date(2000, 1, 1), date(2000, 1, 2), ["2m_temperature", "total_precipitation"],
    )
    completed = [{**requests[0], "variable": ["2m_temperature"], "day": ["01"], "time": [f"{hour:02d}:00" for hour in range(7)]}]
    remaining = planning.subtract_requests(requests, completed)
    actual = _fields(remaining)
    assert set(actual) == set(_fields(requests)) - set(_fields(completed))
    assert len(actual) == len(set(actual)) == 96 - 7
    assert planning.subtract_requests(requests, requests) == []


@pytest.mark.parametrize(
    ("start", "end", "variables", "max_days"),
    [
        (date(2000, 1, 2), date(2000, 1, 1), ["temp"], 7),
        (date(2000, 1, 31), date(2000, 2, 1), ["temp"], 7),
        (date(2000, 1, 1), date(2000, 1, 1), ["temp"], 0),
        (date(2000, 1, 1), date(2000, 1, 1), [], 7),
    ],
)
def test_invalid_month_plan(
    catalog: sqlite3.Connection, start: date, end: date, variables: list[str], max_days: int,
) -> None:
    """Reject dates and limits that cannot describe a monthly download plan."""
    with pytest.raises(ValueError):
        planning.month_requests(catalog, start, end, variables, max_days)
