"""Check that CDS Cartesian selections contain exactly the missing fields."""

from contextlib import closing
from datetime import UTC, date, datetime
from itertools import product
import sqlite3
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
