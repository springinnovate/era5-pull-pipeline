"""Select exact missing ERA5 fields for a bounded single-month request."""

from collections import defaultdict
from datetime import UTC, datetime, timedelta
import sqlite3
from typing import Any

from era5_pull_pipeline.catalog import missing_fields
from era5_pull_pipeline.variables import PARAMETER_IDS


def missing_requests(db: sqlite3.Connection, request: dict[str, Any]) -> list[dict[str, Any]]:
    """Group missing fields into CDS selections without requesting stored hours.

    Args:
        db: Catalog connection protected by the archive writer lock.
        request: Bounded single-month request produced by build_batch_request.

    Returns:
        Requests grouping days with identical missing hour sets, and variables
        with identical day/hour selections. Empty means the selection is stored.

    Raises:
        KeyError: A requested CDS variable is absent from the parameter mapping.
    """
    year, month = int(request["year"][0]), int(request["month"][0])
    parameters = {name: f"param_{PARAMETER_IDS[name]}" for name in request["variable"]}
    times = [
        int(datetime.fromisoformat(
            f"{year:04d}-{month:02d}-{int(day):02d}T{hour}:00+00:00"
        ).timestamp())
        for day in request["day"] for hour in request["time"]
    ]
    missing = missing_fields(db, year, month, list(parameters.values()), times)
    selections: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = defaultdict(list)
    for name, parameter in parameters.items():
        days: dict[str, list[str]] = defaultdict(list)
        for timestamp in sorted(missing.get(parameter, [])):
            time = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=timestamp)
            days[time.strftime("%d")].append(time.strftime("%H:%M"))
        matching_hours: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for day, hours in days.items():
            matching_hours[tuple(hours)].append(day)
        for hours, selected_days in matching_hours.items():
            selections[(tuple(selected_days), hours)].append(name)
    return [
        {**request, "variable": names, "day": list(days), "time": list(hours)}
        for (days, hours), names in selections.items()
    ]
