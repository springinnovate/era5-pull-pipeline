"""Select available missing ERA5 hours in bounded single-month requests."""

from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
import sqlite3
from typing import Any

from era5_pull_pipeline import cds
from era5_pull_pipeline.catalog import missing_fields


def _group_requests(
    request: dict[str, Any], missing: dict[str, list[int]], max_days: int
) -> list[dict[str, Any]]:
    """Partition exact missing fields into bounded Cartesian selections.

    Args:
        request: Single-month selection providing the request settings.
        missing: Requested CDS variable names and their missing UTC epoch hours.
        max_days: Maximum calendar span of each returned request.

    Returns:
        Nonoverlapping requests respecting the day span and CDS field limit.
    """
    requests = []
    first_day, last_day = min(map(int, request["day"])), max(map(int, request["day"]))
    for first in range(first_day, last_day + 1, max_days):
        selections: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = defaultdict(list)
        for name in request["variable"]:
            days: dict[str, list[str]] = defaultdict(list)
            for timestamp in sorted(missing.get(name, [])):
                time = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=timestamp)
                if first <= time.day < first + max_days:
                    days[time.strftime("%d")].append(time.strftime("%H:%M"))
            matching_hours: dict[tuple[str, ...], list[str]] = defaultdict(list)
            for day, hours in days.items():
                matching_hours[tuple(hours)].append(day)
            for hours, selected_days in matching_hours.items():
                selections[(tuple(selected_days), hours)].append(name)
        for (days, hours), names in selections.items():
            for hour_start in range(0, len(hours), cds.MAX_REQUEST_FIELDS):
                selected_hours = list(hours[hour_start:hour_start + cds.MAX_REQUEST_FIELDS])
                variable_limit = cds.MAX_REQUEST_FIELDS // len(selected_hours)
                for variable_start in range(0, len(names), variable_limit):
                    selected_names = names[variable_start:variable_start + variable_limit]
                    day_limit = cds.MAX_REQUEST_FIELDS // (len(selected_names) * len(selected_hours))
                    for day_start in range(0, len(days), day_limit):
                        requests.append({
                            **request,
                            "variable": selected_names,
                            "day": list(days[day_start:day_start + day_limit]),
                            "time": selected_hours,
                        })
    return requests


def missing_requests(db: sqlite3.Connection, request: dict[str, Any]) -> list[dict[str, Any]]:
    """Group missing fields without requesting stored hours.

    Args:
        db: Catalog connection protected by the archive writer lock.
        request: Bounded single-month request produced by build_batch_request.

    Returns:
        Requests containing exactly the missing variable/hour combinations.

    Raises:
        sqlite3.Error: Reading published request coverage fails.
    """
    year, month = int(request["year"][0]), int(request["month"][0])
    times = [
        int(datetime.fromisoformat(
            f"{year:04d}-{month:02d}-{int(day):02d}T{hour}:00+00:00"
        ).timestamp())
        for day in request["day"] for hour in request["time"]
    ]
    missing = missing_fields(db, year, month, request["variable"], times)
    return _group_requests(request, missing, 31)


def month_requests(
    db: sqlite3.Connection,
    start: date,
    end: date,
    variables: Sequence[str],
    max_days: int = cds.DEFAULT_MAX_DAYS,
    availability: list[dict[str, list[str]]] | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Plan available missing hours for one month and count omitted fields.

    Args:
        db: Catalog connection protected by the archive writer lock.
        start: First UTC date, included.
        end: Last UTC date, included and in the same month.
        variables: CDS single-level variable names.
        max_days: Maximum calendar span of each download batch.
        availability: Provider constraint cubes, or None for offline planning
            without an availability filter.

    Returns:
        Download requests, the number of already stored variable/hour pairs,
        and the number of missing pairs unavailable from the provider.

    Raises:
        ValueError: Dates, day limit, or variable selection are invalid, or a
            variable is absent from the provider's reanalysis constraints.
        sqlite3.Error: Reading published request coverage fails.
    """
    if max_days < 1:
        raise ValueError("max_days must be at least 1.")
    if end < start:
        raise ValueError("The end date must be on or after the start date.")
    if (start.year, start.month) != (end.year, end.month):
        raise ValueError("Monthly planning must stay within one calendar month.")
    if not variables or any(not variable for variable in variables):
        raise ValueError("Select at least one nonempty CDS variable name.")
    names = sorted(set(variables))
    # Build the common settings from one field's day before partitioning larger
    # selections; the complete month may exceed either download limit.
    request = cds.build_batch_request(start, start, names[:1])
    request["variable"] = names
    request["day"] = [f"{day:02d}" for day in range(start.day, end.day + 1)]
    first_hour = int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp())
    times = [first_hour + hour * 3600 for hour in range(((end - start).days + 1) * 24)]
    missing = missing_fields(db, start.year, start.month, names, times)
    missing_count = sum(len(hours) for hours in missing.values())
    skipped = len(names) * len(times) - missing_count
    unavailable = 0
    if availability is not None:
        cubes = [cube for cube in availability if "reanalysis" in cube["product_type"]]
        known = {name for cube in cubes for name in cube["variable"]}
        unknown = set(names) - known
        if unknown:
            raise ValueError("Unknown CDS reanalysis variable(s): " + ", ".join(sorted(unknown)))
        available: dict[str, set[int]] = {name: set() for name in missing}
        for cube in cubes:
            if str(start.year) not in cube["year"] or f"{start.month:02d}" not in cube["month"]:
                continue
            selected_names = available.keys() & cube["variable"]
            if not selected_names:
                continue
            selected_days = set(request["day"]) & set(cube["day"])
            hours = {
                int(datetime.fromisoformat(
                    f"{start.year:04d}-{start.month:02d}-{day}T{hour}:00+00:00"
                ).timestamp())
                for day in selected_days for hour in cube["time"]
            }
            for name in selected_names:
                available[name].update(hours)
        missing = {
            name: sorted(set(hours) & available[name])
            for name, hours in missing.items()
        }
        unavailable = missing_count - sum(len(hours) for hours in missing.values())
    return _group_requests(request, missing, max_days), skipped, unavailable


def subtract_requests(
    requests: Sequence[dict[str, Any]], completed: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Remove completed selections from an already bounded download plan.

    Args:
        requests: Available missing requests for a single month.
        completed: Verified staged selections from that month.

    Returns:
        Requests covering exactly the remaining variable/hour combinations.
    """
    covered: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for request in completed:
        for name in request["variable"]:
            covered[name].update((day, hour) for day in request["day"] for hour in request["time"])
    remaining = []
    for request in requests:
        year, month = int(request["year"][0]), int(request["month"][0])
        missing = {
            name: [
                int(datetime.fromisoformat(
                    f"{year:04d}-{month:02d}-{day}T{hour}:00+00:00"
                ).timestamp())
                for day in request["day"] for hour in request["time"]
                if (day, hour) not in covered[name]
            ]
            for name in request["variable"]
        }
        remaining.extend(_group_requests(request, missing, 31))
    return remaining
