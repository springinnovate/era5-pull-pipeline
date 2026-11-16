"""Download global daily ERA5 temperature means and precipitation totals from CDS.

Select inclusive UTC dates with --start-date and --end-date. Each monthly
request returns daily NetCDF data in a ZIP archive, using hourly input samples
on the global 0.25-degree grid. CDS credentials stay in its normal external
configuration. See the README for installation and usage.
"""

from __future__ import annotations

import argparse
from calendar import monthrange
from collections.abc import Iterator, Sequence
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile


DATASET: str = "derived-era5-single-levels-daily-statistics"
PRODUCTS: tuple[tuple[str, str], ...] = (
    ("2m_temperature", "daily_mean"),
    ("total_precipitation", "daily_sum"),
)


def iter_month_ranges(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Yield the inclusive part of each calendar month in an ordered date range.

    Args:
        start: First selected UTC date, no later than end.
        end: Last selected UTC date.

    Yields:
        First and last selected dates within one calendar month.
    """
    first = start
    while first <= end:
        last = min(end, date(first.year, first.month, monthrange(first.year, first.month)[1]))
        yield first, last
        if last == end:
            break
        first = last + timedelta(days=1)


def main(argv: Sequence[str] | None = None) -> int:
    """Download or preview monthly requests for daily ERA5 temperature and precipitation.

    Existing ZIP files are treated as completed and skipped. Downloads use a
    temporary file beside the destination and replace it only after success;
    failed or interrupted temporary files are removed. Dry runs print JSON
    requests without loading credentials, contacting CDS, or writing files.

    Args:
        argv: Command-line arguments, or None to use process arguments.

    Returns:
        Zero after all selected requests have completed or been previewed.

    Raises:
        SystemExit: Arguments are invalid or the optional cdsapi is unavailable.
        Exception: CDS and filesystem failures propagate without reporting the
            affected download as completed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True, metavar="YYYY-MM-DD")
    parser.add_argument(
        "--end-date", type=date.fromisoformat, metavar="YYYY-MM-DD",
        help="Last requested UTC date, inclusive (default: start date).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Directory for ZIP archives (default: era5_daily_<selected dates>).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print requests without downloading, loading credentials, or writing files.",
    )
    args = parser.parse_args(argv)
    start: date = args.start_date
    end: date = args.end_date or start
    if end < start:
        parser.error("--end-date must be on or after --start-date.")
    selection = str(start) if start == end else f"{start}_to_{end}"
    output_dir: Path = args.output_dir or Path(f"era5_daily_{selection}")
    client = None

    for first, last in iter_month_ranges(start, end):
        dates = str(first) if first == last else f"{first}_to_{last}"
        for variable, statistic in PRODUCTS:
            request: dict[str, str | list[str]] = {
                "product_type": "reanalysis",
                "variable": [variable],
                "year": str(first.year),
                "month": [f"{first.month:02d}"],
                "day": [f"{day:02d}" for day in range(first.day, last.day + 1)],
                "daily_statistic": statistic,
                "time_zone": "utc+00:00",
                "frequency": "1_hourly",
            }
            target = output_dir / f"era5_global_{dates}_{variable}_{statistic}.zip"
            if args.dry_run:
                print(json.dumps({"dataset": DATASET, "request": request, "output": str(target)}))
                continue
            if target.is_file():
                print(f"Skipped existing: {target}", flush=True)
                continue
            if client is None:
                try:
                    import cdsapi
                except ModuleNotFoundError as exc:
                    if exc.name != "cdsapi":
                        raise
                    raise SystemExit(
                        'Install the client first: python -m pip install "cdsapi>=0.7.7"'
                    ) from exc
                client = cdsapi.Client()
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {variable}: {statistic} for {dates} (UTC)", flush=True)
            with tempfile.NamedTemporaryFile(
                dir=output_dir, prefix=f"{target.name}.", suffix=".part", delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                client.retrieve(DATASET, request).download(str(temporary_path))
                temporary_path.replace(target)
            finally:
                temporary_path.unlink(missing_ok=True)
            print(f"Saved: {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
