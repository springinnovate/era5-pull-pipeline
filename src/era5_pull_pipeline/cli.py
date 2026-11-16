"""Download global ERA5 and publish hourly or complete UTC daily COG archives."""

import argparse
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
import sys

from era5_pull_pipeline.ncar_download import download_ncar
from era5_pull_pipeline.ncar_ingest import ingest_ncar
from era5_pull_pipeline.pipeline import pull_range
from era5_pull_pipeline.daily import aggregate_daily
from era5_pull_pipeline.daily_pipeline import pull_daily


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected NCAR download or archive command.

    Args:
        argv: Command-line arguments, or None to use the process arguments.

    Returns:
        Zero on success or 130 on interruption. Invalid arguments exit with two.

    Raises:
        Exception: Inventory, transfer, conversion, and storage errors propagate.
    """
    parser = argparse.ArgumentParser(prog="era5-pull", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pull = commands.add_parser(
        "pull", help="Download missing global hours and publish hourly COG files.",
        description=(
            "Fill an inclusive UTC date range, one month at a time. Skip published "
            "hours, resume unfinished downloads, and exit. Refresh requires all "
            "selected hours in each month to be available. No credentials are needed."
        ),
    )
    daily_pull = commands.add_parser(
        "pull-daily", help="Download ERA5 and publish complete UTC daily COG files.",
        description=(
            "Publish daily means and totals over an inclusive UTC date range. "
            "Use a separate daily archive and temporary hourly work directory. "
            "Incomplete days remain unpublished and can be retried."
        ),
    )
    aggregate = commands.add_parser(
        "aggregate-daily", help="Convert an existing hourly archive to UTC daily COG files.",
        description=(
            "Read an hourly archive without changing it and publish complete daily "
            "means and totals in a separate archive. Report days missing hourly inputs."
        ),
    )
    aggregate.add_argument("--source-archive", type=Path, required=True,
                           help="Existing hourly archive containing catalog.sqlite.")
    daily_pull.add_argument("--work-dir", type=Path, required=True,
                            help="Temporary hourly storage outside the daily archive; retain it for resume.")
    download = commands.add_parser(
        "download-ncar", help="Download source files for separate local ingestion.",
        description=(
            "Retain whole global NetCDF files covering an inclusive UTC date range. "
            "This command keeps all selected sources; use pull for bounded bulk backfills."
        ),
    )
    ingest = commands.add_parser(
        "ingest-ncar", help="Publish one month of recorded downloads as hourly COG files.",
        description="Ingest selected UTC hours from local NetCDF files without network access.",
    )
    for selection in (pull, daily_pull, aggregate, download):
        selection.add_argument("--start-date", type=date.fromisoformat, required=True, metavar="YYYY-MM-DD")
        selection.add_argument(
            "--end-date", type=date.fromisoformat, default=datetime.now(UTC).date(),
            metavar="YYYY-MM-DD", help="Last requested UTC date (default: today).",
        )
        selection.add_argument(
            "--variables", nargs="+", required=True, metavar="NAME",
            help="NCAR short names, for example: 2t 2d 10u 10v sp ssrd strd mtpr.",
        )
    for selection in (pull, daily_pull, download):
        selection.add_argument(
            "--download-workers", type=int, default=2,
            help="Maximum simultaneous file transfers (default: 2).",
        )
    pull.add_argument(
        "--processing-workers", type=int, default=4,
        help="Maximum simultaneous native-chunk decoding/encoding jobs (default: 4).",
    )
    pull.add_argument(
        "--staging-dir", type=Path,
        help="Local directory for new downloads and COG conversion; remembered for resume. Omit to reuse configured staging.",
    )
    pull.add_argument(
        "--upload-workers", type=int, default=2,
        help="Maximum simultaneous COG batch uploads from local staging (default: 2).",
    )
    pull.add_argument(
        "--max-source-gb", type=float, default=16,
        help="Maximum reserved NetCDF source bytes in decimal GB (default: 16); conversion needs additional space.",
    )
    pull.add_argument(
        "--max-decoded-gb", type=float, default=2,
        help="Maximum reserved decoded chunk arrays in decimal GB (default: 2); encoding buffers need additional memory.",
    )
    daily_pull.add_argument("--processing-workers", type=int, default=4,
                            help="Maximum simultaneous hourly conversion jobs (default: 4).")
    daily_pull.add_argument("--upload-workers", type=int, default=2,
                            help="Maximum simultaneous COG uploads (default: 2).")
    daily_pull.add_argument("--max-source-gb", type=float, default=16,
                            help="Maximum retained NetCDF bytes in decimal GB (default: 16).")
    daily_pull.add_argument("--max-decoded-gb", type=float, default=2,
                            help="Maximum decoded hourly chunk arrays in decimal GB (default: 2).")
    download.add_argument(
        "--dry-run", action="store_true",
        help="List source files and sizes without downloading or creating local state.",
    )
    ingest.add_argument("--year", type=int, required=True)
    ingest.add_argument("--month", type=int, required=True)
    for storage in (pull, daily_pull, aggregate, ingest):
        storage.add_argument(
            "--refresh", action="store_true", help="Replace the selected stored observations and variables.",
        )
    for command in (pull, download, ingest):
        command.add_argument(
            "--archive", type=Path, default=Path("archive"),
            help="Directory containing catalog.sqlite, staging, and COG observations (default: archive).",
        )
    for command in (daily_pull, aggregate):
        command.add_argument("--archive", type=Path, required=True,
                             help="Separate output directory for the daily catalog and COG files.")
    args = parser.parse_args(argv)
    try:
        if args.command == "pull":
            pull_range(
                args.start_date, args.end_date, args.variables, args.archive,
                download_workers=args.download_workers,
                processing_workers=args.processing_workers,
                staging_dir=args.staging_dir, upload_workers=args.upload_workers,
                max_source_gb=args.max_source_gb, max_decoded_gb=args.max_decoded_gb, refresh=args.refresh,
            )
        elif args.command == "pull-daily":
            pull_daily(
                args.start_date, args.end_date, args.variables, args.archive,
                work_dir=args.work_dir, download_workers=args.download_workers,
                processing_workers=args.processing_workers, upload_workers=args.upload_workers,
                max_source_gb=args.max_source_gb, max_decoded_gb=args.max_decoded_gb,
                refresh=args.refresh,
            )
        elif args.command == "aggregate-daily":
            aggregate_daily(
                args.start_date, args.end_date, args.variables, args.source_archive,
                args.archive, refresh=args.refresh,
            )
        elif args.command == "download-ncar":
            download_ncar(
                args.start_date, args.end_date, args.variables, args.archive,
                dry_run=args.dry_run, download_workers=args.download_workers,
            )
        else:
            published = ingest_ncar(
                args.archive, args.year, args.month, refresh=args.refresh,
            )
            print(f"Published data available at {published}.")
    except KeyboardInterrupt:
        print("Operation interrupted. Run the same command again to resume unfinished work.", file=sys.stderr)
        return 130
    return 0
