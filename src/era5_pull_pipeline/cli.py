"""Download global ERA5 from NCAR and maintain a compressed hourly COG archive."""

import argparse
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
import sys

from era5_pull_pipeline.ncar_download import download_ncar
from era5_pull_pipeline.ncar_ingest import ingest_ncar
from era5_pull_pipeline.pipeline import pull_range


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
    for selection in (pull, download):
        selection.add_argument("--start-date", type=date.fromisoformat, required=True, metavar="YYYY-MM-DD")
        selection.add_argument(
            "--end-date", type=date.fromisoformat, default=datetime.now(UTC).date(),
            metavar="YYYY-MM-DD", help="Last requested UTC date (default: today).",
        )
        selection.add_argument(
            "--variables", nargs="+", required=True, metavar="NAME",
            help="NCAR short names, for example: 2t 2d 10u 10v sp ssrd strd mtpr.",
        )
        selection.add_argument(
            "--download-workers", type=int, default=2,
            help="Maximum simultaneous file transfers (default: 2).",
        )
    pull.add_argument(
        "--processing-workers", type=int, default=4,
        help="Maximum simultaneous native-chunk decoding/encoding jobs (default: 4).",
    )
    pull.add_argument(
        "--max-source-gb", type=float, default=16,
        help="Maximum reserved NetCDF source bytes in decimal GB (default: 16); conversion needs additional space.",
    )
    pull.add_argument(
        "--max-decoded-gb", type=float, default=2,
        help="Maximum reserved decoded chunk arrays in decimal GB (default: 2); encoding buffers need additional memory.",
    )
    download.add_argument(
        "--dry-run", action="store_true",
        help="List source files and sizes without downloading or creating local state.",
    )
    ingest.add_argument("--year", type=int, required=True)
    ingest.add_argument("--month", type=int, required=True)
    for storage in (pull, ingest):
        storage.add_argument(
            "--refresh", action="store_true", help="Replace the selected stored hours and variables.",
        )
    for command in (pull, download, ingest):
        command.add_argument(
            "--archive", type=Path, default=Path("archive"),
            help="Directory containing catalog.sqlite, staging, and COG observations (default: archive).",
        )
    args = parser.parse_args(argv)
    try:
        if args.command == "pull":
            pull_range(
                args.start_date, args.end_date, args.variables, args.archive,
                download_workers=args.download_workers,
                processing_workers=args.processing_workers,
                max_source_gb=args.max_source_gb, max_decoded_gb=args.max_decoded_gb, refresh=args.refresh,
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
