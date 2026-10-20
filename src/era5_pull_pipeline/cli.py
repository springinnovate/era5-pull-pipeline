"""Download and archive global ERA5 date ranges or check CDS access."""

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path
import sys
from uuid import uuid4

from era5_pull_pipeline.cds import DEFAULT_MAX_DAYS, build_batch_request, check_access, download_batch
from era5_pull_pipeline.credentials import load_credentials
from era5_pull_pipeline.pipeline import ingest_batch, pull_batch, pull_range


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested command.

    Args:
        argv: Command-line arguments, or None to use the process arguments.

    Returns:
        Zero on success or 130 on interruption.
        Argument parsing exits with status two when arguments are invalid.

    Raises:
        Exception: File, configuration, CDS, or decoding errors propagate.
    """
    parser = argparse.ArgumentParser(prog="era5-pull", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    access = commands.add_parser(
        "check-access",
        help="Retrieve one global ERA5 field using the working directory's .cdsapirc.",
        description=(
            "Check CDS access by downloading global 2 m temperature at "
            "2000-01-01 00:00 UTC. CDS may queue the request; press Ctrl+C to stop."
        ),
    )
    access.add_argument(
        "--output",
        type=Path,
        default=Path("downloads/era5-access-check.grib"),
        help="GRIB path, overwritten on each check (default: downloads/era5-access-check.grib).",
    )
    batch = commands.add_parser(
        "download-batch",
        help="Download selected variables for one batch of complete UTC days.",
        description=(
            "Download all 24 hourly global fields per selected day. Dates are "
            "inclusive and must stay within one calendar month and --max-days."
        ),
    )
    pull = commands.add_parser(
        "pull-batch",
        help="Download, validate, and publish one batch to a monthly Zarr archive.",
        description=(
            "Fill missing variables and hourly fields within one bounded "
            "calendar-month interval. Stored fields are preserved and skipped."
        ),
    )
    date_range = commands.add_parser(
        "pull",
        help="Fill a date range using resumable batches and monthly publication.",
        description=(
            "Download missing global fields across an inclusive UTC date range, "
            "or replace selected fields with --refresh. Reuse completed staged "
            "work, publish each month once, and exit. Refresh requires all "
            "selected fields in each month to be available."
        ),
    )
    for selection in (batch, pull, date_range):
        selection.add_argument("--start-date", type=date.fromisoformat, required=True, metavar="YYYY-MM-DD")
        selection.add_argument("--end-date", type=date.fromisoformat, required=True, metavar="YYYY-MM-DD")
        selection.add_argument("--variables", nargs="+", required=True, metavar="NAME", help="CDS variable names.")
    for selection in (batch, pull):
        selection.add_argument(
            "--max-days",
            type=int,
            default=DEFAULT_MAX_DAYS,
            help=f"Maximum days in this request (default: {DEFAULT_MAX_DAYS}); not a byte or RAM limit.",
        )
    date_range.add_argument(
        "--batch-days", type=int, default=DEFAULT_MAX_DAYS,
        help=f"Maximum days per download batch (default: {DEFAULT_MAX_DAYS}); not a byte or RAM limit.",
    )
    date_range.add_argument(
        "--refresh", action="store_true",
        help="Download the selected dates and variables again, replacing their stored fields.",
    )
    batch.add_argument(
        "--staging",
        type=Path,
        default=Path("downloads/staging"),
        help="Batch staging directory (default: downloads/staging).",
    )
    ingest = commands.add_parser(
        "ingest-batch",
        help="Publish an existing completed download without contacting CDS.",
    )
    ingest.add_argument("--batch", type=Path, required=True, help="Completed download directory.")
    for storage in (pull, ingest, date_range):
        storage.add_argument(
            "--archive", type=Path, default=Path("archive"),
            help="Archive directory (default: archive).",
        )
        storage.add_argument(
            "--spatial-chunks", nargs=2, type=int, default=(32, 32),
            metavar=("LAT", "LON"), help="Grid cells per spatial read chunk (default: 32 32).",
        )
    args = parser.parse_args(argv)
    try:
        if args.command == "check-access":
            credentials = load_credentials()
            print("Requesting one global ERA5 field. CDS may queue the request; Ctrl+C stops waiting.", flush=True)
            size = check_access(credentials, args.output)
            print(f"CDS access verified: global 2m_temperature at 2000-01-01 00:00 UTC ({size} bytes).")
            print(f"Saved GRIB to {args.output}.")
        elif args.command == "download-batch":
            credentials = load_credentials()
            print(
                f"Requesting global ERA5 data for {args.start_date} through {args.end_date}. "
                "CDS may queue the request; Ctrl+C stops waiting.",
                flush=True,
            )
            completed = download_batch(
                credentials,
                build_batch_request(args.start_date, args.end_date, args.variables, args.max_days),
                args.staging / f"{args.start_date}_{args.end_date}_{uuid4().hex}",
            )
            print(f"Saved completed batch to {completed}.")
        elif args.command == "pull":
            pull_range(
                args.start_date, args.end_date, args.variables, args.archive,
                tuple(args.spatial_chunks), args.batch_days,
                refresh=args.refresh,
            )
        elif args.command == "pull-batch":
            print(
                f"Checking archive coverage for {args.start_date} through {args.end_date}. "
                "If a download is needed, CDS may queue it; Ctrl+C stops waiting.",
                flush=True,
            )
            published = pull_batch(
                args.start_date, args.end_date, args.variables, args.archive,
                tuple(args.spatial_chunks), args.max_days,
            )
            print(f"Published data available at {published}.")
        else:
            print(f"Ingesting completed batch {args.batch}.", flush=True)
            published = ingest_batch(args.batch, args.archive, tuple(args.spatial_chunks))
            print(f"Published data available at {published}.")
    except KeyboardInterrupt:
        print("Operation interrupted. Run it again to recover any pending publication.", file=sys.stderr)
        return 130
    return 0
