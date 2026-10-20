"""Download global ERA5 batches or check CDS access."""

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path
import sys

from era5_pull_pipeline.cds import DEFAULT_MAX_DAYS, check_access, download_batch
from era5_pull_pipeline.credentials import load_credentials


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
    batch.add_argument("--start-date", type=date.fromisoformat, required=True, metavar="YYYY-MM-DD")
    batch.add_argument("--end-date", type=date.fromisoformat, required=True, metavar="YYYY-MM-DD")
    batch.add_argument("--variables", nargs="+", required=True, metavar="NAME", help="CDS variable names.")
    batch.add_argument(
        "--max-days",
        type=int,
        default=DEFAULT_MAX_DAYS,
        help=f"Maximum days in this request (default: {DEFAULT_MAX_DAYS}); not a byte or RAM limit.",
    )
    batch.add_argument(
        "--staging",
        type=Path,
        default=Path("downloads/staging"),
        help="Batch staging directory (default: downloads/staging).",
    )
    args = parser.parse_args(argv)
    try:
        credentials = load_credentials()
        if args.command == "check-access":
            print("Requesting one global ERA5 field. CDS may queue the request; Ctrl+C stops waiting.", flush=True)
            size = check_access(credentials, args.output)
            print(f"CDS access verified: global 2m_temperature at 2000-01-01 00:00 UTC ({size} bytes).")
            print(f"Saved GRIB to {args.output}.")
        else:
            print(
                f"Requesting global ERA5 data for {args.start_date} through {args.end_date}. "
                "CDS may queue the request; Ctrl+C stops waiting.",
                flush=True,
            )
            completed = download_batch(
                credentials, args.start_date, args.end_date, args.variables,
                args.staging, args.max_days,
            )
            print(f"Saved completed batch to {completed}.")
    except KeyboardInterrupt:
        print("Download interrupted. No successful download was reported.", file=sys.stderr)
        return 130
    return 0
