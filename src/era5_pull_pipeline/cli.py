"""Command-line access checks for the global ERA5 pipeline."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from era5_pull_pipeline.cds import AccessCheckError, check_access
from era5_pull_pipeline.credentials import CredentialError, load_credentials


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested command and report a credential-free result.

    Args:
        argv: Command-line arguments, or None to use the process arguments.

    Returns:
        Zero on success, one on failure, or 130 on interruption. Argument
        parsing exits with status two when arguments are invalid.
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
        help="New GRIB path (default: downloads/era5-access-check.grib).",
    )
    args = parser.parse_args(argv)
    try:
        credentials = load_credentials()
        print("Requesting one global ERA5 field. CDS may queue the request; Ctrl+C stops waiting.", flush=True)
        size = check_access(credentials, args.output)
    except (CredentialError, AccessCheckError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Access check interrupted. No successful download was reported.", file=sys.stderr)
        return 130
    print(f"CDS access verified: global 2m_temperature at 2000-01-01 00:00 UTC ({size} bytes).")
    print(f"Saved GRIB to {args.output}.")
    return 0
