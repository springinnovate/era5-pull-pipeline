"""Discover global hourly ERA5 files in NCAR's public S3 inventory."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import re
from xml.etree import ElementTree

from multiurl import robust
import requests

BUCKET_URL = "https://nsf-ncar-era5.s3.us-west-2.amazonaws.com"
SURFACE_COLLECTIONS = (
    "e5.oper.an.sfc",
    "e5.oper.fc.sfc.accumu",
    "e5.oper.fc.sfc.meanflux",
)
FILE_NAME = re.compile(
    r"\.\d+_\d+_(?P<variable>[^.]+)\.ll025sc\."
    r"(?P<start>\d{10})_(?P<end>\d{10})\.nc$"
)


@dataclass(frozen=True)
class SourceFile:
    """Identify a provider file and the inclusive valid UTC hours it contains.

    Attributes:
        key: Object key relative to the public NCAR bucket.
        variable: NCAR's short variable name, taken from its filename.
        size: Object size in bytes.
        etag: Quoted S3 identity token, not necessarily an MD5 checksum.
        last_modified: Provider modification timestamp in S3's ISO format.
        valid_start: First valid UTC hour in the file.
        valid_end: Last valid UTC hour in the file.
    """

    key: str
    variable: str
    size: int
    etag: str
    last_modified: str
    valid_start: datetime
    valid_end: datetime


def _list_month(
    session: requests.Session, collection: str, month: date
) -> Iterator[SourceFile]:
    """Read every inventory page for one surface collection and source month.

    Args:
        session: HTTP session used for anonymous inventory requests.
        collection: NCAR surface collection prefix.
        month: Calendar month in the provider's directory structure.

    Yields:
        Full global 0.25-degree NetCDF files with provider identities and times.

    Raises:
        Exception: HTTP and inventory parsing errors propagate after bounded
            retries for transient HTTP failures.
    """
    get = robust(session.get, maximum_tries=3, retry_after=(2, 30, 2))
    params = {"list-type": "2", "prefix": f"{collection}/{month:%Y%m}/"}
    while True:
        with get(BUCKET_URL, params=params, timeout=30) as response:
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
        for item in root.findall("{*}Contents"):
            key = item.findtext("{*}Key")
            match = FILE_NAME.search(key)
            if match is None:
                continue
            start = datetime.strptime(match["start"], "%Y%m%d%H").replace(tzinfo=UTC)
            end = datetime.strptime(match["end"], "%Y%m%d%H").replace(tzinfo=UTC)
            if collection.startswith("e5.oper.fc."):
                # Forecast filenames begin at the first initialization (06 UTC).
                # Their last timestamp is the final valid hour, not another init.
                start += timedelta(hours=1)
            yield SourceFile(
                key=key,
                variable=match["variable"],
                size=int(item.findtext("{*}Size")),
                etag=item.findtext("{*}ETag"),
                last_modified=item.findtext("{*}LastModified"),
                valid_start=start,
                valid_end=end,
            )
        token = root.findtext("{*}NextContinuationToken")
        if token is None:
            break
        params["continuation-token"] = token


def discover_files(
    start: date, end: date, variables: Sequence[str], session: requests.Session
) -> list[SourceFile]:
    """Find provider files overlapping the requested dates and source variables.

    Dates select valid hours rather than file creation or forecast initialization
    dates. Whole files can contain additional hours outside the selection.

    Args:
        start: First requested UTC date, included.
        end: Last requested UTC date, included.
        variables: NCAR filename short names, for example ``2t``, ``ssrd``, or
            ``mtpr``. Names and parameter identifiers come from the inventory.
        session: HTTP session used for anonymous inventory requests.

    Returns:
        Matching files ordered by first valid hour and variable. Unavailable
        dates or names have no matching files.

    Raises:
        ValueError: The end date precedes the start date.
        Exception: HTTP and inventory parsing errors propagate.
    """
    if end < start:
        raise ValueError("The end date must be on or after the start date.")
    selected = set(variables)
    first_hour = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
    stop = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    files = []
    for collection in SURFACE_COLLECTIONS:
        month = start.replace(day=1)
        if collection.startswith("e5.oper.fc."):
            # The preceding file supplies the first seven hours of a month.
            month = (month - timedelta(days=1)).replace(day=1)
        while month <= end:
            files.extend(
                source for source in _list_month(session, collection, month)
                if source.variable in selected
                and source.valid_start < stop
                and source.valid_end >= first_hour
            )
            month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
    return sorted(files, key=lambda source: (source.valid_start, source.variable, source.key))
