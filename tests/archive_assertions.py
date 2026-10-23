"""Read published COG values through the consumer-facing catalog contract."""

from calendar import monthrange
from collections import defaultdict
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
from numpy.typing import NDArray
import rasterio


def read_raster_values(archive: Path, year: int, month: int) -> dict[str, NDArray[Any]]:
    """Stack a month's actual observations by variable in timestamp order.

    Args:
        archive: Root containing the SQLite catalog and COG files.
        year: UTC calendar year.
        month: UTC calendar month.

    Returns:
        Arrays with time, row, and column axes, excluding absent observations.
    """
    first = datetime(year, month, 1, tzinfo=UTC)
    stop = first + timedelta(days=monthrange(year, month)[1])
    values: dict[str, list[NDArray[Any]]] = defaultdict(list)
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
        for variable, path, band in db.execute(
            """SELECT variable, path, band FROM observations
               WHERE valid_time >= ? AND valid_time < ? ORDER BY variable, valid_time""",
            (int(first.timestamp()), int(stop.timestamp())),
        ):
            with rasterio.open(archive / path) as raster:
                values[variable].append(raster.read(band))
    return {variable: np.stack(arrays) for variable, arrays in values.items()}
