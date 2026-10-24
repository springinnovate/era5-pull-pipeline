# ERA5 pull pipeline

A Python command-line pipeline that downloads full global ERA5 rasters from [NCAR's public archive](https://registry.opendata.aws/nsf-ncar-era5/) and writes one tiled, losslessly compressed Cloud Optimized GeoTIFF (COG) per variable and observation timestamp. Each invocation fills a requested date range and set of variables, publishes the observations, and exits. Repeated runs skip stored hours and resume unfinished downloads. No account or access token is required.

## Installation

Use Python 3.11 or newer. From the repository directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

The examples below use `python` from the environment where the package is installed. On Windows, `.\.venv\Scripts\python.exe` selects the environment explicitly; on macOS or Linux, use `.venv/bin/python`. The installed `era5-pull` command is equivalent to `python -m era5_pull_pipeline`.

## Download a date range

This example retrieves three UTC days of temperature, solar radiation, and precipitation across a year boundary:

```powershell
python -m era5_pull_pipeline pull --start-date 1999-12-31 --end-date 2000-01-02 --variables 2t ssrd mtpr --archive downloads/manual-cog
```

It publishes 216 GeoTIFFs under `1999/12/` and `2000/01/`: one for each of three variables at each selected hour. Every weather field covers the full 721 × 1440 global grid at 0.25° resolution. The pipeline performs no geographic selection, temporal aggregation, or derived ET0 calculation.

Repeat the same command to check resume behavior. Once the selection is complete, it prints `completed=0, skipped=216` and makes no NCAR calls. Counts are **variable-hours**: three variables × three days × 24 hours = 216. The summary also reports unavailable hours, uncompleted hours in an interrupted month, and remaining pending hours.

To add wind for those dates, run the same command with `--variables 10u 10v`. Existing observation files remain unchanged. Extend the dates to fill additional months or interior gaps.

| Option | Meaning |
| --- | --- |
| `--start-date` | First requested UTC date, inclusive; required. |
| `--end-date` | Last requested UTC date, inclusive; defaults to today in UTC. |
| `--variables` | One or more NCAR short variable names; required. |
| `--archive` | Directory containing dataset.yaml, the catalog, staging, and GeoTIFFs; defaults to `archive`. |
| `--download-workers` | Maximum simultaneous file transfers; defaults to 2. |
| `--processing-workers` | Maximum simultaneous native-chunk decoding/encoding jobs; defaults to 4. |
| `--max-decoded-gb` | Maximum reserved decoded chunk arrays in decimal GB; defaults to 2. Encoding buffers and process overhead need additional memory. |
| `--max-source-gb` | Maximum reserved NetCDF source bytes in decimal GB; defaults to 16. Conversion requires additional space. |
| `--refresh` | Retrieve current sources and replace selected stored fields. |

NCAR packages analysis fields as one variable per month, and radiation and precipitation in roughly half-month files. Selecting a single day still downloads whole source files; only selected hours become GeoTIFFs. A forecast file can supply hours on both sides of a month boundary and is downloaded once, then retained until both selected months are published.

Transfers show bytes, speed, elapsed time, and estimated remaining transfer time. Month messages identify availability lookup, overlapping download/processing, independent native decoding chunks, and publication. There is no overall backfill completion estimate.

## Backfill from 1950

To download the selected variables from 1950 through the latest data available from NCAR:

```powershell
python -m era5_pull_pipeline pull --start-date 1950-01-01 --variables 2t 2d 10u 10v sp ssrd strd mtpr --archive archive
```

NCAR updates monthly and typically trails real time by about 3–4 months. Availability is checked separately for each variable. Unavailable hours remain absent from coverage and are checked again on subsequent runs, so the same command extends the archive as NCAR adds data. An explicit `--end-date` limits the requested interval.

The workflow processes one month at a time and prefetches future source files while the current month is processing. Lookahead advances month by month until the source budget fills or an unavailable month is reached. At most `--download-workers` transfers are active across current and prefetched months. Each new file reserves its full size before transfer, including partial and retained sources. Prefetch uses only spare `--max-source-gb` capacity; files that do not fit wait until current-month sources are released. If the required current month cannot fit, the command reports the required size before starting its new transfers. Increase `--max-source-gb` or select fewer variables.

Completed or reused sources are divided at their native NetCDF temporal chunk boundaries. Separate processes decode and encode those chunks concurrently, including multiple chunks from the same source. A job reads its native chunks once, keeps its decoded arrays in memory, and writes validated hourly COGs. There is no full-month decoding barrier or decoded scratch file in `pull`. Decoding, encoding, downloads, and future-month availability lookup can overlap. A dedicated publication thread owns the archive writer lock, SQLite connection, and final file installation. It runs alongside the coordinator, downloads, and decoding processes.

`--processing-workers` bounds active decoding/encoding jobs, and `--max-decoded-gb` bounds their combined decoded-array reservations. Those reservations are released when encoding finishes. A separate limit of twice the processing worker count bounds native COG batches from job submission through durable publication, and the catalog request queue holds at most one request per processing worker. When publication falls behind, these limits pause new encoding while transfer slots continue to refill within the source budget. Common analysis chunks contain 27 hours (about 112 MB decoded at float32); forecast chunks contain 12 hours (about 50 MB). Each job additionally needs compression, validation, and library buffers and may stage one COG per selected hour in its native chunk. These limits bound work independently of the total requested date range. If one native chunk cannot fit, increase `--max-decoded-gb`; the completed source is retained for retry. A 2 GB decoded budget can accommodate sixteen of those analysis chunks, excluding the additional encoding/process memory.

The source budget limits staged NetCDF bytes, not the published archive or processing memory. Use `--processing-workers 1` to run one native chunk job at a time while retaining download overlap and bounded prefetch. More workers can increase CPU use and storage contention; compare the publication rate when choosing a count. Retained source inputs are released after their selected monthly work is published, except files still supplying a later selected month.

## Variables and time semantics

Names come directly from NCAR's file inventory; examples include:

| Name | Variable |
| --- | --- |
| `2t` | 2 metre temperature |
| `2d` | 2 metre dewpoint temperature |
| `10u`, `10v` | 10 metre wind components |
| `sp` | Surface pressure |
| `ssrd`, `strd` | Downward surface solar and thermal radiation |
| `ssr`, `str` | Net surface solar and thermal radiation |
| `mtpr` | Mean total precipitation rate |

The pipeline retains source values, units, grid coordinates, UTC timestamps, decoded precision, and interval metadata. Analysis fields use their valid timestamps directly. Forecast valid time is initialization time plus lead hours.

Radiation accumulations describe the preceding hour and are stored without differencing. The source radiation unit `W m**-2 s` is energy per square metre, equivalent to J/m². `mtpr` describes the mean rate over the preceding hour in `kg m**-2 s**-1`; multiplying by 3,600 gives that hour's water-equivalent precipitation in mm. The archive retains the rate without performing that conversion.

Original source attributes, including upstream precision-processing metadata, remain in provenance. Lossless DEFLATE compression preserves NCAR's supplied decoded values; it does not reverse upstream processing. See the [ERA5 data documentation](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5%2Bdata%2Bdocumentation).

## Stop, resume, and refresh

Press Ctrl+C to stop. No new processing jobs are scheduled; active workers finish before input files are released. Transfers stop between received chunks and retain partial bytes. An idle network read times out after 30 seconds. Run the same ordinary `pull` command to resume. SQLite supplies recorded paths and selections, completed downloads are reused directly, and partial downloads resume from their saved byte position. Transient transfer failures allow up to three attempts with backoff. Requests are pinned to the discovered source identity so bytes from different versions cannot be combined.

The pull command prepares a source's selected observations in one transaction, then publishes each completed native chunk as a batch: one transaction records the validated files as ready, and another records coverage after installation. Files are flushed before the ready commit. A batch is acknowledged only after its coverage commit; an interrupted installation can recover a mix of staged and already installed files. Failed conversion or publication retains its source files, and a restart skips completed observations, rebuilds incomplete files, or finishes a validated publication. NetCDF inputs are removed only after all their selected hours are published and recorded in SQLite. Completed ordinary repeats require no source access.

Use `--refresh` to check current source identities and replace selected stored hours:

```powershell
python -m era5_pull_pipeline pull --start-date 2000-01-01 --end-date 2000-01-01 --variables 2t --archive downloads/manual-cog --refresh
```

This replaces January 1 temperature while preserving the other hours and variables. Every selected hour in that month must be available before downloads start. Each source is checked for its selected hours before writing its observations. A failure can leave earlier observations refreshed; rerunning `--refresh` completes the selection. A newer source replaces any older pending source for the same selected hours after publication succeeds. Unrelated pending selections remain available.

An interrupted refresh reuses staged files when they match the current inventory. Rerunning `--refresh` selects the entire interval again, including months already refreshed. Adjust the start date to continue only the unfinished interval.

## Output contract

```text
archive/
  dataset.yaml
  catalog.sqlite
  1999/12/2_metre_temperature/
    2_metre_temperature_19991231T230000Z.tif
  2000/01/2_metre_temperature/
    2_metre_temperature_20000101T000000Z.tif
  2000/01/mean_total_precipitation_rate/
    mean_total_precipitation_rate_20000101T000000Z.tif
  staging/
    ncar/
    observations/
    decoded/
```

The filename is `<variable>_YYYYMMDDTHHMMSSZ.tif` under `YYYY/MM/<variable>/`. Its timestamp is the observation's valid UTC time. Each file contains one full-resolution band, DEFLATE lossless compression, 512 × 512 tiles, and nearest-neighbor internal display overviews. Calculations read the full-resolution band; overviews contain sampled values for display. Source precision is retained, including float64 when supplied. Floating-point missing cells use NaN NoData. An observation containing only missing cells still has a file and catalog row; an absent observation has neither.

Longitude columns are reordered without changing values from 0…359.75° to sample centers −180…179.75°. Latitude centers remain 90…−90°. The 721 × 1440 grid uses EPSG:4326 and `AREA_OR_POINT=Point`. Its GDAL geotransform is `[-180.125, 0.25, 0, 90.125, 0, -0.25]`. The half-cell outer bounds are −180.125…179.875° longitude and −90.125…90.125° latitude. These edges preserve the original sample centers, including both polar rows; no pixels are shifted, dropped, resampled, or reprojected.

The producer creates this root descriptor:

```yaml
id: era5-ncar
title: ERA5 hourly weather
source:
  format: temporal_cog
  catalog: catalog.sqlite
```

`source.format: temporal_cog` declares a temporal raster archive located through the SQLite `observations` view. All cataloged paths, including retained NetCDF inputs, are relative to the archive root and use `/` separators. Move the complete archive, including its catalog and staging directory, without rewriting paths.

Consumers query `observations`; they do not need to infer metadata from filenames or source parameter codes:

| Column | Meaning |
| --- | --- |
| `variable`, `name`, `units` | Stable descriptive variable identifier, human-readable name, and original source unit string. |
| `path`, `band` | Archive-relative GeoTIFF path and 1-based band number (1). |
| `valid_time` | Observation timestamp, integer Unix seconds in UTC. |
| `reference_time` | Forecast initialization time in the same encoding; NULL for instantaneous analysis. |
| `interval_start`, `interval_end` | UTC interval boundaries in Unix seconds. Instantaneous fields have both equal to `valid_time`. |
| `temporal_type` | `instant`, `mean`, or `accumulation`. Hourly means and accumulations represent the preceding interval `(interval_start, interval_end]`. |
| `dtype`, `nodata` | Decoded NumPy type name and NoData value; `nodata` is the text `NaN` for floating rasters or SQL NULL when absent. |
| `crs`, `transform`, `height`, `width` | CRS identifier, JSON array in GDAL six-value order, and raster dimensions. |
| `start_step`, `end_step`, `step_units` | Forecast leads from reference time; seconds (`s`), or NULL leads for analysis. |
| `source_file_id`, `source_variable`, `object_key`, `etag`, `last_modified` | Provenance linking the observation to the exact provider object. |
| `retrieved_at` | Latest successful retrieval time as an ISO UTC timestamp, retained after the NetCDF download is removed. |

The primary observation key is `(variable, valid_time)`. Source attributes, including upstream precision and packed-value encoding, are retained in `source_files.attributes_json`; variable metadata are in `variables.attributes_json`. The `observations` view joins the catalog's existing coverage, variable, and source records. It is the consumer interface, not a separate metadata database. TIFFs also carry units, variable names, UTC times, interval semantics, and source identifiers as metadata.

Example query and full-resolution read:

```python
from contextlib import closing
from pathlib import Path
import sqlite3
import rasterio

archive = Path("downloads/manual-cog")
with closing(sqlite3.connect(archive / "catalog.sqlite")) as db:
    row = db.execute(
        "SELECT path, band, units FROM observations "
        "WHERE variable = ? ORDER BY valid_time LIMIT 1",
        ("2_metre_temperature",),
    ).fetchone()
with rasterio.open(archive / row[0]) as raster:
    values = raster.read(row[1])
    print(values.shape, values.dtype, row[2], raster.transform)
```

One writer owns the archive. Consumers follow **update → refresh → calculate**: finish the producer update, refresh their observation listing, then calculate while source files remain unchanged. Each COG is finished, read back, and validated before publication. A durable SQLite journal bridges file installation and coverage recording so interrupted publication can finish on restart. Observations awaiting that commit are excluded from the public view. There are no historical raster copies or concurrent-reader snapshots.

## Representative sample

[examples/era5-cog-sample.zip](examples/era5-cog-sample.zip) contains a ready-to-read `era5-cog-sample/` archive: the descriptor, SQLite catalog, and six real global rasters for temperature, accumulated solar radiation, and mean precipitation rate at **1999-12-31 23:00 UTC** and **2000-01-01 00:00 UTC**. It has no pending downloads and uses the same output contract as a full archive.

```powershell
python -m zipfile -e examples/era5-cog-sample.zip downloads
```

Use `downloads/era5-cog-sample/dataset.yaml` for integration, or set the Python inspection example's archive path to `downloads/era5-cog-sample`.

## Separate download and ingestion

To preview whole source files and their sizes without creating local state:

```powershell
python -m era5_pull_pipeline download-ncar --start-date 2000-01-01 --end-date 2000-01-31 --variables 2t ssrd mtpr --dry-run
```

For a local download followed by offline ingestion:

```powershell
python -m era5_pull_pipeline download-ncar --start-date 2000-01-01 --end-date 2000-01-31 --variables 2t --archive downloads/local-ncar
python -m era5_pull_pipeline ingest-ncar --archive downloads/local-ncar --year 2000 --month 1
```

`download-ncar` retains all selected files under `staging/ncar/` and records selected hours in SQLite. It supports `--download-workers` but does not impose the bulk workflow's source budget; use `pull` for long backfills. `ingest-ncar` publishes pending selections for one month without network access, using the serial offline ingestion path and disk-backed decoded scratch arrays. Add `--refresh` to ingestion to replace their stored values.

## Tests

Install the optional test dependencies and run:

```powershell
python -m pip install -e ".[test]"
python -m pytest -q --tb=short
```

Tests use local NetCDF fixtures, real SQLite/COG files, the included global sample, and a local HTTP server. They exercise exact pixel/NoData preservation, georeferencing, time semantics, byte-range resume, concurrency, month boundaries, selected refresh, publication recovery, source cleanup, relocation, and resource limits without contacting external services.
