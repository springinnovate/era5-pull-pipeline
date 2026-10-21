# ERA5 pull pipeline

A Python command-line pipeline that downloads full global ERA5 rasters from [NCAR's public archive](https://registry.opendata.aws/nsf-ncar-era5/) and maintains a compressed monthly Zarr archive. Each invocation fills a requested date range and set of variables, publishes each month, and exits. Repeated runs skip stored hours and resume unfinished downloads. No account or access token is required.

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
python -m era5_pull_pipeline pull --start-date 1999-12-31 --end-date 2000-01-02 --variables 2t ssrd mtpr --archive downloads/manual-ncar-25
```

It publishes `1999/12/era5.zarr` and `2000/01/era5.zarr`, with 24 and 48 selected hours respectively. Every weather field covers the full 721 × 1440 global grid at 0.25° resolution. The pipeline performs no geographic selection, temporal aggregation, or derived ET0 calculation.

Repeat the same command to check resume behavior. Once the selection is complete, it prints `completed=0, skipped=216` and makes no NCAR calls. Counts are **variable-hours**: three variables × three days × 24 hours = 216. The summary also reports unavailable hours, hours in a failed month, and remaining pending hours.

To add wind for those dates, run the same command with `--variables 10u 10v`. The monthly replacements preserve existing variables and hours. Extend the dates to fill additional months or interior gaps.

| Option | Meaning |
| --- | --- |
| `--start-date` | First requested UTC date, inclusive; required. |
| `--end-date` | Last requested UTC date, inclusive; defaults to today in UTC. |
| `--variables` | One or more NCAR short variable names; required. |
| `--archive` | Directory containing the catalog, staging, and monthly stores; defaults to `archive`. |
| `--download-workers` | Maximum simultaneous file transfers; defaults to 2. |
| `--max-source-gb` | Maximum reserved NetCDF source bytes in decimal GB; defaults to 16. Conversion requires additional space. |
| `--spatial-chunks LAT LON` | Grid cells per spatial read chunk; defaults to `32 32`. Must match the archive's existing layout. |
| `--refresh` | Retrieve current sources and replace selected stored fields. |

NCAR packages analysis fields as one variable per month, and radiation and precipitation in roughly half-month files. Selecting a single day still downloads whole source files; only selected hours enter the Zarr archive. A forecast file can supply hours on both sides of a month boundary and is downloaded once, then retained until both selected months are published.

Transfers show bytes, speed, elapsed time, and estimated remaining transfer time. Month messages identify availability lookup, conversion, and publication. There is no overall backfill completion estimate.

## Backfill from 1950

To download the selected variables from 1950 through the latest data available from NCAR:

```powershell
python -m era5_pull_pipeline pull --start-date 1950-01-01 --variables 2t 2d 10u 10v sp ssrd strd mtpr --archive archive
```

NCAR updates monthly and typically trails real time by about 3–4 months. Availability is checked separately for each variable. Unavailable hours remain absent from coverage and are checked again on subsequent runs, so the same command extends the archive as NCAR adds data. An explicit `--end-date` limits the requested interval.

The workflow schedules one month at a time. At most `--download-workers` transfers are active, and conversion and publication run serially after that month's downloads. The source budget reserves full file sizes before transfers start, including files retained from an interrupted run or for a later month. If a month cannot fit, the command stops before downloading it and reports the required size. Increase `--max-source-gb` or request fewer variables at a time.

This budget limits staged NetCDF sources, not total disk use or RAM. Peak working space also includes decoded selected arrays, the new compressed monthly store, and any previous month being replaced. One float32 global hourly field takes about 4.15 MB before compression. Decoded arrays use disk-backed memory mappings; their pages can occupy the operating system's file cache. Conversion reads native NetCDF blocks and writes bounded Zarr shards without loading a whole month into Python heap memory.

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

Original source attributes, including upstream precision-processing metadata, remain in provenance. Lossless Zstandard compression preserves NCAR's supplied decoded values; it does not reverse upstream processing. See the [ERA5 data documentation](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5%2Bdata%2Bdocumentation).

## Stop, resume, and refresh

Press Ctrl+C to stop. Transfers stop between received chunks and retain partial bytes. An idle network read times out after 30 seconds. Run the same ordinary `pull` command to resume. SQLite supplies recorded paths and selections, completed downloads are reused directly, and partial downloads resume from their saved byte position. Transient transfer failures allow up to three attempts with backoff. Requests are pinned to the discovered source identity so bytes from different versions cannot be combined.

Each month commits independently. Failed conversion or publication retains its source files, and a restart rebuilds incomplete conversion or finishes a validated publication. NetCDF inputs are removed only after all their selected hours are published and recorded in SQLite. Completed ordinary repeats require no source access.

Use `--refresh` to check current source identities and replace selected stored hours:

```powershell
python -m era5_pull_pipeline pull --start-date 2000-01-01 --end-date 2000-01-01 --variables 2t --archive downloads/manual-ncar-25 --refresh
```

This replaces January 1 temperature while preserving the other hours and variables. Every selected hour in that month must be available; incomplete availability or an incomplete decoded source prevents publication. A newer source replaces any older pending source for the same selected hours after publication succeeds. Unrelated pending selections remain available.

An interrupted refresh reuses staged files when they match the current inventory. Rerunning `--refresh` selects the entire interval again, including months already refreshed. Adjust the start date to continue only the unfinished interval.

## Inspect the archive

```text
archive/
  catalog.sqlite
  1950/
    01/
      era5.zarr/
    02/
      era5.zarr/
  staging/
    ncar/
    revisions/
```

Each `YYYY/MM/era5.zarr/` is a [Zarr 3 directory](https://zarr.readthedocs.io/en/stable/user-guide/arrays/#persistent-arrays) containing metadata and losslessly compressed arrays. Weather arrays have dimensions `(time, latitude, longitude)` and descriptive source names such as `2_metre_temperature`, `surface_solar_radiation_downwards`, and `mean_total_precipitation_rate`.

Within a month, read chunks span its time axis and a spatial tile. [Zarr sharding](https://zarr.readthedocs.io/en/stable/user-guide/performance/) packs multiple independently readable chunks into larger files to control file counts. Each variable can be read separately; the monthly directories together form the time archive.

Variables can cover different hours within the shared time coordinate. Each `<variable>_coverage` array identifies which hours contain retrieved fields. A false entry means the field is absent; a true entry can still contain spatial missing values from the source. Companion arrays retain forecast reference times and interval boundaries.

Python readers use `open_month` and finish reading within its context:

```python
from pathlib import Path
from era5_pull_pipeline.catalog import open_month

with open_month(Path("downloads/manual-ncar-25"), 2000, 1) as dataset:
    for name, array in dataset.data_vars.items():
        if array.ndim == 3:
            print(name, array.shape, array.dtype, array.attrs["units"])
```

For the three-day example, January's three weather arrays have shape `(48, 721, 1440)` and retain NCAR's float32 precision. December's arrays have shape `(24, 721, 1440)`.

SQLite records published coverage, source selections, identities, retrieval times, and revisions. One writer prepares a replacement month while readers continue to use the published version. Publication coordinates with `open_month` so readers see a complete version. Opening Zarr directly bypasses that coordination. The previous version remains available until publication and catalog recording succeed; interrupted publication is recovered before reading or writing resumes.

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

`download-ncar` retains all selected files under `staging/ncar/` and records selected hours in SQLite. It supports `--download-workers` but does not impose the bulk workflow's source budget; use `pull` for long backfills. `ingest-ncar` publishes pending selections for one month without network access. Add `--refresh` to ingestion to replace their stored values.

## Tests

Install the optional test dependencies and run:

```powershell
python -m pip install -e ".[test]"
python -m pytest -q --tb=short
```

Tests use local NetCDF fixtures, real SQLite and Zarr stores, and a local HTTP server. They exercise interruption, byte-range resume, concurrent transfers, month boundaries, changing availability, historical additions, selected refresh, publication recovery, source cleanup, and resource limits without contacting external services.
