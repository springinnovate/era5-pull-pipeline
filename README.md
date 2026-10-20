# ERA5 pull pipeline

A Python pipeline that runs from the command line, downloads full global ERA5 rasters, and maintains a compressed Zarr archive. Each invocation processes a requested date range and set of variables, fills missing coverage, and exits. The same archive grows from 1950 through the latest available data as subsequent runs add new hours.

## Installation

Use Python 3.11 or newer. From the repository directory, create a virtual environment and install the package:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

These commands use PowerShell on Windows. On macOS or Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

## Inputs

Each run accepts:

| Input | Meaning |
| --- | --- |
| Date range | Inclusive start and end dates in UTC. |
| Variables | The ERA5 single-level variables to retrieve. |
| Output directory | The archive to create or extend. |
| Download batch size | How many days to request at a time. |
| Spatial chunk shape | How many grid cells along latitude and longitude to store in each chunk. |

## Credentials

Access uses a Copernicus Climate Data Store (CDS) account, a personal access token from your [CDS profile](https://cds.climate.copernicus.eu/profile), and acceptance of the [ERA5 dataset's licence](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download). Place a `.cdsapirc` file in the directory from which you run the pipeline:

```yaml
url: https://cds.climate.copernicus.eu/api
key: YOUR_PERSONAL_ACCESS_TOKEN
```

## Check access

From the directory containing `.cdsapirc`, run:

```powershell
.\.venv\Scripts\python.exe -m era5_pull_pipeline check-access
```

The check downloads one full global field of 2 m temperature for **2000-01-01 at 00:00 UTC** to `downloads/era5-access-check.grib` and decodes it with xarray's cfgrib reader. Success prints `CDS access verified` and the saved path. The probe file stays separate from the monthly archive.

CDS may queue the request. Press Ctrl+C to stop waiting locally. Connection settings are passed directly to CDS; file, service, and decoding failures show their original errors.

Repeat the same command to check access again; it replaces the probe file. Use `--output` to choose a different location:

```powershell
.\.venv\Scripts\python.exe -m era5_pull_pipeline check-access --output downloads/check-2.grib
```

The installed `era5-pull check-access` command provides the same check. Use `--help` to see its options.

## Tests

Install the optional test dependencies and run the automated tests from the repository directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest_cache/manual-check
```

The tests use local fixtures and simulated service responses. They do not need credentials or contact CDS; `check-access` is the explicit live test.

## Data

The source is [ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels), using the full global regular 0.25° latitude/longitude grid. Every requested variable and timestep covers this same grid. Geographic selection and clipping belong to downstream consumers.

The archive retains source units, grid coordinates, UTC timestamps, decoded numerical precision, and interval metadata. It preserves each variable's time semantics, including accumulation intervals. See the [ERA5 data documentation](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5%2Bdata%2Bdocumentation).

## Pipeline steps

1. **Load credentials.** Authenticate with CDS using the local `.cdsapirc` file.
2. **Find missing coverage.** Compare the requested dates and variables with the archive's SQLite catalog. Already stored variable/hour combinations are skipped; gaps and newly requested variables are included.
3. **Download GRIB batches.** Split the requested time interval according to the download batch size, stopping at calendar-month boundaries. Every downloaded field covers the full global grid. Batch size is independent of the Zarr storage layout.
4. **Assemble the month.** Decode the GRIB fields into arrays with dimensions `(time, latitude, longitude)`. Combine batches in a staging area for the relevant month. For an existing month, merge new or refreshed values with its published contents, preserving all other variables and hours. During a historical backfill, assemble all requested batches for a month before publishing it.
5. **Validate and publish.** Check the variables, global grid, timestamps, interval metadata, numerical values, and readability of the compressed output. Publish the validated monthly dataset and record its coverage and revision in SQLite.
6. **Remove the GRIB downloads.** Delete the corresponding temporary GRIB files only after publication and the catalog update succeed. The Zarr datasets are the retained data archive.

The staging area holds downloaded GRIB files alongside the monthly Zarr data until publication succeeds, so temporary disk space includes both. The latest month can contain only part of a month. Dates that are not yet available from CDS remain missing, so a subsequent run can retrieve them. The catalog distinguishes partial coverage from a complete requested interval.

## Archive layout

```text
archive/
  catalog.sqlite
  1950/
    01/
      era5.zarr/
    02/
      era5.zarr/
  2026/
    09/
      era5.zarr/
    10/
      era5.zarr/
  staging/
```

Each `YYYY/MM/era5.zarr/` is a [Zarr 3 directory](https://zarr.readthedocs.io/en/stable/user-guide/arrays/#persistent-arrays), containing metadata and compressed array data. Each variable has its own array, with shared time, latitude, and longitude coordinates, and can be read independently.

Within each monthly dataset, read chunks span the calendar month along time and a tile of grid cells in space. These chunks partition the stored global arrays. The spatial chunk shape is configured when creating the archive; Zarr does not require a particular tile size. Zarr sharding packs multiple independently readable chunks into larger files to control file counts.

Chunks use **lossless Zstandard compression**. Storage does not introduce numerical rounding, quantization, or a reduction in decoded precision. Compression reduces storage and transfer volume, while chunks allow portions of an array to be read independently. See [Zarr's chunking and sharding guidance](https://zarr.readthedocs.io/en/stable/user-guide/performance/).

The monthly directories form one continuous archive. A reader selects the relevant months and combines their selected arrays along time. Monthly boundaries also keep validation, recovery, and replacement limited to the affected periods.

## Repeat runs and source updates

SQLite records coverage by variable and hour on the fixed global grid, together with source dataset and grid metadata. The catalog also retains source request details, retrieval times, GRIB checksums, validation results, and published revisions after temporary downloads are removed.

A single writer updates an archive. It prepares a replacement month in staging and retains the previous version until publication and catalog recording succeed. Readers coordinate with publication so they do not read a mixture of old and new arrays. SQLite transactions record state; they do not make filesystem changes atomic. Interrupted publication is reconciled before the affected month becomes readable again.

Adding hours to a partial month or refreshing a selected historical interval produces a replacement version of the affected monthly dataset. This trades rewriting that month for a simple publication and recovery process; the rest of the archive remains untouched. Ordinary catch-up runs fill missing coverage. A deliberate refresh replaces already stored values when ERA5 changes them.
