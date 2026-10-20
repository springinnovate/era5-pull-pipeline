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
| Refresh | Retrieve the selected fields again and replace their stored values. |

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

## Download a date range

From the directory containing `.cdsapirc`, run:

```powershell
.\.venv\Scripts\python.exe -m era5_pull_pipeline pull --start-date 1999-12-31 --end-date 2000-01-02 --variables 2m_temperature total_precipitation --archive downloads/manual-archive-13 --batch-days 1
```

This retrieves three complete UTC days on the global grid, in one-day batches, and publishes `1999/12/era5.zarr` and `2000/01/era5.zarr` beneath the archive directory. `--batch-days` defaults to 7. Requests stop at month boundaries and are subdivided further to respect CDS field limits. It limits calendar days per batch; compressed GRIB sizes vary by variable and data values.

CDS prints the request ID and status changes, such as `accepted`, `running`, and `successful`. While waiting, the pipeline also reports a successful status check approximately every 30 seconds, showing whether the job is queued or running and the elapsed time since the request started. That elapsed time includes earlier invocations when resuming a saved job. For example:

```text
CDS request <job-id>: still queued (accepted); 0:15:30 since request started. Still checking CDS.
```

Once the file transfer begins, a progress bar shows downloaded bytes, transfer rate, and an estimated remaining transfer time. The transfer estimate does not include queue time; CDS does not provide a queue completion estimate.

Repeat the same command to verify coverage: a completed range prints `completed=0, skipped=144` and makes no CDS calls. Counts are **variable-hours**: two variables over three 24-hour days total 144. `completed` counts fields published in this invocation; `skipped` counts already stored fields; `unavailable` counts requested hours absent from CDS's availability snapshot. `failed` identifies a failed batch or monthly publication, and `pending` counts remaining unattempted or staged fields. A failure prints the summary and original error. Hours marked unavailable are requested again on a subsequent run.

To add wind for the same dates, run the command with `--variables 10m_u_component_of_wind 10m_v_component_of_wind`. Existing temperature and precipitation remain in the monthly stores. Use `--spatial-chunks LAT LON` when creating an archive to change the default `32 32` tile shape; subsequent writes must use the archive's existing shape.

Press Ctrl+C to stop. Run the same command to resume: the pipeline uses SQLite's staged batch records to recover submitted CDS jobs and reuse completed downloads and conversions. An interrupted transfer restarts from the beginning; an interrupted conversion is rebuilt from its retained GRIB. Completed artifacts are immutable and reused directly. CDS requests run serially with bounded retries and backoff for transient service failures.

If CDS reports that a saved job has failed or expired, remove that job's `.partial` directory under `staging/batches/` and rerun to submit it again. Its `request.json` identifies the selection and CDS job ID. Completed batches remain reusable.

## Refresh stored data

Add `--refresh` to retrieve the selected dates and variables again, including fields already in the archive:

```powershell
.\.venv\Scripts\python.exe -m era5_pull_pipeline pull --start-date 2000-01-01 --end-date 2000-01-01 --variables 2m_temperature --archive downloads/manual-archive-13 --batch-days 1 --refresh
```

This replaces all 24 temperature fields for January 1 and prints `completed=24, skipped=0`. Other dates, variables, and months retain their data and source information. Gaps within the selection are filled. Each successful refresh records a new retrieval and publication, even if CDS returns identical values. Running without `--refresh` continues to skip stored fields.

Each month's selected fields must all be available from CDS. If availability is incomplete, a download fails, or a response lacks required fields, that month remains unchanged. Absent fields and array fill slots cannot replace stored fields; missing grid cells explicitly encoded in a returned GRIB retain their source meaning.

An interrupted refresh reuses its own staged downloads and conversions, separately from ordinary catch-up work. Publication commits one month at a time. Months already published remain committed if a later month fails; rerunning with `--refresh` selects the entire requested range again, including those completed months. To continue only the unfinished interval, adjust the dates accordingly.

## Tests

Install the optional test dependencies and run the automated tests from the repository directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest_cache/manual-check
```

The tests use local fixtures and simulated service responses. They do not need credentials or contact CDS. `check-access` and `pull` contact the live service when a download is needed.

## Data

The source is [ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels), using the full global regular 0.25° latitude/longitude grid. Every requested variable and timestep covers this same grid. Geographic selection and clipping belong to downstream consumers.

The archive retains source units, grid coordinates, UTC timestamps, decoded numerical precision, and interval metadata. It preserves each variable's time semantics, including accumulation intervals. See the [ERA5 data documentation](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5%2Bdata%2Bdocumentation).

## Pipeline steps

1. **Select coverage.** Compare the requested dates and variables with the archive's SQLite catalog. Ordinary runs skip stored variable/hour combinations and include gaps and newly requested variables. `--refresh` includes every selected field. Match the selection against CDS's availability constraints, including partially available days.
2. **Load credentials when needed.** Authenticate with CDS using the local `.cdsapirc` file when a download is required. A completed repeat needs no credentials or service calls.
3. **Download GRIB batches.** Split the requested time interval according to the download batch size, stopping at calendar-month boundaries. Every downloaded field covers the full global grid. Batch size is independent of the Zarr storage layout.
4. **Assemble the month.** Decode the GRIB fields into arrays with dimensions `(time, latitude, longitude)` and save validated, compressed batch checkpoints. Assemble all selected batches for the month in staging, preserving existing fields unless selected for refresh. Publish each affected month once per invocation.
5. **Validate and publish.** Check the variables, global grid, timestamps, interval metadata, numerical values, and readability of the compressed output. Publish the validated monthly dataset and record its coverage and revision in SQLite.
6. **Remove temporary sources.** Delete the corresponding GRIB files and conversion checkpoints only after publication and the catalog update succeed. The monthly Zarr datasets are the retained data archive.

Processing is serial and limited to one month at a time. Peak disk use includes that month's GRIBs, compressed batch checkpoints, one batch's temporary decoded arrays, the staged monthly replacement, and any previous published month being replaced. Decoded arrays use disk-backed storage; each global float64 field occupies about 8.3 MB before compression. Conversion processes one global field at a time and assembles bounded spatial shards without loading a whole month into RAM. Smaller day batches reduce decoded scratch space. The latest month can contain only part of a month, and unavailable hours remain absent from catalog coverage.

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

Arrays use ecCodes' descriptive names, normalized to lowercase words separated by underscores, such as `2_metre_temperature` and `total_precipitation`. Their attributes retain the original descriptive name, short name, numeric parameter ID, and units; the source request retains the original CDS variable names. Time coordinates identify when fields are valid. Per-variable metadata arrays retain forecast reference times and interval boundaries, including precipitation intervals that begin on the preceding day.

Variables can cover different hours within the shared monthly time coordinate. Each variable's matching `<variable>_coverage` array, such as `2_metre_temperature_coverage`, identifies which hours contain retrieved fields. A false entry means the field is absent; a true entry can still contain spatial missing values. Missing fields have NaN weather values and missing interval metadata. Coverage comes from validated source fields, independently of their numeric values.

Within each monthly dataset, read chunks span the calendar month along time and a tile of grid cells in space. These chunks partition the stored global arrays. The spatial chunk shape is configured when creating the archive; Zarr does not require a particular tile size. Zarr sharding packs multiple independently readable chunks into larger files to control file counts.

Chunks use **lossless Zstandard compression**. Storage does not introduce numerical rounding, quantization, or a reduction in decoded precision. Compression reduces storage and transfer volume, while chunks allow portions of an array to be read independently. See [Zarr's chunking and sharding guidance](https://zarr.readthedocs.io/en/stable/user-guide/performance/).

The monthly directories form one continuous archive. A reader selects the relevant months and combines their selected arrays along time. Monthly boundaries also keep validation, recovery, and replacement limited to the affected periods.

## Repeat runs and source updates

SQLite records each staged request and destination before downloading, then records download and conversion completion. It also records published coverage by variable and hour on the fixed global grid. Source request details, retrieval times, GRIB checksums, validation results, and published revisions remain in the catalog after temporary downloads are removed.

Requesting an additional variable, a longer interval, or an interior gap downloads only missing variable/hour combinations. The planner uses CDS variable names and successfully published source requests to identify completed selections, even when requested in a different combination of variables or dates. Ordinary requests preserve already stored values, including fields whose grid cells are all missing. Published field provenance continues to identify the retrieval that supplied each field through subsequent monthly replacements.

A single writer updates an archive. It prepares a replacement month in staging and retains the previous version until publication and catalog recording succeed. Readers coordinate with publication so they do not read a mixture of old and new arrays. SQLite transactions record state; they do not make filesystem changes atomic. Interrupted publication is reconciled before the affected month becomes readable again.

Python readers use `era5_pull_pipeline.catalog.open_month` as a context manager and finish reading before leaving it. The context holds the publication lock while its xarray dataset is open; opening Zarr directly bypasses that coordination. Downloads and conversion can proceed while a reader is active. Restarting an archive command recovers a validated publication interrupted between its filesystem rename and catalog commit, and retries any pending GRIB cleanup. Unvalidated staging data remain outside published coverage.

Adding hours to a partial month or refreshing a selected historical interval produces a replacement version of the affected monthly dataset. This trades rewriting that month for a simple publication and recovery process; the rest of the archive remains untouched. Ordinary catch-up runs fill missing coverage. A deliberate refresh replaces already stored values when ERA5 changes them.
