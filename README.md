# ERA5 pull pipeline

A Python pipeline that runs from the command line, downloads ERA5 weather data, and maintains a compressed Zarr archive. Each invocation processes a requested date range and set of variables, fills missing coverage, and exits. The same archive grows from 1950 through the latest available data as subsequent runs add new hours.

## Inputs

Each run accepts:

| Input | Meaning |
| --- | --- |
| Date range | Inclusive start and end dates in UTC. |
| Variables | The ERA5 single-level variables to retrieve. |
| Geographic extent | A latitude/longitude bounding box, or global coverage. |
| Output directory | The archive to create or extend. |
| Download batch size | How many days to request at a time. |
| Spatial chunk shape | How many grid cells along latitude and longitude to store in each chunk. |

## Credentials

Access uses a Copernicus Climate Data Store (CDS) account, a personal access token, and acceptance of the dataset's licence. Place a `.cdsapirc` file in the directory from which you run the pipeline:

```yaml
url: https://cds.climate.copernicus.eu/api
key: YOUR_PERSONAL_ACCESS_TOKEN
```

## Data

The source is [ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels), using the regular 0.25° latitude/longitude grid.

The archive retains source units, grid coordinates, UTC timestamps, decoded numerical precision, and interval metadata. It preserves each variable's time semantics, including accumulation intervals. See the [ERA5 data documentation](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5%2Bdata%2Bdocumentation).

## Pipeline steps

1. **Load credentials.** Authenticate with CDS using the local `.cdsapirc` file.
2. **Find missing coverage.** Compare the requested dates and variables with the archive's SQLite catalog. Already stored variable/hour combinations are skipped; gaps and newly requested variables are included.
3. **Download GRIB batches.** Split requests according to the download batch size, stopping at calendar-month boundaries. Batch size is independent of the Zarr storage layout.
4. **Assemble the month.** Decode the GRIB fields into arrays with dimensions `(time, latitude, longitude)`. Combine batches in a staging area for the relevant month. For an existing month, merge new or refreshed values with its published contents, preserving all other variables and hours. During a historical backfill, assemble all requested batches for a month before publishing it.
5. **Validate and publish.** Check the variables, grid, timestamps, interval metadata, numerical values, and readability of the compressed output. Publish the validated monthly dataset and record its coverage and revision in SQLite.
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

Within each monthly dataset, read chunks span the calendar month along time and a tile of grid cells in space. The spatial chunk shape is configured when creating the archive; Zarr does not require a particular tile size. Zarr sharding packs multiple independently readable chunks into larger files to control file counts. A query reads the chunks intersecting its dates and area; it does not load the entire archive.

Chunks use **lossless Zstandard compression**. Storage does not introduce numerical rounding, quantization, or a reduction in decoded precision. Compression reduces storage and transfer volume, while chunk layout determines which data a query needs to read. See [Zarr's chunking and sharding guidance](https://zarr.readthedocs.io/en/stable/user-guide/performance/).

The monthly directories form one continuous archive. A reader selects the relevant months and combines their selected arrays along time. Monthly boundaries also keep validation, recovery, and replacement limited to the affected periods.

## Repeat runs and source updates

SQLite records coverage by variable and hour, together with the dataset, geographic extent, and grid that define the archive. A different extent or grid uses a separate archive. The catalog also retains source request details, retrieval times, GRIB checksums, validation results, and published revisions after temporary downloads are removed.

A single writer updates an archive. It prepares a replacement month in staging and retains the previous version until publication and catalog recording succeed. Readers coordinate with publication so they do not read a mixture of old and new arrays. SQLite transactions record state; they do not make filesystem changes atomic. Interrupted publication is reconciled before the affected month becomes readable again.

Adding hours to a partial month or refreshing a selected historical interval produces a replacement version of the affected monthly dataset. This trades rewriting that month for a simple publication and recovery process; the rest of the archive remains untouched. Ordinary catch-up runs fill missing coverage. A deliberate refresh replaces already stored values when ERA5 changes them.
