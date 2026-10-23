"""Verify the real ERA5 archive supplied for consumer integration."""

from contextlib import closing
from datetime import UTC, datetime
import json
from pathlib import Path, PurePosixPath
import sqlite3
import zipfile

import numpy as np
import rasterio


# Values read independently from the source Zarr fields before COG generation.
# Pixel centers: (-180,90), (-180,0), (0,0), and (-0.25,-90).
SOURCE_PIXELS = {
    ("2t", 946681200): [266.068115234375, 298.931396484375, 298.915771484375, 241.767333984375],
    ("2t", 946684800): [264.69244384765625, 299.21197509765625, 299.08306884765625, 243.44049072265625],
    ("mtpr", 946681200): [4.112720489501953e-05, 0.0, 7.236003875732422e-05, 0.0],
    ("mtpr", 946684800): [2.956390380859375e-05, 0.0, 1.2278556823730469e-05, 0.0],
    ("ssrd", 946681200): [0.0, 3116864.0, 0.0, 1716800.0],
    ("ssrd", 946684800): [0.0, 3554112.0, 0.0, 1716736.0],
}


def test_sample_archive_catalog_files_and_native_values(tmp_path: Path) -> None:
    """Open the portable sample using its catalog and confirm source values."""
    bundle = Path(__file__).resolve().parents[1] / "examples" / "era5-cog-sample.zip"
    with zipfile.ZipFile(bundle) as packed:
        assert len(packed.namelist()) == 8
        packed.extractall(tmp_path)
    archive = tmp_path / "era5-cog-sample"
    assert (archive / "dataset.yaml").read_text(encoding="utf-8") == (
        "id: era5-ncar\ntitle: ERA5 hourly weather\nsource:\n"
        "  format: temporal_cog\n  catalog: catalog.sqlite\n"
    )
    with closing(sqlite3.connect(archive / "catalog.sqlite")) as catalog:
        catalog.row_factory = sqlite3.Row
        rows = catalog.execute("SELECT * FROM observations ORDER BY valid_time, variable").fetchall()
        assert {(row["source_variable"], row["valid_time"]) for row in rows} == set(SOURCE_PIXELS)
        assert catalog.execute("PRAGMA foreign_key_check").fetchall() == []
        assert catalog.execute("SELECT count(*) FROM source_selections").fetchone()[0] == 0
        assert catalog.execute("SELECT count(*) FROM publications").fetchone()[0] == 0
        assert {path.relative_to(archive).as_posix() for path in archive.rglob("*.tif")} == {
            row["path"] for row in rows
        }
        for row in rows:
            timestamp = datetime.fromtimestamp(row["valid_time"], UTC)
            assert row["path"] == f"{timestamp:%Y/%m}/{row['variable']}/{row['variable']}_{timestamp:%Y%m%dT%H%M%SZ}.tif"
            assert not PurePosixPath(row["path"]).is_absolute()
            with rasterio.open(archive / row["path"]) as raster:
                assert (row["height"], row["width"]) == raster.shape == (721, 1440)
                assert row["band"] == raster.count == 1
                assert row["dtype"] == raster.dtypes[0] == "float32"
                assert row["nodata"] == "NaN" and np.isnan(raster.nodata)
                assert row["crs"] == raster.crs.to_string() == "EPSG:4326"
                assert json.loads(row["transform"]) == list(raster.transform.to_gdal()) == [
                    -180.125, 0.25, 0, 90.125, 0, -0.25,
                ]
                assert raster.xy(0, 0) == (-180.0, 90.0)
                assert raster.xy(720, 1439) == (179.75, -90.0)
                assert raster.block_shapes == [(512, 512)]
                assert raster.overviews(1) == [2, 4]
                structure = raster.tags(ns="IMAGE_STRUCTURE")
                assert structure["LAYOUT"] == "COG"
                assert structure["COMPRESSION"] == "DEFLATE"
                assert structure["OVERVIEW_RESAMPLING"] == "NEAREST"
                assert raster.units == (row["units"],)
                tags = raster.tags()
                for key in ("variable", "valid_time", "interval_start", "interval_end", "temporal_type", "source_file_id"):
                    assert tags[key] == str(row[key])
                values = raster.read(1)
                pixels = values[[0, 360, 360, 720], [0, 0, 720, 719]]
                np.testing.assert_array_equal(pixels, SOURCE_PIXELS[row["source_variable"], row["valid_time"]])
            if row["source_variable"] == "2t":
                assert row["units"] == "K" and row["temporal_type"] == "instant"
                assert row["interval_start"] == row["interval_end"] == row["valid_time"]
                assert row["reference_time"] is None
            else:
                expected = ("mean", "kg m**-2 s**-1") if row["source_variable"] == "mtpr" else (
                    "accumulation", "W m**-2 s",
                )
                assert (row["temporal_type"], row["units"]) == expected
                assert row["interval_start"] == row["valid_time"] - 3600
                assert row["interval_end"] == row["valid_time"]
                assert row["reference_time"] == 946663200
            source = catalog.execute("SELECT * FROM source_files WHERE id = ?", (row["source_file_id"],)).fetchone()
            assert source["removed_at"] is not None
            assert source["downloaded_at"] is None
            assert row["retrieved_at"] == source["retrieved_at"]
            assert datetime.fromisoformat(row["retrieved_at"]).tzinfo == UTC
            assert source["path"].startswith("staging/ncar/") and "\\" not in source["path"]
            assert json.loads(source["attributes_json"])["variable"]["units"] == row["units"]
