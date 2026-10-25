"""Exercise failures while copying an encoded COG into its journaled path."""

from datetime import date
from io import BufferedWriter
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from era5_pull_pipeline import pipeline
from test_ncar_pipeline import NcarFiles, files


@pytest.mark.parametrize("local_staging", [False, True])
def test_partial_staged_cog_write_retains_inputs_and_resumes(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch, local_staging: bool,
) -> None:
    """Keep partial encoded output unpublished and rebuild it from retained NetCDF."""
    archive = tmp_path / "archive"
    local = tmp_path / "local" if local_staging else None
    root = local or archive
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    partials: list[Path] = []
    open_file = Path.open

    class InterruptedCogFile(BufferedWriter):
        """Leave real partial bytes before simulating a failed filesystem write."""

        def write(self, data: Any) -> int:
            """Write only the beginning of the encoded COG before failing."""
            super().write(data[:17])
            self.flush()
            raise OSError("staged COG write interrupted")

    def interrupt_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        """Inject a partial-write failure only at the prepared COG destination."""
        if path.is_relative_to(root) and path.suffix == ".tif" and mode == "wb":
            partials.append(path)
            return InterruptedCogFile(open_file(path, mode, buffering=0))
        return open_file(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", interrupt_open)
        with pytest.raises(OSError, match="staged COG write interrupted"):
            pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], archive,
                                staging_dir=local, processing_workers=1)
    with sqlite3.connect(archive / "catalog.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM publications WHERE status='ready'").fetchone()[0] == 0
        path, downloaded = db.execute("SELECT path,downloaded_at FROM source_files").fetchone()
        assert downloaded is not None and (root / path).exists()
    assert partials and all(path.stat().st_size == 17 for path in partials)
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], archive,
                                 processing_workers=1)
    assert result["completed"] == 24 and len(files.downloaded) == 1
    assert all(not path.exists() for path in partials)
    assert not (root / path).exists()
