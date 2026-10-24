"""Exercise local source/COG staging and durable uploads at real boundaries."""

from datetime import date
from pathlib import Path
import sqlite3
from threading import Event, Lock
from typing import Any

import portalocker
import pytest

from archive_assertions import read_raster_values
from test_ncar_pipeline import NcarFiles, files
from era5_pull_pipeline import catalog, ncar_download, ncar_ingest, pipeline, staging
from era5_pull_pipeline.ncar import SourceFile


def test_local_pull_reuses_archive_sources_and_remembers_local_root(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decode existing archive sources and new local sources into local COGs."""
    archive, local = tmp_path / "archive", tmp_path / "ssd"
    first = files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    files.add_file("2t", "2000-01-02T00", "2000-01-02T23")
    with catalog.archive_session(archive) as db:
        rows = catalog.register_sources(db, archive, [first], date(2000, 1, 1), date(2000, 1, 1))
        files.download(db, [first], rows, archive=archive)
    writer, upload = pipeline.write_ncar_chunk_cogs, pipeline.copy_local_cogs
    inputs: set[str] = set()
    copied: list[Path] = []

    def transfer(source: SourceFile, target: Path, **kwargs: Any) -> None:
        """Place every newly downloaded input on the selected local disk."""
        assert target.is_relative_to(local)
        files.transfer(source, target, **kwargs)

    def encode(path: Path, *args: Any) -> list[dict[str, Any]]:
        """Keep both old-source reads and all local conversion paths explicit."""
        inputs.add("local" if path.is_relative_to(local) else "archive")
        assert all(target.is_relative_to(local) for target in args[-1])
        return writer(path, *args)

    def copy(root: Path, destination: Path, paths: list[Path], **kwargs: Any) -> None:
        """Copy exact validated bytes while the NetCDF sources remain retained."""
        assert root == local and destination == archive
        upload(root, destination, paths, **kwargs)
        for path in paths:
            assert path.read_bytes() == (archive / path.relative_to(local)).read_bytes()
        copied.extend(paths)

    monkeypatch.setattr(ncar_download, "download_file", transfer)
    monkeypatch.setattr(pipeline, "write_ncar_chunk_cogs", encode)
    monkeypatch.setattr(pipeline, "copy_local_cogs", copy)
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 2), ["2t"], archive,
                                 staging_dir=local, processing_workers=2)
    assert result["completed"] == 48 and inputs == {"local", "archive"}
    assert len(copied) == 48 and all(not path.exists() for path in copied)
    assert read_raster_values(archive, 2000, 1)["2_metre_temperature"].shape == (48, 3, 4)
    with catalog.archive_session(archive) as db:
        assert catalog.get_staging_dir(db) == local
        assert db.execute("SELECT COUNT(*) FROM local_files").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM source_files WHERE removed_at IS NULL").fetchone()[0] == 0
        assert all(not Path(row[0]).is_absolute() for row in db.execute("SELECT path FROM observations"))
    before = len(files.downloaded)
    repeated = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 2), ["2t"], archive)
    assert repeated["skipped"] == 48 and len(files.downloaded) == before
    moved, new_local = tmp_path / "moved-archive", tmp_path / "new-ssd"
    archive.rename(moved)
    relocated = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 2), ["2t"], moved,
                                    staging_dir=new_local)
    assert relocated["skipped"] == 48 and len(files.downloaded) == before
    assert read_raster_values(moved, 2000, 1)["2_metre_temperature"].shape == (48, 3, 4)
    with catalog.archive_session(moved) as db:
        assert catalog.get_staging_dir(db) == new_local


def test_slow_uploads_bound_outputs_without_stopping_decode_or_downloads(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fill four output batches while two uploads wait and later downloads finish."""
    archive, local = tmp_path / "archive", tmp_path / "ssd"
    files.add_file("2t", "2000-01-01T00", "2000-01-02T23")
    files.add_file("10u", "2000-01-01T00", "2000-01-01T23")
    files.add_file("10u", "2000-01-02T00", "2000-01-02T23")
    upload_started, four_encoded, last_download, checked = Event(), Event(), Event(), Event()
    lock = Lock()
    encoded = transfers = uploads = active = peak = 0
    encode, copy = pipeline.write_ncar_chunk_cogs, pipeline.copy_local_cogs

    def transfer(source: SourceFile, target: Path, **kwargs: Any) -> None:
        """Refill both later downloads while archive upload is held."""
        nonlocal transfers
        transfers += 1
        if transfers > 1:
            assert upload_started.wait(10)
        files.transfer(source, target, **kwargs)
        if transfers == 3:
            last_download.set()

    def write(*args: Any) -> list[dict[str, Any]]:
        """Count complete local batches with only one decoded chunk reserved."""
        nonlocal encoded
        result = encode(*args)
        with lock:
            encoded += 1
            if encoded == 4:
                four_encoded.set()
        return result

    def slow_copy(*args: Any, **kwargs: Any) -> None:
        """Bound concurrent uploads and all unacknowledged local COG batches."""
        nonlocal uploads, active, peak
        with lock:
            uploads += 1
            ordinal = uploads
            active += 1
            peak = max(peak, active)
        try:
            upload_started.set()
            if ordinal == 1:
                assert four_encoded.wait(10), "encoding waited for upload"
                assert last_download.wait(10), "downloads waited for upload"
                Event().wait(.15)
                assert encoded == 4
                assert len(list((local / "staging/observations").glob("*.tif"))) == 24
                with sqlite3.connect(archive / "catalog.sqlite") as db:
                    assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
                checked.set()
            else:
                assert checked.wait(10)
            copy(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(ncar_download, "download_file", transfer)
    monkeypatch.setattr(pipeline, "write_ncar_chunk_cogs", write)
    monkeypatch.setattr(pipeline, "copy_local_cogs", slow_copy)
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 2), ["2t", "10u"], archive,
                                 staging_dir=local, download_workers=1, processing_workers=2, upload_workers=2,
                                 max_source_gb=sum(source.size for source in files.sources) / 1e9,
                                 max_decoded_gb=6 * 3 * 4 * 4 / 1e9)
    assert checked.is_set() and peak == 2 and result["completed"] == 96


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_partial_upload_retains_local_inputs_and_resumes(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException],
) -> None:
    """Never publish partial uploads and rebuild them using retained local inputs."""
    archive, local = tmp_path / "archive", tmp_path / "ssd"
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    partials: list[Path] = []

    def interrupt(root: Path, destination: Path, paths: list[Path], **kwargs: Any) -> None:
        """Fail after putting an incomplete COG on the archive filesystem."""
        path = destination / paths[0].relative_to(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(paths[0].read_bytes()[:17])
        partials.append(path)
        raise failure("upload interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "copy_local_cogs", interrupt)
        with pytest.raises(failure, match="upload interrupted"):
            pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], archive,
                                staging_dir=local, processing_workers=2)
    with sqlite3.connect(archive / "catalog.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
        path, downloaded = db.execute("SELECT path,downloaded_at FROM source_files").fetchone()
        assert downloaded is not None and (local / path).exists() and not (archive / path).exists()
    assert partials and all(path.stat().st_size == 17 for path in partials)
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], archive, processing_workers=2)
    assert result["completed"] == 24 and len(files.downloaded) == 1
    assert all(not path.exists() for path in partials)
    assert not (local / path).exists()


def test_local_cleanup_failure_recovers_ready_archive_copy(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover installed COGs when local cleanup fails before the coverage commit."""
    archive, local = tmp_path / "archive", tmp_path / "ssd"
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    unlink = Path.unlink

    def busy(path: Path, missing_ok: bool = False) -> None:
        """Keep completed local COG cleanup pending until the next invocation."""
        if path.is_relative_to(local) and path.suffix == ".tif":
            raise PermissionError("local COG busy")
        unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", busy)
        with pytest.raises(PermissionError, match="local COG busy"):
            pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], archive,
                                staging_dir=local, processing_workers=1, upload_workers=1)
    with sqlite3.connect(archive / "catalog.sqlite") as db:
        ready = db.execute("SELECT COUNT(*) FROM publications WHERE status='ready'").fetchone()[0]
        assert ready >= 6
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
        source = local / db.execute("SELECT path FROM source_files").fetchone()[0]
        assert source.exists()
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], archive)
    assert result["skipped"] == ready and result["completed"] == 24 - ready
    assert len(files.downloaded) == 1 and not source.exists()


def test_staging_root_change_and_lock_protect_pending_sources(tmp_path: Path, files: NcarFiles) -> None:
    """Keep local ownership stable across commands while unfinished files exist."""
    archive, local = tmp_path / "archive", tmp_path / "ssd"
    source = files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    with catalog.archive_session(archive, staging_dir=local) as db:
        catalog.register_sources(db, archive, [source], date(2000, 1, 1), date(2000, 1, 1))
        with pytest.raises(portalocker.exceptions.LockException):
            with catalog.archive_session(tmp_path / "other-archive", staging_dir=local):
                pytest.fail("two writers shared local scratch storage")
    with pytest.raises(ValueError, match="Local files remain"):
        with catalog.archive_session(archive, staging_dir=tmp_path / "other-ssd"):
            pytest.fail("pending source location changed")
    with pytest.raises(ValueError, match="outside the archive"):
        with catalog.archive_session(archive, staging_dir=archive / "local"):
            pytest.fail("overlapping staging roots accepted")


def test_separate_ingestion_honors_remembered_local_storage(tmp_path: Path, files: NcarFiles) -> None:
    """Use local source paths in the standalone downloader and ingestion command."""
    archive, local = tmp_path / "archive", tmp_path / "ssd"
    source = files.add_file("2t", "2000-01-01T00", "2000-01-01T23")
    with catalog.archive_session(archive, staging_dir=local) as db:
        rows = catalog.register_sources(db, archive, [source], date(2000, 1, 1), date(2000, 1, 1))
        ncar_download.download_registered_sources(db, [source], rows, archive=archive)
        assert (local / rows[0]["path"]).exists()
    ncar_ingest.ingest_ncar(archive, 2000, 1)
    assert read_raster_values(archive, 2000, 1)["2_metre_temperature"].shape == (24, 3, 4)
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM local_files").fetchone()[0] == 0


def test_upload_flush_failure_never_records_ready_state(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain inputs when the actual copy cannot durably flush its destination."""
    archive, local = tmp_path / "archive", tmp_path / "ssd"
    files.add_file("2t", "2000-01-01T00", "2000-01-01T23")

    def failed_flush(descriptor: int) -> None:
        """Fail the upload's durable file boundary."""
        raise OSError("upload flush failed")

    with monkeypatch.context() as patch:
        patch.setattr(staging.os, "fsync", failed_flush)
        with pytest.raises(OSError, match="upload flush failed"):
            pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], archive,
                                staging_dir=local, processing_workers=1, upload_workers=1)
    with sqlite3.connect(archive / "catalog.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM publications WHERE status='ready'").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
        source = local / db.execute("SELECT path FROM source_files").fetchone()[0]
        assert source.exists()
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 1), ["2t"], archive)
    assert result["completed"] == 24 and len(files.downloaded) == 1
    assert not source.exists()
