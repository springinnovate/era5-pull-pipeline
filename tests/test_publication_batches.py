"""Exercise coalesced publication at real queue and durability boundaries."""

from datetime import date
from pathlib import Path
import sqlite3
from threading import Event, Lock
from typing import Any

import pytest

from era5_pull_pipeline import catalog, pipeline
from test_ncar_pipeline import NcarFiles, files


@pytest.mark.parametrize("local_staging", [False, True])
@pytest.mark.parametrize("fail_publication", [False, True])
def test_ready_chunks_share_commits_and_keep_native_batch_bound(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], local_staging: bool, fail_publication: bool,
) -> None:
    """Combine waiting chunks while counting each against the staging allowance."""
    archive = tmp_path / "archive"
    local = tmp_path / "ssd" if local_staging else None
    files.add_file("2t", "2000-01-01T00", "2000-01-02T23")
    first_started, four_encoded, five_encoded, four_uploaded = Event(), Event(), Event(), Event()
    lock = Lock()
    started = encoded = uploaded = 0
    sizes: list[int] = []
    encode, copy, publish = pipeline.write_ncar_chunk_cogs, pipeline.copy_local_cogs, pipeline.publish_observations

    def write(*args: Any) -> list[dict[str, Any]]:
        """Let the first chunk publish before filling the remaining bounded slots."""
        nonlocal started, encoded
        with lock:
            started += 1
            ordinal = started
        if ordinal > 1:
            assert first_started.wait(10)
        result = encode(*args)
        with lock:
            encoded += 1
            if encoded == 4:
                four_encoded.set()
            if encoded == 5:
                five_encoded.set()
        return result

    def upload(*args: Any, **kwargs: Any) -> None:
        """Observe durable uploads before releasing the first publication."""
        nonlocal uploaded
        copy(*args, **kwargs)
        with lock:
            uploaded += 1
            if uploaded == 4:
                four_uploaded.set()

    def held_publish(
        db: sqlite3.Connection, destination: Path,
        observations: list[tuple[str, dict[str, Any]]], **kwargs: Any,
    ) -> list[Path]:
        """Hold two consecutive commits to expose both coalescing and backpressure."""
        sizes.append(len(observations))
        ordinal = len(sizes)
        if ordinal == 1:
            first_started.set()
            assert four_encoded.wait(10), "encoding waited for publication"
            if local_staging:
                assert four_uploaded.wait(10), "uploads waited for publication"
            Event().wait(.2)
            assert encoded == 4
            assert len(observations) == 6
            assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
        elif ordinal == 2:
            assert len(observations) == 18, "ready native chunks were not combined"
            assert five_encoded.wait(10)
            Event().wait(.2)
            assert encoded == 5, "one coalesced request consumed only one staging slot"
            assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 6
            source = db.execute("SELECT path FROM source_files WHERE removed_at IS NULL").fetchone()[0]
            assert ((local or archive) / source).exists()
            if fail_publication:
                raise RuntimeError("coalesced publication failed")
        return publish(db, destination, observations, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "write_ncar_chunk_cogs", write)
        patch.setattr(pipeline, "copy_local_cogs", upload)
        patch.setattr(pipeline, "publish_observations", held_publish)
        if fail_publication:
            with pytest.raises(RuntimeError, match="coalesced publication failed"):
                pipeline.pull_range(
                    date(2000, 1, 1), date(2000, 1, 2), ["2t"], archive,
                    staging_dir=local, processing_workers=2,
                )
            assert "Summary: completed=6, skipped=0, unavailable=0, failed=42, pending=0" in capsys.readouterr().out
            with sqlite3.connect(archive / "catalog.sqlite") as db:
                assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 6
                source = db.execute("SELECT path FROM source_files WHERE removed_at IS NULL").fetchone()[0]
                assert ((local or archive) / source).exists()
        else:
            result = pipeline.pull_range(
                date(2000, 1, 1), date(2000, 1, 2), ["2t"], archive,
                staging_dir=local, processing_workers=2,
            )
            assert result == {"completed": 48, "skipped": 0, "unavailable": 0, "failed": 0, "pending": 0}
            assert sum(sizes) == 48 and len(sizes) < 8
    if fail_publication:
        resumed = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 2), ["2t"], archive, processing_workers=2)
        assert resumed["completed"] == 42 and resumed["skipped"] == 6
    assert len(files.downloaded) == 1
    with catalog.archive_session(archive) as db:
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 48
        assert db.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM source_files WHERE removed_at IS NULL").fetchone()[0] == 0


@pytest.mark.parametrize("local_staging", [False, True])
def test_coalesced_publication_committed_during_shutdown_counts_every_chunk(
    tmp_path: Path, files: NcarFiles, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], local_staging: bool,
) -> None:
    """Count every durable chunk when encoding fails during a combined publication."""
    archive = tmp_path / "archive"
    local = tmp_path / "ssd" if local_staging else None
    files.add_file("2t", "2000-01-01T00", "2000-01-02T23")
    first_started, four_encoded, four_uploaded, second_started, stopping = (Event() for _ in range(5))
    lock = Lock()
    started = encoded = uploaded = publications = 0
    encode, copy, publish = pipeline.write_ncar_chunk_cogs, pipeline.copy_local_cogs, pipeline.publish_observations
    cancel_pending = pipeline.PublicationWorker.cancel_pending

    def write(*args: Any) -> list[dict[str, Any]]:
        """Fail the fifth chunk while three earlier chunks await their joint commit."""
        nonlocal started, encoded
        with lock:
            started += 1
            ordinal = started
        if ordinal > 1:
            assert first_started.wait(10)
        if ordinal == 5:
            assert second_started.wait(10)
            raise RuntimeError("encoding interrupted")
        result = encode(*args)
        with lock:
            encoded += 1
            if encoded == 4:
                four_encoded.set()
        return result

    def upload(*args: Any, **kwargs: Any) -> None:
        """Observe all four completed uploads before allowing coalescing."""
        nonlocal uploaded
        copy(*args, **kwargs)
        with lock:
            uploaded += 1
            if uploaded == 4:
                four_uploaded.set()

    def stop(worker: pipeline.PublicationWorker) -> None:
        """Release the executing publication only after shutdown has begun."""
        stopping.set()
        cancel_pending(worker)

    def held_publish(
        db: sqlite3.Connection, destination: Path,
        observations: list[tuple[str, dict[str, Any]]], **kwargs: Any,
    ) -> list[Path]:
        """Finish the second publication during coordinator error cleanup."""
        nonlocal publications
        publications += 1
        if publications == 1:
            first_started.set()
            assert four_encoded.wait(10)
            if local_staging:
                assert four_uploaded.wait(10)
            Event().wait(.2)
            assert len(observations) == 6
        else:
            assert publications == 2 and len(observations) == 18
            second_started.set()
            assert stopping.wait(10)
        return publish(db, destination, observations, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "write_ncar_chunk_cogs", write)
        patch.setattr(pipeline, "copy_local_cogs", upload)
        patch.setattr(pipeline, "publish_observations", held_publish)
        patch.setattr(pipeline.PublicationWorker, "cancel_pending", stop)
        with pytest.raises(RuntimeError, match="encoding interrupted"):
            pipeline.pull_range(
                date(2000, 1, 1), date(2000, 1, 2), ["2t"], archive,
                staging_dir=local, processing_workers=2,
            )
    assert "Summary: completed=24, skipped=0, unavailable=0, failed=24, pending=0" in capsys.readouterr().out
    with sqlite3.connect(archive / "catalog.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 24
        assert db.execute("SELECT COUNT(*) FROM source_files WHERE removed_at IS NULL").fetchone()[0] == 1
    result = pipeline.pull_range(date(2000, 1, 1), date(2000, 1, 2), ["2t"], archive, processing_workers=2)
    assert result["completed"] == 24 and result["skipped"] == 24
    assert len(files.downloaded) == 1
