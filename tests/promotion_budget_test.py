"""Synthetic inventory and growing streams cannot monopolize promotion work."""

import os

import pytest

from podcast_transcript import safe_io, single_track
from single_track_test import RecordingBackend


def test_promotion_inventory_is_bounded_before_sorting(audio, monkeypatch):
    audio.episode_chunks_dir.mkdir(parents=True)
    for index in range(20):
        (audio.episode_chunks_dir / f"chunk_{index:03}.json").write_text("{}")
    monkeypatch.setattr(safe_io, "TRANSCRIPT_WORK_MAX_ENTRIES", 6, raising=False)
    scan = os.scandir
    consumed = []

    class TrackedScan:
        def __init__(self, path):
            self.entries = scan(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.entries.close()

        def __iter__(self):
            for entry in self.entries:
                consumed.append(entry.name)
                yield entry

    monkeypatch.setattr(os, "scandir", TrackedScan)
    with pytest.raises(RuntimeError, match="Transcript work.*entry"):
        single_track._transcript_files(audio)
    assert len(consumed) == 7


def test_growing_backup_stops_and_preserves_previous_recovery(episode, monkeypatch):
    store, _ = episode
    single_track.transcribe(store.url, RecordingBackend("second", "second"))
    recovery = store.podcast_dir / ".previous-transcripts"
    old_recovery = {p: p.read_bytes() for p in recovery.rglob("*") if p.is_file()}
    current = {
        p: p.read_bytes()
        for p in [*single_track._transcript_files(store), store.cache_metadata_path]
    }
    copy = single_track.copy_file_atomic
    read = os.read
    target_inode = store.output_path(".txt").stat().st_ino
    reads = []
    monkeypatch.setattr(safe_io, "TRANSCRIPT_WORK_MAX_BYTES", 16 * 1024, raising=False)

    def growing(fd, size):
        if os.fstat(fd).st_ino == target_inode:
            reads.append(size)
            if len(reads) > 20:
                pytest.fail("unbounded growing backup read")
            return b"x" * min(size, 1024)
        return read(fd, size)

    def copy_growing(source, target, **kwargs):
        with monkeypatch.context() as patch:
            if any(part.startswith(".backup-") for part in target.parts):
                patch.setattr(os, "read", growing)
            return copy(source, target, **kwargs)

    monkeypatch.setattr(single_track, "copy_file_atomic", copy_growing)
    with pytest.raises(RuntimeError, match="Transcript work.*byte"):
        single_track.transcribe(store.url, RecordingBackend("third", "third"))
    assert reads and len(reads) <= 17
    assert all(p.read_bytes() == data for p, data in current.items())
    assert all(p.read_bytes() == data for p, data in old_recovery.items())
    assert not list(store.podcast_dir.glob(".backup-*"))
    monkeypatch.setattr(single_track, "copy_file_atomic", copy)
    single_track.restore_last_success(store.url)


def test_readability_scan_has_one_elapsed_deadline(episode, monkeypatch):
    store, previous = episode
    now = [0.0]
    read = os.read

    def elapsed(fd, size):
        now[0] += 0.4
        return read(fd, size)

    monkeypatch.setattr(safe_io, "TRANSCRIPT_WORK_TIMEOUT_SECONDS", 1.0, raising=False)
    monkeypatch.setattr(safe_io, "monotonic", lambda: now[0], raising=False)
    monkeypatch.setattr(os, "read", elapsed)
    with pytest.raises(RuntimeError, match="Transcript work.*elapsed"):
        single_track.transcribe(store.url, RecordingBackend("new", "new"))
    assert all(
        (store.podcast_dir / p).read_bytes() == data for p, data in previous.items()
    )


def test_backup_hash_and_copy_share_byte_budget(episode, monkeypatch):
    store, previous = episode
    size = sum(len(data) for data in previous.values())
    # Enough for each file and one complete copy, not copy plus inventory hashing.
    monkeypatch.setattr(safe_io, "TRANSCRIPT_WORK_MAX_BYTES", size + 100, raising=False)
    with pytest.raises(RuntimeError, match="Transcript work.*byte"):
        single_track.transcribe(store.url, RecordingBackend("new", "new"))
    assert all(
        (store.podcast_dir / p).read_bytes() == data for p, data in previous.items()
    )
    assert not (store.podcast_dir / ".previous-transcripts").exists()


def test_standalone_marker_digest_has_finite_work(tmp_path, monkeypatch):
    path = tmp_path / "source.json"
    path.write_bytes(b"{}")
    reads = []
    monkeypatch.setattr(safe_io, "TRANSCRIPT_WORK_MAX_BYTES", 4)

    def growing(fd, size):
        reads.append(size)
        if len(reads) > 6:
            pytest.fail("unbounded marker digest")
        return b"x"

    monkeypatch.setattr(os, "read", growing)
    with pytest.raises(RuntimeError, match="Transcript work.*byte"):
        single_track._file_digest(path)
    assert len(reads) == 5
