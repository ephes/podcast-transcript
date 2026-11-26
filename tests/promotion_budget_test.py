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


def test_promotion_reuses_admitted_inventories_near_entry_limit(
    episode, monkeypatch, mocker
):
    store, previous = episode
    for index in range(20):
        (store.episode_chunks_dir / f"unrelated-{index}").touch()
    # Charge the live and staged inventories and the backup digest scan once.
    live_entries = len(list(store.episode_chunks_dir.iterdir()))
    chunk_transcripts = sum(path.parent.name == "chunks" for path in previous)
    entry_limit = live_entries + (2 + chunk_transcripts) + chunk_transcripts
    monkeypatch.setattr(safe_io, "TRANSCRIPT_WORK_MAX_ENTRIES", entry_limit)
    scans = mocker.spy(single_track, "_transcript_files")

    paths = single_track.transcribe(store.url, RecordingBackend("new", "new"))

    assert paths["plain text"].read_text() == "new\nnew"
    recovery = store.podcast_dir / ".previous-transcripts"
    assert all(
        (recovery / path).read_bytes() == data for path, data in previous.items()
    )
    directories = [call.args[0].podcast_dir for call in scans.call_args_list]
    assert directories.count(store.podcast_dir) == 2  # Readability and admission.
    assert sum(path.name.startswith(".replacement-") for path in directories) == 1
    assert sum(path.name.startswith(".backup-") for path in directories) == 1


def test_promotion_refuses_over_budget_staged_inventory_before_recovery(
    episode, monkeypatch
):
    store, previous = episode
    single_track.transcribe(store.url, RecordingBackend("second", "second"))
    recovery = store.podcast_dir / ".previous-transcripts"
    recovery_manifest = (recovery / "recovery.json").read_bytes()
    current = {path: (store.podcast_dir / path).read_bytes() for path in previous}
    # The live inventory fits, but the staged inventory cannot be admitted.
    monkeypatch.setattr(
        safe_io,
        "TRANSCRIPT_WORK_MAX_ENTRIES",
        len(list(store.episode_chunks_dir.iterdir())),
    )

    with pytest.raises(RuntimeError, match="Transcript work.*entry"):
        single_track.transcribe(store.url, RecordingBackend("third", "third"))

    assert all(
        (store.podcast_dir / path).read_bytes() == data
        for path, data in current.items()
    )
    assert (recovery / "recovery.json").read_bytes() == recovery_manifest
    assert all(
        (recovery / path).read_bytes() == data for path, data in previous.items()
    )


@pytest.mark.parametrize("changed", ["live", "staged"])
def test_promotion_refuses_inventory_changes_during_backup(
    episode, monkeypatch, changed
):
    store, previous = episode
    single_track.transcribe(store.url, RecordingBackend("second", "second"))
    recovery = store.podcast_dir / ".previous-transcripts"
    recovery_manifest = (recovery / "recovery.json").read_bytes()
    current = {path: (store.podcast_dir / path).read_bytes() for path in previous}
    digest = single_track._generation_digest

    def add_chunk_after_digest(audio, **kwargs):
        result = digest(audio, **kwargs)
        directory = store.episode_chunks_dir
        if changed == "staged":
            directory = next(store.podcast_dir.glob(".replacement-*")) / "chunks"
        (directory / "chunk_999.json").write_text("{}")
        return result

    monkeypatch.setattr(single_track, "_generation_digest", add_chunk_after_digest)
    with pytest.raises(RuntimeError, match="Transcript inventory changed"):
        single_track.transcribe(store.url, RecordingBackend("third", "third"))

    assert all(
        (store.podcast_dir / path).read_bytes() == data
        for path, data in current.items()
    )
    assert (recovery / "recovery.json").read_bytes() == recovery_manifest
    assert all(
        (recovery / path).read_bytes() == data for path, data in previous.items()
    )


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


def test_readability_budget_refusal_preserves_published_transcripts(
    episode, monkeypatch
):
    store, previous = episode
    single_track.transcribe(store.url, RecordingBackend("second", "second"))
    recovery = store.podcast_dir / ".previous-transcripts"
    published_paths = [store.podcast_dir / name for name in previous]
    published_paths.extend(path for path in recovery.rglob("*") if path.is_file())
    published = {path: path.read_bytes() for path in published_paths}
    replacement = RecordingBackend("third", "third")
    monkeypatch.setattr(safe_io, "TRANSCRIPT_WORK_MAX_BYTES", 1)

    with pytest.raises(
        RuntimeError,
        match="Transcript work exceeded byte limit.*previous transcripts were preserved",
    ):
        single_track.transcribe(store.url, replacement)

    assert not replacement.calls
    assert not list(store.podcast_dir.glob(".replacement-*"))
    assert all(path.read_bytes() == data for path, data in published.items())


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


@pytest.mark.parametrize(
    "artifact", ["earlier-export", "later-export", "raw", "metadata"]
)
def test_promotion_refuses_file_changes_during_backup(episode, monkeypatch, artifact):
    store, previous = episode
    single_track.transcribe(store.url, RecordingBackend("second", "second"))
    recovery = store.podcast_dir / ".previous-transcripts"
    manifest = (recovery / "recovery.json").read_bytes()
    current = {path: (store.podcast_dir / path).read_bytes() for path in previous}
    first_export = store.output_path(".dote.json")
    changed = {
        "earlier-export": first_export,
        "later-export": store.output_path(".txt"),
        "raw": store.episode_chunks_dir / "chunk_000.json",
        "metadata": store.cache_metadata_path,
    }[artifact]
    relative = changed.relative_to(store.podcast_dir)
    edited = current[relative].replace(b"second", b"edited")
    assert edited != current[relative]
    copy = single_track.copy_file_atomic

    def edit_after_first_export(source, target, **kwargs):
        copy(source, target, **kwargs)
        if source == first_export and target.parent.name.startswith(".backup-"):
            changed.write_bytes(edited)

    monkeypatch.setattr(single_track, "copy_file_atomic", edit_after_first_export)
    with pytest.raises((OSError, RuntimeError), match="Transcript.*changed"):
        single_track.transcribe(store.url, RecordingBackend("third", "third"))

    assert changed.read_bytes() == edited
    assert all(
        (store.podcast_dir / path).read_bytes() == data
        for path, data in current.items()
        if path != relative
    )
    assert (recovery / "recovery.json").read_bytes() == manifest
    assert all(
        (recovery / path).read_bytes() == data for path, data in previous.items()
    )
    assert not list(store.podcast_dir.glob(".backup-*"))
    single_track.restore_last_success(store.url)
    assert all(
        (store.podcast_dir / path).read_bytes() == data
        for path, data in previous.items()
    )
