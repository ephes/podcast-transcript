"""Synthetic local admission and durable completion publication regressions."""

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from podcast_transcript import audio, safe_io
from podcast_transcript.config import settings


@pytest.mark.parametrize("operation", ["hash", "copy"])
def test_local_byte_budget_preserves_completed_files(tmp_path, monkeypatch, operation):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"too-large")
    store = audio.Audio(base_dir=tmp_path / "cache", url=str(source))
    store.podcast_dir.mkdir(parents=True)
    store.source_metadata_path.write_text('{"sha256":"previous"}')
    previous = store.output_path(".txt")
    previous.write_text("previous transcript")
    monkeypatch.setattr(
        audio,
        "resample_audio",
        lambda *args: pytest.fail("admission reached media work"),
    )
    monkeypatch.setattr(settings, "transcript_source_max_bytes", 4)
    with pytest.raises(RuntimeError, match="TRANSCRIPT_SOURCE_MAX_BYTES"):
        if operation == "hash":
            store.prepare_audio_for_transcription()
        else:
            store.make_sure_audio_file_exists()
    assert previous.read_text() == "previous transcript"
    assert store.source_metadata_path.read_text() == '{"sha256":"previous"}'
    assert not store.episode_path.exists()


def test_local_descriptor_refuses_special_file_before_read(tmp_path, monkeypatch):
    source = tmp_path / "synthetic-special"
    source.write_bytes(b"unused")
    real_fstat = os.fstat
    opened = []
    real_open = os.open

    def open_file(path, flags, *args, **kwargs):
        if Path(path) == source:
            opened.append(flags)
        return real_open(path, flags, *args, **kwargs)

    def fstat(fd):
        info = real_fstat(fd)
        return os.stat_result((stat.S_IFIFO | 0o600, *tuple(info)[1:]))

    monkeypatch.setattr(os, "open", open_file)
    monkeypatch.setattr(os, "fstat", fstat)
    with pytest.raises(RuntimeError, match="regular file"):
        audio.Audio._content_fingerprint(source)
    assert opened and opened[0] & getattr(os, "O_NONBLOCK", 0) == getattr(
        os, "O_NONBLOCK", 0
    )


def test_growing_local_descriptor_is_bounded(tmp_path, monkeypatch):
    source = tmp_path / "growing.mp3"
    source.write_bytes(b"abc")
    monkeypatch.setattr(settings, "transcript_source_max_bytes", 5)
    real_read = os.read
    reads = []

    def growing(fd, size):
        block = real_read(fd, min(size, 3))
        reads.append(len(block))
        with source.open("ab") as file:
            file.write(b"abc")
        assert len(reads) < 5, "unbounded source reader"
        return block

    monkeypatch.setattr(os, "read", growing)
    with pytest.raises(RuntimeError, match="TRANSCRIPT_SOURCE_MAX_BYTES"):
        audio.Audio._content_fingerprint(source)
    assert reads == [3, 3]


def test_local_elapsed_budget_checked_during_hash(tmp_path, monkeypatch):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"abc")
    monkeypatch.setattr(settings, "transcript_source_total_timeout", 1)
    ticks = iter([0, 0, 2, 2, 2])
    monkeypatch.setattr(audio, "monotonic", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="TRANSCRIPT_SOURCE_TOTAL_TIMEOUT"):
        audio.Audio._content_fingerprint(source)


def test_local_copy_uses_admitted_descriptor(tmp_path, monkeypatch):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"admitted")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replacement")
    store = audio.Audio(base_dir=tmp_path / "cache", url=str(source))
    real_open = os.open
    swapped = []

    def open_file(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) == source:
            os.replace(replacement, source)
            swapped.append(True)
        return fd

    monkeypatch.setattr(os, "open", open_file)
    store.make_sure_audio_file_exists()
    assert swapped
    assert store.episode_path.read_bytes() == b"admitted"


def record_publication(monkeypatch):
    events = []
    real_sync, real_replace = os.fsync, os.replace

    def sync(fd):
        events.append(
            ("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file", None)
        )
        return real_sync(fd)

    def replace(source, target):
        events.append(("replace", Path(target).name))
        return real_replace(source, target)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    return events


def test_atomic_output_syncs_data_then_rename_then_directory(tmp_path, monkeypatch):
    events = record_publication(monkeypatch)
    safe_io.write_text_atomic(tmp_path / "result.json", "complete")
    assert events[0] == ("file", None)
    assert events[1] == ("replace", "result.json")
    assert events[2][0] == "directory"


@pytest.mark.parametrize("phase", ["file", "directory", "replace"])
def test_publication_failure_does_not_publish_completion(tmp_path, monkeypatch, phase):
    store = audio.Audio(base_dir=tmp_path, url="https://example.test/episode.mp3")
    store.episode_chunks_dir.mkdir(parents=True)
    store.resampled_episode_path.write_bytes(b"audio")
    monkeypatch.setattr(audio, "get_audio_duration", lambda path: 1)
    real_sync, real_replace = os.fsync, os.replace
    faults = []

    def sync(fd):
        kind = "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
        if phase == kind:
            faults.append(kind)
            raise OSError("synthetic sync failure")
        return real_sync(fd)

    def replace(source, target):
        if phase == "replace":
            faults.append(phase)
            raise OSError("synthetic replace failure")
        return real_replace(source, target)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(OSError, match="synthetic"):
        store.split_into_chunks()
    assert faults
    assert not store.chunks_manifest_path.exists()
    assert store._load_chunk_manifest() is None
    assert store.resampled_episode_path.read_bytes() == b"audio"


def test_source_completion_follows_durable_audio_and_chunk_marker(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"synthetic")
    store = audio.Audio(base_dir=tmp_path / "cache", url=str(source))
    monkeypatch.setattr(audio, "resample_audio", shutil.copyfile)
    monkeypatch.setattr(audio, "get_audio_duration", lambda path: 1)
    events = record_publication(monkeypatch)
    store.prepare_audio_for_transcription()
    replacements = [
        (i, name) for i, (kind, name) in enumerate(events) if kind == "replace"
    ]
    assert replacements[-1][1] == "source.json"
    last_chunks = max(i for i, name in replacements if name == "chunks.json")
    assert any(
        kind == "directory" for kind, _ in events[last_chunks + 1 : replacements[-1][0]]
    )
    assert events[-1][0] == "directory"
    assert json.loads(store.source_metadata_path.read_text()) == {
        "sha256": hashlib.sha256(b"synthetic").hexdigest()
    }
