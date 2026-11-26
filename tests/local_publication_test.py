"""Synthetic local admission and durable completion publication regressions."""

import errno
import hashlib
import json
import os
import shutil
import stat
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_transcript import audio, safe_io
from podcast_transcript.config import settings
from podcast_transcript.json_input import load_control_json


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


@pytest.mark.parametrize("limit", [True, 4.5, 4.0, b"4"])
def test_invalid_source_byte_limit_is_refused_before_open(tmp_path, monkeypatch, limit):
    monkeypatch.setattr(settings, "transcript_source_max_bytes", limit)
    with pytest.raises(ValueError, match="TRANSCRIPT_SOURCE_MAX_BYTES"):
        audio.Audio._content_fingerprint(tmp_path / "missing.mp3")


@pytest.mark.parametrize("limit", [4, " +4 "])
def test_source_byte_limit_accepts_exact_number_of_bytes(tmp_path, monkeypatch, limit):
    monkeypatch.setattr(settings, "transcript_source_max_bytes", limit)
    source = tmp_path / "source.mp3"
    source.write_bytes(b"abcd")
    assert (
        audio.Audio._content_fingerprint(source) == hashlib.sha256(b"abcd").hexdigest()
    )

    source.write_bytes(b"abcde")
    with pytest.raises(RuntimeError, match="TRANSCRIPT_SOURCE_MAX_BYTES"):
        audio.Audio._content_fingerprint(source)


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


def test_atomic_output_publishes_near_limit_media_name(tmp_path):
    target = tmp_path / ("a" * 250 + ".mp3")
    with safe_io.atomic_output(target) as stage:
        assert stage.suffix == ".mp3"
        assert stage.parent.parent == target.parent
        stage.write_bytes(b"synthetic audio")

    assert target.read_bytes() == b"synthetic audio"
    assert not stage.parent.exists()


@pytest.mark.parametrize("platform", ["posix", "nt"])
def test_atomic_output_syncs_data_then_rename_then_directory(
    tmp_path, monkeypatch, platform
):
    events = record_publication(monkeypatch)
    record_sync, record_replace = os.fsync, os.replace
    synced_descriptor = None

    def sync(descriptor):
        nonlocal synced_descriptor
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            synced_descriptor = descriptor
        record_sync(descriptor)

    def replace(source, target):
        assert synced_descriptor is not None
        if platform == "nt":
            with pytest.raises(OSError) as raised:
                os.fstat(synced_descriptor)
            assert raised.value.errno == errno.EBADF
        else:
            assert os.fstat(synced_descriptor).st_ino == source.lstat().st_ino
        record_replace(source, target)

    platform_os = SimpleNamespace(
        **{**vars(os), "name": platform, "fsync": sync, "replace": replace}
    )
    monkeypatch.setattr(safe_io, "os", platform_os)
    safe_io.write_text_atomic(tmp_path / "result.json", "complete")
    assert events[0] == ("file", None)
    assert events[1] == ("replace", "result.json")
    assert events[2][0] == "directory"


@pytest.mark.parametrize("change", ["replace", "rewrite", "unlink"])
@pytest.mark.skipif(
    os.name == "nt", reason="Fault mutates a file opened without delete sharing"
)
def test_atomic_output_refuses_stage_changed_after_sync(tmp_path, monkeypatch, change):
    target = tmp_path / "result.json"
    target.write_bytes(b"previous")
    real_sync = os.fsync
    synced = []

    def sync_then_change(descriptor):
        real_sync(descriptor)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return
        synced.append(descriptor)
        if change == "replace":
            replacement = stage.with_name("different.json")
            replacement.write_bytes(b"changed")
            replacement.replace(stage)
        elif change == "rewrite":
            stage.write_bytes(b"changed size")
        else:
            stage.unlink()

    monkeypatch.setattr(os, "fsync", sync_then_change)
    with pytest.raises(RuntimeError, match="Staged output changed before publication"):
        with safe_io.atomic_output(target) as stage:
            stage.write_bytes(b"replacement")
    assert len(synced) == 1
    assert target.read_bytes() == b"previous"
    assert not stage.parent.exists()


@pytest.mark.parametrize(
    ("writer", "data"),
    [
        (safe_io.write_text_atomic, "replacement"),
        (safe_io.write_bytes_atomic, b"replacement"),
    ],
)
def test_atomic_writer_closes_descriptor_when_stream_construction_fails(
    tmp_path, monkeypatch, writer, data
):
    target = tmp_path / "transcript.json"
    target.write_bytes(b"previous")
    error = OSError("synthetic stream construction failure")
    descriptors = []

    def fail_adoption(fd, *args, **kwargs):
        assert stat.S_ISREG(os.fstat(fd).st_mode)
        descriptors.append(fd)
        raise error

    monkeypatch.setattr(os, "fdopen", fail_adoption)
    with pytest.raises(OSError) as raised:
        writer(target, data)

    [descriptor] = descriptors
    try:
        os.fstat(descriptor)
    except OSError as exc:
        assert exc.errno == errno.EBADF
    else:
        os.close(descriptor)
        pytest.fail("staged output descriptor leaked")
    assert raised.value is error
    assert target.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".output-*"))


@pytest.mark.parametrize(
    ("writer", "data"),
    [
        (safe_io.write_text_atomic, '{"cue":"replacement"}'),
        (safe_io.write_bytes_atomic, b'{"cue":"replacement"}'),
    ],
)
def test_atomic_writer_refuses_short_write(tmp_path, monkeypatch, writer, data):
    target = tmp_path / "transcript.json"
    previous = b'{"cue":"previous"}'
    target.write_bytes(previous)
    real_fdopen = os.fdopen

    @contextmanager
    def short_fdopen(fd, mode, **kwargs):
        with real_fdopen(fd, mode, **kwargs) as stream:
            yield SimpleNamespace(write=lambda payload: stream.write(payload[:-1]))

    monkeypatch.setattr(os, "fdopen", short_fdopen)
    with pytest.raises(OSError, match="Incomplete output write"):
        writer(target, data)

    assert target.read_bytes() == previous
    assert not list(tmp_path.glob(".output-*"))


def test_atomic_text_writer_round_trips_utf8_receipt(tmp_path, monkeypatch):
    target = tmp_path / "chunk_000.dote.receipt.json"
    receipt = {"label": "café 日本語"}
    text = json.dumps(receipt, ensure_ascii=False)
    real_fdopen = os.fdopen

    def fdopen_with_ascii_default(fd, mode, *, encoding="ascii", **kwargs):
        return real_fdopen(fd, mode, encoding=encoding, **kwargs)

    monkeypatch.setattr(os, "fdopen", fdopen_with_ascii_default)
    safe_io.write_text_atomic(target, text)

    assert target.read_bytes() == text.encode("utf-8")
    assert load_control_json(target, expected_type=dict) == receipt


@pytest.mark.parametrize("exhaust_budget", [True, False])
def test_atomic_copy_output_close_failure_preserves_destination(
    tmp_path, monkeypatch, caplog, exhaust_budget
):
    source, target = tmp_path / "source.json", tmp_path / "target.json"
    source.write_bytes(b"replacement")
    target.write_bytes(b"previous")
    budget = safe_io.TranscriptWorkBudget()
    if exhaust_budget:
        budget.remaining = 1
    output_descriptor = None
    close_attempts = 0
    open_descriptor, close_descriptor = os.open, os.close
    close_error = OSError(errno.EIO, "synthetic output close failure")

    def remember_output(path, flags, *args, **kwargs):
        nonlocal output_descriptor
        descriptor = open_descriptor(path, flags, *args, **kwargs)
        if flags & os.O_EXCL:
            output_descriptor = descriptor
        return descriptor

    def fail_output_close(descriptor):
        nonlocal output_descriptor, close_attempts
        close_descriptor(descriptor)
        if descriptor == output_descriptor:
            output_descriptor = None
            close_attempts += 1
            raise close_error

    monkeypatch.setattr(os, "open", remember_output)
    monkeypatch.setattr(os, "close", fail_output_close)
    if exhaust_budget:
        with pytest.raises(RuntimeError, match="Transcript work exceeded byte limit"):
            safe_io.copy_file_atomic(source, target, budget=budget)
        assert "Could not close the staged transcript output descriptor" in caplog.text
    else:
        with pytest.raises(OSError) as raised:
            safe_io.copy_file_atomic(source, target, budget=budget)
        assert raised.value is close_error

    assert close_attempts == 1
    assert target.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".output-*"))


@pytest.mark.parametrize("exhaust_budget", [True, False])
def test_atomic_copy_input_close_failure_preserves_destination(
    tmp_path, monkeypatch, caplog, exhaust_budget
):
    source, target = tmp_path / "source.json", tmp_path / "target.json"
    source.write_bytes(b"replacement")
    target.write_bytes(b"previous")
    budget = safe_io.TranscriptWorkBudget()
    if exhaust_budget:
        budget.remaining = 1
    input_descriptor = None
    close_attempts = 0
    open_descriptor, close_descriptor = os.open, os.close
    close_error = OSError(errno.EIO, "synthetic input close failure")

    def remember_input(path, flags, *args, **kwargs):
        nonlocal input_descriptor
        descriptor = open_descriptor(path, flags, *args, **kwargs)
        if path == source:
            input_descriptor = descriptor
        return descriptor

    def fail_input_close(descriptor):
        nonlocal close_attempts
        if descriptor == input_descriptor:
            close_attempts += 1
            close_descriptor(descriptor)
            raise close_error
        close_descriptor(descriptor)

    monkeypatch.setattr(os, "open", remember_input)
    monkeypatch.setattr(os, "close", fail_input_close)
    if exhaust_budget:
        with pytest.raises(RuntimeError, match="Transcript work exceeded byte limit"):
            safe_io.copy_file_atomic(source, target, budget=budget)
        assert "Could not close the transcript input descriptor" in caplog.text
    else:
        with pytest.raises(OSError) as raised:
            safe_io.copy_file_atomic(source, target, budget=budget)
        assert raised.value is close_error

    assert close_attempts == 1
    assert target.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".output-*"))


@pytest.mark.parametrize("failed_parent", ["destination", "staging"])
def test_atomic_output_reports_visible_output_after_directory_sync_failure(
    tmp_path, monkeypatch, failed_parent
):
    target = tmp_path / "result.json"
    target.write_bytes(b"previous")
    original_error = OSError("synthetic directory sync failure")
    real_sync = safe_io.sync_directory
    syncs = []

    def sync_directory(path):
        syncs.append(path)
        if path == failed_directory:
            raise original_error
        real_sync(path)

    monkeypatch.setattr(safe_io, "sync_directory", sync_directory)
    with pytest.raises(OSError) as raised:
        with safe_io.atomic_output(target) as stage:
            failed_directory = (
                target.parent if failed_parent == "destination" else stage.parent
            )
            stage.write_bytes(b"replacement")

    assert target.read_bytes() == b"replacement"
    assert not stage.parent.exists()
    expected_syncs = [target.parent]
    if failed_parent == "staging":
        expected_syncs.append(stage.parent)
    assert syncs == expected_syncs
    error = raised.value
    assert "Durability is uncertain" in str(error)
    assert isinstance(error, safe_io.PublicationUncertainError)
    assert error.target == target
    assert error.directory == failed_directory
    assert error.__cause__ is original_error


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
