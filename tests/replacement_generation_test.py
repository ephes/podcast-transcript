"""Replacement failures and promotion interruption retain a recoverable generation."""

import json

import pytest

from podcast_transcript import audio, single_track
from single_track_test import RecordingBackend


@pytest.mark.parametrize("change", ["prompt", "model", "backend"])
@pytest.mark.parametrize("failure", ["inference", "invalid", "interrupt", "export"])
def test_replacement_failure_leaves_previous_generation_current(
    episode, monkeypatch, change, failure
):
    store, previous = episode
    cls = (
        type("OtherBackend", (RecordingBackend,), {})
        if change == "backend"
        else RecordingBackend
    )
    replacement = cls("new" if change == "model" else "old", "new transcript")
    if change == "prompt":
        replacement.prompt = "new synthetic prompt"
    original = replacement.transcribe
    calls = []

    def infer(chunk, target):
        assert json.loads(store.cache_metadata_path.read_text())["model"] == "old"
        calls.append(chunk.name)
        original(chunk, target)
        if len(calls) == 2:
            if failure == "inference":
                raise RuntimeError("synthetic failure")
            if failure == "interrupt":
                raise KeyboardInterrupt
            if failure == "invalid":
                target.write_text('{"segments":false}')

    monkeypatch.setattr(replacement, "transcribe", infer)
    export = single_track._write_webvtt
    if failure == "export":
        monkeypatch.setattr(
            single_track,
            "_write_webvtt",
            lambda *a: (_ for _ in ()).throw(RuntimeError("synthetic export failure")),
        )
    with pytest.raises((RuntimeError, ValueError, KeyboardInterrupt)):
        single_track.transcribe(store.url, replacement)
    assert all(
        (store.podcast_dir / p).read_bytes() == data for p, data in previous.items()
    )
    assert not list(store.podcast_dir.glob(".replacement-*"))
    # Returning to the previous backend uses the unchanged complete generation.
    old = RecordingBackend("old", "unused")
    single_track.transcribe(store.url, old)
    assert old.calls == []
    monkeypatch.setattr(single_track, "_write_webvtt", export)
    monkeypatch.setattr(replacement, "transcribe", original)
    paths = single_track.transcribe(store.url, replacement)
    assert paths["plain text"].read_text() == "new transcript\nnew transcript"
    recovery = store.podcast_dir / ".previous-transcripts"
    assert all((recovery / p).read_bytes() == data for p, data in previous.items())
    recorded = json.loads(store.cache_metadata_path.read_text())
    assert recorded.pop("chunk_sha256")
    assert recorded == single_track.backend_cache_metadata(replacement)
    replacement.calls.clear()
    single_track.transcribe(store.url, replacement)
    assert replacement.calls == []


def test_interrupted_promotion_keeps_recovery_and_retry_does_not_mix(
    episode, monkeypatch
):
    store, previous = episode
    replacement = RecordingBackend("new", "new transcript")
    # Fail after one root export is replaced, before the new cache marker.
    write = single_track.copy_file_atomic
    promoted = []

    def fail(source, path, **kwargs):
        if path.parent == store.podcast_dir and path.suffix != ".json":
            promoted.append(path)
            if len(promoted) == 2:
                raise OSError("synthetic interrupted promotion")
        return write(source, path, **kwargs)

    monkeypatch.setattr(single_track, "copy_file_atomic", fail)
    with pytest.raises(OSError):
        single_track.transcribe(store.url, replacement)
    assert not store.cache_metadata_path.exists()
    recovery = store.podcast_dir / ".previous-transcripts"
    assert all((recovery / p).read_bytes() == data for p, data in previous.items())
    monkeypatch.setattr(single_track, "copy_file_atomic", write)
    paths = single_track.transcribe(store.url, replacement)
    assert paths["plain text"].read_text() == "new transcript\nnew transcript"
    assert all((recovery / p).read_bytes() == data for p, data in previous.items())
    assert json.loads(store.cache_metadata_path.read_text())["model"] == "new"


def test_promotion_and_recovery_copy_exact_bytes(episode):
    store, previous = episode
    # Lone CR, CRLF and non-UTF-8 bytes must survive both copies unchanged.
    odd = b"old\rline\r\nmore\xff\n"
    text_path = store.output_path(".txt")
    text_path.write_bytes(odd)
    previous[text_path.relative_to(store.podcast_dir)] = odd
    replacement = RecordingBackend("new", "new\r\ntranscript")
    paths = single_track.transcribe(store.url, replacement)
    recovery = store.podcast_dir / ".previous-transcripts"
    assert all((recovery / p).read_bytes() == data for p, data in previous.items())
    staged_text = b"new\r\ntranscript\nnew\r\ntranscript"
    assert paths["plain text"].read_bytes() == staged_text


@pytest.mark.parametrize("damage", ["invalid-json", "not-utf8", "bad-shape"])
def test_unreadable_current_generation_is_replaced_in_place(episode, damage):
    store, _ = episode
    dote = store.output_path(".dote.json")
    dote.write_bytes(
        {"invalid-json": b"{", "not-utf8": b"\xff\xfe", "bad-shape": b'{"lines":1}'}[
            damage
        ]
    )
    replacement = RecordingBackend("new", "new transcript")
    paths = single_track.transcribe(store.url, replacement)
    assert paths["plain text"].read_text() == "new transcript\nnew transcript"
    assert json.loads(store.cache_metadata_path.read_text())["model"] == "new"
    assert not (store.podcast_dir / ".previous-transcripts").exists()


def test_abandoned_staging_is_removed_under_the_episode_lock(episode):
    store, _ = episode
    for name in (".replacement-killed", ".backup-killed"):
        (store.podcast_dir / name / "chunks").mkdir(parents=True)
        single_track._mark_stage(store, store.podcast_dir / name)
        (store.podcast_dir / name / "chunks" / "chunk_000.wav").write_bytes(b"x")
    keep = store.podcast_dir / ".previous-transcripts"
    keep.mkdir()
    old = RecordingBackend("old", "unused")
    single_track.transcribe(store.url, old)
    assert old.calls == []
    assert not list(store.podcast_dir.glob(".replacement-*"))
    assert not list(store.podcast_dir.glob(".backup-*"))
    assert keep.is_dir()


def test_generation_copies_stream_large_binary_export(episode, monkeypatch):
    from pathlib import Path
    import os

    store, _ = episode
    content = b"line\r\n\xff" * (1024 * 1024)
    store.output_path(".txt").write_bytes(content)
    read_bytes = Path.read_bytes
    reads = []
    original_read = os.read

    def bounded_read(fd, size):
        assert 0 < size <= 1024 * 1024
        reads.append(size)
        return original_read(fd, size)

    def forbid_eager_read(path):
        raise AssertionError(f"Whole-file binary read: {path.name}")

    monkeypatch.setattr(Path, "read_bytes", forbid_eager_read)
    monkeypatch.setattr(os, "read", bounded_read)
    single_track.transcribe(store.url, RecordingBackend("new", "replacement"))
    assert reads
    assert (
        read_bytes(
            store.podcast_dir / ".previous-transcripts" / store.output_path(".txt").name
        )
        == content
    )


@pytest.mark.parametrize("failure", ["read", "write"])
def test_failed_stream_copy_preserves_destination(tmp_path, monkeypatch, failure):
    import os
    from podcast_transcript import safe_io

    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"synthetic\r\n\xff" * 200000)
    target.write_bytes(b"previous recovery")
    original = os.read if failure == "read" else os.write
    calls = 0

    def fail(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic stream failure")
        return original(*args)

    monkeypatch.setattr(os, failure, fail)
    with pytest.raises(OSError, match="synthetic stream failure"):
        safe_io.copy_file_atomic(source, target)
    assert target.read_bytes() == b"previous recovery"
    assert not list(tmp_path.glob(".output-*"))


def test_killed_promotion_restores_last_success_without_backend(
    episode, monkeypatch, capsys
):
    import signal
    import subprocess
    import sys
    import podcast_transcript

    store, previous = episode
    script = """
import os, signal, sys, json
from pathlib import Path
from podcast_transcript import audio, single_track
from podcast_transcript.backends import TranscriptionBackend
single_track.settings.transcript_dir = Path(sys.argv[1])
single_track.Audio.prepare_audio_for_transcription = lambda self: sorted(self.episode_chunks_dir.glob("chunk_*.mp3"))
audio.get_audio_duration = lambda _: 1.0006
class Replacement(TranscriptionBackend):
    model_name = "new"
    def transcribe(self, chunk, target):
        target.write_text(json.dumps({"segments": [{"start": 0, "end": 1, "text": "new"}]}))
copy = single_track.copy_file_atomic
def kill(source, target, **kwargs):
    copy(source, target, **kwargs)
    if target.parent.name == sys.argv[2] and target.name.endswith(".dote.json"):
        os.kill(os.getpid(), signal.SIGKILL)
single_track.copy_file_atomic = kill
single_track.transcribe(sys.argv[3], Replacement())
"""
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(store.base_dir),
            store.podcast_dir.name,
            store.url,
        ],
        capture_output=True,
        timeout=20,
    )
    assert process.returncode == -signal.SIGKILL, process.stderr
    assert not store.cache_metadata_path.exists()
    assert (
        store.output_path(".dote.json").read_bytes()
        != previous[store.output_path(".dote.json").relative_to(store.podcast_dir)]
    )

    restart_script = """
import sys
from pathlib import Path
import podcast_transcript
from podcast_transcript import single_track
single_track.settings.transcript_dir = Path(sys.argv[1])
def unavailable(*args):
    raise AssertionError("Backend and source preparation are unavailable")
for name in ("whisper_cpp_from_settings_and_args", "mlx_from_settings", "groq_from_settings", "voxhelm_from_settings"):
    setattr(podcast_transcript, name, unavailable)
single_track.Audio.prepare_audio_for_transcription = unavailable
sys.argv = ["transcribe", "--restore-last-success", sys.argv[2]]
podcast_transcript.transcribe_cli()
"""
    restarted = subprocess.run(
        [sys.executable, "-c", restart_script, str(store.base_dir), store.url],
        capture_output=True,
        timeout=20,
    )
    assert restarted.returncode == 0, restarted.stdout + restarted.stderr
    assert all(
        (store.podcast_dir / p).read_bytes() == data for p, data in previous.items()
    )

    # Repeating restore is safe and still does not construct a backend.
    def unavailable(*args):
        raise AssertionError("Restore must not construct a backend or prepare audio")

    monkeypatch.setattr(
        podcast_transcript, "whisper_cpp_from_settings_and_args", unavailable
    )
    monkeypatch.setattr(audio.Audio, "prepare_audio_for_transcription", unavailable)
    monkeypatch.setattr(
        sys, "argv", ["transcribe", "--restore-last-success", store.url]
    )
    with pytest.raises(SystemExit) as result:
        podcast_transcript.transcribe_cli()
    assert result.value.code == 0, capsys.readouterr().out
    assert all(
        (store.podcast_dir / p).read_bytes() == data for p, data in previous.items()
    )
    assert all(
        (store.podcast_dir / ".previous-transcripts" / p).read_bytes() == data
        for p, data in previous.items()
    )


@pytest.mark.parametrize(
    "damage", ["missing", "corrupt", "symlink", "source-changed", "manifest"]
)
def test_restore_refuses_incomplete_or_changed_generation(episode, damage):
    store, _ = episode
    single_track.transcribe(store.url, RecordingBackend("new", "new"))
    recovery = store.podcast_dir / ".previous-transcripts"
    target = recovery / store.output_path(".txt").name
    if damage == "missing":
        target.unlink()
    elif damage == "corrupt":
        target.write_bytes(b"edited")
    elif damage == "symlink":
        target.unlink()
        target.symlink_to(store.output_path(".txt"))
    elif damage == "source-changed":
        store.source_metadata_path.write_text('{"sha256":"different"}')
    else:
        (recovery / "recovery.json").write_text('{"version":999}')
    before = {
        p: p.read_bytes()
        for p in [store.cache_metadata_path, *single_track._transcript_files(store)]
    }
    with pytest.raises((RuntimeError, ValueError, OSError)):
        single_track.restore_last_success(store.url)
    assert all(p.read_bytes() == value for p, value in before.items())
    assert recovery.is_dir()


def test_failed_restore_keeps_backup_and_retires_marker(episode, monkeypatch):
    store, previous = episode
    single_track.transcribe(store.url, RecordingBackend("new", "new"))
    copy = single_track.copy_file_atomic
    calls = []

    def fail(source, target, **kwargs):
        if not any(part.startswith(".replacement-") for part in target.parts):
            calls.append(target)
            if len(calls) == 2:
                raise OSError("synthetic restore failure")
        copy(source, target, **kwargs)

    monkeypatch.setattr(single_track, "copy_file_atomic", fail)
    with pytest.raises(OSError, match="synthetic restore failure"):
        single_track.restore_last_success(store.url)
    assert not store.cache_metadata_path.exists()
    recovery = store.podcast_dir / ".previous-transcripts"
    assert all((recovery / p).read_bytes() == data for p, data in previous.items())
    monkeypatch.setattr(single_track, "copy_file_atomic", copy)
    single_track.restore_last_success(store.url)
    assert all(
        (store.podcast_dir / p).read_bytes() == data for p, data in previous.items()
    )


def test_failed_new_backup_preserves_previous_recovery(episode, monkeypatch):
    store, previous = episode
    single_track.transcribe(store.url, RecordingBackend("new", "new"))
    copy = single_track.copy_file_atomic

    def fail(source, target, **kwargs):
        if any(part.startswith(".backup-") for part in target.parts):
            raise OSError("synthetic backup failure")
        copy(source, target, **kwargs)

    monkeypatch.setattr(single_track, "copy_file_atomic", fail)
    with pytest.raises(OSError, match="synthetic backup failure"):
        single_track.transcribe(store.url, RecordingBackend("third", "third"))
    recovery = store.podcast_dir / ".previous-transcripts"
    assert all((recovery / p).read_bytes() == data for p, data in previous.items())
    assert json.loads(store.cache_metadata_path.read_text())["model"] == "new"


def test_restore_shares_episode_lock_and_publishes_metadata_last(episode, monkeypatch):
    from filelock import FileLock

    store, _ = episode
    single_track.transcribe(store.url, RecordingBackend("new", "new"))
    lock_path = store.base_dir / ".locks" / f"{store.cache_key}.lock"
    with FileLock(lock_path):
        with pytest.raises(RuntimeError, match="busy"):
            single_track.restore_last_success(store.url)
    copy = single_track.copy_file_atomic
    copies = []

    def record(source, target, **kwargs):
        if not any(part.startswith(".replacement-") for part in target.parts):
            assert not store.cache_metadata_path.exists()
            copies.append(target)
        copy(source, target, **kwargs)

    monkeypatch.setattr(single_track, "copy_file_atomic", record)
    single_track.restore_last_success(store.url)
    assert copies[-1] == store.cache_metadata_path


@pytest.mark.parametrize("artifact", ["export", "raw", "metadata"])
def test_restore_pins_verified_bytes_before_publication(episode, monkeypatch, artifact):
    store, previous = episode
    single_track.transcribe(store.url, RecordingBackend("new", "new"))
    recovery = store.podcast_dir / ".previous-transcripts"
    path = {
        "export": recovery / store.output_path(".txt").name,
        "raw": recovery / "chunks/chunk_000.json",
        "metadata": recovery / "cache.json",
    }[artifact]
    original = path.read_bytes()
    digest = single_track._generation_digest
    replaced = []

    def replace_after_admission(generation, **kwargs):
        result = digest(generation, **kwargs)
        if generation.podcast_dir == recovery and not replaced:
            # Atomic pathname replacement after admission, before restore reopens.
            replacement = path.with_name("replacement")
            replacement.write_bytes(original + b" ")
            replacement.replace(path)
            replaced.append(path)
        return result

    monkeypatch.setattr(single_track, "_generation_digest", replace_after_admission)
    before = {
        p: p.read_bytes()
        for p in [store.cache_metadata_path, *single_track._transcript_files(store)]
    }
    with pytest.raises(RuntimeError, match="(?i)recovery.*changed|changed.*recovery"):
        single_track.restore_last_success(store.url)
    assert all(p.read_bytes() == value for p, value in before.items())
    assert path.read_bytes() == original + b" "  # Restore never edits its backup.
    path.write_bytes(original)
    monkeypatch.setattr(single_track, "_generation_digest", digest)
    single_track.restore_last_success(store.url)
    assert all(
        (store.podcast_dir / p).read_bytes() == data for p, data in previous.items()
    )


@pytest.mark.parametrize(
    "change", ["edit-chunk", "replace-chunk", "resampled", "wav", "legacy"]
)
def test_restore_requires_prepared_audio_byte_provenance(episode, monkeypatch, change):
    store, previous = episode
    # Retained WAVs can be the bytes actually submitted to whisper-cpp.
    wav = store.episode_chunks_dir / "chunk_000.wav"
    wav.write_bytes(b"synthetic prepared wav")
    single_track.transcribe(store.url, RecordingBackend("new", "new"))
    recovery = store.podcast_dir / ".previous-transcripts"
    backup = {
        p.relative_to(recovery): p.read_bytes()
        for p in recovery.rglob("*")
        if p.is_file()
    }
    manifest = recovery / "recovery.json"
    if change == "legacy":
        data = json.loads(manifest.read_text())
        data["version"] = 1
        data["audio"] = {
            k: v
            for k, v in data["audio"].items()
            if k in {"source.json", "chunks/chunks.json"}
        }
        manifest.write_text(json.dumps(data))
        backup[manifest.relative_to(recovery)] = manifest.read_bytes()
    else:
        target = (
            store.resampled_episode_path
            if change == "resampled"
            else wav
            if change == "wav"
            else store.episode_chunks_dir / "chunk_000.mp3"
        )
        if change == "replace-chunk":
            replacement = target.with_suffix(".replacement")
            replacement.write_bytes(b"different audio")
            replacement.replace(target)
        else:
            target.write_bytes(b"different audio")
    before = {
        p: p.read_bytes()
        for p in [store.cache_metadata_path, *single_track._transcript_files(store)]
    }
    with pytest.raises(RuntimeError, match="recovery|prepared audio"):
        single_track.restore_last_success(store.url)
    assert all(p.read_bytes() == value for p, value in before.items())
    assert all((recovery / p).read_bytes() == value for p, value in backup.items())


def test_generation_and_restore_receipts_follow_output_inventory(episode, monkeypatch):
    store, previous = episode
    suffixes = tuple(reversed(single_track.TRANSCRIPT_OUTPUT_SUFFIXES))
    monkeypatch.setattr(single_track, "TRANSCRIPT_OUTPUT_SUFFIXES", suffixes)
    monkeypatch.setattr(audio, "TRANSCRIPT_OUTPUT_SUFFIXES", suffixes)
    expected = {
        "DOTe": store.output_path(".dote.json"),
        "podlove": store.output_path(".podlove.json"),
        "WebVTT": store.output_path(".webvtt"),
        "plain text": store.output_path(".txt"),
    }

    generated = single_track.transcribe(store.url, RecordingBackend("new", "new"))
    assert generated == expected
    assert generated["plain text"].read_text() == "new\nnew"

    restored = single_track.restore_last_success(store.url)
    assert restored == expected
    for receipt in (generated, restored):
        assert len(receipt) == len(set(receipt.values())) == len(suffixes)
        assert set(receipt.values()) == {
            store.output_path(suffix) for suffix in suffixes
        }
    assert all(
        path.read_bytes() == previous[path.relative_to(store.podcast_dir)]
        for path in restored.values()
    )


def test_missing_exports_use_one_dote_snapshot(episode, monkeypatch):
    store, _ = episode
    dote = store.output_path(".dote.json")
    podlove = store.output_path(".podlove.json")
    webvtt = store.output_path(".webvtt")
    plaintext = store.output_path(".txt")
    expected = {path: path.read_bytes() for path in (podlove, webvtt, plaintext)}
    for path in expected:
        path.unlink()

    updated = json.loads(dote.read_text())
    updated["lines"][0]["text"] = "updated transcript"
    replacement = dote.with_name("updated.dote.json")
    replacement.write_text(json.dumps(updated))

    write = single_track.write_text_atomic

    def replace_dote_after_podlove(path, content):
        write(path, content)
        if path == podlove:
            replacement.replace(dote)

    monkeypatch.setattr(single_track, "write_text_atomic", replace_dote_after_podlove)
    backend = RecordingBackend("old", "unused")
    single_track.transcribe(store.url, backend)

    assert not backend.calls
    assert json.loads(dote.read_text()) == updated
    assert {path: path.read_bytes() for path in expected} == expected


@pytest.mark.parametrize("conversion_fails", [False, True])
def test_restore_regenerates_derivatives_from_admitted_dote(
    episode, monkeypatch, conversion_fails
):
    store, previous = episode
    single_track.transcribe(store.url, RecordingBackend("new", "new transcript"))
    recovery = store.podcast_dir / ".previous-transcripts"
    snapshot = single_track._generation_at(store, recovery)
    for suffix in (".podlove.json", ".webvtt", ".txt"):
        snapshot.output_path(suffix).write_bytes(store.output_path(suffix).read_bytes())
    manifest_path = recovery / "recovery.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inventory_sha256"] = single_track._generation_digest(snapshot)
    manifest_path.write_text(json.dumps(manifest))
    evidence = {
        path: path.read_bytes() for path in recovery.rglob("*") if path.is_file()
    }
    current = {path: (store.podcast_dir / path).read_bytes() for path in previous}

    if conversion_fails:

        def fail_conversion(cues, path):
            raise RuntimeError("synthetic export conversion failure")

        monkeypatch.setattr(single_track, "_write_plaintext", fail_conversion)
        with pytest.raises(RuntimeError, match="synthetic export conversion failure"):
            single_track.restore_last_success(store.url)
        assert all(
            (store.podcast_dir / path).read_bytes() == data
            for path, data in current.items()
        )
    else:
        single_track.restore_last_success(store.url)
        assert all(
            (store.podcast_dir / path).read_bytes() == data
            for path, data in previous.items()
        )

    assert {
        path: path.read_bytes() for path in recovery.rglob("*") if path.is_file()
    } == evidence
    assert not list(store.podcast_dir.glob(".replacement-*"))
