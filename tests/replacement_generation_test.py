"""Replacement failures and promotion interruption retain a recoverable generation."""

import json

import pytest

from podcast_transcript import audio, single_track
from single_track_test import RecordingBackend


@pytest.fixture
def episode(tmp_path, monkeypatch):
    monkeypatch.setattr(single_track.settings, "transcript_dir", tmp_path)

    def prepare(self):
        self.episode_chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks = [self.episode_chunks_dir / f"chunk_{i:03}.mp3" for i in range(2)]
        for chunk in chunks:
            chunk.write_bytes(b"synthetic audio")
        return chunks

    monkeypatch.setattr(audio.Audio, "prepare_audio_for_transcription", prepare)
    monkeypatch.setattr(single_track, "get_audio_duration", lambda _: 1.0006)
    source = "https://example.test/audio.mp3"
    store = audio.Audio(base_dir=tmp_path, url=source)
    paths = single_track.transcribe(source, RecordingBackend("old", "old transcript"))
    files = [
        store.cache_metadata_path,
        *paths.values(),
        *store.episode_chunks_dir.glob("*.json"),
    ]
    previous = {p.relative_to(store.podcast_dir): p.read_bytes() for p in files}
    return store, previous


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
    export = single_track.convert_to_webvtt
    if failure == "export":
        monkeypatch.setattr(
            single_track,
            "convert_to_webvtt",
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
    monkeypatch.setattr(single_track, "convert_to_webvtt", export)
    monkeypatch.setattr(replacement, "transcribe", original)
    paths = single_track.transcribe(store.url, replacement)
    assert paths["plain text"].read_text() == "new transcript\nnew transcript"
    recovery = store.podcast_dir / ".previous-transcripts"
    assert all((recovery / p).read_bytes() == data for p, data in previous.items())
    assert json.loads(
        store.cache_metadata_path.read_text()
    ) == single_track.backend_cache_metadata(replacement)
    replacement.calls.clear()
    single_track.transcribe(store.url, replacement)
    assert replacement.calls == []


def test_interrupted_promotion_keeps_recovery_and_retry_does_not_mix(
    episode, monkeypatch
):
    store, previous = episode
    replacement = RecordingBackend("new", "new transcript")
    # Fail after one root export is replaced, before the new cache marker.
    write = single_track.write_text_atomic
    promoted = []

    def fail(path, text):
        if path.parent == store.podcast_dir and path.suffix != ".json":
            promoted.append(path)
            if len(promoted) == 2:
                raise OSError("synthetic interrupted promotion")
        return write(path, text)

    monkeypatch.setattr(single_track, "write_text_atomic", fail)
    with pytest.raises(OSError):
        single_track.transcribe(store.url, replacement)
    assert not store.cache_metadata_path.exists()
    recovery = store.podcast_dir / ".previous-transcripts"
    assert all((recovery / p).read_bytes() == data for p, data in previous.items())
    monkeypatch.setattr(single_track, "write_text_atomic", write)
    paths = single_track.transcribe(store.url, replacement)
    assert paths["plain text"].read_text() == "new transcript\nnew transcript"
    assert all((recovery / p).read_bytes() == data for p, data in previous.items())
    assert json.loads(store.cache_metadata_path.read_text())["model"] == "new"
