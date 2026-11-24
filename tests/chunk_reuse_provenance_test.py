"""Ordinary transcript reuse requires the same prepared chunk bytes."""

import hashlib
import json
import shutil

import pytest

from podcast_transcript import audio, single_track
from single_track_test import RecordingBackend


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(single_track.settings, "transcript_dir", tmp_path / "cache")
    monkeypatch.setattr(audio, "resample_audio", shutil.copyfile)
    monkeypatch.setattr(audio, "get_audio_duration", lambda _: 1.0)
    source = tmp_path / "synthetic.mp3"
    source.write_bytes(b"AAAA")
    store = audio.Audio(base_dir=single_track.settings.transcript_dir, url=str(source))
    single_track.transcribe(store.url, RecordingBackend("model", "original"))
    return store


def published(store):
    return {
        p: p.read_bytes()
        for p in [store.cache_metadata_path, *single_track._transcript_files(store)]
    }


@pytest.mark.parametrize("settings_change", [False, True])
def test_same_size_chunk_edit_refuses_reuse_and_preserves_outputs(
    prepared, settings_change
):
    store = prepared
    before = published(store)
    (store.episode_chunks_dir / "chunk_000.mp3").write_bytes(b"BBBB")
    backend = RecordingBackend("other" if settings_change else "model", "replacement")
    with pytest.raises(RuntimeError, match="Prepared chunk bytes changed"):
        single_track.transcribe(store.url, backend)
    assert not backend.calls
    assert published(store) == before


def test_chunk_digests_recorded_and_unchanged_audio_reuses(prepared):
    store = prepared
    metadata = json.loads(store.cache_metadata_path.read_text())
    assert metadata["chunk_sha256"] == {
        "chunk_000.mp3": hashlib.sha256(b"AAAA").hexdigest()
    }
    backend = RecordingBackend("model", "unused")
    single_track.transcribe(store.url, backend)
    assert not backend.calls


def test_legacy_cache_retranscribes_once_without_discarding_on_failure(prepared):
    store = prepared
    metadata = json.loads(store.cache_metadata_path.read_text())
    metadata.pop("chunk_sha256", None)
    store.cache_metadata_path.write_text(json.dumps(metadata))
    before = published(store)

    class Failing(RecordingBackend):
        def transcribe(self, chunk, target):
            raise RuntimeError("synthetic failure")

    # Keep the same backend identity to isolate the provenance upgrade.
    backend = RecordingBackend("model", "replacement")
    backend.transcribe = Failing.transcribe.__get__(backend)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        single_track.transcribe(store.url, backend)
    assert published(store) == before
    backend = RecordingBackend("model", "replacement")
    single_track.transcribe(store.url, backend)
    assert backend.calls
    assert "chunk_sha256" in json.loads(store.cache_metadata_path.read_text())


def test_chunk_mutation_during_first_inference_never_publishes_raw(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(single_track.settings, "transcript_dir", tmp_path / "cache")
    monkeypatch.setattr(audio, "resample_audio", shutil.copyfile)
    monkeypatch.setattr(audio, "get_audio_duration", lambda _: 1.0)
    source = tmp_path / "synthetic.mp3"
    source.write_bytes(b"AAAA")
    store = audio.Audio(base_dir=single_track.settings.transcript_dir, url=str(source))

    class Mutating(RecordingBackend):
        def transcribe(self, chunk, target):
            super().transcribe(chunk, target)
            chunk.write_bytes(b"BBBB")

    with pytest.raises(RuntimeError, match="Prepared chunk bytes changed"):
        single_track.transcribe(store.url, Mutating("model", "stale"))
    assert not (store.episode_chunks_dir / "chunk_000.json").exists()
    assert not store.output_path(".txt").exists()


def test_out_of_range_raw_cue_is_not_cached_and_retry_exports(tmp_path, monkeypatch):
    monkeypatch.setattr(single_track.settings, "transcript_dir", tmp_path / "cache")
    monkeypatch.setattr(audio, "resample_audio", shutil.copyfile)
    monkeypatch.setattr(audio, "get_audio_duration", lambda _: 1.0)
    source = tmp_path / "synthetic.mp3"
    source.write_bytes(b"AAAA")
    store = audio.Audio(base_dir=single_track.settings.transcript_dir, url=str(source))
    backend = RecordingBackend("model", "corrected")
    valid = backend.transcribe

    def out_of_range(chunk, target):
        valid(chunk, target)
        target.write_text(
            '{"segments":[{"start":0,"end":9007199254740.992,"text":"bad cue"}]}'
        )

    monkeypatch.setattr(backend, "transcribe", out_of_range)
    with pytest.raises(ValueError, match="exceeds 9007199254740991 milliseconds"):
        single_track.transcribe(store.url, backend)
    assert not (store.episode_chunks_dir / "chunk_000.json").exists()
    assert not store.output_path(".dote.json").exists()

    monkeypatch.setattr(backend, "transcribe", valid)
    paths = single_track.transcribe(store.url, backend)
    assert len(backend.calls) == 2
    assert paths["plain text"].read_text() == "corrected"
