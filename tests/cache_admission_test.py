"""Synthetic end-to-end cache admission and reconstruction regressions."""

import json

import pytest

from podcast_transcript import single_track as track
from podcast_transcript.audio import Audio


class Backend:
    def __init__(self):
        self.calls = 0
        self.end = 1.0

    def transcribe(self, source, target):
        self.calls += 1
        target.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "start": 0,
                            "end": self.end,
                            "text": f"generation {self.calls}",
                        }
                    ]
                }
            )
        )


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(track.settings, "transcript_dir", tmp_path)
    monkeypatch.setattr(
        "podcast_transcript.audio.download", lambda _, p: p.write_bytes(b"source")
    )
    monkeypatch.setattr(
        "podcast_transcript.audio.resample_audio", lambda _, p: p.write_bytes(b"audio")
    )
    monkeypatch.setattr("podcast_transcript.audio.get_audio_duration", lambda _: 10.0)
    monkeypatch.setattr(track, "get_audio_duration", lambda _: 10.0)
    url = "https://example.test/synthetic.mp3"
    return url, Audio(base_dir=tmp_path, url=url), Backend()


def test_missing_marker_retires_transcripts(pipeline):
    url, audio, backend = pipeline
    paths = track.transcribe(url, backend)
    assert paths["plain text"].read_text() == "generation 1"
    audio.chunks_manifest_path.unlink()
    # Model the resampled generation that an interrupted rebuild left behind.
    audio.resampled_episode_path.write_bytes(b"changed synthetic audio")
    paths = track.transcribe(url, backend)
    assert backend.calls == 2
    assert paths["plain text"].read_text() == "generation 2"
    assert "generation 2" in paths["DOTe"].read_text()


def test_corrupt_combined_cache_is_never_success(pipeline):
    url, _, backend = pipeline
    paths = track.transcribe(url, backend)
    paths["DOTe"].write_text("{")
    with pytest.raises(ValueError, match="cache JSON"):
        track.transcribe(url, backend)
    assert backend.calls == 1


@pytest.mark.parametrize("name", ["podlove", "WebVTT", "plain text"])
def test_truncated_derived_cache_is_reconstructed(pipeline, name):
    url, _, backend = pipeline
    paths = track.transcribe(url, backend)
    expected = paths[name].read_bytes()
    paths[name].write_bytes(expected[:2])
    repaired = track.transcribe(url, backend)
    assert repaired[name].read_bytes() == expected
    assert backend.calls == 1


@pytest.mark.parametrize("loader", [track.load_whisper_transcript, track.load_dote])
def test_cache_json_read_is_bounded(tmp_path, monkeypatch, loader):
    from podcast_transcript import transcript_validation as validation

    # A small injected cap exercises exact-limit and over-limit real files.
    monkeypatch.setattr(validation, "MAX_CACHE_JSON_BYTES", 64, raising=False)
    body = b'{"segments": [], "lines": []}'
    path = tmp_path / "cache.json"
    path.write_bytes(body.ljust(64, b" "))
    loader(path)
    path.write_bytes(body.ljust(65, b" "))
    with pytest.raises(ValueError, match="cache.json"):
        loader(path)


@pytest.mark.parametrize("loader", [track.load_whisper_transcript, track.load_dote])
def test_cache_json_nesting_errors_are_controlled(tmp_path, loader):
    path = tmp_path / "cache.json"
    path.write_text("[" * 10000 + "]" * 10000)
    with pytest.raises(ValueError, match="cache.json"):
        loader(path)


@pytest.mark.parametrize("loader", [track.load_whisper_transcript, track.load_dote])
def test_growing_cache_read_stops_at_limit(tmp_path, monkeypatch, loader):
    import io
    from pathlib import Path
    from podcast_transcript import transcript_validation as validation

    monkeypatch.setattr(validation, "MAX_CACHE_JSON_BYTES", 64, raising=False)

    class GrowingFile(io.BytesIO):
        def read(self, size=-1):
            assert size == 65
            return b" " * size

    monkeypatch.setattr(Path, "open", lambda *a, **k: GrowingFile())
    with pytest.raises(ValueError, match="cache.json"):
        loader(tmp_path / "cache.json")


@pytest.mark.parametrize("cached", [False, True])
def test_segment_must_fit_its_audio_before_export(pipeline, cached):
    url, audio, backend = pipeline
    if cached:
        track.transcribe(url, backend)
        raw = audio.episode_chunks_dir / "chunk_000.json"
        raw.write_text(
            json.dumps({"segments": [{"start": 0, "end": 30, "text": "private"}]})
        )
    else:
        backend.end = 30
    with pytest.raises(ValueError, match="duration") as error:
        track.transcribe(url, backend)
    assert "private" not in str(error.value)
    if not cached:
        assert not (audio.episode_chunks_dir / "chunk_000.json").exists()
        assert not audio.output_path(".dote.json").exists()


@pytest.mark.parametrize("end", [0, 9, 10, 10.1])
def test_silence_and_timestamp_rounding_tolerance(pipeline, end):
    url, _, backend = pipeline
    backend.end = end
    assert track.transcribe(url, backend)["WebVTT"].exists()


def test_cached_dote_must_fit_its_audio(pipeline):
    url, audio, backend = pipeline
    track.transcribe(url, backend)
    path = audio.episode_chunks_dir / "chunk_000.dote.json"
    data = json.loads(path.read_text())
    data["lines"][0]["endTime"] = "00:00:30,000"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duration"):
        track.transcribe(url, backend)


@pytest.mark.parametrize("loader", [track.load_whisper_transcript, track.load_dote])
@pytest.mark.parametrize("failure", [RecursionError, ValueError, UnicodeDecodeError])
def test_cache_decoder_failures_are_value_free(tmp_path, monkeypatch, loader, failure):
    from podcast_transcript import transcript_validation as validation

    path = tmp_path / "cache.json"
    path.write_text("{}")

    def fail(*args):
        if failure is UnicodeDecodeError:
            raise failure("utf-8", b"private", 0, 1, "private")
        raise failure("private")

    monkeypatch.setattr(validation.json, "loads", fail)
    with pytest.raises(ValueError, match="cache.json") as error:
        loader(path)
    assert "private" not in str(error.value)


def test_multichunk_endpoint_is_bound_to_own_duration(pipeline, monkeypatch):
    url, audio, backend = pipeline
    audio.episode_chunks_dir.mkdir(parents=True)
    chunks = [audio.episode_chunks_dir / f"chunk_{i:03d}.mp3" for i in range(2)]
    for chunk in chunks:
        chunk.write_bytes(b"audio")
    monkeypatch.setattr(Audio, "prepare_audio_for_transcription", lambda _: chunks)
    monkeypatch.setattr(
        track, "get_audio_duration", lambda p: 10.0 if p == chunks[0] else 100.0
    )
    backend.end = 30
    with pytest.raises(ValueError, match="duration"):
        track.transcribe(url, backend)
    assert backend.calls == 1
    assert not chunks[0].with_suffix(".json").exists()
