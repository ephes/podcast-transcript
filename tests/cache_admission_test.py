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
    with pytest.raises(ValueError, match="DOTe JSON"):
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
