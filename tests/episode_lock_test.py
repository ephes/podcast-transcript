import json
import subprocess
import sys
import time

import pytest

from podcast_transcript.audio import Audio
from podcast_transcript.single_track import transcribe
from single_track_test import RecordingBackend, _fake_prepare


@pytest.mark.parametrize("fail", [False, True])
def test_episode_generation_has_one_process_writer(tmp_path, monkeypatch, fail):
    """A second process cannot prepare/invalidate while a backend is working."""
    monkeypatch.setattr(
        "podcast_transcript.single_track.get_audio_duration", lambda _: 10.0
    )
    cache = tmp_path / "cache"
    entered = tmp_path / "entered"
    worker = tmp_path / "worker.py"
    worker.write_text("""
import json
import sys
from pathlib import Path
from podcast_transcript.audio import Audio
from podcast_transcript import single_track
from podcast_transcript.single_track import transcribe, settings
single_track.get_audio_duration = lambda _: 10.0
settings.transcript_dir = Path(sys.argv[1])
def prepare(self):
    self.episode_chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk = self.episode_chunks_dir / "chunk_000.mp3"
    chunk.write_bytes(b"synthetic")
    return [chunk]
Audio.prepare_audio_for_transcription = prepare
class SyntheticBackend:
    model_name = "first"
    def transcribe(self, audio_file, transcript_path):
        Path(sys.argv[2]).write_text("entered")
        sys.stdin.readline()
        if sys.argv[3] == "fail":
            raise RuntimeError("synthetic failure")
        transcript_path.write_text(json.dumps({"segments": [{"start": 0, "end": 1, "text": "first"}]}))
try:
    transcribe("https://example.test/episode.mp3", SyntheticBackend())
except RuntimeError:
    sys.exit(3)
""")
    monkeypatch.setattr(
        "podcast_transcript.single_track.settings.transcript_dir", cache
    )
    monkeypatch.setattr(Audio, "prepare_audio_for_transcription", _fake_prepare)
    child = subprocess.Popen(
        [
            sys.executable,
            str(worker),
            str(cache),
            str(entered),
            "fail" if fail else "ok",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while not entered.exists():
            assert child.poll() is None, child.communicate(timeout=5)[1]
            assert time.monotonic() < deadline, "worker never reached backend"
            time.sleep(0.01)
        # Unrelated episode can finish while the first generation is held.
        other = transcribe(
            "https://example.test/other.mp3", RecordingBackend("other", "other")
        )
        assert other["plain text"].read_text() == "other"
        second = RecordingBackend("second", "second")
        with pytest.raises(RuntimeError, match="busy"):
            transcribe("https://example.test/episode.mp3", second)
        assert second.calls == []
        audio = Audio(base_dir=cache, url="https://example.test/episode.mp3")
        assert json.loads(audio.cache_metadata_path.read_text())["model"] == "first"
        assert not audio.output_path(".txt").exists()
        _, errors = child.communicate("release\n", timeout=15)
        assert child.returncode == (3 if fail else 0), errors
        if not fail:
            assert audio.output_path(".txt").read_text() == "first"
        paths = transcribe(audio.url, second)
        assert paths["plain text"].read_text() == "second"
        assert len(second.calls) == 1
        assert json.loads(audio.cache_metadata_path.read_text())["model"] == "second"
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)


@pytest.mark.parametrize("stage", ["preparation", "export"])
def test_episode_lock_released_on_pipeline_failure(mocker, tmp_path, stage):
    mocker.patch(
        "podcast_transcript.single_track.get_audio_duration", return_value=10.0
    )
    mocker.patch("podcast_transcript.single_track.settings.transcript_dir", tmp_path)
    prepare = mocker.patch.object(
        Audio, "prepare_audio_for_transcription", _fake_prepare
    )
    if stage == "preparation":
        prepare = mocker.patch.object(
            Audio,
            "prepare_audio_for_transcription",
            side_effect=RuntimeError("synthetic failure"),
        )
    else:
        prepare = mocker.patch(
            "podcast_transcript.single_track.convert_to_webvtt",
            side_effect=RuntimeError("synthetic failure"),
        )
    with pytest.raises(RuntimeError, match="synthetic failure"):
        transcribe("https://example.test/episode.mp3", RecordingBackend("m", "text"))
    mocker.stop(prepare)
    paths = transcribe(
        "https://example.test/episode.mp3", RecordingBackend("m", "text")
    )
    assert paths["plain text"].read_text() == "text"
