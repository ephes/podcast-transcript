"""Resampling has an output-byte budget before chunk splitting."""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from podcast_transcript import audio, safe_io, single_track
from single_track_test import RecordingBackend


@pytest.mark.parametrize("size", [31, 32, 33])
def test_resample_rejects_limit_even_on_success(tmp_path, monkeypatch, size):
    monkeypatch.setattr(audio, "RESAMPLED_MAX_BYTES", 32, raising=False)
    target = tmp_path / "prepared.mp3"
    target.write_bytes(b"previous")
    monkeypatch.setattr(
        audio, "run_checked", lambda cmd, **kw: Path(cmd[-1]).write_bytes(b"x" * size)
    )
    if size < 32:
        audio.resample_audio(tmp_path / "source.mp3", target)
        assert target.read_bytes() == b"x" * size
    else:
        with pytest.raises(RuntimeError, match="output byte limit"):
            audio.resample_audio(tmp_path / "source.mp3", target)
        assert target.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".output-*"))


@pytest.mark.parametrize("closed_pipes", [False, True])
def test_growing_resample_is_stopped_before_command_deadline(
    tmp_path, monkeypatch, closed_pipes
):
    monkeypatch.setattr(audio, "RESAMPLED_MAX_BYTES", 32, raising=False)
    monkeypatch.setattr(audio.settings, "transcript_media_timeout", 5)
    target = tmp_path / "prepared.mp3"
    target.write_bytes(b"previous")
    popen = subprocess.Popen
    processes = []
    commands = []

    def launch(cmd, **kwargs):
        commands.append(cmd)
        script = (
            "import os,sys,time; from pathlib import Path; "
            + ("os.close(2); " if closed_pipes else "")
            + "time.sleep(0.1); Path(sys.argv[1]).write_bytes(b'x' * 33); time.sleep(1.5)"
        )
        process = popen([sys.executable, "-c", script, cmd[-1]], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(safe_io.subprocess, "Popen", launch)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="output byte limit"):
        audio.resample_audio(tmp_path / "source.mp3", target)
    assert time.monotonic() - started < 1.5
    assert processes[0].poll() is not None
    assert commands[0][-3:-1] == ["-fs", "32"]
    assert target.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".output-*"))


def test_refused_resample_preserves_previous_prepared_and_published(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(single_track.settings, "transcript_dir", tmp_path / "cache")
    monkeypatch.setattr(audio, "get_audio_duration", lambda _: 1.0)
    resample = audio.resample_audio
    monkeypatch.setattr(audio, "resample_audio", shutil.copyfile)
    source = tmp_path / "synthetic.mp3"
    source.write_bytes(b"old audio")
    single_track.transcribe(str(source), RecordingBackend("model", "previous"))
    store = audio.Audio(base_dir=single_track.settings.transcript_dir, url=str(source))
    before = {
        p: p.read_bytes()
        for p in store.podcast_dir.rglob("*")
        if p.is_file() and p != store.source_metadata_path
    }
    source.write_bytes(b"new audio")
    monkeypatch.setattr(audio, "RESAMPLED_MAX_BYTES", 32, raising=False)
    monkeypatch.setattr(audio, "resample_audio", resample)
    monkeypatch.setattr(
        audio, "run_checked", lambda cmd, **kw: Path(cmd[-1]).write_bytes(b"x" * 33)
    )
    with pytest.raises(RuntimeError, match="output byte limit"):
        single_track.transcribe(str(source), RecordingBackend("model", "replacement"))
    assert all(p.read_bytes() == value for p, value in before.items())
    assert not store.source_metadata_path.exists()
    assert not list(store.podcast_dir.glob(".prepare-*"))
