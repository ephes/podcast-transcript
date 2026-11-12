"""The MLX backend runs inference in an owned, time-bounded worker (synthetic)."""

import json
import os
import sys
import time

import pytest

from podcast_transcript import audio, backends, single_track

SUCCESS = """
def transcribe(path, **kwargs):
    return {"segments": [{"start": 0, "end": 1, "text": kwargs["initial_prompt"]}]}
"""


def hanging(pid_file):
    return f"""
import os, time

def transcribe(path, **kwargs):
    with open({str(pid_file)!r}, "w") as file:
        file.write(str(os.getpid()))
    time.sleep(120)
"""


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_success_runs_in_worker_and_keeps_prompt_out_of_argv(
    monkeypatch, tmp_path, fake_mlx
):
    fake_mlx(SUCCESS)
    commands = []
    real_run = backends.run_checked

    def spy(cmd, **kwargs):
        commands.append(cmd)
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(backends, "run_checked", spy)
    out = tmp_path / "chunk.json"
    backends.MLX(model_name="synthetic-model", prompt="private-prompt").transcribe(
        tmp_path / "chunk.mp3", out
    )
    assert json.loads(out.read_text())["segments"][0]["text"] == "private-prompt"
    assert commands[0][:2] == [sys.executable, "-c"]
    assert commands[0][3] == str(backends.MLX_WORKER_PATH)
    assert "private-prompt" not in " ".join(commands[0])
    assert "mlx_whisper" not in sys.modules
    assert not list(tmp_path.glob(".mlx-*"))


@pytest.mark.parametrize("failure", ["crash", "oversized", "invalid"])
def test_failed_worker_preserves_previous_transcript(
    monkeypatch, tmp_path, fake_mlx, failure
):
    if failure == "crash":
        fake_mlx("def transcribe(path, **kwargs):\n    raise SystemExit(7)\n")
    elif failure == "oversized":
        monkeypatch.setattr(backends, "LOCAL_TRANSCRIPT_MAX_BYTES", 16)
        fake_mlx(SUCCESS)
    else:
        fake_mlx("def transcribe(path, **kwargs):\n    return [1]\n")
    out = tmp_path / "chunk.json"
    out.write_text("previous")
    with pytest.raises((RuntimeError, ValueError)):
        backends.MLX(model_name=None, prompt="p").transcribe(
            tmp_path / "chunk.mp3", out
        )
    assert out.read_text() == "previous"
    assert not list(tmp_path.glob(".mlx-*"))


def test_missing_dependency_fails_without_starting_worker(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    monkeypatch.setattr(
        backends, "run_checked", lambda *a, **k: pytest.fail("worker started")
    )
    with pytest.raises(RuntimeError, match=r"podcast-transcript\[mlx\]"):
        backends.MLX(model_name=None).transcribe(
            tmp_path / "chunk.mp3", tmp_path / "chunk.json"
        )


def test_hung_worker_is_killed_and_previous_transcript_kept(
    monkeypatch, tmp_path, fake_mlx
):
    monkeypatch.setattr(
        backends.settings, "transcript_local_transcription_timeout", "3"
    )
    pid_file = tmp_path / "worker.pid"
    fake_mlx(hanging(pid_file))
    out = tmp_path / "chunk.json"
    out.write_text("previous")
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="MLX transcription"):
        backends.MLX(model_name=None).transcribe(tmp_path / "chunk.mp3", out)
    assert time.monotonic() - started < 30
    assert pid_file.exists(), "worker did not reach inference before the timeout"
    assert not alive(int(pid_file.read_text()))
    assert out.read_text() == "previous"
    assert not list(tmp_path.glob(".mlx-*"))


@pytest.mark.parametrize("safe_path", [False, True])
def test_worker_ignores_modules_planted_in_current_directory(
    monkeypatch, tmp_path, fake_mlx, safe_path
):
    # Safe-path mode (3.11+) omits the current directory; keep PYTHONPATH intact.
    if safe_path:
        monkeypatch.setenv("PYTHONSAFEPATH", "1")
    else:
        monkeypatch.delenv("PYTHONSAFEPATH", raising=False)
    fake_mlx(SUCCESS)
    planted = tmp_path / "cwd"
    planted.mkdir()
    marker = tmp_path / "planted-ran"
    for name in ("json", "pathlib", "runpy", "mlx_whisper"):
        (planted / f"{name}.py").write_text(
            f"open({str(marker)!r}, 'a').write({name!r})\nraise SystemExit(9)\n"
        )
    monkeypatch.chdir(planted)
    out = tmp_path / "chunk.json"
    backends.MLX(model_name=None, prompt="safe").transcribe(tmp_path / "a.mp3", out)
    assert json.loads(out.read_text())["segments"][0]["text"] == "safe"
    assert not marker.exists()


def test_hung_worker_releases_episode_lock_and_retry_succeeds(
    monkeypatch, tmp_path, fake_mlx
):
    monkeypatch.setattr(single_track.settings, "transcript_dir", tmp_path / "cache")
    monkeypatch.setattr(
        backends.settings, "transcript_local_transcription_timeout", "3"
    )

    def prepare(self):
        self.episode_chunks_dir.mkdir(parents=True, exist_ok=True)
        chunk = self.episode_chunks_dir / "chunk_000.mp3"
        chunk.write_bytes(b"synthetic audio")
        return [chunk]

    monkeypatch.setattr(audio.Audio, "prepare_audio_for_transcription", prepare)
    source = "https://example.test/episode.mp3"
    store = audio.Audio(base_dir=single_track.settings.transcript_dir, url=source)
    backend = backends.MLX(model_name=None, prompt="retry")

    fake_mlx(hanging(tmp_path / "worker.pid"))
    with pytest.raises(RuntimeError, match="MLX transcription"):
        single_track.transcribe(source, backend)
    assert not list(store.episode_chunks_dir.glob("chunk_000.json"))

    monkeypatch.setattr(
        backends.settings, "transcript_local_transcription_timeout", "60"
    )
    fake_mlx(SUCCESS)
    paths = single_track.transcribe(source, backend)
    assert paths["plain text"].read_text() == "retry"
