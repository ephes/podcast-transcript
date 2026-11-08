import subprocess
import sys
import time
import threading
from pathlib import Path
import pytest

from podcast_transcript.audio import resample_audio
from podcast_transcript.safe_io import run_checked


@pytest.mark.parametrize("interrupt", [False, True])
def test_media_abort_kills_owned_child_tree_and_preserves_output(
    tmp_path, monkeypatch, interrupt
):
    # A portable Python helper stands in for ffmpeg, with a child inheriting stderr.
    # If the child survives it writes a marker before exiting.
    helper = tmp_path / "helper.py"
    ready = tmp_path / "ready"
    survived = tmp_path / "survived"
    helper.write_text("""
import subprocess, sys, time
from pathlib import Path
sys.stderr.buffer.write(b'x' * 5000000)
sys.stderr.flush()
child = subprocess.Popen([sys.executable, "-c", "import sys,time; from pathlib import Path; Path(sys.argv[1]).write_text('ready'); time.sleep(2); Path(sys.argv[2]).write_text('survived')", sys.argv[2], sys.argv[3]])
Path(sys.argv[1]).write_bytes(b"partial")
child.wait()
""")
    output = tmp_path / "final.mp3"
    output.write_bytes(b"previous good audio")
    popen = subprocess.Popen
    processes = []

    def launch(cmd, **kwargs):
        if cmd[0] != "ffmpeg":
            return popen(cmd, **kwargs)
        process = popen(
            [sys.executable, str(helper), cmd[-1], str(ready), str(survived)], **kwargs
        )
        processes.append(process)
        if interrupt:
            join = threading.Thread.join

            def interrupted(*args, **kw):
                deadline = time.monotonic() + 10
                while not ready.exists():
                    assert time.monotonic() < deadline, "helper child did not start"
                    time.sleep(0.01)
                monkeypatch.setattr(threading.Thread, "join", join)
                raise KeyboardInterrupt

            monkeypatch.setattr(threading.Thread, "join", interrupted)
        return process

    monkeypatch.setattr("podcast_transcript.safe_io.subprocess.Popen", launch)
    # raising=False lets the timeout regression run against the original code.
    monkeypatch.setattr(
        "podcast_transcript.config.settings.transcript_media_timeout",
        1.0,
        raising=False,
    )
    error = KeyboardInterrupt if interrupt else RuntimeError
    start = time.monotonic()
    with pytest.raises(error, **({} if interrupt else {"match": "timed out"})):
        resample_audio(tmp_path / "input.mp3", output)
    assert ready.exists(), "fault must happen after the child was acquired"
    assert time.monotonic() - start < 10
    time.sleep(2.1)  # a surviving child would have written its marker by now
    assert not survived.exists(), "owned child outlived the aborted command"
    assert processes[0].poll() is not None  # direct child was reaped
    assert output.read_bytes() == b"previous good audio"
    assert sorted(p.name for p in tmp_path.glob("*.mp3")) == ["final.mp3"]


def test_run_checked_success_and_bounded_failure():
    run_checked([sys.executable, "-c", "pass"], description="Synthetic helper")
    with pytest.raises(RuntimeError) as error:
        run_checked(
            [
                sys.executable,
                "-c",
                "import sys; sys.stderr.buffer.write(b'x'*10000+b'\\xff'); sys.exit(7)",
            ],
            description="Synthetic helper",
        )
    assert "exit code 7" in str(error.value)
    assert len(str(error.value)) < 3000
    assert "\ufffd" in str(error.value)


@pytest.mark.parametrize("value", [0, -1, "nan", "inf", "bad"])
@pytest.mark.parametrize(
    "attribute", ["transcript_media_timeout", "transcript_local_transcription_timeout"]
)
def test_invalid_command_budgets_fail_before_spawn(monkeypatch, value, attribute):
    from podcast_transcript.config import settings
    from podcast_transcript.backends import WhisperCpp

    monkeypatch.setattr(settings, attribute, value)

    def unexpected_spawn(*args, **kwargs):
        pytest.fail("invalid command budget reached subprocess")

    monkeypatch.setattr(subprocess, "Popen", unexpected_spawn)
    with pytest.raises(ValueError, match="positive finite seconds"):
        if attribute == "transcript_media_timeout":
            run_checked(["synthetic-helper"], description="Invalid budget")
        else:
            WhisperCpp().run_whisper_cli(Path("synthetic.wav"), Path("synthetic.json"))


def test_whisper_cli_uses_separate_transcription_budget(mocker, tmp_path):
    from podcast_transcript.config import settings
    from podcast_transcript.backends import WhisperCpp

    mocker.patch.object(settings, "transcript_media_timeout", 10)
    mocker.patch.object(settings, "transcript_local_transcription_timeout", 100)
    run = mocker.patch("podcast_transcript.backends.run_checked")
    WhisperCpp(processors=3).run_whisper_cli(
        tmp_path / "input.wav", tmp_path / "output"
    )
    assert run.call_args.kwargs["timeout"] == 100
    assert run.call_args.args[0][0] == "whisper-cli"
