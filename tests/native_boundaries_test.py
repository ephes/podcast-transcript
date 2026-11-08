"""Synthetic native tool admission tests."""

import json
import subprocess
import sys

import pytest

from podcast_transcript import audio, backends, safe_io, single_track


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_probe_capture_overflow_reaps_child(monkeypatch, tmp_path, stream):
    popen = subprocess.Popen
    children = []

    # The baseline uses subprocess.run; both paths go through Popen.
    def launch(cmd, **kw):
        script = f"import sys,time; sys.{stream}.buffer.write(b'x'*1000000); sys.{stream}.flush(); time.sleep(30)"
        child = popen([sys.executable, "-c", script], **kw)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(audio, "FFPROBE_TIMEOUT_SECONDS", 10)
    store = audio.Audio(base_dir=tmp_path, url="https://example.test/audio.mp3")
    store.episode_chunks_dir.mkdir(parents=True)
    chunks = [store.episode_chunks_dir / f"chunk_{i:03d}.mp3" for i in range(2)]
    paths = chunks + [
        store.cache_metadata_path,
        store.chunks_manifest_path,
        store.output_path(".txt"),
    ]
    for path in paths:
        path.write_bytes(b"previous")
    monkeypatch.setattr(store, "prepare_audio_for_transcription", lambda: chunks)
    backend = backends.WhisperCpp()
    monkeypatch.setattr(
        backend,
        "transcribe",
        lambda *args: pytest.fail("invalid audio reached inference"),
    )
    with pytest.raises(RuntimeError, match="output.*limit") as error:
        single_track._transcribe_locked(store, backend)
    assert len(str(error.value)) < 2500
    assert children and children[0].poll() is not None
    assert all(path.read_bytes() == b"previous" for path in paths)


def native(timestamp):
    return {
        "transcription": [
            {
                "offsets": {"from": 0},
                "timestamps": {"from": timestamp, "to": timestamp},
                "text": "private-synthetic-text",
            }
        ]
    }


@pytest.mark.parametrize(
    "timestamp",
    [
        "00:99:00,000",
        "00:00:99,000",
        "-1:00:00,000",
        "00:00:00,1",
        "00:00:00,1234",
        "00:00:00.000",
        "0:00:00,000",
        None,
    ],
)
def test_native_invalid_timestamp_preserves_output(tmp_path, timestamp):
    source = tmp_path / "native.json"
    source.write_text(json.dumps(native(timestamp)))
    target = tmp_path / "transcript.json"
    target.write_bytes(b"previous")
    with pytest.raises(ValueError, match="Invalid native transcript") as error:
        backends.WhisperCpp().convert_output_format(source, target)
    assert "private-synthetic-text" not in str(error.value)
    assert target.read_bytes() == b"previous"


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"transcription": None},
        {"transcription": [{}]},
        {"transcription": [None]},
        {"transcription": [{"text": "private-synthetic-text"}]},
    ],
)
def test_native_missing_schema_is_normalized(tmp_path, data):
    source = tmp_path / "native.json"
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid native transcript"):
        backends.WhisperCpp().convert_output_format(source, tmp_path / "out.json")


def test_native_multiday_timecode(tmp_path):
    result = backends.WhisperCpp.transform_transcription(native("123:59:59,999"))
    assert result["segments"][0]["start"] == pytest.approx(446399.999)


def test_probe_small_success_failure_and_timeout(monkeypatch, tmp_path):
    popen = subprocess.Popen
    programs = iter(
        [
            'print(\'{"format": {"duration": "2.5"}}\')',
            "import sys; sys.stderr.write('synthetic failure'); sys.exit(7)",
            "import time; time.sleep(30)",
        ]
    )
    children = []

    def launch(cmd, **kw):
        child = popen([sys.executable, "-c", next(programs)], **kw)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(audio, "FFPROBE_TIMEOUT_SECONDS", 10)
    assert audio.get_audio_duration(tmp_path / "audio.mp3") == 2.5
    with pytest.raises(RuntimeError, match="exit code 7.*synthetic failure"):
        audio.get_audio_duration(tmp_path / "audio.mp3")
    monkeypatch.setattr(audio, "FFPROBE_TIMEOUT_SECONDS", 0.5)
    with pytest.raises(RuntimeError, match="timed out"):
        audio.get_audio_duration(tmp_path / "audio.mp3")
    assert all(child.poll() is not None for child in children)


def test_reaped_failure_does_not_signal_process_group(monkeypatch):
    def unexpected_signal(process):
        pytest.fail("reaped process group was signaled")

    monkeypatch.setattr(safe_io, "_kill_owned_tree", unexpected_signal)
    with pytest.raises(RuntimeError, match="exit code 7"):
        safe_io.capture_checked(
            [sys.executable, "-c", "import sys;sys.exit(7)"],
            description="synthetic",
            timeout=10,
        )


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_probe_capture_requires_finite_timeout(timeout):
    with pytest.raises(ValueError, match="positive finite"):
        safe_io.capture_checked(
            ["never-started"], description="synthetic", timeout=timeout
        )


@pytest.mark.parametrize("capture", [True, False])
def test_probe_parent_exit_keeps_identity_until_pipe_cleanup(monkeypatch, capture):
    import threading

    cleaned = threading.Event()

    class Pipe:
        def read(self, size):
            cleaned.wait(2)
            return b""

        def close(self):
            pass

    class ExitedLeader:
        returncode = None
        stdout = Pipe() if capture else None
        stderr = Pipe()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait(self, timeout=None):
            # Model an exited leader. Waiting would release its reserved PID.
            self.returncode = 0
            return 0

    process = ExitedLeader()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kw: process)

    def cleanup(owned):
        assert owned is process
        assert owned.returncode is None, "leader identity released before pipe cleanup"
        cleaned.set()
        owned.returncode = -1

    monkeypatch.setattr(safe_io, "_kill_owned_tree", cleanup)
    with pytest.raises(RuntimeError, match="timed out"):
        run = safe_io.capture_checked if capture else safe_io.run_checked
        run(["synthetic-probe"], description="synthetic probe", timeout=0.05)
    assert cleaned.is_set()
