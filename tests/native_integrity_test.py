"""Portable synthetic native diagnostics and WAV reuse fixtures."""

import io
import json
import subprocess
import sys
import tracemalloc
import wave
from pathlib import Path

import pytest

from podcast_transcript import backends, safe_io


def wav_bytes(*, channels=1, rate=16000, width=2, frames=80):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(channels)
        writer.setframerate(rate)
        writer.setsampwidth(width)
        writer.writeframes(b"\0" * frames * channels * width)
    return stream.getvalue()


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"not wav",
        wav_bytes(channels=2),
        wav_bytes(rate=44100),
        wav_bytes(width=1),
        wav_bytes()[:-2],
    ],
)
def test_bad_cached_wav_is_rebuilt_before_inference(monkeypatch, tmp_path, content):
    source = tmp_path / "chunk.mp3"
    source.write_bytes(b"original-audio")
    cached = source.with_suffix(".wav")
    cached.write_bytes(content)
    rebuilt = []

    def convert(source, target):
        rebuilt.append(source)
        target.write_bytes(wav_bytes())

    monkeypatch.setattr(backends.WhisperCpp, "convert_to_wav", staticmethod(convert))
    consumed = []

    def infer(self, source, target):
        consumed.append(source.read_bytes())
        Path(str(target) + ".json").write_text(json.dumps({"transcription": []}))

    monkeypatch.setattr(backends.WhisperCpp, "transcribe_wav", infer)
    backends.WhisperCpp().transcribe(source, source.with_suffix(".json"))
    assert rebuilt == [source]
    assert consumed == [wav_bytes()]
    assert source.read_bytes() == b"original-audio"


def test_valid_cached_wav_is_reused(monkeypatch, tmp_path):
    source = tmp_path / "chunk.mp3"
    source.write_bytes(b"original-audio")
    source.with_suffix(".wav").write_bytes(wav_bytes())
    monkeypatch.setattr(
        backends.WhisperCpp,
        "convert_to_wav",
        staticmethod(lambda *a: pytest.fail("valid WAV rebuilt")),
    )

    def infer(self, source, target):
        Path(str(target) + ".json").write_text(json.dumps({"transcription": []}))

    monkeypatch.setattr(backends.WhisperCpp, "transcribe_wav", infer)
    backends.WhisperCpp().transcribe(source, source.with_suffix(".json"))


@pytest.mark.parametrize("outcome", ["conversion-error", "bad-conversion"])
def test_failed_wav_repair_preserves_final_and_source(monkeypatch, tmp_path, outcome):
    source = tmp_path / "chunk.mp3"
    source.write_bytes(b"original-audio")
    cached = source.with_suffix(".wav")
    cached.write_bytes(b"corrupt")
    target = source.with_suffix(".json")
    target.write_bytes(b"previous-transcript")

    def run(cmd, **kw):
        assert cmd[0] == "ffmpeg"
        Path(cmd[-1]).write_bytes(b"partial-or-invalid")
        if outcome == "conversion-error":
            raise RuntimeError("synthetic conversion failure")

    monkeypatch.setattr(backends, "run_checked", run)
    monkeypatch.setattr(
        backends.WhisperCpp,
        "transcribe_wav",
        lambda *a: pytest.fail("bad WAV reached inference"),
    )
    with pytest.raises(RuntimeError):
        backends.WhisperCpp().transcribe(source, target)
    assert target.read_bytes() == b"previous-transcript"
    assert source.read_bytes() == b"original-audio"
    assert cached.read_bytes() == b"corrupt"
    assert not list(tmp_path.glob(".output-*"))


@pytest.mark.parametrize("hang", [False, True])
def test_native_stderr_uses_bounded_memory_tail(monkeypatch, hang):
    popen = subprocess.Popen
    children = []

    def launch(cmd, **kwargs):
        process = popen(cmd, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", launch)

    def no_disk_spool(*a, **kw):
        pytest.fail("native stderr must not spool to disk")

    monkeypatch.setattr(safe_io.tempfile, "TemporaryFile", no_disk_spool)
    script = (
        "import sys,time; sys.stderr.buffer.write(b'x'*5000000+b'\\nlast-synthetic-diagnostic\\n');sys.stderr.flush();"
        + ("time.sleep(30)" if hang else "sys.exit(7)")
    )
    already_tracing = tracemalloc.is_tracing()
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    try:
        with pytest.raises(
            RuntimeError, match="timed out" if hang else "exit code 7"
        ) as error:
            safe_io.run_checked(
                [sys.executable, "-c", script],
                description="Synthetic native tool",
                timeout=2 if hang else 10,
            )
        peak = tracemalloc.get_traced_memory()[1] - baseline
    finally:
        if not already_tracing:
            tracemalloc.stop()
    assert peak < 2 * 1024 * 1024, (
        "stderr retained more than the bounded tail and read buffers"
    )
    assert "last-synthetic-diagnostic" in str(error.value)
    assert len(str(error.value)) < 2500
    assert "truncated" in str(error.value)
    assert children and children[0].poll() is not None


def test_native_stdout_remains_discarded():
    safe_io.run_checked(
        [sys.executable, "-c", "import sys;sys.stdout.buffer.write(b'x'*1000000)"],
        description="Synthetic native output",
        timeout=10,
    )


def test_line_truncation_is_explicit():
    script = "import sys;sys.stderr.write(''.join(f'diagnostic-{i}\\n' for i in range(50)));sys.exit(7)"
    with pytest.raises(RuntimeError) as error:
        safe_io.run_checked(
            [sys.executable, "-c", script],
            description="Synthetic line diagnostics",
            timeout=10,
        )
    message = str(error.value)
    assert "[stderr truncated]" in message
    assert "diagnostic-0\n" not in message
    assert "diagnostic-49" in message
    assert len(message.splitlines()) <= 22
