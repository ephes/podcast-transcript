"""Bounded admission of native JSON output and remote response bodies (synthetic)."""

import gzip
import json
import os
import subprocess
from pathlib import Path

import httpx
import pytest

from podcast_transcript import backends, safe_io
from remote_boundaries_test import Blocks, remote, transport


def native_output() -> bytes:
    return json.dumps(
        {
            "transcription": [
                {
                    "offsets": {"from": 0, "to": 1000},
                    "timestamps": {"from": "00:00:00,000", "to": "00:00:01,000"},
                    "text": " synthetic",
                }
            ]
        }
    ).encode()


def test_native_output_at_exact_limit_is_admitted(monkeypatch, tmp_path):
    data = native_output()
    monkeypatch.setattr(backends, "LOCAL_TRANSCRIPT_MAX_BYTES", len(data))
    source = tmp_path / "native.json"
    source.write_bytes(data)
    target = tmp_path / "out.json"
    backends.WhisperCpp().convert_output_format(source, target)
    assert json.loads(target.read_text())["segments"][0]["text"] == "synthetic"


def test_oversized_native_output_refused_before_parsing(monkeypatch, tmp_path):
    data = native_output()
    monkeypatch.setattr(backends, "LOCAL_TRANSCRIPT_MAX_BYTES", len(data) - 1)

    def forbidden(*args, **kwargs):
        pytest.fail("oversized native output reached the JSON decoder")

    monkeypatch.setattr(backends.json, "loads", forbidden)
    source = tmp_path / "native.json"
    source.write_bytes(data)
    target = tmp_path / "out.json"
    target.write_bytes(b"previous")
    with pytest.raises(RuntimeError, match="Native transcript output exceeded"):
        backends.WhisperCpp().convert_output_format(source, target)
    assert target.read_bytes() == b"previous"


def test_native_output_growth_after_fstat_is_bounded(monkeypatch, tmp_path):
    source = tmp_path / "native.json"
    source.write_bytes(b"x" * 64)
    real_fstat = os.fstat

    def shrunk(descriptor):
        info = list(real_fstat(descriptor))
        info[6] = 1  # st_size: pretend the file grew after inspection
        return os.stat_result(info)

    monkeypatch.setattr(safe_io.os, "fstat", shrunk)
    with pytest.raises(RuntimeError, match="exceeded byte limit"):
        safe_io.read_regular_bounded(source, limit=8, description="Native output")


@pytest.mark.parametrize(
    "limit,reason",
    [(3, "exceeded byte limit"), (4, "could not close after reading")],
)
def test_bounded_read_reports_close_failure_only_without_primary_error(
    monkeypatch, tmp_path, limit, reason
):
    source = tmp_path / "transcript.json"
    source.write_bytes(b"data")
    real_close = os.close
    close_attempts = []

    def close_then_fail(descriptor):
        close_attempts.append(descriptor)
        real_close(descriptor)
        raise OSError("synthetic close detail")

    with monkeypatch.context() as patch:
        patch.setattr(os, "close", close_then_fail)
        with pytest.raises(RuntimeError, match=f"Transcript {reason}") as raised:
            safe_io.read_regular_bounded(source, limit=limit, description="Transcript")
    assert "synthetic close detail" not in str(raised.value)
    assert len(close_attempts) == 1


def test_deeply_nested_native_output_is_normalized(tmp_path):
    source = tmp_path / "native.json"
    source.write_text("[" * 200_000 + "]" * 200_000)
    target = tmp_path / "out.json"
    target.write_bytes(b"previous")
    with pytest.raises(ValueError, match="Invalid native transcript JSON"):
        backends.WhisperCpp().convert_output_format(source, target)
    assert target.read_bytes() == b"previous"


@pytest.mark.parametrize("kind", ["fifo", "symlink", "directory"])
def test_native_output_must_be_regular_file(tmp_path, kind):
    source = tmp_path / "native.json"
    if kind == "fifo":
        os.mkfifo(source)  # no writer: a blocking open would hang here
    elif kind == "symlink":
        real = tmp_path / "real.json"
        real.write_bytes(native_output())
        source.symlink_to(real)
    else:
        source.mkdir()
    with pytest.raises(RuntimeError, match="must be a regular file"):
        backends.WhisperCpp().convert_output_format(source, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("has_previous", [False, True])
@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        pytest.param(
            b"x" * 4097, RuntimeError, "Native transcript output exceeded", id="size"
        ),
        pytest.param(b"{", ValueError, "Invalid native transcript JSON", id="json"),
        pytest.param(
            b'{"transcription": "invalid"}',
            ValueError,
            "require native segments",
            id="schema",
        ),
    ],
)
def test_refused_native_output_preserves_previous_artifacts(
    monkeypatch, tmp_path, has_previous, payload, error, message
):
    from native_integrity_test import wav_bytes

    monkeypatch.setattr(backends, "LOCAL_TRANSCRIPT_MAX_BYTES", 4096)

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            Path(cmd[-1]).write_bytes(wav_bytes())
        else:
            base = Path(cmd[cmd.index("-of") + 1])
            Path(f"{base}.json").write_bytes(payload)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(backends, "run_checked", fake_run)
    chunk = tmp_path / "chunk_000.mp3"
    chunk.write_bytes(b"mp3")
    transcript = tmp_path / "chunk_000.json"
    transcript.write_bytes(b"previous")
    intermediate = tmp_path / "chunk_000.whisper-cpp.json"
    if has_previous:
        intermediate.write_bytes(native_output())
    with pytest.raises(error, match=message):
        backends.WhisperCpp(model_name="model.bin").transcribe(chunk, transcript)
    assert transcript.read_bytes() == b"previous"
    if has_previous:
        assert intermediate.read_bytes() == native_output()
    else:
        assert not intermediate.exists()
    assert chunk.read_bytes() == b"mp3"
    assert not list(tmp_path.glob(".output-*"))


@pytest.mark.parametrize("name", ["groq", "voxhelm"])
@pytest.mark.parametrize("status", [200, 429, 503])
@pytest.mark.parametrize("encoding", ["gzip", "deflate", "br", "identity, gzip"])
def test_compressed_remote_response_refused_before_decoding(
    monkeypatch, tmp_path, name, status, encoding
):
    stream = Blocks([gzip.compress(b"\0" * 1_000_000)])
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            status, stream=stream, headers={"Content-Encoding": encoding}
        )

    transport(monkeypatch, handler)

    def forbidden(*args, **kwargs):
        pytest.fail("compressed remote response reached the decoder")

    monkeypatch.setattr(httpx.Response, "aiter_bytes", forbidden)
    chunk = tmp_path / "chunk.mp3"
    chunk.write_bytes(b"audio")
    out = tmp_path / "chunk.json"
    out.write_bytes(b"previous")
    with pytest.raises(RuntimeError, match="Content-Encoding"):
        remote(name).transcribe(chunk, out)
    assert len(requests) == 1
    assert requests[0].headers["Accept-Encoding"] == "identity"
    assert requests[0].headers["Authorization"] == "Bearer fake"
    assert stream.closed
    assert out.read_bytes() == b"previous"


@pytest.mark.parametrize("name", ["groq", "voxhelm"])
@pytest.mark.parametrize("extra", [0, 1])
def test_identity_remote_body_exact_admission(monkeypatch, tmp_path, name, extra):
    body = json.dumps({"segments": []}).encode()
    monkeypatch.setattr(backends, "REMOTE_RESPONSE_MAX_BYTES", len(body))
    stream = Blocks([body[:5], body[5:] + b" " * extra])
    transport(
        monkeypatch,
        lambda request: httpx.Response(
            200, stream=stream, headers={"Content-Encoding": "identity"}
        ),
    )
    chunk = tmp_path / "chunk.mp3"
    chunk.write_bytes(b"audio")
    out = tmp_path / "chunk.json"
    if extra:
        with pytest.raises(RuntimeError, match="byte limit"):
            remote(name).transcribe(chunk, out)
        assert not out.exists()
    else:
        remote(name).transcribe(chunk, out)
        assert json.loads(out.read_text()) == {"segments": []}
    assert stream.closed
