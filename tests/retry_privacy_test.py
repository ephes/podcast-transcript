"""Synthetic remote retry and diagnostic contracts."""

import json
import sys
import traceback

import httpx
import pytest

import podcast_transcript
from podcast_transcript import backends
from remote_boundaries_test import remote, transport


@pytest.mark.parametrize("duration", ["999999999h", "9" * 400 + "h", "-1s", "0s"])
def test_untrusted_wait_never_sleeps(monkeypatch, tmp_path, duration):
    transport(
        monkeypatch,
        lambda req: httpx.Response(
            429, json={"error": {"message": f"Please try again in {duration}."}}
        ),
    )
    sleeps = []
    monkeypatch.setattr(
        backends.Groq, "sleep_until", staticmethod(lambda end: sleeps.append(end))
    )
    chunk = tmp_path / "audio.mp3"
    chunk.write_bytes(b"synthetic")
    with pytest.raises(RuntimeError, match="rate limit"):
        remote("groq").transcribe(chunk, chunk.with_suffix(".json"))
    assert sleeps == []


def test_sleep_uses_monotonic_clock(monkeypatch):
    clock = [10.0]
    sleeps = []
    monkeypatch.setattr(backends.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(backends.time, "time", lambda: -1000000)

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
        if len(sleeps) > 2:
            raise AssertionError("wait ignored monotonic deadline")

    monkeypatch.setattr(backends.time, "sleep", sleep)
    backends.Groq.sleep_until(10.25)
    assert sleeps == [0.25]


@pytest.mark.parametrize("name", ["groq", "voxhelm"])
@pytest.mark.parametrize("failure", ["status", "rate", "timeout", "transport"])
def test_remote_errors_and_cli_do_not_echo_private_detail(
    monkeypatch, tmp_path, capsys, name, failure
):
    private = (
        "fake-key private-prompt https://example.test/?signature=private-signature"
    )

    def handler(req):
        if failure == "timeout":
            raise httpx.ReadTimeout(private, request=req)
        if failure == "transport":
            raise httpx.ReadError(private, request=req)
        return httpx.Response(
            429 if failure == "rate" else 503, json={"error": {"message": private}}
        )

    transport(monkeypatch, handler)
    backend = remote(name)
    backend.api_key = "fake-key"
    backend.prompt = "private-prompt"
    chunk = tmp_path / "chunk.mp3"
    chunk.write_bytes(b"synthetic")
    output = chunk.with_suffix(".json")
    output.write_text(json.dumps({"segments": []}))
    with pytest.raises(RuntimeError) as error:
        backend.transcribe(chunk, output)
    detail = "".join(traceback.format_exception(error.value))
    monkeypatch.setattr(sys, "argv", ["transcribe", str(chunk)])
    monkeypatch.setattr(
        podcast_transcript,
        "transcribe",
        lambda source, selected: backend.transcribe(chunk, output),
    )
    with pytest.raises(SystemExit) as result:
        podcast_transcript.transcribe_cli()
    assert result.value.code == 1
    captured = capsys.readouterr()
    for marker in ("fake-key", "private-prompt", "private-signature"):
        assert marker not in detail + captured.out + captured.err
    assert json.loads(output.read_text()) == {"segments": []}


def test_retry_wait_budget_is_cumulative(monkeypatch, tmp_path):
    monkeypatch.setattr(backends.settings, "transcript_http_read_timeout", "7")
    transport(
        monkeypatch,
        lambda req: httpx.Response(
            429, json={"error": {"message": "Please try again in 2s."}}
        ),
    )
    sleeps = []
    monkeypatch.setattr(
        backends.Groq, "sleep_until", staticmethod(lambda end: sleeps.append(end))
    )
    chunk = tmp_path / "audio.mp3"
    chunk.write_bytes(b"synthetic")
    with pytest.raises(RuntimeError, match="retry later"):
        remote("groq").transcribe(chunk, chunk.with_suffix(".json"))
    assert len(sleeps) == 1
