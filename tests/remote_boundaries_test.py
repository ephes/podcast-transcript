"""Synthetic HTTP boundaries; no external services or platform-specific tools."""

import gzip
import json
import sys

import httpx
import pytest

import podcast_transcript
from podcast_transcript import backends
from podcast_transcript.audio import download, get_cache_source


def remote(name, base="https://voxhelm.example"):
    if name == "groq":
        return backends.Groq(api_key="fake", model_name=None, language="en", prompt="p")
    return backends.Voxhelm(
        api_base=base, api_key="fake", model_name=None, language=None, prompt=None
    )


def transport(monkeypatch, handler):
    client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: client(transport=httpx.MockTransport(handler), **kw),
    )


@pytest.mark.parametrize(
    "base",
    [
        "https://EXAMPLE.com:443",
        "https://example.com/v1/",
        "https://example.com/custom/v1/",
        "http://example.com/custom",
    ],
)
def test_endpoint_and_identity_share_validated_prefix(base):
    backend = remote("voxhelm", base)
    assert (
        backend.transcription_url
        == backend.cache_service_identity + "/audio/transcriptions"
    )
    assert (
        remote("voxhelm").cache_service_identity
        == remote("voxhelm", "https://voxhelm.example/v1/").cache_service_identity
    )


@pytest.mark.parametrize(
    "base",
    [
        "ftp://example.com",
        "/v1",
        "https:///v1",
        "https://example.com/v1?x=1",
        "https://example.com/#f",
        "https://u:p@example.com",
        "https://example.com/?",
        "https://example.com/#",
    ],
)
def test_invalid_api_base_rejected_before_upload(base):
    with pytest.raises(ValueError, match="VOXHELM_API_BASE"):
        remote("voxhelm", base)


class Blocks(httpx.SyncByteStream):
    def __init__(self, blocks):
        self.blocks = blocks
        self.closed = False

    def __iter__(self):
        yield from self.blocks

    def close(self):
        self.closed = True


@pytest.mark.parametrize("name", ["groq", "voxhelm"])
@pytest.mark.parametrize("status", [200, 503, 429])
@pytest.mark.parametrize("encoding,length", [(None, None), (None, "1"), ("gzip", "1")])
def test_remote_decoded_budget_closes_and_preserves(
    monkeypatch, tmp_path, name, status, encoding, length
):
    monkeypatch.setattr(backends, "REMOTE_RESPONSE_MAX_BYTES", 100, raising=False)
    data = b"x" * 101
    stream = Blocks([gzip.compress(data) if encoding else data])
    headers = {}
    if encoding:
        headers["Content-Encoding"] = encoding
    if length:
        headers["Content-Length"] = length
    transport(
        monkeypatch, lambda req: httpx.Response(status, stream=stream, headers=headers)
    )
    chunk = tmp_path / "chunk.mp3"
    chunk.write_bytes(b"audio")
    out = tmp_path / "chunk.json"
    out.write_bytes(b"old transcript")
    with pytest.raises(RuntimeError, match="response.*byte limit"):
        remote(name).transcribe(chunk, out)
    assert out.read_bytes() == b"old transcript"
    assert stream.closed


@pytest.mark.parametrize("name", ["groq", "voxhelm"])
def test_small_stream_and_bounded_error(monkeypatch, tmp_path, name):
    streams = [
        Blocks([json.dumps({"segments": []}).encode()]),
        Blocks([b"bad" + b"x" * 5000]),
    ]
    responses = iter(
        [httpx.Response(200, stream=streams[0]), httpx.Response(503, stream=streams[1])]
    )
    transport(monkeypatch, lambda req: next(responses))
    chunk = tmp_path / "chunk.mp3"
    chunk.write_bytes(b"audio")
    out = tmp_path / "chunk.json"
    remote(name).transcribe(chunk, out)
    assert json.loads(out.read_text()) == {"segments": []}
    with pytest.raises(RuntimeError, match="status 503: bad") as error:
        remote(name).transcribe(chunk, out)
    assert len(str(error.value)) < 2200
    assert all(s.closed for s in streams)


PRIVATE_URL = "https://invented-user:invented-password@example.test/audio.mp3?key=invented-query#invented-fragment"


@pytest.mark.parametrize("outcome", ["success", "http", "redirect", "transport"])
def test_source_diagnostics_keep_original_request_private(
    monkeypatch, tmp_path, capsys, outcome
):
    calls = []

    def handler(req):
        calls.append(req)
        if outcome == "http":
            return httpx.Response(403)
        if outcome == "redirect":
            return httpx.Response(
                302,
                headers={"Location": "https://redirect.test/file?invented-redirect"},
            )
        if outcome == "transport":
            raise httpx.ReadError(f"failed {PRIVATE_URL}", request=req)
        return httpx.Response(200, content=b"audio")

    transport(monkeypatch, handler)
    monkeypatch.setattr(sys, "argv", ["transcribe", PRIVATE_URL])

    def run(source, backend):
        assert source == PRIVATE_URL
        assert get_cache_source(source) == PRIVATE_URL
        download(source, tmp_path / "audio.mp3")
        return {}

    monkeypatch.setattr(podcast_transcript, "transcribe", run)
    with pytest.raises(SystemExit) as result:
        podcast_transcript.transcribe_cli()
    assert result.value.code == (0 if outcome == "success" else 1)
    assert calls[0].url.query == b"key=invented-query"
    assert calls[0].headers["Authorization"].startswith("Basic ")
    output = capsys.readouterr()
    for marker in (
        "invented-user",
        "invented-password",
        "invented-query",
        "invented-fragment",
        "invented-redirect",
    ):
        assert marker not in output.out + output.err


@pytest.mark.parametrize("name", ["groq", "voxhelm"])
def test_error_stream_keeps_declared_charset(monkeypatch, tmp_path, name):
    stream = Blocks([gzip.compress("synthétique".encode("latin-1"))])
    transport(
        monkeypatch,
        lambda req: httpx.Response(
            503,
            stream=stream,
            headers={
                "Content-Type": "text/plain; charset=iso-8859-1",
                "Content-Encoding": "gzip",
            },
        ),
    )
    chunk = tmp_path / "chunk.mp3"
    chunk.write_bytes(b"audio")
    with pytest.raises(RuntimeError, match="synthétique"):
        remote(name).transcribe(chunk, chunk.with_suffix(".json"))
    assert stream.closed
