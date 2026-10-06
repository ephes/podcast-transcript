import json
import socket
import subprocess
from pathlib import Path

import httpx
import pytest

from podcast_transcript import backends
from podcast_transcript.backends import Groq, Voxhelm, WhisperCpp


def test_groq_model_name_valid():
    # Given a valid model name
    model_name = "whisper-large-v3"

    # When a Groq instance is created
    groq = Groq(api_key="dummy", model_name=model_name, language="en", prompt="dummy")

    # Then the model name is set correctly
    assert groq.model_name == model_name


def test_groq_model_name_invalid():
    # Given an invalid model name
    model_name = "invalid-model"

    # When a Groq instance is created
    with pytest.raises(ValueError):
        Groq(api_key="dummy", model_name=model_name, language="en", prompt="dummy")


def test_groq_audio_file_to_text(mocker, audio):
    # Given a mock response from the Groq API
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hello world"}]
    }
    mocker.patch("httpx.Client.post", return_value=mock_response)

    # And a dummy audio chunk
    audio_chunk = audio.episode_chunks_dir / "chunk_000.mp3"
    audio_chunk.parent.mkdir(parents=True, exist_ok=True)
    audio_chunk.write_bytes(b"dummy audio data")
    transcript_path = audio_chunk.with_suffix(".json")

    # When the audio chunk is transcribed using the Groq API
    groq = Groq(
        api_key="dummy", model_name="whisper-large-v3", language="en", prompt="dummy"
    )
    groq.transcribe(audio_chunk, transcript_path)

    # Then the Groq API is called with the correct parameters
    mock_response.raise_for_status.assert_called_once()
    # And the transcript is written to the transcript path
    assert transcript_path.exists()


def test_voxhelm_audio_file_to_text(mocker, audio):
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "Hello world",
        "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "Hello world"}],
    }
    post = mocker.patch("httpx.Client.post", return_value=mock_response)

    audio_chunk = audio.episode_chunks_dir / "chunk_000.mp3"
    audio_chunk.parent.mkdir(parents=True, exist_ok=True)
    audio_chunk.write_bytes(b"dummy audio data")
    transcript_path = audio_chunk.with_suffix(".json")

    backend = Voxhelm(
        api_base="https://voxhelm.example",
        api_key="secret-token",
        model_name=None,
        language="de",
        prompt="podcast-transcript",
    )
    backend.transcribe(audio_chunk, transcript_path)

    mock_response.raise_for_status.assert_called_once()
    assert transcript_path.exists()
    assert json.loads(transcript_path.read_text()) == mock_response.json.return_value
    assert post.call_args.args[0] == "https://voxhelm.example/v1/audio/transcriptions"
    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer secret-token"}
    assert post.call_args.kwargs["data"] == {
        "model": "gpt-4o-mini-transcribe",
        "response_format": "verbose_json",
        "language": "de",
        "prompt": "podcast-transcript",
    }


def test_voxhelm_uses_existing_v1_prefix(mocker, audio):
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"text": "Hello"}
    post = mocker.patch("httpx.Client.post", return_value=mock_response)

    audio_chunk = audio.episode_chunks_dir / "chunk_000.mp3"
    audio_chunk.parent.mkdir(parents=True, exist_ok=True)
    audio_chunk.write_bytes(b"dummy audio data")

    backend = Voxhelm(
        api_base="https://voxhelm.example/v1",
        api_key="secret-token",
        model_name="whisper-1",
        language=None,
        prompt=None,
    )
    backend.transcribe(audio_chunk, audio_chunk.with_suffix(".json"))

    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer secret-token"}
    assert post.call_args.args[0] == "https://voxhelm.example/v1/audio/transcriptions"
    assert post.call_args.kwargs["data"] == {
        "model": "whisper-1",
        "response_format": "verbose_json",
    }


def test_voxhelm_http_error_includes_status_and_body(mocker, audio):
    mock_response = mocker.MagicMock()
    mock_response.status_code = 503
    mock_response.text = '{"error":"backend unavailable"}'
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "service unavailable",
        request=mocker.MagicMock(),
        response=mock_response,
    )
    mocker.patch("httpx.Client.post", return_value=mock_response)

    audio_chunk = audio.episode_chunks_dir / "chunk_000.mp3"
    audio_chunk.parent.mkdir(parents=True, exist_ok=True)
    audio_chunk.write_bytes(b"dummy audio data")

    backend = Voxhelm(
        api_base="https://voxhelm.example",
        api_key="secret-token",
        model_name=None,
        language=None,
        prompt=None,
    )

    with pytest.raises(RuntimeError, match="status 503:"):
        backend.transcribe(audio_chunk, audio_chunk.with_suffix(".json"))


def _http_error_response(mocker, status_code: int, text: str):
    mock_response = mocker.MagicMock()
    mock_response.status_code = status_code
    mock_response.text = text
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=mocker.MagicMock(), response=mock_response
    )
    return mock_response


def _dummy_chunk(audio):
    audio_chunk = audio.episode_chunks_dir / "chunk_000.mp3"
    audio_chunk.parent.mkdir(parents=True, exist_ok=True)
    audio_chunk.write_bytes(b"dummy audio data")
    return audio_chunk


def test_groq_server_error_raises(mocker, audio):
    mock_response = _http_error_response(mocker, 500, "internal error")
    mocker.patch("httpx.Client.post", return_value=mock_response)
    audio_chunk = _dummy_chunk(audio)
    transcript_path = audio_chunk.with_suffix(".json")

    groq = Groq(
        api_key="dummy", model_name="whisper-large-v3", language="en", prompt="dummy"
    )
    with pytest.raises(RuntimeError, match="status 500: internal error"):
        groq.transcribe(audio_chunk, transcript_path)

    assert not transcript_path.exists()


@pytest.mark.parametrize(
    "message",
    ["Rate limit reached.", "Rate limit reached. Please try again in soon."],
)
def test_groq_rate_limit_without_usable_wait_raises(mocker, audio, message):
    mock_response = _http_error_response(mocker, 429, "")
    mock_response.json.return_value = {"error": {"message": message}}
    post = mocker.patch("httpx.Client.post", return_value=mock_response)
    audio_chunk = _dummy_chunk(audio)
    transcript_path = audio_chunk.with_suffix(".json")

    groq = Groq(
        api_key="dummy", model_name="whisper-large-v3", language="en", prompt="dummy"
    )
    with pytest.raises(RuntimeError, match="rate limit"):
        groq.transcribe(audio_chunk, transcript_path)

    post.assert_called_once()
    assert not transcript_path.exists()


def test_whisper_cpp_ffmpeg_failure_leaves_no_wav(mocker, tmp_path):
    mocker.patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["ffmpeg"], stderr="bad input"),
    )
    wav_path = tmp_path / "chunk_000.wav"

    with pytest.raises(RuntimeError, match="bad input"):
        WhisperCpp.convert_to_wav(tmp_path / "chunk_000.mp3", wav_path)

    assert list(tmp_path.iterdir()) == []


def test_whisper_cpp_writes_transcript_via_temp_base(mocker, tmp_path):
    cpp_output = {
        "transcription": [
            {
                "offsets": {"from": 0, "to": 1000},
                "timestamps": {"from": "00:00:00,000", "to": "00:00:01,000"},
                "text": " Hello",
            }
        ]
    }
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        assert kwargs["check"] is True
        if cmd[0] == "ffmpeg":
            Path(cmd[-1]).write_bytes(b"wav")
        else:
            base = Path(cmd[cmd.index("-of") + 1])
            Path(f"{base}.json").write_text(json.dumps(cpp_output))
        return subprocess.CompletedProcess(cmd, 0)

    mocker.patch("subprocess.run", side_effect=fake_run)
    audio_chunk = tmp_path / "chunk_000.mp3"
    audio_chunk.write_bytes(b"mp3")
    transcript_path = tmp_path / "chunk_000.json"

    WhisperCpp(model_name="model.bin").transcribe(audio_chunk, transcript_path)

    assert json.loads(transcript_path.read_text())["segments"][0]["text"] == "Hello"
    assert (tmp_path / "chunk_000.whisper-cpp.json").exists()
    assert (tmp_path / "chunk_000.wav").exists()
    # No temporary files are left behind
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_whisper_cpp_cli_failure_leaves_no_transcript(mocker, tmp_path):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            Path(cmd[-1]).write_bytes(b"wav")
            return subprocess.CompletedProcess(cmd, 0)
        raise subprocess.CalledProcessError(
            3, cmd, stderr="error: failed to open model"
        )

    mocker.patch("subprocess.run", side_effect=fake_run)
    audio_chunk = tmp_path / "chunk_000.mp3"
    audio_chunk.write_bytes(b"mp3")
    transcript_path = tmp_path / "chunk_000.json"

    with pytest.raises(RuntimeError, match="failed to open model"):
        WhisperCpp(model_name="model.bin").transcribe(audio_chunk, transcript_path)

    assert not transcript_path.exists()
    assert not (tmp_path / "chunk_000.whisper-cpp.json").exists()


@pytest.mark.parametrize(
    "duration, seconds",
    [("2.5s", 2.5), ("480ms", 0.48), ("1m23.456s", 83.456), ("1h2m", 3720.0)],
)
def test_groq_parse_duration(duration, seconds):
    assert Groq.parse_duration(duration) == pytest.approx(seconds)


def test_groq_rate_limit_with_fractional_wait_retries(mocker, audio):
    rate_limited = _http_error_response(mocker, 429, "")
    rate_limited.json.return_value = {
        "error": {"message": "Rate limit reached. Please try again in 2.5s. Visit ..."}
    }
    ok = mocker.MagicMock()
    ok.status_code = 200
    ok.json.return_value = {"segments": []}
    mocker.patch("httpx.Client.post", side_effect=[rate_limited, ok])
    sleep_until = mocker.patch.object(Groq, "sleep_until")
    mocker.patch("podcast_transcript.backends.time.time", return_value=1000.0)
    audio_chunk = _dummy_chunk(audio)
    transcript_path = audio_chunk.with_suffix(".json")

    groq = Groq(
        api_key="dummy", model_name="whisper-large-v3", language="en", prompt="dummy"
    )
    groq.transcribe(audio_chunk, transcript_path)

    sleep_until.assert_called_once_with(pytest.approx(1000.0 + 2.5 + 2))
    assert json.loads(transcript_path.read_text()) == {"segments": []}


@pytest.mark.parametrize(
    "body, json_result",
    [
        ("Too Many Requests", json.JSONDecodeError("Expecting value", "", 0)),
        ('{"detail": "slow down"}', {"detail": "slow down"}),
        ('["unexpected"]', ["unexpected"]),
    ],
)
def test_groq_rate_limit_with_unexpected_body_raises(mocker, audio, body, json_result):
    mock_response = _http_error_response(mocker, 429, body)
    if isinstance(json_result, Exception):
        mock_response.json.side_effect = json_result
    else:
        mock_response.json.return_value = json_result
    mocker.patch("httpx.Client.post", return_value=mock_response)
    audio_chunk = _dummy_chunk(audio)
    transcript_path = audio_chunk.with_suffix(".json")

    groq = Groq(
        api_key="dummy", model_name="whisper-large-v3", language="en", prompt="dummy"
    )
    with pytest.raises(RuntimeError, match="status 429"):
        groq.transcribe(audio_chunk, transcript_path)

    assert not transcript_path.exists()


def _groq():
    return Groq(
        api_key="dummy", model_name="whisper-large-v3", language="en", prompt="dummy"
    )


def _voxhelm(api_base="https://voxhelm.example"):
    return Voxhelm(
        api_base=api_base,
        api_key="token",
        model_name=None,
        language=None,
        prompt=None,
    )


def _mock_transport_client(monkeypatch, handler):
    """Make every httpx.Client in the backends use a MockTransport."""
    real_client = httpx.Client

    def client_factory(*args, **kwargs):
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(backends.httpx, "Client", client_factory)


def _ok_response(mocker):
    ok = mocker.MagicMock()
    ok.status_code = 200
    ok.json.return_value = {"segments": []}
    return ok


EXPECTED_TIMEOUT = httpx.Timeout(connect=10.0, write=300.0, read=1800.0, pool=10.0)


@pytest.mark.parametrize("make_backend", [_groq, _voxhelm])
def test_remote_backends_pass_bounded_timeout(mocker, audio, make_backend):
    post = mocker.patch("httpx.Client.post", return_value=_ok_response(mocker))
    audio_chunk = _dummy_chunk(audio)

    make_backend().transcribe(audio_chunk, audio_chunk.with_suffix(".json"))

    assert post.call_args.kwargs["timeout"] == EXPECTED_TIMEOUT


@pytest.mark.parametrize("make_backend", [_groq, _voxhelm])
def test_remote_backends_use_configured_read_timeout(
    mocker, monkeypatch, audio, make_backend
):
    monkeypatch.setattr(backends.settings, "transcript_http_read_timeout", "90")
    post = mocker.patch("httpx.Client.post", return_value=_ok_response(mocker))
    audio_chunk = _dummy_chunk(audio)

    make_backend().transcribe(audio_chunk, audio_chunk.with_suffix(".json"))

    assert post.call_args.kwargs["timeout"] == httpx.Timeout(
        connect=10.0, write=300.0, read=90.0, pool=10.0
    )


@pytest.mark.parametrize("make_backend, name", [(_groq, "Groq"), (_voxhelm, "Voxhelm")])
@pytest.mark.parametrize(
    "exc_class", [httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout]
)
def test_remote_backend_timeout_raises_runtime_error(
    monkeypatch, audio, make_backend, name, exc_class
):
    def handler(request):
        raise exc_class("timed out", request=request)

    _mock_transport_client(monkeypatch, handler)
    audio_chunk = _dummy_chunk(audio)
    transcript_path = audio_chunk.with_suffix(".json")

    with pytest.raises(RuntimeError, match=f"^{name} transcription request timed out"):
        make_backend().transcribe(audio_chunk, transcript_path)

    assert not transcript_path.exists()


def test_voxhelm_server_that_never_answers_times_out(monkeypatch, audio):
    # A local socket that accepts the connection but never sends a response.
    monkeypatch.setattr(backends.settings, "transcript_http_read_timeout", "0.5")
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        port = server.getsockname()[1]
        audio_chunk = _dummy_chunk(audio)
        with pytest.raises(RuntimeError, match="Voxhelm .*timed out.*ReadTimeout"):
            _voxhelm(f"http://127.0.0.1:{port}").transcribe(
                audio_chunk, audio_chunk.with_suffix(".json")
            )
    finally:
        server.close()


def test_groq_rate_limit_retries_are_capped(mocker, monkeypatch, audio):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            429,
            json={
                "error": {
                    "message": f"Rate limit reached ({len(requests)}). Please try again in 1s."
                }
            },
        )

    _mock_transport_client(monkeypatch, handler)
    sleep_until = mocker.patch.object(Groq, "sleep_until")
    audio_chunk = _dummy_chunk(audio)
    transcript_path = audio_chunk.with_suffix(".json")

    with pytest.raises(
        RuntimeError, match=r"after 10 attempts: Rate limit reached \(10\)"
    ):
        _groq().transcribe(audio_chunk, transcript_path)

    assert len(requests) == backends.GROQ_MAX_ATTEMPTS == 10
    assert sleep_until.call_count == 9
    assert not transcript_path.exists()


def test_groq_rate_limit_retry_resends_audio(mocker, monkeypatch, audio):
    bodies = []

    def handler(request):
        bodies.append(request.read())
        if len(bodies) == 1:
            return httpx.Response(
                429, json={"error": {"message": "Please try again in 1s."}}
            )
        return httpx.Response(200, json={"segments": []})

    _mock_transport_client(monkeypatch, handler)
    mocker.patch.object(Groq, "sleep_until")
    audio_chunk = _dummy_chunk(audio)
    transcript_path = audio_chunk.with_suffix(".json")

    _groq().transcribe(audio_chunk, transcript_path)

    assert len(bodies) == 2
    assert all(b"dummy audio data" in body for body in bodies)
    assert json.loads(transcript_path.read_text()) == {"segments": []}
