import json

import httpx
import pytest

from podcast_transcript.backends import Groq, Voxhelm


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
