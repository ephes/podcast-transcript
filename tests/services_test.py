import pytest

from podcast_transcript.services import Groq


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


def test_groq_audio_file_to_text(mocker, audio_url):
    # Given a mock response from the Groq API
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hello world"}]
    }
    mocker.patch("httpx.Client.post", return_value=mock_response)

    # And a dummy audio chunk
    audio_chunk = audio_url.episode_chunks_dir / "chunk_000.mp3"
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
