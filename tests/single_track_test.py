import pytest
import subprocess
from podcast_transcript.single_track import (
    AudioUrl,
    download,
    resample_audio,
    split_into_chunks,
    audio_chunk_to_text,
    groq_to_dote,
)


@pytest.fixture
def audio_url(tmp_path):
    url = "https://example.com/test.mp3"
    return AudioUrl(base_dir=tmp_path, url=url)


def test_audio_url_initialization(audio_url):
    assert audio_url.url == "https://example.com/test.mp3"
    assert audio_url.title == "test"
    assert audio_url.prefix == "test"
    assert audio_url.podcast_dir == audio_url.base_dir / "test"
    assert audio_url.episode_chunks_dir == audio_url.podcast_dir / "chunks"


def test_download(mocker, audio_url):
    mock_response = mocker.MagicMock()
    mock_response.content = b"audio content"
    mocker.patch("httpx.get", return_value=mock_response)

    target_path = audio_url.episode_path
    download(audio_url.url, target_path)

    assert target_path.exists()
    with target_path.open("rb") as f:
        assert f.read() == b"audio content"


def test_resample_audio(mocker, audio_url):
    input_path = audio_url.episode_path
    output_path = audio_url.resampled_episode_path

    # Create a dummy input file
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"dummy audio data")

    mock_subprocess_run = mocker.patch("subprocess.run")

    resample_audio(input_path, output_path)

    mock_subprocess_run.assert_called_once_with(
        [
            "ffmpeg",
            "-i",
            str(input_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-map",
            "0:a:",
            str(output_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_split_into_chunks_exceeds_limit(mocker, audio_url):
    # Create a mock stat result with st_size exceeding MAX_SIZE_IN_BYTES
    mock_stat = mocker.MagicMock()
    mock_stat.st_size = 26 * 1024 * 1024  # 26 MB

    # Patch 'Path.stat' in the 'single_track' module to return 'mock_stat'
    mocker.patch("podcast_transcript.single_track.Path.stat", return_value=mock_stat)

    # Mock 'subprocess.run' to prevent actual subprocess calls
    mock_subprocess_run = mocker.patch("subprocess.run")

    # Create the resampled audio file directory
    audio_url.resampled_episode_path.parent.mkdir(parents=True, exist_ok=True)
    # Write dummy data to the resampled audio path
    audio_url.resampled_episode_path.write_bytes(b"dummy data")

    # Call the function under test
    split_into_chunks(audio_url)

    # Assert that 'subprocess.run' was called once to split the audio
    mock_subprocess_run.assert_called_once()
    assert "ffmpeg" in mock_subprocess_run.call_args[0][0]


def test_split_into_chunks_within_limit(mocker, audio_url):
    # Mock 'subprocess.run' to prevent actual subprocess calls
    mock_subprocess_run = mocker.patch("subprocess.run")

    # Create the resampled audio file directory
    audio_url.resampled_episode_path.parent.mkdir(parents=True, exist_ok=True)
    # Write dummy data to the resampled audio path
    audio_url.resampled_episode_path.write_bytes(b"dummy data")

    # Ensure the symlink does not already exist
    chunk_symlink = audio_url.episode_chunks_dir / "chunk_000.mp3"
    if chunk_symlink.exists() or chunk_symlink.is_symlink():
        chunk_symlink.unlink()

    # Call the function under test
    chunk_paths = split_into_chunks(audio_url)

    # Assert that 'subprocess.run' was not called since file size is within limit
    mock_subprocess_run.assert_not_called()

    # Check if symlink is created
    assert chunk_symlink.exists()
    assert chunk_symlink.is_symlink()
    assert len(chunk_paths) == 1
    assert chunk_symlink.resolve() == audio_url.resampled_episode_path


def test_groq_to_dote():
    input_data = [
        {"start": 0.0, "end": 1.0, "text": "Hello world"},
        {"start": 1.0, "end": 2.0, "text": "This is a test"},
    ]
    expected_output = {
        "lines": [
            {
                "startTime": "00:00:00,000",
                "endTime": "00:00:01,000",
                "speakerDesignation": "",
                "text": "Hello world",
            },
            {
                "startTime": "00:00:01,000",
                "endTime": "00:00:02,000",
                "speakerDesignation": "",
                "text": "This is a test",
            },
        ]
    }

    output = groq_to_dote(input_data)
    assert output == expected_output


def test_audio_chunk_to_text(mocker, audio_url):
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hello world"}]
    }
    mocker.patch("httpx.Client.post", return_value=mock_response)

    audio_chunk = audio_url.episode_chunks_dir / "chunk_000.mp3"
    audio_chunk.parent.mkdir(parents=True, exist_ok=True)
    audio_chunk.write_bytes(b"dummy audio data")
    transcript_path = audio_chunk.with_suffix(".json")

    audio_chunk_to_text(audio_chunk, transcript_path)

    mock_response.raise_for_status.assert_called_once()
    assert transcript_path.exists()
