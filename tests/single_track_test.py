import subprocess

from podcast_transcript.audio import download, resample_audio
from podcast_transcript.single_track import whisper_to_dote


def test_audio_initialization(audio):
    assert audio.url == "https://example.com/test.mp3"
    assert audio.title == "test"
    assert audio.prefix == "test"
    assert audio.podcast_dir == audio.base_dir / "test"
    assert audio.episode_chunks_dir == audio.podcast_dir / "chunks"


def test_download(mocker, audio):
    mock_response = mocker.MagicMock()
    mock_response.content = b"audio content"
    mocker.patch("httpx.get", return_value=mock_response)

    target_path = audio.episode_path
    target_path.parent.mkdir(exist_ok=True, parents=True)
    download(audio.url, target_path)

    assert target_path.exists()
    with target_path.open("rb") as f:
        assert f.read() == b"audio content"


def test_resample_audio(mocker, audio):
    input_path = audio.episode_path
    output_path = audio.resampled_episode_path

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


def test_split_into_chunks_exceeds_limit(mocker, audio):
    # Create a mock stat result with st_size exceeding MAX_SIZE_IN_BYTES
    mock_stat = mocker.MagicMock()
    mock_stat.st_size = 26 * 1024 * 1024  # 26 MB
    mock_stat.st_mode = 0o040755  # Directory mode (drwxr-xr-x)

    # Patch 'Path.stat' in the 'single_track' module to return 'mock_stat'
    mocker.patch("podcast_transcript.audio.Path.stat", return_value=mock_stat)

    # Mock 'subprocess.run' to prevent actual subprocess calls
    mock_subprocess_run = mocker.patch("subprocess.run")

    # Mock get_audio_duration to return 1 second
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1)

    # Create the resampled audio file directory
    audio.resampled_episode_path.parent.mkdir(parents=True, exist_ok=True)
    # Write dummy data to the resampled audio path
    audio.resampled_episode_path.write_bytes(b"dummy data")

    # Call the function under test
    audio.split_into_chunks()

    # Assert that 'subprocess.run' was called once to split the audio
    mock_subprocess_run.assert_called_once()
    assert "ffmpeg" in mock_subprocess_run.call_args[0][0]


def test_split_into_chunks_within_limit(mocker, audio):
    # Mock 'subprocess.run' to prevent actual subprocess calls
    mock_subprocess_run = mocker.patch("subprocess.run")

    # Mock get_audio_duration to return 1 second
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1)

    # Create the resampled audio file directory
    audio.resampled_episode_path.parent.mkdir(parents=True, exist_ok=True)
    # Write dummy data to the resampled audio path
    audio.resampled_episode_path.write_bytes(b"dummy data")

    # Ensure the symlink does not already exist
    chunk_symlink = audio.episode_chunks_dir / "chunk_000.mp3"
    if chunk_symlink.exists() or chunk_symlink.is_symlink():
        chunk_symlink.unlink()

    # Call the function under test
    chunk_paths = audio.split_into_chunks()

    # Assert that 'subprocess.run' was not called since file size is within limit
    mock_subprocess_run.assert_not_called()

    # Check if symlink is created
    assert chunk_symlink.exists()
    assert chunk_symlink.is_symlink()
    assert len(chunk_paths) == 1
    assert chunk_symlink.resolve() == audio.resampled_episode_path


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

    output = whisper_to_dote(input_data)
    assert output == expected_output
