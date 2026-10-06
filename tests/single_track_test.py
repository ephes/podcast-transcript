import json
import subprocess
from pathlib import Path

import pytest

from podcast_transcript.audio import Audio, get_cache_key, resample_audio
from podcast_transcript.single_track import (
    backend_cache_metadata,
    combine_dote_chunks,
    transcribe,
    whisper_to_dote,
)


def test_audio_initialization(audio):
    assert audio.url == "https://example.com/test.mp3"
    assert audio.title == "test"
    assert audio.prefix == "test"
    assert audio.podcast_dir == audio.base_dir / get_cache_key(audio.url)
    assert audio.podcast_dir.name.startswith("test-")
    assert audio.episode_chunks_dir == audio.podcast_dir / "chunks"


def test_resample_audio(mocker, audio):
    input_path = audio.episode_path
    output_path = audio.resampled_episode_path

    # Create a dummy input file
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"dummy audio data")

    def fake_ffmpeg(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"resampled")
        return subprocess.CompletedProcess(cmd, 0)

    mock_subprocess_run = mocker.patch("subprocess.run", side_effect=fake_ffmpeg)

    resample_audio(input_path, output_path)

    cmd = mock_subprocess_run.call_args.args[0]
    assert cmd[:-1] == [
        "ffmpeg",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-map",
        "0:a:",
    ]
    # ffmpeg writes to a hidden temp file with the same suffix ...
    assert Path(cmd[-1]).name.startswith(".")
    assert Path(cmd[-1]).suffix == ".mp3"
    assert mock_subprocess_run.call_args.kwargs["check"] is True
    # ... which is renamed into place on success
    assert output_path.read_bytes() == b"resampled"
    assert sorted(p.name for p in output_path.parent.iterdir()) == sorted(
        [input_path.name, output_path.name]
    )


def test_resample_audio_failure_raises_and_leaves_no_output(mocker, audio):
    input_path = audio.episode_path
    output_path = audio.resampled_episode_path
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"not really audio")

    def failing_ffmpeg(cmd, **kwargs):
        # ffmpeg leaves a partial file behind before failing
        Path(cmd[-1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(
            1, cmd, stderr="Invalid data found when processing input\n"
        )

    mocker.patch("subprocess.run", side_effect=failing_ffmpeg)

    with pytest.raises(RuntimeError, match="Invalid data found"):
        resample_audio(input_path, output_path)

    assert not output_path.exists()
    assert [p.name for p in output_path.parent.iterdir()] == [input_path.name]


def test_split_into_chunks_exceeds_limit(mocker, audio):
    # Create a mock stat result with st_size exceeding MAX_SIZE_IN_BYTES
    mock_stat = mocker.MagicMock()
    mock_stat.st_size = 26 * 1024 * 1024  # 26 MB
    mock_stat.st_mode = 0o040755  # Directory mode (drwxr-xr-x)

    # Patch 'Path.stat' in the 'single_track' module to return 'mock_stat'
    mocker.patch("podcast_transcript.audio.Path.stat", return_value=mock_stat)

    # Mock 'subprocess.run' to fake an ffmpeg segment run
    def fake_segment(cmd, **kwargs):
        pattern = Path(cmd[-1])
        for idx in range(2):
            (pattern.parent / (pattern.name % idx)).write_bytes(b"chunk")
        return subprocess.CompletedProcess(cmd, 0)

    mock_subprocess_run = mocker.patch("subprocess.run", side_effect=fake_segment)

    # Mock get_audio_duration to return 1 second
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1)

    # Create the resampled audio file directory
    audio.resampled_episode_path.parent.mkdir(parents=True, exist_ok=True)
    # Write dummy data to the resampled audio path
    audio.resampled_episode_path.write_bytes(b"dummy data")

    # Call the function under test
    chunk_paths = audio.split_into_chunks()

    # Assert that 'subprocess.run' was called once to split the audio
    mock_subprocess_run.assert_called_once()
    assert "ffmpeg" in mock_subprocess_run.call_args[0][0]
    assert [p.name for p in chunk_paths] == ["chunk_000.mp3", "chunk_001.mp3"]
    assert all(p.parent == audio.episode_chunks_dir for p in chunk_paths)
    # The scratch directory is cleaned up
    assert not list(audio.episode_chunks_dir.glob(".split-*"))


def test_split_into_chunks_failure_leaves_no_chunks(mocker, audio):
    mock_stat = mocker.MagicMock()
    mock_stat.st_size = 26 * 1024 * 1024
    mock_stat.st_mode = 0o040755
    mocker.patch("podcast_transcript.audio.Path.stat", return_value=mock_stat)
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1)

    def failing_segment(cmd, **kwargs):
        # ffmpeg wrote the first chunk before failing
        pattern = Path(cmd[-1])
        (pattern.parent / (pattern.name % 0)).write_bytes(b"partial chunk")
        raise subprocess.CalledProcessError(1, cmd, stderr="disk full")

    mocker.patch("subprocess.run", side_effect=failing_segment)
    audio.resampled_episode_path.parent.mkdir(parents=True, exist_ok=True)
    audio.resampled_episode_path.write_bytes(b"dummy data")

    with pytest.raises(RuntimeError, match="disk full"):
        audio.split_into_chunks()

    assert list(audio.episode_chunks_dir.glob("chunk_*.mp3")) == []
    assert not list(audio.episode_chunks_dir.glob(".split-*"))
    assert not audio.chunks_manifest_path.exists()


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
    assert json.loads(audio.chunks_manifest_path.read_text()) == ["chunk_000.mp3"]


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


def _write_dote(path: Path, lines: list[tuple[str, str, str]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "lines": [
                    {
                        "startTime": start,
                        "endTime": end,
                        "speakerDesignation": "",
                        "text": text,
                    }
                    for start, end, text in lines
                ]
            }
        )
    )
    return path


def test_combine_dote_chunks_offsets_by_chunk_duration(tmp_path):
    # Given two 2h chunks whose speech ends long before the chunk ends
    # (trailing music / silence)
    first = _write_dote(
        tmp_path / "chunk_000.dote.json",
        [("00:00:00,000", "01:50:00,000", "first chunk")],
    )
    second = _write_dote(
        tmp_path / "chunk_001.dote.json",
        [("00:00:05,000", "00:00:10,500", "second chunk")],
    )
    output_path = tmp_path / "combined.dote.json"

    # When the chunks are combined with their audio durations
    combine_dote_chunks([first, second], output_path, [7200.0, 1800.0])

    # Then the second chunk is shifted by the first chunk's audio duration,
    # not by the end of its last line
    lines = json.loads(output_path.read_text())["lines"]
    assert [(line["startTime"], line["endTime"]) for line in lines] == [
        ("00:00:00,000", "01:50:00,000"),
        ("02:00:05,000", "02:00:10,500"),
    ]


def test_combine_dote_chunks_requires_a_duration_per_chunk(tmp_path):
    first = _write_dote(tmp_path / "a.dote.json", [])
    second = _write_dote(tmp_path / "b.dote.json", [])

    with pytest.raises(ValueError):
        combine_dote_chunks([first, second], tmp_path / "out.dote.json", [7200.0])

    assert not (tmp_path / "out.dote.json").exists()


def test_transcribe_uses_ffprobe_chunk_durations(mocker, tmp_path):
    chunks_dir = tmp_path / "test" / "chunks"
    chunks_dir.mkdir(parents=True)
    chunks = [chunks_dir / "chunk_000.mp3", chunks_dir / "chunk_001.mp3"]
    mocker.patch("podcast_transcript.single_track.settings.transcript_dir", tmp_path)
    mocker.patch(
        "podcast_transcript.single_track.Audio.prepare_audio_for_transcription",
        return_value=chunks,
    )
    durations = {chunks[0]: 7200.25, chunks[1]: 100.0}
    get_duration = mocker.patch(
        "podcast_transcript.single_track.get_audio_duration",
        side_effect=lambda path: durations[path],
    )

    class FakeBackend:
        def transcribe(self, audio_file: Path, transcript_path: Path) -> None:
            transcript_path.write_text(
                json.dumps({"segments": [{"start": 1.0, "end": 2.0, "text": "hi"}]})
            )

    paths = transcribe("https://example.com/test.mp3", FakeBackend())

    assert get_duration.call_count == 2
    lines = json.loads(paths["DOTe"].read_text())["lines"]
    assert [line["startTime"] for line in lines] == ["00:00:01,000", "02:00:01,250"]


def test_transcribe_fails_when_backend_writes_nothing(mocker, tmp_path):
    chunks_dir = tmp_path / "test" / "chunks"
    chunks_dir.mkdir(parents=True)
    mocker.patch("podcast_transcript.single_track.settings.transcript_dir", tmp_path)
    mocker.patch(
        "podcast_transcript.single_track.Audio.prepare_audio_for_transcription",
        return_value=[chunks_dir / "chunk_000.mp3"],
    )

    class SilentBackend:
        def transcribe(self, audio_file: Path, transcript_path: Path) -> None:
            return None

    with pytest.raises(RuntimeError, match="did not produce"):
        transcribe("https://example.com/test.mp3", SilentBackend())


def test_split_into_chunks_reuses_complete_manifest(mocker, audio):
    audio.episode_chunks_dir.mkdir(parents=True)
    for name in ("chunk_000.mp3", "chunk_001.mp3"):
        (audio.episode_chunks_dir / name).write_bytes(b"chunk")
    audio.chunks_manifest_path.write_text(
        json.dumps(["chunk_000.mp3", "chunk_001.mp3"])
    )
    mock_subprocess_run = mocker.patch("subprocess.run")

    chunk_paths = audio.split_into_chunks()

    mock_subprocess_run.assert_not_called()
    assert [p.name for p in chunk_paths] == ["chunk_000.mp3", "chunk_001.mp3"]


def test_split_into_chunks_redoes_partial_chunk_set_without_manifest(mocker, audio):
    # Given an interrupted earlier split: only the first chunk was moved into
    # place (plus a WAV derived from it) and no manifest was written
    audio.episode_chunks_dir.mkdir(parents=True)
    (audio.episode_chunks_dir / "chunk_000.mp3").write_bytes(b"stale")
    (audio.episode_chunks_dir / "chunk_000.wav").write_bytes(b"stale wav")
    audio.resampled_episode_path.write_bytes(b"dummy data")
    mocker.patch.object(type(audio), "exceeds_size_limit", True)

    def fake_segment(cmd, **kwargs):
        pattern = Path(cmd[-1])
        for idx in range(3):
            (pattern.parent / (pattern.name % idx)).write_bytes(b"fresh")
        return subprocess.CompletedProcess(cmd, 0)

    mock_subprocess_run = mocker.patch("subprocess.run", side_effect=fake_segment)

    # When the chunks are requested again
    chunk_paths = audio.split_into_chunks()

    # Then the episode is split again and the full set is recorded
    mock_subprocess_run.assert_called_once()
    assert [p.name for p in chunk_paths] == [
        "chunk_000.mp3",
        "chunk_001.mp3",
        "chunk_002.mp3",
    ]
    assert all(p.read_bytes() == b"fresh" for p in chunk_paths)
    assert not (audio.episode_chunks_dir / "chunk_000.wav").exists()
    assert json.loads(audio.chunks_manifest_path.read_text()) == [
        "chunk_000.mp3",
        "chunk_001.mp3",
        "chunk_002.mp3",
    ]


def test_split_cleanup_keeps_episode_files_named_like_chunks(mocker, tmp_path):
    # Given an episode whose own file names match "chunk_*.mp3"
    audio = Audio(base_dir=tmp_path, url="https://example.com/chunk_episode.mp3")
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.episode_path.write_bytes(b"original")
    audio.resampled_episode_path.write_bytes(b"resampled")
    (audio.episode_chunks_dir / "chunk_000.mp3").write_bytes(b"stale")
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1)
    mocker.patch("subprocess.run")

    # When the chunk set is rebuilt without a manifest
    chunk_paths = audio.split_into_chunks()

    # Then the downloaded and resampled episode files survive
    assert audio.episode_path.read_bytes() == b"original"
    assert audio.resampled_episode_path.read_bytes() == b"resampled"
    assert [p.name for p in chunk_paths] == ["chunk_000.mp3"]
    assert chunk_paths[0].resolve() == audio.resampled_episode_path


def test_episode_named_like_a_chunk_is_not_overwritten(mocker, tmp_path):
    audio = Audio(base_dir=tmp_path, url="https://example.com/chunk_000.mp3")
    assert audio.episode_path.name == "episode_chunk_000.mp3"
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.episode_path.write_bytes(b"original")
    audio.resampled_episode_path.write_bytes(b"resampled")
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1)
    mocker.patch("subprocess.run")

    chunk_paths = audio.split_into_chunks()

    assert audio.episode_path.read_bytes() == b"original"
    assert [p.name for p in chunk_paths] == ["chunk_000.mp3"]
    assert chunk_paths[0].resolve() == audio.resampled_episode_path


@pytest.mark.parametrize(
    "first,second",
    [
        ("https://example.com/123/audio.mp3", "https://example.com/124/audio.mp3"),
        ("https://example.com/ep1.final.mp3", "https://example.com/ep1.draft.mp3"),
        ("https://example.com/a.mp3?id=1", "https://example.com/a.mp3?id=2"),
    ],
)
def test_urls_with_same_stem_get_separate_cache_dirs(tmp_path, first, second):
    first_audio = Audio(base_dir=tmp_path, url=first)
    second_audio = Audio(base_dir=tmp_path, url=second)

    assert first_audio.prefix == second_audio.prefix
    assert first_audio.podcast_dir != second_audio.podcast_dir
    assert first_audio.podcast_dir.name.startswith(f"{first_audio.prefix}-")


def test_local_files_with_same_name_get_separate_cache_dirs(tmp_path, monkeypatch):
    first = Audio(base_dir=tmp_path, url="/podcasts/a/audio.mp3")
    second = Audio(base_dir=tmp_path, url="/podcasts/b/audio.mp3")
    assert first.podcast_dir != second.podcast_dir

    # A relative path and its absolute form share one cache dir
    monkeypatch.chdir(tmp_path)
    relative = Audio(base_dir=tmp_path, url="audio.mp3")
    absolute = Audio(base_dir=tmp_path, url=str(tmp_path / "audio.mp3"))
    assert relative.podcast_dir == absolute.podcast_dir


def test_cache_key_is_stable():
    url = "https://example.com/123/audio.mp3"
    assert get_cache_key(url) == get_cache_key(url)
    assert get_cache_key("https://example.com/").startswith("episode-")


class RecordingBackend:
    def __init__(self, model_name: str, text: str):
        self.model_name = model_name
        self.language = "en"
        self.prompt = "podcast-transcript"
        self.api_key = "secret-token"
        self.text = text
        self.calls: list[Path] = []

    def transcribe(self, audio_file: Path, transcript_path: Path) -> None:
        self.calls.append(audio_file)
        transcript_path.write_text(
            json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": self.text}]})
        )


def _fake_prepare(self):
    self.episode_chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk = self.episode_chunks_dir / "chunk_000.mp3"
    chunk.write_bytes(b"audio")
    return [chunk]


def test_two_episodes_with_same_basename_get_their_own_transcripts(mocker, tmp_path):
    mocker.patch("podcast_transcript.single_track.settings.transcript_dir", tmp_path)
    mocker.patch.object(Audio, "prepare_audio_for_transcription", _fake_prepare)

    first = transcribe(
        "https://example.com/123/audio.mp3", RecordingBackend("m", "episode 123")
    )
    second = transcribe(
        "https://example.com/124/audio.mp3", RecordingBackend("m", "episode 124")
    )

    assert first["plain text"] != second["plain text"]
    assert first["plain text"].read_text() == "episode 123"
    assert second["plain text"].read_text() == "episode 124"


def test_backend_change_retranscribes(mocker, tmp_path):
    mocker.patch("podcast_transcript.single_track.settings.transcript_dir", tmp_path)
    mocker.patch.object(Audio, "prepare_audio_for_transcription", _fake_prepare)
    url = "https://example.com/test.mp3"

    old_backend = RecordingBackend("old-model", "old transcript")
    paths = transcribe(url, old_backend)
    assert paths["plain text"].read_text() == "old transcript"

    # Same settings: everything is reused
    again = RecordingBackend("old-model", "unused")
    transcribe(url, again)
    assert again.calls == []
    assert paths["plain text"].read_text() == "old transcript"

    # Different model: chunk transcripts and combined outputs are redone
    new_backend = RecordingBackend("new-model", "new transcript")
    paths = transcribe(url, new_backend)
    assert len(new_backend.calls) == 1
    for path in paths.values():
        assert "old transcript" not in path.read_text()
    assert paths["plain text"].read_text() == "new transcript"
    audio = Audio(base_dir=tmp_path, url=url)
    metadata = json.loads(audio.cache_metadata_path.read_text())
    assert metadata["model"] == "new-model"
    assert "secret-token" not in audio.cache_metadata_path.read_text()


def test_backend_class_change_retranscribes(mocker, tmp_path):
    mocker.patch("podcast_transcript.single_track.settings.transcript_dir", tmp_path)
    mocker.patch.object(Audio, "prepare_audio_for_transcription", _fake_prepare)
    url = "https://example.com/test.mp3"
    transcribe(url, RecordingBackend("m", "first"))

    class OtherBackend(RecordingBackend):
        pass

    other = OtherBackend("m", "second")
    paths = transcribe(url, other)

    assert len(other.calls) == 1
    assert paths["plain text"].read_text() == "second"


def test_transcript_cache_without_metadata_is_discarded_but_audio_kept(tmp_path):
    audio = Audio(base_dir=tmp_path, url="https://example.com/chunk_000.mp3")
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.episode_path.write_bytes(b"original")
    audio.resampled_episode_path.write_bytes(b"resampled")
    chunks_dir = audio.episode_chunks_dir
    (chunks_dir / "chunk_000.mp3").write_bytes(b"chunk")
    (chunks_dir / "chunk_000.wav").write_bytes(b"wav")
    (chunks_dir / "chunks.json").write_text('["chunk_000.mp3"]')
    stale = [
        chunks_dir / "chunk_000.json",
        chunks_dir / "chunk_000.dote.json",
        chunks_dir / "chunk_000.whisper-cpp.json",
        audio.output_path(".dote.json"),
        audio.output_path(".podlove.json"),
        audio.output_path(".webvtt"),
        audio.output_path(".txt"),
    ]
    for path in stale:
        path.write_text("stale")

    metadata = backend_cache_metadata(RecordingBackend("m", "x"))
    audio.ensure_transcript_cache_matches(metadata)

    assert not any(path.exists() for path in stale)
    for name in ("chunk_000.mp3", "chunk_000.wav", "chunks.json"):
        assert (chunks_dir / name).exists()
    assert audio.episode_path.read_bytes() == b"original"
    assert audio.resampled_episode_path.read_bytes() == b"resampled"
    assert json.loads(audio.cache_metadata_path.read_text()) == metadata
