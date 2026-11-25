import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

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

    mock_media_command = mocker.patch(
        "podcast_transcript.audio.run_checked", side_effect=fake_ffmpeg
    )

    resample_audio(input_path, output_path)

    cmd = mock_media_command.call_args.args[0]
    assert cmd[:-1] == [
        "ffmpeg",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-map",
        "0:a:0",
        "-fs",
        str(2 * 1024**3),
    ]
    # ffmpeg writes to a private staging directory with the same media suffix.
    stage = Path(cmd[-1])
    assert stage.parent.name.startswith(".output-")
    assert stage.parent.parent == output_path.parent
    assert stage.suffix == ".mp3"
    assert "Resampling" in mock_media_command.call_args.kwargs["description"]
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
        raise RuntimeError("Invalid data found when processing input")

    mocker.patch("podcast_transcript.audio.run_checked", side_effect=failing_ffmpeg)

    with pytest.raises(RuntimeError, match="Invalid data found"):
        resample_audio(input_path, output_path)

    assert not output_path.exists()
    assert [p.name for p in output_path.parent.iterdir()] == [input_path.name]


def test_normalized_cache_uses_first_audio_stream(tmp_path):
    source = tmp_path / "two-tracks.mka"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=1",
            "-map",
            "0:a",
            "-map",
            "1:a",
            "-disposition:a:0",
            "0",
            "-disposition:a:1",
            "default",
            "-c:a",
            "pcm_s16le",
            str(source),
        ],
        check=True,
        timeout=10,
    )
    audio = Audio(base_dir=tmp_path / "cache", url=str(source))

    audio.prepare_audio_for_transcription()

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,sample_rate,channels,duration",
            "-of",
            "json",
            str(audio.resampled_episode_path),
        ],
        check=True,
        capture_output=True,
        timeout=10,
    )
    (stream,) = json.loads(probe.stdout)["streams"]
    assert stream["codec_type"] == "audio"
    assert stream["sample_rate"] == "16000"
    assert stream["channels"] == 1
    assert float(stream["duration"]) == pytest.approx(0.25, abs=0.12)


def test_split_into_chunks_exceeds_limit(mocker, audio):
    # Lower the size limit below the dummy file's size instead of mocking
    # 'Path.stat': on Python 3.10/3.11 'Path.exists()' is built on 'stat()', so a
    # global stat mock makes every path look like an existing directory.
    mocker.patch("podcast_transcript.audio.MAX_SIZE_IN_BYTES", 8)

    # Mock 'run_checked' to fake an ffmpeg segment run
    def fake_segment(cmd, **kwargs):
        pattern = Path(cmd[-1])
        for idx in range(2):
            (pattern.parent / (pattern.name % idx)).write_bytes(b"chunk")
        return subprocess.CompletedProcess(cmd, 0)

    mock_media_command = mocker.patch(
        "podcast_transcript.audio.run_checked", side_effect=fake_segment
    )

    # Mock get_audio_duration to return 1 second
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1)

    # Create the resampled audio file directory
    audio.resampled_episode_path.parent.mkdir(parents=True, exist_ok=True)
    # Write dummy data to the resampled audio path
    audio.resampled_episode_path.write_bytes(b"dummy data")

    # Call the function under test
    chunk_paths = audio.split_into_chunks()

    # Assert that 'run_checked' was called once to split the audio
    mock_media_command.assert_called_once()
    assert "ffmpeg" in mock_media_command.call_args[0][0]
    assert [p.name for p in chunk_paths] == ["chunk_000.mp3", "chunk_001.mp3"]
    assert all(p.parent == audio.episode_chunks_dir for p in chunk_paths)
    # The scratch directory is cleaned up
    assert not list(audio.episode_chunks_dir.glob(".split-*"))


def test_split_into_chunks_failure_leaves_no_chunks(mocker, audio):
    # See test_split_into_chunks_exceeds_limit for why 'Path.stat' is not mocked.
    mocker.patch("podcast_transcript.audio.MAX_SIZE_IN_BYTES", 1)
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1)

    def failing_segment(cmd, **kwargs):
        # ffmpeg wrote the first chunk before failing
        pattern = Path(cmd[-1])
        (pattern.parent / (pattern.name % 0)).write_bytes(b"partial chunk")
        raise RuntimeError("disk full")

    mocker.patch("podcast_transcript.audio.run_checked", side_effect=failing_segment)
    audio.resampled_episode_path.parent.mkdir(parents=True, exist_ok=True)
    audio.resampled_episode_path.write_bytes(b"dummy data")

    with pytest.raises(RuntimeError, match="disk full"):
        audio.split_into_chunks()

    assert list(audio.episode_chunks_dir.glob("chunk_*.mp3")) == []
    assert not list(audio.episode_chunks_dir.glob(".split-*"))
    assert not audio.chunks_manifest_path.exists()


def test_split_into_chunks_within_limit(mocker, audio):
    # Mock 'run_checked' to prevent actual subprocess calls
    mock_media_command = mocker.patch("podcast_transcript.audio.run_checked")

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

    # Assert that 'run_checked' was not called since file size is within limit
    mock_media_command.assert_not_called()

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
    for chunk in chunks:
        chunk.write_bytes(b"synthetic audio")
    mocker.patch("podcast_transcript.single_track.settings.transcript_dir", tmp_path)
    mocker.patch(
        "podcast_transcript.single_track.Audio.prepare_audio_for_transcription",
        return_value=chunks,
    )
    durations = {chunks[0]: 7199.25, chunks[1]: 100.0}
    get_duration = mocker.patch(
        "podcast_transcript.audio.get_audio_duration",
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
    assert [line["startTime"] for line in lines] == ["00:00:01,000", "02:00:00,250"]


def test_transcribe_fails_when_backend_writes_nothing(mocker, tmp_path):
    chunks_dir = tmp_path / "test" / "chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunk_000.mp3").write_bytes(b"synthetic audio")
    mocker.patch("podcast_transcript.single_track.settings.transcript_dir", tmp_path)
    mocker.patch(
        "podcast_transcript.single_track.Audio.prepare_audio_for_transcription",
        return_value=[chunks_dir / "chunk_000.mp3"],
    )
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1.0)

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
    mock_media_command = mocker.patch("podcast_transcript.audio.run_checked")

    chunk_paths = audio.split_into_chunks()

    mock_media_command.assert_not_called()
    assert [p.name for p in chunk_paths] == ["chunk_000.mp3", "chunk_001.mp3"]


def test_split_into_chunks_redoes_partial_chunk_set_without_manifest(mocker, audio):
    # Given an interrupted earlier split: only the first chunk was moved into
    # place (plus a WAV derived from it) and no manifest was written
    audio.episode_chunks_dir.mkdir(parents=True)
    (audio.episode_chunks_dir / "chunk_000.mp3").write_bytes(b"stale")
    (audio.episode_chunks_dir / "chunk_000.wav").write_bytes(b"stale wav")
    audio.resampled_episode_path.write_bytes(b"dummy data")
    mocker.patch.object(type(audio), "exceeds_size_limit", True)
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1.0)

    def fake_segment(cmd, **kwargs):
        pattern = Path(cmd[-1])
        for idx in range(3):
            (pattern.parent / (pattern.name % idx)).write_bytes(b"fresh")
        return subprocess.CompletedProcess(cmd, 0)

    mock_media_command = mocker.patch(
        "podcast_transcript.audio.run_checked", side_effect=fake_segment
    )

    # When the chunks are requested again
    chunk_paths = audio.split_into_chunks()

    # Then the episode is split again and the full set is recorded
    mock_media_command.assert_called_once()
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
    mocker.patch("podcast_transcript.audio.run_checked")

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
    mocker.patch("podcast_transcript.audio.run_checked")

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
    self.resampled_episode_path.write_bytes(b"resampled")
    self.chunks_manifest_path.write_text('["chunk_000.mp3"]')
    return [chunk]


def test_two_episodes_with_same_basename_get_their_own_transcripts(mocker, tmp_path):
    mocker.patch("podcast_transcript.single_track.settings.transcript_dir", tmp_path)
    mocker.patch.object(Audio, "prepare_audio_for_transcription", _fake_prepare)
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1.0)

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
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1.0)
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
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1.0)
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


def test_local_source_changes_invalidate_pipeline(mocker, tmp_path):
    import shutil

    mocker.patch(
        "podcast_transcript.single_track.settings.transcript_dir", tmp_path / "cache"
    )
    mocker.patch("podcast_transcript.audio.resample_audio", shutil.copyfile)
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1.0)
    source = tmp_path / "episode.mp3"
    source.write_bytes(b"AAAA")

    class ContentBackend(RecordingBackend):
        def transcribe(self, audio_file, transcript_path):
            self.text = audio_file.read_bytes().decode()
            super().transcribe(audio_file, transcript_path)

    backend = ContentBackend("first", "")
    paths = transcribe(str(source), backend)
    assert paths["plain text"].read_text() == "AAAA"
    transcribe(str(source), backend)
    assert len(backend.calls) == 1
    source.write_bytes(b"BBBB")  # same size, same path
    paths = transcribe(str(source), backend)
    assert paths["plain text"].read_text() == "BBBB"
    assert len(backend.calls) == 2
    backend.model_name = "second"
    transcribe(str(source), backend)
    assert len(backend.calls) == 3


def test_local_source_metadata_written_only_after_preparation(mocker, tmp_path):
    import shutil

    source = tmp_path / "episode.mp3"
    source.write_bytes(b"AAAA")
    audio = Audio(base_dir=tmp_path / "cache", url=str(source))
    mocker.patch(
        "podcast_transcript.audio.resample_audio", side_effect=RuntimeError("failed")
    )
    with pytest.raises(RuntimeError, match="failed"):
        audio.prepare_audio_for_transcription()
    assert not (audio.podcast_dir / "source.json").exists()
    source.write_bytes(b"BBBB")
    mocker.patch("podcast_transcript.audio.resample_audio", shutil.copyfile)
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1.0)
    assert audio.prepare_audio_for_transcription()[0].read_bytes() == b"BBBB"


@pytest.mark.parametrize("failure", ["copy", "resample", "probe"])
def test_changed_source_preparation_preserves_previous_generation(
    mocker, tmp_path, failure
):
    import shutil
    from podcast_transcript import audio as audio_module

    cache = tmp_path / "cache"
    mocker.patch("podcast_transcript.single_track.settings.transcript_dir", cache)
    mocker.patch.object(audio_module, "resample_audio", shutil.copyfile)
    mocker.patch.object(audio_module, "get_audio_duration", return_value=1.0)
    source = tmp_path / "source.mp3"
    source.write_bytes(b"original audio")
    backend = RecordingBackend("model", "original transcript")
    transcribe(str(source), backend)
    audio = Audio(base_dir=cache, url=str(source))
    previous = {
        path.relative_to(audio.podcast_dir): path.read_bytes()
        for path in audio.podcast_dir.rglob("*")
        if path.is_file() and path != audio.source_metadata_path
    }
    source.write_bytes(b"changed corrupt audio")
    target, method = {
        "copy": (Audio, "make_sure_audio_file_exists"),
        "resample": (audio_module, "resample_audio"),
        "probe": (audio_module, "get_audio_duration"),
    }[failure]
    broken = mocker.patch.object(
        target, method, side_effect=RuntimeError("synthetic preparation failure")
    )
    # Both attempts fail rather than returning a transcript for the previous bytes.
    for _ in range(2):
        with pytest.raises(RuntimeError, match="synthetic preparation failure"):
            transcribe(str(source), backend)
        assert len(backend.calls) == 1
        assert not audio.source_metadata_path.exists()
        for relative, content in previous.items():
            assert (audio.podcast_dir / relative).read_bytes() == content
        assert not list(audio.podcast_dir.glob(".prepare-*"))
    mocker.stop(broken)
    backend.text = "replacement transcript"
    paths = transcribe(str(source), backend)
    assert paths["plain text"].read_text() == "replacement transcript"
    assert len(backend.calls) == 2
    assert audio.episode_path.read_bytes() == b"changed corrupt audio"
    assert audio.resampled_episode_path.read_bytes() == b"changed corrupt audio"
    assert json.loads(audio.source_metadata_path.read_text()) == {
        "sha256": audio._content_fingerprint(source)
    }


@pytest.fixture
def fail_staged_text_write(monkeypatch):
    real_open = Path.open

    def install(output, *, successful_writes):
        @contextmanager
        def open_with_write_failure(file):
            with file:
                remaining = successful_writes

                def write(text):
                    nonlocal remaining
                    if remaining == 0:
                        raise OSError("injected stage write failure")
                    remaining -= 1
                    return file.write(text)

                yield SimpleNamespace(write=write)

        def open_path(path, *args, **kwargs):
            file = real_open(path, *args, **kwargs)
            if path.parent.name.startswith(".output-") and path.suffix == output.suffix:
                return open_with_write_failure(file)
            return file

        monkeypatch.setattr(Path, "open", open_path)

    return install


@pytest.mark.parametrize(
    "texts, expected",
    [
        ([], ""),
        (["first", "last"], "first\nlast"),
        (["café\nline two", "", "日本語\n"], "café\nline two\n\n日本語\n"),
    ],
)
def test_plaintext_export_bytes(tmp_path, texts, expected):
    from podcast_transcript.single_track import convert_to_plaintext

    source = _write_dote(
        tmp_path / "input.dote.json",
        [("00:00:00,000", "00:00:01,000", text) for text in texts],
    )
    output = tmp_path / "output.txt"

    convert_to_plaintext(source, output)

    assert output.read_bytes() == expected.replace("\n", os.linesep).encode("utf-8")


def test_plaintext_mid_write_failure_preserves_previous_export(
    tmp_path, fail_staged_text_write
):
    from podcast_transcript.single_track import convert_to_plaintext

    source = _write_dote(
        tmp_path / "input.dote.json",
        [("00:00:00,000", "00:00:01,000", text) for text in ("first", "second")],
    )
    output = tmp_path / "output.txt"
    output.write_bytes(b"previous transcript")
    fail_staged_text_write(output, successful_writes=1)

    with pytest.raises(OSError, match="injected stage write failure"):
        convert_to_plaintext(source, output)

    assert output.read_bytes() == b"previous transcript"
    assert sorted(tmp_path.iterdir()) == [source, output]


def test_webvtt_large_escaped_export_bytes(tmp_path):
    from podcast_transcript.single_track import convert_to_webvtt

    span = ("25:59:59,999", "25:59:59,999")
    payload = "<&\0>" * 128
    escaped = "&lt;&amp;\ufffd&gt;" * 128
    lines = [(*span, f"{i} café {payload}\r\n \t\r\nnext") for i in range(512)]
    lines.insert(1, (*span, " \r\n\t"))
    source = _write_dote(tmp_path / "input.dote.json", lines)
    output = tmp_path / "output.webvtt"

    convert_to_webvtt(source, output)

    expected = "WEBVTT\n" + "".join(
        f"\n25:59:59.999 --> 26:00:00.000\n{i} café {escaped}\nnext\n"
        for i in range(512)
    )
    assert output.read_bytes() == expected.replace("\n", os.linesep).encode("utf-8")


def test_webvtt_mid_write_failure_preserves_previous_export(
    tmp_path, fail_staged_text_write
):
    from podcast_transcript.single_track import convert_to_webvtt

    source = _write_dote(
        tmp_path / "input.dote.json",
        [("00:00:00,000", "00:00:01,000", text) for text in ("first", "second")],
    )
    output = tmp_path / "output.webvtt"
    output.write_bytes(b"previous captions")
    fail_staged_text_write(output, successful_writes=2)

    with pytest.raises(OSError, match="injected stage write failure"):
        convert_to_webvtt(source, output)

    assert output.read_bytes() == b"previous captions"
    assert sorted(tmp_path.iterdir()) == [source, output]


def test_webvtt_preserves_literal_text_in_one_cue(tmp_path):
    from html import unescape
    from podcast_transcript.single_track import convert_to_webvtt

    source = tmp_path / "input.json"
    output = tmp_path / "output.webvtt"
    source.write_text(
        json.dumps(
            {
                "lines": [
                    {
                        "startTime": "00:00:00,000",
                        "endTime": "00:00:01,000",
                        "text": "A & B <literal> -->\r\nnext\r\n\r\n \t\r\nlast\rend",
                    }
                ]
            }
        )
    )
    convert_to_webvtt(source, output)
    blocks = output.read_text().strip().split("\n\n")
    assert blocks[0] == "WEBVTT"
    assert len(blocks) == 2
    timing, payload = blocks[1].split("\n", 1)
    assert timing == "00:00:00.000 --> 00:00:01.000"
    assert "<literal>" not in payload
    assert "&amp;" in payload
    assert "-->" not in payload
    assert unescape(payload) == "A & B <literal> -->\nnext\nlast\nend"


def test_missing_cached_local_source_has_clear_error(mocker, tmp_path):
    import shutil

    source = tmp_path / "episode.mp3"
    source.write_bytes(b"AAAA")
    audio = Audio(base_dir=tmp_path / "cache", url=str(source))
    mocker.patch("podcast_transcript.audio.resample_audio", shutil.copyfile)
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1.0)
    audio.prepare_audio_for_transcription()
    source.unlink()
    with pytest.raises(
        RuntimeError, match="Local source audio is missing:.*episode.mp3"
    ):
        audio.prepare_audio_for_transcription()
    assert audio.episode_path.read_bytes() == b"AAAA"


def test_unchanged_local_source_only_hashes_original(mocker, tmp_path):
    import shutil

    source = tmp_path / "episode.mp3"
    source.write_bytes(b"AAAA")
    audio = Audio(base_dir=tmp_path / "cache", url=str(source))
    mocker.patch("podcast_transcript.audio.resample_audio", shutil.copyfile)
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1.0)
    audio.prepare_audio_for_transcription()
    fingerprint = mocker.spy(Audio, "_content_fingerprint")
    audio.prepare_audio_for_transcription()
    fingerprint.assert_called_once_with(source)


@pytest.mark.parametrize(
    "entries",
    [
        ["OUTSIDE"],
        ["../outside.mp3"],
        [123],
        ["chunk_000.mp3", "chunk_000.mp3"],
        ["chunk_001.mp3", "chunk_000.mp3"],
        ["chunk_001.mp3"],
        ["chunk_000.wav"],
        ["chunk_000.mp3", "chunk_002.mp3"],
    ],
)
def test_invalid_chunk_manifest_rebuilds_before_backend(mocker, tmp_path, entries):
    mocker.patch("podcast_transcript.single_track.settings.transcript_dir", tmp_path)
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1.0)
    url = "https://example.test/episode.mp3"
    audio = Audio(base_dir=tmp_path, url=url)
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.episode_path.write_bytes(b"source")
    audio.resampled_episode_path.write_bytes(b"resampled")
    outside = audio.podcast_dir / "outside.mp3"
    outside.write_bytes(b"private")
    for name in ("chunk_000.mp3", "chunk_001.mp3", "chunk_002.mp3", "chunk_000.wav"):
        (audio.episode_chunks_dir / name).write_bytes(b"stale")
    (audio.episode_chunks_dir / "123").write_bytes(b"private")
    entries = [str(outside) if name == "OUTSIDE" else name for name in entries]
    audio.chunks_manifest_path.write_text(json.dumps(entries))
    backend = RecordingBackend("m", "fresh")
    audio.cache_metadata_path.write_text(json.dumps(backend_cache_metadata(backend)))
    (audio.episode_chunks_dir / "chunk_000.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "text": "stale"}]})
    )
    audio.output_path(".txt").write_text("stale")
    paths = transcribe(url, backend)
    assert paths["plain text"].read_text() == "fresh"
    assert backend.calls == [audio.episode_chunks_dir / "chunk_000.mp3"]
    assert backend.calls[0].read_bytes() == b"resampled"
    assert json.loads(audio.chunks_manifest_path.read_text()) == ["chunk_000.mp3"]
    assert outside.read_bytes() == b"private"


def test_chunk_manifest_symlink_policy(audio, tmp_path):
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.resampled_episode_path.write_bytes(b"resampled")
    chunk = audio.episode_chunks_dir / "chunk_000.mp3"
    chunk.symlink_to(audio.resampled_episode_path)
    audio.chunks_manifest_path.write_text('["chunk_000.mp3"]')
    assert audio._load_chunk_manifest() == [chunk]
    chunk.unlink()
    outside = tmp_path / "outside.mp3"
    outside.write_bytes(b"private")
    chunk.symlink_to(outside)
    assert audio._load_chunk_manifest() is None
    chunk.unlink()
    chunk.mkdir()
    assert audio._load_chunk_manifest() is None
    chunk.rmdir()
    chunk.write_bytes(b"first")
    second = chunk.with_name("chunk_001.mp3")
    second.write_bytes(b"second")
    audio.chunks_manifest_path.write_text('["chunk_000.mp3", "chunk_001.mp3"]')
    assert audio._load_chunk_manifest() == [chunk, second]
    second.unlink()
    second.symlink_to(audio.resampled_episode_path)
    assert audio._load_chunk_manifest() is None


def test_voxhelm_service_identity_controls_cache(mocker, tmp_path, capsys):
    from podcast_transcript.backends import Voxhelm

    mocker.patch("podcast_transcript.single_track.settings.transcript_dir", tmp_path)
    mocker.patch.object(Audio, "prepare_audio_for_transcription", _fake_prepare)
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1.0)
    calls = []

    def record(self, audio_file, transcript_path):
        calls.append(audio_file)
        transcript_path.write_text(json.dumps({"segments": []}))

    mocker.patch.object(Voxhelm, "transcribe", record)

    def backend(base, key="invented-key"):
        return Voxhelm(
            api_base=base, api_key=key, model_name="m", language="en", prompt="p"
        )

    url = "https://example.test/audio.mp3"
    first = backend("https://FIRST.example:443/")
    transcribe(url, first)
    transcribe(url, backend("https://first.example/v1/", "rotated-key"))
    assert len(calls) == 1
    transcribe(url, backend("https://second.example"))
    assert len(calls) == 2
    assert backend_cache_metadata(first) != backend_cache_metadata(
        backend("https://first.example/prefix")
    )
    with pytest.raises(ValueError, match="VOXHELM_API_BASE"):
        backend(
            "https://user:invented-password@first.example/v1?key=invented-query#invented-fragment"
        )
    metadata = Audio(base_dir=tmp_path, url=url).cache_metadata_path.read_text()
    logged = capsys.readouterr().out
    for value in (
        "invented-key",
        "rotated-key",
        "invented-password",
        "invented-query",
        "invented-fragment",
    ):
        assert value not in metadata + logged


@pytest.mark.parametrize("cached_chunks", [False, True])
def test_bad_probe_stops_before_backend_and_preserves_outputs(
    mocker, tmp_path, cached_chunks
):
    mocker.patch("podcast_transcript.single_track.settings.transcript_dir", tmp_path)
    audio = Audio(base_dir=tmp_path, url="https://example.test/audio.mp3")
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.episode_path.write_bytes(b"source")
    audio.resampled_episode_path.write_bytes(b"resampled")
    old_output = audio.output_path(".txt")
    old_output.write_text("previous output")
    if cached_chunks:
        for name in ("chunk_000.mp3", "chunk_001.mp3"):
            (audio.episode_chunks_dir / name).write_bytes(b"chunk")
        audio.chunks_manifest_path.write_text('["chunk_000.mp3", "chunk_001.mp3"]')
    mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            [], 0, stdout='{"format": {"duration": "NaN"}}'
        ),
    )
    backend = RecordingBackend("m", "unused")
    with pytest.raises(RuntimeError, match="ffprobe"):
        transcribe(audio.url, backend)
    assert backend.calls == []
    assert old_output.read_text() == "previous output"
    assert not audio.cache_metadata_path.exists()
    assert not audio.source_metadata_path.exists()
    assert audio.chunks_manifest_path.exists() == cached_chunks


def test_invalid_manifest_survives_interrupted_transcript_cleanup(mocker, tmp_path):
    mocker.patch("podcast_transcript.single_track.settings.transcript_dir", tmp_path)
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1.0)
    url = "https://example.test/episode.mp3"
    audio = Audio(base_dir=tmp_path, url=url)
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.episode_path.write_bytes(b"source")
    audio.resampled_episode_path.write_bytes(b"resampled")
    audio.chunks_manifest_path.write_text('["chunk_001.mp3"]')
    (audio.episode_chunks_dir / "chunk_001.mp3").write_bytes(b"stale")
    backend = RecordingBackend("m", "fresh")
    audio.cache_metadata_path.write_text(json.dumps(backend_cache_metadata(backend)))
    (audio.episode_chunks_dir / "chunk_000.json").write_text(
        json.dumps({"segments": []})
    )
    cleanup = mocker.patch.object(
        Audio, "_remove_cached_transcripts", side_effect=PermissionError("interrupted")
    )
    with pytest.raises(PermissionError, match="interrupted"):
        audio.split_into_chunks()
    assert audio.chunks_manifest_path.exists()
    mocker.stop(cleanup)
    paths = transcribe(url, backend)
    assert paths["plain text"].read_text() == "fresh"
    assert len(backend.calls) == 1


def test_bad_new_chunk_probe_does_not_publish_preparation_markers(mocker, tmp_path):
    import shutil

    mocker.patch(
        "podcast_transcript.single_track.settings.transcript_dir", tmp_path / "cache"
    )
    mocker.patch("podcast_transcript.audio.resample_audio", shutil.copyfile)
    mocker.patch("podcast_transcript.audio.MAX_SIZE_IN_BYTES", 1)
    source = tmp_path / "source.mp3"
    source.write_bytes(b"synthetic")
    audio = Audio(base_dir=tmp_path / "cache", url=str(source))

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffprobe":
            duration = "2.0" if Path(cmd[-1]) == audio.resampled_episode_path else "NaN"
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps({"format": {"duration": duration}})
            )
        pattern = Path(cmd[-1])
        for idx in range(2):
            (pattern.parent / (pattern.name % idx)).write_bytes(b"chunk")
        return subprocess.CompletedProcess(cmd, 0)

    mocker.patch("podcast_transcript.audio.run_checked", side_effect=fake_run)
    mocker.patch("subprocess.run", side_effect=fake_run)
    backend = RecordingBackend("m", "unused")
    with pytest.raises(RuntimeError, match="ffprobe"):
        transcribe(str(source), backend)
    assert backend.calls == []
    assert not audio.source_metadata_path.exists()
    assert not audio.chunks_manifest_path.exists()
    assert not list(audio.episode_chunks_dir.glob(".split-*"))


def test_mlx_word_timestamps_invalidate_only_transcripts(mocker, tmp_path, fake_mlx):
    from podcast_transcript.backends import MLX

    mocker.patch("podcast_transcript.single_track.settings.transcript_dir", tmp_path)
    # Exercise real preparation/cache reuse with synthetic media boundaries.
    mocker.patch(
        "podcast_transcript.audio.download",
        side_effect=lambda _, p: p.write_bytes(b"source"),
    )
    resample = mocker.patch(
        "podcast_transcript.audio.resample_audio",
        side_effect=lambda _, p: p.write_bytes(b"resampled"),
    )
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1.0)
    log = tmp_path / "mlx-calls.log"
    fake_mlx(
        f"""
        def transcribe(path, **kwargs):
            with open({str(log)!r}, "a") as file:
                file.write(str(kwargs["word_timestamps"]) + "\\n")
            text = str(kwargs["word_timestamps"])
            return {{"segments": [{{"start": 0, "end": 1, "text": text}}]}}
        """
    )
    url = "https://example.test/synthetic.mp3"
    backend = MLX(model_name=None)
    paths = transcribe(url, backend)
    audio = Audio(base_dir=tmp_path, url=url)
    # Simulate an old cache missing the new option: invalidate exactly once.
    legacy = backend_cache_metadata(backend)
    legacy.pop("word_timestamps", None)
    audio.cache_metadata_path.write_text(json.dumps(legacy))
    transcribe(url, backend)
    transcribe(url, backend)
    backend.word_timestamps = True
    transcribe(url, backend)
    transcribe(url, backend)
    calls = [line == "True" for line in log.read_text().splitlines()]
    assert calls == [False, False, True]
    assert paths["plain text"].read_text() == "True"
    assert (
        json.loads((audio.episode_chunks_dir / "chunk_000.json").read_text())[
            "segments"
        ][0]["text"]
        == "True"
    )
    assert json.loads(audio.cache_metadata_path.read_text())["word_timestamps"] is True
    resample.assert_called_once()


@pytest.mark.parametrize("duration", [120.0, 7200.0])
def test_split_duration_respects_byte_budget(mocker, audio, duration):
    # Same byte count can be a high-bitrate short input or a two-hour episode.
    limit = 1000
    mocker.patch("podcast_transcript.audio.MAX_SIZE_IN_BYTES", limit)
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.resampled_episode_path.write_bytes(b"x" * 3000)
    mocker.patch(
        "podcast_transcript.audio.get_audio_duration",
        side_effect=lambda p: duration
        if p == audio.resampled_episode_path
        else duration / 4,
    )

    def segment(cmd, **kwargs):
        seconds = float(cmd[cmd.index("-segment_time") + 1])
        assert 0 < seconds <= min(7200, duration * limit / 3000)
        pattern = Path(cmd[-1])
        for i in range(4):
            (pattern.parent / (pattern.name % i)).write_bytes(b"x" * 750)
        return subprocess.CompletedProcess(cmd, 0)

    mocker.patch("podcast_transcript.audio.run_checked", side_effect=segment)
    assert len(audio.split_into_chunks()) == 4
    assert audio.chunks_manifest_path.exists()


@pytest.mark.parametrize(
    "size,duration,error",
    [
        (101, 1, "chunk.*limit"),
        (10, 7201, "chunk.*limit"),
        (0, 1, "empty"),
        (10, 0, "zero duration"),
    ],
)
def test_split_rejects_invalid_chunk_before_publication(
    mocker, audio, size, duration, error
):
    mocker.patch("podcast_transcript.audio.MAX_SIZE_IN_BYTES", 100)
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.resampled_episode_path.write_bytes(b"x" * 300)
    mocker.patch(
        "podcast_transcript.audio.get_audio_duration",
        side_effect=lambda p: 120 if p == audio.resampled_episode_path else duration,
    )

    def segment(cmd, **kwargs):
        pattern = Path(cmd[-1])
        (pattern.parent / (pattern.name % 0)).write_bytes(b"x" * size)
        return subprocess.CompletedProcess(cmd, 0)

    mocker.patch("podcast_transcript.audio.run_checked", side_effect=segment)
    with pytest.raises(RuntimeError, match=error):
        audio.split_into_chunks()
    assert not audio.chunks_manifest_path.exists()
    assert not list(audio.episode_chunks_dir.glob("chunk_*.mp3"))
    assert not list(audio.episode_chunks_dir.glob(".split-*"))


@pytest.mark.parametrize("size", [0, 101])
def test_cached_chunk_with_invalid_size_is_rebuilt(mocker, audio, size):
    mocker.patch("podcast_transcript.audio.MAX_SIZE_IN_BYTES", 100)
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=10)
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.resampled_episode_path.write_bytes(b"short")
    chunk = audio.episode_chunks_dir / "chunk_000.mp3"
    chunk.write_bytes(b"x" * size)
    audio.chunks_manifest_path.write_text('["chunk_000.mp3"]')
    audio.output_path(".txt").write_text("stale")
    assert audio.split_into_chunks() == [chunk]
    assert chunk.is_symlink()
    assert chunk.read_bytes() == b"short"
    assert not audio.output_path(".txt").exists()


@pytest.mark.parametrize("content,duration", [(b"", 1.0), (b"short", 0.0)])
def test_empty_short_audio_is_not_linked(mocker, audio, content, duration):
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.resampled_episode_path.write_bytes(content)
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=duration)

    with pytest.raises(RuntimeError, match="empty|zero duration"):
        audio.split_into_chunks()
    assert not audio.chunks_manifest_path.exists()
    assert not list(audio.episode_chunks_dir.glob("chunk_*.mp3"))


@pytest.mark.parametrize("duration", [0.0, 7200.0, 7200.25])
@pytest.mark.parametrize("count", [1, 2])
def test_cached_chunks_are_checked_before_backend(mocker, audio, duration, count):
    mocker.patch(
        "podcast_transcript.single_track.settings.transcript_dir", audio.base_dir
    )
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.episode_path.write_bytes(b"source")
    audio.resampled_episode_path.write_bytes(b"resampled")
    chunks = [audio.episode_chunks_dir / f"chunk_{i:03d}.mp3" for i in range(count)]
    for chunk in chunks:
        chunk.write_bytes(b"cached audio")
    audio.chunks_manifest_path.write_text(json.dumps([chunk.name for chunk in chunks]))
    audio.output_path(".txt").write_text("previous")
    probe = mocker.patch(
        "podcast_transcript.audio.capture_checked",
        side_effect=lambda cmd, **kwargs: json.dumps(
            {"format": {"duration": duration if cmd[-1] == chunks[-1] else 10.0}}
        ),
    )
    backend = RecordingBackend("m", "admitted")

    if duration == 7200.0:
        transcribe(audio.url, backend)
        assert backend.calls == chunks
    else:
        with pytest.raises(RuntimeError, match="chunk.*duration"):
            transcribe(audio.url, backend)
        assert not backend.calls
        assert audio.output_path(".txt").read_text() == "previous"
    assert [call.args[0][-1] for call in probe.call_args_list] == chunks


@pytest.mark.parametrize("local_source", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_fresh_chunks_are_probed_once(mocker, audio, local_source, split):
    mocker.patch(
        "podcast_transcript.single_track.settings.transcript_dir", audio.base_dir
    )
    mocker.patch("podcast_transcript.audio.MAX_SIZE_IN_BYTES", 100)
    if local_source:
        source = audio.base_dir / "local.mp3"
        source.write_bytes(b"source")
        audio = Audio(base_dir=audio.base_dir, url=str(source))
    else:
        audio.episode_chunks_dir.mkdir(parents=True)
        audio.episode_path.write_bytes(b"source")

    def run(cmd, **kwargs):
        target = Path(cmd[-1])
        if "-segment_time" in cmd:
            for i in range(2):
                (target.parent / (target.name % i)).write_bytes(b"chunk")
        else:
            target.write_bytes(b"x" * (300 if split else 10))

    mocker.patch("podcast_transcript.audio.run_checked", side_effect=run)
    probe = mocker.patch(
        "podcast_transcript.audio.capture_checked",
        return_value='{"format": {"duration": 10.0}}',
    )
    backend = RecordingBackend("m", "fresh")
    transcribe(audio.url, backend)
    expected_names = ["chunk_000.mp3", "chunk_001.mp3"] if split else ["chunk_000.mp3"]
    assert [path.name for path in backend.calls] == expected_names
    assert [call.args[0][-1].name for call in probe.call_args_list] == [
        audio.resampled_episode_path.name,
        *(expected_names if split else []),
    ]


def test_preparation_probe_is_not_reused_for_replaced_chunk(mocker, audio):
    mocker.patch(
        "podcast_transcript.single_track.settings.transcript_dir", audio.base_dir
    )
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.episode_path.write_bytes(b"source")
    audio.resampled_episode_path.write_bytes(b"short")
    prepare = Audio.prepare_audio_for_transcription

    def prepare_and_replace(store):
        chunks = prepare(store)
        replacement = chunks[0].with_suffix(".replacement")
        replacement.write_bytes(b"long!")
        replacement.replace(chunks[0])
        return chunks

    mocker.patch.object(Audio, "prepare_audio_for_transcription", prepare_and_replace)
    probe = mocker.patch(
        "podcast_transcript.audio.get_audio_duration",
        side_effect=lambda path: 10.0 if path.read_bytes() == b"short" else 7200.25,
    )
    backend = RecordingBackend("m", "fresh")

    with pytest.raises(RuntimeError, match="chunk.*duration"):
        transcribe(audio.url, backend)
    assert not backend.calls
    assert [call.args[0].name for call in probe.call_args_list] == [
        audio.resampled_episode_path.name,
        "chunk_000.mp3",
    ]


def test_oversized_zero_duration_audio_fails_without_chunk_manifest(mocker, audio):
    mocker.patch("podcast_transcript.audio.MAX_SIZE_IN_BYTES", 1)
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=0)
    run = mocker.patch("podcast_transcript.audio.run_checked")
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.resampled_episode_path.write_bytes(b"oversized")
    with pytest.raises(RuntimeError, match="zero duration"):
        audio.split_into_chunks()
    run.assert_not_called()
    assert not audio.chunks_manifest_path.exists()


@pytest.mark.parametrize(
    "start, end, dote_start, dote_end, vtt_end",
    [
        (1.0001, 1.0002, "00:00:01,000", "00:00:01,000", "00:00:01.001"),
        (1.0, 1.0, "00:00:01,000", "00:00:01,000", "00:00:01.001"),
        (0.0, 0.0, "00:00:00,000", "00:00:00,000", "00:00:00.001"),
        (59.9991, 59.9992, "00:00:59,999", "00:00:59,999", "00:01:00.000"),
        (
            359999.9991,
            359999.9992,
            "99:59:59,999",
            "99:59:59,999",
            "100:00:00.000",
        ),
        (1.125, 2.875, "00:00:01,125", "00:00:02,875", "00:00:02.875"),
        (
            360000.125,
            360001.875,
            "100:00:00,125",
            "100:00:01,875",
            "100:00:01.875",
        ),
    ],
)
def test_whisper_to_webvtt_keeps_visible_cue_spans(
    tmp_path, start, end, dote_start, dote_end, vtt_end
):
    from podcast_transcript.single_track import (
        convert_to_webvtt,
        whisper_text_chunks_to_dote,
    )

    raw = tmp_path / "chunk_000.json"
    raw_bytes = json.dumps(
        {"segments": [{"start": start, "end": end, "text": "synthetic subtitle"}]}
    ).encode()
    raw.write_bytes(raw_bytes)
    [dote_path] = whisper_text_chunks_to_dote([raw])
    dote_bytes = dote_path.read_bytes()
    [line] = json.loads(dote_bytes)["lines"]
    assert (line["startTime"], line["endTime"]) == (dote_start, dote_end)

    output = tmp_path / "output.webvtt"
    convert_to_webvtt(dote_path, output)

    assert output.read_text() == (
        f"WEBVTT\n\n{dote_start.replace(',', '.')} --> {vtt_end}\nsynthetic subtitle\n"
    )
    assert raw.read_bytes() == raw_bytes
    assert dote_path.read_bytes() == dote_bytes


def test_webvtt_parses_each_dote_endpoint_once(tmp_path, monkeypatch, mocker):
    from podcast_transcript import single_track, transcript_validation

    dote = whisper_to_dote([{"start": 1.0001, "end": 1.0002, "text": "Quantized"}])
    dote["lines"][0].update(startTime="000000:00:01,000", endTime="000000:00:01,000")
    dote["lines"].append(
        {"startTime": "00:00:02,000", "endTime": "00:00:03,000", "text": " "}
    )
    expected = ["WEBVTT\n\n00:00:01.000 --> 00:00:01.001\nQuantized\n"]
    for index in range(64):
        start = f"000123:00:00,{index:03}"
        end = f"000123:00:01,{index:03}"
        dote["lines"].append({"startTime": start, "endTime": end, "text": "Café &"})
        expected.append(
            f"\n123:00:00.{index:03} --> 123:00:01.{index:03}\nCafé &amp;\n"
        )

    source = tmp_path / "input.dote.json"
    original = json.dumps(dote).encode("utf-8")
    source.write_bytes(original)
    output = tmp_path / "output.webvtt"
    parse = mocker.Mock(wraps=transcript_validation.parse_dote_timecode)
    monkeypatch.setattr(transcript_validation, "parse_dote_timecode", parse)

    single_track.convert_to_webvtt(source, output)

    assert output.read_bytes() == "".join(expected).encode("utf-8")
    assert source.read_bytes() == original
    assert parse.call_count == 2 * len(dote["lines"])


@pytest.mark.parametrize("text", ["", " ", "\r\n \t\n"])
def test_webvtt_omits_blank_cues_only_from_rendered_output(tmp_path, text):
    from podcast_transcript.single_track import convert_to_webvtt

    source = _write_dote(
        tmp_path / "input.dote.json",
        [("00:00:00,000", "00:00:00,000", text)],
    )
    original = source.read_bytes()
    output = tmp_path / "output.webvtt"

    convert_to_webvtt(source, output)

    assert output.read_text() == "WEBVTT\n"
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "text, expected",
    [("café\0日本語\0🌍", "café\ufffd日本語\ufffd🌍"), ("\0", "\ufffd")],
)
def test_webvtt_replaces_nul_in_cue_text(tmp_path, text, expected):
    from podcast_transcript.single_track import convert_to_webvtt

    source = _write_dote(
        tmp_path / "input.dote.json",
        [("00:00:00,000", "00:00:01,000", text)],
    )
    original = source.read_bytes()
    output = tmp_path / "output.webvtt"

    convert_to_webvtt(source, output)

    assert output.read_text(encoding="utf-8") == (
        f"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n{expected}\n"
    )
    assert source.read_bytes() == original
