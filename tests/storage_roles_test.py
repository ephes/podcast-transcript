"""External source basenames cannot occupy generated storage roles."""

import shutil

import pytest

from podcast_transcript import audio, single_track
from single_track_test import RecordingBackend


@pytest.mark.parametrize(
    "name",
    [
        "chunks.json",
        "Chunks.JSON",
        "chunk_000.json",
        "chunk_000.dote.json",
        "chunk_000.dote.receipt.json",
        "chunk_000.whisper-cpp.json",
        "chunk_000.mp3",
        "CHUNK_000.WAV",
    ],
)
@pytest.mark.parametrize("remote", [False, True])
def test_reserved_source_survives_prepare_invalidation_and_retry(
    tmp_path, monkeypatch, name, remote
):
    source = tmp_path / name
    source.write_bytes(b"synthetic source")
    url = f"https://example.test/{name}" if remote else str(source)
    monkeypatch.setattr(single_track.settings, "transcript_dir", tmp_path / "cache")
    monkeypatch.setattr(
        audio, "download", lambda _, target: shutil.copyfile(source, target)
    )
    monkeypatch.setattr(audio, "resample_audio", shutil.copyfile)
    monkeypatch.setattr(audio, "get_audio_duration", lambda _: 1.0)
    store = audio.Audio(base_dir=single_track.settings.transcript_dir, url=url)
    assert store.episode_path.name.casefold().startswith("episode_")
    paths = single_track.transcribe(url, RecordingBackend("first", "one"))
    assert store.episode_path.read_bytes() == b"synthetic source"
    single_track.transcribe(url, RecordingBackend("second", "two"))
    assert store.episode_path.read_bytes() == b"synthetic source"
    store.chunks_manifest_path.unlink()
    store.prepare_audio_for_transcription()
    assert store.episode_path.read_bytes() == b"synthetic source"
    single_track.transcribe(url, RecordingBackend("second", "two"))
    assert paths["plain text"].read_text() == "two"
    assert source.read_bytes() == b"synthetic source"


def test_storage_role_preflight_preserves_files_before_conversion_or_cleanup(
    tmp_path, monkeypatch
):
    class Colliding(audio.Audio):
        @property
        def resampled_episode_path(self):
            return self.episode_chunks_dir / "chunks.json"

    store = Colliding(base_dir=tmp_path, url="https://example.test/synthetic.mp3")
    store.episode_chunks_dir.mkdir(parents=True)
    store.chunks_manifest_path.write_text('["chunk_000.mp3"]')
    store.output_path(".txt").write_text("previous")
    monkeypatch.setattr(
        audio,
        "resample_audio",
        lambda *args: pytest.fail("conversion before preflight"),
    )
    for operation in [
        store.prepare_audio_for_transcription,
        store.split_into_chunks,
        store.make_sure_audio_file_is_resampled,
        lambda: store.ensure_transcript_cache_matches({"backend": "new"}),
    ]:
        with pytest.raises(RuntimeError, match="storage roles overlap"):
            operation()
    assert store.chunks_manifest_path.read_text() == '["chunk_000.mp3"]'
    assert store.output_path(".txt").read_text() == "previous"
