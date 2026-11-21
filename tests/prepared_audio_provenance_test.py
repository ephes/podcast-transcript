"""Offline prepared-audio identity and bounded hashing."""

import hashlib
import json

import pytest

from podcast_transcript import audio as audio_module
from podcast_transcript import single_track
from single_track_test import RecordingBackend


@pytest.fixture
def short_audio(audio):
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.resampled_episode_path.write_bytes(b"prepared short audio")
    link = audio.episode_chunks_dir / "chunk_000.mp3"
    link.symlink_to(audio.resampled_episode_path.name)
    audio.chunks_manifest_path.write_text(json.dumps([link.name]))
    return audio, link


def test_short_audio_link_hashes_owned_target_and_rejects_external_link(
    short_audio, tmp_path
):
    audio, link = short_audio
    before = single_track._audio_generation(audio)
    audio.resampled_episode_path.write_bytes(b"edited short audio")
    assert single_track._audio_generation(audio) != before
    external = tmp_path / "external.mp3"
    external.write_bytes(b"edited short audio")
    link.unlink()
    link.symlink_to(external)
    with pytest.raises(RuntimeError, match="inventory"):
        single_track._audio_generation(audio)


@pytest.mark.parametrize("budget", ["bytes", "elapsed", "growth"])
def test_audio_hash_refusal_preserves_public_and_recovery(episode, monkeypatch, budget):
    store, _ = episode
    single_track.transcribe(store.url, RecordingBackend("new", "new"))
    if budget == "growth":
        store.resampled_episode_path.write_bytes(b"x" * 1024 * 1024)
    recovery = store.podcast_dir / ".previous-transcripts"
    before = {
        p: p.read_bytes()
        for p in [
            store.cache_metadata_path,
            *single_track._transcript_files(store),
            *[p for p in recovery.rglob("*") if p.is_file()],
        ]
    }
    if budget == "elapsed":
        clock = iter([0, 3601])
        monkeypatch.setattr(single_track, "monotonic", lambda: next(clock))
    else:
        paths = [
            store.resampled_episode_path,
            *store.episode_chunks_dir.glob("chunk_*.mp3"),
        ]
        size = sum(p.stat().st_size for p in paths)
        monkeypatch.setattr(
            single_track, "PREPARED_AUDIO_MAX_BYTES", size - (budget == "bytes")
        )
        if budget == "growth":
            blocks = single_track.regular_file_blocks
            grown = []

            def grow(path, **kwargs):
                for block in blocks(path, **kwargs):
                    assert len(block) <= 1024 * 1024
                    if path == store.resampled_episode_path and not grown:
                        grown.append(path)
                        # Cross the aggregate limit before the source's EOF check.
                        with path.open("ab") as out:
                            out.write(b"x" * (size - path.stat().st_size + 1))
                    yield block

            monkeypatch.setattr(single_track, "regular_file_blocks", grow)
    with pytest.raises(RuntimeError, match="hashing exceeded"):
        single_track.restore_last_success(store.url)
    assert all(p.read_bytes() == value for p, value in before.items())


def _published(store):
    return {
        p: p.read_bytes()
        for p in [store.cache_metadata_path, *single_track._transcript_files(store)]
    }


@pytest.mark.parametrize("retarget_on_pass", [1, 2])
def test_restore_refuses_short_chunk_retargeted_after_manifest_admission(
    short_audio, tmp_path, monkeypatch, retarget_on_pass
):
    audio, link = short_audio
    monkeypatch.setattr(single_track.settings, "transcript_dir", audio.base_dir)
    monkeypatch.setattr(
        type(audio), "prepare_audio_for_transcription", lambda self: [link]
    )
    monkeypatch.setattr(audio_module, "get_audio_duration", lambda _: 1.0)
    single_track.transcribe(audio.url, RecordingBackend("old", "old transcript"))
    single_track.transcribe(audio.url, RecordingBackend("new", "new transcript"))
    before = _published(audio)

    external = tmp_path / "external.mp3"
    external.write_bytes(b"different audio")
    load_manifest = type(audio)._load_chunk_manifest
    passes = 0

    def retarget_after_load(self):
        nonlocal passes
        chunks = load_manifest(self)
        if chunks == [link]:
            passes += 1
            if passes == retarget_on_pass:
                link.unlink()
                link.symlink_to(external)
        return chunks

    monkeypatch.setattr(type(audio), "_load_chunk_manifest", retarget_after_load)
    if retarget_on_pass == 1:

        def unexpected_copy(*args, **kwargs):
            pytest.fail(
                "Recovery copied transcripts after the short chunk was retargeted"
            )

        monkeypatch.setattr(single_track, "copy_file_atomic", unexpected_copy)
    with pytest.raises(RuntimeError, match="short chunk link"):
        single_track.restore_last_success(audio.url)
    assert _published(audio) == before


def test_short_chunk_alias_preserves_inventory_digest_with_one_read(
    short_audio, monkeypatch
):
    audio, link = short_audio
    source = audio.resampled_episode_path
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    inventory = "".join(
        json.dumps((path.relative_to(audio.podcast_dir).as_posix(), digest)) + "\n"
        for path in (source, link)
    )
    blocks = single_track.regular_file_blocks
    reads = []

    def record_reads(path, **kwargs):
        reads.append(path)
        yield from blocks(path, **kwargs)

    monkeypatch.setattr(single_track, "regular_file_blocks", record_reads)
    assert (
        single_track._prepared_audio_digest(audio)
        == hashlib.sha256(inventory.encode()).hexdigest()
    )
    assert reads == [source]
    monkeypatch.setattr(single_track, "PREPARED_AUDIO_MAX_BYTES", len(payload))
    assert (
        single_track._prepared_audio_digest(audio)
        == hashlib.sha256(inventory.encode()).hexdigest()
    )


def test_short_chunk_alias_refuses_replaced_resample_after_hashing(
    short_audio, tmp_path, monkeypatch
):
    audio, _ = short_audio
    source = audio.resampled_episode_path
    replacement = tmp_path / "replacement.mp3"
    replacement.write_bytes(source.read_bytes())
    blocks = single_track.regular_file_blocks

    def replace_after_read(path, **kwargs):
        yield from blocks(path, **kwargs)
        if path == source and replacement.exists():
            replacement.replace(source)

    monkeypatch.setattr(single_track, "regular_file_blocks", replace_after_read)
    with pytest.raises(RuntimeError, match="Prepared audio changed"):
        single_track._prepared_audio_digest(audio)


def test_unhashable_prepared_audio_refuses_replacement_before_inference(
    episode, monkeypatch
):
    store, _ = episode
    before = _published(store)
    monkeypatch.setattr(single_track, "PREPARED_AUDIO_MAX_BYTES", 1)
    backend = RecordingBackend("new", "new")
    with pytest.raises(RuntimeError, match="hashing exceeded"):
        single_track.transcribe(store.url, backend)
    assert backend.calls == []
    assert _published(store) == before
    assert not (store.podcast_dir / ".previous-transcripts").exists()


def test_prepared_audio_changed_during_replacement_is_not_promoted(episode):
    store, _ = episode
    before = _published(store)

    class Mutating(RecordingBackend):
        def transcribe(self, audio_file, transcript_path):
            store.resampled_episode_path.write_bytes(b"edited during inference")
            super().transcribe(audio_file, transcript_path)

    backend = Mutating("new", "new")
    with pytest.raises(RuntimeError, match="changed during transcription"):
        single_track.transcribe(store.url, backend)
    assert backend.calls
    assert _published(store) == before
    assert not (store.podcast_dir / ".previous-transcripts").exists()
