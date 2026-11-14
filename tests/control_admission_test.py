"""Refused control sidecars preserve prior artifacts before invalidation."""

import json
import os

import pytest

from podcast_transcript import json_input


@pytest.mark.parametrize("marker", ["cache", "source", "chunks"])
@pytest.mark.parametrize(
    "kind", ["oversize", "growth", "fifo", "malformed", "deep", "integer"]
)
def test_control_admission_preserves_generation(
    audio, tmp_path, monkeypatch, marker, kind
):
    audio.episode_chunks_dir.mkdir(parents=True)
    source = tmp_path / "source.mp3"
    source.write_bytes(b"synthetic source")
    audio.url = str(source)
    path = {
        "cache": audio.cache_metadata_path,
        "source": audio.source_metadata_path,
        "chunks": audio.chunks_manifest_path,
    }[marker]
    output = audio.output_path(".txt")
    output.write_bytes(b"previous transcript")
    audio.episode_path.write_bytes(b"previous audio")
    chunk = audio.episode_chunks_dir / "chunk_000.mp3"
    chunk.write_bytes(b"previous chunk")
    monkeypatch.setattr(json_input, "CONTROL_MAX_BYTES", 1024, raising=False)
    if kind == "fifo":
        os.mkfifo(path)
        read_text = type(path).read_text

        def no_block(candidate, *args, **kwargs):
            if candidate == path:
                pytest.fail("control file used blocking read_text on FIFO")
            return read_text(candidate, *args, **kwargs)

        monkeypatch.setattr(type(path), "read_text", no_block)
    else:
        payload = {
            "oversize": '"' + "x" * 1100 + '"',
            "growth": "{}",
            "malformed": "{",
            "deep": "[" * 2000 + "]" * 2000,
            "integer": "9" * 5000,
        }[kind]
        path.write_text(payload)
        if kind == "growth":
            read = os.read

            def growing(fd, count):
                if os.fstat(fd).st_ino == path.stat().st_ino:
                    with path.open("ab") as file:
                        file.write(b" " * 1100)
                return read(fd, count)

            monkeypatch.setattr(os, "read", growing)
    before = None if kind == "fifo" else path.read_bytes()
    with pytest.raises(RuntimeError, match="control"):
        if marker == "cache":
            audio.ensure_transcript_cache_matches({"backend": "new"})
        elif marker == "source":
            audio._local_source_changed()
        else:
            audio.split_into_chunks()
    assert output.read_bytes() == b"previous transcript"
    assert audio.episode_path.read_bytes() == b"previous audio"
    assert chunk.read_bytes() == b"previous chunk"
    if before is not None and kind != "growth":
        assert path.read_bytes() == before
    assert path.exists()


def test_missing_control_recovery_remains_deliberate(audio):
    assert audio._load_chunk_manifest() is None
    audio.ensure_transcript_cache_matches({"backend": "synthetic"})
    assert json.loads(audio.cache_metadata_path.read_text()) == {"backend": "synthetic"}


@pytest.mark.parametrize("marker", ["cache", "chunks"])
def test_control_preflight_precedes_changed_source_retirement(
    audio, tmp_path, monkeypatch, marker
):
    from podcast_transcript import single_track
    from single_track_test import RecordingBackend

    source = tmp_path / "changed.mp3"
    source.write_bytes(b"new audio")
    store = type(audio)(base_dir=audio.base_dir, url=str(source))
    store.episode_chunks_dir.mkdir(parents=True)
    store.source_metadata_path.write_text('{"sha256":"previous"}')
    store.episode_path.write_bytes(b"old audio")
    output = store.output_path(".txt")
    output.write_bytes(b"old transcript")
    path = (
        store.cache_metadata_path if marker == "cache" else store.chunks_manifest_path
    )
    path.write_text("{")
    monkeypatch.setattr(single_track.settings, "transcript_dir", store.base_dir)
    monkeypatch.setattr(
        type(audio),
        "_prepare_changed_local_source",
        lambda *a: pytest.fail("retirement before control admission"),
    )
    with pytest.raises(RuntimeError, match="control"):
        single_track.transcribe(str(source), RecordingBackend("synthetic", "new"))
    assert output.read_bytes() == b"old transcript"
    assert store.episode_path.read_bytes() == b"old audio"
    assert store.source_metadata_path.read_text() == '{"sha256":"previous"}'
