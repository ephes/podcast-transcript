"""Refused control sidecars preserve prior artifacts before invalidation."""

import json
import os
import shutil

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


@pytest.mark.parametrize(
    ("marker", "payload"),
    [
        ("cache", "{"),
        ("chunks", "{"),
        ("cache", "[]"),
        ("source", "false"),
        ("chunks", "{}"),
    ],
)
def test_control_preflight_precedes_changed_source_retirement(
    audio, tmp_path, monkeypatch, marker, payload
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
    path = {
        "cache": store.cache_metadata_path,
        "source": store.source_metadata_path,
        "chunks": store.chunks_manifest_path,
    }[marker]
    path.write_text(payload)
    monkeypatch.setattr(single_track.settings, "transcript_dir", store.base_dir)
    monkeypatch.setattr(
        type(audio),
        "_prepare_changed_local_source",
        lambda *a: pytest.fail("retirement before control admission"),
    )
    with pytest.raises(RuntimeError, match=rf"control JSON in .*{marker}\.json"):
        single_track.transcribe(str(source), RecordingBackend("synthetic", "new"))
    assert output.read_bytes() == b"old transcript"
    assert store.episode_path.read_bytes() == b"old audio"
    assert path.read_text() == payload
    if marker != "source":
        assert store.source_metadata_path.read_text() == '{"sha256":"previous"}'


def test_missing_source_control_rebuilds_prepared_audio(audio, tmp_path, mocker):
    source = tmp_path / "changed.mp3"
    source.write_bytes(b"new audio")
    store = type(audio)(base_dir=audio.base_dir, url=str(source))
    store.episode_chunks_dir.mkdir(parents=True)
    store.episode_path.write_bytes(b"old audio")
    assert not store.source_metadata_path.exists()
    mocker.patch("podcast_transcript.audio.resample_audio", shutil.copyfile)
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1.0)

    (chunk,) = store.prepare_audio_for_transcription()

    assert chunk.read_bytes() == store.episode_path.read_bytes() == b"new audio"
    assert json.loads(store.source_metadata_path.read_text()) == {
        "sha256": store._content_fingerprint(source)
    }


@pytest.mark.parametrize("change", ["rewrite", "truncate", "replace", "unlink"])
@pytest.mark.parametrize("relative", [False, True])
def test_control_read_rejects_changed_generation(
    tmp_path, monkeypatch, change, relative
):
    from podcast_transcript.input_admission import read_regular_bounded

    path = tmp_path / "control.json"
    original = b'{"first":"old","last":"old"}'
    path.write_bytes(original)
    read = os.read
    changed = False
    inode = path.stat().st_ino

    def read_then_change(fd, count):
        nonlocal changed
        block = read(fd, min(count, 8))
        if os.fstat(fd).st_ino == inode and not changed:
            changed = True
            if change == "rewrite":
                path.write_bytes(original.replace(b"old", b"new"))
            elif change == "truncate":
                path.write_bytes(b"{}")
            elif change == "replace":
                replacement = tmp_path / "new.json"
                replacement.write_bytes(original)
                replacement.replace(path)
            else:
                path.unlink()
        return block

    monkeypatch.setattr(os, "read", read_then_change)
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY) if relative else None
    try:
        with pytest.raises(RuntimeError, match="control.*changed"):
            read_regular_bounded(
                type(path)(path.name) if relative else path,
                limit=1024,
                description="control",
                dir_fd=parent,
            )
    finally:
        if parent is not None:
            os.close(parent)


def test_control_rewrite_preserves_previous_cache_generation(audio, monkeypatch):
    audio.episode_chunks_dir.mkdir(parents=True)
    output = audio.output_path(".txt")
    output.write_bytes(b"previous transcript")
    audio.cache_metadata_path.write_bytes(b'{"backend":"old"}')
    read = os.read
    changed = False
    inode = audio.cache_metadata_path.stat().st_ino

    def rewrite(fd, count):
        nonlocal changed
        block = read(fd, min(count, 8))
        if os.fstat(fd).st_ino == inode and not changed:
            changed = True
            audio.cache_metadata_path.write_bytes(b'{"backend":"new"}')
        return block

    monkeypatch.setattr(os, "read", rewrite)
    with pytest.raises(RuntimeError, match="control"):
        audio.ensure_transcript_cache_matches({"backend": "replacement"})
    assert output.read_bytes() == b"previous transcript"
    assert audio.cache_metadata_path.read_bytes() == b'{"backend":"new"}'


def test_unchanged_control_chunked_read(tmp_path, monkeypatch):
    from podcast_transcript.input_admission import read_regular_bounded

    path = tmp_path / "control.json"
    data = b'{"backend":"synthetic"}'
    path.write_bytes(data)
    read = os.read
    monkeypatch.setattr(os, "read", lambda fd, count: read(fd, min(count, 3)))
    assert read_regular_bounded(path, limit=len(data), description="control") == data
