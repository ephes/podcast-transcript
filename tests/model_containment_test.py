"""Mutable directory names never redirect local model hashing outside its root."""

import os

import pytest

from podcast_transcript import audio, backends, single_track
from single_track_test import _fake_prepare


@pytest.mark.parametrize("component", ["root", "queued"])
@pytest.mark.parametrize("barrier", ["enumerated", "leaf-open", "read"])
def test_model_directory_exchange_is_refused_before_inference(
    tmp_path, monkeypatch, component, barrier
):
    model = tmp_path / "model"
    child = model / "queued"
    child.mkdir(parents=True)
    (child / "weights").write_bytes(b"admitted synthetic model")
    outside = tmp_path / "outside"
    outside_child = outside / "queued" if component == "root" else outside
    outside_child.mkdir(parents=True)
    external_file = outside_child / "weights"
    external_file.write_bytes(b"outside synthetic model")
    external_inode = external_file.stat().st_ino
    admitted_inode = (child / "weights").stat().st_ino
    root_inode = model.stat().st_ino
    changed = []
    outside_reads = []

    def exchange():
        if changed:
            return
        path = model if component == "root" else child
        path.rename(tmp_path / "retired")
        path.symlink_to(outside, target_is_directory=True)
        changed.append(path)

    scan = os.scandir

    class ExchangeAfterEnumeration:
        def __init__(self, path):
            self.entries = scan(path)
            info = os.fstat(path) if isinstance(path, int) else os.stat(path)
            self.root = info.st_ino == root_inode

        def __enter__(self):
            return self.entries

        def __exit__(self, *args):
            self.entries.close()
            if self.root and barrier == "enumerated":
                exchange()

    read_model = audio.Audio._read_local_source

    def exchange_at_leaf(path, *args, **kwargs):
        if kwargs.get("model") and barrier == "leaf-open":
            exchange()
        return read_model(path, *args, **kwargs)

    read = os.read

    def guarded_read(fd, size):
        inode = os.fstat(fd).st_ino
        if inode == external_inode:
            outside_reads.append(fd)
            pytest.fail("hashed a model outside the admitted tree")
        block = read(fd, size)
        if inode == admitted_inode and barrier == "read":
            exchange()
        return block

    store = audio.Audio(
        base_dir=tmp_path / "cache", url="https://example.test/audio.mp3"
    )
    store.podcast_dir.mkdir(parents=True)
    previous = store.output_path(".txt")
    previous.write_bytes(b"previous transcript")
    store.cache_metadata_path.write_bytes(b'{"backend":"previous"}')
    backend = backends.MLX(model_name=str(model))
    monkeypatch.setattr(single_track.settings, "transcript_dir", store.base_dir)
    monkeypatch.setattr(audio.Audio, "prepare_audio_for_transcription", _fake_prepare)
    monkeypatch.setattr(os, "scandir", ExchangeAfterEnumeration)
    monkeypatch.setattr(
        audio.Audio, "_read_local_source", staticmethod(exchange_at_leaf)
    )
    monkeypatch.setattr(os, "read", guarded_read)
    monkeypatch.setattr(
        backends.MLX,
        "transcribe",
        lambda *a: pytest.fail("inference before stable model admission"),
    )
    with pytest.raises(RuntimeError, match="[Mm]odel"):
        single_track.transcribe(store.url, backend)
    assert changed
    assert outside_reads == []
    assert previous.read_bytes() == b"previous transcript"
    assert store.cache_metadata_path.read_bytes() == b'{"backend":"previous"}'
    assert external_file.read_bytes() == b"outside synthetic model"
