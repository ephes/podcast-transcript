"""Whole-tree model admission uses cumulative, deterministic budgets."""

import hashlib
import json
import os

import pytest

from podcast_transcript import audio, single_track


def test_directory_identity_remains_compatible(tmp_path):
    (tmp_path / "z").write_bytes(b"last")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "a").write_bytes(b"first")
    files = [
        (".hidden/a", hashlib.sha256(b"first").hexdigest()),
        ("z", hashlib.sha256(b"last").hexdigest()),
    ]
    assert (
        single_track.local_model_identity(tmp_path)
        == hashlib.sha256(json.dumps(files).encode()).hexdigest()
    )


def test_tree_has_one_byte_budget(tmp_path, monkeypatch):
    for name in ("a", "b", "c"):
        (tmp_path / name).write_bytes(b"1234")
    monkeypatch.setattr(audio, "MODEL_MAX_BYTES", 10)
    with pytest.raises(RuntimeError, match="[Mm]odel.*byte"):
        single_track.local_model_identity(tmp_path)


def test_tree_enumeration_is_bounded_before_hashing(tmp_path, monkeypatch):
    for index in range(8):
        (tmp_path / str(index)).mkdir()  # Empty directories also cost traversal.
    monkeypatch.setattr(audio, "MODEL_MAX_ENTRIES", 4, raising=False)
    monkeypatch.setattr(os, "read", lambda *a: pytest.fail("read before admission"))
    with pytest.raises(RuntimeError, match="[Mm]odel.*entry"):
        single_track.local_model_identity(tmp_path)


def test_tree_has_one_deadline(tmp_path, monkeypatch):
    for name in ("a", "b", "c"):
        (tmp_path / name).write_bytes(b"1")
    now = [0.0]
    read = os.read

    def elapsed(fd, size):
        now[0] += 0.4
        return read(fd, size)

    monkeypatch.setattr(audio, "MODEL_HASH_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(audio, "monotonic", lambda: now[0])
    monkeypatch.setattr(os, "read", elapsed)
    with pytest.raises(RuntimeError, match="[Mm]odel.*elapsed"):
        single_track.local_model_identity(tmp_path)


def test_tree_refusal_starts_no_inference(tmp_path, monkeypatch):
    from podcast_transcript.backends import MLX
    from single_track_test import _fake_prepare

    model = tmp_path / "model"
    model.mkdir()
    for name in ("a", "b"):
        (model / name).write_bytes(b"1234")
    backend = MLX(model_name=str(model))
    store = audio.Audio(
        base_dir=tmp_path / "cache", url="https://example.test/audio.mp3"
    )
    store.podcast_dir.mkdir(parents=True)
    previous = store.output_path(".txt")
    previous.write_bytes(b"previous transcript")
    store.cache_metadata_path.write_bytes(b'{"backend":"previous"}')
    monkeypatch.setattr(single_track.settings, "transcript_dir", store.base_dir)
    monkeypatch.setattr(audio.Audio, "prepare_audio_for_transcription", _fake_prepare)
    monkeypatch.setattr(audio, "MODEL_MAX_BYTES", 6)
    monkeypatch.setattr(
        MLX, "transcribe", lambda *a: pytest.fail("refused model reached inference")
    )
    with pytest.raises(RuntimeError, match="[Mm]odel.*byte"):
        single_track.transcribe(store.url, backend)
    assert previous.read_bytes() == b"previous transcript"
    assert store.cache_metadata_path.read_bytes() == b'{"backend":"previous"}'


def test_enumeration_consumes_deadline(tmp_path, monkeypatch):
    for index in range(10):
        (tmp_path / str(index)).mkdir()
    ticks = iter(range(100))
    monkeypatch.setattr(audio, "MODEL_HASH_TIMEOUT_SECONDS", 4)
    monkeypatch.setattr(audio, "monotonic", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="[Mm]odel.*elapsed"):
        single_track.local_model_identity(tmp_path)
