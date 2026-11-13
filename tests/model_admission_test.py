"""One-open model admission with synthetic inode swaps and bounded hashing."""

import os
from pathlib import Path

import pytest

from podcast_transcript import audio, backends, single_track
from single_track_test import _fake_prepare


@pytest.mark.parametrize("replacement", ["fifo", "link"])
def test_model_swapped_at_open_refused_before_inference(
    tmp_path, monkeypatch, replacement
):
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    backend = backends.WhisperCpp(model_name=str(model))
    monkeypatch.setattr(single_track.settings, "transcript_dir", tmp_path / "cache")
    monkeypatch.setattr(audio.Audio, "prepare_audio_for_transcription", _fake_prepare)
    store = audio.Audio(
        base_dir=tmp_path / "cache", url="https://example.test/audio.mp3"
    )
    store.podcast_dir.mkdir(parents=True)
    previous = store.output_path(".txt")
    previous.write_bytes(b"previous transcript")
    store.cache_metadata_path.write_bytes(b'{"backend":"previous"}')
    external = tmp_path / "external"
    external.write_bytes(b"external bytes")
    real_open = os.open
    calls = []

    def opened(path, flags, *args, **kwargs):
        if Path(path) == model:
            calls.append(flags)
            model.unlink()
            if replacement == "fifo":
                os.mkfifo(model)
            else:
                model.symlink_to(external)
        return real_open(path, flags, *args, **kwargs)

    # The old unrestricted Path.open bypasses this descriptor admission seam.
    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(
        backends.WhisperCpp,
        "transcribe",
        lambda *a: pytest.fail("inference started before model admission"),
    )
    with pytest.raises(RuntimeError, match="[Mm]odel"):
        single_track.transcribe(store.url, backend)
    assert len(calls) == 1
    assert calls[0] & os.O_NONBLOCK
    assert previous.read_bytes() == b"previous transcript"
    assert store.cache_metadata_path.read_bytes() == b'{"backend":"previous"}'
    assert external.read_bytes() == b"external bytes"


@pytest.mark.parametrize("kind", ["fifo", "link"])
def test_existing_special_model_is_not_a_remote_id(tmp_path, kind):
    path = tmp_path / "model"
    if kind == "fifo":
        os.mkfifo(path)
    else:
        path.symlink_to(tmp_path / "absent")
    with pytest.raises(RuntimeError, match="[Mm]odel"):
        single_track.local_model_identity(path)


@pytest.mark.parametrize("budget", ["size", "growth", "elapsed"])
def test_model_hash_budgets(tmp_path, monkeypatch, budget):
    path = tmp_path / "model"
    path.write_bytes(b"model")
    if budget == "size":
        monkeypatch.setattr(audio, "MODEL_MAX_BYTES", 4, raising=False)
    elif budget == "growth":
        monkeypatch.setattr(audio, "MODEL_MAX_BYTES", 6, raising=False)
        read = os.read

        def growing(fd, size):
            with path.open("ab") as file:
                file.write(b"growth")
            return read(fd, size)

        monkeypatch.setattr(os, "read", growing)
    else:
        clock = iter([0, 0, 3601])
        monkeypatch.setattr(audio, "monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="[Mm]odel"):
        audio.Audio._content_fingerprint(path, source_budget=False)
