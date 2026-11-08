"""Synthetic inference races through the real episode pipeline."""

import json

import pytest

from podcast_transcript import audio, backends, safe_io, single_track


@pytest.mark.parametrize("backend_name", ["cpp", "mlx-file", "mlx-dir"])
@pytest.mark.parametrize("failure", ["replace", "interrupt"])
@pytest.mark.parametrize("changed_chunk", [0, 1])
def test_model_generation_never_certifies_unchecked_chunks(
    monkeypatch, tmp_path, backend_name, failure, changed_chunk
):
    monkeypatch.setattr(single_track.settings, "transcript_dir", tmp_path / "cache")
    model = tmp_path / "model.bin"
    if backend_name == "mlx-dir":
        model = tmp_path / "model" / "weights.bin"
        model.parent.mkdir()
    model.write_bytes(b"generation-one")
    backend = (
        backends.WhisperCpp(model_name=str(model))
        if backend_name == "cpp"
        else backends.MLX(
            model_name=str(model.parent if backend_name == "mlx-dir" else model)
        )
    )

    def prepare(self):
        self.episode_chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks = [self.episode_chunks_dir / f"chunk_{i:03}.mp3" for i in range(2)]
        for chunk in chunks:
            chunk.write_bytes(b"synthetic audio")
        return chunks

    monkeypatch.setattr(audio.Audio, "prepare_audio_for_transcription", prepare)
    monkeypatch.setattr(single_track, "get_audio_duration", lambda _: 1.0)
    source = "https://example.test/episode.mp3"
    store = audio.Audio(base_dir=single_track.settings.transcript_dir, url=source)
    calls = []
    markers = []
    fail = True

    def infer(self, chunk, target):
        calls.append(model.read_bytes())
        safe_io.write_text_atomic(target, json.dumps({"segments": []}))
        if fail:
            markers.append(store.cache_metadata_path.exists())
            if chunk.name == f"chunk_{changed_chunk:03}.mp3":
                model.write_bytes(b"generation-two")
                if failure == "interrupt":
                    raise KeyboardInterrupt

    monkeypatch.setattr(type(backend), "transcribe", infer)
    expected = KeyboardInterrupt if failure == "interrupt" else RuntimeError
    with pytest.raises(
        expected, match=None if failure == "interrupt" else "model changed"
    ):
        single_track.transcribe(source, backend)
    assert markers == [False] * (changed_chunk + 1)
    assert calls == [b"generation-one"] * (changed_chunk + 1)
    assert not store.cache_metadata_path.exists()
    assert len(list(store.episode_chunks_dir.glob("*.json"))) == changed_chunk
    assert not store.output_path(".dote.json").exists()

    # Both the changed model and the restored identity must start fresh.
    fail = False
    single_track.transcribe(source, backend)
    assert calls == [b"generation-one"] * (changed_chunk + 1) + [b"generation-two"] * 2
    single_track.transcribe(source, backend)
    assert len(calls) == changed_chunk + 3
    model.write_bytes(b"generation-one")
    single_track.transcribe(source, backend)
    assert calls[-2:] == [b"generation-one", b"generation-one"]
    assert len(calls) == changed_chunk + 5
