"""Synthetic model identity and private storage contracts."""

import json
import os
from pathlib import Path

import pytest

from podcast_transcript import audio, backends, safe_io, single_track
from single_track_test import _fake_prepare


@pytest.mark.parametrize("backend_name", ["cpp", "mlx-file", "mlx-dir"])
def test_model_bytes_control_transcript_cache_only(monkeypatch, tmp_path, backend_name):
    monkeypatch.setattr(single_track.settings, "transcript_dir", tmp_path / "cache")
    monkeypatch.setattr(audio.Audio, "prepare_audio_for_transcription", _fake_prepare)
    model = tmp_path / "model.bin"
    if backend_name == "mlx-dir":
        directory = tmp_path / "model-dir"
        directory.mkdir()
        model = directory / "weights.bin"
    model.write_bytes(b"model-v1")
    if backend_name == "cpp":
        backend = backends.WhisperCpp(model_name=str(model))
    else:
        backend = backends.MLX(
            model_name=str(model.parent if backend_name == "mlx-dir" else model)
        )
    calls = []

    def infer(self, chunk, target):
        calls.append(chunk)
        safe_io.write_text_atomic(target, json.dumps({"segments": []}))

    monkeypatch.setattr(type(backend), "transcribe", infer)
    url = "https://example.test/episode.mp3"
    single_track.transcribe(url, backend)
    store = audio.Audio(base_dir=single_track.settings.transcript_dir, url=url)
    prepared = store.episode_chunks_dir / "chunk_000.mp3"
    before = prepared.read_bytes()
    first = json.loads(store.cache_metadata_path.read_text())
    single_track.transcribe(url, backend)
    assert len(calls) == 1
    legacy = dict(first)
    legacy.pop("local_model_sha256")
    store.cache_metadata_path.write_text(json.dumps(legacy))
    single_track.transcribe(url, backend)
    single_track.transcribe(url, backend)
    assert len(calls) == 2
    model.write_bytes(b"model-v2")
    single_track.transcribe(url, backend)
    assert len(calls) == 3
    assert first != json.loads(store.cache_metadata_path.read_text())
    assert prepared.read_bytes() == before


def test_remote_model_ids_and_keys_do_not_hash_local_files(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    model = Path("whisper-large-v3")
    model.write_bytes(b"unrelated")
    backend = backends.Groq(
        api_key="synthetic-key",
        model_name=None,
        language="en",
        prompt="synthetic-prompt",
    )
    before = single_track.backend_cache_metadata(backend)
    model.write_bytes(b"changed")
    backend.api_key = "rotated-key"
    assert single_track.backend_cache_metadata(backend) == before


def test_new_text_files_request_private_creation_mode(monkeypatch, tmp_path):
    real_open = os.open
    creations = []

    def opened(path, flags, mode=0o777, **kw):
        if flags & os.O_CREAT:
            creations.append(mode)
        return real_open(path, flags, mode, **kw)

    monkeypatch.setattr(os, "open", opened)
    target = tmp_path / "cache.json"
    safe_io.write_text_atomic(target, "private-synthetic-prompt")
    assert creations and all(mode & 0o077 == 0 for mode in creations)
    assert target.read_text() == "private-synthetic-prompt"


def test_cache_directories_request_private_mode(monkeypatch, tmp_path):
    real_mkdir = Path.mkdir
    modes = []

    def mkdir(path, mode=0o777, parents=False, exist_ok=False):
        if not path.exists():
            modes.append(mode)
        return real_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    store = audio.Audio(base_dir=tmp_path, url="https://example.test/audio.mp3")
    monkeypatch.setattr(
        audio, "download", lambda source, target: target.write_bytes(b"audio")
    )
    store.make_sure_audio_file_exists()
    assert modes and all(mode & 0o077 == 0 for mode in modes)


def test_cache_change_does_not_log_prompt(capsys, tmp_path):
    store = audio.Audio(base_dir=tmp_path, url="https://example.test/audio.mp3")
    store.ensure_transcript_cache_matches({"prompt": "private-synthetic-prompt"})
    store.ensure_transcript_cache_matches({"prompt": "other-private-prompt"})
    output = capsys.readouterr()
    assert "private" not in output.out + output.err


def test_atomic_publication_private_staging_and_existing_export_mode(
    monkeypatch, tmp_path
):
    import stat

    original_mkdir = os.mkdir
    directories = {}

    def mkdir(path, mode=0o777, **kwargs):
        directories[Path(path)] = mode
        return original_mkdir(path, mode, **kwargs)

    monkeypatch.setattr(os, "mkdir", mkdir)
    target = tmp_path / "export.txt"
    target.write_text("old public export")
    target.chmod(0o640)
    original_mode = stat.S_IMODE(target.stat().st_mode)
    with safe_io.atomic_output(target) as temporary:
        assert temporary.parent != target.parent
        assert directories[temporary.parent] & 0o077 == 0
        temporary.write_text("replacement export")
    assert stat.S_IMODE(target.stat().st_mode) == original_mode
    assert target.read_text() == "replacement export"


def test_split_chunks_are_private_before_promotion(monkeypatch, tmp_path):
    store = audio.Audio(base_dir=tmp_path, url="https://example.test/audio.mp3")
    store.episode_chunks_dir.mkdir(parents=True)
    store.resampled_episode_path.write_bytes(b"resampled")
    monkeypatch.setattr(audio, "get_audio_duration", lambda path: 1)

    def split(cmd, **kwargs):
        Path(cmd[-1].replace("%03d", "000")).write_bytes(b"chunk")

    monkeypatch.setattr(audio, "run_checked", split)
    chmods = {}
    chmod = Path.chmod

    def changed(path, mode, **kwargs):
        chmods[path] = mode
        return chmod(path, mode, **kwargs)

    monkeypatch.setattr(Path, "chmod", changed)
    replace = os.replace

    def promote(source, target):
        assert chmods.get(source) == 0o600
        return replace(source, target)

    monkeypatch.setattr(os, "replace", promote)
    assert store._split_resampled_episode() == ["chunk_000.mp3"]


def test_local_directory_identity_includes_filenames(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "weights.bin"
    weights.write_bytes(b"synthetic")
    before = single_track.local_model_identity(model)
    weights.rename(model / "renamed.bin")
    assert single_track.local_model_identity(model) != before


def test_model_directory_link_policy_without_os_link_privileges(monkeypatch, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "weights.bin"
    weights.write_bytes(b"synthetic-v1")
    # Model public Path link classification; no platform-specific link privilege.
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == weights)
    with pytest.raises(RuntimeError, match="symlinks"):
        single_track.local_model_identity(model)
    directory_link = model / "linked-directory"
    directory_link.mkdir()
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == directory_link)
    with pytest.raises(RuntimeError, match="symlinks"):
        single_track.local_model_identity(model)
