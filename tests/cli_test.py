import importlib.util
import sys

import pytest

import podcast_transcript
from podcast_transcript import voxhelm_from_settings
from podcast_transcript.config import Settings


def test_cli_mlx_missing_deps_shows_actionable_message(monkeypatch, capsys):
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "transcribe",
            "--backend",
            "mlx",
            "https://example.com/audio.mp3",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        podcast_transcript.transcribe_cli()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    normalized = " ".join(combined.split())
    assert "podcast-transcript[mlx]" in normalized
    assert "uv sync" in normalized
    assert "--extra" in normalized


def test_voxhelm_from_settings_requires_api_base(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIPT_HOME", str(tmp_path))
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path))
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    monkeypatch.delenv("VOXHELM_API_BASE", raising=False)
    monkeypatch.setenv("VOXHELM_API_KEY", "secret-token")

    settings = Settings()

    with pytest.raises(ValueError, match="VOXHELM_API_BASE is not set"):
        voxhelm_from_settings(settings)


def test_voxhelm_from_settings_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIPT_HOME", str(tmp_path))
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path))
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    monkeypatch.setenv("VOXHELM_API_BASE", "https://voxhelm.example")
    monkeypatch.delenv("VOXHELM_API_KEY", raising=False)

    settings = Settings()

    with pytest.raises(ValueError, match="VOXHELM_API_KEY is not set"):
        voxhelm_from_settings(settings)
