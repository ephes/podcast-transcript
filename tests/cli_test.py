import importlib.util
import sys

import pytest

import podcast_transcript


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
