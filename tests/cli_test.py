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


@pytest.mark.parametrize(
    "backend_name", [None, "whisper-cpp", "groq", "mlx", "voxhelm"]
)
@pytest.mark.parametrize("custom", [False, True])
def test_cli_successful_backend_dispatch(
    tmp_path, monkeypatch, capsys, backend_name, custom
):
    from podcast_transcript import backends

    # Use real settings/factories with invented values; native MLX is never imported.
    monkeypatch.setenv("TRANSCRIPT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path / "out"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    if backend_name == "groq":
        monkeypatch.setenv("GROQ_API_KEY", "invented-groq-key")
    monkeypatch.setenv("VOXHELM_API_BASE", "https://voxhelm.example.test/v1")
    monkeypatch.setenv("VOXHELM_API_KEY", "invented-voxhelm-key")
    for key in ("TRANSCRIPT_MODEL_NAME", "TRANSCRIPT_LANGUAGE", "TRANSCRIPT_PROMPT"):
        monkeypatch.delenv(key, raising=False)
    if custom:
        monkeypatch.setenv("TRANSCRIPT_MODEL_NAME", "whisper-large-v3-turbo")
        monkeypatch.setenv("TRANSCRIPT_LANGUAGE", "de")
        monkeypatch.setenv("TRANSCRIPT_PROMPT", "Synthetic vocabulary")
    settings = Settings()
    monkeypatch.setattr(podcast_transcript, "settings", settings)
    monkeypatch.setattr(backends, "settings", settings)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    source = tmp_path / "local audio.wav"
    source.write_bytes(b"synthetic source")
    args = ["transcribe", str(source)]
    if backend_name is not None:
        args.extend(["--backend", backend_name])
    if custom:
        args.extend(["--processors", "7"])
    monkeypatch.setattr(sys, "argv", args)
    selected = backend_name or "whisper-cpp"
    expected_types = {
        "whisper-cpp": backends.WhisperCpp,
        "groq": backends.Groq,
        "mlx": backends.MLX,
        "voxhelm": backends.Voxhelm,
    }
    default_models = {
        "whisper-cpp": "ggml-large-v3.bin",
        "groq": "whisper-large-v3",
        "mlx": "mlx-community/whisper-large-v3-mlx",
        "voxhelm": "gpt-4o-mini-transcribe",
    }
    paths = {
        name: tmp_path / filename
        for name, filename in [
            ("DOTe", "out.dote.json"),
            ("podlove", "out.podlove.json"),
            ("WebVTT", "out.webvtt"),
            ("plain text", "out.txt"),
        ]
    }
    calls = []

    def fake_transcribe(url, backend):
        calls.append(url)
        assert url == str(source)
        assert type(backend) is expected_types[selected]
        assert backend.model_name == (
            "whisper-large-v3-turbo" if custom else default_models[selected]
        )
        assert backend.language == ("de" if custom else "en")
        assert backend.prompt == (
            "Synthetic vocabulary" if custom else "podcast-transcript"
        )
        if selected == "whisper-cpp":
            assert backend.processors == (7 if custom else 4)
            assert (
                backend.model_path
                == settings.whisper_cpp_models_dir / backend.model_name
            )
            assert settings.groq_api_key is None
        elif selected == "groq":
            assert backend.api_key == "invented-groq-key"
        elif selected == "voxhelm":
            assert backend.api_base == "https://voxhelm.example.test/v1"
            assert backend.api_key == "invented-voxhelm-key"
        else:
            assert backend.word_timestamps is False
            assert settings.groq_api_key is None
        for path in paths.values():
            path.write_text("synthetic output")
        return paths

    monkeypatch.setattr(podcast_transcript, "transcribe", fake_transcribe)
    with pytest.raises(SystemExit) as error:
        podcast_transcript.transcribe_cli()
    assert error.value.code == 0
    assert calls == [str(source)]
    output = capsys.readouterr().out
    assert "Transcription complete!" in output
    for name, path in paths.items():
        assert name in output
        assert "".join(str(path).split()) in "".join(output.split())


def test_cli_pipeline_failure_exits_one(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["transcribe", "synthetic.wav"])
    calls = []

    def fail(url, backend):
        calls.append(url)
        raise RuntimeError("synthetic pipeline failure")

    monkeypatch.setattr(podcast_transcript, "transcribe", fail)
    with pytest.raises(SystemExit) as error:
        podcast_transcript.transcribe_cli()
    assert error.value.code == 1
    assert calls == ["synthetic.wav"]
    output = capsys.readouterr().out
    assert "synthetic pipeline failure" in output
    assert "Transcription complete" not in output


def test_cli_help_describes_local_and_http_audio(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["transcribe", "--help"])
    with pytest.raises(SystemExit) as error:
        podcast_transcript.transcribe_cli()
    assert error.value.code == 0
    output = " ".join(capsys.readouterr().out.lower().split())
    assert "local file path" in output
    assert "http(s) url" in output
    assert "source" in output


@pytest.mark.parametrize(
    "args,code", [(["--help"], 0), ([], 2), (["--backend", "unknown", "x.wav"], 2)]
)
def test_installed_entrypoint_usage_without_operational_side_effects(
    tmp_path, args, code
):
    import json
    import os
    import subprocess

    home = tmp_path / "unused home"
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    env = dict(
        os.environ,
        HOME=str(tmp_path),
        PATH="",
        TRANSCRIPT_HOME=str(home),
        TRANSCRIPT_DIR=str(blocked / "out"),
    )
    script = (
        "import importlib.metadata, sys; "
        f"sys.argv = ['transcribe'] + {json.dumps(args)}; "
        "entry = next(e for e in importlib.metadata.distribution('podcast-transcript').entry_points "
        "if e.name == 'transcribe'); entry.load()()"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == code, result.stderr + result.stdout
    assert "usage:" in result.stdout + result.stderr
    assert "ffmpeg" not in result.stderr + result.stdout
    assert not home.exists()
    assert blocked.read_text() == "not a directory"


def test_actual_transcription_requires_ffmpeg(tmp_path):
    import os
    import subprocess

    home = tmp_path / "unused home"
    env = dict(
        os.environ, PATH="", TRANSCRIPT_HOME=str(home), TRANSCRIPT_DIR=str(home / "out")
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from podcast_transcript import transcribe_cli; transcribe_cli()",
            "x.wav",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1
    assert "ffmpeg" in result.stdout
    assert not home.exists()


@pytest.mark.parametrize("model_name", [None, "ggml-base.en.bin"])
def test_local_runbook_model_selection(tmp_path, monkeypatch, model_name):
    import argparse
    from pathlib import Path
    from podcast_transcript import backends, whisper_cpp_from_settings_and_args
    from podcast_transcript.safe_io import run_checked

    monkeypatch.setenv("TRANSCRIPT_HOME", str(tmp_path / "local home"))
    monkeypatch.delenv("TRANSCRIPT_MODEL_NAME", raising=False)
    if model_name:
        monkeypatch.setenv("TRANSCRIPT_MODEL_NAME", model_name)
    settings = Settings()
    monkeypatch.setattr(backends, "settings", settings)
    expected = settings.whisper_cpp_models_dir / (model_name or "ggml-large-v3.bin")
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"synthetic model, no inference")
    backend = whisper_cpp_from_settings_and_args(
        settings, argparse.Namespace(processors=2)
    )
    assert backend.model_path == expected
    # Fake CLI launched via Python so executable/shebang rules are platform independent.
    fake_cli = tmp_path / "fake_whisper.py"
    fake_cli.write_text(
        "import sys\nfrom pathlib import Path\n"
        "model = Path(sys.argv[sys.argv.index('-m') + 1])\n"
        "if not model.is_file():\n"
        "    print('Missing local model', file=sys.stderr)\n    sys.exit(2)\n"
        "out = sys.argv[sys.argv.index('-of') + 1]\n"
        "Path(out + '.json').write_text('{}')\n"
    )

    def fake_command(cmd, **kwargs):
        assert cmd[0] == "whisper-cli"
        return run_checked([sys.executable, str(fake_cli), *cmd[1:]], **kwargs)

    monkeypatch.setattr(backends, "run_checked", fake_command)
    source = tmp_path / "synthetic episode.wav"
    source.write_bytes(b"synthetic audio")
    out = tmp_path / "output"
    backend.run_whisper_cli(source, out)
    assert Path(str(out) + ".json").exists()
    expected.unlink()
    with pytest.raises(RuntimeError, match="Missing local model"):
        backend.run_whisper_cli(source, out)


def test_missing_local_executable_diagnostic(monkeypatch, tmp_path):
    from podcast_transcript import backends

    def missing(*args, **kwargs):
        raise FileNotFoundError("whisper-cli not found")

    monkeypatch.setattr("podcast_transcript.safe_io.subprocess.Popen", missing)
    with pytest.raises(FileNotFoundError, match="whisper-cli"):
        backends.WhisperCpp().run_whisper_cli(
            tmp_path / "synthetic.wav", tmp_path / "output"
        )
