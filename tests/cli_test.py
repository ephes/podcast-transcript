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


@pytest.mark.parametrize("value", ["-1", "0"])
def test_processors_rejected_before_backend_or_pipeline(
    mocker, monkeypatch, capsys, value
):
    factory = mocker.patch.object(
        podcast_transcript, "whisper_cpp_from_settings_and_args"
    )
    pipeline = mocker.patch.object(podcast_transcript, "transcribe")
    monkeypatch.setattr(
        sys, "argv", ["transcribe", "--processors", value, "synthetic.wav"]
    )
    with pytest.raises(SystemExit) as error:
        podcast_transcript.transcribe_cli()
    assert error.value.code == 2
    assert "positive integer" in capsys.readouterr().err
    factory.assert_not_called()
    pipeline.assert_not_called()


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_whisper_constructor_requires_positive_integer(value):
    from podcast_transcript.backends import WhisperCpp

    with pytest.raises(ValueError, match="positive integer"):
        WhisperCpp(processors=value)


@pytest.mark.parametrize("args,expected", [([], 4), (["--processors", "1"], 1)])
def test_processor_count_reaches_whisper_argv(
    mocker, monkeypatch, tmp_path, args, expected
):
    from podcast_transcript import backends

    command = mocker.patch.object(backends, "run_checked")

    def pipeline(source, backend):
        backend.run_whisper_cli(tmp_path / "synthetic.wav", tmp_path / "output")
        return {}

    mocker.patch.object(podcast_transcript, "transcribe", side_effect=pipeline)
    monkeypatch.setattr(sys, "argv", ["transcribe", "synthetic.wav", *args])
    with pytest.raises(SystemExit) as error:
        podcast_transcript.transcribe_cli()
    assert error.value.code == 0
    argv = command.call_args.args[0]
    assert argv[argv.index("-p") + 1] == str(expected)
