import pytest

from podcast_transcript.config import Settings, parse_env_line


def test_settings_initialization(tmp_path, monkeypatch):
    # Mock environment variables
    monkeypatch.setenv("TRANSCRIPT_HOME", str(tmp_path))
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path))
    monkeypatch.setenv("GROQ_API_KEY", "test_api_key")
    monkeypatch.setenv("TRANSCRIPT_PROMPT", "a different prompt")
    monkeypatch.setenv("VOXHELM_API_BASE", "https://voxhelm.example")
    monkeypatch.setenv("VOXHELM_API_KEY", "voxhelm-token")

    # Initialize settings
    settings = Settings()

    # Assertions
    assert settings.transcript_dir == tmp_path
    assert settings.groq_api_key == "test_api_key"
    assert settings.transcript_prompt == "a different prompt"
    assert settings.voxhelm_api_base == "https://voxhelm.example"
    assert settings.voxhelm_api_key == "voxhelm-token"
    assert settings.transcript_dir.exists()


def test_env_file_value_containing_equals_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIPT_HOME", str(tmp_path))
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path / "transcripts"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("VOXHELM_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "# a comment line\n"
        "\n"
        "GROQ_API_KEY=gsk_abc==\n"
        'VOXHELM_API_KEY="tok=en=="\n'
        "  # GROQ_API_KEY=commented-out\n"
        "not a setting\n"
    )

    settings = Settings()

    assert settings.groq_api_key == "gsk_abc=="
    assert settings.voxhelm_api_key == "tok=en=="


@pytest.mark.parametrize(
    "line,expected",
    [
        ("KEY=value\n", ("KEY", "value")),
        ("KEY=a=b=c", ("KEY", "a=b=c")),
        ("KEY=", ("KEY", "")),
        (" KEY = value ", ("KEY", "value")),
        ('KEY="quoted = value"', ("KEY", "quoted = value")),
        ("KEY='single'", ("KEY", "single")),
        ('KEY="unbalanced', ("KEY", '"unbalanced')),
        ("export KEY=value", ("KEY", "value")),
        ("KEY=value#notacomment", ("KEY", "value#notacomment")),
        ("# KEY=value", None),
        ("   ", None),
        ("no equals sign", None),
        ("=value", None),
    ],
)
def test_parse_env_line(line, expected):
    assert parse_env_line(line) == expected


def test_http_read_timeout_defaults_to_30_minutes(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIPT_HOME", str(tmp_path))
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path / "transcripts"))
    monkeypatch.delenv("TRANSCRIPT_HTTP_READ_TIMEOUT", raising=False)

    assert Settings().http_read_timeout_seconds == 1800.0


def test_http_read_timeout_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIPT_HOME", str(tmp_path))
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setenv("TRANSCRIPT_HTTP_READ_TIMEOUT", "120.5")

    assert Settings().http_read_timeout_seconds == 120.5


def test_http_read_timeout_from_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIPT_HOME", str(tmp_path))
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path / "transcripts"))
    monkeypatch.delenv("TRANSCRIPT_HTTP_READ_TIMEOUT", raising=False)
    (tmp_path / ".env").write_text("TRANSCRIPT_HTTP_READ_TIMEOUT=600\n")

    assert Settings().http_read_timeout_seconds == 600.0


@pytest.mark.parametrize("value", ["", "soon", "0", "-5", "inf", "nan"])
def test_http_read_timeout_rejects_invalid_values(tmp_path, monkeypatch, value):
    monkeypatch.setenv("TRANSCRIPT_HOME", str(tmp_path))
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setenv("TRANSCRIPT_HTTP_READ_TIMEOUT", value)

    with pytest.raises(ValueError, match="TRANSCRIPT_HTTP_READ_TIMEOUT"):
        Settings().http_read_timeout_seconds


@pytest.mark.parametrize("source", ["default", "env", "file"])
def test_source_download_budget_configuration(tmp_path, monkeypatch, source):
    monkeypatch.setenv("TRANSCRIPT_HOME", str(tmp_path))
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setattr(
        "podcast_transcript.config.shutil.which", lambda name: "/synthetic/ffmpeg"
    )
    monkeypatch.delenv("TRANSCRIPT_SOURCE_MAX_BYTES", raising=False)
    monkeypatch.delenv("TRANSCRIPT_SOURCE_TOTAL_TIMEOUT", raising=False)
    if source == "env":
        monkeypatch.setenv("TRANSCRIPT_SOURCE_MAX_BYTES", "123456")
        monkeypatch.setenv("TRANSCRIPT_SOURCE_TOTAL_TIMEOUT", "123.5")
    elif source == "file":
        (tmp_path / ".env").write_text(
            "TRANSCRIPT_SOURCE_MAX_BYTES=123456\nTRANSCRIPT_SOURCE_TOTAL_TIMEOUT=123.5\n"
        )
    settings = Settings()
    assert settings.source_max_bytes == (2 * 1024**3 if source == "default" else 123456)
    assert settings.source_total_timeout_seconds == (
        3600.0 if source == "default" else 123.5
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("MAX_BYTES", "0"),
        ("MAX_BYTES", "-1"),
        ("MAX_BYTES", "1.5"),
        ("MAX_BYTES", "nan"),
        ("TOTAL_TIMEOUT", "0"),
        ("TOTAL_TIMEOUT", "-1"),
        ("TOTAL_TIMEOUT", "NaN"),
        ("TOTAL_TIMEOUT", "inf"),
        ("TOTAL_TIMEOUT", "later"),
    ],
)
def test_source_download_budget_rejects_invalid_settings(
    tmp_path, monkeypatch, name, value
):
    monkeypatch.setenv("TRANSCRIPT_HOME", str(tmp_path))
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setattr(
        "podcast_transcript.config.shutil.which", lambda name: "/synthetic/ffmpeg"
    )
    monkeypatch.setenv(f"TRANSCRIPT_SOURCE_{name}", value)
    settings = Settings()
    with pytest.raises(ValueError, match=f"TRANSCRIPT_SOURCE_{name}"):
        if name == "MAX_BYTES":
            settings.source_max_bytes
        else:
            settings.source_total_timeout_seconds


def test_dotenv_only_assigns_supported_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIPT_HOME", str(tmp_path))
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setenv("GROQ_API_KEY", "environment-token")
    monkeypatch.setenv("VOXHELM_API_KEY", "environment-voxhelm")
    monkeypatch.setenv("TRANSCRIPT_LANGUAGE", "de")
    (tmp_path / ".env").write_text(
        "READ_ENV_VARS=shadow\nCONSOLE=shadow\nUNRELATED_PROJECT=ignored\n"
        "__DICT__=shadow\nTRANSCRIPT_HOME=wrong\nTRANSCRIPT_DIR=wrong\n"
        "GROQ_API_KEY=file-token\nVOXHELM_API_KEY=file-voxhelm\n"
        "TRANSCRIPT_LANGUAGE=en\nTRANSCRIPT_PROMPT=synthetic prompt\n"
    )
    settings = Settings()
    assert callable(settings.read_env_vars)
    assert callable(settings.console.print)
    assert not hasattr(settings, "unrelated_project")
    assert settings.transcript_home == tmp_path
    assert settings.transcript_dir == tmp_path / "transcripts"
    assert settings.groq_api_key == "environment-token"
    assert settings.voxhelm_api_key == "environment-voxhelm"
    assert settings.transcript_language == "de"
    assert settings.transcript_prompt == "synthetic prompt"


@pytest.mark.parametrize("kind", ["oversize", "invalid-utf8", "growth", "fifo", "link"])
def test_dotenv_refuses_unsafe_input(tmp_path, monkeypatch, kind):
    import os
    from podcast_transcript import config

    monkeypatch.setenv("TRANSCRIPT_HOME", str(tmp_path))
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path / "transcripts"))
    env_file = tmp_path / ".env"
    secret = b"GROQ_API_KEY=synthetic-secret\n"
    # Small budget exercises both huge physical lines and growth after fstat.
    monkeypatch.setattr(config, "DOTENV_MAX_BYTES", 64, raising=False)
    if kind == "fifo":
        os.mkfifo(env_file)
        # Keep the baseline test from blocking forever on its plain open.
        real_open = type(env_file).open

        def no_block(path, *args, **kwargs):
            if path == env_file:
                pytest.fail("dotenv used blocking Path.open on FIFO")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(type(env_file), "open", no_block)
    elif kind == "link":
        target = tmp_path / "secret"
        target.write_bytes(secret)
        env_file.symlink_to(target)
    else:
        env_file.write_bytes(
            secret
            + (
                b"x" * 80
                if kind == "oversize"
                else b"\xff"
                if kind == "invalid-utf8"
                else b""
            )
        )
        if kind == "growth":
            read = os.read

            def growing(fd, count):
                with env_file.open("ab") as file:
                    file.write(b"x" * 80)
                return read(fd, count)

            monkeypatch.setattr(os, "read", growing)
    settings = Settings.__new__(Settings)
    with pytest.raises(RuntimeError, match="dotenv") as error:
        settings.read_env_file(env_file)
    assert "synthetic-secret" not in str(error.value)
    assert not hasattr(settings, "groq_api_key")
    assert not (tmp_path / "transcripts").exists()
