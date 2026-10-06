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
