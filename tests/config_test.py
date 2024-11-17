from podcast_transcript.config import Settings


def test_settings_initialization(tmp_path, monkeypatch):
    # Mock environment variables
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path))
    monkeypatch.setenv("GROQ_API_KEY", "test_api_key")

    # Initialize settings
    settings = Settings()

    # Assertions
    assert settings.transcript_dir == tmp_path
    assert settings.groq_api_key == "test_api_key"
    assert settings.transcript_dir.exists()
