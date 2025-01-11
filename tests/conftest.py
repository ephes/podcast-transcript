import pytest

from podcast_transcript.single_track import AudioUrl


@pytest.fixture
def audio_url(tmp_path):
    url = "https://example.com/test.mp3"
    return AudioUrl(base_dir=tmp_path, url=url)
