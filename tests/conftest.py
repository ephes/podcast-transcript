import pytest

from podcast_transcript.single_track import Audio


@pytest.fixture
def audio(tmp_path):
    url = "https://example.com/test.mp3"
    return Audio(base_dir=tmp_path, url=url)
