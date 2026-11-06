import atexit
import os
import tempfile

import pytest

# Settings are constructed at import time; never open the real transcript store.
_test_home = tempfile.TemporaryDirectory(prefix="transcript-tests-")
atexit.register(_test_home.cleanup)
os.environ["TRANSCRIPT_HOME"] = _test_home.name
os.environ["TRANSCRIPT_DIR"] = os.path.join(_test_home.name, "transcripts")


@pytest.fixture
def audio(tmp_path):
    from podcast_transcript.single_track import Audio

    url = "https://example.com/test.mp3"
    return Audio(base_dir=tmp_path, url=url)
