import atexit
import os
import tempfile

import pytest

# Settings are constructed at import time; use only synthetic configuration.
for _key in (
    "GROQ_API_KEY",
    "VOXHELM_API_BASE",
    "VOXHELM_API_KEY",
    "TRANSCRIPT_MODEL_NAME",
    "TRANSCRIPT_LANGUAGE",
    "TRANSCRIPT_PROMPT",
    "TRANSCRIPT_HTTP_READ_TIMEOUT",
    "TRANSCRIPT_SOURCE_MAX_BYTES",
    "TRANSCRIPT_SOURCE_TOTAL_TIMEOUT",
    "TRANSCRIPT_MEDIA_TIMEOUT",
    "TRANSCRIPT_LOCAL_TRANSCRIPTION_TIMEOUT",
):
    os.environ.pop(_key, None)
# Never open the real transcript store.
_test_home = tempfile.TemporaryDirectory(prefix="transcript-tests-")
atexit.register(_test_home.cleanup)
os.environ["TRANSCRIPT_HOME"] = _test_home.name
os.environ["TRANSCRIPT_DIR"] = os.path.join(_test_home.name, "transcripts")


@pytest.fixture
def audio(tmp_path):
    from podcast_transcript.single_track import Audio

    url = "https://example.com/test.mp3"
    return Audio(base_dir=tmp_path, url=url)


@pytest.fixture
def fake_mlx(tmp_path, monkeypatch):
    """Install a synthetic ``mlx_whisper`` module for the parent and its worker."""
    import sys
    import textwrap

    directory = tmp_path / "fake-mlx"

    def install(source: str):
        directory.mkdir(exist_ok=True)
        (directory / "mlx_whisper.py").write_text(textwrap.dedent(source))
        monkeypatch.syspath_prepend(str(directory))
        existing = os.environ.get("PYTHONPATH")
        monkeypatch.setenv(
            "PYTHONPATH",
            str(directory) + (os.pathsep + existing if existing else ""),
        )
        monkeypatch.delitem(sys.modules, "mlx_whisper", raising=False)
        return directory

    return install
