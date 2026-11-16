"""Exercise the installed entry point without starting media or inference."""

import os
import shutil
import subprocess

import pytest


@pytest.mark.parametrize(
    "arguments, code, text",
    [
        (["--help"], 0, "--backend"),
        (["--processors", "0", "synthetic.mp3"], 2, "positive integer"),
    ],
)
def test_installed_cli(tmp_path, arguments, code, text):
    executable = shutil.which("transcribe")
    assert executable is not None, "install the project's CLI before testing"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("TRANSCRIPT_", "GROQ_", "VOXHELM_"))
    }
    environment["TRANSCRIPT_HOME"] = str(tmp_path / "home")
    environment["TRANSCRIPT_DIR"] = str(tmp_path / "cache")
    result = subprocess.run(
        [executable, *arguments],
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == code
    assert text in result.stdout + result.stderr
    assert not list((tmp_path / "cache").iterdir())
