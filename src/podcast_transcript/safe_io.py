"""
Helpers that make pipeline steps fail closed.

Every step in the pipeline caches its result by checking whether the output
file exists. That is only safe if an output file appears exclusively after the
step succeeded. These helpers write to a temporary path next to the final one
and rename it into place only on success, and they turn failed subprocesses
into exceptions that carry the tool's stderr.
"""

import os
import subprocess
import uuid

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

STDERR_TAIL_LINES = 20


def partial_path_for(path: Path) -> Path:
    """
    Return a unique, hidden temporary path next to ``path``.

    The suffix is kept so tools like ffmpeg can still infer the output format,
    and the leading dot keeps the file out of ``chunk_*.mp3`` style globs.
    """
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.partial{path.suffix}")


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


@contextmanager
def atomic_output(path: Path) -> Iterator[Path]:
    """
    Yield a temporary path to write to and move it to ``path`` on success.

    If the block raises, the temporary file is removed and ``path`` is left
    untouched. If the block finishes without creating the temporary file, a
    ``RuntimeError`` is raised instead of silently caching nothing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = partial_path_for(path)
    try:
        yield tmp_path
        if not tmp_path.exists():
            raise RuntimeError(f"Expected output was not created: {path}")
        os.replace(tmp_path, path)
    finally:
        _unlink_quietly(tmp_path)


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so that a crash never leaves a partial file."""
    with atomic_output(path) as tmp_path:
        tmp_path.write_text(text)


def run_checked(cmd: list[str], *, description: str) -> None:
    """
    Run ``cmd`` and raise ``RuntimeError`` with the stderr tail on failure.
    """
    try:
        subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        tail = "\n".join(stderr.splitlines()[-STDERR_TAIL_LINES:])
        message = f"{description} failed with exit code {exc.returncode}"
        if tail:
            message = f"{message}:\n{tail}"
        raise RuntimeError(message) from exc
