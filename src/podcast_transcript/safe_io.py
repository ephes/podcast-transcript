"""
Helpers that make pipeline steps fail closed.

Every step in the pipeline caches its result by checking whether the output
file exists. That is only safe if an output file appears exclusively after the
step succeeded. These helpers write to a temporary path next to the final one
and rename it into place only on success, and they turn failed subprocesses
into exceptions that carry the tool's stderr.
"""

import math
import os
import signal
import subprocess
import tempfile
import uuid

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .config import settings

STDERR_TAIL_LINES = 20
STDERR_TAIL_CHARS = 2000


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


def _kill_owned_tree(process: subprocess.Popen) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        # Windows has no killpg; taskkill /T is its process-tree API.
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        finally:
            if process.poll() is None:
                process.kill()
    process.wait()


def run_checked(
    cmd: list[str], *, description: str, timeout: float | None = None
) -> None:
    """Run an owned process group with a finite budget and bounded diagnostics."""
    if timeout is None:
        timeout = settings.media_timeout_seconds
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Command timeout must be positive finite seconds")
    # A file keeps stderr capture out of RAM and cannot leave abort cleanup
    # waiting for EOF from a descendant that escaped the owned process group.
    with tempfile.TemporaryFile() as error_log:
        with subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=error_log,
            start_new_session=os.name == "posix",
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        ) as process:
            try:
                process.wait(timeout=timeout)
            except BaseException as exc:
                _kill_owned_tree(process)
                if isinstance(exc, subprocess.TimeoutExpired):
                    raise RuntimeError(
                        f"{description} timed out after {timeout:g} seconds"
                    ) from exc
                raise
            if process.returncode:
                size = error_log.seek(0, os.SEEK_END)
                error_log.seek(max(0, size - STDERR_TAIL_CHARS))
                stderr = error_log.read().decode("utf-8", errors="replace")
                tail = "\n".join(stderr.strip().splitlines()[-STDERR_TAIL_LINES:])
                message = f"{description} failed with exit code {process.returncode}"
                if tail:
                    message = f"{message}:\n{tail}"
                raise RuntimeError(message)
