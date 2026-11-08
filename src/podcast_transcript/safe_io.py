"""
Helpers that make pipeline steps fail closed.

Every step in the pipeline caches its result by checking whether the output
file exists. That is only safe if an output file appears exclusively after the
step succeeded. These helpers stage files in a private directory beside the final output
and rename them into place only on success, and they turn failed subprocesses
into exceptions that carry the tool's stderr.
"""

import math
import os
import signal
import stat
import subprocess
import tempfile
import uuid

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .config import private_mkdir, settings

STDERR_TAIL_LINES = 20
STDERR_TAIL_CHARS = 2000


def partial_path_for(path: Path) -> Path:
    """
    Return a unique, hidden temporary path next to ``path``.

    The suffix is kept so tools like ffmpeg can still infer the output format,
    and the leading dot keeps the file out of ``chunk_*.mp3`` style globs.
    """
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.partial{path.suffix}")


@contextmanager
def atomic_output(path: Path) -> Iterator[Path]:
    """
    Yield a temporary path to write to and move it to ``path`` on success.

    If the block raises, the temporary file is removed and ``path`` is left
    untouched. If the block finishes without creating the temporary file, a
    ``RuntimeError`` is raised instead of silently caching nothing.
    """
    private_mkdir(path.parent)
    # Tools may create their output with ambient permissions. A private staging
    # directory protects those bytes before publication, even in a shared parent.
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    with tempfile.TemporaryDirectory(prefix=".output-", dir=path.parent) as directory:
        tmp_path = Path(directory) / partial_path_for(path).name
        yield tmp_path
        if not tmp_path.exists() and not tmp_path.is_symlink():
            raise RuntimeError(f"Expected output was not created: {path}")
        if not tmp_path.is_symlink():
            tmp_path.chmod(mode)
        os.replace(tmp_path, path)


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so that a crash never leaves a partial file."""
    with atomic_output(path) as tmp_path:
        descriptor = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as file:
            file.write(text)


def _kill_owned_tree(process: subprocess.Popen) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Some sandboxes refuse signaling an already-exited leader. Reap
            # that child, but do not suppress a real failure to stop a live one.
            if process.poll() is None:
                raise
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


def capture_checked(
    cmd: list[str],
    *,
    description: str,
    timeout: float,
    stdout_limit: int = 64 * 1024,
    stderr_limit: int = 64 * 1024,
) -> bytes:
    """Capture small tool results, killing/reaping the owned tree on overflow."""
    import threading
    import time

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Command timeout must be positive finite seconds")
    buffers = [bytearray(), bytearray()]
    overflow = threading.Event()
    stop = threading.Event()
    failures: list[BaseException] = []

    def drain(stream, buffer, limit):
        try:
            while not stop.is_set():
                block = stream.read(8192)
                if not block:
                    break
                remaining = limit - len(buffer)
                buffer.extend(block[:remaining])
                if len(block) > remaining:
                    overflow.set()
        except OSError as exc:
            if not stop.is_set():
                failures.append(exc)
        finally:
            stream.close()

    with subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        start_new_session=os.name == "posix",
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    ) as process:
        threads = [
            threading.Thread(target=drain, args=(stream, buffer, limit), daemon=True)
            for stream, buffer, limit in zip(
                (process.stdout, process.stderr), buffers, (stdout_limit, stderr_limit)
            )
        ]
        deadline = time.monotonic() + timeout
        try:
            for thread in threads:
                thread.start()
            # Keep the leader unreaped until pipe EOF. Its PID cannot be reused
            # while cleanup may still need to signal descendants holding a pipe.
            while True:
                if overflow.is_set():
                    raise RuntimeError(f"{description} output exceeded byte limit")
                active = [thread for thread in threads if thread.is_alive()]
                if not active:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(cmd, timeout)
                active[0].join(timeout=min(0.05, remaining))
            if overflow.is_set():
                raise RuntimeError(f"{description} output exceeded byte limit")
            if failures:
                raise RuntimeError(f"{description} output could not be read")
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except BaseException as exc:
            stop.set()
            if process.returncode is None:
                _kill_owned_tree(process)
            if isinstance(exc, subprocess.TimeoutExpired):
                tail = (
                    bytes(buffers[1][-STDERR_TAIL_CHARS:])
                    .decode("utf-8", errors="replace")
                    .strip()
                )
                raise RuntimeError(
                    f"{description} timed out after {timeout:g} seconds: {tail}"
                ) from None
            raise
        finally:
            stop.set()
            for thread in threads:
                if thread.ident is not None:
                    thread.join(timeout=0.2)

    if process.returncode:
        tail = (
            bytes(buffers[1][-STDERR_TAIL_CHARS:])
            .decode("utf-8", errors="replace")
            .strip()
        )
        raise RuntimeError(
            f"{description} failed with exit code {process.returncode}: {tail}"
        )
    return bytes(buffers[0])
