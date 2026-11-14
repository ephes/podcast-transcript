"""
Helpers that make pipeline steps fail closed.

Every step in the pipeline caches its result by checking whether the output
file exists. That is only safe if an output file appears exclusively after the
step succeeded. These helpers stage files in a private directory beside the final output
and rename them into place only on success, and they turn failed subprocesses
into exceptions that carry the tool's stderr.
"""

import math
import errno
import select
import os
import signal
import stat
import subprocess
import tempfile
import uuid

from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from .config import private_mkdir, settings
from .input_admission import read_regular_bounded as read_regular_bounded

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
        durable_replace(tmp_path, path)


def sync_directory(path: Path) -> None:
    """Persist directory entries; propagate unsupported-filesystem errors."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_file(path: Path) -> None:
    """Sync an admitted regular artifact before its completion is recorded."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("Publication requires a regular file or symlink.")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_replace(source: Path, target: Path) -> None:
    """Persist data before its name; symlinks publish only a directory entry.

    A symlink's target must already be durable. Never open a staged symlink:
    short-audio links are relative to their final parent, not the staging one.
    """
    if not source.is_symlink():
        sync_file(source)
    os.replace(source, target)
    sync_directory(target.parent)
    if source.parent != target.parent:
        sync_directory(source.parent)


def durable_unlink(path: Path) -> None:
    """Retire a completion marker before changing the data it describes."""
    path.unlink(missing_ok=True)
    if path.parent.exists():
        sync_directory(path.parent)


LOCAL_TRANSCRIPT_MAX_BYTES = 64 * 1024 * 1024


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so that a crash never leaves a partial file."""
    with atomic_output(path) as tmp_path:
        descriptor = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as file:
            file.write(text)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` exactly to ``path`` so that a crash never leaves a partial file."""
    with atomic_output(path) as tmp_path:
        descriptor = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)


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

    capture_checked(
        cmd,
        description=description,
        timeout=timeout,
        stdout_limit=None,
        stderr_limit=STDERR_TAIL_CHARS,
        stderr_tail=True,
    )


def capture_checked(
    cmd: list[str],
    *,
    description: str,
    timeout: float,
    stdout_limit: int | None = 64 * 1024,
    stderr_limit: int = 64 * 1024,
    stderr_tail: bool = False,
) -> bytes:
    """Capture small tool results, killing/reaping the owned tree on overflow."""
    import threading
    import time

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Command timeout must be positive finite seconds")
    buffers = [bytearray(), bytearray()]
    overflow = threading.Event()
    stop = threading.Event()
    truncated = threading.Event()
    failures: list[BaseException] = []

    def diagnostic_tail():
        text = bytes(buffers[1][-STDERR_TAIL_CHARS:]).decode("utf-8", errors="replace")
        lines = text.strip().splitlines()
        dropped = truncated.is_set() or len(lines) > STDERR_TAIL_LINES
        text = "\n".join(lines[-STDERR_TAIL_LINES:])
        return ("[stderr truncated]\n" if dropped else "") + text

    def drain(stream, buffer, limit, keep_tail):
        try:
            while not stop.is_set():
                block = stream.read(8192)
                if not block:
                    break
                if keep_tail:
                    if len(buffer) + len(block) > limit:
                        truncated.set()
                    buffer[:] = (buffer + block)[-limit:]
                else:
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
        stdout=subprocess.PIPE if stdout_limit is not None else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=0,
        start_new_session=os.name == "posix",
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    ) as process:
        threads = [
            threading.Thread(
                target=drain, args=(stream, buffer, limit, keep_tail), daemon=True
            )
            for stream, buffer, limit, keep_tail in zip(
                (process.stdout, process.stderr),
                buffers,
                (stdout_limit, stderr_limit),
                (False, stderr_tail),
            )
            if stream is not None
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
            if os.name == "posix":
                # Observe exit without consuming the leader's identity. Even a
                # successful leader can leave pipe-detached children in its group.
                if hasattr(os, "waitid"):
                    while (
                        os.waitid(
                            os.P_PID, process.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG
                        )
                        is None
                    ):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(cmd, timeout)
                        time.sleep(min(0.01, remaining))
                else:
                    # macOS Python 3.10 lacks waitid. kqueue observes exit
                    # without waitpid; ESRCH at registration means the owned,
                    # still-unreaped child already exited before registration.
                    with closing(select.kqueue()) as queue:
                        event = select.kevent(
                            process.pid,
                            filter=select.KQ_FILTER_PROC,
                            flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                            fflags=select.KQ_NOTE_EXIT,
                        )
                        events = queue.control(
                            [event], 1, max(0, deadline - time.monotonic())
                        )
                        if not events:
                            raise subprocess.TimeoutExpired(cmd, timeout)
                        observed = events[0]
                        if (
                            observed.flags & select.KQ_EV_ERROR
                            and observed.data != errno.ESRCH
                        ):
                            raise RuntimeError("Could not observe owned command exit.")
                _kill_owned_tree(process)
            else:
                process.wait(timeout=max(0, deadline - time.monotonic()))
        except BaseException as exc:
            stop.set()
            if process.returncode is None:
                _kill_owned_tree(process)
            if isinstance(exc, subprocess.TimeoutExpired):
                tail = diagnostic_tail()
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
        tail = diagnostic_tail()
        raise RuntimeError(
            f"{description} failed with exit code {process.returncode}: {tail}"
        )
    return bytes(buffers[0])
