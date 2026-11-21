"""
Helpers that make pipeline steps fail closed.

These helpers stage files in a private directory beside the final output and
rename them into place after writing succeeds. They report directory sync
failures after rename separately from failures before publication. Failed
subprocesses raise exceptions that carry the tool's stderr.
"""

import math
import errno
import logging
import select
import os
import signal
import stat
import subprocess
import tempfile

from collections.abc import Iterator, Generator
from contextlib import closing, contextmanager
from pathlib import Path
from time import monotonic

from .config import private_mkdir, settings
from .input_admission import file_generation
from .input_admission import read_regular_bounded as read_regular_bounded

logger = logging.getLogger(__name__)

STDERR_TAIL_LINES = 20
STDERR_TAIL_CHARS = 2000


class PublicationUncertainError(OSError):
    """The replacement became visible, and a directory sync failed."""

    def __init__(self, target: Path, directory: Path) -> None:
        self.target = target
        self.directory = directory
        super().__init__(
            f"Output was published at {target}. "
            f"Durability is uncertain after directory sync failed for {directory}."
        )


@contextmanager
def atomic_output(path: Path) -> Iterator[Path]:
    """
    Yield a temporary path to write to and move it to ``path`` on success.

    Errors before rename leave ``path`` untouched. A missing temporary file
    raises ``RuntimeError``. A directory sync failure after rename raises
    ``PublicationUncertainError`` with the new output already visible at ``path``.
    """
    private_mkdir(path.parent)
    # Tools may create their output with ambient permissions. A private staging
    # directory protects those bytes before publication, even in a shared parent.
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    with tempfile.TemporaryDirectory(prefix=".output-", dir=path.parent) as directory:
        # The directory supplies uniqueness. A single-byte stem keeps the
        # media suffix for tools like ffmpeg without lengthening the filename.
        tmp_path = Path(directory) / f"_{path.suffix}"
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

    Raise ``PublicationUncertainError`` if a directory sync fails after rename.
    """
    if not source.is_symlink():
        sync_file(source)
    os.replace(source, target)
    directory = target.parent
    try:
        sync_directory(directory)
        if source.parent != directory:
            directory = source.parent
            sync_directory(directory)
    except OSError as exc:
        raise PublicationUncertainError(target, directory) from exc


def durable_unlink(path: Path) -> None:
    """Retire a completion marker before changing the data it describes."""
    path.unlink(missing_ok=True)
    if path.parent.exists():
        sync_directory(path.parent)


LOCAL_TRANSCRIPT_MAX_BYTES = 64 * 1024 * 1024


@contextmanager
def _output_descriptor(path: Path) -> Iterator[int]:
    """Own a staged output descriptor until writing and cleanup finish."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        yield descriptor
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            logger.warning("Could not close the staged transcript output descriptor.")
        raise
    else:
        os.close(descriptor)


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so that a crash never leaves a partial file."""
    with atomic_output(path) as tmp_path, _output_descriptor(tmp_path) as descriptor:
        with os.fdopen(descriptor, "w", closefd=False) as file:
            file.write(text)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` exactly to ``path`` so that a crash never leaves a partial file."""
    with atomic_output(path) as tmp_path, _output_descriptor(tmp_path) as descriptor:
        with os.fdopen(descriptor, "wb", closefd=False) as file:
            file.write(data)


COPY_BUFFER_BYTES = 1024 * 1024


TRANSCRIPT_WORK_MAX_ENTRIES = 100_000
TRANSCRIPT_WORK_MAX_BYTES = 64 * 1024**3
TRANSCRIPT_WORK_TIMEOUT_SECONDS = 3600.0


class TranscriptWorkBudget:
    """Aggregate enumeration, hash and copy work for one promotion/restore pass."""

    def __init__(self):
        self.remaining = TRANSCRIPT_WORK_MAX_BYTES
        self.entries = 0
        self.deadline = monotonic() + TRANSCRIPT_WORK_TIMEOUT_SECONDS

    def check(self) -> None:
        if monotonic() >= self.deadline:
            raise RuntimeError(
                "Transcript work exceeded elapsed limit; split the episode."
            )

    def admit_entry(self) -> None:
        self.check()
        self.entries += 1
        if self.entries > TRANSCRIPT_WORK_MAX_ENTRIES:
            raise RuntimeError(
                "Transcript work exceeded directory entry limit; narrow the inventory."
            )

    def charge(self, size: int) -> None:
        self.check()
        self.remaining -= size
        if self.remaining < 0:
            raise RuntimeError(
                "Transcript work exceeded byte limit; split the episode."
            )


def regular_file_blocks(
    path: Path,
    *,
    budget: TranscriptWorkBudget | None = None,
    expected_generation: tuple[int, ...] | None = None,
) -> Generator[bytes, None, None]:
    """Read bounded blocks from a stable regular file of the expected generation."""
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OSError("Transcript copy requires a regular file.")
        if (
            expected_generation is not None
            and file_generation(before) != expected_generation
        ):
            raise OSError("Transcript source changed before read.")
        consumed = 0
        while True:
            if budget is not None:
                budget.check()
            size = (
                COPY_BUFFER_BYTES
                if budget is None
                else min(COPY_BUFFER_BYTES, budget.remaining + 1)
            )
            block = os.read(descriptor, size)
            if budget is not None:
                budget.charge(len(block))
            if not block:
                break
            consumed += len(block)
            yield block
        if (
            file_generation(before) != file_generation(os.fstat(descriptor))
            or consumed != before.st_size
        ):
            raise OSError("Transcript source changed during read.")
    finally:
        os.close(descriptor)


def copy_file_atomic(
    source: Path, target: Path, *, budget: TranscriptWorkBudget | None = None
) -> None:
    """Copy exact bytes with bounded memory into the owned durable writer."""
    with atomic_output(target) as temporary:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with closing(regular_file_blocks(source, budget=budget)) as blocks:
                for block in blocks:
                    remaining = memoryview(block)
                    while remaining:
                        if budget is not None:
                            budget.check()
                        written = os.write(descriptor, remaining)
                        if budget is not None:
                            budget.check()
                        if written <= 0:
                            raise OSError("Transcript copy made no write progress.")
                        remaining = remaining[written:]
        finally:
            os.close(descriptor)
        if budget is not None:
            budget.check()


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
    cmd: list[str],
    *,
    description: str,
    timeout: float | None = None,
    output_limit: tuple[Path, int] | None = None,
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
        output_limit=output_limit,
    )


def capture_checked(
    cmd: list[str],
    *,
    description: str,
    timeout: float,
    stdout_limit: int | None = 64 * 1024,
    stderr_limit: int = 64 * 1024,
    stderr_tail: bool = False,
    output_limit: tuple[Path, int] | None = None,
) -> bytes:
    """Capture small tool results, killing/reaping the owned tree on overflow."""
    import threading
    import time

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Command timeout must be positive finite seconds")
    if output_limit is not None and (
        type(output_limit[1]) is not int or output_limit[1] <= 0
    ):
        raise ValueError("Command output limit must be positive integer bytes")

    def check_output_limit():
        if output_limit is None:
            return
        path, maximum = output_limit
        try:
            info = path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"{description} output must be a regular file")
        if info.st_size >= maximum:
            raise RuntimeError(f"{description} reached output byte limit")

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
                check_output_limit()
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
                        check_output_limit()
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
                        changes = [event]
                        while True:
                            check_output_limit()
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise subprocess.TimeoutExpired(cmd, timeout)
                            events = queue.control(changes, 1, min(0.05, remaining))
                            changes = []
                            if events:
                                break
                        observed = events[0]
                        if (
                            observed.flags & select.KQ_EV_ERROR
                            and observed.data != errno.ESRCH
                        ):
                            raise RuntimeError("Could not observe owned command exit.")
                _kill_owned_tree(process)
            else:
                while True:
                    check_output_limit()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(cmd, timeout)
                    try:
                        process.wait(timeout=min(0.05, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        pass
            check_output_limit()
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
