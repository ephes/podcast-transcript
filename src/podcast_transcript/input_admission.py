"""Small regular-file admission independent of application settings."""

import os
import stat
from pathlib import Path


def file_generation(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_regular_bounded(
    path: Path, *, limit: int, description: str, dir_fd: int | None = None
) -> bytes:
    """Read one regular, non-symlink file of at most ``limit`` bytes.

    The file is admitted through a nonblocking descriptor: FIFOs, devices,
    directories and symlinks are refused before any read, and the read stops
    one byte past the limit even if the file grows after ``fstat``. Bytes are
    returned only if the descriptor generation and named identity stayed stable.
    """
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, dir_fd=dir_fd)
    except FileNotFoundError:
        raise
    except OSError:
        raise RuntimeError(f"{description} must be a regular file.") from None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"{description} must be a regular file.")
        if info.st_size > limit:
            raise RuntimeError(f"{description} exceeded byte limit ({limit} bytes).")
        data = bytearray()
        while len(data) <= limit:
            block = os.read(descriptor, min(1024 * 1024, limit + 1 - len(data)))
            if not block:
                break
            data.extend(block)
        if len(data) > limit:
            raise RuntimeError(f"{description} exceeded byte limit ({limit} bytes).")
        after = os.fstat(descriptor)
        try:
            named = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
        except OSError:
            raise RuntimeError(f"{description} changed during read.") from None

        if (
            file_generation(info) != file_generation(after)
            or file_generation(after) != file_generation(named)
            or not stat.S_ISREG(named.st_mode)
            or len(data) != info.st_size
        ):
            raise RuntimeError(f"{description} changed during read.")
        return bytes(data)
    finally:
        os.close(descriptor)
