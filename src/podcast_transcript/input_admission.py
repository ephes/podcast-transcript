"""Small regular-file admission independent of application settings."""

import os
import stat
from pathlib import Path


def read_regular_bounded(path: Path, *, limit: int, description: str) -> bytes:
    """Read one regular, non-symlink file of at most ``limit`` bytes.

    The file is admitted through a nonblocking descriptor: FIFOs, devices,
    directories and symlinks are refused before any read, and the read stops
    one byte past the limit even if the file grows after ``fstat``.
    """
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
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
        return bytes(data)
    finally:
        os.close(descriptor)
