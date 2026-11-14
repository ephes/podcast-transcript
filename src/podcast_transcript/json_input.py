"""Unambiguous JSON decoding with input-free diagnostics."""

import json
from pathlib import Path

from .input_admission import read_regular_bounded

CONTROL_MAX_BYTES = 64 * 1024


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON member.")
        result[key] = value
    return result


def loads_unique(data: str | bytes):
    return json.loads(data, object_pairs_hook=_unique_object)


def load_control_json(path: Path):
    try:
        raw = read_regular_bounded(
            path, limit=CONTROL_MAX_BYTES, description="Cache control file"
        )
        return loads_unique(raw.decode("utf-8"))
    except FileNotFoundError:
        raise
    except (ValueError, UnicodeError, RecursionError, OSError, RuntimeError):
        raise RuntimeError(
            "Invalid cache control JSON; repair or remove its marker and retry."
        ) from None
