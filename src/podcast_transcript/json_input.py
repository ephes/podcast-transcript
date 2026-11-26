"""Unambiguous JSON decoding with input-free diagnostics."""

import json
import math
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


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("Nonfinite JSON number.")
    return value


def loads_unique(data: str | bytes, *, finite_numbers: bool = False):
    if finite_numbers:
        return json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_float=_finite_float,
            parse_constant=_finite_float,
        )
    return json.loads(data, object_pairs_hook=_unique_object)


def load_control_json(
    path: Path, *, dir_fd: int | None = None, expected_type: type | None = None
):
    try:
        raw = read_regular_bounded(
            path,
            limit=CONTROL_MAX_BYTES,
            description="Cache control file",
            dir_fd=dir_fd,
        )
        value = loads_unique(raw.decode("utf-8"))
        if expected_type is not None and not isinstance(value, expected_type):
            raise ValueError("Unexpected top-level JSON type.")
        return value
    except FileNotFoundError:
        raise
    except (ValueError, UnicodeError, RecursionError, OSError, RuntimeError):
        # The marker's location is not a control value; name it so the
        # documented repair can target the right file.
        marker = (
            Path(path.parent.name) / path.name
            if path.parent.name == "chunks"
            else Path(path.name)
        )
        raise RuntimeError(
            f"Invalid cache control JSON in {marker}; repair or remove this marker and retry."
        ) from None
