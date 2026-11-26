"""Shared admission rules for backend transcripts and cached DOTe artifacts."""

import math
import re
from collections.abc import Iterator
from pathlib import Path
from fractions import Fraction
from typing import cast

from .input_admission import read_regular_bounded
from .json_input import loads_unique
from .safe_io import LOCAL_TRANSCRIPT_MAX_BYTES


_SURROGATE = re.compile(r"[\ud800-\udfff]")

# Largest integer that JSON consumers using IEEE 754 doubles can represent safely.
DOTE_MAX_MILLISECONDS = 2**53 - 1
_DOTE_MAX_HOUR_DIGITS = len(str(DOTE_MAX_MILLISECONDS // 3_600_000))


def exact_seconds(value: object) -> Fraction:
    """Admit millisecond-representable seconds as an exact decimal fraction."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Require finite nonnegative numeric seconds.")
    try:
        valid = value >= 0 and math.isfinite(value * 1000)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("Require finite nonnegative numeric seconds.")
    return Fraction(str(value))


def seconds_to_milliseconds(value: int | float) -> int:
    """Nearest millisecond, ties to even, using the decimal seconds value."""
    return round(exact_seconds(value) * 1000)


def validate_segments(segments: object) -> None:
    if not isinstance(segments, list):
        raise ValueError("Invalid transcript: segments must be a list.")
    previous_start: int | float = -1
    for position, segment in enumerate(segments, 1):
        message = f"Invalid transcript segment {position}: require text and finite, nonnegative start/end with end >= start."
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError(message)
        if _SURROGATE.search(segment["text"]):
            raise ValueError(
                f"Invalid transcript segment {position}: text must contain only Unicode scalar values."
            )
        times: list[int | float] = []
        for key in ("start", "end"):
            value = segment.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(message)
            try:
                milliseconds = seconds_to_milliseconds(value)
            except ValueError:
                raise ValueError(message) from None
            if milliseconds > DOTE_MAX_MILLISECONDS:
                raise ValueError(
                    f"Invalid transcript segment {position}: {key} exceeds "
                    f"{DOTE_MAX_MILLISECONDS} milliseconds."
                )
            times.append(value)
        if times[1] < times[0]:
            raise ValueError(message)
        if times[0] < previous_start:
            raise ValueError(
                f"Invalid transcript segment {position}: starts must be nondecreasing."
            )
        previous_start = times[0]


def validate_whisper_transcript(data: object) -> None:
    if not isinstance(data, dict) or (
        "text" in data and not isinstance(data["text"], str)
    ):
        raise ValueError("Invalid transcript: require a Whisper object with segments.")
    if "text" in data and _SURROGATE.search(data["text"]):
        raise ValueError(
            "Invalid transcript: top-level text must contain only Unicode scalar values."
        )
    validate_segments(data.get("segments"))


def _read_transcript(path: Path, kind: str, *, name: str | None = None) -> bytes:
    try:
        return read_regular_bounded(
            path,
            limit=LOCAL_TRANSCRIPT_MAX_BYTES,
            description=f"{kind} ({name or path.name})",
        )
    except RuntimeError as exc:
        raise ValueError(str(exc)) from None


def load_whisper_transcript(
    path: Path,
    *,
    name: str | None = None,
    cached: bool = True,
    snapshot: bytes | None = None,
) -> dict:
    """Load raw Whisper JSON; ``name`` labels errors and ``cached`` picks the hint."""
    name = name or path.name
    if snapshot is None:
        snapshot = _read_transcript(path, "Raw transcript", name=name)
    hint = (
        "remove the cached raw chunk and retry"
        if cached
        else "it was not cached, so retry once the backend response is corrected"
    )
    try:
        data = loads_unique(snapshot.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError(f"Invalid transcript JSON in {name}; {hint}.") from None
    try:
        validate_whisper_transcript(data)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from None
    return data


def parse_dote_timecode(value: object) -> int:
    """Parse full HH:MM:SS,mmm into a JSON-safe integer millisecond count."""
    match = (
        re.fullmatch(r"([0-9]{2,}):([0-9]{2}):([0-9]{2}),([0-9]{3})", value)
        if isinstance(value, str)
        else None
    )
    if match:
        hours = match[1].lstrip("0")
        minutes, seconds, millis = map(int, match.groups()[1:])
        if minutes < 60 and seconds < 60:
            if len(hours) <= _DOTE_MAX_HOUR_DIGITS:
                whole_seconds = int(hours or "0") * 3600 + minutes * 60 + seconds
                milliseconds = whole_seconds * 1000 + millis
                if milliseconds <= DOTE_MAX_MILLISECONDS:
                    return milliseconds
            raise ValueError(
                f"DOTe timecode exceeds {DOTE_MAX_MILLISECONDS} milliseconds."
            )
    raise ValueError(
        "Invalid DOTe timecode: require HH:MM:SS,mmm with minutes/seconds 00–59."
    )


def format_dote_timecode(milliseconds: int) -> str:
    """Serialize a nonnegative, JSON-safe integer millisecond count."""
    if (
        isinstance(milliseconds, bool)
        or not isinstance(milliseconds, int)
        or milliseconds < 0
    ):
        raise ValueError(
            "Invalid DOTe timecode: require nonnegative integer milliseconds."
        )
    if milliseconds > DOTE_MAX_MILLISECONDS:
        raise ValueError(f"DOTe timecode exceeds {DOTE_MAX_MILLISECONDS} milliseconds.")
    return _format_timecode(milliseconds, separator=",")


def _format_timecode(milliseconds: int, *, separator: str) -> str:
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}{separator}{millis:03}"


def _read_dote_json(path: Path, snapshot: bytes | None = None) -> object:
    if snapshot is None:
        snapshot = _read_transcript(path, "DOTe transcript")
    try:
        return loads_unique(snapshot.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError(f"Invalid DOTe JSON in {path.name}.") from None


def load_dote(
    path: Path, *, snapshot: bytes | None = None, max_lines: int | None = None
) -> dict:
    """Validate DOTe JSON, optionally limiting the line count for episode assembly."""
    data = _read_dote_json(path, snapshot)
    try:
        validate_dote(data, max_lines=max_lines)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from None
    return cast(dict, data)


def _iter_dote_file_cues(path: Path) -> Iterator[tuple[dict, int, int]]:
    """Yield validated cues from one admitted DOTe file."""
    data = _read_dote_json(path)
    try:
        yield from _iter_validated_dote_cues(data)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from None


def validate_dote(data: object, *, max_lines: int | None = None) -> None:
    for _ in _iter_validated_dote_cues(data, max_lines=max_lines):
        pass


def _iter_validated_dote_cues(
    data: object, *, max_lines: int | None = None
) -> Iterator[tuple[dict, int, int]]:
    if not isinstance(data, dict) or not isinstance(data.get("lines"), list):
        raise ValueError("Invalid DOTe: lines must be a list.")
    lines = data["lines"]
    if max_lines is not None and len(lines) > max_lines:
        raise ValueError("Combined transcript exceeds episode line limit.")
    previous_start = -1
    for position, line in enumerate(lines, 1):
        text = line.get("text") if isinstance(line, dict) else None
        if not isinstance(text, str):
            raise ValueError(f"Invalid DOTe line {position}: require a text string.")
        if _SURROGATE.search(text):
            raise ValueError(
                f"Invalid DOTe line {position}: text must contain only Unicode scalar values."
            )
        speaker = line.get("speakerDesignation", "")
        if not isinstance(speaker, str):
            raise ValueError(
                f"Invalid DOTe line {position}: speakerDesignation must be a string."
            )
        if _SURROGATE.search(speaker):
            raise ValueError(
                f"Invalid DOTe line {position}: speakerDesignation must contain only Unicode scalar values."
            )
        try:
            start = parse_dote_timecode(line.get("startTime"))
            end = parse_dote_timecode(line.get("endTime"))
        except ValueError as exc:
            raise ValueError(f"Invalid DOTe line {position}: {exc}") from None
        if end < start:
            raise ValueError(f"Invalid DOTe line {position}: end precedes start.")
        if start < previous_start:
            raise ValueError(
                f"Invalid DOTe line {position}: starts must be nondecreasing."
            )
        previous_start = start
        yield line, start, end
