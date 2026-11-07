"""Shared admission rules for backend transcripts and cached DOTe artifacts."""

import json
import math
import re
from pathlib import Path


def validate_segments(segments: object) -> None:
    if not isinstance(segments, list):
        raise ValueError("Invalid transcript: segments must be a list.")
    for position, segment in enumerate(segments, 1):
        message = f"Invalid transcript segment {position}: require text and finite, nonnegative start/end with end >= start."
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError(message)
        times: list[int | float] = []
        for key in ("start", "end"):
            value = segment.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(message)
            try:
                valid = value >= 0 and math.isfinite(value * 1000)
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(message)
            times.append(value)
        if times[1] < times[0]:
            raise ValueError(message)


def validate_whisper_transcript(data: object) -> None:
    if not isinstance(data, dict) or (
        "text" in data and not isinstance(data["text"], str)
    ):
        raise ValueError("Invalid transcript: require a Whisper object with segments.")
    validate_segments(data.get("segments"))


def load_whisper_transcript(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (ValueError, UnicodeError):
        raise ValueError(
            f"Invalid transcript JSON in {path.name}; remove the cached raw chunk and retry."
        ) from None
    try:
        validate_whisper_transcript(data)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from None
    return data


def parse_dote_timecode(value: object) -> int:
    """Parse full HH:MM:SS,mmm with unbounded hours; return integer milliseconds."""
    match = (
        re.fullmatch(r"([0-9]{2,}):([0-9]{2}):([0-9]{2}),([0-9]{3})", value)
        if isinstance(value, str)
        else None
    )
    if match:
        try:
            hours, minutes, seconds, millis = map(int, match.groups())
            if minutes < 60 and seconds < 60:
                return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis
        except ValueError:
            pass
    raise ValueError(
        "Invalid DOTe timecode: require HH:MM:SS,mmm with minutes/seconds 00–59."
    )


def format_dote_timecode(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def load_dote(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (ValueError, UnicodeError):
        raise ValueError(f"Invalid DOTe JSON in {path.name}.") from None
    try:
        if not isinstance(data, dict) or not isinstance(data.get("lines"), list):
            raise ValueError("Invalid DOTe: lines must be a list.")
        for position, line in enumerate(data["lines"], 1):
            if not isinstance(line, dict) or not isinstance(line.get("text"), str):
                raise ValueError(
                    f"Invalid DOTe line {position}: require a text string."
                )
            if "speakerDesignation" in line and not isinstance(
                line["speakerDesignation"], str
            ):
                raise ValueError(
                    f"Invalid DOTe line {position}: speakerDesignation must be a string."
                )
            start = parse_dote_timecode(line.get("startTime"))
            end = parse_dote_timecode(line.get("endTime"))
            if end < start:
                raise ValueError(f"Invalid DOTe line {position}: end precedes start.")
        return data
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from None
