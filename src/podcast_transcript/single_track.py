import json
import re
import shutil
import tempfile
from datetime import timedelta
from html import escape

from pathlib import Path

from filelock import FileLock, Timeout
from rich import print as rprint

from .audio import Audio, get_audio_duration
from .backends import MLX, TranscriptionBackend, Voxhelm
from .config import settings
from .transcript_validation import (
    validate_segments,
    load_whisper_transcript,
    load_dote,
    parse_dote_timecode,
    format_dote_timecode,
)
from .safe_io import atomic_output, write_text_atomic


def backend_cache_metadata(backend: TranscriptionBackend) -> dict:
    """
    Settings that change what a backend transcribes. They are stored in the
    episode's ``cache.json``; cached transcripts made with different settings
    are discarded. Secrets such as API keys are never included.
    """
    metadata = {
        "backend": type(backend).__name__,
        "model": getattr(backend, "model_name", None),
        "language": getattr(backend, "language", None),
        "prompt": getattr(backend, "prompt", None),
    }
    if isinstance(backend, MLX):
        metadata["word_timestamps"] = backend.word_timestamps
    if isinstance(backend, Voxhelm):
        metadata["service"] = backend.cache_service_identity
    return metadata


def audio_chunks_to_text(
    service: TranscriptionBackend,
    audio_chunks: list[Path],
    chunk_durations: list[float] | None = None,
) -> list[Path]:
    """Convert the audio chunks to text. Only convert if the transcript does not exist yet."""
    file_names = " ".join([chunk.name for chunk in audio_chunks])
    rprint(f"Converting {file_names} to text")

    if chunk_durations is None:
        chunk_durations = [get_audio_duration(chunk) for chunk in audio_chunks]
    if len(audio_chunks) != len(chunk_durations):
        raise ValueError("Require one audio duration per chunk.")
    raw_transcripts = []
    for chunk, duration in zip(audio_chunks, chunk_durations):
        transcript_path = chunk.with_suffix(".json")
        if not transcript_path.exists():
            # Keep backend output and sidecars private until duration admission succeeds.
            with tempfile.TemporaryDirectory(
                prefix=".transcribe-", dir=chunk.parent
            ) as scratch:
                candidate = Path(scratch) / transcript_path.name
                service.transcribe(chunk, candidate)
                if not candidate.exists():
                    raise RuntimeError(
                        f"Transcription backend did not produce {transcript_path}"
                    )
                load_whisper_transcript(candidate, duration)
                with atomic_output(transcript_path) as tmp_path:
                    shutil.copyfile(candidate, tmp_path)
        else:
            load_whisper_transcript(transcript_path, duration)
        raw_transcripts.append(transcript_path)
    return raw_transcripts


def whisper_to_dote(input_data):
    """Convert the Whisper JSON to DOTe format."""

    def format_time(seconds):
        total_milliseconds = int(round(seconds * 1000))
        hours = total_milliseconds // 3600000
        minutes = (total_milliseconds % 3600000) // 60000
        secs = (total_milliseconds % 60000) // 1000
        millis = total_milliseconds % 1000
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    validate_segments(input_data)
    output_data = {"lines": []}
    for item in input_data:
        line = {
            "startTime": format_time(item["start"]),
            "endTime": format_time(item["end"]),
            "speakerDesignation": "",
            "text": item["text"].strip(),
        }
        output_data["lines"].append(line)
    return output_data


def whisper_text_chunks_to_dote(
    raw_text_chunks: list[Path], chunk_durations: list[float] | None = None
) -> list[Path]:
    """Transform the raw Whisper text chunks to DOTe format."""
    dote_paths = []
    for index, chunk in enumerate(raw_text_chunks):
        duration = chunk_durations[index] if chunk_durations is not None else None
        dote_path = chunk.with_suffix(".dote.json")
        if dote_path.exists():
            load_dote(dote_path, duration)
            dote_paths.append(dote_path)
            continue
        whisper_transcript = load_whisper_transcript(chunk)
        rprint(f"Converting {chunk.name} to DOTe format")
        dote_transcript = whisper_to_dote(whisper_transcript["segments"])
        write_text_atomic(dote_path, json.dumps(dote_transcript))
        dote_paths.append(dote_path)
    return dote_paths


def combine_dote_chunks(
    dote_chunks: list[Path], output_path: Path, chunk_durations: list[float]
) -> None:
    """
    Combine the DOTe chunks into a single DOTe file.

    Each chunk's timestamps are shifted by the summed audio durations of the
    chunks before it. Using the audio duration (and not the end of the last
    line) keeps trailing silence or music in a chunk from pulling every later
    timestamp early.
    """
    if len(dote_chunks) != len(chunk_durations):
        raise ValueError(
            f"Got {len(dote_chunks)} DOTe chunks but {len(chunk_durations)} chunk durations"
        )
    if len(dote_chunks) == 1:
        # Copy and return early
        [source_dote_file] = dote_chunks
        load_dote(source_dote_file, chunk_durations[0])
        rprint(f"Copying {source_dote_file} to {output_path}")
        with atomic_output(output_path) as tmp_path:
            shutil.copy(source_dote_file, tmp_path)
        return None

    rprint(f"Combining {len(dote_chunks)} DOTe chunks into {output_path}")

    combined_lines = []
    offset = timedelta()

    for filename, chunk_duration in zip(dote_chunks, chunk_durations):
        data = load_dote(filename, chunk_duration)

        for line in data["lines"]:
            new_line = dict(line)
            start_time = parse_dote_timecode(line["startTime"])
            end_time = parse_dote_timecode(line["endTime"])

            # Adjust times with offset
            new_line["startTime"] = format_dote_timecode(
                start_time + offset // timedelta(milliseconds=1)
            )
            new_line["endTime"] = format_dote_timecode(
                end_time + offset // timedelta(milliseconds=1)
            )
            combined_lines.append(new_line)

        # The next chunk starts where this chunk's audio ends
        offset += timedelta(seconds=chunk_duration)

    write_text_atomic(output_path, json.dumps({"lines": combined_lines}))


def convert_dote_to_podlove(dote_path: Path, podlove_path: Path) -> None:
    dote_data = load_dote(dote_path)

    transcripts = []
    for line in dote_data.get("lines", []):
        start_ms = parse_dote_timecode(line["startTime"])
        end_ms = parse_dote_timecode(line["endTime"])
        transcript = {
            "start": line["startTime"].replace(",", "."),
            "start_ms": start_ms,
            "end": line["endTime"].replace(",", "."),
            "end_ms": end_ms,
            "speaker": line.get("speakerDesignation", ""),
            "voice": "",  # assuming no voice data is available
            "text": line["text"],
        }
        transcripts.append(transcript)

    write_text_atomic(podlove_path, json.dumps({"transcripts": transcripts}))


def convert_to_webvtt(dote_path: Path, vtt_path: Path) -> None:
    """Converts DOTe format to WebVTT format."""
    dote_data = load_dote(dote_path)

    lines = dote_data.get("lines", [])
    output = ["WEBVTT\n"]
    for line in lines:
        start_time = line["startTime"].replace(",", ".")
        end_time = line["endTime"].replace(",", ".")
        text = line["text"].replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n(?:[ \t]*\n)+", "\n", text).strip("\n")
        text = escape(text, quote=False)
        output.append(f"{start_time} --> {end_time}")
        output.append(text)
        output.append("")  # Blank line to separate captions

    write_text_atomic(vtt_path, "\n".join(output))


def convert_to_plaintext(dote_path: Path, plaintext_path: Path) -> None:
    """Converts DOTe format to plain text."""
    dote_data = load_dote(dote_path)

    lines = dote_data.get("lines", [])
    output = []
    for line in lines:
        text = line["text"]
        output.append(text)

    write_text_atomic(plaintext_path, "\n".join(output))


def transcribe(url: str, backend: TranscriptionBackend) -> dict[str, Path]:
    settings.prepare()
    audio = Audio(base_dir=settings.transcript_dir, url=url)
    # Outside the episode directory: deleting/rebuilding episode data must not
    # change the inode used to serialize generations. Leave lock lifecycle to filelock.
    lock_dir = audio.base_dir / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(lock_dir / f"{audio.cache_key}.lock")
    try:
        lock.acquire(timeout=0)
    except Timeout as exc:
        raise RuntimeError(
            "Episode cache is busy; retry after the other transcription finishes."
        ) from exc
    try:
        return _transcribe_locked(audio, backend)
    finally:
        lock.release()


def _transcribe_locked(audio: Audio, backend: TranscriptionBackend) -> dict[str, Path]:
    transcript_paths = {}
    audio_chunks = audio.prepare_audio_for_transcription()
    # Probe before transcription or invalidating outputs, so malformed audio
    # cannot publish transcripts or discard the previous run's combined files.
    chunk_durations = [get_audio_duration(chunk) for chunk in audio_chunks]
    audio.ensure_transcript_cache_matches(backend_cache_metadata(backend))
    text_chunks = audio_chunks_to_text(backend, audio_chunks, chunk_durations)
    dote_chunks = whisper_text_chunks_to_dote(text_chunks, chunk_durations)
    dote_path = audio.output_path(".dote.json")
    if not dote_path.exists():
        combine_dote_chunks(dote_chunks, dote_path, chunk_durations)
    load_dote(dote_path, sum(chunk_durations))
    transcript_paths["DOTe"] = dote_path
    podlove_path = audio.output_path(".podlove.json")
    convert_dote_to_podlove(dote_path, podlove_path)
    transcript_paths["podlove"] = podlove_path
    webvtt_path = audio.output_path(".webvtt")
    convert_to_webvtt(dote_path, webvtt_path)
    transcript_paths["WebVTT"] = webvtt_path
    plaintext_path = audio.output_path(".txt")
    convert_to_plaintext(dote_path, plaintext_path)
    transcript_paths["plain text"] = plaintext_path
    return transcript_paths
