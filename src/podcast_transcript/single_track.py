import json
import hashlib
import re
import shutil
from datetime import timedelta
from html import escape

from pathlib import Path

from filelock import FileLock, Timeout
from rich import print as rprint

from .audio import Audio, get_audio_duration
from .backends import MLX, TranscriptionBackend, Voxhelm, WhisperCpp
from .config import private_mkdir, settings
from .transcript_validation import (
    validate_segments,
    load_whisper_transcript,
    load_dote,
    parse_dote_timecode,
    format_dote_timecode,
)
from .safe_io import atomic_output, write_text_atomic


def local_model_identity(path: Path) -> str | None:
    """Hash local model bytes, including relative filenames for directory models."""
    if path.is_file():
        return Audio._content_fingerprint(path)
    if path.is_dir():
        files = []
        for artifact in sorted(path.rglob("*")):
            if artifact.is_symlink() and artifact.is_dir():
                raise ValueError(
                    "Local model directory must not contain directory symlinks."
                )
            if artifact.is_file():
                files.append(
                    (
                        artifact.relative_to(path).as_posix(),
                        Audio._content_fingerprint(artifact),
                    )
                )
        return hashlib.sha256(json.dumps(files).encode()).hexdigest()
    return None


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
    if isinstance(backend, (WhisperCpp, MLX)):
        model_path = (
            backend.model_path
            if isinstance(backend, WhisperCpp)
            else Path(backend.model_name).expanduser()
        )
        identity = local_model_identity(model_path)
        if identity is not None:
            metadata["local_model_sha256"] = identity
    if isinstance(backend, MLX):
        metadata["word_timestamps"] = backend.word_timestamps
    if isinstance(backend, Voxhelm):
        metadata["service"] = backend.cache_service_identity
    return metadata


def audio_chunks_to_text(
    service: TranscriptionBackend, audio_chunks: list[Path]
) -> list[Path]:
    """Convert the audio chunks to text. Only convert if the transcript does not exist yet."""
    file_names = " ".join([chunk.name for chunk in audio_chunks])
    rprint(f"Converting {file_names} to text")

    raw_transcripts = []
    for chunk in audio_chunks:
        chunk_name = chunk.name.split(".")[0]
        transcript_name = f"{chunk_name}.json"
        transcript_path = chunk.parent / transcript_name
        cached = transcript_path.exists()
        if not cached:
            service.transcribe(chunk, transcript_path)
            if not transcript_path.exists():
                raise RuntimeError(
                    f"Transcription backend did not produce {transcript_path}"
                )
        try:
            load_whisper_transcript(transcript_path)
        except ValueError:
            if not cached:
                transcript_path.unlink(missing_ok=True)
            raise
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


def whisper_text_chunks_to_dote(raw_text_chunks: list[Path]) -> list[Path]:
    """Transform the raw Whisper text chunks to DOTe format."""
    dote_paths = []
    for chunk in raw_text_chunks:
        dote_path = chunk.with_suffix(".dote.json")
        if dote_path.exists():
            load_dote(dote_path)
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
        load_dote(source_dote_file)
        rprint(f"Copying {source_dote_file} to {output_path}")
        with atomic_output(output_path) as tmp_path:
            shutil.copy(source_dote_file, tmp_path)
        return None

    rprint(f"Combining {len(dote_chunks)} DOTe chunks into {output_path}")

    combined_lines = []
    offset = timedelta()

    for filename, chunk_duration in zip(dote_chunks, chunk_durations):
        data = load_dote(filename)

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
    audio = Audio(base_dir=settings.transcript_dir, url=url)
    # Outside the episode directory: deleting/rebuilding episode data must not
    # change the inode used to serialize generations. Leave lock lifecycle to filelock.
    lock_dir = audio.base_dir / ".locks"
    private_mkdir(lock_dir)
    lock = FileLock(lock_dir / f"{audio.cache_key}.lock", mode=0o600)
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
    chunk_durations = (
        [get_audio_duration(chunk) for chunk in audio_chunks]
        if len(audio_chunks) > 1
        else [0.0]
    )
    audio.ensure_transcript_cache_matches(backend_cache_metadata(backend))
    text_chunks = audio_chunks_to_text(backend, audio_chunks)
    dote_chunks = whisper_text_chunks_to_dote(text_chunks)
    dote_path = audio.output_path(".dote.json")
    if not dote_path.exists():
        combine_dote_chunks(dote_chunks, dote_path, chunk_durations)
    transcript_paths["DOTe"] = dote_path
    podlove_path = audio.output_path(".podlove.json")
    if not podlove_path.exists():
        convert_dote_to_podlove(dote_path, podlove_path)
    transcript_paths["podlove"] = podlove_path
    webvtt_path = audio.output_path(".webvtt")
    if not webvtt_path.exists():
        convert_to_webvtt(dote_path, webvtt_path)
    transcript_paths["WebVTT"] = webvtt_path
    plaintext_path = audio.output_path(".txt")
    if not plaintext_path.exists():
        convert_to_plaintext(dote_path, plaintext_path)
    transcript_paths["plain text"] = plaintext_path
    return transcript_paths
