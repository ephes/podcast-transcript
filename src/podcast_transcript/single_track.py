import json
import hashlib
import re
import shutil
import stat
import os
import tempfile
from collections.abc import Callable
from contextlib import nullcontext
from fractions import Fraction
from html import escape

from pathlib import Path

from filelock import FileLock, Timeout
from rich import print as rprint

from .audio import (
    Audio,
    get_audio_duration,
    GENERATED_TRANSCRIPT_RE,
    TRANSCRIPT_OUTPUT_SUFFIXES,
)
from .backends import MLX, TranscriptionBackend, Voxhelm, WhisperCpp
from .config import private_mkdir, settings
from .transcript_validation import (
    validate_segments,
    exact_seconds,
    seconds_to_milliseconds,
    load_whisper_transcript,
    load_dote,
    parse_dote_timecode,
    format_dote_timecode,
)
from .safe_io import atomic_output, write_text_atomic, durable_unlink, sync_directory


def local_model_identity(path: Path) -> str | None:
    """Hash local model bytes, including relative filenames for directory models."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None  # MLX also accepts non-local model identifiers.
    if stat.S_ISREG(info.st_mode):
        return Audio._content_fingerprint(path, source_budget=False)
    if stat.S_ISDIR(info.st_mode):
        files = []
        for artifact in sorted(path.rglob("*")):
            if artifact.is_symlink():
                raise RuntimeError("Local model directory must not contain symlinks.")
            if not artifact.is_dir():
                files.append(
                    (
                        artifact.relative_to(path).as_posix(),
                        Audio._content_fingerprint(artifact, source_budget=False),
                    )
                )
        return hashlib.sha256(json.dumps(files).encode()).hexdigest()
    raise RuntimeError(
        "Local model must be a regular file or directory without symlinks."
    )


def backend_cache_metadata(backend: TranscriptionBackend) -> dict:
    """
    Settings that change what a backend transcribes. They are stored in the
    episode's ``cache.json``; different settings require a new transcript
    generation. Secrets such as API keys are never included.
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
    service: TranscriptionBackend,
    audio_chunks: list[Path],
    *,
    validate_model: Callable[[], None] | None = None,
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
            # A local inference result is published only after the model is
            # rechecked against the identity recorded in cache.json, so a chunk
            # produced by a replaced model never becomes reusable under it.
            staging = (
                atomic_output(transcript_path)
                if validate_model is not None
                else nullcontext(transcript_path)
            )
            with staging as target:
                service.transcribe(chunk, target)
                if validate_model is not None:
                    validate_model()
                    if not target.exists():
                        raise RuntimeError(
                            f"Transcription backend did not produce {transcript_path}"
                        )
                    load_whisper_transcript(target)
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

    validate_segments(input_data)
    output_data = {"lines": []}
    for item in input_data:
        line = {
            "startTime": format_dote_timecode(seconds_to_milliseconds(item["start"])),
            "endTime": format_dote_timecode(seconds_to_milliseconds(item["end"])),
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
    durations = []
    for position, duration in enumerate(chunk_durations, 1):
        try:
            durations.append(exact_seconds(duration))
        except ValueError:
            raise ValueError(
                f"Invalid chunk duration at position {position}."
            ) from None
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
    # Keep decimal durations exact; round only the cumulative boundary.
    offset = Fraction(0)

    for filename, chunk_duration in zip(dote_chunks, durations):
        data = load_dote(filename)

        for line in data["lines"]:
            new_line = dict(line)
            start_time = parse_dote_timecode(line["startTime"])
            end_time = parse_dote_timecode(line["endTime"])

            # Adjust times with offset
            new_line["startTime"] = format_dote_timecode(
                start_time + round(offset * 1000)
            )
            new_line["endTime"] = format_dote_timecode(end_time + round(offset * 1000))
            combined_lines.append(new_line)

        # The next chunk starts where this chunk's audio ends
        offset += chunk_duration

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
    audio_chunks = audio.prepare_audio_for_transcription()
    # Probe before transcription or invalidating outputs, so malformed audio
    # cannot publish transcripts or discard the previous run's combined files.
    chunk_durations = (
        [get_audio_duration(chunk) for chunk in audio_chunks]
        if len(audio_chunks) > 1
        else [0.0]
    )
    metadata = backend_cache_metadata(backend)
    try:
        cached = json.loads(audio.cache_metadata_path.read_text())
    except (FileNotFoundError, ValueError):
        cached = None
    if (
        isinstance(cached, dict)
        and cached != metadata
        and all(
            audio.output_path(suffix).is_file() for suffix in TRANSCRIPT_OUTPUT_SUFFIXES
        )
    ):
        # Do all inference and export work without touching the current generation.
        load_dote(audio.output_path(".dote.json"))
        with tempfile.TemporaryDirectory(
            prefix=".replacement-", dir=audio.podcast_dir
        ) as directory:
            staged = Audio(base_dir=audio.base_dir, url=audio.url, title=audio.title)
            staged.podcast_dir = Path(directory)
            staged.episode_chunks_dir = staged.podcast_dir / "chunks"
            private_mkdir(staged.episode_chunks_dir)
            staged_chunks = []
            for chunk in audio_chunks:
                link = staged.episode_chunks_dir / chunk.name
                link.symlink_to(chunk.resolve())
                staged_chunks.append(link)
            _transcribe_prepared(
                staged, backend, staged_chunks, chunk_durations, metadata
            )
            _promote_transcripts(audio, staged)
        return _transcript_paths(audio)
    return _transcribe_prepared(audio, backend, audio_chunks, chunk_durations, metadata)


def _transcript_paths(audio: Audio) -> dict[str, Path]:
    return dict(
        zip(
            ("DOTe", "podlove", "WebVTT", "plain text"),
            (audio.output_path(suffix) for suffix in TRANSCRIPT_OUTPUT_SUFFIXES),
        )
    )


def _transcript_files(audio: Audio) -> list[Path]:
    return [audio.output_path(suffix) for suffix in TRANSCRIPT_OUTPUT_SUFFIXES] + [
        path
        for path in sorted(audio.episode_chunks_dir.glob("chunk_*"))
        if GENERATED_TRANSCRIPT_RE.fullmatch(path.name)
    ]


def _promote_transcripts(audio: Audio, staged: Audio) -> None:
    # Keep a durable recovery copy before touching any currently published file.
    recovery = audio.podcast_dir / ".previous-transcripts"
    with tempfile.TemporaryDirectory(
        prefix=".backup-", dir=audio.podcast_dir
    ) as directory:
        backup = Path(directory)
        for path in [*_transcript_files(audio), audio.cache_metadata_path]:
            target = backup / path.relative_to(audio.podcast_dir)
            write_text_atomic(target, path.read_text())
        sync_directory(backup)
        if recovery.exists():
            shutil.rmtree(recovery)
        os.replace(backup, recovery)
        sync_directory(audio.podcast_dir)

    # Metadata is the completion claim: a partial multi-file promotion must never
    # make either old or mixed bytes reusable as the replacement generation.
    durable_unlink(audio.cache_metadata_path)
    new_files = _transcript_files(staged)
    targets = {
        audio.podcast_dir / path.relative_to(staged.podcast_dir) for path in new_files
    }
    for path in new_files:
        target = audio.podcast_dir / path.relative_to(staged.podcast_dir)
        write_text_atomic(target, path.read_text())
    for stale in _transcript_files(audio):
        if stale not in targets:
            durable_unlink(stale)
    write_text_atomic(audio.cache_metadata_path, staged.cache_metadata_path.read_text())


def _transcribe_prepared(
    audio: Audio,
    backend: TranscriptionBackend,
    audio_chunks: list[Path],
    chunk_durations: list[float],
    metadata: dict,
) -> dict[str, Path]:
    transcript_paths = {}
    audio.ensure_transcript_cache_matches(metadata)
    validate_model = None
    if "local_model_sha256" in metadata:

        def validate_model() -> None:
            if backend_cache_metadata(backend) != metadata:
                raise RuntimeError(
                    "Local model changed during transcription; retry with a stable model."
                )

    text_chunks = audio_chunks_to_text(
        backend, audio_chunks, validate_model=validate_model
    )
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
