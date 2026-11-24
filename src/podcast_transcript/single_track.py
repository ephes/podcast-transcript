import json
import hashlib
import re
import shutil
import sys
import stat
import os
import tempfile
from collections.abc import Callable
from contextlib import nullcontext, contextmanager, closing
from fractions import Fraction
from html import escape

from pathlib import Path
from time import monotonic

from filelock import FileLock, Timeout
from rich import print as rprint

from .audio import (
    Audio,
    ModelBudget,
    GENERATED_TRANSCRIPT_RE,
    TRANSCRIPT_OUTPUT_LABELS,
    TRANSCRIPT_OUTPUT_SUFFIXES,
)
from .backends import MLX, TranscriptionBackend, Voxhelm, WhisperCpp
from .config import private_mkdir, settings
from .input_admission import file_generation
from .json_input import load_control_json
from .transcript_validation import (
    DOTE_MAX_MILLISECONDS,
    validate_segments,
    validate_dote,
    exact_seconds,
    seconds_to_milliseconds,
    load_whisper_transcript,
    load_dote,
    _load_dote_cues,
    format_dote_timecode,
)
from .safe_io import (
    TranscriptWorkBudget,
    atomic_output,
    durable_unlink,
    sync_directory,
    copy_file_atomic,
    read_regular_bounded,
    regular_file_blocks,
    write_bytes_atomic,
    write_text_atomic,
)


MODEL_DIRECTORY_DESCRIPTORS_SUPPORTED = (
    hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and os.open in os.supports_dir_fd
    and os.scandir in os.supports_fd
)


def local_model_identity(path: Path) -> str | None:
    """Hash local model bytes, including relative filenames for directory models."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None  # MLX also accepts non-local model identifiers.
    if stat.S_ISREG(info.st_mode):
        return Audio._content_fingerprint(path, source_budget=False)
    if stat.S_ISDIR(info.st_mode):
        try:
            return _directory_model_identity(path, info)
        except OSError:
            raise RuntimeError(
                "Local model directory changed or could not be admitted."
            ) from None
    raise RuntimeError(
        "Local model must be a regular file or directory without symlinks."
    )


def _directory_generation(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns)


@contextmanager
def _model_directory_at(
    root: int,
    parts: tuple[str, ...],
    admitted: dict[tuple[str, ...], os.stat_result],
    budget: ModelBudget,
):
    """Reopen components from the pinned root, keeping only two traversal descriptors live."""
    descriptor = os.dup(root)
    try:
        for depth in range(len(parts) + 1):
            budget.check()
            if depth:
                child = os.open(
                    parts[depth - 1],
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = child
            if _directory_generation(os.fstat(descriptor)) != _directory_generation(
                admitted[parts[:depth]]
            ):
                raise RuntimeError("Local model directory changed during hashing.")
        yield descriptor
        budget.check()
        if _directory_generation(os.fstat(descriptor)) != _directory_generation(
            admitted[parts]
        ):
            raise RuntimeError("Local model directory changed during hashing.")
    finally:
        os.close(descriptor)


def _directory_model_identity(path: Path, info: os.stat_result) -> str:
    if not MODEL_DIRECTORY_DESCRIPTORS_SUPPORTED:
        raise RuntimeError(
            "Local model directory requires no-follow directory descriptor support."
        )
    budget = ModelBudget()
    root = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        admitted: dict[tuple[str, ...], os.stat_result] = {(): info}
        artifacts = []
        pending: list[tuple[str, ...]] = [()]
        while pending:
            parts = pending.pop()
            with _model_directory_at(root, parts, admitted, budget) as directory:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        budget.admit_entry()
                        child_parts = (*parts, entry.name)
                        child_info = os.stat(
                            entry.name, dir_fd=directory, follow_symlinks=False
                        )
                        if stat.S_ISDIR(child_info.st_mode):
                            admitted[child_parts] = child_info
                            pending.append(child_parts)
                        elif stat.S_ISREG(child_info.st_mode):
                            artifacts.append(child_parts)
                        else:
                            raise RuntimeError(
                                "Local model directory must contain only regular files and directories without symlinks."
                            )
        # Sorting stays compatible and is bounded by the shared entry budget.
        files = []
        for parts in sorted(artifacts):
            with _model_directory_at(root, parts[:-1], admitted, budget) as directory:
                digest = Audio._read_local_source(
                    Path(parts[-1]), model=True, model_budget=budget, dir_fd=directory
                )
                files.append(("/".join(parts), digest))
        # Reopen every admitted component before accepting any cache identity.
        for parts in admitted:
            with _model_directory_at(root, parts, admitted, budget):
                pass
        if _directory_generation(path.lstat()) != _directory_generation(info):
            raise RuntimeError("Local model directory changed during hashing.")
        budget.check()
        identity = hashlib.sha256(json.dumps(files).encode()).hexdigest()
        budget.check()
        return identity
    finally:
        os.close(root)


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
    validate_chunk: Callable[[Path], None] | None = None,
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
                if validate_model is not None or validate_chunk is not None
                else nullcontext(transcript_path)
            )
            with staging as target:
                service.transcribe(chunk, target)
                if validate_model is not None or validate_chunk is not None:
                    if validate_model is not None:
                        validate_model()
                    if validate_chunk is not None:
                        validate_chunk(chunk)
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
    for position, item in enumerate(input_data, 1):
        start_ms = seconds_to_milliseconds(item["start"])
        end_ms = seconds_to_milliseconds(item["end"])
        if end_ms > DOTE_MAX_MILLISECONDS:
            raise ValueError(
                f"Invalid DOTe line {position}: timecode exceeds {DOTE_MAX_MILLISECONDS} milliseconds."
            )
        line = {
            "startTime": format_dote_timecode(start_ms),
            "endTime": format_dote_timecode(end_ms),
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


# Bound both object overhead and all serialized content (including unknown fields).
COMBINED_MAX_LINES = 250_000
COMBINED_MAX_BYTES = 64 * 1024 * 1024


def _json_size(data: object) -> int:
    return sum(
        len(part.encode("utf-8")) for part in json.JSONEncoder().iterencode(data)
    )


def _check_combined_budget(lines: int, size: int) -> None:
    if lines > COMBINED_MAX_LINES or size > COMBINED_MAX_BYTES:
        raise ValueError("Combined transcript exceeds episode line/byte limit.")


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
    if not dote_chunks:
        raise ValueError("Require at least one DOTe chunk to combine.")
    durations = []
    for position, duration in enumerate(chunk_durations, 1):
        try:
            durations.append(exact_seconds(duration))
        except ValueError:
            raise ValueError(
                f"Invalid chunk duration at position {position}."
            ) from None
    if len(dote_chunks) == 1:
        [source_dote_file] = dote_chunks
        try:
            snapshot = read_regular_bounded(
                source_dote_file,
                limit=COMBINED_MAX_BYTES,
                description=f"Combined transcript ({source_dote_file.name})",
            )
        except RuntimeError as exc:
            raise ValueError(str(exc)) from None
        data = load_dote(source_dote_file, snapshot=snapshot)
        _check_combined_budget(len(data["lines"]), len(snapshot))
        rprint(f"Copying {source_dote_file} to {output_path}")
        write_bytes_atomic(output_path, snapshot)
        return None

    rprint(f"Combining {len(dote_chunks)} DOTe chunks into {output_path}")

    combined_lines: list[dict] = []
    size = _json_size({"lines": []})
    _check_combined_budget(0, size)
    # Keep decimal durations exact; round only the cumulative boundary.
    offset = Fraction(0)

    for position, (filename, chunk_duration) in enumerate(
        zip(dote_chunks, durations), 1
    ):
        offset_ms = round(offset * 1000)
        if offset_ms > DOTE_MAX_MILLISECONDS:
            raise ValueError(
                f"DOTe chunk offset at position {position} exceeds {DOTE_MAX_MILLISECONDS} milliseconds."
            )
        for line, start_ms, end_ms in _load_dote_cues(filename):
            new_line = dict(line)
            new_line["startTime"] = format_dote_timecode(start_ms + offset_ms)
            new_line["endTime"] = format_dote_timecode(end_ms + offset_ms)
            size += _json_size(new_line) + (2 if combined_lines else 0)
            _check_combined_budget(len(combined_lines) + 1, size)
            combined_lines.append(new_line)

        # Only a following chunk uses this cumulative duration.
        offset += chunk_duration

    combined = {"lines": combined_lines}
    validate_dote(combined)
    # Avoid a second episode-sized joined string after admission.
    with atomic_output(output_path) as temporary:
        with temporary.open("w", encoding="utf-8") as target:
            json.dump(combined, target)


def convert_dote_to_podlove(dote_path: Path, podlove_path: Path) -> None:
    cues = _load_dote_cues(dote_path)

    transcripts = []
    for line, start_ms, end_ms in cues:
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
    cues = _load_dote_cues(dote_path)

    with atomic_output(vtt_path) as temporary:
        with temporary.open("x", encoding="utf-8") as target:
            target.write("WEBVTT\n")
            for line, start_ms, end_ms in cues:
                start_time = line["startTime"].replace(",", ".")
                end_time = line["endTime"].replace(",", ".")
                text = line["text"].replace("\0", "\ufffd")
                text = text.replace("\r\n", "\n").replace("\r", "\n")
                text = re.sub(r"\n(?:[ \t]*\n)+", "\n", text).strip("\n")
                if not text.strip():
                    continue
                if end_ms == start_ms:
                    # Quantization can collapse an admitted span. Extend only this
                    # rendered cue; source and DOTe timing evidence stays unchanged.
                    end_time = format_dote_timecode(start_ms + 1).replace(",", ".")
                text = escape(text, quote=False)
                target.write(f"\n{start_time} --> {end_time}\n{text}\n")


def convert_to_plaintext(dote_path: Path, plaintext_path: Path) -> None:
    """Converts DOTe format to plain text."""
    dote_data = load_dote(dote_path)

    with atomic_output(plaintext_path) as temporary:
        with temporary.open("x", encoding="utf-8") as target:
            for index, line in enumerate(dote_data.get("lines", [])):
                if index:
                    target.write("\n")
                target.write(line["text"])


@contextmanager
def _episode_lock(audio: Audio):
    # Outside episode data so replacement cannot change the lock identity.
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
        yield
    finally:
        lock.release()


def transcribe(url: str, backend: TranscriptionBackend) -> dict[str, Path]:
    audio = Audio(base_dir=settings.transcript_dir, url=url)
    with _episode_lock(audio):
        return _transcribe_locked(audio, backend)


def _generation_at(audio: Audio, directory: Path) -> Audio:
    generation = Audio(base_dir=audio.base_dir, url=audio.url, title=audio.title)
    generation.podcast_dir = directory
    generation.episode_chunks_dir = directory / "chunks"
    return generation


def _file_digest(path: Path, *, budget: TranscriptWorkBudget | None = None) -> str:
    budget = budget or TranscriptWorkBudget()
    digest = hashlib.sha256()
    for block in regular_file_blocks(path, budget=budget):
        digest.update(block)
    return digest.hexdigest()


PREPARED_AUDIO_MAX_BYTES = 64 * 1024**3
PREPARED_AUDIO_HASH_TIMEOUT_SECONDS = 3600.0


def _chunk_byte_provenance(audio: Audio, chunks: list[Path]) -> dict[str, str]:
    """Bind cached raw and derived transcripts to each admitted MP3's bytes."""
    deadline = monotonic() + PREPARED_AUDIO_HASH_TIMEOUT_SECONDS
    remaining = PREPARED_AUDIO_MAX_BYTES
    result = {}
    for chunk in chunks:
        source = chunk
        if chunk.is_symlink():
            if not audio._is_owned_short_chunk_link(chunk, chunk_count=len(chunks)):
                raise RuntimeError("Prepared chunk link is not owned resampled audio.")
            source = audio.resampled_episode_path
        digest = hashlib.sha256()
        with closing(regular_file_blocks(source)) as blocks:
            for block in blocks:
                if monotonic() >= deadline:
                    raise RuntimeError("Prepared audio hashing exceeded elapsed limit.")
                remaining -= len(block)
                if remaining < 0:
                    raise RuntimeError("Prepared audio hashing exceeded byte limit.")
                digest.update(block)
        if monotonic() >= deadline:
            raise RuntimeError("Prepared audio hashing exceeded elapsed limit.")
        result[chunk.name] = digest.hexdigest()
    return result


def _prepared_audio_digest(audio: Audio) -> str:
    """Bind recovery to the admitted MP3 inventory and any inference WAV caches."""
    deadline = monotonic() + PREPARED_AUDIO_HASH_TIMEOUT_SECONDS

    def check_time():
        if monotonic() >= deadline:
            raise RuntimeError("Prepared audio hashing exceeded elapsed limit.")

    chunks = audio._load_chunk_manifest()
    if chunks is None:
        raise RuntimeError("Recovery requires a valid prepared audio inventory.")
    chunk_set = set(chunks)
    resampled = audio.resampled_episode_path
    paths = [resampled, *chunks]
    for chunk in chunks:
        wav = chunk.with_suffix(".wav")
        if wav.exists() or wav.is_symlink():
            paths.append(wav)
    remaining = PREPARED_AUDIO_MAX_BYTES

    def hash_audio(
        path: Path, *, expected_generation: tuple[int, ...] | None = None
    ) -> str:
        nonlocal remaining
        digest = hashlib.sha256()
        with closing(
            regular_file_blocks(path, expected_generation=expected_generation)
        ) as blocks:
            for block in blocks:
                check_time()
                remaining -= len(block)
                if remaining < 0:
                    raise RuntimeError("Prepared audio hashing exceeded byte limit.")
                digest.update(block)
        check_time()
        return digest.hexdigest()

    check_time()
    resampled_generation = file_generation(resampled.lstat())
    resampled_digest = hash_audio(resampled, expected_generation=resampled_generation)
    inventory = hashlib.sha256()
    for path in paths:
        check_time()
        if path == resampled:
            digest = resampled_digest
        elif path in chunk_set and path.is_symlink():
            if not audio._is_owned_short_chunk_link(path, chunk_count=len(chunks)):
                raise RuntimeError("Recovery requires an owned short chunk link.")
            if file_generation(resampled.lstat()) != resampled_generation:
                raise RuntimeError(
                    "Prepared audio changed before short chunk digest reuse."
                )
            check_time()
            digest = resampled_digest
        else:
            digest = hash_audio(path)
        inventory.update(
            json.dumps(
                (path.relative_to(audio.podcast_dir).as_posix(), digest)
            ).encode()
        )
        inventory.update(b"\n")
    return inventory.hexdigest()


def _audio_generation(audio: Audio) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("source.json", "chunks/chunks.json"):
        path = audio.podcast_dir / name
        try:
            load_control_json(path)
        except FileNotFoundError:
            result[name] = None
        else:
            result[name] = _file_digest(path)
    result["prepared_audio_sha256"] = _prepared_audio_digest(audio)
    return result


def _generation_digest(
    audio: Audio, *, budget: TranscriptWorkBudget | None = None
) -> str:
    # Hash the derived allowlisted inventory as well as its bytes: missing chunks
    # or exports cannot silently disappear from a restored generation.
    budget = budget or TranscriptWorkBudget()
    files = [*_transcript_files(audio, budget=budget), audio.cache_metadata_path]
    inventory = [
        (
            path.relative_to(audio.podcast_dir).as_posix(),
            _file_digest(path, budget=budget),
        )
        for path in files
    ]
    return hashlib.sha256(json.dumps(inventory).encode()).hexdigest()


def restore_last_success(url: str) -> dict[str, Path]:
    """Restore an admitted backup locally, with no source preparation or backend."""
    audio = Audio(base_dir=settings.transcript_dir, url=url)
    with _episode_lock(audio):
        budget = TranscriptWorkBudget()
        recovery = _generation_at(audio, audio.podcast_dir / ".previous-transcripts")
        if recovery.podcast_dir.is_symlink() or not recovery.podcast_dir.is_dir():
            raise RuntimeError("No admitted last-success recovery generation.")
        if recovery.episode_chunks_dir.is_symlink():
            raise RuntimeError("Recovery chunks must not be a symlink.")
        manifest = load_control_json(recovery.podcast_dir / "recovery.json")
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"version", "audio", "inventory_sha256"}
            or type(manifest["version"]) is not int
            or manifest["version"] != 2
            or manifest["audio"] != _audio_generation(audio)
            or manifest["inventory_sha256"]
            != _generation_digest(recovery, budget=budget)
        ):
            raise RuntimeError(
                "Incomplete recovery generation or changed prepared audio."
            )
        # Own the copied bytes before schema validation/publication. Reopening the
        # recovery paths after their digest check can otherwise read a new generation.
        with tempfile.TemporaryDirectory(
            prefix=".replacement-", dir=audio.podcast_dir
        ) as directory:
            snapshot = _generation_at(audio, Path(directory))
            _mark_stage(audio, snapshot.podcast_dir)
            for path in [
                *_transcript_files(recovery, budget=budget),
                recovery.cache_metadata_path,
            ]:
                copy_file_atomic(
                    path,
                    snapshot.podcast_dir / path.relative_to(recovery.podcast_dir),
                    budget=budget,
                )
            if (
                _generation_digest(snapshot, budget=budget)
                != manifest["inventory_sha256"]
            ):
                raise RuntimeError("Recovery generation changed while copying.")
            metadata = load_control_json(snapshot.cache_metadata_path)
            if not isinstance(metadata, dict) or not isinstance(
                metadata.get("backend"), str
            ):
                raise RuntimeError("Invalid recovery cache metadata.")
            load_dote(snapshot.output_path(".dote.json"))
            for path in _transcript_files(snapshot, budget=budget):
                if path.parent == snapshot.episode_chunks_dir:
                    if path.name.endswith(".dote.json"):
                        load_dote(path)
                    elif not path.name.endswith(".whisper-cpp.json"):
                        load_whisper_transcript(path)
            if manifest["audio"] != _audio_generation(audio):
                raise RuntimeError("Changed prepared audio during recovery.")
            # Keep recovery untouched, even if a later copy/fsync fails.
            _publish_transcripts(
                audio,
                snapshot,
                new_files=_transcript_files(snapshot, budget=budget),
                old_files=_transcript_files(audio, budget=budget),
                budget=budget,
            )
        return _transcript_paths(audio)


STAGING_PREFIXES = (".replacement-", ".backup-")
STAGE_MANIFEST = ".stage-owner.json"


def _stage_manifest(audio: Audio, name: str, info: os.stat_result) -> dict:
    return {
        "version": 1,
        "episode": audio.cache_key,
        "name": name,
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def _mark_stage(audio: Audio, path: Path) -> None:
    write_text_atomic(
        path / STAGE_MANIFEST,
        json.dumps(_stage_manifest(audio, path.name, path.stat())),
    )


def _remove_abandoned_staging(audio: Audio) -> None:
    """Reclaim only admitted stages, with all deletion anchored to their descriptor."""
    if not shutil.rmtree.avoids_symlink_attacks:
        return  # Keep ambiguous stages on platforms without fd-based deletion.
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        parent = os.open(audio.podcast_dir, flags)
    except FileNotFoundError:
        return
    try:
        with os.scandir(parent) as entries:
            names = [
                entry.name
                for entry in entries
                if entry.name.startswith(STAGING_PREFIXES)
            ]
        for name in names:
            try:
                descriptor = os.open(name, flags, dir_fd=parent)
            except OSError:
                continue
            try:
                info = os.fstat(descriptor)
                expected = _stage_manifest(audio, name, info)
                try:
                    manifest = load_control_json(
                        Path(STAGE_MANIFEST), dir_fd=descriptor
                    )
                except (OSError, ValueError, RuntimeError):
                    continue
                if manifest != expected or any(
                    type(manifest[key]) is not type(value)
                    for key, value in expected.items()
                ):
                    continue
                named = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if not os.path.samestat(info, named):
                    continue
                # Even an exchange after this check cannot redirect traversal:
                # child deletion uses the verified opened stage, never its path.
                with os.scandir(descriptor) as entries:
                    children = list(entries)
                for child in children:
                    if child.is_dir(follow_symlinks=False):
                        _remove_tree_at(child.name, descriptor)
                    else:
                        os.unlink(child.name, dir_fd=descriptor)
                named = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if os.path.samestat(info, named):
                    os.rmdir(name, dir_fd=parent)
                    os.fsync(parent)
            except FileNotFoundError:
                continue  # An exchanged/missing name is never repaired.
            finally:
                os.close(descriptor)
    finally:
        os.close(parent)


def _remove_tree_at(name: str, dir_fd: int) -> None:
    """Remove a directory below an opened parent without following symlinks.

    ``shutil.rmtree`` only accepts ``dir_fd`` from Python 3.11; on 3.10 walk the
    tree through descriptors opened with ``O_NOFOLLOW`` instead.
    """
    if sys.version_info >= (3, 11):
        shutil.rmtree(name, dir_fd=dir_fd)
        return
    descriptor = os.open(
        name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd
    )
    try:
        with os.scandir(descriptor) as entries:
            children = list(entries)
        for child in children:
            if child.is_dir(follow_symlinks=False):
                _remove_tree_at(child.name, descriptor)
            else:
                os.unlink(child.name, dir_fd=descriptor)
    finally:
        os.close(descriptor)
    os.rmdir(name, dir_fd=dir_fd)


def _current_generation_readable(audio: Audio) -> bool:
    """Whether the published generation can be kept until its replacement is ready."""
    budget = TranscriptWorkBudget()
    try:
        load_dote(audio.output_path(".dote.json"))
        for path in [
            *_transcript_files(audio, budget=budget),
            audio.cache_metadata_path,
        ]:
            for _ in regular_file_blocks(path, budget=budget):
                pass
    except (OSError, ValueError):
        return False
    return True


def _transcribe_locked(audio: Audio, backend: TranscriptionBackend) -> dict[str, Path]:
    audio.validate_control_files()
    _remove_abandoned_staging(audio)
    audio_chunks = audio.prepare_audio_for_transcription()
    # Probe before transcription or invalidating outputs, so malformed audio
    # cannot publish transcripts or discard the previous run's combined files.
    chunk_durations = [audio.chunk_duration(chunk) for chunk in audio_chunks]
    metadata = backend_cache_metadata(backend)
    try:
        cached = load_control_json(audio.cache_metadata_path)
    except FileNotFoundError:
        cached = None
    provenance = _chunk_byte_provenance(audio, audio_chunks)
    if (
        isinstance(cached, dict)
        and "chunk_sha256" in cached
        and cached["chunk_sha256"] != provenance
    ):
        raise RuntimeError(
            "Prepared chunk bytes changed; previous transcripts were preserved. "
            "Restore the prepared audio or move the episode cache aside and retry."
        )
    metadata["chunk_sha256"] = provenance

    def validate_audio(name: str | None) -> None:
        chunks = [chunk for chunk in audio_chunks if name is None or chunk.name == name]
        expected = {chunk.name: provenance[chunk.name] for chunk in chunks}
        if _chunk_byte_provenance(audio, chunks) != expected:
            raise RuntimeError(
                "Prepared chunk bytes changed during transcription; retry."
            )

    if (
        isinstance(cached, dict)
        and cached != metadata
        and all(
            audio.output_path(suffix).is_file() for suffix in TRANSCRIPT_OUTPUT_SUFFIXES
        )
        # An unreadable or corrupt generation is not worth retaining; replace it
        # in place as before instead of failing every later run.
        and _current_generation_readable(audio)
    ):
        # Bind the recovery copy to the prepared audio before inference, so an
        # unhashable inventory fails fast instead of after every replacement.
        audio_generation = _audio_generation(audio)
        # Do all inference and export work without touching the current generation.
        with tempfile.TemporaryDirectory(
            prefix=".replacement-", dir=audio.podcast_dir
        ) as directory:
            _mark_stage(audio, Path(directory))
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
                staged,
                backend,
                staged_chunks,
                chunk_durations,
                metadata,
                validate_audio=validate_audio,
            )
            _promote_transcripts(audio, staged, audio_generation)
        return _transcript_paths(audio)
    return _transcribe_prepared(
        audio,
        backend,
        audio_chunks,
        chunk_durations,
        metadata,
        validate_audio=validate_audio,
    )


def _transcript_paths(audio: Audio) -> dict[str, Path]:
    return {
        TRANSCRIPT_OUTPUT_LABELS[suffix]: audio.output_path(suffix)
        for suffix in TRANSCRIPT_OUTPUT_SUFFIXES
    }


def _transcript_files(
    audio: Audio, *, budget: TranscriptWorkBudget | None = None
) -> list[Path]:
    budget = budget or TranscriptWorkBudget()
    chunks = []
    try:
        with os.scandir(audio.episode_chunks_dir) as entries:
            for entry in entries:
                budget.admit_entry()  # Even irrelevant entries cost directory work.
                if GENERATED_TRANSCRIPT_RE.fullmatch(entry.name):
                    chunks.append(audio.episode_chunks_dir / entry.name)
    except FileNotFoundError:
        pass
    budget.check()
    return [
        audio.output_path(suffix) for suffix in TRANSCRIPT_OUTPUT_SUFFIXES
    ] + sorted(chunks)


def _chunk_directory_generation(audio: Audio) -> tuple[int, ...] | None:
    try:
        return file_generation(audio.episode_chunks_dir.stat())
    except FileNotFoundError:
        return None


def _promote_transcripts(
    audio: Audio, staged: Audio, audio_generation: dict[str, str | None]
) -> None:
    if _audio_generation(audio) != audio_generation:
        raise RuntimeError("Prepared audio changed during transcription; retry.")
    # Admit both inventories before replacing either recovery or public metadata.
    budget = TranscriptWorkBudget()
    old_generation = _chunk_directory_generation(audio)
    new_generation = _chunk_directory_generation(staged)
    old_files = _transcript_files(audio, budget=budget)
    new_files = _transcript_files(staged, budget=budget)
    # Keep a durable recovery copy before touching any currently published file.
    recovery = audio.podcast_dir / ".previous-transcripts"
    with tempfile.TemporaryDirectory(
        prefix=".backup-", dir=audio.podcast_dir
    ) as directory:
        backup = Path(directory)
        _mark_stage(audio, backup)
        for path in [*old_files, audio.cache_metadata_path]:
            target = backup / path.relative_to(audio.podcast_dir)
            copy_file_atomic(path, target, budget=budget)
        backup_audio = _generation_at(audio, backup)
        write_text_atomic(
            backup / "recovery.json",
            json.dumps(
                {
                    "version": 2,
                    "audio": audio_generation,
                    "inventory_sha256": _generation_digest(backup_audio, budget=budget),
                }
            ),
        )
        sync_directory(backup)
        budget.check()
        # The lock covers cooperating writers. Detect external changes to the
        # enumerated names during admission or backup before committing recovery.
        if old_generation != _chunk_directory_generation(
            audio
        ) or new_generation != _chunk_directory_generation(staged):
            raise RuntimeError("Transcript inventory changed during backup; retry.")
        if recovery.exists():
            shutil.rmtree(recovery)
        os.replace(backup, recovery)
        sync_directory(audio.podcast_dir)

    _publish_transcripts(
        audio, staged, new_files=new_files, old_files=old_files, budget=budget
    )


def _publish_transcripts(
    audio: Audio,
    staged: Audio,
    *,
    new_files: list[Path],
    old_files: list[Path],
    budget: TranscriptWorkBudget,
) -> None:
    # Metadata is the completion claim: a partial multi-file promotion must never
    # make either old or mixed bytes reusable as the replacement generation.
    budget.check()
    durable_unlink(audio.cache_metadata_path)
    targets = {
        audio.podcast_dir / path.relative_to(staged.podcast_dir) for path in new_files
    }
    for path in new_files:
        target = audio.podcast_dir / path.relative_to(staged.podcast_dir)
        copy_file_atomic(path, target, budget=budget)
    for stale in old_files:
        budget.check()
        if stale not in targets:
            durable_unlink(stale)
    copy_file_atomic(
        staged.cache_metadata_path, audio.cache_metadata_path, budget=budget
    )


def _transcribe_prepared(
    audio: Audio,
    backend: TranscriptionBackend,
    audio_chunks: list[Path],
    chunk_durations: list[float],
    metadata: dict,
    *,
    validate_audio: Callable[[str | None], None],
) -> dict[str, Path]:
    audio.ensure_transcript_cache_matches(metadata)

    def validate_inputs(chunk: Path) -> None:
        validate_audio(chunk.name)
        if "local_model_sha256" in metadata and backend_cache_metadata(backend) != {
            key: value for key, value in metadata.items() if key != "chunk_sha256"
        }:
            raise RuntimeError(
                "Local model changed during transcription; retry with a stable model."
            )

    text_chunks = audio_chunks_to_text(
        backend, audio_chunks, validate_chunk=validate_inputs
    )
    validate_audio(None)
    dote_chunks = whisper_text_chunks_to_dote(text_chunks)
    dote_path = audio.output_path(".dote.json")
    if not dote_path.exists():
        combine_dote_chunks(dote_chunks, dote_path, chunk_durations)
    podlove_path = audio.output_path(".podlove.json")
    if not podlove_path.exists():
        convert_dote_to_podlove(dote_path, podlove_path)
    webvtt_path = audio.output_path(".webvtt")
    if not webvtt_path.exists():
        convert_to_webvtt(dote_path, webvtt_path)
    plaintext_path = audio.output_path(".txt")
    if not plaintext_path.exists():
        convert_to_plaintext(dote_path, plaintext_path)
    return _transcript_paths(audio)
