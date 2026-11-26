import hashlib
import json
import math
from collections.abc import Iterator
from contextlib import closing, contextmanager, nullcontext
from itertools import islice
import os
import re
import httpx
import stat
import shutil
import tempfile
import uuid

from pathlib import Path
from time import monotonic
from urllib.parse import unquote, urlparse

from rich import print as rprint

from .config import private_mkdir, settings
from .input_admission import file_generation
from .json_input import load_control_json
from .safe_io import (
    atomic_output,
    capture_checked,
    run_checked,
    write_text_atomic,
    write_complete,
    durable_replace,
    durable_unlink,
    sync_file,
    sync_directory,
)

FFPROBE_TIMEOUT_SECONDS = 60.0
MODEL_MAX_BYTES = 64 * 1024**3
MODEL_HASH_TIMEOUT_SECONDS = 3600.0
MODEL_MAX_ENTRIES = 100_000
RESAMPLED_MAX_BYTES = 2 * 1024**3


class ModelBudget:
    """One budget for enumeration and all reads in a model identity pass."""

    def __init__(self):
        self.remaining = MODEL_MAX_BYTES
        self.deadline = monotonic() + MODEL_HASH_TIMEOUT_SECONDS
        self.entries = 0

    def check(self):
        if monotonic() >= self.deadline:
            raise RuntimeError("Local model exceeded model hashing elapsed budget.")

    def admit_entry(self):
        self.check()
        self.entries += 1
        if self.entries > MODEL_MAX_ENTRIES:
            raise RuntimeError("Local model exceeded directory entry budget.")


GENERATED_CHUNK_RE = re.compile(r"chunk_\d{3,}\.(?:mp3|wav)")
# Per-chunk transcript files written by the transcription step: the backend's
# ``chunk_NNN.json``, whisper-cpp's intermediate ``chunk_NNN.whisper-cpp.json``
# and the derived DOTe chunk and its receipt.
GENERATED_TRANSCRIPT_RE = re.compile(
    r"chunk_\d{3,}\.(?:json|whisper-cpp\.json|dote\.json|dote\.receipt\.json)"
)
CACHE_KEY_HASH_LENGTH = 12
TRANSCRIPT_OUTPUT_LABELS = {
    ".dote.json": "DOTe",
    ".podlove.json": "podlove",
    ".webvtt": "WebVTT",
    ".txt": "plain text",
}
TRANSCRIPT_OUTPUT_SUFFIXES = tuple(TRANSCRIPT_OUTPUT_LABELS)
# Preserve cache paths generated under the original 213-byte filename limit.
CACHE_FILENAME_MAX_BYTES = 213
READABLE_PREFIX_MAX_BYTES = CACHE_FILENAME_MAX_BYTES - max(
    len(suffix) for suffix in (*TRANSCRIPT_OUTPUT_SUFFIXES, "_16khz.mp3")
)


def _byte_prefix(value: str, maximum: int) -> str:
    """Shorten at a UTF-8 boundary, retaining short existing cache names."""
    return value.encode("utf-8")[:maximum].decode("utf-8", errors="ignore")


def _source_filename(name: str) -> str:
    if len(name.encode("utf-8")) <= CACHE_FILENAME_MAX_BYTES:
        return name
    suffix = Path(name).suffix
    # Only keep a bounded extension; unusual long extensions are readable text.
    if len(suffix.encode("utf-8")) > 32:
        suffix = ""
    stem = name[: -len(suffix)] if suffix else name
    return (
        _byte_prefix(stem, CACHE_FILENAME_MAX_BYTES - len(suffix.encode("utf-8")))
        + suffix
    )


def is_url(url: str) -> bool:
    return urlparse(url).scheme.lower() in {"http", "https"}


def safe_display_source(source: str) -> str:
    """Keep local paths, but omit URL credentials, queries and fragments."""
    try:
        url = urlparse(source)
        if url.scheme.lower() in {"http", "https"}:
            return url._replace(
                netloc=url.netloc.rsplit("@", 1)[-1], query="", fragment=""
            ).geturl()
    except ValueError:
        return "<invalid source URL>"
    return source


def safe_diagnostic(message: str) -> str:
    return re.sub(
        r"https?://[^\s\"'<>]+",
        lambda match: safe_display_source(match.group()),
        message,
        flags=re.IGNORECASE,
    )


def get_url_filename(url: str) -> str:
    """A single safe filename from the URL path, excluding query and fragment."""
    name = unquote(urlparse(url).path.rsplit("/", 1)[-1])
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name).strip(".")
    return name or "episode.mp3"


def get_title_from_string(url: str) -> str:
    if is_url(url):
        return get_url_filename(url).split(".")[0]
    else:
        return Path(url).stem


def get_cache_source(url: str) -> str:
    """
    Return the string that identifies an episode's source for caching: the
    full URL (including query string) for URLs, the absolute path for files.
    """
    if is_url(url):
        return url
    return str(Path(url).expanduser().resolve())


def get_cache_key(url: str) -> str:
    """
    Return the cache directory name for an episode: a readable stem plus a
    short hash of the full URL or absolute path. Two episodes that share a
    file name (``.../123/audio.mp3`` and ``.../124/audio.mp3``) or the part
    before the first dot (``ep1.final.mp3`` and ``ep1.draft.mp3``) get
    different directories.
    """
    stem = _byte_prefix(
        get_title_from_string(url) or "episode", READABLE_PREFIX_MAX_BYTES
    )
    digest = hashlib.sha256(get_cache_source(url).encode("utf-8")).hexdigest()
    return f"{stem}-{digest[:CACHE_KEY_HASH_LENGTH]}"


def download(url: str, target_path: Path) -> None:
    """
    Download ``url`` to ``target_path``.

    Redirects are followed (enclosure URLs usually redirect through tracking
    prefixes or CDNs) and HTTP errors raise. The body is streamed to a
    temporary file that is only renamed to ``target_path`` after the whole
    download succeeded, so a failed download is never cached as the episode.
    """
    max_bytes = settings.source_max_bytes
    total_seconds = settings.source_total_timeout_seconds
    deadline = monotonic() + total_seconds

    def check_deadline() -> None:
        if monotonic() >= deadline:
            raise RuntimeError(
                "Source download exceeded TRANSCRIPT_SOURCE_TOTAL_TIMEOUT."
            )

    def check_response_deadline(_response: httpx.Response) -> None:
        check_deadline()

    def check_final_encoding(response: httpx.Response) -> None:
        encoding = response.headers.get("Content-Encoding", "identity")
        if encoding.strip().lower() != "identity":
            raise RuntimeError(
                "Source download requires identity Content-Encoding; "
                "compressed responses are not supported."
            )

    @contextmanager
    def stream_final_response(client: httpx.Client) -> Iterator[httpx.Response]:
        request = client.build_request("GET", url)
        for hop in range(client.max_redirects + 1):
            check_deadline()
            # HTTPX already applied its redirect header rules to next_request.
            # Avoid running URL authentication again on that prepared request.
            response = (
                client.send(request, stream=True, auth=httpx.Auth())
                if hop
                else client.send(request, stream=True)
            )
            with closing(response):
                if response.next_request is None:
                    yield response
                    return
                request = response.next_request
        raise httpx.TooManyRedirects(
            "Exceeded maximum allowed redirects.", request=request
        )

    # Each blocking operation is also bounded, even for very short total budgets.
    timeout = httpx.Timeout(min(60.0, total_seconds), connect=min(30.0, total_seconds))
    rprint(f"Downloading {safe_display_source(url)} to {target_path}")
    try:
        with atomic_output(target_path) as tmp_path:
            with httpx.Client(
                follow_redirects=False,
                timeout=timeout,
                event_hooks={"response": [check_response_deadline]},
                headers={"Accept-Encoding": "identity"},
            ) as client:
                with stream_final_response(client) as response:
                    check_final_encoding(response)
                    response.raise_for_status()
                    check_deadline()
                    if (
                        response.status_code != 200
                        or "Content-Range" in response.headers
                    ):
                        raise RuntimeError(
                            "Source download requires a complete HTTP 200 response "
                            "without Content-Range."
                        )
                    lengths = response.headers.get_list("Content-Length")
                    declared_bytes = None
                    if lengths:
                        if len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]):
                            raise RuntimeError(
                                "Source download requires one decimal Content-Length."
                            )
                        length = lengths[0].lstrip("0") or "0"
                        try:
                            declared_bytes = int(length)
                        except ValueError:
                            # A decimal too large for int also exceeds the byte budget.
                            raise RuntimeError(
                                "Source download exceeded TRANSCRIPT_SOURCE_MAX_BYTES."
                            ) from None
                        if declared_bytes > max_bytes:
                            raise RuntimeError(
                                "Source download exceeded TRANSCRIPT_SOURCE_MAX_BYTES."
                            )
                    received = 0
                    with tmp_path.open("wb") as file:
                        for data in response.iter_bytes():
                            check_deadline()
                            received += len(data)
                            if received > max_bytes:
                                raise RuntimeError(
                                    "Source download exceeded TRANSCRIPT_SOURCE_MAX_BYTES."
                                )
                            write_complete(file.write, data)
                    check_deadline()
                    if declared_bytes is not None and received != declared_bytes:
                        raise RuntimeError(
                            "Source download size does not match Content-Length."
                        )
                    if received == 0:
                        raise RuntimeError("Source download requires nonempty audio.")
    except httpx.HTTPError as exc:
        detail = (
            f"status {exc.response.status_code}"
            if isinstance(exc, httpx.HTTPStatusError)
            else type(exc).__name__
        )
        raise RuntimeError(
            f"Source download failed ({detail}) for {safe_display_source(url)}"
        ) from None


def get_audio_duration(path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",  # Suppress unnecessary output
        "-show_entries",
        "format=duration",  # Show only the duration
        "-of",
        "json",  # Output in JSON format for easy parsing
        path,
    ]
    try:
        output = capture_checked(
            cmd, description=f"ffprobe for {path}", timeout=FFPROBE_TIMEOUT_SECONDS
        )
    except OSError as exc:
        raise RuntimeError(f"Could not run ffprobe for {path}") from exc
    try:
        metadata = json.loads(output)
        raw = metadata["format"]["duration"]
        if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
            raise ValueError("invalid duration type")
        duration = float(raw)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration must be finite and nonnegative")
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise RuntimeError(
            f"ffprobe returned invalid duration metadata for {path}"
        ) from exc
    return duration


def resample_audio(input_path: Path, output_path: Path) -> None:
    rprint(f"Resampling {input_path} to {output_path}")
    # resample the audio file to 16khz
    with atomic_output(output_path) as tmp_path:
        run_checked(
            [
                "ffmpeg",
                "-i",
                str(input_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-map",
                "0:a:0",
                "-fs",
                str(RESAMPLED_MAX_BYTES),
                str(tmp_path),
            ],
            description=f"Resampling {input_path}",
            output_limit=(tmp_path, RESAMPLED_MAX_BYTES),
        )
        # ffmpeg may exit successfully on -fs and can overshoot by a packet.
        # Neither a limit-truncated result nor an oversized result is reusable.
        if tmp_path.stat().st_size >= RESAMPLED_MAX_BYTES:
            raise RuntimeError("Resampling reached output byte limit (2 GiB).")


# 25MB max bytes are allowed
MAX_SIZE_IN_BYTES = 25 * 1024 * 1024
MAX_CHUNK_DURATION = 7200
# Leave room for per-chunk SHA-256 digests in the 64 KiB cache control.
MAX_CHUNK_COUNT = 512


class Audio:
    def __init__(self, *, base_dir: Path, url: str, title: str | None = None):
        self._probed_durations: dict[tuple[int, ...], float] = {}
        self.base_dir = base_dir
        self.url = url
        self._is_http_url = is_url(url)
        source_title = get_title_from_string(url) or "episode"
        self.prefix = _byte_prefix(source_title, READABLE_PREFIX_MAX_BYTES)
        if title is not None:
            self.title = title
        else:
            self.title = source_title
        self.cache_key = get_cache_key(url)
        self.podcast_dir = base_dir / self.cache_key
        self.episode_chunks_dir = self.podcast_dir / "chunks"

    def __repr__(self):
        return self.title

    @property
    def episode_path(self):
        name = get_url_filename(self.url) if self._is_http_url else Path(self.url).name
        # Check the stored (byte-shortened) name: shortening can turn a long,
        # distinct source name into a reserved one.
        stored = _source_filename(name)
        reserved = stored.casefold()
        if (
            reserved == "chunks.json"
            or GENERATED_CHUNK_RE.fullmatch(reserved)
            or GENERATED_TRANSCRIPT_RE.fullmatch(reserved)
            or reserved == self.resampled_episode_path.name.casefold()
        ):
            # Source bytes must never be parsed, overwritten or retired as controls,
            # generated chunk audio, native intermediates or raw/derived transcripts.
            stored = _source_filename(f"episode_{name}")
        return self.episode_chunks_dir / stored

    def validate_storage_roles(self) -> None:
        """Preflight source, prepared, control and transcript namespaces."""
        paths = [
            self.episode_path,
            self.resampled_episode_path,
            self.chunks_manifest_path,
            self.cache_metadata_path,
            self.source_metadata_path,
            *(self.output_path(suffix) for suffix in TRANSCRIPT_OUTPUT_SUFFIXES),
        ]
        identities = [str(path.absolute()).casefold() for path in paths]
        if len(identities) != len(set(identities)):
            raise RuntimeError("Source and generated storage roles overlap.")

    @property
    def resampled_episode_path(self):
        return self.episode_chunks_dir / f"{self.prefix}_16khz.mp3"

    def output_path(self, suffix: str) -> Path:
        """Path of a combined transcript output, e.g. ``output_path(".txt")``."""
        return self.podcast_dir / f"{self.prefix}{suffix}"

    @property
    def cache_metadata_path(self) -> Path:
        return self.podcast_dir / "cache.json"

    def ensure_transcript_cache_matches(self, metadata: dict) -> None:
        """
        Make sure cached transcripts were produced with ``metadata`` (backend,
        model, language, prompt). If ``cache.json`` is missing or records
        different settings, the per-chunk transcripts and the combined outputs
        are deleted so they are transcribed again; the downloaded, resampled
        and chunked audio is kept. ``cache.json`` is written only after the
        stale transcripts are gone. The episode pipeline calls this on private
        staging when replacing a completed generation; it promotes metadata last.
        """
        self.validate_storage_roles()
        try:
            cached = load_control_json(self.cache_metadata_path, expected_type=dict)
        except FileNotFoundError:
            cached = None
        if cached == metadata:
            return
        if cached is not None:
            rprint("Transcription settings changed, discarding cached transcripts")
        self._remove_cached_transcripts()
        private_mkdir(self.podcast_dir)
        write_text_atomic(
            self.cache_metadata_path, json.dumps(metadata, sort_keys=True)
        )

    def _remove_cached_transcripts(self) -> None:
        # Retire the byte-provenance claim before replacing a prepared generation.
        durable_unlink(self.cache_metadata_path)
        keep = {self.episode_path, self.resampled_episode_path}
        if self.episode_chunks_dir.is_dir():
            for path in self.episode_chunks_dir.glob("chunk_*"):
                if path in keep or not GENERATED_TRANSCRIPT_RE.fullmatch(path.name):
                    continue
                path.unlink()
        for suffix in TRANSCRIPT_OUTPUT_SUFFIXES:
            self.output_path(suffix).unlink(missing_ok=True)

    def make_sure_audio_file_exists(self) -> None:
        """
        Make sure the audio file for the episode exists in the local filesystem.
        """
        if not self.episode_path.exists():
            private_mkdir(self.episode_path.parent)
            if self._is_http_url:
                download(self.url, self.episode_path)
            else:
                with atomic_output(self.episode_path) as tmp_path:
                    self._read_local_source(Path(get_cache_source(self.url)), tmp_path)

    def make_sure_audio_file_is_resampled(self) -> None:
        """
        Make sure the audio file is resampled to 16khz.
        """
        self.validate_storage_roles()
        rprint("make sure audio file is resampled: ", self.resampled_episode_path)
        if not self.resampled_episode_path.exists():
            resample_audio(self.episode_path, self.resampled_episode_path)
        if self.resampled_episode_path.stat().st_size >= RESAMPLED_MAX_BYTES:
            raise RuntimeError("Resampled audio reached output byte limit (2 GiB).")

    def _probe_duration(self, path: Path) -> float:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError("Prepared audio must be a regular file.")
        generation = file_generation(info)
        if generation in self._probed_durations:
            return self._probed_durations[generation]
        duration = get_audio_duration(path)
        if generation != file_generation(path.stat()):
            raise RuntimeError("Prepared audio changed during its duration probe.")
        self._probed_durations[generation] = duration
        return duration

    def chunk_duration(self, path: Path) -> float:
        """Admit a prepared chunk for transcription and return its timing."""
        duration = self._probe_duration(path)
        if not 0 < duration <= MAX_CHUNK_DURATION:
            raise RuntimeError(
                f"Prepared chunk {path.name} has a duration outside the allowed "
                f"range (greater than zero and at most {MAX_CHUNK_DURATION} seconds). "
                "Previous transcripts were preserved."
            )
        return duration

    def _publish_prepared_file(self, source: Path, target: Path) -> None:
        # Read the link itself, since only its regular target was probed.
        original = file_generation(source.lstat())
        duration = self._probed_durations.pop(original, None)
        durable_replace(source, target)
        if duration is not None:
            published = file_generation(target.stat())
            # Renaming an owned file can change its ctime. Transfer the probe
            # only if the inode, size and mtime still match across publication.
            if original[:-1] == published[:-1]:
                self._probed_durations[published] = duration

    @property
    def exceeds_size_limit(self) -> bool:
        size = self.resampled_episode_path.stat().st_size
        if size == 0:
            raise RuntimeError("Resampled audio is empty.")
        duration = self._probe_duration(self.resampled_episode_path)
        if duration <= 0:
            raise RuntimeError("Resampled audio has zero duration.")
        return size > MAX_SIZE_IN_BYTES or duration > MAX_CHUNK_DURATION

    def split_into_chunks(self) -> list[Path]:
        """
        If the audio file exceeds the size limit, split it into smaller chunks.
        If not, just create a link to the resampled audio file.
        """
        self.validate_storage_roles()
        cached = self._load_chunk_manifest()
        if cached is not None:
            return cached
        exceeds_limit = self.exceeds_size_limit
        # No valid manifest: either an earlier split was interrupted while the
        # chunks were moved into place, or the cache predates the manifest.
        # Remove leftover chunk audio (and WAV files derived from it) and
        # split again so a partial chunk set is never reused.
        if self.chunks_manifest_path.exists():
            self._remove_cached_transcripts()
        durable_unlink(self.chunks_manifest_path)
        self._remove_generated_chunks()
        if exceeds_limit:
            rprint(f"Splitting {self.resampled_episode_path} into chunks")
            chunk_names = self._split_resampled_episode()
        else:
            sync_file(self.resampled_episode_path)
            sync_directory(self.episode_chunks_dir)
            rprint(f"Creating symlink to {self.resampled_episode_path}")
            with atomic_output(self.episode_chunks_dir / "chunk_000.mp3") as tmp_path:
                tmp_path.symlink_to(self.resampled_episode_path.name)
            chunk_names = ["chunk_000.mp3"]
        # The manifest is written last; it marks the chunk set as complete.
        write_text_atomic(self.chunks_manifest_path, json.dumps(chunk_names))
        return [self.episode_chunks_dir / name for name in chunk_names]

    def _remove_generated_chunks(self) -> None:
        """
        Remove generated ``chunk_NNN.mp3`` files and legacy ``chunk_NNN.wav``
        conversions. The downloaded and resampled episode files are never
        touched, even if their names look similar.
        """
        keep = {self.episode_path, self.resampled_episode_path}
        for path in self.episode_chunks_dir.glob("chunk_*"):
            if path in keep or not GENERATED_CHUNK_RE.fullmatch(path.name):
                continue
            path.unlink()

    @property
    def chunks_manifest_path(self) -> Path:
        return self.episode_chunks_dir / "chunks.json"

    def _is_owned_short_chunk_link(self, path: Path, *, chunk_count: int) -> bool:
        resampled = self.resampled_episode_path
        try:
            return (
                chunk_count == 1
                and path.is_symlink()
                and not resampled.is_symlink()
                and path.resolve() == resampled.resolve()
            )
        except (OSError, RuntimeError):
            return False

    def _read_cached_chunk_names(self) -> list | None:
        try:
            chunk_names = load_control_json(
                self.chunks_manifest_path, expected_type=list
            )
        except FileNotFoundError:
            return None
        if len(chunk_names) > MAX_CHUNK_COUNT:
            raise RuntimeError(
                f"Cache control chunks/chunks.json exceeds the limit of {MAX_CHUNK_COUNT} chunks; "
                "move the episode cache aside and split the source into shorter files."
            )
        return chunk_names

    def _load_chunk_manifest(self) -> list[Path] | None:
        """Return the cached chunk paths if a complete chunk set exists."""
        chunk_names = self._read_cached_chunk_names()
        if not chunk_names:
            return None
        # Exact generated names enforce type, containment, uniqueness and order.
        if chunk_names != [
            f"chunk_{index:03d}.mp3" for index in range(len(chunk_names))
        ]:
            return None
        chunk_paths = [self.episode_chunks_dir / name for name in chunk_names]
        try:
            for path in chunk_paths:
                if (
                    not path.is_file()
                    or not 0 < path.stat().st_size <= MAX_SIZE_IN_BYTES
                ):
                    return None
                if path.is_symlink() and not self._is_owned_short_chunk_link(
                    path, chunk_count=len(chunk_paths)
                ):
                    return None
        except (OSError, RuntimeError):
            return None
        return chunk_paths

    def _split_resampled_episode(self) -> list[str]:
        """
        Split the resampled episode into a hidden scratch directory and only
        move the chunks into place once ffmpeg succeeded. Returns the chunk
        file names; the caller records them in the manifest.
        """
        duration = self._probe_duration(self.resampled_episode_path)
        if duration <= 0:
            raise RuntimeError(
                "Cannot split audio with zero duration to the upload limit"
            )
        size = self.resampled_episode_path.stat().st_size
        # Estimate a byte-safe segment duration; leave headroom for MP3 frames,
        # headers and VBR variation. Actual output limits are checked below.
        segment_seconds = 0.9 * min(
            MAX_CHUNK_DURATION, duration * MAX_SIZE_IN_BYTES / max(size, 1)
        )
        private_mkdir(self.episode_chunks_dir)
        split_dir = self.episode_chunks_dir / f".split-{uuid.uuid4().hex}"
        private_mkdir(split_dir)
        try:
            run_checked(
                [
                    "ffmpeg",
                    "-i",
                    str(self.resampled_episode_path),
                    "-f",
                    "segment",
                    "-segment_time",
                    str(segment_seconds),
                    "-c",
                    "copy",
                    str(split_dir / "chunk_%03d.mp3"),
                ],
                description=f"Splitting {self.resampled_episode_path} into chunks",
            )
            produced = list(islice(split_dir.glob("chunk_*.mp3"), MAX_CHUNK_COUNT + 1))
            if len(produced) > MAX_CHUNK_COUNT:
                raise RuntimeError(
                    f"Splitting produced more than {MAX_CHUNK_COUNT} chunks; "
                    "split the source into shorter files and retry."
                )
            if not produced:
                raise RuntimeError(
                    f"Splitting {self.resampled_episode_path} produced no chunks"
                )
            produced.sort(key=lambda path: int(path.stem.split("_")[1]))
            # Validate every new chunk before publishing audio or completion markers.
            for chunk in produced:
                size = chunk.stat().st_size
                if size == 0:
                    raise RuntimeError("Split chunk is empty.")
                chunk.chmod(0o600)
                chunk_duration = self._probe_duration(chunk)
                if chunk_duration <= 0:
                    raise RuntimeError("Split chunk has zero duration.")
                if chunk_duration > MAX_CHUNK_DURATION or size > MAX_SIZE_IN_BYTES:
                    raise RuntimeError(
                        "Split chunk exceeds the upload limit (25 MiB / 7200 seconds); "
                        "split the source into shorter files and retry. No chunk manifest was published."
                    )
            for chunk in produced:
                self._publish_prepared_file(chunk, self.episode_chunks_dir / chunk.name)
            return [chunk.name for chunk in produced]
        finally:
            shutil.rmtree(split_dir, ignore_errors=True)

    @staticmethod
    def _read_local_source(
        path: Path,
        target: Path | None = None,
        *,
        model: bool = False,
        model_budget: ModelBudget | None = None,
        dir_fd: int | None = None,
    ) -> str:
        """Hash/copy one admitted descriptor with source or model budgets."""
        budget = (model_budget or ModelBudget()) if model else None
        limit = budget.remaining if budget is not None else settings.source_max_bytes
        deadline = (
            budget.deadline
            if budget is not None
            else monotonic() + settings.source_total_timeout_seconds
        )
        label = "Local model" if model else "Local source"
        byte_hint = (
            "64 GiB model byte budget" if model else "TRANSCRIPT_SOURCE_MAX_BYTES"
        )
        time_hint = (
            "model hashing elapsed budget"
            if model
            else "TRANSCRIPT_SOURCE_TOTAL_TIMEOUT"
        )

        def check_deadline():
            if monotonic() >= deadline:
                raise RuntimeError(f"{label} exceeded {time_hint}.")

        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
        if model:
            flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, dir_fd=dir_fd)
        except OSError:
            if model:
                raise RuntimeError(
                    "Local model artifact could not be admitted."
                ) from None
            raise
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError(f"{label} must be a regular file.")
            if info.st_size > limit:
                raise RuntimeError(f"{label} exceeded {byte_hint}.")
            digest = hashlib.sha256()
            received = 0
            # Optional destination is owned staging, never an existing cache file.
            with target.open("wb") if target is not None else nullcontext() as output:
                while True:
                    check_deadline()
                    block = os.read(descriptor, min(1024 * 1024, limit - received + 1))
                    check_deadline()
                    if not block:
                        break
                    received += len(block)
                    if received > limit:
                        raise RuntimeError(f"{label} exceeded {byte_hint}.")
                    digest.update(block)
                    if budget is not None:
                        budget.remaining -= len(block)
                    if output is not None:
                        output.write(block)
            check_deadline()
            return digest.hexdigest()
        finally:
            os.close(descriptor)

    @staticmethod
    def _content_fingerprint(path: Path, *, source_budget: bool = True) -> str:
        return Audio._read_local_source(path, model=not source_budget)

    @property
    def source_metadata_path(self) -> Path:
        return self.podcast_dir / "source.json"

    def _local_source_changed(self) -> bool:
        source_path = Path(get_cache_source(self.url))
        try:
            fingerprint = self._content_fingerprint(source_path)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Local source audio is missing: {source_path}. "
                "Restore the source file before reusing its cache."
            ) from exc
        try:
            cached = load_control_json(self.source_metadata_path, expected_type=dict)
        except FileNotFoundError:
            cached = None
        if cached == {"sha256": fingerprint}:
            return False
        # An absent marker explicitly makes the preserved generation non-current.
        durable_unlink(self.source_metadata_path)
        return True

    def _prepare_changed_local_source(self) -> list[Path]:
        """Prepare fully before retiring any files from the previous generation."""
        with tempfile.TemporaryDirectory(
            prefix=".prepare-", dir=self.podcast_dir
        ) as directory:
            staged = Audio(base_dir=self.base_dir, url=self.url, title=self.title)
            staged.podcast_dir = Path(directory)
            staged.episode_chunks_dir = staged.podcast_dir / "chunks"
            staged.make_sure_audio_file_exists()
            staged.make_sure_audio_file_is_resampled()
            chunks = staged.split_into_chunks()
            fingerprint = self._content_fingerprint(staged.episode_path)

            # Preparation succeeded. A promotion interrupted before source.json
            # is written is retried in staging on the next invocation.
            durable_unlink(self.chunks_manifest_path)
            self._remove_cached_transcripts()
            self._remove_generated_chunks()
            private_mkdir(self.episode_chunks_dir)
            for path in [
                staged.episode_path,
                staged.resampled_episode_path,
                *chunks,
                staged.chunks_manifest_path,
            ]:
                staged._publish_prepared_file(path, self.episode_chunks_dir / path.name)
            write_text_atomic(
                self.source_metadata_path, json.dumps({"sha256": fingerprint})
            )
            self._probed_durations.update(staged._probed_durations)
            return [self.episode_chunks_dir / chunk.name for chunk in chunks]

    def validate_control_files(self) -> None:
        """Admit all present controls before preparation can retire a generation."""
        for path in (self.cache_metadata_path, self.source_metadata_path):
            try:
                load_control_json(path, expected_type=dict)
            except FileNotFoundError:
                pass
        self._read_cached_chunk_names()

    def prepare_audio_for_transcription(self) -> list[Path]:
        """
        Steps needed to prepare an audio file URL for transcription:
            - Download the audio file if it's a URL
            - Resample the audio file to 16khz
            - Split the audio file into smaller chunks if needed
        """
        self._probed_durations.clear()
        self.validate_storage_roles()
        self.validate_control_files()
        private_mkdir(self.podcast_dir)
        if not self._is_http_url and self._local_source_changed():
            return self._prepare_changed_local_source()
        self.make_sure_audio_file_exists()
        self.make_sure_audio_file_is_resampled()
        return self.split_into_chunks()
