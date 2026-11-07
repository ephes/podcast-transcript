import hashlib
import json
import math
import os
import re
import httpx
import shutil
import subprocess
import uuid

from pathlib import Path
from time import monotonic
from urllib.parse import unquote, urlparse

from rich import print as rprint

from .config import settings
from .safe_io import atomic_output, run_checked, write_text_atomic

FFPROBE_TIMEOUT_SECONDS = 60.0
FFPROBE_STDERR_LIMIT = 2000
GENERATED_CHUNK_RE = re.compile(r"chunk_\d{3,}\.(?:mp3|wav)")
# Per-chunk transcript files written by the transcription step: the backend's
# ``chunk_NNN.json``, whisper-cpp's intermediate ``chunk_NNN.whisper-cpp.json``
# and the ``chunk_NNN.dote.json`` derived from it.
GENERATED_TRANSCRIPT_RE = re.compile(
    r"chunk_\d{3,}\.(?:json|whisper-cpp\.json|dote\.json)"
)
CACHE_KEY_HASH_LENGTH = 12
TRANSCRIPT_OUTPUT_SUFFIXES = (".dote.json", ".podlove.json", ".webvtt", ".txt")


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
    stem = get_title_from_string(url) or "episode"
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

    def check_deadline(response: httpx.Response | None = None) -> None:
        if monotonic() >= deadline:
            raise RuntimeError(
                "Source download exceeded TRANSCRIPT_SOURCE_TOTAL_TIMEOUT."
            )

    # Each blocking operation is also bounded, even for very short total budgets.
    timeout = httpx.Timeout(min(60.0, total_seconds), connect=min(30.0, total_seconds))
    rprint(f"Downloading {safe_display_source(url)} to {target_path}")
    try:
        with atomic_output(target_path) as tmp_path:
            with httpx.Client(
                follow_redirects=True,
                timeout=timeout,
                event_hooks={"response": [check_deadline]},
            ) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    check_deadline()
                    length = response.headers.get("Content-Length")
                    if length is not None:
                        try:
                            declared_bytes = int(length)
                        except ValueError:
                            declared_bytes = (
                                0  # Enforce the actual byte count regardless.
                            )
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
                            file.write(data)
                    check_deadline()
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
    # subprocess.run kills and waits for the child if its timeout expires.
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            check=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        tail = stderr[-FFPROBE_STDERR_LIMIT:].strip()
        reason = "timed out" if isinstance(exc, subprocess.TimeoutExpired) else "failed"
        raise RuntimeError(f"ffprobe {reason} for {path}: {tail}") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not run ffprobe for {path}") from exc
    try:
        metadata = json.loads(result.stdout)
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
                "0:a:",
                str(tmp_path),
            ],
            description=f"Resampling {input_path}",
        )


# 25MB max bytes are allowed
MAX_SIZE_IN_BYTES = 25 * 1024 * 1024


class Audio:
    def __init__(self, *, base_dir: Path, url: str, title: str | None = None):
        self.base_dir = base_dir
        self.url = url
        self._is_http_url = is_url(url)
        self.prefix = get_title_from_string(url) or "episode"
        if title is not None:
            self.title = title
        else:
            self.title = self.prefix
        self.cache_key = get_cache_key(url)
        self.podcast_dir = base_dir / self.cache_key
        self.episode_chunks_dir = self.podcast_dir / "chunks"

    def __repr__(self):
        return self.title

    @property
    def episode_path(self):
        name = get_url_filename(self.url) if self._is_http_url else Path(self.url).name
        if GENERATED_CHUNK_RE.fullmatch(name):
            # Keep the source audio from colliding with generated chunk files
            name = f"episode_{name}"
        return self.episode_chunks_dir / name

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
        stale transcripts are gone.
        """
        try:
            cached = json.loads(self.cache_metadata_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            cached = None
        if cached == metadata:
            return
        if cached is not None:
            rprint(
                f"Transcription settings changed ({cached} -> {metadata}), "
                "discarding cached transcripts"
            )
        self._remove_cached_transcripts()
        self.podcast_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(
            self.cache_metadata_path, json.dumps(metadata, sort_keys=True)
        )

    def _remove_cached_transcripts(self) -> None:
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
            self.episode_path.parent.mkdir(exist_ok=True, parents=True)
            if self._is_http_url:
                download(self.url, self.episode_path)
            else:
                with atomic_output(self.episode_path) as tmp_path:
                    shutil.copy(Path(get_cache_source(self.url)), tmp_path)

    def make_sure_audio_file_is_resampled(self) -> None:
        """
        Make sure the audio file is resampled to 16khz.
        """
        rprint("make sure audio file is resampled: ", self.resampled_episode_path)
        if not self.resampled_episode_path.exists():
            resample_audio(self.episode_path, self.resampled_episode_path)

    @property
    def exceeds_size_limit(self) -> bool:
        too_many_bytes = self.resampled_episode_path.stat().st_size > MAX_SIZE_IN_BYTES
        too_long_duration = get_audio_duration(self.resampled_episode_path) > 7200
        return too_many_bytes or too_long_duration

    def split_into_chunks(self) -> list[Path]:
        """
        If the audio file exceeds the size limit, split it into smaller chunks.
        If not, just create a link to the resampled audio file.
        """
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
            self.chunks_manifest_path.unlink()
        self._remove_generated_chunks()
        if exceeds_limit:
            rprint(f"Splitting {self.resampled_episode_path} into chunks")
            chunk_names = self._split_resampled_episode()
        else:
            rprint(f"Creating symlink to {self.resampled_episode_path}")
            with atomic_output(self.episode_chunks_dir / "chunk_000.mp3") as tmp_path:
                tmp_path.symlink_to(self.resampled_episode_path.name)
            chunk_names = ["chunk_000.mp3"]
        # The manifest is written last; it marks the chunk set as complete.
        write_text_atomic(self.chunks_manifest_path, json.dumps(chunk_names))
        return [self.episode_chunks_dir / name for name in chunk_names]

    def _remove_generated_chunks(self) -> None:
        """
        Remove chunk files this tool generated (``chunk_NNN.mp3`` and the
        ``chunk_NNN.wav`` derived from them). The downloaded and resampled
        episode files are never touched, even if their names look similar.
        """
        keep = {self.episode_path, self.resampled_episode_path}
        for path in self.episode_chunks_dir.glob("chunk_*"):
            if path in keep or not GENERATED_CHUNK_RE.fullmatch(path.name):
                continue
            path.unlink()

    @property
    def chunks_manifest_path(self) -> Path:
        return self.episode_chunks_dir / "chunks.json"

    def _load_chunk_manifest(self) -> list[Path] | None:
        """Return the cached chunk paths if a complete chunk set exists."""
        try:
            chunk_names = json.loads(self.chunks_manifest_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(chunk_names, list) or not chunk_names:
            return None
        # Exact generated names enforce type, containment, uniqueness and order.
        if chunk_names != [
            f"chunk_{index:03d}.mp3" for index in range(len(chunk_names))
        ]:
            return None
        chunk_paths = [self.episode_chunks_dir / name for name in chunk_names]
        try:
            for path in chunk_paths:
                if not path.is_file() or path.stat().st_size > MAX_SIZE_IN_BYTES:
                    return None
                if path.is_symlink():
                    # Only the short-episode link to our own resampled audio is valid.
                    if (
                        len(chunk_paths) != 1
                        or self.resampled_episode_path.is_symlink()
                        or path.resolve() != self.resampled_episode_path.resolve()
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
        duration = get_audio_duration(self.resampled_episode_path)
        if duration <= 0:
            raise RuntimeError(
                "Cannot split audio with zero duration to the upload limit"
            )
        size = self.resampled_episode_path.stat().st_size
        # Estimate a byte-safe segment duration; leave headroom for MP3 frames,
        # headers and VBR variation. Actual output limits are checked below.
        segment_seconds = 0.9 * min(7200, duration * MAX_SIZE_IN_BYTES / max(size, 1))
        split_dir = self.episode_chunks_dir / f".split-{uuid.uuid4().hex}"
        split_dir.mkdir(parents=True)
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
            produced = sorted(
                split_dir.glob("chunk_*.mp3"),
                key=lambda path: int(path.stem.split("_")[1]),
            )
            if not produced:
                raise RuntimeError(
                    f"Splitting {self.resampled_episode_path} produced no chunks"
                )
            # Validate every new chunk before publishing audio or completion markers.
            for chunk in produced:
                chunk_duration = get_audio_duration(chunk)
                if chunk_duration > 7200 or chunk.stat().st_size > MAX_SIZE_IN_BYTES:
                    raise RuntimeError(
                        "Split chunk exceeds the upload limit (25 MiB / 7200 seconds); "
                        "split the source into shorter files and retry. No chunk manifest was published."
                    )
            for chunk in produced:
                os.replace(chunk, self.episode_chunks_dir / chunk.name)
            return [chunk.name for chunk in produced]
        finally:
            shutil.rmtree(split_dir, ignore_errors=True)

    @staticmethod
    def _content_fingerprint(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @property
    def source_metadata_path(self) -> Path:
        return self.podcast_dir / "source.json"

    def _invalidate_changed_local_source(self) -> bool:
        source_path = Path(get_cache_source(self.url))
        try:
            fingerprint = self._content_fingerprint(source_path)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Local source audio is missing: {source_path}. "
                "Restore the source file before reusing its cache."
            ) from exc
        try:
            cached = json.loads(self.source_metadata_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            cached = None
        if cached == {"sha256": fingerprint}:
            return False
        # Remove the completion marker first so interrupted preparation is retried.
        self.source_metadata_path.unlink(missing_ok=True)
        self._remove_cached_transcripts()
        self._remove_generated_chunks()
        self.chunks_manifest_path.unlink(missing_ok=True)
        self.resampled_episode_path.unlink(missing_ok=True)
        self.episode_path.unlink(missing_ok=True)
        return True

    def prepare_audio_for_transcription(self) -> list[Path]:
        """
        Steps needed to prepare an audio file URL for transcription:
            - Download the audio file if it's a URL
            - Resample the audio file to 16khz
            - Split the audio file into smaller chunks if needed
        """
        self.podcast_dir.mkdir(parents=True, exist_ok=True)
        source_changed = False
        if not self._is_http_url:
            source_changed = self._invalidate_changed_local_source()
        self.make_sure_audio_file_exists()
        self.make_sure_audio_file_is_resampled()
        chunks = self.split_into_chunks()
        if source_changed:
            write_text_atomic(
                self.source_metadata_path,
                json.dumps({"sha256": self._content_fingerprint(self.episode_path)}),
            )
        return chunks
