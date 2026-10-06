import hashlib
import json
import os
import re
import httpx
import shutil
import subprocess
import uuid

from pathlib import Path
from urllib.parse import urlparse

from rich import print as rprint

from .safe_io import atomic_output, run_checked, write_text_atomic

DOWNLOAD_TIMEOUT = httpx.Timeout(60.0, connect=30.0)
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
    return url.startswith("http")


def get_title_from_string(url: str) -> str:
    if is_url(url):
        parsed_url = urlparse(url)
        return parsed_url.path.split("/")[-1].split(".")[0]
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
    rprint(f"Downloading {url} to {target_path}")
    with atomic_output(target_path) as tmp_path:
        with httpx.Client(follow_redirects=True, timeout=DOWNLOAD_TIMEOUT) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as file:
                    for data in response.iter_bytes():
                        file.write(data)


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
    # Execute the command
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # Return output as string
        check=True,  # Raise CalledProcessError on non-zero exit
    )
    # Parse the JSON output
    metadata = json.loads(result.stdout)
    duration = float(metadata["format"]["duration"])
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
        name = Path(self.url).name
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
                    shutil.copy(Path(self.url), tmp_path)

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
        # No valid manifest: either an earlier split was interrupted while the
        # chunks were moved into place, or the cache predates the manifest.
        # Remove leftover chunk audio (and WAV files derived from it) and
        # split again so a partial chunk set is never reused.
        self._remove_generated_chunks()
        if self.exceeds_size_limit:
            rprint(f"Splitting {self.resampled_episode_path} into chunks")
            chunk_names = self._split_resampled_episode()
        else:
            rprint(f"Creating symlink to {self.resampled_episode_path}")
            with atomic_output(self.episode_chunks_dir / "chunk_000.mp3") as tmp_path:
                tmp_path.symlink_to(self.resampled_episode_path)
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
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        if not isinstance(chunk_names, list) or not chunk_names:
            return None
        chunk_paths = [self.episode_chunks_dir / str(name) for name in chunk_names]
        if not all(path.exists() for path in chunk_paths):
            return None
        return chunk_paths

    def _split_resampled_episode(self) -> list[str]:
        """
        Split the resampled episode into a hidden scratch directory and only
        move the chunks into place once ffmpeg succeeded. Returns the chunk
        file names; the caller records them in the manifest.
        """
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
                    "7200",  # 7200 seconds is the maximum duration allowed by Groq
                    "-c",
                    "copy",
                    str(split_dir / "chunk_%03d.mp3"),
                ],
                description=f"Splitting {self.resampled_episode_path} into chunks",
            )
            produced = sorted(split_dir.glob("chunk_*.mp3"))
            if not produced:
                raise RuntimeError(
                    f"Splitting {self.resampled_episode_path} produced no chunks"
                )
            for chunk in produced:
                os.replace(chunk, self.episode_chunks_dir / chunk.name)
            return [chunk.name for chunk in produced]
        finally:
            shutil.rmtree(split_dir, ignore_errors=True)

    def prepare_audio_for_transcription(self) -> list[Path]:
        """
        Steps needed to prepare an audio file URL for transcription:
            - Download the audio file if it's a URL
            - Resample the audio file to 16khz
            - Split the audio file into smaller chunks if needed
        """
        self.podcast_dir.mkdir(parents=True, exist_ok=True)
        self.make_sure_audio_file_exists()
        self.make_sure_audio_file_is_resampled()
        return self.split_into_chunks()
