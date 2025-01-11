import json
import httpx
import shutil
import subprocess

from pathlib import Path
from urllib.parse import urlparse

from rich import print as rprint


def is_url(url: str) -> bool:
    return url.startswith("http")


def get_title_from_string(url: str) -> str:
    if is_url(url):
        parsed_url = urlparse(url)
        return parsed_url.path.split("/")[-1].split(".")[0]
    else:
        return Path(url).stem


def download(url: str, target_path: Path) -> None:
    rprint(f"Downloading {url} to {target_path}")
    response = httpx.get(url)
    with target_path.open("wb") as file:
        file.write(response.content)


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
    output_path.parent.mkdir(exist_ok=True, parents=True)
    # resample the audio file to 16khz
    subprocess.run(
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
            str(output_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# 25MB max bytes are allowed
MAX_SIZE_IN_BYTES = 25 * 1024 * 1024


class Audio:
    def __init__(self, *, base_dir: Path, url: str, title: str | None = None):
        self.base_dir = base_dir
        self.url = url
        self._is_http_url = is_url(url)
        self.prefix = get_title_from_string(url)
        if title is not None:
            self.title = title
        else:
            self.title = self.prefix
        self.podcast_dir = base_dir / self.prefix
        self.episode_chunks_dir = self.podcast_dir / "chunks"

    def __repr__(self):
        return self.title

    @property
    def episode_path(self):
        return self.episode_chunks_dir / Path(self.url).name

    @property
    def resampled_episode_path(self):
        return self.episode_chunks_dir / f"{self.prefix}_16khz.mp3"

    def make_sure_audio_file_exists(self) -> None:
        """
        Make sure the audio file for the episode exists in the local filesystem.
        """
        if not self.episode_path.exists():
            self.episode_path.parent.mkdir(exist_ok=True, parents=True)
            if self._is_http_url:
                download(self.url, self.episode_path)
            else:
                shutil.copy(Path(self.url), self.episode_path)

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
        chunk_paths = sorted(list(self.episode_chunks_dir.glob("chunk_*.mp3")))
        if len(chunk_paths) > 0:
            return chunk_paths
        if self.exceeds_size_limit:
            rprint(f"Splitting {self.resampled_episode_path} into chunks")
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    self.resampled_episode_path,
                    "-f",
                    "segment",
                    "-segment_time",
                    "7200",  # 7200 seconds is the maximum duration allowed by Groq
                    "-c",
                    "copy",
                    self.episode_chunks_dir / "chunk_%03d.mp3",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            rprint(f"Creating symlink to {self.resampled_episode_path}")
            try:
                (self.episode_chunks_dir / "chunk_000.mp3").symlink_to(
                    self.resampled_episode_path
                )
            except FileExistsError:
                pass
        chunk_paths = sorted(list(self.episode_chunks_dir.glob("chunk_*.mp3")))
        return chunk_paths

    def prepare_audio_for_transcription(self) -> list[Path]:
        """
        Steps needed to prepare an audio file URL for transcription:
            - Download the audio file if it's a URL
            - Resample the audio file to 16khz
            - Split the audio file into smaller chunks if needed
        """
        self.podcast_dir.mkdir(exist_ok=True)
        self.make_sure_audio_file_exists()
        self.make_sure_audio_file_is_resampled()
        return self.split_into_chunks()
