import io
import json
import re
from datetime import timedelta

import httpx
import subprocess

from pathlib import Path
from urllib.parse import urlparse

from rich import print as rprint

from .config import settings

MAX_SIZE_IN_BYTES = 25 * 1024 * 1024  # 25 MB


class AudioUrl:
    def __init__(self, *, base_dir: Path, url: str, title: str | None = None):
        self.base_dir = base_dir
        self.url = url
        if title is not None:
            self.title = title
        else:
            self.title = self.get_title_from_url(url)
        self.prefix = url.split("/")[-1].split(".")[0]
        self.podcast_dir = base_dir / self.prefix
        self.episode_chunks_dir = self.podcast_dir / "chunks"

    @property
    def episode_path(self):
        return self.episode_chunks_dir / f"{self.prefix}.mp3"

    @property
    def resampled_episode_path(self):
        return self.episode_chunks_dir / f"{self.prefix}_16khz.mp3"

    @property
    def is_downloaded(self) -> bool:
        return self.episode_path.exists()

    @property
    def is_resampled(self) -> bool:
        return self.resampled_episode_path.exists()

    @property
    def exceeds_file_size_limit(self) -> bool:
        return self.resampled_episode_path.stat().st_size > MAX_SIZE_IN_BYTES

    @staticmethod
    def get_title_from_url(url: str) -> str:
        parsed_url = urlparse(url)
        return parsed_url.path.split("/")[-1].split(".")[0]

    def __repr__(self):
        return self.title


def download(url: str, target_path: Path) -> None:
    rprint(f"Downloading {url} to {target_path}")
    target_path.parent.mkdir(exist_ok=True, parents=True)
    response = httpx.get(url)
    with target_path.open("wb") as file:
        file.write(response.content)


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


def split_into_chunks(audio: AudioUrl) -> list[Path]:
    """
    If the audio file exceeds the size limit, split it into smaller chunks.
    If not, just create a link to the resampled audio file.
    """
    chunk_paths = list(audio.episode_chunks_dir.glob("*.mp3"))
    if audio.exceeds_file_size_limit:
        rprint(f"Splitting {audio.resampled_episode_path} into chunks")
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                audio.resampled_episode_path,
                "-f",
                "segment",
                "-segment_time",
                "7200",
                "-c",
                "copy",
                audio.episode_chunks_dir / "chunk_%03d.mp3",
            ]
        )  # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        rprint(f"Creating symlink to {audio.resampled_episode_path}")
        try:
            (audio.episode_chunks_dir / "chunk_000.mp3").symlink_to(
                audio.resampled_episode_path
            )
        except FileExistsError:
            pass
    chunk_paths = list(audio.episode_chunks_dir.glob("chunk_*.mp3"))
    return chunk_paths


def prepare_audio_for_transcription(audio: AudioUrl) -> list[Path]:
    """
    Steps needed to prepare an audio file URL for transcription:
        - Download the audio file
        - Resample the audio file to 16khz
        - Split the audio file into smaller chunks if needed
    """
    audio.podcast_dir.mkdir(exist_ok=True)
    if not audio.is_downloaded:
        download(audio.url, audio.episode_path)
    if not audio.is_resampled:
        resample_audio(audio.episode_path, audio.resampled_episode_path)
    return split_into_chunks(audio)


def audio_chunk_to_text(audio_chunk: Path, transcript_path: Path) -> None:
    """
    Convert an audio chunk to text using the Groq API. Use httpx instead of
    groq client to get the response in verbose JSON format. The groq client
    only provides the transcript text.
    """
    rprint("audio chunk to text: ", audio_chunk)
    with audio_chunk.open("rb") as f:
        audio_content = f.read()
    rprint("audio content size: ", len(audio_content))
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {settings.groq_api_key}"}
    audio_file = io.BytesIO(audio_content)
    audio_file.name = "audio.mp3"
    files = {"file": audio_file}
    data = {
        # FIXME make this configurable
        "model": "whisper-large-v3",
        "response_format": "verbose_json",
        "language": "de",
        "prompt": "Florian, Dominik und Jochen sprechen über PyTest ein Testframework für Python",
    }
    with httpx.Client() as client:
        response = client.post(
            url, headers=headers, files=files, data=data, timeout=None
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if response.status_code == 429:
                # rate limit exceeded
                error = response.json()
                rprint("rate limit exceeded: ", error["error"]["message"])
            else:
                rprint("HTTP error: ", e)
                rprint("response: ", response.text)
            return None
        json_transcript = response.json()
    with transcript_path.open("w") as out_file:
        json.dump(json_transcript, out_file)


def audio_chunks_to_text(audio_chunks: list[Path]) -> list[Path]:
    """Convert the audio chunks to text. Only convert if the transcript does not exist yet."""
    file_names = " ".join([chunk.name for chunk in audio_chunks])
    rprint(f"Converting {file_names} to text")

    raw_transcripts = []
    for chunk in audio_chunks:
        chunk_name = chunk.name.split(".")[0]
        transcript_name = f"{chunk_name}.json"
        transcript_path = chunk.parent / transcript_name
        if not transcript_path.exists():
            audio_chunk_to_text(chunk, transcript_path)
        raw_transcripts.append(transcript_path)
    return raw_transcripts


def groq_to_dote(input_data):
    """Convert the Groq JSON to DOTe format."""

    def format_time(seconds):
        total_milliseconds = int(round(seconds * 1000))
        hours = total_milliseconds // 3600000
        minutes = (total_milliseconds % 3600000) // 60000
        secs = (total_milliseconds % 60000) // 1000
        millis = total_milliseconds % 1000
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

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


def groq_text_chunks_to_dote(raw_text_chunks: list[Path]) -> list[Path]:
    """Transform the raw groq text chunks to DOTe format."""
    dote_paths = []
    for chunk in raw_text_chunks:
        dote_path = chunk.with_suffix(".dote.json")
        if dote_path.exists():
            dote_paths.append(dote_path)
            continue
        with chunk.open("r") as file:
            groq_transcript = json.load(file)
        rprint(f"Converting {chunk.name} to DOTe format")
        dote_transcript = groq_to_dote(groq_transcript["segments"])
        dote_path = chunk.with_suffix(".dote.json")
        with dote_path.open("w") as out_file:
            json.dump(dote_transcript, out_file)
        dote_paths.append(dote_path)
    return dote_paths


def combine_dote_chunks(dote_chunks: list[Path], output_path: Path) -> None:
    """Combine the DOTe chunks into a single DOTe file."""
    if len(dote_chunks) == 1:
        # Symlink and return early
        [source_dote_file] = dote_chunks
        rprint(f"Symlink {source_dote_file} to {output_path}")
        try:
            output_path.symlink_to(source_dote_file)
        except FileExistsError:
            pass
        return None

    def parse_timecode(timecode):
        match = re.match(r"(\d+):(\d+):(\d+),(\d+)", timecode)
        if not match:
            raise ValueError(f"Invalid timecode format: {timecode}")
        h, m, s, ms = map(int, match.groups())
        return timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)

    def format_timecode(delta):
        total_seconds = int(delta.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        milliseconds = delta.microseconds // 1000
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    rprint(f"Combining {len(dote_chunks)} DOTe chunks into {output_path}")

    combined_lines = []
    offset = timedelta()

    for filename in dote_chunks:
        with open(filename, "r") as f:
            data = json.load(f)

        for line in data["lines"]:
            new_line = dict(line)
            start_time = parse_timecode(line["startTime"])
            end_time = parse_timecode(line["endTime"])

            # Adjust times with offset
            new_line["startTime"] = format_timecode(start_time + offset)
            new_line["endTime"] = format_timecode(end_time + offset)
            combined_lines.append(new_line)

            # Update offset with the last endTime of this file
            if len(data["lines"]) > 0:
                last_end_time = parse_timecode(data["lines"][-1]["endTime"])
                # print("last_end_time: ", last_end_time)
                offset += last_end_time
                # print("offset: ", offset)

    with open(output_path, "w") as f:
        json.dump({"lines": combined_lines}, f)


def convert_dote_to_podlove(dote_path: Path, podlove_path: Path) -> None:
    def time_to_ms(time_str):
        h, m, s_ms = time_str.split(":")
        s, ms = s_ms.split(",")
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

    with open(dote_path, "r") as infile:
        dote_data = json.load(infile)

    transcripts = []
    for line in dote_data.get("lines", []):
        start_ms = time_to_ms(line["startTime"])
        end_ms = time_to_ms(line["endTime"])
        transcript = {
            "start": line["startTime"].replace(",", "."),
            "start_ms": start_ms,
            "end": line["endTime"].replace(",", "."),
            "end_ms": end_ms,
            "speaker": line["speakerDesignation"],
            "voice": "",  # assuming no voice data is available
            "text": line["text"],
        }
        transcripts.append(transcript)

    with open(podlove_path, "w") as outfile:
        json.dump({"transcripts": transcripts}, outfile)


def convert_to_webvtt(dote_path: Path, vtt_path: Path) -> None:
    """Converts DOTe format to WebVTT format."""
    with open(dote_path, "r") as f:
        dote_data = json.load(f)

    lines = dote_data.get("lines", [])
    output = ["WEBVTT\n"]
    for line in lines:
        start_time = line["startTime"].replace(",", ".")
        end_time = line["endTime"].replace(",", ".")
        text = line["text"]
        output.append(f"{start_time} --> {end_time}")
        output.append(text)
        output.append("")  # Blank line to separate captions

    with vtt_path.open("w") as f:
        f.write("\n".join(output))


def transcribe(url: str) -> None:
    audio = AudioUrl(base_dir=settings.transcript_dir, url=url)
    audio_chunks = prepare_audio_for_transcription(audio)
    text_chunks = audio_chunks_to_text(audio_chunks)
    dote_chunks = groq_text_chunks_to_dote(text_chunks)
    dote_path = audio.podcast_dir / f"{audio.prefix}.dote.json"
    if not dote_path.exists():
        combine_dote_chunks(dote_chunks, dote_path)
    podlove_path = audio.podcast_dir / f"{audio.prefix}.podlove.json"
    if not podlove_path.exists():
        convert_dote_to_podlove(dote_path, podlove_path)
    webvtt_path = audio.podcast_dir / f"{audio.prefix}.webvtt"
    if not webvtt_path.exists():
        convert_to_webvtt(dote_path, webvtt_path)
    rprint(audio.url)
    rprint("single track transcription!")
