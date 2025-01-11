import json
import re
import shutil
from datetime import timedelta

from pathlib import Path

from rich import print as rprint

from .audio import Audio
from .backends import TranscriptionBackend
from .config import settings


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
        if not transcript_path.exists():
            service.transcribe(chunk, transcript_path)
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
            dote_paths.append(dote_path)
            continue
        with chunk.open("r") as file:
            whisper_transcript = json.load(file)
        rprint(f"Converting {chunk.name} to DOTe format")
        dote_transcript = whisper_to_dote(whisper_transcript["segments"])
        dote_path = chunk.with_suffix(".dote.json")
        with dote_path.open("w") as out_file:
            json.dump(dote_transcript, out_file)
        dote_paths.append(dote_path)
    return dote_paths


def combine_dote_chunks(dote_chunks: list[Path], output_path: Path) -> None:
    """Combine the DOTe chunks into a single DOTe file."""
    if len(dote_chunks) == 1:
        # Copy and return early
        [source_dote_file] = dote_chunks
        rprint(f"Copying {source_dote_file} to {output_path}")
        try:
            shutil.copy(source_dote_file, output_path)
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
            offset += last_end_time

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


def convert_to_plaintext(dote_path: Path, plaintext_path: Path) -> None:
    """Converts DOTe format to plain text."""
    with open(dote_path, "r") as f:
        dote_data = json.load(f)

    lines = dote_data.get("lines", [])
    output = []
    for line in lines:
        text = line["text"]
        output.append(text)

    with plaintext_path.open("w") as f:
        f.write("\n".join(output))


def transcribe(url: str, backend: TranscriptionBackend) -> dict[str, Path]:
    transcript_paths = {}
    audio = Audio(base_dir=settings.transcript_dir, url=url)
    audio_chunks = audio.prepare_audio_for_transcription()
    text_chunks = audio_chunks_to_text(backend, audio_chunks)
    dote_chunks = whisper_text_chunks_to_dote(text_chunks)
    dote_path = audio.podcast_dir / f"{audio.prefix}.dote.json"
    if not dote_path.exists():
        combine_dote_chunks(dote_chunks, dote_path)
    transcript_paths["DOTe"] = dote_path
    podlove_path = audio.podcast_dir / f"{audio.prefix}.podlove.json"
    if not podlove_path.exists():
        convert_dote_to_podlove(dote_path, podlove_path)
    transcript_paths["podlove"] = podlove_path
    webvtt_path = audio.podcast_dir / f"{audio.prefix}.webvtt"
    if not webvtt_path.exists():
        convert_to_webvtt(dote_path, webvtt_path)
    transcript_paths["WebVTT"] = webvtt_path
    plaintext_path = audio.podcast_dir / f"{audio.prefix}.txt"
    if not plaintext_path.exists():
        convert_to_plaintext(dote_path, plaintext_path)
    transcript_paths["plain text"] = plaintext_path
    return transcript_paths
