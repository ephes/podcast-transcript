"""
Transcriptions services
"""

import io
import re
import json
import time
import subprocess

from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx

from rich import print as rprint

from .config import settings


@runtime_checkable
class TranscriptionBackend(Protocol):
    def transcribe(self, audio_file: Path, transcript_path: Path) -> None:
        pass


class Groq:
    """
    Transcribe an audio file using the Groq API.
    """

    def __init__(
        self, *, api_key: str, model_name: str | None, language: str, prompt: str
    ):
        self.api_key = api_key
        if model_name is None:
            model_name = "whisper-large-v3"
        self.model_name = self.validate_model(model_name)
        self.language = language
        self.prompt = prompt

    @staticmethod
    def validate_model(model_name: str) -> str:
        supported_models = [
            "whisper-large-v3",
            "whisper-large-v3-turbo",
            "distil-whisper-large-v3-en",
        ]
        if model_name not in set(supported_models):
            raise ValueError(
                f"Invalid model name: {model_name}. Supported models are {supported_models}."
            )
        return model_name

    @staticmethod
    def parse_duration(duration_str):
        total_seconds = 0
        # Find all matches of number and unit
        matches = re.findall(r"(\d+(?:\.\d+)?)([hms])", duration_str)
        for value, unit in matches:
            value = float(value)
            if unit == "h":
                total_seconds += value * 3600
            elif unit == "m":
                total_seconds += value * 60
            elif unit == "s":
                total_seconds += value
        return total_seconds

    @staticmethod
    def sleep_until(end_time):
        """Don't just sleep but also check whether sufficient time has passed."""
        while True:
            now = time.time()
            if now >= end_time:
                break
            time.sleep(min(10, end_time - now))  # Sleep in small increments

    def transcribe(self, audio_file: Path, transcript_path: Path) -> None:
        """
        Convert an audio chunk to text using the Groq API. Use httpx instead of
        groq client to get the response in verbose JSON format. The groq client
        only provides the transcript text.
        """
        rprint("audio chunk to text: ", audio_file)
        with audio_file.open("rb") as f:
            audio_content = f.read()
        rprint("audio content size: ", len(audio_content))
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        upload_file = io.BytesIO(audio_content)
        upload_file.name = "audio.mp3"
        files = {"file": upload_file}
        data = {
            "model": self.model_name,
            "response_format": "verbose_json",
            "language": self.language,
            "prompt": self.prompt,
        }
        while True:
            with httpx.Client() as client:
                response = client.post(
                    url, headers=headers, files=files, data=data, timeout=None
                )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    if response.status_code == 429:
                        # Rate limit exceeded
                        error = response.json()
                        error_message = error["error"]["message"]
                        rprint("rate limit exceeded: ", error_message)
                        # Extract wait time from error message
                        match = re.search(
                            r"Please try again in ([^.]+)\.", error_message
                        )
                        if match:
                            wait_time_str = match.group(1)
                            # Parse wait_time_str
                            wait_seconds = self.parse_duration(wait_time_str)
                            if wait_seconds is not None:
                                rprint(
                                    f"Waiting for {wait_seconds} seconds before retrying..."
                                )
                                end_time = (
                                    time.time() + wait_seconds + 2
                                )  # Add 2 seconds buffer
                                self.sleep_until(end_time)
                                continue  # Retry after waiting
                            else:
                                rprint("Could not parse wait time, exiting.")
                                return None
                        else:
                            rprint(
                                "Could not find wait time in error message, exiting."
                            )
                            return None
                    else:
                        rprint("HTTP error: ", e)
                        rprint("response: ", response.text)
                        return None
                else:
                    # Success
                    json_transcript = response.json()
                    break  # Exit the loop

        with transcript_path.open("w") as out_file:
            json.dump(json_transcript, out_file)


class MLX:
    """
    Transcribe an audio file using the MLX API.
    """

    def __init__(
        self,
        *,
        model_name: str | None,
        word_timestamps: bool = False,
        prompt: str | None = None,
        language: str | None = None,
    ):
        if model_name is None:
            model_name = "mlx-community/whisper-large-v3-mlx"
        # cannot validate model name because it could be a path to a local model
        self.model_name = model_name
        self.word_timestamps = word_timestamps
        self.prompt = prompt
        self.language = language

    def transcribe(self, audio_file: Path, transcript_path: Path) -> None:
        # import only when needed because it's slow (takes 0.5s)
        import mlx_whisper  # type: ignore

        result = mlx_whisper.transcribe(
            str(audio_file),
            path_or_hf_repo=self.model_name,
            word_timestamps=self.word_timestamps,
            initial_prompt=self.prompt,
            language=self.language,  # type: ignore
        )
        with transcript_path.open("w") as file:
            file.write(json.dumps(result, indent=2))


class WhisperCpp:
    """
    Transcribe an audio file using the whisper-cpp library.
    """

    def __init__(self, *, model_name: str | None = None, language: str | None = None):
        if model_name is None:
            model_name = "ggml-large-v3.bin"
        self.model_name = model_name
        self.model_path = settings.whisper_cpp_models_dir / model_name
        self.language = language

    @staticmethod
    def convert_to_wav(input_path: Path, output_path: Path) -> None:
        """
        Convert an audio file to WAV format with specific parameters:
        - Sample rate: 16kHz
        - Channels: Mono (1 channel)
        - Codec: PCM 16-bit little-endian

        Args:
            input_path (Path): Path to the input audio file
            output_path (Path): Path where the output WAV file will be saved
        """
        rprint(f"Converting {input_path} to WAV format at {output_path}")
        output_path.parent.mkdir(exist_ok=True, parents=True)

        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(input_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def transcribe_wav(
        self,
        input_path: Path,
        output_path: Path,
    ) -> None:
        """
        Transcribe audio using whisper-cli with specified parameters.

        Args:
            input_path (Path): Path to the input audio file
            output_path (Path): Path where the JSON transcript will be saved
        """
        rprint(f"Transcribing {input_path} to {output_path}")
        output_path.parent.mkdir(exist_ok=True, parents=True)

        args = [
            "time",  # Note: this might only work on Unix-like systems
            "whisper-cli",
            "-m",
            str(self.model_path),
            "-f",
            str(input_path),
            "-oj",  # Output JSON format
            "-of",
            str(output_path),
        ]
        if self.language is not None:
            args.extend(["-l", self.language])
        subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def transform_transcription(input_data: dict) -> dict:
        """
        Transform transcription data the whisper-cpp JSON to original Whisper format.
        """

        def timestamp_to_seconds(timestamp):
            # Convert "HH:MM:SS,mmm" to seconds
            hours, minutes, seconds = timestamp.split(":")
            seconds, milliseconds = seconds.split(",")
            return (
                float(hours) * 3600
                + float(minutes) * 60
                + float(seconds)
                + float(milliseconds) / 1000
            )

        segments = []
        for idx, entry in enumerate(input_data["transcription"]):
            segment = {
                "id": idx,
                "seek": int(
                    entry["offsets"]["from"]
                ),  # Using 'from' offset as seek position
                "start": timestamp_to_seconds(entry["timestamps"]["from"]),
                "end": timestamp_to_seconds(entry["timestamps"]["to"]),
                "text": entry["text"].strip(),
            }
            segments.append(segment)

        return {"segments": segments}

    def convert_output_format(self, input_path: Path, output_path: Path) -> None:
        with input_path.open("r") as file:
            cpp_transcript = json.load(file)
        transformed_transcript = self.transform_transcription(cpp_transcript)
        with output_path.open("w") as out_file:
            json.dump(transformed_transcript, out_file)

    def transcribe(self, audio_file: Path, transcript_path: Path) -> None:
        # Convert the audio file to WAV format
        wav_file = audio_file.with_suffix(".wav")
        if not wav_file.exists():
            self.convert_to_wav(audio_file, wav_file)
        # Transcribe the WAV file
        whisper_transcript_path = transcript_path.with_suffix(
            ".whisper-cpp"
        )  # .json is automatically appended
        self.transcribe_wav(wav_file, whisper_transcript_path)
        whisper_read_transcript_path = transcript_path.with_suffix(".whisper-cpp.json")
        # Convert the whisper-cpp JSON to original Whisper format
        self.convert_output_format(whisper_read_transcript_path, transcript_path)
