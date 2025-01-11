"""
Transcriptions services
"""

import io
import re
import json
import time

from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx

from rich import print as rprint


@runtime_checkable
class TranscriptionService(Protocol):
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
