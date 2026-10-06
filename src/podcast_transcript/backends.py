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

from .config import settings
from .safe_io import atomic_output, run_checked, write_text_atomic


HTTP_CONNECT_TIMEOUT = 10.0
HTTP_WRITE_TIMEOUT = 300.0
HTTP_POOL_TIMEOUT = 10.0
GROQ_MAX_ATTEMPTS = 10


def http_timeout() -> httpx.Timeout:
    """
    Timeout for remote transcription uploads.

    Connect, write (per chunk sent) and pool waits are short. The read timeout
    covers the time the server needs to transcribe the upload and is set with
    ``TRANSCRIPT_HTTP_READ_TIMEOUT`` (default 30 minutes).
    """
    return httpx.Timeout(
        connect=HTTP_CONNECT_TIMEOUT,
        write=HTTP_WRITE_TIMEOUT,
        read=settings.http_read_timeout_seconds,
        pool=HTTP_POOL_TIMEOUT,
    )


def timeout_error(backend: str, exc: httpx.TimeoutException) -> RuntimeError:
    kind = type(exc).__name__
    detail = str(exc).strip()
    message = f"{backend} transcription request timed out ({kind})"
    if detail:
        message += f": {detail}"
    if isinstance(exc, httpx.ReadTimeout):
        message += ". Raise TRANSCRIPT_HTTP_READ_TIMEOUT if the server needs longer."
    else:
        message += "."
    return RuntimeError(message)


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
        matches = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", duration_str)
        for value, unit in matches:
            value = float(value)
            if unit == "ms":
                total_seconds += value / 1000
            elif unit == "h":
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

    @staticmethod
    def rate_limit_message(response: httpx.Response) -> str | None:
        """Return Groq's rate-limit message, or None if the body has none."""
        try:
            error = response.json()
        except ValueError:
            return None
        if not isinstance(error, dict):
            return None
        details = error.get("error")
        if not isinstance(details, dict):
            return None
        message = details.get("message")
        return message if isinstance(message, str) else None

    @staticmethod
    def http_error(response: httpx.Response) -> RuntimeError:
        detail = response.text.strip()
        if detail:
            return RuntimeError(
                f"Groq transcription failed with status {response.status_code}: {detail}"
            )
        return RuntimeError(
            f"Groq transcription failed with status {response.status_code}."
        )

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
        timeout = http_timeout()
        for attempt in range(1, GROQ_MAX_ATTEMPTS + 1):
            with httpx.Client() as client:
                try:
                    response = client.post(
                        url, headers=headers, files=files, data=data, timeout=timeout
                    )
                except httpx.TimeoutException as e:
                    raise timeout_error("Groq", e) from e
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    if response.status_code == 429:
                        # Rate limit exceeded
                        error_message = self.rate_limit_message(response)
                        if error_message is None:
                            # No usable rate-limit details, report the raw response
                            raise self.http_error(response) from e
                        rprint("rate limit exceeded: ", error_message)
                        # Extract wait time from error message
                        match = re.search(
                            r"Please try again in ((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)",
                            error_message,
                        )
                        if match:
                            wait_time_str = match.group(1)
                            # Parse wait_time_str
                            wait_seconds = self.parse_duration(wait_time_str)
                            if wait_seconds > 0:
                                if attempt == GROQ_MAX_ATTEMPTS:
                                    raise RuntimeError(
                                        f"Groq rate limit still hit after {GROQ_MAX_ATTEMPTS} attempts: "
                                        f"{error_message}"
                                    ) from e
                                rprint(
                                    f"Waiting for {wait_seconds} seconds before retrying..."
                                )
                                end_time = (
                                    time.time() + wait_seconds + 2
                                )  # Add 2 seconds buffer
                                self.sleep_until(end_time)
                                continue  # Retry after waiting
                            raise RuntimeError(
                                f"Groq rate limit hit and the wait time could not be parsed: {error_message}"
                            ) from e
                        raise RuntimeError(
                            f"Groq rate limit hit without a wait time: {error_message}"
                        ) from e
                    raise self.http_error(response) from e
                else:
                    # Success
                    json_transcript = response.json()
                    break  # Exit the loop

        write_text_atomic(transcript_path, json.dumps(json_transcript))


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
        try:
            import mlx_whisper  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                'MLX backend dependencies are not installed. Install with: `pip install "podcast-transcript[mlx]"`.'
            ) from exc

        result = mlx_whisper.transcribe(
            str(audio_file),
            path_or_hf_repo=self.model_name,
            word_timestamps=self.word_timestamps,
            initial_prompt=self.prompt,
            language=self.language,  # type: ignore
        )
        write_text_atomic(transcript_path, json.dumps(result, indent=2))


class Voxhelm:
    """
    Transcribe an audio file using a Voxhelm OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model_name: str | None,
        language: str | None,
        prompt: str | None,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name or "gpt-4o-mini-transcribe"
        self.language = language
        self.prompt = prompt

    @property
    def transcription_url(self) -> str:
        if self.api_base.endswith("/v1"):
            return f"{self.api_base}/audio/transcriptions"
        return f"{self.api_base}/v1/audio/transcriptions"

    def transcribe(self, audio_file: Path, transcript_path: Path) -> None:
        rprint("audio chunk to text: ", audio_file)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": self.model_name,
            "response_format": "verbose_json",
        }
        if self.language:
            data["language"] = self.language
        if self.prompt:
            data["prompt"] = self.prompt

        timeout = http_timeout()
        with audio_file.open("rb") as file_handle:
            files = {"file": (audio_file.name, file_handle)}
            with httpx.Client() as client:
                try:
                    response = client.post(
                        self.transcription_url,
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=timeout,
                    )
                except httpx.TimeoutException as exc:
                    raise timeout_error("Voxhelm", exc) from exc
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    detail = response.text.strip()
                    if detail:
                        raise RuntimeError(
                            f"Voxhelm transcription failed with status {response.status_code}: {detail}"
                        ) from exc
                    raise RuntimeError(
                        f"Voxhelm transcription failed with status {response.status_code}."
                    ) from exc

        write_text_atomic(transcript_path, json.dumps(response.json()))


class WhisperCpp:
    """
    Transcribe an audio file using the whisper-cpp library.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        processors: int = 4,
    ):
        if model_name is None:
            model_name = "ggml-large-v3.bin"
        self.model_name = model_name
        self.model_path = settings.whisper_cpp_models_dir / model_name
        self.language = language
        self.prompt = prompt
        self.processors = processors

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
                    "-c:a",
                    "pcm_s16le",
                    str(tmp_path),
                ],
                description=f"Converting {input_path} to WAV",
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
            output_path (Path): Base path for the JSON transcript; whisper-cli
                appends ``.json`` to it.
        """
        rprint(f"Transcribing {input_path} to {output_path}")
        json_path = output_path.with_name(f"{output_path.name}.json")
        with atomic_output(json_path) as tmp_json_path:
            # whisper-cli appends ".json" to the -of base path
            tmp_base = tmp_json_path.with_suffix("")
            self.run_whisper_cli(input_path, tmp_base)

    def run_whisper_cli(self, input_path: Path, output_base: Path) -> None:
        args = [
            "time",  # Note: this might only work on Unix-like systems
            "whisper-cli",
            "-m",
            str(self.model_path),
            "-f",
            str(input_path),
            "-oj",  # Output JSON format
            "-of",
            str(output_base),
            "-p",
            str(self.processors),
        ]
        if self.language is not None:
            args.extend(["-l", self.language])
        if self.prompt is not None:
            args.extend(["--prompt", self.prompt])
        run_checked(args, description=f"whisper-cli transcription of {input_path}")

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
        write_text_atomic(output_path, json.dumps(transformed_transcript))

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
