"""
Transcriptions services
"""

import asyncio
import importlib.util
import io
import re
import json
import math
import wave
import sys
import tempfile
import time

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, runtime_checkable

import httpx

from rich import print as rprint

from .config import private_mkdir, settings
from .transcript_validation import parse_dote_timecode, validate_whisper_transcript
from .safe_io import (
    LOCAL_TRANSCRIPT_MAX_BYTES,
    atomic_output,
    read_regular_bounded,
    run_checked,
    write_text_atomic,
)


HTTP_CONNECT_TIMEOUT = 10.0
HTTP_WRITE_TIMEOUT = 300.0
HTTP_POOL_TIMEOUT = 10.0
GROQ_MAX_ATTEMPTS = 10
REMOTE_RESPONSE_MAX_BYTES = 32 * 1024 * 1024
REMOTE_ERROR_MAX_CHARS = 2000


def remote_response(backend: str, url: str, **kwargs) -> httpx.Response:
    """Bound the complete request, including server computation and body transfer.

    Only identity-encoded bodies are accepted. httpx decoders inflate each
    received block in one call, so a small compressed block could allocate far
    more than the byte limit before it is counted. Refusing every other
    Content-Encoding before iteration leaves httpx with its identity decoder,
    so the counted bytes are exactly the received bytes.
    """
    headers = dict(kwargs.pop("headers", None) or {})
    headers["Accept-Encoding"] = "identity"

    async def receive() -> httpx.Response:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", url, headers=headers, **kwargs
            ) as response:
                encoding = response.headers.get("Content-Encoding", "identity")
                if encoding.strip().lower() not in {"", "identity"}:
                    raise RuntimeError(
                        f"{backend} transcription response used an unsupported "
                        "Content-Encoding; identity is required."
                    )
                body = bytearray()
                async for block in response.aiter_bytes(chunk_size=64 * 1024):
                    if len(body) + len(block) > REMOTE_RESPONSE_MAX_BYTES:
                        raise RuntimeError(
                            f"{backend} transcription response exceeded byte limit (32 MiB)."
                        )
                    body.extend(block)
                response_headers = dict(response.headers)
                # Retain charset/rate-limit headers; the body is already complete.
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)
                return httpx.Response(
                    response.status_code,
                    headers=response_headers,
                    content=bytes(body),
                    request=response.request,
                )

    async def bounded_receive() -> httpx.Response:
        try:
            return await asyncio.wait_for(receive(), settings.http_read_timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise httpx.ReadTimeout(
                "Elapsed request deadline exceeded TRANSCRIPT_HTTP_READ_TIMEOUT"
            ) from exc

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(bounded_receive())
    # Keep the synchronous backend usable by callers already running an event loop.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(bounded_receive())).result()


def http_timeout() -> httpx.Timeout:
    """
    Timeout for remote transcription uploads.

    Connect, write (per chunk sent) and pool waits are short. The read timeout
    is also the elapsed request budget (upload, computation and response) set
    with ``TRANSCRIPT_HTTP_READ_TIMEOUT`` (default 30 minutes).
    """
    return httpx.Timeout(
        connect=HTTP_CONNECT_TIMEOUT,
        write=HTTP_WRITE_TIMEOUT,
        read=settings.http_read_timeout_seconds,
        pool=HTTP_POOL_TIMEOUT,
    )


def timeout_error(backend: str, exc: httpx.TimeoutException) -> RuntimeError:
    kind = type(exc).__name__
    message = f"{backend} transcription request timed out ({kind})"
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
            now = time.monotonic()
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
        return message[:REMOTE_ERROR_MAX_CHARS] if isinstance(message, str) else None

    @staticmethod
    def http_error(response: httpx.Response) -> RuntimeError:
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
        wait_budget = settings.http_read_timeout_seconds
        for attempt in range(1, GROQ_MAX_ATTEMPTS + 1):
            try:
                response = remote_response(
                    "Groq",
                    url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=timeout,
                )
            except httpx.TimeoutException as e:
                raise timeout_error("Groq", e) from None
            except httpx.HTTPError:
                raise RuntimeError(
                    "Groq transcription request failed (transport error)."
                ) from None
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                if response.status_code == 429:
                    error_message = self.rate_limit_message(response) or ""
                    match = re.search(
                        r"Please try again in ((?:[0-9]+(?:\.[0-9]+)?(?:ms|h|m|s))+)(?![0-9a-z])",
                        error_message,
                    )
                    wait_seconds = self.parse_duration(match.group(1)) if match else 0
                    if attempt == GROQ_MAX_ATTEMPTS:
                        raise RuntimeError(
                            f"Groq rate limit still hit after {GROQ_MAX_ATTEMPTS} attempts; retry later."
                        ) from None
                    delay = wait_seconds + 2
                    if (
                        not math.isfinite(delay)
                        or wait_seconds <= 0
                        or delay > wait_budget
                    ):
                        raise RuntimeError(
                            "Groq rate limit (status 429) hit without a usable wait within TRANSCRIPT_HTTP_READ_TIMEOUT; retry later."
                        ) from None
                    wait_budget -= delay
                    rprint(
                        f"Groq rate limit: waiting {delay:g} seconds before retrying."
                    )
                    self.sleep_until(time.monotonic() + delay)
                    continue
                raise self.http_error(response) from None
            else:
                # Success
                json_transcript = response.json()
                break  # Exit the loop

        validate_whisper_transcript(json_transcript)
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
        # Never import MLX here: inference runs in an owned worker process so a
        # stall is bounded by TRANSCRIPT_LOCAL_TRANSCRIPTION_TIMEOUT.
        if importlib.util.find_spec("mlx_whisper") is None:
            raise RuntimeError(
                'MLX backend dependencies are not installed. Install with: `pip install "podcast-transcript[mlx]"`.'
            )
        private_mkdir(transcript_path.parent)
        with tempfile.TemporaryDirectory(
            prefix=".mlx-", dir=transcript_path.parent
        ) as directory:
            params_path = Path(directory) / "params.json"
            result_path = Path(directory) / "result.json"
            # Parameters (including the prompt) stay out of the process list.
            params_path.write_text(
                json.dumps(
                    {
                        "audio": str(audio_file),
                        "model": self.model_name,
                        "word_timestamps": self.word_timestamps,
                        "prompt": self.prompt,
                        "language": self.language,
                    }
                )
            )
            run_checked(
                [
                    sys.executable,
                    "-m",
                    "podcast_transcript._mlx_worker",
                    str(params_path),
                    str(result_path),
                ],
                description=f"MLX transcription of {audio_file}",
                timeout=settings.local_transcription_timeout_seconds,
            )
            raw = read_regular_bounded(
                result_path,
                limit=LOCAL_TRANSCRIPT_MAX_BYTES,
                description="MLX transcript output",
            )
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError):
            raise ValueError("Invalid MLX transcript JSON.") from None
        validate_whisper_transcript(result)
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
        try:
            url = httpx.URL(api_base)
        except (httpx.InvalidURL, ValueError):
            raise ValueError(
                "Invalid VOXHELM_API_BASE: expected an HTTP(S) service prefix."
            ) from None
        if (
            url.scheme not in {"http", "https"}
            or not url.host
            or url.userinfo
            or b"?" in url.raw_path
            or "#" in api_base
        ):
            raise ValueError(
                "Invalid VOXHELM_API_BASE: require HTTP(S), a host, and no credentials, query or fragment."
            )
        self.api_base = str(url).rstrip("/")
        path = url.raw_path.rstrip(b"/")
        if not path.endswith(b"/v1"):
            path += b"/v1"
        self._api_prefix = url.copy_with(raw_path=path)
        self.api_key = api_key
        self.model_name = model_name or "gpt-4o-mini-transcribe"
        self.language = language
        self.prompt = prompt

    @property
    def cache_service_identity(self) -> str:
        """The same validated, canonical API prefix used for requests."""
        return str(self._api_prefix)

    @property
    def transcription_url(self) -> str:
        return str(
            self._api_prefix.copy_with(
                raw_path=self._api_prefix.raw_path + b"/audio/transcriptions"
            )
        )

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
            try:
                response = remote_response(
                    "Voxhelm",
                    self.transcription_url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=timeout,
                )
            except httpx.TimeoutException as exc:
                raise timeout_error("Voxhelm", exc) from None
            except httpx.HTTPError:
                raise RuntimeError(
                    "Voxhelm transcription request failed (transport error)."
                ) from None
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                raise RuntimeError(
                    f"Voxhelm transcription failed with status {response.status_code}."
                ) from None

        result = response.json()
        validate_whisper_transcript(result)
        write_text_atomic(transcript_path, json.dumps(result))


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
        if (
            isinstance(processors, bool)
            or not isinstance(processors, int)
            or processors < 1
        ):
            raise ValueError("processors must be a positive integer")
        if model_name is None:
            model_name = "ggml-large-v3.bin"
        self.model_name = model_name
        self.model_path = settings.whisper_cpp_models_dir / model_name
        self.language = language
        self.prompt = prompt
        self.processors = processors

    @staticmethod
    def valid_wav(path: Path) -> bool:
        """Validate cached PCM through the public wave API with bounded reads."""
        try:
            with wave.open(str(path), "rb") as source:
                if (
                    source.getnchannels(),
                    source.getframerate(),
                    source.getsampwidth(),
                    source.getcomptype(),
                ) != (1, 16000, 2, "NONE"):
                    return False
                remaining = source.getnframes() * 2
                if remaining <= 0 or remaining > path.stat().st_size:
                    return False
                while remaining:
                    block = source.readframes(min(32768, remaining // 2))
                    if not block or len(block) % 2:
                        return False
                    remaining -= len(block)
                return True
        except (OSError, EOFError, wave.Error):
            return False

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
            if not WhisperCpp.valid_wav(tmp_path):
                raise RuntimeError(
                    "Converted WAV is invalid; require nonempty 16kHz mono PCM16 audio."
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
        run_checked(
            args,
            description=f"whisper-cli transcription of {input_path}",
            timeout=settings.local_transcription_timeout_seconds,
        )

    @staticmethod
    def transform_transcription(input_data: dict) -> dict:
        """
        Transform transcription data the whisper-cpp JSON to original Whisper format.
        """

        def timestamp_to_seconds(timestamp):
            return parse_dote_timecode(timestamp) / 1000

        try:
            if not isinstance(input_data["transcription"], list):
                raise ValueError
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

            result = {"segments": segments}
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            raise ValueError(
                "Invalid native transcript: require native segments and HH:MM:SS,mmm timestamps."
            ) from None
        validate_whisper_transcript(result)
        return result

    def convert_output_format(self, input_path: Path, output_path: Path) -> None:
        raw = read_regular_bounded(
            input_path,
            limit=LOCAL_TRANSCRIPT_MAX_BYTES,
            description="Native transcript output",
        )
        try:
            cpp_transcript = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError):
            raise ValueError("Invalid native transcript JSON.") from None
        transformed_transcript = self.transform_transcription(cpp_transcript)
        validate_whisper_transcript(transformed_transcript)
        write_text_atomic(output_path, json.dumps(transformed_transcript))

    def transcribe(self, audio_file: Path, transcript_path: Path) -> None:
        # Convert the audio file to WAV format
        wav_file = audio_file.with_suffix(".wav")
        if not self.valid_wav(wav_file):
            self.convert_to_wav(audio_file, wav_file)
        # Transcribe the WAV file
        whisper_transcript_path = transcript_path.with_suffix(
            ".whisper-cpp"
        )  # .json is automatically appended
        self.transcribe_wav(wav_file, whisper_transcript_path)
        whisper_read_transcript_path = transcript_path.with_suffix(".whisper-cpp.json")
        # Convert the whisper-cpp JSON to original Whisper format
        try:
            self.convert_output_format(whisper_read_transcript_path, transcript_path)
        except (ValueError, RuntimeError):
            # This run just published the intermediate; drop the refused copy only.
            whisper_read_transcript_path.unlink(missing_ok=True)
            raise
