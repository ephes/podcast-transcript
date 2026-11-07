from __future__ import annotations

import argparse
from typing import TYPE_CHECKING
import importlib.util
from rich.console import Console

if TYPE_CHECKING:
    from .config import Settings
    from . import backends


def __getattr__(name):
    # Preserve the package's existing reexports without preparing a CLI run.
    if name in {"settings", "Settings"}:
        from . import config

        return getattr(config, name)
    if name == "transcribe":
        from .single_track import transcribe

        return transcribe
    raise AttributeError(name)


def groq_from_settings(my_settings: Settings) -> backends.Groq:
    from . import backends

    return backends.Groq(
        api_key=my_settings.groq_api_key,
        model_name=my_settings.transcript_model_name,
        language=my_settings.transcript_language,
        prompt=my_settings.transcript_prompt,
    )


def mlx_from_settings(my_settings: Settings) -> backends.MLX:
    from . import backends

    return backends.MLX(
        model_name=my_settings.transcript_model_name,
        language=my_settings.transcript_language,
        prompt=my_settings.transcript_prompt,
    )


def voxhelm_from_settings(my_settings: Settings) -> backends.Voxhelm:
    from . import backends

    if not my_settings.voxhelm_api_base:
        raise ValueError("VOXHELM_API_BASE is not set.")
    if not my_settings.voxhelm_api_key:
        raise ValueError("VOXHELM_API_KEY is not set.")
    return backends.Voxhelm(
        api_base=my_settings.voxhelm_api_base,
        api_key=my_settings.voxhelm_api_key,
        model_name=my_settings.transcript_model_name,
        language=my_settings.transcript_language,
        prompt=my_settings.transcript_prompt,
    )


def whisper_cpp_from_settings_and_args(
    my_settings: Settings, args: argparse.Namespace
) -> backends.WhisperCpp:
    from . import backends

    return backends.WhisperCpp(
        model_name=my_settings.transcript_model_name,
        language=my_settings.transcript_language,
        prompt=my_settings.transcript_prompt,
        processors=args.processors,
    )


def transcribe_cli():
    console = Console()

    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="Transcribe audio from a local file path or an HTTP(S) URL."
    )

    parser.add_argument(
        "mp3_url",
        metavar="source",
        type=str,
        help="Local file path or HTTP(S) URL of the audio to transcribe.",
    )
    parser.add_argument(
        "--backend",
        choices=["groq", "mlx", "voxhelm", "whisper-cpp"],
        default="whisper-cpp",
        help=(
            "Transcription backend. Choose 'groq' for Groq-based transcription, "
            "'mlx' for MLX-based local transcription, 'voxhelm' for a remote "
            "Voxhelm service, or 'whisper-cpp' for whisper-cpp based local "
            "transcription (default)."
        ),
    )
    parser.add_argument(
        "--processors",
        type=int,
        default="4",
        help=(
            "Number of processors to use for whisper-cpp based local transcription."
            "Defaults to 4. Only applicable when using whisper-cpp backend. When whisper-cpp "
            "uses the GPU instead of the CPU, this argument is also ignored."
        ),
    )
    try:
        args = parser.parse_args()
        mp3_url = args.mp3_url
    except argparse.ArgumentError as e:
        console.print(f"[red]Argument parsing error: {e}[/red]")
        exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error during argument parsing: {e}[/red]")
        exit(1)

    # Start transcription process
    from .audio import safe_display_source, safe_diagnostic

    try:
        from . import settings, transcribe

        console.print(
            f"Starting transcription for: {safe_display_source(mp3_url)}",
            style="blue",
            markup=False,
        )
        if args.backend == "mlx":
            if importlib.util.find_spec("mlx_whisper") is None:
                console.print(
                    "MLX backend dependencies are not installed.", style="red"
                )
                console.print(
                    'Install them with: pip install "podcast-transcript[mlx]" '
                    'or uv pip install "podcast-transcript[mlx]".',
                    markup=False,
                )
                console.print(
                    'In a checkout of this repo: uv pip install -e ".[mlx]" '
                    "or uv sync (or uv sync --extra mlx).",
                    markup=False,
                )
                console.print(
                    'Or run without installing: uvx --from "podcast-transcript[mlx]" transcribe --backend mlx <audio_source>.',
                    markup=False,
                )
                console.print("Note: MLX runs on macOS on Apple Silicon.", style="dim")
                exit(1)
            backend = mlx_from_settings(settings)
        elif args.backend == "voxhelm":
            backend = voxhelm_from_settings(settings)
        elif args.backend == "groq":
            backend = groq_from_settings(settings)
        elif args.backend == "whisper-cpp":
            backend = whisper_cpp_from_settings_and_args(settings, args)
        else:
            console.print("[red]Invalid service argument.[/red]")
            exit(1)
        transcript_paths = transcribe(mp3_url, backend)
        for name, path in transcript_paths.items():
            console.print(
                f"[green]Transcript in {name} format saved to:[/green] {path}"
            )
        console.print("[green]Transcription complete![/green]")
        exit(0)
    except Exception as e:
        console.print(
            f"Error during transcription: {safe_diagnostic(str(e))}",
            style="red",
            markup=False,
        )
        exit(1)


if __name__ == "__main__":
    transcribe_cli()
