import argparse
from rich.console import Console

from .single_track import transcribe
from .config import settings, Settings
from . import backends


def groq_from_settings(my_settings: Settings) -> backends.Groq:
    return backends.Groq(
        api_key=my_settings.groq_api_key,
        model_name=my_settings.transcript_model_name,
        language=my_settings.transcript_language,
        prompt=my_settings.transcript_prompt,
    )


def mlx_from_settings(my_settings: Settings) -> backends.MLX:
    return backends.MLX(
        model_name=my_settings.transcript_model_name,
        language=my_settings.transcript_language,
        prompt=my_settings.transcript_prompt,
    )


def whisper_cpp_from_settings_and_args(
    my_settings: Settings, args: argparse.Namespace
) -> backends.WhisperCpp:
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
        description="Transcribe an MP3 file from a given URL."
    )

    # Add the mp3_url positional argument
    parser.add_argument("mp3_url", type=str, help="URL of the MP3 file to transcribe.")
    parser.add_argument(
        "--backend",
        choices=["groq", "mlx", "whisper-cpp"],
        default="whisper-cpp",
        help=(
            "Transcription backend. Choose 'groq' for Groq-based transcription, "
            "'mlx' for MLX-based local transcription, or 'whisper-cpp' for "
            "whisper-cpp based local transcription (default)."
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
    try:
        console.print(f"[blue]Starting transcription for:[/blue] {mp3_url}")
        if args.backend == "mlx":
            backend = mlx_from_settings(settings)
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
        console.print(f"[red]Error during transcription: {e}[/red]")
        exit(1)


if __name__ == "__main__":
    transcribe_cli()
