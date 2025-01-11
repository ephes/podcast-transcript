import argparse
from rich.console import Console

from .single_track import transcribe
from .config import settings, Settings
from . import services


def groq_from_settings(my_settings: Settings) -> services.Groq:
    return services.Groq(
        api_key=my_settings.groq_api_key,
        model_name=my_settings.transcript_model_name,
        language=my_settings.transcript_language,
        prompt=my_settings.transcript_prompt,
    )


def mlx_from_settings(my_settings: Settings) -> services.MLX:
    return services.MLX(
        model_name=my_settings.transcript_model_name,
        language=my_settings.transcript_language,
        prompt=my_settings.transcript_prompt,
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
        "--service",
        choices=["groq", "local"],
        default="local",
        help="Transcription mode. Choose 'groq' for Groq-based transcription, or 'local' for MLX-based local transcription (default).",
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
        if args.service == "local":
            service = mlx_from_settings(settings)
        elif args.service == "groq":
            service = groq_from_settings(settings)
        else:
            console.print("[red]Invalid service argument.[/red]")
            exit(1)
        transcript_paths = transcribe(mp3_url, service)
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
