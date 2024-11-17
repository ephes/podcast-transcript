import os
import shutil

from pathlib import Path

from rich.console import Console


class Settings:
    """
    A base class for settings holding the applications root dir
    in the users home directory.
    """

    transcript_dir: Path
    groq_api_key: str

    def __init__(self):
        self.console = Console()
        if (transcript_dir := os.getenv("TRANSCRIPT_DIR")) is not None:
            self.transcript_dir = Path(transcript_dir)
        else:
            self.transcript_dir = Path.home() / ".podcast-transcripts"

        # Create the transcript directory if it does not exist
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

        # Read the .env file
        env_file = self.transcript_dir / ".env"
        if env_file.exists():
            self.read_env_file(env_file)

        # Make sure the groq api key is set
        if not hasattr(self, "groq_api_key"):
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            if self.groq_api_key is None:
                self.console.print(
                    "Error: GROQ_API_KEY is not set in the environment variables or .env file.",
                    style="bold red",
                )
                exit(1)

        # Check if ffmpeg is installed
        if shutil.which("ffmpeg") is None:
            self.console.print(
                "Error: ffmpeg is not installed or not found in PATH. Please install ffmpeg and ensure it's in your system's PATH.",
                style="bold red",
            )
            exit(1)

    def read_env_file(self, env_file: Path):
        """
        Read the variables from .env file and set the attributes accordingly.
        """
        with env_file.open("r") as f:
            for line in f:
                key, value = line.strip().split("=")
                setattr(self, key.lower(), value)


settings = Settings()
