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
    transcript_prompt: str = "podcast-transcript"
    transcript_model_name: str | None = None
    transcript_language: str = "en"

    def __init__(self):
        self.console = Console()

        # Set the transcript directory - this is special because we want to create
        # the directory if it does not exist which is why we cannot wait until read_env_vars()
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

        # Read environment variables
        self.read_env_vars()

        # Make sure the groq api key is set - this is special because the api key is required
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
                try:
                    key, value = line.strip().split("=")
                    value = value.strip('"')
                    setattr(self, key.lower(), value)
                except ValueError:
                    pass

    def read_env_vars(self):
        """
        Read the environment variables and set the attributes accordingly.
        """
        transcript_keys = {
            "TRANSCRIPT_PROMPT",
            "TRANSCRIPT_MODEL_NAME",
            "TRANSCRIPT_LANGUAGE",
        }
        for key, value in os.environ.items():
            if key in transcript_keys:
                setattr(self, key.lower(), value)


settings = Settings()
