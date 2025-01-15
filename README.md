# Podcast Transcript

A simple command-line tool to generate transcripts for podcast episodes or other audio files containing speech.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Formats](#output-formats)
- [Development](#development)
  - [Running Tests](#running-tests)
  - [Code Style and Linting](#code-style-and-linting)
- [License](#license)
- [Author](#author)

## Features

- Download and process podcast episodes or other audio content from a given URL or file path.
- Automatically resamples audio to 16kHz mono because Groq will do this anyway.
- Splits large audio files into manageable chunks.
- Transcribes audio locally using [whisper-cpp](https://github.com/ggerganov/whisper.cpp)
- Optionally transcribes audio locally using [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper).
- Optionally transcribes audio using the Groq API.
- Outputs transcripts in multiple formats:
  - DOTe JSON
  - Podlove JSON
  - WebVTT (subtitle format)
  - Plaintext

## Prerequisites

- Python >=3.10 (mlx does not run with 3.13, but works with 3.12)
- [ffmpeg](https://ffmpeg.org/) installed and available in your system’s PATH.
- A [Groq API key](https://groq.com/) for transcription services.

## Installation

1**Install the package**:

```shell
pip install podcast-transcript  # or pipx/uvx install podcast-transcript
```

## Configuration

### Setting the Groq API Key

Using the Groq backend requires a Groq API key to function. You can set the API key in one of the following ways:

- **Environment Variable**:

Set the GROQ_API_KEY environment variable in your shell:
```shell
export GROQ_API_KEY=your_api_key_here
# or
GROQ_API_KEY=your_api_key_here podcast-transcript ...
```

- **.env File**:

Create a .env file in the transcript directory (default is ~/.podcast-transcripts/) and add the following line:
```shell
GROQ_API_KEY=your_api_key_here
```

### Transcript Home

By default, the transcripts home directory is ~/.podcast-transcripts/. You can change this by setting
the TRANSCRIPT_HOME environment variable:

```shell
export TRANSCRIPT_HOME=/path/to/your/transcripts_home
```

The transcript home directory is the place where you could store your .env file. The model files
for the whisper-cpp backend are also stored in the transcript home directory in a directory
called `whisper-cpp-models`. The transcripts themselves are stored in a directory called `transcripts`
unless you specify a different directory.

### Transcripts Directory

By default, transcripts are stored in `~/.podcast-transcripts/transcripts/`.
You can change this by setting the TRANSCRIPT_DIR environment variable:

```shell
export TRANSCRIPT_DIR=/path/to/your/transcripts
```

### Other Configuration Options

You can also set the following environment variables or specify them in the .env file:

- **TRANSCRIPT_MODEL_NAME**: The name of the model to use for the transcript
  (default is "ggml-large-v3.bin" for whisper-cpp, "whisper-large-v3" for Groq and "mlx-community/whisper-large-v3-mlx" for MLX).
- **TRANSCRIPT_PROMPT**: The prompt to use for the transcription (default is "podcast-transcript").
- **TRANSCRIPT_LANGUAGE**: The language code for the transcription (default is en, you could set it to de for example).

## Usage

To transcribe a podcast episode, run the transcribe command followed by the URL of the MP3 file:

```shell
transcribe <mp3_url>
```

Example:

```shell
transcribe https://d2mmy4gxasde9x.cloudfront.net/cast_audio/pp_53.mp3
```

Or if you want to use the Groq API:
```shell
transcribe --backend=groq https://d2mmy4gxasde9x.cloudfront.net/cast_audio/pp_53.mp3
```

## Detailed Steps

The transcription process involves the following steps:

1. Download the audio file from the provided URL or copy it from the file path if one was given.
2. Convert the audio to mp3 and resample to 16kHz mono for optimal transcription.
3. Split the audio into chunks if it exceeds the size limit (25 MB).
4. Transcribe each audio chunk using either whisper-cpp (converts mp3 to wav first), mlx-whisper, or the Groq API.
5. Combine the transcribed chunks into a single transcript.
6. Generate output files in DOTe JSON, Podlove JSON, and WebVTT formats.

The output files are saved in a directory named after the episode, within the transcript directory.

## Output Formats

- **DOTe JSON (*.dote.json)**: A JSON format suitable for further processing or integration with other tools.
- **Podlove JSON (*.podlove.json)**: A JSON format compatible with [Podlove](https://podlove.org/) transcripts.
- **WebVTT (*.vtt)**: A subtitle format that can be used for captioning in media players.
- **Plaintext**: Just the plain text of the transcription.

## Roadmap

- [ ] Support for multitrack transcripts with speaker identification.
- [x] Add support for other transcription backends (e.g., openAI, speechmatics, local whisper via pytorch).
- [x] Add support for other audio formats (e.g., AAC, WAV, FLAC).
- [ ] Add more output formats (e.g., SRT, TTML).

## Development

### Install Development Version

1. **Clone the repository**:

```shell
git clone https://github.com/yourusername/podcast-transcript.git
cd podcast-transcript
```

2. **Create a virtual environment**:

```shell
uv venv
```

3. **Install the package in editable mode**:

```shell
uv pip install -e .
```

### Running Tests

The project uses pytest for testing. To run tests:
```shell
pytest
```

Show coverage:
```shell
coverage run -m pytest && coverage html && open htmlcov/index.html
```

### Code Style and Linting

Install pre-commit hooks to ensure code consistency:
```shell
pre-commit install
```

Check the type hints:
```shell
mypy src/
```

### Publish a Release

Build the distribution package:
```shell
uv build
```

Publish the package to PyPI:
```shell
uv publish --token your_pypi_token
```

## License

This project is licensed under the MIT License.

## Author

- [Jochen Wersdörfer](mailto:jochen-transcript@wersdoerfer.de)
