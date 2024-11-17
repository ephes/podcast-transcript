# Podcast Transcript

A simple command-line tool to generate transcripts for podcast episodes.

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

- Download and process podcast episodes from a given MP3 URL.
- Automatically resamples audio to 16kHz mono because Groq will do this anyway.
- Splits large audio files into manageable chunks.
- Transcribes audio using the Groq API.
- Outputs transcripts in multiple formats:
  - DOTe JSON
  - Podlove JSON
  - WebVTT (subtitle format)

## Prerequisites

- Python 3.10 or higher
- [ffmpeg](https://ffmpeg.org/) installed and available in your system’s PATH.
- A [Groq API key](https://groq.com/) for transcription services.

## Installation

1**Install the package**:

```shell
pip install podcast-transcript  # or pipx/uvx install podcast-transcript
```

## Configuration

### Setting the Groq API Key

The application requires a Groq API key to function. You can set the API key in one of the following ways:

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

### Transcript Directory

By default, transcripts are stored in ~/.podcast-transcripts/.
You can change this by setting the TRANSCRIPT_DIR environment variable:

```shell
export TRANSCRIPT_DIR=/path/to/your/transcripts
```

## Usage

To transcribe a podcast episode, run the transcribe command followed by the URL of the MP3 file:

```shell
transcribe <mp3_url>
```

Example:

```shell
transcribe https://d2mmy4gxasde9x.cloudfront.net/cast_audio/pp_53.mp3
```

## Detailed Steps

The transcription process involves the following steps:

1. Download the MP3 file from the provided URL.
2. Reample the audio to 16kHz mono for optimal transcription.
3. Split the audio into chunks if it exceeds the size limit (25 MB).
4. Transcribe each audio chunk using the Groq API.
5. Combine the transcribed chunks into a single transcript.
6. Generate output files in DOTe JSON, Podlove JSON, and WebVTT formats.

The output files are saved in a directory named after the episode, within the transcript directory.

## Output Formats

- **DOTe JSON (*.dote.json)**: A JSON format suitable for further processing or integration with other tools.
- **Podlove JSON (*.podlove.json)**: A JSON format compatible with [Podlove](https://podlove.org/) transcripts.
- **WebVTT (*.vtt)**: A subtitle format that can be used for captioning in media players.

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
uv sync
```

### Running Tests

The project uses pytest for testing. To run tests:
```shell
pytest
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
