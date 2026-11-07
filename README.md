# Podcast Transcript

A simple command-line tool to generate transcripts for podcast episodes or other audio files containing speech.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Formats](#output-formats)
- [Troubleshooting](#troubleshooting)
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
- Optionally transcribes audio through a remote Voxhelm service.
- Outputs transcripts in multiple formats:
  - DOTe JSON
  - Podlove JSON
  - WebVTT (subtitle format)
  - Plaintext

## Prerequisites

- Python >=3.10
  - MLX backend (via `mlx-whisper`) requires macOS on Apple Silicon and a Python version supported by the `mlx-whisper` / `torch` wheels.
- [ffmpeg](https://ffmpeg.org/) installed and available in your system’s PATH.
- For the default `whisper-cpp` backend: `whisper-cli` on PATH and a model in
  `TRANSCRIPT_HOME/whisper-cpp-models` (see [Configuration](#configuration)).
- Local `whisper-cpp` and MLX backends need no API credentials. Only `--backend groq` requires
  a [Groq API key](https://groq.com/); `--backend voxhelm` requires a Voxhelm API base and key.

Settings currently prints a missing `GROQ_API_KEY` warning even for other backends and `--help`.
This warning is harmless when using `whisper-cpp`, MLX or Voxhelm; a Groq key is required only
when selecting `--backend groq`.

## Installation

1. **Install the package**:

```shell
pip install podcast-transcript
# or install an isolated command-line tool:
pipx install podcast-transcript
# or:
uv tool install podcast-transcript
```

2. **(Optional) Install MLX backend dependencies** (macOS on Apple Silicon only):

```shell
pip install "podcast-transcript[mlx]"
# or
uv pip install "podcast-transcript[mlx]"
```

For a one-shot run without a persistent tool installation:

```shell
uvx --from podcast-transcript transcribe <audio_source>
# or with the optional MLX backend:
uvx --from "podcast-transcript[mlx]" transcribe --backend mlx <audio_source>
```

Example (pin Python + package version):

```shell
time TRANSCRIPT_LANGUAGE=de uvx --python python3.14.2 --from "podcast-transcript[mlx]==0.1.5" transcribe --backend mlx https://d2mmy4gxasde9x.cloudfront.net/cast_audio/pp_67.mp3
```

## Configuration

### Setting the Groq API Key

Using the Groq backend requires a Groq API key to function. You can set the API key in one of the following ways:

- **Environment Variable**:

Set the GROQ_API_KEY environment variable in your shell:
```shell
export GROQ_API_KEY=your_api_key_here
# or
GROQ_API_KEY=your_api_key_here transcribe --backend groq <audio_source>
```

- **.env File**:

Create a .env file in the transcript directory (default is ~/.podcast-transcripts/) and add the following line:
```shell
GROQ_API_KEY=your_api_key_here
```

The .env file has one `KEY=value` per line. Only the first `=` separates key and value, so values may
contain `=`. Blank lines and lines starting with `#` are ignored, an optional `export ` prefix is
allowed, and one pair of surrounding quotes is removed from the value. Comments after a value are not
supported; a `#` there is part of the value.

### Configuring the Voxhelm Backend

Set these variables when using `--backend voxhelm`:

```shell
export VOXHELM_API_BASE=https://voxhelm.home.xn--wersdrfer-47a.de
export VOXHELM_API_KEY=your_voxhelm_token_here
```

`VOXHELM_API_BASE` can point at either the service root or the `/v1` API prefix. The backend uses
`gpt-4o-mini-transcribe` by default, or `TRANSCRIPT_MODEL_NAME` if you override it.

Voxhelm transcript caches include the service's normalized API prefix. Switching services
retranscribes; root and `/v1` aliases share a cache, as do default ports and host casing.
API bases require HTTP(S) and a host; credentials, queries and fragments are rejected before
uploads. Custom path prefixes are supported (for example `/proxy/v1`); a prefix without `/v1`
has `/v1` appended. The validated prefix is shared by requests and cache identity. API keys are excluded from this identity;
key rotation alone does not invalidate transcripts. Existing Voxhelm caches without a service
identity are transcribed again once. Audio caches are retained.

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

Relative `TRANSCRIPT_HOME` and `TRANSCRIPT_DIR` paths are resolved from the working directory
where you run `transcribe`. Short-audio chunk links work with relative paths, including spaces.

### Other Configuration Options

You can also set the following environment variables or specify them in the .env file:

- **TRANSCRIPT_MODEL_NAME**: The name of the model to use for the transcript
  (default is "ggml-large-v3.bin" for whisper-cpp, "whisper-large-v3" for Groq and "mlx-community/whisper-large-v3-mlx" for MLX).
- **TRANSCRIPT_PROMPT**: The prompt to use for the transcription (default is "podcast-transcript").
- **TRANSCRIPT_LANGUAGE**: The language code for the transcription (default is en, you could set it to de for example).
- **TRANSCRIPT_HTTP_READ_TIMEOUT**: Seconds the Groq and Voxhelm backends wait for the server's response after
  the upload (default is 1800, 30 minutes). Connecting is limited to 10 seconds, each upload write to 300 seconds.
  A timeout stops the run with an error naming the backend. The Groq backend retries rate-limited (429) requests
  at most 10 times in total and then stops with Groq's last rate-limit message.

Remote Groq and Voxhelm responses (success and error) are streamed with a 32 MiB decoded-body
limit, independent of Content-Length or compression. Overflow closes the response, fails the run
and preserves previous transcript files. Displayed backend error details are limited to 2,000
characters, including rate-limit messages.

Source progress and request-error diagnostics omit URL credentials, query strings and fragments,
including redirect URLs. Requests and source cache hashing still use the original URL.

HTTP(S) source downloads have separate budgets (environment or `.env`):

- **TRANSCRIPT_SOURCE_MAX_BYTES**: Positive integer maximum for the decoded response body;
  default 2147483648 (2 GiB), suitable for long episodes. Declared oversized responses are rejected
  early; actual streamed bytes are always counted, including when Content-Length is absent.
- **TRANSCRIPT_SOURCE_TOTAL_TIMEOUT**: Positive finite total seconds, default 3600 (one hour).
  Monotonic elapsed time is checked at each response (including redirects), streamed block and EOF.
  A blocked operation can take up to 60 seconds (30 for connecting), capped by this setting,
  before control returns to the deadline check.

Raise these limits for exceptionally large episodes or slow connections, for example
`TRANSCRIPT_SOURCE_MAX_BYTES=4294967296 TRANSCRIPT_SOURCE_TOTAL_TIMEOUT=7200 transcribe <url>`.
Rejected downloads close their response and remove partial files; an existing target is preserved.
These limits apply only to source HTTP(S) downloads, not local copies or backend uploads.

## Usage

The installed executable is `transcribe`. Pass a local audio file path or an HTTP(S) URL;
MP3, AAC, WAV and FLAC input are supported. The default backend is local `whisper-cpp`:

```shell
transcribe <audio_source>
transcribe "/path/to/local audio.wav"
transcribe --help
```

Example:

```shell
transcribe https://d2mmy4gxasde9x.cloudfront.net/cast_audio/pp_53.mp3
```

Or if you want to use the Groq API:
```shell
transcribe --backend=groq https://d2mmy4gxasde9x.cloudfront.net/cast_audio/pp_53.mp3
```

Or if you want to use the MLX backend (requires the `mlx` extra; macOS on Apple Silicon only):
```shell
transcribe --backend=mlx https://d2mmy4gxasde9x.cloudfront.net/cast_audio/pp_53.mp3
```

Or if you want to use a remote Voxhelm service:
```shell
VOXHELM_API_BASE=https://voxhelm.home.xn--wersdrfer-47a.de \
VOXHELM_API_KEY=your_voxhelm_token_here \
transcribe --backend=voxhelm https://d2mmy4gxasde9x.cloudfront.net/cast_audio/pp_53.mp3
```

## Detailed Steps

The transcription process involves the following steps:

1. Download the audio file from the provided URL or copy it from the file path if one was given.
2. Convert the audio to mp3 and resample to 16kHz mono for optimal transcription.
3. Split the audio into chunks if it exceeds 25 MiB or 7,200 seconds; enforce both limits before publishing the chunk manifest.
4. Transcribe each audio chunk using either whisper-cpp (converts mp3 to wav first), mlx-whisper, the Groq API, or Voxhelm.
5. Combine the transcribed chunks into a single transcript.
6. Generate output files in DOTe JSON, Podlove JSON, and WebVTT formats.

The output files are saved in a directory within the transcript directory. The directory name is the
episode's file name (up to the first `.`) plus a short hash of the full URL or absolute file path, for
example `audio-3f2a9c41d0b7`. Episodes that share a file name, such as `.../123/audio.mp3` and
`.../124/audio.mp3`, therefore get separate directories. The output files themselves keep the readable
name (`audio.dote.json`, `audio.txt`, ...).

HTTP(S) downloads use only the URL path for their local filename. Percent escapes are decoded and
characters outside letters, digits, dots, underscores and hyphens become underscores; empty or
dot-only names use `episode.mp3`. Signed URL queries and fragments stay out of filenames, while the
full URL still identifies the cache hash. URL names requiring sanitization now have a different
readable cache-directory stem; their old directories are left untouched and a one-time download and
transcription rebuild occurs. Copy any outputs you need before deleting those old directories. Generated chunk names are reserved and prefixed with `episode_`.

Local files are fingerprinted by SHA-256 on each run. Editing or replacing audio at the same path
invalidates copied, resampled and chunk audio plus all transcripts, even when the size stays the same.
Local caches without a fingerprint are rebuilt once. The fingerprint is recorded after audio
preparation succeeds; backend-setting invalidation remains independent. The original source must
remain available on each run for content verification; a missing source produces an error asking you
to restore it, even if copied audio is cached.

Each step caches its result as a file in that directory and is skipped on later runs if the file
exists. The directory also holds a `cache.json` that records the backend, model, language and prompt
used for the transcripts. If you re-run with different settings (for example `--backend groq` after
`whisper-cpp`, or another `TRANSCRIPT_MODEL_NAME`), the cached chunk transcripts and combined outputs are
discarded and transcribed again; the downloaded, resampled and chunked audio is reused. Steps only move their output into place after they succeeded, so a failed download, ffmpeg
run or transcription raises an error and is retried on the next run. Chunked transcripts are combined
using each chunk's audio duration, so timestamps stay aligned for episodes longer than 2 hours.

MLX caches also include `word_timestamps`. Changing this option retranscribes while keeping
prepared audio; older MLX caches rebuild transcripts once. Remote API key rotation still reuses
transcripts.

Each `transcribe` invocation holds a per-episode OS file lock from preparation through final exports.
A concurrent invocation for the same episode exits with “Episode cache is busy”; retry when the
first run finishes. Other episodes can run concurrently. Locks are released on failures and process
exit. Leave library-managed files in the `.locks` directory under `TRANSCRIPT_DIR` alone
(they persist on Unix); do not manually remove locks or episode data while a transcription is active.

### Transcript and cached-artifact validation

All backend results must be Whisper objects containing a `segments` list. An empty list is a valid
silent transcript. Each segment requires string `text` plus numeric `start` and `end` in seconds;
optional top-level `text` must also be a string. Times must be finite, nonnegative, representable
in milliseconds, and ordered (`end >= start`); booleans and numeric strings are rejected.
Zero-length cues are allowed. Valid fractional seconds retain nearest-millisecond rounding.
Invalid backend results never replace raw transcript caches, so a corrected response can be retried.

Cached raw chunks are validated before reuse. If an older or edited raw chunk is rejected, remove
that `chunks/chunk_NNN.json` and its derived `chunk_NNN.dote.json`, plus the combined episode
outputs, then rerun to retranscribe that chunk. Valid prepared audio and other raw chunks are reused.
Errors identify the cached filename and segment position without including transcript text.

DOTe `speakerDesignation` is optional (defaults to an empty speaker in Podlove); when present it
must be a string. Cached DOTe cues use full `HH:MM:SS,mmm` strings: at least two hour digits, exactly two minute/second
digits (00–59), and exactly three millisecond digits. Hours may exceed 24. Suffixes, malformed
widths and reversed cues are rejected consistently before combining (including single-chunk copies)
or writing any export. Existing output bytes are preserved on validation failure. Correct or remove
the invalid DOTe cache and regenerate its combined exports before retrying.

## Output Formats

- **DOTe JSON (*.dote.json)**: A JSON format suitable for further processing or integration with other tools.
- **Podlove JSON (*.podlove.json)**: A JSON format compatible with [Podlove](https://podlove.org/) transcripts.
- **WebVTT (*.webvtt)**: A subtitle format that can be used for captioning in media players.
  Cue text treats `&`, `<` and `>` as literal text, escaping them as entities. CRLF and CR
  become LF; consecutive blank or whitespace-only lines collapse to one line break, and leading
  and trailing newlines are removed so each segment remains one cue. Other line breaks are kept.
- **Plaintext**: Just the plain text of the transcription.

## Troubleshooting

### Regenerating cached WebVTT exports

Existing `.webvtt` files remain cached. To apply the literal-text escaping fix to an existing export,
delete just that file in the episode cache directory and rerun the same transcription command with
the same backend settings. The cached transcript is reused to generate the corrected WebVTT file.

### Cache directories from versions before 0.1.7

Before 0.1.7 the cache directory was named after the episode's file name only, so episodes with the
same file name shared one directory and a later run could return an earlier episode's transcript.
Those versions could also cache a failed step as if it had succeeded (for example a saved redirect or
error page instead of the episode audio, or a partial ffmpeg output), and multi-chunk transcripts
(episodes longer than 2 hours or larger than 25 MB after resampling) could have timestamps that drift
early.

0.1.7 does not migrate these directories, because the URL that produced a directory is unknown. They
are named without a hash suffix (for example `audio`) and are no longer read or modified: the first
run per episode after upgrading downloads and transcribes the episode again into a new, hashed
directory, so none of the problems above carry over. Copy any transcripts you still need out of the
old directories, then delete them once:

```shell
ls ~/.podcast-transcripts/transcripts/  # or $TRANSCRIPT_DIR
rm -r ~/.podcast-transcripts/transcripts/<old-episode-name>
```

### Chunk upload limits and local command deadlines

Chunks must be at most 25 MiB (26,214,400 bytes) and 7,200 seconds. Stream-copy splitting estimates
segment duration from the resampled file's size and duration, with 10% headroom for frame/header
and bitrate variation. Every new chunk is checked before any is published. If a variable-bitrate
chunk still exceeds either limit, the run fails clearly without a completion manifest; provide
shorter source files and retry. Cached chunks exceeding the byte limit are rebuilt automatically.
Short files within both limits still use a symlink without re-encoding.

Local commands also have finite budgets (positive finite seconds, environment or `.env`):

- `TRANSCRIPT_MEDIA_TIMEOUT`: 3600 (one hour) per ffmpeg resample, split or WAV conversion.
- `TRANSCRIPT_LOCAL_TRANSCRIPTION_TIMEOUT`: 21600 (six hours) per whisper-cli chunk transcription.

Increase these for unusually slow processing. On timeout or interruption the owned command tree is
killed and the direct subprocess reaped (Unix process group; Windows `taskkill /T`). Partial outputs
are removed and previous final outputs preserved. Timeouts raise an error naming the operation;
failed commands report a bounded stderr tail. Retrying reruns the unfinished step.

### Invalid audio duration or a stalled probe

Each `ffprobe` process has a 60-second limit and is killed and reaped on timeout. Duration metadata
must contain a finite, nonnegative number (zero is valid). Missing/malformed metadata and process
failures stop the pipeline with a bounded stderr diagnostic before transcription. Newly split chunks
are probed before publishing chunk/source completion markers. A failed probe on an already cached
chunk preserves existing combined outputs; delete `chunks/chunks.json` to regenerate the chunks
from cached resampled audio, or delete the episode directory to download a corrected URL source.
Fix a failing ffprobe installation before rerunning.

### Recovering an invalid chunk manifest

`chunks/chunks.json` accepts only consecutive generated MP3 basenames starting at
`chunk_000.mp3`, in order, without duplicates. Each must be a regular file; the single short-audio
chunk may be a symlink to the episode's own resampled file. Absolute paths, traversal, other
symlinks and malformed manifests are rejected. Rerun the same command to rebuild an invalid chunk
set and its transcripts automatically; downloaded and resampled audio is reused. Invalid manifests
remain in place until transcript cleanup finishes, so interrupted cleanup is retried.

### Starting an episode over

Failed steps are not cached, so re-running `transcribe` retries them. To force a fresh download and
transcription of one episode, delete its directory and run `transcribe` again:

```shell
rm -r ~/.podcast-transcripts/transcripts/<episode-directory>  # or $TRANSCRIPT_DIR/<episode-directory>
```

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

2. **Sync the project environment**:

```shell
uv sync
```

This creates `.venv` and installs the project plus dev dependencies.

To ensure you’re using the repo’s pinned Python version, you can also run:
```shell
UV_PYTHON=python"$(cat .python-version)" uv sync
```

Common developer commands:
```shell
just lint
just typecheck
just test
```

`just check` runs all three on the pinned `.python-version`. `just test-oldest-python` runs the
test suite on Python 3.10, the oldest version allowed by `requires-python`, in an isolated
environment. GitHub Actions runs both on every push and pull request
(`.github/workflows/ci.yml`).

### Running Tests

The project uses pytest for testing. To run tests:
```shell
just test
# or: pytest
```

Show coverage:
```shell
coverage run -m pytest && coverage html && open htmlcov/index.html
```

CLI tests exercise successful dispatch through all four real backend factories with synthetic
settings, replacing only the transcription boundary; optional MLX native imports are isolated.
They check default backend/model selection, explicit processor and settings forwarding, output paths,
and failure exits. Tests set isolated transcript directories before importing package settings and
use no accounts or real audio.

### Code Style and Linting

Install pre-commit hooks to ensure code consistency:
```shell
pre-commit install
```

Check the type hints:
```shell
just typecheck
# or: mypy src/
```

Run lint/format:
```shell
just lint
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
