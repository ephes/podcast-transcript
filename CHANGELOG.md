0.1.7 - Unreleased
==================

### Features

- Add a `voxhelm` backend for remote transcription via Voxhelm's OpenAI-compatible API.

### Fixes

- The Groq and Voxhelm backends no longer wait forever for a server that never answers. Requests use a
  10 s connect, 300 s write and 10 s pool timeout, and a read timeout set by the new
  `TRANSCRIPT_HTTP_READ_TIMEOUT` setting (seconds, default 1800). A timeout raises a `RuntimeError` naming
  the backend. Groq rate-limit (429) retries stop after 10 attempts with a `RuntimeError` carrying Groq's
  last rate-limit message.
- Make every pipeline step fail closed so a failed step is never cached as a success:
  - Downloads follow redirects, raise on HTTP errors and stream to a temporary file that is only
    renamed into place after the download completed.
  - ffmpeg (resample, chunk split, WAV conversion) and `whisper-cli` run with `check=True`, write to
    temporary outputs that are renamed into place on success, and include the tool's stderr in the error.
  - The chunk split records the complete chunk set in `chunks/chunks.json` after all chunks are in
    place. Chunk directories without this manifest (interrupted splits) are split again once; existing
    per-chunk transcripts are kept.
  - Input files named like generated chunks (`chunk_000.mp3`) are cached as `episode_chunk_000.mp3` so
    the chunk step can no longer overwrite the source audio.
  - The Groq backend raises on non-429 HTTP errors and on rate-limit responses without a usable wait
    time instead of returning without a transcript. Fractional and millisecond rate-limit waits
    (for example `2.5s` or `480ms`) are now parsed correctly.
  - Transcript, DOTe, Podlove, WebVTT and plain-text outputs are written atomically.
- Offset multi-chunk transcripts by each chunk's `ffprobe` audio duration instead of the end of its last
  speech line, which shifted timestamps early for episodes longer than 2 hours.
- Key each episode's cache directory by its readable name plus a short hash of the full URL or absolute
  file path (for example `audio-3f2a9c41d0b7`). Before, the directory was named after the file name up to
  the first `.`, so `.../123/audio.mp3` and `.../124/audio.mp3` (or `ep1.final.mp3` and `ep1.draft.mp3`)
  shared one cache and the second run silently returned the first episode's transcript.
  **Upgrade note:** existing cache directories (without the hash suffix) are not migrated, because the
  URL that produced them is unknown. They are left untouched and no longer used; the first run per
  episode after upgrading downloads and transcribes again. Delete the old directories once (see
  "Troubleshooting" in the README).
- Record the backend, model, language and prompt in a `cache.json` per episode. When they differ from
  the cached run (for example after switching `--backend`), the chunk transcripts and combined outputs
  are discarded and transcribed again instead of silently reusing the old backend's output.
- Parse `.env` lines on the first `=` only, so values containing `=` (for example base64 tokens) are
  no longer silently dropped. Blank lines, `#` comment lines and an optional `export ` prefix are
  handled, and one pair of matching surrounding quotes is stripped.
- Cache directories poisoned by earlier versions (for example a saved redirect or error page instead of the
  episode, or multi-chunk transcripts with drifting timestamps) are not repaired; with the hashed cache
  directories above they are simply no longer used. Delete them once; see "Troubleshooting" in the README.

### Development

- Add a GitHub Actions CI workflow that runs `just check` (pre-commit, mypy, pytest) on pushes and pull
  requests, with SHA-pinned actions, read-only permissions and no secrets.

### Security

- Refresh vulnerable locked dependencies flagged by `pip-audit`: anyio 4.14.2, click 8.3.3, filelock 3.20.3, fsspec 2026.6.0, idna 3.15, pygments 2.20.0, pytest 9.0.3, requests 2.33.0, torch 2.13.0, urllib3 2.8.0 and virtualenv 20.36.1.

### Documentation

- Document `VOXHELM_API_BASE` / `VOXHELM_API_KEY` configuration and `--backend voxhelm` usage.
- Add a README troubleshooting note for cache directories left behind by older versions.
- Document the hashed cache directory names, `cache.json` invalidation and `.env` syntax in the README.

0.1.6 - 2026-02-26
==================

### Fixes

- Fix uv build wheel contents.

### Documentation

- Add uvx --python MLX example.

0.1.5 - 2025-12-18
==================

### Features

- #13 Add parameter for number of processes to whisper-cpp backand.

### Fixes

- Make the `mlx` extra installable only on macOS/Apple Silicon.
- Improve MLX missing-deps CLI guidance (includes `uv sync` / `uv sync --extra mlx`).

### Documentation

- Update development setup instructions to use `uv sync`.
0.1.4 - 2025-01-12
==================

### Fixes

- Fix the wheel build.

0.1.3 - 2025-01-12
==================

### Features

- #13 Add support for whisper-cpp as a local transcription option (new default).

0.1.2 - 2025-01-11
==================

### Features

- #5 Support for plain text format output.
- #8 Retry logic for failed transcription requests when rate limited.
- #7 Copy DOTe transcript JSON instead of creating a symlink.
- #9 Local transcriptions using the MLX-Api
- #10 Transcribe also from files not just from URLs and do not require MP3 format but convert automatically.
- #11 Added coverage reporting.

0.1.1 - 2024-11-23
==================

### Fixes

- Split the audio into chunks if it exceeds the duration limit (7200 seconds).
- Fixed the order in which the chunks are transcribed.
- Exit with status code 0 if the transcription was successful, 1 otherwise.

### Features

- Show generated transcript files in the output.
- Make the model, prompt, and language configurable via environment variables / .env file.

### Documentation

- Added a roadmap section to the README.

0.1.0 - 2024-11-17
==================

### Initial Release

It works for single track mp3 file urls.
