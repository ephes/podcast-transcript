0.1.7 - Unreleased
==================

### Features

- Add a `voxhelm` backend for remote transcription via Voxhelm's OpenAI-compatible API.

### Fixes

- Make every pipeline step fail closed so a failed step is never cached as a success:
  - Downloads follow redirects, raise on HTTP errors and stream to a temporary file that is only
    renamed into place after the download completed.
  - ffmpeg (resample, chunk split, WAV conversion) and `whisper-cli` run with `check=True`, write to
    temporary outputs that are renamed into place on success, and include the tool's stderr in the error.
  - The chunk split records the complete chunk set in `chunks/chunks.json` after all chunks are in
    place. Chunk directories without this manifest (interrupted splits, or caches from earlier versions)
    are split again once; existing per-chunk transcripts are kept.
  - Input files named like generated chunks (`chunk_000.mp3`) are cached as `episode_chunk_000.mp3` so
    the chunk step can no longer overwrite the source audio.
  - The Groq backend raises on non-429 HTTP errors and on rate-limit responses without a usable wait
    time instead of returning without a transcript. Fractional and millisecond rate-limit waits
    (for example `2.5s` or `480ms`) are now parsed correctly.
  - Transcript, DOTe, Podlove, WebVTT and plain-text outputs are written atomically.
- Offset multi-chunk transcripts by each chunk's `ffprobe` audio duration instead of the end of its last
  speech line, which shifted timestamps early for episodes longer than 2 hours.
- Cache directories poisoned by earlier versions (for example a saved redirect or error page instead of the
  episode) are not repaired automatically. Delete the episode's directory once and re-run; see
  "Troubleshooting" in the README.

### Development

- Add a GitHub Actions CI workflow that runs `just check` (pre-commit, mypy, pytest) on pushes and pull
  requests, with SHA-pinned actions, read-only permissions and no secrets.

### Documentation

- Document `VOXHELM_API_BASE` / `VOXHELM_API_KEY` configuration and `--backend voxhelm` usage.
- Add a README troubleshooting note for cache directories left behind by failed steps in older versions.

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
