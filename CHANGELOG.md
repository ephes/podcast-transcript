0.1.7 - Unreleased
==================

### Features

- Add a `voxhelm` backend for remote transcription via Voxhelm's OpenAI-compatible API.

### Fixes

- Validate every combined chunk duration before reading chunks or staging output, including
  single-chunk copies. Reject booleans, negative/nonfinite and overflowing seconds; preserve zero.
- Sum decimal chunk durations exactly and round cumulative offsets to nearest milliseconds,
  ties to even, matching raw cue conversion without per-chunk rounding drift.
- Hash local model artifacts through one inspected nonblocking, no-follow regular-file descriptor,
  with independent 64 GiB and one-hour per-artifact budgets. Refuse model symlinks and special files
  before inference; count growth and preserve prior caches on admission failure.
- Stage replacement backend/model/language/prompt transcripts and all exports before changing a
  completed generation. Failed replacement leaves the old generation current; successful promotion
  retains a durable `.previous-transcripts` recovery copy and publishes new metadata last.
  Interrupted promotion keeps that recovery copy and rebuilds without reusing mixed results.
  Recovery and promotion copy bytes exactly; an unreadable published generation is replaced in
  place instead of blocking every run, and staging left by a killed run is removed under the lock.

- Run MLX inference in an owned worker process bounded by `TRANSCRIPT_LOCAL_TRANSCRIPTION_TIMEOUT`;
  kill and reap stalled workers, release the episode lock, keep previous transcripts, admit the
  result with the bounded native reader and keep prompts out of the process list.

- Stage each newly inferred local-model chunk and publish it only after rechecking the model
  identity recorded for the cache, so chunks from a replaced model are never reused; interrupted
  runs with an unchanged model still resume and prepared audio is retained.

- Admit native whisper-cli JSON through a bounded (64 MiB) nonblocking regular-file reader before
  parsing; normalize excessive nesting, drop only the refused intermediate and keep prior transcripts.
- Request identity encoding for Groq/Voxhelm responses and refuse compressed bodies before decoding,
  so decompression cannot allocate beyond the 32 MiB response limit.

- Admit local sources through a nonblocking regular-file descriptor with the source byte/time budgets,
  preserving completed caches on admission failure.
- Sync staged artifact data and containing directories before completion markers, including durable
  marker retirement and explicit symlink handling; document filesystem power-loss guarantees.

- Drain native command stderr to a bounded memory tail with explicit truncation, avoiding disk spools.
- Validate cached and newly converted WAV samples and shape before native inference; rebuild corrupt
  intermediates while preserving source audio and previous transcripts on conversion failure.

- Bind local model transcript caches to model file/directory bytes; retain prepared audio on changes.
- Create new cache directories owner-only and stage outputs privately with new files owner-only,
  retaining existing destination permissions. Omit prompt-bearing cache metadata from progress logs.

- Bound ffprobe stdout/stderr capture during execution; kill and reap overflowing probes.
- Validate native whisper-cpp timestamp syntax and schema before publishing converted transcripts.

- Bound Groq rate-limit waits to a finite cumulative request-sized budget and use monotonic time.
- Keep remote status, rate-limit, transport and timeout diagnostics free of remote bodies and messages.

- Reject nonpositive `--processors` before backend/audio work; enforce positive integers in
  the public whisper-cpp constructor while preserving the default of four and exact forwarding.
- Request identity HTTP encoding for source audio and reject compressed responses before decoding,
  preserving prior targets and cleaning partial downloads on rejection.
- Apply the configured remote request budget to elapsed upload, computation and response transfer
  time; cancel trickle responses, close transports and preserve previous transcripts on expiry.
- Stage changed local-source audio through copying, resampling and chunk validation before
  retiring the previous generation. Failed preparation keeps completed files recoverable with
  no current-source marker; retry never returns stale transcripts for changed bytes.

- Validate Whisper result envelopes, segment text and finite, nonnegative ordered times before
  caching backend results or reusing raw chunks. Accept silence and zero-length cues; preserve
  millisecond rounding and report cached filenames and bounded segment positions without transcript text.
- Validate cached DOTe timecodes with one strict parser across combination and all exports,
  including single-chunk copies. Reject component overflow, suffix junk and reversed cues while
  preserving multi-day hours and previous outputs; document invalid-cache recovery.

- Validate Voxhelm HTTP(S) API prefixes before uploads and use the same prefix for requests and
  cache identity; reject credentials, query strings and fragments while preserving custom paths.
- Stream Groq/Voxhelm responses with a 32 MiB decoded-byte limit and omit remote bodies
  from failure diagnostics; close responses and preserve previous transcripts on failure.
- Omit source URL credentials, queries and fragments from progress and request-error diagnostics,
  including redirected failures, while retaining original request/cache identity.

- Cover successful CLI dispatch for all four backend factories, including the default local backend,
  processor/settings forwarding, output paths and pipeline error exits, using synthetic fixtures.
- Align README installation and usage examples with the installed `transcribe` executable.
  Help now describes local audio paths and HTTP(S) input; document credentials per backend and
  distinguish persistent tool installation from one-shot `uvx` execution.

- Derive stream-copy chunk duration from both size and duration limits, validate every new chunk
  before publication, and rebuild cached oversized chunks. Fail clearly rather than upload a
  chunk above 25 MiB or 7,200 seconds.
- Bound ffmpeg commands to one hour (`TRANSCRIPT_MEDIA_TIMEOUT`) and whisper-cli to six hours
  (`TRANSCRIPT_LOCAL_TRANSCRIPTION_TIMEOUT`), configurable in the environment or `.env`.
  Kill owned command trees and reap the direct process on timeout/interruption; retain previous
  final files and remove partial outputs. Bound stderr diagnostics and replace invalid bytes.

- Include MLX `word_timestamps` in transcript cache identity; older MLX caches rebuild once,
  retaining prepared audio. Serialize the full pipeline with a per-episode OS file lock;
  competing runs fail with a busy/retry message and unrelated episodes remain concurrent.

- Bound source HTTP(S) downloads by decoded bytes (`TRANSCRIPT_SOURCE_MAX_BYTES`, default 2 GiB)
  and monotonic elapsed time (`TRANSCRIPT_SOURCE_TOTAL_TIMEOUT`, default one hour), in addition
  to per-operation timeouts. Reject oversized headers and streams, close failed responses and
  clean up partial files while preserving existing targets; both budgets support environment
  and `.env` overrides.

- Limit ffprobe to 60 seconds, reap timed-out processes and report bounded diagnostics. Reject
  malformed, negative and nonfinite durations before invoking a backend or replacing combined
  outputs; validate new split chunks before publishing preparation completion markers. Replace
  invalid stderr bytes in bounded diagnostics.

- Include Voxhelm's normalized service identity in transcript cache metadata, excluding URL
  credentials, query strings, fragments and API keys. Root and `/v1` aliases reuse transcripts;
  switching services invalidates them. Existing Voxhelm transcripts rebuild once; audio is kept.

- Make short-audio chunk symlinks relative to their parent, supporting relative cache roots
  (resolved from the command working directory), including paths with spaces.

- Validate cached chunk manifests as ordered, consecutive generated MP3 basenames, rejecting
  duplicates, external paths and unexpected symlinks. Invalid manifests rebuild chunks and
  transcripts while preserving downloaded and resampled audio; interrupted invalid-cache cleanup
  is retried before regenerating chunks.

- Update the locked certifi CA bundle to 2026.10.19 for the eMudhra/emSign CA compromise
  ([GHSA-6q48-6c7w-9gw2](https://github.com/certifi/python-certifi/security/advisories/GHSA-6q48-6c7w-9gw2)).
  Lock-only change: checkout environments get it with `uv sync --frozen`. `uv tool install` does
  not read this lock; upgrade an installed tool with `uv tool upgrade podcast-transcript` and
  confirm its certifi is 2026.10.19 or later.

- Escape literal WebVTT cue text and normalize line endings and blank lines so transcript markup
  and literal arrows remain readable without terminating cues. Delete existing cached `.webvtt`
  files and rerun with the same settings to regenerate them without retranscribing.

- Derive safe download filenames from HTTP(S) URL paths, excluding signed queries and fragments
  while retaining full URL hashing and generated-chunk collision protection. URL names needing
  sanitization change their readable cache-directory stem and rebuild once; old directories stay
  untouched and may be removed after copying any needed outputs.

- Invalidate local audio and transcript caches when source bytes change at the same path; record
  the content fingerprint only after successful preparation. Local sources must remain available
  for verification; missing originals raise a clear error asking users to restore the file.

- The Groq and Voxhelm backends no longer wait forever for a server that never answers. Requests use a
  10 s connect, 300 s write and 10 s pool timeout, and a read timeout set by the new
  `TRANSCRIPT_HTTP_READ_TIMEOUT` setting (seconds, default 1800). A timeout raises a `RuntimeError` naming
  the backend. Groq rate-limit (429) retries stop after 10 attempts with a `RuntimeError`
  suggesting a later retry.
- Make every pipeline step fail closed so a failed step is never cached as a success:
  - Downloads follow redirects, raise on HTTP errors and stream to a temporary file that is only
    renamed into place after the download completed.
  - ffmpeg (resample, chunk split, WAV conversion) and `whisper-cli` check exit status, write to
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
- Keep the declared Python 3.10 support tested: the chunk-splitting tests no longer mock `Path.stat`
  (which broke `Path.exists()` on Python 3.10/3.11), and CI additionally runs the test suite on
  Python 3.10 via `just test-oldest-python`.

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
