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