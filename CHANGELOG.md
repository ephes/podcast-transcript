0.1.2 - 2024-11-30
==================

### Features

- Support for plain text format output.
- Retry logic for failed transcription requests when rate limited.

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