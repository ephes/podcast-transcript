0.1.1 - 2024-11-19
==================

### Fixes

- Split the audio into chunks if it exceeds the duration limit (7200 seconds).
- Fixed the order in which the chunks are transcribed.
- Exit with status code 0 if the transcription was successful, 1 otherwise.

### Features

- Show generated transcript files in the output.

0.1.0 - 2024-11-17
==================

### Initial Release

It works for single track mp3 file urls.