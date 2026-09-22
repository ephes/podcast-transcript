# CLAUDE.md

**IMPORTANT**: Do not run `git commit` or `git push` unless the user explicitly asks you to commit/push.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`podcast-transcript` is a Python CLI tool that downloads/processes audio and generates transcripts (local backends like whisper-cpp / mlx-whisper, or API backends like Groq) and exports multiple formats.

## Project Structure

- `src/podcast_transcript/` – library + CLI entrypoint (`transcribe` script)
- `tests/` – pytest suite
- `pyproject.toml` / `uv.lock` – packaging + dependencies

## Common Commands

```bash
uv venv
uv pip install -e .
just lint
just typecheck
just test
pytest
mypy src/
pre-commit install
pre-commit run -a
```

## Quality Gates (Required)

Do not declare a bugfix/feature finished unless these pass:

```bash
just lint
just typecheck
just test
```

If the `just` shorthands are not available yet, run the underlying equivalents: `pre-commit run -a`, `mypy src/`, `pytest`.
