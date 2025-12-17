# CLAUDE.md

**Note**: This project uses [bd (beads)](https://github.com/steveyegge/beads)
for issue tracking. Use `bd` commands instead of markdown TODOs.
See AGENTS.md for workflow details.

**IMPORTANT**: Do not run `git commit`, `git push`, or `bd sync` unless the user explicitly asks you to commit/push.

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

## Beads (bd) Workflow

Beads (`bd`) is the issue tracker and source of truth for work in this repo. Don’t guess bead IDs—copy/paste them from `bd ready` / `bd show`.

- Find work: `bd ready`
- Start work: `bd update <id> --status in_progress`
- Get context: `bd show <id> --json`, `bd dep tree <id>`
- Finish: `bd close <id> --reason "Done: <summary>"`
- Keep in sync: `bd sync`

### New Machine

After `git clone`:

```bash
bd onboard
```

If `bd onboard` is not available, use:

```bash
bd init
bd hooks install
```

If you previously used `bd init --stealth`, check your global gitignore for `**/.beads/` and remove it (or use `git add -f`) if you intend to commit the Beads ledger.

See `AGENTS.md` for the canonical Beads workflow and ledger conventions.
