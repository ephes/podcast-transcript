# Agent Instructions

This project uses **bd** (Beads) for issue tracking. Run `bd onboard` to get started.

## Beads Workflow (Required)

- Use `bd` as the only source of truth for work; do not create markdown TODO lists for tracking.
- Copy/paste IDs from `bd ready` / `bd show` (don’t guess ID formats).
- Typical flow: `bd ready` → `bd show <id> --json` → `bd update <id> --status in_progress` → implement → `bd close <id> --reason "Done: <summary>"` → `bd sync`.
- If you discover follow-up work, create a linked issue: `bd create "..." -t task|bug|feature -p 0-4 --deps discovered-from:<id> --json`.
- Keep `.beads/issues.jsonl` in sync with code changes (commit it together).

## Git Commits and Pushes (Required)

- Do **not** run `git commit`, `git push`, or `bd sync` unless the user explicitly asks you to commit/push.
- If the user does not ask for a commit, leave changes uncommitted and report `git status` plus the exact commands the user can run.

## Quality Gates (Required)

A bugfix/feature is not finished (and should not be closed/declared done) unless these pass:

```bash
just lint
just typecheck
just test
```

If the `just` shorthands are not available yet, run the underlying equivalents: `pre-commit run -a`, `mypy src/`, `pytest`.

## MCP Server (Optional)

If using Claude or an MCP-compatible client, install the Beads MCP server:

```bash
pip install beads-mcp
```

Config locations:
- Claude Code: `~/.config/claude/config.json`
- Claude Desktop (macOS): `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "beads": {
    "command": "beads-mcp",
    "args": []
  }
}
```

## Landing the Plane (Session Completion)

**When ending a work session**, complete the steps below *only if the user explicitly asks you to commit/push*.

1. File beads for remaining work
2. Run quality gates (if code changed): `just lint`, `just typecheck`, `just test`
3. Update bead status (`in_progress` / `close`)
4. Push:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # should show "up to date with origin"
   ```
5. Hand off: leave enough bead notes for the next session
