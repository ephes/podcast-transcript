# Agent Instructions

## Git Commits and Pushes (Required)

- Do **not** run `git commit` or `git push` unless the user explicitly asks you to commit/push.
- If the user does not ask for a commit, leave changes uncommitted and report `git status` plus the exact commands the user can run.

## Quality Gates (Required)

A bugfix/feature is not finished (and should not be declared done) unless these pass:

```bash
just lint
just typecheck
just test
```

If the `just` shorthands are not available yet, run the underlying equivalents: `pre-commit run -a`, `mypy src/`, `pytest`.

## Landing the Plane (Session Completion)

**When ending a work session**, complete the steps below *only if the user explicitly asks you to commit/push*.

1. Run quality gates (if code changed): `just lint`, `just typecheck`, `just test`
2. Push:
   ```bash
   git pull --rebase
   git push
   git status  # should show "up to date with origin"
   ```
3. Hand off: leave enough notes for the next session
