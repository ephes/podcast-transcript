set quiet := true

default:
  @just --list

# Quality gates
lint *args:
  UV_PYTHON=python"$(tr -d '\n' < .python-version)" uv run pre-commit run -a {{args}}

typecheck *args:
  UV_PYTHON=python"$(tr -d '\n' < .python-version)" uv run mypy {{args}} src/

test *args:
  UV_PYTHON=python"$(tr -d '\n' < .python-version)" uv run pytest {{args}}

# Beads helper: `just bead` -> `bd ready`, `just bead <cmd> ...` -> `bd <cmd> ...`
bead *args:
  @if [ -z "{{args}}" ]; then bd ready; else bd {{args}}; fi
