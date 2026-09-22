set quiet := true

default:
  @just --list

# Quality gates
check:
  just lint
  just typecheck
  just test

lint *args:
  UV_PYTHON=python"$(tr -d '\n' < .python-version)" uv run pre-commit run -a {{args}}

typecheck *args:
  UV_PYTHON=python"$(tr -d '\n' < .python-version)" uv run mypy {{args}} src/

test *args:
  UV_PYTHON=python"$(tr -d '\n' < .python-version)" uv run pytest {{args}}
