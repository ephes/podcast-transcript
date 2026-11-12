"""Run one MLX inference in an owned child process.

``MLX.transcribe`` starts this module with a private parameter file and a
result path. Running inference here gives the CLI a process it can time out,
kill and reap; the parent admits and validates the result before publishing.
"""

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: _mlx_worker PARAMS RESULT", file=sys.stderr)
        return 2
    params = json.loads(Path(argv[0]).read_text())
    try:
        import mlx_whisper  # type: ignore
    except ModuleNotFoundError:
        print("MLX backend dependencies are not installed.", file=sys.stderr)
        return 3
    result = mlx_whisper.transcribe(
        params["audio"],
        path_or_hf_repo=params["model"],
        word_timestamps=params["word_timestamps"],
        initial_prompt=params["prompt"],
        language=params["language"],
    )
    with open(argv[1], "x", encoding="utf-8") as file:
        json.dump(result, file)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
