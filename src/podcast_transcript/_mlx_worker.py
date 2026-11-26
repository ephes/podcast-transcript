"""Run one MLX inference in an owned child process.

``MLX.transcribe`` runs this file (not the package) with a private parameter
file and a result path. It imports only the standard library and
``mlx_whisper``, so the CLI's settings and their side effects are not loaded.
Running inference here gives the CLI a process it can time out, kill and reap;
the parent admits and validates the result before publishing.
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
    remaining = params["result_max_bytes"]
    with open(argv[1], "xb") as file:
        try:
            for chunk in json.JSONEncoder(allow_nan=False).iterencode(result):
                block = chunk.encode("utf-8")
                if len(block) > remaining:
                    print("MLX transcript output exceeded byte limit.", file=sys.stderr)
                    return 4
                file.write(block)
                remaining -= len(block)
        except (TypeError, ValueError, RecursionError):
            print("Invalid MLX transcript JSON.", file=sys.stderr)
            return 5
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
