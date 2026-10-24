"""Run multiple test_skel_wam argument sets in one Python process.

Argument sets are separated by ``--next``. The underlying evaluator's runtime
cache therefore reuses a checkpoint model and VAE when adjacent commands point
to the same checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import test_skel_wam


def split_commands(tokens: list[str]) -> list[list[str]]:
    commands: list[list[str]] = [[]]
    for token in tokens:
        if token == "--next":
            if not commands[-1]:
                raise ValueError("Empty command before --next")
            commands.append([])
        else:
            commands[-1].append(token)
    if not commands[-1]:
        raise ValueError("No arguments after the final --next")
    return commands


def main() -> None:
    commands = split_commands(sys.argv[1:])
    original_argv = sys.argv
    try:
        for index, command in enumerate(commands, start=1):
            print(
                f"[inprocess-test] command {index}/{len(commands)}",
                flush=True,
            )
            sys.argv = ["main/test_skel_wam.py", *command]
            test_skel_wam.main()
    finally:
        sys.argv = original_argv
        test_skel_wam.clear_eval_runtime_cache()


if __name__ == "__main__":
    main()
