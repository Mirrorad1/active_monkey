"""Entry point for the active-loop outer research loop.

Examples:
    uv run --python .venv active-monkey-loop --iterations 3            # mock proposer
    uv run --python .venv active-monkey-loop --iterations 3 --real     # claude CLI proposer
    uv run --python .venv active-monkey-loop                           # NEVER-STOP (real)
"""
from __future__ import annotations

import argparse

from active_loop.loop import run_loop
from active_loop.proposer import MockProposer, ClaudeCliProposer
from active_loop.cli._paths import repo_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=0, help="0 = never stop")
    ap.add_argument("--real", action="store_true", help="use the claude CLI proposer")
    args = ap.parse_args()

    repo = repo_root()
    proposer = ClaudeCliProposer() if args.real else MockProposer(seed=0)
    run_loop(repo, proposer=proposer, iterations=args.iterations)
    print("loop finished; see reports/index.html and world_model/INDEX.md")


if __name__ == "__main__":
    main()
