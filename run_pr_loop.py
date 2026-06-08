"""Entry point for the language-model PR-style autopilot.

    uv run --python .venv python run_pr_loop.py --iterations 3            # mock (no LLM)
    uv run --python .venv python run_pr_loop.py --iterations 3 --real     # claude proposer + critic
"""
from __future__ import annotations

import argparse
from pathlib import Path

from active_loop.pr_loop import run_pr_loop
from active_loop.proposer import LangMockProposer, ClaudeCliProposer
from active_loop.critic import MockCritic, ClaudeCliCritic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=0)
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent
    if args.real:
        proposer = ClaudeCliProposer(
            target_file="active_loop/lang_model_spec.py",
            objective="lower the held-out bits/char (free energy) of this character "
                      "language model without gaming the metric or touching frozen files",
        )
        critic = ClaudeCliCritic()
    else:
        proposer, critic = LangMockProposer(seed=0), MockCritic(approve=True)
    run_pr_loop(repo, proposer, critic, iterations=args.iterations)
    print("PR loop finished; see reports/index.html and world_model/INDEX.md")


if __name__ == "__main__":
    main()
