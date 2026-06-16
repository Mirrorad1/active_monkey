"""Entry point for the affect-model PR-style autopilot.

    uv run --python .venv active-monkey-affect-loop --iterations 3            # mock (no LLM)
    uv run --python .venv active-monkey-affect-loop --iterations 3 --real     # claude proposer + critic
"""
from __future__ import annotations

import argparse

from active_loop.affect_pr_loop import (
    run_affect_pr_loop,
    AffectMockProposer,
    AffectClaudeProposer,
    AffectClaudeCritic,
)
from active_loop.critic import MockCritic
from active_loop.cli._paths import repo_root


def main() -> None:
    ap = argparse.ArgumentParser(description="Affect model PR autopilot")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--real", action="store_true",
                    help="use AffectClaudeProposer + AffectClaudeCritic instead of mocks")
    args = ap.parse_args()
    repo = repo_root()
    if args.real:
        proposer = AffectClaudeProposer()
        critic = AffectClaudeCritic()
    else:
        proposer = AffectMockProposer(seed=0)
        critic = MockCritic(approve=True)
    run_affect_pr_loop(repo, proposer, critic, iterations=args.iterations)
    print("Affect PR loop finished; see reports/affect_index.html and world_model_affect/INDEX.md")


if __name__ == "__main__":
    main()
