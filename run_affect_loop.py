"""Entry point for the affect-model PR-style autopilot.

    uv run --python .venv python run_affect_loop.py --iterations 3            # mock (no LLM)
    uv run --python .venv python run_affect_loop.py --iterations 3 --real     # claude proposer + critic
"""
from __future__ import annotations

import argparse
from pathlib import Path

from active_loop.affect_pr_loop import run_affect_pr_loop, AffectMockProposer
from active_loop.proposer import ClaudeCliProposer
from active_loop.critic import MockCritic, ClaudeCliCritic


def main() -> None:
    ap = argparse.ArgumentParser(description="Affect model PR autopilot")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--real", action="store_true",
                    help="use ClaudeCliProposer + ClaudeCliCritic instead of mocks")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent
    if args.real:
        proposer = ClaudeCliProposer(
            target_file="active_loop/affect_spec.py",
            objective="raise the mean last-third POS-rate (affect metric) while keeping "
                      "guardrails passing and not gaming the evaluator",
        )
        critic = ClaudeCliCritic()
    else:
        proposer = AffectMockProposer(seed=0)
        critic = MockCritic(approve=True)
    run_affect_pr_loop(repo, proposer, critic, iterations=args.iterations)
    print("Affect PR loop finished; see reports/affect_index.html and world_model/INDEX.md")


if __name__ == "__main__":
    main()
