"""The NEVER-STOP outer research loop: propose -> guard -> score -> keep/revert -> grow."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from active_loop import git_ops
from active_loop.frozen_guard import is_frozen_touched
from active_loop.proposer import Proposer
from active_loop.world_model import WorldModel
from active_loop.report_html import render_report


@dataclass
class IterationResult:
    iteration: int
    hypothesis: str
    kept: bool
    reason: str
    metric: float | None
    score: dict | None


def _score(repo: Path) -> dict | None:
    proc = subprocess.run(
        ["uv", "run", "--python", ".venv", "python", "-m", "eval.score_json"],
        cwd=str(repo), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def one_iteration(repo, proposer, iteration, base_metric=None):
    repo = Path(repo)
    base_sha = git_ops.current_sha(repo)
    if base_metric is None:
        base_score = _score(repo)
        base_metric = base_score["metric"] if base_score else float("inf")

    hypothesis = proposer.propose(repo)

    if is_frozen_touched(git_ops.changed_files(repo), repo):
        git_ops.reset_hard(repo)
        return IterationResult(iteration, hypothesis, False, "touched_frozen", None, None)

    score = _score(repo)
    if score is None:
        git_ops.reset_hard(repo)
        return IterationResult(iteration, hypothesis, False, "broken", None, None)

    improved = score["metric"] < base_metric
    if improved and score["verdict"]:
        git_ops.commit_all(repo, f"auto: iter {iteration} keep ({hypothesis}) metric={score['metric']:.4f}")
        return IterationResult(iteration, hypothesis, True, "improved", score["metric"], score)

    git_ops.reset_hard(repo)
    reason = "no_improvement" if not improved else "guardrail_fail"
    return IterationResult(iteration, hypothesis, False, reason, score["metric"], score)


def run_loop(repo, proposer, iterations: int = 0, error_budget: int = 5) -> None:
    repo = Path(repo)
    wm = WorldModel(repo / "world_model")
    history: list[float] = []
    consecutive_errors = 0

    base_score = _score(repo) or {"metric": float("inf"), "success_rate": 0.0,
                                  "ask_rate": 0.0, "guardrails": {}, "verdict": False}
    best_metric = base_score["metric"]
    current_score = base_score

    i = 0
    while iterations == 0 or i < iterations:
        result = one_iteration(repo, proposer, i, base_metric=best_metric)

        wm.append_evidence({
            "iter": i, "hypothesis": result.hypothesis, "kept": result.kept,
            "reason": result.reason, "metric": result.metric,
        })
        if result.kept and result.metric is not None:
            best_metric = result.metric
            current_score = result.score
            history.append(result.metric)
            wm.record_belief(
                f"iter-{i}-{result.reason}",
                f"Change '{result.hypothesis}' lowered free energy to {result.metric:.4f}.",
                supported=True,
            )
            consecutive_errors = 0
        elif result.reason in ("broken", "touched_frozen"):
            consecutive_errors += 1
        else:
            consecutive_errors = 0

        wm.promote_findings()
        wm.rebuild_index()
        render_report(repo / "reports" / "index.html", score=current_score,
                      world_model=wm, history=history)
        (repo / "REPORT.md").write_text(
            f"# active-loop report\n\nlast iter: {i}\nbest metric: {best_metric:.4f}\n"
            f"last result: {result.reason} (kept={result.kept})\n"
        )
        git_ops.commit_paths(repo, ["world_model", "reports", "REPORT.md"],
                             f"loop: iter {i} artifacts (best={best_metric:.4f})")

        if consecutive_errors >= error_budget:
            (repo / "NEEDS_HUMAN.md").write_text(
                f"# Needs human\n\nHalted at iter {i} after {consecutive_errors} consecutive "
                f"broken/frozen-touching proposals.\nLast good commit: {git_ops.current_sha(repo)}\n"
                "Suggested question: is the mutable surface or the proposer prompt mis-scoped?\n"
            )
            git_ops.commit_paths(repo, ["NEEDS_HUMAN.md", "world_model", "reports", "REPORT.md"],
                                 f"loop: halt at iter {i}")
            return
        i += 1
