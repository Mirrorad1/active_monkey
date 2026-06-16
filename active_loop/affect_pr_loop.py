"""PR-style autopilot for the affect model: branch -> propose -> frozen-guard -> tests -> critic -> score -> merge/discard.

Higher metric (mean last-third POS) is better — the opposite direction from the language loop.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from active_loop import git_ops
from active_loop.frozen_guard import is_frozen_touched
from active_loop.world_model import WorldModel
from active_loop.report_html import render_report


@dataclass
class AffectPRResult:
    iteration: int
    hypothesis: str
    merged: bool
    reason: str
    metric: float | None
    critic_reason: str


def _score(repo: Path, score_fn: Callable | None = None) -> dict | None:
    """Return score dict or None on failure.

    If score_fn is provided (for tests), call score_fn(repo) and return the result.
    Otherwise subprocess into eval.affect_score_json.
    """
    if score_fn is not None:
        return score_fn(repo)
    proc = subprocess.run(
        ["uv", "run", "--python", ".venv", "python", "-m", "eval.affect_score_json"],
        cwd=str(repo), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def _tests_pass(repo: Path) -> bool:
    proc = subprocess.run(
        ["uv", "run", "--python", ".venv", "pytest", "-q",
         "tests/test_affect_agent.py", "tests/test_affect_score.py"],
        cwd=str(repo), capture_output=True, text=True,
    )
    return proc.returncode == 0


def one_affect_iteration(
    repo,
    proposer,
    critic,
    iteration: int,
    base_metric: float | None = None,
    score_fn: Callable | None = None,
) -> AffectPRResult:
    repo = Path(repo)
    trunk = git_ops.current_branch(repo)
    if base_metric is None:
        bs = _score(repo, score_fn)
        base_metric = bs["metric"] if bs else float("-inf")

    branch = f"proposal/affect-iter-{iteration}"
    git_ops.create_branch(repo, branch)
    hyp = proposer.propose(repo)

    def _discard(reason: str, metric=None, crit: str = "") -> AffectPRResult:
        git_ops.reset_hard(repo)
        git_ops.checkout(repo, trunk)
        git_ops.delete_branch(repo, branch)
        return AffectPRResult(iteration, hyp, False, reason, metric, crit)

    if is_frozen_touched(git_ops.changed_files(repo), repo):
        return _discard("touched_frozen")

    git_ops.commit_all(repo, f"affect proposal iter {iteration}: {hyp}")

    if not _tests_pass(repo):
        return _discard("tests_failed")

    diff = subprocess.run(
        ["git", "diff", f"{trunk}...{branch}"],
        cwd=str(repo), capture_output=True, text=True,
    ).stdout
    verdict = critic.review(diff, repo)
    if not verdict.approved:
        return _discard("critic_reject", crit=verdict.reason)

    score = _score(repo, score_fn)
    if score is None:
        return _discard("broken", crit=verdict.reason)

    improved = score["metric"] > base_metric
    if improved and score["verdict"]:
        git_ops.checkout(repo, trunk)
        git_ops.merge_no_ff(
            repo, branch,
            f"merge affect iter {iteration} ({hyp}) metric={score['metric']:.4f}",
        )
        git_ops.delete_branch(repo, branch)
        return AffectPRResult(iteration, hyp, True, "improved", score["metric"], verdict.reason)

    return _discard(
        "no_improvement" if not improved else "guardrail_fail",
        metric=score["metric"],
        crit=verdict.reason,
    )


def run_affect_pr_loop(
    repo,
    proposer,
    critic,
    iterations: int = 0,
    error_budget: int = 5,
    score_fn: Callable | None = None,
) -> None:
    repo = Path(repo)
    wm = WorldModel(repo / "world_model")
    history: list[float] = []
    consecutive_errors = 0

    bs = _score(repo, score_fn)
    best_metric = bs["metric"] if bs else float("-inf")
    current = bs or {"metric": best_metric, "verdict": False, "guardrails": {}, "ask_rate": 0.0}

    i = 0
    while iterations == 0 or i < iterations:
        r = one_affect_iteration(
            repo, proposer, critic, i,
            base_metric=best_metric,
            score_fn=score_fn,
        )
        wm.append_evidence({
            "iter": i,
            "hypothesis": r.hypothesis,
            "merged": r.merged,
            "reason": r.reason,
            "metric": r.metric,
            "critic": r.critic_reason,
        })
        if r.merged and r.metric is not None:
            best_metric = r.metric
            history.append(r.metric)
            current = _score(repo, score_fn) or current
            wm.record_belief(
                f"iter-{i}-merged",
                f"'{r.hypothesis}' raised affect metric to {r.metric:.4f}.",
                supported=True,
            )
            consecutive_errors = 0
        elif r.reason in ("broken", "tests_failed", "touched_frozen", "critic_reject"):
            consecutive_errors += 1
        else:
            consecutive_errors = 0

        wm.promote_findings()
        score_view = {
            "metric": current.get("metric", best_metric),
            "success_rate": 0.0,
            "ask_rate": current.get("ask_rate", 0.0),
            "guardrails": current.get("guardrails", {}),
            "verdict": current.get("verdict", False),
        }
        render_report(
            repo / "reports" / "affect_index.html",
            score=score_view,
            world_model=wm,
            history=history,
        )
        (repo / "REPORT.md").write_text(
            f"# active-loop affect report\n\niter {i}\n"
            f"best metric: {best_metric:.4f}\n"
            f"last: {r.reason} (merged={r.merged})\n"
        )
        git_ops.commit_paths(
            repo,
            ["world_model", "reports", "REPORT.md"],
            f"affect loop: iter {i} artifacts",
        )

        if consecutive_errors >= error_budget:
            (repo / "NEEDS_HUMAN.md").write_text(
                f"# Needs human\n\nHalted at iter {i}; {consecutive_errors} consecutive failures.\n"
            )
            git_ops.commit_paths(repo, ["NEEDS_HUMAN.md"], f"affect loop: halt at iter {i}")
            return
        i += 1


# ── Mock proposer for harness validation ────────────────────────────────────

_JITTER_PAT = re.compile(r"rng\.uniform\(0\.0,\s*([\d.]+)")


class AffectMockProposer:
    """Trivially mutates affect_spec.py's jitter magnitude for fast harness testing.

    Nudges the first rng.uniform(0.0, <magnitude>) jitter parameter by a small delta
    drawn from a seed-indexed list.  Validates with compile() before writing.
    """

    _DELTAS = [0.01, -0.01, 0.02, -0.02, 0.03]

    def __init__(self, seed: int = 0):
        self.seed = seed

    def propose(self, repo) -> str:
        path = Path(repo) / "active_loop" / "affect_spec.py"
        src = path.read_text()

        m = _JITTER_PAT.search(src)
        if m:
            old_val = float(m.group(1))
            delta = self._DELTAS[self.seed % len(self._DELTAS)]
            new_val = round(old_val + delta, 4)
            new_src = src[: m.start(1)] + str(new_val) + src[m.end(1):]
            hypothesis = f"nudge affect_spec jitter -> {new_val} (seed {self.seed})"
        else:
            # Fallback: nudge first float literal
            fm = re.search(r"(\d+\.\d+)", src)
            if fm is None:
                raise ValueError("no float literal found in affect_spec.py to nudge")
            old_val = float(fm.group(1))
            delta = self._DELTAS[self.seed % len(self._DELTAS)]
            new_val = round(old_val + delta, 4)
            new_src = src[: fm.start()] + repr(new_val) + src[fm.end():]
            hypothesis = f"nudge affect_spec first float -> {new_val} (seed {self.seed})"

        compile(new_src, "active_loop/affect_spec.py", "exec")
        path.write_text(new_src)
        return hypothesis
