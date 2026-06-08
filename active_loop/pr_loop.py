"""PR-style autopilot: branch -> propose -> frozen-guard -> tests -> critic -> score -> merge/discard."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from active_loop import git_ops
from active_loop.frozen_guard import is_frozen_touched
from active_loop.world_model import WorldModel
from active_loop.report_html import render_report

TRUNK = "master"


@dataclass
class PRResult:
    iteration: int
    hypothesis: str
    merged: bool
    reason: str
    bits_per_char: float | None
    critic_reason: str


def _score(repo: Path) -> dict | None:
    proc = subprocess.run(
        ["uv", "run", "--python", ".venv", "python", "-m", "eval.lang_score_json"],
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
         "tests/test_lang_model_spec.py", "tests/test_alphabet.py"],
        cwd=str(repo), capture_output=True, text=True,
    )
    return proc.returncode == 0


def one_pr_iteration(repo, proposer, critic, iteration, base_bits=None) -> PRResult:
    repo = Path(repo)
    if base_bits is None:
        bs = _score(repo)
        base_bits = bs["bits_per_char"] if bs else float("inf")

    branch = f"proposal/iter-{iteration}"
    git_ops.create_branch(repo, branch)
    hyp = proposer.propose(repo)

    def _discard(reason, bits=None, crit=""):
        git_ops.reset_hard(repo)
        git_ops.checkout(repo, TRUNK)
        git_ops.delete_branch(repo, branch)
        return PRResult(iteration, hyp, False, reason, bits, crit)

    if is_frozen_touched(git_ops.changed_files(repo), repo):
        return _discard("touched_frozen")

    git_ops.commit_all(repo, f"proposal iter {iteration}: {hyp}")

    if not _tests_pass(repo):
        return _discard("tests_failed")

    diff = subprocess.run(["git", "diff", f"{TRUNK}...{branch}"], cwd=str(repo),
                          capture_output=True, text=True).stdout
    verdict = critic.review(diff, repo)
    if not verdict.approved:
        return _discard("critic_reject", crit=verdict.reason)

    score = _score(repo)
    if score is None:
        return _discard("broken", crit=verdict.reason)
    improved = score["bits_per_char"] < base_bits
    if improved and score["verdict"]:
        git_ops.checkout(repo, TRUNK)
        git_ops.merge_no_ff(repo, branch, f"merge iter {iteration} ({hyp}) bits={score['bits_per_char']:.4f}")
        git_ops.delete_branch(repo, branch)
        return PRResult(iteration, hyp, True, "improved", score["bits_per_char"], verdict.reason)
    return _discard("no_improvement" if not improved else "guardrail_fail",
                    bits=score["bits_per_char"], crit=verdict.reason)


def run_pr_loop(repo, proposer, critic, iterations: int = 0, error_budget: int = 5) -> None:
    repo = Path(repo)
    wm = WorldModel(repo / "world_model")
    history: list[float] = []
    consecutive_errors = 0

    bs = _score(repo)
    best_bits = bs["bits_per_char"] if bs else float("inf")
    current = bs or {"bits_per_char": best_bits, "baseline_bits": 0.0, "guardrails": {}, "verdict": False}

    i = 0
    while iterations == 0 or i < iterations:
        r = one_pr_iteration(repo, proposer, critic, i, base_bits=best_bits)
        wm.append_evidence({"iter": i, "hypothesis": r.hypothesis, "merged": r.merged,
                            "reason": r.reason, "bits_per_char": r.bits_per_char,
                            "critic": r.critic_reason})
        if r.merged and r.bits_per_char is not None:
            best_bits = r.bits_per_char
            history.append(r.bits_per_char)
            current = _score(repo) or current
            wm.record_belief(f"iter-{i}-merged",
                             f"'{r.hypothesis}' lowered held-out cost to {r.bits_per_char:.4f} bits/char.",
                             supported=True)
            consecutive_errors = 0
        elif r.reason in ("broken", "tests_failed", "touched_frozen", "critic_reject"):
            consecutive_errors += 1
        else:
            consecutive_errors = 0

        wm.promote_findings()
        score_view = {"metric": current.get("bits_per_char", best_bits),
                      "success_rate": 0.0, "ask_rate": 0.0,
                      "guardrails": current.get("guardrails", {}), "verdict": current.get("verdict", False)}
        render_report(repo / "reports" / "index.html", score=score_view, world_model=wm, history=history)
        (repo / "REPORT.md").write_text(
            f"# active-loop language report\n\niter {i}\nbest bits/char: {best_bits:.4f}\n"
            f"last: {r.reason} (merged={r.merged})\n")
        git_ops.commit_paths(repo, ["world_model", "reports", "REPORT.md"], f"loop: iter {i} artifacts")

        if consecutive_errors >= error_budget:
            (repo / "NEEDS_HUMAN.md").write_text(
                f"# Needs human\n\nHalted at iter {i}; {consecutive_errors} consecutive failures.\n")
            git_ops.commit_paths(repo, ["NEEDS_HUMAN.md"], f"loop: halt at iter {i}")
            return
        i += 1
