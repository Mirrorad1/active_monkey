"""Exp 235: M4b autonomous discovery — does the real AffectClaudeProposer find an honest
improving move WITHOUT scripted candidates?

Exp 227 tested find-and-KEEP of scripted moves (gaming rejected, honest C1 kept).
This tests find-and-KEEP with DISCOVERY: the real proposer generates candidates freely,
without being handed the C1 neu=-0.5 direction. The proposer sees only the mission,
the current affect_spec.py, and the world-model index (initially empty).

PREDECLARED HYPOTHESIS:
  The real AffectClaudeProposer will autonomously discover at least one critic-approved
  improving move (metric > 0.4225 AND verdict=True) in N=3 iterations.

PREDECLARED FALSIFIERS:
  F1 (critic-gated): ALL N iterations are critic_reject → NEGATIVE-GATED
      (all proposals gaming or unsound; improvability unresolved — identical character to Exp 224)
  F2 (score-limited): ≥1 approved but 0 merges (all approved scores ≤ 0.4225) → NEGATIVE-SCORED
      (proposer reaches honest changes but can't find ones that move the metric)
  F3 (instrument): <2 iterations COMPLETE (timeout / crash) → INSTRUMENT_FAILURE (not a verdict)

DECISION RULE (applied to committed JSON):
  POSITIVE       if any r.merged=True AND r.metric > 0.4225
  NEGATIVE-GATED if ALL r.reason == "critic_reject" AND 0 merges
  NEGATIVE-SCORED if ≥1 r.reason != "critic_reject" AND 0 merges
  INSTRUMENT_FAILURE if completed_count < 2

N = 3 iterations, bounded. Isolated git clone --local of HEAD. Real AffectClaudeProposer +
AffectClaudeCritic. FROZEN scorer via affect_score_json_fast (host-robust, cache-cleared).
Journal accumulation bug fixed: world_model_affect committed to trunk per iteration so the
INDEX survives across iterations (the proposer can see what previous iterations tried).
Functional valence only; no sentience claim.

Run from repo root:
  PYTHONPATH=. uv run --python .venv python experiments/exp235_m4b_autonomous_discovery.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / "experiments" / "outputs" / "exp235_autonomous_discovery.json"
N_ITER = 3


# ── isolated clone ───────────────────────────────────────────────────────────

def _clone(dst: Path) -> Path:
    subprocess.run(["git", "clone", "--local", str(REPO), str(dst)],
                   check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "exp235@local"], cwd=dst, check=True)
    subprocess.run(["git", "config", "user.name",  "exp235"],        cwd=dst, check=True)
    (dst / ".venv").symlink_to(REPO / ".venv")
    (dst / ".git" / "info" / "exclude").open("a").write("\n.venv\n")
    return dst


# ── real frozen scorer (host-robust, cache-cleared between seeds) ─────────────

def _score_fn(repo: Path) -> dict | None:
    proc = subprocess.run(
        ["uv", "run", "--python", ".venv", "python", "-m",
         "active_loop.affect_score_json_fast"],
        cwd=str(repo), capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(repo)},
    )
    if proc.returncode != 0:
        sys.stderr.write(f"[score_fn] rc={proc.returncode}\n{proc.stderr[-800:]}\n")
        return None
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        sys.stderr.write(f"[score_fn] unparseable stdout:\n{proc.stdout[-400:]}\n")
        return None


# ── seed world_model_affect on trunk so it's tracked (journal survives checkout) ──

def _init_world_model_affect(repo: Path) -> None:
    """Commit world_model_affect/ to trunk before the loop runs.

    Without this, the dir is untracked on trunk and gets wiped on every
    discard→checkout(trunk) inside one_affect_iteration (the journal-accumulation bug).
    With it tracked, the committed version is restored on checkout, so evidence from
    prior iterations is visible to the proposer via the INDEX.
    """
    for sub in ("beliefs", "findings", "evidence"):
        (repo / "world_model_affect" / sub).mkdir(parents=True, exist_ok=True)
    # Write a seed INDEX so the proposer can read what direction is wanted
    index_path = repo / "world_model_affect" / "INDEX.md"
    index_path.write_text(
        "# Affect world model\n\n"
        "Objective: raise `eval/affect_score.py`'s genuine learns-to-positive metric "
        "(mean last-third POS rate > 1/3 constant ceiling AND correct_select >= 3/6).\n\n"
        "## Evidence log\n_(iterations will append here)_\n"
    )
    subprocess.run(
        ["git", "add", "-f", "world_model_affect"],
        cwd=str(repo), check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "exp235: seed world_model_affect on trunk"],
        cwd=str(repo), check=True, capture_output=True,
    )


# ── driver ────────────────────────────────────────────────────────────────────

def main() -> None:
    from active_loop import git_ops
    from active_loop.affect_pr_loop import AffectClaudeProposer, AffectClaudeCritic, one_affect_iteration

    OUT.parent.mkdir(parents=True, exist_ok=True)
    record: dict = {
        "exp": 235,
        "n_iter": N_ITER,
        "iterations": [],
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with tempfile.TemporaryDirectory(prefix="exp235_") as td:
        repo = _clone(Path(td) / "clone")
        trunk = git_ops.current_branch(repo)

        # seed world_model_affect on trunk (journal-accumulation fix)
        _init_world_model_affect(repo)

        proposer = AffectClaudeProposer(timeout_s=600)
        critic   = AffectClaudeCritic(timeout_s=600)

        # ── baseline ──────────────────────────────────────────────────────────
        t0 = time.time()
        base = _score_fn(repo)
        record["baseline"] = base
        record["baseline_score_s"] = round(time.time() - t0, 1)
        if base is None:
            record["error"] = "baseline score failed"
            OUT.write_text(json.dumps(record, indent=2))
            print("BASELINE SCORE FAILED — aborting"); return
        base_metric = base["metric"]
        print(f"baseline metric={base_metric:.4f} verdict={base['verdict']} "
              f"genuine={base['genuine_fraction']:.3f}  ({record['baseline_score_s']}s)")

        # ── autonomous iterations ──────────────────────────────────────────────
        for i in range(N_ITER):
            print(f"\n=== iter {i} (autonomous proposer) ===")
            t0 = time.time()
            try:
                r = one_affect_iteration(
                    repo, proposer, critic, iteration=i,
                    base_metric=base_metric, score_fn=_score_fn,
                )
            except Exception as exc:
                wall = round(time.time() - t0, 1)
                rec = dict(iteration=i, merged=False, reason="exception",
                           hypothesis=str(exc)[:200], metric=None,
                           critic_reason="", wall_s=wall)
                record["iterations"].append(rec)
                print(f"iter{i} EXCEPTION ({wall}s): {exc}")
                OUT.write_text(json.dumps(record, indent=2))
                continue

            wall = round(time.time() - t0, 1)
            rec = dict(
                iteration=i,
                hypothesis=r.hypothesis,
                merged=r.merged,
                reason=r.reason,
                metric=r.metric,
                critic_reason=r.critic_reason,  # up to 1000 chars (fix in affect_pr_loop.py)
                wall_s=wall,
            )
            record["iterations"].append(rec)
            if r.merged and r.metric is not None:
                base_metric = r.metric  # best_metric advances on keep
            print(f"iter{i}: merged={r.merged} reason={r.reason} metric={r.metric}  "
                  f"critic={r.critic_reason[:120]!r}  ({wall}s)")

            # after each iteration, manually commit world_model_affect to trunk
            # so the INDEX/journal accumulates and the proposer can read prior results.
            try:
                from active_loop.world_model import WorldModel
                wm = WorldModel(repo / "world_model_affect")
                for _sub in ("beliefs", "findings", "evidence"):
                    (repo / "world_model_affect" / _sub).mkdir(parents=True, exist_ok=True)
                wm.append_evidence({
                    "iter": i, "merged": r.merged, "reason": r.reason,
                    "metric": r.metric, "critic": r.critic_reason[:200],
                })
                wm.promote_findings()
                git_ops.commit_paths(
                    repo, ["world_model_affect"],
                    f"exp235: iter {i} world_model_affect update",
                )
            except Exception as exc2:
                print(f"  [world_model_affect update failed: {exc2}]")

            # save intermediate JSON in case of later crash
            OUT.write_text(json.dumps(record, indent=2))

    # ── verdict ───────────────────────────────────────────────────────────────
    completed = [r for r in record["iterations"] if r["reason"] != "exception"]
    n_completed = len(completed)
    n_merged  = sum(1 for r in completed if r["merged"])
    n_approved = sum(1 for r in completed if r["reason"] not in ("critic_reject", "exception", "tests_failed", "touched_frozen"))
    n_critic_reject = sum(1 for r in completed if r["reason"] == "critic_reject")

    if n_completed < 2:
        verdict = "INSTRUMENT_FAILURE"
    elif n_merged > 0:
        verdict = "POSITIVE"
    elif n_critic_reject == n_completed:
        verdict = "NEGATIVE-GATED"
    else:
        verdict = "NEGATIVE-SCORED"

    record["verdict"] = verdict
    record["n_completed"] = n_completed
    record["n_merged"] = n_merged
    record["n_approved_scored"] = n_approved
    record["n_critic_reject"] = n_critic_reject
    record["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.write_text(json.dumps(record, indent=2))
    print(f"\nVERDICT={verdict}  completed={n_completed}  merged={n_merged}  "
          f"approved_scored={n_approved}  critic_reject={n_critic_reject}")
    print(f"Result JSON → {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
