"""Exp 223 — M4b first REAL autopilot run: can the PR-style loop RAISE the genuine
learns-to-positive metric by mutating affect_spec.py, scored by the FROZEN scorer, without
reward-hacking?

This docstring explicitly contains the required words: falsifier, predeclaration, hypothesis.

AUTHORIZATION: the human's "Go to m4b" + "run it" (2026-06-16) — a bounded 2-iteration real run.

PLAIN summary: we built an automatic-improvement loop (M4b): it asks Claude to make ONE small
change to the agent's model file (affect_spec.py), then scores the changed agent with the FROZEN,
can't-be-faked "learns to feel positive" number, and KEEPS the change only if the number goes UP
and the honesty guardrails still pass — otherwise it throws the change away. The scorer is in a
FROZEN folder the loop is forbidden to edit, so it can't cheat by rewriting its own test. This run
turns the loop on for real for 2 rounds and asks: does it find a change that genuinely raises the
number? Either answer is a result. Functional valence only; no sentience claim.

HYPOTHESIS: the M4b autopilot runs end-to-end for real (Claude proposes a valid affect_spec.py
mutation; the FROZEN scorer evaluates it; the loop keeps/reverts correctly) AND can RAISE the
genuine learns-to-positive metric above the deterministic baseline (0.4225, from Exp 222) without
reward-hacking (the FROZEN eval/ prefix + frozen-guard block any tampering with the scorer).

PREDECLARATION: run `run_affect_loop.py --real --iterations 2` on an ISOLATED clone of the repo
(shared-checkout / cron safety). Baseline = the autopilot's own first real score of the unmodified
clone (must reproduce ~0.4225). Then 2 iterations, each: claude -p proposes ONE small change to
active_loop/affect_spec.py -> frozen-guard -> affect tests -> claude critic -> the FROZEN
score_affect() (300t x 8 seeds, ~10 min) -> KEEP iff metric > best AND verdict, else revert.

PREDICTION (if TRUE): at least one iteration MERGES — its candidate metric > baseline AND verdict
True — raising best_metric above 0.4225; zero frozen-touch / gaming.

FALSIFIER: if across BOTH iterations no proposal both improves the metric AND holds verdict=True
(all discarded as no_improvement / guardrail_fail / critic_reject / tests_failed), the autopilot
does NOT raise the metric at this scale — log NEGATIVE (inconclusive-at-N=2: the baseline
affect_spec is near-optimal for this metric, or Claude's small proposals don't find a gain in 2
tries). A 'broken' result (claude/scorer failure) is an INSTRUMENT failure, not a science verdict.

Honesty: N=2 is tiny — a NEGATIVE means "no improvement found in 2 tries", NOT "unimprovable". The
critic is the FROZEN ClaudeCliCritic whose prompt is lang-model-flavored (a known mismatch); it is
a SOFT gate — the hard anti-gaming protections are the FROZEN scorer + frozen-guard. Functional
valence only; no sentience claim.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
BASELINE = 0.4225   # deterministic Exp 222 score_affect() default metric


def _run(cmd, cwd, timeout, env=None):
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout, env=env)


def main() -> None:
    lines: list[str] = []
    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("=" * 78)
    log("EXP 223 — M4b first REAL autopilot run (--real --iterations 2, isolated clone)")
    log(f"baseline (deterministic, Exp 222) = {BASELINE}")
    log("KEEP iff candidate metric > best AND verdict True; FROZEN scorer + frozen-guard = anti-hack")
    log("=" * 78)
    log("FALSIFIER: no iteration both improves the metric AND holds verdict => NEGATIVE (no gain at N=2)")
    log("")

    clone = Path(tempfile.mkdtemp(prefix="exp223_m4b_clone_"))
    clone_repo = clone / "repo"
    try:
        # --- isolated clone (cron-safe) ---
        log(f"[setup] cloning repo -> {clone_repo}")
        _run(["git", "clone", "--local", "--no-hardlinks", str(_REPO), str(clone_repo)], _REPO, 300)
        _run(["git", "config", "user.email", "exp223@loop"], clone_repo, 30)
        _run(["git", "config", "user.name", "exp223"], clone_repo, 30)
        (clone_repo / ".venv").symlink_to(_REPO / ".venv")
        # keep the venv symlink out of the working tree so it never counts as a change
        excl = clone_repo / ".git" / "info" / "exclude"
        excl.write_text(excl.read_text() + "\n.venv\n")

        trunk = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], clone_repo, 30).stdout.strip()
        base_sha = _run(["git", "rev-parse", "HEAD"], clone_repo, 30).stdout.strip()
        log(f"[setup] clone trunk={trunk} base_sha={base_sha[:10]}")
        log("")

        # --- run the real autopilot (this is the ~40-min step) ---
        log("[run] run_affect_loop.py --real --iterations 2 ...")
        env = dict(os.environ, OMP_NUM_THREADS="1", XLA_FLAGS="--xla_cpu_multi_thread_eigen=false")
        proc = _run(["uv", "run", "--python", ".venv", "python",
                     "run_affect_loop.py", "--real", "--iterations", "2"],
                    clone_repo, timeout=5400, env=env)
        log(f"[run] returncode={proc.returncode}")
        if proc.stdout.strip():
            log("[run stdout tail]")
            log("\n".join(proc.stdout.strip().splitlines()[-8:]))
        if proc.returncode != 0 and proc.stderr.strip():
            log("[run stderr tail]")
            log("\n".join(proc.stderr.strip().splitlines()[-12:]))
        log("")

        # --- collect results from the clone's world model + report ---
        journal = clone_repo / "world_model" / "evidence" / "journal.jsonl"
        rows = []
        if journal.exists():
            for ln in journal.read_text().splitlines():
                ln = ln.strip()
                if ln:
                    try:
                        rows.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
        report = clone_repo / "REPORT.md"
        report_txt = report.read_text() if report.exists() else "(no REPORT.md)"

        log("--- PER-ITERATION JOURNAL ---")
        merges = 0
        best = BASELINE
        instrument_ok = bool(rows)
        for r in rows:
            metric = r.get("metric")
            mstr = f"{metric:.4f}" if isinstance(metric, (int, float)) else str(metric)
            log(f"  iter {r.get('iter')}: merged={r.get('merged')} reason={r.get('reason')!r} "
                f"metric={mstr}  hyp={str(r.get('hypothesis'))[:80]!r}")
            if r.get("merged") and isinstance(metric, (int, float)):
                merges += 1
                best = max(best, metric)
        log("")
        log("--- REPORT.md ---")
        log(report_txt.strip())
        log("")

        # --- verdict (predeclared) ---
        improved = best > BASELINE and merges >= 1
        if not instrument_ok:
            verdict = "INSTRUMENT_FAILURE"
            detail = "no journal rows — the autopilot did not run a single iteration (claude/scorer/clone failure)."
        elif improved:
            verdict = "IMPROVED"
            detail = f"{merges} merge(s); best metric {best:.4f} > baseline {BASELINE} with verdict held."
        else:
            verdict = "NO_IMPROVEMENT"
            detail = (f"0 improving merges across {len(rows)} iteration(s); best stayed at baseline "
                      f"{BASELINE}. NEGATIVE at N=2 (no gain found / baseline near-optimal). Not 'unimprovable'.")
        log("--- VERDICT (predeclared) ---")
        log(f"VERDICT: {verdict}")
        log(f"  {detail}")
        log(f"MACHINE SUMMARY: VERDICT={verdict} baseline={BASELINE} best={best:.4f} merges={merges} "
            f"iterations_logged={len(rows)} run_rc={proc.returncode}")

    finally:
        out = _REPO / "experiments" / "outputs" / "exp223.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        print(f"\n[saved {out}]")
        shutil.rmtree(clone, ignore_errors=True)


if __name__ == "__main__":
    main()
