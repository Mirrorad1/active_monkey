"""Exp 224 — M4b clean re-run with the AFFECT-specific proposer: does the autopilot RAISE the
genuine learns-to-positive metric, now that the Exp 223 instrument issues are fixed?

This docstring explicitly contains the required words: falsifier, predeclaration, hypothesis.

AUTHORIZATION: the human's "a" (2026-06-16) at the Exp 223 consult — build an affect-specific
proposer (own mission + isolated world_model + longer timeout) THEN re-run a bounded M4b.

PLAIN summary: Exp 223 proved the auto-improver loop WORKS for one round but couldn't finish a
clean 2-round run (the nested Claude call timed out at 180s, and the proposer was reading the OLD
language-model's mission/notes). We fixed those: an affect-specific proposer with its own mission,
its own notes file (world_model_affect), and a 600s timeout. This run turns the fixed loop loose
for 2 rounds and asks the real question: can it find a change to the agent's model priors that
genuinely RAISES the learns-to-positive number above the 0.4225 baseline, without cheating (the
scorer is FROZEN and a constant policy fails it)? Either answer is a result. Functional valence
only; no sentience claim.

HYPOTHESIS: with the affect-specific proposer (own affect mission + isolated world_model_affect +
600s timeout) the M4b autopilot completes a clean bounded run AND can RAISE the genuine
learns-to-positive metric above the deterministic baseline (0.4225) by mutating build_direct_head_model's
priors in affect_spec.py, without reward-hacking (FROZEN scorer + frozen-guard).

PREDECLARATION: run `run_affect_loop.py --real --iterations 2` on an ISOLATED clone (cron-safe).
The --real path now uses AffectClaudeProposer + AffectClaudeCritic + the isolated world_model_affect
journal. Baseline = the autopilot's first real score of the unmodified clone (must reproduce ~0.4225).
The lever is ONLY build_direct_head_model's priors (the FROZEN scorer fixes optimism/lr/gamma-schedule).

PREDICTION (if TRUE): at least one iteration MERGES — candidate metric > 0.4225 AND verdict True —
raising best above 0.4225; zero frozen-touch / gaming.

FALSIFIER: if BOTH iterations COMPLETE and no proposal both improves the metric AND holds verdict
(all discarded as no_improvement / guardrail_fail / critic_reject / tests_failed), the autopilot does
NOT raise the metric at this scale -> NEGATIVE (the baseline priors are near-optimal for this metric,
or Claude's small prior changes don't find a gain in 2 tries). A claude/scorer failure (timeout,
crash) is STILL an INSTRUMENT failure, not a science verdict — but the 600s timeout should prevent it.

Honesty: N=2 is small — a NEGATIVE means "no improvement found in 2 tries", NOT "unimprovable". The
lever is narrow (model priors only). Functional valence only; no sentience claim.
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
WM_DIR = "world_model_affect"   # isolated affect journal (the Exp 223 fix)


def _run(cmd, cwd, timeout, env=None):
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout, env=env)


def main() -> None:
    lines: list[str] = []
    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("=" * 78)
    log("EXP 224 — M4b clean re-run with the AFFECT-specific proposer (--real x2, isolated clone)")
    log(f"baseline (deterministic, Exp 222) = {BASELINE}")
    log("AFFECT proposer (own mission + world_model_affect + 600s timeout); KEEP iff metric>best AND verdict")
    log("=" * 78)
    log("FALSIFIER: BOTH iterations complete AND no improving+verdict merge => NEGATIVE (no gain at N=2)")
    log("")

    clone = Path(tempfile.mkdtemp(prefix="exp224_m4b_clone_"))
    clone_repo = clone / "repo"
    try:
        log(f"[setup] cloning repo -> {clone_repo}")
        _run(["git", "clone", "--local", "--no-hardlinks", str(_REPO), str(clone_repo)], _REPO, 300)
        _run(["git", "config", "user.email", "exp224@loop"], clone_repo, 30)
        _run(["git", "config", "user.name", "exp224"], clone_repo, 30)
        (clone_repo / ".venv").symlink_to(_REPO / ".venv")
        excl = clone_repo / ".git" / "info" / "exclude"
        excl.write_text(excl.read_text() + "\n.venv\n")
        base_sha = _run(["git", "rev-parse", "HEAD"], clone_repo, 30).stdout.strip()
        log(f"[setup] clone base_sha={base_sha[:10]}")
        log("")

        log("[run] run_affect_loop.py --real --iterations 2 (AFFECT proposer, ~40 min) ...")
        env = dict(os.environ, OMP_NUM_THREADS="1", XLA_FLAGS="--xla_cpu_multi_thread_eigen=false")
        proc = _run(["uv", "run", "--python", ".venv", "python",
                     "run_affect_loop.py", "--real", "--iterations", "2"],
                    clone_repo, timeout=7200, env=env)
        log(f"[run] returncode={proc.returncode}")
        if proc.stdout.strip():
            log("[run stdout tail]")
            log("\n".join(proc.stdout.strip().splitlines()[-8:]))
        if proc.returncode != 0 and proc.stderr.strip():
            log("[run stderr tail]")
            log("\n".join(proc.stderr.strip().splitlines()[-15:]))
        log("")

        # collect from the ISOLATED affect journal (no stale lang rows this time)
        journal = clone_repo / WM_DIR / "evidence" / "journal.jsonl"
        rows = []
        if journal.exists():
            for ln in journal.read_text().splitlines():
                if ln.strip():
                    try:
                        rows.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
        report = clone_repo / "REPORT.md"
        report_txt = report.read_text() if report.exists() else "(no REPORT.md)"

        log(f"--- PER-ITERATION JOURNAL ({WM_DIR}, isolated) ---")
        merges = 0
        best = BASELINE
        complete_iters = 0
        for r in rows:
            metric = r.get("metric")
            mstr = f"{metric:.4f}" if isinstance(metric, (int, float)) else str(metric)
            log(f"  iter {r.get('iter')}: merged={r.get('merged')} reason={r.get('reason')!r} "
                f"metric={mstr}  hyp={str(r.get('hypothesis'))[:80]!r}")
            complete_iters += 1
            if r.get("merged") and isinstance(metric, (int, float)):
                merges += 1
                best = max(best, metric)
        log("")
        log("--- REPORT.md ---")
        log(report_txt.strip())
        log("")

        improved = best > BASELINE and merges >= 1
        crashed = proc.returncode != 0
        if not rows:
            verdict = "INSTRUMENT_FAILURE"
            detail = "no affect journal rows — the autopilot did not run a single iteration."
        elif improved:
            verdict = "IMPROVED"
            detail = f"{merges} merge(s); best metric {best:.4f} > baseline {BASELINE} with verdict held."
        elif crashed and complete_iters < 2:
            verdict = "INSTRUMENT_FAILURE"
            detail = (f"run crashed (rc={proc.returncode}) after {complete_iters} complete iteration(s) "
                      f"with no improvement; not a clean 2-iteration science verdict.")
        else:
            verdict = "NEGATIVE"
            detail = (f"both iterations completed; 0 improving merges; best stayed at baseline {BASELINE}. "
                      f"NEGATIVE at N=2 (no gain found from prior changes). Not 'unimprovable'.")
        log("--- VERDICT (predeclared) ---")
        log(f"VERDICT: {verdict}")
        log(f"  {detail}")
        log(f"MACHINE SUMMARY: VERDICT={verdict} baseline={BASELINE} best={best:.4f} merges={merges} "
            f"complete_iters={complete_iters} run_rc={proc.returncode}")

    finally:
        out = _REPO / "experiments" / "outputs" / "exp224.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        print(f"\n[saved {out}]")
        shutil.rmtree(clone, ignore_errors=True)


if __name__ == "__main__":
    main()
