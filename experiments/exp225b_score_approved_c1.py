"""Exp 225b — does the M4b critic-APPROVED direction actually improve the metric?

The Exp 225 critic diagnostic found the AffectClaudeCritic correctly REJECTS baking the code->intent
structure into A0 (gaming) and APPROVES a mild NEU-aversion on C1 (a legitimate preference lever: the
agent commits to a POS-earning response instead of dithering on the never-positive ASK->NEU). This
scores that APPROVED direction with the FROZEN scorer's EXACT session (only C1 overridden) over the
default 8 seeds x 300t, vs the deterministic baseline 0.4225.

Question: does the critic-approved C1 NEU-aversion BEAT baseline (so the M4b autopilot has a real
improving move it would find+keep), or not (even the legit lever doesn't help at this scale)?
This is the agent's PREFERENCE, not the answer — discrimination (correct_select) stays genuine.
Functional valence only; no sentience claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eval.affect_score import (
    _run_session, CORRECT, CEIL, K, OPTIMISM, LR, SEEDS_DEFAULT, TURNS_DEFAULT,
)
from active_loop.affect_spec import build_direct_head_model, LV
from active_loop.affect_agent import DirectHeadAgent

BASELINE = 0.4225          # C1 neutral = 0.0 (committed affect_spec)
NEU_VALUES = [-0.5, -1.0]  # the approved "mild NEU aversion" direction


def factory_with_neu(neu: float):
    def f(seed: int, turns: int):
        m = build_direct_head_model(seed, k=K)
        m["C"][1] = jnp.array(np.array([-2.0, neu, 3.0])[None])  # override only C1 (length-3 preserved)
        return DirectHeadAgent(
            m, seed=seed, gamma=1.0, alpha=1.0, lr_pA=LR, lv=LV,
            optimism=OPTIMISM, gamma_schedule=(1.0, 8.0, turns),
        )
    return f


def score_neu(neu: float):
    firsts, lasts, csels, genu = [], [], [], 0
    for seed in SEEDS_DEFAULT:
        row = _run_session(factory_with_neu(neu), seed, TURNS_DEFAULT)
        firsts.append(row["first"]); lasts.append(row["last"]); csels.append(row["csel"])
        genu += int(row["csel"] >= 0.5 and row["last"] > CEIL)
    return dict(neu=neu, mean_first=float(np.mean(firsts)), mean_last=float(np.mean(lasts)),
                mean_csel=float(np.mean(csels)), genuine=genu, n=len(SEEDS_DEFAULT))


def main() -> None:
    lines: list[str] = []
    def log(s=""):
        print(s, flush=True); lines.append(s)

    log("=" * 78)
    log("EXP 225b — score the M4b critic-APPROVED C1 NEU-aversion vs baseline 0.4225")
    log(f"FROZEN scorer session (8 seeds x 300t); only C1 overridden; CEIL={CEIL:.3f}")
    log("=" * 78)
    log(f"baseline (C1 neu=0.0) = {BASELINE}  (Exp 222 deterministic)")
    log("")
    results = []
    for neu in NEU_VALUES:
        r = score_neu(neu)
        results.append(r)
        delta = r["mean_last"] - BASELINE
        log(f"C1 neu={neu:+.2f}: mean_last={r['mean_last']:.4f} (delta {delta:+.4f})  "
            f"mean_first={r['mean_first']:.4f}  mean_csel={r['mean_csel']:.3f}  genuine={r['genuine']}/{r['n']}")
    log("")
    best = max(results, key=lambda r: r["mean_last"])
    improved = best["mean_last"] > BASELINE
    verdict = "APPROVED_DIRECTION_IMPROVES" if improved else "APPROVED_DIRECTION_NO_GAIN"
    log("--- VERDICT ---")
    if improved:
        log(f"VERDICT: {verdict} — the critic-approved C1 neu={best['neu']:+.2f} raises mean_last to "
            f"{best['mean_last']:.4f} > baseline {BASELINE} (+{best['mean_last']-BASELINE:.4f}); the M4b "
            f"autopilot has a real improving move it would find+keep.")
    else:
        log(f"VERDICT: {verdict} — no tested C1 neu beats baseline {BASELINE} (best mean_last "
            f"{best['mean_last']:.4f}); even the legit approved lever doesn't improve the metric at this scale.")
    log(f"MACHINE SUMMARY: VERDICT={verdict} baseline={BASELINE} "
        + " ".join(f"neu{r['neu']:+.2f}={r['mean_last']:.4f}(g{r['genuine']})" for r in results))

    out = _REPO / "experiments" / "outputs" / "exp225b.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    (_REPO / "experiments" / "outputs" / "exp225b.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n[saved {out}]")


if __name__ == "__main__":
    main()
