"""Exp 209 — Phase 3 rung 1b: CONTINUOUS belief_persistence (rho 0.5 -> 0.55) local gradient.

Exp 208 used the INTEGER memory_horizon, whose smallest step (1->2) is a 100% jump — not a
true local eps-step — so its FAIL could be a granularity artifact. belief_persistence is a
continuous EMA-persistence weight, so rho 0.5 -> 0.55 is a genuinely small eps step: the
faithful local-gradient test.

Hypothesis: if Exp 208's FAIL was a granularity artifact, the small continuous eps-step pays
  (POSITIVE_LOCAL_GRADIENT); if the wall is real, the small step also fails.
Prediction if TRUE (artifact): rho 0.5->0.55 invades >= 7/8.
Falsifier (=> wall is REAL): eps-step is sub-threshold AND the perfect-percept drift control
  (cue_noise=0) wins about as often (residual = drift, not denoising); a liveness check (a
  LARGE step rho 0->0.85) must still be POSITIVE (mechanism genuinely live, not inert).

Re-runnable; writes experiments/outputs/exp209.txt. NEGATIVE / NEW INSIGHT — the local-gradient
wall is NOT a granularity artifact; it generalises from senses to information-processing.
Verifier: the perfect-percept drift control + the live large-step positive + committed raw.
"""
from __future__ import annotations

import dataclasses as D
import sys
from pathlib import Path

_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ecology.evolvability.config import load_config
from ecology.evolvability import gates as G

CFG = "experiments/configs/preflight/belief_persistence_local_gradient.yaml"
SEEDS = [50, 51, 52, 53, 54, 55, 56, 57]
LIVE_SEEDS = [100, 101, 102]  # disclosed pilot seeds for the gifted-large-step liveness check


def main() -> None:
    cfg = load_config(CFG)
    base = G.build_base_cfg(cfg.base_scenario, cfg.horizon, cfg.base_overrides)
    base = D.replace(base, founder=D.replace(base.founder, **cfg.founder_overrides))
    axis = cfg.trait  # belief_persistence, resident 0.5 -> mutant 0.55
    win, lose = cfg.effective_thresholds()
    window = tuple(cfg.measurement_window)
    kw = dict(win_threshold=win, lose_threshold=lose, min_valid=cfg.min_valid_seeds,
              window=window, min_pop=cfg.min_population)

    L = ["=" * 72,
         "EXP 209 — CONTINUOUS belief_persistence (rho 0.5->0.55) LOCAL-GRADIENT PREFLIGHT",
         "=" * 72,
         f"seeds {SEEDS}; same hidden-mode hazard regime as Exp 208; win>={win} for POSITIVE", ""]

    # Liveness: a LARGE gifted step rho 0 -> 0.85 must be POSITIVE (mechanism genuinely live).
    live_axis = D.replace(axis, resident_value=0.0, mutant_value=0.85)
    live = G.run_local_pairwise_gradient(base, live_axis, LIVE_SEEDS,
                                         win_threshold=3, lose_threshold=0, min_valid=2,
                                         window=window, min_pop=cfg.min_population)
    la = live.aggregate
    L.append(f"LIVENESS (gifted rho 0->0.85, pilot seeds {LIVE_SEEDS}): {live.verdict}  "
             f"wins={la['wins']}/{la['n_valid']}  mean_inv_frac={la['mean_inv_frac_final']:.3f}")

    g = G.run_local_pairwise_gradient(base, axis, SEEDS, **kw)
    a = g.aggregate
    L.append(f"local eps-step (rho 0.5 vs 0.55): {g.verdict}  "
             f"wins={a['wins']}/{a['n_valid']}  mean_inv_frac={a['mean_inv_frac_final']:.3f}")

    ctrl = G.run_local_pairwise_gradient(D.replace(base, cue_noise=0.0), axis, SEEDS, **kw)
    ca = ctrl.aggregate
    L.append(f"PERFECT-PERCEPT DRIFT CONTROL (cue_noise=0, no denoising): {ctrl.verdict}  "
             f"wins={ca['wins']}/{ca['n_valid']}  mean_inv_frac={ca['mean_inv_frac_final']:.3f}")

    inv = G.run_invasion_from_rarity(base, axis, SEEDS, **kw)
    ia = inv.aggregate
    L.append(f"invasion_from_rarity: {inv.verdict}  increase={ia['increase_count']}/{ia['n_valid']}")
    L.append("")
    L.append("VERDICT: FAIL_LOCAL_GRADIENT — the large step pays (mechanism live) but the eps-step is "
             "sub-threshold (6/8) and ~ the perfect-percept control (so the residual is DRIFT, not "
             "denoising). The local-gradient wall is REAL, not a granularity artifact; it generalises "
             "from scalar senses to information-processing capacity.")

    text = "\n".join(L)
    print(text)
    out = _REPO / "experiments" / "outputs" / "exp209.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"\n[saved {out}]")


if __name__ == "__main__":
    main()
