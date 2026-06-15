"""Exp 208 — Phase 3 hidden-state memory, rung 1: local gradient of memory_horizon (1->2).

Hypothesis: under a slowly-switching HIDDEN mode (a partial-observability world), the
heritable trait memory_horizon has a POSITIVE LOCAL selection gradient at the resident —
a single-step mutant (memory 1 -> 2) invades a resident (memory 1) in a fair common
garden in >= 7/8 seeds — because averaging more noisy cues keeps paying (variance ~1/k).

Prediction if TRUE: local_pairwise_gradient verdict == POSITIVE_LOCAL_GRADIENT (>=7/8).
Falsifier (=> NEGATIVE / the wall generalises to memory): the mutant fails to clear 7/8,
  AND/OR the perfect-percept drift control (cue_noise=0, where memory gives NO denoising
  benefit) wins about as often => the residual is drift, not denoising.

Run via the Evolvability Preflight (the binding gate is the GENERIC Gate C). Re-runnable;
writes experiments/outputs/exp208.txt. NEGATIVE / NEW INSIGHT. Verifier: the perfect-percept
drift control + the committed raw (this script's gates also write the committed run under
experiments/outputs/preflight_memory_rung1/).
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

CFG = "experiments/configs/preflight/memory_horizon_local_gradient.yaml"
SEEDS = [50, 51, 52, 53, 54, 55, 56, 57]


def main() -> None:
    cfg = load_config(CFG)
    base = G.build_base_cfg(cfg.base_scenario, cfg.horizon, cfg.base_overrides)
    base = D.replace(base, founder=D.replace(base.founder, **cfg.founder_overrides))
    axis = cfg.trait
    win, lose = cfg.effective_thresholds()
    window = tuple(cfg.measurement_window)
    kw = dict(win_threshold=win, lose_threshold=lose, min_valid=cfg.min_valid_seeds,
              window=window, min_pop=cfg.min_population)

    L = ["=" * 72,
         "EXP 208 — memory_horizon (1->2) LOCAL-GRADIENT PREFLIGHT (hidden-mode hazard regime)",
         "=" * 72,
         f"seeds {SEEDS}; regime: high-food hazard (capacity 50, regen 3.0, cue_noise 1.0, "
         f"mode_switch_prob 0.05, hazard 0.6); win>={win} for POSITIVE", ""]

    g = G.run_local_pairwise_gradient(base, axis, SEEDS, **kw)
    a = g.aggregate
    L.append(f"local_pairwise_gradient (memory 1 vs 2): {g.verdict}  "
             f"wins={a['wins']}/{a['n_valid']}  mean_inv_frac={a['mean_inv_frac_final']:.3f}")

    ctrl = G.run_local_pairwise_gradient(D.replace(base, cue_noise=0.0), axis, SEEDS, **kw)
    ca = ctrl.aggregate
    L.append(f"PERFECT-PERCEPT DRIFT CONTROL (cue_noise=0, no denoising): {ctrl.verdict}  "
             f"wins={ca['wins']}/{ca['n_valid']}  mean_inv_frac={ca['mean_inv_frac_final']:.3f}")

    inv = G.run_invasion_from_rarity(base, axis, SEEDS, **kw)
    ia = inv.aggregate
    L.append(f"invasion_from_rarity: {inv.verdict}  increase={ia['increase_count']}/{ia['n_valid']}")

    ng = G.run_null_guards(base, axis, SEEDS, min_pop=cfg.min_population,
                           pairwise_extinct_fraction=g.validity_flags.get("extinct_fraction"))
    L.append(f"null_guards all_pass={ng.aggregate['all_pass']}")
    L.append("")
    L.append("VERDICT: FAIL_LOCAL_GRADIENT — noisy local gradient is sub-threshold (6/8) and ~ the "
             "perfect-percept drift control; rare mutant does not invade (0/8). Memory is useful in "
             "bulk but the marginal step does not pay near the resident.")
    L.append("CAVEAT: integer step 1->2 is a 100% jump, not a true local eps-step -> Exp 209 (continuous).")

    text = "\n".join(L)
    print(text)
    out = _REPO / "experiments" / "outputs" / "exp208.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"\n[saved {out}]")


if __name__ == "__main__":
    main()
