"""experiments/exp253b_robust_equilibrium.py — Exp 253b: lift the predator equilibrium above
the stochastic-extinction floor to get a ROBUST natural predator-prey equilibrium.

Exp 253 full stack reached t=1500 with both roles alive + stable prey (cv~0.06) but a TINY
predator equilibrium (pred_eq 1-5) that stochastically winks out on some seeds. The boom-bust is
gone; the only remaining problem is small-population fragility. Interference (w) self-limits the
predator (good for damping, bad for size). FIX: LOW w (just enough to damp) + HIGH predator
energetic yield (capR / assimilation / prey K) so the predator equilibrium sits robustly above
the demographic-stochasticity floor.

PREDICTION: some (w, capR, assim) cell yields pred_eq robustly >~10 with BOTH roles alive at
t=1500 on ALL seeds AND min_pred_tail>0 (predator never hits zero) AND low cv_prey — a genuine,
robust, emergent natural equilibrium (no artificial refuge).

PREDECLARED FALSIFIER: if raising predator yield never lifts the predator equilibrium to a robust
non-tiny stable level (every cell is either still tiny/fragile, starved, or tipped back to
boom-bust), then a robust natural equilibrium is not reachable on this substrate.

RAW NUMBERS — controller judges.
"""
import sys
import os
import importlib.util
import dataclasses as D
import itertools

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ecology.engine import Ecology

_spec = importlib.util.spec_from_file_location("exp253", os.path.join(_repo_root, "experiments", "exp253_full_combine.py"))
_e253 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e253)
build_cfg = _e253.build_cfg   # (n_pred, capR, w, h, pred_cap, rate, K, sensing, assim)
run = _e253.run
HORIZON = _e253.HORIZON
SEEDS = _e253.SEEDS


def main():
    L = []
    L.append("=" * 110)
    L.append("Exp 253b — ROBUST natural equilibrium: low interference + high predator yield.")
    L.append(f"fixed: rate=0.5 K=300 pred_cap=5.5 h=4 n_pred=6 sensing=2.5; sweep w x capR x assim; horizon={HORIZON} seeds={SEEDS}")
    L.append("=" * 110)
    L.append(f"{'w':>5} {'capR':>5} {'assim':>6} {'t_end':>7} {'prey_eq':>8} {'pred_eq':>8} "
             f"{'minPredTail':>11} {'cv_prey':>8} {'BOTH@end':>9} {'STABLE':>7} {'ROBUST':>7}")
    L.append("-" * 110)
    robust = []
    for w, capR, assim in itertools.product([0.3, 0.5], [0.6, 0.8], [0.7, 0.9]):
        rs = [run(build_cfg(6, capR, w, h=4.0, pred_cap=5.5, rate=0.5, K=300, assim=assim), s) for s in SEEDS]
        t_end = np.mean([x["t_end"] for x in rs]); prey_eq = np.mean([x["prey_eq"] for x in rs])
        pred_eq = np.mean([x["pred_eq"] for x in rs]); cvp = np.nanmean([x["cv_prey"] for x in rs])
        minpt = np.mean([x["min_pred_tail"] for x in rs]); both_all = all(x["both"] for x in rs)
        is_stable = both_all and all(x["min_pred_tail"] > 0 for x in rs)
        is_robust = is_stable and pred_eq >= 10.0
        L.append(f"{w:>5.1f} {capR:>5.2f} {assim:>6.2f} {t_end:>7.1f} {prey_eq:>8.1f} {pred_eq:>8.1f} "
                 f"{minpt:>11.1f} {cvp:>8.3f} {str(both_all):>9} {str(is_stable):>7} {str(is_robust):>7}")
        if is_robust:
            robust.append((w, capR, assim, prey_eq, pred_eq, cvp))
    L.append("")
    L.append(f"ROBUST stable equilibria (both alive t={HORIZON} all seeds, min_pred_tail>0, pred_eq>=10): {len(robust)}")
    for g in robust:
        L.append(f"    ROBUST: w={g[0]} capR={g[1]} assim={g[2]} prey_eq={g[3]:.1f} pred_eq={g[4]:.1f} cv_prey={g[5]:.3f}")
    out = "\n".join(L)
    print(out)
    with open(os.path.join(_repo_root, "experiments", "outputs", "exp253b_robust_equilibrium.txt"), "w") as f:
        f.write(out + "\n")
    print("\n[written to experiments/outputs/exp253b_robust_equilibrium.txt]")


if __name__ == "__main__":
    main()
