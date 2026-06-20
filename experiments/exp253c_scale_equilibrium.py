"""experiments/exp253c_scale_equilibrium.py — Exp 253c: can a ROBUST (non-tiny) predator
equilibrium be reached now that interference damps the boom?

Exp 253b: the full natural stack gives bounded coexistence (both persist to t=1500, prey stable)
but the predator equilibrium is pinned TINY (~3-6) regardless of capR/assim/K -> stochastically
fragile. Two untested levers, both now safe because interference+Type III cap the boom:
  (1) HIGHER PREDATOR BUFFER (pred_cap): earlier caused boom-bust, but the stabilizers should now
      tolerate it -> faster predator numerical response -> bigger predator equilibrium.
  (2) MORE prey biomass (higher K) + more predators -> bigger ABSOLUTE populations -> robust to the
      demographic-stochasticity floor.

PREDICTION: some (pred_cap, K, w) cell yields a ROBUST equilibrium — pred_eq >= 15, both roles
alive at t=1500 on ALL seeds, min_pred_tail>0, cv_prey low. This would be the genuine, robust,
emergent natural predator-prey equilibrium needed to pose the Red Queen invasion test.

PREDECLARED FALSIFIER: if neither a bigger buffer nor more biomass lifts the predator equilibrium
to a robust non-tiny stable level (it stays tiny/fragile, or a bigger buffer re-tips to boom-bust),
then the coexistence on this substrate is INTRINSICALLY a tiny-predator knife-edge — bounded
coexistence exists but a robust posable equilibrium does not.

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

_spec = importlib.util.spec_from_file_location("exp253", os.path.join(_repo_root, "experiments", "exp253_full_combine.py"))
_e253 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e253)
build_cfg = _e253.build_cfg   # (n_pred, capR, w, h, pred_cap, rate, K, sensing, assim)
run = _e253.run
HORIZON = _e253.HORIZON
SEEDS = _e253.SEEDS


def main():
    L = []
    L.append("=" * 112)
    L.append("Exp 253c — ROBUST predator equilibrium via bigger buffer / more biomass (interference damps boom).")
    L.append(f"fixed: rate=0.5 capR=0.8 assim=0.9 h=4 n_pred=10 sensing=2.5; sweep pred_cap x K x w; horizon={HORIZON} seeds={SEEDS}")
    L.append("=" * 112)
    L.append(f"{'pred_cap':>8} {'K':>5} {'w':>5} {'t_end':>7} {'prey_eq':>8} {'pred_eq':>8} "
             f"{'minPredTail':>11} {'cv_prey':>8} {'BOTH@end':>9} {'STABLE':>7} {'ROBUST':>7}")
    L.append("-" * 112)
    robust = []
    for pred_cap, K, w in itertools.product([5.5, 10.0], [400, 800], [0.5, 1.5]):
        rs = [run(build_cfg(10, 0.8, w, h=4.0, pred_cap=pred_cap, rate=0.5, K=K, assim=0.9), s) for s in SEEDS]
        t_end = np.mean([x["t_end"] for x in rs]); prey_eq = np.mean([x["prey_eq"] for x in rs])
        pred_eq = np.mean([x["pred_eq"] for x in rs]); cvp = np.nanmean([x["cv_prey"] for x in rs])
        minpt = np.mean([x["min_pred_tail"] for x in rs]); both_all = all(x["both"] for x in rs)
        is_stable = both_all and all(x["min_pred_tail"] > 0 for x in rs)
        is_robust = is_stable and pred_eq >= 15.0
        L.append(f"{pred_cap:>8.1f} {K:>5} {w:>5.1f} {t_end:>7.1f} {prey_eq:>8.1f} {pred_eq:>8.1f} "
                 f"{minpt:>11.1f} {cvp:>8.3f} {str(both_all):>9} {str(is_stable):>7} {str(is_robust):>7}")
        if is_robust:
            robust.append((pred_cap, K, w, prey_eq, pred_eq, cvp))
    L.append("")
    L.append(f"ROBUST equilibria (both alive t={HORIZON} all seeds, min_pred_tail>0, pred_eq>=15): {len(robust)}")
    for g in robust:
        L.append(f"    ROBUST: pred_cap={g[0]} K={g[1]} w={g[2]} prey_eq={g[3]:.1f} pred_eq={g[4]:.1f} cv_prey={g[5]:.3f}")
    out = "\n".join(L)
    print(out)
    with open(os.path.join(_repo_root, "experiments", "outputs", "exp253c_scale_equilibrium.txt"), "w") as f:
        f.write(out + "\n")
    print("\n[written to experiments/outputs/exp253c_scale_equilibrium.txt]")


if __name__ == "__main__":
    main()
