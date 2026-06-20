"""experiments/exp254b_stabilize.py — Exp 254b: tighten the predator self-limit to land the
near-equilibrium (pred_eq~19, collapsed at t=600 in Exp 254) into a STABLE-to-horizon regime.

Exp 254 showed predator self-limiting mortality breaks the intake cap (pred_eq~19, vs the ~5
intake-capped ceiling) but the system still slow-oscillates to collapse (~t=600). The collapse is
a SLOW divergent oscillation, so sharper damping should tip it into the stable basin:
  - higher Hill exponent theta (mortality bites SHARPLY near K_P -> tight regulation, less overshoot)
  - lower predator buffer pred_cap (less numerical-response lag)

PREDICTION: some (theta, hmax, pred_cap) cell around K_P=20 lands a STABLE equilibrium —
pred_eq robustly >=12, both alive at t=1500 on ALL seeds, min_pred_tail>0, low cv_prey.

PREDECLARED FALSIFIER: if sharper self-limit + lower buffer still cannot stabilize the
near-equilibrium to the horizon (it stays oscillatory/collapsing or over-damps to predator
starvation), then a clean stable predator-prey FIXED POINT is unreachable on this substrate even
with full doubly-logistic + low-lag tuning — the wall is complete and fully mapped.

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

_spec = importlib.util.spec_from_file_location("exp254", os.path.join(_repo_root, "experiments", "exp254_bazykin_equilibrium.py"))
_e254 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e254)
make_founder = _e254.make_founder
make_cfg = _e254.make_cfg
run = _e254.run
HORIZON = _e254.HORIZON
SEEDS = _e254.SEEDS

from ecology.engine import Ecology


def build_cfg(theta, hmax, pred_cap, Kp=20, capR=0.45, max_cap=2, K_prey=300, rate=0.5,
              n_pred=10, assim=0.7, sensing=2.5):
    prey_geno = D.replace(make_founder(1.0), role="prey")
    pred_geno = D.replace(make_founder(1.4), role="predator", energy_capacity=pred_cap)
    cfg = make_cfg(speed=1.0, cost_slope=0.0, regen_rate=1.0, horizon=HORIZON,
                   founder_mix=((prey_geno, 21), (pred_geno, n_pred)))
    return D.replace(
        cfg,
        enable_decoupled_prey_birth=True, prey_birth_rate=rate, prey_carrying_capacity=float(K_prey),
        enable_predation=True, freeze_prey_speed=True, mutate_predator_speed=False,
        capture_radius=capR, sensing_radius=sensing, assimilation_efficiency=assim,
        pred_start_energy_frac=0.75, max_captures_per_step=max_cap,
        enable_type3_response=False, enable_predator_interference=False,
        enable_predator_self_limit=True, predator_self_limit_kc=float(Kp),
        predator_self_limit_hmax=hmax, predator_self_limit_theta=float(theta),
    )


def main():
    L = []
    L.append("=" * 110)
    L.append("Exp 254b — STABILIZE the near-equilibrium: sharper predator self-limit (theta,hmax) + lower buffer.")
    L.append(f"fixed: K_P=20 capR=0.45 max_cap=2 K_prey=300 rate=0.5 n_pred=10; sweep theta x hmax x pred_cap; horizon={HORIZON} seeds={SEEDS}")
    L.append("=" * 110)
    L.append(f"{'theta':>5} {'hmax':>5} {'pred_cap':>8} {'t_end':>7} {'prey_eq':>8} {'pred_eq':>8} "
             f"{'minPredTail':>11} {'cv_prey':>8} {'BOTH@end':>9} {'STABLE':>7} {'ROBUST':>7}")
    L.append("-" * 110)
    robust = []
    for theta, hmax, pred_cap in itertools.product([2.0, 3.0], [0.3, 0.5], [5.5, 7.0]):
        rs = [run(build_cfg(theta, hmax, pred_cap), s) for s in SEEDS]
        t_end = np.mean([x["t_end"] for x in rs]); prey_eq = np.mean([x["prey_eq"] for x in rs])
        pred_eq = np.mean([x["pred_eq"] for x in rs]); cvp = np.nanmean([x["cv_prey"] for x in rs])
        minpt = np.mean([x["min_pred_tail"] for x in rs]); both_all = all(x["both"] for x in rs)
        is_stable = both_all and all(x["min_pred_tail"] > 0 for x in rs)
        is_robust = is_stable and pred_eq >= 12.0
        L.append(f"{theta:>5.0f} {hmax:>5.2f} {pred_cap:>8.1f} {t_end:>7.1f} {prey_eq:>8.1f} {pred_eq:>8.1f} "
                 f"{minpt:>11.1f} {cvp:>8.3f} {str(both_all):>9} {str(is_stable):>7} {str(is_robust):>7}")
        if is_stable:
            robust.append((theta, hmax, pred_cap, prey_eq, pred_eq, cvp, is_robust))
    L.append("")
    L.append(f"STABLE coexistence cells (both alive t={HORIZON} all seeds, min_pred_tail>0): {len(robust)}  "
             f"[ROBUST (pred_eq>=12): {sum(1 for g in robust if g[6])}]")
    for g in robust:
        tag = "ROBUST" if g[6] else "stable-but-small"
        L.append(f"    {tag}: theta={g[0]} hmax={g[1]} pred_cap={g[2]} prey_eq={g[3]:.1f} pred_eq={g[4]:.1f} cv_prey={g[5]:.3f}")
    out = "\n".join(L)
    print(out)
    with open(os.path.join(_repo_root, "experiments", "outputs", "exp254b_stabilize.txt"), "w") as f:
        f.write(out + "\n")
    print("\n[written to experiments/outputs/exp254b_stabilize.txt]")


if __name__ == "__main__":
    main()
