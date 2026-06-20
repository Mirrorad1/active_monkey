"""experiments/exp253_full_combine.py — Exp 253: the FULL natural-ecology stack — does any
combination yield a genuine persistent predator-prey EQUILIBRIUM?

The diagnosis (Exp 248-252): on this substrate predator viability and system stability are
mutually exclusive — every single stabilizer that damps the boom also starves the predator. The
remaining hypothesis: predator INTERFERENCE (ratio-dependence) caps the boom from the PREDATOR
side while preserving a lone predator's capture success, so combined with the other natural
ingredients it might thread the viable-AND-stable needle.

FULL STACK: decoupled prey births (productivity) + modest predator buffer (low numerical-response
lag) + Type III response (emergent low-density prey refuge + saturation) + predator interference
(ratio-dependent self-limitation). Sweep interference_strength x n_pred x capture_radius.

PREDICTION: some cell yields PERSISTENT STABLE coexistence — both roles alive at t=1500 AND the
predator NEVER hits zero in the tail (min_pred_tail>0) — a genuine emergent equilibrium with NO
artificial fixed refuge.

PREDECLARED FALSIFIER: if NO cell in the full-stack sweep is stable, then NO combination of
natural ecological stabilizers produces a stable two-trophic equilibrium on this substrate — the
predator-viability-vs-stability trade-off is fundamental, and the Red Queen direction closes
(pending a deeper agent-metabolism redesign).

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

_spec = importlib.util.spec_from_file_location("exp250", os.path.join(_repo_root, "experiments", "exp250_decoupled_coexistence.py"))
_e250 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e250)
make_founder = _e250.make_founder
make_cfg = _e250.make_cfg

HORIZON = 1500
SEEDS = [0, 1, 2]


def roles_alive(eco):
    snap = eco.alive_snapshot()
    return (sum(1 for c in snap if c.genotype.role == "prey"),
            sum(1 for c in snap if c.genotype.role == "predator"))


def build_cfg(n_pred, capR, w, h=4.0, pred_cap=5.5, rate=0.5, K=250, sensing=2.5, assim=0.6):
    prey_geno = D.replace(make_founder(1.0), role="prey")
    pred_geno = D.replace(make_founder(1.4), role="predator", energy_capacity=pred_cap)
    cfg = make_cfg(speed=1.0, cost_slope=0.0, regen_rate=1.0, horizon=HORIZON,
                   founder_mix=((prey_geno, 21), (pred_geno, n_pred)))
    return D.replace(
        cfg,
        enable_decoupled_prey_birth=True, prey_birth_rate=rate, prey_carrying_capacity=float(K),
        enable_predation=True, freeze_prey_speed=True, mutate_predator_speed=False,
        capture_radius=capR, sensing_radius=sensing, assimilation_efficiency=assim,
        pred_start_energy_frac=0.75,
        enable_type3_response=True, type3_half_density=h, type3_exponent=2.0,
        enable_predator_interference=True, interference_strength=w, interference_radius=2.5,
    )


def run(cfg, seed):
    eco = Ecology(cfg, seed=seed)
    prey_s, pred_s = [], []
    while eco.has_alive() and not eco.exploded and eco.t < cfg.horizon:
        p, q = roles_alive(eco)
        prey_s.append(p); pred_s.append(q)
        eco.step()
    p, q = roles_alive(eco)
    prey_s.append(p); pred_s.append(q)
    tail_p, tail_q = prey_s[-500:], pred_s[-500:]
    cvp = float(np.std(tail_p) / np.mean(tail_p)) if np.mean(tail_p) > 0 else float("nan")
    return {"t_end": eco.t, "prey_final": p, "pred_final": q,
            "prey_eq": float(np.mean(tail_p)), "pred_eq": float(np.mean(tail_q)),
            "cv_prey": cvp, "min_pred_tail": min(tail_q) if tail_q else 0,
            "both": (p > 0 and q > 0)}


def main():
    L = []
    L.append("=" * 108)
    L.append("Exp 253 — FULL natural stack (decoupled births + low buffer + Type III + interference)")
    L.append(f"fixed: rate=0.5 K=250 pred_cap=5.5 h=4 sensing=2.5; sweep w x n_pred x capR; horizon={HORIZON} seeds={SEEDS}")
    L.append("=" * 108)
    L.append(f"{'w':>5} {'n_pred':>6} {'capR':>5} {'t_end':>7} {'prey_eq':>8} {'pred_eq':>8} "
             f"{'minPredTail':>11} {'cv_prey':>8} {'BOTH@end':>9} {'STABLE':>7}")
    L.append("-" * 108)
    stable = []
    for w, n_pred, capR in itertools.product([0.5, 1.5, 3.0], [4, 8], [0.45, 0.60]):
        rs = [run(build_cfg(n_pred, capR, w), s) for s in SEEDS]
        t_end = np.mean([x["t_end"] for x in rs]); prey_eq = np.mean([x["prey_eq"] for x in rs])
        pred_eq = np.mean([x["pred_eq"] for x in rs]); cvp = np.nanmean([x["cv_prey"] for x in rs])
        minpt = np.mean([x["min_pred_tail"] for x in rs]); both_all = all(x["both"] for x in rs)
        is_stable = both_all and all(x["min_pred_tail"] > 0 for x in rs)
        L.append(f"{w:>5.1f} {n_pred:>6} {capR:>5.2f} {t_end:>7.1f} {prey_eq:>8.1f} {pred_eq:>8.1f} "
                 f"{minpt:>11.1f} {cvp:>8.3f} {str(both_all):>9} {str(is_stable):>7}")
        if is_stable:
            stable.append((w, n_pred, capR, prey_eq, pred_eq, cvp))
    L.append("")
    L.append(f"PERSISTENT STABLE coexistence cells (both alive t={HORIZON}, predator NEVER hit 0 in tail): {len(stable)}")
    for g in stable:
        L.append(f"    STABLE: w={g[0]} n_pred={g[1]} capR={g[2]} prey_eq={g[3]:.1f} pred_eq={g[4]:.1f} cv_prey={g[5]:.3f}")
    out = "\n".join(L)
    print(out)
    with open(os.path.join(_repo_root, "experiments", "outputs", "exp253_full_combine.txt"), "w") as f:
        f.write(out + "\n")
    print("\n[written to experiments/outputs/exp253_full_combine.txt]")


if __name__ == "__main__":
    main()
