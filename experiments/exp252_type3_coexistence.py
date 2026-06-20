"""experiments/exp252_type3_coexistence.py — Exp 252: does a NATURAL Type III predator
functional response (emergent low-density prey refuge + saturation) yield a genuine,
persistent predator-prey EQUILIBRIUM — without any artificial fixed refuge?

HYPOTHESIS / PREDICTION: Exp 251 diagnosed the boom-bust latent state as (1) predator
energy-buffer lag (dampable: cv 0.56->0.16 as buffer shrinks) + (2) a functional response with
no low-density refuge / no saturation (predator finds rare prey -> drives extinction; booms on
dense prey). Exp 252 combines the three NATURAL ingredients: decoupled prey births (productivity),
a modest predator buffer (cap~5.5, the cv~0.16 point), and a Type III response (emergent refuge +
saturation). PREDICTION: some (half_density h, capR, n_pred) cell yields a PERSISTENT bounded
coexistence — both roles alive at the horizon AND the predator NEVER hits zero in the tail
(min_pred_tail>0) — a genuine equilibrium/limit cycle, not a long transient, with no artificial
protection.

PREDECLARED FALSIFIER: if NO cell keeps both roles alive at t=1200 with min_pred_tail>0 across the
sweep, the Type III natural stabilizer does not produce a stable equilibrium on this substrate
either, and the next lever is predator interference (ratio-dependence) or accepting the close.

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

HORIZON = 1200
SEEDS = [0, 1, 2]
PREY_BIRTH_RATE = 0.5
K_PREY = 250
PRED_CAP = 5.5          # modest buffer (Exp 251 cv~0.16 point); threshold 4.2 -> buffer 1.3


def roles_alive(eco):
    snap = eco.alive_snapshot()
    return (sum(1 for c in snap if c.genotype.role == "prey"),
            sum(1 for c in snap if c.genotype.role == "predator"))


def build_cfg(n_pred, capR, h, sensing=2.5, assim=0.6, k=2.0, type3=True, pred_cap=PRED_CAP):
    prey_geno = D.replace(make_founder(1.0), role="prey")
    pred_geno = D.replace(make_founder(1.4), role="predator", energy_capacity=pred_cap)
    cfg = make_cfg(speed=1.0, cost_slope=0.0, regen_rate=1.0, horizon=HORIZON,
                   founder_mix=((prey_geno, 21), (pred_geno, n_pred)))
    return D.replace(
        cfg,
        enable_decoupled_prey_birth=True, prey_birth_rate=PREY_BIRTH_RATE,
        prey_carrying_capacity=float(K_PREY), enable_predation=True,
        freeze_prey_speed=True, mutate_predator_speed=False,
        capture_radius=capR, sensing_radius=sensing, assimilation_efficiency=assim,
        pred_start_energy_frac=0.75,
        enable_type3_response=type3, type3_half_density=float(h), type3_exponent=k,
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
    tail_p, tail_q = prey_s[-400:], pred_s[-400:]
    cvp = float(np.std(tail_p) / np.mean(tail_p)) if np.mean(tail_p) > 0 else float("nan")
    return {"t_end": eco.t, "prey_final": p, "pred_final": q,
            "prey_eq": float(np.mean(tail_p)), "pred_eq": float(np.mean(tail_q)),
            "cv_prey": cvp, "min_pred_tail": min(tail_q) if tail_q else 0,
            "both": (p > 0 and q > 0)}


def main():
    L = []
    L.append("=" * 104)
    L.append(f"Exp 252 — Type III response + decoupled births + modest buffer: persistent equilibrium?")
    L.append(f"decoupled rate={PREY_BIRTH_RATE} K={K_PREY} pred_cap={PRED_CAP}(buffer~1.3) type3 k=2; horizon={HORIZON} seeds={SEEDS}")
    L.append("=" * 104)
    L.append(f"{'h':>4} {'n_pred':>6} {'capR':>5} {'t_end':>7} {'prey_eq':>8} {'pred_eq':>8} "
             f"{'minPredTail':>11} {'cv_prey':>8} {'BOTH@end':>9} {'STABLE':>7}")
    L.append("-" * 104)
    stable = []
    for h, n_pred, capR in itertools.product([2.0, 4.0, 8.0], [3, 5], [0.35, 0.45]):
        rs = [run(build_cfg(n_pred, capR, h), s) for s in SEEDS]
        t_end = np.mean([x["t_end"] for x in rs]); prey_eq = np.mean([x["prey_eq"] for x in rs])
        pred_eq = np.mean([x["pred_eq"] for x in rs]); cvp = np.nanmean([x["cv_prey"] for x in rs])
        minpt = np.mean([x["min_pred_tail"] for x in rs]); both_all = all(x["both"] for x in rs)
        is_stable = both_all and all(x["min_pred_tail"] > 0 for x in rs)
        L.append(f"{h:>4.0f} {n_pred:>6} {capR:>5.2f} {t_end:>7.1f} {prey_eq:>8.1f} {pred_eq:>8.1f} "
                 f"{minpt:>11.1f} {cvp:>8.3f} {str(both_all):>9} {str(is_stable):>7}")
        if is_stable:
            stable.append((h, n_pred, capR, prey_eq, pred_eq, cvp))
    L.append("")
    L.append(f"PERSISTENT STABLE coexistence cells (both alive t={HORIZON}, predator NEVER hit 0 in tail): {len(stable)}")
    for g in stable:
        L.append(f"    STABLE: h={g[0]} n_pred={g[1]} capR={g[2]} prey_eq={g[3]:.1f} pred_eq={g[4]:.1f} cv_prey={g[5]:.3f}")
    out = "\n".join(L)
    print(out)
    with open(os.path.join(_repo_root, "experiments", "outputs", "exp252_type3_coexistence.txt"), "w") as f:
        f.write(out + "\n")
    print("\n[written to experiments/outputs/exp252_type3_coexistence.txt]")


if __name__ == "__main__":
    main()
