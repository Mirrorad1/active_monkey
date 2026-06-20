"""experiments/exp254_bazykin_equilibrium.py — Exp 254: a doubly-self-limited (Bazykin)
predator-prey system — does predator self-limiting MORTALITY (caps NUMBER, not intake) yield a
ROBUST stable equilibrium where every intake-cutting stabilizer failed?

Exp 253c pinned the latent cause: every working stabilizer (Type III, interference, handling-time
saturation) caps the predator's INTAKE -> predator equilibrium intrinsically tiny/fragile. Exp 254
adds predator self-limiting mortality: predators eat FULLY (well-fed, can grow) but die faster at
high predator density -> caps predator NUMBER without cutting intake. Combined with decoupled prey
births (logistic prey), this is the doubly-logistic Bazykin model -> textbook ROBUST stable
coexistence. Type III + interference are turned OFF (simple full predation); max_captures lifted so
predators eat well; predator number controlled by mortality.

PREDICTION: some (K_P, hmax, max_captures) cell yields a ROBUST equilibrium — pred_eq >= 15, both
roles alive at t=1500 on ALL seeds, min_pred_tail>0, low cv_prey. This is the robust posable
predator population the Red Queen invasion test needs.

PREDECLARED FALSIFIER: if predator self-limit ALSO fails to produce a robust stable equilibrium
(predator still tiny/fragile, or boom-bust), then NO mechanism — intake-cutting OR number-cutting —
yields robust two-trophic coexistence on this substrate, and the wall is complete.

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


def build_cfg(Kp, hmax, max_cap, capR=0.5, K_prey=300, rate=0.5, n_pred=10,
              pred_cap=8.0, assim=0.7, sensing=2.5):
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
        enable_type3_response=False, enable_predator_interference=False,   # clean Bazykin: no intake cuts
        enable_predator_self_limit=True, predator_self_limit_kc=float(Kp),
        predator_self_limit_hmax=hmax, predator_self_limit_theta=1.0,
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
    L.append("=" * 112)
    L.append("Exp 254 — Bazykin (logistic prey + predator SELF-LIMIT mortality, full predation, no intake cuts).")
    L.append(f"fixed: rate=0.5 K_prey=300 pred_cap=8 capR=0.5 assim=0.7 n_pred=10; sweep K_P x hmax x max_cap; horizon={HORIZON} seeds={SEEDS}")
    L.append("=" * 112)
    L.append(f"{'K_P':>5} {'hmax':>5} {'maxcap':>6} {'t_end':>7} {'prey_eq':>8} {'pred_eq':>8} "
             f"{'minPredTail':>11} {'cv_prey':>8} {'BOTH@end':>9} {'STABLE':>7} {'ROBUST':>7}")
    L.append("-" * 112)
    robust = []
    for Kp, hmax, mc in itertools.product([20, 40], [0.1, 0.2], [2, 3]):
        rs = [run(build_cfg(Kp, hmax, mc), s) for s in SEEDS]
        t_end = np.mean([x["t_end"] for x in rs]); prey_eq = np.mean([x["prey_eq"] for x in rs])
        pred_eq = np.mean([x["pred_eq"] for x in rs]); cvp = np.nanmean([x["cv_prey"] for x in rs])
        minpt = np.mean([x["min_pred_tail"] for x in rs]); both_all = all(x["both"] for x in rs)
        is_stable = both_all and all(x["min_pred_tail"] > 0 for x in rs)
        is_robust = is_stable and pred_eq >= 15.0
        L.append(f"{Kp:>5} {hmax:>5.2f} {mc:>6} {t_end:>7.1f} {prey_eq:>8.1f} {pred_eq:>8.1f} "
                 f"{minpt:>11.1f} {cvp:>8.3f} {str(both_all):>9} {str(is_stable):>7} {str(is_robust):>7}")
        if is_robust:
            robust.append((Kp, hmax, mc, prey_eq, pred_eq, cvp))
    L.append("")
    L.append(f"ROBUST equilibria (both alive t={HORIZON} all seeds, min_pred_tail>0, pred_eq>=15): {len(robust)}")
    for g in robust:
        L.append(f"    ROBUST: K_P={g[0]} hmax={g[1]} max_cap={g[2]} prey_eq={g[3]:.1f} pred_eq={g[4]:.1f} cv_prey={g[5]:.3f}")
    out = "\n".join(L)
    print(out)
    with open(os.path.join(_repo_root, "experiments", "outputs", "exp254_bazykin_equilibrium.txt"), "w") as f:
        f.write(out + "\n")
    print("\n[written to experiments/outputs/exp254_bazykin_equilibrium.txt]")


if __name__ == "__main__":
    main()
