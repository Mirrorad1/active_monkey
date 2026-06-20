"""experiments/exp251_latent_state.py — Exp 251: diagnose the LATENT STATE behind the
predator-prey boom-bust, and test the NATURAL/GENERAL fix (low-buffer demography) — NOT an
artificial refuge.

DIAGNOSIS HYPOTHESIS: Exp 250b showed that with decoupled prey growth the predator becomes
viable but the system PARADOX-OF-ENRICHMENT boom-busts (diverging oscillation -> collapse), with
no stable band. The latent cause is hypothesized to be the ENERGY-BUFFER LAG in the predator's
numerical response: the predator stores energy during the prey boom and keeps reproducing AFTER
the prey crash (delayed response) -> overshoot -> divergence. This is the predator-side twin of
the Exp-247 prey oscillation ("oscillation is buffer-driven, not artificially fixable").

NATURAL GENERAL FIX (not a refuge): shrink the predator's energy BUFFER (energy_capacity toward
its reproduction threshold) so its demography tracks CURRENT intake closely (near-instantaneous,
low-storage consumer — ecologically natural). PREDICTION: as the predator buffer shrinks, the
diverging oscillation should DAMP; at some low-buffer point a persistent bounded coexistence
(both roles alive at the horizon, NOT collapsing) should emerge WITHOUT any artificial protection.

PREDECLARED FALSIFIER: if shrinking the predator buffer across the full range never yields
persistent coexistence (every buffer either boom-bust collapses or starves the predator out),
then the boom-bust is NOT predator-buffer-lag-driven and low-buffer demography is not the fix.

RAW NUMBERS — controller judges. NOTE: a low predator buffer also lowers predator viability, so
expect a NON-MONOTONE response (too much buffer -> boom-bust; too little -> predator starves;
the hypothesis is a stable MIDDLE).
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
PRED_THRESHOLD = 4.2   # make_founder reproduction_energy_threshold; buffer = cap - threshold


def roles_alive(eco):
    snap = eco.alive_snapshot()
    return (sum(1 for c in snap if c.genotype.role == "prey"),
            sum(1 for c in snap if c.genotype.role == "predator"))


def build_cfg(K, birth_rate, n_pred, capR, pred_cap, prey_cap=10.0, sensing=2.5, assim=0.6):
    prey_geno = D.replace(make_founder(1.0), role="prey", energy_capacity=prey_cap)
    pred_geno = D.replace(make_founder(1.4), role="predator", energy_capacity=pred_cap)
    cfg = make_cfg(speed=1.0, cost_slope=0.0, regen_rate=1.0, horizon=HORIZON,
                   founder_mix=((prey_geno, 21), (pred_geno, n_pred)))
    return D.replace(
        cfg,
        enable_decoupled_prey_birth=True, prey_birth_rate=birth_rate,
        prey_carrying_capacity=float(K), enable_predation=True,
        freeze_prey_speed=True, mutate_predator_speed=False,
        capture_radius=capR, sensing_radius=sensing, assimilation_efficiency=assim,
        pred_start_energy_frac=0.75,
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
    tail_p, tail_q = prey_s[-300:], pred_s[-300:]
    cvp = float(np.std(tail_p) / np.mean(tail_p)) if np.mean(tail_p) > 0 else float("nan")
    cvq = float(np.std(tail_q) / np.mean(tail_q)) if np.mean(tail_q) > 0 else float("nan")
    return {"t_end": eco.t, "prey_final": p, "pred_final": q,
            "prey_eq": float(np.mean(tail_p)), "pred_eq": float(np.mean(tail_q)),
            "cv_prey": cvp, "cv_pred": cvq, "min_pred_tail": min(tail_q) if tail_q else 0,
            "both": (p > 0 and q > 0)}


def main():
    L = []
    L.append("=" * 104)
    L.append(f"Exp 251 — LATENT-STATE diagnostic: predator energy-BUFFER sweep in a boom-bust regime.")
    L.append(f"buffer = pred energy_capacity - threshold({PRED_THRESHOLD}); horizon={HORIZON} seeds={SEEDS}")
    L.append("hypothesis: shrinking the predator buffer (less numerical-response lag) DAMPS the boom-bust")
    L.append("=" * 104)

    # Two boom-bust regimes from Exp 250b, each x a predator-buffer sweep.
    regimes = [
        dict(K=250, birth_rate=0.5, n_pred=3, capR=0.45),
        dict(K=300, birth_rate=0.5, n_pred=4, capR=0.40),
    ]
    pred_caps = [10.0, 7.0, 5.5, 4.8, 4.4]  # buffer 5.8 -> 0.2 (toward instantaneous)

    L.append(f"{'K':>5} {'rate':>5} {'n_pred':>6} {'capR':>5} {'pred_cap':>8} {'buffer':>7} "
             f"{'t_end':>7} {'prey_eq':>8} {'pred_eq':>8} {'minPredTail':>11} {'cv_prey':>8} {'BOTH@end':>9}")
    L.append("-" * 104)
    persist = []
    for reg in regimes:
        for pc in pred_caps:
            cfg_kw = dict(reg, pred_cap=pc)
            rs = [run(build_cfg(**cfg_kw), s) for s in SEEDS]
            t_end = np.mean([x["t_end"] for x in rs]); prey_eq = np.mean([x["prey_eq"] for x in rs])
            pred_eq = np.mean([x["pred_eq"] for x in rs]); cvp = np.nanmean([x["cv_prey"] for x in rs])
            minpt = np.mean([x["min_pred_tail"] for x in rs]); both_all = all(x["both"] for x in rs)
            L.append(f"{reg['K']:>5} {reg['birth_rate']:>5.2f} {reg['n_pred']:>6} {reg['capR']:>5.2f} "
                     f"{pc:>8.1f} {pc - PRED_THRESHOLD:>7.1f} {t_end:>7.1f} {prey_eq:>8.1f} {pred_eq:>8.1f} "
                     f"{minpt:>11.1f} {cvp:>8.3f} {str(both_all):>9}")
            if both_all:
                persist.append((reg, pc, prey_eq, pred_eq, minpt, cvp))
        L.append("")

    L.append(f"PERSISTENT coexistence cells (both roles alive at t={HORIZON}, ALL seeds, min_pred_tail>0): "
             f"{sum(1 for x in persist if x[4] > 0)}")
    for g in persist:
        flag = "STABLE-ish" if g[4] > 0 else "(predator hit 0 in tail — fragile)"
        L.append(f"    K={g[0]['K']} rate={g[0]['birth_rate']} capR={g[0]['capR']} pred_cap={g[1]} "
                 f"prey_eq={g[2]:.1f} pred_eq={g[3]:.1f} min_pred_tail={g[4]:.1f} cv_prey={g[5]:.3f} {flag}")
    out = "\n".join(L)
    print(out)
    with open(os.path.join(_repo_root, "experiments", "outputs", "exp251_latent_state.txt"), "w") as f:
        f.write(out + "\n")
    print("\n[written to experiments/outputs/exp251_latent_state.txt]")


if __name__ == "__main__":
    main()
