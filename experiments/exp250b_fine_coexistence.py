"""experiments/exp250b_fine_coexistence.py — Exp 250b: FINE transition-zone sweep.

HYPOTHESIS / PREDICTION: Exp 250 (additive decoupled prey births) moved the failure mode from
"predator starves" (capR=0.3) to paradox-of-enrichment "boom-bust collapse" (capR=0.5), with the
longest survivors (rate0.5,K250: t_end~463-518) NEARLY reaching the horizon. The coarse grid
{0.3,0.5} STRADDLED the likely stable band. PREDICTION: a finer capR sweep through the transition
zone (~0.35-0.45) at higher K, run to a LONGER horizon, will reveal a persistent coexistence cell
(both roles alive at t=1000) — a stable predator-prey equilibrium or bounded limit cycle.

PREDECLARED FALSIFIER: if NO cell in the fine transition-zone grid keeps BOTH roles alive at
t=1000 on all seeds (every cell still either starves the predator out or collapses before t=1000),
then even decoupled prey growth yields only UNSTABLE (boom-bust) two-trophic dynamics on this
substrate — no stable coexistence band — and the spatial refuge is the last lever before closing.

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

HORIZON = 1000
SEEDS = [0, 1, 2]


def roles_alive(eco):
    snap = eco.alive_snapshot()
    return (sum(1 for c in snap if c.genotype.role == "prey"),
            sum(1 for c in snap if c.genotype.role == "predator"))


def build_cfg(n_prey, n_pred, K, birth_rate, capR, sensing=2.5, assim=0.6):
    prey_geno = D.replace(make_founder(1.0), role="prey")
    pred_geno = D.replace(make_founder(1.4), role="predator")
    cfg = make_cfg(speed=1.0, cost_slope=0.0, regen_rate=1.0, horizon=HORIZON,
                   founder_mix=((prey_geno, n_prey), (pred_geno, n_pred)))
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
    tail_p, tail_q = prey_s[-200:], pred_s[-200:]
    cvp = float(np.std(tail_p) / np.mean(tail_p)) if np.mean(tail_p) > 0 else float("nan")
    min_pred_tail = min(tail_q) if tail_q else 0
    return {"t_end": eco.t, "prey_final": p, "pred_final": q,
            "prey_eq": float(np.mean(tail_p)), "pred_eq": float(np.mean(tail_q)),
            "cv_prey": cvp, "min_pred_tail": min_pred_tail, "both": (p > 0 and q > 0)}


def main():
    L = []
    L.append("=" * 100)
    L.append(f"Exp 250b — FINE transition-zone coexistence sweep (decoupled births). horizon={HORIZON} seeds={SEEDS}")
    L.append("=" * 100)
    L.append(f"{'rate':>5} {'K':>5} {'n_pred':>6} {'capR':>5} {'t_end':>7} {'prey_eq':>8} {'pred_eq':>8} "
             f"{'minPredTail':>11} {'cv_prey':>8} {'BOTH@end':>9}")
    L.append("-" * 100)
    coexist = []
    for rate, K, n_pred, capR in itertools.product([0.4, 0.5], [200, 300], [3, 4], [0.35, 0.40, 0.45]):
        rs = [run(build_cfg(21, n_pred, K, rate, capR), s) for s in SEEDS]
        t_end = np.mean([x["t_end"] for x in rs]); prey_eq = np.mean([x["prey_eq"] for x in rs])
        pred_eq = np.mean([x["pred_eq"] for x in rs]); cvp = np.nanmean([x["cv_prey"] for x in rs])
        minpt = np.mean([x["min_pred_tail"] for x in rs]); both_all = all(x["both"] for x in rs)
        L.append(f"{rate:>5.2f} {K:>5} {n_pred:>6} {capR:>5.2f} {t_end:>7.1f} {prey_eq:>8.1f} {pred_eq:>8.1f} "
                 f"{minpt:>11.1f} {cvp:>8.3f} {str(both_all):>9}")
        if both_all:
            coexist.append((rate, K, n_pred, capR, prey_eq, pred_eq, minpt, cvp))
    L.append("")
    L.append(f"COEXISTENCE regimes (both roles alive at t={HORIZON}, ALL seeds): {len(coexist)}")
    for g in coexist:
        L.append(f"    COEXIST: rate={g[0]} K={g[1]} n_pred={g[2]} capR={g[3]} prey_eq={g[4]:.1f} "
                 f"pred_eq={g[5]:.1f} min_pred_tail={g[6]:.1f} cv_prey={g[7]:.3f}")
    out = "\n".join(L)
    print(out)
    with open(os.path.join(_repo_root, "experiments", "outputs", "exp250b_fine_coexistence.txt"), "w") as f:
        f.write(out + "\n")
    print("\n[written to experiments/outputs/exp250b_fine_coexistence.txt]")


if __name__ == "__main__":
    main()
