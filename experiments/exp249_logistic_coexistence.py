"""experiments/exp249_logistic_coexistence.py — Exp 249 Rung 2: does decoupled (logistic)
prey growth enable two-trophic predator-prey COEXISTENCE that Exp 248 could not reach?

Exp 248 CAN'T-POSE: on the Exp-240 viable prey base, NO predation regime gave persistent
two-trophic coexistence (0/42+1900 runs) — predators either starve out or boom-bust collapse.
Mechanism: the prey's only density regulation is via the LAGGY energy/resource buffer, so a
predator boom overshoots and crashes the prey before regulation bites.

Exp 249 adds `enable_logistic_prey_growth`: an INSTANTANEOUS, lag-free density-dependent
suppression of prey births (reproduce only if rng < max(0, 1 - N_prey/K)). This gives prey a
resource-independent carrying capacity K with NO buffer lag — the Rosenzweig-MacArthur
stabilizer. PRE-REGISTERED FALSIFIER: if NO (K_prey, predation-pressure) cell gives persistent
coexistence (both roles alive at the horizon on all seeds), logistic prey growth is NOT the
escape -> try the spatial refuge next. If some cell coexists -> the Red Queen invasion test
becomes posable; proceed to expressibility + the co-evolving-vs-static invasion.

RAW NUMBERS — controller judges go/abort.
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


def _load(mod_name, rel):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_repo_root, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_e242 = _load("exp242_regulated_ess", "experiments/exp242_regulated_ess.py")
make_founder = _e242._make_founder
make_cfg = _e242._make_cfg
REGEN_RATE = 1.0
COST_SLOPE = 0.0
HORIZON = 600
SEEDS = [0, 1, 2]


def roles_alive(eco):
    snap = eco.alive_snapshot()
    p = sum(1 for c in snap if c.genotype.role == "prey")
    q = sum(1 for c in snap if c.genotype.role == "predator")
    return p, q


def build_cfg(n_prey, n_pred, K_prey, capture_radius, sensing_radius=2.5, assimilation=0.6,
              enable_pred=True):
    prey_geno = D.replace(make_founder(1.0), role="prey")
    pred_geno = D.replace(make_founder(1.4), role="predator")
    mix = ((prey_geno, n_prey), (pred_geno, n_pred)) if enable_pred else ((prey_geno, n_prey),)
    cfg = make_cfg(speed=1.0, cost_slope=COST_SLOPE, regen_rate=REGEN_RATE,
                   horizon=HORIZON, founder_mix=mix)
    return D.replace(
        cfg,
        enable_logistic_prey_growth=True,
        prey_carrying_capacity=float(K_prey),
        enable_predation=enable_pred,
        freeze_prey_speed=True,
        mutate_predator_speed=False,
        capture_radius=capture_radius,
        sensing_radius=sensing_radius,
        assimilation_efficiency=assimilation,
        pred_start_energy_frac=0.75,
    )


def run(cfg, seed):
    eco = Ecology(cfg, seed=seed)
    prey_s, pred_s = [], []
    while eco.has_alive() and not eco.exploded and eco.t < cfg.horizon:
        p, q = roles_alive(eco)
        prey_s.append(p)
        pred_s.append(q)
        eco.step()
    p, q = roles_alive(eco)
    prey_s.append(p)
    pred_s.append(q)
    tail_p = prey_s[max(0, len(prey_s) - 150):]
    tail_q = pred_s[max(0, len(pred_s) - 150):]
    cvp = float(np.std(tail_p) / np.mean(tail_p)) if tail_p and np.mean(tail_p) > 0 else float("nan")
    return {"t_end": eco.t, "prey_final": p, "pred_final": q,
            "prey_eq": float(np.mean(tail_p)), "pred_eq": float(np.mean(tail_q)),
            "cv_prey": cvp, "exploded": eco.exploded, "both": (p > 0 and q > 0)}


def main():
    lines = []
    lines.append("=" * 98)
    lines.append("Exp 249 — does LOGISTIC prey growth enable two-trophic coexistence? RAW — controller judges.")
    lines.append(f"prey base: Exp-240 viable founder + enable_logistic_prey_growth; regen={REGEN_RATE} horizon={HORIZON} seeds={SEEDS}")
    lines.append("=" * 98)

    # (A) sanity: logistic prey-only persists and caps near K
    lines.append("(A) PREY-ONLY with logistic ON (must persist; should cap near K):")
    for K in (60, 150):
        cfg = build_cfg(21, 0, K, 0.4, enable_pred=False)
        rs = [run(cfg, s) for s in SEEDS]
        lines.append(f"    K={K:>4}: t_end={np.mean([r['t_end'] for r in rs]):.0f} "
                     f"prey_eq={np.mean([r['prey_eq'] for r in rs]):.1f} "
                     f"prey_final={np.mean([r['prey_final'] for r in rs]):.1f}")
    lines.append("")

    # (B) two-trophic coexistence sweep: K_prey x predation pressure
    lines.append("(B) TWO-TROPHIC coexistence sweep (logistic prey + predator):")
    lines.append(f"{'K_prey':>6} {'n_pred':>6} {'capR':>5} {'t_end':>7} {'prey_eq':>8} {'pred_eq':>8} "
                 f"{'cv_prey':>8} {'BOTH@end':>9} {'explode':>8}")
    lines.append("-" * 98)
    Ks = [60, 100, 150, 250]
    n_preds = [2, 4, 6]
    capRs = [0.2, 0.35, 0.5]
    coexist = []
    for K, n_pred, capR in itertools.product(Ks, n_preds, capRs):
        cfg = build_cfg(21, n_pred, K, capR)
        rs = [run(cfg, s) for s in SEEDS]
        t_end = np.mean([r["t_end"] for r in rs])
        prey_eq = np.mean([r["prey_eq"] for r in rs])
        pred_eq = np.mean([r["pred_eq"] for r in rs])
        cvp = np.nanmean([r["cv_prey"] for r in rs])
        both_all = all(r["both"] for r in rs)
        explode = any(r["exploded"] for r in rs)
        lines.append(
            f"{K:>6} {n_pred:>6} {capR:>5.2f} {t_end:>7.1f} {prey_eq:>8.1f} {pred_eq:>8.1f} "
            f"{cvp:>8.3f} {str(both_all):>9} {str(explode):>8}"
        )
        if both_all and not explode:
            coexist.append((K, n_pred, capR, prey_eq, pred_eq, cvp))
    lines.append("")
    lines.append(f"COEXISTENCE regimes (both roles alive at t={HORIZON}, ALL seeds): {len(coexist)}")
    for g in coexist:
        lines.append(f"    COEXIST: K={g[0]} n_pred={g[1]} capR={g[2]} prey_eq={g[3]:.1f} "
                     f"pred_eq={g[4]:.1f} cv_prey={g[5]:.3f}")
    out = "\n".join(lines)
    print(out)
    out_dir = os.path.join(_repo_root, "experiments", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp249_logistic_coexistence.txt"), "w") as f:
        f.write(out + "\n")
    print(f"\n[written to experiments/outputs/exp249_logistic_coexistence.txt]")


if __name__ == "__main__":
    main()
