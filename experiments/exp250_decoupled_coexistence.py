"""experiments/exp250_decoupled_coexistence.py — Exp 250: does ADDITIVE resource-independent
prey growth enable two-trophic predator-prey COEXISTENCE?

HYPOTHESIS / PREDICTION: Exp 248 (no decoupling) and Exp 249 (logistic SUPPRESSION) both gave
0 coexistence — predators starve out or boom-bust collapse. The binding constraint is prey
PRODUCTIVITY at low density (suppression only caps the top; it cannot raise low-density growth,
which is resource-refill-limited). Exp 250 adds `enable_decoupled_prey_birth`: a resource-
INDEPENDENT logistic birth stream (prob prey_birth_rate*(1-N/K) per prey/step, child energy free)
that RAISES low-density prey productivity decoupled from the depletable resource (textbook
Rosenzweig-MacArthur logistic prey). PREDICTION: there exists a (prey_birth_rate, K, predation)
band with persistent two-trophic coexistence.

PREDECLARED FALSIFIER: if NO cell across the prey_birth_rate x K x predation sweep gives
persistent coexistence (both roles alive at the horizon on ALL seeds) — i.e. every cell either
starves the predator out or boom-bust collapses — then decoupled prey growth is NOT the escape
either, and the spatial refuge is the last lever before closing the Red Queen direction.

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
HORIZON = 600
SEEDS = [0, 1, 2]


def roles_alive(eco):
    snap = eco.alive_snapshot()
    return (sum(1 for c in snap if c.genotype.role == "prey"),
            sum(1 for c in snap if c.genotype.role == "predator"))


def build_cfg(n_prey, n_pred, K, birth_rate, capR, sensing=2.5, assim=0.6, enable_pred=True):
    prey_geno = D.replace(make_founder(1.0), role="prey")
    pred_geno = D.replace(make_founder(1.4), role="predator")
    mix = ((prey_geno, n_prey), (pred_geno, n_pred)) if enable_pred else ((prey_geno, n_prey),)
    cfg = make_cfg(speed=1.0, cost_slope=0.0, regen_rate=REGEN_RATE, horizon=HORIZON, founder_mix=mix)
    return D.replace(
        cfg,
        enable_decoupled_prey_birth=True,
        prey_birth_rate=birth_rate,
        prey_carrying_capacity=float(K),
        enable_predation=enable_pred,
        freeze_prey_speed=True,
        mutate_predator_speed=False,
        capture_radius=capR,
        sensing_radius=sensing,
        assimilation_efficiency=assim,
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
    tail_p, tail_q = prey_s[-150:], pred_s[-150:]
    cvp = float(np.std(tail_p) / np.mean(tail_p)) if np.mean(tail_p) > 0 else float("nan")
    return {"t_end": eco.t, "prey_final": p, "pred_final": q,
            "prey_eq": float(np.mean(tail_p)), "pred_eq": float(np.mean(tail_q)),
            "cv_prey": cvp, "exploded": eco.exploded, "both": (p > 0 and q > 0)}


def main():
    L = []
    L.append("=" * 100)
    L.append("Exp 250 — does ADDITIVE resource-independent prey growth enable coexistence? RAW.")
    L.append(f"prey base: Exp-240 viable founder + enable_decoupled_prey_birth; regen={REGEN_RATE} horizon={HORIZON} seeds={SEEDS}")
    L.append("=" * 100)

    L.append("(A) PREY-ONLY with decoupled births ON (sanity: persists, bounded near K):")
    for K, r in ((120, 0.3), (250, 0.5)):
        rs = [run(build_cfg(10, 0, K, r, 0.3, enable_pred=False), s) for s in SEEDS]
        L.append(f"    K={K:>4} r={r}: t_end={np.mean([x['t_end'] for x in rs]):.0f} "
                 f"prey_eq={np.mean([x['prey_eq'] for x in rs]):.1f} "
                 f"exploded={any(x['exploded'] for x in rs)}")
    L.append("")

    L.append("(B) TWO-TROPHIC coexistence sweep (decoupled prey births + predator):")
    L.append(f"{'rate':>5} {'K':>5} {'n_pred':>6} {'capR':>5} {'t_end':>7} {'prey_eq':>8} {'pred_eq':>8} "
             f"{'cv_prey':>8} {'BOTH@end':>9} {'explode':>8}")
    L.append("-" * 100)
    coexist = []
    for r, K, n_pred, capR in itertools.product([0.2, 0.35, 0.5], [120, 250], [3, 6], [0.3, 0.5]):
        rs = [run(build_cfg(21, n_pred, K, r, capR), s) for s in SEEDS]
        t_end = np.mean([x["t_end"] for x in rs]); prey_eq = np.mean([x["prey_eq"] for x in rs])
        pred_eq = np.mean([x["pred_eq"] for x in rs]); cvp = np.nanmean([x["cv_prey"] for x in rs])
        both_all = all(x["both"] for x in rs); explode = any(x["exploded"] for x in rs)
        L.append(f"{r:>5.2f} {K:>5} {n_pred:>6} {capR:>5.2f} {t_end:>7.1f} {prey_eq:>8.1f} {pred_eq:>8.1f} "
                 f"{cvp:>8.3f} {str(both_all):>9} {str(explode):>8}")
        if both_all and not explode:
            coexist.append((r, K, n_pred, capR, prey_eq, pred_eq, cvp))
    L.append("")
    L.append(f"COEXISTENCE regimes (both roles alive at t={HORIZON}, ALL seeds): {len(coexist)}")
    for g in coexist:
        L.append(f"    COEXIST: rate={g[0]} K={g[1]} n_pred={g[2]} capR={g[3]} "
                 f"prey_eq={g[4]:.1f} pred_eq={g[5]:.1f} cv_prey={g[6]:.3f}")
    out = "\n".join(L)
    print(out)
    out_dir = os.path.join(_repo_root, "experiments", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp250_decoupled_coexistence.txt"), "w") as f:
        f.write(out + "\n")
    print(f"\n[written to experiments/outputs/exp250_decoupled_coexistence.txt]")


if __name__ == "__main__":
    main()
