"""experiments/exp254c_scale_cycle.py — Exp 254c: is the predator-prey OSCILLATION a stable
(bounded) LIMIT CYCLE killed by small-N stochastic trough-extinction, or an intrinsically
DIVERGING oscillation? Scale the populations up and run long to decide.

Exp 254 gave a real predator-prey oscillation (pred_eq~19) that collapsed at t~600. A stable limit
cycle is valid — canonical — Red Queen coexistence. If the collapse is small-N stochastic trough-
extinction, SCALING the populations up (bigger K_prey + K_P, gentle hmax, more founders) should let
the cycle PERSIST with troughs that stay well above zero. If the oscillation is intrinsically
DIVERGENT, scaling only delays collapse.

DISCRIMINATOR: run to a long horizon (2000) and classify the oscillation amplitude trend
(cv of 2nd-half vs 1st-half of the predator series). BOUNDED (cv not growing) + reaches horizon +
min_pred_tail > 0 on all seeds => robust persistent cycle = posable Red Queen resident.

PREDICTION: at sufficient scale, a robust bounded cycle persists (both roles alive at t=2000, all
seeds, predator trough > 0, amplitude not diverging).

PREDECLARED FALSIFIER: if even at large scale the oscillation still collapses (predator trough
hits 0 / amplitude diverges) on all cells, the instability is intrinsic (divergent), not small-N
stochastic, and the two-trophic coexistence wall is complete in the dynamical (cycle) sense too.

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

HORIZON = 2000
SEEDS = [0, 1, 2]


def roles_alive(eco):
    snap = eco.alive_snapshot()
    return (sum(1 for c in snap if c.genotype.role == "prey"),
            sum(1 for c in snap if c.genotype.role == "predator"))


def build_cfg(K_prey, K_P, hmax, n_pred, capR=0.5, max_cap=2, rate=0.5, pred_cap=6.0,
              assim=0.7, sensing=2.5):
    prey_geno = D.replace(make_founder(1.0), role="prey")
    pred_geno = D.replace(make_founder(1.4), role="predator", energy_capacity=pred_cap)
    cfg = make_cfg(speed=1.0, cost_slope=0.0, regen_rate=1.0, horizon=HORIZON,
                   founder_mix=((prey_geno, 40), (pred_geno, n_pred)))
    return D.replace(
        cfg,
        enable_decoupled_prey_birth=True, prey_birth_rate=rate, prey_carrying_capacity=float(K_prey),
        enable_predation=True, freeze_prey_speed=True, mutate_predator_speed=False,
        capture_radius=capR, sensing_radius=sensing, assimilation_efficiency=assim,
        pred_start_energy_frac=0.75, max_captures_per_step=max_cap,
        enable_type3_response=False, enable_predator_interference=False,
        enable_predator_self_limit=True, predator_self_limit_kc=float(K_P),
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
    # amplitude-trend: cv of predator series 1st-half vs 2nd-half (after a 200-step burn-in)
    series = pred_s[200:] if len(pred_s) > 400 else pred_s
    half = len(series) // 2
    def _cv(xs):
        m = np.mean(xs)
        return float(np.std(xs) / m) if m > 0 else float("nan")
    cv1 = _cv(series[:half]) if half > 5 else float("nan")
    cv2 = _cv(series[half:]) if half > 5 else float("nan")
    tail_q = pred_s[-400:]
    return {"t_end": eco.t, "prey_final": p, "pred_final": q,
            "pred_eq": float(np.mean(tail_q)) if tail_q else 0.0,
            "min_pred_tail": min(tail_q) if tail_q else 0,
            "cv1": cv1, "cv2": cv2, "both": (p > 0 and q > 0)}


def main():
    L = []
    L.append("=" * 116)
    L.append("Exp 254c — SCALE-UP: is the predator-prey oscillation a robust BOUNDED cycle or intrinsically DIVERGENT?")
    L.append(f"decoupled prey + predator self-limit, full predation; sweep K_prey x K_P x hmax; horizon={HORIZON} seeds={SEEDS}")
    L.append("=" * 116)
    L.append(f"{'K_prey':>6} {'K_P':>5} {'hmax':>5} {'n_pred':>6} {'t_end':>7} {'pred_eq':>8} "
             f"{'minPredTail':>11} {'cv_1st':>7} {'cv_2nd':>7} {'trend':>9} {'BOTH@T':>7} {'CYCLE':>6}")
    L.append("-" * 116)
    cycles = []
    for K_prey, K_P, hmax in itertools.product([500, 1000], [60, 120], [0.06, 0.12]):
        n_pred = max(20, int(K_P * 0.5))
        rs = [run(build_cfg(K_prey, K_P, hmax, n_pred), s) for s in SEEDS]
        t_end = np.mean([x["t_end"] for x in rs]); pred_eq = np.mean([x["pred_eq"] for x in rs])
        minpt = np.mean([x["min_pred_tail"] for x in rs])
        cv1 = np.nanmean([x["cv1"] for x in rs]); cv2 = np.nanmean([x["cv2"] for x in rs])
        both_T = all(x["t_end"] >= HORIZON and x["both"] for x in rs)
        diverging = (cv2 > cv1 * 1.3) if (cv1 == cv1 and cv2 == cv2 and cv1 > 0) else True
        trend = "DIVERGE" if diverging else "bounded"
        # robust persistent cycle: reaches horizon, both alive, predator never hit 0, not diverging
        is_cycle = both_T and all(x["min_pred_tail"] > 0 for x in rs) and not diverging
        L.append(f"{K_prey:>6} {K_P:>5} {hmax:>5.2f} {n_pred:>6} {t_end:>7.1f} {pred_eq:>8.1f} "
                 f"{minpt:>11.1f} {cv1:>7.3f} {cv2:>7.3f} {trend:>9} {str(both_T):>7} {str(is_cycle):>6}")
        if is_cycle:
            cycles.append((K_prey, K_P, hmax, pred_eq, minpt, cv2))
    L.append("")
    L.append(f"ROBUST PERSISTENT CYCLES (reach t={HORIZON} all seeds, predator trough>0, amplitude not diverging): {len(cycles)}")
    for g in cycles:
        L.append(f"    CYCLE: K_prey={g[0]} K_P={g[1]} hmax={g[2]} pred_eq={g[3]:.1f} min_pred_tail={g[4]:.1f} cv2={g[5]:.3f}")
    out = "\n".join(L)
    print(out)
    with open(os.path.join(_repo_root, "experiments", "outputs", "exp254c_scale_cycle.txt"), "w") as f:
        f.write(out + "\n")
    print("\n[written to experiments/outputs/exp254c_scale_cycle.txt]")


if __name__ == "__main__":
    main()
