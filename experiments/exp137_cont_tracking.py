"""
Exp 137 — continuous-substrate rung 5: non-stationary tracking (NIW-as-learning-rate
vs a fixed-learning-rate baseline; the Exp 88 window-law cross-link).

Hypothesis (worked out before any code): NIW with per-step precision forgetting at
decay lam IS an exponential moving average with rate ~(1-lam) at steady state, so on
constant-velocity drift the best forgetting setting TIES the best fixed learning rate
(same estimator family) — while at v=0 the unforgetting conjugate annealer (lr ~ 1/t)
beats EVERY fixed rate. The tracking-error-vs-window curve at v>0 is U-shaped (the
Exp 88 forgetting window, on a new substrate: noise floor below, lag ceiling above)
with optimum at the EMA ramp-tracking law N* = (sigma^2 / (2 v^2))^(1/3) — predicted
N* ~ 62.6 / 24.8 / 9.9 for v = 0.0005 / 0.002 / 0.008. The card's FAIL clause ("no
hyperparameter setting beats a fixed-learning-rate baseline") means LOSING; the
predicted v>0 tie plus the v=0 win is the characterized NIW-as-learning-rate verdict,
not a FAIL.

Setup: d=2; one emission Gaussian whose true center drifts mu_true(t) = v*t*(1,0);
data x_t = mu_true(t) + N(0, 0.35^2 I); T = 4000 steps, RMS tracking error
||m_t - mu_true(t)|| over the last 3000 (burn-in 1000); v in {0, 0.0005, 0.002,
0.008}; NIW agents: N_eff in {5,10,20,40,80,160,320} via lam = 1 - 1/N_eff, plus
lam = 1 (no forgetting); NIW prior m0=(0,0), kappa0=1, nu0=4, S0=0.35^2*(nu0-d-1)*I,
decayed toward this prior each step (update then decay). Fixed-lr baselines:
m_t = m_{t-1} + alpha*(x_t - m_{t-1}), alpha = 1/N_eff over the same N_eff grid,
m_0 = (0,0). Seeds 0..7 per (v, agent-setting) cell.

Predictions (TRUE iff all):
- P1 two-regime: at v=0, no-forgetting NIW has lower RMS than EVERY fixed-lr alpha
  in >= 7/8 seeds (per alpha); at every v>0, no-forgetting RMS >= 3x the best-lam RMS
  (cell means).
- P2 steady-state tie: at every v>0, best-lam NIW cell-mean RMS within +-15% of
  best-alpha fixed-lr cell-mean RMS.
- P3 window law: at every v>0 the RMS-vs-N_eff curve (lam agents, cell means) is
  U-shaped with strictly interior argmin over the finite grid, argmin N* within a
  factor of 2.5 of (sigma^2/(2 v^2))^(1/3), and N* non-increasing in v across the
  three v>0 levels.

Falsifier (any triggers NEGATIVE): NIW-forgetting LOSES to fixed-lr by > 15% at any
v>0 (the card's FAIL clause — log it as such), OR the v=0 conjugate advantage is
absent (P1a fails), OR no U-shape (argmin at a grid edge at any v>0), OR N* off by
> 2.5x or increasing in v. A tie within +-15% at v>0 plus a v=0 win is the
predicted POSITIVE, per the hypothesis paragraph.
"""
from __future__ import annotations

import json
import os
import time
import numpy as np

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

D = 2
SIGMA = 0.35
T = 4000
BURN = 1000
EVAL_LEN = T - BURN  # 3000

V_LIST = [0.0, 0.0005, 0.002, 0.008]
N_EFF_GRID = [5, 10, 20, 40, 80, 160, 320]
N_SEEDS = 8

KAPPA0 = 1.0
NU0 = float(D + 2)  # = 4.0
M0 = np.zeros(D)
S0 = SIGMA ** 2 * (NU0 - D - 1) * np.eye(D)  # = 0.35^2 * 1 * I

# Predicted N* via ramp-tracking law: (sigma^2 / (2 v^2))^(1/3)
def predicted_nstar(v):
    if v == 0:
        return float("inf")
    return (SIGMA ** 2 / (2.0 * v ** 2)) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# Inline NIW update+decay arithmetic (same formulas as NIW class; used for speed)
# The NIW class and its tests remain the canonical reference.
# State: kappa (float), nu (float), m (array d), S (array d x d)
# ---------------------------------------------------------------------------

def niw_update_inline(kappa, nu, m, S, x):
    """NIW update for a single observation x (same formula as NIW.update)."""
    kappa_new = kappa + 1.0
    nu_new    = nu + 1.0
    m_new     = (kappa * m + x) / kappa_new
    diff      = x - m
    S_new     = S + (kappa / kappa_new) * np.outer(diff, diff)
    return kappa_new, nu_new, m_new, S_new


def niw_decay_inline(kappa, nu, m, S, lam, pk, pnu, pm, pS):
    """NIW decay toward prior (pk, pnu, pm, pS) (same formula as NIW.decay)."""
    kappa_new = pk + lam * (kappa - pk)
    nu_new    = pnu + lam * (nu - pnu)
    m_new     = (lam * kappa * m + (1.0 - lam) * pk * pm) / kappa_new
    S_new     = pS + lam * (S - pS)
    return kappa_new, nu_new, m_new, S_new


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    t0 = time.time()
    rows = []

    # Prior scalars for inline decay
    pk, pnu, pm, pS = KAPPA0, NU0, M0.copy(), S0.copy()

    for vi, v in enumerate(V_LIST):
        # Precompute x_t streams for all seeds (shared across all agents)
        # rng = np.random.default_rng(seed * 40 + vi)
        streams = {}
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed * 40 + vi)
            mu_true = v * np.arange(T)[:, None] * np.array([1.0, 0.0])  # (T, 2)
            noise = rng.standard_normal((T, D)) * SIGMA
            streams[seed] = mu_true + noise  # (T, 2)

        # ---- NIW agents ----
        # N_eff grid + no-forgetting (lam=1)
        niw_settings = [(n, 1.0 - 1.0 / n) for n in N_EFF_GRID] + [("inf", 1.0)]

        for (neff, lam) in niw_settings:
            for seed in range(N_SEEDS):
                xs = streams[seed]
                # Inline state
                kappa, nu, m, S = pk, pnu, pm.copy(), pS.copy()

                sq_errs = []
                for t_idx in range(T):
                    x_t = xs[t_idx]
                    # update then decay
                    kappa, nu, m, S = niw_update_inline(kappa, nu, m, S, x_t)
                    if lam < 1.0:
                        kappa, nu, m, S = niw_decay_inline(kappa, nu, m, S, lam, pk, pnu, pm, pS)
                    # track after-decay mean
                    if t_idx >= BURN:
                        mu_true_t = v * t_idx * np.array([1.0, 0.0])
                        sq_errs.append(np.sum((m - mu_true_t) ** 2))

                rms = float(np.sqrt(np.mean(sq_errs)))
                rows.append({
                    "exp": 137, "rung": 5, "agent": "niw", "seed": seed,
                    "step": T, "metric": "rms", "value": rms,
                    "params": {"v": v, "N_eff": neff}
                })

        # ---- Fixed-lr baselines ----
        for n_eff in N_EFF_GRID:
            alpha = 1.0 / n_eff
            for seed in range(N_SEEDS):
                xs = streams[seed]
                m_lr = np.zeros(D)
                sq_errs = []
                for t_idx in range(T):
                    x_t = xs[t_idx]
                    m_lr = m_lr + alpha * (x_t - m_lr)
                    if t_idx >= BURN:
                        mu_true_t = v * t_idx * np.array([1.0, 0.0])
                        sq_errs.append(np.sum((m_lr - mu_true_t) ** 2))
                rms = float(np.sqrt(np.mean(sq_errs)))
                rows.append({
                    "exp": 137, "rung": 5, "agent": "fixed_lr", "seed": seed,
                    "step": T, "metric": "rms", "value": rms,
                    "params": {"v": v, "N_eff": n_eff}
                })

    elapsed = time.time() - t0
    print(f"Simulation wall time: {elapsed:.1f}s\n")
    return rows


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(rows):
    import collections

    # Organize by (v, agent, neff) -> list of rms values
    cells = collections.defaultdict(list)
    for r in rows:
        key = (r["params"]["v"], r["agent"], r["params"]["N_eff"])
        cells[key].append(r["value"])

    def cell_mean(v, agent, neff):
        return float(np.mean(cells[(v, agent, neff)]))

    # ---- Per-v tables ----
    for v in V_LIST:
        print(f"=== v = {v} ===")
        print(f"{'N_eff':>8}  {'NIW cell-mean RMS':>20}  {'FixedLR cell-mean RMS':>22}")
        # N_eff grid rows
        for n in N_EFF_GRID:
            niw_rms  = cell_mean(v, "niw",      n)
            lr_rms   = cell_mean(v, "fixed_lr", n)
            print(f"{n:>8}  {niw_rms:>20.5f}  {lr_rms:>22.5f}")
        # No-forgetting row (NIW only)
        nf_rms = cell_mean(v, "niw", "inf")
        print(f"{'inf(no-fgt)':>8}  {nf_rms:>20.5f}  {'n/a':>22}")
        print()

    # ---- Per-v summary ----
    for v in V_LIST:
        niw_rms_by_n  = {n: cell_mean(v, "niw",      n) for n in N_EFF_GRID}
        lr_rms_by_n   = {n: cell_mean(v, "fixed_lr", n) for n in N_EFF_GRID}
        nf_rms        = cell_mean(v, "niw", "inf")

        best_n_niw  = min(N_EFF_GRID, key=lambda n: niw_rms_by_n[n])
        best_rms_niw = niw_rms_by_n[best_n_niw]
        best_n_lr    = min(N_EFF_GRID, key=lambda n: lr_rms_by_n[n])
        best_rms_lr  = lr_rms_by_n[best_n_lr]

        ratio = best_rms_niw / best_rms_lr if best_rms_lr > 0 else float("nan")
        pstar = predicted_nstar(v)

        # U-shape check: argmin is strictly interior (not first or last grid point)
        niw_curve = [niw_rms_by_n[n] for n in N_EFF_GRID]
        argmin_idx = int(np.argmin(niw_curve))
        ushape_interior = (argmin_idx > 0) and (argmin_idx < len(N_EFF_GRID) - 1)

        print(f"v={v}: best-lam N_eff={best_n_niw} RMS={best_rms_niw:.5f} | "
              f"best-alpha N_eff={best_n_lr} RMS={best_rms_lr:.5f} | "
              f"ratio(NIW/LR)={ratio:.3f} | "
              f"pred N*={pstar:.1f} | obs N*={best_n_niw} | "
              f"U-shape interior={'yes' if ushape_interior else 'no'}")
    print()

    # ---- P1, P2, P3 tallies ----
    # P1a: v=0, no-forgetting NIW < every fixed-lr alpha, in >= 7/8 seeds
    v0 = 0.0
    p1a_per_alpha = {}
    for n in N_EFF_GRID:
        niw_seeds  = cells[(v0, "niw",      "inf")]
        lr_seeds   = cells[(v0, "fixed_lr", n)]
        wins = sum(1 for a, b in zip(niw_seeds, lr_seeds) if a < b)
        p1a_per_alpha[n] = wins
    p1a_pass = all(w >= 7 for w in p1a_per_alpha.values())
    print("P1a (v=0, no-fgt NIW < every fixed-lr in >=7/8 seeds):")
    for n, w in p1a_per_alpha.items():
        print(f"  alpha=1/{n}: {w}/8 seeds NIW wins  {'OK' if w >= 7 else 'FAIL'}")
    print(f"  P1a overall: {'PASS' if p1a_pass else 'FAIL'}")
    print()

    # P1b: at every v>0, no-forgetting RMS >= 3x best-lam RMS (cell means)
    p1b_results = {}
    for v in V_LIST:
        if v == 0.0:
            continue
        nf_rms = cell_mean(v, "niw", "inf")
        best_lam_rms = min(cell_mean(v, "niw", n) for n in N_EFF_GRID)
        ratio_1b = nf_rms / best_lam_rms if best_lam_rms > 0 else float("nan")
        p1b_results[v] = (ratio_1b, ratio_1b >= 3.0)
        print(f"P1b v={v}: no-fgt RMS={nf_rms:.5f}, best-lam RMS={best_lam_rms:.5f}, "
              f"ratio={ratio_1b:.3f}  {'OK' if ratio_1b >= 3.0 else 'FAIL (< 3x)'}")
    p1b_pass = all(ok for _, ok in p1b_results.values())
    print(f"  P1b overall: {'PASS' if p1b_pass else 'FAIL'}")
    print()

    p1_pass = p1a_pass and p1b_pass

    # P2: at every v>0, best-lam NIW within +-15% of best-alpha LR
    p2_results = {}
    print("P2 (v>0, best-lam NIW within +-15% of best-alpha LR):")
    for v in V_LIST:
        if v == 0.0:
            continue
        niw_rms_by_n = {n: cell_mean(v, "niw", n) for n in N_EFF_GRID}
        lr_rms_by_n  = {n: cell_mean(v, "fixed_lr", n) for n in N_EFF_GRID}
        best_rms_niw = min(niw_rms_by_n.values())
        best_rms_lr  = min(lr_rms_by_n.values())
        ratio = best_rms_niw / best_rms_lr if best_rms_lr > 0 else float("nan")
        ok = 0.85 <= ratio <= 1.15
        p2_results[v] = ok
        print(f"  v={v}: NIW={best_rms_niw:.5f}, LR={best_rms_lr:.5f}, ratio={ratio:.4f}  "
              f"{'OK' if ok else 'FAIL (> 15% gap)'}")
    p2_pass = all(p2_results.values())
    print(f"  P2 overall: {'PASS' if p2_pass else 'FAIL'}")
    print()

    # P3: U-shaped, strictly interior argmin, N* within 2.5x predicted, N* non-increasing in v
    print("P3 (window law: U-shaped, interior argmin, N* within 2.5x pred, non-increasing in v):")
    p3_nstar_obs = {}
    p3_ushape = {}
    p3_within_2p5x = {}
    for v in V_LIST:
        if v == 0.0:
            continue
        niw_curve = [cell_mean(v, "niw", n) for n in N_EFF_GRID]
        argmin_idx = int(np.argmin(niw_curve))
        ushape_interior = (argmin_idx > 0) and (argmin_idx < len(N_EFF_GRID) - 1)
        obs_nstar = N_EFF_GRID[argmin_idx]
        pred_nstar = predicted_nstar(v)
        ratio_star = obs_nstar / pred_nstar
        within = (ratio_star <= 2.5) and (ratio_star >= 1.0 / 2.5)
        p3_nstar_obs[v] = obs_nstar
        p3_ushape[v] = ushape_interior
        p3_within_2p5x[v] = within
        print(f"  v={v}: argmin_idx={argmin_idx} N*={obs_nstar}, pred={pred_nstar:.1f}, "
              f"ratio={ratio_star:.2f}  U-interior={'yes' if ushape_interior else 'no'}  "
              f"within2.5x={'yes' if within else 'no'}")

    # N* non-increasing in v
    v_pos = [v for v in V_LIST if v > 0]
    nstar_vals = [p3_nstar_obs[v] for v in v_pos]
    nonincreasing = all(nstar_vals[i] >= nstar_vals[i+1] for i in range(len(nstar_vals)-1))
    print(f"  N* sequence across v>0: {nstar_vals}  non-increasing={'yes' if nonincreasing else 'no'}")

    p3_pass = (
        all(p3_ushape.values()) and
        all(p3_within_2p5x.values()) and
        nonincreasing
    )
    print(f"  P3 overall: {'PASS' if p3_pass else 'FAIL'}")
    print()

    # ---- Falsifier check (per spec) ----
    falsifiers = []

    # Falsifier 1: NIW-forgetting LOSES to fixed-lr by > 15% at any v>0
    for v in V_LIST:
        if v == 0.0:
            continue
        niw_rms_by_n = {n: cell_mean(v, "niw", n) for n in N_EFF_GRID}
        lr_rms_by_n  = {n: cell_mean(v, "fixed_lr", n) for n in N_EFF_GRID}
        best_rms_niw = min(niw_rms_by_n.values())
        best_rms_lr  = min(lr_rms_by_n.values())
        ratio = best_rms_niw / best_rms_lr if best_rms_lr > 0 else float("nan")
        if ratio > 1.15:
            falsifiers.append(f"NIW loses to fixed-lr by {(ratio-1)*100:.1f}% at v={v} (FAIL clause)")

    # Falsifier 2: P1a fails
    if not p1a_pass:
        falsifiers.append("v=0 conjugate advantage absent (P1a fails)")

    # Falsifier 3: no U-shape (argmin at grid edge at any v>0)
    for v in V_LIST:
        if v == 0.0:
            continue
        if not p3_ushape[v]:
            falsifiers.append(f"No U-shape at v={v} (argmin at grid edge)")

    # Falsifier 4: N* off by > 2.5x
    for v in V_LIST:
        if v == 0.0:
            continue
        if not p3_within_2p5x[v]:
            falsifiers.append(f"N* off by > 2.5x at v={v}")

    # Falsifier 5: N* increasing in v
    if not nonincreasing:
        falsifiers.append(f"N* increasing in v: {list(zip(v_pos, nstar_vals))}")

    # ---- VERDICT ----
    print("=" * 70)
    if falsifiers:
        print("VERDICT: NEGATIVE")
        for f in falsifiers:
            print(f"  FALSIFIER: {f}")
    elif p1_pass and p2_pass and p3_pass:
        print("VERDICT: POSITIVE")
        print("  All conjuncts of P1, P2, P3 hold.")
        print("  v=0 conjugate annealer advantage confirmed (P1a).")
        print("  No-forgetting 3x degradation at v>0 confirmed (P1b).")
        print("  Best-lam ties best fixed-lr within +-15% at all v>0 (P2).")
        print("  U-shaped window law with N* within 2.5x prediction (P3).")
    else:
        print("VERDICT: MIXED")
        if not p1_pass:
            print(f"  P1 FAIL: p1a={p1a_pass}, p1b={p1b_pass}")
        if not p2_pass:
            print(f"  P2 FAIL: {p2_results}")
        if not p3_pass:
            print(f"  P3 FAIL: ushape={p3_ushape}, within2.5x={p3_within_2p5x}, nonincreasing={nonincreasing}")
        print("  Note: 'Not a falsifier' never counts toward POSITIVE.")
    print("=" * 70)

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure output directory exists
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    rows = run_experiment()
    analyze(rows)

    # Write JSON rows
    out_path = os.path.join(out_dir, "exp137_rows.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nJSON rows written to {out_path}")
