"""
Exp 137 mechanism check — is the tracking NEGATIVE attributable to the decay FORM
(mean re-anchoring toward a static prior), not to conjugate forgetting per se?

Exp 137's predeclared decay blends m toward the STATIC natal prior mean (0,0) every
step: m' = (lam*kappa*m + (1-lam)*kappa0*m0) / kappa'. Under sustained drift the
truth walks away from m0, so each decay step adds a position-dependent pull toward
the origin — a bias absent from a pure EMA, growing with v*t, and inflating
small-N_eff error (pushing the observed argmin up). Predicted before running this
check: a KEEP-MEAN decay (counts and scatter decay toward prior magnitudes, m kept:
kappa' = kappa0 + lam*(kappa - kappa0), m' = m) is analytically an EMA with
alpha ~ 1/(kappa_ss + 1) and therefore:
- C1: keep-mean NIW ties the matched fixed-lr baseline within +-5% (cell means) at
  every v > 0 on the IDENTICAL streams of Exp 137 (same rng scheme, seeds 0..7);
- C2: keep-mean argmin N* returns to within a factor 2.5 of the cube-root law
  (sigma^2/(2 v^2))^(1/3) at ALL three v > 0 (including v=0.008 where the static-
  prior form was off 4x), and is non-increasing in v;
- C3: the static-prior form's excess RMS over keep-mean (best-cell to best-cell)
  GROWS with v (monotone over the three v > 0 levels) — the bias signature.
Falsifier: C1 fails (keep-mean still loses > 5% — the negative is NOT the re-anchor
form; conjugate forgetting itself is deficient) — log whichever way it falls.
"""
from __future__ import annotations

import json
import os
import time
import numpy as np

# ---------------------------------------------------------------------------
# Experiment parameters (identical to exp137)
# ---------------------------------------------------------------------------

D = 2
SIGMA = 0.35
T = 4000
BURN = 1000
EVAL_LEN = T - BURN  # 3000

V_LIST = [0.0, 0.0005, 0.002, 0.008]
V_POS = [0.0005, 0.002, 0.008]
N_EFF_GRID = [5, 10, 20, 40, 80, 160, 320]
N_SEEDS = 8

KAPPA0 = 1.0
M0 = np.zeros(D)

# Predicted N* from Exp 137: (sigma^2 / (2 v^2))^(1/3)
PRED_NSTAR = {0.0005: 62.6, 0.002: 24.8, 0.008: 9.9}


def predicted_nstar(v):
    if v == 0:
        return float("inf")
    return (SIGMA ** 2 / (2.0 * v ** 2)) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# Stream generation — MUST use vi as index in ORIGINAL V_LIST [0, 0.0005, 0.002, 0.008]
# so streams are bit-identical to exp137
# ---------------------------------------------------------------------------

def make_streams(v, vi):
    """Return dict seed -> (T, 2) array. vi is index in full V_LIST."""
    streams = {}
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed * 40 + vi)
        mu_true = v * np.arange(T)[:, None] * np.array([1.0, 0.0])
        noise = rng.standard_normal((T, D)) * SIGMA
        streams[seed] = mu_true + noise
    return streams


# ---------------------------------------------------------------------------
# Three agents inline (kappa scalar, m 2-vector; S not needed for mean tracking)
# ---------------------------------------------------------------------------

def run_static_prior(xs, lam, kappa0=KAPPA0, m0=M0):
    """Exp 137's original form: decay blends m toward static prior mean m0."""
    kappa = kappa0
    m = m0.copy()
    sq_errs = []
    v_t = None  # extracted from stream construction — we recompute mu_true inline
    # We don't have v here; we'll pass xs as precomputed with true positions baked in.
    # Actually we need mu_true(t) for error; pass it separately — see run_cell.
    raise NotImplementedError("Use run_cell instead")


def run_cell(xs, lam, agent, v):
    """
    Run one agent for all seeds over xs streams.
    agent: 'static_prior' | 'keep_mean' | 'fixed_lr'
    Returns list of N_SEEDS rms floats.
    """
    kappa0 = KAPPA0
    m0 = M0.copy()
    n_eff = round(1.0 / (1.0 - lam)) if lam < 1.0 else None

    rms_list = []
    for seed in range(N_SEEDS):
        x_stream = xs[seed]  # (T, 2)

        if agent == "fixed_lr":
            alpha = lam  # alpha = 1/N_eff; lam passed as alpha here
            m = np.zeros(D)
            sq_errs = []
            for t_idx in range(T):
                x_t = x_stream[t_idx]
                m = m + alpha * (x_t - m)
                if t_idx >= BURN:
                    mu_true_t = v * t_idx * np.array([1.0, 0.0])
                    sq_errs.append(np.sum((m - mu_true_t) ** 2))
            rms_list.append(float(np.sqrt(np.mean(sq_errs))))

        elif agent == "static_prior":
            # Exp 137 form: update then decay m toward m0=(0,0)
            kappa = kappa0
            m = m0.copy()
            sq_errs = []
            for t_idx in range(T):
                x_t = x_stream[t_idx]
                # update
                kappa_new = kappa + 1.0
                m = (kappa * m + x_t) / kappa_new
                kappa = kappa_new
                # decay
                kappa_d = kappa0 + lam * (kappa - kappa0)
                m = (lam * kappa * m + (1.0 - lam) * kappa0 * m0) / kappa_d
                kappa = kappa_d
                if t_idx >= BURN:
                    mu_true_t = v * t_idx * np.array([1.0, 0.0])
                    sq_errs.append(np.sum((m - mu_true_t) ** 2))
            rms_list.append(float(np.sqrt(np.mean(sq_errs))))

        elif agent == "keep_mean":
            # Keep-mean form: decay kappa only; m kept
            kappa = kappa0
            m = m0.copy()
            sq_errs = []
            for t_idx in range(T):
                x_t = x_stream[t_idx]
                # update (same as static_prior)
                kappa_new = kappa + 1.0
                m = (kappa * m + x_t) / kappa_new
                kappa = kappa_new
                # decay kappa only; m kept
                kappa = kappa0 + lam * (kappa - kappa0)
                # m unchanged
                if t_idx >= BURN:
                    mu_true_t = v * t_idx * np.array([1.0, 0.0])
                    sq_errs.append(np.sum((m - mu_true_t) ** 2))
            rms_list.append(float(np.sqrt(np.mean(sq_errs))))

        else:
            raise ValueError(f"Unknown agent: {agent}")

    return rms_list


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    t0 = time.time()
    rows = []

    for vi, v in enumerate(V_LIST):
        if v == 0.0:
            continue  # skip v=0 per spec (mechanism check is v>0 only)

        streams = make_streams(v, vi)  # vi in {1,2,3} for v in {0.0005,0.002,0.008}

        for n_eff in N_EFF_GRID:
            lam = 1.0 - 1.0 / n_eff
            alpha_lr = 1.0 / n_eff

            # (a) static_prior
            sp_rms = run_cell(streams, lam, "static_prior", v)
            for seed, rms in enumerate(sp_rms):
                rows.append({
                    "exp": 137, "rung": 5, "agent": "static_prior", "seed": seed,
                    "step": T, "metric": "rms", "value": rms,
                    "params": {"v": v, "N_eff": n_eff}
                })

            # (b) keep_mean
            km_rms = run_cell(streams, lam, "keep_mean", v)
            for seed, rms in enumerate(km_rms):
                rows.append({
                    "exp": 137, "rung": 5, "agent": "keep_mean", "seed": seed,
                    "step": T, "metric": "rms", "value": rms,
                    "params": {"v": v, "N_eff": n_eff}
                })

            # (c) fixed_lr (alpha = 1/N_eff; pass as lam arg — run_cell uses it directly)
            fl_rms = run_cell(streams, alpha_lr, "fixed_lr", v)
            for seed, rms in enumerate(fl_rms):
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

def cell_mean_from_rows(rows, v, agent, n_eff):
    vals = [r["value"] for r in rows
            if r["params"]["v"] == v and r["agent"] == agent and r["params"]["N_eff"] == n_eff]
    return float(np.mean(vals)) if vals else float("nan")


def analyze(rows):
    # ---- Per-v tables ----
    for v in V_POS:
        print(f"=== v = {v} ===")
        print(f"{'N_eff':>8}  {'static RMS':>12}  {'keepmean RMS':>14}  {'fixedlr RMS':>13}")
        for n in N_EFF_GRID:
            sp = cell_mean_from_rows(rows, v, "static_prior", n)
            km = cell_mean_from_rows(rows, v, "keep_mean", n)
            fl = cell_mean_from_rows(rows, v, "fixed_lr", n)
            print(f"{n:>8}  {sp:>12.5f}  {km:>14.5f}  {fl:>13.5f}")
        print()

    # ---- Per-v best cells ----
    print("=== Per-v best cells and ratios ===")
    best = {}  # v -> {agent: (n_eff, rms)}
    for v in V_POS:
        best[v] = {}
        for agent in ("static_prior", "keep_mean", "fixed_lr"):
            by_n = {n: cell_mean_from_rows(rows, v, agent, n) for n in N_EFF_GRID}
            best_n = min(by_n, key=by_n.get)
            best[v][agent] = (best_n, by_n[best_n])
        sp_n, sp_rms = best[v]["static_prior"]
        km_n, km_rms = best[v]["keep_mean"]
        fl_n, fl_rms = best[v]["fixed_lr"]
        print(f"v={v}: static_prior best N={sp_n} RMS={sp_rms:.5f} | "
              f"keep_mean best N={km_n} RMS={km_rms:.5f} | "
              f"fixed_lr best N={fl_n} RMS={fl_rms:.5f}")
        print(f"       km/fl ratio={km_rms/fl_rms:.4f}  sp/fl ratio={sp_rms/fl_rms:.4f}  "
              f"sp/km ratio={sp_rms/km_rms:.4f}")
    print()

    # ---- Argmins vs predicted N* ----
    print("=== Argmins vs predicted N* (62.6 / 24.8 / 9.9) ===")
    km_nstar_obs = {}
    for v in V_POS:
        km_curve = [cell_mean_from_rows(rows, v, "keep_mean", n) for n in N_EFF_GRID]
        argmin_idx = int(np.argmin(km_curve))
        obs_n = N_EFF_GRID[argmin_idx]
        pred_n = predicted_nstar(v)
        ratio = obs_n / pred_n
        interior = (argmin_idx > 0) and (argmin_idx < len(N_EFF_GRID) - 1)
        km_nstar_obs[v] = obs_n
        print(f"  v={v}: keep_mean argmin N*={obs_n}  pred={pred_n:.1f}  "
              f"ratio={ratio:.2f}  interior={'yes' if interior else 'NO'}")
    print()

    # ---- C1 tally ----
    print("=== C1: keep-mean ties fixed-lr within +-5% at every v>0 ===")
    c1_results = {}
    for v in V_POS:
        km_rms = best[v]["keep_mean"][1]
        fl_rms = best[v]["fixed_lr"][1]
        ratio = km_rms / fl_rms if fl_rms > 0 else float("nan")
        ok = 0.95 <= ratio <= 1.05
        c1_results[v] = ok
        print(f"  v={v}: keep_mean={km_rms:.5f}, fixed_lr={fl_rms:.5f}, "
              f"ratio={ratio:.4f}  {'PASS' if ok else 'FAIL (outside +-5%)'}")
    c1_pass = all(c1_results.values())
    print(f"  C1 overall: {'PASS' if c1_pass else 'FAIL'}")
    print()

    # ---- C2 tally ----
    print("=== C2: keep-mean N* within 2.5x cube-root law, non-increasing in v ===")
    c2_within = {}
    for v in V_POS:
        obs_n = km_nstar_obs[v]
        pred_n = predicted_nstar(v)
        ratio = obs_n / pred_n
        within = (ratio <= 2.5) and (ratio >= 1.0 / 2.5)
        c2_within[v] = within
        print(f"  v={v}: N*={obs_n}, pred={pred_n:.1f}, ratio={ratio:.2f}  "
              f"{'within 2.5x' if within else 'OUTSIDE 2.5x'}")
    nstar_vals_km = [km_nstar_obs[v] for v in V_POS]
    nonincreasing = all(nstar_vals_km[i] >= nstar_vals_km[i+1]
                        for i in range(len(nstar_vals_km) - 1))
    print(f"  N* sequence: {nstar_vals_km}  non-increasing={'yes' if nonincreasing else 'NO'}")
    c2_pass = all(c2_within.values()) and nonincreasing
    print(f"  C2 overall: {'PASS' if c2_pass else 'FAIL'}")
    print()

    # ---- C3 tally ----
    print("=== C3: static-prior excess over keep-mean GROWS with v (monotone) ===")
    excess = {}
    for v in V_POS:
        sp_rms = best[v]["static_prior"][1]
        km_rms = best[v]["keep_mean"][1]
        excess[v] = sp_rms - km_rms
        print(f"  v={v}: static_prior={sp_rms:.5f}, keep_mean={km_rms:.5f}, "
              f"excess={excess[v]:.5f}")
    v_list_pos = list(V_POS)
    monotone_growing = all(
        excess[v_list_pos[i]] <= excess[v_list_pos[i + 1]]
        for i in range(len(v_list_pos) - 1)
    )
    print(f"  Excess sequence: {[f'{excess[v]:.5f}' for v in V_POS]}  "
          f"monotone-growing={'yes' if monotone_growing else 'NO'}")
    c3_pass = monotone_growing
    print(f"  C3 overall: {'PASS' if c3_pass else 'FAIL'}")
    print()

    # ---- VERDICT ----
    print("=" * 70)
    print(f"C1={'PASS' if c1_pass else 'FAIL'}  C2={'PASS' if c2_pass else 'FAIL'}  "
          f"C3={'PASS' if c3_pass else 'FAIL'}")
    all_pass = c1_pass and c2_pass and c3_pass
    if all_pass:
        verdict = "CONFIRMED"
        note = ("Keep-mean NIW ties fixed-lr (C1), N* returns to cube-root law (C2), "
                "and static-prior excess grows with v (C3). "
                "The Exp 137 tracking negative IS attributable to the re-anchoring form.")
    elif not c1_pass and not c2_pass and not c3_pass:
        verdict = "NOT-CONFIRMED"
        note = "All three conjuncts fail — conjugate forgetting itself is deficient."
    elif not c1_pass:
        verdict = "NOT-CONFIRMED"
        note = ("C1 fails: keep-mean still loses > 5% to fixed-lr. "
                "The negative is NOT solely the re-anchor form; "
                "conjugate forgetting itself is deficient.")
    else:
        # partial
        failing = [c for c, ok in [("C1", c1_pass), ("C2", c2_pass), ("C3", c3_pass)] if not ok]
        verdict = "PARTIAL"
        note = f"Failing arms: {failing}. Not silently promoted to CONFIRMED."
    print(f"VERDICT: {verdict}")
    print(f"  {note}")
    print("=" * 70)
    print()

    return c1_pass, c2_pass, c3_pass


# ---------------------------------------------------------------------------
# Bit-identity check vs exp137
# ---------------------------------------------------------------------------

def bit_identity_check(rows):
    # Load exp137 rows
    exp137_path = os.path.join(os.path.dirname(__file__), "outputs", "exp137_rows.json")
    try:
        with open(exp137_path) as f:
            exp137_rows = json.load(f)
    except FileNotFoundError:
        print("BIT-IDENTITY CHECK: exp137_rows.json not found — skipping.")
        return

    # Find static_prior v=0.002, N_eff=40, seed=0 from THIS run
    this_val = None
    for r in rows:
        if (r["agent"] == "static_prior" and r["params"]["v"] == 0.002
                and r["params"]["N_eff"] == 40 and r["seed"] == 0):
            this_val = r["value"]
            break

    # Find NIW v=0.002, N_eff=40, seed=0 from exp137
    ref_val = None
    for r in exp137_rows:
        if (r["agent"] == "niw" and r["params"]["v"] == 0.002
                and r["params"]["N_eff"] == 40 and r["seed"] == 0):
            ref_val = r["value"]
            break

    print("=== BIT-IDENTITY CHECK ===")
    print(f"  This script  (static_prior, v=0.002, N_eff=40, seed=0): {this_val}")
    print(f"  exp137_rows  (niw,          v=0.002, N_eff=40, seed=0): {ref_val}")
    if this_val is not None and ref_val is not None:
        if this_val == ref_val:
            print("  MATCH: bit-identical.")
        else:
            diff = abs(this_val - ref_val)
            rel = diff / abs(ref_val) if ref_val != 0 else float("inf")
            print(f"  MISMATCH: abs diff={diff:.2e}, rel diff={rel:.2e}")
            assert False, f"Bit-identity FAILED: {this_val} != {ref_val}"
    else:
        print("  Could not locate one or both values.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    rows = run_experiment()
    analyze(rows)
    bit_identity_check(rows)

    # Write JSON rows
    out_path = os.path.join(out_dir, "exp137_mech_rows.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"JSON rows written to {out_path}")
