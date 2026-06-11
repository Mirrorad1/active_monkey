"""
Exp 136 — continuous-substrate rung 4: dimensionality scaling (d = 2, 4, 8, 16, 32).

Hypothesis: under the ratified confound control (pairwise Mahalanobis separation held
constant across d), closed-form precision accumulation's inference QUALITY is
d-invariant and its COST grows polynomially but is nowhere near dead by d ~ 8 — the
card's FAIL clause is predicted not to trigger at toy scale.

Confound control (declared): 6 concept centers drawn uniformly at random on the unit
hypersphere S^{d-1} per seed; footprint scale sigma_d = (min pairwise center
distance) / 3.0, so the minimum Mahalanobis separation is exactly 3.0 at every d;
Sigma_k = sigma_d^2 I. Raw separations are logged (uncontrolled, high d would enjoy
near-orthogonality — the blessing is deliberately controlled away; what remains is
the substrate question). Anchor GIVEN (rung-1 task at each d): true concept = center
0, words from the normalized mixture at it; prior N(0, 4I_d); T = 200 train + 200
held-out; seeds 0..7 per d; tabular twin (6 states, true table, log-space per the
card rule — trivially, state posterior via product) on IDENTICAL streams.

Predictions (TRUE iff all):
- P1 quality d-invariant: final Mahalanobis localization error
  ||mu_post - s*|| / sigma_d < 1.0 in >= 7/8 seeds at EVERY d.
- P2 twin gap d-invariant: mean held-out NLL over the first 50 steps, continuous
  minus tabular, <= 0.15 nats (cell mean) at EVERY d.
- P3 cost alive: (a) full-run wall clock (200 observations + 20 checkpoint
  solves + predictive evals) at d=8 is < 100x that at d=2; (b) full run at d=32
  takes < 1.0 s; (c) microbenchmark log-log slopes over d in {8,16,32}:
  observe() kernel slope in [0.5, 2.5], Sigma-solve (the _cho_inv via .Sigma
  property) slope in [1.5, 3.5] (numpy overhead blurs exact exponents at toy
  scale — wide honest bands; medians of 3 timing repeats).

Falsifier (any triggers NEGATIVE): P1 fails (>= 2/8 seeds at Mahalanobis error >= 1)
at any d >= 8 — quality dead; OR P2 gap > 0.5 nats at any d — the twin decisively
better at scale; OR P3(a) fails — cost dead by d ~ 8 (the card's FAIL clause); OR
P3(b) fails. Slope bands outside their ranges are logged as MIXED-grade boundary
notes, not falsifiers (timing noise), unless BOTH slopes exceed 4 (super-cubic —
something is wrong).
"""
from __future__ import annotations

import json
import time
import math
from pathlib import Path

import numpy as np
from numpy.linalg import solve

from active_loop.continuous import GaussianBelief, predictive_word_logprobs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM_LIST = [2, 4, 8, 16, 32]
N_CONCEPTS = 6
N_SEEDS = 8
T_TRAIN = 200
T_HOLDOUT = 200
CHECKPOINT_EVERY = 10
EARLY_CUTOFF = 50   # checkpoints with t <= 50 count toward "early NLL"
N_OBS_REPS = 5000   # microbenchmark observe() repetitions per block
N_SOL_REPS = 1000   # microbenchmark Sigma-solve reps per block
N_TIMING_BLOCKS = 3  # median of this many blocks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unit_vectors(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """Draw n unit vectors on S^{d-1}."""
    v = rng.standard_normal((n, d))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def min_pairwise_dist(centers: np.ndarray) -> float:
    """Minimum Euclidean distance between any pair of rows in centers (n x d)."""
    n = centers.shape[0]
    min_d = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            if dist < min_d:
                min_d = dist
    return min_d


def word_probs(s_star: np.ndarray, centers: np.ndarray, sigma_d: float) -> np.ndarray:
    """Softmax mixture probs: p_k propto exp(-||s* - mu_k||^2 / (2 sigma_d^2))."""
    log_p = np.array([
        -np.dot(s_star - centers[k], s_star - centers[k]) / (2.0 * sigma_d ** 2)
        for k in range(len(centers))
    ])
    log_p -= float(np.logaddexp.reduce(log_p))
    return np.exp(log_p)


def draw_words(rng: np.random.Generator, s_star: np.ndarray, centers: np.ndarray,
               sigma_d: float, n: int) -> np.ndarray:
    """Draw n word indices from the mixture distribution anchored at s_star."""
    probs = word_probs(s_star, centers, sigma_d)
    return rng.choice(len(centers), size=n, p=probs)


def tabular_logA(centers: np.ndarray, sigma_d: float) -> np.ndarray:
    """Compute log-softmax emission table logA[k, c] = log p(word=k | state=c).

    Per the card rule: tabular twin uses log-space; state posterior via product.
    p(w=k | s=c) propto exp(-||mu_c - mu_k||^2 / (2 sigma_d^2)).
    """
    M = len(centers)
    logA = np.zeros((M, M))
    for c in range(M):
        log_row = np.array([
            -np.dot(centers[c] - centers[k], centers[c] - centers[k])
            / (2.0 * sigma_d ** 2)
            for k in range(M)
        ])
        # log-softmax normalize over k (emission probs for state c)
        log_row -= float(np.logaddexp.reduce(log_row))
        logA[:, c] = log_row
    return logA


def tabular_holdout_nll(logA: np.ndarray, logq: np.ndarray,
                         holdout_words: np.ndarray) -> float:
    """Mean NLL for tabular twin on holdout words given current log-posterior logq.

    p(w) = sum_c exp(logA[w, c] + logq[c])   (logq already normalized)
    NLL per word = -log p(w)
    """
    nll_sum = 0.0
    # normalize logq just in case
    logq_norm = logq - float(np.logaddexp.reduce(logq))
    for w in holdout_words:
        log_pw = float(np.logaddexp.reduce(logA[w, :] + logq_norm))
        nll_sum += -log_pw
    return nll_sum / len(holdout_words)


def continuous_holdout_nll(belief: GaussianBelief, centers: np.ndarray,
                            sigma_d: float, holdout_words: np.ndarray) -> float:
    """Mean NLL for continuous agent on holdout words given current belief."""
    mu_post = belief.mu
    Sigma_post = belief.Sigma
    d = len(mu_post)
    word_Sigmas = [sigma_d ** 2 * np.eye(d) for _ in range(len(centers))]
    log_probs = predictive_word_logprobs(mu_post, Sigma_post, list(centers), word_Sigmas)
    nll_sum = 0.0
    for w in holdout_words:
        nll_sum += -log_probs[w]
    return nll_sum / len(holdout_words)


# ---------------------------------------------------------------------------
# Per (d, seed) run — returns a dict of results
# ---------------------------------------------------------------------------

def run_one(d: int, seed: int, di: int) -> dict:
    rng = np.random.default_rng(seed * 10 + di)

    # Draw concept centers on S^{d-1}
    centers = unit_vectors(rng, N_CONCEPTS, d)
    min_sep = min_pairwise_dist(centers)
    sigma_d = min_sep / 3.0

    # True concept = center 0
    s_star = centers[0]

    # Generate training and holdout streams
    train_words = draw_words(rng, s_star, centers, sigma_d, T_TRAIN)
    holdout_words = draw_words(rng, s_star, centers, sigma_d, T_HOLDOUT)

    # Continuous agent setup
    Lambda_obs = (1.0 / sigma_d ** 2) * np.eye(d)  # precision of each observation
    belief = GaussianBelief(np.zeros(d), 4.0 * np.eye(d))

    # Tabular twin setup
    logA = tabular_logA(centers, sigma_d)
    logq = np.zeros(N_CONCEPTS)   # uniform log prior (un-normalized, fine — normalized on use)

    # Storage for checkpoint NLL values (for "early" mean)
    early_nll_cont = []
    early_nll_tab = []

    # Timing: full run wall-clock
    t0 = time.perf_counter()

    for t in range(1, T_TRAIN + 1):
        w = int(train_words[t - 1])
        mu_k = centers[w]

        # Continuous update
        belief.observe(mu_k, Lambda_obs)

        # Tabular update (log-space product)
        logq = logq + logA[w, :]

        # Checkpoint every CHECKPOINT_EVERY steps
        if t % CHECKPOINT_EVERY == 0:
            nll_c = continuous_holdout_nll(belief, centers, sigma_d, holdout_words)
            nll_t = tabular_holdout_nll(logA, logq, holdout_words)
            if t <= EARLY_CUTOFF:
                early_nll_cont.append(nll_c)
                early_nll_tab.append(nll_t)

    t1 = time.perf_counter()
    full_run_s = t1 - t0

    # Final Mahalanobis localization error
    mu_post = belief.mu
    maha_err = float(np.linalg.norm(mu_post - s_star) / sigma_d)

    # Early NLL means (checkpoints at t=10,20,...,50 if EARLY_CUTOFF=50)
    early_nll_cont_mean = float(np.mean(early_nll_cont)) if early_nll_cont else float("nan")
    early_nll_tab_mean = float(np.mean(early_nll_tab)) if early_nll_tab else float("nan")
    gap = early_nll_cont_mean - early_nll_tab_mean

    return {
        "d": d,
        "seed": seed,
        "sigma_d": sigma_d,
        "min_sep": min_sep,
        "mahalanobis_err": maha_err,
        "early_nll_cont": early_nll_cont_mean,
        "early_nll_tab": early_nll_tab_mean,
        "gap": gap,
        "full_run_s": full_run_s,
    }


# ---------------------------------------------------------------------------
# Microbenchmarks for observe() and Sigma-solve per d
# ---------------------------------------------------------------------------

def microbenchmark_observe(d: int) -> float:
    """Median over N_TIMING_BLOCKS of (N_OBS_REPS observe() calls / N_OBS_REPS).

    Recreate belief every 100 reps to avoid precision blowup changing flops.
    Returns median microseconds per operation.
    """
    Lambda_obs = (1.0 / 0.1) * np.eye(d)
    mu_obs = np.zeros(d)
    times = []
    for _ in range(N_TIMING_BLOCKS):
        belief = GaussianBelief(np.zeros(d), np.eye(d))
        t0 = time.perf_counter()
        for i in range(N_OBS_REPS):
            if i % 100 == 0:
                belief = GaussianBelief(np.zeros(d), np.eye(d))
            belief.observe(mu_obs, Lambda_obs)
        t1 = time.perf_counter()
        times.append((t1 - t0) / N_OBS_REPS * 1e6)
    return float(np.median(times))


def microbenchmark_sigma_solve(d: int) -> float:
    """Median over N_TIMING_BLOCKS of (N_SOL_REPS .Sigma property / N_SOL_REPS).

    Returns median microseconds per operation.
    """
    belief = GaussianBelief(np.zeros(d), np.eye(d))
    # Do a few updates so Lambda is non-trivially dense
    Lambda_obs = 0.1 * np.eye(d)
    for _ in range(10):
        belief.observe(np.zeros(d), Lambda_obs)
    times = []
    for _ in range(N_TIMING_BLOCKS):
        t0 = time.perf_counter()
        for _ in range(N_SOL_REPS):
            _ = belief.Sigma
        t1 = time.perf_counter()
        times.append((t1 - t0) / N_SOL_REPS * 1e6)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Log-log slope via OLS on log(d) -> log(time) for d in {8, 16, 32}
# ---------------------------------------------------------------------------

def log_log_slope(xs: list[float], ys: list[float]) -> float:
    """OLS slope of log(y) ~ slope * log(x) + intercept."""
    lx = np.log(xs)
    ly = np.log(ys)
    # OLS: slope = cov(lx, ly) / var(lx)
    lx_mean = np.mean(lx)
    ly_mean = np.mean(ly)
    num = np.sum((lx - lx_mean) * (ly - ly_mean))
    den = np.sum((lx - lx_mean) ** 2)
    if den == 0:
        return float("nan")
    return float(num / den)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rows_path = Path("experiments/outputs/exp136_rows.json")
    rows_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    # Per-d summary storage
    per_d: dict[int, dict] = {}

    print("=" * 80)
    print("Exp 136 — continuous-substrate rung 4: dimensionality scaling")
    print("=" * 80)

    for di, d in enumerate(DIM_LIST):
        seed_results = []
        for seed in range(N_SEEDS):
            res = run_one(d, seed, di)
            seed_results.append(res)
            # JSON row: mahalanobis_err
            all_rows.append({
                "exp": 136, "rung": 4, "agent": "continuous",
                "seed": seed, "step": T_TRAIN, "metric": "mahalanobis_err",
                "value": res["mahalanobis_err"],
                "params": {"d": d, "sigma_d": res["sigma_d"], "min_sep": res["min_sep"]},
            })
            # JSON row: early_nll continuous
            all_rows.append({
                "exp": 136, "rung": 4, "agent": "continuous",
                "seed": seed, "step": EARLY_CUTOFF, "metric": "early_nll",
                "value": res["early_nll_cont"],
                "params": {"d": d, "sigma_d": res["sigma_d"], "min_sep": res["min_sep"]},
            })
            # JSON row: early_nll tabular
            all_rows.append({
                "exp": 136, "rung": 4, "agent": "tabular",
                "seed": seed, "step": EARLY_CUTOFF, "metric": "early_nll",
                "value": res["early_nll_tab"],
                "params": {"d": d, "sigma_d": res["sigma_d"], "min_sep": res["min_sep"]},
            })
            # JSON row: gap
            all_rows.append({
                "exp": 136, "rung": 4, "agent": "cont_minus_tab",
                "seed": seed, "step": EARLY_CUTOFF, "metric": "gap",
                "value": res["gap"],
                "params": {"d": d, "sigma_d": res["sigma_d"], "min_sep": res["min_sep"]},
            })
            # JSON row: full_run_seconds
            all_rows.append({
                "exp": 136, "rung": 4, "agent": "continuous",
                "seed": seed, "step": T_TRAIN, "metric": "full_run_seconds",
                "value": res["full_run_s"],
                "params": {"d": d, "sigma_d": res["sigma_d"], "min_sep": res["min_sep"]},
            })

        # Aggregate
        maha_errs = [r["mahalanobis_err"] for r in seed_results]
        p1_pass = sum(1 for e in maha_errs if e < 1.0)
        mean_sigma_d = float(np.mean([r["sigma_d"] for r in seed_results]))
        mean_min_sep = float(np.mean([r["min_sep"] for r in seed_results]))
        mean_early_nll_cont = float(np.mean([r["early_nll_cont"] for r in seed_results]))
        mean_early_nll_tab = float(np.mean([r["early_nll_tab"] for r in seed_results]))
        mean_gap = float(np.mean([r["gap"] for r in seed_results]))
        full_run_times = [r["full_run_s"] for r in seed_results]
        median_full_run = float(np.median(full_run_times))

        # Microbenchmarks for this d
        obs_us = microbenchmark_observe(d)
        sol_us = microbenchmark_sigma_solve(d)

        # JSON rows for kernel timings (seed=-1)
        all_rows.append({
            "exp": 136, "rung": 4, "agent": "continuous",
            "seed": -1, "step": 0, "metric": "observe_us_per_op",
            "value": obs_us,
            "params": {"d": d, "sigma_d": mean_sigma_d, "min_sep": mean_min_sep},
        })
        all_rows.append({
            "exp": 136, "rung": 4, "agent": "continuous",
            "seed": -1, "step": 0, "metric": "sigma_solve_us_per_op",
            "value": sol_us,
            "params": {"d": d, "sigma_d": mean_sigma_d, "min_sep": mean_min_sep},
        })

        per_d[d] = {
            "mean_sigma_d": mean_sigma_d,
            "mean_min_sep": mean_min_sep,
            "p1_pass": p1_pass,
            "mean_early_nll_cont": mean_early_nll_cont,
            "mean_early_nll_tab": mean_early_nll_tab,
            "mean_gap": mean_gap,
            "median_full_run": median_full_run,
            "obs_us": obs_us,
            "sol_us": sol_us,
            "maha_errs": maha_errs,
        }

    # ------------------------------------------------------------------
    # Print per-d table
    # ------------------------------------------------------------------
    print()
    print(f"{'d':>4}  {'mean sigma_d':>12}  {'mean min sep':>12}  "
          f"{'P1 pass/8':>10}  {'NLL cont':>9}  {'NLL tab':>8}  "
          f"{'gap':>7}  {'med full-run s':>14}  {'obs us/op':>10}  {'solve us/op':>11}")
    print("-" * 120)
    for d in DIM_LIST:
        r = per_d[d]
        print(f"{d:>4}  {r['mean_sigma_d']:>12.4f}  {r['mean_min_sep']:>12.4f}  "
              f"{r['p1_pass']:>10d}  {r['mean_early_nll_cont']:>9.4f}  "
              f"{r['mean_early_nll_tab']:>8.4f}  {r['mean_gap']:>7.4f}  "
              f"{r['median_full_run']:>14.4f}  {r['obs_us']:>10.4f}  {r['sol_us']:>11.4f}")

    # ------------------------------------------------------------------
    # Slope fits over d in {8, 16, 32}
    # ------------------------------------------------------------------
    slope_ds = [8, 16, 32]
    obs_us_vals = [per_d[d]["obs_us"] for d in slope_ds]
    sol_us_vals = [per_d[d]["sol_us"] for d in slope_ds]
    obs_slope = log_log_slope([float(d) for d in slope_ds], obs_us_vals)
    sol_slope = log_log_slope([float(d) for d in slope_ds], sol_us_vals)

    print()
    print("Log-log slopes over d in {8, 16, 32}:")
    print(f"  observe() kernel slope:        {obs_slope:.3f}  (target band [0.5, 2.5])")
    print(f"  Sigma-solve (.Sigma) slope:    {sol_slope:.3f}  (target band [1.5, 3.5])")

    # ------------------------------------------------------------------
    # P1 / P2 / P3 tallies
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("TALLIES")
    print("=" * 80)

    print()
    print("P1 — quality d-invariant (Mahalanobis err < 1.0 in >= 7/8 seeds at EVERY d):")
    p1_conjuncts = []
    for d in DIM_LIST:
        pass_count = per_d[d]["p1_pass"]
        ok = pass_count >= 7
        tag = "PASS" if ok else "FAIL"
        print(f"  d={d:>2}: {pass_count}/8 seeds pass  [{tag}]"
              f"  errs={[f'{e:.3f}' for e in per_d[d]['maha_errs']]}")
        p1_conjuncts.append(ok)
    p1_holds = all(p1_conjuncts)

    print()
    print("P2 — twin gap d-invariant (mean early NLL gap cont-tab <= 0.15 nats at EVERY d):")
    p2_conjuncts = []
    for d in DIM_LIST:
        gap = per_d[d]["mean_gap"]
        ok = gap <= 0.15
        tag = "PASS" if ok else "FAIL"
        print(f"  d={d:>2}: gap = {gap:.4f} nats  [{tag}]")
        p2_conjuncts.append(ok)
    p2_holds = all(p2_conjuncts)

    print()
    print("P3 — cost alive:")
    t_d2 = per_d[2]["median_full_run"]
    t_d8 = per_d[8]["median_full_run"]
    t_d32 = per_d[32]["median_full_run"]
    p3a_ratio = t_d8 / t_d2 if t_d2 > 0 else float("inf")
    p3a_ok = p3a_ratio < 100.0
    p3b_ok = t_d32 < 1.0
    p3c_obs_ok = 0.5 <= obs_slope <= 2.5
    p3c_sol_ok = 1.5 <= sol_slope <= 3.5
    p3a_tag = "PASS" if p3a_ok else "FAIL"
    p3b_tag = "PASS" if p3b_ok else "FAIL"
    p3c_obs_tag = "PASS" if p3c_obs_ok else "OUT-OF-BAND"
    p3c_sol_tag = "PASS" if p3c_sol_ok else "OUT-OF-BAND"
    print(f"  P3(a): d=8 full-run / d=2 full-run = {p3a_ratio:.2f}x  (< 100x required)  [{p3a_tag}]")
    print(f"  P3(b): d=32 median full-run = {t_d32:.4f} s  (< 1.0 s required)  [{p3b_tag}]")
    print(f"  P3(c): observe() slope = {obs_slope:.3f}  (target [0.5, 2.5])  [{p3c_obs_tag}]")
    print(f"  P3(c): Sigma-solve slope = {sol_slope:.3f}  (target [1.5, 3.5])  [{p3c_sol_tag}]")
    # Slope-band misses are not falsifiers, but the docstring's "TRUE iff all"
    # makes P3(c) a conjunct of the POSITIVE verdict: out-of-band slopes -> MIXED.
    p3_holds = p3a_ok and p3b_ok and p3c_obs_ok and p3c_sol_ok

    # ------------------------------------------------------------------
    # Falsifier checks and verdict
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    falsifiers_triggered = []

    # P1 falsifier: >= 2/8 seeds at error >= 1 at any d >= 8
    for d in DIM_LIST:
        if d >= 8:
            fail_count = N_SEEDS - per_d[d]["p1_pass"]
            if fail_count >= 2:
                falsifiers_triggered.append(
                    f"P1 FALSIFIER: d={d}, {fail_count}/8 seeds have Mahalanobis err >= 1.0"
                    f" (quality dead)"
                )

    # P2 falsifier: gap > 0.5 nats at any d
    for d in DIM_LIST:
        gap = per_d[d]["mean_gap"]
        if gap > 0.5:
            falsifiers_triggered.append(
                f"P2 FALSIFIER: d={d}, gap={gap:.4f} > 0.5 nats"
                f" (twin decisively better at scale)"
            )

    # P3(a) falsifier
    if not p3a_ok:
        falsifiers_triggered.append(
            f"P3(a) FALSIFIER: d=8/d=2 ratio = {p3a_ratio:.2f}x >= 100x"
            f" (cost dead by d~8, card FAIL clause triggered)"
        )

    # P3(b) falsifier
    if not p3b_ok:
        falsifiers_triggered.append(
            f"P3(b) FALSIFIER: d=32 full-run = {t_d32:.4f} s >= 1.0 s"
        )

    # Slope band misses — MIXED notes, not falsifiers, unless both > 4
    slope_notes = []
    if not p3c_obs_ok:
        slope_notes.append(
            f"MIXED-grade boundary note: observe() slope = {obs_slope:.3f} outside [0.5, 2.5]"
        )
    if not p3c_sol_ok:
        slope_notes.append(
            f"MIXED-grade boundary note: Sigma-solve slope = {sol_slope:.3f} outside [1.5, 3.5]"
        )
    both_super_cubic = (obs_slope > 4.0 and sol_slope > 4.0)
    if both_super_cubic:
        falsifiers_triggered.append(
            f"P3(c) FALSIFIER: both slopes super-cubic"
            f" (observe={obs_slope:.3f}, Sigma-solve={sol_slope:.3f} — both > 4)"
        )

    if falsifiers_triggered:
        verdict = "NEGATIVE"
    elif p1_holds and p2_holds and p3_holds:
        verdict = "POSITIVE"
    else:
        verdict = "MIXED"

    if falsifiers_triggered:
        print()
        for f in falsifiers_triggered:
            print(f"  [FALSIFIER] {f}")

    if slope_notes:
        print()
        for note in slope_notes:
            print(f"  [NOTE] {note}")

    all_conj_ok = p1_holds and p2_holds and p3a_ok and p3b_ok
    print()
    print(f"  P1 holds across all d:     {'YES' if p1_holds else 'NO'}")
    print(f"  P2 holds across all d:     {'YES' if p2_holds else 'NO'}")
    print(f"  P3(a) d8/d2 < 100x:        {'YES' if p3a_ok else 'NO'}")
    print(f"  P3(b) d32 < 1.0 s:         {'YES' if p3b_ok else 'NO'}")
    print(f"  P3(c) obs slope in band:   {'YES' if p3c_obs_ok else 'NO (see notes)'}")
    print(f"  P3(c) sol slope in band:   {'YES' if p3c_sol_ok else 'NO (see notes)'}")
    print()
    print(f"VERDICT: {verdict}")

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    with open(rows_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print()
    print(f"JSON rows written to {rows_path}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
