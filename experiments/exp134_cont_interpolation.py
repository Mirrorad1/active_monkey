"""
Exp 134 — continuous-substrate rung 2: interpolation to unseen blends (the unimodal-
approximation boundary).

Hypothesis: under precision accumulation, a blended word stream from two opposite
concepts lands the posterior mean at the midpoint (no corner-snapping anywhere in the
sweep), but the posterior does NOT widen — the unimodal conjugate posterior hides the
ambiguity, and its cost surfaces as a held-out NLL gap vs the tabular twin that grows
with concept separation and saturates at ln 2.

Analysis behind the predictions (made before running): mu_post is the precision-
weighted average of prior mean and observed word centers, so with equal Sigma_k it
converges to the empirical mean of the two corners = the midpoint — snap is
structurally impossible for this update rule. The accumulated precision
Lambda_0 + sum_t Lambda_{o_t} is independent of WHICH words arrive when all Sigma_k
are equal, so tr(Sigma_post) is bit-identical between blended and pure streams — the
card's naive "Sigma should widen" is predicted FALSE for this rule. The tabular twin's
categorical posterior CAN represent the blend bimodally (q ~ (1/2,0,0,1/2)), so at
large separation its blended held-out NLL approaches ln 2 while the continuous agent's
midpoint-unimodal predictive approaches uniform over 4 words = ln 4: predicted gap
saturates at ln 2 ~ 0.693 nats and shrinks to ~0 when footprints overlap (small L/sigma).

Sweep (predeclared grid): L in {0.5, 1.0, 2.0, 4.0} x sigma in {0.175, 0.35, 0.7},
4 word-Gaussians at square corners (0,0),(L,0),(0,L),(L,L), Sigma_k = sigma^2 I;
blend = iid 50/50 words from corners A=(0,0) and D=(L,L); prior N((0,0), 4I);
T = 200 train + 200 held-out per seed; seeds 0..7 per cell (96 runs); per cell also
a PURE corner-A stream (same length, same seeds) for the P2 control.

Predictions (hypothesis TRUE iff all hold):
- P1 interpolation, no snap: snapped (mu_post closer to some corner than to the
  midpoint) in 0/96 blended runs, AND ||mu_post - midpoint|| < 0.15*L in >= 7/8 seeds
  in >= 11/12 cells.
- P2 no-widening (structural): |tr(Sigma_post_blend) - tr(Sigma_post_pure)| < 1e-9
  in all 96 seed-cell pairs.
- P3 cost boundary (shape-level, per the ratified sweep rules): Spearman rho between
  cell-mean NLL gap (continuous - tabular, blended held-out) and log(L/sigma) >= 0.8,
  AND gap at the largest L/sigma within 0.15 nats of ln 2, AND gap at the smallest
  L/sigma < 0.10 nats.

Falsifier (any triggers NEGATIVE): any cell with >= 2/8 snapped runs (snap regime
exists — the structural analysis is wrong), OR P2 violated anywhere (Sigma DOES
respond to stream composition), OR P3 shape fails (gap non-monotone in L/sigma, or
saturates far from ln 2, or stays large when footprints overlap). The card's own FAIL
clause (snap across the WHOLE sweep) is a strict subset of the first falsifier.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

# Ensure repo root on path
sys.path.insert(0, ".")

from active_loop.continuous import GaussianBelief, predictive_word_logprobs

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
# Cell index enumerates (L, sigma) in row-major order: L outer, sigma inner.
# L in {0.5, 1.0, 2.0, 4.0}, sigma in {0.175, 0.35, 0.7}
# => cell 0: (0.5, 0.175), cell 1: (0.5, 0.35), cell 2: (0.5, 0.7),
#    cell 3: (1.0, 0.175), ..., cell 11: (4.0, 0.7)

L_VALUES = [0.5, 1.0, 2.0, 4.0]
SIGMA_VALUES = [0.175, 0.35, 0.7]
CELLS = [(L, sigma) for L in L_VALUES for sigma in SIGMA_VALUES]  # 12 cells
N_SEEDS = 8
TRAIN_LEN = 200
HOLDOUT_LEN = 200
LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Spearman rank correlation (numpy-only fallback)
# ---------------------------------------------------------------------------

def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation using average ranks for ties, then Pearson on ranks."""
    # Try scipy.stats first; fall back to numpy implementation
    try:
        from scipy.stats import spearmanr  # type: ignore
        result = spearmanr(x, y)
        return float(result.statistic if hasattr(result, "statistic") else result.correlation)
    except Exception:
        pass

    def _rank(v: np.ndarray) -> np.ndarray:
        """Average ranks (1-based) with tie averaging."""
        n = len(v)
        order = np.argsort(v, kind="stable")
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i + 1
            while j < n and v[order[j]] == v[order[i]]:
                j += 1
            avg = (i + 1 + j) / 2.0  # average of 1-based positions i+1..j
            for k in range(i, j):
                ranks[order[k]] = avg
            i = j
        return ranks

    rx = _rank(np.asarray(x, dtype=float))
    ry = _rank(np.asarray(y, dtype=float))
    # Pearson on ranks
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = math.sqrt(float(np.dot(rx_c, rx_c)) * float(np.dot(ry_c, ry_c)))
    if denom == 0.0:
        return 0.0
    return float(np.dot(rx_c, ry_c)) / denom


# ---------------------------------------------------------------------------
# Per-cell, per-seed computation
# ---------------------------------------------------------------------------

def run_cell_seed(cell_index: int, seed: int, L: float, sigma: float) -> dict:
    """Run one (cell, seed) pair; return dict of scalar results."""
    rng = np.random.default_rng(seed * 1000 + cell_index)

    # Geometry
    d = 2
    corners = np.array([[0.0, 0.0], [L, 0.0], [0.0, L], [L, L]])  # A, B, C, D
    midpoint = np.array([L / 2.0, L / 2.0])
    Sigma_k = (sigma ** 2) * np.eye(d)
    Lambda_k = np.linalg.inv(Sigma_k)  # = (1/sigma^2) * I

    # Prior: N((0,0), 4I)
    prior_mu = np.zeros(d)
    prior_Sigma = 4.0 * np.eye(d)

    # Word lists for predictive_word_logprobs
    word_mus = [corners[k] for k in range(4)]
    word_Sigmas = [Sigma_k for _ in range(4)]

    # --- Streams ---
    # Blended: iid 50/50 from corners A=0 and D=3
    blend_train = rng.choice([0, 3], size=TRAIN_LEN)
    blend_holdout = rng.choice([0, 3], size=HOLDOUT_LEN)
    # Pure-A: all word 0 (deterministic)
    pure_train = np.zeros(TRAIN_LEN, dtype=int)

    # --- Continuous blended agent ---
    belief_blend = GaussianBelief(prior_mu, prior_Sigma)
    for w in blend_train:
        belief_blend.observe(corners[w], Lambda_k)
    mu_post = belief_blend.mu
    Sigma_post = belief_blend.Sigma
    tr_blend = belief_blend.trace_sigma

    # Snap check: snapped if any corner is closer to mu_post than the midpoint
    dist_to_corners = np.array([np.linalg.norm(mu_post - corners[k]) for k in range(4)])
    dist_to_mid = np.linalg.norm(mu_post - midpoint)
    snapped = int(np.min(dist_to_corners) < dist_to_mid)
    midpoint_err = np.linalg.norm(mu_post - midpoint)
    midpoint_err_norm = midpoint_err / L if L > 0 else midpoint_err

    # Continuous blended held-out NLL
    log_probs = predictive_word_logprobs(mu_post, Sigma_post, word_mus, word_Sigmas)
    cont_nll = float(-np.mean([log_probs[w] for w in blend_holdout]))

    # --- Continuous pure-A agent (for P2 control) ---
    belief_pure = GaussianBelief(prior_mu, prior_Sigma)
    for w in pure_train:
        belief_pure.observe(corners[w], Lambda_k)
    tr_pure = belief_pure.trace_sigma
    tr_diff_abs = abs(tr_blend - tr_pure)

    # --- Tabular twin (blended) ---
    # Transition matrix A_table[k, c] = N(corner_c; corner_k, Sigma_k), column-normalised
    # i.e. A_table[word, :] = unnorm emission row, then normalise columns
    # Actually: tabular posterior update uses the emission row for word w:
    # q *= emission[w, :], where emission[w, c] = N(corner_c; corner_w, Sigma_k)
    # But we need the transition matrix column-normalised as specified.
    # Spec: "A_table[k, c] = normalized Gaussian density N(corner_c; corner_k, Sigma_k),
    # column-normalized"
    # Build raw: raw[k, c] = N(corner_c; corner_k, Sigma_k)
    A_raw = np.zeros((4, 4))
    for k in range(4):
        for c in range(4):
            diff = corners[c] - corners[k]
            log_p = -0.5 * diff @ Lambda_k @ diff
            # drop constant normalisation factor (same for all; will cancel in column norm)
            A_raw[k, c] = math.exp(log_p)
    # Column-normalise
    col_sums = A_raw.sum(axis=0)
    A_table = A_raw / col_sums[np.newaxis, :]

    # Tabular belief update: q starts uniform, update q *= A_table[w, :] per word
    def _tab_update(stream: np.ndarray) -> np.ndarray:
        q = np.ones(4) / 4.0
        for w in stream:
            q = q * A_table[w, :]
            q_sum = q.sum()
            if q_sum < 1e-300:
                q = np.ones(4) / 4.0
            else:
                q = q / q_sum
        return q

    q_tab = _tab_update(blend_train)

    # Tabular blended held-out NLL
    tab_nll_terms = []
    q_eval = q_tab.copy()
    for w in blend_holdout:
        pred = A_table[w, :] @ q_eval
        if pred < 1e-300:
            pred = 1e-300
        tab_nll_terms.append(-math.log(pred))
        # advance q
        q_eval = q_eval * A_table[w, :]
        q_sum = q_eval.sum()
        if q_sum < 1e-300:
            q_eval = np.ones(4) / 4.0
        else:
            q_eval = q_eval / q_sum

    tab_nll = float(np.mean(tab_nll_terms))
    nll_gap = cont_nll - tab_nll

    return {
        "midpoint_err_norm": midpoint_err_norm,
        "snapped": snapped,
        "tr_blend": tr_blend,
        "tr_pure": tr_pure,
        "tr_diff_abs": tr_diff_abs,
        "cont_nll": cont_nll,
        "tab_nll": tab_nll,
        "nll_gap": nll_gap,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_rows = []
    # Per-cell aggregates for printing
    cell_results = []

    for cell_index, (L, sigma) in enumerate(CELLS):
        ratio = L / sigma
        seed_data = []
        for seed in range(N_SEEDS):
            r = run_cell_seed(cell_index, seed, L, sigma)
            seed_data.append(r)

            # Emit JSON rows
            base_params = {"L": L, "sigma": sigma, "ratio": ratio}
            for metric in [
                "midpoint_err_norm", "snapped", "tr_blend", "tr_pure",
                "tr_diff_abs",
            ]:
                all_rows.append({
                    "exp": 134, "rung": 2, "agent": "continuous_blend",
                    "seed": seed, "step": TRAIN_LEN,
                    "metric": metric, "value": r[metric], "params": base_params,
                })
            all_rows.append({
                "exp": 134, "rung": 2, "agent": "continuous_blend",
                "seed": seed, "step": TRAIN_LEN,
                "metric": "holdout_nll", "value": r["cont_nll"], "params": base_params,
            })
            all_rows.append({
                "exp": 134, "rung": 2, "agent": "tabular_blend",
                "seed": seed, "step": TRAIN_LEN,
                "metric": "holdout_nll", "value": r["tab_nll"], "params": base_params,
            })
            all_rows.append({
                "exp": 134, "rung": 2, "agent": "continuous_blend",
                "seed": seed, "step": TRAIN_LEN,
                "metric": "nll_gap", "value": r["nll_gap"], "params": base_params,
            })

        # Cell-level aggregates
        snap_count = sum(r["snapped"] for r in seed_data)
        mid_err_pass = sum(r["midpoint_err_norm"] < 0.15 for r in seed_data)
        max_tr_diff = max(r["tr_diff_abs"] for r in seed_data)
        mean_cont_nll = float(np.mean([r["cont_nll"] for r in seed_data]))
        mean_tab_nll = float(np.mean([r["tab_nll"] for r in seed_data]))
        mean_gap = float(np.mean([r["nll_gap"] for r in seed_data]))

        cell_results.append({
            "L": L, "sigma": sigma, "ratio": ratio,
            "snap_count": snap_count,
            "mid_err_pass": mid_err_pass,
            "max_tr_diff": max_tr_diff,
            "mean_cont_nll": mean_cont_nll,
            "mean_tab_nll": mean_tab_nll,
            "mean_gap": mean_gap,
        })

    # Write JSON rows
    import os
    os.makedirs("experiments/outputs", exist_ok=True)
    with open("experiments/outputs/exp134_rows.json", "w") as fh:
        for row in all_rows:
            fh.write(json.dumps(row) + "\n")

    # ---------------------------------------------------------------------------
    # Print cell table
    # ---------------------------------------------------------------------------
    print("=" * 92)
    print("Exp 134 — Cell Table (12 cells, 8 seeds each)")
    print("=" * 92)
    hdr = (
        f"{'L':>5}  {'sigma':>6}  {'L/sig':>6}  "
        f"{'snap/8':>7}  {'mid<0.15/8':>11}  {'max_tr_diff':>12}  "
        f"{'cont_NLL':>9}  {'tab_NLL':>8}  {'gap':>8}"
    )
    print(hdr)
    print("-" * 92)
    for cr in cell_results:
        print(
            f"{cr['L']:>5.2f}  {cr['sigma']:>6.3f}  {cr['ratio']:>6.2f}  "
            f"{cr['snap_count']:>7d}  {cr['mid_err_pass']:>11d}  "
            f"{cr['max_tr_diff']:>12.3e}  "
            f"{cr['mean_cont_nll']:>9.4f}  {cr['mean_tab_nll']:>8.4f}  {cr['mean_gap']:>8.4f}"
        )
    print("=" * 92)

    # ---------------------------------------------------------------------------
    # P1: No-snap + midpoint proximity
    # ---------------------------------------------------------------------------
    total_snapped = sum(cr["snap_count"] for cr in cell_results)
    # P1a: 0 snapped in all 96 runs
    p1a_pass = (total_snapped == 0)
    # P1b: cells with >= 2 snaps (falsifier threshold)
    cells_ge2_snaps = sum(cr["snap_count"] >= 2 for cr in cell_results)
    # P1c: midpoint proximity >=7/8 in >=11/12 cells
    cells_mid_pass = sum(cr["mid_err_pass"] >= 7 for cr in cell_results)
    p1c_pass = (cells_mid_pass >= 11)

    p1_pass = p1a_pass and p1c_pass

    print()
    print("--- P1: Interpolation / No-Snap ---")
    print(f"  Total snapped runs (96 total): {total_snapped}  => p1a={'PASS' if p1a_pass else 'FAIL'}")
    print(f"  Cells with snap_count >= 2: {cells_ge2_snaps}  (falsifier: any => NEGATIVE)")
    print(f"  Cells with midpoint_err_norm < 0.15 in >=7/8 seeds: {cells_mid_pass}/12  => p1c={'PASS' if p1c_pass else 'FAIL'}")
    print(f"  P1 overall: {'PASS' if p1_pass else 'FAIL'}")

    # ---------------------------------------------------------------------------
    # P2: No-widening (structural)
    # ---------------------------------------------------------------------------
    # All 96 seed-cell pairs must have tr_diff_abs < 1e-9
    max_tr_diff_global = max(cr["max_tr_diff"] for cr in cell_results)
    p2_pass = (max_tr_diff_global < 1e-9)

    print()
    print("--- P2: No-Widening (Structural) ---")
    print(f"  Max |tr_blend - tr_pure| across all 96 pairs: {max_tr_diff_global:.6e}")
    print(f"  Threshold: 1e-9  => P2={'PASS' if p2_pass else 'FAIL'}")

    # ---------------------------------------------------------------------------
    # P3: Cost boundary shape
    # ---------------------------------------------------------------------------
    ratios = np.array([cr["ratio"] for cr in cell_results])
    gaps = np.array([cr["mean_gap"] for cr in cell_results])
    log_ratios = np.log(ratios)

    rho = spearman_rho(log_ratios, gaps)
    p3a_pass = (rho >= 0.8)

    # Gap at largest L/sigma
    max_ratio_idx = int(np.argmax(ratios))
    gap_at_max = gaps[max_ratio_idx]
    p3b_pass = (abs(gap_at_max - LN2) <= 0.15)

    # Gap at smallest L/sigma
    min_ratio_idx = int(np.argmin(ratios))
    gap_at_min = gaps[min_ratio_idx]
    p3c_pass = (gap_at_min < 0.10)

    p3_pass = p3a_pass and p3b_pass and p3c_pass

    print()
    print("--- P3: Cost Boundary Shape ---")
    print(f"  Spearman rho(log(L/sigma), cell_mean_gap) = {rho:.4f}  (need >= 0.8)  => p3a={'PASS' if p3a_pass else 'FAIL'}")
    print(f"  Gap at max L/sigma (ratio={ratios[max_ratio_idx]:.2f}): {gap_at_max:.4f}  ln2={LN2:.4f}  |diff|={abs(gap_at_max-LN2):.4f}  (need within 0.15)  => p3b={'PASS' if p3b_pass else 'FAIL'}")
    print(f"  Gap at min L/sigma (ratio={ratios[min_ratio_idx]:.2f}): {gap_at_min:.4f}  (need < 0.10)  => p3c={'PASS' if p3c_pass else 'FAIL'}")
    print(f"  P3 overall: {'PASS' if p3_pass else 'FAIL'}")

    # ---------------------------------------------------------------------------
    # Falsifier evaluation
    # ---------------------------------------------------------------------------
    falsifier_snap = (cells_ge2_snaps > 0)   # any cell with >= 2/8 snapped
    falsifier_p2 = not p2_pass                 # P2 violated anywhere
    falsifier_p3 = not p3_pass                 # P3 shape fails

    any_falsifier = falsifier_snap or falsifier_p2 or falsifier_p3

    print()
    print("--- Falsifier Evaluation ---")
    print(f"  Snap falsifier (any cell >= 2/8 snapped):    {'TRIGGERED' if falsifier_snap else 'clear'}")
    print(f"  P2 falsifier (tr_diff >= 1e-9 anywhere):     {'TRIGGERED' if falsifier_p2 else 'clear'}")
    print(f"  P3 falsifier (shape fails):                  {'TRIGGERED' if falsifier_p3 else 'clear'}")

    # ---------------------------------------------------------------------------
    # VERDICT
    # ---------------------------------------------------------------------------
    # POSITIVE requires ALL conjuncts: P1 (p1a AND p1c) AND P2 AND P3 (p3a AND p3b AND p3c)
    # NEGATIVE if any falsifier triggered
    # MIXED otherwise

    print()
    print("=" * 92)
    all_pass = p1a_pass and p1c_pass and p2_pass and p3a_pass and p3b_pass and p3c_pass

    if any_falsifier:
        verdict = "NEGATIVE"
        triggered = []
        if falsifier_snap:
            triggered.append(f"snap falsifier (cells_ge2={cells_ge2_snaps})")
        if falsifier_p2:
            triggered.append(f"P2 falsifier (max_tr_diff={max_tr_diff_global:.3e})")
        if falsifier_p3:
            details = []
            if not p3a_pass:
                details.append(f"rho={rho:.4f}<0.8")
            if not p3b_pass:
                details.append(f"gap_max={gap_at_max:.4f} not within 0.15 of ln2")
            if not p3c_pass:
                details.append(f"gap_min={gap_at_min:.4f}>=0.10")
            triggered.append("P3 falsifier (" + "; ".join(details) + ")")
        print(f"VERDICT: {verdict}")
        print(f"  Triggered falsifiers: {'; '.join(triggered)}")
    elif all_pass:
        verdict = "POSITIVE"
        print(f"VERDICT: {verdict}")
        print(
            f"  All conjuncts pass: "
            f"P1a(snaps=0) AND P1c(cells_mid={cells_mid_pass}/12>=11) AND "
            f"P2(max_tr_diff={max_tr_diff_global:.2e}<1e-9) AND "
            f"P3a(rho={rho:.4f}>=0.8) AND P3b(gap_max={gap_at_max:.4f} within 0.15 of ln2) AND "
            f"P3c(gap_min={gap_at_min:.4f}<0.10)"
        )
    else:
        verdict = "MIXED"
        failing = []
        if not p1a_pass:
            failing.append(f"P1a(snaps={total_snapped}>0)")
        if not p1c_pass:
            failing.append(f"P1c(cells_mid={cells_mid_pass}/12<11)")
        if not p2_pass:
            failing.append(f"P2(max_tr_diff={max_tr_diff_global:.2e}>=1e-9)")
        if not p3a_pass:
            failing.append(f"P3a(rho={rho:.4f}<0.8)")
        if not p3b_pass:
            failing.append(f"P3b(gap_max={gap_at_max:.4f})")
        if not p3c_pass:
            failing.append(f"P3c(gap_min={gap_at_min:.4f}>=0.10)")
        print(f"VERDICT: {verdict}")
        print(f"  Failing conjuncts (no falsifier triggered): {'; '.join(failing)}")
    print("=" * 92)
    print(f"ln2 reference = {LN2:.6f}")
    print(f"JSON rows written to: experiments/outputs/exp134_rows.json")


if __name__ == "__main__":
    main()
