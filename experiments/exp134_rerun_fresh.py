"""
Exp 134 rerun — fresh-seed confirmation of the rung-2 surprise + mechanism check.

The Exp 134 P3 falsifier triggered with a SIGN REVERSAL (continuous BEATS tabular on
blends, catastrophically so at large L/sigma). Per the persona rule (rerun a surprise
once) and VALIDATION.md's fresh-seed requirement, this rerun uses seeds 8..15 (never
run) on the identical predeclared grid, plus a mechanism check.

Predicted mechanism (declared before this rerun): the tabular static-state posterior
q_c propto prod_t A[o_t, c] is order-independent; its collapse is driven by the
BINOMIAL COUNT IMBALANCE n0 - n3 between the two blend words — each unit of imbalance
multiplies the A-vs-D odds by (A[0,A]/A[0,D])^1, astronomical at large L/sigma — so
q concentrates on the majority-count corner and pays unbounded NLL on the minority
word. The continuous posterior interpolates instead and degrades only via the
midpoint offset.

Predictions (confirmation iff all): (C1) 0 snapped runs and max |tr_blend - tr_pure|
< 1e-9, as before; (C2) cell-mean NLL gap (cont - tab) negative in all cells with
L/sigma >= 2.86 and Spearman rho(log ratio, gap) <= -0.8 (the reversed sign,
confirmed out-of-sample); (C3) mechanism: in every run at L/sigma >= 5.71, the
tabular argmax state is the majority-count corner whenever n0 != n3, and per-cell
the correlation between |n0 - n3| and tabular NLL is positive.
Falsifier: any of C1-C3 fails on fresh seeds -> the Exp 134 surprise does not
replicate; log as such.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")

from active_loop.continuous import GaussianBelief, predictive_word_logprobs

# ---------------------------------------------------------------------------
# Grid definition (identical to exp134)
# ---------------------------------------------------------------------------
L_VALUES = [0.5, 1.0, 2.0, 4.0]
SIGMA_VALUES = [0.175, 0.35, 0.7]
CELLS = [(L, sigma) for L in L_VALUES for sigma in SIGMA_VALUES]  # 12 cells

SEEDS = list(range(8, 16))   # seeds 8..15 (never run before)
TRAIN_LEN = 200
HOLDOUT_LEN = 200
LN2 = math.log(2.0)

# C2 threshold: cells with L/sigma >= 2.86 must all have negative gap
C2_RATIO_THRESH = 2.86
# C3 threshold: cells with L/sigma >= 5.71 used for mechanism check
C3_RATIO_THRESH = 5.71


# ---------------------------------------------------------------------------
# Spearman rank correlation (reused verbatim from exp134)
# ---------------------------------------------------------------------------

def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation using average ranks for ties, then Pearson on ranks."""
    try:
        from scipy.stats import spearmanr  # type: ignore
        result = spearmanr(x, y)
        return float(result.statistic if hasattr(result, "statistic") else result.correlation)
    except Exception:
        pass

    def _rank(v: np.ndarray) -> np.ndarray:
        n = len(v)
        order = np.argsort(v, kind="stable")
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i + 1
            while j < n and v[order[j]] == v[order[i]]:
                j += 1
            avg = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg
            i = j
        return ranks

    rx = _rank(np.asarray(x, dtype=float))
    ry = _rank(np.asarray(y, dtype=float))
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
    # Same rng scheme as exp134: default_rng(seed*1000 + cell_index)
    rng = np.random.default_rng(seed * 1000 + cell_index)

    # Geometry
    d = 2
    corners = np.array([[0.0, 0.0], [L, 0.0], [0.0, L], [L, L]])  # A=0, B=1, C=2, D=3
    midpoint = np.array([L / 2.0, L / 2.0])
    Sigma_k = (sigma ** 2) * np.eye(d)
    Lambda_k = np.linalg.inv(Sigma_k)

    prior_mu = np.zeros(d)
    prior_Sigma = 4.0 * np.eye(d)

    word_mus = [corners[k] for k in range(4)]
    word_Sigmas = [Sigma_k for _ in range(4)]

    # Streams: blended (words 0,3) train + holdout; pure-A train for C1 tr check
    blend_train = rng.choice([0, 3], size=TRAIN_LEN)
    blend_holdout = rng.choice([0, 3], size=HOLDOUT_LEN)
    pure_train = np.zeros(TRAIN_LEN, dtype=int)

    # Count imbalance
    n0 = int(np.sum(blend_train == 0))
    n3 = int(np.sum(blend_train == 3))

    # --- Continuous blended agent ---
    belief_blend = GaussianBelief(prior_mu, prior_Sigma)
    for w in blend_train:
        belief_blend.observe(corners[w], Lambda_k)
    mu_post = belief_blend.mu
    Sigma_post = belief_blend.Sigma
    tr_blend = belief_blend.trace_sigma

    # Snap check
    dist_to_corners = np.array([np.linalg.norm(mu_post - corners[k]) for k in range(4)])
    dist_to_mid = np.linalg.norm(mu_post - midpoint)
    snapped = int(np.min(dist_to_corners) < dist_to_mid)

    # Continuous blended held-out NLL
    log_probs = predictive_word_logprobs(mu_post, Sigma_post, word_mus, word_Sigmas)
    cont_nll = float(-np.mean([log_probs[w] for w in blend_holdout]))

    # --- Continuous pure-A agent (C1 tr check) ---
    belief_pure = GaussianBelief(prior_mu, prior_Sigma)
    for w in pure_train:
        belief_pure.observe(corners[w], Lambda_k)
    tr_pure = belief_pure.trace_sigma
    tr_diff_abs = abs(tr_blend - tr_pure)

    # --- Tabular twin (blended) ---
    # A_table[k, c] = N(corner_c; corner_k, Sigma_k), column-normalised
    A_raw = np.zeros((4, 4))
    for k in range(4):
        for c in range(4):
            diff = corners[c] - corners[k]
            log_p = -0.5 * diff @ Lambda_k @ diff
            A_raw[k, c] = math.exp(log_p)
    col_sums = A_raw.sum(axis=0)
    A_table = A_raw / col_sums[np.newaxis, :]

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

    # Tabular argmax state
    tab_argmax = int(np.argmax(q_tab))

    # Tabular blended held-out NLL
    tab_nll_terms = []
    q_eval = q_tab.copy()
    for w in blend_holdout:
        pred = A_table[w, :] @ q_eval
        if pred < 1e-300:
            pred = 1e-300
        tab_nll_terms.append(-math.log(pred))
        q_eval = q_eval * A_table[w, :]
        q_sum = q_eval.sum()
        if q_sum < 1e-300:
            q_eval = np.ones(4) / 4.0
        else:
            q_eval = q_eval / q_sum

    tab_nll = float(np.mean(tab_nll_terms))
    nll_gap = cont_nll - tab_nll

    # C3: majority-count corner is 0 if n0>n3 else 3
    majority_corner = 0 if n0 > n3 else 3  # only meaningful when n0 != n3

    return {
        "snapped": snapped,
        "tr_blend": tr_blend,
        "tr_pure": tr_pure,
        "tr_diff_abs": tr_diff_abs,
        "cont_nll": cont_nll,
        "tab_nll": tab_nll,
        "nll_gap": nll_gap,
        "n0": n0,
        "n3": n3,
        "tab_argmax": tab_argmax,
        "majority_corner": majority_corner,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cell_results = []

    for cell_index, (L, sigma) in enumerate(CELLS):
        ratio = L / sigma
        seed_data = []
        for seed in SEEDS:
            r = run_cell_seed(cell_index, seed, L, sigma)
            seed_data.append(r)

        snap_count = sum(r["snapped"] for r in seed_data)
        max_tr_diff = max(r["tr_diff_abs"] for r in seed_data)
        mean_cont_nll = float(np.mean([r["cont_nll"] for r in seed_data]))
        mean_tab_nll = float(np.mean([r["tab_nll"] for r in seed_data]))
        mean_gap = float(np.mean([r["nll_gap"] for r in seed_data]))

        cell_results.append({
            "L": L, "sigma": sigma, "ratio": ratio,
            "snap_count": snap_count,
            "max_tr_diff": max_tr_diff,
            "mean_cont_nll": mean_cont_nll,
            "mean_tab_nll": mean_tab_nll,
            "mean_gap": mean_gap,
            "seed_data": seed_data,
        })

    # ---------------------------------------------------------------------------
    # Print cell table
    # ---------------------------------------------------------------------------
    print("=" * 96)
    print("Exp 134 RERUN — Cell Table (12 cells, 8 seeds 8..15)")
    print("=" * 96)
    hdr = (
        f"{'L':>5}  {'sigma':>6}  {'L/sig':>6}  "
        f"{'snaps/8':>8}  {'max_tr_diff':>12}  "
        f"{'cont_NLL':>9}  {'tab_NLL':>9}  {'gap':>9}"
    )
    print(hdr)
    print("-" * 96)
    for cr in cell_results:
        print(
            f"{cr['L']:>5.2f}  {cr['sigma']:>6.3f}  {cr['ratio']:>6.2f}  "
            f"{cr['snap_count']:>8d}  {cr['max_tr_diff']:>12.3e}  "
            f"{cr['mean_cont_nll']:>9.4f}  {cr['mean_tab_nll']:>9.4f}  {cr['mean_gap']:>9.4f}"
        )
    print("=" * 96)

    # ---------------------------------------------------------------------------
    # C1: No-snap + tr structural check
    # ---------------------------------------------------------------------------
    total_snapped = sum(cr["snap_count"] for cr in cell_results)
    c1_snap_pass = (total_snapped == 0)
    max_tr_diff_global = max(cr["max_tr_diff"] for cr in cell_results)
    c1_tr_pass = (max_tr_diff_global < 1e-9)
    c1_pass = c1_snap_pass and c1_tr_pass

    print()
    print("--- C1: No-snap + tr structural check ---")
    print(f"  Total snapped runs ({len(SEEDS) * len(CELLS)} total): {total_snapped}"
          f"  => snap={'PASS' if c1_snap_pass else 'FAIL'}")
    print(f"  Max |tr_blend - tr_pure| across all runs: {max_tr_diff_global:.6e}"
          f"  (need < 1e-9) => tr={'PASS' if c1_tr_pass else 'FAIL'}")
    print(f"  C1 overall: {'PASS' if c1_pass else 'FAIL'}")

    # ---------------------------------------------------------------------------
    # C2: Reversed sign confirmation
    # ---------------------------------------------------------------------------
    # (a) All cells with L/sigma >= 2.86 must have negative gap (cont < tab)
    high_ratio_cells = [cr for cr in cell_results if cr["ratio"] >= C2_RATIO_THRESH]
    c2a_negatives = [cr["mean_gap"] < 0 for cr in high_ratio_cells]
    c2a_pass = all(c2a_negatives)

    # (b) Spearman rho(log ratio, gap) <= -0.8  (reversed sign vs exp134 P3 which required >= 0.8)
    ratios_arr = np.array([cr["ratio"] for cr in cell_results])
    gaps_arr = np.array([cr["mean_gap"] for cr in cell_results])
    log_ratios_arr = np.log(ratios_arr)
    rho = spearman_rho(log_ratios_arr, gaps_arr)
    c2b_pass = (rho <= -0.8)

    c2_pass = c2a_pass and c2b_pass

    print()
    print("--- C2: Reversed sign (cont BEATS tabular at large L/sigma) ---")
    print(f"  Cells with L/sigma >= {C2_RATIO_THRESH}: {len(high_ratio_cells)}")
    for cr in high_ratio_cells:
        sign_ok = "neg" if cr["mean_gap"] < 0 else "POS(FAIL)"
        print(f"    L={cr['L']:.2f} sigma={cr['sigma']:.3f} ratio={cr['ratio']:.2f}"
              f"  gap={cr['mean_gap']:+.4f}  [{sign_ok}]")
    print(f"  All high-ratio gaps negative: {sum(c2a_negatives)}/{len(high_ratio_cells)}"
          f"  => c2a={'PASS' if c2a_pass else 'FAIL'}")
    print(f"  Spearman rho(log(L/sigma), gap) = {rho:.4f}  (need <= -0.8)"
          f"  => c2b={'PASS' if c2b_pass else 'FAIL'}")
    print(f"  C2 overall: {'PASS' if c2_pass else 'FAIL'}")

    # ---------------------------------------------------------------------------
    # C3: Mechanism check
    # ---------------------------------------------------------------------------
    # For every run at L/sigma >= 5.71: argmax == majority corner (when n0 != n3)
    # Per-cell: Pearson corr between |n0-n3| and tab NLL

    c3_cells = [cr for cr in cell_results if cr["ratio"] >= C3_RATIO_THRESH]
    argmax_hits = 0
    argmax_total = 0   # runs with n0 != n3
    argmax_skipped = 0  # runs with n0 == n3

    print()
    print("--- C3: Mechanism check (argmax + imbalance correlation) ---")
    print(f"  Cells with L/sigma >= {C3_RATIO_THRESH}: {len(c3_cells)}")

    c3_pearson_per_cell = []
    all_pearson_positive = True

    for cr in c3_cells:
        seed_data = cr["seed_data"]
        # Argmax tally
        for r in seed_data:
            if r["n0"] == r["n3"]:
                argmax_skipped += 1
            else:
                argmax_total += 1
                if r["tab_argmax"] == r["majority_corner"]:
                    argmax_hits += 1

        # Per-cell Pearson corr between |n0-n3| and tab NLL
        imbalances = np.array([abs(r["n0"] - r["n3"]) for r in seed_data], dtype=float)
        tab_nlls = np.array([r["tab_nll"] for r in seed_data], dtype=float)

        if imbalances.std() == 0 or tab_nlls.std() == 0:
            pearson_r = float("nan")
        else:
            im_c = imbalances - imbalances.mean()
            tn_c = tab_nlls - tab_nlls.mean()
            denom = math.sqrt(float(np.dot(im_c, im_c)) * float(np.dot(tn_c, tn_c)))
            pearson_r = float(np.dot(im_c, tn_c)) / denom if denom > 0 else float("nan")

        c3_pearson_per_cell.append((cr["L"], cr["sigma"], cr["ratio"], pearson_r))

        if math.isnan(pearson_r) or pearson_r <= 0:
            all_pearson_positive = False

        print(f"    L={cr['L']:.2f} sigma={cr['sigma']:.3f} ratio={cr['ratio']:.2f}"
              f"  Pearson(|n0-n3|, tab_NLL) = {pearson_r:+.4f}")

    c3a_pass = (argmax_total > 0) and (argmax_hits == argmax_total)
    c3b_pass = all_pearson_positive and (len(c3_pearson_per_cell) > 0)
    c3_pass = c3a_pass and c3b_pass

    print(f"  Argmax == majority corner: {argmax_hits}/{argmax_total} runs"
          f"  (skipped n0==n3: {argmax_skipped})"
          f"  => c3a={'PASS' if c3a_pass else 'FAIL'}")
    print(f"  All per-cell Pearson(|n0-n3|, tab_NLL) > 0: {'YES' if all_pearson_positive else 'NO'}"
          f"  => c3b={'PASS' if c3b_pass else 'FAIL'}")
    print(f"  C3 overall: {'PASS' if c3_pass else 'FAIL'}")

    # ---------------------------------------------------------------------------
    # VERDICT
    # ---------------------------------------------------------------------------
    all_confirmed = c1_pass and c2_pass and c3_pass

    print()
    print("=" * 96)
    if all_confirmed:
        print("VERDICT: CONFIRMED")
        print(
            f"  C1: snaps=0, max_tr_diff={max_tr_diff_global:.2e}<1e-9  |  "
            f"C2: all high-ratio gaps negative ({sum(c2a_negatives)}/{len(high_ratio_cells)}), "
            f"rho={rho:.4f}<=-0.8  |  "
            f"C3: argmax={argmax_hits}/{argmax_total}, all Pearson>0"
        )
    else:
        print("VERDICT: NOT-CONFIRMED")
        failing = []
        if not c1_pass:
            details = []
            if not c1_snap_pass:
                details.append(f"snaps={total_snapped}>0")
            if not c1_tr_pass:
                details.append(f"max_tr_diff={max_tr_diff_global:.2e}>=1e-9")
            failing.append("C1(" + "; ".join(details) + ")")
        if not c2_pass:
            details = []
            if not c2a_pass:
                details.append(f"not all high-ratio gaps negative ({sum(c2a_negatives)}/{len(high_ratio_cells)})")
            if not c2b_pass:
                details.append(f"rho={rho:.4f}>-0.8")
            failing.append("C2(" + "; ".join(details) + ")")
        if not c3_pass:
            details = []
            if not c3a_pass:
                details.append(f"argmax_hits={argmax_hits}/{argmax_total}")
            if not c3b_pass:
                details.append("not all Pearson>0")
            failing.append("C3(" + "; ".join(details) + ")")
        print(f"  Failing conjuncts: {'; '.join(failing)}")
    print("=" * 96)
    print(f"ln2 reference = {LN2:.6f}")
    print(f"Seeds used: {SEEDS[0]}..{SEEDS[-1]}")


if __name__ == "__main__":
    main()
