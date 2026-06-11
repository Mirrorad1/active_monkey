"""Exp 162 — N3 rung 1, candidate 3: the WINDOW BLIND SPOT. Hidden-context
alternation SLOWER than the classifier's fixed window makes evaluated
windows context-pure, so the typed diagnosis is persistently wrong — a
learnable structured world is labeled noise/fine — and the failure is
pinned to a theta_N2 dial (the window length itself): the SAME correctness
stream re-windowed at 2x the period recovers STRUCTURAL.

Hypothesis (window-purity arithmetic, near-analytic): with half-period
H = 400 and window W = 200 evaluated every 100 steps, most full windows lie
entirely inside one context. Pure-A windows: error 0 => OK. Pure-B windows:
error ~1.0 with ZERO variance => lag1_rho defined 0 => NOISE (the dangerous
mislabel: irreducible-noise suppresses repair). Only switch-straddling
windows (~1-2 per switch) read STRUCTURAL. The classifier's majority
diagnosis is therefore NOT STRUCTURAL, persistently — while the identical
stream re-windowed at W' = 800 = 2H mixes both contexts in every window
(error ~0.5, lag1_rho >> 0.3) and reads STRUCTURAL everywhere. Unlike the
adaptive channel (Exp 160-161), a fixed window cannot self-heal: the wrong
diagnosis is sustained for the life of the regime.

World R-SLOW (provided): mirro's standard world with StructCmap alternating
base cmap (context A) and a derangement (context B) every H = 400 steps,
starting in A; derangement RANDOMIZED per fork from {[1,2,0], [2,0,1]}
(rng 120_000 + fork_seed). 4000 steps = 5 full cycles.

Design: forks of mirro only (spine never saved), FRESH seeds 58-65, exp155
step-loop semantics. Classifier exactly as Exp 158 (W=200, every 100 steps
once full; error<0.05 OK; rho>0.3 STRUCTURAL; else NOISE). Re-windowed
control: the SAME recorded correctness stream evaluated post-hoc with
W'=800 (every 100 steps once full), identical thresholds. The Exp 157 EWMA
channel rides along UNGATED (its AUROC in this world is a diagnostic).
Ground-truth context per step logged for window-purity accounting.

Preconditions (instrument-grade; failure => "PRECONDITION FAILED", rows
written, no verdict): PC1 ahat_drift < 0.05 per fork; PC2 >= 30 evaluated
W=200 windows per fork; PC3 derangement bites: pooled error rate over
PURE context-B windows >= 0.5 (input validity).

Predeclared properties and falsifiers:
  P1 (the typed diagnosis is systematically wrong): per fork, STRUCTURAL
     fraction over evaluated W=200 windows <= 0.35 AND the majority label
     != STRUCTURAL; both in >= 7/8 forks.
     FALSIFIER F1: pooled STRUCTURAL fraction >= 0.5 (the classifier
     catches the structure after all — candidate dead).
  P2 (the failure is pinned to the window dial): at W' = 800, per-fork
     STRUCTURAL fraction >= 0.7 in >= 7/8 forks.
     FALSIFIER F2: pooled W'=800 STRUCTURAL fraction < 0.5 (the information
     was not recoverable by re-windowing — the pin fails).
  P3 (the dangerous error type is present): pooled NOISE fraction over
     W=200 windows >= 0.2 (the structured world is being called
     irreducible noise, the label that suppresses growth/repair).
     FALSIFIER F3: pooled NOISE fraction <= 0.05.
  VERDICT: POSITIVE (GATE PASSES — the discriminating perturbation EXISTS,
  pinned at theta_N2's window) iff P1, P2, P3 all pass. NEGATIVE iff any
  falsifier fires. Otherwise MIXED. "Not a falsifier" never counts toward
  POSITIVE. A NEGATIVE kills candidate 3 only; candidate 4 (OK-bar-hugging
  error rates) remains.

Ungated diagnostics: full label distribution at W=200 and W'=800;
window-purity accounting (pure-A / pure-B / straddle counts and their label
breakdown); per-fork derangement; EWMA channel AUROC in this world; label
timeline for one fork (fork 58) for the record.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import deque as _deque
from pathlib import Path

import numpy as np

from active_loop.creature import (
    Creature,
    SURPRISE_WINDOW,
)
from active_loop.verdict import write_verdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS = list(range(58, 66))   # fresh seeds — exp155-161 used 0-57

N_STEPS = 4000
N_CHUNKS = 40
CHUNK_SIZE = N_STEPS // N_CHUNKS    # 100

# R-SLOW world parameters
HALF_PERIOD = 400               # H = 400 steps per context
DERANGEMENT_OPTIONS = [[1, 2, 0], [2, 0, 1]]
RSLOW_SEED_OFFSET = 120_000     # rng 120_000 + fork_seed for derangement draw

# Classifier (W=200) parameters — verbatim from Exp 158
CLASSIFIER_WINDOW = 200         # W = 200 (same as SURPRISE_WINDOW)
EVAL_EVERY = 100                # evaluate every 100 steps once window full
ERROR_RATE_OK_THRESH = 0.05     # error_rate < 0.05 => OK
LAG1_RHO_STRUCT_THRESH = 0.3   # lag1_rho > 0.3 => STRUCTURAL (else NOISE)

# Re-windowed control: W' = 800 = 2 * H
REWINDOW_W = 800                # W' = 800 = 2 * HALF_PERIOD

# EWMA channel parameters — verbatim from Exp 157/160
ALPHA = 0.05        # EWMA learning rate
EWMA_INIT = 0.5     # initial EWMA value per cell

# Precondition thresholds
PRECONDITION_AHAT_DRIFT = 0.05      # PC1: ahat_drift < 0.05
PRECONDITION_MIN_WINDOWS = 30       # PC2: >= 30 evaluated W=200 windows per fork
PRECONDITION_PURE_B_ERR_MIN = 0.5   # PC3: pooled error rate over pure-B windows >= 0.5

# P1 thresholds (typed diagnosis is systematically wrong)
P1_STRUCT_FRAC_MAX = 0.35           # per-fork STRUCTURAL fraction <= 0.35
P1_FORKS_MIN = 7                    # >= 7/8 forks with struct_frac <= 0.35 AND majority != STRUCTURAL
F1_POOLED_STRUCT_FRAC_MIN = 0.5     # F1: pooled STRUCTURAL fraction >= 0.5

# P2 thresholds (failure pinned to window dial)
P2_REWINDOW_STRUCT_FRAC_MIN = 0.7   # per-fork W'=800 STRUCTURAL fraction >= 0.7
P2_FORKS_MIN = 7                    # >= 7/8 forks
F2_POOLED_REWINDOW_STRUCT_FRAC_MAX = 0.5  # F2: pooled W'=800 STRUCTURAL fraction < 0.5

# P3 thresholds (dangerous mislabel present)
P3_POOLED_NOISE_FRAC_MIN = 0.2      # pooled NOISE fraction over W=200 windows >= 0.2
F3_POOLED_NOISE_FRAC_MAX = 0.05     # F3: pooled NOISE fraction <= 0.05

# EWMA AUROC minimum class count (ungated)
AUROC_MIN_CLASS = 50


# ---------------------------------------------------------------------------
# StructCmap — verbatim from exp158
# ---------------------------------------------------------------------------

class StructCmap:
    """Hidden-context cmap: alternates every half_period steps.

    Context A = base cmap (standard).
    Context B = derangement applied to cell->color lookup.

    The step counter is maintained externally; caller must set self.current_step
    before each __getitem__ call.
    """

    def __init__(self, base_cmap, n_colors, derangement, half_period=100):
        self.base = list(base_cmap)
        self.n_colors = int(n_colors)
        self.derangement = list(derangement)  # sigma[old_color] = new_color
        self.half_period = int(half_period)
        self.current_step = 0  # caller sets before __getitem__

    def __getitem__(self, s):
        ctx = (self.current_step // self.half_period) % 2
        true_color = self.base[s]
        if ctx == 0:
            return true_color
        else:
            return self.derangement[true_color]

    def __len__(self):
        return len(self.base)


# ---------------------------------------------------------------------------
# make_derangement — verbatim from exp158
# ---------------------------------------------------------------------------

def make_derangement(n_colors: int) -> list[int]:
    """Return a fixed derangement (no fixed points) of range(n_colors).

    For n_colors=3 returns [1, 2, 0].
    For general n: cyclic shift by 1 is always a derangement.
    """
    return [(i + 1) % n_colors for i in range(n_colors)]


# ---------------------------------------------------------------------------
# lag1_rho — verbatim from exp158
# ---------------------------------------------------------------------------

def lag1_rho(seq: np.ndarray) -> float:
    """Lag-1 Pearson autocorrelation of a 1-D array.

    Returns 0.0 when denom < 1e-300 (zero-variance window).
    """
    n = len(seq)
    if n < 2:
        return float("nan")
    x = seq[:-1].astype(float)
    y = seq[1:].astype(float)
    mu_x = x.mean()
    mu_y = y.mean()
    num = ((x - mu_x) * (y - mu_y)).sum()
    denom = math.sqrt(((x - mu_x) ** 2).sum() * ((y - mu_y) ** 2).sum())
    if denom < 1e-300:
        return 0.0
    return float(num / denom)


# ---------------------------------------------------------------------------
# Classifier: evaluate a correctness window and return label + diagnostics
# (verbatim from exp158)
# ---------------------------------------------------------------------------

def classify_window(window: np.ndarray) -> tuple[str, float, float]:
    """Classify a correctness window.

    Returns (label, error_rate, rho) where:
      label: "OK" | "STRUCTURAL" | "NOISE"
      error_rate: 1 - mean(window)
      rho: lag1_rho(window) — computed for every evaluated window regardless
           of label (diagnostic value even for OK-labeled windows).

    lag1_rho of a zero-variance window returns 0.0 (the lag1_rho function
    above already returns 0.0 when denom < 1e-300).
    """
    error_rate = 1.0 - float(window.mean())
    rho = lag1_rho(window)
    if error_rate < ERROR_RATE_OK_THRESH:
        label = "OK"
    elif rho > LAG1_RHO_STRUCT_THRESH:
        label = "STRUCTURAL"
    else:
        label = "NOISE"
    return label, error_rate, rho


# ---------------------------------------------------------------------------
# Mann-Whitney AUROC — verbatim from exp160
# ---------------------------------------------------------------------------

def mannwhitney_auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """AUROC via Mann-Whitney U statistic.

    pos_scores: confidence values where correct_t = 1
    neg_scores: confidence values where correct_t = 0
    Returns float in [0, 1].
    """
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    u = 0.0
    for p in pos_scores:
        u += np.sum(p > neg_scores) + 0.5 * np.sum(p == neg_scores)
    return float(u / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# Determine window purity given context array over a window's span
# ---------------------------------------------------------------------------

def window_purity(ctx_window: np.ndarray) -> str:
    """Return purity category for a window of context labels.

    Returns:
      "pure-A"   — all context values are 0
      "pure-B"   — all context values are 1
      "straddle" — mixed
    """
    if np.all(ctx_window == 0):
        return "pure-A"
    elif np.all(ctx_window == 1):
        return "pure-B"
    else:
        return "straddle"


# ---------------------------------------------------------------------------
# Run one fork in R-SLOW regime — step loop verbatim from exp158 R-STRUCT branch
# ---------------------------------------------------------------------------

def run_fork(
    mirro: Creature,
    fork_seed: int,
    base_cmap: list,
    n_colors: int,
    n_chunks: int = N_CHUNKS,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """Run one fresh fork of mirro in R-SLOW (H=400, randomized derangement).

    Step loop replicates exp155/exp158 run_fork_regime semantics EXACTLY
    (R-STRUCT branch):
      - A_hat pre-step
      - pred_probs = A_hat @ qs
      - o_hat = argmax(pred_probs)
      - conf = ewma[argmax(qs)] (alpha 0.05 init 0.5, updated after correctness)
      - cmap_obj.current_step = t (before observation)
      - obs via StructCmap with half_period=400
      - correct_t = (o_hat == obs)
      - classifier deque(200) evaluated at (t+1)%100==0 when full
      - surprise = -ln(A_hat[obs,:] @ qs) pre-update
      - Bayes update
      - pA Dirichlet count update
      - value accumulation (Exp 26 mechanism)
      - chunk rng action (chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF)
      - move
      - advance qs through B
      - age += 1
      AFTER correctness known:
        ewma[bel_cell] = (1 - ALPHA)*ewma[bel_cell] + ALPHA*correct_t
    """
    assert SURPRISE_WINDOW == 200, (
        f"SURPRISE_WINDOW from creature.py == {SURPRISE_WINDOW}, expected 200"
    )
    assert CLASSIFIER_WINDOW == SURPRISE_WINDOW, (
        f"CLASSIFIER_WINDOW={CLASSIFIER_WINDOW} must equal SURPRISE_WINDOW={SURPRISE_WINDOW}"
    )

    fork_name = f"exp162_s{fork_seed}_RSLOW"
    c = mirro.fork(fork_name)

    n_cells = c.world.n_cells
    n_actions = 4

    # --- Draw derangement for this fork (from {[1,2,0], [2,0,1]}, rng 120_000+fork_seed) ---
    geom_rng = np.random.default_rng(RSLOW_SEED_OFFSET + fork_seed)
    derangement = DERANGEMENT_OPTIONS[int(geom_rng.integers(0, len(DERANGEMENT_OPTIONS)))]

    # --- Build R-SLOW cmap ---
    cmap_obj = StructCmap(
        base_cmap=base_cmap,
        n_colors=n_colors,
        derangement=derangement,
        half_period=HALF_PERIOD,
    )

    B = c.world.transition_matrix()   # (n_cells, n_cells, 4)

    n_total = n_chunks * chunk_size   # 4000

    # A_hat at start for drift check
    A_hat_start = c._A_hat().copy()

    # --- Per-cell EWMA state (verbatim from exp160) ---
    ewma = np.full(n_cells, EWMA_INIT, dtype=np.float64)

    # --- Per-step storage (for post-hoc W'=800 pass and AUROC) ---
    correct_arr = np.empty(n_total, dtype=np.int32)
    context_arr = np.empty(n_total, dtype=np.int32)   # 0=A, 1=B
    conf_arr = np.empty(n_total, dtype=np.float64)    # EWMA conf per step

    # --- Classifier state: rolling correctness window (W=200) ---
    correct_window = _deque(maxlen=CLASSIFIER_WINDOW)

    # --- Per-window records at W=200 ---
    window_labels_200: list[str] = []
    window_error_rates_200: list[float] = []
    window_rhos_200: list[float] = []
    window_at_step_200: list[int] = []   # step t when evaluated

    # --- Step loop ---
    global_step = 0
    eps = 1e-300

    for chunk_idx in range(n_chunks):
        # Deterministic per-chunk seed derived from fork_seed and chunk index
        chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF
        rng = np.random.default_rng(chunk_seed)

        for _step_in_chunk in range(chunk_size):
            t = global_step

            # --- Set cmap step counter BEFORE reading cmap ---
            cmap_obj.current_step = t

            # --- Ground-truth context for this step ---
            ctx_t = int((t // HALF_PERIOD) % 2)   # 0=A, 1=B

            # --- Compute A_hat (pre-step) ---
            A_hat = c._A_hat()   # shape (n_colors, n_cells)

            # --- Predicted observation (BEFORE update) ---
            pred_probs = A_hat @ c.qs   # shape (n_colors,)
            o_hat = int(np.argmax(pred_probs))

            # --- bel_cell and EWMA confidence (pre-update, for EWMA channel) ---
            bel_cell = int(np.argmax(c.qs))
            conf = float(ewma[bel_cell])

            # --- Observe (from StructCmap with cmap_obj.current_step = t) ---
            obs = int(cmap_obj[c.true_pos])

            # --- Correctness ---
            correct_t = 1 if (o_hat == obs) else 0

            # --- Store per-step records ---
            correct_arr[t] = correct_t
            context_arr[t] = ctx_t
            conf_arr[t] = conf

            # --- Append to rolling window ---
            correct_window.append(correct_t)

            # --- Evaluate classifier at every EVAL_EVERY steps once window full ---
            if (t + 1) % EVAL_EVERY == 0 and len(correct_window) == CLASSIFIER_WINDOW:
                win_arr = np.array(correct_window, dtype=np.float64)
                label, err_rate, rho = classify_window(win_arr)
                window_labels_200.append(label)
                window_error_rates_200.append(err_rate)
                window_rhos_200.append(rho)
                window_at_step_200.append(t)

            # --- Surprise (for trajectory consistency; not used in classifier or channel) ---
            p_o = float(A_hat[obs, :] @ c.qs)
            _ = -math.log(p_o + eps)  # computed but not stored

            # --- Belief update: qs ∝ A_hat[obs,:] * qs ---
            likelihood = A_hat[obs, :]
            qs_updated = likelihood * c.qs
            denom = qs_updated.sum()
            if denom > 0:
                qs_updated = qs_updated / denom
            else:
                qs_updated = np.ones(n_cells) / n_cells

            # --- Dirichlet count update: pA[obs, :] += qs_updated ---
            c.pA[obs, :] += qs_updated

            # --- Value accumulation (Exp 26 mechanism) ---
            map_cell = int(np.argmax(qs_updated))
            predicted_obs_dist = A_hat[:, map_cell]
            h_predicted = -np.sum(
                predicted_obs_dist * np.log(predicted_obs_dist + 1e-12)
            )
            predictability_weight = math.exp(-h_predicted)
            c.value_counts[obs] += predictability_weight

            # --- Random action ---
            action = int(rng.integers(0, n_actions))

            # --- Move ---
            c.true_pos = c.world.move(c.true_pos, action)

            # --- Advance qs through B ---
            c.qs = B[:, :, action] @ qs_updated

            c.age_steps += 1

            # --- EWMA update AFTER correctness known (verbatim from exp160) ---
            ewma[bel_cell] = (1.0 - ALPHA) * ewma[bel_cell] + ALPHA * correct_t

            global_step += 1

    # A_hat at end for drift check
    A_hat_end = c._A_hat()
    ahat_drift = float(np.abs(A_hat_end - A_hat_start).max())

    # --- Post-hoc W'=800 pass over the recorded correctness array ---
    window_labels_800: list[str] = []
    window_error_rates_800: list[float] = []
    window_rhos_800: list[float] = []
    window_at_step_800: list[int] = []

    for t in range(REWINDOW_W - 1, n_total):
        if (t + 1) % EVAL_EVERY == 0:
            win_arr = correct_arr[t - REWINDOW_W + 1: t + 1].astype(np.float64)
            label, err_rate, rho = classify_window(win_arr)
            window_labels_800.append(label)
            window_error_rates_800.append(err_rate)
            window_rhos_800.append(rho)
            window_at_step_800.append(t)

    # --- Per-window purity accounting at W=200 ---
    # For each evaluated window, determine purity from context_arr over its span
    purity_200: list[str] = []
    for t in window_at_step_200:
        ctx_win = context_arr[t - CLASSIFIER_WINDOW + 1: t + 1]
        purity_200.append(window_purity(ctx_win))

    # --- Pooled error rate over pure-B windows (PC3) ---
    pure_b_errors: list[float] = []
    for label, err_rate, purity in zip(window_labels_200, window_error_rates_200, purity_200):
        if purity == "pure-B":
            pure_b_errors.append(err_rate)
    pooled_pure_b_err = float(np.mean(pure_b_errors)) if pure_b_errors else float("nan")

    # --- W=200 label aggregates ---
    n_windows_200 = len(window_labels_200)
    n_ok_200 = window_labels_200.count("OK")
    n_noise_200 = window_labels_200.count("NOISE")
    n_struct_200 = window_labels_200.count("STRUCTURAL")
    frac_ok_200 = n_ok_200 / n_windows_200 if n_windows_200 > 0 else float("nan")
    frac_noise_200 = n_noise_200 / n_windows_200 if n_windows_200 > 0 else float("nan")
    frac_struct_200 = n_struct_200 / n_windows_200 if n_windows_200 > 0 else float("nan")

    # --- Majority label at W=200 ---
    if n_windows_200 > 0:
        counts_200 = {"OK": n_ok_200, "NOISE": n_noise_200, "STRUCTURAL": n_struct_200}
        majority_200 = max(counts_200, key=lambda k: counts_200[k])
    else:
        majority_200 = "NONE"

    # --- W'=800 label aggregates ---
    n_windows_800 = len(window_labels_800)
    n_ok_800 = window_labels_800.count("OK")
    n_noise_800 = window_labels_800.count("NOISE")
    n_struct_800 = window_labels_800.count("STRUCTURAL")
    frac_ok_800 = n_ok_800 / n_windows_800 if n_windows_800 > 0 else float("nan")
    frac_noise_800 = n_noise_800 / n_windows_800 if n_windows_800 > 0 else float("nan")
    frac_struct_800 = n_struct_800 / n_windows_800 if n_windows_800 > 0 else float("nan")

    # --- Purity counts and per-purity label breakdown at W=200 ---
    purity_counts = {"pure-A": 0, "pure-B": 0, "straddle": 0}
    purity_label_counts: dict[str, dict[str, int]] = {
        "pure-A": {"OK": 0, "NOISE": 0, "STRUCTURAL": 0},
        "pure-B": {"OK": 0, "NOISE": 0, "STRUCTURAL": 0},
        "straddle": {"OK": 0, "NOISE": 0, "STRUCTURAL": 0},
    }
    for label, purity in zip(window_labels_200, purity_200):
        purity_counts[purity] += 1
        purity_label_counts[purity][label] += 1

    # --- EWMA channel AUROC over all 4000 steps (ungated) ---
    pos_conf = conf_arr[correct_arr == 1]
    neg_conf = conf_arr[correct_arr == 0]
    if len(pos_conf) >= AUROC_MIN_CLASS and len(neg_conf) >= AUROC_MIN_CLASS:
        ewma_auroc = mannwhitney_auroc(pos_conf, neg_conf)
    else:
        ewma_auroc = float("nan")

    return {
        "fork_seed": fork_seed,
        "derangement": derangement,
        "n_total": n_total,
        "ahat_drift": ahat_drift,
        # W=200 aggregates
        "n_windows_200": n_windows_200,
        "n_ok_200": n_ok_200,
        "n_noise_200": n_noise_200,
        "n_struct_200": n_struct_200,
        "frac_ok_200": frac_ok_200,
        "frac_noise_200": frac_noise_200,
        "frac_struct_200": frac_struct_200,
        "majority_200": majority_200,
        # W'=800 aggregates
        "n_windows_800": n_windows_800,
        "n_ok_800": n_ok_800,
        "n_noise_800": n_noise_800,
        "n_struct_800": n_struct_800,
        "frac_ok_800": frac_ok_800,
        "frac_noise_800": frac_noise_800,
        "frac_struct_800": frac_struct_800,
        # Purity accounting
        "purity_counts": purity_counts,
        "purity_label_counts": purity_label_counts,
        "pooled_pure_b_err": pooled_pure_b_err,
        # EWMA channel AUROC (ungated)
        "ewma_auroc": ewma_auroc,
        # Per-window records for timeline (stored for fork_58 printing in smoke)
        "_window_labels_200": window_labels_200,
        "_window_at_step_200": window_at_step_200,
        "_purity_200": purity_200,
        "_window_labels_800": window_labels_800,
        "_window_at_step_800": window_at_step_800,
    }


# ---------------------------------------------------------------------------
# Run all seeds
# ---------------------------------------------------------------------------

def run_all(
    seeds: list[int],
    mirro: Creature,
    base_cmap: list,
    n_colors: int,
    n_chunks: int = N_CHUNKS,
    verbose: bool = True,
) -> list[dict]:
    """Run all seeds; return list of result dicts."""
    rows: list[dict] = []
    for seed in seeds:
        if verbose:
            print(f"  Running seed={seed} regime=R-SLOW ...", flush=True)
        r = run_fork(
            mirro=mirro,
            fork_seed=seed,
            base_cmap=base_cmap,
            n_colors=n_colors,
            n_chunks=n_chunks,
            chunk_size=CHUNK_SIZE,
        )
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Print per-fork table
# ---------------------------------------------------------------------------

def print_fork_table(rows: list[dict]) -> None:
    hdr = (
        f"{'seed':>5}  "
        f"{'derang':>12}  "
        f"{'n_w200':>6}  "
        f"{'fok200':>7}  "
        f"{'fno200':>7}  "
        f"{'fst200':>7}  "
        f"{'maj200':>10}  "
        f"{'n_w800':>6}  "
        f"{'fst800':>7}  "
        f"{'pur_B_err':>9}  "
        f"{'ewma_au':>7}  "
        f"{'drift':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def _fmt(v: float, w: int = 7) -> str:
            if isinstance(v, float) and math.isnan(v):
                return "nan".rjust(w)
            return f"{v:.4f}".rjust(w)

        print(
            f"{r['fork_seed']:>5}  "
            f"{str(r['derangement']):>12}  "
            f"{r['n_windows_200']:>6}  "
            f"{_fmt(r['frac_ok_200'], 7)}  "
            f"{_fmt(r['frac_noise_200'], 7)}  "
            f"{_fmt(r['frac_struct_200'], 7)}  "
            f"{r['majority_200']:>10}  "
            f"{r['n_windows_800']:>6}  "
            f"{_fmt(r['frac_struct_800'], 7)}  "
            f"{_fmt(r['pooled_pure_b_err'], 9)}  "
            f"{_fmt(r['ewma_auroc'], 7)}  "
            f"{r['ahat_drift']:>7.4f}"
        )


# ---------------------------------------------------------------------------
# Print label timeline for one fork (fork 58, smoke diagnostic)
# ---------------------------------------------------------------------------

def print_label_timeline(r: dict) -> None:
    """Print the W=200 label timeline for one fork, with purity annotation."""
    fork_seed = r["fork_seed"]
    labels = r["_window_labels_200"]
    steps = r["_window_at_step_200"]
    purities = r["_purity_200"]
    labels_800 = r["_window_labels_800"]
    steps_800 = r["_window_at_step_800"]

    print(f"Label timeline (W=200) for fork {fork_seed}:")
    print(
        f"  {'step':>6}  {'ctx_phase':>10}  {'purity':>9}  {'label_200':>10}"
    )
    for t, label, purity in zip(steps, labels, purities):
        # Determine context phase span at this eval point (window t-199..t)
        ctx_start = (t - CLASSIFIER_WINDOW + 1) // HALF_PERIOD
        ctx_end = t // HALF_PERIOD
        ctx_phase = f"A{ctx_start//1}-A{ctx_end//1}" if ctx_start == ctx_end else f"A{ctx_start}-A{ctx_end}"
        print(f"  {t:>6}  {ctx_phase:>10}  {purity:>9}  {label:>10}")
    print()

    print(f"Label timeline (W'=800) for fork {fork_seed}:")
    print(f"  {'step':>6}  {'label_800':>10}")
    for t, label in zip(steps_800, labels_800):
        print(f"  {t:>6}  {label:>10}")
    print()


# ---------------------------------------------------------------------------
# Print purity accounting table
# ---------------------------------------------------------------------------

def print_purity_table(rows: list[dict]) -> None:
    """Print window-purity accounting: counts and per-purity label breakdown."""
    print("Window-purity accounting (W=200):")
    hdr = (
        f"{'seed':>5}  "
        f"{'pure-A':>7}  {'A_ok':>5}  {'A_no':>5}  {'A_st':>5}  "
        f"{'pure-B':>7}  {'B_ok':>5}  {'B_no':>5}  {'B_st':>5}  "
        f"{'stradl':>7}  {'S_ok':>5}  {'S_no':>5}  {'S_st':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        pc = r["purity_counts"]
        plc = r["purity_label_counts"]
        print(
            f"{r['fork_seed']:>5}  "
            f"{pc['pure-A']:>7}  "
            f"{plc['pure-A']['OK']:>5}  "
            f"{plc['pure-A']['NOISE']:>5}  "
            f"{plc['pure-A']['STRUCTURAL']:>5}  "
            f"{pc['pure-B']:>7}  "
            f"{plc['pure-B']['OK']:>5}  "
            f"{plc['pure-B']['NOISE']:>5}  "
            f"{plc['pure-B']['STRUCTURAL']:>5}  "
            f"{pc['straddle']:>7}  "
            f"{plc['straddle']['OK']:>5}  "
            f"{plc['straddle']['NOISE']:>5}  "
            f"{plc['straddle']['STRUCTURAL']:>5}"
        )
    print()


# ---------------------------------------------------------------------------
# Precondition check
# ---------------------------------------------------------------------------

def check_preconditions(rows: list[dict]) -> tuple[bool, list[str]]:
    """Check PC1, PC2, PC3.  Returns (all_pass, list_of_failures)."""
    failures: list[str] = []

    for r in rows:
        seed = r["fork_seed"]

        # PC1: ahat_drift < 0.05 per fork
        if r["ahat_drift"] >= PRECONDITION_AHAT_DRIFT:
            failures.append(
                f"PC1 FAIL: seed={seed} ahat_drift={r['ahat_drift']:.4f} "
                f">= {PRECONDITION_AHAT_DRIFT}"
            )

        # PC2: >= 30 evaluated W=200 windows per fork
        if r["n_windows_200"] < PRECONDITION_MIN_WINDOWS:
            failures.append(
                f"PC2 FAIL: seed={seed} n_windows_200={r['n_windows_200']} "
                f"< {PRECONDITION_MIN_WINDOWS}"
            )

    # PC3: pooled error rate over PURE context-B windows >= 0.5 (derangement bites)
    all_pure_b_errs: list[float] = []
    for r in rows:
        plc = r["purity_label_counts"]
        pc = r["purity_counts"]
        if pc["pure-B"] > 0:
            # reconstruct per-window errors from labels: for pure-B, error ~ 1 if NOISE/STRUCTURAL
            # We stored pooled_pure_b_err per fork; aggregate all windows across forks
            pass
    # Collect pooled_pure_b_err values from forks that have pure-B windows
    per_fork_pure_b_errs = [
        r["pooled_pure_b_err"] for r in rows
        if not (isinstance(r["pooled_pure_b_err"], float) and math.isnan(r["pooled_pure_b_err"]))
    ]
    if per_fork_pure_b_errs:
        # Weighted pooled: combine using purity_counts["pure-B"] as weight
        # But we only stored per-fork mean; use unweighted mean of fork means
        pooled_pure_b_err_all = float(np.mean(per_fork_pure_b_errs))
    else:
        pooled_pure_b_err_all = float("nan")

    if math.isnan(pooled_pure_b_err_all) or pooled_pure_b_err_all < PRECONDITION_PURE_B_ERR_MIN:
        failures.append(
            f"PC3 FAIL: pooled pure-B window error rate={pooled_pure_b_err_all:.4f} "
            f"< {PRECONDITION_PURE_B_ERR_MIN} (derangement not biting)"
        )

    all_pass = len(failures) == 0
    return all_pass, failures, pooled_pure_b_err_all  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Evaluate predeclared outcome map (P1/F1, P2/F2, P3/F3)
# ---------------------------------------------------------------------------

def evaluate(rows: list[dict]) -> dict:
    """Evaluate P1/F1, P2/F2, P3/F3; return evaluation dict."""
    n_forks = len(rows)  # 8

    # --- Pooled W=200 fractions ---
    total_windows_200 = sum(r["n_windows_200"] for r in rows)
    total_struct_200 = sum(r["n_struct_200"] for r in rows)
    total_noise_200 = sum(r["n_noise_200"] for r in rows)
    pooled_struct_frac_200 = (
        total_struct_200 / total_windows_200 if total_windows_200 > 0 else float("nan")
    )
    pooled_noise_frac_200 = (
        total_noise_200 / total_windows_200 if total_windows_200 > 0 else float("nan")
    )

    # --- Pooled W'=800 fractions ---
    total_windows_800 = sum(r["n_windows_800"] for r in rows)
    total_struct_800 = sum(r["n_struct_800"] for r in rows)
    pooled_struct_frac_800 = (
        total_struct_800 / total_windows_800 if total_windows_800 > 0 else float("nan")
    )

    # --- P1 / F1: typed diagnosis is systematically wrong ---
    # P1: per fork, STRUCTURAL fraction <= 0.35 AND majority != STRUCTURAL; both in >= 7/8 forks
    p1_forks_pass_count = sum(
        1 for r in rows
        if (
            not math.isnan(r["frac_struct_200"])
            and r["frac_struct_200"] <= P1_STRUCT_FRAC_MAX
            and r["majority_200"] != "STRUCTURAL"
        )
    )
    p1_pass = p1_forks_pass_count >= P1_FORKS_MIN

    # F1: pooled STRUCTURAL fraction >= 0.5
    f1 = (
        not math.isnan(pooled_struct_frac_200)
        and pooled_struct_frac_200 >= F1_POOLED_STRUCT_FRAC_MIN
    )

    # --- P2 / F2: failure pinned to window dial ---
    # P2: at W'=800, per-fork STRUCTURAL fraction >= 0.7 in >= 7/8 forks
    p2_forks_pass_count = sum(
        1 for r in rows
        if (
            not math.isnan(r["frac_struct_800"])
            and r["frac_struct_800"] >= P2_REWINDOW_STRUCT_FRAC_MIN
        )
    )
    p2_pass = p2_forks_pass_count >= P2_FORKS_MIN

    # F2: pooled W'=800 STRUCTURAL fraction < 0.5
    f2 = (
        not math.isnan(pooled_struct_frac_800)
        and pooled_struct_frac_800 < F2_POOLED_REWINDOW_STRUCT_FRAC_MAX
    )

    # --- P3 / F3: dangerous mislabel present ---
    # P3: pooled NOISE fraction over W=200 windows >= 0.2
    p3_pass = (
        not math.isnan(pooled_noise_frac_200)
        and pooled_noise_frac_200 >= P3_POOLED_NOISE_FRAC_MIN
    )

    # F3: pooled NOISE fraction <= 0.05
    f3 = (
        not math.isnan(pooled_noise_frac_200)
        and pooled_noise_frac_200 <= F3_POOLED_NOISE_FRAC_MAX
    )

    # --- Verdict ---
    positive = p1_pass and p2_pass and p3_pass
    negative = f1 or f2 or f3
    if positive:
        verdict_str = "POSITIVE"
    elif negative:
        verdict_str = "NEGATIVE"
    else:
        verdict_str = "MIXED"

    return {
        # W=200 pooled
        "total_windows_200": total_windows_200,
        "total_struct_200": total_struct_200,
        "total_noise_200": total_noise_200,
        "pooled_struct_frac_200": pooled_struct_frac_200,
        "pooled_noise_frac_200": pooled_noise_frac_200,
        # P1 / F1
        "p1_forks_pass_count": p1_forks_pass_count,
        "p1_pass": p1_pass,
        "f1": f1,
        # W'=800 pooled
        "total_windows_800": total_windows_800,
        "total_struct_800": total_struct_800,
        "pooled_struct_frac_800": pooled_struct_frac_800,
        # P2 / F2
        "p2_forks_pass_count": p2_forks_pass_count,
        "p2_pass": p2_pass,
        "f2": f2,
        # P3 / F3
        "p3_pass": p3_pass,
        "f3": f3,
        # Verdict
        "verdict_str": verdict_str,
        "n_forks": n_forks,
    }


# ---------------------------------------------------------------------------
# Write JSONL rows + summary
# ---------------------------------------------------------------------------

def write_json_rows(rows: list[dict], path: Path, ev: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _nan_to_none(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    with path.open("w") as fh:
        for r in rows:
            row: dict = {
                "exp": 162,
                "fork_seed": r["fork_seed"],
                "derangement": r["derangement"],
                "n_total": r["n_total"],
                "ahat_drift": r["ahat_drift"],
                # W=200
                "n_windows_200": r["n_windows_200"],
                "n_ok_200": r["n_ok_200"],
                "n_noise_200": r["n_noise_200"],
                "n_struct_200": r["n_struct_200"],
                "frac_ok_200": _nan_to_none(r["frac_ok_200"]),
                "frac_noise_200": _nan_to_none(r["frac_noise_200"]),
                "frac_struct_200": _nan_to_none(r["frac_struct_200"]),
                "majority_200": r["majority_200"],
                # W'=800
                "n_windows_800": r["n_windows_800"],
                "n_ok_800": r["n_ok_800"],
                "n_noise_800": r["n_noise_800"],
                "n_struct_800": r["n_struct_800"],
                "frac_ok_800": _nan_to_none(r["frac_ok_800"]),
                "frac_noise_800": _nan_to_none(r["frac_noise_800"]),
                "frac_struct_800": _nan_to_none(r["frac_struct_800"]),
                # Purity accounting
                "purity_counts": r["purity_counts"],
                "purity_label_counts": r["purity_label_counts"],
                "pooled_pure_b_err": _nan_to_none(r["pooled_pure_b_err"]),
                # EWMA AUROC (ungated)
                "ewma_auroc": _nan_to_none(r["ewma_auroc"]),
            }
            fh.write(json.dumps(row) + "\n")

        if ev is not None:
            summary: dict = {
                "exp": 162,
                "row_type": "summary",
                "total_windows_200": ev["total_windows_200"],
                "pooled_struct_frac_200": _nan_to_none(ev["pooled_struct_frac_200"]),
                "pooled_noise_frac_200": _nan_to_none(ev["pooled_noise_frac_200"]),
                "p1_forks_pass_count": ev["p1_forks_pass_count"],
                "p1_pass": ev["p1_pass"],
                "f1": ev["f1"],
                "total_windows_800": ev["total_windows_800"],
                "pooled_struct_frac_800": _nan_to_none(ev["pooled_struct_frac_800"]),
                "p2_forks_pass_count": ev["p2_forks_pass_count"],
                "p2_pass": ev["p2_pass"],
                "f2": ev["f2"],
                "p3_pass": ev["p3_pass"],
                "f3": ev["f3"],
                "verdict": ev["verdict_str"],
            }
            fh.write(json.dumps(summary) + "\n")


# ---------------------------------------------------------------------------
# Print verdict block (gate-style, verbatim from exp160)
# ---------------------------------------------------------------------------

def print_verdict_block(ev: dict, rows: list[dict]) -> None:
    n = ev["n_forks"]

    def _fmt_frac(v: float) -> str:
        if math.isnan(v):
            return "nan"
        return f"{v:.4f}"

    print("=" * 80)
    print("PREDECLARED OUTCOME MAP")
    print("=" * 80)
    print()

    # Per-fork W=200 fractions
    struct_fracs_200 = [
        f"{r['frac_struct_200']:.4f}" if not math.isnan(r["frac_struct_200"]) else "nan"
        for r in rows
    ]
    noise_fracs_200 = [
        f"{r['frac_noise_200']:.4f}" if not math.isnan(r["frac_noise_200"]) else "nan"
        for r in rows
    ]
    majorities = [r["majority_200"] for r in rows]

    print(f"P1 (typed diagnosis is systematically wrong at W=200):")
    print(f"  STRUCTURAL frac per fork: [{', '.join(struct_fracs_200)}]")
    print(f"  Majority label per fork:  [{', '.join(majorities)}]")
    print(f"  Forks with struct_frac <= {P1_STRUCT_FRAC_MAX} AND majority != STRUCTURAL: "
          f"{ev['p1_forks_pass_count']}/{n} (need >={P1_FORKS_MIN})")
    print(f"  Pooled STRUCTURAL frac (W=200): {_fmt_frac(ev['pooled_struct_frac_200'])}")
    print(f"  FALSIFIER F1: pooled struct frac >= {F1_POOLED_STRUCT_FRAC_MIN}: "
          f"{'YES — F1 FIRES' if ev['f1'] else 'no'}")
    p1_status = "PASS" if ev["p1_pass"] else ("FALSIFIER F1" if ev["f1"] else "MIXED (not P1, not F1)")
    print(f"  => P1: {p1_status}")
    print()

    # Per-fork W'=800 fractions
    struct_fracs_800 = [
        f"{r['frac_struct_800']:.4f}" if not math.isnan(r["frac_struct_800"]) else "nan"
        for r in rows
    ]

    print(f"P2 (failure pinned to window dial — W'=800 recovers STRUCTURAL):")
    print(f"  STRUCTURAL frac (W'=800) per fork: [{', '.join(struct_fracs_800)}]")
    print(f"  Forks with W'=800 struct_frac >= {P2_REWINDOW_STRUCT_FRAC_MIN}: "
          f"{ev['p2_forks_pass_count']}/{n} (need >={P2_FORKS_MIN})")
    print(f"  Pooled STRUCTURAL frac (W'=800): {_fmt_frac(ev['pooled_struct_frac_800'])}")
    print(f"  FALSIFIER F2: pooled W'=800 struct frac < {F2_POOLED_REWINDOW_STRUCT_FRAC_MAX}: "
          f"{'YES — F2 FIRES' if ev['f2'] else 'no'}")
    p2_status = "PASS" if ev["p2_pass"] else ("FALSIFIER F2" if ev["f2"] else "MIXED (not P2, not F2)")
    print(f"  => P2: {p2_status}")
    print()

    print(f"P3 (dangerous mislabel present — structured world called NOISE):")
    print(f"  NOISE frac per fork: [{', '.join(noise_fracs_200)}]")
    print(f"  Pooled NOISE frac (W=200): {_fmt_frac(ev['pooled_noise_frac_200'])} "
          f"(need >={P3_POOLED_NOISE_FRAC_MIN})")
    print(f"  FALSIFIER F3: pooled NOISE frac <= {F3_POOLED_NOISE_FRAC_MAX}: "
          f"{'YES — F3 FIRES' if ev['f3'] else 'no'}")
    p3_status = "PASS" if ev["p3_pass"] else ("FALSIFIER F3" if ev["f3"] else "MIXED (not P3, not F3)")
    print(f"  => P3: {p3_status}")
    print()

    # --- Conjunct summary + VERDICT (gate-style from exp160) ---
    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    print(f"P1 (struct_frac_200 <= {P1_STRUCT_FRAC_MAX} AND majority != STRUCTURAL "
          f"in >={P1_FORKS_MIN}/{n} forks): {p1_status}")
    print(f"P2 (struct_frac_800 >= {P2_REWINDOW_STRUCT_FRAC_MIN} "
          f"in >={P2_FORKS_MIN}/{n} forks): {p2_status}")
    print(f"P3 (pooled NOISE frac >= {P3_POOLED_NOISE_FRAC_MIN}): {p3_status}")
    print()
    if ev["verdict_str"] == "POSITIVE":
        print("VERDICT: POSITIVE")
        print("GATE PASSES — the discriminating perturbation EXISTS, pinned at theta_N2's window")
    elif ev["verdict_str"] == "NEGATIVE":
        print("VERDICT: NEGATIVE")
        print("GATE: this candidate FAILED")
    else:
        print("VERDICT: MIXED")
    print()
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 162 — N3 rung 1, candidate 3: the WINDOW BLIND SPOT"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Smoke run: seed=[58] only, full 4000 steps, "
            "prints everything including fork-58 label timeline. "
            "No verdict written."
        ),
    )
    args = parser.parse_args()

    smoke = args.smoke
    seeds = [58] if smoke else SEEDS

    print("=" * 80)
    print("Exp 162 — N3 rung 1, candidate 3: the WINDOW BLIND SPOT")
    print("          R-SLOW world: H=400, W=200 classifier vs W'=800 re-windowed control")
    print("=" * 80)
    print()
    print(f"SURPRISE_WINDOW={SURPRISE_WINDOW}  CLASSIFIER_WINDOW={CLASSIFIER_WINDOW}  "
          f"[must be equal; asserted]")
    assert CLASSIFIER_WINDOW == SURPRISE_WINDOW
    print(f"EVAL_EVERY={EVAL_EVERY}  ERROR_RATE_OK_THRESH={ERROR_RATE_OK_THRESH}  "
          f"LAG1_RHO_STRUCT_THRESH={LAG1_RHO_STRUCT_THRESH}")
    print(f"HALF_PERIOD={HALF_PERIOD}  REWINDOW_W={REWINDOW_W} (=2*H)  "
          f"RSLOW_SEED_OFFSET={RSLOW_SEED_OFFSET}")
    print(f"ALPHA={ALPHA}  EWMA_INIT={EWMA_INIT}  (EWMA channel, ungated)")
    print(f"Seeds: {seeds}  N_CHUNKS={N_CHUNKS}  "
          f"CHUNK_SIZE={CHUNK_SIZE}  N_STEPS={N_CHUNKS * CHUNK_SIZE}")
    print()
    print(f"Window-purity arithmetic (H={HALF_PERIOD}, W={CLASSIFIER_WINDOW}, "
          f"eval every {EVAL_EVERY}):")
    print(f"  Pure-A windows (steps entirely in ctx A): error~0 => OK")
    print(f"  Pure-B windows (steps entirely in ctx B): error~1, var~0 => rho=0 => NOISE")
    print(f"  Straddle windows (~1-2 per switch): mixed error, rho>0 => STRUCTURAL")
    print(f"  W'={REWINDOW_W} = 2*H spans both contexts => error~0.5, rho>>0 => STRUCTURAL")
    print()

    # --- Load mirro spine (read-only) ---
    mirro = Creature.load("creature/state/mirro")
    print(f"Loaded mirro: age={mirro.age_steps}, world={mirro.world.rows}x{mirro.world.cols}, "
          f"n_colors={mirro.world.n_colors}, n_cells={mirro.world.n_cells}")
    base_cmap = list(mirro.world.cmap)
    n_colors = mirro.world.n_colors
    print(f"Base cmap: {base_cmap}")
    print()

    # --- Run ---
    rows = run_all(
        seeds=seeds,
        mirro=mirro,
        base_cmap=base_cmap,
        n_colors=n_colors,
        n_chunks=N_CHUNKS,
        verbose=True,
    )
    print()

    # --- Per-fork table ---
    print("Per-fork table:")
    print_fork_table(rows)
    print()

    # --- Window-purity accounting ---
    print_purity_table(rows)

    # --- EWMA channel AUROC (ungated diagnostic) ---
    print("Ungated diagnostics — EWMA channel AUROC in R-SLOW world:")
    for r in rows:
        au = r["ewma_auroc"]
        au_str = f"{au:.4f}" if not math.isnan(au) else "nan"
        print(f"  seed={r['fork_seed']}  derangement={r['derangement']}  "
              f"ewma_auroc={au_str}  "
              f"frac_ok={r['frac_ok_200']:.4f}  "
              f"frac_noise={r['frac_noise_200']:.4f}  "
              f"frac_struct={r['frac_struct_200']:.4f}  "
              f"(W=200 majority={r['majority_200']})")
    print()

    # --- Fork 58 label timeline (always printed — smoke or full) ---
    fork58_rows = [r for r in rows if r["fork_seed"] == 58]
    if fork58_rows:
        print("Fork 58 label timeline (for the record):")
        print_label_timeline(fork58_rows[0])

    # --- Smoke exit ---
    if smoke:
        print("SMOKE ONLY — no verdict")
        return

    # --- Precondition check ---
    pc_pass, pc_failures, pooled_pure_b_err_all = check_preconditions(rows)
    if pc_failures:
        print("PRECONDITION FAILED:")
        for f in pc_failures:
            print(f"  {f}")
        print()
        print("PRECONDITION FAILED — no verdict.")
        out_rows_path = Path(__file__).parent / "outputs" / "exp162_rows.json"
        write_json_rows(rows, out_rows_path, ev=None)
        print(f"JSON rows written to {out_rows_path} (no verdict)")
        return

    print(f"Preconditions: all PASS  (pooled pure-B error={pooled_pure_b_err_all:.4f})")
    print()

    # --- Evaluate predeclared outcome map ---
    ev = evaluate(rows)
    n = ev["n_forks"]

    print_verdict_block(ev, rows)

    # --- Write JSON rows ---
    out_rows_path = Path(__file__).parent / "outputs" / "exp162_rows.json"
    write_json_rows(rows, out_rows_path, ev=ev)
    print(f"JSON rows written to {out_rows_path}")

    # --- Write verdict JSON ---
    def _fmt_frac(v: float) -> str:
        if math.isnan(v):
            return "nan"
        return f"{v:.4f}"

    arms = {
        "P1_diagnosis_wrong": {
            "pass": bool(ev["p1_pass"]),
            "reason": (
                f"W=200 struct_frac <= {P1_STRUCT_FRAC_MAX} AND majority != STRUCTURAL "
                f"in {ev['p1_forks_pass_count']}/{n} forks (need {P1_FORKS_MIN}); "
                f"pooled struct frac={_fmt_frac(ev['pooled_struct_frac_200'])}; "
                f"F1 fired={ev['f1']}"
            ),
        },
        "P2_window_pinned": {
            "pass": bool(ev["p2_pass"]),
            "reason": (
                f"W'=800 struct_frac >= {P2_REWINDOW_STRUCT_FRAC_MIN} "
                f"in {ev['p2_forks_pass_count']}/{n} forks (need {P2_FORKS_MIN}); "
                f"pooled W'=800 struct frac={_fmt_frac(ev['pooled_struct_frac_800'])}; "
                f"F2 fired={ev['f2']}"
            ),
        },
        "P3_dangerous_mislabel": {
            "pass": bool(ev["p3_pass"]),
            "reason": (
                f"Pooled NOISE frac (W=200)={_fmt_frac(ev['pooled_noise_frac_200'])} "
                f"(need >={P3_POOLED_NOISE_FRAC_MIN}); "
                f"F3 fired={ev['f3']}"
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp162_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp162",
        arms=arms,
        verdict=ev["verdict_str"],
        halted=False,
        notes=(
            "N3 rung 1 candidate 3: window blind spot. R-SLOW world with H=400 "
            "(half-period > W=200 classifier window). Pure context-B windows read "
            "NOISE (zero variance => rho=0 => dangerous mislabel). The failure is "
            "pinned to theta_N2's window dial: re-windowing at W'=800=2H recovers "
            "STRUCTURAL. Fixed window cannot self-heal unlike the adaptive EWMA channel. "
            "POSITIVE => GATE PASSES, discriminating perturbation exists. "
            "NEGATIVE kills candidate 3 only; candidate 4 remains."
        ),
    )
    print(f"Verdict JSON written to {verdict_path}")


if __name__ == "__main__":
    main()
