"""Exp 161 — N3 rung 1, candidate 2: the LAG-MATCHED swap. At swap tempos
matched to the EWMA's own relearn time, every swap — including returns to
the original placement — inverts a freshly re-learned map, so the
anti-calibration that decayed at T=250 (Exp 160) should be SUSTAINED.

Hypothesis (mechanism-derived from Exp 160's segment traces): Exp 160's
fast alternation failed because the EWMA never relearned between swaps —
P0-return segments re-validated the old map (AUROC 0.72-0.80) and repeated
fast swaps blurred it to uninformative (pooled 0.5408). At T_swap ~ relearn
time, the map converts to the CURRENT placement within each segment, so the
next swap (either direction) opens with the Exp 160 first-swap-class
inversion (AUROC ~0.22). Two declared arms: T_swap = 500 and T_swap = 750.

World R-SWAP-LAG (provided): identical to Exp 160's R-SWAP except the swap
period — random 13-of-25 noisy placement P0 (per-fork rng 110_000 +
fork_seed; p_true 0.55, noise rng 90_000 + fork_seed); P1 = complement;
burn-in steps 0-999 on P0; then P0 <-> P1 swaps every T_swap steps for
steps 1000-3999 (T=500: 6 swaps; T=750: 4 swaps). 4000 steps total.

Design: forks of mirro only (spine never saved), FRESH seeds 50-57, both
arms run on the same seeds (independent forks per arm), exp155 step-loop
semantics; N2 channel / oracle channel / classifier exactly as Exp 160.
Gated AUROCs use POST-BURN-IN trials (steps 1000-3999).

Preconditions per arm (instrument-grade; failure => "PRECONDITION FAILED",
rows written, no verdict): PC1 >= 50 correct AND >= 50 incorrect
post-burn-in per fork; PC2 ahat_drift < 0.05 per fork; PC3 burn-in N2 AUROC
(steps 200-999) > 0.6 per fork in >= 7/8 forks.

Predeclared properties and falsifiers:
  P1 (sustained deception in at least one arm): in arm T=500 OR arm T=750,
     post-burn-in N2 AUROC pooled <= 0.45 AND per-fork < 0.5 in >= 7/8.
     FALSIFIER F1: BOTH arms pooled >= 0.5 (candidate 2 dead). Pooled
     values between 0.45 and 0.5 in the best arm with neither P1 nor F1
     met => MIXED.
  P2 (discrimination possible): oracle pooled AUROC >= 0.7 in every arm
     that passes P1 (if no arm passes P1, evaluate on both arms for the
     record). FALSIFIER F2: oracle pooled < 0.6 in BOTH arms.
  VERDICT: POSITIVE (GATE PASSES — the discriminating perturbation EXISTS)
  iff P1 and P2 both pass. NEGATIVE iff F1 or F2 fires. Otherwise MIXED.
  "Not a falsifier" never counts toward POSITIVE. A NEGATIVE kills
  candidate 2 only; candidates 3 (period >= classifier window) and 4
  (OK-bar-hugging error rates) remain.

Ungated diagnostics: per-segment N2 AUROC traces per arm (the mechanism
prediction: deceived openings in BOTH directions, not only the first swap);
fraction of segments with AUROC < 0.5 per arm (Exp 160 comparison value:
decaying — only early segments); classifier post-burn-in label fractions
(expectation: NOISE-dominant); terminal EWMA-vs-P0 correlation per arm.
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

SEEDS = list(range(50, 58))   # fresh seeds — exp155-160 used 0-49

ARMS = [500, 750]   # T_swap values for the two declared arms

N_STEPS = 4000
N_CHUNKS = 40
CHUNK_SIZE = N_STEPS // N_CHUNKS    # 100

# Burn-in boundary
BURN_IN_END = 1000          # steps 0-999 are burn-in; post-burn-in = 1000-3999
BURN_IN_START_EVAL = 200    # PC3 burn-in AUROC evaluated on steps 200-999

# Classifier window / evaluation parameters (verbatim from exp158/exp159)
CLASSIFIER_WINDOW = 200       # W = 200 (same as SURPRISE_WINDOW)
EVAL_EVERY = 100              # evaluate once window is full, every 100 steps
ERROR_RATE_OK_THRESH = 0.05   # error_rate < 0.05 => OK
LAG1_RHO_STRUCT_THRESH = 0.3  # lag1_rho > 0.3 => STRUCTURAL (else NOISE)

# EWMA channel parameters (verbatim from exp157/exp159)
ALPHA = 0.05       # EWMA learning rate
EWMA_INIT = 0.5    # initial EWMA value per cell

# R-SWAP-LAG world parameters
P_TRUE_NOISY = 0.55          # p_true for noisy cells
N_NOISY_P0 = 13              # number of noisy cells in P0 (P1 has 25-13=12)

# Seed offsets for R-SWAP-LAG
SWAP_PLACE_SEED_OFFSET = 110_000   # per-fork rng for drawing P0: 110_000 + fork_seed
SWAP_NOISE_SEED_OFFSET = 90_000    # per-fork noise rng: 90_000 + fork_seed

# Swap schedule
SWAP_PHASE_START = 1000      # first post-burn-in segment starts at step 1000
# T_swap is per-arm (see ARMS); segment index = (t - 1000) // T_swap for t >= 1000
# NOTE on swap convention (same as Exp 160):
#   phase_idx = (step - 1000) // T_swap  for step >= 1000
#   noisy_set = P1 if (phase_idx % 2 == 0) else P0
#   => the FIRST post-burn-in segment (phase_idx=0) uses P1 (inverted from burn-in)
#   => subsequent segments alternate P0/P1 at the arm's T_swap tempo

# Precondition thresholds
PRECONDITION_MIN_CLASS = 50          # PC1: >= 50 correct AND incorrect post-burn-in per fork
PRECONDITION_AHAT_DRIFT = 0.05       # PC2: max abs A_hat change < 0.05 per fork
PC3_BURNIN_AUROC_THRESH = 0.6        # PC3: burn-in AUROC > 0.6 per fork
PC3_BURNIN_FORKS_MIN = 7             # PC3: >= 7/8 forks must pass burn-in AUROC check

# P1 thresholds (sustained deception in at least one arm)
P1_POOLED_MAX = 0.45                 # P1: post-burn-in pooled N2 AUROC <= 0.45
P1_PER_FORK_THRESH = 0.5             # P1: per-fork threshold
P1_FORKS_BELOW_HALF_MIN = 7          # P1: >= 7/8 forks with AUROC < 0.5
F1_POOLED_MIN = 0.5                  # F1: BOTH arms pooled >= 0.5 => F1 fires

# P2 thresholds (discrimination possible via oracle)
P2_ORACLE_POOLED_MIN = 0.7           # P2: oracle pooled AUROC >= 0.7
F2_ORACLE_POOLED_MAX = 0.6           # F2: oracle pooled < 0.6 in BOTH arms => F2 fires


def _n_segments(T_swap: int) -> int:
    """Number of post-burn-in T_swap-step segments in steps 1000-3999."""
    # steps 1000-3999 = 3000 steps
    return 3000 // T_swap


# ---------------------------------------------------------------------------
# SwapPlaceNoiseCmap — reliability-swap world (lag-matched variant)
# ---------------------------------------------------------------------------

class SwapPlaceNoiseCmap:
    """Color map wrapper implementing the R-SWAP-LAG reliability-swap world.

    P0: 13 cells drawn uniformly w/o replacement (per-fork rng 110_000+fork_seed).
    P1: the complementary 12 cells.

    Swap schedule (exact):
      - steps 0-999 (burn-in): noisy set = P0 (EWMA learns P0's map)
      - steps 1000+: the FIRST post-burn-in segment uses P1 (same convention as
        Exp 160, so phase_idx=0 => P1). Formally, for step >= 1000:
          phase_idx = (step - 1000) // T_swap
          noisy_set = P1 if (phase_idx % 2 == 0) else P0
        At T_swap=500: 6 swaps (segments); at T_swap=750: 4 swaps (segments).

    Caller must set self.current_step before each __getitem__ call.

    The noise rng is private (seed 90_000+fork_seed); trajectory is untouched.
    """

    def __init__(self, base_cmap, n_colors, noisy_cells_p0: list[int],
                 p_true: float, noise_seed: int, T_swap: int):
        self.base = list(base_cmap)
        self.n_colors = int(n_colors)
        self.noisy_set_P0 = set(noisy_cells_p0)
        # P1 = complement of P0 over range(25)
        all_cells = set(range(len(base_cmap)))
        self.noisy_set_P1 = all_cells - self.noisy_set_P0
        self.p_true = float(p_true)
        self.rng = np.random.default_rng(noise_seed)
        self.current_step = 0   # caller sets before __getitem__
        self.T_swap = int(T_swap)

    def _current_noisy_set(self) -> set:
        """Return the currently active noisy set based on current_step."""
        t = self.current_step
        if t < SWAP_PHASE_START:
            return self.noisy_set_P0
        phase_idx = (t - SWAP_PHASE_START) // self.T_swap
        # phase_idx=0 => P1 (first swap, inverted from burn-in), phase_idx=1 => P0, alternating
        if phase_idx % 2 == 0:
            return self.noisy_set_P1
        else:
            return self.noisy_set_P0

    def is_noisy(self, cell: int) -> bool:
        """True iff cell is currently in the active noisy set."""
        return cell in self._current_noisy_set()

    def current_reliability(self, cell: int) -> float:
        """Oracle: true current reliability (p_true if noisy, 1.0 if clean)."""
        if self.is_noisy(cell):
            return self.p_true
        return 1.0

    def __getitem__(self, s: int) -> int:
        true = self.base[s]
        if self.is_noisy(s):
            if self.rng.random() < self.p_true:
                return true
            others = [c for c in range(self.n_colors) if c != true]
            return int(self.rng.choice(others))
        return true

    def __len__(self) -> int:
        return len(self.base)


# ---------------------------------------------------------------------------
# lag1_rho — verbatim from exp158/exp159
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
# (verbatim from exp158/exp159)
# ---------------------------------------------------------------------------

def classify_window(window: np.ndarray) -> tuple[str, float, float]:
    """Classify a correctness window.

    Returns (label, error_rate, rho) where:
      label: "OK" | "STRUCTURAL" | "NOISE"
      error_rate: 1 - mean(window)
      rho: lag1_rho(window)
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
# Mann-Whitney AUROC — verbatim from exp157/exp159
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


def pooled_auroc(conf_list: list[np.ndarray],
                 correct_list: list[np.ndarray]) -> float:
    """Compute AUROC on all trials concatenated across forks."""
    all_conf = np.concatenate(conf_list)
    all_correct = np.concatenate(correct_list)
    pos = all_conf[all_correct == 1]
    neg = all_conf[all_correct == 0]
    return mannwhitney_auroc(pos, neg)


# ---------------------------------------------------------------------------
# Draw P0 cell list for a given fork seed
# ---------------------------------------------------------------------------

def draw_p0_cells(fork_seed: int) -> list[int]:
    """Draw the P0 noisy cell list (13 cells) using rng 110_000 + fork_seed."""
    rng = np.random.default_rng(SWAP_PLACE_SEED_OFFSET + fork_seed)
    cells_arr = rng.choice(25, size=N_NOISY_P0, replace=False)
    return sorted(int(x) for x in cells_arr)


# ---------------------------------------------------------------------------
# Run one fork — step loop verbatim from exp160, generalised for T_swap arm
# ---------------------------------------------------------------------------

def run_fork(
    mirro: Creature,
    fork_seed: int,
    T_swap: int,
    base_cmap: list,
    n_colors: int,
    n_chunks: int = N_CHUNKS,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """Run one fresh fork of mirro in the R-SWAP-LAG regime for arm T_swap.

    Step loop replicates exp155/exp158/exp159/exp160 run_fork semantics EXACTLY:
      - A_hat pre-step
      - pred_probs = A_hat @ qs
      - o_hat = argmax(pred_probs)
      - bel_cell = argmax(qs) pre-update
      - conf_n2 = ewma[bel_cell] (EWMA channel, pre-update)
      - cmap.current_step = t (before observation)
      - conf_oracle = cmap.current_reliability(c.true_pos) (ORACLE, pre-update)
      - obs from cmap_obj
      - correct_t = (o_hat == obs)
      - append to classifier window; evaluate at (t+1) % 100 == 0 when full
      - surprise = -ln(A_hat[obs,:] @ qs) pre-update
      - Bayes update
      - pA Dirichlet count update
      - value accumulation (Exp 26 mechanism)
      - random action from per-chunk rng
        (chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF)
      - move
      - advance qs through B
      - age += 1
      AFTER correctness known:
        ewma[bel_cell] = (1-ALPHA)*ewma[bel_cell] + ALPHA*correct_t
    """
    assert SURPRISE_WINDOW == 200, (
        f"SURPRISE_WINDOW from creature.py == {SURPRISE_WINDOW}, expected 200"
    )
    assert CLASSIFIER_WINDOW == SURPRISE_WINDOW, (
        f"CLASSIFIER_WINDOW={CLASSIFIER_WINDOW} must equal SURPRISE_WINDOW={SURPRISE_WINDOW}"
    )

    fork_name = f"exp161_s{fork_seed}_T{T_swap}"
    c = mirro.fork(fork_name)

    n_cells = c.world.n_cells
    n_actions = 4

    # --- Build R-SWAP-LAG cmap for this arm ---
    p0_cells = draw_p0_cells(fork_seed)
    noise_seed = SWAP_NOISE_SEED_OFFSET + fork_seed
    cmap_obj = SwapPlaceNoiseCmap(
        base_cmap=base_cmap,
        n_colors=n_colors,
        noisy_cells_p0=p0_cells,
        p_true=P_TRUE_NOISY,
        noise_seed=noise_seed,
        T_swap=T_swap,
    )

    B = c.world.transition_matrix()   # (n_cells, n_cells, 4)

    n_total = n_chunks * chunk_size

    # A_hat at start for drift check
    A_hat_start = c._A_hat().copy()

    # --- Per-cell EWMA state (verbatim from exp157/exp159) ---
    ewma = np.full(n_cells, EWMA_INIT, dtype=np.float64)

    # --- Per-step storage ---
    conf_n2_arr = np.empty(n_total, dtype=np.float64)
    conf_oracle_arr = np.empty(n_total, dtype=np.float64)
    correct_arr = np.empty(n_total, dtype=np.int32)
    in_burnin_arr = np.zeros(n_total, dtype=bool)     # True for t < BURN_IN_END
    segment_arr = np.full(n_total, -1, dtype=np.int32)  # -1 for burn-in steps

    # --- Classifier state: rolling correctness window (verbatim from exp158/exp159) ---
    correct_window = _deque(maxlen=CLASSIFIER_WINDOW)

    # --- Per-window records (ALL steps; post-burn-in filter applied in aggregation) ---
    window_labels: list[str] = []
    window_error_rates: list[float] = []
    window_rhos: list[float] = []
    window_at_step: list[int] = []   # step t when this window was evaluated

    # --- Post-burn-in accumulators for precondition check ---
    n_correct_post = 0
    n_incorrect_post = 0

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

            # --- Compute A_hat (pre-step) ---
            A_hat = c._A_hat()   # shape (n_colors, n_cells)

            # --- Predicted observation (BEFORE update) ---
            pred_probs = A_hat @ c.qs   # shape (n_colors,)
            o_hat = int(np.argmax(pred_probs))

            # --- bel_cell and N2 EWMA channel confidence (BEFORE update) ---
            bel_cell = int(np.argmax(c.qs))
            conf_n2 = float(ewma[bel_cell])

            # --- Oracle channel (reads generator, NOT creature) ---
            true_pos_t = c.true_pos
            conf_oracle = cmap_obj.current_reliability(true_pos_t)

            # --- Observe (from R-SWAP-LAG cmap) ---
            obs = int(cmap_obj[true_pos_t])

            # --- Correctness ---
            correct_t = 1 if (o_hat == obs) else 0

            # --- Store per-step records ---
            conf_n2_arr[t] = conf_n2
            conf_oracle_arr[t] = conf_oracle
            correct_arr[t] = correct_t
            is_burnin = t < BURN_IN_END
            in_burnin_arr[t] = is_burnin
            if not is_burnin:
                # segment index = (t - 1000) // T_swap for t >= 1000
                seg_idx = (t - BURN_IN_END) // T_swap
                segment_arr[t] = seg_idx
                if correct_t == 1:
                    n_correct_post += 1
                else:
                    n_incorrect_post += 1

            # --- Append to classifier rolling window ---
            correct_window.append(correct_t)

            # --- Evaluate classifier at every EVAL_EVERY steps once window full ---
            if (t + 1) % EVAL_EVERY == 0 and len(correct_window) == CLASSIFIER_WINDOW:
                win_arr = np.array(correct_window, dtype=np.float64)
                label, err_rate, rho = classify_window(win_arr)
                window_labels.append(label)
                window_error_rates.append(err_rate)
                window_rhos.append(rho)
                window_at_step.append(t)

            # --- Surprise (for trajectory consistency; not used in channel or classifier) ---
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

            # --- EWMA update AFTER correctness known (verbatim from exp157/exp159) ---
            ewma[bel_cell] = (1.0 - ALPHA) * ewma[bel_cell] + ALPHA * correct_t

            global_step += 1

    # A_hat at end for drift check
    A_hat_end = c._A_hat()
    ahat_drift = float(np.abs(A_hat_end - A_hat_start).max())

    # --- Burn-in AUROC (steps BURN_IN_START_EVAL to BURN_IN_END-1) for PC3 ---
    burnin_mask = in_burnin_arr & (np.arange(n_total) >= BURN_IN_START_EVAL)
    burnin_n2_conf = conf_n2_arr[burnin_mask]
    burnin_correct = correct_arr[burnin_mask]
    if len(burnin_n2_conf) >= PRECONDITION_MIN_CLASS:
        bi_pos = burnin_n2_conf[burnin_correct == 1]
        bi_neg = burnin_n2_conf[burnin_correct == 0]
        burnin_auroc = mannwhitney_auroc(bi_pos, bi_neg)
    else:
        burnin_auroc = float("nan")

    # --- Post-burn-in per-fork arrays for pooled AUROC ---
    post_mask = ~in_burnin_arr
    post_n2_conf = conf_n2_arr[post_mask]
    post_oracle_conf = conf_oracle_arr[post_mask]
    post_correct = correct_arr[post_mask]

    # --- Post-burn-in per-fork AUROCs ---
    if n_correct_post >= PRECONDITION_MIN_CLASS and n_incorrect_post >= PRECONDITION_MIN_CLASS:
        post_pos_n2 = post_n2_conf[post_correct == 1]
        post_neg_n2 = post_n2_conf[post_correct == 0]
        post_auroc_n2 = mannwhitney_auroc(post_pos_n2, post_neg_n2)

        post_pos_oracle = post_oracle_conf[post_correct == 1]
        post_neg_oracle = post_oracle_conf[post_correct == 0]
        post_auroc_oracle = mannwhitney_auroc(post_pos_oracle, post_neg_oracle)
    else:
        post_auroc_n2 = float("nan")
        post_auroc_oracle = float("nan")

    # --- Per-segment N2 AUROC trace (ungated diagnostic) ---
    n_segs = _n_segments(T_swap)
    segment_n2_aurocs: list[float] = []
    for seg in range(n_segs):
        seg_mask = segment_arr == seg
        seg_n2 = conf_n2_arr[seg_mask]
        seg_correct = correct_arr[seg_mask]
        if len(seg_n2) > 0:
            s_pos = seg_n2[seg_correct == 1]
            s_neg = seg_n2[seg_correct == 0]
            seg_auroc = mannwhitney_auroc(s_pos, s_neg)
        else:
            seg_auroc = float("nan")
        segment_n2_aurocs.append(seg_auroc)

    # --- Post-burn-in classifier label distribution (ungated diagnostic) ---
    post_window_labels = [
        lab for lab, step in zip(window_labels, window_at_step)
        if step >= BURN_IN_END
    ]
    n_post_windows = len(post_window_labels)
    n_ok_post = post_window_labels.count("OK")
    n_noise_post = post_window_labels.count("NOISE")
    n_structural_post = post_window_labels.count("STRUCTURAL")
    frac_ok_post = n_ok_post / n_post_windows if n_post_windows > 0 else float("nan")
    frac_noise_post = n_noise_post / n_post_windows if n_post_windows > 0 else float("nan")
    frac_structural_post = n_structural_post / n_post_windows if n_post_windows > 0 else float("nan")

    # --- Terminal EWMA calibration vs CURRENT truth (ungated diagnostic) ---
    # For diagnostics: terminal EWMA vs P0 (burn-in) reliability map
    p0_cell_set = cmap_obj.noisy_set_P0
    terminal_ewma_vs_p0_reliability: list[float] = [
        P_TRUE_NOISY if (c_idx in p0_cell_set) else 1.0 for c_idx in range(n_cells)
    ]
    x_ewma = ewma
    y_p0 = np.array(terminal_ewma_vs_p0_reliability)
    xm = x_ewma - x_ewma.mean()
    ym = y_p0 - y_p0.mean()
    denom_r = math.sqrt(float((xm ** 2).sum()) * float((ym ** 2).sum()))
    if denom_r > 1e-300:
        ewma_vs_p0_r = float((xm * ym).sum() / denom_r)
    else:
        ewma_vs_p0_r = 0.0

    # --- Fraction of post-burn-in segments with AUROC < 0.5 (ungated diagnostic) ---
    n_segs_valid = sum(
        1 for au in segment_n2_aurocs
        if not (isinstance(au, float) and math.isnan(au))
    )
    n_segs_deceived = sum(
        1 for au in segment_n2_aurocs
        if not (isinstance(au, float) and math.isnan(au)) and au < 0.5
    )
    frac_segs_deceived = n_segs_deceived / n_segs_valid if n_segs_valid > 0 else float("nan")

    return {
        "fork_seed": fork_seed,
        "T_swap": T_swap,
        "arm": T_swap,
        "regime": "R-SWAP-LAG",
        "n_total": n_total,
        "n_correct_post": n_correct_post,
        "n_incorrect_post": n_incorrect_post,
        "ahat_drift": ahat_drift,
        "burnin_auroc": burnin_auroc,
        "post_auroc_n2": post_auroc_n2,
        "post_auroc_oracle": post_auroc_oracle,
        "segment_n2_aurocs": segment_n2_aurocs,
        "n_segs": n_segs,
        "n_segs_deceived": n_segs_deceived,
        "frac_segs_deceived": frac_segs_deceived,
        "n_post_windows": n_post_windows,
        "n_ok_post": n_ok_post,
        "n_noise_post": n_noise_post,
        "n_structural_post": n_structural_post,
        "frac_ok_post": frac_ok_post,
        "frac_noise_post": frac_noise_post,
        "frac_structural_post": frac_structural_post,
        "ewma_vs_p0_r": ewma_vs_p0_r,
        "p0_cells": sorted(p0_cell_set),
        # Full post-burn-in arrays for pooled AUROC
        "_post_n2_conf": post_n2_conf,
        "_post_oracle_conf": post_oracle_conf,
        "_post_correct": post_correct,
    }


# ---------------------------------------------------------------------------
# Run all seeds for one arm
# ---------------------------------------------------------------------------

def run_arm(
    T_swap: int,
    seeds: list[int],
    mirro: Creature,
    base_cmap: list,
    n_colors: int,
    n_chunks: int = N_CHUNKS,
    verbose: bool = True,
) -> list[dict]:
    """Run all seeds for one arm (T_swap); return list of result dicts."""
    rows: list[dict] = []
    for seed in seeds:
        if verbose:
            p0 = draw_p0_cells(seed)
            print(
                f"  T_swap={T_swap}  seed={seed}  P0(first5)={p0[:5]}...  "
                f"(P1 has {25 - len(p0)} cells)",
                flush=True,
            )
        r = run_fork(
            mirro=mirro,
            fork_seed=seed,
            T_swap=T_swap,
            base_cmap=base_cmap,
            n_colors=n_colors,
            n_chunks=n_chunks,
            chunk_size=CHUNK_SIZE,
        )
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Print per-fork table for one arm
# ---------------------------------------------------------------------------

def print_fork_table(rows: list[dict], T_swap: int) -> None:
    print(f"  Per-fork table (arm T_swap={T_swap}):")
    hdr = (
        f"  {'seed':>5}  "
        f"{'n_cor_post':>10}  "
        f"{'n_inc_post':>10}  "
        f"{'burnin_au':>9}  "
        f"{'post_au_n2':>10}  "
        f"{'post_au_or':>10}  "
        f"{'frac_ok':>8}  "
        f"{'frac_no':>8}  "
        f"{'frac_st':>8}  "
        f"{'ewma_r':>8}  "
        f"{'drift':>8}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        def _fmt(v: float, w: int = 9) -> str:
            if isinstance(v, float) and math.isnan(v):
                return "nan".rjust(w)
            return f"{v:.4f}".rjust(w)

        print(
            f"  {r['fork_seed']:>5}  "
            f"{r['n_correct_post']:>10}  "
            f"{r['n_incorrect_post']:>10}  "
            f"{_fmt(r['burnin_auroc'], 9)}  "
            f"{_fmt(r['post_auroc_n2'], 10)}  "
            f"{_fmt(r['post_auroc_oracle'], 10)}  "
            f"{_fmt(r['frac_ok_post'], 8)}  "
            f"{_fmt(r['frac_noise_post'], 8)}  "
            f"{_fmt(r['frac_structural_post'], 8)}  "
            f"{_fmt(r['ewma_vs_p0_r'], 8)}  "
            f"{r['ahat_drift']:>8.4f}"
        )


# ---------------------------------------------------------------------------
# Precondition check (per arm)
# ---------------------------------------------------------------------------

def check_preconditions(rows: list[dict], T_swap: int) -> tuple[bool, list[str]]:
    """Check PC1, PC2, PC3 for one arm.  Returns (all_pass, list_of_failures)."""
    failures: list[str] = []

    n_forks = len(rows)
    burnin_auroc_pass_count = 0

    for r in rows:
        seed = r["fork_seed"]

        # PC1: >= 50 correct AND >= 50 incorrect post-burn-in per fork
        if r["n_correct_post"] < PRECONDITION_MIN_CLASS:
            failures.append(
                f"PC1 FAIL: T={T_swap} seed={seed} n_correct_post={r['n_correct_post']} "
                f"< {PRECONDITION_MIN_CLASS}"
            )
        if r["n_incorrect_post"] < PRECONDITION_MIN_CLASS:
            failures.append(
                f"PC1 FAIL: T={T_swap} seed={seed} n_incorrect_post={r['n_incorrect_post']} "
                f"< {PRECONDITION_MIN_CLASS}"
            )

        # PC2: ahat_drift < 0.05
        if r["ahat_drift"] >= PRECONDITION_AHAT_DRIFT:
            failures.append(
                f"PC2 FAIL: T={T_swap} seed={seed} ahat_drift={r['ahat_drift']:.4f} "
                f">= {PRECONDITION_AHAT_DRIFT}"
            )

        # PC3: burn-in AUROC > 0.6 per fork
        bi_au = r["burnin_auroc"]
        if not (isinstance(bi_au, float) and math.isnan(bi_au)) and bi_au > PC3_BURNIN_AUROC_THRESH:
            burnin_auroc_pass_count += 1

    # PC3 aggregate: >= 7/8 forks must pass burn-in AUROC
    if burnin_auroc_pass_count < PC3_BURNIN_FORKS_MIN:
        failures.append(
            f"PC3 FAIL: T={T_swap} only {burnin_auroc_pass_count}/{n_forks} forks have "
            f"burn-in AUROC > {PC3_BURNIN_AUROC_THRESH} "
            f"(need >= {PC3_BURNIN_FORKS_MIN})"
        )

    all_pass = len(failures) == 0
    return all_pass, failures


# ---------------------------------------------------------------------------
# Per-arm evaluation (P1/F1, P2/F2)
# ---------------------------------------------------------------------------

def evaluate_arm(rows: list[dict], T_swap: int) -> dict:
    """Evaluate P1/F1 and P2/F2 for one arm; return evaluation dict."""
    n_forks = len(rows)

    # --- Pooled post-burn-in AUROCs ---
    pool_n2_auroc = pooled_auroc(
        [r["_post_n2_conf"] for r in rows],
        [r["_post_correct"] for r in rows],
    )
    pool_oracle_auroc = pooled_auroc(
        [r["_post_oracle_conf"] for r in rows],
        [r["_post_correct"] for r in rows],
    )

    # --- P1 / F1: sustained deception in this arm ---
    # P1a: pooled N2 AUROC <= 0.45
    p1_pooled_pass = (
        not (isinstance(pool_n2_auroc, float) and math.isnan(pool_n2_auroc))
        and pool_n2_auroc <= P1_POOLED_MAX
    )
    # P1b: per-fork N2 AUROC < 0.5 in >= 7/8 forks
    p1_forks_below_count = sum(
        1 for r in rows
        if not (isinstance(r["post_auroc_n2"], float) and math.isnan(r["post_auroc_n2"]))
        and r["post_auroc_n2"] < P1_PER_FORK_THRESH
    )
    p1_forks_pass = p1_forks_below_count >= P1_FORKS_BELOW_HALF_MIN
    p1_pass = p1_pooled_pass and p1_forks_pass

    # F1 per arm: pooled >= 0.5
    f1_arm = (
        not (isinstance(pool_n2_auroc, float) and math.isnan(pool_n2_auroc))
        and pool_n2_auroc >= F1_POOLED_MIN
    )

    # --- P2 / F2: oracle discrimination possible in this arm ---
    p2_pass = (
        not (isinstance(pool_oracle_auroc, float) and math.isnan(pool_oracle_auroc))
        and pool_oracle_auroc >= P2_ORACLE_POOLED_MIN
    )
    f2_arm = (
        not (isinstance(pool_oracle_auroc, float) and math.isnan(pool_oracle_auroc))
        and pool_oracle_auroc < F2_ORACLE_POOLED_MAX
    )

    # --- Burn-in AUROC summary ---
    burnin_aurocs = [
        r["burnin_auroc"] for r in rows
        if not (isinstance(r["burnin_auroc"], float) and math.isnan(r["burnin_auroc"]))
    ]
    mean_burnin_auroc = float(np.mean(burnin_aurocs)) if burnin_aurocs else float("nan")

    # --- Fraction of segments deceived (pooled across forks, for diagnostics) ---
    all_segs_deceived = sum(r["n_segs_deceived"] for r in rows)
    all_segs_valid_count = sum(r["n_segs"] for r in rows)
    frac_segs_deceived_pooled = (
        all_segs_deceived / all_segs_valid_count if all_segs_valid_count > 0 else float("nan")
    )

    return {
        "T_swap": T_swap,
        # P1 / F1
        "pool_n2_auroc": pool_n2_auroc,
        "p1_pooled_pass": p1_pooled_pass,
        "p1_forks_below_count": p1_forks_below_count,
        "p1_forks_pass": p1_forks_pass,
        "p1_pass": p1_pass,
        "f1_arm": f1_arm,
        # P2 / F2
        "pool_oracle_auroc": pool_oracle_auroc,
        "p2_pass": p2_pass,
        "f2_arm": f2_arm,
        # Burn-in summary
        "mean_burnin_auroc": mean_burnin_auroc,
        # Segment diagnostic
        "frac_segs_deceived_pooled": frac_segs_deceived_pooled,
        # Counts
        "n_forks": n_forks,
    }


# ---------------------------------------------------------------------------
# Two-arm verdict (P1/F1 across both arms, P2/F2 applied per arm)
# ---------------------------------------------------------------------------

def evaluate_two_arms(ev500: dict, ev750: dict) -> dict:
    """Evaluate global P1/F1/P2/F2 verdict across both arms.

    P1: sustained deception in AT LEAST ONE arm (T=500 OR T=750).
    F1: BOTH arms pooled >= 0.5 (candidate 2 dead).
    P2: oracle pooled >= 0.7 in every arm that passes P1.
         If no arm passes P1, evaluate on both arms for the record.
    F2: oracle pooled < 0.6 in BOTH arms.
    VERDICT: POSITIVE iff P1 and P2 both pass.
             NEGATIVE iff F1 or F2 fires.
             Otherwise MIXED.
    """
    p1_pass = ev500["p1_pass"] or ev750["p1_pass"]

    # F1: BOTH arms pooled >= 0.5
    f1 = ev500["f1_arm"] and ev750["f1_arm"]

    # P2: evaluate on arms that pass P1; if none pass P1, both arms
    arms_for_p2 = [ev for ev in [ev500, ev750] if ev["p1_pass"]]
    if not arms_for_p2:
        arms_for_p2 = [ev500, ev750]
    p2_pass = all(ev["p2_pass"] for ev in arms_for_p2)

    # F2: oracle pooled < 0.6 in BOTH arms
    f2 = ev500["f2_arm"] and ev750["f2_arm"]

    positive = p1_pass and p2_pass
    negative = f1 or f2
    if positive:
        verdict_str = "POSITIVE"
    elif negative:
        verdict_str = "NEGATIVE"
    else:
        verdict_str = "MIXED"

    return {
        "p1_pass": p1_pass,
        "f1": f1,
        "p2_pass": p2_pass,
        "f2": f2,
        "verdict_str": verdict_str,
    }


# ---------------------------------------------------------------------------
# Write JSONL rows + per-arm summaries + overall summary
# ---------------------------------------------------------------------------

def write_json_rows(
    rows_per_arm: dict[int, list[dict]],
    evs_per_arm: dict[int, dict] | None,
    overall: dict | None,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _nan_to_none(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    with path.open("w") as fh:
        for T_swap, rows in rows_per_arm.items():
            for r in rows:
                row: dict = {
                    "exp": 161,
                    "fork_seed": r["fork_seed"],
                    "arm": T_swap,
                    "regime": r["regime"],
                    "n_total": r["n_total"],
                    "n_correct_post": r["n_correct_post"],
                    "n_incorrect_post": r["n_incorrect_post"],
                    "ahat_drift": r["ahat_drift"],
                    "burnin_auroc": _nan_to_none(r["burnin_auroc"]),
                    "post_auroc_n2": _nan_to_none(r["post_auroc_n2"]),
                    "post_auroc_oracle": _nan_to_none(r["post_auroc_oracle"]),
                    "segment_n2_aurocs": [_nan_to_none(v) for v in r["segment_n2_aurocs"]],
                    "n_segs": r["n_segs"],
                    "n_segs_deceived": r["n_segs_deceived"],
                    "frac_segs_deceived": _nan_to_none(r["frac_segs_deceived"]),
                    "n_post_windows": r["n_post_windows"],
                    "n_ok_post": r["n_ok_post"],
                    "n_noise_post": r["n_noise_post"],
                    "n_structural_post": r["n_structural_post"],
                    "frac_ok_post": _nan_to_none(r["frac_ok_post"]),
                    "frac_noise_post": _nan_to_none(r["frac_noise_post"]),
                    "frac_structural_post": _nan_to_none(r["frac_structural_post"]),
                    "ewma_vs_p0_r": r["ewma_vs_p0_r"],
                    "p0_cells": r["p0_cells"],
                }
                fh.write(json.dumps(row) + "\n")

        if evs_per_arm is not None:
            for T_swap, ev in evs_per_arm.items():
                arm_summary: dict = {
                    "exp": 161,
                    "row_type": "arm_summary",
                    "arm": T_swap,
                    "pool_n2_auroc": _nan_to_none(ev["pool_n2_auroc"]),
                    "p1_pooled_pass": ev["p1_pooled_pass"],
                    "p1_forks_below_count": ev["p1_forks_below_count"],
                    "p1_forks_pass": ev["p1_forks_pass"],
                    "p1_pass": ev["p1_pass"],
                    "f1_arm": ev["f1_arm"],
                    "pool_oracle_auroc": _nan_to_none(ev["pool_oracle_auroc"]),
                    "p2_pass": ev["p2_pass"],
                    "f2_arm": ev["f2_arm"],
                    "mean_burnin_auroc": _nan_to_none(ev["mean_burnin_auroc"]),
                    "frac_segs_deceived_pooled": _nan_to_none(ev["frac_segs_deceived_pooled"]),
                }
                fh.write(json.dumps(arm_summary) + "\n")

        if overall is not None:
            overall_summary: dict = {
                "exp": 161,
                "row_type": "overall_summary",
                "p1_pass": overall["p1_pass"],
                "f1": overall["f1"],
                "p2_pass": overall["p2_pass"],
                "f2": overall["f2"],
                "verdict": overall["verdict_str"],
            }
            fh.write(json.dumps(overall_summary) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 161 — N3 rung 1, candidate 2: lag-matched swap (R-SWAP-LAG)"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Smoke run: seed=[50] only, BOTH arms (T=500 and T=750), full 4000 steps, "
            "prints per-arm numbers including segment AUROC traces. "
            "No verdict written."
        ),
    )
    args = parser.parse_args()

    smoke = args.smoke
    seeds = [50] if smoke else SEEDS

    print("=" * 80)
    print("Exp 161 — N3 rung 1, candidate 2: LAG-MATCHED SWAP (R-SWAP-LAG)")
    print("          Two arms: T_swap = 500 and T_swap = 750")
    print("=" * 80)
    print()
    print(f"SURPRISE_WINDOW={SURPRISE_WINDOW}  CLASSIFIER_WINDOW={CLASSIFIER_WINDOW}  "
          f"[must be equal; asserted]")
    assert CLASSIFIER_WINDOW == SURPRISE_WINDOW
    print(f"EVAL_EVERY={EVAL_EVERY}  ERROR_RATE_OK_THRESH={ERROR_RATE_OK_THRESH}  "
          f"LAG1_RHO_STRUCT_THRESH={LAG1_RHO_STRUCT_THRESH}")
    print(f"ALPHA={ALPHA}  EWMA_INIT={EWMA_INIT}  (EWMA channel, verbatim exp157/exp159)")
    print(f"P_TRUE_NOISY={P_TRUE_NOISY}  N_NOISY_P0={N_NOISY_P0}")
    print(f"BURN_IN_END={BURN_IN_END}  SWAP_PHASE_START={SWAP_PHASE_START}  ARMS={ARMS}")
    print(f"SWAP_PLACE_SEED_OFFSET={SWAP_PLACE_SEED_OFFSET}  "
          f"SWAP_NOISE_SEED_OFFSET={SWAP_NOISE_SEED_OFFSET}")
    print(f"Seeds: {seeds}  N_CHUNKS={N_CHUNKS}  "
          f"CHUNK_SIZE={CHUNK_SIZE}  N_STEPS={N_CHUNKS * CHUNK_SIZE}")
    print()
    print("Swap schedule (both arms, same convention as Exp 160):")
    print("  steps 0-999: noisy set = P0 (burn-in; EWMA learns the map)")
    print("  step 1000: FIRST post-burn-in segment uses P1 (inverted from burn-in)")
    print("  subsequent segments: alternate P0/P1 at each arm's T_swap tempo")
    print("  T=500: 6 segments (6 swaps); T=750: 4 segments (4 swaps)")
    print()

    # --- Load mirro spine (read-only) ---
    mirro = Creature.load("creature/state/mirro")
    print(f"Loaded mirro: age={mirro.age_steps}, world={mirro.world.rows}x{mirro.world.cols}, "
          f"n_colors={mirro.world.n_colors}, n_cells={mirro.world.n_cells}")
    base_cmap = list(mirro.world.cmap)
    n_colors = mirro.world.n_colors
    print(f"Base cmap: {base_cmap}")
    print()

    # --- Run both arms ---
    rows_per_arm: dict[int, list[dict]] = {}
    for T_swap in ARMS:
        print(f"--- ARM T_swap={T_swap} ---")
        rows = run_arm(
            T_swap=T_swap,
            seeds=seeds,
            mirro=mirro,
            base_cmap=base_cmap,
            n_colors=n_colors,
            n_chunks=N_CHUNKS,
            verbose=True,
        )
        rows_per_arm[T_swap] = rows
        print()

    # --- Per-fork tables ---
    for T_swap, rows in rows_per_arm.items():
        print_fork_table(rows, T_swap)
        print()

    # --- Ungated diagnostics ---
    print("Ungated diagnostics:")
    print("  Named ungated expectations:")
    print("    - segment N2 AUROC trace: deceived openings in BOTH directions "
          "(not only the first swap, unlike Exp 160)")
    print("    - fraction of segments with AUROC < 0.5 per arm "
          "(Exp 160 comparison: decaying, only early segments)")
    print("    - classifier post-burn-in: NOISE-dominant")
    print("    - terminal EWMA: negatively correlated with CURRENT truth "
          "(inverted relative to P0 burn-in map)")
    print()

    for T_swap, rows in rows_per_arm.items():
        n_segs = _n_segments(T_swap)
        print(f"  [ARM T_swap={T_swap}, n_segs={n_segs}]")
        for r in rows:
            seg_strs = []
            for i, au in enumerate(r["segment_n2_aurocs"]):
                phase = "P1" if i % 2 == 0 else "P0"
                if isinstance(au, float) and not math.isnan(au):
                    seg_strs.append(f"seg{i}({phase}):{au:.3f}")
                else:
                    seg_strs.append(f"seg{i}({phase}):nan")
            print(f"    seed={r['fork_seed']}  burnin_auroc={r['burnin_auroc']:.4f}"
                  f"  post_n2={r['post_auroc_n2']:.4f}"
                  f"  post_oracle={r['post_auroc_oracle']:.4f}"
                  f"  frac_segs_dec={r['frac_segs_deceived']:.3f}")
            # Print segments in rows of up to 4
            for chunk_start in range(0, len(seg_strs), 4):
                chunk = seg_strs[chunk_start:chunk_start + 4]
                print(f"      {', '.join(chunk)}")
            print(f"      post-burn-in classifier: "
                  f"frac_ok={r['frac_ok_post']:.4f}  "
                  f"frac_noise={r['frac_noise_post']:.4f}  "
                  f"frac_struct={r['frac_structural_post']:.4f}  "
                  f"(n_win={r['n_post_windows']})")
            print(f"      ewma_vs_P0_r={r['ewma_vs_p0_r']:.4f}  "
                  f"p0_cells(first5)={r['p0_cells'][:5]}...")
        print()

    # --- Smoke exit ---
    if smoke:
        print("SMOKE ONLY — no verdict")
        return

    # --- Precondition checks per arm ---
    all_pc_pass = True
    all_pc_failures: list[str] = []
    for T_swap, rows in rows_per_arm.items():
        pc_pass, pc_failures = check_preconditions(rows, T_swap)
        if not pc_pass:
            all_pc_pass = False
            all_pc_failures.extend(pc_failures)

    if all_pc_failures:
        print("PRECONDITION FAILED:")
        for f in all_pc_failures:
            print(f"  {f}")
        print()
        print("PRECONDITION FAILED — no verdict.")
        out_rows_path = Path(__file__).parent / "outputs" / "exp161_rows.json"
        write_json_rows(rows_per_arm, evs_per_arm=None, overall=None, path=out_rows_path)
        print(f"JSON rows written to {out_rows_path} (no verdict)")
        return

    print("Preconditions: all PASS")
    print()

    # --- Evaluate per-arm ---
    evs_per_arm: dict[int, dict] = {}
    for T_swap, rows in rows_per_arm.items():
        evs_per_arm[T_swap] = evaluate_arm(rows, T_swap)

    ev500 = evs_per_arm[500]
    ev750 = evs_per_arm[750]

    print("=" * 80)
    print("PREDECLARED OUTCOME MAP")
    print("=" * 80)
    print()

    for T_swap, ev in evs_per_arm.items():
        rows = rows_per_arm[T_swap]
        n = ev["n_forks"]

        # Burn-in summary
        bi_aurocs_str = ", ".join(
            f"{r['burnin_auroc']:.4f}" if not (isinstance(r['burnin_auroc'], float) and math.isnan(r['burnin_auroc'])) else "nan"
            for r in rows
        )
        print(f"[ARM T_swap={T_swap}]")
        print(f"PC3 burn-in N2 AUROC (steps {BURN_IN_START_EVAL}-{BURN_IN_END - 1}):")
        print(f"  Per fork: [{bi_aurocs_str}]")
        print(f"  Mean: {ev['mean_burnin_auroc']:.4f}")
        print()

        pool_n2_str = (
            f"{ev['pool_n2_auroc']:.4f}"
            if not (isinstance(ev["pool_n2_auroc"], float) and math.isnan(ev["pool_n2_auroc"]))
            else "nan"
        )
        post_n2_aurocs_str = ", ".join(
            f"{r['post_auroc_n2']:.4f}" if not (isinstance(r['post_auroc_n2'], float) and math.isnan(r['post_auroc_n2'])) else "nan"
            for r in rows
        )
        print(f"P1 (sustained deception in arm T={T_swap}):")
        print(f"  Post-burn-in N2 AUROC per fork: [{post_n2_aurocs_str}]")
        print(f"  Forks with N2 AUROC < {P1_PER_FORK_THRESH}: "
              f"{ev['p1_forks_below_count']}/{n} (need >={P1_FORKS_BELOW_HALF_MIN} for P1)")
        print(f"  Pooled N2 AUROC: {pool_n2_str} (need <={P1_POOLED_MAX} for P1)")
        print(f"  FALSIFIER F1 (this arm): pooled >= {F1_POOLED_MIN}: "
              f"{'YES' if ev['f1_arm'] else 'no'}")
        p1_status = "PASS" if ev["p1_pass"] else ("F1 (this arm)" if ev["f1_arm"] else "MIXED (not P1, not F1)")
        print(f"  => P1 (arm T={T_swap}): {p1_status}")
        print()

        pool_oracle_str = (
            f"{ev['pool_oracle_auroc']:.4f}"
            if not (isinstance(ev["pool_oracle_auroc"], float) and math.isnan(ev["pool_oracle_auroc"]))
            else "nan"
        )
        post_oracle_aurocs_str = ", ".join(
            f"{r['post_auroc_oracle']:.4f}" if not (isinstance(r['post_auroc_oracle'], float) and math.isnan(r['post_auroc_oracle'])) else "nan"
            for r in rows
        )
        print(f"P2 (discrimination possible in arm T={T_swap}):")
        print(f"  Post-burn-in oracle AUROC per fork: [{post_oracle_aurocs_str}]")
        print(f"  Pooled oracle AUROC: {pool_oracle_str} (need >={P2_ORACLE_POOLED_MIN} for P2)")
        print(f"  FALSIFIER F2 (this arm): oracle pooled < {F2_ORACLE_POOLED_MAX}: "
              f"{'YES' if ev['f2_arm'] else 'no'}")
        p2_status = "PASS" if ev["p2_pass"] else ("F2 (this arm)" if ev["f2_arm"] else "MIXED (not P2, not F2)")
        print(f"  => P2 (arm T={T_swap}): {p2_status}")
        print()

    # --- Two-arm global verdict ---
    overall = evaluate_two_arms(ev500, ev750)

    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    print(f"  T=500 P1: {'PASS' if ev500['p1_pass'] else 'FAIL'}  "
          f"pooled_n2={ev500['pool_n2_auroc']:.4f}  "
          f"forks_below={ev500['p1_forks_below_count']}/{ev500['n_forks']}")
    print(f"  T=750 P1: {'PASS' if ev750['p1_pass'] else 'FAIL'}  "
          f"pooled_n2={ev750['pool_n2_auroc']:.4f}  "
          f"forks_below={ev750['p1_forks_below_count']}/{ev750['n_forks']}")
    print(f"  P1 (at least one arm passes): {'PASS' if overall['p1_pass'] else 'FAIL'}")
    print(f"  FALSIFIER F1 (BOTH arms pooled >= {F1_POOLED_MIN}): "
          f"{'YES — F1 FIRES' if overall['f1'] else 'no'}")
    print()
    print(f"  T=500 P2: {'PASS' if ev500['p2_pass'] else 'FAIL'}  "
          f"pooled_oracle={ev500['pool_oracle_auroc']:.4f}")
    print(f"  T=750 P2: {'PASS' if ev750['p2_pass'] else 'FAIL'}  "
          f"pooled_oracle={ev750['pool_oracle_auroc']:.4f}")
    print(f"  P2 (oracle discriminable in P1-passing arms): "
          f"{'PASS' if overall['p2_pass'] else 'FAIL'}")
    print(f"  FALSIFIER F2 (oracle pooled < {F2_ORACLE_POOLED_MAX} in BOTH arms): "
          f"{'YES — F2 FIRES' if overall['f2'] else 'no'}")
    print()

    if overall["verdict_str"] == "POSITIVE":
        print("VERDICT: POSITIVE")
        print("GATE PASSES — the discriminating perturbation EXISTS")
    elif overall["verdict_str"] == "NEGATIVE":
        print("VERDICT: NEGATIVE")
        print("GATE: this candidate FAILED")
    else:
        print("VERDICT: MIXED")
    print()
    print("=" * 80)

    # --- Write JSON rows ---
    out_rows_path = Path(__file__).parent / "outputs" / "exp161_rows.json"
    write_json_rows(rows_per_arm, evs_per_arm=evs_per_arm, overall=overall, path=out_rows_path)
    print(f"JSON rows written to {out_rows_path}")

    # --- Write verdict JSON ---
    arms_verdict = {
        "P1_sustained_anticalibration": {
            "pass": bool(overall["p1_pass"]),
            "reason": (
                f"T=500 pooled_n2={ev500['pool_n2_auroc']:.4f} "
                f"(forks_below={ev500['p1_forks_below_count']}/{ev500['n_forks']}); "
                f"T=750 pooled_n2={ev750['pool_n2_auroc']:.4f} "
                f"(forks_below={ev750['p1_forks_below_count']}/{ev750['n_forks']}); "
                f"F1(both_arms_pooled>={F1_POOLED_MIN})={overall['f1']}"
            ),
        },
        "P2_oracle_discriminable": {
            "pass": bool(overall["p2_pass"]),
            "reason": (
                f"T=500 pooled_oracle={ev500['pool_oracle_auroc']:.4f}; "
                f"T=750 pooled_oracle={ev750['pool_oracle_auroc']:.4f}; "
                f"F2(both_arms_oracle<{F2_ORACLE_POOLED_MAX})={overall['f2']}"
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp161_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp161",
        arms=arms_verdict,
        verdict=overall["verdict_str"],
        halted=False,
        notes=(
            "N3 rung 1 gate, candidate 2 (lag-matched): R-SWAP-LAG world swaps which "
            "13-of-25 cells are noisy every T_swap steps — two arms T=500 and T=750, "
            "both matched to the EWMA relearn time (~500 steps) after a 1000-step burn-in. "
            "Mechanism prediction: at these tempos the EWMA fully converts within each "
            "segment, so the next swap (either direction) inverts a freshly-learned map, "
            "sustaining anti-calibration unlike the decaying sawtooth of Exp 160 (T=250). "
            "POSITIVE => GATE PASSES. NEGATIVE kills candidate 2 only; candidates 3 and 4 remain."
        ),
    )
    print(f"Verdict JSON written to {verdict_path}")


if __name__ == "__main__":
    main()
