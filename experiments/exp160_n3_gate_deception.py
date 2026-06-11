"""Exp 160 — N3 rung 1, the discriminating-perturbation GATE: a reliability-
swap world makes the N2 confidence channel systematically ANTI-calibrated
(trusting it is worse than ignoring it), while an oracle control proves the
discrimination was achievable — the constructible regime where N2 is wrong,
whose existence the anti-regress rule demands before N3 can be a layer.

Hypothesis: the Exp 157 EWMA channel is history-indexed; a world that SWAPS
which cells are unreliable faster than the EWMA can relearn (time constant
1/alpha = 20 visits/cell ~ 500 steps at this walk's ~1 visit per cell per 25
steps) inverts the learned reliability map persistently: high-confidence
(learned-clean) cells produce errors, low-confidence cells don't — type-2
AUROC systematically BELOW 0.5. An oracle channel reading the TRUE current
reliability shows the world itself is discriminable, so the failure is N2's.

World R-SWAP (provided): random 13-of-25 noisy placement P0 (per-fork rng
110_000 + fork_seed; p_true = 0.55, noise rng 90_000 + fork_seed); P1 = the
complementary 12 cells. Phase 0 (burn-in): steps 0-999 on P0 (the EWMA
learns the map). Then the placement SWAPS P0 <-> P1 every 250 steps for
steps 1000-3999 (12 swaps; 250 << 500-step relearn time). Total 4000 steps.

Design: forks of mirro only (spine never saved), FRESH seeds 42-49, exp155
step-loop semantics. Channels computed per step pre-update as in Exp 157/159:
  N2 channel:    conf = EWMA[argmax(qs)] (alpha 0.05, init 0.5; updated with
                 correctness after each step; runs from step 0).
  ORACLE channel: conf_oracle = true current reliability of the TRUE cell
                 (1.0 if currently clean, 0.55 if currently noisy) — reads
                 the generator, NOT the creature; control only.
  Classifier:    exp158 form (window 200, every 100 steps; error<0.05 OK;
                 rho>0.3 STRUCTURAL; else NOISE) — runs throughout; labels
                 logged (UNGATED diagnostic).
Gated AUROCs use POST-BURN-IN trials only (steps 1000-3999).

Preconditions (instrument-grade; failure => "PRECONDITION FAILED", rows
written, no verdict): PC1 >= 50 correct AND >= 50 incorrect post-burn-in
trials per fork; PC2 ahat_drift < 0.05 per fork; PC3 burn-in calibration:
type-2 AUROC of the N2 channel on burn-in trials (steps 200-999) > 0.6 per
fork in >= 7/8 forks (the channel must have LEARNED the map before the
deception can invert it — input validity).

Predeclared properties and falsifiers:
  P1 (deception exists): post-burn-in N2-channel type-2 AUROC pooled <= 0.45
     AND per-fork < 0.5 in >= 7/8 forks.
     FALSIFIER F1: pooled >= 0.5 (no systematic anti-calibration — this
     candidate fails the gate).
  P2 (discrimination possible — the failure is N2's, not the world's):
     post-burn-in ORACLE-channel pooled AUROC >= 0.7.
     FALSIFIER F2: oracle pooled < 0.6 (the world itself is undiscriminable;
     the deception claim is unlicensed).
  VERDICT: POSITIVE (GATE PASSES — the discriminating perturbation EXISTS)
  iff P1 and P2 both pass. NEGATIVE iff F1 or F2 fires. Otherwise MIXED.
  "Not a falsifier" never counts toward POSITIVE. A NEGATIVE here kills THIS
  candidate only; the card's other candidates (period >= window; OK-bar-
  hugging error rates) remain to be tried before any "no N2-failure regime"
  conclusion.

Ungated diagnostics: per-250-step-segment N2 AUROC trace (the predicted
anti-calibration sawtooth); classifier label distribution post-burn-in
(named ungated expectation: NOISE-dominant — the typed alarm also misses
the swap structure); EWMA terminal calibration vs CURRENT truth; per-fork
P0 cell lists.
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

SEEDS = list(range(42, 50))   # fresh seeds — exp155-159 used 0-41

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

# R-SWAP world parameters
P_TRUE_NOISY = 0.55          # p_true for noisy cells
N_NOISY_P0 = 13              # number of noisy cells in P0 (P1 has 25-13=12)

# Seed offsets for R-SWAP
SWAP_PLACE_SEED_OFFSET = 110_000   # per-fork rng for drawing P0: 110_000 + fork_seed
SWAP_NOISE_SEED_OFFSET = 90_000    # per-fork noise rng: 90_000 + fork_seed

# Swap schedule
SWAP_PHASE_START = 1000      # first swap at step 1000
SWAP_INTERVAL = 250          # swap every 250 steps
# At step 1000: FIRST swap fires (steps 1000-1249 use P1)
# At step 1250: second swap (steps 1250-1499 use P0), ...alternating

# Precondition thresholds
PRECONDITION_MIN_CLASS = 50          # PC1: >= 50 correct AND incorrect post-burn-in per fork
PRECONDITION_AHAT_DRIFT = 0.05       # PC2: max abs A_hat change < 0.05 per fork
PC3_BURNIN_AUROC_THRESH = 0.6        # PC3: burn-in AUROC > 0.6 per fork
PC3_BURNIN_FORKS_MIN = 7             # PC3: >= 7/8 forks must pass burn-in AUROC check

# P1 thresholds (deception exists)
P1_POOLED_MAX = 0.45                 # P1: post-burn-in pooled N2 AUROC <= 0.45
P1_PER_FORK_THRESH = 0.5             # P1: per-fork threshold
P1_FORKS_BELOW_HALF_MIN = 7          # P1: >= 7/8 forks with AUROC < 0.5
F1_POOLED_MIN = 0.5                  # F1: pooled >= 0.5 => F1 fires

# P2 thresholds (discrimination possible via oracle)
P2_ORACLE_POOLED_MIN = 0.7           # P2: oracle pooled AUROC >= 0.7
F2_ORACLE_POOLED_MAX = 0.6           # F2: oracle pooled < 0.6 => F2 fires

# Number of post-burn-in segments (each 250 steps)
N_SWAP_SEGMENTS = 12    # steps 1000-3999 => 12 segments of 250 steps


# ---------------------------------------------------------------------------
# SwapPlaceNoiseCmap — reliability-swap world
# ---------------------------------------------------------------------------

class SwapPlaceNoiseCmap:
    """Color map wrapper implementing the R-SWAP reliability-swap world.

    P0: 13 cells drawn uniformly w/o replacement (per-fork rng 110_000+fork_seed).
    P1: the complementary 12 cells.

    Swap schedule (exact):
      - steps 0-999 (burn-in): noisy set = P0 (EWMA learns P0's map)
      - steps 1000+: the FIRST SWAP fires at step 1000 (P0->P1), then alternates
        every 250 steps. Formally, for step >= 1000:
          phase_idx = (step - 1000) // 250
          noisy_set = P1 if (phase_idx % 2 == 0) else P0
        So steps 1000-1249 (phase_idx=0) => P1 (inverted from burn-in)
           steps 1250-1499 (phase_idx=1) => P0 (back to burn-in)
           steps 1500-1749 (phase_idx=2) => P1
           ... alternating every 250 steps for 12 segments total.

    Caller must set self.current_step before each __getitem__ call.

    The noise rng is private (seed 90_000+fork_seed); trajectory is untouched.
    """

    def __init__(self, base_cmap, n_colors, noisy_cells_p0: list[int],
                 p_true: float, noise_seed: int):
        self.base = list(base_cmap)
        self.n_colors = int(n_colors)
        self.noisy_set_P0 = set(noisy_cells_p0)
        # P1 = complement of P0 over range(25)
        all_cells = set(range(len(base_cmap)))
        self.noisy_set_P1 = all_cells - self.noisy_set_P0
        self.p_true = float(p_true)
        self.rng = np.random.default_rng(noise_seed)
        self.current_step = 0   # caller sets before __getitem__

    def _current_noisy_set(self) -> set:
        """Return the currently active noisy set based on current_step."""
        t = self.current_step
        if t < SWAP_PHASE_START:
            return self.noisy_set_P0
        phase_idx = (t - SWAP_PHASE_START) // SWAP_INTERVAL
        # phase_idx=0 => P1 (first swap), phase_idx=1 => P0, alternating
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
# Run one fork — step loop verbatim from exp159, extended with R-SWAP world
# ---------------------------------------------------------------------------

def run_fork(
    mirro: Creature,
    fork_seed: int,
    base_cmap: list,
    n_colors: int,
    n_chunks: int = N_CHUNKS,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """Run one fresh fork of mirro in the R-SWAP regime.

    Step loop replicates exp155/exp158/exp159 run_fork_regime semantics EXACTLY:
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

    fork_name = f"exp160_s{fork_seed}_RSWAP"
    c = mirro.fork(fork_name)

    n_cells = c.world.n_cells
    n_actions = 4

    # --- Build R-SWAP cmap ---
    p0_cells = draw_p0_cells(fork_seed)
    noise_seed = SWAP_NOISE_SEED_OFFSET + fork_seed
    cmap_obj = SwapPlaceNoiseCmap(
        base_cmap=base_cmap,
        n_colors=n_colors,
        noisy_cells_p0=p0_cells,
        p_true=P_TRUE_NOISY,
        noise_seed=noise_seed,
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

            # --- Observe (from R-SWAP cmap) ---
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
                seg_idx = (t - BURN_IN_END) // SWAP_INTERVAL
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

    # --- Per-segment (250-step) N2 AUROC trace (ungated diagnostic) ---
    segment_n2_aurocs: list[float] = []
    for seg in range(N_SWAP_SEGMENTS):
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

    return {
        "fork_seed": fork_seed,
        "regime": "R-SWAP",
        "n_total": n_total,
        "n_correct_post": n_correct_post,
        "n_incorrect_post": n_incorrect_post,
        "ahat_drift": ahat_drift,
        "burnin_auroc": burnin_auroc,
        "post_auroc_n2": post_auroc_n2,
        "post_auroc_oracle": post_auroc_oracle,
        "segment_n2_aurocs": segment_n2_aurocs,
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
            p0 = draw_p0_cells(seed)
            print(
                f"  seed={seed}  P0(first5)={p0[:5]}...  "
                f"(P1 has {25 - len(p0)} cells)",
                flush=True,
            )
            print(f"    Running seed={seed} regime=R-SWAP ...", flush=True)
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
    print("-" * len(hdr))
    for r in rows:
        def _fmt(v: float, w: int = 9) -> str:
            if isinstance(v, float) and math.isnan(v):
                return "nan".rjust(w)
            return f"{v:.4f}".rjust(w)

        print(
            f"{r['fork_seed']:>5}  "
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
# Precondition check
# ---------------------------------------------------------------------------

def check_preconditions(rows: list[dict]) -> tuple[bool, list[str]]:
    """Check PC1, PC2, PC3.  Returns (all_pass, list_of_failures)."""
    failures: list[str] = []

    n_forks = len(rows)
    burnin_auroc_pass_count = 0

    for r in rows:
        seed = r["fork_seed"]

        # PC1: >= 50 correct AND >= 50 incorrect post-burn-in per fork
        if r["n_correct_post"] < PRECONDITION_MIN_CLASS:
            failures.append(
                f"PC1 FAIL: seed={seed} n_correct_post={r['n_correct_post']} "
                f"< {PRECONDITION_MIN_CLASS}"
            )
        if r["n_incorrect_post"] < PRECONDITION_MIN_CLASS:
            failures.append(
                f"PC1 FAIL: seed={seed} n_incorrect_post={r['n_incorrect_post']} "
                f"< {PRECONDITION_MIN_CLASS}"
            )

        # PC2: ahat_drift < 0.05
        if r["ahat_drift"] >= PRECONDITION_AHAT_DRIFT:
            failures.append(
                f"PC2 FAIL: seed={seed} ahat_drift={r['ahat_drift']:.4f} "
                f">= {PRECONDITION_AHAT_DRIFT}"
            )

        # PC3: burn-in AUROC > 0.6 per fork
        bi_au = r["burnin_auroc"]
        if not (isinstance(bi_au, float) and math.isnan(bi_au)) and bi_au > PC3_BURNIN_AUROC_THRESH:
            burnin_auroc_pass_count += 1

    # PC3 aggregate: >= 7/8 forks must pass burn-in AUROC
    if burnin_auroc_pass_count < PC3_BURNIN_FORKS_MIN:
        failures.append(
            f"PC3 FAIL: only {burnin_auroc_pass_count}/{n_forks} forks have "
            f"burn-in AUROC > {PC3_BURNIN_AUROC_THRESH} "
            f"(need >= {PC3_BURNIN_FORKS_MIN})"
        )

    all_pass = len(failures) == 0
    return all_pass, failures


# ---------------------------------------------------------------------------
# Evaluate predeclared outcome map (P1/F1, P2/F2)
# ---------------------------------------------------------------------------

def evaluate(rows: list[dict]) -> dict:
    """Evaluate P1/F1 and P2/F2; return evaluation dict."""
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

    # --- P1 / F1: deception exists ---
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

    # F1: pooled >= 0.5
    f1 = (
        not (isinstance(pool_n2_auroc, float) and math.isnan(pool_n2_auroc))
        and pool_n2_auroc >= F1_POOLED_MIN
    )

    # --- P2 / F2: oracle discrimination possible ---
    # P2: pooled oracle AUROC >= 0.7
    p2_pass = (
        not (isinstance(pool_oracle_auroc, float) and math.isnan(pool_oracle_auroc))
        and pool_oracle_auroc >= P2_ORACLE_POOLED_MIN
    )
    # F2: oracle pooled < 0.6
    f2 = (
        not (isinstance(pool_oracle_auroc, float) and math.isnan(pool_oracle_auroc))
        and pool_oracle_auroc < F2_ORACLE_POOLED_MAX
    )

    # --- Verdict ---
    positive = p1_pass and p2_pass
    negative = f1 or f2
    if positive:
        verdict_str = "POSITIVE"
    elif negative:
        verdict_str = "NEGATIVE"
    else:
        verdict_str = "MIXED"

    # --- Burn-in AUROC summary for reporting ---
    burnin_aurocs = [
        r["burnin_auroc"] for r in rows
        if not (isinstance(r["burnin_auroc"], float) and math.isnan(r["burnin_auroc"]))
    ]
    mean_burnin_auroc = float(np.mean(burnin_aurocs)) if burnin_aurocs else float("nan")

    return {
        # P1 / F1
        "pool_n2_auroc": pool_n2_auroc,
        "p1_pooled_pass": p1_pooled_pass,
        "p1_forks_below_count": p1_forks_below_count,
        "p1_forks_pass": p1_forks_pass,
        "p1_pass": p1_pass,
        "f1": f1,
        # P2 / F2
        "pool_oracle_auroc": pool_oracle_auroc,
        "p2_pass": p2_pass,
        "f2": f2,
        # Burn-in summary
        "mean_burnin_auroc": mean_burnin_auroc,
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
                "exp": 160,
                "fork_seed": r["fork_seed"],
                "regime": r["regime"],
                "n_total": r["n_total"],
                "n_correct_post": r["n_correct_post"],
                "n_incorrect_post": r["n_incorrect_post"],
                "ahat_drift": r["ahat_drift"],
                "burnin_auroc": _nan_to_none(r["burnin_auroc"]),
                "post_auroc_n2": _nan_to_none(r["post_auroc_n2"]),
                "post_auroc_oracle": _nan_to_none(r["post_auroc_oracle"]),
                "segment_n2_aurocs": [_nan_to_none(v) for v in r["segment_n2_aurocs"]],
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

        if ev is not None:
            summary: dict = {
                "exp": 160,
                "row_type": "summary",
                "pool_n2_auroc": _nan_to_none(ev["pool_n2_auroc"]),
                "p1_pooled_pass": ev["p1_pooled_pass"],
                "p1_forks_below_count": ev["p1_forks_below_count"],
                "p1_forks_pass": ev["p1_forks_pass"],
                "p1_pass": ev["p1_pass"],
                "f1": ev["f1"],
                "pool_oracle_auroc": _nan_to_none(ev["pool_oracle_auroc"]),
                "p2_pass": ev["p2_pass"],
                "f2": ev["f2"],
                "mean_burnin_auroc": _nan_to_none(ev["mean_burnin_auroc"]),
                "verdict": ev["verdict_str"],
            }
            fh.write(json.dumps(summary) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 160 — N3 rung 1 gate: R-SWAP reliability-inversion deception"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Smoke run: seed=[42] only, full 4000 steps, "
            "prints per-fork numbers including segment AUROC trace. "
            "No verdict written."
        ),
    )
    args = parser.parse_args()

    smoke = args.smoke
    seeds = [42] if smoke else SEEDS

    print("=" * 80)
    print("Exp 160 — N3 rung 1, discriminating-perturbation GATE")
    print("          R-SWAP: reliability-inversion deception world")
    print("=" * 80)
    print()
    print(f"SURPRISE_WINDOW={SURPRISE_WINDOW}  CLASSIFIER_WINDOW={CLASSIFIER_WINDOW}  "
          f"[must be equal; asserted]")
    assert CLASSIFIER_WINDOW == SURPRISE_WINDOW
    print(f"EVAL_EVERY={EVAL_EVERY}  ERROR_RATE_OK_THRESH={ERROR_RATE_OK_THRESH}  "
          f"LAG1_RHO_STRUCT_THRESH={LAG1_RHO_STRUCT_THRESH}")
    print(f"ALPHA={ALPHA}  EWMA_INIT={EWMA_INIT}  (EWMA channel, verbatim exp157/exp159)")
    print(f"P_TRUE_NOISY={P_TRUE_NOISY}  N_NOISY_P0={N_NOISY_P0}")
    print(f"BURN_IN_END={BURN_IN_END}  SWAP_PHASE_START={SWAP_PHASE_START}  "
          f"SWAP_INTERVAL={SWAP_INTERVAL}  N_SWAP_SEGMENTS={N_SWAP_SEGMENTS}")
    print(f"SWAP_PLACE_SEED_OFFSET={SWAP_PLACE_SEED_OFFSET}  "
          f"SWAP_NOISE_SEED_OFFSET={SWAP_NOISE_SEED_OFFSET}")
    print(f"Seeds: {seeds}  N_CHUNKS={N_CHUNKS}  "
          f"CHUNK_SIZE={CHUNK_SIZE}  N_STEPS={N_CHUNKS * CHUNK_SIZE}")
    print()
    print("Swap schedule:")
    print("  steps 0-999: noisy set = P0 (burn-in; EWMA learns the map)")
    print("  step 1000: FIRST SWAP — noisy set = P1 (inverted)")
    print("  step 1250: SECOND SWAP — noisy set = P0")
    print("  ... alternating every 250 steps (250 << 500-step relearn time)")
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

    # --- Ungated diagnostics ---
    print("Ungated diagnostics:")
    print("  Named ungated expectations:")
    print("    - segment N2 AUROC trace: predicted sawtooth (anti-calibration "
          "alternates with partial recovery every 250 steps)")
    print("    - classifier post-burn-in: NOISE-dominant (the typed alarm misses "
          "the swap structure — no lag-1 rho signal from swap timing)")
    print("    - terminal EWMA: negatively correlated with CURRENT truth "
          "(inverted relative to P0 burn-in map)")
    print()
    for r in rows:
        seg_strs = []
        for i, au in enumerate(r["segment_n2_aurocs"]):
            phase = "P1" if i % 2 == 0 else "P0"
            if isinstance(au, float) and not math.isnan(au):
                seg_strs.append(f"seg{i}({phase}):{au:.3f}")
            else:
                seg_strs.append(f"seg{i}({phase}):nan")
        print(f"  seed={r['fork_seed']}  burnin_auroc={r['burnin_auroc']:.4f}"
              f"  post_n2={r['post_auroc_n2']:.4f}"
              f"  post_oracle={r['post_auroc_oracle']:.4f}")
        print(f"    segment N2 AUROCs: {', '.join(seg_strs[:6])}")
        if len(seg_strs) > 6:
            print(f"                      {', '.join(seg_strs[6:])}")
        print(f"    post-burn-in classifier: "
              f"frac_ok={r['frac_ok_post']:.4f}  "
              f"frac_noise={r['frac_noise_post']:.4f}  "
              f"frac_struct={r['frac_structural_post']:.4f}  "
              f"(n_win={r['n_post_windows']})")
        print(f"    ewma_vs_P0_r={r['ewma_vs_p0_r']:.4f}  "
              f"p0_cells(first5)={r['p0_cells'][:5]}...")
        print()

    # --- Smoke exit ---
    if smoke:
        print("SMOKE ONLY — no verdict")
        return

    # --- Precondition check ---
    pc_pass, pc_failures = check_preconditions(rows)
    if pc_failures:
        print("PRECONDITION FAILED:")
        for f in pc_failures:
            print(f"  {f}")
        print()
        print("PRECONDITION FAILED — no verdict.")
        out_rows_path = Path(__file__).parent / "outputs" / "exp160_rows.json"
        write_json_rows(rows, out_rows_path, ev=None)
        print(f"JSON rows written to {out_rows_path} (no verdict)")
        return

    print("Preconditions: all PASS")
    print()

    # --- Evaluate predeclared outcome map ---
    ev = evaluate(rows)
    n = ev["n_forks"]

    print("=" * 80)
    print("PREDECLARED OUTCOME MAP")
    print("=" * 80)
    print()

    # Burn-in summary
    bi_aurocs_str = ", ".join(
        f"{r['burnin_auroc']:.4f}" if not (isinstance(r['burnin_auroc'], float) and math.isnan(r['burnin_auroc'])) else "nan"
        for r in rows
    )
    print(f"PC3 burn-in N2 AUROC (steps {BURN_IN_START_EVAL}-{BURN_IN_END - 1}):")
    print(f"  Per fork: [{bi_aurocs_str}]")
    print(f"  Mean: {ev['mean_burnin_auroc']:.4f}")
    print()

    # P1 detail
    pool_n2_str = (
        f"{ev['pool_n2_auroc']:.4f}"
        if not (isinstance(ev["pool_n2_auroc"], float) and math.isnan(ev["pool_n2_auroc"]))
        else "nan"
    )
    post_n2_aurocs_str = ", ".join(
        f"{r['post_auroc_n2']:.4f}" if not (isinstance(r['post_auroc_n2'], float) and math.isnan(r['post_auroc_n2'])) else "nan"
        for r in rows
    )
    print(f"P1 (deception exists — N2 anti-calibrated post-burn-in):")
    print(f"  Post-burn-in N2 AUROC per fork: [{post_n2_aurocs_str}]")
    print(f"  Forks with N2 AUROC < {P1_PER_FORK_THRESH}: "
          f"{ev['p1_forks_below_count']}/{n} (need >={P1_FORKS_BELOW_HALF_MIN} for P1)")
    print(f"  Pooled N2 AUROC: {pool_n2_str} (need <={P1_POOLED_MAX} for P1)")
    print(f"  FALSIFIER F1: pooled >= {F1_POOLED_MIN}: {'YES — F1 FIRES' if ev['f1'] else 'no'}")
    p1_status = "PASS" if ev["p1_pass"] else ("FALSIFIER F1" if ev["f1"] else "MIXED (not P1, not F1)")
    print(f"  => P1: {p1_status}")
    print()

    # P2 detail
    pool_oracle_str = (
        f"{ev['pool_oracle_auroc']:.4f}"
        if not (isinstance(ev["pool_oracle_auroc"], float) and math.isnan(ev["pool_oracle_auroc"]))
        else "nan"
    )
    post_oracle_aurocs_str = ", ".join(
        f"{r['post_auroc_oracle']:.4f}" if not (isinstance(r['post_auroc_oracle'], float) and math.isnan(r['post_auroc_oracle'])) else "nan"
        for r in rows
    )
    print(f"P2 (discrimination possible — oracle channel discriminates post-burn-in):")
    print(f"  Post-burn-in oracle AUROC per fork: [{post_oracle_aurocs_str}]")
    print(f"  Pooled oracle AUROC: {pool_oracle_str} (need >={P2_ORACLE_POOLED_MIN} for P2)")
    print(f"  FALSIFIER F2: oracle pooled < {F2_ORACLE_POOLED_MAX}: "
          f"{'YES — F2 FIRES' if ev['f2'] else 'no'}")
    p2_status = "PASS" if ev["p2_pass"] else ("FALSIFIER F2" if ev["f2"] else "MIXED (not P2, not F2)")
    print(f"  => P2: {p2_status}")
    print()

    # --- Conjunct summary + VERDICT ---
    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    print(f"P1 (pooled N2 AUROC <= {P1_POOLED_MAX} AND per-fork < {P1_PER_FORK_THRESH} "
          f"in >={P1_FORKS_BELOW_HALF_MIN}/{n}): {p1_status}")
    print(f"P2 (oracle pooled AUROC >= {P2_ORACLE_POOLED_MIN}): {p2_status}")
    print()
    if ev["verdict_str"] == "POSITIVE":
        print(f"VERDICT: POSITIVE")
        print("GATE PASSES — the discriminating perturbation EXISTS")
    elif ev["verdict_str"] == "NEGATIVE":
        print(f"VERDICT: NEGATIVE")
        print("GATE: this candidate FAILED")
    else:
        print(f"VERDICT: MIXED")
    print()
    print("=" * 80)

    # --- Write JSON rows ---
    out_rows_path = Path(__file__).parent / "outputs" / "exp160_rows.json"
    write_json_rows(rows, out_rows_path, ev=ev)
    print(f"JSON rows written to {out_rows_path}")

    # --- Write verdict JSON ---
    pool_n2_for_reason = pool_n2_str
    pool_oracle_for_reason = pool_oracle_str
    arms = {
        "P1_anticalibration": {
            "pass": bool(ev["p1_pass"]),
            "reason": (
                f"Post-burn-in pooled N2 AUROC={pool_n2_for_reason} "
                f"(need <={P1_POOLED_MAX}); "
                f"forks with N2 AUROC < {P1_PER_FORK_THRESH}: "
                f"{ev['p1_forks_below_count']}/{n} (need >={P1_FORKS_BELOW_HALF_MIN}); "
                f"F1 fired={ev['f1']}"
            ),
        },
        "P2_oracle_discriminable": {
            "pass": bool(ev["p2_pass"]),
            "reason": (
                f"Post-burn-in pooled oracle AUROC={pool_oracle_for_reason} "
                f"(need >={P2_ORACLE_POOLED_MIN}); "
                f"F2 fired={ev['f2']}"
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp160_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp160",
        arms=arms,
        verdict=ev["verdict_str"],
        halted=False,
        notes=(
            "N3 rung 1 gate: R-SWAP world swaps which 13-of-25 cells are noisy "
            "every 250 steps (250 << 500-step EWMA relearn time) after a 1000-step "
            "burn-in. P0 burn-in lets EWMA learn the map; post-burn-in swap inverts "
            "the learned reliability map. N2 channel (EWMA) is history-indexed and "
            "cannot follow the swap — predicted anti-calibration (type-2 AUROC < 0.5). "
            "Oracle channel reads the TRUE current reliability per step — serves as "
            "control proving the world itself is discriminable. "
            "POSITIVE => GATE PASSES, discriminating perturbation exists, N3 rung 2 opens. "
            "NEGATIVE kills this candidate only; other candidates remain."
        ),
    )
    print(f"Verdict JSON written to {verdict_path}")


if __name__ == "__main__":
    main()
