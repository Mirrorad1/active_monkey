"""Exp 159 — the N2 prereq RE-CONFIRMATION: with the built channel (Exp 157)
and classifier (Exp 158) in place of the channels that failed Exp 155, the
mirro body satisfies the meta-calibration-n3 card's N2 prereq — tested under
per-fork randomization of every regime parameter (the Exp 155 verifier's
determinism flag). PASS opens N3 rung 1.

Hypothesis: (a) the Exp 157 expected-uncertainty channel yields meta-d' > 0
where discrimination is possible, even under RANDOMIZED noise placement;
(b) the Exp 158 classifier separates irreducible noise from structural
mismatch under randomized noise levels and structural geometries.

Design: forks of mirro only (spine never saved), FRESH seeds 34-41 (Exp
155-158 used 0-33), 4000 steps per (fork, regime), exp155 step-loop
semantics. Per-fork randomization rng = np.random.default_rng(100_000 +
fork_seed). Regimes:
  R-CTRL:   standard world.
  R-NOISE:  NoisyCmap with p_true DRAWN from {0.6, 0.7, 0.8} per fork
            (cmap rng seed 70_000 + fork_seed).
  R-STRUCT: StructCmap with derangement DRAWN from {[1,2,0],[2,0,1]} and
            half_period DRAWN from {60, 100, 140} per fork.
  R-PLACE:  place-conditional noise with RANDOM placement — exactly 13 of
            the 25 cells drawn uniformly (without replacement, per-fork rng)
            are noisy (p_true = 0.55, cmap rng seed 90_000 + fork_seed),
            the other 12 clean. Random placement kills Exp 158's
            checkerboard-parity geometry.
All drawn parameters are logged per fork.

Instruments (both PROVIDED forms, contents self-formed from the creature's
own stream — exp157/exp158 definitions verbatim):
  Channel: per-cell EWMA of correctness (alpha 0.05, init 0.5), confidence
           read at argmax(qs) pre-update; type-2 AUROC vs correctness.
  Classifier: window 200 over correctness, evaluated every 100 steps;
           error_rate < 0.05 => OK; elif lag1_rho > 0.3 => STRUCTURAL;
           else NOISE. Majority label per (fork, regime) = most common
           label over its evaluated windows.

Preconditions (instrument-grade; failure => "PRECONDITION FAILED", rows
written, no verdict): PC1 >= 50 correct AND >= 50 incorrect per fork in
R-NOISE, R-STRUCT, R-PLACE; PC2 ahat_drift < 0.05 per (fork, regime);
PC3 >= 30 evaluated windows per (fork, regime); PC4 placement validity in
R-PLACE: pooled clean-cell minus noisy-cell accuracy >= 0.3 (classified by
TRUE position).

Predeclared properties and falsifiers (P1 uses Exp 155's original bands):
  P1 (meta-d' > 0 where discrimination is possible): in R-PLACE, channel
     type-2 AUROC > 0.55 per fork in >= 6/8 forks AND pooled AUROC >= 0.55.
     FALSIFIER F1: <= 4/8 forks with AUROC > 0.5, OR pooled <= 0.5.
  P2 (noise-vs-structure separation): majority label correct per regime —
     R-CTRL majority OK in >= 7/8 forks AND R-NOISE majority NOISE in
     >= 7/8 forks AND R-STRUCT majority STRUCTURAL in >= 7/8 forks — AND
     pooled window-level STRUCTURAL-fraction in R-NOISE <= 0.05.
     FALSIFIER F2: any of the three majority counts <= 5/8, OR pooled
     R-NOISE STRUCTURAL-fraction > 0.2.
  VERDICT: POSITIVE (prereq SATISFIED) iff P1 and P2 both pass. NEGATIVE
  iff F1 or F2 fires. Otherwise MIXED. "Not a falsifier" never counts
  toward POSITIVE.

Ungated diagnostics: R-PLACE classifier label distribution and mean lag1
rho per fork, with the NAMED UNGATED PREDICTION: random placement removes
the parity anti-correlation, so R-PLACE mean rho rises toward ~0 (vs
-0.14..-0.19 on the Exp 158 checkerboard) and the majority label stays
NOISE. Also: R-PLACE EWMA calibration Pearson r (terminal EWMA vs true
per-cell reliability); drawn parameters per fork; per-regime pooled mean
error rate and rho.
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

SEEDS = list(range(34, 42))   # fresh seeds — exp155-158 used 0-33

N_STEPS = 4000
N_CHUNKS = 40
CHUNK_SIZE = N_STEPS // N_CHUNKS    # 100

# Classifier window / evaluation parameters (verbatim from exp158)
CLASSIFIER_WINDOW = 200       # W = 200 (same as SURPRISE_WINDOW)
EVAL_EVERY = 100              # evaluate once window is full, every 100 steps
ERROR_RATE_OK_THRESH = 0.05   # error_rate < 0.05 => OK
LAG1_RHO_STRUCT_THRESH = 0.3  # lag1_rho > 0.3 => STRUCTURAL (else NOISE)

# EWMA channel parameters (verbatim from exp157)
ALPHA = 0.05       # EWMA learning rate
EWMA_INIT = 0.5    # initial EWMA value per cell

# Regime seed offsets (matching exp158 conventions)
NOISE_SEED_OFFSET = 70_000        # R-NOISE NoisyCmap rng: seed 70_000+fork_seed
STRUCT_GEOM_SEED_OFFSET = 80_000  # R-STRUCT geometry rng: seed 80_000+fork_seed
PLACE_SEED_OFFSET = 90_000        # R-PLACE cmap rng: seed 90_000+fork_seed
FORK_PARAM_SEED_OFFSET = 100_000  # Per-fork randomization rng: seed 100_000+fork_seed

# R-NOISE: p_true drawn from this set per fork
P_TRUE_NOISE_OPTIONS = [0.6, 0.7, 0.8]

# R-STRUCT geometry options
DERANGEMENT_OPTIONS = [[1, 2, 0], [2, 0, 1]]
HALF_PERIOD_OPTIONS = [60, 100, 140]

# R-PLACE parameters
P_TRUE_PLACE = 0.55
N_NOISY_CELLS = 13    # exactly 13 of 25 cells drawn as noisy per fork

# Precondition thresholds
PRECONDITION_MIN_CLASS = 50       # PC1: >= 50 correct AND incorrect per fork in alarm regimes
PRECONDITION_AHAT_DRIFT = 0.05    # PC2: max abs A_hat change < 0.05
PRECONDITION_MIN_WINDOWS = 30     # PC3: >= 30 evaluated windows per (fork, regime)
PC4_ACC_DIFF_MIN = 0.3            # PC4: pooled clean_acc - noisy_acc >= 0.3 in R-PLACE

# P1 thresholds (Exp 155's original bands)
P1_FORKS_AUROC_MIN = 6            # >= 6/8 forks with channel AUROC > 0.55
P1_PER_FORK_THRESH = 0.55         # per-fork AUROC threshold for P1 count
P1_POOLED_MIN = 0.55              # pooled AUROC >= 0.55
F1_FORKS_MAX = 4                  # F1a: <= 4/8 forks with AUROC > 0.5 => F1
F1_PER_FORK_THRESH = 0.5          # per-fork AUROC threshold for F1 count
F1_POOLED_MAX = 0.5               # F1b: pooled <= 0.5 => F1

# P2 thresholds
P2_CTRL_OK_FORKS_MIN = 7          # >= 7/8 forks with majority OK in R-CTRL
P2_NOISE_NOISE_FORKS_MIN = 7      # >= 7/8 forks with majority NOISE in R-NOISE
P2_STRUCT_STRUCT_FORKS_MIN = 7    # >= 7/8 forks with majority STRUCTURAL in R-STRUCT
P2_POOLED_STRUCT_FRAC_MAX = 0.05  # pooled R-NOISE STRUCTURAL-fraction <= 0.05
F2_MAJORITY_COUNT_MAX = 5         # F2: any majority count <= 5/8 fires F2
F2_POOLED_STRUCT_FRAC_HIGH = 0.2  # F2: pooled R-NOISE STRUCTURAL-fraction > 0.2

ALL_REGIMES = ("R-CTRL", "R-NOISE", "R-STRUCT", "R-PLACE")
SMOKE_REGIMES = ("R-NOISE", "R-PLACE")


# ---------------------------------------------------------------------------
# NoisyCmap — verbatim from exp158 (which copies exp155)
# ---------------------------------------------------------------------------

class NoisyCmap:
    """Color map wrapper with irreducible observation noise.

    Each lookup returns the true color with probability p_true, otherwise a
    uniform draw among the other colors.  Uses its OWN seeded rng so the
    creature's action stream is untouched (trajectory determinism preserved).
    """

    def __init__(self, base_cmap, n_colors, p_true, seed):
        self.base = list(base_cmap)
        self.n_colors = int(n_colors)
        self.p_true = float(p_true)
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, s):
        true = self.base[s]
        if self.rng.random() < self.p_true:
            return true
        others = [c for c in range(self.n_colors) if c != true]
        return int(self.rng.choice(others))

    def __len__(self):
        return len(self.base)


# ---------------------------------------------------------------------------
# StructCmap — verbatim from exp158 (which copies exp155)
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
# make_derangement — verbatim from exp158 (which copies exp155)
# ---------------------------------------------------------------------------

def make_derangement(n_colors: int) -> list[int]:
    """Return a fixed derangement (no fixed points) of range(n_colors).

    For n_colors=3 returns [1, 2, 0].
    For general n: cyclic shift by 1 is always a derangement.
    """
    return [(i + 1) % n_colors for i in range(n_colors)]


# ---------------------------------------------------------------------------
# RandomPlaceNoiseCmap — per-fork random-placement noisy cells
# ---------------------------------------------------------------------------

class RandomPlaceNoiseCmap:
    """Color map wrapper with per-fork randomly-placed noisy cells.

    noisy_cells: sorted list of 13 cell indices drawn uniformly w/o replacement
                 by the per-fork randomization rng (passed in as noisy_cells).
    Noisy cells: true color returned with p_true = 0.55; otherwise uniform
                 draw among the other n_colors-1 colors.
    Clean cells: true color always returned (no rng draw).

    The rng is private (seeded 90_000+fork_seed); action stream is untouched.
    Random placement (rather than checkerboard) removes the parity
    anti-correlation that Exp 158's R-PLACE-NOISE exploited.
    """

    def __init__(self, base_cmap, n_colors, noisy_cells: list[int], p_true, seed):
        self.base = list(base_cmap)
        self.n_colors = int(n_colors)
        self.noisy_cells = set(noisy_cells)
        self.p_true = float(p_true)
        self.rng = np.random.default_rng(seed)

    def is_noisy(self, cell: int) -> bool:
        """True iff this cell is in the randomly-drawn noisy set."""
        return cell in self.noisy_cells

    def __getitem__(self, s: int) -> int:
        true = self.base[s]
        if self.is_noisy(s):
            if self.rng.random() < self.p_true:
                return true
            others = [c for c in range(self.n_colors) if c != true]
            return int(self.rng.choice(others))
        else:
            return true

    def __len__(self) -> int:
        return len(self.base)


# ---------------------------------------------------------------------------
# lag1_rho — verbatim from exp158 (which copies exp155)
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
      rho: lag1_rho(window)

    lag1_rho of a zero-variance window returns 0.0.
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
# Mann-Whitney AUROC — verbatim from exp157 (which copies exp155)
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
# Draw per-fork regime parameters
# ---------------------------------------------------------------------------

def draw_fork_params(fork_seed: int) -> dict:
    """Draw per-fork randomization parameters using rng = default_rng(100_000 + fork_seed).

    Returns dict with:
      p_true_noise: float (one of P_TRUE_NOISE_OPTIONS)
      derangement: list[int]
      half_period: int
      noisy_cells: sorted list[int] of 13 cell indices
    """
    rng = np.random.default_rng(FORK_PARAM_SEED_OFFSET + fork_seed)

    # R-NOISE: p_true drawn from {0.6, 0.7, 0.8}
    p_true_noise = float(
        P_TRUE_NOISE_OPTIONS[int(rng.integers(0, len(P_TRUE_NOISE_OPTIONS)))]
    )

    # R-STRUCT: derangement and half_period
    derangement = list(DERANGEMENT_OPTIONS[int(rng.integers(0, len(DERANGEMENT_OPTIONS)))])
    half_period = int(HALF_PERIOD_OPTIONS[int(rng.integers(0, len(HALF_PERIOD_OPTIONS)))])

    # R-PLACE: 13 noisy cells drawn without replacement from range(25)
    noisy_cells_arr = rng.choice(25, size=N_NOISY_CELLS, replace=False)
    noisy_cells = sorted(int(x) for x in noisy_cells_arr)

    return {
        "p_true_noise": p_true_noise,
        "derangement": derangement,
        "half_period": half_period,
        "noisy_cells": noisy_cells,
    }


# ---------------------------------------------------------------------------
# Run one fork in one regime — step loop verbatim from exp158 (which copies exp155)
# Extended with per-cell EWMA channel (verbatim from exp157)
# ---------------------------------------------------------------------------

def run_fork_regime(
    mirro: Creature,
    fork_seed: int,
    regime: str,
    base_cmap: list,
    n_colors: int,
    fork_params: dict,
    n_chunks: int = N_CHUNKS,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """Run one fresh fork of mirro in one regime.

    Returns aggregates and per-step arrays for evaluation.

    Step loop replicates exp155/exp158 run_fork_regime semantics EXACTLY:
      - A_hat pre-step
      - pred_probs = A_hat @ qs
      - o_hat = argmax(pred_probs)
      - bel_cell = argmax(qs) pre-update
      - conf_channel = ewma[bel_cell] (EWMA channel, pre-update)
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

    fork_name = f"exp159_s{fork_seed}_{regime}"
    c = mirro.fork(fork_name)

    n_cells = c.world.n_cells
    n_actions = 4

    # --- Build regime-specific cmap using pre-drawn fork params ---
    if regime == "R-CTRL":
        cmap_obj = base_cmap  # plain list
        drawn_params: dict = {}
    elif regime == "R-NOISE":
        p_true_noise = fork_params["p_true_noise"]
        noise_seed = NOISE_SEED_OFFSET + fork_seed
        cmap_obj = NoisyCmap(base_cmap, n_colors, p_true_noise, seed=noise_seed)
        drawn_params = {"p_true": p_true_noise}
    elif regime == "R-STRUCT":
        derangement = fork_params["derangement"]
        half_period = fork_params["half_period"]
        cmap_obj = StructCmap(base_cmap, n_colors, derangement, half_period=half_period)
        drawn_params = {"derangement": derangement, "half_period": half_period}
    elif regime == "R-PLACE":
        noisy_cells = fork_params["noisy_cells"]
        place_seed = PLACE_SEED_OFFSET + fork_seed
        cmap_obj = RandomPlaceNoiseCmap(
            base_cmap=base_cmap,
            n_colors=n_colors,
            noisy_cells=noisy_cells,
            p_true=P_TRUE_PLACE,
            seed=place_seed,
        )
        drawn_params = {"noisy_cells": noisy_cells}
    else:
        raise ValueError(f"Unknown regime: {regime}")

    B = c.world.transition_matrix()   # (n_cells, n_cells, 4)

    n_total = n_chunks * chunk_size

    # A_hat at start for drift check
    A_hat_start = c._A_hat().copy()

    # --- Per-cell EWMA state (the internal confidence channel, verbatim from exp157) ---
    ewma = np.full(n_cells, EWMA_INIT, dtype=np.float64)

    # --- Per-step storage for AUROC computation ---
    conf_channel_arr = np.empty(n_total, dtype=np.float64)
    correct_arr = np.empty(n_total, dtype=np.int32)
    true_pos_arr = np.empty(n_total, dtype=np.int32)

    # --- Classifier state: rolling correctness window (verbatim from exp158) ---
    correct_window = _deque(maxlen=CLASSIFIER_WINDOW)

    # --- Per-window records ---
    window_labels: list[str] = []
    window_error_rates: list[float] = []
    window_rhos: list[float] = []

    # --- Accumulators for preconditions ---
    n_correct_total = 0
    n_incorrect_total = 0

    # --- Step loop ---
    global_step = 0
    eps = 1e-300

    for chunk_idx in range(n_chunks):
        # Deterministic per-chunk seed derived from fork_seed and chunk index
        chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF
        rng = np.random.default_rng(chunk_seed)

        for _step_in_chunk in range(chunk_size):
            t = global_step

            # Update step counter for R-STRUCT before reading cmap
            if regime == "R-STRUCT":
                cmap_obj.current_step = t

            # --- Compute A_hat (pre-step) ---
            A_hat = c._A_hat()   # shape (n_colors, n_cells)

            # --- Predicted observation (BEFORE update) ---
            pred_probs = A_hat @ c.qs   # shape (n_colors,)
            o_hat = int(np.argmax(pred_probs))

            # --- bel_cell and EWMA channel confidence (BEFORE update) ---
            bel_cell = int(np.argmax(c.qs))
            conf_channel = float(ewma[bel_cell])

            # --- Observe (from regime-specific cmap) ---
            true_pos_t = c.true_pos
            obs = int(cmap_obj[true_pos_t])

            # --- Correctness ---
            correct_t = 1 if (o_hat == obs) else 0
            if correct_t == 1:
                n_correct_total += 1
            else:
                n_incorrect_total += 1

            # --- Store per-step records ---
            conf_channel_arr[t] = conf_channel
            correct_arr[t] = correct_t
            true_pos_arr[t] = true_pos_t

            # --- Append to classifier rolling window ---
            correct_window.append(correct_t)

            # --- Evaluate classifier at every EVAL_EVERY steps once window full ---
            if (t + 1) % EVAL_EVERY == 0 and len(correct_window) == CLASSIFIER_WINDOW:
                win_arr = np.array(correct_window, dtype=np.float64)
                label, err_rate, rho = classify_window(win_arr)
                window_labels.append(label)
                window_error_rates.append(err_rate)
                window_rhos.append(rho)

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

            # --- EWMA update AFTER correctness known (verbatim from exp157) ---
            ewma[bel_cell] = (1.0 - ALPHA) * ewma[bel_cell] + ALPHA * correct_t

            global_step += 1

    # A_hat at end for drift check
    A_hat_end = c._A_hat()
    ahat_drift = float(np.abs(A_hat_end - A_hat_start).max())

    # --- Per-(fork, regime) aggregates ---
    n_windows = len(window_labels)
    n_ok = window_labels.count("OK")
    n_noise = window_labels.count("NOISE")
    n_structural = window_labels.count("STRUCTURAL")

    frac_ok = n_ok / n_windows if n_windows > 0 else float("nan")
    frac_noise = n_noise / n_windows if n_windows > 0 else float("nan")
    frac_structural = n_structural / n_windows if n_windows > 0 else float("nan")

    majority_label: str
    if n_windows > 0:
        counts = {"OK": n_ok, "NOISE": n_noise, "STRUCTURAL": n_structural}
        majority_label = max(counts, key=lambda k: counts[k])
    else:
        majority_label = "NONE"

    mean_error_rate = float(np.mean(window_error_rates)) if window_error_rates else float("nan")
    mean_rho = float(np.mean(window_rhos)) if window_rhos else float("nan")

    # --- AUROC (channel type-2 AUROC vs correctness) ---
    pos_conf = conf_channel_arr[correct_arr == 1]
    neg_conf = conf_channel_arr[correct_arr == 0]
    if n_correct_total >= PRECONDITION_MIN_CLASS and n_incorrect_total >= PRECONDITION_MIN_CLASS:
        auroc = mannwhitney_auroc(pos_conf, neg_conf)
    else:
        auroc = float("nan")

    # --- R-PLACE: clean vs noisy accuracy and EWMA calibration ---
    clean_acc = float("nan")
    noisy_acc = float("nan")
    ewma_calibration_r = float("nan")
    terminal_ewma: list = []
    cell_true_reliability: list = []
    if regime == "R-PLACE":
        noisy_set = set(fork_params["noisy_cells"])
        cell_is_noisy_arr = np.array(
            [c_idx in noisy_set for c_idx in range(n_cells)], dtype=bool
        )
        true_cell_noisy = cell_is_noisy_arr[true_pos_arr]
        clean_mask = ~true_cell_noisy
        noisy_mask = true_cell_noisy
        n_clean = int(clean_mask.sum())
        n_noisy_steps = int(noisy_mask.sum())
        clean_acc = float(correct_arr[clean_mask].mean()) if n_clean > 0 else float("nan")
        noisy_acc = float(correct_arr[noisy_mask].mean()) if n_noisy_steps > 0 else float("nan")
        # Terminal EWMA vs true per-cell reliability — Pearson r over 25 cells
        terminal_ewma = ewma.tolist()
        cell_true_reliability = [
            P_TRUE_PLACE if (c_idx in noisy_set) else 1.0 for c_idx in range(n_cells)
        ]
        x = ewma
        y = np.array(cell_true_reliability)
        xm = x - x.mean()
        ym = y - y.mean()
        denom_r = math.sqrt(float((xm ** 2).sum()) * float((ym ** 2).sum()))
        if denom_r > 1e-300:
            ewma_calibration_r = float((xm * ym).sum() / denom_r)
        else:
            ewma_calibration_r = 0.0

    result: dict = {
        "fork_seed": fork_seed,
        "regime": regime,
        "n_total": n_total,
        "n_correct": n_correct_total,
        "n_incorrect": n_incorrect_total,
        "ahat_drift": ahat_drift,
        "n_windows": n_windows,
        "n_ok": n_ok,
        "n_noise": n_noise,
        "n_structural": n_structural,
        "frac_ok": frac_ok,
        "frac_noise": frac_noise,
        "frac_structural": frac_structural,
        "majority_label": majority_label,
        "mean_error_rate": mean_error_rate,
        "mean_rho": mean_rho,
        "auroc": auroc,
        "drawn_params": drawn_params,
        # For R-PLACE diagnostics
        "clean_acc": clean_acc,
        "noisy_acc": noisy_acc,
        "ewma_calibration_r": ewma_calibration_r,
        # Full arrays for pooled AUROC (R-PLACE)
        "_conf_channel": conf_channel_arr,
        "_correct": correct_arr,
    }
    if terminal_ewma:
        result["terminal_ewma"] = terminal_ewma
    if cell_true_reliability:
        result["cell_true_reliability"] = cell_true_reliability
    return result


# ---------------------------------------------------------------------------
# Run all seeds for a given set of regimes
# ---------------------------------------------------------------------------

def run_all(
    seeds: list[int],
    mirro: Creature,
    base_cmap: list,
    n_colors: int,
    regimes: tuple[str, ...],
    n_chunks: int = N_CHUNKS,
    verbose: bool = True,
) -> list[dict]:
    """Run all (seed, regime) pairs; return list of result dicts."""
    rows: list[dict] = []
    for seed in seeds:
        # Draw per-fork parameters once for all regimes of this fork
        fork_params = draw_fork_params(seed)
        if verbose:
            print(
                f"  seed={seed} params: "
                f"p_true_noise={fork_params['p_true_noise']}  "
                f"derangement={fork_params['derangement']}  "
                f"half_period={fork_params['half_period']}  "
                f"noisy_cells(first5)={fork_params['noisy_cells'][:5]}...",
                flush=True,
            )
        for regime in regimes:
            if verbose:
                print(f"    Running seed={seed} regime={regime} ...", flush=True)
            r = run_fork_regime(
                mirro=mirro,
                fork_seed=seed,
                regime=regime,
                base_cmap=base_cmap,
                n_colors=n_colors,
                fork_params=fork_params,
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
        f"{'regime':>8}  "
        f"{'n_cor':>6}  "
        f"{'n_inc':>6}  "
        f"{'auroc':>7}  "
        f"{'n_win':>6}  "
        f"{'frac_ok':>8}  "
        f"{'frac_no':>8}  "
        f"{'frac_st':>8}  "
        f"{'majority':>10}  "
        f"{'mean_err':>9}  "
        f"{'mean_rho':>9}  "
        f"{'drift':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        au = f"{r['auroc']:.4f}" if not (isinstance(r['auroc'], float) and math.isnan(r['auroc'])) else "   nan"
        fo = f"{r['frac_ok']:.4f}" if not (isinstance(r['frac_ok'], float) and math.isnan(r['frac_ok'])) else "   nan"
        fn = f"{r['frac_noise']:.4f}" if not (isinstance(r['frac_noise'], float) and math.isnan(r['frac_noise'])) else "   nan"
        fs = f"{r['frac_structural']:.4f}" if not (isinstance(r['frac_structural'], float) and math.isnan(r['frac_structural'])) else "   nan"
        me = f"{r['mean_error_rate']:.4f}" if not (isinstance(r['mean_error_rate'], float) and math.isnan(r['mean_error_rate'])) else "    nan"
        mr = f"{r['mean_rho']:.4f}" if not (isinstance(r['mean_rho'], float) and math.isnan(r['mean_rho'])) else "    nan"
        print(
            f"{r['fork_seed']:>5}  "
            f"{r['regime']:>8}  "
            f"{r['n_correct']:>6}  "
            f"{r['n_incorrect']:>6}  "
            f"{au:>7}  "
            f"{r['n_windows']:>6}  "
            f"{fo:>8}  "
            f"{fn:>8}  "
            f"{fs:>8}  "
            f"{r['majority_label']:>10}  "
            f"{me:>9}  "
            f"{mr:>9}  "
            f"{r['ahat_drift']:>8.4f}"
        )


# ---------------------------------------------------------------------------
# Precondition check
# ---------------------------------------------------------------------------

def check_preconditions(rows: list[dict]) -> tuple[bool, list[str]]:
    """Check PC1, PC2, PC3, PC4.  Returns (all_pass, list_of_failures)."""
    failures: list[str] = []

    for r in rows:
        seed = r["fork_seed"]
        regime = r["regime"]

        # PC1: >= 50 correct AND >= 50 incorrect in alarm regimes
        if regime in ("R-NOISE", "R-STRUCT", "R-PLACE"):
            if r["n_correct"] < PRECONDITION_MIN_CLASS:
                failures.append(
                    f"PC1 FAIL: seed={seed} {regime} n_correct={r['n_correct']} "
                    f"< {PRECONDITION_MIN_CLASS}"
                )
            if r["n_incorrect"] < PRECONDITION_MIN_CLASS:
                failures.append(
                    f"PC1 FAIL: seed={seed} {regime} n_incorrect={r['n_incorrect']} "
                    f"< {PRECONDITION_MIN_CLASS}"
                )

        # PC2: ahat_drift < 0.05
        if r["ahat_drift"] >= PRECONDITION_AHAT_DRIFT:
            failures.append(
                f"PC2 FAIL: seed={seed} {regime} ahat_drift={r['ahat_drift']:.4f} "
                f">= {PRECONDITION_AHAT_DRIFT}"
            )

        # PC3: >= 30 evaluated windows per (fork, regime)
        if r["n_windows"] < PRECONDITION_MIN_WINDOWS:
            failures.append(
                f"PC3 FAIL: seed={seed} {regime} n_windows={r['n_windows']} "
                f"< {PRECONDITION_MIN_WINDOWS}"
            )

    # PC4: R-PLACE pooled clean_acc - noisy_acc >= 0.3
    place_rows = [r for r in rows if r["regime"] == "R-PLACE"]
    if place_rows:
        clean_accs = [
            r["clean_acc"] for r in place_rows
            if not (isinstance(r["clean_acc"], float) and math.isnan(r["clean_acc"]))
        ]
        noisy_accs = [
            r["noisy_acc"] for r in place_rows
            if not (isinstance(r["noisy_acc"], float) and math.isnan(r["noisy_acc"]))
        ]
        if clean_accs and noisy_accs:
            pooled_clean = float(np.mean(clean_accs))
            pooled_noisy = float(np.mean(noisy_accs))
            acc_diff = pooled_clean - pooled_noisy
            if acc_diff < PC4_ACC_DIFF_MIN:
                failures.append(
                    f"PC4 FAIL: R-PLACE pooled clean_acc({pooled_clean:.4f}) - "
                    f"noisy_acc({pooled_noisy:.4f}) = {acc_diff:.4f} < {PC4_ACC_DIFF_MIN}"
                )
        else:
            failures.append("PC4 FAIL: could not compute pooled R-PLACE clean/noisy accuracy (nan)")

    all_pass = len(failures) == 0
    return all_pass, failures


# ---------------------------------------------------------------------------
# Evaluate predeclared outcome map (P1/F1, P2/F2)
# ---------------------------------------------------------------------------

def evaluate(rows: list[dict]) -> dict:
    """Evaluate P1/F1 and P2/F2; return evaluation dict."""
    ctrl_rows = [r for r in rows if r["regime"] == "R-CTRL"]
    noise_rows = [r for r in rows if r["regime"] == "R-NOISE"]
    struct_rows = [r for r in rows if r["regime"] == "R-STRUCT"]
    place_rows = [r for r in rows if r["regime"] == "R-PLACE"]

    n_forks = len(SEEDS)  # 8

    # --- P1 / F1: R-PLACE channel AUROC ---
    # P1: channel AUROC > 0.55 per fork in >= 6/8 forks AND pooled >= 0.55
    p1_forks_pass_count = sum(
        1 for r in place_rows
        if not (isinstance(r["auroc"], float) and math.isnan(r["auroc"]))
        and r["auroc"] > P1_PER_FORK_THRESH
    )
    p1_forks_pass = p1_forks_pass_count >= P1_FORKS_AUROC_MIN

    # Pooled R-PLACE AUROC (concatenated trials)
    pool_place_auroc = pooled_auroc(
        [r["_conf_channel"] for r in place_rows],
        [r["_correct"] for r in place_rows],
    ) if place_rows else float("nan")
    p1_pooled_pass = (
        not (isinstance(pool_place_auroc, float) and math.isnan(pool_place_auroc))
        and pool_place_auroc >= P1_POOLED_MIN
    )
    p1_pass = p1_forks_pass and p1_pooled_pass

    # F1a: <= 4/8 forks with AUROC > 0.5
    f1_forks_low_count = sum(
        1 for r in place_rows
        if not (isinstance(r["auroc"], float) and math.isnan(r["auroc"]))
        and r["auroc"] > F1_PER_FORK_THRESH
    )
    f1a = f1_forks_low_count <= F1_FORKS_MAX

    # F1b: pooled <= 0.5
    f1b = (
        not (isinstance(pool_place_auroc, float) and math.isnan(pool_place_auroc))
        and pool_place_auroc <= F1_POOLED_MAX
    )
    f1 = f1a or f1b

    # --- P2 / F2: majority label classification ---
    # P2a: R-CTRL majority OK in >= 7/8 forks
    p2_ctrl_ok_count = sum(
        1 for r in ctrl_rows if r["majority_label"] == "OK"
    )
    p2_ctrl_pass = p2_ctrl_ok_count >= P2_CTRL_OK_FORKS_MIN

    # P2b: R-NOISE majority NOISE in >= 7/8 forks
    p2_noise_noise_count = sum(
        1 for r in noise_rows if r["majority_label"] == "NOISE"
    )
    p2_noise_pass = p2_noise_noise_count >= P2_NOISE_NOISE_FORKS_MIN

    # P2c: R-STRUCT majority STRUCTURAL in >= 7/8 forks
    p2_struct_struct_count = sum(
        1 for r in struct_rows if r["majority_label"] == "STRUCTURAL"
    )
    p2_struct_pass = p2_struct_struct_count >= P2_STRUCT_STRUCT_FORKS_MIN

    # P2d: pooled window-level STRUCTURAL-fraction in R-NOISE <= 0.05
    total_noise_windows = sum(r["n_windows"] for r in noise_rows)
    total_noise_structural = sum(r["n_structural"] for r in noise_rows)
    pooled_noise_struct_frac = (
        total_noise_structural / total_noise_windows
        if total_noise_windows > 0 else float("nan")
    )
    p2_pooled_pass = (
        not (isinstance(pooled_noise_struct_frac, float) and math.isnan(pooled_noise_struct_frac))
        and pooled_noise_struct_frac <= P2_POOLED_STRUCT_FRAC_MAX
    )
    p2_pass = p2_ctrl_pass and p2_noise_pass and p2_struct_pass and p2_pooled_pass

    # F2: any majority count <= 5/8 OR pooled R-NOISE STRUCTURAL > 0.2
    f2_ctrl = p2_ctrl_ok_count <= F2_MAJORITY_COUNT_MAX
    f2_noise = p2_noise_noise_count <= F2_MAJORITY_COUNT_MAX
    f2_struct = p2_struct_struct_count <= F2_MAJORITY_COUNT_MAX
    f2_pooled = (
        not (isinstance(pooled_noise_struct_frac, float) and math.isnan(pooled_noise_struct_frac))
        and pooled_noise_struct_frac > F2_POOLED_STRUCT_FRAC_HIGH
    )
    f2 = f2_ctrl or f2_noise or f2_struct or f2_pooled

    # --- Verdict ---
    positive = p1_pass and p2_pass
    negative = f1 or f2
    if positive:
        verdict_str = "POSITIVE"
    elif negative:
        verdict_str = "NEGATIVE"
    else:
        verdict_str = "MIXED"

    # --- Per-regime pooled diagnostics ---
    def _pooled_mean(rows_list: list[dict], field: str) -> float:
        vals = [
            r[field] for r in rows_list
            if not (isinstance(r[field], float) and math.isnan(r[field]))
        ]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        # P1 / F1
        "p1_forks_pass_count": p1_forks_pass_count,
        "p1_forks_pass": p1_forks_pass,
        "pool_place_auroc": pool_place_auroc,
        "p1_pooled_pass": p1_pooled_pass,
        "p1_pass": p1_pass,
        "f1_forks_low_count": f1_forks_low_count,
        "f1a": f1a,
        "f1b": f1b,
        "f1": f1,
        # P2 / F2
        "p2_ctrl_ok_count": p2_ctrl_ok_count,
        "p2_ctrl_pass": p2_ctrl_pass,
        "p2_noise_noise_count": p2_noise_noise_count,
        "p2_noise_pass": p2_noise_pass,
        "p2_struct_struct_count": p2_struct_struct_count,
        "p2_struct_pass": p2_struct_pass,
        "pooled_noise_struct_frac": pooled_noise_struct_frac,
        "p2_pooled_pass": p2_pooled_pass,
        "p2_pass": p2_pass,
        "f2_ctrl": f2_ctrl,
        "f2_noise": f2_noise,
        "f2_struct": f2_struct,
        "f2_pooled": f2_pooled,
        "f2": f2,
        # Per-regime pooled diagnostics
        "ctrl_mean_error_rate": _pooled_mean(ctrl_rows, "mean_error_rate"),
        "ctrl_mean_rho": _pooled_mean(ctrl_rows, "mean_rho"),
        "noise_mean_error_rate": _pooled_mean(noise_rows, "mean_error_rate"),
        "noise_mean_rho": _pooled_mean(noise_rows, "mean_rho"),
        "struct_mean_error_rate": _pooled_mean(struct_rows, "mean_error_rate"),
        "struct_mean_rho": _pooled_mean(struct_rows, "mean_rho"),
        "place_mean_error_rate": _pooled_mean(place_rows, "mean_error_rate"),
        "place_mean_rho": _pooled_mean(place_rows, "mean_rho"),
        # Verdict
        "verdict_str": verdict_str,
        "n_forks": n_forks,
    }


# ---------------------------------------------------------------------------
# Write JSONL rows + summary
# ---------------------------------------------------------------------------

def write_json_rows(rows: list[dict], path: Path, ev: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            row: dict = {
                "exp": 159,
                "fork_seed": r["fork_seed"],
                "regime": r["regime"],
                "n_total": r["n_total"],
                "n_correct": r["n_correct"],
                "n_incorrect": r["n_incorrect"],
                "ahat_drift": r["ahat_drift"],
                "n_windows": r["n_windows"],
                "n_ok": r["n_ok"],
                "n_noise": r["n_noise"],
                "n_structural": r["n_structural"],
                "frac_ok": r["frac_ok"] if not (isinstance(r["frac_ok"], float) and math.isnan(r["frac_ok"])) else None,
                "frac_noise": r["frac_noise"] if not (isinstance(r["frac_noise"], float) and math.isnan(r["frac_noise"])) else None,
                "frac_structural": r["frac_structural"] if not (isinstance(r["frac_structural"], float) and math.isnan(r["frac_structural"])) else None,
                "majority_label": r["majority_label"],
                "mean_error_rate": r["mean_error_rate"] if not (isinstance(r["mean_error_rate"], float) and math.isnan(r["mean_error_rate"])) else None,
                "mean_rho": r["mean_rho"] if not (isinstance(r["mean_rho"], float) and math.isnan(r["mean_rho"])) else None,
                "auroc": r["auroc"] if not (isinstance(r["auroc"], float) and math.isnan(r["auroc"])) else None,
                "drawn_params": r["drawn_params"],
            }
            if r["regime"] == "R-PLACE":
                row["clean_acc"] = r["clean_acc"] if not (isinstance(r["clean_acc"], float) and math.isnan(r["clean_acc"])) else None
                row["noisy_acc"] = r["noisy_acc"] if not (isinstance(r["noisy_acc"], float) and math.isnan(r["noisy_acc"])) else None
                row["ewma_calibration_r"] = r["ewma_calibration_r"] if not (isinstance(r["ewma_calibration_r"], float) and math.isnan(r["ewma_calibration_r"])) else None
            fh.write(json.dumps(row) + "\n")

        if ev is not None:
            pnsf = ev["pooled_noise_struct_frac"]
            summary: dict = {
                "exp": 159,
                "row_type": "summary",
                "p1_forks_pass_count": ev["p1_forks_pass_count"],
                "p1_forks_pass": ev["p1_forks_pass"],
                "pool_place_auroc": ev["pool_place_auroc"] if not (isinstance(ev["pool_place_auroc"], float) and math.isnan(ev["pool_place_auroc"])) else None,
                "p1_pooled_pass": ev["p1_pooled_pass"],
                "p1_pass": ev["p1_pass"],
                "f1_forks_low_count": ev["f1_forks_low_count"],
                "f1a": ev["f1a"],
                "f1b": ev["f1b"],
                "f1": ev["f1"],
                "p2_ctrl_ok_count": ev["p2_ctrl_ok_count"],
                "p2_ctrl_pass": ev["p2_ctrl_pass"],
                "p2_noise_noise_count": ev["p2_noise_noise_count"],
                "p2_noise_pass": ev["p2_noise_pass"],
                "p2_struct_struct_count": ev["p2_struct_struct_count"],
                "p2_struct_pass": ev["p2_struct_pass"],
                "pooled_noise_struct_frac": pnsf if not (isinstance(pnsf, float) and math.isnan(pnsf)) else None,
                "p2_pooled_pass": ev["p2_pooled_pass"],
                "p2_pass": ev["p2_pass"],
                "f2_ctrl": ev["f2_ctrl"],
                "f2_noise": ev["f2_noise"],
                "f2_struct": ev["f2_struct"],
                "f2_pooled": ev["f2_pooled"],
                "f2": ev["f2"],
                "verdict": ev["verdict_str"],
            }
            fh.write(json.dumps(summary) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 159 — N2 prereq re-confirmation (randomized parameters)"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Smoke run: seed=[34] only, full 40 chunks, "
            "regimes R-NOISE and R-PLACE only, no verdict written."
        ),
    )
    args = parser.parse_args()

    smoke = args.smoke
    seeds = [34] if smoke else SEEDS
    regimes = SMOKE_REGIMES if smoke else ALL_REGIMES

    print("=" * 80)
    print("Exp 159 — N2 prereq re-confirmation (Exp 157 channel + Exp 158 classifier,")
    print("          per-fork randomized regime parameters)")
    print("=" * 80)
    print()
    print(f"SURPRISE_WINDOW={SURPRISE_WINDOW}  CLASSIFIER_WINDOW={CLASSIFIER_WINDOW}  "
          f"[must be equal; asserted]")
    assert CLASSIFIER_WINDOW == SURPRISE_WINDOW
    print(f"EVAL_EVERY={EVAL_EVERY}  ERROR_RATE_OK_THRESH={ERROR_RATE_OK_THRESH}  "
          f"LAG1_RHO_STRUCT_THRESH={LAG1_RHO_STRUCT_THRESH}")
    print(f"ALPHA={ALPHA}  EWMA_INIT={EWMA_INIT}  (EWMA channel, verbatim exp157)")
    print(f"P_TRUE_NOISE_OPTIONS={P_TRUE_NOISE_OPTIONS}  "
          f"HALF_PERIOD_OPTIONS={HALF_PERIOD_OPTIONS}")
    print(f"DERANGEMENT_OPTIONS={DERANGEMENT_OPTIONS}")
    print(f"P_TRUE_PLACE={P_TRUE_PLACE}  N_NOISY_CELLS={N_NOISY_CELLS}  "
          f"(random placement, not checkerboard)")
    print(f"FORK_PARAM_SEED_OFFSET={FORK_PARAM_SEED_OFFSET}  "
          f"NOISE_SEED_OFFSET={NOISE_SEED_OFFSET}  "
          f"PLACE_SEED_OFFSET={PLACE_SEED_OFFSET}")
    print(f"Seeds: {seeds}  Regimes: {regimes}  N_CHUNKS={N_CHUNKS}  "
          f"CHUNK_SIZE={CHUNK_SIZE}  N_STEPS={N_CHUNKS * CHUNK_SIZE}")
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
        regimes=regimes,
        n_chunks=N_CHUNKS,
        verbose=True,
    )
    print()

    # --- Per-fork table ---
    print("Per-fork table:")
    print_fork_table(rows)
    print()

    # --- Ungated diagnostics (R-PLACE label distribution, mean rho, EWMA calibration) ---
    place_rows = [r for r in rows if r["regime"] == "R-PLACE"]
    if place_rows:
        print("Ungated diagnostics (R-PLACE — random placement):")
        print("  Named ungated prediction: random placement removes parity anti-correlation;")
        print("  R-PLACE mean_rho expected ~0 (vs -0.14..-0.19 Exp 158 checkerboard);")
        print("  majority label expected NOISE (not STRUCTURAL).")
        for r in place_rows:
            ca = f"{r['clean_acc']:.4f}" if not (isinstance(r['clean_acc'], float) and math.isnan(r['clean_acc'])) else "nan"
            na = f"{r['noisy_acc']:.4f}" if not (isinstance(r['noisy_acc'], float) and math.isnan(r['noisy_acc'])) else "nan"
            er = f"{r['ewma_calibration_r']:.4f}" if not (isinstance(r['ewma_calibration_r'], float) and math.isnan(r['ewma_calibration_r'])) else "nan"
            au = f"{r['auroc']:.4f}" if not (isinstance(r['auroc'], float) and math.isnan(r['auroc'])) else "nan"
            print(
                f"  seed={r['fork_seed']}  n_win={r['n_windows']}  "
                f"majority={r['majority_label']}  "
                f"frac_noise={r['frac_noise']:.4f}  frac_struct={r['frac_structural']:.4f}  "
                f"mean_rho={r['mean_rho']:.4f}  auroc={au}  "
                f"clean_acc={ca}  noisy_acc={na}  ewma_r={er}  "
                f"noisy_cells={r['drawn_params'].get('noisy_cells', [])[:3]}..."
            )
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
        out_rows_path = Path(__file__).parent / "outputs" / "exp159_rows.json"
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

    # P1 detail (R-PLACE channel AUROC)
    place_aurocs = [
        f"{r['auroc']:.4f}" if not (isinstance(r['auroc'], float) and math.isnan(r['auroc'])) else "nan"
        for r in place_rows
    ]
    pool_s = (
        f"{ev['pool_place_auroc']:.4f}"
        if not (isinstance(ev['pool_place_auroc'], float) and math.isnan(ev['pool_place_auroc']))
        else "nan"
    )
    print(f"P1 (meta-d' > 0 in R-PLACE via EWMA channel, randomized placement):")
    print(f"  AUROC per fork: [{', '.join(place_aurocs)}]")
    print(f"  Forks with AUROC > {P1_PER_FORK_THRESH}: "
          f"{ev['p1_forks_pass_count']}/{n} (need >={P1_FORKS_AUROC_MIN} for pass)")
    print(f"  Pooled AUROC: {pool_s} (need >={P1_POOLED_MIN} for pass)")
    print(f"  F1a: forks with AUROC > {F1_PER_FORK_THRESH}: "
          f"{ev['f1_forks_low_count']}/{n} (fires if <={F1_FORKS_MAX})")
    print(f"  F1b: pooled <= {F1_POOLED_MAX}: {'YES' if ev['f1b'] else 'no'}")
    p1_status = "PASS" if ev["p1_pass"] else ("FALSIFIER F1" if ev["f1"] else "MIXED")
    print(f"  => P1: {p1_status}")
    print()

    # P2 detail (classifier majority labels)
    ctrl_majorities = [r["majority_label"] for r in rows if r["regime"] == "R-CTRL"]
    noise_majorities = [r["majority_label"] for r in rows if r["regime"] == "R-NOISE"]
    struct_majorities = [r["majority_label"] for r in rows if r["regime"] == "R-STRUCT"]
    pnsf_s = (
        f"{ev['pooled_noise_struct_frac']:.4f}"
        if not (isinstance(ev["pooled_noise_struct_frac"], float) and math.isnan(ev["pooled_noise_struct_frac"]))
        else "nan"
    )
    print(f"P2 (noise-vs-structure separation via classifier, randomized params):")
    print(f"  R-CTRL  majority labels: {ctrl_majorities}")
    print(f"  R-CTRL  majority=OK count: {ev['p2_ctrl_ok_count']}/{n} "
          f"(need >={P2_CTRL_OK_FORKS_MIN}; F2 fires if <={F2_MAJORITY_COUNT_MAX})")
    print(f"  R-NOISE majority labels: {noise_majorities}")
    print(f"  R-NOISE majority=NOISE count: {ev['p2_noise_noise_count']}/{n} "
          f"(need >={P2_NOISE_NOISE_FORKS_MIN}; F2 fires if <={F2_MAJORITY_COUNT_MAX})")
    print(f"  R-STRUCT majority labels: {struct_majorities}")
    print(f"  R-STRUCT majority=STRUCTURAL count: {ev['p2_struct_struct_count']}/{n} "
          f"(need >={P2_STRUCT_STRUCT_FORKS_MIN}; F2 fires if <={F2_MAJORITY_COUNT_MAX})")
    print(f"  Pooled R-NOISE STRUCTURAL-fraction: {pnsf_s} "
          f"(need <={P2_POOLED_STRUCT_FRAC_MAX}; F2 fires if >{F2_POOLED_STRUCT_FRAC_HIGH})")
    print(f"  F2 sub-flags: ctrl={ev['f2_ctrl']} noise={ev['f2_noise']} "
          f"struct={ev['f2_struct']} pooled={ev['f2_pooled']}")
    print(f"  Pooled diagnostics:")
    print(f"    R-CTRL  mean_error_rate={ev['ctrl_mean_error_rate']:.4f}  "
          f"mean_rho={ev['ctrl_mean_rho']:.4f}")
    print(f"    R-NOISE mean_error_rate={ev['noise_mean_error_rate']:.4f}  "
          f"mean_rho={ev['noise_mean_rho']:.4f}")
    print(f"    R-STRUCT mean_error_rate={ev['struct_mean_error_rate']:.4f}  "
          f"mean_rho={ev['struct_mean_rho']:.4f}")
    print(f"    R-PLACE mean_error_rate={ev['place_mean_error_rate']:.4f}  "
          f"mean_rho={ev['place_mean_rho']:.4f}")
    p2_status = "PASS" if ev["p2_pass"] else ("FALSIFIER F2" if ev["f2"] else "MIXED")
    print(f"  => P2: {p2_status}")
    print()

    # --- Conjunct summary + VERDICT ---
    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    print(f"P1 (R-PLACE channel AUROC > {P1_PER_FORK_THRESH} in "
          f">={P1_FORKS_AUROC_MIN}/{n} forks AND pooled >={P1_POOLED_MIN}): {p1_status}")
    print(f"P2 (classifier majority correct per regime in >={P2_CTRL_OK_FORKS_MIN}/{n} forks "
          f"AND pooled R-NOISE STRUCTURAL <={P2_POOLED_STRUCT_FRAC_MAX}): {p2_status}")
    print()
    if ev["verdict_str"] == "POSITIVE":
        print(f"VERDICT: POSITIVE — PREREQ SATISFIED")
    elif ev["verdict_str"] == "NEGATIVE":
        print(f"VERDICT: NEGATIVE — PREREQ NOT SATISFIED")
    else:
        print(f"VERDICT: MIXED")
    print()
    print("=" * 80)

    # --- Write JSON rows ---
    out_rows_path = Path(__file__).parent / "outputs" / "exp159_rows.json"
    write_json_rows(rows, out_rows_path, ev=ev)
    print(f"JSON rows written to {out_rows_path}")

    # --- Write verdict JSON ---
    pnsf_for_reason = (
        f"{ev['pooled_noise_struct_frac']:.4f}"
        if not (isinstance(ev["pooled_noise_struct_frac"], float) and math.isnan(ev["pooled_noise_struct_frac"]))
        else "nan"
    )
    arms = {
        "P1_meta_d_prime_randomized": {
            "pass": bool(ev["p1_pass"]),
            "reason": (
                f"R-PLACE EWMA channel AUROC > {P1_PER_FORK_THRESH} in "
                f"{ev['p1_forks_pass_count']}/{n} forks (need {P1_FORKS_AUROC_MIN}); "
                f"pooled AUROC={pool_s} (need >={P1_POOLED_MIN}); "
                f"F1a forks>{F1_PER_FORK_THRESH}={ev['f1_forks_low_count']}/{n}; "
                f"F1b pooled<={F1_POOLED_MAX}={ev['f1b']}"
            ),
        },
        "P2_separation_randomized": {
            "pass": bool(ev["p2_pass"]),
            "reason": (
                f"R-CTRL majority=OK: {ev['p2_ctrl_ok_count']}/{n}; "
                f"R-NOISE majority=NOISE: {ev['p2_noise_noise_count']}/{n}; "
                f"R-STRUCT majority=STRUCTURAL: {ev['p2_struct_struct_count']}/{n}; "
                f"pooled R-NOISE STRUCTURAL-frac={pnsf_for_reason} "
                f"(need <={P2_POOLED_STRUCT_FRAC_MAX}); "
                f"F2 sub-flags: ctrl={ev['f2_ctrl']} noise={ev['f2_noise']} "
                f"struct={ev['f2_struct']} pooled={ev['f2_pooled']}"
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp159_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp159",
        arms=arms,
        verdict=ev["verdict_str"],
        halted=False,
        notes=(
            "N2 prereq re-confirmation with Exp 157 EWMA channel and Exp 158 classifier. "
            "Per-fork randomization of all regime parameters (p_true_noise, derangement, "
            "half_period, noisy_cell_placement) addresses Exp 155 verifier's determinism flag. "
            "R-PLACE uses 13 randomly-placed noisy cells (not checkerboard) to remove parity "
            "anti-correlation; named ungated prediction: R-PLACE mean_rho ~0 and majority NOISE. "
            "POSITIVE opens N3 rung 1."
        ),
    )
    print(f"Verdict JSON written to {verdict_path}")


if __name__ == "__main__":
    main()
