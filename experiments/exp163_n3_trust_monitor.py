"""Exp 163 — N3 rung 2: the third-order trust monitor. N2's diagnoses make
implicit forecasts; a monitor that scores those forecasts against realized
outcomes — internal signals only, no generator access — should lose trust
in N2 exactly where N2 is broken (R-SLOW, the Exp 162 regime) and keep it
where N2 is valid (the Exp 159 regime set).

Hypothesis: in R-SLOW the classifier's labels cycle OK -> STRUCTURAL ->
NOISE with the world's hidden phase, so the forecasts implicit in OK
("errors stay rare") and NOISE ("this error level is irreducible and
persists") are repeatedly violated at phase boundaries — while in R-CTRL /
R-NOISE / R-PLACE the same forecasts hold. Scoring those violations yields
a third-order trust signal with metacognitive sensitivity OVER N2.

N3 monitor (PROVIDED form; contents self-formed from the creature's own
stream): at each classifier evaluation (W=200, every 100 steps) with label
L and window error rate e_w, score the forecast against the NEXT 100 steps'
realized error rate e_next (internal: own predictions vs own observations):
  L = OK:    violated iff e_next >= 0.15  (the "errors stay rare" claim).
  L = NOISE: violated iff |e_next - e_w| >= 0.3  (the "this level is
             irreducible and persists" claim).
  L = STRUCTURAL: NOT scored (its forecast is weaker-typed; declared
             exclusion, logged in diagnostics).
Evaluations without a full next-100 horizon are excluded. N3-trust per
(fork, regime) = 1 - (violations / scored labels).

Design: forks of mirro only (spine never saved), FRESH seeds 66-73, 4000
steps per (fork, regime), exp155 step-loop semantics. Regimes (drawn
params logged; per-fork rng 100_000 + fork_seed as in Exp 159):
  R-SLOW:  Exp 162 form — StructCmap half_period 400, derangement drawn
           from {[1,2,0],[2,0,1]} (BROKEN regime, by Exp 162's verdict).
  R-CTRL:  standard world (valid).
  R-NOISE: NoisyCmap p_true drawn from {0.6,0.7,0.8} (valid).
  R-PLACE: random 13-of-25 noisy placement, p_true 0.55 (valid).
The EWMA channel rides along; its rolling-Brier trust variant is logged
UNGATED for comparison (channel_trust = brier(baseline) - brier(channel)
over the run, baseline = window-mean correctness).

Preconditions (instrument-grade; failure => "PRECONDITION FAILED", rows
written, no verdict): PC1 >= 25 scored labels per (fork, regime); PC2
ahat_drift < 0.05 per (fork, regime); PC3 R-SLOW pure-B-window error >= 0.5
pooled (the Exp 162 input-validity check).

Predeclared properties and falsifiers:
  P1 (trust drops where N2 is broken): per fork, trust(R-SLOW) is the
     STRICT MINIMUM among the four regimes, in >= 7/8 forks.
     FALSIFIER F1 (the card's "no metacognitive sensitivity over N2"):
     pooled trust(R-SLOW) >= min over valid regimes of pooled trust, OR
     strict-minimum in < 3 forks (the card's own >= 3 floor).
  P2 (magnitude): pooled gap = (min valid-regime pooled trust) - (pooled
     trust(R-SLOW)) >= 0.15. FALSIFIER F2: pooled gap <= 0.02.
  VERDICT: POSITIVE (RUNG 2 PASSES — N3 detects N2's miscalibration) iff
  P1 and P2 both pass. NEGATIVE iff F1 or F2 fires. Otherwise MIXED.
  "Not a falsifier" never counts toward POSITIVE.

Ungated diagnostics: per-(fork, regime) violation rates split by label
type (OK / NOISE) and scored-label counts; STRUCTURAL label counts
(unscored); rolling trust trace for fork 66 in R-SLOW (window of last 10
scored labels) for the record; the channel-Brier trust variant per cell;
drawn parameters per fork.
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

SEEDS = list(range(66, 74))   # fresh seeds — exp155-162 used 0-65

N_STEPS = 4000
N_CHUNKS = 40
CHUNK_SIZE = N_STEPS // N_CHUNKS    # 100

# Classifier window / evaluation parameters (verbatim from exp158/159/162)
CLASSIFIER_WINDOW = 200       # W = 200 (same as SURPRISE_WINDOW)
EVAL_EVERY = 100              # evaluate once window is full, every 100 steps
ERROR_RATE_OK_THRESH = 0.05   # error_rate < 0.05 => OK
LAG1_RHO_STRUCT_THRESH = 0.3  # lag1_rho > 0.3 => STRUCTURAL (else NOISE)

# EWMA channel parameters (verbatim from exp157/159)
ALPHA = 0.05       # EWMA learning rate
EWMA_INIT = 0.5    # initial EWMA value per cell

# N3 monitor forecast-scoring thresholds (from docstring)
OK_VIOLATION_THRESH = 0.15     # L=OK violated iff e_next >= 0.15
NOISE_VIOLATION_DELTA = 0.30   # L=NOISE violated iff |e_next - e_w| >= 0.30

# Rolling trust window for the trace diagnostic (fork 66, R-SLOW)
ROLLING_TRUST_WINDOW = 10

# R-SLOW parameters (verbatim from exp162)
HALF_PERIOD_RSLOW = 400
RSLOW_DERANG_SEED_OFFSET = 120_000   # rng 120_000 + fork_seed (Exp 162 convention)

# Per-fork randomization rng seed offset (Exp 159 convention)
FORK_PARAM_SEED_OFFSET = 100_000

# Regime seed offsets (matching exp159)
NOISE_SEED_OFFSET = 70_000        # R-NOISE NoisyCmap rng: seed 70_000+fork_seed
PLACE_SEED_OFFSET = 90_000        # R-PLACE cmap rng: seed 90_000+fork_seed

# R-NOISE: p_true drawn from this set per fork
P_TRUE_NOISE_OPTIONS = [0.6, 0.7, 0.8]

# R-SLOW derangement options
DERANGEMENT_OPTIONS = [[1, 2, 0], [2, 0, 1]]

# R-PLACE parameters
P_TRUE_PLACE = 0.55
N_NOISY_CELLS = 13    # exactly 13 of 25 cells drawn as noisy per fork

# Precondition thresholds
PRECONDITION_MIN_SCORED = 25          # PC1: >= 25 scored labels per (fork, regime)
PRECONDITION_AHAT_DRIFT = 0.05        # PC2: ahat_drift < 0.05 per (fork, regime)
PRECONDITION_PURE_B_ERR_MIN = 0.5     # PC3: R-SLOW pooled pure-B error >= 0.5

# P1 thresholds
P1_FORKS_STRICT_MIN = 7              # >= 7/8 forks where trust(R-SLOW) is strict minimum
F1_STRICT_MIN_FLOOR = 3              # F1b: strict-minimum in < 3 forks fires F1

# P2 thresholds
P2_GAP_MIN = 0.15                    # pooled gap >= 0.15
F2_GAP_MAX = 0.02                    # F2: pooled gap <= 0.02

ALL_REGIMES = ("R-SLOW", "R-CTRL", "R-NOISE", "R-PLACE")
SMOKE_REGIMES = ("R-SLOW", "R-NOISE")


# ---------------------------------------------------------------------------
# NoisyCmap — verbatim from exp159
# ---------------------------------------------------------------------------

class NoisyCmap:
    """Color map wrapper with irreducible observation noise."""

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
# StructCmap — verbatim from exp159/exp162
# ---------------------------------------------------------------------------

class StructCmap:
    """Hidden-context cmap: alternates every half_period steps."""

    def __init__(self, base_cmap, n_colors, derangement, half_period=100):
        self.base = list(base_cmap)
        self.n_colors = int(n_colors)
        self.derangement = list(derangement)
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
# RandomPlaceNoiseCmap — verbatim from exp159
# ---------------------------------------------------------------------------

class RandomPlaceNoiseCmap:
    """Color map wrapper with per-fork randomly-placed noisy cells."""

    def __init__(self, base_cmap, n_colors, noisy_cells: list[int], p_true, seed):
        self.base = list(base_cmap)
        self.n_colors = int(n_colors)
        self.noisy_cells = set(noisy_cells)
        self.p_true = float(p_true)
        self.rng = np.random.default_rng(seed)

    def is_noisy(self, cell: int) -> bool:
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
# lag1_rho — verbatim from exp159/exp162
# ---------------------------------------------------------------------------

def lag1_rho(seq: np.ndarray) -> float:
    """Lag-1 Pearson autocorrelation of a 1-D array."""
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
# Classifier window evaluation (verbatim from exp159/162)
# ---------------------------------------------------------------------------

def classify_window(window: np.ndarray) -> tuple[str, float, float]:
    """Classify a correctness window. Returns (label, error_rate, rho)."""
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
# window_purity — for PC3 pure-B accounting (from exp162)
# ---------------------------------------------------------------------------

def window_purity(ctx_window: np.ndarray) -> str:
    if np.all(ctx_window == 0):
        return "pure-A"
    elif np.all(ctx_window == 1):
        return "pure-B"
    else:
        return "straddle"


# ---------------------------------------------------------------------------
# Draw per-fork regime parameters
# ---------------------------------------------------------------------------

def draw_fork_params(fork_seed: int) -> dict:
    """Draw per-fork randomization parameters.

    Uses rng = default_rng(100_000 + fork_seed) as in Exp 159.
    For R-SLOW derangement uses rng = default_rng(120_000 + fork_seed) as in Exp 162.
    """
    rng = np.random.default_rng(FORK_PARAM_SEED_OFFSET + fork_seed)

    # R-NOISE: p_true drawn from {0.6, 0.7, 0.8}
    p_true_noise = float(
        P_TRUE_NOISE_OPTIONS[int(rng.integers(0, len(P_TRUE_NOISE_OPTIONS)))]
    )

    # R-PLACE: 13 noisy cells drawn without replacement from range(25)
    noisy_cells_arr = rng.choice(25, size=N_NOISY_CELLS, replace=False)
    noisy_cells = sorted(int(x) for x in noisy_cells_arr)

    # R-SLOW derangement: use Exp 162 convention (rng 120_000 + fork_seed)
    rslow_rng = np.random.default_rng(RSLOW_DERANG_SEED_OFFSET + fork_seed)
    derangement = list(
        DERANGEMENT_OPTIONS[int(rslow_rng.integers(0, len(DERANGEMENT_OPTIONS)))]
    )

    return {
        "p_true_noise": p_true_noise,
        "derangement": derangement,
        "half_period_rslow": HALF_PERIOD_RSLOW,
        "noisy_cells": noisy_cells,
    }


# ---------------------------------------------------------------------------
# Run one fork in one regime
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

    Step loop replicates exp155/exp158/exp159/exp162 semantics EXACTLY:
      - A_hat pre-step
      - pred_probs = A_hat @ qs
      - o_hat = argmax(pred_probs)
      - bel_cell = argmax(qs) pre-update
      - conf_channel = ewma[bel_cell] (EWMA channel, pre-update)
      - set current_step for StructCmap regimes
      - obs from cmap_obj
      - correct_t = (o_hat == obs)
      - append to classifier deque(200); evaluate at (t+1)%100==0 when full
      - surprise = -ln(A_hat[obs,:] @ qs) pre-update
      - Bayes update
      - pA Dirichlet count update
      - value accumulation (Exp 26 mechanism)
      - chunk rng action (chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF)
      - move
      - advance qs through B
      - age += 1
      AFTER correctness known:
        ewma[bel_cell] = (1-ALPHA)*ewma[bel_cell] + ALPHA*correct_t

    N3 monitor scored post-hoc over the recorded correctness array.
    """
    assert SURPRISE_WINDOW == 200, (
        f"SURPRISE_WINDOW from creature.py == {SURPRISE_WINDOW}, expected 200"
    )
    assert CLASSIFIER_WINDOW == SURPRISE_WINDOW, (
        f"CLASSIFIER_WINDOW={CLASSIFIER_WINDOW} must equal SURPRISE_WINDOW={SURPRISE_WINDOW}"
    )

    fork_name = f"exp163_s{fork_seed}_{regime}"
    c = mirro.fork(fork_name)

    n_cells = c.world.n_cells
    n_actions = 4

    # --- Build regime-specific cmap ---
    if regime == "R-CTRL":
        cmap_obj = base_cmap   # plain list
        drawn_params: dict = {}
    elif regime == "R-NOISE":
        p_true_noise = fork_params["p_true_noise"]
        noise_seed = NOISE_SEED_OFFSET + fork_seed
        cmap_obj = NoisyCmap(base_cmap, n_colors, p_true_noise, seed=noise_seed)
        drawn_params = {"p_true": p_true_noise}
    elif regime == "R-SLOW":
        derangement = fork_params["derangement"]
        half_period = fork_params["half_period_rslow"]
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

    # --- Per-cell EWMA state ---
    ewma = np.full(n_cells, EWMA_INIT, dtype=np.float64)

    # --- Per-step storage ---
    correct_arr = np.empty(n_total, dtype=np.int32)
    conf_arr = np.empty(n_total, dtype=np.float64)   # EWMA conf per step

    # For PC3 (R-SLOW pure-B accounting)
    context_arr: np.ndarray | None = None
    if regime == "R-SLOW":
        context_arr = np.empty(n_total, dtype=np.int32)

    # --- Classifier state: rolling correctness deque(200) ---
    correct_window = _deque(maxlen=CLASSIFIER_WINDOW)

    # --- Per-evaluation records (for N3 monitor post-hoc scoring) ---
    eval_records: list[dict] = []   # {"t": int, "label": str, "e_w": float}

    # --- Step loop ---
    global_step = 0
    eps = 1e-300

    for chunk_idx in range(n_chunks):
        chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF
        rng = np.random.default_rng(chunk_seed)

        for _step_in_chunk in range(chunk_size):
            t = global_step

            # Set step counter for StructCmap before reading cmap
            if regime == "R-SLOW":
                cmap_obj.current_step = t  # type: ignore[union-attr]

            # Ground-truth context for R-SLOW
            if context_arr is not None:
                context_arr[t] = int((t // HALF_PERIOD_RSLOW) % 2)

            # --- A_hat pre-step ---
            A_hat = c._A_hat()

            # --- Predicted observation ---
            pred_probs = A_hat @ c.qs
            o_hat = int(np.argmax(pred_probs))

            # --- bel_cell and EWMA channel confidence (pre-update) ---
            bel_cell = int(np.argmax(c.qs))
            conf_channel = float(ewma[bel_cell])

            # --- Observe ---
            true_pos_t = c.true_pos
            obs = int(cmap_obj[true_pos_t])

            # --- Correctness ---
            correct_t = 1 if (o_hat == obs) else 0

            # --- Store per-step records ---
            correct_arr[t] = correct_t
            conf_arr[t] = conf_channel

            # --- Append to classifier rolling window ---
            correct_window.append(correct_t)

            # --- Evaluate classifier at every EVAL_EVERY steps once window full ---
            if (t + 1) % EVAL_EVERY == 0 and len(correct_window) == CLASSIFIER_WINDOW:
                win_arr = np.array(correct_window, dtype=np.float64)
                label, err_rate, rho = classify_window(win_arr)
                eval_records.append({"t": t, "label": label, "e_w": err_rate})

            # --- Surprise (for trajectory consistency) ---
            p_o = float(A_hat[obs, :] @ c.qs)
            _ = -math.log(p_o + eps)

            # --- Belief update ---
            likelihood = A_hat[obs, :]
            qs_updated = likelihood * c.qs
            denom = qs_updated.sum()
            if denom > 0:
                qs_updated = qs_updated / denom
            else:
                qs_updated = np.ones(n_cells) / n_cells

            # --- Dirichlet count update ---
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

            # --- EWMA update AFTER correctness known ---
            ewma[bel_cell] = (1.0 - ALPHA) * ewma[bel_cell] + ALPHA * correct_t

            global_step += 1

    # A_hat at end for drift check
    A_hat_end = c._A_hat()
    ahat_drift = float(np.abs(A_hat_end - A_hat_start).max())

    # -----------------------------------------------------------------------
    # N3 monitor: post-hoc scoring of eval records
    # -----------------------------------------------------------------------
    # For each evaluation at step t (with a full window ending at t), the
    # "next 100 steps" horizon is steps t+1..t+100. These are fully within
    # the run iff t <= n_total - 101, i.e. t <= n_total - 101.
    # n_total = 4000; eval steps are at t = 199, 299, ..., 3999.
    # t <= 3899 iff t+100 <= 3999 = n_total - 1. (Spec says t <= 3899.)
    HORIZON_STEPS = 100
    MAX_SCORABLE_T = n_total - 1 - HORIZON_STEPS   # = 3899

    n_scored = 0
    n_violations = 0
    n_ok_labels = 0
    n_noise_labels = 0
    n_struct_labels = 0
    n_ok_violations = 0
    n_ok_scored = 0
    n_noise_violations = 0
    n_noise_scored = 0

    # Rolling trust trace: list of (scored_label_index, rolling_trust)
    # Tracks last ROLLING_TRUST_WINDOW scored outcomes (1=not violated, 0=violated)
    trust_trace: list[tuple[int, float]] = []
    rolling_outcomes: _deque[int] = _deque(maxlen=ROLLING_TRUST_WINDOW)
    scored_label_idx = 0

    for rec in eval_records:
        t = rec["t"]
        label = rec["label"]
        e_w = rec["e_w"]

        if label == "STRUCTURAL":
            n_struct_labels += 1
            continue

        if t > MAX_SCORABLE_T:
            # No full next-100 horizon; excluded from scoring
            continue

        # Compute e_next = mean error of steps t+1..t+100
        next_slice = correct_arr[t + 1: t + 1 + HORIZON_STEPS]
        e_next = 1.0 - float(next_slice.mean())

        if label == "OK":
            n_ok_labels += 1
            violated = int(e_next >= OK_VIOLATION_THRESH)
            n_ok_scored += 1
            n_ok_violations += violated
        else:  # NOISE
            n_noise_labels += 1
            violated = int(abs(e_next - e_w) >= NOISE_VIOLATION_DELTA)
            n_noise_scored += 1
            n_noise_violations += violated

        n_scored += 1
        n_violations += violated

        # Rolling trust trace
        rolling_outcomes.append(1 - violated)
        scored_label_idx += 1
        if len(rolling_outcomes) > 0:
            rolling_trust = float(sum(rolling_outcomes) / len(rolling_outcomes))
            trust_trace.append((scored_label_idx, rolling_trust))

    # N3 trust = 1 - (violations / scored)
    if n_scored > 0:
        trust = 1.0 - n_violations / n_scored
    else:
        trust = float("nan")

    viol_rate_ok = (n_ok_violations / n_ok_scored) if n_ok_scored > 0 else float("nan")
    viol_rate_noise = (n_noise_violations / n_noise_scored) if n_noise_scored > 0 else float("nan")

    # -----------------------------------------------------------------------
    # Channel-Brier trust (ungated): brier_channel - brier_baseline
    # brier_channel = mean((conf_t - correct_t)^2)
    # brier_baseline = mean((mean_correct - correct_t)^2)
    # channel_trust = brier_baseline - brier_channel
    # -----------------------------------------------------------------------
    mean_correct = float(correct_arr.mean())
    brier_channel = float(np.mean((conf_arr - correct_arr.astype(np.float64)) ** 2))
    brier_baseline = float(np.mean((mean_correct - correct_arr.astype(np.float64)) ** 2))
    channel_trust = brier_baseline - brier_channel

    # -----------------------------------------------------------------------
    # PC3 pure-B window error (R-SLOW only)
    # -----------------------------------------------------------------------
    pure_b_window_errors: list[float] = []
    if regime == "R-SLOW" and context_arr is not None:
        for rec in eval_records:
            t_eval = rec["t"]
            ctx_win = context_arr[t_eval - CLASSIFIER_WINDOW + 1: t_eval + 1]
            if np.all(ctx_win == 1):  # pure-B
                pure_b_window_errors.append(rec["e_w"])

    pooled_pure_b_err = (
        float(np.mean(pure_b_window_errors))
        if pure_b_window_errors else float("nan")
    )

    return {
        "fork_seed": fork_seed,
        "regime": regime,
        "n_total": n_total,
        "ahat_drift": ahat_drift,
        "drawn_params": drawn_params,
        # N3 monitor aggregates
        "n_scored": n_scored,
        "n_violations": n_violations,
        "trust": trust,
        "n_ok_labels": n_ok_labels,
        "n_noise_labels": n_noise_labels,
        "n_struct_labels": n_struct_labels,
        "n_ok_scored": n_ok_scored,
        "n_ok_violations": n_ok_violations,
        "n_noise_scored": n_noise_scored,
        "n_noise_violations": n_noise_violations,
        "viol_rate_ok": viol_rate_ok,
        "viol_rate_noise": viol_rate_noise,
        # Channel-Brier trust (ungated)
        "channel_trust": channel_trust,
        "brier_channel": brier_channel,
        "brier_baseline": brier_baseline,
        # PC3 (R-SLOW only)
        "pooled_pure_b_err": pooled_pure_b_err,
        # Rolling trust trace (for smoke diagnostic printing)
        "_trust_trace": trust_trace,
        # Eval records (for diagnostics)
        "_eval_records": eval_records,
    }


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
        fork_params = draw_fork_params(seed)
        if verbose:
            print(
                f"  seed={seed} params: "
                f"p_true_noise={fork_params['p_true_noise']}  "
                f"derangement={fork_params['derangement']}  "
                f"half_period_rslow={fork_params['half_period_rslow']}  "
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
# Print per-fork trust table: rows = fork, columns = regime
# ---------------------------------------------------------------------------

def print_trust_table(rows: list[dict], regimes: tuple[str, ...]) -> None:
    """Print per-fork trust table with regime columns and strict-min marker."""
    def _fmt(v: float, w: int = 7) -> str:
        if isinstance(v, float) and math.isnan(v):
            return "nan".rjust(w)
        return f"{v:.4f}".rjust(w)

    # Build seed -> regime -> trust mapping
    trust_map: dict[int, dict[str, float]] = {}
    seeds_seen: list[int] = []
    for r in rows:
        s = r["fork_seed"]
        if s not in trust_map:
            trust_map[s] = {}
            seeds_seen.append(s)
        trust_map[s][r["regime"]] = r["trust"]

    col_width = 9
    regime_hdr = "  ".join(reg.rjust(col_width) for reg in regimes)
    print(f"  {'seed':>5}  {regime_hdr}  {'strict_min?':>12}")
    print("  " + "-" * (7 + 2 + (col_width + 2) * len(regimes) + 14))

    for s in seeds_seen:
        trust_values = trust_map[s]
        all_regimes_present = all(reg in trust_values for reg in regimes)
        valid_regimes = [reg for reg in regimes if reg != "R-SLOW"]

        # Mark if trust(R-SLOW) is strictly minimum across all regimes for this fork
        strict_min = False
        if all_regimes_present and "R-SLOW" in trust_values:
            t_slow = trust_values["R-SLOW"]
            if not math.isnan(t_slow):
                strict_min = all(
                    (math.isnan(trust_values.get(reg, float("nan"))) or
                     t_slow < trust_values.get(reg, float("nan")))
                    for reg in valid_regimes
                    if reg in trust_values
                )

        cells = "  ".join(_fmt(trust_values.get(reg, float("nan")), col_width) for reg in regimes)
        marker = "YES" if strict_min else "no"
        print(f"  {s:>5}  {cells}  {marker:>12}")
    print()


# ---------------------------------------------------------------------------
# Precondition check
# ---------------------------------------------------------------------------

def check_preconditions(rows: list[dict]) -> tuple[bool, list[str]]:
    """Check PC1, PC2, PC3. Returns (all_pass, list_of_failures)."""
    failures: list[str] = []

    for r in rows:
        seed = r["fork_seed"]
        regime = r["regime"]

        # PC1: >= 25 scored labels per (fork, regime)
        if r["n_scored"] < PRECONDITION_MIN_SCORED:
            failures.append(
                f"PC1 FAIL: seed={seed} {regime} n_scored={r['n_scored']} "
                f"< {PRECONDITION_MIN_SCORED}"
            )

        # PC2: ahat_drift < 0.05
        if r["ahat_drift"] >= PRECONDITION_AHAT_DRIFT:
            failures.append(
                f"PC2 FAIL: seed={seed} {regime} ahat_drift={r['ahat_drift']:.4f} "
                f">= {PRECONDITION_AHAT_DRIFT}"
            )

    # PC3: R-SLOW pooled pure-B window error >= 0.5
    slow_rows = [r for r in rows if r["regime"] == "R-SLOW"]
    pure_b_errs = [
        r["pooled_pure_b_err"] for r in slow_rows
        if not (isinstance(r["pooled_pure_b_err"], float) and math.isnan(r["pooled_pure_b_err"]))
    ]
    if slow_rows:
        if pure_b_errs:
            pooled = float(np.mean(pure_b_errs))
        else:
            pooled = float("nan")
        if math.isnan(pooled) or pooled < PRECONDITION_PURE_B_ERR_MIN:
            failures.append(
                f"PC3 FAIL: R-SLOW pooled pure-B window error={pooled:.4f} "
                f"< {PRECONDITION_PURE_B_ERR_MIN} (derangement not biting / no pure-B windows)"
            )

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Evaluate predeclared outcome map (P1/F1, P2/F2)
# ---------------------------------------------------------------------------

def evaluate(rows: list[dict]) -> dict:
    """Evaluate P1/F1 and P2/F2; return evaluation dict."""

    # --- Pooled trust per regime ---
    regime_groups: dict[str, list[dict]] = {}
    for r in rows:
        regime_groups.setdefault(r["regime"], []).append(r)

    def _pooled_trust(regime_rows: list[dict]) -> float:
        total_scored = sum(r["n_scored"] for r in regime_rows)
        total_violations = sum(r["n_violations"] for r in regime_rows)
        if total_scored == 0:
            return float("nan")
        return 1.0 - total_violations / total_scored

    pooled_trust: dict[str, float] = {}
    for regime in ALL_REGIMES:
        if regime in regime_groups:
            pooled_trust[regime] = _pooled_trust(regime_groups[regime])
        else:
            pooled_trust[regime] = float("nan")

    valid_regimes = [reg for reg in ALL_REGIMES if reg != "R-SLOW"]
    valid_pooled = [pooled_trust[reg] for reg in valid_regimes if not math.isnan(pooled_trust[reg])]
    min_valid_pooled = min(valid_pooled) if valid_pooled else float("nan")

    # --- P1 / F1: trust(R-SLOW) strict minimum per fork ---
    # Build per-fork trust across regimes
    trust_by_seed: dict[int, dict[str, float]] = {}
    for r in rows:
        trust_by_seed.setdefault(r["fork_seed"], {})[r["regime"]] = r["trust"]

    strict_min_count = 0
    for seed, t_map in trust_by_seed.items():
        t_slow = t_map.get("R-SLOW", float("nan"))
        if math.isnan(t_slow):
            continue
        valid_vals = [
            t_map[reg] for reg in valid_regimes
            if reg in t_map and not math.isnan(t_map[reg])
        ]
        if valid_vals and all(t_slow < v for v in valid_vals):
            strict_min_count += 1

    n_forks = len(trust_by_seed)
    p1_pass = strict_min_count >= P1_FORKS_STRICT_MIN

    # F1a: pooled trust(R-SLOW) >= min over valid regimes of pooled trust
    t_slow_pooled = pooled_trust.get("R-SLOW", float("nan"))
    f1a = (
        not math.isnan(t_slow_pooled) and not math.isnan(min_valid_pooled)
        and t_slow_pooled >= min_valid_pooled
    )
    # F1b: strict-minimum in < 3 forks
    f1b = strict_min_count < F1_STRICT_MIN_FLOOR
    f1 = f1a or f1b

    # --- P2 / F2: pooled gap >= 0.15 ---
    if not math.isnan(min_valid_pooled) and not math.isnan(t_slow_pooled):
        pooled_gap = min_valid_pooled - t_slow_pooled
    else:
        pooled_gap = float("nan")

    p2_pass = not math.isnan(pooled_gap) and pooled_gap >= P2_GAP_MIN
    f2 = not math.isnan(pooled_gap) and pooled_gap <= F2_GAP_MAX

    # --- Verdict ---
    positive = p1_pass and p2_pass
    negative = f1 or f2
    if positive:
        verdict_str = "POSITIVE"
    elif negative:
        verdict_str = "NEGATIVE"
    else:
        verdict_str = "MIXED"

    # Per-regime ungated diagnostics
    def _pooled_mean_field(regime_rows: list[dict], field: str) -> float:
        vals = [
            r[field] for r in regime_rows
            if not (isinstance(r[field], float) and math.isnan(r[field]))
        ]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "pooled_trust": pooled_trust,
        "min_valid_pooled": min_valid_pooled,
        "pooled_gap": pooled_gap,
        "strict_min_count": strict_min_count,
        "n_forks": n_forks,
        # P1
        "p1_pass": p1_pass,
        "f1a": f1a,
        "f1b": f1b,
        "f1": f1,
        # P2
        "p2_pass": p2_pass,
        "f2": f2,
        # Verdict
        "verdict_str": verdict_str,
        # Per-regime channel trust
        "channel_trust_by_regime": {
            regime: _pooled_mean_field(regime_groups.get(regime, []), "channel_trust")
            for regime in ALL_REGIMES
        },
    }


# ---------------------------------------------------------------------------
# Write JSONL rows + summary
# ---------------------------------------------------------------------------

def _nan_to_none(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def write_json_rows(rows: list[dict], path: Path, ev: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            row: dict = {
                "exp": 163,
                "fork_seed": r["fork_seed"],
                "regime": r["regime"],
                "n_total": r["n_total"],
                "ahat_drift": _nan_to_none(r["ahat_drift"]),
                "drawn_params": r["drawn_params"],
                # N3 monitor
                "n_scored": r["n_scored"],
                "n_violations": r["n_violations"],
                "trust": _nan_to_none(r["trust"]),
                "n_ok_labels": r["n_ok_labels"],
                "n_noise_labels": r["n_noise_labels"],
                "n_struct_labels": r["n_struct_labels"],
                "n_ok_scored": r["n_ok_scored"],
                "n_ok_violations": r["n_ok_violations"],
                "n_noise_scored": r["n_noise_scored"],
                "n_noise_violations": r["n_noise_violations"],
                "viol_rate_ok": _nan_to_none(r["viol_rate_ok"]),
                "viol_rate_noise": _nan_to_none(r["viol_rate_noise"]),
                # Channel Brier
                "channel_trust": r["channel_trust"],
                "brier_channel": r["brier_channel"],
                "brier_baseline": r["brier_baseline"],
                # PC3
                "pooled_pure_b_err": _nan_to_none(r["pooled_pure_b_err"]),
            }
            fh.write(json.dumps(row) + "\n")

        if ev is not None:
            summary: dict = {
                "exp": 163,
                "row_type": "summary",
                "pooled_trust_rslow": _nan_to_none(ev["pooled_trust"].get("R-SLOW")),
                "pooled_trust_rctrl": _nan_to_none(ev["pooled_trust"].get("R-CTRL")),
                "pooled_trust_rnoise": _nan_to_none(ev["pooled_trust"].get("R-NOISE")),
                "pooled_trust_rplace": _nan_to_none(ev["pooled_trust"].get("R-PLACE")),
                "min_valid_pooled": _nan_to_none(ev["min_valid_pooled"]),
                "pooled_gap": _nan_to_none(ev["pooled_gap"]),
                "strict_min_count": ev["strict_min_count"],
                "p1_pass": ev["p1_pass"],
                "f1a": ev["f1a"],
                "f1b": ev["f1b"],
                "f1": ev["f1"],
                "p2_pass": ev["p2_pass"],
                "f2": ev["f2"],
                "verdict": ev["verdict_str"],
            }
            fh.write(json.dumps(summary) + "\n")


# ---------------------------------------------------------------------------
# Print rolling trust trace for fork 66 in R-SLOW
# ---------------------------------------------------------------------------

def print_rolling_trust_trace(rows: list[dict]) -> None:
    """Print rolling trust trace (window=10 scored labels) for fork 66, R-SLOW."""
    target = [r for r in rows if r["fork_seed"] == 66 and r["regime"] == "R-SLOW"]
    if not target:
        print("  (fork 66 R-SLOW not found in rows)")
        return
    r = target[0]
    trace = r["_trust_trace"]
    print(f"Rolling trust trace (window={ROLLING_TRUST_WINDOW}) for fork 66, R-SLOW:")
    print(f"  {'scored_label_idx':>16}  {'rolling_trust':>13}")
    for idx, rt in trace:
        print(f"  {idx:>16}  {rt:>13.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 163 — N3 rung 2: third-order trust monitor"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Smoke run: seed=[66] only, regimes R-SLOW and R-NOISE only, "
            "full 4000 steps, prints per-cell numbers + rolling trust trace. "
            "No verdict written."
        ),
    )
    args = parser.parse_args()

    smoke = args.smoke
    seeds = [66] if smoke else SEEDS
    regimes = SMOKE_REGIMES if smoke else ALL_REGIMES

    print("=" * 80)
    print("Exp 163 — N3 rung 2: the third-order trust monitor")
    print("          N3 monitor scores N2's forecast violations; trust drops")
    print("          in R-SLOW (broken regime) but not in valid regimes.")
    print("=" * 80)
    print()
    print(f"SURPRISE_WINDOW={SURPRISE_WINDOW}  CLASSIFIER_WINDOW={CLASSIFIER_WINDOW}  "
          f"[must be equal; asserted]")
    assert CLASSIFIER_WINDOW == SURPRISE_WINDOW
    print(f"EVAL_EVERY={EVAL_EVERY}  ERROR_RATE_OK_THRESH={ERROR_RATE_OK_THRESH}  "
          f"LAG1_RHO_STRUCT_THRESH={LAG1_RHO_STRUCT_THRESH}")
    print(f"N3 monitor: OK_VIOLATION_THRESH={OK_VIOLATION_THRESH}  "
          f"NOISE_VIOLATION_DELTA={NOISE_VIOLATION_DELTA}")
    print(f"ALPHA={ALPHA}  EWMA_INIT={EWMA_INIT}  "
          f"(EWMA channel, ungated Brier trust)")
    print(f"HALF_PERIOD_RSLOW={HALF_PERIOD_RSLOW}  "
          f"RSLOW_DERANG_SEED_OFFSET={RSLOW_DERANG_SEED_OFFSET}  "
          f"FORK_PARAM_SEED_OFFSET={FORK_PARAM_SEED_OFFSET}")
    print(f"P_TRUE_NOISE_OPTIONS={P_TRUE_NOISE_OPTIONS}  "
          f"P_TRUE_PLACE={P_TRUE_PLACE}  N_NOISY_CELLS={N_NOISY_CELLS}")
    print(f"DERANGEMENT_OPTIONS={DERANGEMENT_OPTIONS}")
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

    # --- Per-fork trust table ---
    print("Per-fork trust table (rows=fork, columns=regime, strict_min? = trust(R-SLOW) < all valid):")
    print_trust_table(rows, regimes)

    # --- Per-cell numbers (smoke: detailed per-(fork, regime) rows) ---
    print("Per-(fork, regime) N3 monitor numbers:")
    hdr = (
        f"{'seed':>5}  {'regime':>8}  "
        f"{'n_scored':>8}  {'n_viol':>7}  {'trust':>7}  "
        f"{'n_ok_sc':>7}  {'ok_viol':>7}  {'vr_ok':>7}  "
        f"{'n_no_sc':>7}  {'no_viol':>7}  {'vr_no':>7}  "
        f"{'n_struct':>8}  {'ch_trust':>8}  {'drift':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def _f(v: float, w: int = 7) -> str:
            if isinstance(v, float) and math.isnan(v):
                return "nan".rjust(w)
            return f"{v:.4f}".rjust(w)
        print(
            f"{r['fork_seed']:>5}  {r['regime']:>8}  "
            f"{r['n_scored']:>8}  {r['n_violations']:>7}  {_f(r['trust'], 7)}  "
            f"{r['n_ok_scored']:>7}  {r['n_ok_violations']:>7}  {_f(r['viol_rate_ok'], 7)}  "
            f"{r['n_noise_scored']:>7}  {r['n_noise_violations']:>7}  {_f(r['viol_rate_noise'], 7)}  "
            f"{r['n_struct_labels']:>8}  {_f(r['channel_trust'], 8)}  {r['ahat_drift']:>7.4f}"
        )
    print()

    # --- Rolling trust trace for fork 66 in R-SLOW (always printed) ---
    fork66_slow = [r for r in rows if r["fork_seed"] == 66 and r["regime"] == "R-SLOW"]
    if fork66_slow:
        print_rolling_trust_trace(rows)

    # --- Ungated diagnostics: drawn parameters and channel Brier ---
    print("Ungated diagnostics — drawn parameters and channel Brier trust:")
    for r in rows:
        dp = r["drawn_params"]
        param_str = (
            f"p_true={dp.get('p_true', '-')}"
            if r["regime"] == "R-NOISE" else
            f"derang={dp.get('derangement', '-')}  hp={dp.get('half_period', '-')}"
            if r["regime"] == "R-SLOW" else
            f"noisy_cells(first3)={dp.get('noisy_cells', [])[:3]}..."
            if r["regime"] == "R-PLACE" else
            "(none)"
        )
        print(
            f"  seed={r['fork_seed']}  {r['regime']:>8}  "
            f"ch_trust={r['channel_trust']:+.4f}  "
            f"brier_ch={r['brier_channel']:.4f}  brier_base={r['brier_baseline']:.4f}  "
            f"pure_b_err={_nan_to_none(r['pooled_pure_b_err'])}  "
            f"params: {param_str}"
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
        out_rows_path = Path(__file__).parent / "outputs" / "exp163_rows.json"
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

    # Pooled trust per regime
    pt = ev["pooled_trust"]
    print("Pooled N3-trust per regime:")
    for regime in ALL_REGIMES:
        v = pt.get(regime, float("nan"))
        v_s = f"{v:.4f}" if not math.isnan(v) else "nan"
        valid_marker = " (valid)" if regime != "R-SLOW" else " (BROKEN)"
        print(f"  {regime:>8}: {v_s}{valid_marker}")
    min_valid_s = f"{ev['min_valid_pooled']:.4f}" if not math.isnan(ev["min_valid_pooled"]) else "nan"
    gap_s = f"{ev['pooled_gap']:.4f}" if not math.isnan(ev["pooled_gap"]) else "nan"
    print(f"  Min(valid-regime pooled trust) = {min_valid_s}")
    print(f"  Pooled gap = min_valid - trust(R-SLOW) = {gap_s}")
    print()

    print(f"P1 (trust drops where N2 is broken — strict minimum in >= {P1_FORKS_STRICT_MIN}/{n} forks):")
    print(f"  Forks where trust(R-SLOW) is strict minimum: {ev['strict_min_count']}/{n}")
    print(f"  (need >={P1_FORKS_STRICT_MIN} for P1 pass)")
    print(f"  FALSIFIER F1a: pooled trust(R-SLOW) >= min valid pooled: "
          f"{'YES — F1a FIRES' if ev['f1a'] else 'no'}")
    print(f"  FALSIFIER F1b: strict-minimum in < {F1_STRICT_MIN_FLOOR} forks: "
          f"{'YES — F1b FIRES' if ev['f1b'] else 'no'}")
    p1_status = "PASS" if ev["p1_pass"] else ("FALSIFIER F1" if ev["f1"] else "MIXED (not P1, not F1)")
    print(f"  => P1: {p1_status}")
    print()

    print(f"P2 (magnitude — pooled gap >= {P2_GAP_MIN}):")
    print(f"  Pooled gap = {gap_s} (need >={P2_GAP_MIN})")
    print(f"  FALSIFIER F2: pooled gap <= {F2_GAP_MAX}: "
          f"{'YES — F2 FIRES' if ev['f2'] else 'no'}")
    p2_status = "PASS" if ev["p2_pass"] else ("FALSIFIER F2" if ev["f2"] else "MIXED (not P2, not F2)")
    print(f"  => P2: {p2_status}")
    print()

    # --- Channel Brier trust ungated ---
    print("Ungated — Channel-Brier trust per regime (brier_baseline - brier_channel):")
    for regime in ALL_REGIMES:
        v = ev["channel_trust_by_regime"].get(regime, float("nan"))
        v_s = f"{v:+.4f}" if not math.isnan(v) else "nan"
        print(f"  {regime:>8}: {v_s}")
    print()

    # --- Per-fork trust table (conjunct display) ---
    print("Per-fork trust table:")
    print_trust_table(rows, ALL_REGIMES)

    # --- Conjunct summary + VERDICT ---
    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    print(f"P1 (trust(R-SLOW) strict minimum in >={P1_FORKS_STRICT_MIN}/{n} forks): {p1_status}")
    print(f"P2 (pooled gap >= {P2_GAP_MIN}): {p2_status}")
    print()
    if ev["verdict_str"] == "POSITIVE":
        print("VERDICT: POSITIVE")
        print("RUNG 2 PASSES — N3 detects N2's miscalibration")
    elif ev["verdict_str"] == "NEGATIVE":
        print("VERDICT: NEGATIVE")
        print("RUNG 2: candidate monitor FAILED")
    else:
        print("VERDICT: MIXED")
    print()
    print("=" * 80)

    # --- Write JSON rows ---
    out_rows_path = Path(__file__).parent / "outputs" / "exp163_rows.json"
    write_json_rows(rows, out_rows_path, ev=ev)
    print(f"JSON rows written to {out_rows_path}")

    # --- Write verdict JSON ---
    t_slow_pooled = ev["pooled_trust"].get("R-SLOW", float("nan"))
    t_slow_s = f"{t_slow_pooled:.4f}" if not math.isnan(t_slow_pooled) else "nan"
    arms = {
        "P1_trust_drops_in_broken": {
            "pass": bool(ev["p1_pass"]),
            "reason": (
                f"trust(R-SLOW) strict minimum in {ev['strict_min_count']}/{n} forks "
                f"(need >={P1_FORKS_STRICT_MIN}); "
                f"pooled trust(R-SLOW)={t_slow_s}; "
                f"F1a(pooled_rslow>=min_valid)={ev['f1a']}; "
                f"F1b(strict_min<{F1_STRICT_MIN_FLOOR})={ev['f1b']}"
            ),
        },
        "P2_gap_magnitude": {
            "pass": bool(ev["p2_pass"]),
            "reason": (
                f"pooled gap = min_valid_trust - trust(R-SLOW) = {gap_s} "
                f"(need >={P2_GAP_MIN}); "
                f"min_valid_pooled={min_valid_s}; "
                f"F2(gap<={F2_GAP_MAX})={ev['f2']}"
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp163_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp163",
        arms=arms,
        verdict=ev["verdict_str"],
        halted=False,
        notes=(
            "N3 rung 2: third-order trust monitor. N3 monitor scores N2's implicit "
            "forecasts (OK => 'errors stay rare'; NOISE => 'error level persists') "
            "against realized outcomes from the creature's own correctness stream. "
            "STRUCTURAL excluded (weaker-typed forecast). Tested over 4 regimes: "
            "R-SLOW (broken, Exp 162 window-blind-spot), R-CTRL / R-NOISE / R-PLACE "
            "(valid, Exp 159 set). POSITIVE: RUNG 2 PASSES — N3 detects N2's miscalibration. "
            "NEGATIVE: RUNG 2: candidate monitor FAILED."
        ),
    )
    print(f"Verdict JSON written to {verdict_path}")


if __name__ == "__main__":
    main()
