"""Exp 172 — N3 K-endogenization gap-seeking horizon.

RATCHET LINEAGE:
  Exp 167 introduced the N2+N3 LOCK controller.  Exp 168 found THE RATCHET LAW.
  Exp 169 (Delta A + Delta B) fixed this: MIXED — boundary transient reduced.
  Exp 170 (tier A): global P95 of completed run lengths.  FAILED — permanent
  ceiling (plateau-captured; the tier-A collapse).
  Exp 171 (tier B): recency median x2.  MIXED — K dipped below the SLOW
  half-period at fast-cycling phases, causing premature stationary locks inside
  broken segments (tier B's failure signature); broken margin halved.  Posted
  the lineage's best valid margin.

THE GAP LAW: the lock horizon must sit just above the cycling mode's
  demonstrated run lengths (1–9) and stay untouched by the plateau mode (24–84).

THIS EXPERIMENT — GAP-SEEKING HORIZON (everything else verbatim from exp171):

  GAP-SEEKING DERIVED K (replaces exp171's 2x-median rule in arm bB):
    K(t) = clamp(max(last M=12 completed run lengths) + 1, 3, 16)
    with no completed runs: K(t) = 3.
    Run completion/discard rules UNCHANGED (complete on class change;
    discard on dial change).
    Rationale: the horizon sits ONE ABOVE the longest run the stream has
    RECENTLY shown can end — by construction it can never lock on any run
    pattern the stream has recently demonstrated (fixing tier B's sub-phase
    locks inside SLOW segments), while plateaus rarely complete so they rarely
    touch it, and when one does the K=16 spike decays within one window of 12
    completions (fixing tier A's permanent capture).  The clamp [3,16] and
    M=12 remain provided policy, declared.

  DELTAS FROM EXP 171:
    - SEEDS = list(range(138, 146)) (exp171 used 130-137)
    - Output IDs: exp172 (rows -> experiments/outputs/exp172_rows.json,
      verdict -> exp172_verdict.json, experiment="exp172")
    - Gap-seeking K formula replaces 2x-median; M=12 window unchanged.

  EVERYTHING ELSE VERBATIM FROM EXP 171:
    Three same-schedule arms, mixed schedule, StructCmap/NoisyCmap/
    RandomPlaceNoisyCmap, scoring, descent driver (Delta A), no-wrap (Delta B),
    JSONL/write_verdict, preconditions, --smoke.

READ-ONLY-DIAL DECLARATION:
  One creature run per fork; per-step correctness stream recorded; agents (b8)
  and (bB) are post-hoc readouts over that SAME stream.  The creature is never
  re-run or influenced by the dial policy.

SEEDS = list(range(138, 146)) — 8 fresh forks; exp171 used 130-137.

MIXED SESSION — one per fork (verbatim from exp168/169/170/171):
  24000 steps = 6 contiguous segments of 4000 steps each.
  Per-fork rng 140_000+seed draws the segment ORDER.
  SLOW = StructCmap half_period 1000; derangement FORCED by seed parity.
  NOISE p_true from {0.6, 0.7, 0.8}.  PLACE = random 13-of-25 noisy cells.

AGENTS (post-hoc readouts; read-only-dial):
  (a)  N2-only: fixed W=200 baseline.
  (b8) N2+N3 COMPLETED controller with constant K=8 (verbatim exp169 Delta A+B).
  (bB) N2+N3 COMPLETED controller with GAP-SEEKING derived K(t).

SCORING per segment (verbatim from exp168/169/170/171):
  Eval points every 100 steps; window evaluates if enough history under the
  agent's current dial (for b8/bB) or fixed W=200 (for a).
  Ground-truth label per segment type:
    SLOW  -> STRUCTURAL   CTRL -> OK   NOISE/PLACE -> NOISE
  Segment correctness = fraction of that segment's evaluated windows bearing
  the correct label.

PRECONDITIONS (verbatim from exp168/169/170/171):
  PC1 ahat_drift < 0.30.  PC2 >= 30 eval points per segment at W=200.
  PC3 SLOW deranged-phase per-step error >= 0.9 pooled.

PREDECLARED PROPERTIES AND FALSIFIERS:
  GATES on (bB):
    P1 concentration >= 0.75 (F1 <= 0.5)
    P2 responsiveness >= 7/8 (F2 <= 4/8)
    P4 broken margin >= +0.2 (F4 <= +0.05)
    F_HARM valid deficit > 0.15

  P5 (gap-seeking claim, load-bearing): each pooled metric of bB —
    concentration, responsiveness count, valid margin, broken margin — within
    0.05 of b8's SAME-RUN pooled values OR better.
    FALSIFIER F5: any metric worse than b8's by > 0.10.

  P7 (the regulation claim):
    K(t) takes >= 2 distinct values within the session in >= 7/8 forks AND the
    fraction of evals with K pegged at a clamp bound (3 or 16) is < 0.5 pooled.
    FALSIFIER F7: pegged fraction > 0.9 (degenerate — tier-A collapse repeated).

  P3 reported-not-gating (declared, as exp170/171).
  P6 (straddle projection, FOURTH test, ungated): bB valid margin stays < -0.05.

  VERDICT:
    POSITIVE (the gap-seeking horizon endogenizes K at no cost — the human's
             stronger claim holds)
             iff P1, P2, P4, P5, P7 all pass and F_HARM silent.
    NEGATIVE iff any falsifier fires (halt for a word).
    Otherwise MIXED.

UNGATED DIAGNOSTICS:
  K(t) trajectories for bB (values, time-weighted mean, pegged fraction);
  b8-vs-bB per-segment score table; completed-run histograms; descent latencies
  for both controllers; the P3/P6 report; everything exp171 logged.
  PLUS: count of bB stationary locks occurring INSIDE SLOW segments (tier B's
  failure signature — expected ~0 now); K behavior at plateau completions (spike
  height and decay length in completions).

--smoke: seed [138] only, full 24000 steps, prints schedule, K trajectory,
  dial trajectories, per-segment scores for all three arms,
  "SMOKE ONLY — no verdict".
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

SEEDS = list(range(138, 146))   # fresh seeds — exp171 used 130-137

N_SEGMENTS = 6
SEGMENT_STEPS = 4000
N_STEPS = N_SEGMENTS * SEGMENT_STEPS   # 24000
N_CHUNKS = N_STEPS // 100              # 240  (chunk_size=100)
CHUNK_SIZE = 100

# Per-fork rng offset for the segment schedule draw
SCHEDULE_SEED_OFFSET = 140_000         # rng 140_000 + fork_seed

# Noise rng offset for the PLACE world (verbatim from exp163)
PLACE_NOISE_SEED_OFFSET = 90_000       # rng 90_000 + fork_seed

# Classifier parameters (verbatim from exp158/159/162/163/165/166/167/168/169/170)
CLASSIFIER_WINDOW_DEFAULT = 200        # starting W for agents (b8) and (bB) and (a)
EVAL_EVERY = 100                       # evaluate every 100 steps once window full
ERROR_RATE_OK_THRESH = 0.05            # error_rate < 0.05 => OK
LAG1_RHO_STRUCT_THRESH = 0.3          # lag1_rho > 0.3 => STRUCTURAL (else NOISE)

# Dial candidate set (ordered for ascent; saturation at 1600 — Delta B from exp169)
DIAL_CANDIDATES = [200, 400, 800, 1600]

# EWMA channel parameters (verbatim from exp157/159/163/165/166/167/168/169/170)
ALPHA = 0.05        # EWMA learning rate
EWMA_INIT = 0.5     # initial EWMA value per cell

# N3 forecast-scoring thresholds (verbatim from exp165/167/168/169/170)
OK_VIOLATION_THRESH = 0.15             # L=OK violated iff e_next >= 0.15
NOISE_VIOLATION_DELTA = 0.30           # L=NOISE violated iff |e_next - e_w| >= 0.30
# STRUCTURAL: not scored

ROLLING_TRUST_WINDOW = 10              # last 10 scored labels
TRUST_FIRE_THRESH = 0.85               # advance dial when trust < 0.85 (>= 2 violations)
MIN_SCORED_FOR_TRUST = 3               # < 3 scored since last dial change => NO-EVIDENCE

# CONSTANT K for agent (b8) — verbatim exp169
LOCK_K_B8 = 8

# GAP-SEEKING DERIVED K parameters for agent (bB)
# K(t) = clamp(max(last M completed run lengths) + 1, K_MIN, K_MAX)
# with no completed runs: K(t) = K_MIN
K_WINDOW_M = 12                        # sliding window: last M=12 completed runs
K_MIN = 3                              # clamp lower bound
K_MAX = 16                             # clamp upper bound
# NOTE: constant LOCK_K for bB is intentionally NOT defined; K is always derived.

# SLOW segment parameters
HALF_PERIOD_SLOW = 1000                # H=1000 (requires W > 1000 to detect)
# Derangement FORCED by seed parity (fixes Exp 167 rng accident):
#   seed even => [1,2,0]; seed odd => [2,0,1]
DERANGEMENT_OPTIONS = [[1, 2, 0], [2, 0, 1]]

# NOISE p_true options
P_TRUE_NOISE_OPTIONS = [0.6, 0.7, 0.8]

# PLACE parameters (verbatim from exp163)
P_TRUE_PLACE = 0.55
N_NOISY_CELLS = 13                     # 13 of 25 cells drawn as noisy per fork

# Precondition thresholds (verbatim from exp168/169/170)
PC1_AHAT_DRIFT_MAX = 0.30             # PC1 (24000-step horizon-scaled bound)
PC2_MIN_EVAL_POINTS = 30              # PC2: >= 30 eval points per segment at W=200
PC3_SLOW_DERANGED_ERR_MIN = 0.9       # PC3: SLOW deranged-phase error >= 0.9 pooled

# P1 thresholds (override concentration) — on bB
P1_CONC_MIN = 0.75                    # fraction of dial changes in SLOW >= 0.75
F1_CONC_MAX = 0.50                    # F1: fraction <= 0.50

# P2 thresholds (responsiveness) — on bB
P2_FORKS_MIN = 7                      # >= 7/8 forks respond on first SLOW segment
F2_FORKS_MAX = 4                      # F2: <= 4/8 forks

# P3 thresholds (reported but NOT gating — see docstring) — on bB
P3_NO_HARM_MARGIN = 0.05              # (bB) >= (a) - 0.05 (pooled): REPORTED ONLY
F_HARM_MARGIN = 0.15                  # F_HARM: (a) - (bB) > 0.15 (gates verdict)

# P4 thresholds (overrides earn their keep) — on bB
P4_IMPROVE_MIN = 0.20                 # (bB) >= (a) + 0.20 (pooled)
F4_IMPROVE_MAX = 0.05                 # F4: (bB) <= (a) + 0.05

# P5 thresholds (tier-B claim: bB within 0.05 of b8's SAME-RUN pooled values OR better)
P5_TOLERANCE = 0.05                   # within 0.05 of b8 OR better
F5_TOLERANCE = 0.10                   # worse by > 0.10 => F5 fires

# P6 threshold (straddle projection — ungated)
P6_VALID_MARGIN_THRESH = -0.05        # bB valid margin < -0.05 => projection holds

# P7 thresholds (regulation claim)
P7_FORKS_MIN = 7                      # >= 7/8 forks have >= 2 distinct K values
P7_PEGGED_MAX = 0.5                   # pegged fraction < 0.5 pooled
F7_PEGGED_DEGENERATE = 0.9            # F7: pegged fraction > 0.9


# ---------------------------------------------------------------------------
# StructCmap — verbatim from exp162/163/164/165/166/167/168/169/170
# ---------------------------------------------------------------------------

class StructCmap:
    """Hidden-context cmap: alternates every half_period steps."""

    def __init__(self, base_cmap, n_colors, derangement, half_period=1000):
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
# NoisyCmap — verbatim from exp155/163/168/169/170
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
# RandomPlaceNoiseCmap — verbatim from exp159/163/168/169/170
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
# lag1_rho — verbatim from exp162/163/164/165/166/167/168/169/170
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
# classify_window — verbatim from exp162/163/164/165/166/167/168/169/170
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
# Tier-B derived K computation
# ---------------------------------------------------------------------------

def compute_k_gap(completed_run_lengths: list[int]) -> int:
    """Compute the gap-seeking lock horizon K(t) for agent (bB).

    K(t) = clamp(max(last K_WINDOW_M completed run lengths) + 1, K_MIN, K_MAX)
    with no completed runs: K(t) = K_MIN (= 3).

    The horizon sits ONE ABOVE the longest run the stream has RECENTLY shown can
    end — by construction it can never lock on any run pattern the stream has
    recently demonstrated (fixing tier B's sub-phase locks inside SLOW segments),
    while plateaus rarely complete so they rarely touch it, and when one does
    the K=16 spike decays within one window of 12 completions (fixing tier A's
    permanent capture).  The clamp [3,16] and M=12 remain provided policy,
    declared.

    Args:
      completed_run_lengths: accumulated list of completed run lengths; only the
        last K_WINDOW_M entries are used.

    Returns:
      K(t): int in [K_MIN, K_MAX].
    """
    if not completed_run_lengths:
        return K_MIN
    window = completed_run_lengths[-K_WINDOW_M:]
    raw = max(window) + 1
    return int(max(K_MIN, min(K_MAX, raw)))


# ---------------------------------------------------------------------------
# Draw per-fork schedule parameters — verbatim from exp168/169/170
# ---------------------------------------------------------------------------

def draw_fork_schedule(fork_seed: int, n_cells: int) -> dict:
    """Draw the per-fork mixed session schedule and parameters."""
    rng = np.random.default_rng(SCHEDULE_SEED_OFFSET + fork_seed)

    derangement = DERANGEMENT_OPTIONS[fork_seed % 2]
    p_true_noise = float(P_TRUE_NOISE_OPTIONS[int(rng.integers(0, len(P_TRUE_NOISE_OPTIONS)))])

    noisy_cells_arr = rng.choice(n_cells, size=N_NOISY_CELLS, replace=False)
    noisy_cells = sorted(int(x) for x in noisy_cells_arr)

    valid_types = ["CTRL", "NOISE", "PLACE"]
    valid_perm = rng.permutation(3)
    valid_assigned = [valid_types[int(i)] for i in valid_perm]

    segment_pool = ["SLOW", "SLOW", "SLOW"] + valid_assigned
    order_perm = rng.permutation(N_SEGMENTS)
    segment_types = [segment_pool[int(i)] for i in order_perm]

    segment_params = []
    for seg_type in segment_types:
        if seg_type == "SLOW":
            segment_params.append({
                "type": "SLOW",
                "half_period": HALF_PERIOD_SLOW,
                "derangement": derangement,
            })
        elif seg_type == "CTRL":
            segment_params.append({"type": "CTRL"})
        elif seg_type == "NOISE":
            segment_params.append({"type": "NOISE", "p_true": p_true_noise})
        elif seg_type == "PLACE":
            segment_params.append({
                "type": "PLACE",
                "p_true": P_TRUE_PLACE,
                "noisy_cells": noisy_cells,
            })
        else:
            raise ValueError(f"Unknown segment type: {seg_type}")

    first_slow_idx = next(i for i, t in enumerate(segment_types) if t == "SLOW")

    return {
        "segment_types": segment_types,
        "segment_params": segment_params,
        "derangement": derangement,
        "p_true_noise": p_true_noise,
        "noisy_cells": noisy_cells,
        "first_slow_segment_idx": first_slow_idx,
    }


# ---------------------------------------------------------------------------
# Build segment-aware cmap objects — verbatim from exp168/169/170
# ---------------------------------------------------------------------------

def build_segment_cmaps(
    base_cmap: list,
    n_colors: int,
    schedule: dict,
    fork_seed: int,
) -> list:
    """Build one cmap object per segment."""
    segment_types = schedule["segment_types"]
    derangement = schedule["derangement"]
    p_true_noise = schedule["p_true_noise"]
    noisy_cells = schedule["noisy_cells"]

    cmaps = []
    noise_seg_counter = 0
    place_seg_counter = 0
    for seg_idx, seg_type in enumerate(segment_types):
        if seg_type == "SLOW":
            cmaps.append(StructCmap(base_cmap, n_colors, derangement, HALF_PERIOD_SLOW))
        elif seg_type == "CTRL":
            cmaps.append(base_cmap)
        elif seg_type == "NOISE":
            noise_seed = SCHEDULE_SEED_OFFSET + fork_seed + 1000 * (noise_seg_counter + 1)
            cmaps.append(NoisyCmap(base_cmap, n_colors, p_true_noise, seed=noise_seed))
            noise_seg_counter += 1
        elif seg_type == "PLACE":
            place_seed = PLACE_NOISE_SEED_OFFSET + fork_seed
            cmaps.append(RandomPlaceNoiseCmap(
                base_cmap, n_colors, noisy_cells=noisy_cells,
                p_true=P_TRUE_PLACE, seed=place_seed,
            ))
            place_seg_counter += 1
        else:
            raise ValueError(f"Unknown segment type: {seg_type}")
    return cmaps


# ---------------------------------------------------------------------------
# Run one fork (continuous across all 6 segments) — verbatim from exp168/169/170
# ---------------------------------------------------------------------------

def run_fork(
    mirro: Creature,
    fork_seed: int,
    base_cmap: list,
    n_colors: int,
    schedule: dict,
) -> dict:
    """Run one continuous fork across all 6 segments (24000 steps)."""
    assert SURPRISE_WINDOW == 200, (
        f"SURPRISE_WINDOW from creature.py == {SURPRISE_WINDOW}, expected 200"
    )

    fork_name = f"exp172_s{fork_seed}"
    c = mirro.fork(fork_name)

    n_cells = c.world.n_cells
    n_actions = 4

    segment_types = schedule["segment_types"]
    cmaps = build_segment_cmaps(base_cmap, n_colors, schedule, fork_seed)

    B = c.world.transition_matrix()

    A_hat_start = c._A_hat().copy()
    ewma = np.full(n_cells, EWMA_INIT, dtype=np.float64)

    correct_arr = np.empty(N_STEPS, dtype=np.int32)
    context_arr = np.full(N_STEPS, -1, dtype=np.int32)

    global_step = 0
    eps = 1e-300

    for seg_idx in range(N_SEGMENTS):
        seg_type = segment_types[seg_idx]
        cmap_obj = cmaps[seg_idx]
        seg_start = seg_idx * SEGMENT_STEPS

        if seg_type == "SLOW":
            cmap_obj.current_step = 0  # type: ignore[union-attr]

        seg_n_chunks = SEGMENT_STEPS // CHUNK_SIZE

        for chunk_in_seg in range(seg_n_chunks):
            chunk_idx = seg_idx * seg_n_chunks + chunk_in_seg
            chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF
            rng = np.random.default_rng(chunk_seed)

            for _step_in_chunk in range(CHUNK_SIZE):
                t = global_step
                step_in_seg = t - seg_start

                if seg_type == "SLOW":
                    cmap_obj.current_step = step_in_seg  # type: ignore[union-attr]
                    context_arr[t] = int((step_in_seg // HALF_PERIOD_SLOW) % 2)

                A_hat = c._A_hat()
                pred_probs = A_hat @ c.qs
                o_hat = int(np.argmax(pred_probs))
                bel_cell = int(np.argmax(c.qs))

                true_pos_t = c.true_pos
                if seg_type == "CTRL":
                    obs = int(base_cmap[true_pos_t])
                else:
                    obs = int(cmap_obj[true_pos_t])

                correct_t = 1 if (o_hat == obs) else 0
                correct_arr[t] = correct_t

                p_o = float(A_hat[obs, :] @ c.qs)
                _ = -math.log(p_o + eps)

                likelihood = A_hat[obs, :]
                qs_updated = likelihood * c.qs
                denom = qs_updated.sum()
                if denom > 0:
                    qs_updated = qs_updated / denom
                else:
                    qs_updated = np.ones(n_cells) / n_cells

                c.pA[obs, :] += qs_updated

                map_cell = int(np.argmax(qs_updated))
                predicted_obs_dist = A_hat[:, map_cell]
                h_predicted = -np.sum(
                    predicted_obs_dist * np.log(predicted_obs_dist + 1e-12)
                )
                predictability_weight = math.exp(-h_predicted)
                c.value_counts[obs] += predictability_weight

                action = int(rng.integers(0, n_actions))
                c.true_pos = c.world.move(c.true_pos, action)
                c.qs = B[:, :, action] @ qs_updated
                c.age_steps += 1

                ewma[bel_cell] = (1.0 - ALPHA) * ewma[bel_cell] + ALPHA * correct_t

                global_step += 1

    A_hat_end = c._A_hat()
    ahat_drift = float(np.abs(A_hat_end - A_hat_start).max())

    return {
        "fork_seed": fork_seed,
        "n_total": N_STEPS,
        "ahat_drift": ahat_drift,
        "correct_arr": correct_arr,
        "context_arr": context_arr,
        "segment_types": segment_types,
        "schedule": schedule,
    }


# ---------------------------------------------------------------------------
# Segment boundaries (helper)
# ---------------------------------------------------------------------------

def segment_boundaries() -> list[tuple[int, int]]:
    """Return list of (start_t, end_t_inclusive) for each segment (0-based)."""
    return [(i * SEGMENT_STEPS, (i + 1) * SEGMENT_STEPS - 1) for i in range(N_SEGMENTS)]


# ---------------------------------------------------------------------------
# Post-hoc: evaluate fixed-dial agent (a) per segment — verbatim from exp168/169/170
# ---------------------------------------------------------------------------

def eval_fixed_dial_segments(correct_arr: np.ndarray, W: int, segment_types: list[str]) -> dict:
    """Evaluate a fixed-dial agent (W constant) and return per-segment eval records."""
    n_total = len(correct_arr)

    all_eval_records: list[dict] = []
    seg_eval_records: dict[int, list[dict]] = {i: [] for i in range(N_SEGMENTS)}

    for t in range(EVAL_EVERY - 1, n_total, EVAL_EVERY):
        if t + 1 < W:
            continue
        win_arr = correct_arr[max(0, t - W + 1): t + 1].astype(np.float64)
        label, err_rate, rho = classify_window(win_arr)
        seg_idx = t // SEGMENT_STEPS
        rec = {"t": t, "label": label, "e_w": err_rate, "rho": rho, "seg_idx": seg_idx}
        all_eval_records.append(rec)
        seg_eval_records[seg_idx].append(rec)

    seg_correctness: dict[int, float] = {}
    for i in range(N_SEGMENTS):
        recs = seg_eval_records[i]
        if not recs:
            seg_correctness[i] = float("nan")
            continue
        gt_label = _ground_truth_label(segment_types[i])
        n_correct = sum(1 for r in recs if r["label"] == gt_label)
        seg_correctness[i] = n_correct / len(recs)

    return {
        "W": W,
        "all_eval_records": all_eval_records,
        "seg_eval_records": seg_eval_records,
        "seg_correctness": seg_correctness,
    }


def _ground_truth_label(seg_type: str) -> str:
    """Ground-truth correct label for a segment type (Exp 158 taxonomy)."""
    if seg_type == "SLOW":
        return "STRUCTURAL"
    elif seg_type == "CTRL":
        return "OK"
    elif seg_type in ("NOISE", "PLACE"):
        return "NOISE"
    else:
        raise ValueError(f"Unknown segment type: {seg_type}")


# ---------------------------------------------------------------------------
# Core N2+N3 COMPLETED controller simulation (parameterised by K policy)
# ---------------------------------------------------------------------------

def _run_n3_controller(
    correct_arr: np.ndarray,
    segment_types: list[str],
    *,
    use_tier_b_k: bool,
) -> dict:
    """Simulate N2+N3 COMPLETED controller.

    If use_tier_b_k=False: uses constant LOCK_K_B8=8 (agent b8, verbatim exp169).
    If use_tier_b_k=True:  uses tier-B derived K(t) (agent bB).

    Delta A (descent driver) and Delta B (no wrap) are verbatim from exp169 in both.

    Returns a dict with eval_records, dial_trajectory, lock_events, descent_events,
    seg_eval_records, seg_correctness, change_by_seg_type, change_seg_idxs,
    total_dial_changes, descent_seg_idxs, descent_by_seg_type, total_descent_changes,
    wrap_count, final_dial, final_locked, final_lock_class, final_k (for bB; 8 for b8),
    k_trajectory, completed_run_lengths, k_at_lock_events, k_at_dial_changes,
    k_distinct_per_fork, pegged_eval_count, total_eval_count.
    """
    n_total = len(correct_arr)
    HORIZON = 100
    MAX_SCORABLE_T = n_total - 1 - HORIZON

    current_W = DIAL_CANDIDATES[0]
    dial_trajectory: list[dict] = [{"t": -1, "dial_change_to": current_W,
                                     "trust_at_change": None, "lock_event": None}]

    rolling_outcomes: _deque[int] = _deque(maxlen=ROLLING_TRUST_WINDOW)
    pending: list[dict] = []
    scored_since_change = 0

    label_run: list[str] = []
    locked = False
    lock_class: str | None = None

    # Derived K state (used by both; b8 just always returns 8 from helper)
    completed_run_lengths: list[int] = []
    current_run_len = 0
    current_run_class: str | None = None

    eval_records: list[dict] = []
    lock_events: list[dict] = []
    descent_events: list[dict] = []

    k_trajectory: list[dict] = []
    k_at_lock_events: list[dict] = []
    k_at_dial_changes: list[dict] = []
    k_values_seen: set[int] = set()

    last_logged_k: int | None = None

    change_seg_idxs: list[int] = []
    descent_seg_idxs: list[int] = []

    wrap_count = 0
    pegged_eval_count = 0
    total_eval_count = 0

    def _get_k() -> int:
        if not use_tier_b_k:
            return LOCK_K_B8
        return compute_k_gap(completed_run_lengths)

    for t in range(EVAL_EVERY - 1, n_total, EVAL_EVERY):

        k_t = _get_k()
        k_values_seen.add(k_t)
        if last_logged_k is None or k_t != last_logged_k:
            k_trajectory.append({"t": t, "k": k_t,
                                  "n_completed": len(completed_run_lengths),
                                  "reason": "k_changed_or_first"})
            last_logged_k = k_t

        # Step 1: mature pending forecasts
        newly_matured: list[dict] = []
        still_pending: list[dict] = []
        for pf in pending:
            if t >= pf["t_eval"] + HORIZON:
                newly_matured.append(pf)
            else:
                still_pending.append(pf)
        pending = still_pending

        for pf in newly_matured:
            t_eval = pf["t_eval"]
            label = pf["label"]
            e_w = pf["e_w"]
            next_slice = correct_arr[t_eval + 1: t_eval + 1 + HORIZON]
            e_next = 1.0 - float(next_slice.mean())
            if label == "OK":
                violated = int(e_next >= OK_VIOLATION_THRESH)
            elif label == "NOISE":
                violated = int(abs(e_next - e_w) >= NOISE_VIOLATION_DELTA)
            else:
                continue
            rolling_outcomes.append(1 - violated)
            scored_since_change += 1

        # Step 2: check trust rule (ASCENT) — only if NOT LOCKED, Delta B
        if (not locked and
                scored_since_change >= MIN_SCORED_FOR_TRUST and
                len(rolling_outcomes) > 0):
            trust = float(sum(rolling_outcomes) / len(rolling_outcomes))
            if trust < TRUST_FIRE_THRESH:
                idx = DIAL_CANDIDATES.index(current_W)
                new_idx = min(idx + 1, len(DIAL_CANDIDATES) - 1)
                if new_idx == idx:
                    pass
                else:
                    current_W = DIAL_CANDIDATES[new_idx]
                    label_run = []
                    locked = False
                    lock_class = None
                    # Discard in-progress run on dial change
                    current_run_len = 0
                    current_run_class = None
                rolling_outcomes.clear()
                scored_since_change = 0
                pending = []
                seg_idx_now = t // SEGMENT_STEPS
                change_seg_idxs.append(seg_idx_now)
                k_at_dial_changes.append({"t": t, "k": k_t, "event": "ASCENT",
                                           "to_W": current_W})
                dial_trajectory.append({
                    "t": t,
                    "dial_change_to": current_W,
                    "trust_at_change": trust,
                    "lock_event": None,
                    "seg_idx": seg_idx_now,
                    "seg_type": segment_types[seg_idx_now],
                    "event_type": "ASCENT",
                    "k_at_change": k_t,
                })

        # Step 3: evaluate current window under current_W
        if t + 1 >= current_W:
            win_arr = correct_arr[max(0, t - current_W + 1): t + 1].astype(np.float64)
            label, err_rate, rho = classify_window(win_arr)
            seg_idx_now = t // SEGMENT_STEPS
            eval_records.append({
                "t": t, "label": label, "e_w": err_rate, "rho": rho,
                "W_used": current_W, "locked": locked, "seg_idx": seg_idx_now,
                "k_t": k_t,
            })
            total_eval_count += 1
            if k_t in (K_MIN, K_MAX):
                pegged_eval_count += 1

            if label in ("OK", "NOISE") and t <= MAX_SCORABLE_T:
                pending.append({"t_eval": t, "label": label, "e_w": err_rate})

            # Update in-progress run counter
            if current_run_class is None:
                current_run_class = label
                current_run_len = 1
            elif label == current_run_class:
                current_run_len += 1
            else:
                completed_run_lengths.append(current_run_len)
                current_run_class = label
                current_run_len = 1
                k_t = _get_k()
                k_values_seen.add(k_t)
                if k_t != last_logged_k:
                    k_trajectory.append({"t": t, "k": k_t,
                                         "n_completed": len(completed_run_lengths),
                                         "reason": "run_completed"})
                    last_logged_k = k_t

            label_run.append(label)

            # LOCK test using k_t
            prev_locked = locked
            if len(label_run) >= k_t and len(set(label_run[-k_t:])) == 1:
                new_lock_class = label_run[-1]
                if not locked or lock_class != new_lock_class:
                    locked = True
                    lock_class = new_lock_class
                    lock_ev = {"t": t, "event": "LOCKED", "on_class": lock_class,
                               "dial": current_W, "seg_idx": seg_idx_now,
                               "seg_type": segment_types[seg_idx_now],
                               "k_at_lock": k_t}
                    lock_events.append(lock_ev)
                    dial_trajectory[-1] = dict(dial_trajectory[-1], lock_event=lock_ev)
                    k_at_lock_events.append({"t": t, "k": k_t, "event": "LOCKED",
                                             "on_class": lock_class})
            else:
                if locked:
                    lock_ev = {"t": t, "event": "UNLOCKED", "on_class": lock_class,
                               "dial": current_W, "seg_idx": seg_idx_now,
                               "seg_type": segment_types[seg_idx_now],
                               "k_at_unlock": k_t}
                    lock_events.append(lock_ev)
                    dial_trajectory.append({
                        "t": t, "dial_change_to": current_W,
                        "trust_at_change": None, "lock_event": lock_ev,
                        "seg_idx": seg_idx_now, "seg_type": segment_types[seg_idx_now],
                        "event_type": "UNLOCK",
                        "k_at_change": k_t,
                    })
                locked = False
                lock_class = None

            # Delta A: DESCENT DRIVER
            if locked and lock_class in ("OK", "NOISE") and current_W > 200:
                from_W = current_W
                idx = DIAL_CANDIDATES.index(current_W)
                current_W = DIAL_CANDIDATES[idx - 1]
                seg_idx_now2 = t // SEGMENT_STEPS
                descent_ev = {
                    "t": t,
                    "from_W": from_W,
                    "to_W": current_W,
                    "lock_class": lock_class,
                    "seg_idx": seg_idx_now2,
                    "seg_type": segment_types[seg_idx_now2],
                    "k_at_descent": k_t,
                }
                descent_events.append(descent_ev)
                descent_seg_idxs.append(seg_idx_now2)
                k_at_dial_changes.append({"t": t, "k": k_t, "event": "DESCENT",
                                           "from_W": from_W, "to_W": current_W})
                dial_trajectory.append({
                    "t": t,
                    "dial_change_to": current_W,
                    "trust_at_change": None,
                    "lock_event": None,
                    "seg_idx": seg_idx_now2,
                    "seg_type": segment_types[seg_idx_now2],
                    "event_type": "DESCENT",
                    "from_W": from_W,
                    "lock_class": lock_class,
                    "k_at_change": k_t,
                })

    # Build per-segment eval records and correctness
    seg_eval_records: dict[int, list[dict]] = {i: [] for i in range(N_SEGMENTS)}
    for rec in eval_records:
        seg_eval_records[rec["seg_idx"]].append(rec)

    seg_correctness: dict[int, float] = {}
    for i in range(N_SEGMENTS):
        recs = seg_eval_records[i]
        if not recs:
            seg_correctness[i] = float("nan")
            continue
        gt_label = _ground_truth_label(segment_types[i])
        n_correct = sum(1 for r in recs if r["label"] == gt_label)
        seg_correctness[i] = n_correct / len(recs)

    change_by_seg_type: dict[str, int] = {"SLOW": 0, "CTRL": 0, "NOISE": 0, "PLACE": 0}
    for seg_idx in change_seg_idxs:
        seg_type = segment_types[seg_idx]
        change_by_seg_type[seg_type] = change_by_seg_type.get(seg_type, 0) + 1

    descent_by_seg_type: dict[str, int] = {"SLOW": 0, "CTRL": 0, "NOISE": 0, "PLACE": 0}
    for seg_idx in descent_seg_idxs:
        seg_type = segment_types[seg_idx]
        descent_by_seg_type[seg_type] = descent_by_seg_type.get(seg_type, 0) + 1

    return {
        "eval_records": eval_records,
        "dial_trajectory": dial_trajectory,
        "lock_events": lock_events,
        "descent_events": descent_events,
        "seg_eval_records": seg_eval_records,
        "seg_correctness": seg_correctness,
        "change_seg_idxs": change_seg_idxs,
        "change_by_seg_type": change_by_seg_type,
        "total_dial_changes": len(change_seg_idxs),
        "descent_seg_idxs": descent_seg_idxs,
        "descent_by_seg_type": descent_by_seg_type,
        "total_descent_changes": len(descent_seg_idxs),
        "wrap_count": wrap_count,
        "final_dial": current_W,
        "final_locked": locked,
        "final_lock_class": lock_class,
        "final_k": k_t if use_tier_b_k else LOCK_K_B8,
        # K-derived diagnostics
        "k_trajectory": k_trajectory,
        "completed_run_lengths": completed_run_lengths,
        "k_at_lock_events": k_at_lock_events,
        "k_at_dial_changes": k_at_dial_changes,
        "k_distinct_per_fork": len(k_values_seen),
        "k_values_seen": sorted(k_values_seen),
        "pegged_eval_count": pegged_eval_count,
        "total_eval_count": total_eval_count,
    }


# ---------------------------------------------------------------------------
# Post-hoc: evaluate all three agents for one fork
# ---------------------------------------------------------------------------

def compute_fork_scores(run_result: dict) -> dict:
    """Compute per-segment correctness for agents (a), (b8), (bB).

    All three readouts operate on the SAME correct_arr from run_result.
    """
    correct_arr = run_result["correct_arr"]
    segment_types = run_result["segment_types"]

    # Agent (a): fixed W=200
    a_result = eval_fixed_dial_segments(correct_arr, W=200, segment_types=segment_types)

    # Agent (b8): constant K=8, verbatim exp169
    b8_result = _run_n3_controller(correct_arr, segment_types, use_tier_b_k=False)

    # Agent (bB): tier-B derived K
    bB_result = _run_n3_controller(correct_arr, segment_types, use_tier_b_k=True)

    return {
        "segment_types": segment_types,
        # Agent (a)
        "a_seg_correctness": a_result["seg_correctness"],
        "a_seg_eval_records": a_result["seg_eval_records"],
        # Agent (b8)
        "b8_seg_correctness": b8_result["seg_correctness"],
        "b8_seg_eval_records": b8_result["seg_eval_records"],
        "b8_eval_records": b8_result["eval_records"],
        "b8_dial_trajectory": b8_result["dial_trajectory"],
        "b8_lock_events": b8_result["lock_events"],
        "b8_descent_events": b8_result["descent_events"],
        "b8_change_by_seg_type": b8_result["change_by_seg_type"],
        "b8_change_seg_idxs": b8_result["change_seg_idxs"],
        "b8_total_dial_changes": b8_result["total_dial_changes"],
        "b8_descent_by_seg_type": b8_result["descent_by_seg_type"],
        "b8_total_descent_changes": b8_result["total_descent_changes"],
        "b8_wrap_count": b8_result["wrap_count"],
        "b8_final_dial": b8_result["final_dial"],
        "b8_final_locked": b8_result["final_locked"],
        "b8_final_lock_class": b8_result["final_lock_class"],
        # Agent (bB)
        "bB_seg_correctness": bB_result["seg_correctness"],
        "bB_seg_eval_records": bB_result["seg_eval_records"],
        "bB_eval_records": bB_result["eval_records"],
        "bB_dial_trajectory": bB_result["dial_trajectory"],
        "bB_lock_events": bB_result["lock_events"],
        "bB_descent_events": bB_result["descent_events"],
        "bB_change_by_seg_type": bB_result["change_by_seg_type"],
        "bB_change_seg_idxs": bB_result["change_seg_idxs"],
        "bB_total_dial_changes": bB_result["total_dial_changes"],
        "bB_descent_by_seg_type": bB_result["descent_by_seg_type"],
        "bB_total_descent_changes": bB_result["total_descent_changes"],
        "bB_wrap_count": bB_result["wrap_count"],
        "bB_final_dial": bB_result["final_dial"],
        "bB_final_locked": bB_result["final_locked"],
        "bB_final_lock_class": bB_result["final_lock_class"],
        "bB_final_k": bB_result["final_k"],
        # K-derived diagnostics (bB)
        "bB_k_trajectory": bB_result["k_trajectory"],
        "bB_completed_run_lengths": bB_result["completed_run_lengths"],
        "bB_k_at_lock_events": bB_result["k_at_lock_events"],
        "bB_k_at_dial_changes": bB_result["k_at_dial_changes"],
        "bB_k_distinct_per_fork": bB_result["k_distinct_per_fork"],
        "bB_k_values_seen": bB_result["k_values_seen"],
        "bB_pegged_eval_count": bB_result["pegged_eval_count"],
        "bB_total_eval_count": bB_result["total_eval_count"],
    }


# ---------------------------------------------------------------------------
# Precondition checks — verbatim from exp168/169/170
# ---------------------------------------------------------------------------

def check_preconditions(
    run_results: list[dict],
    fork_scores: dict,
    seeds: list[int],
) -> tuple[bool, list[str]]:
    """Check PC1 (ahat_drift), PC2 (eval points per segment), PC3 (SLOW deranged error)."""
    failures: list[str] = []

    for rr in run_results:
        seed = rr["fork_seed"]
        if rr["ahat_drift"] >= PC1_AHAT_DRIFT_MAX:
            failures.append(
                f"PC1 FAIL: seed={seed} ahat_drift={rr['ahat_drift']:.4f} "
                f"(bound={PC1_AHAT_DRIFT_MAX})"
            )
        fs = fork_scores[seed]
        a_seg_recs = fs["a_seg_eval_records"]
        segment_types = rr["segment_types"]
        for seg_idx in range(N_SEGMENTS):
            n_pts = len(a_seg_recs.get(seg_idx, []))
            if n_pts < PC2_MIN_EVAL_POINTS:
                failures.append(
                    f"PC2 FAIL: seed={seed} seg={seg_idx}({segment_types[seg_idx]}) "
                    f"n_eval={n_pts} < {PC2_MIN_EVAL_POINTS}"
                )

    slow_deranged_errors: list[float] = []
    for rr in run_results:
        context = rr["context_arr"]
        correct = rr["correct_arr"]
        segment_types = rr["segment_types"]
        for seg_idx, seg_type in enumerate(segment_types):
            if seg_type != "SLOW":
                continue
            seg_start = seg_idx * SEGMENT_STEPS
            seg_end = seg_start + SEGMENT_STEPS
            seg_ctx = context[seg_start:seg_end]
            seg_corr = correct[seg_start:seg_end]
            deranged_mask = seg_ctx == 1
            if deranged_mask.sum() > 0:
                err = 1.0 - float(seg_corr[deranged_mask].mean())
                slow_deranged_errors.append(err)

    if slow_deranged_errors:
        pooled_slow_err = float(np.mean(slow_deranged_errors))
        if pooled_slow_err < PC3_SLOW_DERANGED_ERR_MIN:
            failures.append(
                f"PC3 FAIL: SLOW pooled deranged-phase error={pooled_slow_err:.4f} "
                f"< {PC3_SLOW_DERANGED_ERR_MIN}"
            )
    else:
        failures.append("PC3 FAIL: no SLOW deranged steps found")

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Evaluate predeclared properties
# ---------------------------------------------------------------------------

def _pool_seg_scores(run_results, fork_scores, seeds, seg_filter):
    """Pool per-segment correctness for agent (a), (b8), (bB) by segment type filter."""
    a_vals: list[float] = []
    b8_vals: list[float] = []
    bB_vals: list[float] = []
    for seed in seeds:
        rr = next(r for r in run_results if r["fork_seed"] == seed)
        segment_types = rr["segment_types"]
        fs = fork_scores[seed]
        for seg_idx, seg_type in enumerate(segment_types):
            if not seg_filter(seg_type):
                continue
            a_c = fs["a_seg_correctness"].get(seg_idx, float("nan"))
            b8_c = fs["b8_seg_correctness"].get(seg_idx, float("nan"))
            bB_c = fs["bB_seg_correctness"].get(seg_idx, float("nan"))
            if not math.isnan(a_c):
                a_vals.append(a_c)
            if not math.isnan(b8_c):
                b8_vals.append(b8_c)
            if not math.isnan(bB_c):
                bB_vals.append(bB_c)
    pa = float(np.mean(a_vals)) if a_vals else float("nan")
    pb8 = float(np.mean(b8_vals)) if b8_vals else float("nan")
    pbB = float(np.mean(bB_vals)) if bB_vals else float("nan")
    return pa, pb8, pbB


def evaluate(
    run_results: list[dict],
    fork_scores: dict,
    seeds: list[int],
) -> dict:
    """Evaluate P1/F1, P2/F2, P3(reported), P4/F4, P5/F5, P7/F7, F_HARM, P6(ungated)."""

    # --- P1: override concentration (bB ascent changes only) ---
    total_changes_all = 0
    slow_changes_all = 0
    for seed in seeds:
        fs = fork_scores[seed]
        cbt = fs["bB_change_by_seg_type"]
        tc = fs["bB_total_dial_changes"]
        total_changes_all += tc
        slow_changes_all += cbt.get("SLOW", 0)

    p1_conc = slow_changes_all / total_changes_all if total_changes_all > 0 else float("nan")
    p1_pass = not math.isnan(p1_conc) and p1_conc >= P1_CONC_MIN
    f1 = not math.isnan(p1_conc) and p1_conc <= F1_CONC_MAX

    # --- P2: responsiveness (bB first SLOW segment) ---
    p2_fork_pass: list[bool] = []
    for seed in seeds:
        rr = next(r for r in run_results if r["fork_seed"] == seed)
        schedule = rr["schedule"]
        first_slow_seg = schedule["first_slow_segment_idx"]
        fs = fork_scores[seed]
        change_seg_idxs = fs["bB_change_seg_idxs"]
        responded = any(idx == first_slow_seg for idx in change_seg_idxs)
        p2_fork_pass.append(responded)

    n_p2_forks = sum(1 for x in p2_fork_pass if x)
    p2_pass = n_p2_forks >= P2_FORKS_MIN
    f2 = n_p2_forks <= F2_FORKS_MAX

    # --- Valid and slow pooled scores (a, b8, bB) ---
    pooled_valid_a, pooled_valid_b8, pooled_valid_bB = _pool_seg_scores(
        run_results, fork_scores, seeds, lambda st: st != "SLOW")
    pooled_slow_a, pooled_slow_b8, pooled_slow_bB = _pool_seg_scores(
        run_results, fork_scores, seeds, lambda st: st == "SLOW")

    # Valid margins
    p3_margin_bB = (pooled_valid_bB - pooled_valid_a
                    if not (math.isnan(pooled_valid_bB) or math.isnan(pooled_valid_a))
                    else float("nan"))
    p3_margin_b8 = (pooled_valid_b8 - pooled_valid_a
                    if not (math.isnan(pooled_valid_b8) or math.isnan(pooled_valid_a))
                    else float("nan"))

    # P3 reported (bB vs a)
    p3_pass = not math.isnan(p3_margin_bB) and p3_margin_bB >= -P3_NO_HARM_MARGIN
    f_harm = not math.isnan(p3_margin_bB) and (pooled_valid_a - pooled_valid_bB) > F_HARM_MARGIN

    # Broken margins
    p4_margin_bB = (pooled_slow_bB - pooled_slow_a
                    if not (math.isnan(pooled_slow_bB) or math.isnan(pooled_slow_a))
                    else float("nan"))
    p4_margin_b8 = (pooled_slow_b8 - pooled_slow_a
                    if not (math.isnan(pooled_slow_b8) or math.isnan(pooled_slow_a))
                    else float("nan"))

    # P4 (bB broken margin vs a)
    p4_pass = not math.isnan(p4_margin_bB) and p4_margin_bB >= P4_IMPROVE_MIN
    f4 = not math.isnan(p4_margin_bB) and p4_margin_bB <= F4_IMPROVE_MAX

    # b8 concentration and responsiveness for P5 reference
    b8_total_changes = sum(fork_scores[s]["b8_total_dial_changes"] for s in seeds)
    b8_slow_changes = sum(fork_scores[s]["b8_change_by_seg_type"].get("SLOW", 0) for s in seeds)
    b8_conc = b8_slow_changes / b8_total_changes if b8_total_changes > 0 else float("nan")

    b8_p2_fork_pass: list[bool] = []
    for seed in seeds:
        rr = next(r for r in run_results if r["fork_seed"] == seed)
        schedule = rr["schedule"]
        first_slow_seg = schedule["first_slow_segment_idx"]
        fs = fork_scores[seed]
        change_seg_idxs = fs["b8_change_seg_idxs"]
        responded = any(idx == first_slow_seg for idx in change_seg_idxs)
        b8_p2_fork_pass.append(responded)
    b8_n_p2 = sum(1 for x in b8_p2_fork_pass if x)

    # --- P5: bB within 0.05 of b8's SAME-RUN pooled values OR better ---
    # Metrics: concentration, responsiveness count, valid margin (bB-a vs b8-a),
    #          broken margin (bB-a vs b8-a)
    p5_metrics: dict[str, dict] = {}
    p5_all_pass = True

    def _p5_check(name, bB_val, b8_val, higher_is_better=True):
        if math.isnan(bB_val) or math.isnan(b8_val):
            return {"value": bB_val, "b8_ref": b8_val, "pass": False, "f5": False,
                    "note": "nan"}
        if higher_is_better:
            ok = bB_val >= b8_val - P5_TOLERANCE
            f5 = bB_val < b8_val - F5_TOLERANCE
        else:
            # lower is better; flip: ok if bB <= b8 + tolerance
            ok = bB_val <= b8_val + P5_TOLERANCE
            f5 = bB_val > b8_val + F5_TOLERANCE
        return {"value": bB_val, "b8_ref": b8_val, "pass": ok, "f5": f5}

    conc_check = _p5_check("conc", p1_conc, b8_conc, higher_is_better=True)
    p5_metrics["conc"] = conc_check
    if not conc_check["pass"]:
        p5_all_pass = False

    resp_check = _p5_check("resp", float(n_p2_forks), float(b8_n_p2), higher_is_better=True)
    p5_metrics["resp"] = resp_check
    if not resp_check["pass"]:
        p5_all_pass = False

    valid_check = _p5_check("valid_margin", p3_margin_bB, p3_margin_b8, higher_is_better=True)
    p5_metrics["valid_margin"] = valid_check
    if not valid_check["pass"]:
        p5_all_pass = False

    broken_check = _p5_check("broken_margin", p4_margin_bB, p4_margin_b8, higher_is_better=True)
    p5_metrics["broken_margin"] = broken_check
    if not broken_check["pass"]:
        p5_all_pass = False

    p5_pass = p5_all_pass
    f5 = any(m["f5"] for m in p5_metrics.values())

    # --- P6: straddle projection (ungated) ---
    p6_projection_holds = not math.isnan(p3_margin_bB) and p3_margin_bB < P6_VALID_MARGIN_THRESH
    p6_projection_refuted = not math.isnan(p3_margin_bB) and p3_margin_bB >= P6_VALID_MARGIN_THRESH

    # --- P7: regulation claim ---
    # K(t) takes >= 2 distinct values in >= 7/8 forks
    # AND pegged fraction < 0.5 pooled
    p7_fork_diverse: list[bool] = []
    total_pegged = 0
    total_evals = 0
    for seed in seeds:
        fs = fork_scores[seed]
        k_distinct = fs["bB_k_distinct_per_fork"]
        p7_fork_diverse.append(k_distinct >= 2)
        total_pegged += fs["bB_pegged_eval_count"]
        total_evals += fs["bB_total_eval_count"]

    n_p7_diverse_forks = sum(1 for x in p7_fork_diverse if x)
    pooled_pegged_frac = total_pegged / total_evals if total_evals > 0 else float("nan")

    p7_diverse_ok = n_p7_diverse_forks >= P7_FORKS_MIN
    p7_pegged_ok = not math.isnan(pooled_pegged_frac) and pooled_pegged_frac < P7_PEGGED_MAX
    p7_pass = p7_diverse_ok and p7_pegged_ok
    f7 = not math.isnan(pooled_pegged_frac) and pooled_pegged_frac > F7_PEGGED_DEGENERATE

    # --- Verdict ---
    positive = p1_pass and p2_pass and p4_pass and p5_pass and p7_pass and not f_harm
    negative = f1 or f2 or f4 or f5 or f7 or f_harm
    if positive:
        verdict_str = "POSITIVE"
    elif negative:
        verdict_str = "NEGATIVE"
    else:
        verdict_str = "MIXED"

    # --- Diagnostics ---
    derang_variants: set[tuple] = set()
    for seed in seeds:
        rr = next(r for r in run_results if r["fork_seed"] == seed)
        d = tuple(rr["schedule"]["derangement"])
        derang_variants.add(d)

    all_descent_events_bB: list[dict] = []
    all_descent_events_b8: list[dict] = []
    for seed in seeds:
        fs = fork_scores[seed]
        for de in fs["bB_descent_events"]:
            all_descent_events_bB.append({"seed": seed, **de})
        for de in fs["b8_descent_events"]:
            all_descent_events_b8.append({"seed": seed, **de})

    total_wrap_bB = sum(fork_scores[s]["bB_wrap_count"] for s in seeds)
    total_wrap_b8 = sum(fork_scores[s]["b8_wrap_count"] for s in seeds)

    # CTRL-segment margin
    _, _, _ = _pool_seg_scores(run_results, fork_scores, seeds, lambda st: st == "CTRL")
    ctrl_a, ctrl_b8, ctrl_bB = _pool_seg_scores(
        run_results, fork_scores, seeds, lambda st: st == "CTRL")
    ctrl_margin_bB = (ctrl_bB - ctrl_a
                      if not (math.isnan(ctrl_bB) or math.isnan(ctrl_a)) else float("nan"))
    ctrl_margin_b8 = (ctrl_b8 - ctrl_a
                      if not (math.isnan(ctrl_b8) or math.isnan(ctrl_a)) else float("nan"))

    all_completed_bB: list[int] = []
    for seed in seeds:
        all_completed_bB.extend(fork_scores[seed]["bB_completed_run_lengths"])

    pooled_k_final_bB = [fork_scores[s]["bB_final_k"] for s in seeds]

    # --- Exp 172 ungated diagnostic: stationary locks inside SLOW segments (tier B failure sig) ---
    # A stationary lock inside SLOW = bB lock event on STRUCTURAL class while inside a SLOW segment
    # and the lock is NOT immediately followed by a dial change (i.e. it is a within-segment lock
    # that stays locked without being unlocked by a segment boundary).
    # Simpler operational count: lock events with on_class == "STRUCTURAL" inside SLOW segments.
    slow_locks_bB: list[dict] = []
    for seed in seeds:
        fs = fork_scores[seed]
        rr = next(r for r in run_results if r["fork_seed"] == seed)
        segment_types = rr["segment_types"]
        for le in fs["bB_lock_events"]:
            if le["event"] == "LOCKED" and le.get("seg_type") == "SLOW":
                slow_locks_bB.append({"seed": seed, **le})
    n_slow_locks_bB = len(slow_locks_bB)

    # --- Exp 172 ungated diagnostic: K spike/decay at plateau completions ---
    # A plateau completion is a completed run whose length >= HALF_PERIOD_SLOW (1000 evals = ~half-period
    # in evals at EVAL_EVERY=100: half_period steps / eval_every = 1000/100 = 10 evals).
    # We identify runs >= 10 eval-steps as "plateau-scale". Spike height = K after that completion.
    # Decay length = how many subsequent completions until K drops back below K_MAX.
    PLATEAU_EVAL_THRESHOLD = HALF_PERIOD_SLOW // EVAL_EVERY  # 10 eval-steps
    plateau_k_spikes: list[dict] = []
    for seed in seeds:
        fs = fork_scores[seed]
        crls = fs["bB_completed_run_lengths"]
        ktraj = fs["bB_k_trajectory"]
        for i, rl in enumerate(crls):
            if rl >= PLATEAU_EVAL_THRESHOLD:
                # K after this completion = compute_k_gap(crls[:i+1])
                window = crls[max(0, i + 1 - K_WINDOW_M): i + 1]
                k_after = int(max(K_MIN, min(K_MAX, max(window) + 1)))
                # Decay: how many completions after this until max(window) < K_MAX - 1
                decay_len = 0
                for j in range(i + 1, min(i + 1 + K_WINDOW_M + 1, len(crls))):
                    w2 = crls[max(0, j + 1 - K_WINDOW_M): j + 1]
                    k_j = int(max(K_MIN, min(K_MAX, max(w2) + 1)))
                    if k_j < K_MAX:
                        break
                    decay_len += 1
                else:
                    decay_len = -1  # never decayed within window
                plateau_k_spikes.append({
                    "seed": seed, "completion_idx": i, "run_length": rl,
                    "k_after": k_after, "decay_completions": decay_len,
                })

    return {
        "n_forks": len(seeds),
        # P1
        "total_changes_all": total_changes_all,
        "slow_changes_all": slow_changes_all,
        "p1_conc": p1_conc,
        "p1_pass": p1_pass,
        "f1": f1,
        # P2
        "p2_fork_pass": p2_fork_pass,
        "n_p2_forks": n_p2_forks,
        "p2_pass": p2_pass,
        "f2": f2,
        # P3 (reported, not gating)
        "pooled_valid_a": pooled_valid_a,
        "pooled_valid_b8": pooled_valid_b8,
        "pooled_valid_bB": pooled_valid_bB,
        "p3_margin_bB": p3_margin_bB,
        "p3_margin_b8": p3_margin_b8,
        "p3_pass": p3_pass,
        "f_harm": f_harm,
        # P4
        "pooled_slow_a": pooled_slow_a,
        "pooled_slow_b8": pooled_slow_b8,
        "pooled_slow_bB": pooled_slow_bB,
        "p4_margin_bB": p4_margin_bB,
        "p4_margin_b8": p4_margin_b8,
        "p4_pass": p4_pass,
        "f4": f4,
        # P5
        "p5_pass": p5_pass,
        "f5": f5,
        "p5_metrics": p5_metrics,
        "b8_conc": b8_conc,
        "b8_n_p2": b8_n_p2,
        # P6 (projection)
        "p6_projection_holds": p6_projection_holds,
        "p6_projection_refuted": p6_projection_refuted,
        # P7
        "p7_fork_diverse": p7_fork_diverse,
        "n_p7_diverse_forks": n_p7_diverse_forks,
        "pooled_pegged_frac": pooled_pegged_frac,
        "p7_diverse_ok": p7_diverse_ok,
        "p7_pegged_ok": p7_pegged_ok,
        "p7_pass": p7_pass,
        "f7": f7,
        # Verdict
        "verdict_str": verdict_str,
        # Diagnostics
        "derang_variants": [list(d) for d in sorted(derang_variants)],
        "all_descent_events_bB": all_descent_events_bB,
        "all_descent_events_b8": all_descent_events_b8,
        "total_wrap_bB": total_wrap_bB,
        "total_wrap_b8": total_wrap_b8,
        "ctrl_a": ctrl_a,
        "ctrl_b8": ctrl_b8,
        "ctrl_bB": ctrl_bB,
        "ctrl_margin_bB": ctrl_margin_bB,
        "ctrl_margin_b8": ctrl_margin_b8,
        "all_completed_run_lengths_bB": all_completed_bB,
        "pooled_k_final_bB": pooled_k_final_bB,
        # Exp 172 new ungated diagnostics
        "slow_locks_bB": slow_locks_bB,
        "n_slow_locks_bB": n_slow_locks_bB,
        "plateau_k_spikes": plateau_k_spikes,
    }


# ---------------------------------------------------------------------------
# Write JSONL rows + summary
# ---------------------------------------------------------------------------

def _nan_to_none(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def write_json_rows(
    run_results: list[dict],
    fork_scores: dict,
    ev: dict | None,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for rr in run_results:
            seed = rr["fork_seed"]
            fs = fork_scores[seed]
            segment_types = rr["segment_types"]
            schedule = rr["schedule"]

            seg_summary = []
            for i in range(N_SEGMENTS):
                seg_summary.append({
                    "seg_idx": i,
                    "seg_type": segment_types[i],
                    "a_correctness": _nan_to_none(fs["a_seg_correctness"].get(i)),
                    "b8_correctness": _nan_to_none(fs["b8_seg_correctness"].get(i)),
                    "bB_correctness": _nan_to_none(fs["bB_seg_correctness"].get(i)),
                    "n_eval_a": len(fs["a_seg_eval_records"].get(i, [])),
                    "n_eval_b8": len(fs["b8_seg_eval_records"].get(i, [])),
                    "n_eval_bB": len(fs["bB_seg_eval_records"].get(i, [])),
                })

            row: dict = {
                "exp": 172,
                "fork_seed": seed,
                "n_total": rr["n_total"],
                "ahat_drift": _nan_to_none(rr["ahat_drift"]),
                "segment_types": segment_types,
                "derangement": schedule["derangement"],
                "p_true_noise": _nan_to_none(schedule["p_true_noise"]),
                "noisy_cells_first5": schedule["noisy_cells"][:5],
                "first_slow_segment_idx": schedule["first_slow_segment_idx"],
                "segment_scores": seg_summary,
                # b8 stats
                "b8_total_dial_changes": fs["b8_total_dial_changes"],
                "b8_change_by_seg_type": fs["b8_change_by_seg_type"],
                "b8_total_descent_changes": fs["b8_total_descent_changes"],
                "b8_wrap_count": fs["b8_wrap_count"],
                "b8_final_dial": fs["b8_final_dial"],
                # bB stats
                "bB_total_dial_changes": fs["bB_total_dial_changes"],
                "bB_change_by_seg_type": fs["bB_change_by_seg_type"],
                "bB_change_seg_idxs": fs["bB_change_seg_idxs"],
                "bB_total_descent_changes": fs["bB_total_descent_changes"],
                "bB_wrap_count": fs["bB_wrap_count"],
                "bB_final_dial": fs["bB_final_dial"],
                "bB_final_locked": fs["bB_final_locked"],
                "bB_final_lock_class": fs["bB_final_lock_class"],
                "bB_final_k": fs["bB_final_k"],
                "bB_k_trajectory": fs["bB_k_trajectory"],
                "bB_completed_run_lengths": fs["bB_completed_run_lengths"],
                "bB_k_at_lock_events": fs["bB_k_at_lock_events"],
                "bB_k_at_dial_changes": fs["bB_k_at_dial_changes"],
                "bB_k_distinct_per_fork": fs["bB_k_distinct_per_fork"],
                "bB_k_values_seen": fs["bB_k_values_seen"],
                "bB_pegged_eval_count": fs["bB_pegged_eval_count"],
                "bB_total_eval_count": fs["bB_total_eval_count"],
                "bB_dial_trajectory_summary": [
                    {
                        "t": d["t"],
                        "to": d["dial_change_to"],
                        "trust": _nan_to_none(d.get("trust_at_change")),
                        "seg_idx": d.get("seg_idx"),
                        "seg_type": d.get("seg_type"),
                        "event_type": d.get("event_type"),
                        "from_W": d.get("from_W"),
                        "lock_class": d.get("lock_class"),
                        "k_at_change": d.get("k_at_change"),
                    }
                    for d in fs["bB_dial_trajectory"]
                ],
                "bB_descent_events": [
                    {"t": de["t"], "from_W": de["from_W"], "to_W": de["to_W"],
                     "lock_class": de["lock_class"],
                     "seg_idx": de["seg_idx"], "seg_type": de["seg_type"],
                     "k_at_descent": de.get("k_at_descent")}
                    for de in fs["bB_descent_events"]
                ],
            }
            fh.write(json.dumps(row) + "\n")

        if ev is not None:
            def _fv(v):
                if isinstance(v, float) and math.isnan(v):
                    return None
                return v

            summary: dict = {
                "exp": 172,
                "row_type": "summary",
                "p1_conc_bB": _fv(ev["p1_conc"]),
                "p1_pass": ev["p1_pass"],
                "f1": ev["f1"],
                "n_p2_forks_bB": ev["n_p2_forks"],
                "p2_pass": ev["p2_pass"],
                "f2": ev["f2"],
                "pooled_valid_a": _fv(ev["pooled_valid_a"]),
                "pooled_valid_b8": _fv(ev["pooled_valid_b8"]),
                "pooled_valid_bB": _fv(ev["pooled_valid_bB"]),
                "p3_margin_bB": _fv(ev["p3_margin_bB"]),
                "p3_margin_b8": _fv(ev["p3_margin_b8"]),
                "p3_pass": ev["p3_pass"],
                "p3_note": "REPORTED ONLY — does not gate this verdict",
                "f_harm": ev["f_harm"],
                "pooled_slow_a": _fv(ev["pooled_slow_a"]),
                "pooled_slow_b8": _fv(ev["pooled_slow_b8"]),
                "pooled_slow_bB": _fv(ev["pooled_slow_bB"]),
                "p4_margin_bB": _fv(ev["p4_margin_bB"]),
                "p4_margin_b8": _fv(ev["p4_margin_b8"]),
                "p4_pass": ev["p4_pass"],
                "f4": ev["f4"],
                "p5_pass": ev["p5_pass"],
                "f5": ev["f5"],
                "p5_metrics": ev["p5_metrics"],
                "b8_conc": _fv(ev["b8_conc"]),
                "b8_n_p2": ev["b8_n_p2"],
                "p6_projection_holds": ev["p6_projection_holds"],
                "p6_projection_refuted": ev["p6_projection_refuted"],
                "n_p7_diverse_forks": ev["n_p7_diverse_forks"],
                "pooled_pegged_frac": _fv(ev["pooled_pegged_frac"]),
                "p7_pass": ev["p7_pass"],
                "f7": ev["f7"],
                "verdict": ev["verdict_str"],
                "ctrl_margin_bB": _fv(ev["ctrl_margin_bB"]),
                "ctrl_margin_b8": _fv(ev["ctrl_margin_b8"]),
                "total_wrap_bB": ev["total_wrap_bB"],
                "total_wrap_b8": ev["total_wrap_b8"],
                "n_descent_events_bB": len(ev["all_descent_events_bB"]),
                "n_descent_events_b8": len(ev["all_descent_events_b8"]),
                "n_completed_run_lengths_bB": len(ev["all_completed_run_lengths_bB"]),
                "pooled_k_final_bB": ev["pooled_k_final_bB"],
            }
            fh.write(json.dumps(summary) + "\n")


# ---------------------------------------------------------------------------
# Print K trajectory (ungated diagnostic) for bB
# ---------------------------------------------------------------------------

def print_k_trajectory(run_results: list[dict], fork_scores: dict, seeds: list[int]) -> None:
    """Print K(t) trajectory per fork for agent (bB)."""
    print("K(t) trajectories for bB (gap-seeking lock horizon — ungated diagnostic):")
    for seed in seeds:
        fs = fork_scores[seed]
        ktraj = fs["bB_k_trajectory"]
        completed = fs["bB_completed_run_lengths"]
        k_final = fs["bB_final_k"]
        k_at_locks = fs["bB_k_at_lock_events"]
        k_at_dials = fs["bB_k_at_dial_changes"]
        k_distinct = fs["bB_k_distinct_per_fork"]
        k_seen = fs["bB_k_values_seen"]
        pegged = fs["bB_pegged_eval_count"]
        total_e = fs["bB_total_eval_count"]
        pegged_frac = pegged / total_e if total_e > 0 else float("nan")

        print(f"  seed={seed}  final_K={k_final}  n_completed_runs={len(completed)}"
              f"  k_distinct={k_distinct}  k_values_seen={k_seen}"
              f"  pegged_frac={pegged_frac:.3f}({pegged}/{total_e})")
        if completed:
            window = completed[-K_WINDOW_M:]
            mx = max(window)
            print(f"    completed_run_lengths(last{K_WINDOW_M}): {window}")
            print(f"    max={mx}  => max+1={mx+1}"
                  f"  => K=clamp({mx+1},{K_MIN},{K_MAX})={k_final}")
        else:
            print(f"    no completed runs => K={K_MIN} (clamp min)")

        if ktraj:
            traj_strs = [f"t={e['t']} K={e['k']}({e['reason']})" for e in ktraj[:10]]
            if len(ktraj) > 10:
                traj_strs.append(f"...({len(ktraj)} total)")
            print(f"    K trajectory: {'; '.join(traj_strs)}")

        if k_at_locks:
            lock_strs = [f"t={e['t']} K={e['k']} on {e['on_class']}" for e in k_at_locks[:5]]
            print(f"    K at lock events: {'; '.join(lock_strs)}")

        if k_at_dials:
            dial_strs = [f"t={e['t']} K={e['k']} {e['event']}" for e in k_at_dials[:5]]
            print(f"    K at dial changes: {'; '.join(dial_strs)}")
        print()
    print()


# ---------------------------------------------------------------------------
# Print dial trajectories for b8 and bB
# ---------------------------------------------------------------------------

def print_dial_trajectories(run_results: list[dict], fork_scores: dict, seeds: list[int],
                             agent_key: str, agent_label: str) -> None:
    """Print dial trajectory for one agent (b8 or bB)."""
    print(f"Agent ({agent_label}) dial trajectories + LOCK/DESCENT events:")
    for seed in seeds:
        rr = next(r for r in run_results if r["fork_seed"] == seed)
        fs = fork_scores[seed]
        segment_types = rr["segment_types"]
        schedule = rr["schedule"]

        seg_str = "  ".join(f"[{i}:{t}]" for i, t in enumerate(segment_types))
        print(f"  seed={seed}  schedule: {seg_str}  "
              f"derang={schedule['derangement']}  "
              f"p_noise={schedule['p_true_noise']:.2f}")

        traj = fs[f"{agent_key}_dial_trajectory"]
        traj_strs = []
        for d in traj:
            s = f"t={d['t']} W={d['dial_change_to']}"
            ev_type = d.get("event_type", "")
            k_ann = f" K={d.get('k_at_change', '?')}" if d.get("k_at_change") is not None else ""
            if ev_type == "ASCENT":
                seg_label = (f" seg={d.get('seg_type', '?')}"
                             if d.get("seg_type") else "")
                s += f" [ASCENT trust={d['trust_at_change']:.3f}{seg_label}{k_ann}]"
            elif ev_type == "DESCENT":
                s += (f" [DESCENT {d.get('from_W')}->{d['dial_change_to']}"
                      f" on_class={d.get('lock_class')} seg={d.get('seg_type','?')}{k_ann}]")
            elif ev_type == "UNLOCK":
                le = d.get("lock_event", {})
                s += f" [UNLOCK from {le.get('on_class','?')} seg={le.get('seg_type','?')}{k_ann}]"
            if d.get("lock_event") and ev_type not in ("DESCENT", "UNLOCK"):
                le = d["lock_event"]
                k_lock = f" K={le.get('k_at_lock','?')}" if le.get("k_at_lock") is not None else ""
                s += f" [LOCK:{le['event']} on {le['on_class']} seg={le.get('seg_type','?')}{k_lock}]"
            traj_strs.append(s)
        print(f"    traj: {' -> '.join(traj_strs)}")

        des_evs = fs[f"{agent_key}_descent_events"]
        if des_evs:
            des_strs = [f"t={de['t']} {de['from_W']}->{de['to_W']} "
                        f"({de['seg_type']} lock_class={de['lock_class']} K={de.get('k_at_descent','?')})"
                        for de in des_evs]
            print(f"    descent_events: {', '.join(des_strs)}")
        else:
            print("    descent_events: none")

        seg_lines = []
        for i in range(N_SEGMENTS):
            a_c = fs["a_seg_correctness"].get(i, float("nan"))
            b_c = fs[f"{agent_key}_seg_correctness"].get(i, float("nan"))
            st = segment_types[i]
            seg_lines.append(f"  seg{i}({st}): a={a_c:.3f} {agent_label}={b_c:.3f}")
        print("    " + "  ".join(seg_lines))

        cbt = fs[f"{agent_key}_change_by_seg_type"]
        dbt = fs[f"{agent_key}_descent_by_seg_type"]
        print(f"    ascent_changes: total={fs[f'{agent_key}_total_dial_changes']} by_type={cbt}")
        print(f"    descent_changes: total={fs[f'{agent_key}_total_descent_changes']} by_type={dbt}")
        print(f"    wrap_count={fs[f'{agent_key}_wrap_count']} (should be 0)")
        print(f"    final: W={fs[f'{agent_key}_final_dial']} "
              f"locked={fs[f'{agent_key}_final_locked']}({fs[f'{agent_key}_final_lock_class']})")
        print()
    print()


# ---------------------------------------------------------------------------
# Print per-segment correctness table (all three agents)
# ---------------------------------------------------------------------------

def print_score_table(run_results: list[dict], fork_scores: dict, seeds: list[int],
                      ev: dict) -> None:
    """Print pooled per-segment-type correctness table (a) vs (b8) vs (bB)."""
    def _f(v, w=7):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "nan".rjust(w)
        return f"{v:.4f}".rjust(w)

    print("Pooled per-segment-type correctness (a=W200 / b8=K8 / bB=tier-B):")
    print(f"  {'seg_type':>10}  {'a_pool':>8}  {'b8_pool':>8}  {'bB_pool':>8}"
          f"  {'bB-a':>7}  {'bB-b8':>7}  {'gt_label':>10}")
    print("  " + "-" * 70)

    for seg_type in ["SLOW", "CTRL", "NOISE", "PLACE"]:
        pa, pb8, pbB = _pool_seg_scores(
            run_results, fork_scores, seeds, lambda st, s=seg_type: st == s)
        diff_a = pbB - pa if not (math.isnan(pa) or math.isnan(pbB)) else float("nan")
        diff_b8 = pbB - pb8 if not (math.isnan(pb8) or math.isnan(pbB)) else float("nan")
        gt = _ground_truth_label(seg_type)
        print(f"  {seg_type:>10}  {_f(pa)}  {_f(pb8)}  {_f(pbB)}"
              f"  {_f(diff_a)}  {_f(diff_b8)}  {gt:>10}")
    print()

    print(f"  Valid (CTRL+NOISE+PLACE):")
    print(f"    a={_f(ev['pooled_valid_a'])}  b8={_f(ev['pooled_valid_b8'])}"
          f"  bB={_f(ev['pooled_valid_bB'])}")
    print(f"    bB-a margin={_f(ev['p3_margin_bB'])}  b8-a margin={_f(ev['p3_margin_b8'])}")
    print(f"  SLOW:")
    print(f"    a={_f(ev['pooled_slow_a'])}  b8={_f(ev['pooled_slow_b8'])}"
          f"  bB={_f(ev['pooled_slow_bB'])}")
    print(f"    bB-a margin={_f(ev['p4_margin_bB'])}  b8-a margin={_f(ev['p4_margin_b8'])}")
    print(f"  CTRL specifically (exp168 harm locus):")
    print(f"    a={_f(ev['ctrl_a'])}  b8={_f(ev['ctrl_b8'])}  bB={_f(ev['ctrl_bB'])}")
    print(f"    bB-a={_f(ev['ctrl_margin_bB'])}  b8-a={_f(ev['ctrl_margin_b8'])}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exp 172 — K-endogenization gap-seeking horizon. "
            "Three same-schedule arms: (a) W=200, (b8) K=8, (bB) gap-seeking K."
        )
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Smoke run: seed=[138] only, full 24000 steps, prints segment schedule, "
            "K trajectory (bB), dial trajectories, per-segment scores "
            "for all three arms. No verdict."
        ),
    )
    args = parser.parse_args()

    smoke = args.smoke
    seeds = [138] if smoke else SEEDS

    print("=" * 80)
    print("Exp 172 — K-ENDOGENIZATION GAP-SEEKING HORIZON")
    print("          K(t) = clamp(max(last 12 completed runs) + 1, 3, 16)")
    print("          Three arms: (a) W=200, (b8) K=8, (bB) gap-seeking; same schedule/stream.")
    print("=" * 80)
    print()
    print(f"SURPRISE_WINDOW={SURPRISE_WINDOW}  [must be 200; asserted]")
    assert SURPRISE_WINDOW == 200, f"SURPRISE_WINDOW={SURPRISE_WINDOW}, expected 200"
    print(f"CLASSIFIER_WINDOW_DEFAULT={CLASSIFIER_WINDOW_DEFAULT}  EVAL_EVERY={EVAL_EVERY}")
    print(f"ERROR_RATE_OK_THRESH={ERROR_RATE_OK_THRESH}  LAG1_RHO_STRUCT_THRESH={LAG1_RHO_STRUCT_THRESH}")
    print(f"DIAL_CANDIDATES={DIAL_CANDIDATES}  starting W={CLASSIFIER_WINDOW_DEFAULT}")
    print(f"GAP-SEEKING K: window_M={K_WINDOW_M}  K_MIN={K_MIN}  K_MAX={K_MAX}")
    print(f"  K(t) = clamp(max(last {K_WINDOW_M} runs) + 1, {K_MIN}, {K_MAX})")
    print(f"  with no completed runs: K(t) = {K_MIN}")
    print(f"AGENT (b8): constant K={LOCK_K_B8} (verbatim exp169)")
    print(f"DELTA B (verbatim exp169) — NO WRAP: ascent saturates at 1600")
    print(f"DELTA A (verbatim exp169) — DESCENT: locked on OK/NOISE AND W>200 => step down")
    print(f"N3 trust: TRUST_FIRE_THRESH={TRUST_FIRE_THRESH}  ROLLING_TRUST_WINDOW={ROLLING_TRUST_WINDOW}  "
          f"MIN_SCORED_FOR_TRUST={MIN_SCORED_FOR_TRUST}")
    print(f"N3 forecast: OK_VIOLATION_THRESH={OK_VIOLATION_THRESH}  "
          f"NOISE_VIOLATION_DELTA={NOISE_VIOLATION_DELTA}  STRUCTURAL=NOT scored")
    print(f"P5 reference: b8's SAME-RUN pooled values (schedule confound eliminated)")
    print(f"P5 tolerance: within {P5_TOLERANCE} of b8 or better; F5 if worse by > {F5_TOLERANCE}")
    print(f"P7 regulation: >= {P7_FORKS_MIN}/8 forks with >= 2 distinct K AND "
          f"pegged_frac < {P7_PEGGED_MAX}; F7 if pegged > {F7_PEGGED_DEGENERATE}")
    print(f"P3: REPORTED ONLY — does not gate this verdict")
    print(f"F_HARM: valid deficit > {F_HARM_MARGIN} gates verdict")
    print(f"SLOW: HALF_PERIOD_SLOW={HALF_PERIOD_SLOW}  derang forced by seed parity")
    print(f"N_STEPS={N_STEPS}  N_SEGMENTS={N_SEGMENTS}  SEGMENT_STEPS={SEGMENT_STEPS}")
    print(f"SCHEDULE_SEED_OFFSET={SCHEDULE_SEED_OFFSET}  PLACE_NOISE_SEED_OFFSET={PLACE_NOISE_SEED_OFFSET}")
    print(f"PC1_AHAT_DRIFT_MAX={PC1_AHAT_DRIFT_MAX}")
    print(f"Seeds: {seeds}")
    print()

    # --- Load mirro spine (read-only) ---
    mirro = Creature.load("creature/state/mirro")
    print(f"Loaded mirro: age={mirro.age_steps}, world={mirro.world.rows}x{mirro.world.cols}, "
          f"n_colors={mirro.world.n_colors}, n_cells={mirro.world.n_cells}")
    base_cmap = list(mirro.world.cmap)
    n_colors = mirro.world.n_colors
    n_cells = mirro.world.n_cells
    print(f"Base cmap: {base_cmap}")
    print()

    # --- Run all seeds ---
    run_results: list[dict] = []
    fork_scores: dict = {}

    for seed in seeds:
        schedule = draw_fork_schedule(fork_seed=seed, n_cells=n_cells)
        print(
            f"  seed={seed}  schedule={schedule['segment_types']}  "
            f"derang={schedule['derangement']}  p_noise={schedule['p_true_noise']:.2f}  "
            f"noisy_cells(5)={schedule['noisy_cells'][:5]}  "
            f"first_slow_seg={schedule['first_slow_segment_idx']}",
            flush=True,
        )
        print(f"    Running seed={seed} (24000 steps, 6 segments) ...", flush=True)
        rr = run_fork(
            mirro=mirro,
            fork_seed=seed,
            base_cmap=base_cmap,
            n_colors=n_colors,
            schedule=schedule,
        )
        run_results.append(rr)
        fs = compute_fork_scores(rr)
        fork_scores[seed] = fs
        print(f"    ahat_drift={rr['ahat_drift']:.4f}  "
              f"b8: ascent={fs['b8_total_dial_changes']} desc={fs['b8_total_descent_changes']} "
              f"wrap={fs['b8_wrap_count']}  "
              f"bB: ascent={fs['bB_total_dial_changes']} desc={fs['bB_total_descent_changes']} "
              f"wrap={fs['bB_wrap_count']}  "
              f"bB_final_K={fs['bB_final_k']}  "
              f"bB_k_distinct={fs['bB_k_distinct_per_fork']}  "
              f"bB_k_seen={fs['bB_k_values_seen']}  "
              f"n_completed_runs={len(fs['bB_completed_run_lengths'])}",
              flush=True)
    print()

    # --- Print K trajectories for bB (ungated diagnostic) ---
    print_k_trajectory(run_results, fork_scores, seeds)

    # --- Print dial trajectories for both controllers ---
    print_dial_trajectories(run_results, fork_scores, seeds, "b8", "b8")
    print_dial_trajectories(run_results, fork_scores, seeds, "bB", "bB")

    # --- Derangement variant audit ---
    variants = set(tuple(rr["schedule"]["derangement"]) for rr in run_results)
    print(f"Derangement variants in run: {[list(v) for v in sorted(variants)]}")
    if len(variants) == 2:
        print("  => Both variants present (GOOD)")
    elif len(seeds) <= 1:
        print(f"  => Single seed smoke run (seed parity forced variant)")
    else:
        print(f"  => WARNING: only {len(variants)} variant(s) present (expected 2 for 8 forks)")
    print()

    # --- Smoke exit ---
    if smoke:
        print("=" * 80)
        print("SMOKE ONLY — no verdict")
        print("=" * 80)
        return

    # --- Precondition check ---
    pc_pass, pc_failures = check_preconditions(run_results, fork_scores, seeds)
    if pc_failures:
        print("PRECONDITION FAILED:")
        for f in pc_failures:
            print(f"  {f}")
        print()
        print("PRECONDITION FAILED — no verdict.")
        out_rows_path = Path(__file__).parent / "outputs" / "exp172_rows.json"
        write_json_rows(run_results, fork_scores, ev=None, path=out_rows_path)
        print(f"JSON rows written to {out_rows_path} (no verdict)")
        return

    print("Preconditions: all PASS")
    print()

    # --- Evaluate predeclared outcome map ---
    ev = evaluate(run_results, fork_scores, seeds)
    n = ev["n_forks"]

    # --- Print score table ---
    print_score_table(run_results, fork_scores, seeds, ev)

    def _fv(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "nan"
        return f"{v:.4f}"

    print("=" * 80)
    print("PREDECLARED OUTCOME MAP")
    print("=" * 80)
    print()

    print(f"P1 (bB override concentration >= {P1_CONC_MIN}, time-share chance 0.5):")
    print(f"  Total bB ascent changes = {ev['total_changes_all']}  "
          f"in SLOW = {ev['slow_changes_all']}  "
          f"pooled fraction = {_fv(ev['p1_conc'])}  (need >= {P1_CONC_MIN})")
    print(f"  FALSIFIER F1: fraction <= {F1_CONC_MAX}: "
          f"{'YES — F1 FIRES' if ev['f1'] else 'no'}")
    p1_status = "PASS" if ev["p1_pass"] else ("FALSIFIER F1" if ev["f1"] else "MIXED (not P1, not F1)")
    print(f"  => P1: {p1_status}")
    print()

    print(f"P2 (bB responsiveness: >= {P2_FORKS_MIN}/{n} forks respond on first SLOW seg):")
    for i, seed in enumerate(seeds):
        print(f"  seed={seed}: responded_on_first_slow={ev['p2_fork_pass'][i]}")
    print(f"  Forks responding: {ev['n_p2_forks']}/{n}  (need >= {P2_FORKS_MIN})")
    print(f"  FALSIFIER F2: <= {F2_FORKS_MAX}/{n} forks: "
          f"{'YES — F2 FIRES' if ev['f2'] else 'no'}")
    p2_status = "PASS" if ev["p2_pass"] else ("FALSIFIER F2" if ev["f2"] else "MIXED (not P2, not F2)")
    print(f"  => P2: {p2_status}")
    print()

    print(f"P3 (no harm in valid regimes: bB >= a - {P3_NO_HARM_MARGIN})")
    print(f"  [REPORTED ONLY — does NOT gate this verdict]")
    print(f"  pooled_valid_a={_fv(ev['pooled_valid_a'])}  "
          f"pooled_valid_bB={_fv(ev['pooled_valid_bB'])}  "
          f"margin(bB-a)={_fv(ev['p3_margin_bB'])}  (threshold: >= -{P3_NO_HARM_MARGIN})")
    p3_status = "PASS" if ev["p3_pass"] else "NOT MET"
    print(f"  => P3: {p3_status} (reported; not gating)")
    print(f"  F_HARM: (a)-(bB) > {F_HARM_MARGIN} (GATES VERDICT): "
          f"{'YES — F_HARM FIRES' if ev['f_harm'] else 'no'}")
    print()

    print(f"P4 (bB overrides earn keep: pooled_slow_bB >= pooled_slow_a + {P4_IMPROVE_MIN}):")
    print(f"  pooled_slow_a={_fv(ev['pooled_slow_a'])}  "
          f"pooled_slow_bB={_fv(ev['pooled_slow_bB'])}  "
          f"margin(bB-a)={_fv(ev['p4_margin_bB'])}  (need >= {P4_IMPROVE_MIN})")
    print(f"  FALSIFIER F4: (bB)-(a) <= {F4_IMPROVE_MAX}: "
          f"{'YES — F4 FIRES' if ev['f4'] else 'no'}")
    p4_status = "PASS" if ev["p4_pass"] else ("FALSIFIER F4" if ev["f4"] else "MIXED (not P4, not F4)")
    print(f"  => P4: {p4_status}")
    print()

    print(f"P5 (GAP-SEEKING CLAIM: bB within {P5_TOLERANCE} of b8's SAME-RUN pooled values OR better):")
    print(f"  b8 reference values (same-run pooled): "
          f"conc={_fv(ev['b8_conc'])}  resp={ev['b8_n_p2']}  "
          f"valid_margin={_fv(ev['p3_margin_b8'])}  broken_margin={_fv(ev['p4_margin_b8'])}")
    for metric_name, m in ev["p5_metrics"].items():
        vstr = _fv(m["value"]) if not isinstance(m["value"], str) else m["value"]
        b8str = _fv(m["b8_ref"]) if not isinstance(m["b8_ref"], str) else m["b8_ref"]
        status_str = "PASS" if m["pass"] else ("F5" if m["f5"] else "FAIL-soft")
        print(f"  {metric_name}: bB={vstr}  b8_ref={b8str}  {status_str}")
    print(f"  FALSIFIER F5 (any metric worse by > {F5_TOLERANCE}): "
          f"{'YES — F5 FIRES' if ev['f5'] else 'no'}")
    p5_status = "PASS" if ev["p5_pass"] else ("FALSIFIER F5" if ev["f5"] else "FAIL (soft)")
    print(f"  => P5: {p5_status}")
    print()

    print(f"P6 (VERIFIER PROJECTION — predeclared falsifiable, UNGATED):")
    print(f"  Projection: bB valid margin stays < {P6_VALID_MARGIN_THRESH}")
    print(f"  Current bB valid margin = {_fv(ev['p3_margin_bB'])}")
    if ev["p6_projection_refuted"]:
        print(f"  => STRADDLE-DOMINANCE PROJECTION REFUTED: margin >= {P6_VALID_MARGIN_THRESH}")
    elif ev["p6_projection_holds"]:
        print(f"  => Projection HOLDS: margin < {P6_VALID_MARGIN_THRESH} as predicted")
    else:
        print(f"  => margin indeterminate")
    print()

    print(f"P7 (REGULATION CLAIM):")
    print(f"  >= {P7_FORKS_MIN}/8 forks with >= 2 distinct K values:")
    for i, seed in enumerate(seeds):
        fs = fork_scores[seed]
        print(f"    seed={seed}: k_distinct={fs['bB_k_distinct_per_fork']} "
              f"k_seen={fs['bB_k_values_seen']}")
    print(f"  Forks with >= 2 distinct K: {ev['n_p7_diverse_forks']}/{n}  "
          f"(need >= {P7_FORKS_MIN})")
    print(f"  Pooled pegged fraction: {_fv(ev['pooled_pegged_frac'])}  "
          f"(need < {P7_PEGGED_MAX}; F7 if > {F7_PEGGED_DEGENERATE})")
    print(f"  FALSIFIER F7: pegged > {F7_PEGGED_DEGENERATE}: "
          f"{'YES — F7 FIRES' if ev['f7'] else 'no'}")
    p7_status = "PASS" if ev["p7_pass"] else ("FALSIFIER F7" if ev["f7"] else "MIXED (not P7, not F7)")
    print(f"  => P7: {p7_status}")
    print()

    # --- K(t) summary diagnostics ---
    all_crls = ev["all_completed_run_lengths_bB"]
    print(f"K(t) summary diagnostics (bB pooled across all forks):")
    print(f"  n_completed_run_lengths_total = {len(all_crls)}")
    if all_crls:
        print(f"  pooled: min={min(all_crls)}  max={max(all_crls)}  "
              f"mean={np.mean(all_crls):.2f}  "
              f"median={np.median(all_crls):.2f}")
    print(f"  final K per fork (bB): {ev['pooled_k_final_bB']}")
    print()

    # --- Exp 172 new ungated: stationary locks inside SLOW (tier B failure signature) ---
    print(f"Stationary locks inside SLOW segments (bB) — tier B failure signature (expected ~0):")
    print(f"  total SLOW lock events (bB): {ev['n_slow_locks_bB']}")
    for le in ev["slow_locks_bB"][:10]:
        print(f"  seed={le['seed']} t={le['t']} on_class={le.get('on_class')} "
              f"K={le.get('k_at_lock','?')} seg={le.get('seg_idx')}({le.get('seg_type')})")
    if ev["n_slow_locks_bB"] > 10:
        print(f"  ... ({ev['n_slow_locks_bB']} total, first 10 shown)")
    print()

    # --- Exp 172 new ungated: K spike/decay at plateau completions ---
    print(f"K spike/decay at plateau completions (bB) — plateau scale >= {HALF_PERIOD_SLOW // EVAL_EVERY} evals:")
    spikes = ev["plateau_k_spikes"]
    print(f"  total plateau completions: {len(spikes)}")
    for sp in spikes[:10]:
        print(f"  seed={sp['seed']} completion_idx={sp['completion_idx']} "
              f"run_length={sp['run_length']} k_after={sp['k_after']} "
              f"decay_completions={sp['decay_completions']}")
    if len(spikes) > 10:
        print(f"  ... ({len(spikes)} total, first 10 shown)")
    print()

    # --- Descent diagnostic ---
    print(f"Descent events bB: total={len(ev['all_descent_events_bB'])}")
    print(f"Descent events b8: total={len(ev['all_descent_events_b8'])}")
    for de in ev["all_descent_events_bB"][:20]:
        print(f"  seed={de['seed']} t={de['t']} {de['from_W']}->{de['to_W']} "
              f"lock_class={de['lock_class']} seg={de['seg_idx']}({de['seg_type']}) "
              f"K={de.get('k_at_descent','?')}")
    if len(ev["all_descent_events_bB"]) > 20:
        print(f"  ... ({len(ev['all_descent_events_bB'])} total, first 20 shown)")
    print()

    # --- Wrap confirmation ---
    print(f"Wrap count bB (should be 0): {ev['total_wrap_bB']}")
    print(f"Wrap count b8 (should be 0): {ev['total_wrap_b8']}")
    print()

    # --- Conjunct summary + VERDICT ---
    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    print(f"P1  (bB override concentration >= {P1_CONC_MIN}): {p1_status}")
    print(f"P2  (bB responsiveness >= {P2_FORKS_MIN}/{n} forks): {p2_status}")
    print(f"P3  [REPORTED ONLY — not gating]: {p3_status}")
    print(f"F_HARM (bB valid deficit > {F_HARM_MARGIN} — GATES): {'FIRED' if ev['f_harm'] else 'silent'}")
    print(f"P4  (bB overrides earn keep, margin >= +{P4_IMPROVE_MIN}): {p4_status}")
    print(f"P5  (gap-seeking: bB within {P5_TOLERANCE} of b8 same-run values or better): {p5_status}")
    print(f"P6  [verifier projection — ungated]: "
          f"{'REFUTED' if ev['p6_projection_refuted'] else 'holds'}")
    print(f"P7  (regulation: K diverse in >= {P7_FORKS_MIN}/8 forks AND pegged < {P7_PEGGED_MAX}): "
          f"{p7_status}")
    print()
    if ev["verdict_str"] == "POSITIVE":
        print("VERDICT: POSITIVE")
        print("GAP-SEEKING HORIZON SUPPORTED — the gap-seeking horizon endogenizes K at no")
        print("cost vs the magic constant (K=8 same-run arm). The human's stronger claim holds.")
    elif ev["verdict_str"] == "NEGATIVE":
        print("VERDICT: NEGATIVE")
        print("A falsifier fired — halt for a word.")
    else:
        print("VERDICT: MIXED")
    print()
    print("=" * 80)

    # --- Write JSON rows ---
    out_rows_path = Path(__file__).parent / "outputs" / "exp172_rows.json"
    write_json_rows(run_results, fork_scores, ev=ev, path=out_rows_path)
    print(f"JSON rows written to {out_rows_path}")

    # --- Write verdict JSON ---
    def _fv2(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "nan"
        return f"{v:.4f}"

    arms = {
        "P1_override_concentration_bB": {
            "pass": bool(ev["p1_pass"]),
            "reason": (
                f"bB pooled fraction of ascent changes in SLOW = {_fv2(ev['p1_conc'])} "
                f"(need >= {P1_CONC_MIN}); "
                f"total_ascent={ev['total_changes_all']} slow_ascent={ev['slow_changes_all']}; "
                f"F1 fired={ev['f1']}"
            ),
        },
        "P2_responsiveness_bB": {
            "pass": bool(ev["p2_pass"]),
            "reason": (
                f"{ev['n_p2_forks']}/{ev['n_forks']} bB forks responded on first SLOW segment "
                f"(need >= {P2_FORKS_MIN}); F2 fired={ev['f2']}"
            ),
        },
        "P3_no_harm_valid_REPORTED_ONLY": {
            "pass": bool(ev["p3_pass"]),
            "reason": (
                f"REPORTED ONLY — does not gate verdict. "
                f"pooled_valid_a={_fv2(ev['pooled_valid_a'])} "
                f"pooled_valid_bB={_fv2(ev['pooled_valid_bB'])} "
                f"margin(bB-a)={_fv2(ev['p3_margin_bB'])}; "
                f"ctrl_margin_bB={_fv2(ev['ctrl_margin_bB'])}; "
                f"F_HARM fired={ev['f_harm']} (gates verdict)"
            ),
        },
        "P4_overrides_earn_keep_bB": {
            "pass": bool(ev["p4_pass"]),
            "reason": (
                f"pooled_slow_a={_fv2(ev['pooled_slow_a'])} "
                f"pooled_slow_bB={_fv2(ev['pooled_slow_bB'])} "
                f"margin(bB-a)={_fv2(ev['p4_margin_bB'])} (need >= +{P4_IMPROVE_MIN}); "
                f"F4 fired={ev['f4']}"
            ),
        },
        "P5_gap_seeking_k_endogenization": {
            "pass": bool(ev["p5_pass"]),
            "reason": (
                f"bB within {P5_TOLERANCE} of b8 same-run pooled values on all 4 metrics. "
                f"conc: bB={_fv2(ev['p1_conc'])} b8={_fv2(ev['b8_conc'])}; "
                f"resp: bB={ev['n_p2_forks']} b8={ev['b8_n_p2']}; "
                f"valid_margin: bB={_fv2(ev['p3_margin_bB'])} b8={_fv2(ev['p3_margin_b8'])}; "
                f"broken_margin: bB={_fv2(ev['p4_margin_bB'])} b8={_fv2(ev['p4_margin_b8'])}; "
                f"F5 fired={ev['f5']}"
            ),
        },
        "P6_straddle_projection_ungated": {
            "pass": bool(ev["p6_projection_refuted"]),
            "reason": (
                f"UNGATED (FOURTH test). Projection: bB valid margin < {P6_VALID_MARGIN_THRESH}. "
                f"bB margin={_fv2(ev['p3_margin_bB'])}; "
                f"projection_holds={ev['p6_projection_holds']}; "
                f"projection_refuted={ev['p6_projection_refuted']}"
            ),
        },
        "P7_regulation_claim": {
            "pass": bool(ev["p7_pass"]),
            "reason": (
                f"{ev['n_p7_diverse_forks']}/{ev['n_forks']} forks with >= 2 distinct K "
                f"(need >= {P7_FORKS_MIN}); "
                f"pooled_pegged_frac={_fv2(ev['pooled_pegged_frac'])} "
                f"(need < {P7_PEGGED_MAX}; F7 if > {F7_PEGGED_DEGENERATE}); "
                f"F7 fired={ev['f7']}"
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp172_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp172",
        arms=arms,
        verdict=ev["verdict_str"],
        halted=False,
        notes=(
            "N3 K-endogenization gap-seeking horizon. "
            f"K(t)=clamp(max(last {K_WINDOW_M} runs)+1,{K_MIN},{K_MAX}). "
            "Three arms on same schedule/stream: (a) W=200, (b8) K=8, (bB) gap-seeking. "
            "P5 compares bB vs b8 same-run pooled values (schedule confound eliminated). "
            "P7 checks K regulation: diverse values and non-degenerate pegging. "
            "P3 REPORTED ONLY; F_HARM gates instead. "
            "Seeds 138-145 (exp171 used 130-137). "
            f"POSITIVE: GAP-SEEKING HORIZON SUPPORTED. NEGATIVE: falsifier fired. "
            f"wrap_bB={ev['total_wrap_bB']} wrap_b8={ev['total_wrap_b8']} "
            f"ctrl_margin_bB={_fv2(ev['ctrl_margin_bB'])}."
        ),
    )
    print(f"Verdict JSON written to {verdict_path}")


if __name__ == "__main__":
    main()
