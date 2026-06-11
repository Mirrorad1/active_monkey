"""Exp 173 — N3 multi-tempo regulated horizon (TIER-3 GATE).

RATCHET LINEAGE:
  Exp 167 introduced the N2+N3 LOCK controller.  Exp 168 found THE RATCHET LAW.
  Exp 169 (Delta A + Delta B) fixed this: MIXED — boundary transient reduced.
  Exp 170 (tier A): global P95 of completed run lengths.  FAILED — permanent
  ceiling (plateau-captured; the tier-A collapse).
  Exp 171 (tier B): recency median x2.  MIXED — K dipped below the SLOW
  half-period at fast-cycling phases, causing premature stationary locks inside
  broken segments (tier B's failure signature); broken margin halved.  Posted
  the lineage's best valid margin.
  Exp 172 (gap-seeking): K(t) = clamp(max(last 12 completed runs)+1, 3, 16).
  POSITIVE — gap-seeking horizon endogenizes K at no cost vs magic constant.
  Exp 172 testability verdict: at SINGLE-TEMPO richness a constant lock horizon
  in the gap is unbeatable, so the tier-3 claim (N3 regulates its own evidence
  horizon from the stream's reliability and tempo) needs a world where the gap
  MOVES.

THIS EXPERIMENT — MULTI-TEMPO REGULATED HORIZON (THE TIER-3 GATE):

  THE CLAIM: N3 usefully regulates its own evidence horizon where the gap moves.
  In a session containing both fast (SLOW600, half_period=600) and slow
  (SLOW1400, half_period=1400) broken segments, a TIME-WINDOWED regulator
  outperforms any constant K because:
    (a) fast-broken segments need a lower K (short run lengths can end quickly);
    (b) slow-broken segments need a higher K (run lengths are longer);
    (c) after a tempo change, a count-window retains stale evidence from the
        prior tempo; a time window expires that evidence within one period.
  A constant K is either too high for fast tempos (slow detection) or too low
  for slow tempos (premature stationary locks), or both.

  REGULATED K (arm bR):
    K(t) = clamp(max(completed runs within the last T_exp=3000 steps) + 1, 3, 16)
    with no live completions: K(t) = 3.
    Completions older than 3000 steps expire from the window (TIME-decayed, not
    count-decayed).  T_exp = segment length is the declared provided policy.
    Rationale: completion RATE varies across regimes, so Exp 172's count-window
    could not track tempo — old plateaus never expired; a time window is the
    natural recency statistic for a TEMPO measure: after a tempo change, the old
    tempo's evidence is gone within one expiry period.

  FOUR ARMS (post-hoc readouts over the same recorded stream):
    (a)   fixed W=200 baseline
    (c8)  N2+N3 COMPLETED controller, constant K=8
    (c16) N2+N3 COMPLETED controller, constant K=16
    (bR)  N2+N3 COMPLETED controller, TIME-REGULATED K(t)

  MULTI-TEMPO SESSION — one per fork:
    24000 steps = 8 contiguous segments of 3000 steps each.
    Per-fork rng (150_000+seed) shuffles the segment pool:
      [SLOW600, SLOW600, SLOW1400, SLOW1400, CTRL, NOISE, PLACE, CTRL]
      (two fast-broken, two slow-broken, four valid; derangement alternates
      by seed parity; each SLOW segment starts in context A with its own
      internal step offset).
    SLOW600 = StructCmap half_period=600; SLOW1400 = StructCmap half_period=1400.
    NOISE p_true drawn from {0.6, 0.7, 0.8}.  PLACE = random 13-of-25 cells.
    All draws logged.

READ-ONLY-DIAL DECLARATION:
  One creature run per fork; per-step correctness stream recorded; arms (c8),
  (c16), and (bR) are post-hoc readouts over that SAME stream.  The creature
  is never re-run or influenced by the dial policy.

SEEDS = list(range(146, 154)) — 8 fresh forks.

SCORING per segment:
  Eval points every 100 steps; window evaluates if enough history under the
  agent's current dial (for c8/c16/bR) or fixed W=200 (for a).
  Ground-truth label per segment type:
    SLOW600  -> STRUCTURAL   SLOW1400 -> STRUCTURAL
    CTRL     -> OK           NOISE/PLACE -> NOISE
  Per-segment class labels for scoring: broken600 (SLOW600 segments),
  broken1400 (SLOW1400 segments), valid (CTRL/NOISE/PLACE segments).

PRECONDITIONS:
  PC1 ahat_drift < 0.30.
  PC2 >= 20 eval points per segment at W=200 (segments are 3000 steps => ~30 evals).
  PC3 SLOW deranged-phase per-step error >= 0.9 pooled (both SLOW classes pooled).

PREDECLARED PROPERTIES AND FALSIFIERS:

  P_reg (THE TIER-3 GATE, load-bearing):
    pooled combined(bR) - max(combined(c8), combined(c16)) >= 0.10
    AND combined(bR) > both constants in >= 6/8 forks.
    FALSIFIER F_reg: some constant's combined within 0.05 of bR pooled
    (regulation is config even where the gap moves — the tier-3 claim REJECTED).

  P_resp (responsiveness):
    bR responds (>= 1 ascent) in the first broken segment of each fork
    in >= 7/8 forks.
    FALSIFIER F_resp: <= 4/8 forks respond.

  F_HARM:
    bR valid-class deficit vs (a) > 0.15 (gates verdict).

  P_K (regulation reality-check):
    bR's K(t) differs between the two broken tempos — pooled mean K during
    SLOW600 segments < pooled mean K during SLOW1400 segments by >= 2
    (the horizon tracks tempo).
    FALSIFIER F_K: difference < 0.5 (the regulator does not see tempo).

  VERDICT:
    POSITIVE (the tier-3 claim SUPPORTED — N3 usefully regulates its own
             evidence horizon where the gap moves)
             iff P_reg, P_resp, P_K all pass and F_HARM silent.
    NEGATIVE iff any falsifier fires (halt for a word).
    Otherwise MIXED.

  combined(arm) = min(broken600_pooled, broken1400_pooled, valid_pooled).

UNGATED DIAGNOSTICS:
  K(t) trajectories with segment boundaries; per-arm per-segment-class table;
  mean K by segment class; lock/descent latencies by tempo; expiry events.

--smoke: seed [146] only, full 24000 steps, prints schedule, K trajectory with
  mean-K-by-class, four-arm per-segment table, "SMOKE ONLY — no verdict".
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

SEEDS = list(range(146, 154))   # 8 fresh forks — exp172 used 138-145

N_SEGMENTS = 8
SEGMENT_STEPS = 3000
N_STEPS = N_SEGMENTS * SEGMENT_STEPS   # 24000
N_CHUNKS = N_STEPS // 100              # 240  (chunk_size=100)
CHUNK_SIZE = 100

# Per-fork rng offset for the segment schedule draw
SCHEDULE_SEED_OFFSET = 150_000         # rng 150_000 + fork_seed

# Noise rng offset for the PLACE world (verbatim from exp163)
PLACE_NOISE_SEED_OFFSET = 90_000       # rng 90_000 + fork_seed

# Classifier parameters (verbatim from exp158/159/162/163/165/166/167/168/169/170/172)
CLASSIFIER_WINDOW_DEFAULT = 200        # starting W for all agents
EVAL_EVERY = 100                       # evaluate every 100 steps once window full
ERROR_RATE_OK_THRESH = 0.05            # error_rate < 0.05 => OK
LAG1_RHO_STRUCT_THRESH = 0.3          # lag1_rho > 0.3 => STRUCTURAL (else NOISE)

# Dial candidate set (ordered for ascent; saturation at 1600 — Delta B from exp169)
DIAL_CANDIDATES = [200, 400, 800, 1600]

# EWMA channel parameters (verbatim from exp157/159/163/165/166/167/168/169/170/172)
ALPHA = 0.05        # EWMA learning rate
EWMA_INIT = 0.5     # initial EWMA value per cell

# N3 forecast-scoring thresholds (verbatim from exp165/167/168/169/170/172)
OK_VIOLATION_THRESH = 0.15             # L=OK violated iff e_next >= 0.15
NOISE_VIOLATION_DELTA = 0.30           # L=NOISE violated iff |e_next - e_w| >= 0.30
# STRUCTURAL: not scored

ROLLING_TRUST_WINDOW = 10              # last 10 scored labels
TRUST_FIRE_THRESH = 0.85               # advance dial when trust < 0.85 (>= 2 violations)
MIN_SCORED_FOR_TRUST = 3               # < 3 scored since last dial change => NO-EVIDENCE

# CONSTANT K for agents (c8) and (c16)
LOCK_K_C8 = 8
LOCK_K_C16 = 16

# TIME-REGULATED K parameters for agent (bR)
# K(t) = clamp(max(completed runs within last T_exp steps) + 1, K_MIN, K_MAX)
# with no live completions: K(t) = K_MIN
T_EXP = SEGMENT_STEPS                 # 3000 steps — one segment length (declared policy)
K_MIN = 3                              # clamp lower bound
K_MAX = 16                             # clamp upper bound

# SLOW segment parameters — TWO tempos
HALF_PERIOD_SLOW600 = 600              # fast-broken (requires W > 600 to detect)
HALF_PERIOD_SLOW1400 = 1400            # slow-broken (requires W > 1400 to detect)

# Derangement FORCED by seed parity (fixes Exp 167 rng accident):
#   seed even => [1,2,0]; seed odd => [2,0,1]
DERANGEMENT_OPTIONS = [[1, 2, 0], [2, 0, 1]]

# NOISE p_true options
P_TRUE_NOISE_OPTIONS = [0.6, 0.7, 0.8]

# PLACE parameters (verbatim from exp163)
P_TRUE_PLACE = 0.55
N_NOISY_CELLS = 13                     # 13 of 25 cells drawn as noisy per fork

# Segment pool: 2x SLOW600, 2x SLOW1400, 2x CTRL, 1x NOISE, 1x PLACE
SEGMENT_POOL_TEMPLATE = ["SLOW600", "SLOW600", "SLOW1400", "SLOW1400",
                          "CTRL", "NOISE", "PLACE", "CTRL"]

# Precondition thresholds
PC1_AHAT_DRIFT_MAX = 0.30             # PC1 (24000-step horizon-scaled bound)
PC2_MIN_EVAL_POINTS = 20              # PC2: >= 20 eval points per segment at W=200
                                       # (3000 steps / 100 = ~30 evals, so 20 is conservative)
PC3_SLOW_DERANGED_ERR_MIN = 0.9       # PC3: SLOW deranged-phase error >= 0.9 pooled

# P_reg thresholds (THE TIER-3 GATE)
P_REG_MARGIN_MIN = 0.10               # combined(bR) - max(c8,c16) >= 0.10
P_REG_FORKS_MIN = 6                   # bR > both constants in >= 6/8 forks
F_REG_WITHIN = 0.05                   # F_reg: some constant within 0.05 of bR pooled

# P_resp thresholds
P_RESP_FORKS_MIN = 7                  # >= 7/8 forks
F_RESP_FORKS_MAX = 4                  # F_resp: <= 4/8

# F_HARM
F_HARM_MARGIN = 0.15                  # bR valid deficit vs (a) > 0.15

# P_K thresholds
P_K_DIFF_MIN = 2.0                    # mean K SLOW600 < mean K SLOW1400 by >= 2
F_K_DIFF_MAX = 0.5                    # F_K: difference < 0.5


# ---------------------------------------------------------------------------
# StructCmap — verbatim from exp162/163/164/165/166/167/168/169/170/172
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
# NoisyCmap — verbatim from exp155/163/168/169/170/172
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
# RandomPlaceNoiseCmap — verbatim from exp159/163/168/169/170/172
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
# lag1_rho — verbatim from exp162/163/164/165/166/167/168/169/170/172
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
# classify_window — verbatim from exp162/163/164/165/166/167/168/169/170/172
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
# Ground-truth label — extended for two SLOW tempos
# ---------------------------------------------------------------------------

def _ground_truth_label(seg_type: str) -> str:
    """Ground-truth correct label for a segment type."""
    if seg_type in ("SLOW600", "SLOW1400"):
        return "STRUCTURAL"
    elif seg_type == "CTRL":
        return "OK"
    elif seg_type in ("NOISE", "PLACE"):
        return "NOISE"
    else:
        raise ValueError(f"Unknown segment type: {seg_type}")


# ---------------------------------------------------------------------------
# Segment class labels for scoring (broken600, broken1400, valid)
# ---------------------------------------------------------------------------

def _segment_class(seg_type: str) -> str:
    """Return broken600 / broken1400 / valid for a segment type."""
    if seg_type == "SLOW600":
        return "broken600"
    elif seg_type == "SLOW1400":
        return "broken1400"
    else:
        return "valid"


# ---------------------------------------------------------------------------
# Draw per-fork schedule parameters
# ---------------------------------------------------------------------------

def draw_fork_schedule(fork_seed: int, n_cells: int) -> dict:
    """Draw the per-fork multi-tempo session schedule and parameters."""
    rng = np.random.default_rng(SCHEDULE_SEED_OFFSET + fork_seed)

    derangement = DERANGEMENT_OPTIONS[fork_seed % 2]
    p_true_noise = float(P_TRUE_NOISE_OPTIONS[int(rng.integers(0, len(P_TRUE_NOISE_OPTIONS)))])

    noisy_cells_arr = rng.choice(n_cells, size=N_NOISY_CELLS, replace=False)
    noisy_cells = sorted(int(x) for x in noisy_cells_arr)

    # Shuffle the fixed segment pool
    pool = list(SEGMENT_POOL_TEMPLATE)  # 8 segments
    order_perm = rng.permutation(N_SEGMENTS)
    segment_types = [pool[int(i)] for i in order_perm]

    # Build per-segment params
    segment_params = []
    noise_seg_counter = 0
    place_seg_counter = 0
    for seg_type in segment_types:
        if seg_type == "SLOW600":
            segment_params.append({
                "type": "SLOW600",
                "half_period": HALF_PERIOD_SLOW600,
                "derangement": derangement,
            })
        elif seg_type == "SLOW1400":
            segment_params.append({
                "type": "SLOW1400",
                "half_period": HALF_PERIOD_SLOW1400,
                "derangement": derangement,
            })
        elif seg_type == "CTRL":
            segment_params.append({"type": "CTRL"})
        elif seg_type == "NOISE":
            segment_params.append({"type": "NOISE", "p_true": p_true_noise})
            noise_seg_counter += 1
        elif seg_type == "PLACE":
            segment_params.append({
                "type": "PLACE",
                "p_true": P_TRUE_PLACE,
                "noisy_cells": noisy_cells,
            })
            place_seg_counter += 1
        else:
            raise ValueError(f"Unknown segment type: {seg_type}")

    # First broken segment (either SLOW600 or SLOW1400) for P_resp
    first_broken_idx = next(
        i for i, t in enumerate(segment_types) if t in ("SLOW600", "SLOW1400")
    )

    return {
        "segment_types": segment_types,
        "segment_params": segment_params,
        "derangement": derangement,
        "p_true_noise": p_true_noise,
        "noisy_cells": noisy_cells,
        "first_broken_segment_idx": first_broken_idx,
    }


# ---------------------------------------------------------------------------
# Build segment-aware cmap objects
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
        if seg_type == "SLOW600":
            cmaps.append(StructCmap(base_cmap, n_colors, derangement, HALF_PERIOD_SLOW600))
        elif seg_type == "SLOW1400":
            cmaps.append(StructCmap(base_cmap, n_colors, derangement, HALF_PERIOD_SLOW1400))
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
# Run one fork (continuous across all 8 segments) — adapted from exp172
# ---------------------------------------------------------------------------

def run_fork(
    mirro: Creature,
    fork_seed: int,
    base_cmap: list,
    n_colors: int,
    schedule: dict,
) -> dict:
    """Run one continuous fork across all 8 segments (24000 steps)."""
    assert SURPRISE_WINDOW == 200, (
        f"SURPRISE_WINDOW from creature.py == {SURPRISE_WINDOW}, expected 200"
    )

    fork_name = f"exp173_s{fork_seed}"
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

        if seg_type in ("SLOW600", "SLOW1400"):
            cmap_obj.current_step = 0  # type: ignore[union-attr]

        seg_n_chunks = SEGMENT_STEPS // CHUNK_SIZE

        for chunk_in_seg in range(seg_n_chunks):
            chunk_idx = seg_idx * seg_n_chunks + chunk_in_seg
            chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF
            rng = np.random.default_rng(chunk_seed)

            for _step_in_chunk in range(CHUNK_SIZE):
                t = global_step
                step_in_seg = t - seg_start

                if seg_type in ("SLOW600", "SLOW1400"):
                    half_period = cmap_obj.half_period  # type: ignore[union-attr]
                    cmap_obj.current_step = step_in_seg  # type: ignore[union-attr]
                    context_arr[t] = int((step_in_seg // half_period) % 2)

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
# Post-hoc: evaluate fixed-dial agent (a) per segment
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


# ---------------------------------------------------------------------------
# Time-regulated K computation for agent (bR)
# ---------------------------------------------------------------------------

def compute_k_regulated(
    completed_run_entries: list[tuple[int, int]],
    current_t: int,
) -> tuple[int, int]:
    """Compute time-regulated K(t) and count of expired entries.

    K(t) = clamp(max(completed run lengths within last T_exp steps) + 1, K_MIN, K_MAX)
    with no live completions: K(t) = K_MIN (= 3).

    Args:
      completed_run_entries: list of (t_completed, run_length) pairs, ordered by
        t_completed.
      current_t: current time step.

    Returns:
      (K_t, n_expired): the computed K and number of entries that have expired.
    """
    cutoff = current_t - T_EXP
    live = [(tc, rl) for tc, rl in completed_run_entries if tc > cutoff]
    n_expired = len(completed_run_entries) - len(live)
    if not live:
        return K_MIN, n_expired
    max_len = max(rl for _, rl in live)
    raw = max_len + 1
    return int(max(K_MIN, min(K_MAX, raw))), n_expired


# ---------------------------------------------------------------------------
# Core N2+N3 COMPLETED controller simulation (parameterised by K policy)
# ---------------------------------------------------------------------------

# K policy tags
K_POLICY_CONSTANT_8 = "c8"
K_POLICY_CONSTANT_16 = "c16"
K_POLICY_REGULATED = "bR"


def _run_n3_controller(
    correct_arr: np.ndarray,
    segment_types: list[str],
    *,
    k_policy: str,
) -> dict:
    """Simulate N2+N3 COMPLETED controller.

    k_policy:
      "c8"  -> constant K=8 (agent c8)
      "c16" -> constant K=16 (agent c16)
      "bR"  -> time-regulated K(t) (agent bR)

    Delta A (descent driver) and Delta B (no wrap) are verbatim from exp169.

    Returns a dict with eval_records, dial_trajectory, lock_events, descent_events,
    seg_eval_records, seg_correctness, change_by_seg_type, change_seg_idxs,
    total_dial_changes, descent_seg_idxs, descent_by_seg_type, total_descent_changes,
    wrap_count, final_dial, final_locked, final_lock_class, final_k,
    k_trajectory, completed_run_entries (bR) / completed_run_lengths (c8/c16),
    k_at_lock_events, k_at_dial_changes, k_values_seen_sorted,
    expiry_events (bR only, else []),
    k_by_seg_type_accumulator (for mean K by segment class).
    """
    assert k_policy in (K_POLICY_CONSTANT_8, K_POLICY_CONSTANT_16, K_POLICY_REGULATED)

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

    # Run-tracking state
    # For bR: list of (t_completed, run_length) with timestamps for expiry
    # For c8/c16: list of int run lengths (no timestamps needed)
    completed_run_entries: list[tuple[int, int]] = []   # bR only
    completed_run_lengths: list[int] = []               # c8/c16
    current_run_len = 0
    current_run_class: str | None = None

    eval_records: list[dict] = []
    lock_events: list[dict] = []
    descent_events: list[dict] = []
    expiry_events: list[dict] = []

    k_trajectory: list[dict] = []
    k_at_lock_events: list[dict] = []
    k_at_dial_changes: list[dict] = []
    k_values_seen: set[int] = set()

    last_logged_k: int | None = None
    prev_n_entries = 0  # track expiries for bR

    change_seg_idxs: list[int] = []
    descent_seg_idxs: list[int] = []

    wrap_count = 0

    # Accumulate K values per segment class for mean-K diagnostic
    # {seg_class: [k_values...]}
    k_by_seg_class: dict[str, list[int]] = {
        "broken600": [], "broken1400": [], "valid": [],
    }

    def _get_k(t: int) -> tuple[int, int]:
        """Return (k_t, n_expired). n_expired only nonzero for bR."""
        if k_policy == K_POLICY_CONSTANT_8:
            return LOCK_K_C8, 0
        elif k_policy == K_POLICY_CONSTANT_16:
            return LOCK_K_C16, 0
        else:  # bR
            return compute_k_regulated(completed_run_entries, t)

    for t in range(EVAL_EVERY - 1, n_total, EVAL_EVERY):

        k_t, n_expired = _get_k(t)
        k_values_seen.add(k_t)

        # Track expiry events for bR
        if k_policy == K_POLICY_REGULATED and n_expired > 0:
            cutoff = t - T_EXP
            expired_list = [(tc, rl) for tc, rl in completed_run_entries if tc <= cutoff]
            if expired_list:
                expiry_events.append({
                    "t": t,
                    "n_expired": n_expired,
                    "expired_entries": expired_list[:5],  # log first 5
                    "k_after_expiry": k_t,
                })
            # Prune the list for efficiency (keep only live entries)
            cutoff_prune = t - T_EXP
            completed_run_entries = [(tc, rl) for tc, rl in completed_run_entries if tc > cutoff_prune]

        if last_logged_k is None or k_t != last_logged_k:
            n_live = len(completed_run_entries) if k_policy == K_POLICY_REGULATED else len(completed_run_lengths)
            k_trajectory.append({"t": t, "k": k_t,
                                  "n_completed": n_live,
                                  "reason": "k_changed_or_first"})
            last_logged_k = k_t

        # Accumulate K by segment class
        seg_idx_t = t // SEGMENT_STEPS
        seg_cls = _segment_class(segment_types[seg_idx_t])
        k_by_seg_class[seg_cls].append(k_t)

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

            if label in ("OK", "NOISE") and t <= MAX_SCORABLE_T:
                pending.append({"t_eval": t, "label": label, "e_w": err_rate})

            # Update in-progress run counter
            if current_run_class is None:
                current_run_class = label
                current_run_len = 1
            elif label == current_run_class:
                current_run_len += 1
            else:
                # Run completed
                if k_policy == K_POLICY_REGULATED:
                    completed_run_entries.append((t, current_run_len))
                else:
                    completed_run_lengths.append(current_run_len)
                current_run_class = label
                current_run_len = 1
                # Recompute K after new completion
                k_t, _ = _get_k(t)
                k_values_seen.add(k_t)
                if k_t != last_logged_k:
                    n_live = (len(completed_run_entries) if k_policy == K_POLICY_REGULATED
                              else len(completed_run_lengths))
                    k_trajectory.append({"t": t, "k": k_t,
                                         "n_completed": n_live,
                                         "reason": "run_completed"})
                    last_logged_k = k_t

            label_run.append(label)

            # LOCK test using k_t
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

    change_by_seg_type: dict[str, int] = {
        "SLOW600": 0, "SLOW1400": 0, "CTRL": 0, "NOISE": 0, "PLACE": 0
    }
    for seg_idx in change_seg_idxs:
        seg_type = segment_types[seg_idx]
        change_by_seg_type[seg_type] = change_by_seg_type.get(seg_type, 0) + 1

    descent_by_seg_type: dict[str, int] = {
        "SLOW600": 0, "SLOW1400": 0, "CTRL": 0, "NOISE": 0, "PLACE": 0
    }
    for seg_idx in descent_seg_idxs:
        seg_type = segment_types[seg_idx]
        descent_by_seg_type[seg_type] = descent_by_seg_type.get(seg_type, 0) + 1

    # Mean K by segment class
    mean_k_by_class: dict[str, float] = {}
    for cls, vals in k_by_seg_class.items():
        mean_k_by_class[cls] = float(np.mean(vals)) if vals else float("nan")

    # Final K
    k_t_final, _ = _get_k(n_total - 1)

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
        "final_k": k_t_final,
        # K-derived diagnostics
        "k_trajectory": k_trajectory,
        "completed_run_entries": completed_run_entries if k_policy == K_POLICY_REGULATED else [],
        "completed_run_lengths": completed_run_lengths if k_policy != K_POLICY_REGULATED else [],
        "k_at_lock_events": k_at_lock_events,
        "k_at_dial_changes": k_at_dial_changes,
        "k_values_seen_sorted": sorted(k_values_seen),
        "expiry_events": expiry_events,
        "mean_k_by_class": mean_k_by_class,
        "k_by_seg_class": k_by_seg_class,
    }


# ---------------------------------------------------------------------------
# Post-hoc: evaluate all four agents for one fork
# ---------------------------------------------------------------------------

def compute_fork_scores(run_result: dict) -> dict:
    """Compute per-segment correctness for agents (a), (c8), (c16), (bR).

    All four readouts operate on the SAME correct_arr from run_result.
    """
    correct_arr = run_result["correct_arr"]
    segment_types = run_result["segment_types"]

    # Agent (a): fixed W=200
    a_result = eval_fixed_dial_segments(correct_arr, W=200, segment_types=segment_types)

    # Agent (c8): constant K=8
    c8_result = _run_n3_controller(correct_arr, segment_types, k_policy=K_POLICY_CONSTANT_8)

    # Agent (c16): constant K=16
    c16_result = _run_n3_controller(correct_arr, segment_types, k_policy=K_POLICY_CONSTANT_16)

    # Agent (bR): time-regulated K
    bR_result = _run_n3_controller(correct_arr, segment_types, k_policy=K_POLICY_REGULATED)

    def _pack(prefix: str, r: dict) -> dict:
        return {
            f"{prefix}_seg_correctness": r["seg_correctness"],
            f"{prefix}_seg_eval_records": r["seg_eval_records"],
            f"{prefix}_eval_records": r["eval_records"],
            f"{prefix}_dial_trajectory": r["dial_trajectory"],
            f"{prefix}_lock_events": r["lock_events"],
            f"{prefix}_descent_events": r["descent_events"],
            f"{prefix}_change_by_seg_type": r["change_by_seg_type"],
            f"{prefix}_change_seg_idxs": r["change_seg_idxs"],
            f"{prefix}_total_dial_changes": r["total_dial_changes"],
            f"{prefix}_descent_by_seg_type": r["descent_by_seg_type"],
            f"{prefix}_total_descent_changes": r["total_descent_changes"],
            f"{prefix}_wrap_count": r["wrap_count"],
            f"{prefix}_final_dial": r["final_dial"],
            f"{prefix}_final_locked": r["final_locked"],
            f"{prefix}_final_lock_class": r["final_lock_class"],
            f"{prefix}_final_k": r["final_k"],
            f"{prefix}_k_trajectory": r["k_trajectory"],
            f"{prefix}_k_at_lock_events": r["k_at_lock_events"],
            f"{prefix}_k_at_dial_changes": r["k_at_dial_changes"],
            f"{prefix}_k_values_seen_sorted": r["k_values_seen_sorted"],
            f"{prefix}_mean_k_by_class": r["mean_k_by_class"],
            f"{prefix}_k_by_seg_class": r["k_by_seg_class"],
        }

    out = {
        "segment_types": segment_types,
        # Agent (a)
        "a_seg_correctness": a_result["seg_correctness"],
        "a_seg_eval_records": a_result["seg_eval_records"],
    }
    out.update(_pack("c8", c8_result))
    out.update(_pack("c16", c16_result))
    out.update(_pack("bR", bR_result))
    # Extra bR-specific
    out["bR_completed_run_entries"] = bR_result["completed_run_entries"]
    out["bR_expiry_events"] = bR_result["expiry_events"]

    return out


# ---------------------------------------------------------------------------
# Pool segment scores by class
# ---------------------------------------------------------------------------

def _pool_seg_scores_by_class(
    run_results: list[dict],
    fork_scores: dict,
    seeds: list[int],
    seg_class: str,
) -> tuple[float, float, float, float]:
    """Pool per-segment correctness for agents (a), (c8), (c16), (bR) by segment class.

    seg_class: "broken600", "broken1400", "valid" (matches SLOW600, SLOW1400,
    or CTRL/NOISE/PLACE respectively).

    Returns (pa, pc8, pc16, pbR).
    """
    def _seg_matches(seg_type: str) -> bool:
        return _segment_class(seg_type) == seg_class

    a_vals: list[float] = []
    c8_vals: list[float] = []
    c16_vals: list[float] = []
    bR_vals: list[float] = []

    for seed in seeds:
        rr = next(r for r in run_results if r["fork_seed"] == seed)
        segment_types = rr["segment_types"]
        fs = fork_scores[seed]
        for seg_idx, seg_type in enumerate(segment_types):
            if not _seg_matches(seg_type):
                continue
            a_c = fs["a_seg_correctness"].get(seg_idx, float("nan"))
            c8_c = fs["c8_seg_correctness"].get(seg_idx, float("nan"))
            c16_c = fs["c16_seg_correctness"].get(seg_idx, float("nan"))
            bR_c = fs["bR_seg_correctness"].get(seg_idx, float("nan"))
            if not math.isnan(a_c):
                a_vals.append(a_c)
            if not math.isnan(c8_c):
                c8_vals.append(c8_c)
            if not math.isnan(c16_c):
                c16_vals.append(c16_c)
            if not math.isnan(bR_c):
                bR_vals.append(bR_c)

    pa = float(np.mean(a_vals)) if a_vals else float("nan")
    pc8 = float(np.mean(c8_vals)) if c8_vals else float("nan")
    pc16 = float(np.mean(c16_vals)) if c16_vals else float("nan")
    pbR = float(np.mean(bR_vals)) if bR_vals else float("nan")
    return pa, pc8, pc16, pbR


def _combined(broken600: float, broken1400: float, valid: float) -> float:
    """combined = min(broken600, broken1400, valid)."""
    vals = [v for v in (broken600, broken1400, valid) if not math.isnan(v)]
    if not vals:
        return float("nan")
    return min(vals)


# ---------------------------------------------------------------------------
# Precondition checks — adapted from exp172
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

    # PC3: pooled deranged-phase error for both SLOW classes
    slow_deranged_errors: list[float] = []
    for rr in run_results:
        context = rr["context_arr"]
        correct = rr["correct_arr"]
        segment_types = rr["segment_types"]
        for seg_idx, seg_type in enumerate(segment_types):
            if seg_type not in ("SLOW600", "SLOW1400"):
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

def evaluate(
    run_results: list[dict],
    fork_scores: dict,
    seeds: list[int],
) -> dict:
    """Evaluate P_reg/F_reg, P_resp/F_resp, F_HARM, P_K/F_K."""

    # --- Pool by class ---
    pa_b600, pc8_b600, pc16_b600, pbR_b600 = _pool_seg_scores_by_class(
        run_results, fork_scores, seeds, "broken600")
    pa_b1400, pc8_b1400, pc16_b1400, pbR_b1400 = _pool_seg_scores_by_class(
        run_results, fork_scores, seeds, "broken1400")
    pa_valid, pc8_valid, pc16_valid, pbR_valid = _pool_seg_scores_by_class(
        run_results, fork_scores, seeds, "valid")

    # combined per arm
    combined_a = _combined(pa_b600, pa_b1400, pa_valid)
    combined_c8 = _combined(pc8_b600, pc8_b1400, pc8_valid)
    combined_c16 = _combined(pc16_b600, pc16_b1400, pc16_valid)
    combined_bR = _combined(pbR_b600, pbR_b1400, pbR_valid)

    max_constant_combined = (
        max(combined_c8, combined_c16)
        if not (math.isnan(combined_c8) or math.isnan(combined_c16))
        else (combined_c8 if not math.isnan(combined_c8) else combined_c16)
    )

    # P_reg: pooled combined(bR) - max(c8, c16) >= 0.10
    #        AND combined(bR) > both constants in >= 6/8 forks
    p_reg_margin = (combined_bR - max_constant_combined
                    if not (math.isnan(combined_bR) or math.isnan(max_constant_combined))
                    else float("nan"))

    # Per-fork: bR combined > both constants
    p_reg_forks_pass: list[bool] = []
    for seed in seeds:
        rr = next(r for r in run_results if r["fork_seed"] == seed)
        segment_types = rr["segment_types"]
        fs = fork_scores[seed]

        def _fork_combined(prefix: str) -> float:
            b600_vals, b1400_vals, valid_vals = [], [], []
            for seg_idx, seg_type in enumerate(segment_types):
                cls = _segment_class(seg_type)
                sc = fs[f"{prefix}_seg_correctness"].get(seg_idx, float("nan"))
                if math.isnan(sc):
                    continue
                if cls == "broken600":
                    b600_vals.append(sc)
                elif cls == "broken1400":
                    b1400_vals.append(sc)
                else:
                    valid_vals.append(sc)
            b600 = float(np.mean(b600_vals)) if b600_vals else float("nan")
            b1400 = float(np.mean(b1400_vals)) if b1400_vals else float("nan")
            valid = float(np.mean(valid_vals)) if valid_vals else float("nan")
            return _combined(b600, b1400, valid)

        fork_bR = _fork_combined("bR")
        fork_c8 = _fork_combined("c8")
        fork_c16 = _fork_combined("c16")
        if math.isnan(fork_bR) or math.isnan(fork_c8) or math.isnan(fork_c16):
            p_reg_forks_pass.append(False)
        else:
            p_reg_forks_pass.append(fork_bR > fork_c8 and fork_bR > fork_c16)

    n_p_reg_forks = sum(1 for x in p_reg_forks_pass if x)

    p_reg_margin_ok = not math.isnan(p_reg_margin) and p_reg_margin >= P_REG_MARGIN_MIN
    p_reg_forks_ok = n_p_reg_forks >= P_REG_FORKS_MIN
    p_reg = p_reg_margin_ok and p_reg_forks_ok

    # F_reg: some constant's combined within 0.05 of bR pooled
    f_reg = False
    if not math.isnan(combined_bR):
        if not math.isnan(combined_c8) and abs(combined_c8 - combined_bR) <= F_REG_WITHIN:
            f_reg = True
        if not math.isnan(combined_c16) and abs(combined_c16 - combined_bR) <= F_REG_WITHIN:
            f_reg = True

    # --- P_resp: bR responds (>= 1 ascent) in first broken segment >= 7/8 forks ---
    p_resp_forks_pass: list[bool] = []
    for seed in seeds:
        rr = next(r for r in run_results if r["fork_seed"] == seed)
        schedule = rr["schedule"]
        first_broken_seg = schedule["first_broken_segment_idx"]
        fs = fork_scores[seed]
        change_seg_idxs = fs["bR_change_seg_idxs"]
        responded = any(idx == first_broken_seg for idx in change_seg_idxs)
        p_resp_forks_pass.append(responded)

    n_p_resp_forks = sum(1 for x in p_resp_forks_pass if x)
    p_resp = n_p_resp_forks >= P_RESP_FORKS_MIN
    f_resp = n_p_resp_forks <= F_RESP_FORKS_MAX

    # --- F_HARM: bR valid deficit vs (a) > 0.15 ---
    valid_deficit = (pa_valid - pbR_valid
                     if not (math.isnan(pa_valid) or math.isnan(pbR_valid))
                     else float("nan"))
    f_harm = not math.isnan(valid_deficit) and valid_deficit > F_HARM_MARGIN

    # --- P_K: mean K during SLOW600 < mean K during SLOW1400 by >= 2 ---
    # Pool all per-eval K values by class from bR
    all_k_b600: list[float] = []
    all_k_b1400: list[float] = []
    for seed in seeds:
        fs = fork_scores[seed]
        k_by = fs["bR_k_by_seg_class"]
        all_k_b600.extend(k_by.get("broken600", []))
        all_k_b1400.extend(k_by.get("broken1400", []))

    mean_k_b600 = float(np.mean(all_k_b600)) if all_k_b600 else float("nan")
    mean_k_b1400 = float(np.mean(all_k_b1400)) if all_k_b1400 else float("nan")
    k_diff = (mean_k_b1400 - mean_k_b600
              if not (math.isnan(mean_k_b600) or math.isnan(mean_k_b1400))
              else float("nan"))

    p_k = not math.isnan(k_diff) and k_diff >= P_K_DIFF_MIN
    f_k = not math.isnan(k_diff) and k_diff < F_K_DIFF_MAX

    # --- Verdict ---
    positive = p_reg and p_resp and p_k and not f_harm
    negative = f_reg or f_resp or f_harm or f_k
    if positive:
        verdict_str = "POSITIVE"
    elif negative:
        verdict_str = "NEGATIVE"
    else:
        verdict_str = "MIXED"

    # --- P_K additional: per-class mean K for all arms ---
    all_k_bR_valid: list[float] = []
    for seed in seeds:
        fs = fork_scores[seed]
        all_k_bR_valid.extend(fs["bR_k_by_seg_class"].get("valid", []))
    mean_k_bR_valid = float(np.mean(all_k_bR_valid)) if all_k_bR_valid else float("nan")

    # Descent event collection
    all_descent_events_bR: list[dict] = []
    all_descent_events_c8: list[dict] = []
    all_descent_events_c16: list[dict] = []
    for seed in seeds:
        fs = fork_scores[seed]
        for de in fs["bR_descent_events"]:
            all_descent_events_bR.append({"seed": seed, **de})
        for de in fs["c8_descent_events"]:
            all_descent_events_c8.append({"seed": seed, **de})
        for de in fs["c16_descent_events"]:
            all_descent_events_c16.append({"seed": seed, **de})

    # Expiry event count
    total_expiry_events = sum(
        len(fork_scores[s]["bR_expiry_events"]) for s in seeds
    )

    return {
        "n_forks": len(seeds),
        # Pool scores
        "pa_b600": pa_b600, "pc8_b600": pc8_b600, "pc16_b600": pc16_b600, "pbR_b600": pbR_b600,
        "pa_b1400": pa_b1400, "pc8_b1400": pc8_b1400, "pc16_b1400": pc16_b1400, "pbR_b1400": pbR_b1400,
        "pa_valid": pa_valid, "pc8_valid": pc8_valid, "pc16_valid": pc16_valid, "pbR_valid": pbR_valid,
        # Combined
        "combined_a": combined_a,
        "combined_c8": combined_c8,
        "combined_c16": combined_c16,
        "combined_bR": combined_bR,
        "max_constant_combined": max_constant_combined,
        # P_reg
        "p_reg_margin": p_reg_margin,
        "p_reg_margin_ok": p_reg_margin_ok,
        "p_reg_forks_pass": p_reg_forks_pass,
        "n_p_reg_forks": n_p_reg_forks,
        "p_reg_forks_ok": p_reg_forks_ok,
        "p_reg": p_reg,
        "f_reg": f_reg,
        # P_resp
        "p_resp_forks_pass": p_resp_forks_pass,
        "n_p_resp_forks": n_p_resp_forks,
        "p_resp": p_resp,
        "f_resp": f_resp,
        # F_HARM
        "valid_deficit_bR_vs_a": valid_deficit,
        "f_harm": f_harm,
        # P_K
        "mean_k_b600": mean_k_b600,
        "mean_k_b1400": mean_k_b1400,
        "mean_k_bR_valid": mean_k_bR_valid,
        "k_diff": k_diff,
        "p_k": p_k,
        "f_k": f_k,
        # Verdict
        "verdict_str": verdict_str,
        # Diagnostics
        "all_descent_events_bR": all_descent_events_bR,
        "all_descent_events_c8": all_descent_events_c8,
        "all_descent_events_c16": all_descent_events_c16,
        "total_expiry_events": total_expiry_events,
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
                    "seg_class": _segment_class(segment_types[i]),
                    "a_correctness": _nan_to_none(fs["a_seg_correctness"].get(i)),
                    "c8_correctness": _nan_to_none(fs["c8_seg_correctness"].get(i)),
                    "c16_correctness": _nan_to_none(fs["c16_seg_correctness"].get(i)),
                    "bR_correctness": _nan_to_none(fs["bR_seg_correctness"].get(i)),
                    "n_eval_a": len(fs["a_seg_eval_records"].get(i, [])),
                    "n_eval_c8": len(fs["c8_seg_eval_records"].get(i, [])),
                    "n_eval_c16": len(fs["c16_seg_eval_records"].get(i, [])),
                    "n_eval_bR": len(fs["bR_seg_eval_records"].get(i, [])),
                })

            row: dict = {
                "exp": 173,
                "fork_seed": seed,
                "n_total": rr["n_total"],
                "ahat_drift": _nan_to_none(rr["ahat_drift"]),
                "segment_types": segment_types,
                "derangement": schedule["derangement"],
                "p_true_noise": _nan_to_none(schedule["p_true_noise"]),
                "noisy_cells_first5": schedule["noisy_cells"][:5],
                "first_broken_segment_idx": schedule["first_broken_segment_idx"],
                "segment_scores": seg_summary,
                # c8 stats
                "c8_total_dial_changes": fs["c8_total_dial_changes"],
                "c8_change_by_seg_type": fs["c8_change_by_seg_type"],
                "c8_total_descent_changes": fs["c8_total_descent_changes"],
                "c8_final_dial": fs["c8_final_dial"],
                # c16 stats
                "c16_total_dial_changes": fs["c16_total_dial_changes"],
                "c16_change_by_seg_type": fs["c16_change_by_seg_type"],
                "c16_total_descent_changes": fs["c16_total_descent_changes"],
                "c16_final_dial": fs["c16_final_dial"],
                # bR stats
                "bR_total_dial_changes": fs["bR_total_dial_changes"],
                "bR_change_by_seg_type": fs["bR_change_by_seg_type"],
                "bR_change_seg_idxs": fs["bR_change_seg_idxs"],
                "bR_total_descent_changes": fs["bR_total_descent_changes"],
                "bR_final_dial": fs["bR_final_dial"],
                "bR_final_locked": fs["bR_final_locked"],
                "bR_final_lock_class": fs["bR_final_lock_class"],
                "bR_final_k": fs["bR_final_k"],
                "bR_k_trajectory": fs["bR_k_trajectory"],
                "bR_k_at_lock_events": fs["bR_k_at_lock_events"],
                "bR_k_at_dial_changes": fs["bR_k_at_dial_changes"],
                "bR_k_values_seen_sorted": fs["bR_k_values_seen_sorted"],
                "bR_mean_k_by_class": fs["bR_mean_k_by_class"],
                "bR_n_expiry_events": len(fs["bR_expiry_events"]),
                "bR_dial_trajectory_summary": [
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
                    for d in fs["bR_dial_trajectory"]
                ],
                "bR_descent_events": [
                    {"t": de["t"], "from_W": de["from_W"], "to_W": de["to_W"],
                     "lock_class": de["lock_class"],
                     "seg_idx": de["seg_idx"], "seg_type": de["seg_type"],
                     "k_at_descent": de.get("k_at_descent")}
                    for de in fs["bR_descent_events"]
                ],
            }
            fh.write(json.dumps(row) + "\n")

        if ev is not None:
            def _fv(v):
                if isinstance(v, float) and math.isnan(v):
                    return None
                return v

            summary: dict = {
                "exp": 173,
                "row_type": "summary",
                # Pool scores
                "pa_b600": _fv(ev["pa_b600"]),
                "pc8_b600": _fv(ev["pc8_b600"]),
                "pc16_b600": _fv(ev["pc16_b600"]),
                "pbR_b600": _fv(ev["pbR_b600"]),
                "pa_b1400": _fv(ev["pa_b1400"]),
                "pc8_b1400": _fv(ev["pc8_b1400"]),
                "pc16_b1400": _fv(ev["pc16_b1400"]),
                "pbR_b1400": _fv(ev["pbR_b1400"]),
                "pa_valid": _fv(ev["pa_valid"]),
                "pc8_valid": _fv(ev["pc8_valid"]),
                "pc16_valid": _fv(ev["pc16_valid"]),
                "pbR_valid": _fv(ev["pbR_valid"]),
                # Combined
                "combined_a": _fv(ev["combined_a"]),
                "combined_c8": _fv(ev["combined_c8"]),
                "combined_c16": _fv(ev["combined_c16"]),
                "combined_bR": _fv(ev["combined_bR"]),
                # P_reg
                "p_reg_margin": _fv(ev["p_reg_margin"]),
                "n_p_reg_forks": ev["n_p_reg_forks"],
                "p_reg": ev["p_reg"],
                "f_reg": ev["f_reg"],
                # P_resp
                "n_p_resp_forks": ev["n_p_resp_forks"],
                "p_resp": ev["p_resp"],
                "f_resp": ev["f_resp"],
                # F_HARM
                "valid_deficit_bR_vs_a": _fv(ev["valid_deficit_bR_vs_a"]),
                "f_harm": ev["f_harm"],
                # P_K
                "mean_k_b600": _fv(ev["mean_k_b600"]),
                "mean_k_b1400": _fv(ev["mean_k_b1400"]),
                "mean_k_bR_valid": _fv(ev["mean_k_bR_valid"]),
                "k_diff": _fv(ev["k_diff"]),
                "p_k": ev["p_k"],
                "f_k": ev["f_k"],
                "verdict": ev["verdict_str"],
                "total_expiry_events": ev["total_expiry_events"],
            }
            fh.write(json.dumps(summary) + "\n")


# ---------------------------------------------------------------------------
# Print K trajectory by segment class for bR
# ---------------------------------------------------------------------------

def print_k_trajectory_by_class(
    run_results: list[dict],
    fork_scores: dict,
    seeds: list[int],
) -> None:
    """Print K(t) trajectory + mean K by segment class for agent (bR)."""
    print("bR K(t) trajectories with mean K by segment class (time-regulated):")
    for seed in seeds:
        rr = next(r for r in run_results if r["fork_seed"] == seed)
        fs = fork_scores[seed]
        segment_types = rr["segment_types"]

        ktraj = fs["bR_k_trajectory"]
        k_seen = fs["bR_k_values_seen_sorted"]
        k_final = fs["bR_final_k"]
        mean_k = fs["bR_mean_k_by_class"]
        n_expiry = len(fs["bR_expiry_events"])

        seg_str = "  ".join(f"[{i}:{t}]" for i, t in enumerate(segment_types))
        print(f"  seed={seed}  schedule: {seg_str}")
        print(f"    final_K={k_final}  k_seen={k_seen}  n_expiry_events={n_expiry}")
        print(f"    mean_K: broken600={mean_k.get('broken600', float('nan')):.2f}  "
              f"broken1400={mean_k.get('broken1400', float('nan')):.2f}  "
              f"valid={mean_k.get('valid', float('nan')):.2f}")

        if ktraj:
            traj_strs = [f"t={e['t']} K={e['k']}({e['reason'][:8]})" for e in ktraj[:12]]
            if len(ktraj) > 12:
                traj_strs.append(f"...({len(ktraj)} total)")
            print(f"    K trajectory: {'; '.join(traj_strs)}")
        print()
    print()


# ---------------------------------------------------------------------------
# Print four-arm per-segment correctness table
# ---------------------------------------------------------------------------

def print_four_arm_table(
    run_results: list[dict],
    fork_scores: dict,
    seeds: list[int],
    ev: dict | None = None,
) -> None:
    """Print four-arm per-segment-class correctness table."""
    def _f(v, w=7):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "nan".rjust(w)
        return f"{v:.4f}".rjust(w)

    print("Four-arm per-segment-class correctness (a=W200 / c8=K8 / c16=K16 / bR=regulated):")
    print(f"  {'class':>10}  {'a_pool':>8}  {'c8_pool':>8}  {'c16_pool':>8}  {'bR_pool':>8}"
          f"  {'bR-c8':>7}  {'bR-c16':>7}  {'gt_label':>10}")
    print("  " + "-" * 85)

    for seg_class, seg_display in [
        ("broken600", "SLOW600"), ("broken1400", "SLOW1400"),
        ("valid", "valid(CTRL/NOISE/PLACE)")
    ]:
        pa, pc8, pc16, pbR = _pool_seg_scores_by_class(
            run_results, fork_scores, seeds, seg_class)
        diff_c8 = pbR - pc8 if not (math.isnan(pc8) or math.isnan(pbR)) else float("nan")
        diff_c16 = pbR - pc16 if not (math.isnan(pc16) or math.isnan(pbR)) else float("nan")
        gt = "STRUCTURAL" if seg_class.startswith("broken") else "OK/NOISE"
        print(f"  {seg_class:>10}  {_f(pa)}  {_f(pc8)}  {_f(pc16)}  {_f(pbR)}"
              f"  {_f(diff_c8)}  {_f(diff_c16)}  {gt:>10}")

    if ev is not None:
        print()
        print(f"  combined(a)  = {_f(ev['combined_a'])}")
        print(f"  combined(c8) = {_f(ev['combined_c8'])}")
        print(f"  combined(c16)= {_f(ev['combined_c16'])}")
        print(f"  combined(bR) = {_f(ev['combined_bR'])}")
        print(f"  bR - max(c8,c16) = {_f(ev['p_reg_margin'])}  "
              f"(need >= {P_REG_MARGIN_MIN} for P_reg)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exp 173 — N3 multi-tempo regulated horizon (TIER-3 GATE). "
            "Four same-schedule arms: (a) W=200, (c8) K=8, (c16) K=16, (bR) time-regulated K."
        )
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Smoke run: seed=[146] only, full 24000 steps, prints schedule, "
            "K trajectory with mean-K-by-class, four-arm per-segment table. "
            "No verdict."
        ),
    )
    args = parser.parse_args()

    smoke = args.smoke
    seeds = [146] if smoke else SEEDS

    print("=" * 80)
    print("Exp 173 — N3 MULTI-TEMPO REGULATED HORIZON (TIER-3 GATE)")
    print("          K(t) = clamp(max(completed runs within last T_exp steps)+1, 3, 16)")
    print("          Four arms: (a) W=200, (c8) K=8, (c16) K=16, (bR) time-regulated;")
    print("          same schedule/stream.")
    print("=" * 80)
    print()
    print(f"SURPRISE_WINDOW={SURPRISE_WINDOW}  [must be 200; asserted]")
    assert SURPRISE_WINDOW == 200, f"SURPRISE_WINDOW={SURPRISE_WINDOW}, expected 200"
    print(f"CLASSIFIER_WINDOW_DEFAULT={CLASSIFIER_WINDOW_DEFAULT}  EVAL_EVERY={EVAL_EVERY}")
    print(f"ERROR_RATE_OK_THRESH={ERROR_RATE_OK_THRESH}  LAG1_RHO_STRUCT_THRESH={LAG1_RHO_STRUCT_THRESH}")
    print(f"DIAL_CANDIDATES={DIAL_CANDIDATES}  starting W={CLASSIFIER_WINDOW_DEFAULT}")
    print(f"TIME-REGULATED K (bR): T_exp={T_EXP}  K_MIN={K_MIN}  K_MAX={K_MAX}")
    print(f"  K(t) = clamp(max(completed runs within last {T_EXP} steps)+1, {K_MIN}, {K_MAX})")
    print(f"  with no live completions: K(t) = {K_MIN}")
    print(f"  T_exp = segment length = {SEGMENT_STEPS} steps (declared policy)")
    print(f"AGENTS: (a) W=200 | (c8) K=8 | (c16) K=16 | (bR) time-regulated")
    print(f"DELTA B (verbatim exp169) — NO WRAP: ascent saturates at 1600")
    print(f"DELTA A (verbatim exp169) — DESCENT: locked on OK/NOISE AND W>200 => step down")
    print(f"N3 trust: TRUST_FIRE_THRESH={TRUST_FIRE_THRESH}  ROLLING_TRUST_WINDOW={ROLLING_TRUST_WINDOW}  "
          f"MIN_SCORED_FOR_TRUST={MIN_SCORED_FOR_TRUST}")
    print(f"N3 forecast: OK_VIOLATION_THRESH={OK_VIOLATION_THRESH}  "
          f"NOISE_VIOLATION_DELTA={NOISE_VIOLATION_DELTA}  STRUCTURAL=NOT scored")
    print(f"MULTI-TEMPO: SLOW600 half_period={HALF_PERIOD_SLOW600}  SLOW1400 half_period={HALF_PERIOD_SLOW1400}")
    print(f"N_SEGMENTS={N_SEGMENTS}  SEGMENT_STEPS={SEGMENT_STEPS}  N_STEPS={N_STEPS}")
    print(f"Segment pool: {SEGMENT_POOL_TEMPLATE}")
    print(f"SCHEDULE_SEED_OFFSET={SCHEDULE_SEED_OFFSET}  PLACE_NOISE_SEED_OFFSET={PLACE_NOISE_SEED_OFFSET}")
    print(f"PC1_AHAT_DRIFT_MAX={PC1_AHAT_DRIFT_MAX}")
    print(f"PC2_MIN_EVAL_POINTS={PC2_MIN_EVAL_POINTS}  PC3_SLOW_DERANGED_ERR_MIN={PC3_SLOW_DERANGED_ERR_MIN}")
    print(f"P_reg: margin >= {P_REG_MARGIN_MIN} AND bR>both in >= {P_REG_FORKS_MIN}/8 forks")
    print(f"P_resp: >= {P_RESP_FORKS_MIN}/8 forks  P_K: k_diff >= {P_K_DIFF_MIN}  "
          f"F_HARM: deficit > {F_HARM_MARGIN}")
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
            f"first_broken_seg={schedule['first_broken_segment_idx']}",
            flush=True,
        )
        print(f"    Running seed={seed} (24000 steps, 8 segments x 3000) ...", flush=True)
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
              f"c8: ascent={fs['c8_total_dial_changes']} desc={fs['c8_total_descent_changes']}  "
              f"c16: ascent={fs['c16_total_dial_changes']} desc={fs['c16_total_descent_changes']}  "
              f"bR: ascent={fs['bR_total_dial_changes']} desc={fs['bR_total_descent_changes']}  "
              f"bR_final_K={fs['bR_final_k']}  "
              f"bR_k_seen={fs['bR_k_values_seen_sorted']}  "
              f"bR_mean_K={fs['bR_mean_k_by_class']}",
              flush=True)
    print()

    # --- Print K trajectories for bR with mean K by class ---
    print_k_trajectory_by_class(run_results, fork_scores, seeds)

    # --- Print four-arm per-segment table ---
    print_four_arm_table(run_results, fork_scores, seeds, ev=None)

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
        out_rows_path = Path(__file__).parent / "outputs" / "exp173_rows.json"
        write_json_rows(run_results, fork_scores, ev=None, path=out_rows_path)
        print(f"JSON rows written to {out_rows_path} (no verdict)")
        return

    print("Preconditions: all PASS")
    print()

    # --- Evaluate predeclared outcome map ---
    ev = evaluate(run_results, fork_scores, seeds)
    n = ev["n_forks"]

    # --- Print score table with combined ---
    print_four_arm_table(run_results, fork_scores, seeds, ev=ev)

    def _fv(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "nan"
        return f"{v:.4f}"

    print("=" * 80)
    print("PREDECLARED OUTCOME MAP")
    print("=" * 80)
    print()

    print(f"P_reg (THE TIER-3 GATE — pooled combined(bR) - max(c8,c16) >= {P_REG_MARGIN_MIN}):")
    print(f"  combined(a)={_fv(ev['combined_a'])}  combined(c8)={_fv(ev['combined_c8'])}"
          f"  combined(c16)={_fv(ev['combined_c16'])}  combined(bR)={_fv(ev['combined_bR'])}")
    print(f"  max(c8,c16)={_fv(ev['max_constant_combined'])}  "
          f"bR - max(c8,c16) = {_fv(ev['p_reg_margin'])}  (need >= {P_REG_MARGIN_MIN})")
    print(f"  bR > both constants per fork: {sum(1 for x in ev['p_reg_forks_pass'] if x)}/{n}  "
          f"(need >= {P_REG_FORKS_MIN})")
    for i, seed in enumerate(seeds):
        print(f"    seed={seed}: bR>both={ev['p_reg_forks_pass'][i]}")
    print(f"  FALSIFIER F_reg (constant within {F_REG_WITHIN} of bR pooled): "
          f"{'YES — F_reg FIRES' if ev['f_reg'] else 'no'}")
    p_reg_status = "PASS" if ev["p_reg"] else ("FALSIFIER F_reg" if ev["f_reg"] else "FAIL (soft)")
    print(f"  => P_reg: {p_reg_status}")
    print()

    print(f"P_resp (bR responds on first broken seg: >= {P_RESP_FORKS_MIN}/{n} forks):")
    for i, seed in enumerate(seeds):
        print(f"  seed={seed}: responded={ev['p_resp_forks_pass'][i]}")
    print(f"  Forks responding: {ev['n_p_resp_forks']}/{n}  (need >= {P_RESP_FORKS_MIN})")
    print(f"  FALSIFIER F_resp (<=  {F_RESP_FORKS_MAX}/{n}): "
          f"{'YES — F_resp FIRES' if ev['f_resp'] else 'no'}")
    p_resp_status = "PASS" if ev["p_resp"] else ("FALSIFIER F_resp" if ev["f_resp"] else "FAIL (soft)")
    print(f"  => P_resp: {p_resp_status}")
    print()

    print(f"F_HARM (bR valid deficit vs a > {F_HARM_MARGIN} — GATES VERDICT):")
    print(f"  pooled_valid_a={_fv(ev['pa_valid'])}  pooled_valid_bR={_fv(ev['pbR_valid'])}")
    print(f"  valid_deficit(a-bR)={_fv(ev['valid_deficit_bR_vs_a'])}  "
          f"(fires if > {F_HARM_MARGIN})")
    print(f"  F_HARM: {'YES — F_HARM FIRES' if ev['f_harm'] else 'silent'}")
    print()

    print(f"P_K (K tracks tempo: mean_K_SLOW1400 - mean_K_SLOW600 >= {P_K_DIFF_MIN}):")
    print(f"  mean_K_SLOW600={_fv(ev['mean_k_b600'])}  "
          f"mean_K_SLOW1400={_fv(ev['mean_k_b1400'])}  "
          f"mean_K_valid={_fv(ev['mean_k_bR_valid'])}")
    print(f"  k_diff(1400-600)={_fv(ev['k_diff'])}  (need >= {P_K_DIFF_MIN})")
    print(f"  FALSIFIER F_K (diff < {F_K_DIFF_MAX}): "
          f"{'YES — F_K FIRES' if ev['f_k'] else 'no'}")
    p_k_status = "PASS" if ev["p_k"] else ("FALSIFIER F_K" if ev["f_k"] else "FAIL (soft)")
    print(f"  => P_K: {p_k_status}")
    print()

    # --- Ungated diagnostics ---
    print(f"Ungated diagnostics:")
    print(f"  Total expiry events (bR): {ev['total_expiry_events']}")
    print(f"  Descent events bR: {len(ev['all_descent_events_bR'])}")
    print(f"  Descent events c8: {len(ev['all_descent_events_c8'])}")
    print(f"  Descent events c16: {len(ev['all_descent_events_c16'])}")
    print()

    # --- Conjunct summary + VERDICT ---
    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    print(f"P_reg (tier-3 gate: bR combined >= max(c8,c16) + {P_REG_MARGIN_MIN}): {p_reg_status}")
    print(f"P_resp (bR responsiveness >= {P_RESP_FORKS_MIN}/{n} forks): {p_resp_status}")
    print(f"F_HARM (bR valid deficit > {F_HARM_MARGIN} — GATES): "
          f"{'FIRED' if ev['f_harm'] else 'silent'}")
    print(f"P_K (K tracks tempo by >= {P_K_DIFF_MIN}): {p_k_status}")
    print()
    if ev["verdict_str"] == "POSITIVE":
        print("VERDICT: POSITIVE")
        print("TIER-3 CLAIM SUPPORTED — N3 usefully regulates its own evidence horizon")
        print("where the gap moves. Time-regulated K outperforms both constant horizons.")
    elif ev["verdict_str"] == "NEGATIVE":
        print("VERDICT: NEGATIVE")
        print("A falsifier fired — halt for a word.")
    else:
        print("VERDICT: MIXED")
    print()
    print("=" * 80)

    # --- Write JSON rows ---
    out_rows_path = Path(__file__).parent / "outputs" / "exp173_rows.json"
    write_json_rows(run_results, fork_scores, ev=ev, path=out_rows_path)
    print(f"JSON rows written to {out_rows_path}")

    # --- Write verdict JSON ---
    def _fv2(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "nan"
        return f"{v:.4f}"

    arms = {
        "P_reg_tier3_gate": {
            "pass": bool(ev["p_reg"]),
            "reason": (
                f"combined(bR)={_fv2(ev['combined_bR'])}  "
                f"max(c8,c16)={_fv2(ev['max_constant_combined'])}  "
                f"margin={_fv2(ev['p_reg_margin'])} (need >= {P_REG_MARGIN_MIN}); "
                f"bR>both in {ev['n_p_reg_forks']}/{ev['n_forks']} forks "
                f"(need >= {P_REG_FORKS_MIN}); F_reg fired={ev['f_reg']}"
            ),
        },
        "P_resp_responsiveness": {
            "pass": bool(ev["p_resp"]),
            "reason": (
                f"{ev['n_p_resp_forks']}/{ev['n_forks']} forks responded on first broken segment "
                f"(need >= {P_RESP_FORKS_MIN}); F_resp fired={ev['f_resp']}"
            ),
        },
        "F_HARM_valid_deficit": {
            "pass": not bool(ev["f_harm"]),
            "reason": (
                f"valid_deficit(a-bR)={_fv2(ev['valid_deficit_bR_vs_a'])} "
                f"(fires if > {F_HARM_MARGIN}); F_HARM={'FIRED' if ev['f_harm'] else 'silent'}"
            ),
        },
        "P_K_tempo_tracking": {
            "pass": bool(ev["p_k"]),
            "reason": (
                f"mean_K_SLOW600={_fv2(ev['mean_k_b600'])}  "
                f"mean_K_SLOW1400={_fv2(ev['mean_k_b1400'])}  "
                f"k_diff={_fv2(ev['k_diff'])} (need >= {P_K_DIFF_MIN}); "
                f"F_K fired={ev['f_k']}"
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp173_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp173",
        arms=arms,
        verdict=ev["verdict_str"],
        halted=False,
        notes=(
            "N3 multi-tempo regulated horizon (TIER-3 GATE). "
            f"K(t)=clamp(max(completed within last {T_EXP} steps)+1,{K_MIN},{K_MAX}). "
            "Four arms on same schedule/stream: (a) W=200, (c8) K=8, (c16) K=16, (bR) regulated. "
            "8 segments x 3000 steps = 24000 total; two SLOW tempos (600, 1400). "
            f"POSITIVE: TIER-3 CLAIM SUPPORTED. NEGATIVE: falsifier fired. MIXED: partial. "
            f"Seeds 146-153 (exp172 used 138-145). "
            f"combined(bR)={_fv2(ev['combined_bR'])} combined(c8)={_fv2(ev['combined_c8'])} "
            f"combined(c16)={_fv2(ev['combined_c16'])} k_diff={_fv2(ev['k_diff'])}."
        ),
    )
    print(f"Verdict JSON written to {verdict_path}")


if __name__ == "__main__":
    main()
