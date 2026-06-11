"""Exp 168 — N3 rung 4 MIXED SCHEDULE: independent variance / override concentration.

RUNG 4 CONTEXT — The meta-calibration-n3 card's FINAL rung.
Rungs 1–3 established: (1) a world where N2 is systematically wrong; (2) N3's
trust monitor detects that mismatch; (3) N3's lock controller corrects it without
harming valid regimes.  Rung 4 asks: does N3 have INDEPENDENT VARIANCE?  Over a
MIXED schedule of valid and broken regimes, do N3's overrides (dial changes) land
in broken segments — or does N3 fire indiscriminately (epiphenomenal noise), or
never fire at all (collapsed into N2)?  Passing grades the card's entire central
hypothesis as SUPPORTED.

READ-ONLY-DIAL DECLARATION:
  One creature run per fork; per-step correctness stream recorded; agent (b)'s
  dial policy is a post-hoc readout over that SAME stream.  The creature is never
  re-run or influenced by the dial policy.

SEEDS = list(range(106, 114)) — 8 fresh forks; exp167 used 98-105.

MIXED SESSION — one per fork:
  24000 steps = 6 contiguous segments of 4000 steps each.
  Per-fork rng 140_000+seed draws the segment ORDER:
    random shuffle of [SLOW, SLOW, SLOW, VALID1, VALID2, VALID3]
    where VALID1/VALID2/VALID3 are CTRL/NOISE/PLACE in shuffled assignment.
    NOISE p_true drawn from {0.6, 0.7, 0.8} (uniform, logged).
    PLACE = random 13-of-25 noisy cells (p=0.55, noise rng 90_000+seed, logged).
    SLOW = StructCmap half_period 1000, derangement FORCED by seed parity:
           [1,2,0] if seed even, [2,0,1] if seed odd.
           (Fixing Exp 167's rng accident where all forks drew one variant.)
  One CONTINUOUS creature run across all 6 segments; the controller has NO
  segment knowledge.  StructCmap current_step within a SLOW segment uses the
  step offset WITHIN that segment (each SLOW segment starts in context A).

AGENTS (post-hoc readouts; read-only-dial):
  (a) N2-only:  fixed W=200 baseline.
  (b) N2+N3 LOCK controller (Exp 167 verbatim): starts W=200; scores only
      OK and NOISE forecasts (STRUCTURAL not scored); LOCK disables trigger
      when last K=8 labels under current dial are all same class; advances
      200->400->800->1600->200 cyclically.

SCORING per segment:
  Eval points every 100 steps; a window evaluates if enough history under the
  agent's current dial (for agent b) or fixed W=200 (for agent a).
  Ground-truth label per segment type:
    SLOW  -> STRUCTURAL (correct label)
    CTRL  -> OK         (correct label)
    NOISE -> NOISE      (correct label)
    PLACE -> NOISE      (correct label; Exp 158 taxonomy confirmed)
  Segment correctness = fraction of that segment's evaluated windows bearing
  the correct label.
  Valid-segment pooled correctness = mean over all CTRL/NOISE/PLACE segments
  and forks.
  Broken-segment pooled correctness = mean over all SLOW segments and forks.
  Dial changes of agent (b) are attributed to the segment type they occur in.

PRECONDITIONS:
  PC1 ahat_drift < 0.30 (horizon-scaled: 6× the 4000-step 0.05 bound;
      Exp 166/167 measured ~0.05-0.06 at 12000, so 0.30 gives headroom
      at 24000 steps).
  PC2 >= 30 eval points per segment at W=200 (agent a per-segment check).
  PC3 SLOW deranged-phase per-step error >= 0.9 pooled.

PREDECLARED PROPERTIES AND FALSIFIERS:
  P1 (override concentration):
    Pooled fraction of (b)'s dial changes occurring within SLOW segments >= 0.75.
    Time-share chance = 0.5 (3 SLOW / 6 total segments).
    FALSIFIER F1: fraction <= 0.5.
  P2 (responsiveness):
    In >= 7/8 forks, at least one dial change occurs within the FIRST SLOW
    segment the fork encounters.
    FALSIFIER F2: <= 4/8 forks.
  P3 (no harm in valid regimes):
    Pooled valid-segment correctness of (b) >= that of (a) - 0.05.
    FALSIFIER F3: (a) - (b) > 0.15 (indiscriminate overriding hurts).
  P4 (overrides earn their keep):
    Pooled SLOW-segment correctness of (b) >= that of (a) + 0.20.
    FALSIFIER F4: (b) <= (a) + 0.05 (zero effective independent variance —
    epiphenomenal).

  VERDICT:
    POSITIVE (RUNG 4 PASSES — with rungs 1-3, the card's central hypothesis
             is SUPPORTED) iff P1, P2, P3, P4 all pass.
    NEGATIVE iff any falsifier fires.
    Otherwise MIXED.
    "Not a falsifier" never counts toward POSITIVE.

UNGATED DIAGNOSTICS:
  Full dial trajectory + lock events per fork with segment boundaries annotated.
  Per-segment correctness table (a) vs (b).
  Change counts per segment type.
  Descent question: does the dial ever come DOWN after a SLOW segment ends
    (cycle wraps 1600->200)?  Does staying high hurt valid segments (visible
    in P3)?
  Derangement variants confirmed to span both options.

--smoke: seed [106] only, full 24000 steps, prints segment schedule, dial
  trajectory, per-segment scores for both agents, lock states.
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

SEEDS = list(range(106, 114))   # fresh seeds — exp167 used 98-105

N_SEGMENTS = 6
SEGMENT_STEPS = 4000
N_STEPS = N_SEGMENTS * SEGMENT_STEPS   # 24000
N_CHUNKS = N_STEPS // 100              # 240  (chunk_size=100)
CHUNK_SIZE = 100

# Per-fork rng offset for the segment schedule draw
SCHEDULE_SEED_OFFSET = 140_000         # rng 140_000 + fork_seed

# Noise rng offset for the PLACE world (verbatim from exp163)
PLACE_NOISE_SEED_OFFSET = 90_000       # rng 90_000 + fork_seed

# Classifier parameters (verbatim from exp158/159/162/163/165/166/167)
CLASSIFIER_WINDOW_DEFAULT = 200        # starting W for agent (b) and agent (a)
EVAL_EVERY = 100                       # evaluate every 100 steps once window full
ERROR_RATE_OK_THRESH = 0.05            # error_rate < 0.05 => OK
LAG1_RHO_STRUCT_THRESH = 0.3          # lag1_rho > 0.3 => STRUCTURAL (else NOISE)

# Dial candidate set (ordered for cyclic advance; verbatim from exp165/166/167)
DIAL_CANDIDATES = [200, 400, 800, 1600]

# EWMA channel parameters (verbatim from exp157/159/163/165/166/167)
ALPHA = 0.05        # EWMA learning rate
EWMA_INIT = 0.5     # initial EWMA value per cell

# N3 forecast-scoring thresholds (verbatim from exp165/167 — STRUCTURAL not scored)
OK_VIOLATION_THRESH = 0.15             # L=OK violated iff e_next >= 0.15
NOISE_VIOLATION_DELTA = 0.30           # L=NOISE violated iff |e_next - e_w| >= 0.30
# STRUCTURAL: not scored

ROLLING_TRUST_WINDOW = 10              # last 10 scored labels
TRUST_FIRE_THRESH = 0.85               # advance dial when trust < 0.85 (>= 2 violations)
MIN_SCORED_FOR_TRUST = 3               # < 3 scored since last dial change => NO-EVIDENCE

# LOCK parameters (verbatim from exp167)
LOCK_K = 8   # lock engages when last K labels under current dial are all same class

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

# Precondition thresholds
# PC1: horizon-scaled — 6× the 4000-step 0.05 bound (exp166/167 measured ~0.05-0.06 at 12000)
PC1_AHAT_DRIFT_MAX = 0.30             # PC1 (24000-step horizon-scaled bound)
PC2_MIN_EVAL_POINTS = 30              # PC2: >= 30 eval points per segment at W=200
PC3_SLOW_DERANGED_ERR_MIN = 0.9       # PC3: SLOW deranged-phase error >= 0.9 pooled

# P1 thresholds (override concentration)
P1_CONC_MIN = 0.75                    # fraction of dial changes in SLOW >= 0.75
F1_CONC_MAX = 0.50                    # F1: fraction <= 0.50

# P2 thresholds (responsiveness)
P2_FORKS_MIN = 7                      # >= 7/8 forks respond on first SLOW segment
F2_FORKS_MAX = 4                      # F2: <= 4/8 forks

# P3 thresholds (no harm in valid regimes)
P3_NO_HARM_MARGIN = 0.05              # (b) >= (a) - 0.05 (pooled)
F3_HARM_MARGIN = 0.15                 # F3: (a) - (b) > 0.15

# P4 thresholds (overrides earn their keep)
P4_IMPROVE_MIN = 0.20                 # (b) >= (a) + 0.20 (pooled)
F4_IMPROVE_MAX = 0.05                 # F4: (b) <= (a) + 0.05


# ---------------------------------------------------------------------------
# StructCmap — verbatim from exp162/163/164/165/166/167
# ---------------------------------------------------------------------------

class StructCmap:
    """Hidden-context cmap: alternates every half_period steps.

    current_step is set by caller BEFORE __getitem__ is called.
    """

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
# NoisyCmap — verbatim from exp155/163
# ---------------------------------------------------------------------------

class NoisyCmap:
    """Color map wrapper with irreducible observation noise.

    Each lookup returns the true color with probability p_true, otherwise a
    uniform draw among the other colors.  Uses its OWN seeded rng so the
    creature's action stream is untouched.
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
# RandomPlaceNoiseCmap — verbatim from exp159/163
# ---------------------------------------------------------------------------

class RandomPlaceNoiseCmap:
    """Color map wrapper with per-fork randomly-placed noisy cells.

    noisy_cells: set of cell indices that are noisy (p_true < 1).
    Clean cells always return the true color.
    """

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
# lag1_rho — verbatim from exp162/163/164/165/166/167
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
# classify_window — verbatim from exp162/163/164/165/166/167
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
# Draw per-fork schedule parameters
# ---------------------------------------------------------------------------

def draw_fork_schedule(fork_seed: int, n_cells: int) -> dict:
    """Draw the per-fork mixed session schedule and parameters.

    Uses rng 140_000 + fork_seed for ALL draws (logged for audit).

    Returns:
      segment_types: list of 6 segment type strings in execution order,
                     e.g. ["SLOW", "CTRL", "SLOW", "NOISE", "PLACE", "SLOW"]
      segment_params: list of 6 per-segment param dicts
      derangement: the SLOW derangement (forced by seed parity)
      p_true_noise: float, NOISE world's p_true
      noisy_cells: list[int], PLACE world's noisy cell indices
      first_slow_segment_idx: 0-based index of first SLOW segment in order
    """
    rng = np.random.default_rng(SCHEDULE_SEED_OFFSET + fork_seed)

    # SLOW derangement: FORCED by seed parity (fixes Exp 167 rng accident)
    derangement = DERANGEMENT_OPTIONS[fork_seed % 2]

    # NOISE p_true: drawn from {0.6, 0.7, 0.8}
    p_true_noise = float(P_TRUE_NOISE_OPTIONS[int(rng.integers(0, len(P_TRUE_NOISE_OPTIONS)))])

    # PLACE noisy cells: 13 of n_cells drawn without replacement
    noisy_cells_arr = rng.choice(n_cells, size=N_NOISY_CELLS, replace=False)
    noisy_cells = sorted(int(x) for x in noisy_cells_arr)

    # Segment order: shuffle [SLOW, SLOW, SLOW, VALID1, VALID2, VALID3]
    # where VALID slots are CTRL/NOISE/PLACE in shuffled assignment
    valid_types = ["CTRL", "NOISE", "PLACE"]
    valid_perm = rng.permutation(3)
    valid_assigned = [valid_types[int(i)] for i in valid_perm]  # e.g. [NOISE, CTRL, PLACE]

    # Build the pool: 3 SLOWs + 3 VALIDs
    segment_pool = ["SLOW", "SLOW", "SLOW"] + valid_assigned
    order_perm = rng.permutation(N_SEGMENTS)
    segment_types = [segment_pool[int(i)] for i in order_perm]

    # Build per-segment param dicts (logged for audit)
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
# Build segment-aware cmap objects
# ---------------------------------------------------------------------------

def build_segment_cmaps(
    base_cmap: list,
    n_colors: int,
    schedule: dict,
    fork_seed: int,
) -> list:
    """Build one cmap object per segment.

    SLOW segments share the same derangement (forced by seed parity) but each
    gets its OWN StructCmap instance (current_step reset per segment).
    NOISE uses NoisyCmap with rng 140_000+seed+segment_idx (deterministic,
    independent per segment — the outer schedule rng has already been consumed).
    PLACE uses RandomPlaceNoiseCmap with rng PLACE_NOISE_SEED_OFFSET+seed.
    CTRL uses the base_cmap list directly.
    """
    segment_types = schedule["segment_types"]
    derangement = schedule["derangement"]
    p_true_noise = schedule["p_true_noise"]
    noisy_cells = schedule["noisy_cells"]

    cmaps = []
    noise_seg_counter = 0   # count NOISE segments for unique rng
    place_seg_counter = 0   # count PLACE segments for unique rng
    for seg_idx, seg_type in enumerate(segment_types):
        if seg_type == "SLOW":
            cmaps.append(StructCmap(base_cmap, n_colors, derangement, HALF_PERIOD_SLOW))
        elif seg_type == "CTRL":
            cmaps.append(base_cmap)   # plain list; no step counter needed
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
# Run one fork (continuous across all 6 segments)
# ---------------------------------------------------------------------------

def run_fork(
    mirro: Creature,
    fork_seed: int,
    base_cmap: list,
    n_colors: int,
    schedule: dict,
) -> dict:
    """Run one continuous fork across all 6 segments (24000 steps).

    Returns per-step correctness array, per-step deranged-phase membership
    (SLOW segments only), ahat_drift, and segment boundary info.
    """
    assert SURPRISE_WINDOW == 200, (
        f"SURPRISE_WINDOW from creature.py == {SURPRISE_WINDOW}, expected 200"
    )

    fork_name = f"exp168_s{fork_seed}"
    c = mirro.fork(fork_name)

    n_cells = c.world.n_cells
    n_actions = 4

    segment_types = schedule["segment_types"]
    cmaps = build_segment_cmaps(base_cmap, n_colors, schedule, fork_seed)

    B = c.world.transition_matrix()   # (n_cells, n_cells, 4)

    A_hat_start = c._A_hat().copy()
    ewma = np.full(n_cells, EWMA_INIT, dtype=np.float64)

    correct_arr = np.empty(N_STEPS, dtype=np.int32)
    # Per-step context: for SLOW segments, 0=context_A, 1=context_B (deranged)
    #                   for valid segments, -1
    context_arr = np.full(N_STEPS, -1, dtype=np.int32)

    global_step = 0
    eps = 1e-300

    for seg_idx in range(N_SEGMENTS):
        seg_type = segment_types[seg_idx]
        cmap_obj = cmaps[seg_idx]
        seg_start = seg_idx * SEGMENT_STEPS

        # For SLOW segments: reset StructCmap.current_step to 0 at seg start
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

                # Set step counter for StructCmap (SLOW segments)
                if seg_type == "SLOW":
                    cmap_obj.current_step = step_in_seg  # type: ignore[union-attr]
                    context_arr[t] = int((step_in_seg // HALF_PERIOD_SLOW) % 2)

                # A_hat pre-step
                A_hat = c._A_hat()

                # Predicted observation
                pred_probs = A_hat @ c.qs
                o_hat = int(np.argmax(pred_probs))

                # bel_cell for EWMA update
                bel_cell = int(np.argmax(c.qs))

                # Observe
                true_pos_t = c.true_pos
                if seg_type == "CTRL":
                    obs = int(base_cmap[true_pos_t])
                else:
                    obs = int(cmap_obj[true_pos_t])

                # Correctness
                correct_t = 1 if (o_hat == obs) else 0
                correct_arr[t] = correct_t

                # Surprise (for trajectory consistency)
                p_o = float(A_hat[obs, :] @ c.qs)
                _ = -math.log(p_o + eps)

                # Belief update
                likelihood = A_hat[obs, :]
                qs_updated = likelihood * c.qs
                denom = qs_updated.sum()
                if denom > 0:
                    qs_updated = qs_updated / denom
                else:
                    qs_updated = np.ones(n_cells) / n_cells

                # Dirichlet count update
                c.pA[obs, :] += qs_updated

                # Value accumulation (Exp 26 mechanism)
                map_cell = int(np.argmax(qs_updated))
                predicted_obs_dist = A_hat[:, map_cell]
                h_predicted = -np.sum(
                    predicted_obs_dist * np.log(predicted_obs_dist + 1e-12)
                )
                predictability_weight = math.exp(-h_predicted)
                c.value_counts[obs] += predictability_weight

                # Random action
                action = int(rng.integers(0, n_actions))

                # Move
                c.true_pos = c.world.move(c.true_pos, action)

                # Advance qs through B
                c.qs = B[:, :, action] @ qs_updated

                c.age_steps += 1

                # EWMA update AFTER correctness known
                ewma[bel_cell] = (1.0 - ALPHA) * ewma[bel_cell] + ALPHA * correct_t

                global_step += 1

    A_hat_end = c._A_hat()
    ahat_drift = float(np.abs(A_hat_end - A_hat_start).max())

    return {
        "fork_seed": fork_seed,
        "n_total": N_STEPS,
        "ahat_drift": ahat_drift,
        "correct_arr": correct_arr,
        "context_arr": context_arr,       # -1 outside SLOW, 0/1 inside SLOW
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
    """Evaluate a fixed-dial agent (W constant) and return per-segment eval records.

    Returns:
      all_eval_records: list of all eval records
      seg_eval_records: dict seg_idx -> list of eval records in that segment
      seg_correctness: dict seg_idx -> fraction correct (by ground-truth label)
    """
    n_total = len(correct_arr)
    bounds = segment_boundaries()
    seg_to_idx = {i: i for i in range(N_SEGMENTS)}

    all_eval_records: list[dict] = []
    seg_eval_records: dict[int, list[dict]] = {i: [] for i in range(N_SEGMENTS)}

    for t in range(EVAL_EVERY - 1, n_total, EVAL_EVERY):
        if t + 1 < W:
            continue
        win_arr = correct_arr[max(0, t - W + 1): t + 1].astype(np.float64)
        label, err_rate, rho = classify_window(win_arr)

        # Find which segment this eval point belongs to
        seg_idx = t // SEGMENT_STEPS

        rec = {"t": t, "label": label, "e_w": err_rate, "rho": rho, "seg_idx": seg_idx}
        all_eval_records.append(rec)
        seg_eval_records[seg_idx].append(rec)

    # Compute per-segment correctness
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
# Post-hoc: simulate N2+N3 LOCK agent (b) — Exp 167 verbatim
# ---------------------------------------------------------------------------

def eval_n3_agent_segments(correct_arr: np.ndarray, segment_types: list[str]) -> dict:
    """Simulate N2+N3 LOCK agent (b) over the correctness stream (24000 steps).

    The controller runs CONTINUOUSLY across all segments — it has NO segment
    knowledge.  Same logic as exp167 eval_n3_agent verbatim.

    Returns eval_records (per eval point), dial_trajectory, lock_events,
    seg_eval_records, seg_correctness, change_by_seg_type, first_slow_responded
    (set of fork-level: whether the first SLOW segment saw a dial change).
    """
    n_total = len(correct_arr)
    HORIZON = 100   # forecast horizon (next 100 steps)
    MAX_SCORABLE_T = n_total - 1 - HORIZON   # = 23899

    current_W = DIAL_CANDIDATES[0]   # start at 200
    dial_trajectory: list[dict] = [{"t": -1, "dial_change_to": current_W,
                                     "trust_at_change": None, "lock_event": None}]

    # Rolling scored outcomes since last dial change
    rolling_outcomes: _deque[int] = _deque(maxlen=ROLLING_TRUST_WINDOW)
    # Pending forecasts: (t_eval, label, e_w) not yet scorable
    pending: list[dict] = []
    # Count of scored labels since last dial change
    scored_since_change = 0

    # LOCK state (verbatim from exp167)
    label_run: _deque[str] = _deque(maxlen=LOCK_K)
    locked = False
    lock_class: str | None = None

    eval_records: list[dict] = []
    lock_events: list[dict] = []

    # Track which segment each dial change occurs in
    # (segment index = t // SEGMENT_STEPS)
    change_seg_idxs: list[int] = []   # segment index for each dial change

    for t in range(EVAL_EVERY - 1, n_total, EVAL_EVERY):

        # --- Step 1: mature any pending forecasts where t+100 has elapsed ---
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
                # STRUCTURAL: not scored (verbatim exp167)
                continue
            rolling_outcomes.append(1 - violated)
            scored_since_change += 1

        # --- Step 2: check trust rule — only if NOT LOCKED ---
        if (not locked and
                scored_since_change >= MIN_SCORED_FOR_TRUST and
                len(rolling_outcomes) > 0):
            trust = float(sum(rolling_outcomes) / len(rolling_outcomes))
            if trust < TRUST_FIRE_THRESH:
                # Advance dial cyclically
                idx = DIAL_CANDIDATES.index(current_W)
                current_W = DIAL_CANDIDATES[(idx + 1) % len(DIAL_CANDIDATES)]
                rolling_outcomes.clear()
                scored_since_change = 0
                pending = []
                label_run.clear()
                locked = False
                lock_class = None
                seg_idx_now = t // SEGMENT_STEPS
                change_seg_idxs.append(seg_idx_now)
                dial_trajectory.append({
                    "t": t,
                    "dial_change_to": current_W,
                    "trust_at_change": trust,
                    "lock_event": None,
                    "seg_idx": seg_idx_now,
                    "seg_type": segment_types[seg_idx_now],
                })

        # --- Step 3: evaluate current window under current_W ---
        if t + 1 >= current_W:
            win_arr = correct_arr[max(0, t - current_W + 1): t + 1].astype(np.float64)
            label, err_rate, rho = classify_window(win_arr)
            seg_idx_now = t // SEGMENT_STEPS
            eval_records.append({
                "t": t, "label": label, "e_w": err_rate, "rho": rho,
                "W_used": current_W, "locked": locked, "seg_idx": seg_idx_now,
            })

            # Register forecast for OK and NOISE only
            if label in ("OK", "NOISE") and t <= MAX_SCORABLE_T:
                pending.append({"t_eval": t, "label": label, "e_w": err_rate})

            # --- Update label_run and re-evaluate lock ---
            label_run.append(label)

            if len(label_run) == LOCK_K and len(set(label_run)) == 1:
                new_lock_class = label_run[-1]
                if not locked or lock_class != new_lock_class:
                    locked = True
                    lock_class = new_lock_class
                    lock_ev = {"t": t, "event": "LOCKED", "on_class": lock_class,
                               "dial": current_W, "seg_idx": seg_idx_now,
                               "seg_type": segment_types[seg_idx_now]}
                    lock_events.append(lock_ev)
                    dial_trajectory[-1] = dict(dial_trajectory[-1], lock_event=lock_ev)
            else:
                if locked:
                    lock_ev = {"t": t, "event": "UNLOCKED", "on_class": lock_class,
                               "dial": current_W, "seg_idx": seg_idx_now,
                               "seg_type": segment_types[seg_idx_now]}
                    lock_events.append(lock_ev)
                    dial_trajectory.append({
                        "t": t, "dial_change_to": current_W,
                        "trust_at_change": None, "lock_event": lock_ev,
                        "seg_idx": seg_idx_now, "seg_type": segment_types[seg_idx_now],
                    })
                locked = False
                lock_class = None

    # --- Build per-segment eval records and correctness ---
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

    # --- Count dial changes by segment TYPE ---
    change_by_seg_type: dict[str, int] = {"SLOW": 0, "CTRL": 0, "NOISE": 0, "PLACE": 0}
    for seg_idx in change_seg_idxs:
        seg_type = segment_types[seg_idx]
        change_by_seg_type[seg_type] = change_by_seg_type.get(seg_type, 0) + 1

    total_dial_changes = len(change_seg_idxs)

    return {
        "eval_records": eval_records,
        "dial_trajectory": dial_trajectory,
        "lock_events": lock_events,
        "seg_eval_records": seg_eval_records,
        "seg_correctness": seg_correctness,
        "change_seg_idxs": change_seg_idxs,
        "change_by_seg_type": change_by_seg_type,
        "total_dial_changes": total_dial_changes,
        "final_dial": current_W,
        "final_locked": locked,
        "final_lock_class": lock_class,
    }


# ---------------------------------------------------------------------------
# Compute scores for one fork
# ---------------------------------------------------------------------------

def compute_fork_scores(run_result: dict) -> dict:
    """Compute per-segment correctness for agents (a) and (b), plus P1-P4 inputs."""
    correct_arr = run_result["correct_arr"]
    segment_types = run_result["segment_types"]

    # Agent (a): fixed W=200
    a_result = eval_fixed_dial_segments(correct_arr, W=200, segment_types=segment_types)

    # Agent (b): N2+N3 LOCK controller
    b_result = eval_n3_agent_segments(correct_arr, segment_types=segment_types)

    return {
        "segment_types": segment_types,
        "a_seg_correctness": a_result["seg_correctness"],
        "a_seg_eval_records": a_result["seg_eval_records"],
        "b_seg_correctness": b_result["seg_correctness"],
        "b_seg_eval_records": b_result["seg_eval_records"],
        "b_eval_records": b_result["eval_records"],
        "dial_trajectory": b_result["dial_trajectory"],
        "lock_events": b_result["lock_events"],
        "change_by_seg_type": b_result["change_by_seg_type"],
        "change_seg_idxs": b_result["change_seg_idxs"],
        "total_dial_changes": b_result["total_dial_changes"],
        "final_dial": b_result["final_dial"],
        "final_locked": b_result["final_locked"],
        "final_lock_class": b_result["final_lock_class"],
    }


# ---------------------------------------------------------------------------
# Precondition checks
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
        # PC1
        if rr["ahat_drift"] >= PC1_AHAT_DRIFT_MAX:
            failures.append(
                f"PC1 FAIL: seed={seed} ahat_drift={rr['ahat_drift']:.4f} "
                f"(bound={PC1_AHAT_DRIFT_MAX}, 24000-step horizon-scaled)"
            )
        # PC2: >= 30 eval points per segment at W=200
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

    # PC3: SLOW deranged-phase per-step error >= 0.9 pooled
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
# Evaluate predeclared properties P1-P4 / falsifiers F1-F4
# ---------------------------------------------------------------------------

def evaluate(
    run_results: list[dict],
    fork_scores: dict,
    seeds: list[int],
) -> dict:
    """Evaluate P1/F1, P2/F2, P3/F3, P4/F4."""

    # --- P1: override concentration ---
    # Pooled fraction of (b)'s dial changes occurring in SLOW segments
    total_changes_all = 0
    slow_changes_all = 0
    for seed in seeds:
        fs = fork_scores[seed]
        cbt = fs["change_by_seg_type"]
        tc = fs["total_dial_changes"]
        total_changes_all += tc
        slow_changes_all += cbt.get("SLOW", 0)

    if total_changes_all > 0:
        p1_conc = slow_changes_all / total_changes_all
    else:
        p1_conc = float("nan")

    p1_pass = not math.isnan(p1_conc) and p1_conc >= P1_CONC_MIN
    f1 = not math.isnan(p1_conc) and p1_conc <= F1_CONC_MAX

    # --- P2: responsiveness ---
    # For each fork, find the first SLOW segment index in the schedule,
    # then check if at least one dial change occurs in that segment.
    p2_fork_pass: list[bool] = []
    for seed in seeds:
        rr = next(r for r in run_results if r["fork_seed"] == seed)
        schedule = rr["schedule"]
        first_slow_seg = schedule["first_slow_segment_idx"]
        fs = fork_scores[seed]
        change_seg_idxs = fs["change_seg_idxs"]
        responded = any(idx == first_slow_seg for idx in change_seg_idxs)
        p2_fork_pass.append(responded)

    n_p2_forks = sum(1 for x in p2_fork_pass if x)
    p2_pass = n_p2_forks >= P2_FORKS_MIN
    f2 = n_p2_forks <= F2_FORKS_MAX

    # --- P3: no harm in valid regimes ---
    valid_a_scores: list[float] = []
    valid_b_scores: list[float] = []
    for seed in seeds:
        rr = next(r for r in run_results if r["fork_seed"] == seed)
        segment_types = rr["segment_types"]
        fs = fork_scores[seed]
        for seg_idx, seg_type in enumerate(segment_types):
            if seg_type == "SLOW":
                continue
            a_corr = fs["a_seg_correctness"].get(seg_idx, float("nan"))
            b_corr = fs["b_seg_correctness"].get(seg_idx, float("nan"))
            if not math.isnan(a_corr):
                valid_a_scores.append(a_corr)
            if not math.isnan(b_corr):
                valid_b_scores.append(b_corr)

    pooled_valid_a = float(np.mean(valid_a_scores)) if valid_a_scores else float("nan")
    pooled_valid_b = float(np.mean(valid_b_scores)) if valid_b_scores else float("nan")
    if not math.isnan(pooled_valid_a) and not math.isnan(pooled_valid_b):
        p3_margin = pooled_valid_b - pooled_valid_a   # positive = (b) better
    else:
        p3_margin = float("nan")

    p3_pass = not math.isnan(p3_margin) and p3_margin >= -P3_NO_HARM_MARGIN
    f3 = not math.isnan(p3_margin) and (pooled_valid_a - pooled_valid_b) > F3_HARM_MARGIN

    # --- P4: overrides earn their keep ---
    slow_a_scores: list[float] = []
    slow_b_scores: list[float] = []
    for seed in seeds:
        rr = next(r for r in run_results if r["fork_seed"] == seed)
        segment_types = rr["segment_types"]
        fs = fork_scores[seed]
        for seg_idx, seg_type in enumerate(segment_types):
            if seg_type != "SLOW":
                continue
            a_corr = fs["a_seg_correctness"].get(seg_idx, float("nan"))
            b_corr = fs["b_seg_correctness"].get(seg_idx, float("nan"))
            if not math.isnan(a_corr):
                slow_a_scores.append(a_corr)
            if not math.isnan(b_corr):
                slow_b_scores.append(b_corr)

    pooled_slow_a = float(np.mean(slow_a_scores)) if slow_a_scores else float("nan")
    pooled_slow_b = float(np.mean(slow_b_scores)) if slow_b_scores else float("nan")
    if not math.isnan(pooled_slow_a) and not math.isnan(pooled_slow_b):
        p4_margin = pooled_slow_b - pooled_slow_a   # positive = (b) better
    else:
        p4_margin = float("nan")

    p4_pass = not math.isnan(p4_margin) and p4_margin >= P4_IMPROVE_MIN
    f4 = not math.isnan(p4_margin) and p4_margin <= F4_IMPROVE_MAX

    # --- Verdict ---
    positive = p1_pass and p2_pass and p3_pass and p4_pass
    negative = f1 or f2 or f3 or f4
    if positive:
        verdict_str = "POSITIVE"
    elif negative:
        verdict_str = "NEGATIVE"
    else:
        verdict_str = "MIXED"

    # --- Derangement variant audit ---
    derang_variants: set[tuple] = set()
    for seed in seeds:
        rr = next(r for r in run_results if r["fork_seed"] == seed)
        d = tuple(rr["schedule"]["derangement"])
        derang_variants.add(d)

    # --- Descent question (ungated diagnostic) ---
    # Does (b) ever cycle 1600->200 back down after SLOW ends?
    # Visible as a dial change TO 200 occurring in a valid segment.
    descent_events: list[dict] = []
    for seed in seeds:
        fs = fork_scores[seed]
        rr = next(r for r in run_results if r["fork_seed"] == seed)
        segment_types = rr["segment_types"]
        for d in fs["dial_trajectory"]:
            if d.get("trust_at_change") is not None and d["dial_change_to"] == 200:
                seg_idx = d.get("seg_idx")
                if seg_idx is not None and segment_types[seg_idx] != "SLOW":
                    descent_events.append({
                        "seed": seed,
                        "t": d["t"],
                        "seg_idx": seg_idx,
                        "seg_type": segment_types[seg_idx],
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
        # P3
        "pooled_valid_a": pooled_valid_a,
        "pooled_valid_b": pooled_valid_b,
        "p3_margin": p3_margin,
        "p3_pass": p3_pass,
        "f3": f3,
        # P4
        "pooled_slow_a": pooled_slow_a,
        "pooled_slow_b": pooled_slow_b,
        "p4_margin": p4_margin,
        "p4_pass": p4_pass,
        "f4": f4,
        # Verdict
        "verdict_str": verdict_str,
        # Diagnostics
        "derang_variants": [list(d) for d in sorted(derang_variants)],
        "descent_events": descent_events,
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

            # Build per-segment summary
            seg_summary = []
            for i in range(N_SEGMENTS):
                seg_summary.append({
                    "seg_idx": i,
                    "seg_type": segment_types[i],
                    "a_correctness": _nan_to_none(fs["a_seg_correctness"].get(i)),
                    "b_correctness": _nan_to_none(fs["b_seg_correctness"].get(i)),
                    "n_eval_a": len(fs["a_seg_eval_records"].get(i, [])),
                    "n_eval_b": len(fs["b_seg_eval_records"].get(i, [])),
                })

            row: dict = {
                "exp": 168,
                "fork_seed": seed,
                "n_total": rr["n_total"],
                "ahat_drift": _nan_to_none(rr["ahat_drift"]),
                "segment_types": segment_types,
                "derangement": schedule["derangement"],
                "p_true_noise": _nan_to_none(schedule["p_true_noise"]),
                "noisy_cells_first5": schedule["noisy_cells"][:5],
                "first_slow_segment_idx": schedule["first_slow_segment_idx"],
                "segment_scores": seg_summary,
                "total_dial_changes": fs["total_dial_changes"],
                "change_by_seg_type": fs["change_by_seg_type"],
                "change_seg_idxs": fs["change_seg_idxs"],
                "final_dial": fs["final_dial"],
                "final_locked": fs["final_locked"],
                "final_lock_class": fs["final_lock_class"],
                "dial_trajectory_summary": [
                    {
                        "t": d["t"],
                        "to": d["dial_change_to"],
                        "trust": _nan_to_none(d.get("trust_at_change")),
                        "seg_idx": d.get("seg_idx"),
                        "seg_type": d.get("seg_type"),
                        "lock_event": d.get("lock_event"),
                    }
                    for d in fs["dial_trajectory"]
                ],
            }
            fh.write(json.dumps(row) + "\n")

        if ev is not None:
            def _fv(v):
                if isinstance(v, float) and math.isnan(v):
                    return None
                return v

            summary: dict = {
                "exp": 168,
                "row_type": "summary",
                "p1_conc": _fv(ev["p1_conc"]),
                "p1_pass": ev["p1_pass"],
                "f1": ev["f1"],
                "n_p2_forks": ev["n_p2_forks"],
                "p2_pass": ev["p2_pass"],
                "f2": ev["f2"],
                "pooled_valid_a": _fv(ev["pooled_valid_a"]),
                "pooled_valid_b": _fv(ev["pooled_valid_b"]),
                "p3_margin": _fv(ev["p3_margin"]),
                "p3_pass": ev["p3_pass"],
                "f3": ev["f3"],
                "pooled_slow_a": _fv(ev["pooled_slow_a"]),
                "pooled_slow_b": _fv(ev["pooled_slow_b"]),
                "p4_margin": _fv(ev["p4_margin"]),
                "p4_pass": ev["p4_pass"],
                "f4": ev["f4"],
                "verdict": ev["verdict_str"],
            }
            fh.write(json.dumps(summary) + "\n")


# ---------------------------------------------------------------------------
# Print dial trajectories + segment annotations
# ---------------------------------------------------------------------------

def print_dial_trajectories(run_results: list[dict], fork_scores: dict, seeds: list[int]) -> None:
    """Print agent (b)'s dial trajectory per fork with segment annotations."""
    print("Agent (b) dial trajectories + LOCK events (segment-annotated):")
    bounds = segment_boundaries()

    for seed in seeds:
        rr = next(r for r in run_results if r["fork_seed"] == seed)
        fs = fork_scores[seed]
        segment_types = rr["segment_types"]
        schedule = rr["schedule"]

        # Print segment schedule
        seg_str = "  ".join(f"[{i}:{t}]" for i, t in enumerate(segment_types))
        print(f"  seed={seed}  schedule: {seg_str}  "
              f"derang={schedule['derangement']}  "
              f"p_noise={schedule['p_true_noise']:.2f}  "
              f"noisy_cells(5)={schedule['noisy_cells'][:5]}")
        print(f"    first_slow_seg={schedule['first_slow_segment_idx']}")

        # Dial trajectory
        traj = fs["dial_trajectory"]
        traj_strs = []
        for d in traj:
            s = f"t={d['t']} W={d['dial_change_to']}"
            if d.get("trust_at_change") is not None:
                seg_label = (f" seg={d.get('seg_type', '?')}"
                             if d.get("seg_type") else "")
                s += f" [trust={d['trust_at_change']:.3f}{seg_label}]"
            if d.get("lock_event"):
                le = d["lock_event"]
                s += f" [LOCK:{le['event']} on {le['on_class']} seg={le.get('seg_type','?')}]"
            traj_strs.append(s)
        print(f"    traj: {' -> '.join(traj_strs)}")

        # Per-segment correctness
        seg_lines = []
        for i in range(N_SEGMENTS):
            a_c = fs["a_seg_correctness"].get(i, float("nan"))
            b_c = fs["b_seg_correctness"].get(i, float("nan"))
            st = segment_types[i]
            seg_lines.append(f"  seg{i}({st}): a={a_c:.3f} b={b_c:.3f}")
        print("    " + "  ".join(seg_lines))

        # Change summary
        cbt = fs["change_by_seg_type"]
        print(f"    changes: total={fs['total_dial_changes']} by_type={cbt}")
        print(f"    final: W={fs['final_dial']} locked={fs['final_locked']}({fs['final_lock_class']})")
        print()

    print()


# ---------------------------------------------------------------------------
# Print per-segment correctness table
# ---------------------------------------------------------------------------

def print_score_table(run_results: list[dict], fork_scores: dict, seeds: list[int],
                      ev: dict) -> None:
    """Print pooled per-segment-type correctness table (a) vs (b)."""
    def _f(v, w=7):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "nan".rjust(w)
        return f"{v:.4f}".rjust(w)

    print("Pooled per-segment-type correctness (agent a=W200 vs b=N3-LOCK):")
    print(f"  {'seg_type':>10}  {'a_pooled':>9}  {'b_pooled':>9}  {'b-a':>7}  {'gt_label':>10}")
    print("  " + "-" * 55)

    for seg_type in ["SLOW", "CTRL", "NOISE", "PLACE"]:
        a_vals: list[float] = []
        b_vals: list[float] = []
        for seed in seeds:
            rr = next(r for r in run_results if r["fork_seed"] == seed)
            fs = fork_scores[seed]
            segment_types = rr["segment_types"]
            for seg_idx, st in enumerate(segment_types):
                if st != seg_type:
                    continue
                a_c = fs["a_seg_correctness"].get(seg_idx, float("nan"))
                b_c = fs["b_seg_correctness"].get(seg_idx, float("nan"))
                if not math.isnan(a_c):
                    a_vals.append(a_c)
                if not math.isnan(b_c):
                    b_vals.append(b_c)
        pa = float(np.mean(a_vals)) if a_vals else float("nan")
        pb = float(np.mean(b_vals)) if b_vals else float("nan")
        diff = pb - pa if not math.isnan(pa) and not math.isnan(pb) else float("nan")
        gt = _ground_truth_label(seg_type)
        print(f"  {seg_type:>10}  {_f(pa)}  {_f(pb)}  {_f(diff)}  {gt:>10}")
    print()

    print(f"  Valid (CTRL+NOISE+PLACE): a={_f(ev['pooled_valid_a'])} "
          f"b={_f(ev['pooled_valid_b'])} margin(b-a)={_f(ev['p3_margin'])}")
    print(f"  SLOW:                     a={_f(ev['pooled_slow_a'])} "
          f"b={_f(ev['pooled_slow_b'])} margin(b-a)={_f(ev['p4_margin'])}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exp 168 — N3 rung 4, MIXED SCHEDULE: independent variance / override concentration."
        )
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Smoke run: seed=[106] only, full 24000 steps, prints segment schedule, "
            "dial trajectory, per-segment scores for both agents, lock states. No verdict."
        ),
    )
    args = parser.parse_args()

    smoke = args.smoke
    seeds = [106] if smoke else SEEDS

    print("=" * 80)
    print("Exp 168 — N3 rung 4: MIXED SCHEDULE (independent variance / override concentration)")
    print("          Rungs 1–3 PASSED (Exp 162–167). This is the card's FINAL rung.")
    print("=" * 80)
    print()
    print(f"SURPRISE_WINDOW={SURPRISE_WINDOW}  [must be 200; asserted]")
    assert SURPRISE_WINDOW == 200, f"SURPRISE_WINDOW={SURPRISE_WINDOW}, expected 200"
    print(f"CLASSIFIER_WINDOW_DEFAULT={CLASSIFIER_WINDOW_DEFAULT}  EVAL_EVERY={EVAL_EVERY}")
    print(f"ERROR_RATE_OK_THRESH={ERROR_RATE_OK_THRESH}  LAG1_RHO_STRUCT_THRESH={LAG1_RHO_STRUCT_THRESH}")
    print(f"DIAL_CANDIDATES={DIAL_CANDIDATES}  starting W={CLASSIFIER_WINDOW_DEFAULT}")
    print(f"N3 trust: TRUST_FIRE_THRESH={TRUST_FIRE_THRESH}  ROLLING_TRUST_WINDOW={ROLLING_TRUST_WINDOW}  "
          f"MIN_SCORED_FOR_TRUST={MIN_SCORED_FOR_TRUST}")
    print(f"N3 forecast (verbatim exp167): OK_VIOLATION_THRESH={OK_VIOLATION_THRESH}  "
          f"NOISE_VIOLATION_DELTA={NOISE_VIOLATION_DELTA}  STRUCTURAL=NOT scored")
    print(f"LOCK: LOCK_K={LOCK_K}")
    print(f"SLOW: HALF_PERIOD_SLOW={HALF_PERIOD_SLOW}  derang forced by seed parity "
          f"(even->[1,2,0], odd->[2,0,1])")
    print(f"N_STEPS={N_STEPS}  N_SEGMENTS={N_SEGMENTS}  SEGMENT_STEPS={SEGMENT_STEPS}")
    print(f"SCHEDULE_SEED_OFFSET={SCHEDULE_SEED_OFFSET}  PLACE_NOISE_SEED_OFFSET={PLACE_NOISE_SEED_OFFSET}")
    print(f"PC1_AHAT_DRIFT_MAX={PC1_AHAT_DRIFT_MAX}  (24000-step horizon-scaled bound, 6x 4000-step)")
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
              f"total_dial_changes={fs['total_dial_changes']}  "
              f"change_by_seg={fs['change_by_seg_type']}",
              flush=True)
    print()

    # --- Print dial trajectories (ungated diagnostic) ---
    print_dial_trajectories(run_results, fork_scores, seeds)

    # --- Derangement variant audit ---
    variants = set(tuple(rr["schedule"]["derangement"]) for rr in run_results)
    print(f"Derangement variants in run: {[list(v) for v in sorted(variants)]}")
    if len(variants) == 2:
        print("  => Both variants present (GOOD — fixes Exp 167 rng accident)")
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
        out_rows_path = Path(__file__).parent / "outputs" / "exp168_rows.json"
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

    # --- P1 / F1 ---
    def _fv(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "nan"
        return f"{v:.4f}"

    print("=" * 80)
    print("PREDECLARED OUTCOME MAP")
    print("=" * 80)
    print()

    print(f"P1 (override concentration >= {P1_CONC_MIN}, time-share chance 0.5):")
    print(f"  Total dial changes = {ev['total_changes_all']}  "
          f"in SLOW = {ev['slow_changes_all']}  "
          f"pooled fraction = {_fv(ev['p1_conc'])}  (need >= {P1_CONC_MIN})")
    print(f"  FALSIFIER F1: fraction <= {F1_CONC_MAX}: "
          f"{'YES — F1 FIRES' if ev['f1'] else 'no'}")
    p1_status = "PASS" if ev["p1_pass"] else ("FALSIFIER F1" if ev["f1"] else "MIXED (not P1, not F1)")
    print(f"  => P1: {p1_status}")
    print()

    print(f"P2 (responsiveness: >= {P2_FORKS_MIN}/{n} forks respond on first SLOW seg):")
    for i, seed in enumerate(seeds):
        print(f"  seed={seed}: responded_on_first_slow={ev['p2_fork_pass'][i]}")
    print(f"  Forks responding: {ev['n_p2_forks']}/{n}  (need >= {P2_FORKS_MIN})")
    print(f"  FALSIFIER F2: <= {F2_FORKS_MAX}/{n} forks: "
          f"{'YES — F2 FIRES' if ev['f2'] else 'no'}")
    p2_status = "PASS" if ev["p2_pass"] else ("FALSIFIER F2" if ev["f2"] else "MIXED (not P2, not F2)")
    print(f"  => P2: {p2_status}")
    print()

    print(f"P3 (no harm in valid regimes: pooled_b >= pooled_a - {P3_NO_HARM_MARGIN}):")
    print(f"  pooled_valid_a={_fv(ev['pooled_valid_a'])}  "
          f"pooled_valid_b={_fv(ev['pooled_valid_b'])}  "
          f"margin(b-a)={_fv(ev['p3_margin'])}  (need >= -{P3_NO_HARM_MARGIN})")
    print(f"  FALSIFIER F3: (a)-(b) > {F3_HARM_MARGIN}: "
          f"{'YES — F3 FIRES' if ev['f3'] else 'no'}")
    p3_status = "PASS" if ev["p3_pass"] else ("FALSIFIER F3" if ev["f3"] else "MIXED (not P3, not F3)")
    print(f"  => P3: {p3_status}")
    print()

    print(f"P4 (overrides earn keep: pooled_slow_b >= pooled_slow_a + {P4_IMPROVE_MIN}):")
    print(f"  pooled_slow_a={_fv(ev['pooled_slow_a'])}  "
          f"pooled_slow_b={_fv(ev['pooled_slow_b'])}  "
          f"margin(b-a)={_fv(ev['p4_margin'])}  (need >= {P4_IMPROVE_MIN})")
    print(f"  FALSIFIER F4: (b)-(a) <= {F4_IMPROVE_MAX}: "
          f"{'YES — F4 FIRES' if ev['f4'] else 'no'}")
    p4_status = "PASS" if ev["p4_pass"] else ("FALSIFIER F4" if ev["f4"] else "MIXED (not P4, not F4)")
    print(f"  => P4: {p4_status}")
    print()

    # --- Descent diagnostic ---
    if ev["descent_events"]:
        print("Descent events (dial 1600->200 after SLOW ends, in valid segment):")
        for de in ev["descent_events"]:
            print(f"  seed={de['seed']} t={de['t']} seg={de['seg_idx']}({de['seg_type']})")
    else:
        print("Descent events: none (dial did not cycle down in valid segments)")
    print()

    # --- Conjunct summary + VERDICT ---
    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    print(f"P1  (override concentration >= {P1_CONC_MIN}): {p1_status}")
    print(f"P2  (responsiveness >= {P2_FORKS_MIN}/{n} forks): {p2_status}")
    print(f"P3  (no harm in valid regimes, margin >= -{P3_NO_HARM_MARGIN}): {p3_status}")
    print(f"P4  (overrides earn keep, margin >= +{P4_IMPROVE_MIN}): {p4_status}")
    print()
    if ev["verdict_str"] == "POSITIVE":
        print("VERDICT: POSITIVE")
        print("RUNG 4 PASSES — with rungs 1-3, the meta-calibration-n3 card's central")
        print("hypothesis is SUPPORTED: N3 is a genuine regime-adaptive control surface")
        print("over N2 with independent variance concentrated in broken regimes.")
    elif ev["verdict_str"] == "NEGATIVE":
        print("VERDICT: NEGATIVE")
        print("A falsifier fired — see above for which property failed.")
    else:
        print("VERDICT: MIXED")
    print()
    print("=" * 80)

    # --- Write JSON rows ---
    out_rows_path = Path(__file__).parent / "outputs" / "exp168_rows.json"
    write_json_rows(run_results, fork_scores, ev=ev, path=out_rows_path)
    print(f"JSON rows written to {out_rows_path}")

    # --- Write verdict JSON ---
    def _fv2(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "nan"
        return f"{v:.4f}"

    arms = {
        "P1_override_concentration": {
            "pass": bool(ev["p1_pass"]),
            "reason": (
                f"pooled fraction of changes in SLOW = {_fv2(ev['p1_conc'])} "
                f"(need >= {P1_CONC_MIN}); "
                f"total_changes={ev['total_changes_all']} slow_changes={ev['slow_changes_all']}; "
                f"F1 fired={ev['f1']}"
            ),
        },
        "P2_responsiveness": {
            "pass": bool(ev["p2_pass"]),
            "reason": (
                f"{ev['n_p2_forks']}/{ev['n_forks']} forks responded on first SLOW segment "
                f"(need >= {P2_FORKS_MIN}); "
                f"F2 fired={ev['f2']}"
            ),
        },
        "P3_no_harm_valid": {
            "pass": bool(ev["p3_pass"]),
            "reason": (
                f"pooled_valid_a={_fv2(ev['pooled_valid_a'])} "
                f"pooled_valid_b={_fv2(ev['pooled_valid_b'])} "
                f"margin(b-a)={_fv2(ev['p3_margin'])} (need >= -{P3_NO_HARM_MARGIN}); "
                f"F3 fired={ev['f3']}"
            ),
        },
        "P4_overrides_earn_keep": {
            "pass": bool(ev["p4_pass"]),
            "reason": (
                f"pooled_slow_a={_fv2(ev['pooled_slow_a'])} "
                f"pooled_slow_b={_fv2(ev['pooled_slow_b'])} "
                f"margin(b-a)={_fv2(ev['p4_margin'])} (need >= +{P4_IMPROVE_MIN}); "
                f"F4 fired={ev['f4']}"
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp168_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp168",
        arms=arms,
        verdict=ev["verdict_str"],
        halted=False,
        notes=(
            "N3 rung 4: MIXED SCHEDULE — 6 segments (3 SLOW + 3 valid) in random order. "
            "Exp 167 lock controller (verbatim) runs continuously across all segments. "
            "Tests independent variance: overrides concentrated in broken regimes (P1), "
            "responsive on first SLOW encounter (P2), no harm to valid regimes (P3), "
            "overrides improve SLOW-segment diagnosis (P4). "
            "POSITIVE: RUNG 4 PASSES, card hypothesis SUPPORTED. "
            "NEGATIVE: a falsifier fired, epiphenomenal collapse or indiscriminate firing."
        ),
    )
    print(f"Verdict JSON written to {verdict_path}")


if __name__ == "__main__":
    main()
