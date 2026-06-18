"""Exp 231 — self-other-modeling Rung 2-sweep: does the goal-inference edge GROW with the other's rate of goal change? (turning Exp 230's localization into a law)

Hypothesis: because goal-inference's advantage over a fair adaptive learned baseline is LOCALIZED to post-change
re-tracking (Exp 230), the OVERALL edge should GROW as the other changes its goal more often (more post-change
windows per run). Predeclared SHAPE: the overall ToM edge is MONOTONE NON-INCREASING in the switch PERIOD P
(monotone NON-DECREASING in change frequency 1/P), with a real magnitude.

Setup: Exp 230 substrate + predictors (M3a adaptive goal-inference; M1.5a adaptive learned transition = the FAIR
control). SWEEP P in {100, 200, 400, 800, STATIONARY (no switches)}; for each P run 8 seeds x H=2000 and measure
mean overall edge = M3a_overall - M1.5a_overall, plus M3a_overall, M1.5a_overall, mean q_goal[current_goal].
P=400 is the Exp 230 cell; STATIONARY = goal fixed at the initial favorite color for the whole run (the fair
steady-state cell: Exp 229-like but with the ADAPTIVE baseline). Goal cycling for finite P: goal_color(step) =
all_colors[(step//P) % 3]. For STATIONARY: goal_color = all_colors[0]-equivalent (the initial favorite, color 2),
fixed.

Prediction (LAW TRUE): (a) the sequence edge(P) for P=[100,200,400,800,STATIONARY] is MONOTONE NON-INCREASING
within a tolerance (no adjacent pair INCREASES by > 0.01 as P grows); AND (b) magnitude: edge(100) - edge(STAT)
>= 0.05 (the law has real size, not just direction); AND (c) mean q_goal[current_goal] >= 0.60 at EVERY P.

Falsifier (LAW FALSE):
  F1: the edge is NOT monotone non-increasing in P (some adjacent pair INCREASES by > 0.01 as P grows) -> the
      'ToM pays in proportion to non-stationarity' law is false (relationship non-monotone/flat).
  F2: edge(100) - edge(STAT) < 0.05 -> the law is FLAT: change-rate does not materially modulate the ToM edge.
  F3: mean q_goal[current_goal] < 0.60 at some P -> goal-inference fails to track at that change rate (instrument
      limit; bounds the law's validity to slower changes).
  F4: any committed spine state_hash changes -> continuity bug, halt.
"""
from __future__ import annotations

import copy
import sys
from collections import deque
from pathlib import Path

import numpy as np

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H = 2000                          # steps per seed
SEEDS = list(range(8))            # seeds 0-7
MIRRO_DIR = Path("creature/state/mirro")
VELA_DIR = Path("creature/state/vela")
M2_EPSILON = 1e-6                 # smoothing for M2* oracle
OUTPUT_PATH = Path("experiments/outputs/exp231.txt")

# Comfort-gate parameters (verbatim from exp230/exp229/exp228/exp70)
EPS_POLICY = 0.2                  # epsilon-greedy exploration
ALPHA_EMA = 0.1                   # comfort-EMA alpha
THRESH_GATE = 0.75                # comfort gate threshold
LAMBDA_RELAX = 0.01               # away-from-source EMA relaxation toward 1.0
EST_INIT = 1.0                    # initial comfort estimate

# M1.5a adaptive decay (verbatim from exp230)
TRANS_DECAY = 0.99                # trans_a *= TRANS_DECAY each step

# M3a leak parameters (verbatim from exp230)
Q_LEAK_RETAIN = 0.98              # q_goal *= Q_LEAK_RETAIN each step after Bayes update
Q_LEAK_UNIFORM = 0.02             # (1 - Q_LEAK_RETAIN) added as uniform

# SWEEP: periods to evaluate (None = STATIONARY sentinel)
PERIODS = [100, 200, 400, 800, None]   # None = STATIONARY
PERIOD_LABELS = ["100", "200", "400", "800", "STAT"]

# ---------------------------------------------------------------------------
# Load committed spines (read-only — NEVER call .live() or .save() on these)
# ---------------------------------------------------------------------------

mirro_spine = Creature.load(MIRRO_DIR)
vela_spine = Creature.load(VELA_DIR)

hash_mirro_before = mirro_spine._state_hash()
hash_vela_before = vela_spine._state_hash()

# ---------------------------------------------------------------------------
# BFS helpers (verbatim from exp230)
# ---------------------------------------------------------------------------

def build_bfs_dist(world: World, source_cell: int) -> list:
    """BFS distance field from source_cell over the 4-neighbour wall-clamped grid."""
    n = world.n_cells
    dist = [-1] * n
    dist[source_cell] = 0
    q: deque[int] = deque([source_cell])
    while q:
        cell = q.popleft()
        for a in range(4):
            nb = world.move(cell, a)
            if dist[nb] == -1:
                dist[nb] = dist[cell] + 1
                q.append(nb)
    return dist


def policy_action(
    world: World,
    dist: list,
    current: int,
    est: float,
    rng: np.random.Generator,
) -> int:
    """Comfort-gated epsilon-greedy BFS policy (verbatim from exp230).

    Draw order (for determinism):
      1. u = rng.random()  (epsilon check)
      2a. if u < EPS_POLICY: rng.integers(0, 4)  (random)
      2b. elif est < THRESH_GATE: rng.integers(0, 4)  (wander due to gate)
      2c. else: rng.integers(0, len(minimizers))  (greedy BFS, tie-break)
    """
    u = rng.random()
    if u < EPS_POLICY:
        return int(rng.integers(0, 4))
    if est < THRESH_GATE:
        return int(rng.integers(0, 4))
    best_d = min(dist[world.move(current, a)] for a in range(4))
    minimizers = [a for a in range(4) if dist[world.move(current, a)] == best_d]
    return int(minimizers[rng.integers(0, len(minimizers))])


def goal_next_dist(world: World, dist_fields_for_color: list, b_pos: int, n_cells: int) -> np.ndarray:
    """Compute the next-cell distribution P(next | b_pos, goal=c) for a given color's dist field.

    Uses eps-greedy BFS: prob (1-EPS_POLICY) on BFS-greedy minimizer cells (uniform over ties),
    prob EPS_POLICY spread uniformly over the 4 wall-clamped neighbor cells.
    """
    neighbors = [world.move(b_pos, a) for a in range(4)]

    # Greedy component: find BFS minimizers
    best_d = min(dist_fields_for_color[nb] for nb in neighbors)
    minimizers = [nb for nb in neighbors if dist_fields_for_color[nb] == best_d]

    # Base distribution: eps/4 per neighbor for the random component
    pred = np.zeros(n_cells)
    for nb in neighbors:
        pred[nb] += EPS_POLICY / 4.0

    # Greedy component: (1-eps) / |minimizers| for each minimizer neighbor
    weight = (1.0 - EPS_POLICY) / len(minimizers)
    for nb in minimizers:
        pred[nb] += weight

    return pred


# ---------------------------------------------------------------------------
# Main experiment setup (verbatim from exp230)
# ---------------------------------------------------------------------------

world = vela_spine.world
n_cells = world.n_cells

# Number of colors and color list
n_colors = len(set(world.cmap))
all_colors = sorted(set(world.cmap))  # e.g. [0, 1, 2]

# Pre-compute per-color source cells and dist fields
color_source_cells: dict[int, list[int]] = {}
color_dist_fields: dict[int, dict[int, list]] = {}
for c in all_colors:
    srcs = [i for i, col in enumerate(world.cmap) if col == c]
    color_source_cells[c] = srcs
    color_dist_fields[c] = {sc: build_bfs_dist(world, sc) for sc in srcs}

# Helper: get the nearest-source dist field for a given color at b_pos
def nearest_dist_for_color(c: int, b_pos: int) -> list:
    srcs = color_source_cells[c]
    dfs = color_dist_fields[c]
    nearest_src = min(srcs, key=lambda sc: dfs[sc][b_pos])
    return dfs[nearest_src]

# STATIONARY favorite color: argmax(vela.value_counts) = the most common color = color 2
# (consistent with spec: the initial favorite color, all_colors[0]-equivalent per spec note)
# We use argmax of value_counts to be robust; this is color 2 as per spec.
from collections import Counter
color_counts = Counter(world.cmap)
STATIONARY_GOAL_COLOR = max(color_counts, key=lambda c: color_counts[c])

# ---------------------------------------------------------------------------
# per-seed run function
# ---------------------------------------------------------------------------

def run_one(seed: int, period) -> dict:
    """Run one (seed, period) cell. period is int P or None (STATIONARY).

    Returns dict with m3a_overall, m15a_overall, q_current_mean, edge.
    The rng is seeded ONLY by seed, so same draws per step regardless of period
    (period only affects which goal_color is selected, not rng draw order).
    """
    rng = np.random.default_rng(seed)

    # Fork both spines; NEVER save them
    b_creature = vela_spine.fork(f"exp231-vela-fork-seed{seed}")
    a_creature = mirro_spine.fork(f"exp231-mirro-fork-seed{seed}")

    # --- Per-seed state ---

    # M1.5a: adaptive learned transition (float, decays each step)
    trans_a = np.zeros((n_cells, n_cells), dtype=np.float64)

    # M3a: adaptive goal posterior (uniform prior, with leak after Bayes update)
    q_goal = np.ones(n_colors) / n_colors  # indexed by all_colors list position

    # Comfort estimate for B's gate
    comfort_est = EST_INIT

    # Per-step accumulators: overall
    m15a_correct_total = 0
    m3a_correct_total = 0

    # Track q_goal[current_goal] each step for mean
    q_goal_current_sum = 0.0

    for step in range(H):
        b_pos = b_creature.true_pos

        # --- Determine current goal color for this step ---
        if period is None:
            # STATIONARY: fixed to the initial favorite color
            goal_color = STATIONARY_GOAL_COLOR
        else:
            # Cycling: goal_color(step) = all_colors[(step // P) % n_colors]
            goal_color = all_colors[(step // period) % n_colors]

        goal_color_idx = all_colors.index(goal_color)

        # Source cells for current goal
        source_cells_cur = color_source_cells[goal_color]

        # Nearest dist field for current goal color (for B's actual policy)
        dist_to_use = nearest_dist_for_color(goal_color, b_pos)

        # --- M1.5a: adaptive learned transition baseline ---
        # Predict BEFORE updating: (trans_a[cur] + 1.0) normalized
        row_a = trans_a[b_pos] + 1.0
        m15a_pred = row_a / row_a.sum()

        # --- M3a: adaptive goal-inference (ToM treatment) ---
        # Record q_goal[current_goal] BEFORE predicting (pre-update value)
        q_goal_current_sum += q_goal[goal_color_idx]

        # Compute per-color next-cell distributions at b_pos
        per_color_pred = []
        for ci, c in enumerate(all_colors):
            dist_for_c = nearest_dist_for_color(c, b_pos)
            pred_c = goal_next_dist(world, dist_for_c, b_pos, n_cells)
            per_color_pred.append(pred_c)

        # M3a prediction: marginalize over goal posterior
        m3a_pred = np.zeros(n_cells)
        for ci in range(n_colors):
            m3a_pred += q_goal[ci] * per_color_pred[ci]

        # --- B moves (the real policy) ---
        # Update comfort estimate (update BEFORE policy action, mirrors exp230)
        at_source = b_pos in source_cells_cur
        if at_source:
            comfort_est = (1 - ALPHA_EMA) * comfort_est + ALPHA_EMA * 1.0
        else:
            comfort_est += LAMBDA_RELAX * (1.0 - comfort_est)

        action = policy_action(world, dist_to_use, b_pos, comfort_est, rng)
        actual_next = world.move(b_pos, action)

        # --- Update M3a q_goal AFTER observing B's move (no leakage) ---
        lik = np.zeros(n_colors)
        for ci in range(n_colors):
            lik[ci] = per_color_pred[ci][actual_next] + 1e-9  # floor to avoid zeros
        q_goal = q_goal * lik
        q_sum = q_goal.sum()
        if q_sum > 0:
            q_goal = q_goal / q_sum
        else:
            q_goal = np.ones(n_colors) / n_colors  # reset on degenerate case

        # Apply forgetting leak AFTER Bayes update
        q_goal = Q_LEAK_RETAIN * q_goal + Q_LEAK_UNIFORM * (1.0 / n_colors)
        # Re-normalize (guard float drift)
        q_goal = q_goal / q_goal.sum()

        # --- Update M1.5a adaptive transition AFTER B moves ---
        # Decay first, then add new observation
        trans_a *= TRANS_DECAY
        trans_a[b_pos, actual_next] += 1.0

        # Update B's true position
        b_creature.true_pos = actual_next

        # --- Score predictors ---
        m15a_argmax = int(np.argmax(m15a_pred))
        m3a_argmax = int(np.argmax(m3a_pred))

        m15a_hit = int(m15a_argmax == actual_next)
        m3a_hit = int(m3a_argmax == actual_next)

        m15a_correct_total += m15a_hit
        m3a_correct_total += m3a_hit

    # Per-seed accuracy
    m15a_overall = m15a_correct_total / H
    m3a_overall = m3a_correct_total / H
    edge = m3a_overall - m15a_overall
    q_current_mean = q_goal_current_sum / H

    return {
        "seed": seed,
        "m15a_overall": m15a_overall,
        "m3a_overall": m3a_overall,
        "edge": edge,
        "q_current_mean": q_current_mean,
    }


# ---------------------------------------------------------------------------
# SWEEP: run all (period, seed) cells
# ---------------------------------------------------------------------------

# period_results[i] = list of 8 per-seed dicts for PERIODS[i]
period_results: list[list[dict]] = []

for period in PERIODS:
    seed_dicts = []
    for seed in SEEDS:
        result = run_one(seed, period)
        seed_dicts.append(result)
    period_results.append(seed_dicts)

# ---------------------------------------------------------------------------
# Spine integrity check (F4)
# ---------------------------------------------------------------------------

hash_mirro_after = mirro_spine._state_hash()
hash_vela_after = vela_spine._state_hash()

f4_mirro_ok = hash_mirro_before == hash_mirro_after
f4_vela_ok = hash_vela_before == hash_vela_after
f4_ok = f4_mirro_ok and f4_vela_ok

# ---------------------------------------------------------------------------
# Per-P aggregates
# ---------------------------------------------------------------------------

# Compute per-P means
per_p_mean_edge = []
per_p_mean_m3a = []
per_p_mean_m15a = []
per_p_mean_q = []

for i, period in enumerate(PERIODS):
    dicts = period_results[i]
    mean_edge = float(np.mean([d["edge"] for d in dicts]))
    mean_m3a = float(np.mean([d["m3a_overall"] for d in dicts]))
    mean_m15a = float(np.mean([d["m15a_overall"] for d in dicts]))
    mean_q = float(np.mean([d["q_current_mean"] for d in dicts]))
    per_p_mean_edge.append(mean_edge)
    per_p_mean_m3a.append(mean_m3a)
    per_p_mean_m15a.append(mean_m15a)
    per_p_mean_q.append(mean_q)

# Edge sequence E = [edge(100), edge(200), edge(400), edge(800), edge(STAT)]
E = per_p_mean_edge  # length 5, same order as PERIODS

# ---------------------------------------------------------------------------
# Evaluate conjuncts F1/F2/F3/F4
# ---------------------------------------------------------------------------

# (a) Monotone non-increasing in P with tolerance 0.01:
# For each adjacent pair (E[i], E[i+1]), check E[i+1] <= E[i] + 0.01
# (no increase > 0.01 as P grows, i.e., as change frequency decreases)
adjacent_deltas = [E[i + 1] - E[i] for i in range(len(E) - 1)]  # positive = increase as P grows
monotone_violations = [(i, delta) for i, delta in enumerate(adjacent_deltas) if delta > 0.01]
f1_fired = len(monotone_violations) > 0  # F1: some adjacent pair increases by > 0.01

# (b) Magnitude: edge(100) - edge(STAT) >= 0.05
magnitude = E[0] - E[4]  # edge at P=100 minus edge at STATIONARY
f2_fired = magnitude < 0.05

# (c) min over P of mean q_current >= 0.60
min_q = min(per_p_mean_q)
f3_fired = min_q < 0.60

# ---------------------------------------------------------------------------
# Build output string
# ---------------------------------------------------------------------------

lines = []
lines.append("=" * 100)
lines.append("Exp 231 — self-other-modeling Rung 2-sweep: does goal-inference edge GROW with change rate?")
lines.append("=" * 100)
lines.append("")
lines.append("Implementation notes:")
lines.append("  - Shared world: vela's world (vela-fork is B, lives here natively)")
lines.append(f"  - n_cells = {n_cells}, n_colors = {n_colors}, all_colors = {all_colors}")
lines.append(f"  - STATIONARY_GOAL_COLOR = {STATIONARY_GOAL_COLOR} (argmax of vela color_counts = {dict(color_counts)})")
lines.append(f"  - SWEEP periods P: {PERIOD_LABELS} (None=STATIONARY)")
lines.append(f"  - For finite P: goal_color(step) = all_colors[(step//P) % n_colors]")
lines.append(f"  - For STATIONARY: goal_color = {STATIONARY_GOAL_COLOR} (fixed every step)")
lines.append(f"  - EPS_POLICY = {EPS_POLICY}, ALPHA_EMA = {ALPHA_EMA}, THRESH_GATE = {THRESH_GATE}")
lines.append(f"  - LAMBDA_RELAX = {LAMBDA_RELAX}, EST_INIT = {EST_INIT}")
lines.append(f"  - M1.5a: adaptive trans float, decay={TRANS_DECAY} per step, +1 Laplace predict (BINDING control)")
lines.append(f"  - M3a: q_goal uniform init; PREDICT then observe then Bayes update, then leak:")
lines.append(f"         q_goal = {Q_LEAK_RETAIN}*q_goal + {Q_LEAK_UNIFORM}*(1/{n_colors} each), re-normalized")
lines.append(f"  - q_goal[current_goal] tracked at PREDICT time (pre-update, no leakage)")
lines.append(f"  - Seeds are PAIRED across P: same rng(seed) draws regardless of period (rng draw order is identical)")
lines.append(f"  - P=400 cell corresponds directly to Exp 230's result")
lines.append(f"  - H = {H} steps per (seed, period) cell; {len(PERIODS)} x {len(SEEDS)} = {len(PERIODS)*len(SEEDS)} total runs")
lines.append("")

# Per-P summary table
lines.append("--- PER-P SUMMARY TABLE ---")
lines.append("")
hdr = f"{'P':>6}  {'mean_edge':>10}  {'mean_M3a':>10}  {'mean_M1.5a':>10}  {'mean_q[cur]':>12}"
lines.append(hdr)
lines.append("-" * 60)
for i, label in enumerate(PERIOD_LABELS):
    row = (
        f"  {label:>4}  {per_p_mean_edge[i]:>10.5f}  {per_p_mean_m3a[i]:>10.5f}  "
        f"{per_p_mean_m15a[i]:>10.5f}  {per_p_mean_q[i]:>12.5f}"
    )
    lines.append(row)
lines.append("")

# Per-seed edges per P
lines.append("--- PER-SEED EDGES PER P (spread visible) ---")
lines.append("")
seed_hdr = f"{'P':>6}  " + "  ".join(f"s{s:>1}" for s in SEEDS)
lines.append(seed_hdr)
lines.append("-" * 80)
for i, label in enumerate(PERIOD_LABELS):
    edges_str = "  ".join(f"{period_results[i][s]['edge']:>6.4f}" for s in range(len(SEEDS)))
    lines.append(f"  {label:>4}  {edges_str}")
lines.append("")

# Per-seed m3a_overall per P
lines.append("--- PER-SEED M3a_OVERALL PER P ---")
lines.append("")
lines.append(seed_hdr)
lines.append("-" * 80)
for i, label in enumerate(PERIOD_LABELS):
    vals_str = "  ".join(f"{period_results[i][s]['m3a_overall']:>6.4f}" for s in range(len(SEEDS)))
    lines.append(f"  {label:>4}  {vals_str}")
lines.append("")

# Per-seed m15a_overall per P
lines.append("--- PER-SEED M1.5a_OVERALL PER P ---")
lines.append("")
lines.append(seed_hdr)
lines.append("-" * 80)
for i, label in enumerate(PERIOD_LABELS):
    vals_str = "  ".join(f"{period_results[i][s]['m15a_overall']:>6.4f}" for s in range(len(SEEDS)))
    lines.append(f"  {label:>4}  {vals_str}")
lines.append("")

# Per-seed q_current_mean per P
lines.append("--- PER-SEED q_CURRENT_MEAN PER P ---")
lines.append("")
lines.append(seed_hdr)
lines.append("-" * 80)
for i, label in enumerate(PERIOD_LABELS):
    vals_str = "  ".join(f"{period_results[i][s]['q_current_mean']:>6.4f}" for s in range(len(SEEDS)))
    lines.append(f"  {label:>4}  {vals_str}")
lines.append("")

# Edge sequence
lines.append("--- EDGE SEQUENCE E = [edge(100), edge(200), edge(400), edge(800), edge(STAT)] ---")
lines.append("")
e_str = ", ".join(f"{e:.5f}" for e in E)
lines.append(f"  E = [{e_str}]")
lines.append("")

# Monotonicity check (adjacent deltas)
lines.append("--- MONOTONICITY CHECK (adjacent deltas as P grows, tolerance 0.01) ---")
lines.append("")
lines.append("  For the law to hold (F1 NOT FIRED), each delta must be <= +0.01.")
lines.append("  delta[i] = E[i+1] - E[i]  (positive = edge INCREASES as P grows = violation)")
lines.append("")
for i, delta in enumerate(adjacent_deltas):
    pair_label = f"edge({PERIOD_LABELS[i]}) -> edge({PERIOD_LABELS[i+1]})"
    violation = " *** VIOLATION (delta > 0.01)" if delta > 0.01 else ""
    lines.append(f"  delta[{i}]: {pair_label:>30}  delta = {delta:>+.5f}{violation}")
lines.append("")

# Magnitude check
lines.append("--- MAGNITUDE CHECK (F2) ---")
lines.append("")
lines.append(f"  edge(100) - edge(STAT) = {E[0]:.5f} - {E[4]:.5f} = {magnitude:.5f}  (threshold: >= 0.05)")
lines.append(f"  F2 {'NOT FIRED' if not f2_fired else 'FIRED'}: magnitude {'>=0.05 (law has real size)' if not f2_fired else '<0.05 (law is FLAT)'}")
lines.append("")

# Min q_current
lines.append("--- MIN q_CURRENT CHECK (F3) ---")
lines.append("")
for i, label in enumerate(PERIOD_LABELS):
    marker = " <-- MIN" if per_p_mean_q[i] == min_q else ""
    lines.append(f"  P={label:>4}: mean_q_current = {per_p_mean_q[i]:.5f}{marker}")
lines.append(f"  min over all P = {min_q:.5f}  (threshold: >= 0.60)")
lines.append(f"  F3 {'NOT FIRED' if not f3_fired else 'FIRED'}: min_q {'>=0.60' if not f3_fired else '<0.60 at some P (instrument fails at that rate)'}")
lines.append("")

# Spine integrity
lines.append("--- SPINE INTEGRITY (F4) ---")
lines.append("")
lines.append(f"  mirro before: {hash_mirro_before[:24]}...")
lines.append(f"  mirro after:  {hash_mirro_after[:24]}...")
lines.append(f"  mirro intact: {'YES' if f4_mirro_ok else 'NO --- F4 FIRED'}")
lines.append(f"  vela before:  {hash_vela_before[:24]}...")
lines.append(f"  vela after:   {hash_vela_after[:24]}...")
lines.append(f"  vela intact:  {'YES' if f4_vela_ok else 'NO --- F4 FIRED'}")
lines.append(f"  F4: {'NOT FIRED' if f4_ok else 'FIRED (continuity bug)'}")
lines.append("")

# Falsifier evaluation
lines.append("--- FALSIFIER EVALUATION ---")
lines.append("")
lines.append(
    f"  F1 (edge NOT monotone non-increasing in P): "
    f"{'NOT FIRED' if not f1_fired else 'FIRED'}  "
    f"[violations: {len(monotone_violations)}/{len(adjacent_deltas)} adjacent pairs increase by > 0.01]"
)
if f1_fired:
    for i, delta in monotone_violations:
        lines.append(
            f"       violation at {PERIOD_LABELS[i]}->{PERIOD_LABELS[i+1]}: delta={delta:+.5f} > 0.01"
        )
lines.append(
    f"  F2 (magnitude edge(100)-edge(STAT) < 0.05): "
    f"{'NOT FIRED' if not f2_fired else 'FIRED'}  "
    f"[magnitude={magnitude:.5f} (need>=0.05)]"
)
lines.append(
    f"  F3 (min q_current < 0.60 at some P): "
    f"{'NOT FIRED' if not f3_fired else 'FIRED'}  "
    f"[min_q={min_q:.5f} (need>=0.60)]"
)
lines.append(
    f"  F4 (spine hash changed): "
    f"{'NOT FIRED' if f4_ok else 'FIRED'}  "
    f"[mirro {'OK' if f4_mirro_ok else 'CHANGED'}, vela {'OK' if f4_vela_ok else 'CHANGED'}]"
)
lines.append("")

# Prediction conjunct evaluation
lines.append("--- PREDICTION CONJUNCT EVALUATION ---")
lines.append("")
pred_a = not f1_fired
pred_b = magnitude >= 0.05
pred_c = not f3_fired
lines.append(
    f"  (a) edge sequence monotone non-increasing (tol=0.01):  {'TRUE' if pred_a else 'FALSE'}  "
    f"[{len(monotone_violations)} violations]"
)
lines.append(
    f"  (b) edge(100)-edge(STAT) >= 0.05 (real magnitude):    {'TRUE' if pred_b else 'FALSE'}  "
    f"[magnitude={magnitude:.5f}]"
)
lines.append(
    f"  (c) min q_current >= 0.60 at every P:                 {'TRUE' if pred_c else 'FALSE'}  "
    f"[min_q={min_q:.5f}]"
)
lines.append("")

# Determine verdict
if not f4_ok:
    verdict = "VERDICT: HALT/F4 — spine integrity violated; continuity bug"
elif f1_fired and f2_fired:
    verdict = (
        f"VERDICT: FALSE/F1+F2 — edge is non-monotone in P (F1: {len(monotone_violations)} violation(s)) "
        f"AND law is flat (F2: magnitude={magnitude:.5f} < 0.05); "
        f"'ToM pays in proportion to non-stationarity' law is FALSE"
    )
elif f1_fired:
    verdict = (
        f"VERDICT: FALSE/F1 — edge is NOT monotone non-increasing in P "
        f"({len(monotone_violations)} violation(s), max delta={max(d for _, d in monotone_violations):+.5f}); "
        f"law shape is false (relationship non-monotone)"
    )
elif f2_fired:
    verdict = (
        f"VERDICT: FALSE/F2 — law is FLAT: edge(100)-edge(STAT)={magnitude:.5f} < 0.05; "
        f"change-rate does not materially modulate the ToM edge (direction may hold but size is trivial)"
    )
elif pred_a and pred_b and pred_c:
    verdict = (
        f"VERDICT: TRUE — all conjuncts hold; "
        f"edge sequence monotone non-increasing (0 violations, max_delta={max(adjacent_deltas):+.5f}); "
        f"magnitude={magnitude:.5f} >= 0.05 (law has real size); "
        f"min_q={min_q:.5f} >= 0.60 (instrument tracks at all rates); "
        f"'ToM edge GROWS with change frequency' LAW CONFIRMED"
    )
elif not pred_c:
    verdict = (
        f"VERDICT: NO_VERDICT/F3 — goal posterior fails to track at some change rate "
        f"(min_q={min_q:.5f} < 0.60); instrument limit reached; "
        f"law validity bounded to rates where tracking succeeds"
    )
else:
    failed = [lbl for lbl, v in [("a", pred_a), ("b", pred_b), ("c", pred_c)] if not v]
    verdict = (
        f"VERDICT: FALSE — conjunct(s) {failed} not met; "
        f"'ToM pays in proportion to non-stationarity' law not established"
    )

lines.append(verdict)
lines.append("")
lines.append("=" * 100)

output_text = "\n".join(lines)

# ---------------------------------------------------------------------------
# Print and write
# ---------------------------------------------------------------------------

print(output_text)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(output_text + "\n")

print(f"\nOutput written to: {OUTPUT_PATH}")

# Halt on F4
if not f4_ok:
    print("HALT: F4 fired — spine integrity violated")
    sys.exit(1)
