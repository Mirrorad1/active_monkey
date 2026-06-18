"""Exp 229 — self-other-modeling Rung 2: latent-goal inference vs a LEARNED-transition baseline (the binding control).

Hypothesis: a modeler that INFERS B's latent goal (which color B seeks) from B's trajectory via a PROVIDED
goal-directed-policy model, and predicts B's next cell by goal-marginalization, beats a LEARNED empirical
first-order transition baseline P(B_next|B_current) ESPECIALLY EARLY (sample efficiency: the goal posterior
concentrates in a few steps and generalizes across cells, while the transition model must observe each cell's
transitions) — a functional theory-of-mind benefit. If the learned-transition baseline instead matches the
goal-inferrer, then 'learning B's dynamics' suffices and there is NO ToM benefit at this scale (the
trivial-provision concern from Exp 228, realized).

Setup: the Exp 228 substrate (vela-fork B goal-directed via comfort-gated eps-greedy BFS toward favorite
color=2; mirro-fork A observes B's cell each step; forks NEVER saved). H=2000 steps, seeds 0-7. Five one-step
predictors of B's NEXT cell, scored by argmax accuracy:
  M0 occupancy marginal; M1 position-tracker (innate random-walk diffusion, Exp 228 baseline);
  M1.5 LEARNED transition (online counts trans[current, next] += 1, predict row P(next|current) with +1 smoothing)
       — THE BINDING non-ToM control (learns B's dynamics, no goal latent);
  M3 goal-inference (the ToM treatment): q_goal = posterior over the 3 colors, started UNIFORM (1/3 each),
     updated each step by the likelihood of B's OBSERVED move under the PROVIDED policy P(move | pos, goal=c) =
     eps-greedy BFS toward the nearest color-c cell (eps 0.2, same form A is given; the GOAL color c is NOT
     provided — only the policy form is). Predict next cell = sum_c q_goal[c] * P(next | pos, BFS-toward-c).
  M2 goal-oracle (knows the true goal; upper bound, Exp 228).
Windows: EARLY = first 100 steps; OVERALL = all 2000.

Prediction (ToM benefit TRUE): (a) mean(M3_early_acc - M1.5_early_acc) >= 0.05 AND in >=6/8 seeds; AND
(b) M3_overall_acc >= M1.5_overall_acc - 0.02 (not worse overall); AND (c) q_goal concentrates on the true
goal: mean over the run of q_goal[true_goal=2] >= 0.60.

Falsifier (NO ToM benefit / FALSE):
  F1: mean(M3_early - M1.5_early) < 0.05 OR the per-seed count (M3_early>M1.5_early) < 6/8 -> no sample-efficiency
      edge: learning B's dynamics suffices, NO ToM benefit at this scale (trivial-provision concern realized).
  F2: mean q_goal[true_goal] < 0.60 -> goal-inference did not concentrate (broken/uninformative instrument; NO_VERDICT).
  F3: M3_overall < M1.5_overall - 0.02 -> the goal model HURTS asymptotically (misspecification dominates).
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
EARLY_WINDOW = 100                # first 100 steps for EARLY metric
SEEDS = list(range(8))            # seeds 0-7
MIRRO_DIR = Path("creature/state/mirro")
VELA_DIR = Path("creature/state/vela")
M2_EPSILON = 1e-6                 # smoothing for M2 log-loss
OUTPUT_PATH = Path("experiments/outputs/exp229.txt")

# Comfort-gate parameters (reuse exp228/exp70 constants)
EPS_POLICY = 0.2                  # epsilon-greedy exploration
ALPHA_EMA = 0.1                   # comfort-EMA alpha
THRESH_GATE = 0.75                # comfort gate threshold
LAMBDA_RELAX = 0.01               # away-from-source EMA relaxation toward 1.0
EST_INIT = 1.0                    # initial comfort estimate

# ---------------------------------------------------------------------------
# Load committed spines (read-only — NEVER call .live() or .save() on these)
# ---------------------------------------------------------------------------

mirro_spine = Creature.load(MIRRO_DIR)
vela_spine = Creature.load(VELA_DIR)

hash_mirro_before = mirro_spine._state_hash()
hash_vela_before = vela_spine._state_hash()

# ---------------------------------------------------------------------------
# BFS helpers (verbatim from exp228)
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
    """Comfort-gated epsilon-greedy BFS policy (mirrors exp228/exp70's policy_action).

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
    """Compute the next-cell distribution P(next | b_pos, goal=c) for a given color's dist_fields.

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
# Main experiment loop
# ---------------------------------------------------------------------------

# Use VELA's world as the shared world (B lives here natively) — same as exp228
world = vela_spine.world
n_cells = world.n_cells

# Innate transition matrix for M1: B_innate[s', s, a] = 1 if move(s,a)==s'
B_innate = world.transition_matrix()  # shape (n_cells, n_cells, 4)

# B's favorite-color: argmax(value_counts); should be color 2
fav_color = int(np.argmax(vela_spine.value_counts))
source_cells_fav = [i for i, c in enumerate(world.cmap) if c == fav_color]

# Dist fields for BFS toward nearest source of the true goal color (for B's actual movement)
dist_fields_fav = {sc: build_bfs_dist(world, sc) for sc in source_cells_fav}

# Number of colors in world (should be 3)
n_colors = len(set(world.cmap))
all_colors = sorted(set(world.cmap))  # e.g. [0, 1, 2]

# Pre-compute per-color dist fields for M3: for each color c, list of source cells
# and combined dist field toward nearest cell of that color (we need per-source-cell
# dist fields, then pick nearest at each step)
color_source_cells: dict[int, list[int]] = {}
color_dist_fields: dict[int, dict[int, list]] = {}
for c in all_colors:
    srcs = [i for i, col in enumerate(world.cmap) if col == c]
    color_source_cells[c] = srcs
    color_dist_fields[c] = {sc: build_bfs_dist(world, sc) for sc in srcs}

true_goal = fav_color  # = 2

seed_results = []  # list of dicts, one per seed

for seed in SEEDS:
    rng = np.random.default_rng(seed)

    # Fork both spines; never save them
    b_creature = vela_spine.fork(f"exp229-vela-fork-seed{seed}")
    a_creature = mirro_spine.fork(f"exp229-mirro-fork-seed{seed}")

    # --- Per-seed state ---

    # M0: running histogram of B's actual positions (smoothed with +1 uniform prior)
    pos_histogram = np.ones(n_cells)  # +1 smoothing

    # M1.5: learned first-order transition counts
    trans = np.zeros((n_cells, n_cells), dtype=np.int64)

    # M3: goal posterior (uniform prior over all colors)
    q_goal = np.ones(n_colors) / n_colors  # indexed by all_colors list position

    # Comfort estimate for B's gate
    comfort_est = EST_INIT

    # Per-step accumulators
    m0_correct_early = 0
    m1_correct_early = 0
    m15_correct_early = 0
    m3_correct_early = 0
    m2_correct_early = 0

    m0_correct_total = 0
    m1_correct_total = 0
    m15_correct_total = 0
    m3_correct_total = 0
    m2_correct_total = 0

    # q_goal[true_goal] tracking: sum over all steps for mean
    q_goal_true_sum = 0.0

    for step in range(H):
        b_pos = b_creature.true_pos

        # Update M0 histogram with current observed position BEFORE predicting
        pos_histogram[b_pos] += 1.0

        # --- M0: normalize running histogram -> distribution over next cell ---
        m0_pred = pos_histogram / pos_histogram.sum()

        # --- M1: one-hot at current b_pos, advance through uniform random walk ---
        m1_pred = B_innate[:, b_pos, :].mean(axis=1)  # shape (n_cells,)

        # --- M1.5: learned transition baseline ---
        # Predict BEFORE updating (no leakage): +1 Laplace smoothing
        row = trans[b_pos].astype(float) + 1.0
        m15_pred = row / row.sum()

        # --- M3: goal-inference (ToM treatment) ---
        # PREDICT with CURRENT q_goal (used only moves up to t-1)
        # Accumulate q_goal[true_goal] BEFORE updating q_goal
        true_goal_idx = all_colors.index(true_goal)
        q_goal_true_sum += q_goal[true_goal_idx]

        # Compute per-color next-cell distributions at b_pos
        per_color_pred = []
        for ci, c in enumerate(all_colors):
            # Nearest source of color c to b_pos
            srcs = color_source_cells[c]
            dfs = color_dist_fields[c]
            nearest_src_c = min(srcs, key=lambda sc: dfs[sc][b_pos])
            dist_for_c = dfs[nearest_src_c]
            pred_c = goal_next_dist(world, dist_for_c, b_pos, n_cells)
            per_color_pred.append(pred_c)

        # M3 prediction: marginalize over goal posterior
        m3_pred = np.zeros(n_cells)
        for ci in range(n_colors):
            m3_pred += q_goal[ci] * per_color_pred[ci]

        # --- M2: goal-oracle (knows true goal, uses BFS-greedy near-one-hot) ---
        nearest_src_fav = min(source_cells_fav, key=lambda sc: dist_fields_fav[sc][b_pos])
        dist_to_use = dist_fields_fav[nearest_src_fav]
        best_d = min(dist_to_use[world.move(b_pos, a)] for a in range(4))
        minimizers = [a for a in range(4) if dist_to_use[world.move(b_pos, a)] == best_d]
        m2_pred = np.full(n_cells, M2_EPSILON / n_cells)
        weight = (1.0 - M2_EPSILON) / len(minimizers)
        for a in minimizers:
            nc = world.move(b_pos, a)
            m2_pred[nc] += weight

        # --- B moves (the real policy) ---
        # Update comfort estimate (mirrors exp228: update BEFORE policy action)
        at_source = b_pos in source_cells_fav
        if at_source:
            comfort_est = (1 - ALPHA_EMA) * comfort_est + ALPHA_EMA * 1.0
        else:
            comfort_est += LAMBDA_RELAX * (1.0 - comfort_est)

        action = policy_action(world, dist_to_use, b_pos, comfort_est, rng)
        actual_next = world.move(b_pos, action)

        # --- Update M3 q_goal AFTER observing B's move (no leakage) ---
        lik = np.zeros(n_colors)
        for ci in range(n_colors):
            # Likelihood of actual_next given pos=b_pos and goal=all_colors[ci]
            lik[ci] = per_color_pred[ci][actual_next] + 1e-9  # floor to avoid zeros
        q_goal = q_goal * lik
        q_goal_sum = q_goal.sum()
        if q_goal_sum > 0:
            q_goal = q_goal / q_goal_sum
        else:
            q_goal = np.ones(n_colors) / n_colors  # reset on degenerate case

        # --- Update M1.5 transition counts AFTER B moves ---
        b_pos_prev = b_pos
        trans[b_pos_prev, actual_next] += 1

        # Update B's true position
        b_creature.true_pos = actual_next

        # --- Score predictors ---
        m0_argmax = int(np.argmax(m0_pred))
        m1_argmax = int(np.argmax(m1_pred))
        m15_argmax = int(np.argmax(m15_pred))
        m3_argmax = int(np.argmax(m3_pred))
        m2_argmax = int(np.argmax(m2_pred))

        m0_hit = int(m0_argmax == actual_next)
        m1_hit = int(m1_argmax == actual_next)
        m15_hit = int(m15_argmax == actual_next)
        m3_hit = int(m3_argmax == actual_next)
        m2_hit = int(m2_argmax == actual_next)

        if step < EARLY_WINDOW:
            m0_correct_early += m0_hit
            m1_correct_early += m1_hit
            m15_correct_early += m15_hit
            m3_correct_early += m3_hit
            m2_correct_early += m2_hit

        m0_correct_total += m0_hit
        m1_correct_total += m1_hit
        m15_correct_total += m15_hit
        m3_correct_total += m3_hit
        m2_correct_total += m2_hit

    # Per-seed results
    m0_early_acc = m0_correct_early / EARLY_WINDOW
    m1_early_acc = m1_correct_early / EARLY_WINDOW
    m15_early_acc = m15_correct_early / EARLY_WINDOW
    m3_early_acc = m3_correct_early / EARLY_WINDOW
    m2_early_acc = m2_correct_early / EARLY_WINDOW

    m0_overall_acc = m0_correct_total / H
    m1_overall_acc = m1_correct_total / H
    m15_overall_acc = m15_correct_total / H
    m3_overall_acc = m3_correct_total / H
    m2_overall_acc = m2_correct_total / H

    mean_q_goal_true = q_goal_true_sum / H

    seed_results.append({
        "seed": seed,
        "m0_early": m0_early_acc,
        "m1_early": m1_early_acc,
        "m15_early": m15_early_acc,
        "m3_early": m3_early_acc,
        "m2_early": m2_early_acc,
        "m0_overall": m0_overall_acc,
        "m1_overall": m1_overall_acc,
        "m15_overall": m15_overall_acc,
        "m3_overall": m3_overall_acc,
        "m2_overall": m2_overall_acc,
        "mean_q_goal_true": mean_q_goal_true,
        "early_edge": m3_early_acc - m15_early_acc,
        "m3_early_beats_m15": m3_early_acc > m15_early_acc,
    })

# ---------------------------------------------------------------------------
# Spine integrity check (F4)
# ---------------------------------------------------------------------------

hash_mirro_after = mirro_spine._state_hash()
hash_vela_after = vela_spine._state_hash()

f4_mirro_ok = hash_mirro_before == hash_mirro_after
f4_vela_ok = hash_vela_before == hash_vela_after
f4_ok = f4_mirro_ok and f4_vela_ok

# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------

mean_early_edge = float(np.mean([r["early_edge"] for r in seed_results]))
count_m3_beats_m15_early = sum(1 for r in seed_results if r["m3_early_beats_m15"])
mean_m3_overall = float(np.mean([r["m3_overall"] for r in seed_results]))
mean_m15_overall = float(np.mean([r["m15_overall"] for r in seed_results]))
mean_m0_overall = float(np.mean([r["m0_overall"] for r in seed_results]))
mean_m1_overall = float(np.mean([r["m1_overall"] for r in seed_results]))
mean_m2_overall = float(np.mean([r["m2_overall"] for r in seed_results]))
mean_q_goal_true_all = float(np.mean([r["mean_q_goal_true"] for r in seed_results]))

mean_m3_early = float(np.mean([r["m3_early"] for r in seed_results]))
mean_m15_early = float(np.mean([r["m15_early"] for r in seed_results]))
mean_m2_early = float(np.mean([r["m2_early"] for r in seed_results]))

# ---------------------------------------------------------------------------
# Verdict evaluation
# ---------------------------------------------------------------------------

# Prediction conjuncts
pred_a = (mean_early_edge >= 0.05) and (count_m3_beats_m15_early >= 6)
pred_b = mean_m3_overall >= mean_m15_overall - 0.02
pred_c = mean_q_goal_true_all >= 0.60

# Falsifiers
f1_fired = not ((mean_early_edge >= 0.05) and (count_m3_beats_m15_early >= 6))
f2_fired = mean_q_goal_true_all < 0.60
f3_fired = mean_m3_overall < mean_m15_overall - 0.02
# F4 evaluated above

# ---------------------------------------------------------------------------
# Build output string
# ---------------------------------------------------------------------------

lines = []
lines.append("=" * 80)
lines.append("Exp 229 — self-other-modeling Rung 2: latent-goal inference vs LEARNED-transition baseline")
lines.append("=" * 80)
lines.append("")
lines.append("Implementation notes:")
lines.append("  - Shared world: vela's world (vela-fork is B, lives here natively)")
lines.append(f"  - B's favorite color (true goal): argmax(vela.value_counts) = color {true_goal}")
lines.append(f"  - Source cells for true goal: {source_cells_fav}")
lines.append(f"  - Number of colors: {n_colors}, all_colors = {all_colors}")
lines.append(f"  - EPS_POLICY = {EPS_POLICY}, ALPHA_EMA = {ALPHA_EMA}, THRESH_GATE = {THRESH_GATE}")
lines.append(f"  - LAMBDA_RELAX = {LAMBDA_RELAX}, EST_INIT = {EST_INIT}")
lines.append("  - M1.5: online trans counts +1 Laplace, predict BEFORE update, update AFTER B moves")
lines.append("  - M3: q_goal uniform init (1/3 each); PREDICT then observe then update (no leakage)")
lines.append("  - M3 per-color policy: eps-greedy toward nearest source of that color (EPS=0.2)")
lines.append("  - BFS tie-break: uniform over minimizer neighbor cells for both M2 and M3")
lines.append("  - At-source behavior: BFS dist=0 at source -> best_d = min over 4 wall-clamped neighbors")
lines.append("    (may include source itself if wall clamped); same logic for both M2 and M3 per-color dist")
lines.append("  - q_goal floor: 1e-9 added to each likelihood to avoid collapse")
lines.append("  - M0 initialized with +1 smoothing (uniform prior)")
lines.append("  - EARLY = first 100 steps; OVERALL = all 2000 steps")
lines.append(f"  - M2_EPSILON (oracle smoothing) = {M2_EPSILON}")
lines.append("")

lines.append("--- PER-SEED TABLE ---")
lines.append("")
hdr = (
    f"{'seed':>4}  {'M15_e':>7}  {'M3_e':>7}  {'edge_e':>7}  "
    f"{'M15_ov':>7}  {'M3_ov':>7}  {'q[true]':>8}  {'M3e>M15e':>9}"
)
lines.append(hdr)
lines.append("-" * 78)
for r in seed_results:
    row = (
        f"  {r['seed']:>2}  {r['m15_early']:>7.4f}  {r['m3_early']:>7.4f}  {r['early_edge']:>7.4f}  "
        f"{r['m15_overall']:>7.4f}  {r['m3_overall']:>7.4f}  {r['mean_q_goal_true']:>8.4f}  "
        f"{'YES' if r['m3_early_beats_m15'] else 'NO':>9}"
    )
    lines.append(row)
lines.append("-" * 78)
lines.append(
    f"  {'mean':>4}  {mean_m15_early:>7.4f}  {mean_m3_early:>7.4f}  {mean_early_edge:>7.4f}  "
    f"{mean_m15_overall:>7.4f}  {mean_m3_overall:>7.4f}  {mean_q_goal_true_all:>8.4f}"
)
lines.append("")

lines.append("--- FULL PREDICTOR ACCURACY TABLE ---")
lines.append("")
hdr2 = (
    f"{'seed':>4}  {'M0_e':>7}  {'M1_e':>7}  {'M15_e':>7}  {'M3_e':>7}  {'M2_e':>7}  "
    f"{'M0_ov':>7}  {'M1_ov':>7}  {'M15_ov':>7}  {'M3_ov':>7}  {'M2_ov':>7}"
)
lines.append(hdr2)
lines.append("-" * 100)
for r in seed_results:
    row2 = (
        f"  {r['seed']:>2}  {r['m0_early']:>7.4f}  {r['m1_early']:>7.4f}  {r['m15_early']:>7.4f}  "
        f"{r['m3_early']:>7.4f}  {r['m2_early']:>7.4f}  "
        f"{r['m0_overall']:>7.4f}  {r['m1_overall']:>7.4f}  {r['m15_overall']:>7.4f}  "
        f"{r['m3_overall']:>7.4f}  {r['m2_overall']:>7.4f}"
    )
    lines.append(row2)
lines.append("-" * 100)
lines.append(
    f"  {'mean':>4}  {mean_m0_overall:>7.4f}  {mean_m1_overall:>7.4f}  {mean_m15_overall:>7.4f}  "
    f"{mean_m3_overall:>7.4f}  {mean_m2_overall:>7.4f}  (overall means)"
)
lines.append("")

lines.append("--- AGGREGATES ---")
lines.append("")
lines.append(f"  mean(M3_early - M1.5_early):   {mean_early_edge:.4f}  (threshold for pred_a: >=0.05)")
lines.append(f"  count(M3_early > M1.5_early):  {count_m3_beats_m15_early}/8  (threshold for pred_a: >=6/8)")
lines.append(f"  mean M3_overall:               {mean_m3_overall:.4f}")
lines.append(f"  mean M1.5_overall:             {mean_m15_overall:.4f}")
lines.append(f"  M3_overall - M1.5_overall:     {mean_m3_overall - mean_m15_overall:.4f}  (pred_b threshold: >=-0.02)")
lines.append(f"  mean q_goal[true_goal=2]:      {mean_q_goal_true_all:.4f}  (pred_c threshold: >=0.60)")
lines.append(f"  mean M0_overall:               {mean_m0_overall:.4f}")
lines.append(f"  mean M1_overall:               {mean_m1_overall:.4f}")
lines.append(f"  mean M2_overall (oracle):      {mean_m2_overall:.4f}")
lines.append("")

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

lines.append("--- FALSIFIER EVALUATION ---")
lines.append("")
lines.append(
    f"  F1 (no sample-efficiency edge): "
    f"{'NOT FIRED' if not f1_fired else 'FIRED'}  "
    f"[mean_early_edge={mean_early_edge:.4f} (need>=0.05), count={count_m3_beats_m15_early}/8 (need>=6)]"
)
lines.append(
    f"  F2 (goal inference did not concentrate): "
    f"{'NOT FIRED' if not f2_fired else 'FIRED'}  "
    f"[mean q_goal[true]={mean_q_goal_true_all:.4f} (need>=0.60)]"
)
lines.append(
    f"  F3 (ToM hurts asymptotically): "
    f"{'NOT FIRED' if not f3_fired else 'FIRED'}  "
    f"[M3_overall-M1.5_overall={mean_m3_overall-mean_m15_overall:.4f} (need>=-0.02)]"
)
lines.append(
    f"  F4 (spine hash changed): "
    f"{'NOT FIRED' if f4_ok else 'FIRED'}  "
    f"[mirro {'OK' if f4_mirro_ok else 'CHANGED'}, vela {'OK' if f4_vela_ok else 'CHANGED'}]"
)
lines.append("")

lines.append("--- PREDICTION CONJUNCT EVALUATION ---")
lines.append("")
lines.append(
    f"  (a) mean_early_edge>=0.05 AND count>=6/8: {'TRUE' if pred_a else 'FALSE'}  "
    f"[edge={mean_early_edge:.4f}, count={count_m3_beats_m15_early}/8]"
)
lines.append(
    f"  (b) M3_overall >= M1.5_overall - 0.02:   {'TRUE' if pred_b else 'FALSE'}  "
    f"[{mean_m3_overall:.4f} >= {mean_m15_overall:.4f} - 0.02 = {mean_m15_overall-0.02:.4f}]"
)
lines.append(
    f"  (c) mean q_goal[true] >= 0.60:           {'TRUE' if pred_c else 'FALSE'}  "
    f"[{mean_q_goal_true_all:.4f}]"
)
lines.append("")

# Determine verdict
if not f4_ok:
    verdict = "VERDICT: HALT/F4 — spine integrity violated; continuity bug"
elif f2_fired:
    verdict = (
        f"VERDICT: NO_VERDICT/F2 — goal posterior did not concentrate "
        f"(mean q_goal[true]={mean_q_goal_true_all:.4f} < 0.60); instrument broken/uninformative"
    )
elif f1_fired and f3_fired:
    verdict = (
        f"VERDICT: FALSE/F1+F3 — no ToM benefit early (edge={mean_early_edge:.4f}, count={count_m3_beats_m15_early}/8) "
        f"AND ToM hurts asymptotically ({mean_m3_overall:.4f} < {mean_m15_overall:.4f}-0.02); "
        f"trivial-provision concern realized + misspecification dominates"
    )
elif f1_fired:
    verdict = (
        f"VERDICT: FALSE/F1 — no sample-efficiency edge for ToM "
        f"(mean_early_edge={mean_early_edge:.4f}, count={count_m3_beats_m15_early}/8); "
        f"learning B's dynamics suffices; NO ToM benefit at this scale"
    )
elif f3_fired:
    verdict = (
        f"VERDICT: FALSE/F3 — ToM goal model HURTS asymptotically "
        f"(M3_overall={mean_m3_overall:.4f} < M1.5_overall={mean_m15_overall:.4f}-0.02); "
        f"misspecification dominates"
    )
elif pred_a and pred_b and pred_c:
    verdict = (
        f"VERDICT: TRUE — all conjuncts hold; goal posterior concentrates (q[true]={mean_q_goal_true_all:.4f}); "
        f"M3 early edge={mean_early_edge:.4f} in {count_m3_beats_m15_early}/8 seeds; "
        f"M3_overall={mean_m3_overall:.4f} >= M1.5_overall={mean_m15_overall:.4f}-0.02; "
        f"functional ToM benefit confirmed at this scale"
    )
else:
    # Some conjuncts false but no explicit falsifier fires (edge case)
    failed = [l for l, v in [("a", pred_a), ("b", pred_b), ("c", pred_c)] if not v]
    verdict = (
        f"VERDICT: FALSE — conjunct(s) {failed} not met; F1-F4 not fired explicitly; "
        f"no ToM benefit established"
    )

lines.append(verdict)
lines.append("")
lines.append("=" * 80)

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
