"""Exp 230 — self-other-modeling Rung 2-hard: does the goal-inference benefit SURVIVE a NON-STATIONARY goal, vs a FAIR adaptive learned baseline?

Hypothesis: Exp 229's ToM benefit was sample-efficiency in a STATIONARY world (an adaptive learned-transition
model nearly caught up). Make the other's goal NON-STATIONARY (B's favorite color CHANGES every 400 steps) and
compare adaptive goal-inference (q_goal that re-concentrates after a switch) against a FAIR adaptive learned-
transition baseline (decaying transition counts). If structured goal re-inference re-tracks the changed goal
faster than unstructured transition re-learning, the ToM benefit is SUSTAINED (not just early) -> robust. If the
adaptive learned baseline matches it, then Exp 229's win does NOT survive a fair adaptive comparison in a
non-stationary world -> the ToM benefit is an artifact of easy stationary inference (a real deflation).

Setup: Exp 229 substrate; forks NEVER saved. B's goal color CYCLES: goal_color(step) = all_colors[(step//400) % 3]
(5 segments, 4 switches in H=2000); B navigates (comfort-gated eps-greedy BFS) toward the CURRENT goal color's
nearest cell. seeds 0-7. Predictors of B's NEXT cell, argmax accuracy:
  M2* oracle (knows the CURRENT goal_color; near-one-hot at the BFS-greedy cell for the current goal);
  M1.5s LEARNED-stationary transition (Exp 229's trans counts +1, NO forgetting) -- reference, expected to DEGRADE;
  M1.5a LEARNED-adaptive transition: trans = 0.99*trans + onehot(prev->next) each step; predict row (trans[cur]+1)
        normalized -- the FAIR adaptive non-ToM baseline (BINDING control);
  M3a goal-inference-ADAPTIVE: q_goal over 3 colors with a forgetting leak each step q_goal = 0.98*q_goal + 0.02*uniform
      (applied AFTER the Bayes update) so it can re-concentrate after a switch; predict/observe/update, NO leakage.
Windows: OVERALL = all 2000; POST-SWITCH = the 50 steps immediately after each of the 4 switches (200 steps total).

Prediction (SUSTAINED ToM benefit TRUE): (a) mean(M3a_overall - M1.5a_overall) >= 0.05 AND in >=6/8 seeds;
(b) M3a re-tracks: mean over the run of q_goal[current_goal] >= 0.60; (c) M3a_postswitch_acc > M1.5a_postswitch_acc
(structured re-inference re-tracks faster right after a goal change).

Falsifier (NO sustained benefit / FALSE):
  F1: mean(M3a_overall - M1.5a_overall) < 0.05 OR <6/8 seeds -> the adaptive learned baseline matches goal-inference
      in a non-stationary world; Exp 229's benefit does NOT survive a fair adaptive comparison (DEFLATION).
  F2: mean q_goal[current_goal] < 0.60 -> goal-inference cannot track the changing goal (instrument / leak mis-tuned).
  F3: M3a_postswitch <= M1.5a_postswitch -> no faster re-tracking (the mechanism's claimed edge is absent).
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
OUTPUT_PATH = Path("experiments/outputs/exp230.txt")

# Comfort-gate parameters (reuse from exp229/exp228/exp70)
EPS_POLICY = 0.2                  # epsilon-greedy exploration
ALPHA_EMA = 0.1                   # comfort-EMA alpha
THRESH_GATE = 0.75                # comfort gate threshold
LAMBDA_RELAX = 0.01               # away-from-source EMA relaxation toward 1.0
EST_INIT = 1.0                    # initial comfort estimate

# Non-stationary goal switching: goal changes every SWITCH_PERIOD steps
SWITCH_PERIOD = 400               # 5 segments in H=2000
SWITCH_STEPS = [400, 800, 1200, 1600]  # 4 switch points
POST_SWITCH_WINDOW = 50           # 50 steps after each switch scored as post-switch

# M1.5a adaptive decay
TRANS_DECAY = 0.99                # trans_a *= TRANS_DECAY each step

# M3a leak parameters
Q_LEAK_RETAIN = 0.98              # q_goal *= Q_LEAK_RETAIN each step after Bayes update
Q_LEAK_UNIFORM = 0.02             # (1 - Q_LEAK_RETAIN) added as uniform

# ---------------------------------------------------------------------------
# Load committed spines (read-only — NEVER call .live() or .save() on these)
# ---------------------------------------------------------------------------

mirro_spine = Creature.load(MIRRO_DIR)
vela_spine = Creature.load(VELA_DIR)

hash_mirro_before = mirro_spine._state_hash()
hash_vela_before = vela_spine._state_hash()

# ---------------------------------------------------------------------------
# BFS helpers (verbatim from exp229)
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
    """Comfort-gated epsilon-greedy BFS policy (mirrors exp229).

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
# Main experiment setup
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

# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

seed_results = []  # list of dicts, one per seed

for seed in SEEDS:
    rng = np.random.default_rng(seed)

    # Fork both spines; NEVER save them
    b_creature = vela_spine.fork(f"exp230-vela-fork-seed{seed}")
    a_creature = mirro_spine.fork(f"exp230-mirro-fork-seed{seed}")

    # --- Per-seed state ---

    # M1.5s: stationary learned transition (exp229's counts, no decay)
    trans_s = np.zeros((n_cells, n_cells), dtype=np.int64)

    # M1.5a: adaptive learned transition (float, decays each step)
    trans_a = np.zeros((n_cells, n_cells), dtype=np.float64)

    # M3a: adaptive goal posterior (uniform prior, with leak after Bayes update)
    q_goal = np.ones(n_colors) / n_colors  # indexed by all_colors list position

    # Comfort estimate for B's gate
    comfort_est = EST_INIT

    # Per-step accumulators: overall
    m2star_correct_total = 0
    m15s_correct_total = 0
    m15a_correct_total = 0
    m3a_correct_total = 0

    # Post-switch accumulators
    m3a_correct_postswitch = 0
    m15a_correct_postswitch = 0
    postswitch_total_steps = 0

    # Track q_goal[current_goal] each step for mean
    q_goal_current_sum = 0.0

    for step in range(H):
        b_pos = b_creature.true_pos

        # --- Non-stationary goal: determine current goal color for this step ---
        goal_color = all_colors[(step // SWITCH_PERIOD) % n_colors]
        goal_color_idx = all_colors.index(goal_color)

        # Source cells for current goal
        source_cells_cur = color_source_cells[goal_color]

        # Nearest dist field for current goal color (for B's actual policy)
        dist_to_use = nearest_dist_for_color(goal_color, b_pos)

        # --- M1.5s: stationary learned transition baseline ---
        # Predict BEFORE updating (no leakage): +1 Laplace smoothing
        row_s = trans_s[b_pos].astype(float) + 1.0
        m15s_pred = row_s / row_s.sum()

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

        # --- M2* oracle: knows current goal_color ---
        best_d = min(dist_to_use[world.move(b_pos, a)] for a in range(4))
        minimizers_oracle = [a for a in range(4) if dist_to_use[world.move(b_pos, a)] == best_d]
        m2star_pred = np.full(n_cells, M2_EPSILON / n_cells)
        weight_oracle = (1.0 - M2_EPSILON) / len(minimizers_oracle)
        for a in minimizers_oracle:
            nc = world.move(b_pos, a)
            m2star_pred[nc] += weight_oracle

        # --- B moves (the real policy) ---
        # Update comfort estimate (update BEFORE policy action, mirrors exp229)
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
        # Re-normalize (the above already sums to 1 if q_goal sums to 1, but guard float drift)
        q_goal = q_goal / q_goal.sum()

        # --- Update M1.5s transition counts AFTER B moves ---
        trans_s[b_pos, actual_next] += 1

        # --- Update M1.5a adaptive transition AFTER B moves ---
        # Decay first, then add new observation
        trans_a *= TRANS_DECAY
        trans_a[b_pos, actual_next] += 1.0

        # Update B's true position
        b_creature.true_pos = actual_next

        # --- Score predictors ---
        m2star_argmax = int(np.argmax(m2star_pred))
        m15s_argmax = int(np.argmax(m15s_pred))
        m15a_argmax = int(np.argmax(m15a_pred))
        m3a_argmax = int(np.argmax(m3a_pred))

        m2star_hit = int(m2star_argmax == actual_next)
        m15s_hit = int(m15s_argmax == actual_next)
        m15a_hit = int(m15a_argmax == actual_next)
        m3a_hit = int(m3a_argmax == actual_next)

        m2star_correct_total += m2star_hit
        m15s_correct_total += m15s_hit
        m15a_correct_total += m15a_hit
        m3a_correct_total += m3a_hit

        # --- Post-switch window scoring ---
        in_postswitch = any(sw <= step < sw + POST_SWITCH_WINDOW for sw in SWITCH_STEPS)
        if in_postswitch:
            m3a_correct_postswitch += m3a_hit
            m15a_correct_postswitch += m15a_hit
            postswitch_total_steps += 1

    # Per-seed accuracy
    m2star_overall = m2star_correct_total / H
    m15s_overall = m15s_correct_total / H
    m15a_overall = m15a_correct_total / H
    m3a_overall = m3a_correct_total / H

    m3a_postswitch = m3a_correct_postswitch / postswitch_total_steps if postswitch_total_steps > 0 else float("nan")
    m15a_postswitch = m15a_correct_postswitch / postswitch_total_steps if postswitch_total_steps > 0 else float("nan")

    mean_q_goal_current = q_goal_current_sum / H

    edge = m3a_overall - m15a_overall

    seed_results.append({
        "seed": seed,
        "m2star_overall": m2star_overall,
        "m15s_overall": m15s_overall,
        "m15a_overall": m15a_overall,
        "m3a_overall": m3a_overall,
        "edge": edge,
        "m3a_postswitch": m3a_postswitch,
        "m15a_postswitch": m15a_postswitch,
        "postswitch_steps": postswitch_total_steps,
        "mean_q_goal_current": mean_q_goal_current,
        "m3a_beats_m15a_overall": m3a_overall > m15a_overall,
        "m3a_beats_m15a_postswitch": (
            m3a_postswitch > m15a_postswitch
            if not (np.isnan(m3a_postswitch) or np.isnan(m15a_postswitch))
            else False
        ),
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

mean_m2star_overall = float(np.mean([r["m2star_overall"] for r in seed_results]))
mean_m15s_overall = float(np.mean([r["m15s_overall"] for r in seed_results]))
mean_m15a_overall = float(np.mean([r["m15a_overall"] for r in seed_results]))
mean_m3a_overall = float(np.mean([r["m3a_overall"] for r in seed_results]))
mean_edge = float(np.mean([r["edge"] for r in seed_results]))
count_m3a_beats_m15a_overall = sum(1 for r in seed_results if r["m3a_beats_m15a_overall"])
mean_q_goal_current_all = float(np.mean([r["mean_q_goal_current"] for r in seed_results]))

# Post-switch: mean over seeds (filter nan if any)
ps_m3a_vals = [r["m3a_postswitch"] for r in seed_results if not np.isnan(r["m3a_postswitch"])]
ps_m15a_vals = [r["m15a_postswitch"] for r in seed_results if not np.isnan(r["m15a_postswitch"])]
mean_m3a_postswitch = float(np.mean(ps_m3a_vals)) if ps_m3a_vals else float("nan")
mean_m15a_postswitch = float(np.mean(ps_m15a_vals)) if ps_m15a_vals else float("nan")
total_postswitch_steps_all = sum(r["postswitch_steps"] for r in seed_results)

# ---------------------------------------------------------------------------
# Verdict evaluation
# ---------------------------------------------------------------------------

# Prediction conjuncts
pred_a = (mean_edge >= 0.05) and (count_m3a_beats_m15a_overall >= 6)
pred_b = mean_q_goal_current_all >= 0.60
pred_c = (
    mean_m3a_postswitch > mean_m15a_postswitch
    if not (np.isnan(mean_m3a_postswitch) or np.isnan(mean_m15a_postswitch))
    else False
)

# Falsifiers
f1_fired = not ((mean_edge >= 0.05) and (count_m3a_beats_m15a_overall >= 6))
f2_fired = mean_q_goal_current_all < 0.60
f3_fired = not (
    mean_m3a_postswitch > mean_m15a_postswitch
    if not (np.isnan(mean_m3a_postswitch) or np.isnan(mean_m15a_postswitch))
    else False
)
# F4 evaluated above

# ---------------------------------------------------------------------------
# Build output string
# ---------------------------------------------------------------------------

lines = []
lines.append("=" * 90)
lines.append("Exp 230 — self-other-modeling Rung 2-hard: NON-STATIONARY goal vs FAIR adaptive learned baseline")
lines.append("=" * 90)
lines.append("")
lines.append("Implementation notes:")
lines.append("  - Shared world: vela's world (vela-fork is B, lives here natively)")
lines.append(f"  - n_cells = {n_cells}, n_colors = {n_colors}, all_colors = {all_colors}")
lines.append(f"  - B's goal CYCLES: goal_color(step) = all_colors[(step//400) % 3]")
lines.append(f"  - Switch steps: {SWITCH_STEPS}; post-switch window = {POST_SWITCH_WINDOW} steps each")
lines.append(f"  - EPS_POLICY = {EPS_POLICY}, ALPHA_EMA = {ALPHA_EMA}, THRESH_GATE = {THRESH_GATE}")
lines.append(f"  - LAMBDA_RELAX = {LAMBDA_RELAX}, EST_INIT = {EST_INIT}")
lines.append(f"  - M1.5s: stationary trans counts +1 Laplace, no decay (exp229 reference)")
lines.append(f"  - M1.5a: adaptive trans float, decay={TRANS_DECAY} per step, +1 Laplace predict (BINDING control)")
lines.append(f"  - M3a: q_goal uniform init; PREDICT then observe then Bayes update, then leak:")
lines.append(f"         q_goal = {Q_LEAK_RETAIN}*q_goal + {Q_LEAK_UNIFORM}*(1/3 each), re-normalized")
lines.append(f"  - M2*: oracle knows CURRENT goal_color; near-one-hot (eps={M2_EPSILON})")
lines.append(f"  - q_goal[current_goal] tracked at PREDICT time (pre-update, no leakage)")
lines.append(f"  - BFS tie-break: uniform over minimizer neighbor cells")
lines.append(f"  - q_goal floor: 1e-9 added to each likelihood before Bayes update")
lines.append("")

lines.append("--- PER-SEED TABLE ---")
lines.append("")
hdr = (
    f"{'seed':>4}  {'M2*_ov':>7}  {'M15s_ov':>8}  {'M15a_ov':>8}  {'M3a_ov':>7}  "
    f"{'edge':>7}  {'M3a_ps':>7}  {'M15a_ps':>8}  {'q[cur]':>7}  {'M3a>M15a':>9}"
)
lines.append(hdr)
lines.append("-" * 95)
for r in seed_results:
    row = (
        f"  {r['seed']:>2}  {r['m2star_overall']:>7.4f}  {r['m15s_overall']:>8.4f}  "
        f"{r['m15a_overall']:>8.4f}  {r['m3a_overall']:>7.4f}  {r['edge']:>7.4f}  "
        f"{r['m3a_postswitch']:>7.4f}  {r['m15a_postswitch']:>8.4f}  "
        f"{r['mean_q_goal_current']:>7.4f}  "
        f"{'YES' if r['m3a_beats_m15a_overall'] else 'NO':>9}"
    )
    lines.append(row)
lines.append("-" * 95)
lines.append(
    f"  {'mean':>4}  {mean_m2star_overall:>7.4f}  {mean_m15s_overall:>8.4f}  "
    f"{mean_m15a_overall:>8.4f}  {mean_m3a_overall:>7.4f}  {mean_edge:>7.4f}  "
    f"{mean_m3a_postswitch:>7.4f}  {mean_m15a_postswitch:>8.4f}  "
    f"{mean_q_goal_current_all:>7.4f}"
)
lines.append(f"  [post-switch total steps across all seeds: {total_postswitch_steps_all}]")
lines.append("")

lines.append("--- AGGREGATES ---")
lines.append("")
lines.append(f"  mean(M3a_overall - M1.5a_overall) [edge]:  {mean_edge:.4f}  (threshold for pred_a: >=0.05)")
lines.append(f"  count(M3a_overall > M1.5a_overall):        {count_m3a_beats_m15a_overall}/8  (threshold for pred_a: >=6/8)")
lines.append(f"  mean M3a_overall:                          {mean_m3a_overall:.4f}")
lines.append(f"  mean M1.5a_overall (adaptive baseline):    {mean_m15a_overall:.4f}")
lines.append(f"  mean M1.5s_overall (stationary ref):       {mean_m15s_overall:.4f}  (expected to degrade)")
lines.append(f"  mean M2*_overall (oracle):                 {mean_m2star_overall:.4f}")
lines.append(f"  mean q_goal[current_goal]:                 {mean_q_goal_current_all:.4f}  (pred_b threshold: >=0.60)")
lines.append(f"  mean M3a_postswitch:                       {mean_m3a_postswitch:.4f}")
lines.append(f"  mean M1.5a_postswitch:                     {mean_m15a_postswitch:.4f}")
lines.append(f"  M3a_postswitch - M1.5a_postswitch:         {mean_m3a_postswitch - mean_m15a_postswitch:.4f}  (pred_c threshold: >0)")
lines.append(f"  M1.5s_overall vs M1.5a_overall delta:      {mean_m15s_overall - mean_m15a_overall:.4f}  (degradation reference)")
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
    f"  F1 (adaptive baseline matches goal-inference): "
    f"{'NOT FIRED' if not f1_fired else 'FIRED'}  "
    f"[mean_edge={mean_edge:.4f} (need>=0.05), count={count_m3a_beats_m15a_overall}/8 (need>=6)]"
)
lines.append(
    f"  F2 (goal-inference cannot track changing goal): "
    f"{'NOT FIRED' if not f2_fired else 'FIRED'}  "
    f"[mean q_goal[current]={mean_q_goal_current_all:.4f} (need>=0.60)]"
)
lines.append(
    f"  F3 (no faster post-switch re-tracking): "
    f"{'NOT FIRED' if not f3_fired else 'FIRED'}  "
    f"[M3a_ps={mean_m3a_postswitch:.4f}, M1.5a_ps={mean_m15a_postswitch:.4f} (need M3a_ps > M1.5a_ps)]"
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
    f"  (a) mean_edge>=0.05 AND count>=6/8:          {'TRUE' if pred_a else 'FALSE'}  "
    f"[edge={mean_edge:.4f}, count={count_m3a_beats_m15a_overall}/8]"
)
lines.append(
    f"  (b) mean q_goal[current] >= 0.60:            {'TRUE' if pred_b else 'FALSE'}  "
    f"[{mean_q_goal_current_all:.4f}]"
)
lines.append(
    f"  (c) M3a_postswitch > M1.5a_postswitch:       {'TRUE' if pred_c else 'FALSE'}  "
    f"[M3a={mean_m3a_postswitch:.4f}, M1.5a={mean_m15a_postswitch:.4f}]"
)
lines.append("")

# Determine verdict
if not f4_ok:
    verdict = "VERDICT: HALT/F4 — spine integrity violated; continuity bug"
elif f2_fired:
    verdict = (
        f"VERDICT: NO_VERDICT/F2 — goal posterior could not track changing goal "
        f"(mean q_goal[current]={mean_q_goal_current_all:.4f} < 0.60); instrument broken/leak mis-tuned"
    )
elif f1_fired and f3_fired:
    verdict = (
        f"VERDICT: FALSE/F1+F3 — adaptive baseline matches goal-inference overall (edge={mean_edge:.4f}, "
        f"count={count_m3a_beats_m15a_overall}/8) AND no faster post-switch re-tracking "
        f"(M3a_ps={mean_m3a_postswitch:.4f} <= M1.5a_ps={mean_m15a_postswitch:.4f}); "
        f"Exp 229 ToM benefit DOES NOT SURVIVE fair adaptive comparison (DEFLATION)"
    )
elif f1_fired:
    verdict = (
        f"VERDICT: FALSE/F1 — adaptive learned baseline matches goal-inference overall "
        f"(mean_edge={mean_edge:.4f}, count={count_m3a_beats_m15a_overall}/8); "
        f"ToM overall benefit does not survive fair adaptive comparison (DEFLATION)"
    )
elif f3_fired:
    verdict = (
        f"VERDICT: FALSE/F3 — no faster post-switch re-tracking "
        f"(M3a_ps={mean_m3a_postswitch:.4f} <= M1.5a_ps={mean_m15a_postswitch:.4f}); "
        f"structured goal re-inference does NOT re-track faster after a switch"
    )
elif pred_a and pred_b and pred_c:
    verdict = (
        f"VERDICT: TRUE — all conjuncts hold; q_goal tracks changing goal ({mean_q_goal_current_all:.4f}); "
        f"M3a overall edge={mean_edge:.4f} in {count_m3a_beats_m15a_overall}/8 seeds; "
        f"M3a re-tracks faster post-switch (M3a_ps={mean_m3a_postswitch:.4f} > M1.5a_ps={mean_m15a_postswitch:.4f}); "
        f"ToM benefit SUSTAINED in non-stationary world"
    )
else:
    failed = [lbl for lbl, v in [("a", pred_a), ("b", pred_b), ("c", pred_c)] if not v]
    verdict = (
        f"VERDICT: FALSE — conjunct(s) {failed} not met; no falsifier fires explicitly; "
        f"ToM sustained benefit not established"
    )

lines.append(verdict)
lines.append("")
lines.append("=" * 90)

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
