"""Exp 233 — self-other-modeling Rung 2-ambiguous: does the ToM re-tracking benefit SURVIVE a GENUINELY hard (ambiguous) inference?

Hypothesis: Exp 232's '25-way' goal stayed EASY (q->0.93) because the 4 corner targets were trivially separable
(L38: difficulty = behavioral separability, not cardinality). Make the inference GENUINELY hard by using AMBIGUOUS
targets whose BFS trajectories OVERLAP -- the tight cluster {7,11,13,17} (the 4 cells orthogonally adjacent to
center cell 12 of the 5x5 grid), so B heading toward any one looks nearly identical until it is close, keeping the
target posterior diffuse. THEN test whether goal-inference still beats the adaptive learned baseline on post-switch
re-tracking when the inference is hard. If NOT, the arc's ToM benefit needed EASY inference.

MANIPULATION CHECK (predeclared, L38): mean q_target[true_target] < 0.60 (the targets are genuinely ambiguous; the
inference is hard). If this FAILS (q >= 0.60), these targets were not ambiguous enough -> NO_VERDICT on the
hard-inference question (re-design needed); do NOT read the post-switch arm as evidence either way.

Setup: Exp 232 substrate but CYCLE = [7,11,13,17] (the ambiguous center-adjacent cluster), 4 switches in H=2000,
seeds 0-7. Predictors as Exp 232: M2* oracle (current target); M1.5a adaptive learned transition (the FAIR control);
M3t 25-way target-posterior goal-inference (uniform init, forgetting leak, no leakage). Windows OVERALL=2000,
POST-SWITCH=50 steps after each of the 4 switches.

Prediction (ToM SURVIVES hard inference / strong-positive):
  (manip) mean q_target[true_target] < 0.60 (genuinely ambiguous -- the manipulation worked), AND
  (a) mean(M3t_postswitch - M1.5a_postswitch) >= 0.05 AND in >=6/8 seeds (ToM re-tracks faster EVEN when hard).

Verdicts:
  - (manip) FALSE (q >= 0.60): NO_VERDICT -- targets not ambiguous (manipulation failed, L38); re-design.
  - (manip) TRUE and (a) TRUE: POSITIVE -- the ToM benefit SURVIVES a genuinely hard inference (caveat overcome).
  - (manip) TRUE and (a) FALSE: NEGATIVE (F1) -- ToM does NOT survive hard inference; the benefit needed EASY
    inference (the deflation the arc has circled).
Falsifiers: F1 = (a) fails while (manip) holds; F2 = any committed spine state_hash changes.
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
OUTPUT_PATH = Path("experiments/outputs/exp233.txt")

# Comfort-gate parameters (reuse from exp230/232)
EPS_POLICY = 0.2                  # epsilon-greedy exploration
ALPHA_EMA = 0.1                   # comfort-EMA alpha
THRESH_GATE = 0.75                # comfort gate threshold
LAMBDA_RELAX = 0.01               # away-from-source EMA relaxation toward 1.0
EST_INIT = 1.0                    # initial comfort estimate

# Non-stationary goal switching: target cell changes every SWITCH_PERIOD steps
SWITCH_PERIOD = 400               # 5 segments in H=2000
SWITCH_STEPS = [400, 800, 1200, 1600]  # 4 switch points
POST_SWITCH_WINDOW = 50           # 50 steps after each switch scored as post-switch

# M1.5a adaptive decay
TRANS_DECAY = 0.99                # trans_a *= TRANS_DECAY each step

# M3t leak parameters
Q_LEAK_RETAIN = 0.98              # q_target *= Q_LEAK_RETAIN each step after Bayes update
Q_LEAK_UNIFORM = 0.02             # (1 - Q_LEAK_RETAIN) added as uniform

# THE ONLY CHANGE FROM EXP 232: ambiguous center-adjacent cluster instead of 4 corners
# Cell 12 is the center of the 5x5 grid; cells 7,11,13,17 are the 4 orthogonal neighbors.
# BFS trajectories to any of these overlap until B is very close, keeping posterior diffuse.
CYCLE = [7, 11, 13, 17]           # ambiguous center-adjacent cluster

# ---------------------------------------------------------------------------
# Load committed spines (read-only — NEVER call .live() or .save() on these)
# ---------------------------------------------------------------------------

mirro_spine = Creature.load(MIRRO_DIR)
vela_spine = Creature.load(VELA_DIR)

hash_mirro_before = mirro_spine._state_hash()
hash_vela_before = vela_spine._state_hash()

# ---------------------------------------------------------------------------
# BFS helpers (verbatim from exp232)
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
    """Comfort-gated epsilon-greedy BFS policy (mirrors exp230/232).

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


def target_next_dist(world: World, dist_field: list, b_pos: int, n_cells: int) -> np.ndarray:
    """Compute the next-cell distribution P(next | b_pos, target=t) for a given target's dist field.

    Uses eps-greedy BFS: prob (1-EPS_POLICY) on BFS-greedy minimizer cells (uniform over ties),
    prob EPS_POLICY spread uniformly over the 4 wall-clamped neighbor cells.
    """
    neighbors = [world.move(b_pos, a) for a in range(4)]

    # Greedy component: find BFS minimizers among neighbors
    best_d = min(dist_field[nb] for nb in neighbors)
    minimizers = [nb for nb in neighbors if dist_field[nb] == best_d]

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
n_cells = world.n_cells  # expected 25 for 5x5 grid

# Candidate target cells: ALL 25 cells (0..24) — identical to exp232
n_targets = n_cells  # 25

# Pre-compute BFS dist fields for ALL 25 candidate target cells (shared across seeds)
# dist_fields_all[t] = BFS distance field from cell t
dist_fields_all: list[list] = [build_bfs_dist(world, t) for t in range(n_targets)]

# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

seed_results = []  # list of dicts, one per seed

for seed in SEEDS:
    rng = np.random.default_rng(seed)

    # Fork both spines; NEVER save them
    b_creature = vela_spine.fork(f"exp233-vela-fork-seed{seed}")
    a_creature = mirro_spine.fork(f"exp233-mirro-fork-seed{seed}")

    # --- Per-seed state ---

    # M1.5a: adaptive learned transition (float, decays each step)
    trans_a = np.zeros((n_cells, n_cells), dtype=np.float64)

    # M3t: 25-way target posterior (uniform prior)
    q_target = np.ones(n_targets, dtype=np.float64) / n_targets

    # Comfort estimate for B's gate
    comfort_est = EST_INIT

    # Per-step accumulators: overall
    m2star_correct_total = 0
    m15a_correct_total = 0
    m3t_correct_total = 0

    # Post-switch accumulators
    m3t_correct_postswitch = 0
    m15a_correct_postswitch = 0
    postswitch_total_steps = 0

    # Track q_target[current_target] each step for mean (pre-update, at PREDICT time)
    q_target_current_sum = 0.0

    for step in range(H):
        b_pos = b_creature.true_pos

        # --- Non-stationary goal: determine current target cell for this step ---
        current_target = CYCLE[(step // SWITCH_PERIOD) % len(CYCLE)]

        # Dist field for current target (for B's actual policy)
        dist_to_use = dist_fields_all[current_target]

        # --- M1.5a: adaptive learned transition baseline ---
        # Predict BEFORE updating: (trans_a[cur] + 1.0) normalized
        row_a = trans_a[b_pos] + 1.0
        m15a_pred = row_a / row_a.sum()

        # --- M3t: adaptive 25-way target-inference (ToM treatment) ---
        # Record q_target[current_target] BEFORE predicting (pre-update, no leakage)
        q_target_current_sum += q_target[current_target]

        # Compute per-target next-cell distributions at b_pos (vectorized over all 25 targets)
        # per_target_pred[t] = P(next | b_pos, target=t)  shape: (n_targets, n_cells)
        per_target_pred = np.zeros((n_targets, n_cells), dtype=np.float64)
        for t in range(n_targets):
            per_target_pred[t] = target_next_dist(world, dist_fields_all[t], b_pos, n_cells)

        # M3t prediction: marginalize over target posterior
        m3t_pred = q_target @ per_target_pred  # shape (n_cells,)

        # --- M2* oracle: knows current target cell; near-one-hot at BFS-greedy next cell ---
        neighbors_oracle = [world.move(b_pos, a) for a in range(4)]
        best_d_oracle = min(dist_to_use[nb] for nb in neighbors_oracle)
        minimizers_oracle = [nb for nb in neighbors_oracle if dist_to_use[nb] == best_d_oracle]
        m2star_pred = np.full(n_cells, M2_EPSILON / n_cells)
        weight_oracle = (1.0 - M2_EPSILON) / len(minimizers_oracle)
        for nb in minimizers_oracle:
            m2star_pred[nb] += weight_oracle

        # --- B moves (the real policy) ---
        # Update comfort estimate (update BEFORE policy action, mirrors exp230/232)
        at_target = (b_pos == current_target)
        if at_target:
            comfort_est = (1 - ALPHA_EMA) * comfort_est + ALPHA_EMA * 1.0
        else:
            comfort_est += LAMBDA_RELAX * (1.0 - comfort_est)

        action = policy_action(world, dist_to_use, b_pos, comfort_est, rng)
        actual_next = world.move(b_pos, action)

        # --- Update M3t q_target AFTER observing B's move (no leakage) ---
        # Likelihood of actual_next under each candidate target's policy
        lik = per_target_pred[:, actual_next] + 1e-9  # shape (n_targets,), floor to avoid zeros
        q_target = q_target * lik
        q_sum = q_target.sum()
        if q_sum > 0:
            q_target = q_target / q_sum
        else:
            q_target = np.ones(n_targets, dtype=np.float64) / n_targets  # reset on degenerate

        # Apply forgetting leak AFTER Bayes update
        q_target = Q_LEAK_RETAIN * q_target + Q_LEAK_UNIFORM * (1.0 / n_targets)
        # Re-normalize (guards float drift)
        q_target = q_target / q_target.sum()

        # --- Update M1.5a adaptive transition AFTER B moves ---
        # Decay first, then add new observation
        trans_a *= TRANS_DECAY
        trans_a[b_pos, actual_next] += 1.0

        # Update B's true position
        b_creature.true_pos = actual_next

        # --- Score predictors ---
        m2star_argmax = int(np.argmax(m2star_pred))
        m15a_argmax = int(np.argmax(m15a_pred))
        m3t_argmax = int(np.argmax(m3t_pred))

        m2star_hit = int(m2star_argmax == actual_next)
        m15a_hit = int(m15a_argmax == actual_next)
        m3t_hit = int(m3t_argmax == actual_next)

        m2star_correct_total += m2star_hit
        m15a_correct_total += m15a_hit
        m3t_correct_total += m3t_hit

        # --- Post-switch window scoring ---
        in_postswitch = any(sw <= step < sw + POST_SWITCH_WINDOW for sw in SWITCH_STEPS)
        if in_postswitch:
            m3t_correct_postswitch += m3t_hit
            m15a_correct_postswitch += m15a_hit
            postswitch_total_steps += 1

    # Per-seed accuracy
    m2star_overall = m2star_correct_total / H
    m15a_overall = m15a_correct_total / H
    m3t_overall = m3t_correct_total / H

    m3t_postswitch = m3t_correct_postswitch / postswitch_total_steps if postswitch_total_steps > 0 else float("nan")
    m15a_postswitch = m15a_correct_postswitch / postswitch_total_steps if postswitch_total_steps > 0 else float("nan")

    mean_q_target_current = q_target_current_sum / H

    edge_ov = m3t_overall - m15a_overall
    postswitch_edge = (
        m3t_postswitch - m15a_postswitch
        if not (np.isnan(m3t_postswitch) or np.isnan(m15a_postswitch))
        else float("nan")
    )

    seed_results.append({
        "seed": seed,
        "m2star_overall": m2star_overall,
        "m15a_overall": m15a_overall,
        "m3t_overall": m3t_overall,
        "edge_ov": edge_ov,
        "m3t_postswitch": m3t_postswitch,
        "m15a_postswitch": m15a_postswitch,
        "postswitch_edge": postswitch_edge,
        "postswitch_steps": postswitch_total_steps,
        "mean_q_target_current": mean_q_target_current,
        "postswitch_edge_ge_05": (
            postswitch_edge >= 0.05
            if not np.isnan(postswitch_edge)
            else False
        ),
    })

# ---------------------------------------------------------------------------
# Spine integrity check (F2)
# ---------------------------------------------------------------------------

hash_mirro_after = mirro_spine._state_hash()
hash_vela_after = vela_spine._state_hash()

f2_mirro_ok = hash_mirro_before == hash_mirro_after
f2_vela_ok = hash_vela_before == hash_vela_after
f2_spine_ok = f2_mirro_ok and f2_vela_ok

# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------

mean_m2star_overall = float(np.mean([r["m2star_overall"] for r in seed_results]))
mean_m15a_overall = float(np.mean([r["m15a_overall"] for r in seed_results]))
mean_m3t_overall = float(np.mean([r["m3t_overall"] for r in seed_results]))
mean_edge_ov = float(np.mean([r["edge_ov"] for r in seed_results]))
mean_q_target_current_all = float(np.mean([r["mean_q_target_current"] for r in seed_results]))

# Post-switch: mean over seeds (filter nan)
ps_m3t_vals = [r["m3t_postswitch"] for r in seed_results if not np.isnan(r["m3t_postswitch"])]
ps_m15a_vals = [r["m15a_postswitch"] for r in seed_results if not np.isnan(r["m15a_postswitch"])]
mean_m3t_postswitch = float(np.mean(ps_m3t_vals)) if ps_m3t_vals else float("nan")
mean_m15a_postswitch = float(np.mean(ps_m15a_vals)) if ps_m15a_vals else float("nan")
mean_postswitch_edge = (
    mean_m3t_postswitch - mean_m15a_postswitch
    if not (np.isnan(mean_m3t_postswitch) or np.isnan(mean_m15a_postswitch))
    else float("nan")
)

total_postswitch_steps_all = sum(r["postswitch_steps"] for r in seed_results)
count_postswitch_edge_ge_05 = sum(1 for r in seed_results if r["postswitch_edge_ge_05"])

# ---------------------------------------------------------------------------
# Verdict evaluation (Exp 233 logic — gated on manipulation check)
# ---------------------------------------------------------------------------

# MANIPULATION CHECK: mean q_target[true_target] < 0.60
manip_ok = mean_q_target_current_all < 0.60

# (a) post-switch edge
a_ok = (
    (not np.isnan(mean_postswitch_edge)) and
    (mean_postswitch_edge >= 0.05) and
    (count_postswitch_edge_ge_05 >= 6)
)

# ---------------------------------------------------------------------------
# Build output string
# ---------------------------------------------------------------------------

lines = []
lines.append("=" * 90)
lines.append("Exp 233 — self-other-modeling Rung 2-ambiguous: ToM re-tracking with GENUINELY HARD (ambiguous) inference")
lines.append("=" * 90)
lines.append("")
lines.append("Implementation notes:")
lines.append("  - Substrate: identical to Exp 232 (vela-fork is B, mirro-fork is A)")
lines.append(f"  - n_cells = {n_cells}, n_targets = {n_targets} (all cells are candidate targets, as in Exp 232)")
lines.append(f"  - B's CYCLE target cells: {CYCLE}  (ambiguous center-adjacent cluster — cells orthogonal to center 12)")
lines.append(f"  - B's goal TARGET CELL = CYCLE[(step//400) % 4]; 4 switches in H={H}")
lines.append(f"  - Switch steps: {SWITCH_STEPS}; post-switch window = {POST_SWITCH_WINDOW} steps each")
lines.append(f"  - EPS_POLICY = {EPS_POLICY}, ALPHA_EMA = {ALPHA_EMA}, THRESH_GATE = {THRESH_GATE}")
lines.append(f"  - LAMBDA_RELAX = {LAMBDA_RELAX}, EST_INIT = {EST_INIT}")
lines.append(f"  - M1.5a: adaptive trans float, decay={TRANS_DECAY} per step, +1 Laplace predict (BINDING control)")
lines.append(f"  - M3t: q_target uniform init over {n_targets} cells; PREDICT then observe then Bayes update, then leak:")
lines.append(f"         q_target = {Q_LEAK_RETAIN}*q_target + {Q_LEAK_UNIFORM}*(1/{n_targets} each), re-normalized")
lines.append(f"  - M2*: oracle knows CURRENT target cell; near-one-hot at BFS-greedy next cell (eps={M2_EPSILON})")
lines.append(f"  - BFS dist fields for all {n_targets} candidate targets precomputed once (shared across seeds)")
lines.append(f"  - q_target[current_target] tracked at PREDICT time (pre-update, no leakage)")
lines.append(f"  - BFS tie-break: uniform over minimizer neighbor cells")
lines.append(f"  - Likelihood floor: 1e-9 added to each per-target likelihood before Bayes update")
lines.append(f"  - Comfort gate: at-target means b_pos == current_target cell")
lines.append(f"  - MANIPULATION CHECK threshold: mean q_target[true] < 0.60 (targets genuinely ambiguous)")
lines.append("")

lines.append("--- PER-SEED TABLE ---")
lines.append("")
hdr = (
    f"{'seed':>4}  {'M2*_ov':>7}  {'M15a_ov':>8}  {'M3t_ov':>7}  "
    f"{'edge_ov':>8}  {'M3t_ps':>7}  {'M15a_ps':>8}  {'ps_edge':>8}  {'q[cur]':>7}  {'ps>=.05':>8}"
)
lines.append(hdr)
lines.append("-" * 100)
for r in seed_results:
    row = (
        f"  {r['seed']:>2}  {r['m2star_overall']:>7.4f}  {r['m15a_overall']:>8.4f}  "
        f"{r['m3t_overall']:>7.4f}  {r['edge_ov']:>8.4f}  "
        f"{r['m3t_postswitch']:>7.4f}  {r['m15a_postswitch']:>8.4f}  "
        f"{r['postswitch_edge']:>8.4f}  "
        f"{r['mean_q_target_current']:>7.4f}  "
        f"{'YES' if r['postswitch_edge_ge_05'] else 'NO':>8}"
    )
    lines.append(row)
lines.append("-" * 100)
lines.append(
    f"  {'mean':>4}  {mean_m2star_overall:>7.4f}  {mean_m15a_overall:>8.4f}  "
    f"{mean_m3t_overall:>7.4f}  {mean_edge_ov:>8.4f}  "
    f"{mean_m3t_postswitch:>7.4f}  {mean_m15a_postswitch:>8.4f}  "
    f"{mean_postswitch_edge:>8.4f}  "
    f"{mean_q_target_current_all:>7.4f}"
)
lines.append(f"  [post-switch total steps across all seeds: {total_postswitch_steps_all}]")
lines.append(f"  [count seeds with postswitch_edge >= 0.05: {count_postswitch_edge_ge_05}/8]")
lines.append("")

lines.append("--- AGGREGATES ---")
lines.append("")
lines.append(f"  mean(M3t_postswitch - M1.5a_postswitch) [ps edge]: {mean_postswitch_edge:.4f}  (threshold: >=0.05 AND >=6/8 seeds)")
lines.append(f"  count seeds postswitch_edge >= 0.05:               {count_postswitch_edge_ge_05}/8  (threshold: >=6/8)")
lines.append(f"  mean M3t_overall:                                   {mean_m3t_overall:.4f}")
lines.append(f"  mean M1.5a_overall (adaptive baseline):             {mean_m15a_overall:.4f}")
lines.append(f"  mean(M3t_overall - M1.5a_overall) [overall edge]:  {mean_edge_ov:.4f}")
lines.append(f"  mean M2*_overall (oracle):                          {mean_m2star_overall:.4f}")
lines.append(f"  mean q_target[true_target]:                         {mean_q_target_current_all:.4f}  [MANIPULATION CHECK: need < 0.60]")
lines.append("")

lines.append("--- MANIPULATION CHECK ---")
lines.append("")
lines.append(f"  mean q_target[true_target] = {mean_q_target_current_all:.4f}")
lines.append(f"  Threshold: < 0.60 (targets genuinely ambiguous; inference is hard)")
lines.append(f"  Result: {'PASSED (q < 0.60 -- targets ARE ambiguous)' if manip_ok else 'FAILED (q >= 0.60 -- targets NOT ambiguous enough)'}")
lines.append("")

lines.append("--- SPINE INTEGRITY (F2) ---")
lines.append("")
lines.append(f"  mirro before: {hash_mirro_before[:24]}...")
lines.append(f"  mirro after:  {hash_mirro_after[:24]}...")
lines.append(f"  mirro intact: {'YES' if f2_mirro_ok else 'NO --- F2 FIRED'}")
lines.append(f"  vela before:  {hash_vela_before[:24]}...")
lines.append(f"  vela after:   {hash_vela_after[:24]}...")
lines.append(f"  vela intact:  {'YES' if f2_vela_ok else 'NO --- F2 FIRED'}")
lines.append(f"  F2 (spine hash): {'NOT FIRED' if f2_spine_ok else 'FIRED (continuity bug)'}")
lines.append("")

lines.append("--- VERDICT LOGIC EVALUATION ---")
lines.append("")
lines.append(f"  MANIPULATION CHECK (manip_ok): mean q[true] < 0.60? {'YES' if manip_ok else 'NO'}  [{mean_q_target_current_all:.4f}]")
lines.append(f"  (a) ps_edge >= 0.05 AND count >= 6/8:  {'YES' if a_ok else 'NO'}  [ps_edge={mean_postswitch_edge:.4f}, count={count_postswitch_edge_ge_05}/8]")
lines.append("")
lines.append("  Decision tree:")
lines.append(f"    F2 (spine changed)? {'YES -- overrides' if not f2_spine_ok else 'NO'}")
lines.append(f"    manip_ok (q < 0.60)? {'YES' if manip_ok else 'NO (-> NO_VERDICT)'}")
lines.append(f"    a_ok (ps edge)? {'YES (-> POSITIVE)' if a_ok else 'NO (-> NEGATIVE/F1)'}")
lines.append("")

# Determine verdict
if not f2_spine_ok:
    verdict = "VERDICT: HALT/F2 — spine integrity violated; continuity bug (overrides all)"
elif not manip_ok:
    verdict = (
        f"VERDICT: NO_VERDICT (manipulation failed: q={mean_q_target_current_all:.4f} >= 0.60, targets not ambiguous); "
        f"re-design needed"
    )
elif a_ok:
    verdict = (
        f"VERDICT: POSITIVE (ToM survives hard inference) -- "
        f"manip PASSED (q={mean_q_target_current_all:.4f} < 0.60); "
        f"ps_edge={mean_postswitch_edge:.4f} >= 0.05 in {count_postswitch_edge_ge_05}/8 seeds; "
        f"ToM re-tracking benefit SURVIVES genuinely ambiguous inference"
    )
else:
    verdict = (
        f"VERDICT: NEGATIVE/F1 (ToM does NOT survive hard inference) -- "
        f"manip PASSED (q={mean_q_target_current_all:.4f} < 0.60) but "
        f"ps_edge={mean_postswitch_edge:.4f} [need>=0.05], count={count_postswitch_edge_ge_05}/8 [need>=6]; "
        f"the ToM benefit needed EASY inference (deflation of the arc)"
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

# Halt on F2 (spine integrity)
if not f2_spine_ok:
    print("HALT: F2 fired — spine integrity violated")
    sys.exit(1)
