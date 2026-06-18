"""Exp 228 — self-other-modeling Rung 1: goal-directed-other substrate + position-tracking baseline + headroom bound.

Hypothesis: a clade-mate B navigating toward its own favorite-color source is (i) more predictable one step
ahead from its CURRENT position + innate dynamics than from a static occupancy prior, and (ii) leaves measurable
HEADROOM that knowing B's latent GOAL would fill — bounding what Rung-2 goal-inference could achieve.
CONSOLIDATION / plumbing — NO theory-of-mind claim; this builds the baseline Rung 2 must beat.

Setup: shared 5x5 world; mirro and vela loaded READ-ONLY and forked as the two creatures; B (the 'other') moves
goal-directed via comfort-gated BFS toward the nearest cell of color argmax(B.value_counts); A observes B's exact
cell each step (a PROVIDED obs_other_pos modality). Three one-step predictors of B's NEXT cell, scored over the run,
fresh fork-seeds {0-7} (8 seeds), H steps each:
  M0 occupancy-marginal: B's running position histogram (ignores current cell).
  M1 position-tracker (BASELINE): q_other = delta(observed B cell) advanced one step through the SHARED INNATE
     random-walk B (uniform over the 4 wall-clamped moves) — goal-AGNOSTIC local dynamics.
  M2 goal-oracle (UPPER BOUND): knows B's true goal + B's actual BFS policy -> near-exact next-cell prediction.
Predictor score = mean one-step next-cell accuracy (argmax prediction == B's actual next cell) and mean log-loss.

Prediction (hypothesis TRUE): M1_acc > M0_acc in >=7/8 seeds AND mean(M2_acc - M1_acc) >= 0.05 (headroom exists).
Trunk integrity: mirro and vela _state_hash() byte-identical before vs after (the spines are NEVER saved).

Falsifier (FALSE / halt):
  F1: M1_acc <= M0_acc (position baseline not better than a static prior -> substrate mis-specified).
  F2: mean(M2_acc - M1_acc) < 0.05 (NO headroom: goal-knowledge does not beat position-tracking -> Rung 2
      goal-inference has a near-zero ceiling; an honest predicted-negative BOUND, not a pass).
  F3: any committed spine _state_hash() changes (a fork mutated the trunk -> continuity bug, halt).
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
M2_EPSILON = 1e-6                 # smoothing for M2 log-loss
OUTPUT_PATH = Path("experiments/outputs/exp228.txt")

# Comfort-gate parameters (reuse exp70 constants)
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
# BFS helpers: build distance field and nearest-source policy for a given world
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


def bfs_next_cell(world: World, dist: list, current: int, rng: np.random.Generator) -> tuple[int, int]:
    """Return (action, next_cell) from greedy BFS toward the source, with RNG tie-break.

    This is the deterministic (non-epsilon, non-comfort-gated) BFS move used by M2
    to predict the oracle next cell.  Tie-breaking is deterministic given the rng.
    """
    best_d = min(dist[world.move(current, a)] for a in range(4))
    minimizers = [a for a in range(4) if dist[world.move(current, a)] == best_d]
    action = int(minimizers[rng.integers(0, len(minimizers))])
    next_cell = world.move(current, action)
    return action, next_cell


def policy_action(
    world: World,
    dist: list,
    current: int,
    est: float,
    rng: np.random.Generator,
) -> int:
    """Comfort-gated epsilon-greedy BFS policy (mirrors exp70's policy_action).

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


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

# Implementation choice: use VELA's world (vela-fork is creature B, it lives in its own world).
# Both worlds are 5x5/3-color; the choice affects only whose cmap B uses for comfort.
# B (vela-fork) navigates toward the nearest cell of color argmax(B.value_counts) in vela's world.
# A (mirro-fork) is the observer/predictor; it uses the same shared world (vela's) so the
# transition matrix B_innate is the same for all predictors.

seed_results = []  # list of dicts, one per seed

for seed in SEEDS:
    rng = np.random.default_rng(seed)

    # Fork both spines; never save them
    b_creature = vela_spine.fork(f"exp228-vela-fork-seed{seed}")
    a_creature = mirro_spine.fork(f"exp228-mirro-fork-seed{seed}")

    # Use vela's world as the shared world (B lives here natively)
    world = vela_spine.world
    n_cells = world.n_cells

    # Innate transition matrix: B_innate[s', s, a] = 1 if move(s,a)==s'
    # M1: q_next = mean over actions of B_innate[:, current, a]
    B_innate = world.transition_matrix()  # shape (n_cells, n_cells, 4)

    # B's favorite-color source: argmax(value_counts) -> find all matching cells,
    # BFS from each and pick the one closest to B's current position.
    # We build a BFS distance field from each candidate source; at each step we re-eval
    # the nearest one (since B's position changes, nearest target may change).
    # Implementation: compute a dist field from each source cell, then at each step
    # pick the source whose dist[current_pos] is minimal, use that dist field for policy.
    fav_color = int(np.argmax(b_creature.value_counts))
    source_cells = [i for i, c in enumerate(world.cmap) if c == fav_color]

    # Pre-compute a dist field per source cell
    dist_fields = {sc: build_bfs_dist(world, sc) for sc in source_cells}

    # M0: running histogram of B's actual positions (smoothed with +1 uniform prior)
    pos_histogram = np.ones(n_cells)  # +1 smoothing

    # Comfort estimate for B's gate (mirrors exp70)
    comfort_est = EST_INIT

    # Per-step accumulators
    m0_correct = 0
    m1_correct = 0
    m2_correct = 0
    m0_logloss = 0.0
    m1_logloss = 0.0
    m2_logloss = 0.0

    for step in range(H):
        b_pos = b_creature.true_pos

        # Update M0 histogram with current observed position BEFORE predicting
        # (The histogram represents B's empirical marginal, ignoring current cell)
        pos_histogram[b_pos] += 1.0

        # --- M0: normalize running histogram -> distribution over next cell ---
        m0_pred = pos_histogram / pos_histogram.sum()

        # --- M1: one-hot at current b_pos, advance through uniform random walk ---
        # q_next[s'] = mean_a B_innate[s', b_pos, a]
        m1_pred = B_innate[:, b_pos, :].mean(axis=1)  # shape (n_cells,)

        # --- M2: goal-oracle ---
        # Find nearest source to B's current position
        nearest_src = min(source_cells, key=lambda sc: dist_fields[sc][b_pos])
        dist_to_use = dist_fields[nearest_src]

        # Determine the actual action B will take (same policy, same rng state)
        # We need to predict using the same rng draw that B will use.
        # Implementation choice: fork the rng state for M2 prediction.
        # We use a separate oracle-rng seeded the same way to peek at B's next action.
        # However, since the rng is shared between policy and oracle, we need a cleaner
        # approach: the M2 oracle knows B's policy deterministically.
        # We simulate: peek at the rng state without advancing it for B.
        # Implementation: use a separate oracle_rng (seeded identically per step)
        # that is NOT the real B rng — instead, re-derive the expected next cell.
        # Since the policy has epsilon and gate randomness, and tie-breaks use rng,
        # the oracle can only predict the BFS greedy next cell (accounting for epsilon).
        # For M2 to be a near-one-hot upper bound, we assign:
        #   - prob (1-EPS) * gate_open -> BFS-greedy argmax (deterministic if no tie)
        #   - remaining mass spread over 4 random actions
        # Then argmax -> BFS greedy cell if gate open and not epsilon.
        # However, the spec says "near-exact next-cell prediction" with small epsilon smoothing.
        # Simpler correct interpretation: M2 produces near-one-hot at the BFS-next cell.
        # We implement as: M2_pred = (1-M2_EPSILON) * one_hot(bfs_next) + M2_EPSILON/n_cells.
        # This is the "knows B's true goal + BFS policy" oracle.
        if dist_to_use[b_pos] == 0:
            # Already at source: any move may be taken (random walk step from source)
            # BFS "greedy" from source = all 4 actions have dist >= 0; some stay at 0 (wall-clamped)
            # Take the move that keeps distance 0 if possible, else minimum
            best_d = min(dist_to_use[world.move(b_pos, a)] for a in range(4))
            minimizers = [a for a in range(4) if dist_to_use[world.move(b_pos, a)] == best_d]
            # For M2, predict uniform over neighbors tied for best BFS move
            m2_pred = np.full(n_cells, M2_EPSILON / n_cells)
            weight = (1.0 - M2_EPSILON) / len(minimizers)
            for a in minimizers:
                nc = world.move(b_pos, a)
                m2_pred[nc] += weight
        else:
            best_d = min(dist_to_use[world.move(b_pos, a)] for a in range(4))
            minimizers = [a for a in range(4) if dist_to_use[world.move(b_pos, a)] == best_d]
            # M2: near-one-hot distributed over BFS-greedy next cells
            m2_pred = np.full(n_cells, M2_EPSILON / n_cells)
            weight = (1.0 - M2_EPSILON) / len(minimizers)
            for a in minimizers:
                nc = world.move(b_pos, a)
                m2_pred[nc] += weight

        # --- B moves (the real policy) ---
        # Update comfort estimate (mirrors exp70: update BEFORE policy action)
        at_source = (b_pos == nearest_src)
        if at_source:
            comfort_est = (1 - ALPHA_EMA) * comfort_est + ALPHA_EMA * 1.0
        else:
            comfort_est += LAMBDA_RELAX * (1.0 - comfort_est)

        action = policy_action(world, dist_to_use, b_pos, comfort_est, rng)
        actual_next = world.move(b_pos, action)

        # Update B's true position
        b_creature.true_pos = actual_next

        # --- Score predictors ---
        m0_argmax = int(np.argmax(m0_pred))
        m1_argmax = int(np.argmax(m1_pred))
        m2_argmax = int(np.argmax(m2_pred))

        m0_correct += int(m0_argmax == actual_next)
        m1_correct += int(m1_argmax == actual_next)
        m2_correct += int(m2_argmax == actual_next)

        m0_logloss += -np.log(m0_pred[actual_next] + 1e-12)
        m1_logloss += -np.log(m1_pred[actual_next] + 1e-12)
        m2_logloss += -np.log(m2_pred[actual_next] + 1e-12)

    m0_acc = m0_correct / H
    m1_acc = m1_correct / H
    m2_acc = m2_correct / H
    m0_ll = m0_logloss / H
    m1_ll = m1_logloss / H
    m2_ll = m2_logloss / H

    seed_results.append({
        "seed": seed,
        "m0_acc": m0_acc,
        "m1_acc": m1_acc,
        "m2_acc": m2_acc,
        "m0_ll": m0_ll,
        "m1_ll": m1_ll,
        "m2_ll": m2_ll,
        "headroom": m2_acc - m1_acc,
        "m1_beats_m0": m1_acc > m0_acc,
    })

# ---------------------------------------------------------------------------
# Spine integrity check (F3)
# ---------------------------------------------------------------------------

hash_mirro_after = mirro_spine._state_hash()
hash_vela_after = vela_spine._state_hash()

f3_mirro_ok = hash_mirro_before == hash_mirro_after
f3_vela_ok = hash_vela_before == hash_vela_after
f3_ok = f3_mirro_ok and f3_vela_ok

# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------

m1_beats_m0_count = sum(1 for r in seed_results if r["m1_beats_m0"])
mean_headroom = float(np.mean([r["headroom"] for r in seed_results]))
mean_m0_acc = float(np.mean([r["m0_acc"] for r in seed_results]))
mean_m1_acc = float(np.mean([r["m1_acc"] for r in seed_results]))
mean_m2_acc = float(np.mean([r["m2_acc"] for r in seed_results]))

# ---------------------------------------------------------------------------
# Hypothesis evaluation
# ---------------------------------------------------------------------------

# Prediction TRUE requires BOTH:
#   (a) M1_acc > M0_acc in >=7/8 seeds
#   (b) mean(M2_acc - M1_acc) >= 0.05
pred_a_holds = m1_beats_m0_count >= 7
pred_b_holds = mean_headroom >= 0.05

# Falsifiers
f1_fired = not pred_a_holds   # M1 not better than M0 in >=7/8 seeds
f2_fired = not pred_b_holds   # headroom < 0.05
# F3 evaluated above

hypothesis_true = pred_a_holds and pred_b_holds and f3_ok

# ---------------------------------------------------------------------------
# Build output string
# ---------------------------------------------------------------------------

lines = []
lines.append("=" * 72)
lines.append("Exp 228 — self-other-modeling Rung 1")
lines.append("=" * 72)
lines.append("")
lines.append("Implementation notes:")
lines.append("  - Shared world: vela's world (vela-fork is B, lives here natively)")
lines.append("  - B's favorite color: argmax(vela.value_counts) = color 2")
lines.append("  - Source cells: all cells with cmap[i]==2 in vela's world")
lines.append("  - At each step, nearest source = argmin dist[b_pos] over source cells")
lines.append("  - BFS tie-break (M2): uniform over minimizer actions -> uniform over")
lines.append("    tied next cells (weighted by 1-M2_EPSILON)")
lines.append(f"  - M2_EPSILON = {M2_EPSILON}")
lines.append("  - Comfort gate: ALPHA=0.1, THRESH=0.75, LAMBDA=0.01, EPS_policy=0.2")
lines.append("  - M0 initialized with +1 smoothing (uniform prior)")
lines.append("  - Scoring: BEFORE B moves, predictors output; AFTER B moves, score")
lines.append("")

lines.append("--- PER-SEED TABLE ---")
lines.append("")
hdr = (
    f"{'seed':>4}  {'M0_acc':>8}  {'M1_acc':>8}  {'M2_acc':>8}  "
    f"{'headroom':>10}  {'M0_ll':>8}  {'M1_ll':>8}  {'M2_ll':>8}  {'M1>M0':>5}"
)
lines.append(hdr)
lines.append("-" * 80)
for r in seed_results:
    row = (
        f"  {r['seed']:>2}  {r['m0_acc']:>8.4f}  {r['m1_acc']:>8.4f}  {r['m2_acc']:>8.4f}  "
        f"{r['headroom']:>10.4f}  {r['m0_ll']:>8.4f}  {r['m1_ll']:>8.4f}  {r['m2_ll']:>8.4f}  "
        f"{'YES' if r['m1_beats_m0'] else 'NO':>5}"
    )
    lines.append(row)

lines.append("-" * 80)
lines.append(
    f"  {'mean':>4}  {mean_m0_acc:>8.4f}  {mean_m1_acc:>8.4f}  {mean_m2_acc:>8.4f}  "
    f"{mean_headroom:>10.4f}"
)
lines.append("")

lines.append("--- AGGREGATES ---")
lines.append("")
lines.append(f"  M1 > M0 seeds:        {m1_beats_m0_count}/8  (threshold: >=7/8)")
lines.append(f"  mean(M2 - M1) headroom: {mean_headroom:.4f}  (threshold: >=0.05)")
lines.append(f"  mean M0 accuracy:     {mean_m0_acc:.4f}")
lines.append(f"  mean M1 accuracy:     {mean_m1_acc:.4f}")
lines.append(f"  mean M2 accuracy:     {mean_m2_acc:.4f}")
lines.append("")

lines.append("--- SPINE INTEGRITY (F3) ---")
lines.append("")
lines.append(f"  mirro before: {hash_mirro_before[:24]}...")
lines.append(f"  mirro after:  {hash_mirro_after[:24]}...")
lines.append(f"  mirro intact: {'YES' if f3_mirro_ok else 'NO --- F3 FIRED'}")
lines.append(f"  vela before:  {hash_vela_before[:24]}...")
lines.append(f"  vela after:   {hash_vela_after[:24]}...")
lines.append(f"  vela intact:  {'YES' if f3_vela_ok else 'NO --- F3 FIRED'}")
lines.append(f"  F3: {'NOT FIRED' if f3_ok else 'FIRED (continuity bug)'}")
lines.append("")

lines.append("--- FALSIFIER EVALUATION ---")
lines.append("")
lines.append(
    f"  F1 (M1 not better than M0 in >=7/8 seeds): "
    f"{'NOT FIRED' if not f1_fired else 'FIRED'}  "
    f"[M1>M0 in {m1_beats_m0_count}/8 seeds, need >=7]"
)
lines.append(
    f"  F2 (mean headroom M2-M1 < 0.05): "
    f"{'NOT FIRED' if not f2_fired else 'FIRED'}  "
    f"[mean headroom = {mean_headroom:.4f}, threshold 0.05]"
)
lines.append(
    f"  F3 (spine hash changed): "
    f"{'NOT FIRED' if f3_ok else 'FIRED'}  "
    f"[mirro {'OK' if f3_mirro_ok else 'CHANGED'}, vela {'OK' if f3_vela_ok else 'CHANGED'}]"
)
lines.append("")

lines.append("--- PREDICTION CONJUNCT EVALUATION ---")
lines.append("")
lines.append(
    f"  (a) M1>M0 in >=7/8 seeds: {'TRUE' if pred_a_holds else 'FALSE'}  "
    f"({m1_beats_m0_count}/8)"
)
lines.append(
    f"  (b) mean(M2-M1) >= 0.05:  {'TRUE' if pred_b_holds else 'FALSE'}  "
    f"({mean_headroom:.4f})"
)
lines.append("")

if hypothesis_true:
    verdict = "VERDICT: TRUE — both conjuncts hold; F1, F2, F3 not fired; Rung-1 substrate confirmed"
elif f3_ok and not f1_fired and not f2_fired:
    verdict = "VERDICT: TRUE — both conjuncts hold; no falsifiers fired"
elif f3_ok and f1_fired and not f2_fired:
    verdict = "VERDICT: FALSE/F1 — position baseline not better than static prior; substrate mis-specified"
elif f3_ok and not f1_fired and f2_fired:
    verdict = (
        "VERDICT: FALSE/F2 — no headroom; goal-knowledge does not beat position-tracking; "
        "honest predicted-negative bound; Rung-2 ceiling near-zero"
    )
elif f3_ok and f1_fired and f2_fired:
    verdict = "VERDICT: FALSE/F1+F2 — both falsifiers fired"
else:
    verdict = "VERDICT: HALT/F3 — spine integrity violated; continuity bug"

lines.append(verdict)
lines.append("")
lines.append("=" * 72)

output_text = "\n".join(lines)

# ---------------------------------------------------------------------------
# Print and write
# ---------------------------------------------------------------------------

print(output_text)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(output_text + "\n")

print(f"\nOutput written to: {OUTPUT_PATH}")

# Halt on F3
if not f3_ok:
    print("HALT: F3 fired — spine integrity violated")
    sys.exit(1)
