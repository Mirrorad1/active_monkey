"""Exp 58 — cascade round 2, the kill test: the mass-vs-tempo plasticity law.

Correction to Exp 57's queued criterion (owned): 'low-mass newborn recovers >= 6/8'
mis-translates the law -- counts accumulate DURING the schedule, so every non-decaying
creature crosses the ~20-count threshold within a segment or two. The law's honest test
is PER-OUTCOME prediction:
  LAW: a creature recovers segment k iff R_k < 1, where R_k = (mean old-color column
  mass on the cells whose color changes at segment k) / 20 (= 500 steps / 25 cells of
  per-segment evidence at each cell).
Tests, all predeclared:
(1) Mass-swept cohort: newborns with settling 50 / 250 / 1000 steps (seeds 91/92/93,
    settled in the segment-0 world), then the 8-segment schedule. Before each segment
    the script computes R_k from the creature's own pA (mass of the OUTGOING color on
    cells about to change). PASS iff (R_k < 1) == recovered in >= 80% of the 24
    (subject, segment) cells (segments with no changed cells count as R_k = 0,
    predicted-recover).
(2) Forgetting counterfactual: a 1000-step-settled creature (seed 94) whose pA is
    multiplied by 0.9 after every step (DECLARED harness modification = exponential
    forgetting; effective mass saturates ~ 1/(1-0.9) x per-step increment ~ 10 < 20).
    The law predicts it TRACKS the drift: recovery in >= 6/8 segments, where every
    non-decaying creature freezes.
Verdict (predeclared): BOTH pass -> the NOVELTY-CANDIDATE DIES as the lawful
consequence of non-decaying counts; the finding is the unified plasticity law
(Exp 48 values + Exp 56/57 maps): lifelong adaptation requires forgetting once
accumulated evidence outruns world tempo. EITHER fails -> candidate survives round 2;
cascade checks exhausted; consult the human.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from active_loop.creature import Creature, World

ROWS, COLS = 5, 5
N_COLORS = 3
N_CELLS = ROWS * COLS
CORNER_CYCLE = [(0, 0), (0, 2), (2, 2), (2, 0)]
N_SEGMENTS = 8
SEG_STEPS = 500
CHECK_EVERY = 25
RECOVERY_THRESH = 0.92
LAG_MAX = 450
MASS_THRESHOLD = 20.0  # 500 steps / 25 cells


def build_cmap(corner):
    """5x5 cmap: background = i%2; 3x3 patch at corner = color 2."""
    cmap = [(i % 2) for i in range(N_CELLS)]
    r0, c0 = corner
    for dr in range(3):
        for dc in range(3):
            r, c = r0 + dr, c0 + dc
            cmap[r * COLS + c] = 2
    return cmap


# Build all 8 segment cmaps (corner cycles mod 4)
seg_cmaps = [build_cmap(CORNER_CYCLE[k % 4]) for k in range(N_SEGMENTS)]


def changed_cells(k):
    """Cells whose color differs between segment k-1 and segment k.
    For k=0: compared to segment-0 world itself -> empty set.
    """
    if k == 0:
        return []
    prev = seg_cmaps[k - 1]
    curr = seg_cmaps[k]
    return [cell for cell in range(N_CELLS) if prev[cell] != curr[cell]]


def compute_R(c, k):
    """R_k = mean over changed_cells(k) of pA[outgoing_color, cell] / MASS_THRESHOLD."""
    cells = changed_cells(k)
    if not cells:
        return 0.0
    prev_cmap = seg_cmaps[k - 1]
    masses = [c.pA[prev_cmap[cell], cell] for cell in cells]
    return float(np.mean(masses)) / MASS_THRESHOLD


def run_segment(c, k, base_seed, use_decay=False):
    """Run one 500-step segment; return (recovered, lag)."""
    world = World(rows=ROWS, cols=COLS, cmap=seg_cmaps[k], n_colors=N_COLORS)
    c.world = world
    lag = None
    for step_i in range(SEG_STEPS):
        global_idx = k * SEG_STEPS + step_i
        seed = base_seed * 1000003 + global_idx
        c.live(1, seed=seed)
        if use_decay:
            c.pA *= 0.9
            c.pA = np.maximum(c.pA, 0.01)
        if (step_i + 1) % CHECK_EVERY == 0:
            acc = c.map_accuracy()
            if lag is None and acc >= RECOVERY_THRESH:
                lag = step_i + 1
    recovered = lag is not None and lag <= LAG_MAX
    return recovered, lag


def settle(c, steps, base_seed):
    """Settle creature for given steps in the segment-0 world."""
    world = World(rows=ROWS, cols=COLS, cmap=seg_cmaps[0], n_colors=N_COLORS)
    c.world = world
    for i in range(steps):
        c.live(1, seed=base_seed * 1000003 + i)


# ---------------------------------------------------------------------------
# Test 1: Mass-swept cohort
# ---------------------------------------------------------------------------
print("=" * 60)
print("TEST 1: Mass-swept cohort")
print("=" * 60)

cohort = [(50, 91, 191), (250, 92, 192), (1000, 93, 193)]
total_match = 0
total_cells = 0

for settle_steps, birth_seed, base_seed in cohort:
    world0 = World(rows=ROWS, cols=COLS, cmap=seg_cmaps[0], n_colors=N_COLORS)
    c = Creature.birth(f"cohort_{settle_steps}", world0, seed=birth_seed)
    settle(c, settle_steps, base_seed)

    for k in range(N_SEGMENTS):
        R_k = compute_R(c, k)
        predicted_rec = R_k < 1.0
        recovered, lag = run_segment(c, k, base_seed)
        actual_rec = recovered
        match = predicted_rec == actual_rec
        lag_disp = lag if lag is not None else ">500"
        print(
            f"settle{settle_steps} seg{k} R={R_k:.2f} "
            f"predicted={'rec' if predicted_rec else 'frozen'} "
            f"actual={'rec' if actual_rec else 'frozen'} "
            f"match={match} lag={lag_disp}"
        )
        total_match += int(match)
        total_cells += 1

frac1 = total_match / total_cells
pass1 = frac1 >= 0.80
print(f"\nrule_match={total_match}/{total_cells} ({frac1:.3f}) -> {'PASS' if pass1 else 'FAIL'}")

# ---------------------------------------------------------------------------
# Test 2: Forgetting counterfactual
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 2: Forgetting counterfactual (decay=0.9/step)")
print("=" * 60)

world0 = World(rows=ROWS, cols=COLS, cmap=seg_cmaps[0], n_colors=N_COLORS)
c_decay = Creature.birth("forgetting", world0, seed=94)

# Settle 1000 steps with decay
base_seed_decay = 194
settle_world = World(rows=ROWS, cols=COLS, cmap=seg_cmaps[0], n_colors=N_COLORS)
c_decay.world = settle_world
for i in range(1000):
    c_decay.live(1, seed=base_seed_decay * 1000003 + i)
    c_decay.pA *= 0.9
    c_decay.pA = np.maximum(c_decay.pA, 0.01)

n_recovered_decay = 0
for k in range(N_SEGMENTS):
    R_k = compute_R(c_decay, k)
    recovered, lag = run_segment(c_decay, k, base_seed_decay, use_decay=True)
    lag_disp = lag if lag is not None else ">500"
    print(
        f"decay seg{k} R={R_k:.2f} "
        f"actual={'rec' if recovered else 'frozen'} lag={lag_disp}"
    )
    if recovered:
        n_recovered_decay += 1

pass2 = n_recovered_decay >= 6
print(f"\nforgetting variant recovered {n_recovered_decay}/8 -> {'PASS' if pass2 else 'FAIL'}")

# ---------------------------------------------------------------------------
# Summary & Verdict
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"(1) rule_match={total_match}/{total_cells} ({frac1:.3f}) -> {'PASS' if pass1 else 'FAIL'}")
print(f"(2) forgetting variant recovered {n_recovered_decay}/8 -> {'PASS' if pass2 else 'FAIL'}")
print()
if pass1 and pass2:
    print(
        "VERDICT: candidate DIES — unified mass-vs-tempo plasticity law confirmed "
        "(adaptation requires forgetting)"
    )
else:
    fails = []
    if not pass1:
        fails.append("(1) rule_match")
    if not pass2:
        fails.append("(2) forgetting variant")
    print(
        f"VERDICT: candidate SURVIVES round 2 ({', '.join(fails)} failed) — "
        "cascade exhausted, consulting human"
    )
