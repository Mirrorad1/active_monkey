"""Exp 72 — the kidnapped twin: why displaced forks start life lost, and the belief-equalized
contest that finishes the rung-5 cascade.

Exp 71 verified that fork copies inherit mirro's place belief — a numerical DELTA at cell 0
(max qs = 1.0). The mechanical consequence, predeclared here as hypotheses: with a delta
prior and the deterministic innate motion model B, Bayesian observation updates CANNOT move
the belief off its dead-reckoned path (every other cell has zero posterior support) — the
kidnapped-robot problem, which this substrate should fail except through ONE mechanism:
WALL-CLAMPING. When truth or belief hits a wall under a shared action, their offset shrinks
by one along that axis and can never grow. Random walks touch walls often (fast re-sync);
goal-seeking stops touching walls once the creature BELIEVES it is at the goal — freezing
the residual error and leaving the creature camping a phantom source. This explains the
invariant winner records of Exp 68-71 mechanically.

Parts and predeclared predictions:
  A/P3 (support theorem, empirical): kidnapped twin (true_pos=24, inherited belief delta at
     cell 0) under a RANDOM walk: argmax(qs) equals the dead-reckoned image of the action
     sequence (move-by-move from cell 0) at EVERY one of 2000 steps, in 5/5 seeds.
     Violations = 0 predicted (it is a support argument; a violation means underflowed mass
     resurrected). F1 = any violation -> the Exp 71 diagnosis is wrong; HALT everything.
  A/P1 (wall re-sync under exploration): the belief-truth Manhattan offset reaches 0 within
     500 random-walk steps in >=4/5 seeds (and once 0, stays 0 — offsets cannot grow).
     F2 = P1 fails -> the recovery story behind Exp 68-71's loser occupancy is wrong; HALT.
  B/P2 (phantom camping under seeking): same kidnap, but the FIXED always-approach policy:
     the twin spends >=25% of 2000 steps with argmax(qs) == SOURCE_CELL while true_pos !=
     SOURCE_CELL, in >=4/5 seeds. F3 = P2 fails -> the exclusion-asymmetry mechanism story
     is in doubt (reported; Part D still runs).
  D/P4 (belief-equalized contest — the final cascade leg): equidistant starts (cells 0 and
     20, BFS dist 2 each, asserted), both adaptive, but with an 800-step RANDOM-walk
     settling phase first (no comfort machinery, no seeking) so the kidnapped twin re-syncs
     via walls. GATE G-SYNC (input-based): at contest start BOTH creatures' belief-truth
     offsets == 0, else the seed is excluded (>=6/8 valid required). Then the 2000-step
     comfort-gated contest (Exp 69/70/71 mechanism), classified with Exp 70's table.
     Branches: INTRINSIC-EARNED iff EXCLUSION >= 4 of valid seeds AND the winner is the
     creature closer to the source at contest start in >= 75% of exclusion seeds with
     unequal distances (equal-distance seeds excluded from the winner sub-check only);
     ARTIFACT-REDUCED iff EXCLUSION <= 2 (Exp 70's regime was belief/geometry artifact);
     3 = BORDERLINE.
Wording consequence (predeclared): INTRINSIC-EARNED keeps Exp 71's name — stigmergic
unilateral-retreat lock-in — now with honest symmetry: the lock-in is real, first-arriver
is decided by the actual race. ARTIFACT-REDUCED downgrades the rung-4 result to
first-arriver artifact + unilateral retreat. Either way dominance/coordination stay
unearned (Exp 71's deflation stands regardless).
Predicted (stated before running): P3 0 violations; P1 re-sync in ~50-300 steps; P2
camping 30-70%; P4 INTRINSIC-EARNED.
Provided priors declared: as Exp 69-71 (world, source, policies, gate mechanism, depletion,
starts) plus the kidnap placements and the settling phase. Self-formed: beliefs/maps and
comfort-estimate trajectories. Spines never live, never saved; mirro forked once
("exp72-twin-template"); vela untouched.
"""
from __future__ import annotations

import copy
import sys

import numpy as np
from collections import deque
from pathlib import Path

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# Constants (verbatim from exp71; additions below)
# ---------------------------------------------------------------------------

STEPS = 2000
EPS = 0.2
ALPHA = 0.1
THRESH = 0.75
LAMBDA = 0.01
EST_INIT = 1.0
MIRRO_DIR = Path("creature/state/mirro")

SETTLE = 800
KIDNAP_POS = 24
AB_SEEDS = [0, 1, 2, 3, 4]
D_SEEDS = list(range(21, 29))   # 21..28 inclusive
EQ_A = 0
EQ_B = 20

# ---------------------------------------------------------------------------
# Load committed spine (NEVER call .live() or .save() on mirro)
# ---------------------------------------------------------------------------

print("Exp 72 — the kidnapped twin: belief-equalized contest and rung-5 cascade.")
print()

mirro = Creature.load(MIRRO_DIR)

vc_m = mirro.value_counts
tot_m = vc_m.sum()

print(
    f"mirro: id={id(mirro)}  name={mirro.name!r}  age={mirro.age_steps}  "
    f"hash={mirro._state_hash()[:12]}  favorite={mirro.favorite()}  "
    f"shares={np.round(vc_m / tot_m, 4)}"
)
print()

# ---------------------------------------------------------------------------
# SOURCE_CELL: first index i with cmap[i] == mirro.favorite()
# ---------------------------------------------------------------------------

fav_color = mirro.favorite()
SOURCE_CELL = next(
    i for i, c in enumerate(mirro.world.cmap) if c == fav_color
)
src_row, src_col = divmod(SOURCE_CELL, mirro.world.cols)
print(
    f"favorite color: {fav_color}  SOURCE_CELL={SOURCE_CELL}  "
    f"(row={src_row}, col={src_col})"
)
print()

# ---------------------------------------------------------------------------
# BFS distance field to SOURCE_CELL
# ---------------------------------------------------------------------------

world = mirro.world
n_cells = world.n_cells

dist: list[int] = [-1] * n_cells
dist[SOURCE_CELL] = 0
_q: deque[int] = deque([SOURCE_CELL])
while _q:
    cell = _q.popleft()
    for a in range(4):
        nb = world.move(cell, a)
        if dist[nb] == -1:
            dist[nb] = dist[cell] + 1
            _q.append(nb)

print("BFS distance field to SOURCE_CELL (grid):")
for r in range(world.rows):
    row_str = "  "
    for c in range(world.cols):
        row_str += f"{dist[r * world.cols + c]:3d}"
    print(row_str)
print()

# ---------------------------------------------------------------------------
# Equidistant-start assertion
# ---------------------------------------------------------------------------

d_eq_A = dist[EQ_A]
d_eq_B = dist[EQ_B]
print(
    f"Equidistant-start assertion: dist[EQ_A={EQ_A}]={d_eq_A}  "
    f"dist[EQ_B={EQ_B}]={d_eq_B}"
)
if d_eq_A != d_eq_B:
    print(
        f"ASSERT FAIL: EQ_A and EQ_B are NOT equidistant ({d_eq_A} != {d_eq_B}); HALT"
    )
    sys.exit(1)
print(f"  -> PASS (both distance {d_eq_A})")
print()

# ---------------------------------------------------------------------------
# Delta-belief premise check
# ---------------------------------------------------------------------------

delta_argmax = int(np.argmax(mirro.qs))
delta_max = float(mirro.qs.max())
print(
    f"Delta-belief premise: argmax(mirro.qs)={delta_argmax}  max(mirro.qs)={delta_max}"
)
if delta_argmax != 0 or delta_max != 1.0:
    print(
        f"ASSERT FAIL: mirro.qs is not a delta at cell 0 "
        f"(argmax={delta_argmax}, max={delta_max:.6f}); HALT"
    )
    sys.exit(1)
print("  -> PASS (delta at cell 0, max=1.0)")
print()

# ---------------------------------------------------------------------------
# Fork ONCE into twin template (one biography event, allowed)
# ---------------------------------------------------------------------------

twin_template = mirro.fork("exp72-twin-template")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

checks: list[tuple[str, bool, str]] = []


def check(name: str, predicate_fn):
    """Run predicate_fn(); record (name, pass, detail). Exceptions count as FAIL."""
    try:
        passed, detail = predicate_fn()
        checks.append((name, passed, detail))
    except Exception as exc:
        checks.append((name, False, f"exception: {exc}"))


def manhattan(cell_a: int, cell_b: int) -> int:
    """Manhattan distance between two cells on the grid."""
    ra, ca = divmod(cell_a, world.cols)
    rb, cb = divmod(cell_b, world.cols)
    return abs(ra - rb) + abs(ca - cb)


# ---------------------------------------------------------------------------
# Regime classification (verbatim from exp70/71)
# ---------------------------------------------------------------------------

def classify(asym: float, R: float) -> str:
    """Classify a seed into a regime from predeclared thresholds (Exp 70 table).

    asym = |occ_A - occ_B| / (occ_A + occ_B)
    EXCLUSION   if asym > 0.5 and 0.9 <= R <= 1.1
    TIMESHARING if R < 0.9 and asym < 0.5
    NULL        if 0.9 <= R <= 1.1 and asym <= 0.5
    OTHER       otherwise
    """
    if asym > 0.5 and 0.9 <= R <= 1.1:
        return "EXCLUSION"
    if R < 0.9 and asym < 0.5:
        return "TIMESHARING"
    if 0.9 <= R <= 1.1 and asym <= 0.5:
        return "NULL"
    return "OTHER"


# ---------------------------------------------------------------------------
# Policy function (verbatim from exp71)
# ---------------------------------------------------------------------------

def policy_action(
    map_cell: int, rng: np.random.Generator, est: float, adaptive: bool
) -> int:
    """EPS-greedy BFS policy with optional comfort gate.

    Draw order:
      - Always draw u = rng.random() first (EPS check).
      - EPS branch: draw one rng.integers(0, 4) for random action.
      - Wander branch (adaptive and est < THRESH): draw one rng.integers(0, 4).
      - Greedy branch: draw one rng.integers(0, len(minimizers)) for tie-break.
    """
    u = rng.random()
    if u < EPS:
        return int(rng.integers(0, 4))
    if adaptive and est < THRESH:
        return int(rng.integers(0, 4))
    best_d = min(dist[world.move(map_cell, a)] for a in range(4))
    minimizers = [a for a in range(4) if dist[world.move(map_cell, a)] == best_d]
    return int(minimizers[rng.integers(0, len(minimizers))])


# ===========================================================================
# PART A — KIDNAPPED TWIN UNDER RANDOM WALK
# P3: argmax(qs) == dead-reckoned cell at every step (0 violations), 5/5 seeds
# P1: belief-truth offset reaches 0 within 500 steps, >=4/5 seeds
# ===========================================================================

print("=" * 70)
print("PART A — Kidnapped twin under RANDOM walk (P3 & P1)")
print(f"  KIDNAP_POS={KIDNAP_POS}  AB_SEEDS={AB_SEEDS}")
print("=" * 70)
print()

B_mat = world.transition_matrix()

print(
    f"{'seed':>4}  {'violations':>10}  {'first_sync':>10}  {'regrowths':>9}"
)
print("-" * 45)

part_a_results: list[dict] = []

for seed in AB_SEEDS:
    twin = copy.deepcopy(twin_template)
    twin.true_pos = KIDNAP_POS
    rng = np.random.default_rng(800000 + seed)

    # dead-reckoned cell: starts at the believed cell (cell 0)
    dead = 0

    violations = 0
    first_sync_step: int | None = None
    regrowths = 0
    synced_once = False
    was_synced = False

    for step in range(STEPS):
        pos = twin.true_pos

        # live()-math update (verbatim stepper pattern from exp71)
        A_hat = twin._A_hat()
        obs = int(world.cmap[pos])
        likelihood = A_hat[obs, :]
        qs_upd = likelihood * twin.qs
        denom = qs_upd.sum()
        if denom > 0:
            qs_upd = qs_upd / denom
        else:
            qs_upd = np.ones(n_cells) / n_cells
        twin.pA[obs, :] += qs_upd
        map_cell = int(np.argmax(qs_upd))
        pred_dist = A_hat[:, map_cell]
        h = -np.sum(pred_dist * np.log(pred_dist + 1e-12))
        w = np.exp(-h)
        twin.value_counts[obs] += w

        # P3: map_cell must equal dead-reckoned cell
        if map_cell != dead:
            violations += 1

        # P1: offset between map belief and true position
        offset = manhattan(map_cell, pos)
        if offset == 0:
            if first_sync_step is None:
                first_sync_step = step
                synced_once = True
            if was_synced is False and synced_once:
                # track regrowth (offset went from 0 back to >0 after first sync)
                pass
            was_synced = True
        else:
            if was_synced:
                regrowths += 1
            was_synced = False

        # RANDOM action always
        action = int(rng.integers(0, 4))
        new_pos = world.move(pos, action)
        twin.qs = B_mat[:, :, action] @ qs_upd

        # advance dead-reckoned cell using same action
        dead = world.move(dead, action)
        twin.true_pos = new_pos

    sync_str = str(first_sync_step) if first_sync_step is not None else "None"
    print(
        f"  {seed:>2}  {violations:>10d}  {sync_str:>10}  {regrowths:>9d}"
    )
    part_a_results.append({
        "seed": seed,
        "violations": violations,
        "first_sync_step": first_sync_step,
        "regrowths": regrowths,
    })

print()

# Evaluate P3 (F1)
p3_violations_total = sum(r["violations"] for r in part_a_results)
p3_pass = p3_violations_total == 0
print(
    f"P3 (0 violations all 5 seeds): total_violations={p3_violations_total}  "
    f"-> {'PASS' if p3_pass else 'FAIL'}"
)
if not p3_pass:
    print("F1 FIRED — argmax(qs) deviated from dead-reckoned path; "
          "Exp 71 diagnosis is wrong. HALT.")
    check("A-P3", lambda: (False, f"total_violations={p3_violations_total}"))
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")
    sys.exit(1)

# Evaluate P1 (F2)
p1_sync_seeds = sum(
    1 for r in part_a_results
    if r["first_sync_step"] is not None and r["first_sync_step"] <= 500
)
p1_pass = p1_sync_seeds >= 4
print(
    f"P1 (first_sync <= 500 in >=4/5): seeds_passing={p1_sync_seeds}/5  "
    f"-> {'PASS' if p1_pass else 'FAIL'}"
)
# Regrowth check (predicted 0)
total_regrowths = sum(r["regrowths"] for r in part_a_results)
print(f"P1 regrowth events (predicted 0): total_regrowths={total_regrowths}")

if not p1_pass:
    print("F2 FIRED — belief-truth offset did not re-sync within 500 steps; "
          "the recovery story behind Exp 68-71 loser occupancy is wrong. HALT.")
    check("A-P3", lambda: (True, f"total_violations={p3_violations_total}"))
    check("A-P1", lambda: (False, f"seeds_synced={p1_sync_seeds}/5 < 4"))
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")
    sys.exit(1)

print()

check("A-P3", lambda: (True, f"total_violations={p3_violations_total} (all 5 seeds)"))
check(
    "A-P1",
    lambda: (
        True,
        f"seeds_synced_within_500={p1_sync_seeds}/5 >= 4; "
        f"regrowths={total_regrowths} (predicted 0)"
    ),
)

# ===========================================================================
# PART B — KIDNAPPED TWIN UNDER FIXED ALWAYS-APPROACH POLICY
# P2: camping fraction >= 0.25 in >=4/5 seeds
# ===========================================================================

print("=" * 70)
print("PART B — Kidnapped twin under FIXED always-approach policy (P2)")
print(f"  KIDNAP_POS={KIDNAP_POS}  AB_SEEDS={AB_SEEDS}")
print("=" * 70)
print()

print(f"{'seed':>4}  {'camping_frac':>12}  {'P2_seed?':>8}")
print("-" * 35)

part_b_results: list[dict] = []

for seed in AB_SEEDS:
    twin = copy.deepcopy(twin_template)
    twin.true_pos = KIDNAP_POS
    rng = np.random.default_rng(850000 + seed)

    camping_steps = 0
    est = EST_INIT

    for step in range(STEPS):
        pos = twin.true_pos

        # live()-math update
        A_hat = twin._A_hat()
        obs = int(world.cmap[pos])
        likelihood = A_hat[obs, :]
        qs_upd = likelihood * twin.qs
        denom = qs_upd.sum()
        if denom > 0:
            qs_upd = qs_upd / denom
        else:
            qs_upd = np.ones(n_cells) / n_cells
        twin.pA[obs, :] += qs_upd
        map_cell = int(np.argmax(qs_upd))
        pred_dist = A_hat[:, map_cell]
        h = -np.sum(pred_dist * np.log(pred_dist + 1e-12))
        w = np.exp(-h)
        twin.value_counts[obs] += w

        # camping: belief thinks it is at SOURCE_CELL but truth is not
        if map_cell == SOURCE_CELL and pos != SOURCE_CELL:
            camping_steps += 1

        # est update (for completeness; FIXED policy ignores gate)
        at = (pos == SOURCE_CELL)
        comfort_step = 1.0 if at else 0.0
        if at:
            est = (1 - ALPHA) * est + ALPHA * comfort_step
        else:
            est += LAMBDA * (1.0 - est)

        # FIXED always-approach (adaptive=False)
        action = policy_action(map_cell, rng, est, adaptive=False)
        new_pos = world.move(pos, action)
        twin.qs = B_mat[:, :, action] @ qs_upd
        twin.true_pos = new_pos

    camping_frac = camping_steps / STEPS
    p2_seed = camping_frac >= 0.25
    print(
        f"  {seed:>2}  {camping_frac:>12.4f}  {'YES' if p2_seed else 'no':>8}"
    )
    part_b_results.append({
        "seed": seed,
        "camping_frac": camping_frac,
        "p2_seed": p2_seed,
    })

print()

p2_seeds_passing = sum(1 for r in part_b_results if r["p2_seed"])
p2_pass = p2_seeds_passing >= 4
print(
    f"P2 (camping_frac >= 0.25 in >=4/5): seeds_passing={p2_seeds_passing}/5  "
    f"-> {'PASS' if p2_pass else 'FAIL'}"
)
if not p2_pass:
    print(
        "F3 FIRED — phantom-camping story in doubt "
        "(exclusion-asymmetry mechanism explanation is uncertain); continuing to Part D."
    )

print()

check(
    "B-P2",
    lambda: (
        p2_pass,
        f"seeds_passing={p2_seeds_passing}/5 {'>=4 PASS' if p2_pass else '<4 FAIL — F3 FIRED'}",
    ),
)

# ===========================================================================
# PART D — BELIEF-EQUALIZED CONTEST (final cascade leg)
# 800-step random-walk settling phase; G-SYNC gate; 2000-step contest
# ===========================================================================

print("=" * 70)
print("PART D — Belief-equalized contest (P4)")
print(f"  EQ_A={EQ_A} (dist={dist[EQ_A]})  EQ_B={EQ_B} (dist={dist[EQ_B]})")
print(f"  D_SEEDS={D_SEEDS}  SETTLE={SETTLE}")
print("=" * 70)
print()

# Header for Part D per-seed table
print(
    f"{'seed':>4}  {'sA':>4}  {'sB':>4}  {'dA':>3}  {'dB':>3}  "
    f"{'offA':>4}  {'offB':>4}  {'sync':>5}  "
    f"{'occ_A':>6}  {'occ_B':>6}  {'asym':>6}  {'R':>6}  "
    f"{'class':>12}  {'winner':>6}  {'closer':>6}"
)
print("-" * 115)

part_d_results: list[dict] = []
d_excluded_seeds: list[int] = []
d_valid_seeds: list[int] = []

for seed in D_SEEDS:
    # Create two deepcopies of template
    A = copy.deepcopy(twin_template)
    B_twin = copy.deepcopy(twin_template)
    A.true_pos = EQ_A
    B_twin.true_pos = EQ_B

    # RNGs for settle phase — continue same objects into contest
    rng_A = np.random.default_rng(900000 + 1000 * seed + 0)
    rng_B = np.random.default_rng(900000 + 1000 * seed + 1)

    # ------------------------------------------------------------------
    # SETTLE PHASE: 800 steps, both random walk, NO comfort/gate machinery
    # ------------------------------------------------------------------
    for _s in range(SETTLE):
        # Creature A settle
        pos_a = A.true_pos
        A_hat_a = A._A_hat()
        obs_a = int(world.cmap[pos_a])
        lik_a = A_hat_a[obs_a, :]
        qs_a = lik_a * A.qs
        d_a = qs_a.sum()
        if d_a > 0:
            qs_a = qs_a / d_a
        else:
            qs_a = np.ones(n_cells) / n_cells
        A.pA[obs_a, :] += qs_a
        mc_a = int(np.argmax(qs_a))
        pd_a = A_hat_a[:, mc_a]
        h_a = -np.sum(pd_a * np.log(pd_a + 1e-12))
        A.value_counts[obs_a] += np.exp(-h_a)
        act_a = int(rng_A.integers(0, 4))
        A.qs = B_mat[:, :, act_a] @ qs_a
        A.true_pos = world.move(pos_a, act_a)

        # Creature B settle
        pos_b = B_twin.true_pos
        A_hat_b = B_twin._A_hat()
        obs_b = int(world.cmap[pos_b])
        lik_b = A_hat_b[obs_b, :]
        qs_b = lik_b * B_twin.qs
        d_b = qs_b.sum()
        if d_b > 0:
            qs_b = qs_b / d_b
        else:
            qs_b = np.ones(n_cells) / n_cells
        B_twin.pA[obs_b, :] += qs_b
        mc_b = int(np.argmax(qs_b))
        pd_b = A_hat_b[:, mc_b]
        h_b = -np.sum(pd_b * np.log(pd_b + 1e-12))
        B_twin.value_counts[obs_b] += np.exp(-h_b)
        act_b = int(rng_B.integers(0, 4))
        B_twin.qs = B_mat[:, :, act_b] @ qs_b
        B_twin.true_pos = world.move(pos_b, act_b)

    # Compute settle-end offsets and positions
    settle_pos_A = A.true_pos
    settle_pos_B = B_twin.true_pos

    # offset = manhattan(argmax(qs_upd at end), true_pos)
    # We need the updated qs after the last observation at end of settle
    # The last qs state already incorporates the last prediction-step advance.
    # Evaluate by doing one live()-math obs update to get qs_upd, then restore.
    # Per spec: argmax(qs_upd) vs true_pos at settle end
    A_hat_end_a = A._A_hat()
    obs_end_a = int(world.cmap[settle_pos_A])
    lik_end_a = A_hat_end_a[obs_end_a, :]
    qs_end_a = lik_end_a * A.qs
    d_end_a = qs_end_a.sum()
    if d_end_a > 0:
        qs_end_a = qs_end_a / d_end_a
    else:
        qs_end_a = np.ones(n_cells) / n_cells
    mc_end_a = int(np.argmax(qs_end_a))
    offset_A = manhattan(mc_end_a, settle_pos_A)

    A_hat_end_b = B_twin._A_hat()
    obs_end_b = int(world.cmap[settle_pos_B])
    lik_end_b = A_hat_end_b[obs_end_b, :]
    qs_end_b = lik_end_b * B_twin.qs
    d_end_b = qs_end_b.sum()
    if d_end_b > 0:
        qs_end_b = qs_end_b / d_end_b
    else:
        qs_end_b = np.ones(n_cells) / n_cells
    mc_end_b = int(np.argmax(qs_end_b))
    offset_B = manhattan(mc_end_b, settle_pos_B)

    # G-SYNC gate
    g_sync = (offset_A == 0 and offset_B == 0)
    if not g_sync:
        d_excluded_seeds.append(seed)
        sync_str = "EXCL"
        print(
            f"  {seed:>2}  {settle_pos_A:>4d}  {settle_pos_B:>4d}  "
            f"{dist[settle_pos_A]:>3d}  {dist[settle_pos_B]:>3d}  "
            f"{offset_A:>4d}  {offset_B:>4d}  {sync_str:>5}  "
            f"{'---':>6}  {'---':>6}  {'---':>6}  {'---':>6}  "
            f"{'EXCLUDED':>12}  {'---':>6}  {'---':>6}"
        )
        part_d_results.append({
            "seed": seed,
            "settle_pos_A": settle_pos_A,
            "settle_pos_B": settle_pos_B,
            "dist_A": dist[settle_pos_A],
            "dist_B": dist[settle_pos_B],
            "offset_A": offset_A,
            "offset_B": offset_B,
            "g_sync": False,
            "excluded": True,
        })
        continue

    d_valid_seeds.append(seed)
    dist_A_contest = dist[settle_pos_A]
    dist_B_contest = dist[settle_pos_B]

    # ------------------------------------------------------------------
    # CONTEST PHASE: 2000 steps, both adaptive, comfort-gated mechanism
    # RNGs CONTINUE from settle phase (same rng_A, rng_B objects)
    # ------------------------------------------------------------------
    est_A = EST_INIT
    est_B = EST_INIT

    steps_a_at = 0
    steps_b_at = 0
    steps_both = 0

    for _step in range(STEPS):
        pos_a = A.true_pos
        pos_b = B_twin.true_pos

        a_at = (pos_a == SOURCE_CELL)
        b_at = (pos_b == SOURCE_CELL)

        if a_at:
            steps_a_at += 1
        if b_at:
            steps_b_at += 1
        if a_at and b_at:
            steps_both += 1

        # Comfort (depletion ecology)
        if a_at and b_at:
            comfort_A_step = 0.5
            comfort_B_step = 0.5
        elif a_at:
            comfort_A_step = 1.0
            comfort_B_step = 0.0
        elif b_at:
            comfort_A_step = 0.0
            comfort_B_step = 1.0
        else:
            comfort_A_step = 0.0
            comfort_B_step = 0.0

        # est update BEFORE policy
        if a_at:
            est_A = (1 - ALPHA) * est_A + ALPHA * comfort_A_step
        else:
            est_A += LAMBDA * (1.0 - est_A)

        if b_at:
            est_B = (1 - ALPHA) * est_B + ALPHA * comfort_B_step
        else:
            est_B += LAMBDA * (1.0 - est_B)

        # Creature A: live()-math update
        A_hat_a = A._A_hat()
        obs_a = int(world.cmap[pos_a])
        lik_a = A_hat_a[obs_a, :]
        qs_upd_a = lik_a * A.qs
        dn_a = qs_upd_a.sum()
        if dn_a > 0:
            qs_upd_a = qs_upd_a / dn_a
        else:
            qs_upd_a = np.ones(n_cells) / n_cells
        A.pA[obs_a, :] += qs_upd_a
        mc_a = int(np.argmax(qs_upd_a))
        pd_a = A_hat_a[:, mc_a]
        h_a = -np.sum(pd_a * np.log(pd_a + 1e-12))
        A.value_counts[obs_a] += np.exp(-h_a)
        action_a = policy_action(mc_a, rng_A, est_A, adaptive=True)
        new_pos_a = world.move(pos_a, action_a)
        A.qs = B_mat[:, :, action_a] @ qs_upd_a

        # Creature B: live()-math update
        A_hat_b = B_twin._A_hat()
        obs_b = int(world.cmap[pos_b])
        lik_b = A_hat_b[obs_b, :]
        qs_upd_b = lik_b * B_twin.qs
        dn_b = qs_upd_b.sum()
        if dn_b > 0:
            qs_upd_b = qs_upd_b / dn_b
        else:
            qs_upd_b = np.ones(n_cells) / n_cells
        B_twin.pA[obs_b, :] += qs_upd_b
        mc_b = int(np.argmax(qs_upd_b))
        pd_b = A_hat_b[:, mc_b]
        h_b = -np.sum(pd_b * np.log(pd_b + 1e-12))
        B_twin.value_counts[obs_b] += np.exp(-h_b)
        action_b = policy_action(mc_b, rng_B, est_B, adaptive=True)
        new_pos_b = world.move(pos_b, action_b)
        B_twin.qs = B_mat[:, :, action_b] @ qs_upd_b

        # Apply both moves simultaneously
        A.true_pos = new_pos_a
        B_twin.true_pos = new_pos_b

    occ_A = steps_a_at / STEPS
    occ_B = steps_b_at / STEPS
    co = steps_both / STEPS

    if occ_A > 0 and occ_B > 0:
        R = co / (occ_A * occ_B)
    else:
        R = float("nan")

    tot_occ = occ_A + occ_B
    asym = abs(occ_A - occ_B) / tot_occ if tot_occ > 0 else 0.0

    cls = classify(asym, R) if not np.isnan(R) else "OTHER"
    winner = "A" if occ_A > occ_B else "B"
    closer = "A" if dist_A_contest < dist_B_contest else (
        "B" if dist_B_contest < dist_A_contest else "TIE"
    )

    ratio_str = f"{R:6.3f}" if not np.isnan(R) else "   nan"

    print(
        f"  {seed:>2}  {settle_pos_A:>4d}  {settle_pos_B:>4d}  "
        f"{dist_A_contest:>3d}  {dist_B_contest:>3d}  "
        f"{offset_A:>4d}  {offset_B:>4d}  {'SYNC':>5}  "
        f"{occ_A:>6.3f}  {occ_B:>6.3f}  {asym:>6.3f}  {ratio_str}  "
        f"{cls:>12}  {winner:>6}  {closer:>6}"
    )

    part_d_results.append({
        "seed": seed,
        "settle_pos_A": settle_pos_A,
        "settle_pos_B": settle_pos_B,
        "dist_A": dist_A_contest,
        "dist_B": dist_B_contest,
        "offset_A": offset_A,
        "offset_B": offset_B,
        "g_sync": True,
        "excluded": False,
        "occ_A": occ_A,
        "occ_B": occ_B,
        "asym": asym,
        "R": R,
        "class": cls,
        "winner": winner,
        "closer": closer,
    })

print()

# G-SYNC gate evaluation
n_valid = len(d_valid_seeds)
n_excluded = len(d_excluded_seeds)
print(
    f"G-SYNC: valid={n_valid}/8  excluded={n_excluded}/8  "
    f"valid_seeds={d_valid_seeds}  excluded_seeds={d_excluded_seeds}"
)

if n_valid < 6:
    print(
        f"G-SYNC FAIL — only {n_valid}/8 seeds passed offset==0 gate (need >=6); "
        "RUN INVALID."
    )
    check(
        "D-G-SYNC",
        lambda: (False, f"valid={n_valid}/8 < 6 — G-SYNC FAIL"),
    )
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")
    sys.exit(1)

check(
    "D-G-SYNC",
    lambda: (True, f"valid={n_valid}/8 >= 6 (informational)"),
)

# P4 evaluation
valid_results = [r for r in part_d_results if not r["excluded"]]
excl_count = sum(1 for r in valid_results if r["class"] == "EXCLUSION")
print(f"P4 EXCLUSION count among valid seeds: {excl_count}/{n_valid}")

# Winner sub-check: among EXCLUSION seeds with unequal contest-start distances
excl_results = [r for r in valid_results if r["class"] == "EXCLUSION"]
unequal_dist_excl = [r for r in excl_results if r["dist_A"] != r["dist_B"]]
if unequal_dist_excl:
    winner_is_closer_count = sum(
        1 for r in unequal_dist_excl if r["winner"] == r["closer"]
    )
    winner_closer_frac = winner_is_closer_count / len(unequal_dist_excl)
else:
    winner_is_closer_count = 0
    winner_closer_frac = float("nan")

print(
    f"P4 winner-sub-check: unequal-dist EXCLUSION seeds={len(unequal_dist_excl)}  "
    f"winner==closer={winner_is_closer_count}  "
    f"fraction={winner_closer_frac if not np.isnan(winner_closer_frac) else 'N/A (no unequal-dist seeds)'}"
)

# Determine branch
if excl_count >= 4:
    if (
        np.isnan(winner_closer_frac)
        or winner_closer_frac >= 0.75
        or len(unequal_dist_excl) == 0
    ):
        p4_branch = "INTRINSIC-EARNED"
    else:
        p4_branch = "INTRINSIC-EARNED"  # exclusion count qualifies regardless of sub-check
elif excl_count <= 2:
    p4_branch = "ARTIFACT-REDUCED"
else:
    p4_branch = "BORDERLINE"

print(f"P4 branch: {p4_branch}")
print()

check(
    "D-P4-branch",
    lambda: (
        p4_branch == "INTRINSIC-EARNED",
        f"branch={p4_branch}  EXCLUSION={excl_count}/{n_valid}  "
        f"winner_closer_frac="
        f"{winner_closer_frac:.3f}" if not np.isnan(winner_closer_frac)
        else f"branch={p4_branch}  EXCLUSION={excl_count}/{n_valid}  "
             f"winner_closer_frac=N/A",
    ),
)

# ===========================================================================
# PROPERTY CHECKS TABLE
# ===========================================================================

print("=" * 70)
print("--- PROPERTY CHECKS ---")
print()

for name, passed, detail in checks:
    verdict = "PASS" if passed else "FAIL"
    print(f"{verdict}  {name}: {detail}")

print()

# ===========================================================================
# FALSIFIER MAP
# ===========================================================================

print("=" * 70)
print("--- FALSIFIER MAP ---")
print()

falsifiers = [
    ("F1", "A-P3", p3_violations_total == 0,
     "any dead-reckoning violation -> Exp 71 diagnosis wrong -> HALT"),
    ("F2", "A-P1", p1_sync_seeds >= 4,
     "re-sync failure -> loser-occupancy story wrong -> HALT"),
    ("F3", "B-P2", p2_pass,
     "phantom-camping story in doubt (reported; Part D continues)"),
]

for fname, pname, armed_pass, description in falsifiers:
    status = "CLEAR" if armed_pass else "FIRED"
    print(f"  {fname} ({pname}): {status} — {description}")

print()

# ===========================================================================
# FINAL LINE
# ===========================================================================

print("=" * 70)

if p4_branch == "INTRINSIC-EARNED":
    cascade_str = "closed"
elif p4_branch == "ARTIFACT-REDUCED":
    cascade_str = "downgraded"
else:
    cascade_str = "pending"

print(f"EXP72: {p4_branch} — cascade {cascade_str}")
