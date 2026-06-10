"""Exp 71 — rung-5 cascade for the exclusion departure: reproduce, then try to deflate it.

Exp 70 found (out-of-sample) that twins coupled only through a shared depletable source
depart from independence, modal regime an exclusion-like lock-in — with the first-arriver
confound provably load-bearing (closer starter won 5/5). Per the functional-emergence
cascade discipline, the deviation must survive reproduction and a deflationary sweep
before any dominance/coordination vocabulary is used. Three predeclared legs:

R3/D2 — instrument leg, runs FIRST (gate idleness without contest): each policy variant
  alone in its own world (no partner, crowding impossible). Predeclared: solo-ADAPT
  occupancy within +-0.10 of solo-FIXED occupancy for the same seed in >= 6/8 seeds.
  FAILURE = the comfort gate self-fires without contest -> instrument bug -> HALT, no
  verdict on anything.
R1 — equidistant reproduction (removes the first-arriver confound): both adaptive, starts
  at EQUAL BFS distance from the source (cells 0 and 20, both distance 2 — asserted
  in-script). 8 fresh seeds (13-20; never run in Exp 69/70). Classify each seed with
  Exp 70's predeclared table. Outcome branches (predeclared): EXCLUSION-INTRINSIC iff
  EXCLUSION classifications >= 4/8 (symmetry breaks stochastically); GEOMETRY-REDUCED iff
  EXCLUSION <= 2/8 with modal NULL or TIMESHARING; 3/8 = borderline, reported as such.
  Diagnostic (not a falsifier): among EXCLUSION seeds the winner identity should be
  roughly balanced (neither > 75%) — a persistent winner under equidistance would
  indicate residual asymmetry in the substrate (e.g. tie-break order), reported honestly.
R2/D1 — unilateral deflation (is mutual adaptation needed?): A FIXED (always-approach),
  B ADAPTIVE, original asymmetric starts (0 and 24). Predeclared: DEFLATION SUCCEEDS iff
  the exclusion signature (asym > 0.5 AND winner = A-the-fixed) appears in >= 6/8 seeds —
  i.e. ONE adapting agent suffices and the Exp 70 regime is unilateral-retreat lock-in,
  not mutual structure. DEFLATION FAILS iff <= 3/8 (mutual adaptation is load-bearing).

Predeclared WORDING RULES (the cascade's output is the honest name):
  - R1 EXCLUSION-INTRINSIC and R2 deflation succeeds -> "stigmergic unilateral-retreat
    lock-in, intrinsic to contest+retreat feedback" — NOT mutual social structure; the
    words dominance/coordination remain unearned.
  - R1 GEOMETRY-REDUCED and R2 deflation succeeds -> Exp 70's regime reduces fully to
    first-arriver advantage + unilateral retreat (maximal deflation).
  - R2 deflation FAILS and R1 EXCLUSION-INTRINSIC -> the candidate SURVIVES as mutual
    exclusion structure (strongest outcome; further cascade rounds then required).
  - Any other combination: reported per-leg, no aggregate name claimed.
Predicted (stated before running): R3 passes; R2 deflation succeeds; R1 uncertain,
leaning EXCLUSION-INTRINSIC.
Gates: G1 solo-FIXED sanity occupancy >= 0.20 in >= 7/8 seeds (instrument); G3' input
gate for the PAIRED legs: >= 20 crowded steps per run in >= 6/8 runs of each leg —
runs below 20 are excluded from that leg's classification (reported); a leg with < 6
valid runs is INVALID (no verdict for that leg).
Provided priors declared: everything Exp 69/70 declared. Self-formed: beliefs/maps and
comfort-estimate trajectories. Spines never live, never saved; mirro forked once
("exp71-twin-template"); vela untouched.
"""
from __future__ import annotations

import copy
import sys

import numpy as np
from collections import Counter, deque
from pathlib import Path

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# Constants (verbatim from exp70)
# ---------------------------------------------------------------------------

STEPS = 2000
EPS = 0.2
ALPHA = 0.1
THRESH = 0.75
LAMBDA = 0.01
EST_INIT = 1.0
MIRRO_DIR = Path("creature/state/mirro")

# Seeds for each leg
SEEDS = [13, 14, 15, 16, 17, 18, 19, 20]   # fresh; never run in Exp 69/70

# Start positions
EQ_START_A = 0
EQ_START_B = 20
ASYM_START_A = 0
ASYM_START_B = 24
SOLO_START = 0

# rng_base values (avoid collisions with prior exps; exp70 used 400000)
RNG_BASE_SOLO = 500000
RNG_BASE_R1   = 600000
RNG_BASE_R2   = 700000

# ---------------------------------------------------------------------------
# Load committed spine (read-only — NEVER call .live() or .save() on mirro)
# ---------------------------------------------------------------------------

print("Exp 71 — rung-5 cascade for the exclusion departure: reproduce, then try to deflate it.")
print()

mirro = Creature.load(MIRRO_DIR)

vc_m = mirro.value_counts
tot_m = vc_m.sum()

print(
    f"mirro: name={mirro.name!r} age={mirro.age_steps} "
    f"hash={mirro._state_hash()[:12]} favorite={mirro.favorite()} "
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
# BFS distance field to SOURCE_CELL over the grid
# ---------------------------------------------------------------------------

world = mirro.world
n_cells = world.n_cells

dist: list[int] = [-1] * n_cells
dist[SOURCE_CELL] = 0
q: deque[int] = deque([SOURCE_CELL])
while q:
    cell = q.popleft()
    for a in range(4):
        nb = world.move(cell, a)
        if dist[nb] == -1:
            dist[nb] = dist[cell] + 1
            q.append(nb)

# Print BFS distance field as grid
print("BFS distance field to SOURCE_CELL (grid):")
for r in range(world.rows):
    row_str = "  "
    for c in range(world.cols):
        row_str += f"{dist[r * world.cols + c]:3d}"
    print(row_str)
print()

# Assert equidistant starts
d_eq_A = dist[EQ_START_A]
d_eq_B = dist[EQ_START_B]
print(
    f"Equidistant-start assertion: dist[EQ_START_A={EQ_START_A}]={d_eq_A}  "
    f"dist[EQ_START_B={EQ_START_B}]={d_eq_B}"
)
if d_eq_A != d_eq_B:
    print(
        f"ASSERT FAIL: EQ_START_A and EQ_START_B are NOT equidistant "
        f"({d_eq_A} != {d_eq_B}); HALT"
    )
    sys.exit(1)
print(f"  -> PASS (both distance {d_eq_A})")
print()

# ---------------------------------------------------------------------------
# Fork ONCE into twin template (one biography event, allowed)
# ---------------------------------------------------------------------------

twin_t = mirro.fork("exp71-twin-template")

# ---------------------------------------------------------------------------
# Policy function (verbatim from exp70)
# ---------------------------------------------------------------------------

def policy_action(map_cell: int, rng: np.random.Generator, est: float, adaptive: bool) -> int:
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


# ---------------------------------------------------------------------------
# Regime classification (verbatim from exp70)
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
# Stepper: paired run (generalised — per-creature adaptive flags)
# ---------------------------------------------------------------------------

def run_pair(
    seed: int,
    adaptive_A: bool,
    adaptive_B: bool,
    start_A: int,
    start_B: int,
    rng_base: int,
) -> dict:
    """Run one paired seed; return per-run result dict.

    rng_A = default_rng(rng_base + 1000*seed + 0)
    rng_B = default_rng(rng_base + 1000*seed + 1)
    Each creature has its own adaptive flag and est.
    """
    A = copy.deepcopy(twin_t)
    B_twin = copy.deepcopy(twin_t)
    A.true_pos = start_A
    B_twin.true_pos = start_B

    rng_A = np.random.default_rng(rng_base + 1000 * seed + 0)
    rng_B = np.random.default_rng(rng_base + 1000 * seed + 1)

    B_mat = world.transition_matrix()

    steps_a_at = 0
    steps_b_at = 0
    steps_both = 0

    comfort_A_total = 0.0
    comfort_B_total = 0.0

    est_A = EST_INIT
    est_B = EST_INIT

    gate_steps_A = 0
    gate_steps_B = 0
    prev_above_A = True
    prev_above_B = True

    for _step in range(STEPS):
        pos_A = A.true_pos
        pos_B = B_twin.true_pos

        a_at = (pos_A == SOURCE_CELL)
        b_at = (pos_B == SOURCE_CELL)

        if a_at:
            steps_a_at += 1
        if b_at:
            steps_b_at += 1
        if a_at and b_at:
            steps_both += 1

        # Comfort this step (depletion ecology: 1.0 alone / 0.5 each crowded)
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

        comfort_A_total += comfort_A_step
        comfort_B_total += comfort_B_step

        # comfort_est update BEFORE policy (declared order)
        if a_at:
            est_A = (1 - ALPHA) * est_A + ALPHA * comfort_A_step
        else:
            est_A += LAMBDA * (1.0 - est_A)

        if b_at:
            est_B = (1 - ALPHA) * est_B + ALPHA * comfort_B_step
        else:
            est_B += LAMBDA * (1.0 - est_B)

        curr_above_A = (est_A >= THRESH)
        curr_above_B = (est_B >= THRESH)
        prev_above_A = curr_above_A
        prev_above_B = curr_above_B

        if not curr_above_A:
            gate_steps_A += 1
        if not curr_above_B:
            gate_steps_B += 1

        # Creature A: live()-math update
        A_hat_A = A._A_hat()
        obs_A = int(world.cmap[pos_A])
        likelihood_A = A_hat_A[obs_A, :]
        qs_upd_A = likelihood_A * A.qs
        denom_A = qs_upd_A.sum()
        if denom_A > 0:
            qs_upd_A = qs_upd_A / denom_A
        else:
            qs_upd_A = np.ones(n_cells) / n_cells
        A.pA[obs_A, :] += qs_upd_A
        map_cell_A = int(np.argmax(qs_upd_A))
        pred_dist_A = A_hat_A[:, map_cell_A]
        h_A = -np.sum(pred_dist_A * np.log(pred_dist_A + 1e-12))
        w_A = np.exp(-h_A)
        A.value_counts[obs_A] += w_A
        action_A = policy_action(map_cell_A, rng_A, est_A, adaptive_A)
        new_pos_A = world.move(pos_A, action_A)
        A.qs = B_mat[:, :, action_A] @ qs_upd_A

        # Creature B: live()-math update
        A_hat_B = B_twin._A_hat()
        obs_B = int(world.cmap[pos_B])
        likelihood_B = A_hat_B[obs_B, :]
        qs_upd_B = likelihood_B * B_twin.qs
        denom_B = qs_upd_B.sum()
        if denom_B > 0:
            qs_upd_B = qs_upd_B / denom_B
        else:
            qs_upd_B = np.ones(n_cells) / n_cells
        B_twin.pA[obs_B, :] += qs_upd_B
        map_cell_B = int(np.argmax(qs_upd_B))
        pred_dist_B = A_hat_B[:, map_cell_B]
        h_B = -np.sum(pred_dist_B * np.log(pred_dist_B + 1e-12))
        w_B = np.exp(-h_B)
        B_twin.value_counts[obs_B] += w_B
        action_B = policy_action(map_cell_B, rng_B, est_B, adaptive_B)
        new_pos_B = world.move(pos_B, action_B)
        B_twin.qs = B_mat[:, :, action_B] @ qs_upd_B

        # Apply both moves simultaneously
        A.true_pos = new_pos_A
        B_twin.true_pos = new_pos_B

    occ_A = steps_a_at / STEPS
    occ_B = steps_b_at / STEPS
    co = steps_both / STEPS

    if occ_A > 0 and occ_B > 0:
        ratio = co / (occ_A * occ_B)
    else:
        ratio = float("nan")

    total_source_steps = steps_a_at + steps_b_at
    if total_source_steps > 0:
        comfort_per_source_step = (comfort_A_total + comfort_B_total) / total_source_steps
    else:
        comfort_per_source_step = float("nan")

    winner = "A" if occ_A > occ_B else "B"

    tot_occ = occ_A + occ_B
    asym = abs(occ_A - occ_B) / tot_occ if tot_occ > 0 else 0.0

    return {
        "seed": seed,
        "occ_A": occ_A,
        "occ_B": occ_B,
        "co_occ": co,
        "ratio": ratio,
        "asym": asym,
        "winner": winner,
        "comfort_per_source_step": comfort_per_source_step,
        "gate_frac_A": gate_steps_A / STEPS,
        "gate_frac_B": gate_steps_B / STEPS,
        "crowded_steps": steps_both,
    }

# ---------------------------------------------------------------------------
# Stepper: solo run (single creature, no partner)
# ---------------------------------------------------------------------------

def run_solo(seed: int, adaptive: bool, rng_base: int) -> dict:
    """Run one creature alone; return occupancy and gate_frac.

    Solo comfort at source is always 1.0 (alone by construction).
    est updates: at-source EMA toward 1.0, away recovery toward 1.0 (lambda).
    So est should stay ~1 and the gate should stay open (no crowding possible).
    """
    C = copy.deepcopy(twin_t)
    C.true_pos = SOLO_START

    rng = np.random.default_rng(rng_base + 1000 * seed + 0)

    B_mat = world.transition_matrix()

    steps_at = 0
    est = EST_INIT
    gate_steps = 0

    for _step in range(STEPS):
        pos = C.true_pos
        at = (pos == SOURCE_CELL)

        if at:
            steps_at += 1

        # Solo comfort: always 1.0 when at source (alone by construction)
        comfort_step = 1.0 if at else 0.0

        # est update
        if at:
            est = (1 - ALPHA) * est + ALPHA * comfort_step
        else:
            est += LAMBDA * (1.0 - est)

        if est < THRESH:
            gate_steps += 1

        # live()-math update
        A_hat = C._A_hat()
        obs = int(world.cmap[pos])
        likelihood = A_hat[obs, :]
        qs_upd = likelihood * C.qs
        denom = qs_upd.sum()
        if denom > 0:
            qs_upd = qs_upd / denom
        else:
            qs_upd = np.ones(n_cells) / n_cells
        C.pA[obs, :] += qs_upd
        map_cell = int(np.argmax(qs_upd))
        pred_dist = A_hat[:, map_cell]
        h = -np.sum(pred_dist * np.log(pred_dist + 1e-12))
        w = np.exp(-h)
        C.value_counts[obs] += w
        action = policy_action(map_cell, rng, est, adaptive)
        new_pos = world.move(pos, action)
        C.qs = B_mat[:, :, action] @ qs_upd

        C.true_pos = new_pos

    occupancy = steps_at / STEPS
    gate_frac = gate_steps / STEPS

    return {
        "seed": seed,
        "adaptive": adaptive,
        "occupancy": occupancy,
        "gate_frac": gate_frac,
    }

# ===========================================================================
# LEG R3/D2 — INSTRUMENT LEG (solo runs, FIRST)
# ===========================================================================

print("=" * 70)
print("LEG R3/D2 — INSTRUMENT: solo runs (no partner)")
print("=" * 70)
print()
print(
    f"{'seed':>4}  {'arm':>5}  {'occ':>6}  {'gate_frac':>9}"
)
print("-" * 35)

solo_results: dict[tuple[int, bool], dict] = {}

for seed in SEEDS:
    for adaptive in [False, True]:
        r = run_solo(seed, adaptive, RNG_BASE_SOLO)
        solo_results[(seed, adaptive)] = r
        arm_label = "ADAPT" if adaptive else "FIXED"
        print(
            f"  {seed:>2}  {arm_label:>5}  {r['occupancy']:>6.3f}  {r['gate_frac']:>9.4f}"
        )

print()

# G1: solo-FIXED sanity — occupancy >= 0.20 in >= 7/8 seeds
g1_pass_count = 0
g1_details = []
for seed in SEEDS:
    r = solo_results[(seed, False)]
    ok = r["occupancy"] >= 0.20
    if ok:
        g1_pass_count += 1
    g1_details.append(f"s{seed}:occ={r['occupancy']:.3f}->{'OK' if ok else 'FAIL'}")

g1_passed = g1_pass_count >= 7
print(
    f"G1 solo-FIXED sanity (occ>=0.20 in >=7/8): "
    f"seeds_passing={g1_pass_count}/{len(SEEDS)}  [{'; '.join(g1_details)}]"
)
print(f"G1: {'PASS' if g1_passed else 'FAIL'}")
print()

# D2: solo-ADAPT occ within +-0.10 of solo-FIXED occ for same seed, >= 6/8
d2_within_count = 0
d2_details = []
for seed in SEEDS:
    r_fixed = solo_results[(seed, False)]
    r_adapt = solo_results[(seed, True)]
    diff = abs(r_adapt["occupancy"] - r_fixed["occupancy"])
    ok = diff <= 0.10
    if ok:
        d2_within_count += 1
    d2_details.append(
        f"s{seed}:fixed={r_fixed['occupancy']:.3f},adapt={r_adapt['occupancy']:.3f},"
        f"diff={diff:.3f}->{'OK' if ok else 'FAIL'}"
    )

d2_passed = d2_within_count >= 6
print(
    f"D2 solo-ADAPT vs solo-FIXED occupancy within +-0.10 in >=6/8: "
    f"seeds_passing={d2_within_count}/{len(SEEDS)}"
)
for detail in d2_details:
    print(f"  {detail}")
print(f"D2: {'PASS' if d2_passed else 'FAIL'}")
print()

if not g1_passed or not d2_passed:
    failing = []
    if not g1_passed:
        failing.append("G1")
    if not d2_passed:
        failing.append("D2")
    print(
        f"R3/D2 FAIL — gate self-fires (or instrument broken); "
        f"HALT, no verdict ({', '.join(failing)} failed)"
    )
    sys.exit(1)

print("R3/D2: PASS — gate does not self-fire without contest.")
print()

# Record instrument checks
check("R3-D2-instrument", lambda: (d2_passed, f"within_count={d2_within_count}/8 >= 6"))
check("G1-solo-fixed-sanity", lambda: (g1_passed, f"seeds_passing={g1_pass_count}/8 >= 7"))

# ===========================================================================
# LEG R1 — EQUIDISTANT REPRODUCTION (seeds 13-20, both adaptive, eq starts)
# ===========================================================================

print("=" * 70)
print("LEG R1 — EQUIDISTANT REPRODUCTION (both adaptive, eq starts)")
print(f"  EQ_START_A={EQ_START_A} (dist={dist[EQ_START_A]})  "
      f"EQ_START_B={EQ_START_B} (dist={dist[EQ_START_B]})")
print("=" * 70)
print()
print(
    f"{'seed':>4}  {'occ_A':>6}  {'occ_B':>6}  {'asym':>6}  {'R':>6}  "
    f"{'crowded':>7}  {'gfrac_A':>7}  {'gfrac_B':>7}  {'winner':>6}  {'class':>12}"
)
print("-" * 95)

r1_results: dict[int, dict] = {}

for seed in SEEDS:
    r = run_pair(seed, True, True, EQ_START_A, EQ_START_B, RNG_BASE_R1)
    r1_results[seed] = r
    asym = r["asym"]
    R = r["ratio"]
    if np.isnan(R):
        cls = "OTHER"
        ratio_str = "   nan"
    else:
        cls = classify(asym, R)
        ratio_str = f"{R:6.3f}"
    r["class"] = cls
    print(
        f"  {seed:>2}  {r['occ_A']:>6.3f}  {r['occ_B']:>6.3f}  {asym:>6.3f}  "
        f"{ratio_str}  {r['crowded_steps']:>7d}  {r['gate_frac_A']:>7.3f}  "
        f"{r['gate_frac_B']:>7.3f}  {r['winner']:>6}  {cls:>12}"
    )

print()

# G3' for R1: exclude runs with < 20 crowded steps
r1_valid_seeds = [s for s in SEEDS if r1_results[s]["crowded_steps"] >= 20]
r1_excluded = [s for s in SEEDS if r1_results[s]["crowded_steps"] < 20]
r1_valid_count = len(r1_valid_seeds)

print(
    f"G3' R1: valid runs (crowded>=20): {r1_valid_count}/8  "
    f"valid={r1_valid_seeds}  excluded={r1_excluded}"
)

r1_leg_valid = r1_valid_count >= 6
if not r1_leg_valid:
    print(f"R1 LEG INVALID: only {r1_valid_count}/8 valid runs (need >= 6); no verdict for R1.")
    r1_branch = "INVALID"
else:
    # Classify valid runs
    r1_classes = [r1_results[s]["class"] for s in r1_valid_seeds]
    r1_excl_count = r1_classes.count("EXCLUSION")
    r1_null_count = r1_classes.count("NULL")
    r1_ts_count = r1_classes.count("TIMESHARING")

    # Modal among valid
    named_r1 = [c for c in r1_classes if c in ("EXCLUSION", "TIMESHARING", "NULL")]
    if named_r1:
        r1_modal_counter = Counter(named_r1)
        r1_max = max(r1_modal_counter.values())
        r1_modal_cands = [c for c, n in r1_modal_counter.items() if n == r1_max]
        r1_modal = r1_modal_cands[0] if len(r1_modal_cands) == 1 else None
    else:
        r1_modal = None

    print(
        f"R1 valid classes: {r1_classes}  EXCLUSION={r1_excl_count}  "
        f"NULL={r1_null_count}  TIMESHARING={r1_ts_count}  modal={r1_modal}"
    )

    if r1_excl_count >= 4:
        r1_branch = "EXCLUSION-INTRINSIC"
    elif r1_excl_count <= 2 and r1_modal in ("NULL", "TIMESHARING"):
        r1_branch = "GEOMETRY-REDUCED"
    elif r1_excl_count == 3:
        r1_branch = "BORDERLINE"
    else:
        r1_branch = "BORDERLINE"

    print(f"R1 branch: {r1_branch}")

print()

# Winner-balance diagnostic among R1 EXCLUSION seeds
r1_excl_seeds = [s for s in r1_valid_seeds if r1_results[s]["class"] == "EXCLUSION"]
if r1_excl_seeds:
    print("R1 DIAGNOSTIC — winner identity among EXCLUSION seeds (equidistant starts):")
    r1_winner_A = sum(1 for s in r1_excl_seeds if r1_results[s]["winner"] == "A")
    r1_winner_B = len(r1_excl_seeds) - r1_winner_A
    for s in r1_excl_seeds:
        r = r1_results[s]
        print(
            f"  seed={s}: occ_A={r['occ_A']:.3f}  occ_B={r['occ_B']:.3f}  "
            f"winner={r['winner']}"
        )
    frac_A = r1_winner_A / len(r1_excl_seeds) if r1_excl_seeds else float("nan")
    print(
        f"  A wins: {r1_winner_A}/{len(r1_excl_seeds)} ({frac_A:.2f})  "
        f"B wins: {r1_winner_B}/{len(r1_excl_seeds)}"
    )
    if frac_A > 0.75:
        print(
            "  DIAGNOSTIC NOTE: A wins > 75% under equidistance — residual substrate "
            "asymmetry (e.g. tie-break order) may be present; reported honestly."
        )
    elif (1.0 - frac_A) > 0.75:
        print(
            "  DIAGNOSTIC NOTE: B wins > 75% under equidistance — residual substrate "
            "asymmetry (e.g. tie-break order) may be present; reported honestly."
        )
    else:
        print("  Winner identity roughly balanced (neither > 75%) — no residual asymmetry signal.")
    print()
else:
    print("R1 DIAGNOSTIC — no EXCLUSION seeds in valid set.")
    print()

check(
    "R1-branch",
    lambda: (
        r1_leg_valid,
        f"branch={r1_branch}  valid_runs={r1_valid_count}/8  "
        f"EXCLUSION={r1_classes.count('EXCLUSION') if r1_leg_valid else 'N/A'}"
        if r1_leg_valid
        else f"branch=INVALID  valid_runs={r1_valid_count}/8"
    ),
)

# ===========================================================================
# LEG R2/D1 — UNILATERAL DEFLATION (A fixed, B adaptive, asym starts)
# ===========================================================================

print("=" * 70)
print("LEG R2/D1 — UNILATERAL DEFLATION (A fixed, B adaptive, asym starts)")
print(f"  ASYM_START_A={ASYM_START_A} (dist={dist[ASYM_START_A]})  "
      f"ASYM_START_B={ASYM_START_B} (dist={dist[ASYM_START_B]})")
print("=" * 70)
print()
print(
    f"{'seed':>4}  {'occ_A':>6}  {'occ_B':>6}  {'asym':>6}  {'R':>6}  "
    f"{'crowded':>7}  {'gfrac_B':>7}  {'winner':>6}  {'sig?':>5}"
)
print("-" * 80)

r2_results: dict[int, dict] = {}

for seed in SEEDS:
    r = run_pair(seed, False, True, ASYM_START_A, ASYM_START_B, RNG_BASE_R2)
    r2_results[seed] = r
    asym = r["asym"]
    R = r["ratio"]
    ratio_str = f"{R:6.3f}" if not np.isnan(R) else "   nan"
    # Exclusion signature: asym > 0.5 AND winner == 'A' (the fixed creature)
    sig = (asym > 0.5 and r["winner"] == "A")
    r["deflation_sig"] = sig
    print(
        f"  {seed:>2}  {r['occ_A']:>6.3f}  {r['occ_B']:>6.3f}  {asym:>6.3f}  "
        f"{ratio_str}  {r['crowded_steps']:>7d}  {r['gate_frac_B']:>7.3f}  "
        f"{r['winner']:>6}  {'YES' if sig else 'no':>5}"
    )

print()

# G3' for R2: exclude runs with < 20 crowded steps
r2_valid_seeds = [s for s in SEEDS if r2_results[s]["crowded_steps"] >= 20]
r2_excluded = [s for s in SEEDS if r2_results[s]["crowded_steps"] < 20]
r2_valid_count = len(r2_valid_seeds)

print(
    f"G3' R2: valid runs (crowded>=20): {r2_valid_count}/8  "
    f"valid={r2_valid_seeds}  excluded={r2_excluded}"
)

r2_leg_valid = r2_valid_count >= 6
if not r2_leg_valid:
    print(f"R2 LEG INVALID: only {r2_valid_count}/8 valid runs (need >= 6); no verdict for R2.")
    r2_branch = "INVALID"
    r2_deflation_count = 0
else:
    r2_deflation_count = sum(
        1 for s in r2_valid_seeds if r2_results[s]["deflation_sig"]
    )
    print(
        f"R2 deflation signature count (asym>0.5 AND winner=A): "
        f"{r2_deflation_count}/{r2_valid_count}"
    )
    if r2_deflation_count >= 6:
        r2_branch = "DEFLATION-SUCCEEDS"
    elif r2_deflation_count <= 3:
        r2_branch = "DEFLATION-FAILS"
    else:
        r2_branch = "INDETERMINATE"

    print(f"R2 branch: {r2_branch}")

print()

# R2 diagnostic: B's gate-closed fraction per seed
print("R2 DIAGNOSTIC — B's gate-closed fraction per seed (adaptive B):")
print(f"{'seed':>4}  {'gfrac_B':>9}  {'crowded':>7}  {'valid':>5}")
print("-" * 35)
for seed in SEEDS:
    r = r2_results[seed]
    is_valid = r["crowded_steps"] >= 20
    print(
        f"  {seed:>2}  {r['gate_frac_B']:>9.4f}  {r['crowded_steps']:>7d}  "
        f"{'YES' if is_valid else 'no':>5}"
    )
print()

check(
    "R2-branch",
    lambda: (
        r2_leg_valid,
        f"branch={r2_branch}  deflation_sig={r2_deflation_count}/{r2_valid_count}  "
        f"valid_runs={r2_valid_count}/8"
        if r2_leg_valid
        else f"branch=INVALID  valid_runs={r2_valid_count}/8"
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
# VERDICT SECTION — apply WORDING RULES
# ===========================================================================

print("=" * 70)
print("--- VERDICT ---")
print()

print(f"R3/D2 (instrument): PASS")
print(f"G1 (solo-FIXED sanity): {'PASS' if g1_passed else 'FAIL'}")
print(f"R1 branch: {r1_branch}  (valid_runs={r1_valid_count}/8)")
print(f"R2 branch: {r2_branch}  (valid_runs={r2_valid_count}/8)")
print()

# Apply predeclared wording rules
aggregate_name: str | None = None

if r1_branch == "EXCLUSION-INTRINSIC" and r2_branch == "DEFLATION-SUCCEEDS":
    aggregate_name = (
        "stigmergic unilateral-retreat lock-in, intrinsic to contest+retreat feedback"
    )
    print(
        "WORDING RULE 1 fires: R1=EXCLUSION-INTRINSIC + R2=DEFLATION-SUCCEEDS"
    )
    print(
        f"Aggregate name: \"{aggregate_name}\""
    )
    print(
        "NOTE: NOT mutual social structure; words dominance/coordination remain unearned."
    )
elif r1_branch == "GEOMETRY-REDUCED" and r2_branch == "DEFLATION-SUCCEEDS":
    aggregate_name = (
        "Exp 70 regime reduces fully to first-arriver advantage + unilateral retreat "
        "(maximal deflation)"
    )
    print(
        "WORDING RULE 2 fires: R1=GEOMETRY-REDUCED + R2=DEFLATION-SUCCEEDS"
    )
    print(f"Aggregate name: \"{aggregate_name}\"")
elif r2_branch == "DEFLATION-FAILS" and r1_branch == "EXCLUSION-INTRINSIC":
    aggregate_name = (
        "mutual exclusion structure (candidate SURVIVES; further cascade rounds required)"
    )
    print(
        "WORDING RULE 3 fires: R2=DEFLATION-FAILS + R1=EXCLUSION-INTRINSIC"
    )
    print(f"Aggregate name: \"{aggregate_name}\"")
    print(
        "NOTE: Strongest outcome — mutual adaptation is load-bearing."
    )
else:
    print(
        "No single wording rule fires — reporting per-leg, no aggregate name claimed."
    )
    print(
        f"  R1: {r1_branch}  |  R2: {r2_branch}"
    )

print()

if aggregate_name is not None:
    final_summary = f"CASCADE COMPLETE — {aggregate_name}"
else:
    final_summary = (
        f"CASCADE COMPLETE — per-leg: R1={r1_branch}, R2={r2_branch}; "
        f"no aggregate name"
    )

print(f"EXP71: {final_summary}")
