"""Exp 70 — rung 4 resolved out-of-sample: classifying the depletion-coupling regimes.

Exp 69 ran the comfort-gated mechanism but self-invalidated on a mis-specified gate
(output-conditioned; see the VALIDATION.md rule it produced). Its gate-invalid seeds 0-4
showed two apparent regimes — EXCLUSION (one creature monopolizes the source) and
ALTERNATION/TIMESHARING — but those observations are POST-HOC for that data: re-running
the same seeds with only the gate changed would reproduce identical trajectories, so any
"test" on them would be circular. Exp 70 therefore tests OUT-OF-SAMPLE: the regime
definitions below were fixed from Exp 69's data (disclosed), and are tested on EIGHT
FRESH seeds (5-12) never run before.

Mechanism (identical to Exp 69, declared): comfort-gated approach — EMA (alpha=0.1,
init 1.0) of comfort experienced at the source; below THRESH=0.75 wander instead of seek;
away-steps relax the estimate toward 1.0 (lambda=0.01); EPS=0.2 exploration; depletion
ecology (1.0 alone / 0.5 each crowded); coupling is resource-mediated (stigmergy) — no
dedicated other-agent sense (declared in Exp 69).

Per-seed classification (predeclared; asym = |occ_A - occ_B| / (occ_A + occ_B)):
  EXCLUSION   if asym > 0.5 and 0.9 <= R <= 1.1
  TIMESHARING if R < 0.9 and asym < 0.5
  NULL        if 0.9 <= R <= 1.1 and asym <= 0.5
  OTHER       otherwise
Gates (input-based per the VALIDATION.md rule from Exp 69):
  G1 fixed-arm null sanity: R_fixed in [0.75, 1.33] in >= 7/8 seeds.
  G3' mechanism input present: >= 20 crowded steps (both-at-source events) in every
      ADAPT run (counts events, not threshold states).
Predeclared verdict logic:
  RESULT requires >= 6/8 ADAPT seeds classifying into named (non-OTHER) categories AND a
  unique modal category among them. Modal NULL -> rung 4 NEGATIVE (indistinguishable from
  solipsists at this substrate; named next substrate: distal other-agent sensing). Modal
  EXCLUSION or TIMESHARING -> a real departure from independent behavior; reported as an
  ASYMMETRIC-EXCLUSION-LIKE or TIMESHARING-LIKE DEPARTURE respectively — the rung-5
  cascade (>= 3 reproductions + deflationary sweep) is REQUIRED before the words
  dominance or coordination are used as findings.
  P-MECH (first-arriver story, evaluated only if modal == EXCLUSION): the
  higher-occupancy creature is A (the closer starter, START_A=0 at BFS distance 2 vs
  START_B=24 at distance 6) in >= 80% of the EXCLUSION-classified seeds.
Falsifiers:
  F1 = >= 3/8 seeds OTHER -> the regimes are not captured by this classification;
     instrument iteration again, no rung-4 verdict.
  F2 = G1 fails -> null unstable, halt.
Predicted (stated before running, from Exp 69's seeds 0-4): modal EXCLUSION around 6/8
with 1-2 TIMESHARING; P-MECH holds (the dist-2 starter wins).
Provided priors declared: as Exp 69 (world, source rule, policy core, gate mechanism
parameters, depletion ecology, starts). Self-formed: maps/beliefs; each creature's
comfort-estimate trajectory. Spines never live, never saved; mirro forked once; vela
untouched.
"""
from __future__ import annotations

import copy
import sys

import numpy as np
from collections import deque
from pathlib import Path

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEPS = 2000
SEEDS = [5, 6, 7, 8, 9, 10, 11, 12]
EPS = 0.2
ALPHA = 0.1
THRESH = 0.75
LAMBDA = 0.01
EST_INIT = 1.0
START_A = 0
START_B = 24
MIRRO_DIR = Path("creature/state/mirro")

# ---------------------------------------------------------------------------
# Load committed spine (read-only — NEVER call .live() or .save() on mirro)
# ---------------------------------------------------------------------------

print("Exp 70 — rung 4 resolved out-of-sample: classifying the depletion-coupling regimes.")
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

dist = [-1] * n_cells
dist[SOURCE_CELL] = 0
q: deque[int] = deque([SOURCE_CELL])
while q:
    cell = q.popleft()
    for a in range(4):
        nb = world.move(cell, a)
        if dist[nb] == -1:
            dist[nb] = dist[cell] + 1
            q.append(nb)

# Print BFS distance field as 5x5 grid
print("BFS distance field to SOURCE_CELL (5x5 grid):")
for r in range(world.rows):
    row_str = "  "
    for c in range(world.cols):
        row_str += f"{dist[r * world.cols + c]:3d}"
    print(row_str)
print()

# ---------------------------------------------------------------------------
# Fork ONCE into twin template (one biography event, allowed)
# ---------------------------------------------------------------------------

twin_t = mirro.fork("exp70-twin-template")

# ---------------------------------------------------------------------------
# Policy function
# ---------------------------------------------------------------------------

def policy_action(map_cell: int, rng: np.random.Generator, est: float, adaptive: bool) -> int:
    """EPS-greedy BFS policy with optional comfort gate.

    Draw order:
      - Always draw u = rng.random() first (EPS check).
      - EPS branch: draw one rng.integers(0, 4) for random action.
      - Wander branch (adaptive and est < THRESH): draw one rng.integers(0, 4).
      - Greedy branch: draw one rng.integers(0, len(minimizers)) for tie-break.
    RNG-draw parity note: wander and greedy branches each draw exactly one integer
    after the uniform u draw, but from different ranges. Arms have their own rngs;
    draw-count divergence across arms is expected and intentional — do NOT try to
    match draw counts across ADAPT and FIXED.
    """
    u = rng.random()
    if u < EPS:
        return int(rng.integers(0, 4))
    # Adaptive gate: if est below threshold, wander
    if adaptive and est < THRESH:
        return int(rng.integers(0, 4))
    # Greedy: find actions minimizing dist[world.move(map_cell, a)]
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
# Per-run stepper
# ---------------------------------------------------------------------------

def run_pair(seed: int, adaptive: bool) -> dict:
    """Run one seed/arm pair; return per-run result dict.

    Both ADAPT and FIXED use rngs seeded at 400000+1000*seed+{0,1}. The same
    seed numbers feed both arms; divergence between arms arises purely from
    policy logic (wander vs greedy after the EPS draw) — no injected noise.
    """
    A = copy.deepcopy(twin_t)
    B_twin = copy.deepcopy(twin_t)
    A.true_pos = START_A
    B_twin.true_pos = START_B

    # Same rng seed formula for both arms — divergence comes from policy branch
    rng_A = np.random.default_rng(400000 + 1000 * seed + 0)
    rng_B = np.random.default_rng(400000 + 1000 * seed + 1)

    B_mat = world.transition_matrix()  # (n_cells, n_cells, 4)

    steps_a_at = 0
    steps_b_at = 0
    steps_both = 0

    comfort_A = 0.0
    comfort_B = 0.0

    est_A = EST_INIT
    est_B = EST_INIT

    min_est_A = EST_INIT
    min_est_B = EST_INIT

    # Gate closure counters: transitions from >=THRESH to <THRESH
    gate_closures_A = 0
    gate_closures_B = 0
    gate_steps_A = 0
    gate_steps_B = 0
    # Track whether we were above threshold at end of previous step
    prev_above_A = True  # EST_INIT=1.0 >= THRESH
    prev_above_B = True

    # For co-occupancy time series (20 buckets of 100 steps)
    bucket_size = STEPS // 20
    co_buckets = [0] * 20

    for step in range(STEPS):
        # Read start-of-step positions simultaneously
        pos_A = A.true_pos
        pos_B = B_twin.true_pos

        # Occupancy indicators
        a_at = (pos_A == SOURCE_CELL)
        b_at = (pos_B == SOURCE_CELL)

        if a_at:
            steps_a_at += 1
        if b_at:
            steps_b_at += 1
        if a_at and b_at:
            steps_both += 1
            co_buckets[step // bucket_size] += 1

        # Comfort this step
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

        comfort_A += comfort_A_step
        comfort_B += comfort_B_step

        # --- comfort_est update BEFORE policy (declared order) ---
        if a_at:
            est_A = (1 - ALPHA) * est_A + ALPHA * comfort_A_step
        else:
            est_A += LAMBDA * (1.0 - est_A)

        if b_at:
            est_B = (1 - ALPHA) * est_B + ALPHA * comfort_B_step
        else:
            est_B += LAMBDA * (1.0 - est_B)

        # Track minimum estimates
        if est_A < min_est_A:
            min_est_A = est_A
        if est_B < min_est_B:
            min_est_B = est_B

        # Track gate closures (>=THRESH -> <THRESH transitions)
        curr_above_A = (est_A >= THRESH)
        curr_above_B = (est_B >= THRESH)
        if prev_above_A and not curr_above_A:
            gate_closures_A += 1
        if prev_above_B and not curr_above_B:
            gate_closures_B += 1
        prev_above_A = curr_above_A
        prev_above_B = curr_above_B

        # Gate-closed step counters (diagnostic: fraction of steps spent below THRESH)
        if not curr_above_A:
            gate_steps_A += 1
        if not curr_above_B:
            gate_steps_B += 1

        # --- Creature A: natural live() update EXACTLY (as exp68) ---
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
        # Policy action from post-observation MAP cell
        action_A = policy_action(map_cell_A, rng_A, est_A, adaptive)
        new_pos_A = world.move(pos_A, action_A)
        A.qs = B_mat[:, :, action_A] @ qs_upd_A

        # --- Creature B: natural live() update EXACTLY (as exp68) ---
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
        # Policy action from post-observation MAP cell
        action_B = policy_action(map_cell_B, rng_B, est_B, adaptive)
        new_pos_B = world.move(pos_B, action_B)
        B_twin.qs = B_mat[:, :, action_B] @ qs_upd_B

        # Apply both moves simultaneously
        A.true_pos = new_pos_A
        B_twin.true_pos = new_pos_B

    # Per-run metrics
    occ_A = steps_a_at / STEPS
    occ_B = steps_b_at / STEPS
    co = steps_both / STEPS

    if occ_A > 0 and occ_B > 0:
        ratio = co / (occ_A * occ_B)
    else:
        ratio = float("nan")

    total_source_steps = steps_a_at + steps_b_at
    if total_source_steps > 0:
        comfort_per_source_step = (comfort_A + comfort_B) / total_source_steps
    else:
        comfort_per_source_step = float("nan")

    return {
        "seed": seed,
        "adaptive": adaptive,
        "occ_A": occ_A,
        "occ_B": occ_B,
        "co_occ": co,
        "ratio": ratio,
        "comfort_A": comfort_A,
        "comfort_B": comfort_B,
        "comfort_per_source_step": comfort_per_source_step,
        "min_est_A": min_est_A,
        "min_est_B": min_est_B,
        "gate_closures_A": gate_closures_A,
        "gate_closures_B": gate_closures_B,
        "gate_frac_A": gate_steps_A / STEPS,
        "gate_frac_B": gate_steps_B / STEPS,
        "steps_a_at": steps_a_at,
        "steps_b_at": steps_b_at,
        "steps_both": steps_both,
        "crowded_steps": steps_both,
        "co_buckets": co_buckets,
    }


# ---------------------------------------------------------------------------
# Per-seed regime classification
# ---------------------------------------------------------------------------

def classify(asym: float, R: float) -> str:
    """Classify a seed into a regime from predeclared thresholds.

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
# Main run loop
# ---------------------------------------------------------------------------

print("--- RUNS ---")
print(
    f"{'seed':>4}  {'arm':>5}  {'occ_A':>6}  {'occ_B':>6}  {'co_occ':>6}  "
    f"{'R':>6}  {'cpss':>6}  {'min_estA':>8}  {'min_estB':>8}  "
    f"{'closA':>5}  {'closB':>5}"
)
print("-" * 90)

# Results keyed by (seed, adaptive)
run_results: dict[tuple[int, bool], dict] = {}

for seed in SEEDS:
    for adaptive in [False, True]:
        r = run_pair(seed, adaptive)
        run_results[(seed, adaptive)] = r
        arm_label = "ADAPT" if adaptive else "FIXED"
        ratio_str = f"{r['ratio']:6.3f}" if not np.isnan(r['ratio']) else "   nan"
        cpss_str = f"{r['comfort_per_source_step']:6.3f}" if not np.isnan(r['comfort_per_source_step']) else "   nan"
        if adaptive:
            est_str = f"{r['min_est_A']:8.4f}  {r['min_est_B']:8.4f}  {r['gate_closures_A']:5d}  {r['gate_closures_B']:5d}"
        else:
            est_str = f"{'---':>8}  {'---':>8}  {'---':>5}  {'---':>5}"
        print(
            f"  {seed:>2}  {arm_label:>5}  {r['occ_A']:>6.3f}  {r['occ_B']:>6.3f}  "
            f"{r['co_occ']:>6.3f}  {ratio_str}  {cpss_str}  {est_str}"
        )

print()

# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

print("--- GATES ---")
print()

# G1: FIXED-arm sanity — R_fixed in [0.75, 1.33] in >=7/8 seeds
g1_pass_count = 0
g1_details = []
for seed in SEEDS:
    r = run_results[(seed, False)]
    R = r["ratio"]
    if np.isnan(R):
        ok = False
        g1_details.append(f"s{seed}:R=nan->FAIL")
    else:
        ok = 0.75 <= R <= 1.33
        if ok:
            g1_pass_count += 1
        g1_details.append(f"s{seed}:R={R:.3f}->{'OK' if ok else 'FAIL'}")

g1_passed = g1_pass_count >= 7
print(f"G1 FIXED-arm sanity: seeds_passing={g1_pass_count}/{len(SEEDS)}  [{'; '.join(g1_details)}]")
if not g1_passed:
    print("F2/G1 FAIL — null unstable; HALT")
    sys.exit(1)
print("G1: PASS")
print()

# G3': mechanism input present — in every ADAPT run, crowded_steps >= 20
print("G3' mechanism input present: >= 20 crowded steps in every ADAPT run")
g3p_details = []
g3p_all_ok = True
for seed in SEEDS:
    r = run_results[(seed, True)]
    cs = r["crowded_steps"]
    ok = cs >= 20
    if not ok:
        g3p_all_ok = False
    g3p_details.append(f"s{seed}:crowded={cs}->{'OK' if ok else 'FAIL'}")

print(f"G3' all_seeds_ok={'OK' if g3p_all_ok else 'FAIL'}  [{'; '.join(g3p_details)}]")
if not g3p_all_ok:
    print("G3' FAIL — mechanism input absent; RUN INVALID")
    sys.exit(1)
print("G3': PASS")
print()

# ---------------------------------------------------------------------------
# Per-seed ADAPT classification table
# ---------------------------------------------------------------------------

print("--- PER-SEED ADAPT CLASSIFICATION ---")
print()
print(
    f"{'seed':>4}  {'occ_A':>6}  {'occ_B':>6}  {'asym':>6}  {'R':>6}  "
    f"{'crowded':>7}  {'gate_frac_A':>11}  {'gate_frac_B':>11}  {'class':>12}"
)
print("-" * 90)

# Compute per-seed classification data
seed_classes: dict[int, str] = {}
seed_asym: dict[int, float] = {}

for seed in SEEDS:
    r = run_results[(seed, True)]
    occ_A = r["occ_A"]
    occ_B = r["occ_B"]
    R = r["ratio"]
    cs = r["crowded_steps"]
    gfA = r["gate_frac_A"]
    gfB = r["gate_frac_B"]

    tot_occ = occ_A + occ_B
    if tot_occ > 0:
        asym = abs(occ_A - occ_B) / tot_occ
    else:
        asym = 0.0

    if np.isnan(R):
        cls = "OTHER"
    else:
        cls = classify(asym, R)

    seed_classes[seed] = cls
    seed_asym[seed] = asym

    ratio_str = f"{R:6.3f}" if not np.isnan(R) else "   nan"
    print(
        f"  {seed:>2}  {occ_A:>6.3f}  {occ_B:>6.3f}  {asym:>6.3f}  {ratio_str}  "
        f"{cs:>7d}  {gfA:>11.3f}  {gfB:>11.3f}  {cls:>12}"
    )

print()

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------

# Regime tally
named_categories = ["EXCLUSION", "TIMESHARING", "NULL"]
class_list = [seed_classes[s] for s in SEEDS]
other_count = class_list.count("OTHER")
named_count = sum(1 for c in class_list if c in named_categories)

# Compute modal category among named
from collections import Counter
named_only = [c for c in class_list if c in named_categories]
if named_only:
    counts = Counter(named_only)
    max_count = max(counts.values())
    modal_candidates = [c for c, n in counts.items() if n == max_count]
    modal_unique = (len(modal_candidates) == 1)
    modal = modal_candidates[0] if modal_unique else None
else:
    modal_unique = False
    modal = None


def _c1():
    """C1-classifiable: >= 6/8 named AND unique modal."""
    n_named = named_count
    passes = n_named >= 6 and modal_unique and modal is not None
    detail = (
        f"named={n_named}/8  other={other_count}/8  "
        f"classes={class_list}  modal={'TIE' if not modal_unique else modal}  "
        f"modal_count={counts.get(modal, 0) if modal else 0}"
    )
    return passes, detail


def _c2_pmech():
    """C2-pmech: if modal==EXCLUSION, fraction of EXCLUSION seeds with occ_A > occ_B >= 0.8."""
    if modal != "EXCLUSION":
        return True, f"n/a — modal is {modal!r}, not EXCLUSION (passed=True)"

    excl_seeds = [s for s in SEEDS if seed_classes[s] == "EXCLUSION"]
    if not excl_seeds:
        return True, "n/a — no EXCLUSION seeds (passed=True)"

    winner_A_count = 0
    winner_details = []
    for s in excl_seeds:
        r = run_results[(s, True)]
        winner_is_A = r["occ_A"] > r["occ_B"]
        if winner_is_A:
            winner_A_count += 1
        winner_details.append(f"s{s}:winner={'A' if winner_is_A else 'B'}")

    frac = winner_A_count / len(excl_seeds)
    passes = frac >= 0.8
    detail = (
        f"EXCLUSION_seeds={excl_seeds}  A_wins={winner_A_count}/{len(excl_seeds)}={frac:.3f}  "
        f"threshold=0.8  [{'; '.join(winner_details)}]"
    )
    return passes, detail


check("C1-classifiable->=6/8-named-unique-modal", _c1)
check("C2-pmech-A-wins-EXCLUSION->=0.8", _c2_pmech)

# ---------------------------------------------------------------------------
# Print property check results
# ---------------------------------------------------------------------------

print("--- PROPERTY CHECKS ---")
print()

failed_names: list[str] = []
for name, passed, detail in checks:
    verdict = "PASS" if passed else "FAIL"
    print(f"{verdict}  {name}: {detail}")
    if not passed:
        failed_names.append(name)

print()

# ---------------------------------------------------------------------------
# Falsifier map
# ---------------------------------------------------------------------------

print("--- FALSIFIER MAP ---")
print()

c1_passed = checks[0][1]
c2_passed = checks[1][1]

# F1: >= 3/8 seeds OTHER
if other_count >= 3:
    print(
        f"F1 FIRED: regimes unclassifiable — instrument iteration, no rung-4 verdict "
        f"(OTHER count={other_count}/8 >= 3)"
    )
else:
    print(f"F1: not fired (OTHER count={other_count}/8 < 3)")

if c1_passed:
    if modal == "NULL":
        print(
            f"MODAL REGIME: NULL ({counts.get('NULL', 0)}/8) — "
            "rung 4 NEGATIVE (solipsist-indistinguishable); next substrate: distal sensing"
        )
    elif modal in ("EXCLUSION", "TIMESHARING"):
        label = "ASYMMETRIC-EXCLUSION-LIKE" if modal == "EXCLUSION" else "TIMESHARING-LIKE"
        print(
            f"MODAL REGIME: {modal} ({counts.get(modal, 0)}/8) — "
            f"DEPARTURE FROM INDEPENDENT BEHAVIOR — rung-5 cascade required before dominance/coordination language"
        )

print()

# ---------------------------------------------------------------------------
# Diagnostics: co-occupancy time series for first EXCLUSION and first TIMESHARING seed
# ---------------------------------------------------------------------------

print("--- DIAGNOSTICS ---")
print()

bucket_size = STEPS // 20

first_excl_seed = next((s for s in SEEDS if seed_classes[s] == "EXCLUSION"), None)
first_ts_seed = next((s for s in SEEDS if seed_classes[s] == "TIMESHARING"), None)

if first_excl_seed is not None:
    r_excl = run_results[(first_excl_seed, True)]
    excl_series = [f"{v / bucket_size:.2f}" for v in r_excl["co_buckets"]]
    print(f"Co-occupancy time series — first EXCLUSION seed={first_excl_seed}, ADAPT arm (20 buckets of 100 steps):")
    print(f"  {' '.join(excl_series)}")
    print()
else:
    print("Co-occupancy time series — no EXCLUSION seed found.")
    print()

if first_ts_seed is not None:
    r_ts = run_results[(first_ts_seed, True)]
    ts_series = [f"{v / bucket_size:.2f}" for v in r_ts["co_buckets"]]
    print(f"Co-occupancy time series — first TIMESHARING seed={first_ts_seed}, ADAPT arm (20 buckets of 100 steps):")
    print(f"  {' '.join(ts_series)}")
    print()
else:
    print("Co-occupancy time series — no TIMESHARING seed found.")
    print()

# Per-seed winner identity for EXCLUSION seeds
excl_seeds_all = [s for s in SEEDS if seed_classes[s] == "EXCLUSION"]
if excl_seeds_all:
    print("Winner identity for EXCLUSION seeds (higher-occupancy creature):")
    for s in excl_seeds_all:
        r = run_results[(s, True)]
        winner = "A" if r["occ_A"] > r["occ_B"] else "B"
        print(
            f"  seed={s}: occ_A={r['occ_A']:.3f}  occ_B={r['occ_B']:.3f}  winner={winner}"
        )
    print()

# Gate-closed fractions per ADAPT run
print("Gate-closed fractions per ADAPT run (fraction of steps each creature had est<THRESH):")
print(f"{'seed':>4}  {'frac_closed_A':>13}  {'frac_closed_B':>13}  {'closures_A':>10}  {'closures_B':>10}")
print("-" * 60)
for seed in SEEDS:
    r = run_results[(seed, True)]
    print(
        f"  {seed:>2}  {r['gate_frac_A']:>13.3f}  {r['gate_frac_B']:>13.3f}  "
        f"{r['gate_closures_A']:>10}  {r['gate_closures_B']:>10}"
        f"   (min_est A={r['min_est_A']:.3f} B={r['min_est_B']:.3f})"
    )
print()

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

if other_count >= 3:
    print("EXP70: FAIL [F1 — regimes unclassifiable, instrument iteration required]")
elif not c1_passed:
    fail_detail = "; ".join(failed_names)
    print(f"EXP70: FAIL [{fail_detail}]")
elif modal == "NULL":
    print("EXP70: NEGATIVE (modal NULL)")
else:
    print(f"EXP70: DEPARTURE ({modal}, cascade queued)")
