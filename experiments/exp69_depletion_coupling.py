"""Exp 69 — rung 4 part 2: does experienced depletion bend the pair away from independence?

Social-emergence direction, rung 4: Exp 68 validated the instrument and measured the
mutually-blind FIXED-policy null at R = 0.996-1.016. This experiment adds ONE new
mechanism — comfort-gated approach with optimistic recovery — and asks whether the pair's
co-occupancy departs from the independence null in a consistent direction.

Honesty note on the coupling channel (declared): a dedicated other-agent-here sense adds
nothing at this substrate, because it fires only on co-location and at-source co-location
IS the depletion event the creature already experiences as halved comfort. The coupling
tested here is therefore RESOURCE-MEDIATED (stigmergy-like): each creature senses the
other only through the comfort it actually receives. Distal other-agent sensing is a
different substrate; if rung 4 is NEGATIVE here, that is the named next substrate.

The mechanism (provided FORM, declared; learned CONTENT): each creature keeps
comfort_est, an EMA of the comfort received on at-source steps (alpha=0.1, init 1.0,
optimistic). Policy per step: with prob EPS=0.2 random; else if comfort_est >= THRESH=0.75
the BFS-greedy approach of Exp 68; else a uniform random action (wander). On steps NOT at
the source, comfort_est relaxes toward 1.0: est += LAMBDA*(1.0-est), LAMBDA=0.01
(optimistic recovery — without it a single bad stretch causes permanent avoidance and the
gate never re-opens). Pre-run dynamics sketch (stated): crowding halves comfort, the EMA
crosses the gate after ~7 crowded steps, recovery takes ~50 away-steps; whether the twins
settle into avoidance/timesharing (R < 1), synchronized oscillation (R > 1), or wash out
(R ~ 1) is genuinely uncertain — the direction is NOT predicted.

Arms (same seeds, paired): ADAPT (both creatures comfort-gated) and FIXED (Exp 68's
always-approach policy re-run with this script's rngs as the paired control); Exp 68's
committed null (R = 0.996-1.016) is the reference for the FIXED arm's sanity.

Gates (instrument validity): G1 FIXED-arm sanity: R_fixed in [0.75, 1.33] in >=4/5 seeds
(replicates Exp 68's null under fresh rngs). G3 mechanism engaged: in every ADAPT run,
EACH creature's comfort_est dips below THRESH at least once (else the gate never fired —
instrument idle, run INVALID, redesign).
Predeclared predictions:
  P1 (departure, two-sided, direction-consistent): |R_adapt - 1| > 0.10 in >=4/5 seeds,
     AND the deviation has the SAME SIGN in all seeds where it exceeds 0.10.
  P2 (functional payoff — evaluated ONLY if P1 passes with R < 1): comfort per
     at-source step is higher in ADAPT than FIXED for the same seed in >=4/5 seeds
     (timesharing should raise per-capita comfort at the source). If P1 passes with
     R > 1, P2 is reported as n/a and the synchronized-oscillation reading is logged.
Falsifiers:
  F1 = P1 fails -> rung 4 NEGATIVE at this substrate: resource-mediated coupling does not
     produce measurable departure from independence; logged as a real negative with the
     named next substrate (distal other-agent sensing).
  F2 = G1 fails -> the null itself is unstable; halt and investigate before any verdict.
CASCADE DISCIPLINE (binding): if P1 passes, the entry reports a DEPARTURE FROM
INDEPENDENCE and queues the functional-emergence rung-5 cascade (>=3 fork reproductions +
deflationary sweep) — the word coordination is NOT used for this result until the cascade
completes.
Provided priors declared: everything Exp 68 declared (world, source rule, BFS policy core,
EPS, depletion ecology, starts) plus the gate mechanism (alpha, THRESH, LAMBDA, init).
Self-formed: maps/beliefs consumed by navigation; each creature's comfort_est trajectory
(its own experienced depletion). The committed spines never live and are never saved;
mirro is forked once into the twin template; vela untouched.
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
SEEDS = [0, 1, 2, 3, 4]
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

print("Exp 69 — rung 4 part 2: does experienced depletion bend the pair away from independence?")
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

twin_t = mirro.fork("exp69-twin-template")

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
        "co_buckets": co_buckets,
    }


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

# G1: FIXED-arm sanity — R_fixed in [0.75, 1.33] in >=4/5 seeds
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

g1_passed = g1_pass_count >= 4
print(f"G1 FIXED-arm sanity: seeds_passing={g1_pass_count}/{len(SEEDS)}  [{'; '.join(g1_details)}]")
if not g1_passed:
    print("F2/G1 FAIL — null unstable; HALT")
    sys.exit(1)
print("G1: PASS")
print()

# G3: mechanism engaged — in every ADAPT run, EACH creature's min_est < THRESH
g3_details = []
g3_all_ok = True
for seed in SEEDS:
    r = run_results[(seed, True)]
    ok_A = r["min_est_A"] < THRESH
    ok_B = r["min_est_B"] < THRESH
    ok = ok_A and ok_B
    if not ok:
        g3_all_ok = False
    g3_details.append(
        f"s{seed}:minA={r['min_est_A']:.4f}({'OK' if ok_A else 'FAIL'})"
        f",minB={r['min_est_B']:.4f}({'OK' if ok_B else 'FAIL'})"
    )

print(f"G3 mechanism engaged: all_seeds_both_creatures={'OK' if g3_all_ok else 'FAIL'}  [{'; '.join(g3_details)}]")
if not g3_all_ok:
    print("G3 FAIL — gate never fired; RUN INVALID")
    sys.exit(1)
print("G3: PASS")
print()

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------

def _p1():
    """P1: |R_adapt - 1| > 0.10 in >=4/5 seeds, same sign in all exceeding seeds."""
    devs = []
    details = []
    for seed in SEEDS:
        r = run_results[(seed, True)]
        R = r["ratio"]
        if np.isnan(R):
            devs.append(float("nan"))
            details.append(f"s{seed}:R=nan->FAIL")
        else:
            dev = R - 1.0
            devs.append(dev)
            exceed = abs(dev) > 0.10
            details.append(f"s{seed}:dev={dev:+.3f}->{'exceed' if exceed else 'within'}")

    # Filter non-nan deviations that exceed threshold
    exceed_devs = [d for d in devs if not np.isnan(d) and abs(d) > 0.10]
    n_exceed = len(exceed_devs)
    n_pass_threshold = n_exceed >= 4

    # Check same sign among exceeding
    if n_exceed > 0:
        signs = set(1 if d > 0 else -1 for d in exceed_devs)
        same_sign = len(signs) == 1
        common_sign = (1 if list(signs)[0] > 0 else -1) if same_sign else 0
    else:
        same_sign = False
        common_sign = 0

    passes = n_pass_threshold and same_sign
    detail = (
        f"n_exceeding={n_exceed}/{len(SEEDS)}  same_sign={'YES' if same_sign else 'NO'}  "
        f"common_sign={'+' if common_sign > 0 else ('-' if common_sign < 0 else '?')}  "
        f"devs=[{'; '.join(details)}]"
    )
    # Attach sign to check record via closure
    _p1.common_sign = common_sign
    return passes, detail

_p1.common_sign = 0


def _p2():
    """P2: comfort_per_source_step(ADAPT) > (FIXED) in >=4/5 seeds — only if P1 passed with R<1."""
    p1_result = next((r for name, r, _ in checks if name.startswith("P1")), None)
    p1_passed_val = p1_result if p1_result is not None else False

    if not p1_passed_val:
        return True, "n/a — P1 failed"

    sign = _p1.common_sign
    if sign > 0:
        # P1 passed with R > 1: synchronized-oscillation reading
        return True, "n/a — R>1 synchronized-oscillation reading (P1 passed, R>1)"

    # P1 passed with R < 1: check functional payoff
    n_pass = 0
    details = []
    for seed in SEEDS:
        r_adapt = run_results[(seed, True)]
        r_fixed = run_results[(seed, False)]
        cpss_a = r_adapt["comfort_per_source_step"]
        cpss_f = r_fixed["comfort_per_source_step"]
        if np.isnan(cpss_a) or np.isnan(cpss_f):
            details.append(f"s{seed}:nan->FAIL")
        else:
            ok = cpss_a > cpss_f
            if ok:
                n_pass += 1
            details.append(f"s{seed}:adapt={cpss_a:.3f},fixed={cpss_f:.3f}->{'OK' if ok else 'FAIL'}")
    passes = n_pass >= 4
    detail = (
        f"seeds_passing={n_pass}/{len(SEEDS)}  threshold>=4  "
        f"[{'; '.join(details)}]"
    )
    return passes, detail


check("P1-departure-|R-1|>0.10->=4/5-seeds-same-sign", _p1)
check("P2-comfort-per-source-step-adapt>fixed->=4/5", _p2)

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

p1_passed = checks[0][1]
p2_passed = checks[1][1]
sign = _p1.common_sign

if not p1_passed:
    print(
        "F1 FIRED: rung 4 NEGATIVE at this substrate — resource-mediated coupling does not "
        "bend the pair off independence; next substrate would be distal other-agent sensing"
    )
else:
    sign_label = "positive" if sign > 0 else "negative"
    print(
        f"DEPARTURE FROM INDEPENDENCE (sign={sign_label}) — "
        "rung-5 cascade REQUIRED before any coordination claim"
    )

print()

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

if p1_passed:
    print("EXP69: PASS (departure, cascade queued)")
else:
    print("EXP69: NEGATIVE (no departure)")

print()

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

print("--- DIAGNOSTICS ---")
print()

# Per-seed ADAPT arm: fraction of steps each creature spent gate-closed (est < THRESH)
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

# 20-bucket (100-step) time series of co-occupancy fraction for seed 0: ADAPT vs FIXED
print("Co-occupancy fraction time series — seed 0, 20 buckets of 100 steps each:")
r_adapt_s0 = run_results[(0, True)]
r_fixed_s0 = run_results[(0, False)]

bucket_size = STEPS // 20
adapt_series = [f"{v / bucket_size:.2f}" for v in r_adapt_s0["co_buckets"]]
fixed_series = [f"{v / bucket_size:.2f}" for v in r_fixed_s0["co_buckets"]]

print(f"ADAPT: {' '.join(adapt_series)}")
print(f"FIXED: {' '.join(fixed_series)}")
print()
