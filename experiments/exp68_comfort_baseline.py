"""Exp 68 — rung 4 part 1: the comfort-source instrument and the mutually-blind null.

Social-emergence direction, rung 4 (loop/directions/social-emergence.md) asks whether two
clade-mates valuing ONE comfort source develop measurable coordination versus two creatures
who cannot sense each other. That comparison needs a validated instrument first (Exp 61's
lesson): a value-driven seeking policy, a depletable source, occupancy/co-occupancy metrics,
and the INDEPENDENCE NULL measured on a mutually-blind pair. This experiment builds and
validates exactly that — part 2 (Exp 69+) adds the inter-agent sensing/learning coupling
and tests for coordination AGAINST this baseline. No coordination claim is made here.

Setup: two twin forks of mirro (same self-formed favorite, color 2 — so one source both
value, by construction) in mirro's world; the comfort source is the FIRST cell whose color
is mirro's favorite (declared rule). Provided policy (declared, the Exp 62 VI pattern):
each step, with prob EPS=0.2 a uniform random action, else the action stepping the
creature's MAP-estimate cell along a shortest path toward the source (BFS distance field on
the innate B; ties broken uniformly with the creature's policy rng). Navigation runs from
BELIEF (argmax qs), not ground truth. Depletion ecology (declared): a creature at the
source gains comfort 1.0 per step if alone there, 0.5 if both are there. The pair is BLIND:
no other-agent modality, no channel — their processes interact only through the (measured)
depletion ledger, never through behavior. The natural live() learning math (belief update,
pA soft counts, value accumulation) continues untouched; the comfort ledger is
measurement-only and feeds nothing back in part 1. Starts: opposite corners (0 and 24).

Predeclared predictions:
  P1 (seeking works, navigation from belief): each creature's at-source occupancy
     fraction >= 0.20 (= 5x the 1/25 uniform-random baseline) in >=4/5 seeds.
  P2 (the blind null): independence ratio R = P(both at source) / (P(A at source) *
     P(B at source)) in [0.75, 1.33] in >=4/5 seeds — mutually-blind seekers must show
     no spurious coordination/avoidance (autocorrelated occupancy puts this genuinely
     at stake).
  ID1 (instrument identity, all runs): each creature's realized comfort ==
     steps-alone-at-source * 1.0 + steps-both-at-source * 0.5, exactly.
Falsifiers:
  F1 = P1 fails -> the policy/instrument cannot seek from beliefs; substrate broken;
     fix before any rung-4 part 2.
  F2 = P2 fails -> the independence null is violated for BLIND agents; part 2's planned
     baseline comparison is invalid as designed; investigate before climbing.
Predicted (stated before running): occupancy ~0.35-0.55 (the policy has no stay action,
so at-source dynamics bounce); R ~ 1.0 +- 0.15; ID1 exact.
Provided priors declared: the policy form and EPS, the source-cell rule, the depletion
ecology, start positions, the world. Self-formed: the creatures' maps/beliefs (which the
navigation consumes) and their value ledgers (untouched mechanics). The committed spines
never live and are never saved; mirro is forked once into the twin template; vela untouched.
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
START_A = 0
START_B = 24
MIRRO_DIR = Path("creature/state/mirro")

# ---------------------------------------------------------------------------
# Load committed spine (read-only — NEVER call .live() or .save() on mirro)
# ---------------------------------------------------------------------------

print("Exp 68 — rung 4 part 1: the comfort-source instrument and the mutually-blind null.")
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

twin_t = mirro.fork("exp68-twin-template")

# ---------------------------------------------------------------------------
# Policy function
# ---------------------------------------------------------------------------

def policy_action(map_cell: int, rng: np.random.Generator) -> int:
    """EPS-greedy BFS policy: with prob EPS random, else BFS-greedy toward SOURCE_CELL.

    Ties broken uniformly via rng. Uses dist[] precomputed above.
    EPS branch: draw u first; if u < EPS random, else greedy.
    """
    u = rng.random()
    if u < EPS:
        return int(rng.integers(0, 4))
    # greedy: find actions minimizing dist[world.move(map_cell, a)]
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

def run_seed(seed: int) -> dict:
    """Run one seed; return per-run result dict."""
    A = copy.deepcopy(twin_t)
    B_twin = copy.deepcopy(twin_t)
    A.true_pos = START_A
    B_twin.true_pos = START_B

    rng_A = np.random.default_rng(300000 + 1000 * seed + 0)
    rng_B = np.random.default_rng(300000 + 1000 * seed + 1)

    B_mat = world.transition_matrix()  # (n_cells, n_cells, 4)

    steps_a_at = 0
    steps_b_at = 0
    steps_both = 0

    comfort_A = 0.0
    comfort_B = 0.0

    # For ID1 verification
    alone_A = 0   # steps A alone at source
    alone_B = 0   # steps B alone at source
    both_steps = 0  # steps both at source simultaneously

    for _ in range(STEPS):
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

        # Comfort ledger (measurement-only, feeds nothing back)
        if a_at and b_at:
            comfort_A += 0.5
            comfort_B += 0.5
            both_steps += 1
        elif a_at:
            comfort_A += 1.0
            alone_A += 1
        elif b_at:
            comfort_B += 1.0
            alone_B += 1

        # --- Creature A: natural live() update EXACTLY ---
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
        action_A = policy_action(map_cell_A, rng_A)
        new_pos_A = world.move(pos_A, action_A)
        A.qs = B_mat[:, :, action_A] @ qs_upd_A

        # --- Creature B: natural live() update EXACTLY ---
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
        action_B = policy_action(map_cell_B, rng_B)
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

    # ID1: exact comfort ledger identity check (within 1e-9)
    id1_A = abs(comfort_A - (alone_A * 1.0 + both_steps * 0.5)) < 1e-9
    id1_B = abs(comfort_B - (alone_B * 1.0 + both_steps * 0.5)) < 1e-9
    id1_ok = id1_A and id1_B

    # End map accuracies (diagnostic)
    map_acc_A = A.map_accuracy()
    map_acc_B = B_twin.map_accuracy()

    return {
        "seed": seed,
        "occ_A": occ_A,
        "occ_B": occ_B,
        "co_occ": co,
        "ratio": ratio,
        "comfort_A": comfort_A,
        "comfort_B": comfort_B,
        "alone_A": alone_A,
        "alone_B": alone_B,
        "both_steps": both_steps,
        "id1_ok": id1_ok,
        "map_acc_A": map_acc_A,
        "map_acc_B": map_acc_B,
    }


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

print("--- RUNS ---")
print(
    f"{'seed':>4}  {'occ_A':>6}  {'occ_B':>6}  {'co_occ':>6}  "
    f"{'ratio':>6}  {'comf_A':>7}  {'comf_B':>7}  {'id1_ok':>6}"
)
print("-" * 70)

run_results: dict[int, dict] = {}

for seed in SEEDS:
    r = run_seed(seed)
    run_results[seed] = r
    ratio_str = f"{r['ratio']:6.3f}" if not np.isnan(r['ratio']) else "   nan"
    print(
        f"  {r['seed']:>2}  {r['occ_A']:>6.3f}  {r['occ_B']:>6.3f}  "
        f"{r['co_occ']:>6.3f}  {ratio_str}  "
        f"{r['comfort_A']:>7.1f}  {r['comfort_B']:>7.1f}  "
        f"{'Y' if r['id1_ok'] else 'N':>6}"
    )

print()

# ---------------------------------------------------------------------------
# Diagnostics: per-seed end map accuracies
# ---------------------------------------------------------------------------

print("--- DIAGNOSTICS: END MAP ACCURACIES ---")
print(f"{'seed':>4}  {'map_acc_A':>9}  {'map_acc_B':>9}")
print("-" * 30)
for seed in SEEDS:
    r = run_results[seed]
    print(f"  {seed:>2}  {r['map_acc_A']:>9.3f}  {r['map_acc_B']:>9.3f}")
print()

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------

def _p1():
    # occ >= 0.20 for BOTH creatures; count seeds passing; need >=4/5
    n_pass = 0
    details = []
    for seed in SEEDS:
        r = run_results[seed]
        ok = (r["occ_A"] >= 0.20) and (r["occ_B"] >= 0.20)
        if ok:
            n_pass += 1
        details.append(
            f"s{seed}:A={r['occ_A']:.3f},B={r['occ_B']:.3f}->{'OK' if ok else 'FAIL'}"
        )
    passes = n_pass >= 4
    detail = (
        f"seeds_passing={n_pass}/{len(SEEDS)}  threshold>=4  "
        f"[{'; '.join(details)}]"
    )
    return passes, detail


def _p2():
    # 0.75 <= R <= 1.33 in >=4/5 seeds
    n_pass = 0
    details = []
    for seed in SEEDS:
        r = run_results[seed]
        R = r["ratio"]
        if np.isnan(R):
            ok = False
            details.append(f"s{seed}:R=nan->FAIL")
        else:
            ok = 0.75 <= R <= 1.33
            if ok:
                n_pass += 1
            details.append(f"s{seed}:R={R:.3f}->{'OK' if ok else 'FAIL'}")
    passes = n_pass >= 4
    detail = (
        f"seeds_passing={n_pass}/{len(SEEDS)}  threshold>=4  "
        f"[{'; '.join(details)}]"
    )
    return passes, detail


def _id1():
    # exact in all 10 ledgers (5 seeds x 2 creatures)
    all_ok = all(run_results[seed]["id1_ok"] for seed in SEEDS)
    details = [
        f"s{seed}:{'OK' if run_results[seed]['id1_ok'] else 'FAIL'}"
        for seed in SEEDS
    ]
    detail = f"all_10_ledgers={'OK' if all_ok else 'FAIL'}  [{'; '.join(details)}]"
    return all_ok, detail


check("P1-seeking-works-occ>=0.20-both->=4/5-seeds", _p1)
check("P2-blind-null-R-in-[0.75,1.33]->=4/5-seeds", _p2)
check("ID1-comfort-ledger-exact-all-10", _id1)

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
id1_passed = checks[2][1]

if not p1_passed:
    print(
        "F1 FIRED: seeking-from-belief broken — fix substrate before rung-4 part 2"
    )

if not p2_passed:
    print(
        "F2 FIRED: blind-pair null violated — part 2 baseline invalid as designed"
    )

if p1_passed and p2_passed and id1_passed:
    print(
        "INSTRUMENT VALIDATED: part 2 (sensing+learning coupling) may proceed "
        "against this baseline."
    )

if p1_passed and p2_passed and id1_passed:
    print("  No falsifiers fired.")

print()

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

if p1_passed and p2_passed and id1_passed:
    print("EXP68: PASS")
else:
    fail_tags = [name.split("-")[0] for name, passed, _ in checks if not passed]
    print(f"EXP68: FAIL {fail_tags}")
