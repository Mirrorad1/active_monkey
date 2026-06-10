"""Exp 85 — graded-uncertainty rung 1: count decay does no harm in a static world.

Opening experiment of loop/directions/graded-uncertainty.md (the card-sanctioned
direction on the six-times-motivated forgetting substrate; the M4a build remains
reserved for the Exp 74 CONSULT's human answer). Mechanism M-A (provided, declared):
per-step count decay pA *= LAMBDA with LAMBDA=0.997 (Exp 60's in-window value) and a
0.01 floor, implemented in the experiment stepper — the creature class is untouched.

Before the mechanism may be tested as a healer (scars, rung 2) it must first be shown
costless where there is nothing to heal: in a STATIC world, a decaying-counts creature
must match a non-decaying twin's competence. The equilibrium arithmetic: each step adds
total mass 1.0 to pA, so decayed total mass converges to ~1/(1-LAMBDA) = 333 — a sharp
sanity check on the implementation.

Predeclared (8 seeds and effect-size bands per the Exp 79 VALIDATION rule):
  P1 (map parity + absolute competence): |map_acc_lambda - map_acc_control| <= 0.08 for
     the same birth seed in >= 6/8 seeds, AND mean(map_acc_lambda) >= 0.85.
  P2 (localization parity): end localize_bits <= 0.1 in BOTH arms in >= 6/8 seeds.
  P3 (equilibrium mass): lambda-arm total pA mass at end in [250, 420] (333 +- 25
     percent) in >= 6/8 seeds.
Falsifiers:
  F1 = P1 fails -> LAMBDA=0.997 costs competence even in the stationary case; the rung-1
     gate fails and a calibration experiment (lambda sweep) must precede everything else
     on this ladder.
  F2 = P3 fails -> the equilibrium arithmetic (or the implementation) is wrong; fix
     before any verdict.
Provided priors declared: the mechanism (LAMBDA, floor), the world (mirro's committed
cmap layout, used as a static layout for fresh births — the spine itself is untouched
and not loaded beyond reading its world), raising length, start cell, birth seeds
600-607. Fresh separate-root newborns only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEPS = 3000
SEEDS = list(range(8))        # offsets 0..7; actual birth seed = BIRTH_BASE + offset
BIRTH_BASE = 600
START = 12
LAMBDA = 0.997
FLOOR = 0.01

# ---------------------------------------------------------------------------
# World: read mirro's committed cmap layout, fresh births only — spine untouched
# ---------------------------------------------------------------------------

_manifest = json.loads(
    Path("creature/state/mirro/manifest.json").read_text()
)
world = World.from_dict(_manifest["world"])

print("Exp 85 — graded-uncertainty rung 1: count decay does no harm in a static world.")
print()
print("World (mirro's committed cmap layout, used as static layout for fresh newborns):")
print(f"  rows={world.rows}  cols={world.cols}  n_colors={world.n_colors}")
print("  cmap grid:")
for r in range(world.rows):
    row_vals = world.cmap[r * world.cols : (r + 1) * world.cols]
    print("   ", " ".join(str(v) for v in row_vals))
print()
print(f"  STEPS={STEPS}  LAMBDA={LAMBDA}  FLOOR={FLOOR}  START={START}")
print(f"  Birth seeds: {[BIRTH_BASE + s for s in SEEDS]}")
print()

# ---------------------------------------------------------------------------
# Precompute B once (shared, world is static)
# ---------------------------------------------------------------------------

B = world.transition_matrix()   # (n_cells, n_cells, 4)

# ---------------------------------------------------------------------------
# Per-seed measurement storage
# ---------------------------------------------------------------------------

results = []  # list of dicts

for seed_offset in SEEDS:
    birth_seed = BIRTH_BASE + seed_offset

    # -----------------------------------------------------------------------
    # CONTROL arm: standard live() path
    # -----------------------------------------------------------------------
    ctrl = Creature.birth(f"exp85-ctrl-s{seed_offset}", world, seed=birth_seed)
    ctrl.true_pos = START
    ctrl.live(STEPS)

    acc_ctrl = ctrl.map_accuracy()
    loc_ctrl = ctrl.localize_bits()
    mass_ctrl = float(ctrl.pA.sum())

    # -----------------------------------------------------------------------
    # LAMBDA arm: custom stepper replicating live() EXACTLY plus one decay line
    #
    # Ordering declared: decay-then-observe (pA decayed before A_hat is computed
    # each step, so the normalization in _A_hat always sees the post-decay counts).
    #
    # RNG derivation is identical to live(): combined_seed = (c._seed * 1_000_003 +
    # c.rng_counter) & 0xFFFFFFFFFFFFFFFF, then rng.integers(0, 4) once per step.
    # The decay operation does NOT consume rng draws, so with the same seed derivation
    # the action sequences of the two arms are IDENTICAL for the same birth seed —
    # matched trajectories. This is intentional: any competence difference is caused
    # solely by the pA decay, not by different random walks.
    # -----------------------------------------------------------------------
    lam = Creature.birth(f"exp85-lam-s{seed_offset}", world, seed=birth_seed)
    lam.true_pos = START

    # Derive RNG identically to live() (rng_counter=0 at birth, same as ctrl)
    combined_seed = (lam._seed * 1_000_003 + lam.rng_counter) & 0xFFFFFFFFFFFFFFFF
    rng = np.random.default_rng(combined_seed)

    for _ in range(STEPS):
        # --- DECAY FIRST (declared ordering: decay-then-observe) ---
        lam.pA *= LAMBDA
        np.maximum(lam.pA, FLOOR, out=lam.pA)

        # --- A_hat from (decayed) pA ---
        A = lam.pA.copy()
        col_sums = A.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        A_hat = A / col_sums

        # --- observe ---
        obs = int(world.cmap[lam.true_pos])

        # --- belief update: qs ∝ likelihood(obs) * prior(qs) ---
        likelihood = A_hat[obs, :]
        qs_upd = likelihood * lam.qs
        denom = qs_upd.sum()
        if denom > 0:
            qs_upd = qs_upd / denom
        else:
            qs_upd = np.ones(world.n_cells) / world.n_cells

        # --- Dirichlet count learning: pA[obs, :] += qs_upd ---
        lam.pA[obs, :] += qs_upd

        # --- Value accumulation (Exp 26 mechanism, matching live()) ---
        map_cell = int(np.argmax(qs_upd))
        predicted_obs_dist = A_hat[:, map_cell]
        h_predicted = -np.sum(
            predicted_obs_dist * np.log(predicted_obs_dist + 1e-12)
        )
        predictability_weight = np.exp(-h_predicted)
        lam.value_counts[obs] += predictability_weight

        # --- choose action, move ---
        action = int(rng.integers(0, 4))
        lam.true_pos = world.move(lam.true_pos, action)

        # --- advance belief through movement model ---
        lam.qs = B[:, :, action] @ qs_upd

    # Advance bookkeeping to match live()
    lam.age_steps += STEPS
    lam.rng_counter += 1

    acc_lam = lam.map_accuracy()
    loc_lam = lam.localize_bits()
    mass_lam = float(lam.pA.sum())

    results.append({
        "seed_offset": seed_offset,
        "birth_seed": birth_seed,
        "acc_ctrl": acc_ctrl,
        "acc_lam": acc_lam,
        "diff": abs(acc_lam - acc_ctrl),
        "loc_ctrl": loc_ctrl,
        "loc_lam": loc_lam,
        "mass_ctrl": mass_ctrl,
        "mass_lam": mass_lam,
    })

# ---------------------------------------------------------------------------
# Print per-seed table
# ---------------------------------------------------------------------------

print("--- PER-SEED TABLE ---")
header = (
    f"  {'seed':>4}  {'acc_ctrl':>9}  {'acc_lam':>8}  {'|diff|':>7}  "
    f"{'loc_ctrl':>9}  {'loc_lam':>8}  {'mass_ctrl':>10}  {'mass_lam':>10}"
)
print(header)
print("  " + "-" * (len(header) - 2))
for r in results:
    print(
        f"  {r['birth_seed']:>4}  {r['acc_ctrl']:>9.4f}  {r['acc_lam']:>8.4f}  "
        f"{r['diff']:>7.4f}  {r['loc_ctrl']:>9.4f}  {r['loc_lam']:>8.4f}  "
        f"{r['mass_ctrl']:>10.2f}  {r['mass_lam']:>10.2f}"
    )
print()

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------

EQ_MASS_LO = 250.0
EQ_MASS_HI = 420.0
P1_DIFF_THRESH = 0.08
P1_ACC_MIN = 0.85
P2_LOC_MAX = 0.1
PASS_THRESHOLD = 6  # >= 6/8 seeds

def check(label: str, value: bool) -> bool:
    print(f"  {'PASS' if value else 'FAIL'}  {label}")
    return value

print("--- PROPERTY CHECKS ---")

# P1: map parity + absolute competence
p1_parity_per_seed = [r["diff"] <= P1_DIFF_THRESH for r in results]
p1_parity_count = sum(p1_parity_per_seed)
mean_acc_lam = float(np.mean([r["acc_lam"] for r in results]))
P1_parity_ok = p1_parity_count >= PASS_THRESHOLD
P1_abs_ok = mean_acc_lam >= P1_ACC_MIN
P1_ok = P1_parity_ok and P1_abs_ok
check(
    f"P1a (parity): |diff|<={P1_DIFF_THRESH} in {p1_parity_count}/8 seeds "
    f"(need {PASS_THRESHOLD}): {'PASS' if P1_parity_ok else 'FAIL'}",
    P1_parity_ok,
)
check(
    f"P1b (abs competence): mean(acc_lam)={mean_acc_lam:.4f} >= {P1_ACC_MIN}: "
    f"{'PASS' if P1_abs_ok else 'FAIL'}",
    P1_abs_ok,
)
check(f"P1 (both): {'PASS' if P1_ok else 'FAIL'}", P1_ok)

print()

# P2: localization parity in BOTH arms
p2_per_seed = [
    r["loc_ctrl"] <= P2_LOC_MAX and r["loc_lam"] <= P2_LOC_MAX
    for r in results
]
p2_count = sum(p2_per_seed)
P2_ok = p2_count >= PASS_THRESHOLD
check(
    f"P2 (localize_bits <= {P2_LOC_MAX} in both arms): {p2_count}/8 seeds "
    f"(need {PASS_THRESHOLD}): {'PASS' if P2_ok else 'FAIL'}",
    P2_ok,
)

print()

# P3: equilibrium mass in lambda arm
p3_per_seed = [EQ_MASS_LO <= r["mass_lam"] <= EQ_MASS_HI for r in results]
p3_count = sum(p3_per_seed)
P3_ok = p3_count >= PASS_THRESHOLD
eq_theory = 1.0 / (1.0 - LAMBDA)
check(
    f"P3 (mass_lam in [{EQ_MASS_LO}, {EQ_MASS_HI}]; theory={eq_theory:.0f}): "
    f"{p3_count}/8 seeds (need {PASS_THRESHOLD}): {'PASS' if P3_ok else 'FAIL'}",
    P3_ok,
)

print()

# ---------------------------------------------------------------------------
# Falsifier map
# ---------------------------------------------------------------------------

print("--- FALSIFIER MAP ---")

F1_fired = not P1_ok
F2_fired = not P3_ok

if not F1_fired:
    print(
        f"  F1 did not fire (P1 PASS — lambda costs no competence in the stationary case; "
        f"parity in {p1_parity_count}/8, mean_acc_lam={mean_acc_lam:.4f})."
    )
else:
    print(
        f"  F1 FIRED: P1 failed "
        f"(parity {p1_parity_count}/8, mean_acc_lam={mean_acc_lam:.4f}; "
        f"abs_ok={P1_abs_ok}) "
        f"-> LAMBDA={LAMBDA} costs competence even in the stationary case; "
        f"rung-1 gate fails; calibration experiment (lambda sweep) must precede "
        f"everything else on this ladder."
    )

if not F2_fired:
    print(
        f"  F2 did not fire (P3 PASS — equilibrium mass in [{EQ_MASS_LO}, {EQ_MASS_HI}] "
        f"in {p3_count}/8 seeds; theory={eq_theory:.0f})."
    )
else:
    print(
        f"  F2 FIRED: P3 failed ({p3_count}/8 seeds in [{EQ_MASS_LO}, {EQ_MASS_HI}]; "
        f"theory={eq_theory:.0f}) "
        f"-> equilibrium arithmetic or implementation is wrong; fix before any verdict."
    )

print()

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

if not F1_fired and not F2_fired:
    print("EXP85: NO-HARM PASS — rung 2 (scar healing) unlocked")
elif F1_fired and not F2_fired:
    print("EXP85: F1 — lambda costs competence; calibration rung required")
elif not F1_fired and F2_fired:
    print("EXP85: F2 — equilibrium arithmetic wrong")
else:
    print("EXP85: F1 + F2 — lambda costs competence AND equilibrium arithmetic wrong")
