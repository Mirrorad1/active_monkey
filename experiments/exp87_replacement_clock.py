"""Exp 87 — the replacement clock, out-of-sample: the corrected scar-healing rate law.

Exp 86's F2 halt: the mechanism healed scars to ~zero but my predeclared half-life used
the foreign-MASS decay clock (231 steps) on a RATIO metric whose denominator also decays;
the corrected arithmetic says the scar fraction S falls on the REPLACEMENT timescale
(fresh accrual overtaking decayed old mass). Per the fresh-seeds rule, the corrected
numbers — read off Exp 86's trajectories — must pass on seeds never run: this experiment.

Protocol: Exp 86's implant verbatim (1500 steps world A = mirro's layout, 1500 steps
world B = colors permuted, no decay, S0 ~ 0.49), then the 3000-step lambda healing phase
(0.997 / floor 0.01, decay stepper with live()-identical rng bookkeeping in 250-step
chunks), S sampled every 250. No control arm (Exp 86 settled the dilution law 8/8; all
claims here are lambda-arm claims). Fresh seeds 800-807.

Predeclared (the corrected rate law, stated in Exp 86's entry before this run):
  P1 (early ratio): S(250) in [0.40, 0.50] in >= 6/8 seeds.
  P2 (half-crossing on the replacement clock): the first sample with S <= S0/2 occurs at
     t in [750, 1500] in >= 6/8 seeds.
  P3 (end-state replication): S(3000) <= 0.05 in >= 6/8 seeds.
Falsifier:
  F1 = P1 or P2 fails -> the replacement-clock model is ALSO wrong; the rung-2 rate law
     has no validated form and the ladder stays halted for a real diagnosis (fitting is
     not validation).
Provided priors declared: as Exp 86 (worlds, implant, mechanism, lengths); fresh birth
seeds 800-807. Fresh separate-root newborns; spines untouched.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMPLANT_A = 1500
IMPLANT_B = 1500
HEAL = 3000
SAMPLE = 250          # measurement interval in steps
N_CHUNKS = HEAL // SAMPLE  # 12
SEEDS = list(range(8))     # offsets 0..7; actual birth seed = BIRTH_BASE + offset
BIRTH_BASE = 800
START = 12
LAMBDA = 0.997
FLOOR = 0.01

# ---------------------------------------------------------------------------
# World A: read mirro's committed cmap layout, fresh births only — spine untouched
# ---------------------------------------------------------------------------

_manifest = json.loads(
    Path("creature/state/mirro/manifest.json").read_text()
)
world_A = World.from_dict(_manifest["world"])

# World B: cmap permuted 0->1->2->0 (every cell changes color — maximal mixture scar)
cmap_B = [(c + 1) % world_A.n_colors for c in world_A.cmap]
world_B = World(
    rows=world_A.rows,
    cols=world_A.cols,
    cmap=cmap_B,
    n_colors=world_A.n_colors,
)

print("Exp 87 — the replacement clock, out-of-sample: the corrected scar-healing rate law.")
print()
print("World A (mirro's committed cmap layout):")
print(f"  rows={world_A.rows}  cols={world_A.cols}  n_colors={world_A.n_colors}")
print("  cmap grid:")
for r in range(world_A.rows):
    row_vals = world_A.cmap[r * world_A.cols : (r + 1) * world_A.cols]
    print("   ", " ".join(str(v) for v in row_vals))
print()
print("World B (cmap permuted 0->1->2->0 from A — every cell changes color):")
print(f"  rows={world_B.rows}  cols={world_B.cols}  n_colors={world_B.n_colors}")
print("  cmap grid:")
for r in range(world_B.rows):
    row_vals = world_B.cmap[r * world_B.cols : (r + 1) * world_B.cols]
    print("   ", " ".join(str(v) for v in row_vals))
print()
print(
    f"  IMPLANT_A={IMPLANT_A}  IMPLANT_B={IMPLANT_B}  HEAL={HEAL}  SAMPLE={SAMPLE}"
    f"  LAMBDA={LAMBDA}  FLOOR={FLOOR}  START={START}"
)
print(f"  Birth seeds: {[BIRTH_BASE + s for s in SEEDS]}")
print()

# ---------------------------------------------------------------------------
# Scar metric
# ---------------------------------------------------------------------------

def scar(c: Creature, cmap_b: list) -> float:
    """S = mean over cells s of (1 - pA[cmap_B[s], s] / pA[:, s].sum()).

    Measures the share of off-true-color mass in world B's terms.
    """
    n_cells = c.pA.shape[1]
    col_totals = c.pA.sum(axis=0)          # (n_cells,)
    # avoid division by zero (should never happen with floor>0)
    safe_totals = np.where(col_totals == 0, 1.0, col_totals)
    true_mass = np.array([c.pA[cmap_b[s], s] for s in range(n_cells)])
    off_share = 1.0 - true_mass / safe_totals
    return float(np.mean(off_share))


# ---------------------------------------------------------------------------
# Lambda stepper: replicates live() EXACTLY plus decay at step start.
#
# RNG derivation: identical to live() — combined_seed = (c._seed * 1_000_003 +
# c.rng_counter) & 0xFFFFFFFFFFFFFFFF.  Decay does NOT consume rng draws so
# action sequences match live() for the same birth seed.
#
# Chunking mirrors live(250) x 12: each call derives rng from current
# rng_counter, runs 250 steps, then increments age_steps and rng_counter by
# the same amounts live(250) would — so rng_counter state is identical.
# ---------------------------------------------------------------------------

def _step_n_chunk(c: Creature, n: int) -> None:
    """Run n lambda-decay steps, consuming exactly one rng_counter slot (like live(n))."""
    world = c.world
    B = world.transition_matrix()   # (n_cells, n_cells, 4)
    combined_seed = (c._seed * 1_000_003 + c.rng_counter) & 0xFFFFFFFFFFFFFFFF
    rng = np.random.default_rng(combined_seed)

    for _ in range(n):
        # --- DECAY FIRST (declared ordering: decay-then-observe) ---
        c.pA *= LAMBDA
        np.maximum(c.pA, FLOOR, out=c.pA)

        # --- A_hat from (decayed) pA ---
        A = c.pA.copy()
        col_sums = A.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        A_hat = A / col_sums

        # --- observe ---
        obs = int(world.cmap[c.true_pos])

        # --- belief update: qs ∝ likelihood(obs) * prior(qs) ---
        likelihood = A_hat[obs, :]
        qs_upd = likelihood * c.qs
        denom = qs_upd.sum()
        if denom > 0:
            qs_upd = qs_upd / denom
        else:
            qs_upd = np.ones(world.n_cells) / world.n_cells

        # --- Dirichlet count learning: pA[obs, :] += qs_upd ---
        c.pA[obs, :] += qs_upd

        # --- Value accumulation (matching live()) ---
        map_cell = int(np.argmax(qs_upd))
        predicted_obs_dist = A_hat[:, map_cell]
        h_predicted = -np.sum(
            predicted_obs_dist * np.log(predicted_obs_dist + 1e-12)
        )
        predictability_weight = np.exp(-h_predicted)
        c.value_counts[obs] += predictability_weight

        # --- choose action, move ---
        action = int(rng.integers(0, 4))
        c.true_pos = world.move(c.true_pos, action)

        # --- advance belief through movement model ---
        c.qs = B[:, :, action] @ qs_upd

    # Advance bookkeeping to match live(n)
    c.age_steps += n
    c.rng_counter += 1


# ---------------------------------------------------------------------------
# Per-seed loop (lambda arm only — no control arm)
# ---------------------------------------------------------------------------

results = []   # list of dicts

for seed_offset in SEEDS:
    birth_seed = BIRTH_BASE + seed_offset

    # -----------------------------------------------------------------------
    # Implant phase (NO decay)
    # -----------------------------------------------------------------------
    c = Creature.birth(f"exp87-s{seed_offset}", world_A, seed=birth_seed)
    c.true_pos = START
    c.live(IMPLANT_A)                      # learn world A

    # Switch world to B
    c.world = world_B
    c.live(IMPLANT_B)                      # live in B without decay — imprints the scar

    S0 = scar(c, cmap_B)

    # Deepcopy for the lambda healing arm
    lam = copy.deepcopy(c)

    # -----------------------------------------------------------------------
    # LAMBDA arm: custom decay stepper, 12 chunks of 250.
    # -----------------------------------------------------------------------
    lam_traj = []    # S after each chunk (12 values)
    for _ in range(N_CHUNKS):
        _step_n_chunk(lam, SAMPLE)
        lam_traj.append(scar(lam, cmap_B))

    # First sampled t with S <= S0/2 (250, 500, ..., 3000), or None
    crossing_t = None
    for chunk_idx, s_val in enumerate(lam_traj):
        t = SAMPLE * (chunk_idx + 1)
        if s_val <= S0 / 2.0:
            crossing_t = t
            break

    results.append({
        "seed_offset": seed_offset,
        "birth_seed": birth_seed,
        "S0": S0,
        "S_lam_250": lam_traj[0],
        "S_lam_end": lam_traj[-1],
        "crossing_t": crossing_t,
        "lam_traj": lam_traj,
    })

# ---------------------------------------------------------------------------
# Seed 0 full trajectory (13 samples: S0 + 12 chunks)
# ---------------------------------------------------------------------------

r0 = results[0]
print("--- SEED 0 FULL TRAJECTORY (S0 + 12 x 250-step samples) ---")
print(f"  S0 = {r0['S0']:.6f}")
traj_steps = [SAMPLE * (i + 1) for i in range(N_CHUNKS)]
print(f"  {'step':>5}  {'S_lam':>9}")
for step, sl in zip(traj_steps, r0["lam_traj"]):
    print(f"  {step:>5}  {sl:>9.6f}")
print()

# ---------------------------------------------------------------------------
# Per-seed table
# ---------------------------------------------------------------------------

print("--- PER-SEED TABLE ---")
header = (
    f"  {'seed':>4}  {'S0':>8}  {'S_lam_250':>9}  {'crossing_t':>10}  {'S_lam_end':>9}"
)
print(header)
print("  " + "-" * (len(header) - 2))
for r in results:
    ct = str(r["crossing_t"]) if r["crossing_t"] is not None else "None"
    print(
        f"  {r['birth_seed']:>4}  {r['S0']:>8.5f}  {r['S_lam_250']:>9.5f}  "
        f"{ct:>10}  {r['S_lam_end']:>9.5f}"
    )
print()

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------

PASS_THRESHOLD = 6   # >= 6/8 seeds

P1_LO = 0.40
P1_HI = 0.50
P2_T_LO = 750
P2_T_HI = 1500
P3_END_MAX = 0.05


def check(label: str, value: bool) -> bool:
    print(f"  {'PASS' if value else 'FAIL'}  {label}")
    return value


print("--- PROPERTY CHECKS ---")

# P1: S(250) in [0.40, 0.50] in >= 6/8 seeds
p1_per_seed = [P1_LO <= r["S_lam_250"] <= P1_HI for r in results]
p1_count = sum(p1_per_seed)
P1_ok = p1_count >= PASS_THRESHOLD
check(
    f"P1 (S(250) in [{P1_LO}, {P1_HI}]): {p1_count}/8 seeds "
    f"(need {PASS_THRESHOLD}): {'PASS' if P1_ok else 'FAIL'}",
    P1_ok,
)

print()

# P2: first crossing t in [750, 1500] in >= 6/8 seeds
p2_per_seed = [
    r["crossing_t"] is not None and P2_T_LO <= r["crossing_t"] <= P2_T_HI
    for r in results
]
p2_count = sum(p2_per_seed)
P2_ok = p2_count >= PASS_THRESHOLD
check(
    f"P2 (crossing_t in [{P2_T_LO}, {P2_T_HI}]): {p2_count}/8 seeds "
    f"(need {PASS_THRESHOLD}): {'PASS' if P2_ok else 'FAIL'}",
    P2_ok,
)

print()

# P3: S(3000) <= 0.05 in >= 6/8 seeds
p3_per_seed = [r["S_lam_end"] <= P3_END_MAX for r in results]
p3_count = sum(p3_per_seed)
P3_ok = p3_count >= PASS_THRESHOLD
check(
    f"P3 (S(3000) <= {P3_END_MAX}): {p3_count}/8 seeds "
    f"(need {PASS_THRESHOLD}): {'PASS' if P3_ok else 'FAIL'}",
    P3_ok,
)

print()

# ---------------------------------------------------------------------------
# Falsifier map
# ---------------------------------------------------------------------------

print("--- FALSIFIER MAP ---")

# F1 = P1 or P2 fails
F1_fired = not P1_ok or not P2_ok

if not F1_fired:
    print(
        f"  F1 did not fire (P1 PASS — S(250) in [{P1_LO}, {P1_HI}] in {p1_count}/8 seeds; "
        f"P2 PASS — crossing_t in [{P2_T_LO}, {P2_T_HI}] in {p2_count}/8 seeds)."
    )
else:
    reasons = []
    if not P1_ok:
        reasons.append(
            f"P1 failed ({p1_count}/8 seeds with S(250) in [{P1_LO}, {P1_HI}])"
        )
    if not P2_ok:
        reasons.append(
            f"P2 failed ({p2_count}/8 seeds with crossing_t in [{P2_T_LO}, {P2_T_HI}])"
        )
    print(
        f"  F1 FIRED: {'; '.join(reasons)} "
        f"-> the replacement-clock model is ALSO wrong; the rung-2 rate law "
        f"has no validated form and the ladder stays halted for a real diagnosis "
        f"(fitting is not validation)."
    )

if P3_ok:
    print(
        f"  P3 holds (S(3000) <= {P3_END_MAX} in {p3_count}/8 seeds — end-state replication confirmed)."
    )
else:
    print(
        f"  P3 failed (S(3000) <= {P3_END_MAX} in only {p3_count}/8 seeds — end-state not replicated)."
    )

print()

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

if not F1_fired and P3_ok:
    print("EXP87: REPLACEMENT CLOCK VALIDATED — ladder resumes (rung 3 next)")
else:
    print("EXP87: F1 — rate law still unvalidated; ladder stays halted")
