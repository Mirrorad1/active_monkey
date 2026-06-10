"""Exp 86 — graded-uncertainty rung 2: implanted scars heal at the predicted half-life.

Exp 78 proved scars are permanent under non-decaying counts (mirro's drift-era residue,
28.6 percent foreign mass, preserved forever). Exp 85 proved the count-decay mechanism
(LAMBDA=0.997, floor 0.01) is costless in static worlds. This rung tests the healing
claim with the arithmetic's own numbers: a scar of foreign mass m decays as m*LAMBDA^t
(half-life ln2/ln(1/0.997) ~ 231 steps), so an implanted scar should halve within
[150, 400] steps and be essentially gone (S <= 0.05) within 3000 — while a non-decaying
control heals only by dilution (S_end/S_0 ~ 0.5 after doubling its post-implant counts).

Scar implant (shared EXACTLY across arms): a fresh creature lives 1500 steps in world A
(mirro's committed layout), then its world is switched to world B (A's cmap permuted
0->1->2->0 — every cell changes color, maximal mixture scar), where it lives 1500 more
steps (all WITHOUT decay). The implanted creature is then deepcopied into the two arms,
so the healing phase starts from the bit-identical scar.

Scar metric: S = mean over cells of the off-true-color column mass share w.r.t. world B
(Exp 78's metric): S = mean_s (1 - pA[cmap_B[s], s] / pA[:, s].sum()). Sampled every 250
steps through the 3000-step healing phase (lived in static world B).

Predeclared predictions (8 seeds, 700-707; effect sizes per the Exp 79 rule):
  P1 (control = dilution only): S_ctrl(end)/S_ctrl(0) in [0.35, 0.65] in >= 6/8 seeds.
  P2 (lambda heals at the arithmetic's rate): the lambda arm's S crosses S(0)/2 within
     [150, 400] steps AND S(end) <= 0.05, both in >= 6/8 seeds.
  P3 (healing shows where scars live): lambda-arm end mean gate (mean over cells of
     exp(-H(A_hat[:,s]))) >= control's + 0.05, in >= 6/8 seeds.
Falsifiers:
  F1 = P1 fails -> the dilution model of the control is wrong; interpretation of Exp 78
     revisited.
  F2 = P2 fails -> the mechanism does NOT heal at the claimed rate (or at all) — the
     direction's central premise fails at rung 2; halt the ladder for diagnosis.
  F3 = P3 fails -> scars shrink by the metric but gates do not recover; the metric and
     the functional quantity disagree; diagnose before rung 3.
Provided priors declared: worlds A and B, the implant protocol, the mechanism (LAMBDA,
floor), phase lengths, birth seeds. Fresh separate-root newborns; the spines untouched
(mirro's manifest read only for its world layout).
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
BIRTH_BASE = 700
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

print("Exp 86 — graded-uncertainty rung 2: implanted scars heal at the predicted half-life.")
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
# Scar metric and gate metric
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


def mean_gate(c: Creature) -> float:
    """Mean over cells of exp(-H(A_hat[:,s])) — sharpness of learned map."""
    A_hat = c._A_hat()                     # (n_colors, n_cells)
    # entropy per cell
    H = -np.sum(A_hat * np.log(A_hat + 1e-12), axis=0)  # (n_cells,)
    return float(np.mean(np.exp(-H)))


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
# Per-seed loop
# ---------------------------------------------------------------------------

# NOTE on the half-life crossing check with 250-step sampling:
# The arithmetic half-life is ln2/ln(1/0.997) ~ 231 steps, which is < 250.
# With SAMPLE=250 the [150,400] window means S(250) should already be <= S0/2
# (231 < 250 means the crossing happened before the first sample point).
# P2 check: S_lam(250) <= S0/2 AND S_lam(end) <= 0.05.
# We also report S(500) for completeness.

results = []   # list of dicts

for seed_offset in SEEDS:
    birth_seed = BIRTH_BASE + seed_offset

    # -----------------------------------------------------------------------
    # Implant phase (shared, NO decay in either arm yet)
    # -----------------------------------------------------------------------
    c = Creature.birth(f"exp86-s{seed_offset}", world_A, seed=birth_seed)
    c.true_pos = START
    c.live(IMPLANT_A)                      # learn world A

    # Switch world to B (world object is read-only; safe to share across creatures)
    c.world = world_B
    c.live(IMPLANT_B)                      # live in B without decay — imprints the scar

    S0 = scar(c, cmap_B)

    # Deepcopy into the two arms so both start from the bit-identical scar
    ctrl = copy.deepcopy(c)
    lam = copy.deepcopy(c)

    # -----------------------------------------------------------------------
    # CONTROL arm: standard live() in world B, 12 chunks of 250
    # Each ctrl.live(250) call increments rng_counter by 1 — 12 calls total.
    # -----------------------------------------------------------------------
    ctrl_traj = []   # S after each chunk (12 values)
    for _ in range(N_CHUNKS):
        ctrl.live(SAMPLE)
        ctrl_traj.append(scar(ctrl, cmap_B))

    S_ctrl_end = ctrl_traj[-1]
    ratio_ctrl = S_ctrl_end / S0 if S0 > 0 else float("nan")
    gate_ctrl_end = mean_gate(ctrl)

    # -----------------------------------------------------------------------
    # LAMBDA arm: custom decay stepper, 12 chunks of 250.
    # _step_n_chunk mirrors live(250) exactly: same rng derivation from
    # rng_counter (which started identical to ctrl's post-implant counter),
    # increments age_steps and rng_counter identically.
    # -----------------------------------------------------------------------
    lam_traj = []    # S after each chunk (12 values)
    for _ in range(N_CHUNKS):
        _step_n_chunk(lam, SAMPLE)
        lam_traj.append(scar(lam, cmap_B))

    S_lam_250 = lam_traj[0]    # S after first 250 steps
    S_lam_end = lam_traj[-1]
    gate_lam_end = mean_gate(lam)
    gate_diff = gate_lam_end - gate_ctrl_end

    results.append({
        "seed_offset": seed_offset,
        "birth_seed": birth_seed,
        "S0": S0,
        "S_ctrl_end": S_ctrl_end,
        "ratio_ctrl": ratio_ctrl,
        "S_lam_250": S_lam_250,
        "S_lam_500": lam_traj[1] if len(lam_traj) > 1 else float("nan"),
        "S_lam_end": S_lam_end,
        "gate_ctrl_end": gate_ctrl_end,
        "gate_lam_end": gate_lam_end,
        "gate_diff": gate_diff,
        "ctrl_traj": ctrl_traj,
        "lam_traj": lam_traj,
    })

# ---------------------------------------------------------------------------
# Seed 0 full trajectory (13 samples: S0 + 12 chunks)
# ---------------------------------------------------------------------------

r0 = results[0]
print("--- SEED 0 FULL TRAJECTORY (S0 + 12 x 250-step samples) ---")
print(f"  S0 = {r0['S0']:.6f}")
traj_steps = [SAMPLE * (i + 1) for i in range(N_CHUNKS)]
print(f"  {'step':>5}  {'S_ctrl':>9}  {'S_lam':>9}")
for step, sc, sl in zip(traj_steps, r0["ctrl_traj"], r0["lam_traj"]):
    print(f"  {step:>5}  {sc:>9.6f}  {sl:>9.6f}")
print()

# ---------------------------------------------------------------------------
# Per-seed table
# ---------------------------------------------------------------------------

print("--- PER-SEED TABLE ---")
header = (
    f"  {'seed':>4}  {'S0':>8}  {'S_ctrl_end':>10}  {'ratio_ctrl':>10}  "
    f"{'S_lam_250':>9}  {'S_lam_end':>9}  {'gate_ctrl':>9}  {'gate_lam':>9}  {'gate_diff':>9}"
)
print(header)
print("  " + "-" * (len(header) - 2))
for r in results:
    print(
        f"  {r['birth_seed']:>4}  {r['S0']:>8.5f}  {r['S_ctrl_end']:>10.5f}  "
        f"{r['ratio_ctrl']:>10.5f}  {r['S_lam_250']:>9.5f}  {r['S_lam_end']:>9.5f}  "
        f"{r['gate_ctrl_end']:>9.5f}  {r['gate_lam_end']:>9.5f}  {r['gate_diff']:>9.5f}"
    )
print()

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------

PASS_THRESHOLD = 6   # >= 6/8 seeds

P1_RATIO_LO = 0.35
P1_RATIO_HI = 0.65
P2_END_MAX = 0.05
P3_GATE_DIFF_MIN = 0.05


def check(label: str, value: bool) -> bool:
    print(f"  {'PASS' if value else 'FAIL'}  {label}")
    return value


print("--- PROPERTY CHECKS ---")

# P1: S_ctrl(end)/S_ctrl(0) in [0.35, 0.65] in >= 6/8 seeds
p1_per_seed = [P1_RATIO_LO <= r["ratio_ctrl"] <= P1_RATIO_HI for r in results]
p1_count = sum(p1_per_seed)
P1_ok = p1_count >= PASS_THRESHOLD
check(
    f"P1 (ratio_ctrl in [{P1_RATIO_LO}, {P1_RATIO_HI}]): {p1_count}/8 seeds "
    f"(need {PASS_THRESHOLD}): {'PASS' if P1_ok else 'FAIL'}",
    P1_ok,
)

print()

# P2: S_lam(250) <= S0/2 AND S_lam(end) <= 0.05, both in >= 6/8 seeds
# Half-life ~ 231 steps < 250, so S(250) should already be <= S0/2.
p2_per_seed = [
    (r["S_lam_250"] <= r["S0"] / 2.0) and (r["S_lam_end"] <= P2_END_MAX)
    for r in results
]
p2_count = sum(p2_per_seed)
P2_ok = p2_count >= PASS_THRESHOLD

# Breakdown counts for reporting
p2a_count = sum(r["S_lam_250"] <= r["S0"] / 2.0 for r in results)
p2b_count = sum(r["S_lam_end"] <= P2_END_MAX for r in results)
check(
    f"P2a (S_lam(250) <= S0/2 — half-life ~231 < 250): {p2a_count}/8 seeds "
    f"(need {PASS_THRESHOLD}): {'PASS' if p2a_count >= PASS_THRESHOLD else 'FAIL'}",
    p2a_count >= PASS_THRESHOLD,
)
check(
    f"P2b (S_lam(end) <= {P2_END_MAX}): {p2b_count}/8 seeds "
    f"(need {PASS_THRESHOLD}): {'PASS' if p2b_count >= PASS_THRESHOLD else 'FAIL'}",
    p2b_count >= PASS_THRESHOLD,
)
check(
    f"P2 (both): {p2_count}/8 seeds "
    f"(need {PASS_THRESHOLD}): {'PASS' if P2_ok else 'FAIL'}",
    P2_ok,
)

print()

# P3: gate_lam_end >= gate_ctrl_end + 0.05 in >= 6/8 seeds
p3_per_seed = [r["gate_diff"] >= P3_GATE_DIFF_MIN for r in results]
p3_count = sum(p3_per_seed)
P3_ok = p3_count >= PASS_THRESHOLD
mean_gate_diff = float(np.mean([r["gate_diff"] for r in results]))
check(
    f"P3 (gate_lam >= gate_ctrl + {P3_GATE_DIFF_MIN}; mean_diff={mean_gate_diff:.5f}): "
    f"{p3_count}/8 seeds (need {PASS_THRESHOLD}): {'PASS' if P3_ok else 'FAIL'}",
    P3_ok,
)

print()

# ---------------------------------------------------------------------------
# Falsifier map
# ---------------------------------------------------------------------------

print("--- FALSIFIER MAP ---")

F1_fired = not P1_ok
F2_fired = not P2_ok
F3_fired = not P3_ok

if not F1_fired:
    print(
        f"  F1 did not fire (P1 PASS — dilution model holds; ratio_ctrl in "
        f"[{P1_RATIO_LO}, {P1_RATIO_HI}] in {p1_count}/8 seeds)."
    )
else:
    print(
        f"  F1 FIRED: P1 failed ({p1_count}/8 seeds with ratio_ctrl in "
        f"[{P1_RATIO_LO}, {P1_RATIO_HI}]) "
        f"-> the dilution model of the control is wrong; interpretation of Exp 78 revisited."
    )

if not F2_fired:
    print(
        f"  F2 did not fire (P2 PASS — lambda heals at the arithmetic's rate; "
        f"half-life crossing in {p2a_count}/8, end<=0.05 in {p2b_count}/8, both in {p2_count}/8 seeds)."
    )
else:
    print(
        f"  F2 FIRED: P2 failed (half-life crossing {p2a_count}/8, end<=0.05 {p2b_count}/8, "
        f"combined {p2_count}/8 seeds) "
        f"-> the mechanism does NOT heal at the claimed rate (or at all) — "
        f"the direction's central premise fails at rung 2; halt the ladder for diagnosis."
    )

if not F3_fired:
    print(
        f"  F3 did not fire (P3 PASS — gate recovers with scar healing; "
        f"gate_diff >= {P3_GATE_DIFF_MIN} in {p3_count}/8 seeds, mean_diff={mean_gate_diff:.5f})."
    )
else:
    print(
        f"  F3 FIRED: P3 failed (gate_diff >= {P3_GATE_DIFF_MIN} only in {p3_count}/8 seeds; "
        f"mean_diff={mean_gate_diff:.5f}) "
        f"-> scars shrink by the metric but gates do not recover; "
        f"the metric and the functional quantity disagree; diagnose before rung 3."
    )

print()

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

n_fired = sum([F1_fired, F2_fired, F3_fired])

if n_fired == 0:
    print("EXP86: SCARS HEAL AS PREDICTED — rung 3 (the tradeoff curve) unlocked")
elif F2_fired and not F1_fired and not F3_fired:
    print("EXP86: F2 — healing mechanism does not work at the claimed rate; halt ladder")
elif F1_fired and not F2_fired and not F3_fired:
    print("EXP86: F1 — dilution model wrong; revisit Exp 78 interpretation")
elif F3_fired and not F1_fired and not F2_fired:
    print("EXP86: F3 — scar metric and gate quantity disagree; diagnose before rung 3")
elif F1_fired and F2_fired and not F3_fired:
    print("EXP86: F1 + F2 — dilution model wrong AND healing rate wrong")
elif F1_fired and F3_fired and not F2_fired:
    print("EXP86: F1 + F3 — dilution model wrong AND metric/gate disagree")
elif F2_fired and F3_fired and not F1_fired:
    print("EXP86: F2 + F3 — healing rate wrong AND metric/gate disagree")
else:
    print("EXP86: F1 + F2 + F3 — all three falsifiers fired; full diagnosis required")
