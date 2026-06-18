"""Exp 234 — self-other-modeling Rung 3: does ACTING on a model of the other (proactive avoidance) beat REACTIVE stigmergy at coordinating on a shared resource?

Hypothesis: the arc (Exp 229-233) showed ToM PREDICTS the other well; does it PAY in BEHAVIOR? Two clade-mates
share one comfort source; CROWDING wastes it (both present -> 0.3 each; sole present -> 1.0; none -> 0.0). A
creature that PROACTIVELY avoids the source when it observes the OTHER already there (model-based yield) should
produce LESS wasteful crowding -> higher JOINT comfort -- than a REACTIVE stigmergic creature that only retreats
AFTER crowding lowers its own comfort (the Exp 70 baseline). If proactive avoidance does NOT beat reactive
stigmergy, the model-of-the-other is predictively useful but behaviorally INERT (the Exp 70/71 finding --
coordination is stigmergic, ToM adds nothing behavioral).

Setup: shared 5x5 (vela's world), ONE source cell S=12 (center). C1=mirro-fork, C2=vela-fork (forks NEVER saved).
Comfort per step for a creature: both creatures at S -> 0.3 each; exactly one at S -> 1.0 for that one; neither at
S -> 0.0. Two arms x 8 PAIRED seeds x H=2000:
  BASE: both creatures use comfort-gated BFS-toward-S (Exp 70 reactive policy: approach S; comfort-EMA gate ->
        wander when own comfort EMA is low). No model of the other.
  TOM:  both creatures use the SAME approach BUT proactively YIELD: if the OTHER is currently AT S and self is NOT
        at S, wander (do not approach) this step; otherwise behave as BASE. (Observed other-position is a provided
        modality.)
Metrics: JOINT = mean over steps of (comfort_C1 + comfort_C2); CO_OCCUPATION = fraction of steps both at S;
FAIRNESS = min(sum_comfort_C1, sum_comfort_C2) / max(sum_comfort_C1, sum_comfort_C2) (1.0 equal, 0 one starves).

Prediction (ToM PAYS in efficiency / TRUE): (a) mean(TOM_joint - BASE_joint) >= 0.10 AND in >=6/8 seeds; (b)
mean TOM_co_occupation < mean BASE_co_occupation; (c) sanity: mean BASE_co_occupation > 0.05 (the reactive
baseline DOES crowd, so there is room to improve).
Report FAIRNESS for both arms as an honest diagnostic: if TOM wins JOINT but at LOWER fairness (one creature
monopolizes via the other's yield), that is EFFICIENT EXCLUSION (echoing Exp 71's stigmergic lock-in), NOT fair
turn-taking.

Falsifier (ToM behaviorally INERT / FALSE):
  F1: mean(TOM_joint - BASE_joint) < 0.10 OR <6/8 -> proactive model-based avoidance does NOT beat reactive
      stigmergy; the model of the other is predictively useful but behaviorally inert (Exp 70/71 holds).
  F2: mean TOM_co_occupation >= mean BASE_co_occupation -> the proactive-yield mechanism does not reduce crowding.
  F3: mean BASE_co_occupation <= 0.05 -> the baseline does not crowd (no room to test; re-design).
  F4: any committed spine state_hash changes.
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
OUTPUT_PATH = Path("experiments/outputs/exp234.txt")

# Comfort-gate parameters (reuse from exp70 / exp230)
EPS_POLICY = 0.2                  # epsilon-greedy exploration
ALPHA_EMA = 0.1                   # comfort-EMA alpha
THRESH_GATE = 0.75                # comfort gate threshold
LAMBDA_RELAX = 0.01               # away-from-source EMA relaxation toward 1.0
EST_INIT = 1.0                    # initial comfort estimate

# Comfort rule per step (from docstring)
COMFORT_CROWDED = 0.3             # both at S
COMFORT_SOLE = 1.0                # exactly one at S
COMFORT_NONE = 0.0                # neither at S

# Source cell S (center of 5x5 = cell 12)
# This is vela's world; we declare S=12 (center) as stated in docstring.
SOURCE_CELL = 12

# ---------------------------------------------------------------------------
# Load committed spines (read-only — NEVER call .live() or .save() on these)
# ---------------------------------------------------------------------------

mirro_spine = Creature.load(MIRRO_DIR)
vela_spine = Creature.load(VELA_DIR)

hash_mirro_before = mirro_spine._state_hash()
hash_vela_before = vela_spine._state_hash()

# ---------------------------------------------------------------------------
# World setup (vela's world)
# ---------------------------------------------------------------------------

world = vela_spine.world
n_cells = world.n_cells

# BFS distance field to SOURCE_CELL
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

dist_to_source = build_bfs_dist(world, SOURCE_CELL)

# ---------------------------------------------------------------------------
# Policy functions
# ---------------------------------------------------------------------------

def base_action(
    pos: int,
    comfort_est: float,
    rng: np.random.Generator,
) -> int:
    """Comfort-gated epsilon-greedy BFS policy (Exp 70 reactive policy).

    Draw order (mirrors exp70 / exp230 for determinism):
      1. u = rng.random()  (epsilon check)
      2a. if u < EPS_POLICY: rng.integers(0, 4)  (random)
      2b. elif comfort_est < THRESH_GATE: rng.integers(0, 4)  (wander due to gate)
      2c. else: rng.integers(0, len(minimizers))  (greedy BFS, tie-break)
    """
    u = rng.random()
    if u < EPS_POLICY:
        return int(rng.integers(0, 4))
    if comfort_est < THRESH_GATE:
        return int(rng.integers(0, 4))
    best_d = min(dist_to_source[world.move(pos, a)] for a in range(4))
    minimizers = [a for a in range(4) if dist_to_source[world.move(pos, a)] == best_d]
    return int(minimizers[rng.integers(0, len(minimizers))])


def tom_action(
    pos: int,
    other_pos: int,
    comfort_est: float,
    rng: np.random.Generator,
) -> int:
    """ToM policy = BASE + proactive yield check.

    If other_pos == SOURCE_CELL AND self_pos != SOURCE_CELL:
        yield: wander (random action) this step; consume 2 rng draws to match BASE draw parity
        (draw u first, then draw random integer, so draw count matches BASE wander branch)
    Else: behave exactly as BASE.

    RNG draw parity: the yield branch draws u (rng.random()) then rng.integers(0,4) — exactly
    2 draws, same as the BASE wander branch. This ensures paired-seed determinism: for steps
    where the yield fires, the two policy paths both consume 2 draws.
    """
    if other_pos == SOURCE_CELL and pos != SOURCE_CELL:
        # Proactive yield: wander (random action), consuming same draws as wander branch
        _u = rng.random()  # consume epsilon draw
        return int(rng.integers(0, 4))
    # Otherwise: identical to BASE
    return base_action(pos, comfort_est, rng)

# ---------------------------------------------------------------------------
# Per-seed, per-arm runner
# ---------------------------------------------------------------------------

def run_arm(seed: int, arm: str) -> dict:
    """Run one seed/arm pair. arm is 'BASE' or 'TOM'.

    Both creatures start at their loaded true_pos (mirro-fork at mirro.true_pos,
    vela-fork at vela.true_pos). The spec says 'start at loaded true_pos or a fixed
    non-source start; state the choice.' We use the loaded true_pos values, which are
    mirro.true_pos=11, vela.true_pos=22 (neither equals SOURCE_CELL=12).

    RNG seeding: BASE arm seeded at 700000 + seed*10 + {0,1} for C1/C2;
                 TOM  arm seeded at 700001 + seed*10 + {0,1} for C1/C2.
    This pairs the two arms via the same offset, giving independent per-arm streams.
    Both creatures run with independent RNGs (per-creature RNG, as in exp70).
    """
    # Fork both spines (NEVER save)
    c1 = copy.deepcopy(mirro_spine)  # mirro-fork
    c2 = copy.deepcopy(vela_spine)   # vela-fork
    # Start at loaded true_pos (mirro=11, vela=22; neither is SOURCE_CELL=12)
    c1.true_pos = mirro_spine.true_pos
    c2.true_pos = vela_spine.true_pos

    # Per-arm, per-creature RNG seeds (paired across arms by construction)
    if arm == "BASE":
        rng1 = np.random.default_rng(700000 + seed * 10 + 0)
        rng2 = np.random.default_rng(700000 + seed * 10 + 1)
    else:  # TOM
        rng1 = np.random.default_rng(700001 + seed * 10 + 0)
        rng2 = np.random.default_rng(700001 + seed * 10 + 1)

    # Comfort estimates (EMA)
    est1 = EST_INIT
    est2 = EST_INIT

    # Accumulators
    sum_comfort1 = 0.0
    sum_comfort2 = 0.0
    sum_joint = 0.0
    steps_both = 0

    for step in range(H):
        pos1 = c1.true_pos
        pos2 = c2.true_pos

        # --- Compute comfort for this step from START-OF-STEP positions ---
        at1 = (pos1 == SOURCE_CELL)
        at2 = (pos2 == SOURCE_CELL)

        if at1 and at2:
            comfort1_step = COMFORT_CROWDED
            comfort2_step = COMFORT_CROWDED
            steps_both += 1
        elif at1:
            comfort1_step = COMFORT_SOLE
            comfort2_step = COMFORT_NONE
        elif at2:
            comfort1_step = COMFORT_NONE
            comfort2_step = COMFORT_SOLE
        else:
            comfort1_step = COMFORT_NONE
            comfort2_step = COMFORT_NONE

        sum_comfort1 += comfort1_step
        sum_comfort2 += comfort2_step
        sum_joint += comfort1_step + comfort2_step

        # --- Update comfort EMA BEFORE policy (declared order, mirrors exp70/exp230) ---
        if at1:
            est1 = (1 - ALPHA_EMA) * est1 + ALPHA_EMA * comfort1_step
        else:
            est1 += LAMBDA_RELAX * (1.0 - est1)

        if at2:
            est2 = (1 - ALPHA_EMA) * est2 + ALPHA_EMA * comfort2_step
        else:
            est2 += LAMBDA_RELAX * (1.0 - est2)

        # --- Pick actions ---
        if arm == "BASE":
            a1 = base_action(pos1, est1, rng1)
            a2 = base_action(pos2, est2, rng2)
        else:  # TOM: each creature observes the other's current position
            a1 = tom_action(pos1, pos2, est1, rng1)
            a2 = tom_action(pos2, pos1, est2, rng2)

        # --- Apply both moves simultaneously ---
        c1.true_pos = world.move(pos1, a1)
        c2.true_pos = world.move(pos2, a2)

    # Per-seed metrics
    mean_joint = sum_joint / H
    co_occupation = steps_both / H
    if sum_comfort1 > 0 and sum_comfort2 > 0:
        fairness = min(sum_comfort1, sum_comfort2) / max(sum_comfort1, sum_comfort2)
    elif sum_comfort1 == 0 and sum_comfort2 == 0:
        fairness = 1.0  # both starved equally
    else:
        fairness = 0.0  # one starved completely

    return {
        "seed": seed,
        "arm": arm,
        "mean_joint": mean_joint,
        "co_occupation": co_occupation,
        "fairness": fairness,
        "sum_comfort1": sum_comfort1,
        "sum_comfort2": sum_comfort2,
    }

# ---------------------------------------------------------------------------
# Main run loop (paired seeds)
# ---------------------------------------------------------------------------

print("Exp 234 — self-other-modeling Rung 3: proactive avoidance vs reactive stigmergy")
print()

seed_results = {}  # (seed, arm) -> result dict

for seed in SEEDS:
    for arm in ["BASE", "TOM"]:
        r = run_arm(seed, arm)
        seed_results[(seed, arm)] = r

# ---------------------------------------------------------------------------
# Spine integrity check (F4)
# ---------------------------------------------------------------------------

hash_mirro_after = mirro_spine._state_hash()
hash_vela_after = vela_spine._state_hash()

f4_mirro_ok = hash_mirro_before == hash_mirro_after
f4_vela_ok = hash_vela_before == hash_vela_after
f4_ok = f4_mirro_ok and f4_vela_ok

# ---------------------------------------------------------------------------
# Per-seed paired table
# ---------------------------------------------------------------------------

per_seed = []
for seed in SEEDS:
    base = seed_results[(seed, "BASE")]
    tom = seed_results[(seed, "TOM")]
    joint_edge = tom["mean_joint"] - base["mean_joint"]
    per_seed.append({
        "seed": seed,
        "base_joint": base["mean_joint"],
        "tom_joint": tom["mean_joint"],
        "joint_edge": joint_edge,
        "base_coocc": base["co_occupation"],
        "tom_coocc": tom["co_occupation"],
        "base_fair": base["fairness"],
        "tom_fair": tom["fairness"],
        "tom_beats_base_by_010": joint_edge >= 0.10,
        "tom_beats_base": joint_edge > 0.0,
    })

# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------

mean_joint_edge = float(np.mean([ps["joint_edge"] for ps in per_seed]))
count_tom_beats_010 = sum(1 for ps in per_seed if ps["tom_beats_base_by_010"])
count_tom_beats_any = sum(1 for ps in per_seed if ps["tom_beats_base"])
mean_base_coocc = float(np.mean([ps["base_coocc"] for ps in per_seed]))
mean_tom_coocc = float(np.mean([ps["tom_coocc"] for ps in per_seed]))
mean_base_fair = float(np.mean([ps["base_fair"] for ps in per_seed]))
mean_tom_fair = float(np.mean([ps["tom_fair"] for ps in per_seed]))

# Prediction conjuncts
pred_a = (mean_joint_edge >= 0.10) and (count_tom_beats_010 >= 6)
pred_b = mean_tom_coocc < mean_base_coocc
pred_c_sanity = mean_base_coocc > 0.05

# Falsifiers
f1_fired = not ((mean_joint_edge >= 0.10) and (count_tom_beats_010 >= 6))
f2_fired = mean_tom_coocc >= mean_base_coocc
f3_fired = mean_base_coocc <= 0.05
# F4 evaluated above

# Fairness diagnostic: TOM wins JOINT but at lower fairness?
tom_wins_joint = mean_joint_edge > 0
fairness_lower_in_tom = mean_tom_fair < mean_base_fair
efficient_exclusion = tom_wins_joint and fairness_lower_in_tom

# ---------------------------------------------------------------------------
# Build output string
# ---------------------------------------------------------------------------

lines = []
lines.append("=" * 90)
lines.append("Exp 234 — self-other-modeling Rung 3: proactive avoidance (ToM) vs reactive stigmergy (BASE)")
lines.append("=" * 90)
lines.append("")
lines.append("Implementation notes:")
lines.append("  - Shared world: vela's 5x5 world (n_cells=25)")
lines.append(f"  - SOURCE_CELL S = {SOURCE_CELL} (center cell, row=2 col=2)")
lines.append(f"  - Comfort rule: both@S -> {COMFORT_CROWDED} each; sole@S -> {COMFORT_SOLE}; neither -> {COMFORT_NONE}")
lines.append(f"  - C1 start: mirro.true_pos = {mirro_spine.true_pos}  (not S)")
lines.append(f"  - C2 start: vela.true_pos  = {vela_spine.true_pos}  (not S)")
lines.append(f"  - EPS_POLICY={EPS_POLICY}, ALPHA_EMA={ALPHA_EMA}, THRESH_GATE={THRESH_GATE}")
lines.append(f"  - LAMBDA_RELAX={LAMBDA_RELAX}, EST_INIT={EST_INIT}")
lines.append(f"  - H={H} steps, {len(SEEDS)} seeds (0-7), PAIRED: BASE and TOM run same seed")
lines.append(f"  - BASE arm RNG seeds: 700000+seed*10+{{0,1}} per creature")
lines.append(f"  - TOM  arm RNG seeds: 700001+seed*10+{{0,1}} per creature")
lines.append(f"  - Comfort EMA updated BEFORE policy action (mirrors exp70/exp230)")
lines.append(f"  - TOM yield: if other_pos==S and self_pos!=S -> wander (random action)")
lines.append(f"  - TOM yield RNG draw parity: draws u + integers(0,4), matching BASE wander branch")
lines.append(f"  - JOINT computed from START-OF-STEP positions (before move)")
lines.append(f"  - FAIRNESS = min(sum_c1,sum_c2)/max(sum_c1,sum_c2); 1.0=equal, 0.0=one starves")
lines.append(f"  - Spine forks via copy.deepcopy; NEVER saved")
lines.append("")

lines.append("--- PER-SEED TABLE ---")
lines.append("")
hdr = (
    f"{'seed':>4}  {'BASE_joint':>10}  {'TOM_joint':>9}  {'joint_edge':>10}  "
    f"{'BASE_coocc':>10}  {'TOM_coocc':>9}  {'BASE_fair':>9}  {'TOM_fair':>8}  {'TOM>=0.10':>9}"
)
lines.append(hdr)
lines.append("-" * 100)
for ps in per_seed:
    row = (
        f"  {ps['seed']:>2}  {ps['base_joint']:>10.4f}  {ps['tom_joint']:>9.4f}  "
        f"{ps['joint_edge']:>10.4f}  {ps['base_coocc']:>10.4f}  {ps['tom_coocc']:>9.4f}  "
        f"{ps['base_fair']:>9.4f}  {ps['tom_fair']:>8.4f}  "
        f"{'YES' if ps['tom_beats_base_by_010'] else 'NO':>9}"
    )
    lines.append(row)
lines.append("-" * 100)
lines.append(
    f"  {'mean':>4}  {float(np.mean([ps['base_joint'] for ps in per_seed])):>10.4f}  "
    f"{float(np.mean([ps['tom_joint'] for ps in per_seed])):>9.4f}  "
    f"{mean_joint_edge:>10.4f}  {mean_base_coocc:>10.4f}  {mean_tom_coocc:>9.4f}  "
    f"{mean_base_fair:>9.4f}  {mean_tom_fair:>8.4f}"
)
lines.append("")

lines.append("--- AGGREGATES ---")
lines.append("")
lines.append(f"  mean(TOM_joint - BASE_joint) [joint edge]:  {mean_joint_edge:.4f}  (threshold for pred_a: >=0.10)")
lines.append(f"  count(joint_edge >= 0.10) / 8:             {count_tom_beats_010}/8  (threshold for pred_a: >=6/8)")
lines.append(f"  count(TOM_joint > BASE_joint) / 8:         {count_tom_beats_any}/8  (any positive edge)")
lines.append(f"  mean BASE_co_occupation:                    {mean_base_coocc:.4f}  (sanity pred_c threshold: >0.05)")
lines.append(f"  mean TOM_co_occupation:                     {mean_tom_coocc:.4f}  (pred_b: must be < BASE)")
lines.append(f"  mean BASE_fairness:                         {mean_base_fair:.4f}")
lines.append(f"  mean TOM_fairness:                          {mean_tom_fair:.4f}")
lines.append("")

lines.append("--- FAIRNESS DIAGNOSTIC ---")
lines.append("")
lines.append(f"  TOM wins JOINT overall: {'YES' if tom_wins_joint else 'NO'}  (mean_joint_edge={mean_joint_edge:.4f})")
lines.append(f"  TOM fairness < BASE fairness: {'YES' if fairness_lower_in_tom else 'NO'}  (TOM_fair={mean_tom_fair:.4f}, BASE_fair={mean_base_fair:.4f})")
if efficient_exclusion:
    lines.append(
        f"  DIAGNOSIS: EFFICIENT EXCLUSION — TOM wins JOINT but at LOWER fairness."
        f" One creature monopolizes via the other's yield. This echoes Exp 71's"
        f" stigmergic lock-in: proactive yield creates a dominant/yielder asymmetry."
    )
elif tom_wins_joint and not fairness_lower_in_tom:
    lines.append(
        f"  DIAGNOSIS: FAIR TURN-TAKING — TOM wins JOINT and maintains equal or better"
        f" fairness. Proactive yield enables efficient alternation without exclusion."
    )
else:
    lines.append(
        f"  DIAGNOSIS: TOM does not win JOINT overall; fairness comparison is secondary."
    )
lines.append("")

lines.append("--- SPINE INTEGRITY (F4) ---")
lines.append("")
lines.append(f"  mirro before: {hash_mirro_before[:24]}...")
lines.append(f"  mirro after:  {hash_mirro_after[:24]}...")
lines.append(f"  mirro intact: {'YES' if f4_mirro_ok else 'NO --- F4 FIRED'}")
lines.append(f"  vela before:  {hash_vela_before[:24]}...")
lines.append(f"  vela after:   {hash_vela_after[:24]}...")
lines.append(f"  vela intact:  {'YES' if f4_vela_ok else 'NO --- F4 FIRED'}")
lines.append(f"  F4: {'NOT FIRED' if f4_ok else 'FIRED (continuity bug)'}")
lines.append("")

lines.append("--- FALSIFIER EVALUATION ---")
lines.append("")
lines.append(
    f"  F1 (TOM does NOT beat BASE by >=0.10 in >=6/8 seeds): "
    f"{'NOT FIRED' if not f1_fired else 'FIRED'}  "
    f"[mean_edge={mean_joint_edge:.4f} (need>=0.10), count={count_tom_beats_010}/8 (need>=6/8)]"
)
lines.append(
    f"  F2 (TOM co_occupation >= BASE co_occupation): "
    f"{'NOT FIRED' if not f2_fired else 'FIRED'}  "
    f"[TOM_coocc={mean_tom_coocc:.4f}, BASE_coocc={mean_base_coocc:.4f}]"
)
lines.append(
    f"  F3 (BASE does not crowd: co_occupation <= 0.05): "
    f"{'NOT FIRED' if not f3_fired else 'FIRED'}  "
    f"[mean BASE_coocc={mean_base_coocc:.4f} (need >0.05)]"
)
lines.append(
    f"  F4 (spine hash changed): "
    f"{'NOT FIRED' if f4_ok else 'FIRED'}  "
    f"[mirro {'OK' if f4_mirro_ok else 'CHANGED'}, vela {'OK' if f4_vela_ok else 'CHANGED'}]"
)
lines.append("")

lines.append("--- PREDICTION CONJUNCT EVALUATION ---")
lines.append("")
lines.append(
    f"  (a) mean_edge>=0.10 AND count>=6/8:    {'TRUE' if pred_a else 'FALSE'}  "
    f"[edge={mean_joint_edge:.4f}, count={count_tom_beats_010}/8]"
)
lines.append(
    f"  (b) mean TOM_coocc < mean BASE_coocc:  {'TRUE' if pred_b else 'FALSE'}  "
    f"[TOM={mean_tom_coocc:.4f}, BASE={mean_base_coocc:.4f}]"
)
lines.append(
    f"  (c) sanity BASE_coocc > 0.05:          {'TRUE' if pred_c_sanity else 'FALSE'}  "
    f"[BASE_coocc={mean_base_coocc:.4f}]"
)
lines.append("")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

if not f4_ok:
    verdict = "VERDICT: HALT/F4 — spine integrity violated; continuity bug"
elif f3_fired:
    verdict = (
        f"VERDICT: NO_VERDICT/F3 — BASE does not crowd (BASE_coocc={mean_base_coocc:.4f} <= 0.05); "
        f"no room to improve; re-design required"
    )
elif f1_fired and f2_fired:
    verdict = (
        f"VERDICT: FALSE/F1+F2 — proactive ToM yield does NOT beat reactive stigmergy "
        f"(mean_edge={mean_joint_edge:.4f} < 0.10, count={count_tom_beats_010}/8 < 6/8) "
        f"AND does not reduce crowding (TOM_coocc={mean_tom_coocc:.4f} >= BASE_coocc={mean_base_coocc:.4f}); "
        f"model-of-the-other is behaviorally INERT (Exp 70/71 holds)"
    )
elif f1_fired:
    verdict = (
        f"VERDICT: FALSE/F1 — proactive ToM yield does NOT beat reactive stigmergy by >=0.10 "
        f"(mean_edge={mean_joint_edge:.4f}, count={count_tom_beats_010}/8); "
        f"model-of-the-other is behaviorally INERT (Exp 70/71 holds)"
    )
elif f2_fired:
    verdict = (
        f"VERDICT: FALSE/F2 — TOM co_occupation not reduced "
        f"(TOM_coocc={mean_tom_coocc:.4f} >= BASE_coocc={mean_base_coocc:.4f}); "
        f"proactive yield mechanism does not reduce crowding"
    )
elif pred_a and pred_b and pred_c_sanity:
    if efficient_exclusion:
        verdict = (
            f"VERDICT: TRUE/EFFICIENT-EXCLUSION — all conjuncts hold: "
            f"mean_edge={mean_joint_edge:.4f} in {count_tom_beats_010}/8 seeds; "
            f"TOM_coocc={mean_tom_coocc:.4f} < BASE_coocc={mean_base_coocc:.4f}; "
            f"BASE crowds ({mean_base_coocc:.4f} > 0.05). "
            f"BUT TOM fairness ({mean_tom_fair:.4f}) < BASE fairness ({mean_base_fair:.4f}): "
            f"EFFICIENT EXCLUSION, not fair turn-taking — one creature monopolizes via the other's yield"
        )
    else:
        verdict = (
            f"VERDICT: TRUE/FAIR-TURN-TAKING — all conjuncts hold: "
            f"mean_edge={mean_joint_edge:.4f} in {count_tom_beats_010}/8 seeds; "
            f"TOM_coocc={mean_tom_coocc:.4f} < BASE_coocc={mean_base_coocc:.4f}; "
            f"BASE crowds ({mean_base_coocc:.4f} > 0.05); "
            f"fairness maintained (TOM={mean_tom_fair:.4f} >= BASE={mean_base_fair:.4f}); "
            f"proactive avoidance PAYS and enables fair coordination"
        )
else:
    failed = []
    if not pred_a:
        failed.append(f"a(edge={mean_joint_edge:.4f},{count_tom_beats_010}/8)")
    if not pred_b:
        failed.append(f"b(TOM_coocc={mean_tom_coocc:.4f}>=BASE={mean_base_coocc:.4f})")
    if not pred_c_sanity:
        failed.append(f"c(BASE_coocc={mean_base_coocc:.4f}<=0.05)")
    verdict = (
        f"VERDICT: FALSE — conjunct(s) {failed} not met; "
        f"no single falsifier fires cleanly; proactive avoidance not established"
    )

lines.append(verdict)
lines.append("")
lines.append("=" * 90)

output_text = "\n".join(lines)

# ---------------------------------------------------------------------------
# Print and write
# ---------------------------------------------------------------------------

print(output_text)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(output_text + "\n")

print(f"\nOutput written to: {OUTPUT_PATH}")

# Halt on F4
if not f4_ok:
    print("HALT: F4 fired — spine integrity violated")
    sys.exit(1)
