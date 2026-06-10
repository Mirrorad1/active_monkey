"""Exp 90 — graded-uncertainty rung 4b: the empty fridge meets the window arithmetic —
which predicts an IMPOSSIBILITY.

Exp 62 (the empty-fridge death): under real scarcity, every map type died standing at the
old food cell after the food moved — including the in-window decay arm (lambda=0.997),
because unlearning could not beat the viability clock. The window framework now makes
that quantitative: camped at the empty fridge, contradictory evidence accrues ~1/step, so
the stale belief flips on the half-life t_half = ln2/ln(1/lambda) — 231 steps at 0.997,
69 at 0.99, 34 at 0.98, 23 at 0.97 — against a post-move survival budget of roughly
25-40 steps. Meanwhile the ACCURACY FLOOR (Exp 60) demands enough equilibrium mass to
hold a map at all: at lambda=0.97 a non-camped cell holds ~1.3 counts. The two
requirements may exclude each other: the map-decay window that beats this viability
clock may not exist — which would prove Exp 62's other named mechanism (failure-driven
exploration) NECESSARY rather than optional. Either outcome of P3 is the finding.

Protocol: Exp 62's IN-DECAY arm replicated exactly (world, food cell, energy decay,
policy trigger, settling, single food-move event, budget), with the map-decay rate swept
over LAMBDAS = [0.997, 0.99, 0.98, 0.97], 8 fresh seeds per arm.

Predeclared:
  P1 (viability side of the window): lambda = 0.997 and 0.99 arms die post-move in
     >= 6/8 seeds each (t_half 231 and 69 >> the post-move budget).
  P2 (monotonicity): post-move survival counts are non-decreasing as lambda decreases
     (faster decay never hurts survival here).
  P3 (the question; interpretation rule fixed in advance): let S = max survival count
     over the 0.98 and 0.97 arms. S >= 6/8 -> a viable map-window EXISTS for this
     ecology (the empty fridge is solvable by forgetting alone; the exploration override
     is an optimization, not a necessity). S <= 2/8 -> the EMPTY WINDOW is confirmed
     (no lambda both holds a map and unlearns within the viability budget; the
     exploration override is NECESSARY). S in [3, 5] -> borderline; reported as a dose
     curve, no binary verdict.
Falsifiers:
  F1 = P1 fails (a slow-decay arm survives >= 3/8) -> the half-life arithmetic is wrong
     at the fridge; diagnose.
  F2 = P2 fails -> survival is not monotone in decay speed; the window story is
     incomplete; diagnose.
Provided priors declared: everything Exp 62 declared (its ecology and policy coupling
are harness-provided), plus the lambda sweep and fresh seeds. Fresh births with Exp 62's
settling; spines untouched (nothing loaded).
"""
from __future__ import annotations

import math

import numpy as np

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# Constants — Exp 62 IN-DECAY arm protocol, exactly
# ---------------------------------------------------------------------------

ROWS, COLS = 5, 5
N_CELLS = ROWS * COLS
N_COLORS = 3
BUDGET = 2000
SETTLE = 800
DECAY_FLOOR = 0.01
ENERGY_START = 1.0
ENERGY_DECAY = 0.02          # 50-step autonomy
ENERGY_THRESH = 0.5
GAMMA = 0.9
FOOD_MOVE_STEP = 1000        # food cell relocates at this step
FOOD_PRE = 0                 # cell 0 = corner (0,0)
FOOD_POST = 24               # cell 24 = corner (4,4)

# Exp 90 sweep — fresh seeds (base 1000 + arm_index*100 + seed_offset)
LAMBDAS = [0.997, 0.99, 0.98, 0.97]
N_SEEDS = 8


# ---------------------------------------------------------------------------
# World construction — identical to Exp 62 build_world()
# ---------------------------------------------------------------------------

def build_world(food_cell: int) -> World:
    """Background: color (i%2). Food cell gets color 2."""
    cmap = [i % 2 for i in range(N_CELLS)]
    cmap[food_cell] = 2
    return World(rows=ROWS, cols=COLS, cmap=cmap, n_colors=N_COLORS)


PRE_WORLD = build_world(FOOD_PRE)
POST_WORLD = build_world(FOOD_POST)


# ---------------------------------------------------------------------------
# VI helper — identical to Exp 62
# ---------------------------------------------------------------------------

def vi_toward_food(believed_food: list[int], world: World) -> np.ndarray:
    """Value-iterate: reward=1 at believed food cells, gamma=0.9.
    Returns Q[cell, action]."""
    T = np.zeros((N_CELLS, 4), dtype=int)
    for cell in range(N_CELLS):
        for a in range(4):
            T[cell, a] = world.move(cell, a)
    R = np.zeros(N_CELLS)
    for fc in believed_food:
        R[fc] = 1.0
    V = np.zeros(N_CELLS)
    for _ in range(80):
        V_new = V.copy()
        for cell in range(N_CELLS):
            best = -1e9
            for a in range(4):
                nxt = T[cell, a]
                val = R[nxt] + GAMMA * V[nxt]
                if val > best:
                    best = val
            V_new[cell] = best
        V = V_new
    Q = np.zeros((N_CELLS, 4))
    for cell in range(N_CELLS):
        for a in range(4):
            nxt = T[cell, a]
            Q[cell, a] = R[nxt] + GAMMA * V[nxt]
    return Q


# ---------------------------------------------------------------------------
# Single-run function — Exp 62 IN-DECAY arm protocol parameterised by lambda
# ---------------------------------------------------------------------------

def run_agent(lam: float, seed_offset: int, arm_index: int) -> dict:
    """Replicate Exp 62's IN-DECAY arm exactly with lambda=lam and fresh seeds."""
    birth_seed = 1000 + arm_index * 100 + seed_offset
    arm_name = f"exp90-lam{lam}-s{seed_offset}"
    agent = Creature.birth(arm_name, PRE_WORLD, seed=birth_seed)

    # ------------------------------------------------------------------
    # Settle phase: 800 random-walk steps in the pre-move world
    # (identical to Exp 62: observe->learn->decay->random-action->move->belief)
    # ------------------------------------------------------------------
    B0 = PRE_WORLD.transition_matrix()
    settle_rng = np.random.default_rng((arm_index + 1) * 1000 + seed_offset)
    for _ in range(SETTLE):
        A_hat = agent._A_hat()
        obs = int(PRE_WORLD.cmap[agent.true_pos])
        likelihood = A_hat[obs, :]
        qs_up = likelihood * agent.qs
        denom = qs_up.sum()
        qs_up = qs_up / denom if denom > 0 else np.ones(N_CELLS) / N_CELLS
        agent.pA[obs, :] += qs_up
        # map-decay applied at Exp-62's point: after learning, before next step
        agent.pA = np.maximum(agent.pA * lam, DECAY_FLOOR)
        action = int(settle_rng.integers(0, 4))
        agent.true_pos = PRE_WORLD.move(agent.true_pos, action)
        agent.qs = B0[:, :, action] @ qs_up

    # Post-settle believed-food set (diagnostic)
    A_hat_post = agent._A_hat()
    believed_food_settle = [c for c in range(N_CELLS)
                            if int(np.argmax(A_hat_post[:, c])) == 2]

    # ------------------------------------------------------------------
    # Survival phase: 2000 steps; world switches at step 1000
    # ------------------------------------------------------------------
    E = ENERGY_START
    dead = False
    death_step = None
    death_pos = None
    current_world = PRE_WORLD

    for step in range(BUDGET):
        # World switch at step 1000 (food relocates)
        if step == FOOD_MOVE_STEP:
            current_world = POST_WORLD
            agent.world = current_world
        B = current_world.transition_matrix()

        # Observe + learn + decay (Exp 62 ordering)
        A_hat = agent._A_hat()
        obs = int(current_world.cmap[agent.true_pos])
        likelihood = A_hat[obs, :]
        qs_up = likelihood * agent.qs
        denom = qs_up.sum()
        qs_up = qs_up / denom if denom > 0 else np.ones(N_CELLS) / N_CELLS
        agent.pA[obs, :] += qs_up
        agent.pA = np.maximum(agent.pA * lam, DECAY_FLOOR)

        # Action selection: intero ON, explore if E >= thresh, VI if E < thresh
        rng_act = np.random.default_rng(birth_seed * 1_000_003 + SETTLE + step + 1)
        if E < ENERGY_THRESH:
            A_hat_curr = agent._A_hat()
            believed_food = [c for c in range(N_CELLS)
                             if int(np.argmax(A_hat_curr[:, c])) == 2]
            if believed_food:
                Q = vi_toward_food(believed_food, current_world)
                q_row = Q[agent.true_pos]
                max_q = q_row.max()
                best_actions = np.where(q_row == max_q)[0]
                action = int(rng_act.choice(best_actions))
            else:
                action = int(rng_act.integers(0, 4))
        else:
            action = int(rng_act.integers(0, 4))

        # Move
        agent.true_pos = current_world.move(agent.true_pos, action)

        # Advance belief
        agent.qs = B[:, :, action] @ qs_up

        # Energy update
        E -= ENERGY_DECAY
        if current_world.cmap[agent.true_pos] == 2:
            E = ENERGY_START   # refill to 1.0 on real food cell

        if E <= 0:
            dead = True
            death_step = step + 1
            death_pos = agent.true_pos
            break

    survived = not dead

    # Believed-food set at end/death
    A_hat_end = agent._A_hat()
    believed_food_end = [c for c in range(N_CELLS)
                         if int(np.argmax(A_hat_end[:, c])) == 2]

    died_post_move = (dead and death_step is not None and death_step > FOOD_MOVE_STEP)
    death_at_old_food = (dead and death_pos == FOOD_PRE)

    return {
        "survived": survived,
        "death_step": death_step if dead else None,
        "death_pos": death_pos,
        "died_post_move": died_post_move,
        "death_at_old_food": death_at_old_food,
        "believed_food_settle": believed_food_settle,
        "believed_food_end": believed_food_end,
        "seed_offset": seed_offset,
        "lam": lam,
    }


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

print("Exp 90 — graded-uncertainty rung 4b: empty fridge meets window arithmetic")
print(f"Budget={BUDGET}  settle={SETTLE}  n_seeds={N_SEEDS}  "
      f"energy_thresh={ENERGY_THRESH}  energy_decay={ENERGY_DECAY}")
print(f"food_pre=cell{FOOD_PRE}  food_post=cell{FOOD_POST}  "
      f"food_move_step={FOOD_MOVE_STEP}")
print(f"LAMBDAS={LAMBDAS}")
print()

# Half-life reference for each lambda
print("Half-life reference: t_half = ln2 / ln(1/lambda)")
for lam in LAMBDAS:
    t_half = math.log(2) / math.log(1.0 / lam)
    print(f"  lambda={lam}: t_half={t_half:.1f} steps")
print()

# ---------------------------------------------------------------------------
# Run all arms
# ---------------------------------------------------------------------------

arm_results: dict[float, list[dict]] = {}

for arm_index, lam in enumerate(LAMBDAS):
    results = []
    for seed_offset in range(N_SEEDS):
        r = run_agent(lam, seed_offset, arm_index)
        results.append(r)
        birth_seed = 1000 + arm_index * 100 + seed_offset
        pos_str = str(r["death_pos"]) if r["death_step"] is not None else "-"
        dstep_str = str(r["death_step"]) if r["death_step"] is not None else "-"
        print(
            f"lambda={lam}  seed={seed_offset}  birth={birth_seed}  "
            f"survived={r['survived']}  death_step={dstep_str}  "
            f"death_pos={pos_str}  "
            f"died_post_move={r['died_post_move']}  "
            f"death_at_old_food={r['death_at_old_food']}  "
            f"believed_food_end={r['believed_food_end']}"
        )
    arm_results[lam] = results
    surv_k = sum(1 for r in results if r["survived"])
    post_move_deaths = sum(1 for r in results if r["died_post_move"])
    old_food_deaths = sum(1 for r in results if r["death_at_old_food"])
    print(
        f"  -> lambda={lam}: survived {surv_k}/{N_SEEDS}  "
        f"post_move_deaths={post_move_deaths}/{N_SEEDS}  "
        f"died_at_old_food={old_food_deaths}/{N_SEEDS}"
    )
    print()

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("--- SUMMARY TABLE ---")
print(f"  {'lambda':>7}  {'survived':>8}  {'post_move_deaths':>16}  {'at_old_food':>12}")
print("  " + "-" * 50)
surv_by_lam: dict[float, int] = {}
for lam in LAMBDAS:
    res = arm_results[lam]
    surv_k = sum(1 for r in res if r["survived"])
    pm_deaths = sum(1 for r in res if r["died_post_move"])
    old_deaths = sum(1 for r in res if r["death_at_old_food"])
    surv_by_lam[lam] = surv_k
    print(
        f"  {lam:>7}  {surv_k:>5}/{N_SEEDS}    "
        f"{pm_deaths:>8}/{N_SEEDS}         "
        f"{old_deaths:>5}/{N_SEEDS}"
    )
print()

# ---------------------------------------------------------------------------
# Predeclared checks
# ---------------------------------------------------------------------------

print("--- PREDECLARED CHECKS ---")

# P1: lambda=0.997 and lambda=0.99 each die post-move in >= 6/8
# (equivalently survived <= 2/8 in both)
pm_deaths_0997 = sum(1 for r in arm_results[0.997] if r["died_post_move"])
pm_deaths_099  = sum(1 for r in arm_results[0.99]  if r["died_post_move"])
surv_0997 = surv_by_lam[0.997]
surv_099  = surv_by_lam[0.99]
P1_0997 = pm_deaths_0997 >= 6
P1_099  = pm_deaths_099  >= 6
P1_ok   = P1_0997 and P1_099
print(
    f"  P1a (lambda=0.997 post-move deaths >= 6/8): "
    f"{pm_deaths_0997}/8  {'PASS' if P1_0997 else 'FAIL'}"
)
print(
    f"  P1b (lambda=0.99  post-move deaths >= 6/8): "
    f"{pm_deaths_099}/8   {'PASS' if P1_099 else 'FAIL'}"
)
print(f"  P1 overall: {'PASS' if P1_ok else 'FAIL'}")
print()

# P2: survival counts non-decreasing as lambda decreases
# LAMBDAS = [0.997, 0.99, 0.98, 0.97] — decreasing lambda, so we want
# surv_by_lam[0.997] <= surv_by_lam[0.99] <= surv_by_lam[0.98] <= surv_by_lam[0.97]
surv_sequence = [surv_by_lam[lam] for lam in LAMBDAS]
P2_ok = all(surv_sequence[i] <= surv_sequence[i + 1]
            for i in range(len(surv_sequence) - 1))
print(
    f"  P2 (survival non-decreasing as lambda decreases): "
    f"{surv_sequence}  {'PASS' if P2_ok else 'FAIL'}"
)
print()

# P3: S = max(surv(0.98), surv(0.97))
S = max(surv_by_lam[0.98], surv_by_lam[0.97])
if S >= 6:
    p3_verdict = "VIABLE WINDOW EXISTS"
    p3_label = f"S={S}/8 >= 6 => viable window exists"
elif S <= 2:
    p3_verdict = "EMPTY WINDOW"
    p3_label = f"S={S}/8 <= 2 => empty window confirmed"
else:
    p3_verdict = "BORDERLINE"
    p3_label = f"S={S}/8 in [3,5] => borderline; dose curve reported"
print(f"  P3 (the window question): {p3_label}")
print(
    f"    surv(0.98)={surv_by_lam[0.98]}/8  surv(0.97)={surv_by_lam[0.97]}/8  S={S}/8"
)
print()

# Dose curve (always shown for P3 context)
print("  Dose curve (lambda -> survival count):")
for lam in LAMBDAS:
    t_half = math.log(2) / math.log(1.0 / lam)
    print(
        f"    lambda={lam}  t_half={t_half:6.1f}  survived={surv_by_lam[lam]}/{N_SEEDS}"
    )
print()

# ---------------------------------------------------------------------------
# Falsifier map
# ---------------------------------------------------------------------------

print("--- FALSIFIER MAP ---")

F1_fired = not P1_ok
F2_fired = not P2_ok

if not F1_fired:
    print(
        f"  F1 did not fire (P1 PASS — slow-decay arms both died post-move "
        f">= 6/8; half-life arithmetic consistent with fridge data)."
    )
else:
    slow_surv = []
    if surv_0997 >= 3:
        slow_surv.append(f"lambda=0.997 survived {surv_0997}/8")
    if surv_099 >= 3:
        slow_surv.append(f"lambda=0.99 survived {surv_099}/8")
    print(
        f"  F1 FIRED: P1 failed ({'; '.join(slow_surv)}) -> "
        f"the half-life arithmetic is wrong at the fridge; diagnose."
    )

if not F2_fired:
    print(
        f"  F2 did not fire (P2 PASS — survival {surv_sequence} non-decreasing "
        f"as lambda decreases)."
    )
else:
    print(
        f"  F2 FIRED: P2 failed — survival {surv_sequence} is NOT monotone in "
        f"decay speed; the window story is incomplete; diagnose."
    )

print()

# ---------------------------------------------------------------------------
# Final verdict line
# ---------------------------------------------------------------------------

if F1_fired or F2_fired:
    f_tags = []
    if F1_fired:
        f_tags.append(f"F1(half-life arithmetic wrong at fridge)")
    if F2_fired:
        f_tags.append(f"F2(survival not monotone)")
    print(f"EXP90: FALSIFIER(S) FIRED — {'; '.join(f_tags)} (S={S}/8)")
elif p3_verdict == "VIABLE WINDOW EXISTS":
    print(
        f"EXP90: VIABLE WINDOW EXISTS (S={S}/8)"
    )
elif p3_verdict == "EMPTY WINDOW":
    print(
        f"EXP90: EMPTY WINDOW CONFIRMED — exploration override is NECESSARY (S={S}/8)"
    )
else:
    print(
        f"EXP90: BORDERLINE (S={S}/8) — dose curve reported"
    )
