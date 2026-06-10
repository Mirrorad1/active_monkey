"""Exp 62 — the interoceptive stake v2: scarcity makes it binding (rung 6, attempt 2).

Exp 61's ecology never engaged the stake (mean E ~0.90 everywhere). v2, predeclared in
Exp 61's entry with one stated refinement (single food move for a clean before/after):
ONE food cell (4% of the world), E decay 0.02/step (50-step autonomy), trigger 0.5,
refill to 1.0 on the REAL food cell; 2000-step budget; the food cell RELOCATES once at
step 1000 from cell 0 (corner (0,0)) to cell 24 (corner (4,4)). Background cmap: color
(i%2) everywhere except the food cell = color 2.
Same 2x2 arms x 5 seeds (births 40+s, settling 800 steps in the pre-move world):
  IN-DECAY  intero ON,  map decay lambda=0.997 (floor 0.01)
  IN-FROZEN intero ON,  non-decaying map
  NO-DECAY  intero OFF (always random walk), decay map
  NO-FROZEN intero OFF, frozen map
New diagnostic: post-move visits to the OLD food cell (the empty-fridge signature).
Predeclared: (i) IN-DECAY survives >= 4/5 (honestly uncertain: after the move its
believed-food set empties until exploration re-discovers one cell within 50-step
autonomy windows -- a failure is a re-discovery finding, not noise); (ii) IN-FROZEN
dies post-move >= 4/5 with repeated old-cell revisits; (iii) both intero arms outlive
both no-intero arms on median survival in the PRE-move phase economy (operationalized:
median survival steps of intero arms > no-intero arms overall).
Falsifiers: IN-FROZEN survives >= 3/5 (the law's survival consequence fails); intero
median <= no-intero median (rung 6's honest FAIL, machinery now actually engaged).
All outcomes logged honestly. All machinery provided (declared).
"""
from __future__ import annotations

import numpy as np

from active_loop.creature import Creature, World

ROWS, COLS = 5, 5
N_CELLS = ROWS * COLS
N_COLORS = 3
BUDGET = 2000
SETTLE = 800
SEEDS = list(range(5))      # s = 0..4 -> birth seed 40+s
DECAY_RATE = 0.997
DECAY_FLOOR = 0.01
ENERGY_START = 1.0
ENERGY_DECAY = 0.02         # 50-step autonomy
ENERGY_THRESH = 0.5
GAMMA = 0.9
FOOD_MOVE_STEP = 1000       # food cell relocates at this step
FOOD_PRE = 0                # cell 0 = corner (0,0)
FOOD_POST = 24              # cell 24 = corner (4,4)


def build_world(food_cell: int) -> World:
    """Background: color (i%2). Food cell gets color 2."""
    cmap = [i % 2 for i in range(N_CELLS)]
    cmap[food_cell] = 2
    return World(rows=ROWS, cols=COLS, cmap=cmap, n_colors=N_COLORS)


PRE_WORLD = build_world(FOOD_PRE)
POST_WORLD = build_world(FOOD_POST)


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


def run_agent(arm_name: str, use_intero: bool, use_decay: bool,
              seed: int, arm_index: int) -> dict:
    birth_seed = 40 + seed
    agent = Creature.birth(arm_name, PRE_WORLD, seed=birth_seed)

    # Settle phase: 800 random-walk steps learning in the pre-move world
    B0 = PRE_WORLD.transition_matrix()
    settle_rng = np.random.default_rng((arm_index + 1) * 1000 + seed)
    for _ in range(SETTLE):
        A_hat = agent._A_hat()
        obs = int(agent.world.cmap[agent.true_pos])
        likelihood = A_hat[obs, :]
        qs_up = likelihood * agent.qs
        denom = qs_up.sum()
        qs_up = qs_up / denom if denom > 0 else np.ones(N_CELLS) / N_CELLS
        agent.pA[obs, :] += qs_up
        if use_decay:
            agent.pA = np.maximum(agent.pA * DECAY_RATE, DECAY_FLOOR)
        action = int(settle_rng.integers(0, 4))
        agent.true_pos = agent.world.move(agent.true_pos, action)
        agent.qs = B0[:, :, action] @ qs_up

    # Report post-settle believed-food set
    A_hat_post = agent._A_hat()
    believed_food_settle = [c for c in range(N_CELLS)
                            if int(np.argmax(A_hat_post[:, c])) == 2]

    # Survival phase: 2000 steps; world switches at step 1000
    E = ENERGY_START
    energies = []
    dead = False
    death_step = None
    death_pos = None
    oldcell_visits_postmove = 0
    current_world = PRE_WORLD

    for step in range(BUDGET):
        # World switch at step 1000
        if step == FOOD_MOVE_STEP:
            current_world = POST_WORLD
            agent.world = current_world
        B = current_world.transition_matrix()

        # Observe + learn
        A_hat = agent._A_hat()
        obs = int(current_world.cmap[agent.true_pos])
        likelihood = A_hat[obs, :]
        qs_up = likelihood * agent.qs
        denom = qs_up.sum()
        qs_up = qs_up / denom if denom > 0 else np.ones(N_CELLS) / N_CELLS
        agent.pA[obs, :] += qs_up
        if use_decay:
            agent.pA = np.maximum(agent.pA * DECAY_RATE, DECAY_FLOOR)

        # Action selection
        rng_act = np.random.default_rng(birth_seed * 1_000_003 + SETTLE + step + 1)
        if use_intero and E < ENERGY_THRESH:
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
            E = min(ENERGY_START, E + (ENERGY_START - E) + (ENERGY_START - E))
            E = ENERGY_START  # refill to 1.0 on real food cell
        energies.append(E)

        # Track post-move visits to old food cell (cell 0)
        if step >= FOOD_MOVE_STEP and agent.true_pos == FOOD_PRE:
            oldcell_visits_postmove += 1

        if E <= 0:
            dead = True
            death_step = step + 1
            death_pos = agent.true_pos
            break

    survived = not dead
    steps_lived = death_step if dead else BUDGET
    mean_E = float(np.mean(energies)) if energies else ENERGY_START

    # Believed-food set at end/death
    A_hat_end = agent._A_hat()
    believed_food_end = [c for c in range(N_CELLS)
                         if int(np.argmax(A_hat_end[:, c])) == 2]

    return {
        "survived": survived,
        "steps": steps_lived,
        "dead": dead,
        "death_pos": death_pos,
        "mean_E": mean_E,
        "oldcell_visits_postmove": oldcell_visits_postmove,
        "believed_food_end": believed_food_end,
        "believed_food_settle": believed_food_settle,
        "seed": seed,
        # Time below threshold (for machinery engagement report)
        "steps_below_thresh": sum(1 for e in energies if e < ENERGY_THRESH),
    }


ARMS = [
    ("IN-DECAY",  True,  True,  0),
    ("IN-FROZEN", True,  False, 1),
    ("NO-DECAY",  False, True,  2),
    ("NO-FROZEN", False, False, 3),
]

print("Exp 62 — interoceptive stake v2: scarcity makes it binding (rung 6, attempt 2)")
print(f"Budget={BUDGET} settle={SETTLE} seeds={SEEDS} energy_thresh={ENERGY_THRESH} "
      f"energy_decay={ENERGY_DECAY} food_pre=cell{FOOD_PRE} food_post=cell{FOOD_POST} "
      f"food_move_step={FOOD_MOVE_STEP}")
print()

arm_results: dict[str, list[dict]] = {}
for arm_name, use_intero, use_decay, arm_idx in ARMS:
    results = []
    for s in SEEDS:
        r = run_agent(arm_name, use_intero, use_decay, s, arm_idx)
        results.append(r)
        pos_str = str(r["death_pos"]) if r["dead"] else "-"
        print(
            f"arm={arm_name} seed={s} survived={r['survived']} "
            f"steps={r['steps']} death_pos={pos_str} mean_E={r['mean_E']:.3f} "
            f"oldcell_visits_postmove={r['oldcell_visits_postmove']} "
            f"believed_food_end={r['believed_food_end']}"
        )
    arm_results[arm_name] = results
    survived_k = sum(1 for r in results if r["survived"])
    step_list = [r["steps"] for r in results]
    median_steps = float(np.median(step_list))
    print(f"  -> {arm_name}: survived {survived_k}/5, median steps {median_steps:.0f}")
    print()

print("--- SUMMARY ---")

# (i) IN-DECAY survived >= 4/5
in_decay_surv = sum(1 for r in arm_results["IN-DECAY"] if r["survived"])
pass_i = in_decay_surv >= 4
print(f"(i) IN-DECAY survived {in_decay_surv}/5  {'PASS' if pass_i else 'FAIL'}")

# (ii) IN-FROZEN died post-move (step>1000) in >= 4/5 AND mean oldcell_visits >= 2
in_frozen_res = arm_results["IN-FROZEN"]
post_move_deaths = [r for r in in_frozen_res if r["dead"] and r["steps"] > FOOD_MOVE_STEP]
mean_oldcell = float(np.mean([r["oldcell_visits_postmove"] for r in in_frozen_res]))
pass_ii = len(post_move_deaths) >= 4 and mean_oldcell >= 2.0
print(
    f"(ii) IN-FROZEN post-move deaths (step>{FOOD_MOVE_STEP}): {len(post_move_deaths)}/5  "
    f"mean_oldcell_visits={mean_oldcell:.2f}  "
    f"({'PASS' if pass_ii else 'FAIL'})"
)

# (iii) median(intero arms combined) > median(no-intero arms combined)
intero_steps = ([r["steps"] for r in arm_results["IN-DECAY"]] +
                [r["steps"] for r in arm_results["IN-FROZEN"]])
nointero_steps = ([r["steps"] for r in arm_results["NO-DECAY"]] +
                  [r["steps"] for r in arm_results["NO-FROZEN"]])
intero_med = float(np.median(intero_steps))
nointero_med = float(np.median(nointero_steps))
pass_iii = intero_med > nointero_med
print(
    f"(iii) intero combined median={intero_med:.0f} > "
    f"no-intero combined median={nointero_med:.0f}  "
    f"{'PASS' if pass_iii else 'FAIL'}"
)

# Machinery engagement: mean time-below-threshold for intero arms
intero_below = float(np.mean(
    [r["steps_below_thresh"] for r in arm_results["IN-DECAY"]] +
    [r["steps_below_thresh"] for r in arm_results["IN-FROZEN"]]
))
print(f"  machinery engagement: mean steps_below_thresh (intero arms) = {intero_below:.1f}")

print()
if pass_i and pass_ii and pass_iii:
    verdict = ("VERDICT: stake BINDING: law's survival consequence + "
               "interoceptive advantage confirmed")
elif not pass_iii:
    verdict = (
        f"VERDICT: rung 6 honest FAIL: interoception adds nothing even under scarcity "
        f"(intero_med={intero_med:.0f} <= nointero_med={nointero_med:.0f}; "
        f"mean_steps_below_thresh={intero_below:.1f} — machinery was engaged)"
    )
else:
    failed = []
    if not pass_i:
        failed.append("(i) IN-DECAY survival")
    if not pass_ii:
        failed.append("(ii) IN-FROZEN post-move starvation + oldcell signature")
    verdict = f"VERDICT: MIXED: {failed}"
print(verdict)
