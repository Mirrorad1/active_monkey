"""Exp 61 — rung 6: the interoceptive stake, as a survival test of the plasticity law.

Toy allostasis (ALL machinery provided, declared): energy E starts 1.0, decays
0.01/step (100-step autonomy), refills to 1.0 on a REAL food cell (the 3x3 color-2
patch). Policy switch (provided coupling): E < 0.4 -> value-iterate toward the nearest
BELIEVED food cell (argmax of the agent's own learned A_hat = color 2); E >= 0.4 ->
random-walk explore. The world drifts: patch moves every 500 steps (corner cycle);
budget 2000 steps. Death = E <= 0.
2x2 arms x 5 seeds (births seeded 30+s, settling 1000 steps in the segment-0 world so
every agent starts knowing the initial food location):
  IN-DECAY   interoception ON,  map learning with in-window decay (pA*=0.997/step,
             floor 0.01)        -- predicted: survives full budget >= 4/5.
  IN-FROZEN  interoception ON,  non-decaying map -- predicted: dies after the first
             patch move >= 4/5 (starves seeking REMEMBERED food: deaths cluster at
             step > 500 with the agent at/near the OLD patch).
  NO-DECAY   interoception OFF (always explore), decay map    -- forage by luck.
  NO-FROZEN  interoception OFF, non-decay map                 -- forage by luck.
  Predicted: both NO arms die earlier than IN-DECAY (the channel buys survival).
Falsifiers: IN-FROZEN survives >= 3/5 (the law's consequence fails to bite in a closed
loop); or IN-DECAY <= the NO arms (interoception adds nothing -- rung 6 FAIL branch).
All outcomes logged honestly. The self-formed component is each agent's learned map
(and its staleness); everything else is provided harness.
"""
from __future__ import annotations

import numpy as np
from collections import deque

from active_loop.creature import Creature, World

ROWS, COLS = 5, 5
N_CELLS = ROWS * COLS
N_COLORS = 3
CORNER_CYCLE = [(0, 0), (0, 2), (2, 2), (2, 0)]
BUDGET = 2000
SETTLE = 1000
SEEDS = list(range(5))          # s = 0..4 -> birth seed 30+s
DECAY_RATE = 0.997
DECAY_FLOOR = 0.01
ENERGY_START = 1.0
ENERGY_DECAY = 0.01
ENERGY_THRESH = 0.4
GAMMA = 0.9


def build_cmap(corner):
    cmap = [(i % 2) for i in range(N_CELLS)]
    r0, c0 = corner
    for dr in range(3):
        for dc in range(3):
            r, c = r0 + dr, c0 + dc
            cmap[r * COLS + c] = 2
    return cmap


def seg_world(k):
    corner = CORNER_CYCLE[k % 4]
    return World(rows=ROWS, cols=COLS, cmap=build_cmap(corner), n_colors=N_COLORS)


def patch_cells(k):
    """Return set of real food cells for segment k."""
    r0, c0 = CORNER_CYCLE[k % 4]
    return {(r0 + dr) * COLS + (c0 + dc) for dr in range(3) for dc in range(3)}


def patch_center(k):
    """Return center cell of patch for segment k."""
    r0, c0 = CORNER_CYCLE[k % 4]
    return (r0 + 1) * COLS + (c0 + 1)


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


def run_agent(arm_name: str, use_intero: bool, use_decay: bool, seed: int) -> dict:
    birth_seed = 30 + seed
    w0 = seg_world(0)
    agent = Creature.birth(arm_name, w0, seed=birth_seed)

    # Settle phase: 1000 steps random walk in segment-0 world (learn initial food location)
    B0 = w0.transition_matrix()
    rng_settle = np.random.default_rng(birth_seed * 1_000_003 + 0)
    for i in range(SETTLE):
        A_hat = agent._A_hat()
        obs = int(agent.world.cmap[agent.true_pos])
        likelihood = A_hat[obs, :]
        qs_up = likelihood * agent.qs
        denom = qs_up.sum()
        qs_up = qs_up / denom if denom > 0 else np.ones(N_CELLS) / N_CELLS
        agent.pA[obs, :] += qs_up
        if use_decay:
            agent.pA = np.maximum(agent.pA * DECAY_RATE, DECAY_FLOOR)
        action = int(rng_settle.integers(0, 4))
        agent.true_pos = agent.world.move(agent.true_pos, action)
        agent.qs = B0[:, :, action] @ qs_up

    # Survival phase
    E = ENERGY_START
    energies = []
    dead = False
    death_step = None
    death_pos = None
    last_switch_seg = 0

    for step in range(BUDGET):
        # Determine active segment (switches at 500, 1000, 1500)
        seg = step // 500
        if seg != last_switch_seg:
            agent.world = seg_world(seg)
            last_switch_seg = seg
        B = agent.world.transition_matrix()

        # Observe + learn
        A_hat = agent._A_hat()
        obs = int(agent.world.cmap[agent.true_pos])
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
            # Find believed food cells from learned A_hat
            A_hat_curr = agent._A_hat()
            believed_food = [c for c in range(N_CELLS)
                             if int(np.argmax(A_hat_curr[:, c])) == 2]
            if believed_food:
                Q = vi_toward_food(believed_food, agent.world)
                q_row = Q[agent.true_pos]
                max_q = q_row.max()
                best_actions = np.where(q_row == max_q)[0]
                action = int(rng_act.choice(best_actions))
            else:
                action = int(rng_act.integers(0, 4))
        else:
            action = int(rng_act.integers(0, 4))

        # Move
        agent.true_pos = agent.world.move(agent.true_pos, action)

        # Advance belief
        agent.qs = B[:, :, action] @ qs_up

        # Energy update
        E -= ENERGY_DECAY
        if agent.world.cmap[agent.true_pos] == 2:
            E = ENERGY_START
        energies.append(E)

        if E <= 0:
            dead = True
            death_step = step + 1  # 1-indexed
            death_pos = agent.true_pos
            break

    survived = not dead
    steps_lived = death_step if dead else BUDGET
    mean_E = float(np.mean(energies)) if energies else ENERGY_START

    # Distance to old patch center at death (the patch active before the most recent switch)
    dist_to_old = None
    if dead and death_step is not None:
        old_seg = max(0, (death_step - 1) // 500 - 1)
        old_center = patch_center(old_seg)
        dr = abs(death_pos // COLS - old_center // COLS)
        dc = abs(death_pos % COLS - old_center % COLS)
        dist_to_old = dr + dc

    return {
        "survived": survived,
        "steps": steps_lived,
        "dead": dead,
        "death_pos": death_pos,
        "dist_to_old": dist_to_old,
        "mean_E": mean_E,
        "seed": seed,
    }


ARMS = [
    ("IN-DECAY",  True,  True),
    ("IN-FROZEN", True,  False),
    ("NO-DECAY",  False, True),
    ("NO-FROZEN", False, False),
]

print("Exp 61 — interoceptive stake (rung 6 survival test)")
print(f"Budget={BUDGET} settle={SETTLE} seeds={SEEDS} energy_thresh={ENERGY_THRESH}")
print()

arm_results = {}
for arm_name, use_intero, use_decay in ARMS:
    results = []
    for s in SEEDS:
        r = run_agent(arm_name, use_intero, use_decay, s)
        results.append(r)
        pos_str = str(r["death_pos"]) if r["dead"] else "-"
        print(
            f"arm={arm_name} seed={s} survived={r['survived']} "
            f"steps={r['steps']} death_pos={pos_str} mean_E={r['mean_E']:.2f}"
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

# (ii) IN-FROZEN died after step 500 in >= 4/5 (post-move starvation)
in_frozen_res = arm_results["IN-FROZEN"]
post_move_deaths = [r for r in in_frozen_res if r["dead"] and r["steps"] > 500]
pass_ii = len(post_move_deaths) >= 4
dists = [r["dist_to_old"] for r in post_move_deaths if r["dist_to_old"] is not None]
mean_dist = float(np.mean(dists)) if dists else float("nan")
in_frozen_died = sum(1 for r in in_frozen_res if r["dead"])
print(
    f"(ii) IN-FROZEN post-move deaths (step>500): {len(post_move_deaths)}/5  "
    f"({'PASS' if pass_ii else 'FAIL'})  "
    f"mean_dist_to_old_patch={mean_dist:.2f} "
    f"[total died={in_frozen_died}/5]"
)

# (iii) IN-DECAY median survival > both NO arms
in_decay_med = float(np.median([r["steps"] for r in arm_results["IN-DECAY"]]))
no_decay_med = float(np.median([r["steps"] for r in arm_results["NO-DECAY"]]))
no_frozen_med = float(np.median([r["steps"] for r in arm_results["NO-FROZEN"]]))
pass_iii = in_decay_med > no_decay_med and in_decay_med > no_frozen_med
print(
    f"(iii) IN-DECAY median={in_decay_med:.0f} > NO-DECAY median={no_decay_med:.0f} "
    f"AND > NO-FROZEN median={no_frozen_med:.0f}  {'PASS' if pass_iii else 'FAIL'}"
)

print()
if pass_i and pass_ii and pass_iii:
    verdict = "law's survival consequence + interoceptive advantage CONFIRMED"
else:
    failed = []
    if not pass_i:
        failed.append("(i) IN-DECAY survival")
    if not pass_ii:
        failed.append("(ii) IN-FROZEN post-move starvation")
    if not pass_iii:
        failed.append("(iii) IN-DECAY median > NO arms")
    verdict = f"falsifier HIT: {failed}"
print(f"VERDICT: {verdict}")
