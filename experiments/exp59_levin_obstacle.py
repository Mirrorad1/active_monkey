"""Exp 59 — rung 4: Levin obstacle transplant (error tolerance + delayed gratification).

PROVIDED (declared): the whole navigation harness -- value-iteration planning (Exp 30
pattern) over the 5x5 grid with a task-assigned goal (reward +1 at cell 0), the innate
movement model, softmax action selection (tau=0.3), and GENERIC failure-learning: a
move that fails (locked target) marks that single transition blocked in the planner's
model -- ordinary world-model updating, not a lock-specific handler. The subject is a
disposable fork of mirro (age 10700); mirro untouched.
Geometry: start cell 2 (0,2); goal cell 0 (0,0); LOCKED cell 1 (0,1) -- a corner pocket
whose only detour is strictly longer (4 steps vs 2), so 'against-gradient' is
well-defined: a step that INCREASES believed-unlocked geodesic distance to the goal.
Conditions, 5 softmax seeds each (101..105):
  A) VI agent: full value iteration on the current (failure-updated) model, replan
     each step.
  B) GREEDY baseline (the predeclared Levin-critics control): 1-step lookahead
     (immediate believed distance), same softmax, same failure-learning.
Predeclared properties:
  (a) error tolerance: A reaches the goal in >= 4/5 seeds within 60 steps.
  (b) delayed gratification: >= 4/5 of A's successful runs contain >= 1
      against-gradient step.
  (c) horizon gap: mean steps(B) >= 1.5 x mean steps(A); without the gap the DG label
      is REFUSED (jitter, not horizon).
Falsifiers: (a) fails; (b) fails (pure greedy descent suffices); (c) fails (no gap).
All three verdict combinations logged honestly.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# Setup: load mirro, unbind, fork
# ---------------------------------------------------------------------------
MIRRO_DIR = "creature/state/mirro"
mirro = Creature.load(MIRRO_DIR)
mirro._state_dir = None  # unbind immediately — mirro must not be mutated

subject = mirro.fork("levin_subject")
# Navigation state is position only; start position declared as cell 2
START = 2
GOAL = 0
LOCKED = 1
BUDGET = 60
TAU = 0.3
SEEDS = list(range(101, 106))

# World for movement (5x5, any cmap)
world = World(rows=5, cols=5, cmap=list(range(25)), n_colors=25)
N = world.n_cells  # 25


# ---------------------------------------------------------------------------
# BFS geodesic distance on the ORIGINAL grid (no knowledge of lock)
# Used for "believed-unlocked" distance label only
# ---------------------------------------------------------------------------
def bfs_distances(goal: int) -> list[int]:
    """BFS distances from all cells to goal on the plain grid (no obstacles)."""
    dist = [-1] * N
    dist[goal] = 0
    q = deque([goal])
    while q:
        cell = q.popleft()
        for a in range(4):
            nb = world.move(cell, a)
            if dist[nb] == -1:
                dist[nb] = dist[cell] + 1
                q.append(nb)
    return dist


UNLOCKED_DIST = bfs_distances(GOAL)  # distance ignoring lock


# ---------------------------------------------------------------------------
# Value iteration on the current model (blocked dict)
# ---------------------------------------------------------------------------
def run_vi(blocked: dict[tuple[int, int], bool], gamma: float = 0.9, sweeps: int = 100) -> np.ndarray:
    """Returns Q[cell, action] using VI on the failure-updated transition model."""
    # Build transition under blocked model
    # T[cell, a] = next cell (stay if blocked)
    T = np.zeros((N, 4), dtype=int)
    for cell in range(N):
        for a in range(4):
            if blocked.get((cell, a), False):
                T[cell, a] = cell  # stay
            else:
                T[cell, a] = world.move(cell, a)

    R = np.zeros(N)
    R[GOAL] = 1.0

    V = np.zeros(N)
    for _ in range(sweeps):
        V_new = V.copy()
        for cell in range(N):
            if cell == GOAL:
                V_new[cell] = 1.0
                continue
            best = -1e9
            for a in range(4):
                nxt = T[cell, a]
                val = R[nxt] + gamma * V[nxt]
                if val > best:
                    best = val
            V_new[cell] = best
        V = V_new

    Q = np.zeros((N, 4))
    for cell in range(N):
        for a in range(4):
            nxt = T[cell, a]
            Q[cell, a] = R[nxt] + gamma * V[nxt]
    return Q, T


def softmax_sample(q_row: np.ndarray, tau: float, rng: np.random.Generator) -> int:
    logits = q_row / tau
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# Run one trial
# ---------------------------------------------------------------------------
def run_trial(condition: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    blocked: dict[tuple[int, int], bool] = {}
    pos = START
    trajectory = [pos]
    against_gradient = 0
    reached = False

    for step in range(BUDGET):
        if pos == GOAL:
            reached = True
            break

        # Build Q for this step
        if condition == "A":
            Q, T_mat = run_vi(blocked)
            q_row = Q[pos]
        else:  # B: greedy 1-step lookahead
            q_row = np.zeros(4)
            for a in range(4):
                if blocked.get((pos, a), False):
                    nxt = pos
                else:
                    nxt = world.move(pos, a)
                q_row[a] = -UNLOCKED_DIST[nxt]

        action = softmax_sample(q_row, TAU, rng)

        # Execute action
        intended = world.move(pos, action) if not blocked.get((pos, action), False) else pos
        if intended == LOCKED:
            # Execution: stay, record failure
            actual_next = pos
            blocked[(pos, action)] = True
        else:
            actual_next = intended

        # Against-gradient: does this step increase believed-unlocked distance to goal?
        dist_before = UNLOCKED_DIST[pos]
        dist_after = UNLOCKED_DIST[actual_next]
        if dist_after > dist_before:
            against_gradient += 1

        pos = actual_next
        trajectory.append(pos)

    if pos == GOAL:
        reached = True

    return {
        "condition": condition,
        "seed": seed,
        "reached": reached,
        "steps": len(trajectory) - 1,
        "against_gradient_steps": against_gradient,
        "trajectory": trajectory,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print(f"Exp 59 — Levin obstacle transplant")
print(f"Subject: levin_subject (fork of mirro age {mirro.age_steps})")
print(f"Geometry: start={START}, goal={GOAL}, locked={LOCKED}")
print(f"Believed-unlocked distances: {UNLOCKED_DIST[:5]}  (cells 0-4)")
print()

results_A = []
results_B = []

for seed in SEEDS:
    for cond, store in [("A", results_A), ("B", results_B)]:
        r = run_trial(cond, seed)
        store.append(r)
        traj_str = "->".join(str(c) for c in r["trajectory"])
        print(
            f"  {cond} seed={seed}  reached={r['reached']}  "
            f"steps={r['steps']}  ag_steps={r['against_gradient_steps']}  "
            f"traj=[{traj_str}]"
        )

print()

# (a) error tolerance
a_reached = sum(1 for r in results_A if r["reached"])
pass_a = a_reached >= 4
print(f"(a) A reached {a_reached}/5 (within {BUDGET})  {'PASS' if pass_a else 'FAIL'}")

# (b) delayed gratification
a_successes = [r for r in results_A if r["reached"]]
a_with_ag = sum(1 for r in a_successes if r["against_gradient_steps"] >= 1)
pass_b = (len(a_successes) >= 4) and (a_with_ag >= 4)
print(f"(b) A successful runs with >=1 against-gradient step: {a_with_ag}/{len(a_successes)}  {'PASS' if pass_b else 'FAIL'}")

# (c) horizon gap
mean_A = np.mean([r["steps"] if not r["reached"] else r["steps"] for r in results_A])
# count unreached as 60
steps_A = [r["steps"] if r["reached"] else BUDGET for r in results_A]
steps_B = [r["steps"] if r["reached"] else BUDGET for r in results_B]
mean_A_adj = np.mean(steps_A)
mean_B_adj = np.mean(steps_B)
ratio = mean_B_adj / mean_A_adj if mean_A_adj > 0 else float("inf")
pass_c = ratio >= 1.5
print(f"(c) mean_steps: A={mean_A_adj:.2f} B={mean_B_adj:.2f} ratio={ratio:.3f}  {'PASS' if pass_c else 'FAIL'}")

print()

# Verdict
if pass_a and pass_b and pass_c:
    verdict = (
        f"DG DEMONSTRATED (error-tolerant routing with horizon-dependent against-gradient steps) "
        f"[a={a_reached}/5, b={a_with_ag}/{len(a_successes)}, ratio={ratio:.3f}]"
    )
elif pass_a and pass_b and not pass_c:
    verdict = (
        f"ROUTING WITHOUT HORIZON (DG label refused: no gap vs greedy) "
        f"[a={a_reached}/5, b={a_with_ag}/{len(a_successes)}, ratio={ratio:.3f}]"
    )
elif pass_a and not pass_b:
    verdict = (
        f"PURE GREEDY SUFFICES (no against-gradient needed) "
        f"[a={a_reached}/5, b={a_with_ag}/{len(a_successes)}, ratio={ratio:.3f}]"
    )
else:
    verdict = (
        f"ERROR TOLERANCE FAILED "
        f"[a={a_reached}/5, b={a_with_ag}/{len(a_successes)}, ratio={ratio:.3f}]"
    )

print(f"VERDICT: {verdict}")
