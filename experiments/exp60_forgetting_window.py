"""Exp 60 — cascade round 3 (human-authorized option a): the corrected forgetting
counterfactual, bracketing the plasticity window.

The mass-vs-tempo law predicts adaptation requires accumulated evidence mass in a
WINDOW: above an accuracy floor (enough counts to hold a sharp map), below the tempo
ceiling (~20 = drift period x visit rate). Exp 58's decay arm (lambda=0.9/step, mass~1)
sat BELOW the floor -- uninformative. Two corrected arms bracket the window:
  ARM-IN  (lambda=0.997/step, per-visit retention ~0.93, steady-state mass ~14, inside
           the window): the law's signature prediction -- it TRACKS the drift,
           recovery >= 6/8 segments.
  ARM-HIGH (lambda=0.9999/step, mass ~hundreds, above the ceiling): decay present but
           too weak -- FROZEN, recovery <= 2/8.
With Exp 58's lambda=0.9 as the below-floor reference, three lambda values bracket the
window. Predeclared verdict: BOTH arms correct -> the NOVELTY-CANDIDATE DIES as the
lawful consequence of non-decaying counts; the unified plasticity-window law is logged
(consolidating Exp 48/56/57/58): lifelong adaptation requires forgetting, with mass
held between accuracy floor and tempo ceiling. ARM-IN failing to track -> the candidate
SURVIVES with the mechanism unexplained; the thread halts for explicit human input.
Subjects: fresh births, settling 1000 steps with decay active (seeds 95/96, per-step
action seed bases 195/196). Decay applied as pA *= lambda after every live(1), floored
at 0.01.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from active_loop.creature import Creature, World

ROWS, COLS = 5, 5
N_COLORS = 3
N_CELLS = ROWS * COLS
CORNER_CYCLE = [(0, 0), (0, 2), (2, 2), (2, 0)]
N_SEGMENTS = 8
SEG_STEPS = 500
CHECK_EVERY = 25
RECOVERY_THRESH = 0.92
LAG_MAX = 450


def build_cmap(corner):
    """5x5 cmap: background = i%2; 3x3 patch at corner = color 2."""
    cmap = [(i % 2) for i in range(N_CELLS)]
    r0, c0 = corner
    for dr in range(3):
        for dc in range(3):
            r, c = r0 + dr, c0 + dc
            cmap[r * COLS + c] = 2
    return cmap


seg_cmaps = [build_cmap(CORNER_CYCLE[k % 4]) for k in range(N_SEGMENTS)]


def run_arm(name, lam, seed, base):
    print("=" * 60)
    print(f"ARM: {name}  lambda={lam}  birth_seed={seed}  base={base}")
    print("=" * 60)

    # Birth in segment-0 world
    world0 = World(rows=ROWS, cols=COLS, cmap=seg_cmaps[0], n_colors=N_COLORS)
    c = Creature.birth(name, world0, seed=seed)

    # Settle 1000 steps with decay active
    settle_world = World(rows=ROWS, cols=COLS, cmap=seg_cmaps[0], n_colors=N_COLORS)
    c.world = settle_world
    for i in range(1000):
        c.live(1, seed=base * 1000003 + i)
        c.pA *= lam
        c.pA = np.maximum(c.pA, 0.01)

    post_acc = c.map_accuracy()
    mean_mass = float(np.mean(c.pA))
    print(f"{name} post-settling: map_accuracy={post_acc:.4f}  mean_pA_mass={mean_mass:.4f}")

    # 8-segment drift schedule
    n_recovered = 0
    for k in range(N_SEGMENTS):
        world_k = World(rows=ROWS, cols=COLS, cmap=seg_cmaps[k], n_colors=N_COLORS)
        c.world = world_k
        lag = None
        for step_i in range(SEG_STEPS):
            global_idx = 1000 + k * SEG_STEPS + step_i
            c.live(1, seed=base * 1000003 + global_idx)
            c.pA *= lam
            c.pA = np.maximum(c.pA, 0.01)
            if (step_i + 1) % CHECK_EVERY == 0:
                acc = c.map_accuracy()
                if lag is None and acc >= RECOVERY_THRESH:
                    lag = step_i + 1
        recovered = lag is not None and lag <= LAG_MAX
        end_acc = c.map_accuracy()
        lag_disp = lag if lag is not None else ">500"
        print(f"{name} seg{k} lag={lag_disp} end_acc={end_acc:.4f}")
        if recovered:
            n_recovered += 1

    print(f"{name} recovered {n_recovered}/8")
    return n_recovered


# Run both arms
n_in = run_arm("ARM-IN", 0.997, 95, 195)
print()
n_high = run_arm("ARM-HIGH", 0.9999, 96, 196)

# Summary
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
pass_in = n_in >= 6
pass_high = n_high <= 2
print(f"ARM-IN  recovered {n_in}/8  (predicted >= 6) -> {'PASS' if pass_in else 'FAIL'}")
print(f"ARM-HIGH recovered {n_high}/8  (predicted <= 2) -> {'PASS' if pass_high else 'FAIL'}")
print()
if pass_in and pass_high:
    print(
        "VERDICT: candidate DIES — plasticity-window law confirmed "
        "(in-window decay tracks; weak decay freezes; Exp 58's strong decay sat below the floor)"
    )
elif not pass_in:
    print(
        "VERDICT: candidate SURVIVES (ARM-IN failed to track) — "
        "thread halts for human"
    )
else:
    print(
        "VERDICT: MIXED (ARM-HIGH unexpectedly tracked) — "
        "window ceiling wrong, report"
    )
