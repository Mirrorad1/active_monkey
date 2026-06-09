"""Exp 47 — accumulating values: fork-twins of one life diverge by comfort history.

Hypothesis: divergent comfort histories produce divergent self-formed favorites in
fork-twins of the same creature -- Exp 26's history->opinion result, now expressed as
counterfactual twins of mirro's one accumulated life (age 1300).
Design: 3 fork-pairs from mirro's committed snapshot. Within pair i, both twins live
1200 steps with the SAME action seed (identical movement trajectories) but in
differently-recolored 3x3 worlds: world-X cmap=[0,0,0,0,0,1,2,1,2] (color 0 dominant),
world-Y cmap=[2,2,2,2,2,1,0,1,0] (color 2 dominant). The only difference within a pair
is the color experience -- the comfort history.
Prediction if TRUE: in each pair, fork-X favorite=0 and fork-Y favorite=2; >= 2 of 3
pairs diverge (card threshold; directional expectation 3/3).
Falsifier: 0/3 pairs diverge = FAIL; 1/3 = below threshold (logged as negative-leaning).
Controls: same fork source (hash recorded), same within-pair action seed; mirro itself
is NEVER mutated or saved by this script (hash checked before/after). Forks are
disposable and never saved.
Seeds: pair i uses live seed 100+i (i=1,2,3). All reported.
"""

import numpy as np
from active_loop.creature import Creature, World

STATE = "creature/state/mirro"

# --- Step 1: Load mirro, record fork-source hash ---
mirro = Creature.load(STATE)
hash_before = mirro._state_hash()
print(f"fork-source: name={mirro.name} age={mirro.age_steps} hash={hash_before[:16]}")

# --- Step 2: Build world_X and world_Y ---
world_X = World(rows=3, cols=3, cmap=[0,0,0,0,0,1,2,1,2], n_colors=3)
world_Y = World(rows=3, cols=3, cmap=[2,2,2,2,2,1,0,1,0], n_colors=3)

# --- Note on fork() state_dir ---
# Inspected source: fork() sets twin._state_dir = None already.
# Confirmed: forks are unbound; live() will not write to mirro's state dir.
# NOTE: fork() does call self._bio_append on mirro (the parent) per source line ~539-544,
# which appends to mirro's BIOGRAPHY.jsonl. This is the normal fork-event log; it does
# NOT modify mirro's state arrays (pA, qs, value_counts). Hash is preserved.
print("fork() note: twin._state_dir set to None by source; live() on forks will not write to creature/state/mirro/")

# --- Step 3: 3 fork-pairs ---
results = []
for i in range(1, 4):
    seed = 100 + i

    fx = mirro.fork(f"fork_x{i}")
    fx.world = world_X
    fy = mirro.fork(f"fork_y{i}")
    fy.world = world_Y

    # Confirm unbound (should already be None from fork())
    if fx._state_dir is not None:
        fx._state_dir = None
        print(f"  pair {i}: WARNING fx had bound state_dir — unbound manually")
    if fy._state_dir is not None:
        fy._state_dir = None
        print(f"  pair {i}: WARNING fy had bound state_dir — unbound manually")

    fx.live(1200, seed=seed)
    fy.live(1200, seed=seed)

    fav_x = fx.favorite()
    fav_y = fy.favorite()
    conv_x = fx.conviction()
    conv_y = fy.conviction()
    diverged = fav_x != fav_y
    results.append((fav_x, fav_y, diverged))

    vc_x = fx.value_counts / (fx.value_counts.sum() + 1e-12) * 100
    vc_y = fy.value_counts / (fy.value_counts.sum() + 1e-12) * 100

    print(f"pair {i}: X favorite={fav_x} conviction={conv_x:.3f} | "
          f"Y favorite={fav_y} conviction={conv_y:.3f} | diverged={diverged}")
    print(f"  X value_counts%: " + " ".join(f"c{j}={v:.1f}" for j, v in enumerate(vc_x)))
    print(f"  Y value_counts%: " + " ".join(f"c{j}={v:.1f}" for j, v in enumerate(vc_y)))

# --- Step 4: Re-load mirro and verify no mutation ---
mirro2 = Creature.load(STATE)
hash_after = mirro2._state_hash()
print(f"mirro_untouched={hash_after == hash_before}  (before={hash_before[:16]} after={hash_after[:16]})")

# --- Step 5: Summary ---
n_diverged = sum(1 for r in results if r[2])
n_directional = sum(1 for fav_x, fav_y, _ in results if fav_x == 0 and fav_y == 2)
print(f"\npairs_diverged={n_diverged}/3")
print(f"directional_match={n_directional}/3  (X fav==0 AND Y fav==2)")
if n_diverged >= 2:
    print(f"VERDICT: prediction CONFIRMED ({n_diverged}/3 diverged, threshold >=2)")
elif n_diverged == 0:
    print("VERDICT: falsifier HIT (0/3 diverged)")
else:
    print("VERDICT: below threshold (1/3)")
