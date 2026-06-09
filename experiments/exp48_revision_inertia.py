"""Exp 48 — revision with inertia in ONE life (episode 4; permanently mutates mirro).

Mirro at age 1300 holds a WEAK favorite (color 2, conviction 0.340, counts
393.0/376.2/397.0 -- near-tie). Inertia is only testable against an entrenched
opinion (as in Exp 40), so the episode has two predeclared phases of mirro's real life:
Phase 1 (entrenchment): world becomes green-rich (cmap [2,2,2,2,2,1,0,1,0]); live 1500
steps. Gate: conviction >= 0.42 AND counts gap c2-c0 >= 200. If the gate fails, STOP
and report -- phase 2 is not run.
Phase 2 (revision): world becomes red-rich (cmap [0,0,0,0,0,1,2,1,2]); live up to 2500
steps in 100-step checkpoints, recording favorite + conviction at each.
Predeclared properties: REVISION = favorite becomes 0 and holds for the final 3
checkpoints reached; INERTIA = >= 3 checkpoints after the world change still favoring 2.
Falsifiers: favorite already 0 at checkpoint 1 (no inertia), or no revision within 2500
steps. Prediction: revision with ~8-13 checkpoints of holdout (rate estimate from the
value-accrual mechanism: gap ~(5/9-2/9)*w*1500, flip rate ~0.3/step).
Correction (cites Exp 47 entry): Exp 47's line 'mirro's accumulated counts leaned
color 0' was wrong -- the loaded age-1300 favorite is 2 (near-tie). The 3/3 divergence
claim is unaffected (it compares X-twins vs Y-twins, not inheritance).
Deterministic continuation of mirro's committed rng state; no explicit seeds.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from active_loop.creature import Creature, World

MIRRO_DIR = Path("creature/state/mirro")

# --- Load mirro ---
c = Creature.load(MIRRO_DIR)
print(f"BEFORE: age={c.age_steps} fav={c.favorite()} conv={c.conviction():.4f} "
      f"counts={np.round(c.value_counts, 1)} hash={c._state_hash()[:16]}")

# --- Build worlds ---
world_green = World(rows=3, cols=3, cmap=[2,2,2,2,2,1,0,1,0], n_colors=3)
world_red   = World(rows=3, cols=3, cmap=[0,0,0,0,0,1,2,1,2], n_colors=3)

# --- Phase 1: entrenchment ---
c._bio_append({"event": "world_change", "age_steps": c.age_steps,
               "summary": "world -> green-rich (entrenchment phase, Exp 48)"})
c.world = world_green

for k in range(1, 16):
    c.live(100)
    print(f"P1 ck{k}: age={c.age_steps} fav={c.favorite()} "
          f"conv={c.conviction():.4f} counts={np.round(c.value_counts, 1)}")

# Gate check
conv_ok  = c.conviction() >= 0.42
gap      = float(c.value_counts[2] - c.value_counts[0])
gap_ok   = gap >= 200
gate_pass = conv_ok and gap_ok
print(f"entrenchment_gate={'PASS' if gate_pass else 'FAIL'} "
      f"(conv={c.conviction():.4f}, gap={gap:.1f})")

if not gate_pass:
    c.save(MIRRO_DIR)
    print("VERDICT: gate FAILED — phase 2 not run (episode logged honestly)")
    sys.exit(0)

# --- Phase 2: revision ---
c._bio_append({"event": "world_change", "age_steps": c.age_steps,
               "summary": "world -> red-rich (revision phase, Exp 48)"})
c.world = world_red

holdout_checkpoints = 0          # leading checkpoints still fav==2
counted_holdout = False           # once fav leaves 2, stop counting holdout
consec_zero = 0                   # consecutive checkpoints with fav==0
total_p2_ck = 0
revised = False
transition_ck = []                # checkpoints with fav neither 0 nor 2

for k in range(1, 26):
    c.live(100)
    total_p2_ck = k
    fav = c.favorite()
    conv = c.conviction()
    gap_c0_c2 = float(c.value_counts[0] - c.value_counts[2])
    print(f"P2 ck{k}: fav={fav} conv={conv:.4f} gap_c0_minus_c2={gap_c0_c2:.1f}")

    if not counted_holdout:
        if fav == 2:
            holdout_checkpoints += 1
        else:
            counted_holdout = True  # fav has moved away from 2

    if fav == 0:
        consec_zero += 1
    else:
        consec_zero = 0

    if fav not in (0, 2):
        transition_ck.append(k)

    if consec_zero >= 3:
        revised = True
        break

# --- Save mirro ---
c.save(MIRRO_DIR)
print(f"AFTER:  age={c.age_steps} fav={c.favorite()} conv={c.conviction():.4f} "
      f"counts={np.round(c.value_counts, 1)} hash={c._state_hash()[:16]}")

# --- Summary ---
print(f"revised={revised}")
print(f"holdout_checkpoints={holdout_checkpoints}")
if transition_ck:
    print(f"transition_checkpoints (fav neither 0 nor 2)={transition_ck}")

if revised and holdout_checkpoints >= 3:
    print(f"VERDICT: prediction CONFIRMED (revision with inertia, holdout={holdout_checkpoints})")
elif revised and holdout_checkpoints == 0:
    print("VERDICT: falsifier HIT (instant revision, holdout=0)")
elif not revised:
    print("VERDICT: falsifier HIT (no revision in 2500 steps)")
else:
    print(f"VERDICT: MIXED (revised with weak inertia, holdout={holdout_checkpoints})")
