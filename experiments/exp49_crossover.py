"""Exp 49 — episode 4b: the crossover. Mirro's red-world life continues to the flip.

Exp 48 ended with the predeclared budget exhausted: favorite still 2 after 2500
red-world steps, value gap c0-c2 eroded -420.8 -> -53.2 (linear ~16/checkpoint, with
one late reversal at ck25). Hypothesis: the linear erosion holds and mirro revises
within 2500 further steps. Prediction: favorite flips to 0 and holds 3 consecutive
100-step checkpoints, crossover within the first ~9 checkpoints of this continuation;
afterwards conviction-for-0 begins rebuilding (secondary, reported not gated).
Falsifiers: (a) no flip within 25 checkpoints (5000 cumulative opposing-evidence steps
-- freeze stronger than linear); (b) the gap re-widens toward color 2 (mechanism quirk
flagged by Exp 48's ck25 uptick) -- logged as such if seen.
Deterministic continuation of mirro's committed rng state; no explicit seeds.
World is NOT changed by this script (it persisted red-rich from Exp 48).
"""
import sys
from active_loop.creature import Creature

# ---------------------------------------------------------------------------
# Load and verify initial state
# ---------------------------------------------------------------------------
c = Creature.load("creature/state/mirro")

vc = c.value_counts
gap_before = vc[0] - vc[2]
cmap = list(c.world.cmap)

print("BEFORE:")
print(f"  age={c.age_steps}")
print(f"  favorite={c.favorite()}")
print(f"  conviction={c.conviction():.4f}")
print(f"  value_counts={[round(v, 1) for v in vc]}")
print(f"  gap_c0_minus_c2={gap_before:.1f}")
print(f"  hash={c._state_hash()[:16]}")
print(f"  world_cmap={cmap}")

if cmap != [0, 0, 0, 0, 0, 1, 2, 1, 2]:
    print("WORLD PERSISTENCE BUG")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Bio entry
# ---------------------------------------------------------------------------
c._bio_append({
    "event": "episode",
    "age_steps": c.age_steps,
    "summary": "Exp 49: continue red-world life to the crossover (episode 4b)",
})

# ---------------------------------------------------------------------------
# Checkpoint loop: up to 25 checkpoints; stop 5 after 3-consecutive fav==0
# ---------------------------------------------------------------------------
crossover_ck = None
consecutive_fav0 = 0
post_flip_remaining = 0
done = False

prev_gap = gap_before
gap_rewidened = False

for k in range(1, 26):
    c.live(100)
    vc = c.value_counts
    gap = vc[0] - vc[2]
    fav = c.favorite()
    conv = c.conviction()

    print(f"ck{k}: age={c.age_steps} fav={fav} conv={conv:.4f} gap_c0_minus_c2={gap:.1f}")

    # Track gap re-widening (gap moves more negative by > 10)
    if gap - prev_gap < -10:
        gap_rewidened = True
    prev_gap = gap

    # Track crossover
    if fav == 0:
        if crossover_ck is None:
            crossover_ck = k
        consecutive_fav0 += 1
    else:
        consecutive_fav0 = 0

    # Once 3 consecutive fav==0, run 5 more then stop
    if consecutive_fav0 >= 3 and not done:
        post_flip_remaining = 5
        done = True

    if done:
        post_flip_remaining -= 1
        if post_flip_remaining <= 0:
            break

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
c.save("creature/state/mirro")

vc_after = c.value_counts
gap_after = vc_after[0] - vc_after[2]
print()
print("AFTER:")
print(f"  age={c.age_steps}")
print(f"  favorite={c.favorite()}")
print(f"  conviction={c.conviction():.4f}")
print(f"  value_counts={[round(v, 1) for v in vc_after]}")
print(f"  gap_c0_minus_c2={gap_after:.1f}")
print(f"  hash={c._state_hash()[:16]}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
revised = crossover_ck is not None and done  # held 3 consecutive
print()
print(f"crossover_ck={crossover_ck}")
print(f"revised={revised}")
print(f"gap_rewidened={gap_rewidened}")

if revised and crossover_ck <= 9:
    verdict = f"VERDICT: prediction CONFIRMED (crossover at ck {crossover_ck})"
elif revised:
    verdict = f"VERDICT: MIXED (revised late, crossover at ck {crossover_ck})"
else:
    verdict = "VERDICT: falsifier HIT (no revision in 5000 cumulative steps)"

if gap_rewidened:
    verdict += " + gap re-widening observed"

print(verdict)
