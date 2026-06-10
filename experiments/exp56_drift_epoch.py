"""Exp 56 — enriched-world epoch: a drifting comfort source (rung 3; mutates mirro).

ONE new declared mechanism (provided, like taught labels): a 3x3 patch of color 2 whose
top-left corner cycles [(0,0),(0,2),(2,2),(2,0)] around the 5x5 world, moving at each
500-step segment boundary; background cells carry a fixed 0/1 pattern. 8 segments =
4000 steps of mirro's real life (age 6700 -> 10700), biography recording every segment.
Drift-aware ledger (exp56_ledger.json, frozen pre-epoch + declared schedule):
  E1 map-vs-CURRENT-world recovers to >= 0.92 by each segment end; recovery lag
     (steps from segment start to first reaching 0.92, checked every 25 steps)
     within [0, 450].
  E2 occupancy entropy per segment in [4.40, 4.644].
  E3 color-2 value share rises by > +0.02 over the epoch (values track the source).
  E4 favorite == 2 at epoch end (moderate confidence; failure = value-lag finding).
Falsifiers: any segment fails E1 (adaptation does not track the drift); E3 fails (the
value system cannot follow a moving source); occupancy anomaly (E2). All expectations
met = the rung-3 REAL NEGATIVE: no novelty at this richness, adaptation as-designed --
logged as such, no post-hoc anomaly shopping.
Deterministic continuation of mirro's committed rng (no explicit seeds).
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from active_loop.creature import Creature, World

STATE_DIR = Path("creature/state/mirro")
CORNER_CYCLE = [(0, 0), (0, 2), (2, 2), (2, 0)]
ROWS, COLS = 5, 5
N_COLORS = 3


def build_cmap(corner):
    """5x5 cmap: background = i%2 (0/1); 3x3 patch at corner = color 2."""
    cmap = [(i % 2) for i in range(ROWS * COLS)]
    r0, c0 = corner
    for dr in range(3):
        for dc in range(3):
            r, c = r0 + dr, c0 + dc
            cmap[r * COLS + c] = 2
    return cmap


def occupancy_entropy(positions):
    """Shannon entropy of cell-visit counts, in bits."""
    counts = np.bincount(positions, minlength=ROWS * COLS).astype(float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return float(-np.sum(p * np.log2(p + 1e-300)))


def value_shares(c):
    total = c.value_counts.sum()
    if total == 0:
        return [0.0] * N_COLORS
    return [float(c.value_counts[i] / total) for i in range(N_COLORS)]


# --- Load mirro ---
c = Creature.load(STATE_DIR)
pre_shares = value_shares(c)
print(f"BEFORE: age={c.age_steps} favorite={c.favorite()} "
      f"shares=[{pre_shares[0]:.4f},{pre_shares[1]:.4f},{pre_shares[2]:.4f}] "
      f"hash16={c._state_hash()[:16]} cmap={list(c.world.cmap)}")

# Epoch-start biography event
c._bio_append({
    "event": "epoch_start",
    "age_steps": c.age_steps,
    "summary": "Exp 56: drifting comfort source epoch (8x500 steps, declared schedule)",
})

# --- Epoch: 8 segments x 500 steps ---
lags = []
ent_vals = []
seg_c2_shares = []

for seg in range(8):
    corner = CORNER_CYCLE[seg % 4]
    cmap = build_cmap(corner)
    c.world = World(rows=ROWS, cols=COLS, cmap=cmap, n_colors=N_COLORS)

    positions = []
    lag = None

    for step_i in range(500):
        c.live(1)
        positions.append(c.true_pos)
        # Check map accuracy every 25 steps (at step 24, 49, ...)
        if (step_i + 1) % 25 == 0:
            acc = c.map_accuracy()
            if lag is None and acc >= 0.92:
                lag = step_i + 1  # steps from segment start

    end_acc = c.map_accuracy()
    occ_ent = occupancy_entropy(positions)
    shares = value_shares(c)
    c2_share = shares[2]
    fav = c.favorite()

    lag_display = lag if lag is not None else ">500 FAIL"
    lags.append(lag)
    ent_vals.append(occ_ent)
    seg_c2_shares.append(c2_share)

    print(f"seg{seg} corner={corner} lag={lag_display} end_acc={end_acc:.4f} "
          f"occ_ent={occ_ent:.4f} c2_share={c2_share:.4f} fav={fav}")

    c._bio_append({
        "event": "segment_end",
        "age_steps": c.age_steps,
        "summary": f"Exp 56 seg{seg}: corner={corner} lag={lag_display} "
                   f"end_acc={end_acc:.4f} occ_ent={occ_ent:.4f} "
                   f"c2_share={c2_share:.4f} fav={fav}",
    })

# --- Save ---
c.save(STATE_DIR)
post_shares = value_shares(c)
print(f"AFTER:  age={c.age_steps} favorite={c.favorite()} "
      f"shares=[{post_shares[0]:.4f},{post_shares[1]:.4f},{post_shares[2]:.4f}] "
      f"hash16={c._state_hash()[:16]}")

# --- Summary ---
print()
# E1
e1_ok = [lag is not None and lag <= 450 for lag in lags]
e1_pass = all(e1_ok)
e1_n = sum(e1_ok)
print(f"E1 segment_recovery: {e1_n}/8 segments recovered (lags={lags}) "
      f"{'PASS' if e1_pass else 'FAIL'}")

# E2
e2_ok = [4.40 <= e <= 4.644 for e in ent_vals]
e2_n = sum(e2_ok)
e2_pass = e2_n == 8
print(f"E2 occupancy: {e2_n}/8 in band [{', '.join(f'{e:.4f}' for e in ent_vals)}] "
      f"{'PASS' if e2_pass else 'FAIL'}")

# E3
pre_c2 = pre_shares[2]
post_c2 = post_shares[2]
delta_c2 = post_c2 - pre_c2
e3_pass = delta_c2 > 0.02
print(f"E3 c2_share drift: pre={pre_c2:.4f} post={post_c2:.4f} delta={delta_c2:+.4f} "
      f"{'PASS' if e3_pass else 'FAIL'}")

# E4
fav_end = c.favorite()
print(f"E4 favorite_end={fav_end} (expected 2, report only)")

# Verdict
deviations = []
if not e1_pass:
    deviations.append("E1(recovery)")
if not e2_pass:
    deviations.append("E2(occupancy)")
if not e3_pass:
    deviations.append("E3(c2_drift)")

if not deviations:
    print("VERDICT: rung-3 REAL NEGATIVE (adaptation as-designed, no undeclared deviation)")
else:
    print(f"VERDICT: DEVIATION(S): {deviations} -> enters the novelty cascade (rung 5) next iteration")
