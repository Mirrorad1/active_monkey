"""Exp 52 — surprise ledger v2: revalidation + sensitivity floor (rung 1, second attempt).

One change from v1 (Exp 51's falsifier hit): the conviction-drift band is DERIVED, before
any epoch runs, from the current world's composition equilibrium --
  eq_share = (count of favorite-colored cells)/n_cells  (all cells equally predictable
  at convergence, so equilibrium accrual share = visit share);
  center = (eq_share - cur_share) * steps/(total_value_counts + steps);
  band = center +/- 0.015 (visit-share random-walk noise margin).
The formula is analytic from the diagnosis, not fitted to Exp 51's failed reading; it
must cover that reading (-0.00834) naturally or v2 is already suspect. All other bands
unchanged (they passed v1). Scorer identical and blind across epochs.
Four fork epochs, fresh per-step action seeds (NOT mirro's stream; mirro untouched):
  baseline (seed base 41)        -- must stay quiet.
  strong anomaly (10 recolored cells, base 42)  -- must flag.
  subtle anomaly (3 recolored cells, base 43)   -- must flag (22/25=0.88 < 0.92 map band).
  floor probe (2 recolored cells, base 44)      -- PREDICTED MISS (23/25=0.92 sits on the
  band edge, >= passes): measures the detection floor; a miss here is data, not failure.
Falsifier: baseline false alarm OR strong/subtle anomaly missed.
Prediction: VALIDATED with a documented 3-cell detection floor.
"""

import time
from pathlib import Path

import numpy as np

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# Load mirro (read-only), record reference state
# ---------------------------------------------------------------------------

MIRRO_DIR = Path("creature/state/mirro")
mirro = Creature.load(MIRRO_DIR)
ref_hash_before = mirro._state_hash()[:16]
ref_cmap = list(mirro.world.cmap)
ref_favorite = mirro.favorite()
ref_conviction_share = float(mirro.value_counts[ref_favorite] / mirro.value_counts.sum())
total_counts = float(mirro.value_counts.sum())

print(f"mirro loaded: age={mirro.age_steps}, favorite={ref_favorite}, "
      f"conviction_share={ref_conviction_share:.4f}, hash16={ref_hash_before}")
print(f"ref_cmap: {ref_cmap}")
print()

# ---------------------------------------------------------------------------
# Derive conviction-drift band (BEFORE any epoch)
# ---------------------------------------------------------------------------

N_CELLS = len(ref_cmap)
STEPS = 1000
NOISE_MARGIN = 0.015

eq_share = ref_cmap.count(ref_favorite) / N_CELLS
drift_center = (eq_share - ref_conviction_share) * STEPS / (total_counts + STEPS)
drift_lo = drift_center - NOISE_MARGIN
drift_hi = drift_center + NOISE_MARGIN

print(f"derived conviction_drift band:")
print(f"  eq_share={eq_share:.6f} ({ref_cmap.count(ref_favorite)}/{N_CELLS})")
print(f"  cur_share={ref_conviction_share:.6f}, total_counts={total_counts:.1f}")
print(f"  center={drift_center:.6f}, band=[{drift_lo:.6f}, {drift_hi:.6f}]")
exp51_reading = -0.00834
covers = drift_lo <= exp51_reading <= drift_hi
print(f"  covers_exp51_baseline_reading={covers}  (reading={exp51_reading})")
print()


# ---------------------------------------------------------------------------
# run_epoch: explicit per-step seeds (fresh stream, not mirro's rng)
# ---------------------------------------------------------------------------

def run_epoch(fork, base_seed, steps=1000):
    n_cells = fork.world.n_cells
    visit_counts = np.zeros(n_cells, dtype=float)
    localize_sum = 0.0
    for i in range(steps):
        fork.live(1, seed=base_seed * 1000003 + i)
        visit_counts[fork.true_pos] += 1
        localize_sum += fork.localize_bits()
    return visit_counts, localize_sum / steps


# ---------------------------------------------------------------------------
# score: generic, blind (uses derived band for conviction_drift)
# ---------------------------------------------------------------------------

def score(props):
    flags = []
    occ = props["occupancy_entropy_bits"]
    if not (4.40 <= occ <= 4.644):
        flags.append(f"occupancy_entropy_bits={occ:.4f} outside [4.40, 4.644]")
    macc = props["map_accuracy_vs_reference"]
    if not (0.92 <= macc <= 1.0):
        flags.append(f"map_accuracy_vs_reference={macc:.4f} outside [0.92, 1.0]")
    mlb = props["mean_localize_bits"]
    if not (0.0 <= mlb <= 0.15):
        flags.append(f"mean_localize_bits={mlb:.4f} outside [0.0, 0.15]")
    cd = props["conviction_drift"]
    if not (drift_lo <= cd <= drift_hi):
        flags.append(f"conviction_drift={cd:.5f} outside [{drift_lo:.6f}, {drift_hi:.6f}]")
    if props["favorite_changed"]:
        flags.append(f"favorite_changed: pre={props['fav_pre']} post={props['fav_post']}")
    return flags


def compute_props(fork, visit_counts, mean_lb):
    total_visits = visit_counts.sum()
    if total_visits > 0:
        p = visit_counts / total_visits
        occ_ent = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    else:
        occ_ent = 0.0
    learned_map = fork.sensory_map()
    map_acc = sum(l == r for l, r in zip(learned_map, ref_cmap)) / len(ref_cmap)
    fav_share_post = float(fork.value_counts[ref_favorite] / fork.value_counts.sum())
    conv_drift = fav_share_post - ref_conviction_share
    fav_post = fork.favorite()
    return {
        "occupancy_entropy_bits": occ_ent,
        "map_accuracy_vs_reference": map_acc,
        "mean_localize_bits": mean_lb,
        "conviction_drift": conv_drift,
        "favorite_changed": fav_post != ref_favorite,
        "fav_pre": ref_favorite,
        "fav_post": fav_post,
    }


def make_world(recolor_cells, new_color=2):
    cmap = list(ref_cmap)
    for idx in recolor_cells:
        cmap[idx] = new_color
    return World(rows=mirro.world.rows, cols=mirro.world.cols,
                 cmap=cmap, n_colors=mirro.world.n_colors)


# ---------------------------------------------------------------------------
# Four epochs
# ---------------------------------------------------------------------------

EPOCHS = [
    ("baseline",      [],                    41),
    ("strong_anomaly",[0,1,5,6,10,13,16,18,21,23], 42),
    ("subtle_anomaly",[3, 13, 19],            43),
    ("floor_probe",   [7, 22],               44),
]

results = {}

for name, recolor_cells, base_seed in EPOCHS:
    print(f"=== {name} (base_seed={base_seed}) ===")
    t0 = time.time()
    fork = mirro.fork(f"v2_{name}")
    if recolor_cells:
        fork.world = make_world(recolor_cells)
        print(f"  recolored cells {recolor_cells} -> color 2")
        print(f"  world_cmap: {list(fork.world.cmap)}")
    visits, mean_lb = run_epoch(fork, base_seed)
    elapsed = time.time() - t0
    props = compute_props(fork, visits, mean_lb)
    flags = score(props)
    results[name] = flags
    print(f"  occupancy_entropy_bits : {props['occupancy_entropy_bits']:.4f}")
    print(f"  map_accuracy_vs_ref    : {props['map_accuracy_vs_reference']:.4f}")
    print(f"  mean_localize_bits     : {props['mean_localize_bits']:.4f}")
    print(f"  conviction_drift       : {props['conviction_drift']:.5f}")
    print(f"  favorite_changed       : {props['favorite_changed']} (fav={props['fav_post']})")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  {name}_flags={len(flags)} {flags}")
    print()

# ---------------------------------------------------------------------------
# Verify mirro untouched
# ---------------------------------------------------------------------------

mirro2 = Creature.load(MIRRO_DIR)
ref_hash_after = mirro2._state_hash()[:16]
mirro_untouched = (ref_hash_before == ref_hash_after)
print(f"mirro_untouched={mirro_untouched} (hash_before={ref_hash_before}, hash_after={ref_hash_after})")
print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=== SUMMARY ===")
for name, flags in results.items():
    print(f"  {name}: flags={len(flags)} {flags}")

floor_3cell = len(results["subtle_anomaly"]) >= 1
floor_2cell = len(results["floor_probe"]) >= 1
print(f"detection_floor: 3-cell flagged={floor_3cell}, 2-cell flagged={floor_2cell}")

false_alarm   = len(results["baseline"]) > 0
strong_missed = len(results["strong_anomaly"]) == 0
subtle_missed = len(results["subtle_anomaly"]) == 0

if not false_alarm and not strong_missed and not subtle_missed:
    print("VERDICT: instrument VALIDATED (quiet baseline; 10-cell and 3-cell anomalies flagged; "
          f"floor at <=2 cells documented)")
else:
    parts = []
    if false_alarm:
        parts.append(f"false alarm on baseline: {results['baseline']}")
    if strong_missed:
        parts.append("strong anomaly missed")
    if subtle_missed:
        parts.append("subtle anomaly missed")
    print(f"VERDICT: falsifier HIT ({'; '.join(parts)})")
