"""Exp 51 — the surprise ledger: an instrument that can certify 'unexpected' (rung 1).

Functional-emergence rung 1: novelty claims need an instrument that (a) stays quiet on a
normal epoch and (b) flags a planted anomaly it was not told about. Ledger ranges are
PREDECLARED in experiments/exp51_ledger.json (written before any epoch ran; analytic +
Exp 49-calibrated): occupancy entropy [4.40, 4.644] bits (clamped random walk on the 5x5
is doubly stochastic -> uniform stationary, log2(25)=4.644); map accuracy vs the
pre-epoch reference world [0.92, 1.0]; mean localize bits < 0.15; conviction drift
[-0.005, +0.04] (Exp 49 measured ~0.0014/100 steps); favorite must not change.
The scorer is GENERIC: identical code runs on both epochs; it is never told which is the
anomaly. Controls (both on disposable forks; mirro untouched):
  baseline epoch -- fork lives 1000 steps in mirro's current world. Expect 0 flags.
  planted anomaly -- fork lives 1000 steps in a world with 10 cells recolored to color 2
  (the perturbation the scorer is blind to). Expect >= 1 flag (reference-map mismatch
  and/or conviction-drift breach / favorite flip).
Prediction: ledger passes both controls (quiet baseline, anomaly flagged).
Falsifier (card): misses the planted anomaly OR false-alarms on the baseline. A
miscalibrated predeclared range that false-alarms IS a failure of the instrument.
Forks live with explicit seeds 31 (baseline) and 32 (anomaly); single pair, reported.
"""

import copy
import json
import math
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

print(f"mirro loaded: age={mirro.age_steps}, favorite={ref_favorite}, "
      f"conviction_share={ref_conviction_share:.4f}, hash16={ref_hash_before}")
print(f"ref_cmap: {ref_cmap}")
print()

# Note: forks share mirro's _seed; with live(1) called without explicit seed,
# each call increments rng_counter and derives a fresh seed from (self._seed, rng_counter).
# Baseline and anomaly forks start from the SAME rng_counter value, so they get
# IDENTICAL action streams — matched trajectories, isolating the world-map effect.

# ---------------------------------------------------------------------------
# Load ledger
# ---------------------------------------------------------------------------

LEDGER_PATH = Path("experiments/exp51_ledger.json")
ledger = json.loads(LEDGER_PATH.read_text())
L = ledger["properties"]


def run_epoch(fork, steps=1000):
    """Run fork for steps×live(1); return visit_counts array and mean localize bits."""
    n_cells = fork.world.n_cells
    visit_counts = np.zeros(n_cells, dtype=float)
    localize_sum = 0.0
    for _ in range(steps):
        fork.live(1)
        visit_counts[fork.true_pos] += 1
        localize_sum += fork.localize_bits()
    return visit_counts, localize_sum / steps


def score(props, label):
    """Score properties dict against ledger; return list of flag strings."""
    flags = []
    occ = props["occupancy_entropy_bits"]
    if not (L["occupancy_entropy_bits"]["min"] <= occ <= L["occupancy_entropy_bits"]["max"]):
        flags.append(f"occupancy_entropy_bits={occ:.4f} outside [{L['occupancy_entropy_bits']['min']}, {L['occupancy_entropy_bits']['max']}]")
    macc = props["map_accuracy_vs_reference"]
    if not (L["map_accuracy_vs_reference"]["min"] <= macc <= L["map_accuracy_vs_reference"]["max"]):
        flags.append(f"map_accuracy_vs_reference={macc:.4f} outside [{L['map_accuracy_vs_reference']['min']}, {L['map_accuracy_vs_reference']['max']}]")
    mlb = props["mean_localize_bits"]
    if not (L["mean_localize_bits"]["min"] <= mlb <= L["mean_localize_bits"]["max"]):
        flags.append(f"mean_localize_bits={mlb:.4f} outside [{L['mean_localize_bits']['min']}, {L['mean_localize_bits']['max']}]")
    cd = props["conviction_drift"]
    if not (L["conviction_drift"]["min"] <= cd <= L["conviction_drift"]["max"]):
        flags.append(f"conviction_drift={cd:.5f} outside [{L['conviction_drift']['min']}, {L['conviction_drift']['max']}]")
    if props["favorite_changed"]:
        flags.append(f"favorite_changed: pre={props['fav_pre']} post={props['fav_post']}")
    return flags


def compute_props(fork, visit_counts, mean_lb):
    """Compute all scored properties from fork post-epoch."""
    # occupancy entropy
    total_visits = visit_counts.sum()
    if total_visits > 0:
        p = visit_counts / total_visits
        occ_ent = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    else:
        occ_ent = 0.0

    # map accuracy vs REFERENCE cmap (not fork's current world — scorer is blind to anomaly)
    learned_map = fork.sensory_map()
    map_acc = sum(l == r for l, r in zip(learned_map, ref_cmap)) / len(ref_cmap)

    # conviction drift: track ref_favorite's share
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


# ---------------------------------------------------------------------------
# Baseline epoch
# ---------------------------------------------------------------------------

print("=== BASELINE epoch ===")
print("(fork in mirro's unmodified world, 1000×live(1))")
t0 = time.time()
fb = mirro.fork("ledger_baseline")
visits_b, mlb_b = run_epoch(fb, steps=1000)
elapsed_b = time.time() - t0

props_b = compute_props(fb, visits_b, mlb_b)
flags_b = score(props_b, "baseline")

print(f"  occupancy_entropy_bits : {props_b['occupancy_entropy_bits']:.4f}")
print(f"  map_accuracy_vs_ref    : {props_b['map_accuracy_vs_reference']:.4f}")
print(f"  mean_localize_bits     : {props_b['mean_localize_bits']:.4f}")
print(f"  conviction_drift       : {props_b['conviction_drift']:.5f}")
print(f"  favorite_changed       : {props_b['favorite_changed']} (fav={props_b['fav_post']})")
print(f"  elapsed: {elapsed_b:.1f}s")
print(f"  baseline_flags={len(flags_b)} {flags_b}")
print()

# ---------------------------------------------------------------------------
# Anomaly epoch — perturbed world (10 cells recolored to color 2)
# ---------------------------------------------------------------------------

print("=== ANOMALY epoch ===")
PERTURB_CELLS = [0, 1, 5, 6, 10, 13, 16, 18, 21, 23]
print(f"(fork in world with cells {PERTURB_CELLS} recolored to color 2)")
t1 = time.time()
fa = mirro.fork("ledger_anomaly")
# Build perturbed world
perturbed_cmap = list(mirro.world.cmap)
for idx in PERTURB_CELLS:
    perturbed_cmap[idx] = 2
fa.world = World(
    rows=mirro.world.rows,
    cols=mirro.world.cols,
    cmap=perturbed_cmap,
    n_colors=mirro.world.n_colors,
)
print(f"  perturbed_cmap: {perturbed_cmap}")
visits_a, mlb_a = run_epoch(fa, steps=1000)
elapsed_a = time.time() - t1

props_a = compute_props(fa, visits_a, mlb_a)
flags_a = score(props_a, "anomaly")

print(f"  occupancy_entropy_bits : {props_a['occupancy_entropy_bits']:.4f}")
print(f"  map_accuracy_vs_ref    : {props_a['map_accuracy_vs_reference']:.4f}")
print(f"  mean_localize_bits     : {props_a['mean_localize_bits']:.4f}")
print(f"  conviction_drift       : {props_a['conviction_drift']:.5f}")
print(f"  favorite_changed       : {props_a['favorite_changed']} (fav={props_a['fav_post']})")
print(f"  elapsed: {elapsed_a:.1f}s")
print(f"  anomaly_flags={len(flags_a)} {flags_a}")
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
print(f"baseline_flags={len(flags_b)} {flags_b}")
print(f"anomaly_flags={len(flags_a)} {flags_a}")

false_alarm = len(flags_b) > 0
anomaly_missed = len(flags_a) == 0

if not false_alarm and not anomaly_missed:
    print("VERDICT: instrument VALIDATED (quiet baseline, anomaly flagged)")
elif false_alarm and anomaly_missed:
    print(f"VERDICT: falsifier HIT (false alarm on baseline: {flags_b})")
    print("VERDICT: falsifier HIT (planted anomaly missed)")
elif false_alarm:
    print(f"VERDICT: falsifier HIT (false alarm on baseline: {flags_b})")
else:
    print("VERDICT: falsifier HIT (planted anomaly missed)")
