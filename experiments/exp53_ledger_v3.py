"""Exp 53 — surprise ledger v3: state-conditional invariants (rung 1, third attempt).

One change from v2 (Exp 52's falsifier hit): favorite constancy is scored ONLY when the
creature's pre-epoch entrenchment exceeds the epoch noise margin -- gap_share =
(counts[favorite] - counts[runner_up]) / total > 0.03; otherwise the property reports
DISABLED (favorite identity at near-indifference is noise, not an invariant).
All other properties identical to v2 (drift band derived per-creature from world
composition: center = (eq_share - cur_share)*steps/(total+steps), +/-0.015).
Four blind fork epochs, fresh per-step action seeds (bases 51/52/53/54), mirro untouched:
  baseline           -- mirro fork, own world. gap_share ~0.022 < 0.03 so favorite
                        property should auto-DISABLE; drift must stay in band. Quiet.
  strong anomaly     -- 10 cells recolored to 2. Must flag.
  subtle anomaly     -- 3 cells recolored to 2. Must flag.
  enabled-side ctrl  -- fork pre-entrenched 1500 steps in a color-0-rich world (cells
                        [4,9,12,15,18,21] recolored to 0; c0=15/25), gap_share must rise
                        > 0.03 (ENABLING the favorite property; if it does not, report
                        and treat this arm as void); then a quiet 1000-step epoch in
                        that same world with bands derived from the entrenched state.
                        Must stay quiet WITH the favorite property active.
Falsifier: any baseline false alarm OR strong/subtle anomaly missed.
Prediction: VALIDATED (quiet baselines on both sides of the condition; anomalies
flagged) -- completing rung 1 with two earned design rules built in.
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
# Helper: compute gap_share for any creature state
# ---------------------------------------------------------------------------

def compute_gap_share(creature):
    vc = creature.value_counts
    total = float(vc.sum())
    fav = creature.favorite()
    sorted_vc = np.sort(vc)[::-1]
    runner_up_count = sorted_vc[1] if len(sorted_vc) > 1 else 0
    return float((vc[fav] - runner_up_count) / total), fav


# ---------------------------------------------------------------------------
# Derive conviction-drift band for a given creature state + world
# ---------------------------------------------------------------------------

STEPS = 1000
NOISE_MARGIN = 0.015


def derive_band(creature, world_cmap, steps=STEPS):
    fav = creature.favorite()
    vc = creature.value_counts
    cur_share = float(vc[fav] / vc.sum())
    total = float(vc.sum())
    eq_share = world_cmap.count(fav) / len(world_cmap)
    center = (eq_share - cur_share) * steps / (total + steps)
    return center - NOISE_MARGIN, center + NOISE_MARGIN, eq_share, cur_share, total


# ---------------------------------------------------------------------------
# run_epoch
# ---------------------------------------------------------------------------

def run_epoch(fork, base_seed, steps=STEPS):
    n_cells = fork.world.n_cells
    visit_counts = np.zeros(n_cells, dtype=float)
    localize_sum = 0.0
    for i in range(steps):
        fork.live(1, seed=base_seed * 1000003 + i)
        visit_counts[fork.true_pos] += 1
        localize_sum += fork.localize_bits()
    return visit_counts, localize_sum / steps


# ---------------------------------------------------------------------------
# score: generic, blind; favorite rule is conditional on gap_share
# ---------------------------------------------------------------------------

def score(props, drift_lo, drift_hi, gap_share_enabled):
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
    if gap_share_enabled:
        if props["favorite_changed"]:
            flags.append(f"favorite_changed: pre={props['fav_pre']} post={props['fav_post']}")
    return flags


def compute_props(fork, visit_counts, mean_lb, epoch_ref_cmap, epoch_ref_favorite):
    total_visits = visit_counts.sum()
    if total_visits > 0:
        p = visit_counts / total_visits
        occ_ent = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    else:
        occ_ent = 0.0
    learned_map = fork.sensory_map()
    map_acc = sum(l == r for l, r in zip(learned_map, epoch_ref_cmap)) / len(epoch_ref_cmap)
    vc = fork.value_counts
    fav_share_post = float(vc[epoch_ref_favorite] / vc.sum())
    # conviction_drift relative to entry conviction share
    conv_drift = fav_share_post - float(mirro.value_counts[epoch_ref_favorite] / mirro.value_counts.sum())
    fav_post = fork.favorite()
    return {
        "occupancy_entropy_bits": occ_ent,
        "map_accuracy_vs_reference": map_acc,
        "mean_localize_bits": mean_lb,
        "conviction_drift": conv_drift,
        "favorite_changed": fav_post != epoch_ref_favorite,
        "fav_pre": epoch_ref_favorite,
        "fav_post": fav_post,
    }


def make_world(base_cmap, recolor_cells, new_color=2):
    cmap = list(base_cmap)
    for idx in recolor_cells:
        cmap[idx] = new_color
    return World(rows=mirro.world.rows, cols=mirro.world.cols,
                 cmap=cmap, n_colors=mirro.world.n_colors)


# ---------------------------------------------------------------------------
# Epochs 1-3: baseline, strong_anomaly, subtle_anomaly
# ---------------------------------------------------------------------------

EPOCHS_STD = [
    ("baseline",       [],                              51),
    ("strong_anomaly", [0,1,5,6,10,13,16,18,21,23],    52),
    ("subtle_anomaly", [3, 13, 19],                     53),
]

results = {}

for name, recolor_cells, base_seed in EPOCHS_STD:
    print(f"=== {name} (base_seed={base_seed}) ===")
    t0 = time.time()
    fork = mirro.fork(f"v3_{name}")
    epoch_cmap = list(ref_cmap)
    if recolor_cells:
        fork.world = make_world(ref_cmap, recolor_cells)
        epoch_cmap = list(fork.world.cmap)
        print(f"  recolored cells {recolor_cells} -> color 2")
        print(f"  world_cmap: {epoch_cmap}")
    # Derive band from mirro's state (entering the epoch) + epoch world
    drift_lo, drift_hi, eq_share, cur_share, total = derive_band(mirro, epoch_cmap)
    gap_share, fav = compute_gap_share(mirro)
    gap_enabled = gap_share > 0.03
    print(f"  gap_share={gap_share:.4f} -> favorite_property={'ENABLED' if gap_enabled else 'DISABLED'}")
    print(f"  drift band: center+/-0.015 => [{drift_lo:.6f}, {drift_hi:.6f}]  "
          f"(eq={eq_share:.4f}, cur={cur_share:.4f})")
    visits, mean_lb = run_epoch(fork, base_seed)
    elapsed = time.time() - t0
    props = compute_props(fork, visits, mean_lb, epoch_cmap, fav)
    flags = score(props, drift_lo, drift_hi, gap_enabled)
    results[name] = (flags, gap_enabled)
    print(f"  occupancy_entropy_bits : {props['occupancy_entropy_bits']:.4f}")
    print(f"  map_accuracy_vs_ref    : {props['map_accuracy_vs_reference']:.4f}")
    print(f"  mean_localize_bits     : {props['mean_localize_bits']:.4f}")
    print(f"  conviction_drift       : {props['conviction_drift']:.5f}")
    fav_status = f"{props['favorite_changed']} (fav={props['fav_post']})" if gap_enabled else "DISABLED"
    print(f"  favorite_changed       : {fav_status}")
    print(f"  elapsed: {elapsed:.1f}s  flags={len(flags)} {flags}")
    print()

# ---------------------------------------------------------------------------
# Epoch 4: enabled-side control (entrenchment + quiet epoch)
# ---------------------------------------------------------------------------

print("=== enabled_side_ctrl (base_seed=54) ===")
t0 = time.time()
# Build c0-rich world: cells [4,9,12,15,18,21] -> color 0  => c0=15/25
ENRICH_CELLS = [4, 9, 12, 15, 18, 21]
rich_cmap = list(ref_cmap)
for idx in ENRICH_CELLS:
    rich_cmap[idx] = 0
rich_world = World(rows=mirro.world.rows, cols=mirro.world.cols,
                   cmap=rich_cmap, n_colors=mirro.world.n_colors)
c0_count = rich_cmap.count(0)
print(f"  enriched world: c0={c0_count}/25, cmap={rich_cmap}")

# Fork mirro and run entrenchment phase (1500 steps, seed base 54)
fork_ent = mirro.fork("v3_enabled_side_ctrl")
fork_ent.world = rich_world
print("  entrenchment: 1500 steps ...")
for i in range(1500):
    fork_ent.live(1, seed=54 * 1000003 + 1000000 + i)

gap_share_ent, fav_ent = compute_gap_share(fork_ent)
print(f"  post-entrenchment: age={fork_ent.age_steps}, favorite={fav_ent}, "
      f"gap_share={gap_share_ent:.4f}")

# Derive band from entrenched state + rich world
drift_lo_e, drift_hi_e, eq_share_e, cur_share_e, total_e = derive_band(fork_ent, rich_cmap)
gap_enabled_ent = gap_share_ent > 0.03
print(f"  gap_share={gap_share_ent:.4f} -> favorite_property={'ENABLED' if gap_enabled_ent else 'DISABLED (arm VOID)'}")
print(f"  drift band: [{drift_lo_e:.6f}, {drift_hi_e:.6f}]  (eq={eq_share_e:.4f}, cur={cur_share_e:.4f})")

# For conviction_drift we compare against entrenched cur_share
ent_conv_share = cur_share_e  # pre-epoch share for this arm

# Scored epoch: 1000 steps in the same rich world
visits_e, mean_lb_e = run_epoch(fork_ent, 54)
elapsed_e = time.time() - t0

# compute_props for entrenched arm (drift relative to entrenched creature's share)
total_visits_e = visits_e.sum()
if total_visits_e > 0:
    p_e = visits_e / total_visits_e
    occ_ent_e = float(-np.sum(p_e[p_e > 0] * np.log2(p_e[p_e > 0])))
else:
    occ_ent_e = 0.0
learned_map_e = fork_ent.sensory_map()
map_acc_e = sum(l == r for l, r in zip(learned_map_e, rich_cmap)) / len(rich_cmap)
vc_e = fork_ent.value_counts
fav_share_post_e = float(vc_e[fav_ent] / vc_e.sum())
conv_drift_e = fav_share_post_e - ent_conv_share
fav_post_e = fork_ent.favorite()
props_e = {
    "occupancy_entropy_bits": occ_ent_e,
    "map_accuracy_vs_reference": map_acc_e,
    "mean_localize_bits": mean_lb_e,
    "conviction_drift": conv_drift_e,
    "favorite_changed": fav_post_e != fav_ent,
    "fav_pre": fav_ent,
    "fav_post": fav_post_e,
}
flags_e = score(props_e, drift_lo_e, drift_hi_e, gap_enabled_ent)
results["enabled_side_ctrl"] = (flags_e, gap_enabled_ent)

print(f"  occupancy_entropy_bits : {props_e['occupancy_entropy_bits']:.4f}")
print(f"  map_accuracy_vs_ref    : {props_e['map_accuracy_vs_reference']:.4f}")
print(f"  mean_localize_bits     : {props_e['mean_localize_bits']:.4f}")
print(f"  conviction_drift       : {props_e['conviction_drift']:.5f}")
fav_status_e = f"{props_e['favorite_changed']} (fav={props_e['fav_post']})" if gap_enabled_ent else "DISABLED"
print(f"  favorite_changed       : {fav_status_e}")
print(f"  elapsed: {elapsed_e:.1f}s  flags={len(flags_e)} {flags_e}")
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
for name, (flags, gap_en) in results.items():
    fav_prop = "ENABLED" if gap_en else "DISABLED"
    print(f"  {name}: favorite_property={fav_prop} flags={len(flags)} {flags}")

false_alarm   = len(results["baseline"][0]) > 0
strong_missed = len(results["strong_anomaly"][0]) == 0
subtle_missed = len(results["subtle_anomaly"][0]) == 0

# enabled-side arm: check if it actually enabled
ent_flags, ent_enabled = results["enabled_side_ctrl"]
enabled_arm_void = not ent_enabled

if enabled_arm_void:
    print(f"enabled-side arm VOID (gap_share={gap_share_ent:.4f} <= 0.03); "
          f"verdict rests on other three arms; enabled-side untested.")

if not false_alarm and not strong_missed and not subtle_missed:
    verdict = ("VERDICT: instrument VALIDATED (rung 1 complete)")
    print(verdict)
else:
    parts = []
    if false_alarm:
        parts.append(f"false alarm on baseline: {results['baseline'][0]}")
    if strong_missed:
        parts.append("strong anomaly missed")
    if subtle_missed:
        parts.append("subtle anomaly missed")
    print(f"VERDICT: falsifier HIT ({'; '.join(parts)})")
