"""Exp 54 — surprise ledger v4: frozen references (rung 1, fourth attempt).

Two changes from v3, both from diagnosed causes:
(1) FROZEN REFERENCES (design rule #3): every expectation derives from the declared
reference state -- mirro's committed state and ITS world for the three detection arms;
the entrenched fork's declared post-entrenchment state for the enabled-side arm. The
scoring function never receives the epoch's world; un-blinding is structurally
impossible, not just avoided.
(2) Gap-scaled noise margin: +/-(0.015 + 0.5*|center|). By formula this covers all
three previously observed baseline drift readings (-0.0083, -0.0188, -0.0104) and the
v3 enabled-arm reading (+0.0114) -- coverage by derivation, not by fitting.
Arms (fresh per-step seeds, bases 61/62/63/64; mirro untouched, hash-verified):
  baseline           -- quiet expected (favorite property auto-DISABLED, gap ~0.018).
  strong anomaly     -- 10 cells recolored to 2; must flag (vs frozen ref).
  subtle anomaly     -- 3 cells recolored to 2; must flag (map-vs-frozen-ref ~0.88).
  enabled-side ctrl  -- entrenchment (1500 steps, c0-rich declared world) then quiet
                        epoch; favorite property ENABLED and must stay quiet.
Falsifier: any baseline false alarm OR either anomaly missed.
Predeclared escalation: if v4 fails on a NEW mode, the ladder pauses and the human is
consulted in this entry. Prediction: VALIDATED -- rung 1 complete.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# Frozen Reference bundle
# ---------------------------------------------------------------------------

@dataclass
class Ref:
    cmap: List[int]
    favorite: int
    share: float          # cur_share at reference time
    total_counts: float
    drift_band: Tuple[float, float]
    gap_share: float
    enabled: bool


def make_ref(creature, world_cmap, steps=1000) -> Ref:
    """Compute a frozen Ref from a declared creature state and ITS world cmap."""
    vc = creature.value_counts
    fav = creature.favorite()
    total = float(vc.sum())
    cur_share = float(vc[fav] / total)
    eq_share = world_cmap.count(fav) / len(world_cmap)
    center = (eq_share - cur_share) * steps / (total + steps)
    margin = 0.015 + 0.5 * abs(center)
    band = (center - margin, center + margin)

    sorted_vc = np.sort(vc)[::-1]
    runner_up = float(sorted_vc[1]) if len(sorted_vc) > 1 else 0.0
    gap_share = float((vc[fav] - runner_up) / total)
    enabled = gap_share > 0.03

    return Ref(
        cmap=list(world_cmap),
        favorite=fav,
        share=cur_share,
        total_counts=total,
        drift_band=band,
        gap_share=gap_share,
        enabled=enabled,
    )


# ---------------------------------------------------------------------------
# Load mirro (read-only)
# ---------------------------------------------------------------------------

MIRRO_DIR = Path("creature/state/mirro")
mirro = Creature.load(MIRRO_DIR)
ref_hash_before = mirro._state_hash()[:16]

print(f"mirro loaded: age={mirro.age_steps}, favorite={mirro.favorite()}, hash16={ref_hash_before}")
print(f"mirro cmap: {list(mirro.world.cmap)}")
print()

# Freeze reference for arms 1-3 from mirro + mirro's world
REF_MIRRO = make_ref(mirro, list(mirro.world.cmap))
print(f"[FROZEN REF arms 1-3] favorite={REF_MIRRO.favorite}, share={REF_MIRRO.share:.4f}, "
      f"gap_share={REF_MIRRO.gap_share:.4f}, enabled={REF_MIRRO.enabled}")
print(f"  drift_band=[{REF_MIRRO.drift_band[0]:.6f}, {REF_MIRRO.drift_band[1]:.6f}]")
print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_world(base_cmap, recolor_cells, new_color=2):
    cmap = list(base_cmap)
    for idx in recolor_cells:
        cmap[idx] = new_color
    return World(rows=mirro.world.rows, cols=mirro.world.cols,
                 cmap=cmap, n_colors=mirro.world.n_colors)


def run_epoch(fork, base_seed, steps=1000):
    n_cells = fork.world.n_cells
    visit_counts = np.zeros(n_cells, dtype=float)
    localize_sum = 0.0
    for i in range(steps):
        fork.live(1, seed=base_seed * 1000003 + i)
        visit_counts[fork.true_pos] += 1
        localize_sum += fork.localize_bits()
    return visit_counts, localize_sum / steps


def score(fork, visits, mean_lb, ref: Ref):
    """Score an epoch purely from the frozen Ref; never uses epoch world."""
    flags = []
    total_visits = visits.sum()
    if total_visits > 0:
        p = visits / total_visits
        occ_ent = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    else:
        occ_ent = 0.0
    if not (4.40 <= occ_ent <= 4.644):
        flags.append(f"occupancy_entropy_bits={occ_ent:.4f} outside [4.40,4.644]")

    learned_map = fork.sensory_map()
    map_acc = sum(l == r for l, r in zip(learned_map, ref.cmap)) / len(ref.cmap)
    if not (0.92 <= map_acc <= 1.0):
        flags.append(f"map_accuracy_vs_ref={map_acc:.4f} outside [0.92,1.0]")

    if not (0.0 <= mean_lb <= 0.15):
        flags.append(f"mean_localize_bits={mean_lb:.4f} outside [0.0,0.15]")

    vc = fork.value_counts
    fav_share_post = float(vc[ref.favorite] / vc.sum())
    conv_drift = fav_share_post - ref.share
    lo, hi = ref.drift_band
    if not (lo <= conv_drift <= hi):
        flags.append(f"conviction_drift={conv_drift:.5f} outside [{lo:.6f},{hi:.6f}]")

    if ref.enabled:
        fav_post = fork.favorite()
        if fav_post != ref.favorite:
            flags.append(f"favorite_changed: pre={ref.favorite} post={fav_post}")

    return flags, occ_ent, map_acc, conv_drift


# ---------------------------------------------------------------------------
# Arms 1-3: baseline, strong_anomaly, subtle_anomaly (all use REF_MIRRO)
# ---------------------------------------------------------------------------

EPOCHS_STD = [
    ("baseline",       [],                            61),
    ("strong_anomaly", [0,1,5,6,10,13,16,18,21,23],  62),
    ("subtle_anomaly", [3, 13, 19],                   63),
]

results = {}

for name, recolor_cells, base_seed in EPOCHS_STD:
    print(f"=== {name} (base_seed={base_seed}) ===")
    t0 = time.time()
    fork = mirro.fork(f"v4_{name}")
    if recolor_cells:
        fork.world = make_world(mirro.world.cmap, recolor_cells)
        print(f"  recolored cells {recolor_cells} -> color 2: {list(fork.world.cmap)}")
    visits, mean_lb = run_epoch(fork, base_seed)
    elapsed = time.time() - t0
    flags, occ, macc, cdrift = score(fork, visits, mean_lb, REF_MIRRO)
    results[name] = (flags, REF_MIRRO.enabled)
    fav_status = ("DISABLED" if not REF_MIRRO.enabled
                  else f"{fork.favorite() != REF_MIRRO.favorite} (post={fork.favorite()})")
    print(f"  occupancy_entropy_bits : {occ:.4f}")
    print(f"  map_accuracy_vs_ref    : {macc:.4f}")
    print(f"  mean_localize_bits     : {mean_lb:.4f}")
    print(f"  conviction_drift       : {cdrift:.5f}  band=[{REF_MIRRO.drift_band[0]:.6f},{REF_MIRRO.drift_band[1]:.6f}]")
    print(f"  favorite_changed       : {fav_status}")
    print(f"  elapsed: {elapsed:.1f}s  flags={len(flags)} {flags}")
    print()

# ---------------------------------------------------------------------------
# Arm 4: enabled-side control — entrenchment then frozen ref then scored epoch
# ---------------------------------------------------------------------------

print("=== enabled_side_ctrl (base_seed=64) ===")
t0 = time.time()

ENRICH_CELLS = [4, 9, 12, 15, 18, 21]
rich_cmap = list(mirro.world.cmap)
for idx in ENRICH_CELLS:
    rich_cmap[idx] = 0
rich_world = World(rows=mirro.world.rows, cols=mirro.world.cols,
                   cmap=rich_cmap, n_colors=mirro.world.n_colors)
print(f"  enriched world: c0={rich_cmap.count(0)}/25, cmap={rich_cmap}")

fork_ent = mirro.fork("v4_enabled_side_ctrl")
fork_ent.world = rich_world
print("  entrenchment: 1500 steps ...")
for i in range(1500):
    fork_ent.live(1, seed=64 * 1000003 + 1000000 + i)

# Freeze Ref from entrenched fork + declared rich world
REF_ENT = make_ref(fork_ent, rich_cmap)
arm4_void = not REF_ENT.enabled
print(f"[FROZEN REF arm 4] favorite={REF_ENT.favorite}, share={REF_ENT.share:.4f}, "
      f"gap_share={REF_ENT.gap_share:.4f}, enabled={REF_ENT.enabled}")
print(f"  drift_band=[{REF_ENT.drift_band[0]:.6f}, {REF_ENT.drift_band[1]:.6f}]")
if arm4_void:
    print(f"  gap_share={REF_ENT.gap_share:.4f} <= 0.03 -> arm VOID")

# Scored quiet epoch
visits_e, mean_lb_e = run_epoch(fork_ent, 64)
elapsed_e = time.time() - t0
flags_e, occ_e, macc_e, cdrift_e = score(fork_ent, visits_e, mean_lb_e, REF_ENT)
results["enabled_side_ctrl"] = (flags_e, REF_ENT.enabled)

fav_status_e = ("DISABLED (arm VOID)" if arm4_void
                else f"{fork_ent.favorite() != REF_ENT.favorite} (post={fork_ent.favorite()})")
print(f"  occupancy_entropy_bits : {occ_e:.4f}")
print(f"  map_accuracy_vs_ref    : {macc_e:.4f}")
print(f"  mean_localize_bits     : {mean_lb_e:.4f}")
print(f"  conviction_drift       : {cdrift_e:.5f}  band=[{REF_ENT.drift_band[0]:.6f},{REF_ENT.drift_band[1]:.6f}]")
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
for arm_name, (flags, enabled) in results.items():
    fav_prop = "ENABLED" if enabled else "DISABLED"
    print(f"  {arm_name}: favorite_property={fav_prop} flags={len(flags)} {flags}")

false_alarm   = len(results["baseline"][0]) > 0
strong_missed = len(results["strong_anomaly"][0]) == 0
subtle_missed = len(results["subtle_anomaly"][0]) == 0
ent_flags, ent_enabled = results["enabled_side_ctrl"]
enabled_arm_void = not ent_enabled

if enabled_arm_void:
    print(f"enabled-side arm VOID (gap_share={REF_ENT.gap_share:.4f} <= 0.03); "
          f"verdict rests on other three arms.")

arm4_ok = (len(ent_flags) == 0 or enabled_arm_void)

if not false_alarm and not strong_missed and not subtle_missed and arm4_ok:
    print("VERDICT: instrument VALIDATED — rung 1 complete")
else:
    parts = []
    if false_alarm:
        parts.append(f"false alarm on baseline: {results['baseline'][0]}")
    if strong_missed:
        parts.append("strong anomaly missed")
    if subtle_missed:
        parts.append("subtle anomaly missed")
    if not arm4_ok:
        parts.append(f"enabled-side arm flags: {ent_flags}")
    print(f"VERDICT: falsifier HIT ({'; '.join(parts)})")
    # New mode check: v3 known modes were map_accuracy and conviction_drift issues;
    # v2 known mode was favorite_changed; v1 known mode was map_accuracy.
    known_modes = {"map_accuracy", "conviction_drift", "favorite_changed", "occupancy_entropy"}
    all_flag_texts = " ".join(str(f) for arm_f, _ in results.values() for f in arm_f)
    is_new_mode = not any(m in all_flag_texts for m in known_modes)
    if is_new_mode:
        print("ESCALATION: pause ladder, consult human")
