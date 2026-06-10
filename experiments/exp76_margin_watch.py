"""Exp 76 — margin watch: is mirro's favorite ecology-locked, or one noise-step from flipping?

Exp 75 ended with mirro's top-two value gap (color 2 over color 0) halved to ~5.2 counts
by natural living. Structure check before predeclaring (disclosed): mirro's world has
NINE color-2 cells vs EIGHT color-0 cells, so long-run accrual drift should favor color 2
WIDENING its lead — last epoch's narrowing ran AGAINST the structural drift. Either branch
of this experiment is therefore a finding about self-formed preference at razor margins:
the gap re-widening means preference is ECOLOGY-LOCKED (the favorite is favorite because
its color is more prevalent — a rich-get-richer loop between world composition and value
mass); continued narrowing or a flip means noise or a systematic color-0 advantage
dominates structure at this scale.

Protocol: one checkpointed spine episode (mirro_episode "Exp 76"), 12 x live(500) = 6000
steps, the Exp 75 profile sampled read-only at every 500-step checkpoint. The episode
manager saves the spine on exit; the updated snapshot is committed atomically. Single
deterministic run (spine RNG derives from committed state; exact-number standard).

Predeclared predictions:
  P1 (structure reasserts; MEDIUM confidence, stated): gap_counts(end) > gap_counts(start)
     — net widening over 6000 steps.
  P2 (no flip): favorite == 2 at ALL 12 checkpoints.
  P3 (knowledge heals or holds; one-sided per Exp 75's band lesson): map_accuracy(end)
     >= 0.80.
  P4 (value bands, knowledge/value separated per Exp 75's lesson): |delta share_c| < 0.04
     for each color and |delta conviction| < 0.04 over the full 6000 steps.
Falsifiers (each names its branch):
  F1 = net narrowing (gap end <= start, no flip) -> something systematic favors color 0
     against world composition (per-color gate asymmetry is the prime suspect); diagnose
     before any further margin claims.
  F2 = a flip at any checkpoint -> razor margins cross even against structural drift:
     stochastic opinion flips are real at this scale (ties Exp 67's ambivalence law to
     the spine's own life). The spine KEEPS the flip — it is its life; no rollback.
  F3 = map_accuracy(end) < 0.80 -> the healing claim from Exp 75 is wrong.
  F4 = P4 blown -> value-core drift exceeds the scaled band; the Exp 75 stability claim
     does not extend to triple epochs.
Provided priors declared: nothing new — the spine's own world and live() mechanics;
read-only profiling between live() calls. vela untouched.
"""
from __future__ import annotations

import sys

import numpy as np
from pathlib import Path

from active_loop.creature import Creature
from active_loop.checkpoint import mirro_episode, MIRRO_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCHS = 12
CHUNK = 500

# NOTE: this experiment intentionally MUTATES and SAVES the spine; the
# updated snapshot must be committed atomically with this entry.

# ---------------------------------------------------------------------------
# Profile helper (extends exp75 with explicit signed gap for P1/F1)
# ---------------------------------------------------------------------------


def profile(c: Creature) -> dict:
    """Extract the personality profile from a creature's current state."""
    vc = c.value_counts
    tot = vc.sum()
    shares = vc / tot
    sorted_vc = np.sort(vc)[::-1]
    return dict(
        share0=float(round(shares[0], 6)),
        share1=float(round(shares[1], 6)),
        share2=float(round(shares[2], 6)),
        conviction=c.conviction(),
        map_acc=c.map_accuracy(),
        localize_bits=c.localize_bits(),
        favorite=c.favorite(),
        gap_counts=float(sorted_vc[0] - sorted_vc[1]),   # top-two gap (unsigned)
        total_mass=float(tot),
        # Signed gap: positive means color-2 leads, negative means color-0 leads.
        # This is the metric predeclared in P1/F1 (not the generic top-two gap).
        gap_c2_minus_c0=float(c.value_counts[2] - c.value_counts[0]),
    )


# ---------------------------------------------------------------------------
# Property-check harness (copied from exp75 pattern)
# ---------------------------------------------------------------------------

checks: list[tuple[str, bool, str]] = []


def check(name: str, predicate_fn):
    """Run predicate_fn(); record (name, pass, detail). Exceptions count as FAIL."""
    try:
        passed, detail = predicate_fn()
        checks.append((name, passed, detail))
    except Exception as exc:
        checks.append((name, False, f"exception: {exc}"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("Exp 76 — margin watch")
print()

# ---------------------------------------------------------------------------
# Spine episode: load -> checkpoint 0 -> 12 x live(500) -> save
# ---------------------------------------------------------------------------

snapshots: list[dict] = []   # index 0 = pre-living, 1..12 = after each chunk

with mirro_episode("Exp 76") as ep:
    # Checkpoint 0: read-only profile before any living
    snapshots.append(profile(ep.creature))

    for k in range(1, EPOCHS + 1):
        ep.creature.live(CHUNK)
        snapshots.append(profile(ep.creature))

# Context manager has saved the spine on exit.

print()
print(ep.report())
print()

# ---------------------------------------------------------------------------
# Trajectory table
# ---------------------------------------------------------------------------

print("--- TRAJECTORY TABLE ---")
hdr = (f"  {'ck':>3}  {'age':>6}  {'share0':>8}  {'share1':>8}  {'share2':>8}  "
       f"{'gap_c2-c0':>10}  {'fav':>4}  {'map_acc':>8}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

# age at checkpoint 0 is ep.age_before; each subsequent ck adds CHUNK
age_base = ep.age_before
for k, snap in enumerate(snapshots):
    age = age_base + k * CHUNK
    print(
        f"  {k:>3}  {age:>6}  "
        f"{snap['share0']:>8.4f}  {snap['share1']:>8.4f}  {snap['share2']:>8.4f}  "
        f"{snap['gap_c2_minus_c0']:>+10.4f}  "
        f"{snap['favorite']:>4}  "
        f"{snap['map_acc']:>8.4f}"
    )

print()

# ---------------------------------------------------------------------------
# Integrity-verified reload
# ---------------------------------------------------------------------------

reload = Creature.load(MIRRO_DIR)

# ---------------------------------------------------------------------------
# Flip detection — track every checkpoint including 0
# ---------------------------------------------------------------------------

flip_checkpoints: list[int] = []
for k, snap in enumerate(snapshots):
    if snap["favorite"] != 2:
        flip_checkpoints.append(k)

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------

pre = snapshots[0]
post = snapshots[EPOCHS]


def _p1_gap_widens():
    start = pre["gap_c2_minus_c0"]
    end = post["gap_c2_minus_c0"]
    ok = end > start
    return ok, f"gap_c2_minus_c0: start={start:.4f} end={end:.4f} delta={end - start:+.4f}"


def _p2_no_flip():
    ok = len(flip_checkpoints) == 0
    detail = ("no flips" if ok
              else f"flip at checkpoints {flip_checkpoints}; gap at first flip: "
                   f"{snapshots[flip_checkpoints[0]]['gap_c2_minus_c0']:+.4f}")
    return ok, detail


def _p3_map_acc():
    ok = post["map_acc"] >= 0.80
    return ok, f"map_acc_end={post['map_acc']:.6f}"


def _p4_shares():
    details = []
    ok = True
    for color in ("share0", "share1", "share2"):
        d = abs(post[color] - pre[color])
        band_ok = d < 0.04
        if not band_ok:
            ok = False
        details.append(f"{color}:|delta|={d:.6f} ({'ok' if band_ok else 'FAIL'})")
    return ok, "; ".join(details)


def _p4_conviction():
    d = abs(post["conviction"] - pre["conviction"])
    ok = d < 0.04
    return ok, f"|delta_conviction|={d:.6f}"


def _spine_integrity():
    expected_after = ep.age_before + EPOCHS * CHUNK  # 18700
    age_ok = (
        reload.age_steps == ep.age_after == expected_after
        and ep.age_after - ep.age_before == EPOCHS * CHUNK
    )
    return age_ok, (
        f"reload.age={reload.age_steps}  ep.age_after={ep.age_after}  "
        f"ep.age_before={ep.age_before}  delta={ep.age_after - ep.age_before}  "
        f"expected_after={expected_after}"
    )


check("P1-gap-widens", _p1_gap_widens)
check("P2-no-flip", _p2_no_flip)
check("P3-map-accuracy-end", _p3_map_acc)
check("P4-share-stability", _p4_shares)
check("P4-conviction-stability", _p4_conviction)
check("SPINE-age-18700-delta-6000", _spine_integrity)

# ---------------------------------------------------------------------------
# Print property check results
# ---------------------------------------------------------------------------

print("--- PROPERTY CHECKS ---")
failed_names: list[str] = []
for name, passed, detail in checks:
    verdict = "PASS" if passed else "FAIL"
    print(f"  {verdict}  {name}: {detail}")
    if not passed:
        failed_names.append(name)

print()

# ---------------------------------------------------------------------------
# Falsifier map
# ---------------------------------------------------------------------------

f1_fired = (
    "P1-gap-widens" in failed_names
    and "P2-no-flip" not in failed_names        # no flip, but gap narrowed
)
f2_fired = "P2-no-flip" in failed_names         # any flip at any checkpoint
f3_fired = "P3-map-accuracy-end" in failed_names
f4_fired = (
    "P4-share-stability" in failed_names
    or "P4-conviction-stability" in failed_names
)
spine_failed = "SPINE-age-18700-delta-6000" in failed_names

print("--- FALSIFIER MAP ---")
if spine_failed:
    print("  SPINE INTEGRITY FAILED: episode manager did not save correctly or "
          "reload integrity check failed. Halt.")
if f1_fired:
    print("  F1 FIRED: net narrowing without flip — something systematic favors "
          "color 0 against world composition; per-color gate asymmetry is the "
          "prime suspect. Diagnose before any further margin claims.")
if f2_fired:
    first_flip = flip_checkpoints[0]
    print(f"  F2 FIRED: NATURAL FLIP at checkpoint {first_flip} "
          f"(gap_c2_minus_c0={snapshots[first_flip]['gap_c2_minus_c0']:+.4f}) — "
          "stochastic opinion flips are real at this scale; ties Exp 67 "
          "ambivalence law to the spine's own life. Spine keeps the flip.")
if f3_fired:
    print("  F3 FIRED: map_accuracy(end) < 0.80 — the healing claim from Exp 75 "
          "does not hold over triple epochs.")
if f4_fired:
    print("  F4 FIRED: value-core drift exceeds the scaled band — the Exp 75 "
          "stability claim does not extend to triple epochs.")
if not (f1_fired or f2_fired or f3_fired or f4_fired or spine_failed):
    print("  No falsifiers fired.")
print()

# ---------------------------------------------------------------------------
# Final verdict line
# ---------------------------------------------------------------------------

if spine_failed:
    print("EXP76: HALT (spine integrity)")
    sys.exit(1)

# Base verdict
if f2_fired:
    first_flip = flip_checkpoints[0]
    verdict = f"EXP76: F2 — NATURAL FLIP at checkpoint {first_flip}"
elif f1_fired:
    verdict = "EXP76: F1 — systematic narrowing, diagnose"
else:
    verdict = "EXP76: ECOLOGY-LOCKED (gap widened, no flip)"

# Append additional falsifier tags
if f3_fired:
    verdict += " + F3"
if f4_fired:
    verdict += " + F4"

print(verdict)
