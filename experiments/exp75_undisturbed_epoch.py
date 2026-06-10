"""Exp 75 — the undisturbed epoch: is mirro's personality stable when nobody interferes?

Functional-emergence rung 2's open branch (Exp 55 found profile INSTABILITY across ages
that bracketed engineered value reversals and a world-growth event — whether the profile
is stable across an UNDISTURBED epoch was never answered). This experiment also resumes
the spine's continuous life (untouched at age 10700 / hash 21ccb619f063 through the
fork-only social chapter, Exp 64-74) via the checkpointed episode discipline.

The sharp hook (stated before running): mirro's favorite sits on a razor-thin margin —
value shares 0.3623 (color 0) vs 0.3641 (color 2), a gap of ~11 counts in ~6,189 — so by
this program's own ambivalence law (Exp 67: persuadability and lability are gated by the
top-two gap, not age), mirro is itself a near-ambivalent adult, and ~1,300 counts of
natural accrual noise could flip its favorite without any intervention.

Protocol: read-only profile from the committed state -> ep.creature.live(2000) in its own
world (no perturbations, no teaching, no channel — undisturbed) -> profile after; the
episode context manager records checkpoint before/after and saves the spine (the updated
creature/state/mirro/ snapshot is committed atomically with this entry).

Profile dims: value shares (3), conviction, map_accuracy, localize_bits, top-two value
gap in counts (entrenchment).

Predeclared predictions:
  P1 (quantitative stability bands): |delta share_c| < 0.02 for each color; |delta
     conviction| < 0.02; |delta map_accuracy| <= 0.04; localize_bits <= 0.2 after.
  P-FAV (predicted UNCHANGED, explicitly LOW confidence): favorite stays color 2.
  P2 (spine integrity): age 10700 -> 12700; clean save; Creature.load() integrity-verified
     reload; biography gains the episode events.
Falsifiers (each names its branch):
  F1 = any P1 band blown -> the profile is NOT stable under undisturbed living — the
     rung-2 'current-state readout' branch, logged as such (a real finding, not a bug).
  F2 = P-FAV fails (favorite flips) -> the razor-margin favorite is current-state
     readout; ties the ambivalence law (Exp 67) back to the spine itself: mirro is an
     ambivalent adult whose 'opinion' can drift on its own.
  F3 = P2 fails -> halt; spine integrity problem (the episode manager did not save, or
     reload fails integrity).
Provided priors declared: nothing new — the spine's own world, the live() mechanics, and
the profile readouts; no interventions of any kind. Single deterministic run (the spine's
RNG derives from committed state; exact-number standard applies per VALIDATION.md).
vela untouched.
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

EPOCH_STEPS = 2000

# NOTE: this experiment intentionally MUTATES and SAVES the spine (the whole
# point); the snapshot must be committed with the entry.

# ---------------------------------------------------------------------------
# Profile helper
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
        gap_counts=float(sorted_vc[0] - sorted_vc[1]),
        total_mass=float(tot),
    )


# ---------------------------------------------------------------------------
# Property-check harness (copied from exp64 pattern)
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

print("Exp 75 — the undisturbed epoch")
print()

# ---------------------------------------------------------------------------
# Spine episode: load -> measure -> live -> measure -> save
# ---------------------------------------------------------------------------

with mirro_episode("Exp 75") as ep:
    pre = profile(ep.creature)

    print("--- PRE-EPOCH PROFILE (committed state) ---")
    for k, v in pre.items():
        if isinstance(v, float):
            print(f"  {k:<18}: {v:.6f}")
        else:
            print(f"  {k:<18}: {v}")
    print()

    # Undisturbed life — no perturbations, no teaching, no channel
    ep.creature.live(EPOCH_STEPS)

    post = profile(ep.creature)

    print("--- POST-EPOCH PROFILE ---")
    for k, v in post.items():
        if isinstance(v, float):
            print(f"  {k:<18}: {v:.6f}")
        else:
            print(f"  {k:<18}: {v}")
    print()

# Context manager has saved the spine and printed ep.report() on clean exit.
# Print the report again for the log block.
print()
print(ep.report())
print()

# ---------------------------------------------------------------------------
# Integrity-verified reload
# ---------------------------------------------------------------------------

reload = Creature.load(MIRRO_DIR)

# ---------------------------------------------------------------------------
# Delta table
# ---------------------------------------------------------------------------

print("--- DELTA TABLE ---")
BANDS = {
    "share0":       ("< 0.02",   lambda d: abs(d) < 0.02),
    "share1":       ("< 0.02",   lambda d: abs(d) < 0.02),
    "share2":       ("< 0.02",   lambda d: abs(d) < 0.02),
    "conviction":   ("< 0.02",   lambda d: abs(d) < 0.02),
    "map_acc":      ("<= 0.04",  lambda d: abs(d) <= 0.04),
    "localize_bits":("<= 0.2 (post abs)", lambda _: post["localize_bits"] <= 0.2),
}

print(f"  {'component':<18}  {'before':>10}  {'after':>10}  {'delta':>10}  {'band':<24}  ok?")
print("  " + "-" * 84)

for comp, (band_str, band_fn) in BANDS.items():
    before_val = pre[comp]
    after_val = post[comp]
    delta = after_val - before_val
    ok = band_fn(delta)
    ok_str = "PASS" if ok else "FAIL"
    print(
        f"  {comp:<18}  {before_val:>10.6f}  {after_val:>10.6f}  "
        f"{delta:>+10.6f}  {band_str:<24}  {ok_str}"
    )

print()
print(f"  {'favorite':<18}  {pre['favorite']:>10}  {post['favorite']:>10}  "
      f"{'n/a':>10}  {'== 2 (low conf)':24}  "
      f"{'PASS' if post['favorite'] == pre['favorite'] == 2 else 'FLIP'}")
print()

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------


def _p1_shares():
    details = []
    ok = True
    for c in ("share0", "share1", "share2"):
        d = abs(post[c] - pre[c])
        band_ok = d < 0.02
        if not band_ok:
            ok = False
        details.append(f"{c}:|delta|={d:.6f} ({'ok' if band_ok else 'FAIL'})")
    return ok, "; ".join(details)


def _p1_conviction():
    d = abs(post["conviction"] - pre["conviction"])
    ok = d < 0.02
    return ok, f"|delta_conviction|={d:.6f}"


def _p1_map_acc():
    d = abs(post["map_acc"] - pre["map_acc"])
    ok = d <= 0.04
    return ok, f"|delta_map_acc|={d:.6f}"


def _p1_localize():
    ok = post["localize_bits"] <= 0.2
    return ok, f"localize_bits_after={post['localize_bits']:.6f}"


def _pfav():
    ok = (post["favorite"] == pre["favorite"] == 2)
    return ok, f"pre_fav={pre['favorite']} post_fav={post['favorite']}"


def _p2_age():
    age_ok = (
        reload.age_steps == ep.age_after == 12700
        and ep.age_after - ep.age_before == EPOCH_STEPS
    )
    return age_ok, (
        f"reload.age={reload.age_steps}  ep.age_after={ep.age_after}  "
        f"ep.age_before={ep.age_before}  delta={ep.age_after - ep.age_before}"
    )


check("P1-share-stability", _p1_shares)
check("P1-conviction-stability", _p1_conviction)
check("P1-map-accuracy-stability", _p1_map_acc)
check("P1-localize-bits-after", _p1_localize)
check("P-FAV-favorite-unchanged", _pfav)
check("P2-spine-integrity", _p2_age)

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

p1_names = {"P1-share-stability", "P1-conviction-stability",
            "P1-map-accuracy-stability", "P1-localize-bits-after"}
f1_fired = any(n in failed_names for n in p1_names)
f2_fired = "P-FAV-favorite-unchanged" in failed_names
f3_fired = "P2-spine-integrity" in failed_names

print("--- FALSIFIER MAP ---")
if f3_fired:
    print("  F3 FIRED: P2 spine-integrity failure — episode manager did not save "
          "correctly or reload integrity check failed.")
if f1_fired:
    print("  F1 FIRED: profile is NOT stable under undisturbed living — "
          "rung-2 current-state readout branch.")
if f2_fired:
    print("  F2 FIRED: razor-margin favorite flipped — mirro is an ambivalent adult "
          "whose opinion can drift on its own (ties Exp 67 ambivalence law back to "
          "the spine itself).")
if not (f1_fired or f2_fired or f3_fired):
    print("  No falsifiers fired.")
print()

# ---------------------------------------------------------------------------
# Final verdict line
# ---------------------------------------------------------------------------

if f3_fired:
    print("EXP75: HALT (F3)")
    sys.exit(1)
elif f1_fired:
    print("EXP75: UNSTABLE (F1 branch)")
elif f2_fired and not f1_fired:
    print("EXP75: P1 STABLE, FAVORITE FLIPPED (F2 branch)")
else:
    print("EXP75: STABLE (P1+P-FAV)")
