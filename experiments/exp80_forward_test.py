"""Exp 80 — the forward test: the accrual law predicts the spine's future, pre-registered.

Exp 77 validated the accrual law retrodictively (it explained Exp 76's flip after the
fact). This experiment makes the law predict FORWARD: from the committed snapshot (age
18700, hash 52f6e814bfe6, gap c2-c0 = -32.06, favorite 0), the law's drift rate
(R(2)-R(0))*steps ~ -43.8 per 6000 steps forecasts the gap after one more 6000-step
epoch at about -76. A second-order prediction comes from the non-decay arithmetic: the
foreign (scar) count mass is FIXED while pure new counts accumulate, so column entropies
fall, both colors' gates rise, and the per-color gate gap R(0)-R(2) should NARROW over
the epoch (scar dilution ~ 1/total-counts).

Protocol: measure R(c) analytically (read-only) from the loaded spine; one checkpointed
episode live(6000); measure R(c) and the value profile again. The episode manager saves
the spine; the snapshot is committed atomically. Single deterministic run (exact-number
standard; resume-from-snapshot is the reproducibility unit).

Predeclared predictions:
  P1 (forward drift): gap_c2_minus_c0 at age 24700 lands in [-116, -36] (center -76,
     band +-40 from Exp 76's observed oscillation amplitude).
  P2 (consolidation): favorite == 0 at age 24700.
  P3 (scar dilution): (R(0) - R(2)) at age 24700 is SMALLER than at age 18700 (the gate
     gap narrows as fixed scar mass dilutes under growing pure counts).
Falsifiers:
  F1 = P1 fails toward zero or a re-cross -> the accrual law does NOT hold forward on
     the spine (a major negative for Exp 77's law; halt the thread for diagnosis).
  F2 = P3 fails (gate gap widens) -> the dilution arithmetic is wrong; the scars are not
     behaving as fixed mass under growing totals; diagnose.
  (P2 failing with P1 passing is possible only near the band's upper edge; report as
   borderline consolidation.)
Provided priors declared: nothing new — the spine's world, live() mechanics, analytic
gate readouts. vela untouched.
"""
from __future__ import annotations

import sys

import numpy as np

from active_loop.creature import Creature
from active_loop.checkpoint import mirro_episode, MIRRO_DIR

# ---------------------------------------------------------------------------
# Constants
# NOTE: mutates and saves the spine; commit snapshot atomically.
# ---------------------------------------------------------------------------

EPOCH = 6000
EXPECT_START_AGE = 18700
EXPECT_START_HASH = "52f6e814bfe6"
BAND = (-116.0, -36.0)


# ---------------------------------------------------------------------------
# Analytic gate-rate helper
# ---------------------------------------------------------------------------

def rates(c: Creature) -> dict:
    """Compute per-color analytic accrual rates from the committed A_hat.

    Returns dict with keys:
      R0, R1, R2   — per-color accrual rate (sum of w(s) for cells of that color / 25)
      mean_gate_0, mean_gate_1, mean_gate_2 — mean gate per color
    """
    A_hat = c._A_hat()                                     # shape (n_colors, n_cells)
    H_cells = -np.sum(A_hat * np.log(A_hat + 1e-12), axis=0)  # (n_cells,)
    w_cells = np.exp(-H_cells)                             # (n_cells,)
    cmap = np.array(c.world.cmap)
    n_cells = c.world.n_cells                              # 25
    n_colors = c.world.n_colors                            # 3

    result: dict = {}
    R_vals = np.zeros(n_colors)
    for color in range(n_colors):
        mask = cmap == color
        w_c = w_cells[mask]
        n_c = int(mask.sum())
        R_c = float(w_c.sum()) / n_cells
        R_vals[color] = R_c
        result[f"R{color}"] = R_c
        result[f"mean_gate_{color}"] = float(w_c.mean()) if n_c > 0 else 0.0

    return result


# ---------------------------------------------------------------------------
# Signed gap helper: value_counts[2] - value_counts[0]
# ---------------------------------------------------------------------------

def gap(c: Creature) -> float:
    return float(c.value_counts[2] - c.value_counts[0])


# ---------------------------------------------------------------------------
# Property-check harness
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
# Main — spine episode
# ---------------------------------------------------------------------------

print("Exp 80 — the forward test")
print()

with mirro_episode("Exp 80") as ep:
    # --- Guard: verify we are starting from the correct committed snapshot ---
    if ep.age_before != EXPECT_START_AGE or ep.hash_before[:12] != EXPECT_START_HASH:
        print(
            f"SNAPSHOT MISMATCH: expected age={EXPECT_START_AGE} "
            f"hash={EXPECT_START_HASH}, "
            f"got age={ep.age_before} hash={ep.hash_before[:12]}"
        )
        sys.exit(1)

    # --- Pre-epoch analytic readouts ---
    r_start = rates(ep.creature)
    start_gap = gap(ep.creature)
    start_fav = ep.creature.favorite()
    r0_r2_start = r_start["R0"] - r_start["R2"]

    print("--- START STATE (age 18700) ---")
    print(f"  gap_c2_minus_c0 : {start_gap:+.4f}")
    print(f"  favorite        : {start_fav}")
    print(f"  R(0)            : {r_start['R0']:.6f}")
    print(f"  R(1)            : {r_start['R1']:.6f}")
    print(f"  R(2)            : {r_start['R2']:.6f}")
    print(f"  R(0)-R(2)       : {r0_r2_start:.6f}")
    print()

    # --- Predicted end gap from accrual law ---
    predicted_end_gap = start_gap + (r_start["R2"] - r_start["R0"]) * EPOCH
    print(f"  predicted_end_gap (law) = {start_gap:+.4f} + "
          f"({r_start['R2']:.6f} - {r_start['R0']:.6f}) * {EPOCH} "
          f"= {predicted_end_gap:+.4f}")
    print()

    # --- Live one epoch ---
    ep.creature.live(EPOCH)

    # --- Post-epoch analytic readouts ---
    r_end = rates(ep.creature)
    end_gap = gap(ep.creature)
    end_fav = ep.creature.favorite()
    r0_r2_end = r_end["R0"] - r_end["R2"]

    print("--- END STATE (age 24700) ---")
    print(f"  gap_c2_minus_c0 : {end_gap:+.4f}")
    print(f"  favorite        : {end_fav}")
    print(f"  R(0)            : {r_end['R0']:.6f}")
    print(f"  R(1)            : {r_end['R1']:.6f}")
    print(f"  R(2)            : {r_end['R2']:.6f}")
    print(f"  R(0)-R(2)       : {r0_r2_end:.6f}")
    print()

# Context manager has saved the spine on exit; ep.report() was already printed by it.

# --- Predicted vs actual ---
actual_end_gap = end_gap
prediction_error = actual_end_gap - predicted_end_gap
print()
print("--- PREDICTED VS ACTUAL ---")
print(f"  predicted_end_gap : {predicted_end_gap:+.4f}")
print(f"  actual_end_gap    : {actual_end_gap:+.4f}")
print(f"  error (actual-predicted) : {prediction_error:+.4f}")
print()

# ---------------------------------------------------------------------------
# Integrity-verified reload
# ---------------------------------------------------------------------------

reload = Creature.load(MIRRO_DIR)

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------


def _p1_forward_drift():
    ok = BAND[0] <= actual_end_gap <= BAND[1]
    return ok, (
        f"gap_c2_minus_c0={actual_end_gap:+.4f}  "
        f"band=[{BAND[0]:.1f}, {BAND[1]:.1f}]  "
        f"in_band={ok}"
    )


def _p2_consolidation():
    ok = end_fav == 0
    return ok, f"favorite_at_24700={end_fav}  expected=0"


def _p3_scar_dilution():
    ok = r0_r2_end < r0_r2_start
    return ok, (
        f"R(0)-R(2) start={r0_r2_start:.6f}  end={r0_r2_end:.6f}  "
        f"narrowed={ok}"
    )


def _spine_integrity():
    expected_after = EXPECT_START_AGE + EPOCH  # 24700
    age_ok = (
        reload.age_steps == ep.age_after == expected_after
        and ep.age_after - ep.age_before == EPOCH
    )
    return age_ok, (
        f"reload.age={reload.age_steps}  ep.age_after={ep.age_after}  "
        f"ep.age_before={ep.age_before}  delta={ep.age_after - ep.age_before}  "
        f"expected_after={expected_after}"
    )


check("P1-forward-drift", _p1_forward_drift)
check("P2-consolidation", _p2_consolidation)
check("P3-scar-dilution", _p3_scar_dilution)
check("SPINE-age-24700-delta-6000", _spine_integrity)

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

p1_passed = "P1-forward-drift" not in failed_names
p2_passed = "P2-consolidation" not in failed_names
p3_passed = "P3-scar-dilution" not in failed_names
spine_failed = "SPINE-age-24700-delta-6000" in failed_names

f1_fired = not p1_passed
f2_fired = not p3_passed

print("--- FALSIFIER MAP ---")
if spine_failed:
    print("  SPINE INTEGRITY FAILED: episode manager did not save correctly or "
          "reload integrity check failed. Halt.")
if f1_fired:
    print("  F1 FIRED: P1 fails — gap_c2_minus_c0 outside band "
          f"[{BAND[0]:.1f}, {BAND[1]:.1f}]; the accrual law does NOT hold forward "
          "on the spine. Halt this thread for diagnosis.")
if f2_fired:
    print("  F2 FIRED: P3 fails — gate gap R(0)-R(2) widened rather than narrowed; "
          "the dilution arithmetic is wrong. The scars are not behaving as fixed mass "
          "under growing totals; diagnose.")
if p1_passed and not p2_passed:
    print("  NOTE: P2 failed with P1 passing — borderline consolidation "
          "(gap landed near upper edge of band; favorite did not hold at 0).")
if not (f1_fired or f2_fired or spine_failed):
    print("  No falsifiers fired.")
print()

# ---------------------------------------------------------------------------
# Final verdict line
# ---------------------------------------------------------------------------

if spine_failed:
    print("EXP80: HALT (spine integrity)")
    sys.exit(1)

if f1_fired and f2_fired:
    print("EXP80: F1 — law fails forward, HALT THREAD + F2 — dilution arithmetic wrong")
elif f1_fired:
    print("EXP80: F1 — law fails forward, HALT THREAD")
elif f2_fired:
    print("EXP80: F2 — dilution arithmetic wrong")
else:
    print("EXP80: LAW HOLDS FORWARD (P1+P2+P3)")
