"""Exp 81 — the law on the other line: predicting vela's future from vela's state.

Exp 80 validated the accrual law forward on mirro. This is the out-of-individual
generalization: the SAME analytic procedure, applied with zero tuning to the other
committed line — vela (the Exp 63 peer spine: mirro's fork raised in the REVERSED world,
untouched since Exp 65, age 12750, hash 875a c30d715a-prefixed) — must predict vela's
preference dynamics from vela's own committed state.

Disclosed read-only pre-step: vela's gates give R(0)=0.174749, R(1)=0.157858,
R(2)=0.169871 (all lower than mirro's ~0.20 — the immigrant carries TWO worlds' scar
mass); gap c2-c0 = +62.03, favorite 2; predicted drift (R(2)-R(0))*6000 = -29.3.
The law therefore makes a call DIFFERENT from mirro's: vela's favorite SURVIVES this
epoch (predicted end gap +32.7, no flip), while drifting toward an eventual flip.

Predeclared predictions:
  P1 (forward drift): gap_c2_minus_c0 at age 18750 lands in [-7.3, +72.7] (center
     +32.7, band +-40 as in Exp 80).
  P2 (no-flip call, separate from P1): favorite == 2 at age 18750.
  P3 (scar dilution): R(0)-R(2) at age 18750 is SMALLER than 0.004878 (the start gap).
Falsifiers:
  F1 = P1 fails -> the law does NOT generalize out-of-individual; halt for diagnosis.
  F2 = P3 fails -> dilution arithmetic wrong on the immigrant's heavier scars.
  (P1 passing with P2 failing = marginal early flip inside the band: MIXED, reported.)
Provided priors declared: nothing new — vela's own world (the reversed cmap), live()
mechanics, analytic gate readouts. This experiment advances VELA's continuous line via
the generic checkpointed episode (state_dir=creature/state/vela); the updated vela
snapshot is committed atomically. mirro is untouched (not loaded). Single deterministic
run (exact-number standard; resume-from-snapshot is the reproducibility unit).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from active_loop.creature import Creature
from active_loop.checkpoint import mirro_episode

# ---------------------------------------------------------------------------
# Constants
# NOTE: advances and saves VELA's line; commit snapshot atomically.
# ---------------------------------------------------------------------------

VELA_DIR = Path("creature/state/vela")
EPOCH = 6000
EXPECT_START_AGE = 12750
EXPECT_START_HASH = "875ac30d715a"
BAND = (-7.3, 72.7)
START_GATE_GAP = 0.004878


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
# Main — vela episode
# ---------------------------------------------------------------------------

print("Exp 81 — the law on the other line")
print()

with mirro_episode("Exp 81", state_dir=VELA_DIR) as ep:
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

    print("--- START STATE (age 12750) ---")
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

    print("--- END STATE (age 18750) ---")
    print(f"  gap_c2_minus_c0 : {end_gap:+.4f}")
    print(f"  favorite        : {end_fav}")
    print(f"  R(0)            : {r_end['R0']:.6f}")
    print(f"  R(1)            : {r_end['R1']:.6f}")
    print(f"  R(2)            : {r_end['R2']:.6f}")
    print(f"  R(0)-R(2)       : {r0_r2_end:.6f}")
    print()

# Context manager has saved vela's spine on exit; ep.report() was already printed by it.

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

reload = Creature.load(VELA_DIR)

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


def _p2_no_flip():
    ok = end_fav == 2
    return ok, f"favorite_at_18750={end_fav}  expected=2"


def _p3_scar_dilution():
    ok = r0_r2_end < START_GATE_GAP
    return ok, (
        f"R(0)-R(2) start={r0_r2_start:.6f}  end={r0_r2_end:.6f}  "
        f"threshold={START_GATE_GAP:.6f}  below_threshold={ok}"
    )


def _spine_integrity():
    expected_after = EXPECT_START_AGE + EPOCH  # 18750
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
check("P2-no-flip", _p2_no_flip)
check("P3-scar-dilution", _p3_scar_dilution)
check("SPINE-age-18750-delta-6000", _spine_integrity)

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
p2_passed = "P2-no-flip" not in failed_names
p3_passed = "P3-scar-dilution" not in failed_names
spine_failed = "SPINE-age-18750-delta-6000" in failed_names

f1_fired = not p1_passed
f2_fired = not p3_passed

print("--- FALSIFIER MAP ---")
if spine_failed:
    print("  SPINE INTEGRITY FAILED: episode manager did not save correctly or "
          "reload integrity check failed. Halt.")
if f1_fired:
    print("  F1 FIRED: P1 fails — gap_c2_minus_c0 outside band "
          f"[{BAND[0]:.1f}, {BAND[1]:.1f}]; the law does NOT generalize "
          "out-of-individual. Halt for diagnosis.")
if f2_fired:
    print("  F2 FIRED: P3 fails — gate gap R(0)-R(2) not below start threshold "
          f"{START_GATE_GAP:.6f}; dilution arithmetic wrong on the immigrant's "
          "heavier scars; diagnose.")
if p1_passed and not p2_passed:
    print("  NOTE: P2 failed with P1 passing — marginal early flip inside the band; "
          "MIXED result (favorite did not hold at 2).")
if not (f1_fired or f2_fired or spine_failed):
    print("  No falsifiers fired.")
print()

# ---------------------------------------------------------------------------
# Final verdict line
# ---------------------------------------------------------------------------

if spine_failed:
    print("EXP81: HALT (spine integrity)")
    sys.exit(1)

if f1_fired and f2_fired:
    print("EXP81: F1 — law fails out-of-individual, HALT + F2 — dilution wrong on heavy scars")
elif f1_fired:
    print("EXP81: F1 — law fails out-of-individual, HALT")
elif f2_fired:
    print("EXP81: F2 — dilution wrong on heavy scars")
elif p1_passed and not p2_passed:
    print("EXP81: MIXED — in-band early flip")
else:
    print("EXP81: LAW GENERALIZES (P1+P2+P3)")
