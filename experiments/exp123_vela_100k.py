"""Exp 123 — idle-mode life epoch: vela crosses one hundred thousand steps (age 96750 ->
102750).

Declared idle-mode (Exp 92), vela's turn — and the second 100k milestone: the peer line,
born as a fork at mirro@10700 and raised in the mirror world, passes 100,000 steps of its
own continuous experience (inheriting 10,700 from the trunk; ~92,000 lived on its own
line through immigration, five flips including one three-sigma lurch, and its long
settling). The milestone is noted, not celebrated into the bands. vela enters at +423.8
(deep bin) -> recorded call: HOLD (razor books open per Exp 117).

Predeclared:
  P1 (knowledge, one-sided): map_accuracy at 102750 >= 0.96 (currently 1.000).
  P2 (value core): |delta share_c| < 0.04 each and |delta conviction| < 0.04.
  P3 (noise-calibrated forecast): end gap c2-c0 within predicted +- 302.5 (the
     registered band; deep tails read against Exp 112).
  P4 (recorded depth call): predicted HOLD; outcome recorded (razor books open per
     Exp 117).
Falsifiers: F1 knowledge degraded; F2 value-core instability; F3 a >2.5-sigma event
(interpreted against Exp 112's state-dependent sigma finding).
Single deterministic run; vela's line advances, snapshot committed atomically. mirro
untouched (not loaded).
"""
from __future__ import annotations

import sys

import numpy as np

from active_loop.creature import Creature
from active_loop.checkpoint import mirro_episode

# ---------------------------------------------------------------------------
# Constants
# NOTE: mutates and saves the spine; commit snapshot atomically.
# ---------------------------------------------------------------------------

VELA_DIR = "creature/state/vela"
EPOCH = 6000
EXPECT_START_AGE = 96750
SIGMA = 121.0          # Exp 83 noise model
N_SIGMA = 2.5
BAND_HALF = N_SIGMA * SIGMA  # 302.5


# ---------------------------------------------------------------------------
# Analytic gate-rate helper (from exp80 pattern)
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
# Profile helper (from exp75 pattern)
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
        gap_c2c0=float(c.value_counts[2] - c.value_counts[0]),
        total_mass=float(tot),
    )


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

print("Exp 123 — idle-mode life epoch (vela age 96750 -> 102750)")
print()

with mirro_episode("Exp 123", state_dir=VELA_DIR) as ep:
    # --- Guard: verify we are starting from the correct committed snapshot ---
    if ep.age_before != EXPECT_START_AGE:
        print(
            f"SNAPSHOT MISMATCH: expected age={EXPECT_START_AGE}, "
            f"got age={ep.age_before}"
        )
        print(f"  hash_before={ep.hash_before[:12]}")
        sys.exit(1)

    print(f"  hash_before = {ep.hash_before[:12]}")
    print()

    # --- Pre-epoch profile and rates ---
    pre = profile(ep.creature)
    r_start = rates(ep.creature)
    start_gap = pre["gap_c2c0"]
    start_fav = pre["favorite"]
    start_map_acc = pre["map_acc"]

    print("--- START PROFILE (age 96750) ---")
    for k, v in pre.items():
        if isinstance(v, float):
            print(f"  {k:<18}: {v:.6f}")
        else:
            print(f"  {k:<18}: {v}")
    print()
    print(f"  R(0)            : {r_start['R0']:.6f}")
    print(f"  R(1)            : {r_start['R1']:.6f}")
    print(f"  R(2)            : {r_start['R2']:.6f}")
    print()

    # --- Noise-calibrated forecast (P3) ---
    predicted_end_gap = start_gap + (r_start["R2"] - r_start["R0"]) * EPOCH
    band_lo = predicted_end_gap - BAND_HALF
    band_hi = predicted_end_gap + BAND_HALF
    print(f"  predicted_end_gap = {start_gap:+.4f} + "
          f"({r_start['R2']:.6f} - {r_start['R0']:.6f}) * {EPOCH} "
          f"= {predicted_end_gap:+.4f}")
    print(f"  P3 band (2.5-sigma = {BAND_HALF:.1f}): [{band_lo:+.4f}, {band_hi:+.4f}]")
    print()

    # --- Live one epoch ---
    ep.creature.live(EPOCH)

    # --- Post-epoch profile and rates ---
    post = profile(ep.creature)
    r_end = rates(ep.creature)
    end_gap = post["gap_c2c0"]
    end_fav = post["favorite"]

    print("--- END PROFILE (age 102750) ---")
    for k, v in post.items():
        if isinstance(v, float):
            print(f"  {k:<18}: {v:.6f}")
        else:
            print(f"  {k:<18}: {v}")
    print()
    print(f"  R(0)            : {r_end['R0']:.6f}")
    print(f"  R(1)            : {r_end['R1']:.6f}")
    print(f"  R(2)            : {r_end['R2']:.6f}")
    print()

# Context manager has saved the spine on exit.

# --- Predicted vs actual ---
prediction_error = end_gap - predicted_end_gap
print()
print("--- PREDICTED VS ACTUAL ---")
print(f"  predicted_end_gap : {predicted_end_gap:+.4f}")
print(f"  actual_end_gap    : {end_gap:+.4f}")
print(f"  error (actual-predicted) : {prediction_error:+.4f}")
print(f"  |error| / sigma   : {abs(prediction_error) / SIGMA:.3f}")
print()

# ---------------------------------------------------------------------------
# Integrity-verified reload
# ---------------------------------------------------------------------------

from pathlib import Path
reload = Creature.load(Path(VELA_DIR))

# ---------------------------------------------------------------------------
# Delta table
# ---------------------------------------------------------------------------

print("--- DELTA TABLE ---")
# P1 uses absolute one-sided threshold: map_acc at 102750 >= 0.96
BANDS = {
    "share0":     ("< 0.04", lambda d: abs(d) < 0.04),
    "share1":     ("< 0.04", lambda d: abs(d) < 0.04),
    "share2":     ("< 0.04", lambda d: abs(d) < 0.04),
    "conviction": ("< 0.04", lambda d: abs(d) < 0.04),
    "map_acc":    (">= 0.96 (absolute one-sided)   ", lambda _: post["map_acc"] >= 0.96),
}

print(f"  {'component':<18}  {'before':>10}  {'after':>10}  {'delta':>10}  {'band':<34}  ok?")
print("  " + "-" * 94)

for comp, (band_str, band_fn) in BANDS.items():
    before_val = pre[comp]
    after_val = post[comp]
    delta = after_val - before_val
    ok = band_fn(delta)
    ok_str = "PASS" if ok else "FAIL"
    print(
        f"  {comp:<18}  {before_val:>10.6f}  {after_val:>10.6f}  "
        f"{delta:>+10.6f}  {band_str:<34}  {ok_str}"
    )

print()
# Favorite: recorded depth call (P4) — predicted HOLD, deep bin entry at +423.8
held_as_called = end_fav == start_fav
p4_outcome = "HELD as called" if held_as_called else "FLIPPED against the call"
print(f"  favorite: {start_fav} -> {end_fav}  (RECORDED depth call: HOLD — deep bin |gap| >= 120)")
print(f"  P4 recorded-call outcome: {p4_outcome} (recorded; razor books open per Exp 117)")
print()

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------


def _p1_map_accuracy():
    threshold = 0.96
    ok = post["map_acc"] >= threshold
    return ok, (
        f"map_acc_at_102750={post['map_acc']:.6f}  "
        f"threshold={threshold:.6f} (absolute one-sided)  ok={ok}"
    )


def _p2_shares():
    details = []
    ok = True
    for color_key in ("share0", "share1", "share2"):
        d = abs(post[color_key] - pre[color_key])
        band_ok = d < 0.04
        if not band_ok:
            ok = False
        details.append(f"{color_key}:|delta|={d:.6f} ({'ok' if band_ok else 'FAIL'})")
    return ok, "; ".join(details)


def _p2_conviction():
    d = abs(post["conviction"] - pre["conviction"])
    ok = d < 0.04
    return ok, f"|delta_conviction|={d:.6f}  threshold=0.04"


def _p3_forecast():
    ok = band_lo <= end_gap <= band_hi
    return ok, (
        f"end_gap={end_gap:+.4f}  predicted={predicted_end_gap:+.4f}  "
        f"band=[{band_lo:+.4f}, {band_hi:+.4f}]  "
        f"|error|/sigma={abs(prediction_error)/SIGMA:.3f}  in_band={ok}"
    )


def _spine_integrity():
    expected_after = EXPECT_START_AGE + EPOCH  # 102750
    age_ok = (
        reload.age_steps == ep.age_after == expected_after
        and ep.age_after - ep.age_before == EPOCH
    )
    return age_ok, (
        f"reload.age={reload.age_steps}  ep.age_after={ep.age_after}  "
        f"ep.age_before={ep.age_before}  delta={ep.age_after - ep.age_before}  "
        f"expected_after={expected_after}"
    )


check("P1-map-accuracy-102750", _p1_map_accuracy)
check("P2-share-stability", _p2_shares)
check("P2-conviction-stability", _p2_conviction)
check("P3-noise-calibrated-forecast", _p3_forecast)
check("SPINE-age-102750-delta-6000", _spine_integrity)

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

f1_fired = "P1-map-accuracy-102750" in failed_names
f2_fired = (
    "P2-share-stability" in failed_names
    or "P2-conviction-stability" in failed_names
)
f3_fired = "P3-noise-calibrated-forecast" in failed_names
spine_failed = "SPINE-age-102750-delta-6000" in failed_names

print("--- FALSIFIER MAP ---")
if spine_failed:
    print("  SPINE INTEGRITY FAILED: episode manager did not save correctly or "
          "reload integrity check failed. Halt.")
if f1_fired:
    print("  F1 FIRED: P1 fails — map_accuracy at 102750 below 0.96; knowledge "
          "degraded — something broke.")
if f2_fired:
    print("  F2 FIRED: P2 fails — value-core instability at a horizon it has "
          "repeatedly held; new finding.")
if f3_fired:
    sigma_count = abs(prediction_error) / SIGMA
    print(f"  F3 FIRED: P3 fails — gap outside 2.5-sigma band "
          f"(|error|={abs(prediction_error):.1f}, {sigma_count:.2f} sigma); "
          "interpreted against Exp 112's state-dependent sigma finding.")
if not (f1_fired or f2_fired or f3_fired or spine_failed):
    print("  No falsifiers fired.")
print()

# ---------------------------------------------------------------------------
# Final verdict line
# ---------------------------------------------------------------------------

if spine_failed:
    print("EXP123: HALT (spine integrity)")
    sys.exit(1)

f_lines = []
if f1_fired:
    f_lines.append("F1 — knowledge degraded")
if f2_fired:
    f_lines.append("F2 — value-core unstable")
if f3_fired:
    sigma_count = abs(prediction_error) / SIGMA
    f_lines.append(f"F3 — forecast miss ({sigma_count:.2f} sigma)")

if f_lines:
    print("EXP123: " + "; ".join(f_lines))
else:
    print("EXP123: EPOCH CLEAN (bands held)")
