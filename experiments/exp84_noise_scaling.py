"""Exp 84 — the noise model: does the drift's sigma scale as sqrt(t), and can the law
ever forecast?

Exp 83 measured sigma = 121 per 6000 steps (4x the expected drift) and concluded
per-epoch forecasting is uninformative — ASSUMING sigma grows as sqrt(t) while the drift
mean grows linearly-then-decaying. That scaling assumption is the load-bearing,
unverified piece: if sigma grows slower, a forecast horizon exists; if sqrt(t) or faster,
with dilution decaying the rate, the law can never call an individual's future. This
experiment measures sigma at three horizons from the same start state (vela@12750,
git-recovered as Exp 82/83): T in {1500, 6000, 24000}, 12 counterfactual branches each
(explicit action seeds 4000+i, disjoint per horizon).

Predeclared predictions:
  P1 (sqrt-t scaling): sigma(6000)/sigma(1500) in [1.5, 2.7] AND
     sigma(24000)/sigma(6000) in [1.5, 2.7] (sqrt(4)=2 with +-35 percent for n=12
     std-estimation error).
  P2 (the horizon verdict; LOW confidence, stated): SNR(24000) = |mean D(24000)| /
     sigma(24000) < 1.0 — even at 4x Exp 83's horizon, the forecast stays uninformative
     (the dilution arithmetic decays the rate while noise keeps growing).
Falsifiers:
  F1 = either ratio outside [1.5, 2.7] -> the noise is NOT sqrt(t) (walk autocorrelation
     structure matters); the noise model needs a different form — report the measured
     exponent estimate log(sigma ratio)/log(4).
  F2 = P2 fails (SNR(24000) >= 1.0) -> a usable forecast horizon EXISTS after all;
     Exp 83's "may never arrive" is corrected to a measured horizon estimate.
Either falsifier is itself the finding; this is a measurement experiment with banded
expectations. In-memory deepcopies only; neither committed line touched; scratch snapshot
untracked and derivable.
"""
from __future__ import annotations

import copy
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from active_loop.creature import Creature

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMIT_BEFORE = "e7220c1~1"       # parent of the Exp 81 commit = pre-epoch snapshot
EXPECT_START_AGE = 12750
EXPECT_START_HASH = "875ac30d715a"

HORIZONS = [1500, 6000, 24000]
N = 12

# Seeds: for horizon index h (0,1,2), seeds = [4000 + 100*h + i for i in range(12)]
# This gives disjoint seed sets: [4000..4011], [4100..4111], [4200..4211]
SEEDS = {
    T: [4000 + 100 * h + i for i in range(N)]
    for h, T in enumerate(HORIZONS)
}

# ---------------------------------------------------------------------------
# Step 1: recover pre-epoch snapshot from git history (same pattern as Exp 82/83)
# ---------------------------------------------------------------------------

scratch = Path("experiments/outputs/exp84_pre_snapshot")
scratch.mkdir(parents=True, exist_ok=True)

print("Exp 84 — noise scaling: does sigma ~ sqrt(t)?")
print()
print(f"Recovering pre-epoch vela snapshot from {COMMIT_BEFORE}...")
print(f"  (derivable via: git show {COMMIT_BEFORE}:creature/state/vela/{{manifest.json,arrays.npz}})")
print(f"  scratch dir: {scratch}  [untracked; not committed — derivable from git history]")
print()

for fname in ["manifest.json", "arrays.npz"]:
    result = subprocess.run(
        ["git", "show", f"{COMMIT_BEFORE}:creature/state/vela/{fname}"],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"ERROR: git show failed for {fname}: {result.stderr.decode()}")
        sys.exit(1)
    (scratch / fname).write_bytes(result.stdout)

# Load and verify snapshot
v = Creature.load(scratch)
print(f"Loaded vela: age={v.age_steps}  hash={v._state_hash()[:12]}")

if v.age_steps != EXPECT_START_AGE or v._state_hash()[:12] != EXPECT_START_HASH:
    print(
        f"SNAPSHOT MISMATCH: expected age={EXPECT_START_AGE} hash={EXPECT_START_HASH}, "
        f"got age={v.age_steps} hash={v._state_hash()[:12]}"
    )
    sys.exit(1)

print()

# ---------------------------------------------------------------------------
# Step 2: run counterfactual branches for each horizon
# ---------------------------------------------------------------------------

D_by_horizon: dict[int, list[float]] = {}

for T in HORIZONS:
    seeds = SEEDS[T]
    print(f"Running {N} branches at T={T} (seeds {seeds[0]}..{seeds[-1]})...")
    D_values: list[float] = []
    for s in seeds:
        b = copy.deepcopy(v)
        d0 = float(b.value_counts[2] - b.value_counts[0])
        b.live(T, seed=s)
        d_end = float(b.value_counts[2] - b.value_counts[0])
        D_k = d_end - d0
        D_values.append(D_k)
        print(f"  T={T}  seed={s}  D={D_k:+.4f}")
    D_by_horizon[T] = D_values
    print()

# ---------------------------------------------------------------------------
# Step 3: per-horizon statistics
# ---------------------------------------------------------------------------

stats: dict[int, dict] = {}

print("--- SUMMARY TABLE ---")
print(f"  {'T':>6}  {'mean(D)':>10}  {'std(D)':>10}  {'SNR':>8}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")

for T in HORIZONS:
    arr = np.array(D_by_horizon[T])
    mean_D = float(np.mean(arr))
    std_D = float(np.std(arr, ddof=1))
    snr = abs(mean_D) / std_D if std_D > 0.0 else float("inf")
    stats[T] = {"mean": mean_D, "std": std_D, "snr": snr}
    print(f"  {T:>6}  {mean_D:>+10.4f}  {std_D:>10.4f}  {snr:>8.4f}")

print()

# ---------------------------------------------------------------------------
# Step 4: scaling ratios and exponent estimates
# ---------------------------------------------------------------------------

std_1500 = stats[1500]["std"]
std_6000 = stats[6000]["std"]
std_24000 = stats[24000]["std"]

r1 = std_6000 / std_1500    # should be ~sqrt(4) = 2 if sqrt-t scaling
r2 = std_24000 / std_6000   # should be ~sqrt(4) = 2 if sqrt-t scaling

# exponent: sigma ~ t^alpha  =>  r = (T2/T1)^alpha  =>  alpha = log(r)/log(T2/T1)
# T ratio is 6000/1500 = 4 and 24000/6000 = 4 for both steps
e1 = math.log(r1) / math.log(4) if r1 > 0 else float("nan")
e2 = math.log(r2) / math.log(4) if r2 > 0 else float("nan")

print("--- SCALING RATIOS ---")
print(f"  sigma(6000) / sigma(1500)   = {r1:.4f}   (sqrt-t predicts ~2.00; band [1.5, 2.7])")
print(f"  sigma(24000) / sigma(6000)  = {r2:.4f}   (sqrt-t predicts ~2.00; band [1.5, 2.7])")
print(f"  exponent estimate e1 = log(r1)/log(4) = {e1:.4f}  (sqrt-t => 0.50)")
print(f"  exponent estimate e2 = log(r2)/log(4) = {e2:.4f}  (sqrt-t => 0.50)")
print()

# ---------------------------------------------------------------------------
# Step 5: property checks
# ---------------------------------------------------------------------------

RATIO_LO, RATIO_HI = 1.5, 2.7

P1_r1_ok = RATIO_LO <= r1 <= RATIO_HI
P1_r2_ok = RATIO_LO <= r2 <= RATIO_HI
P1_ok = P1_r1_ok and P1_r2_ok
P2_ok = stats[24000]["snr"] < 1.0

snr_24000 = stats[24000]["snr"]

print("--- PROPERTY CHECKS ---")
print(f"  {'PASS' if P1_r1_ok else 'FAIL'}  P1a: r1={r1:.4f} in [{RATIO_LO}, {RATIO_HI}]")
print(f"  {'PASS' if P1_r2_ok else 'FAIL'}  P1b: r2={r2:.4f} in [{RATIO_LO}, {RATIO_HI}]")
print(f"  {'PASS' if P1_ok else 'FAIL'}  P1 (both ratios): {'PASS' if P1_ok else 'FAIL'}")
print(f"  {'PASS' if P2_ok else 'FAIL'}  P2: SNR(24000)={snr_24000:.4f} < 1.0 "
      f"-> forecast {'uninformative' if P2_ok else 'INFORMATIVE (horizon exists)'}")
print()

# ---------------------------------------------------------------------------
# Step 6: falsifier map
# ---------------------------------------------------------------------------

print("--- FALSIFIER MAP ---")

F1_fired = not P1_ok
F2_fired = not P2_ok

if not F1_fired:
    print(f"  F1 did not fire (P1 PASS — both ratios in [{RATIO_LO}, {RATIO_HI}]; noise is sqrt-t).")
else:
    bad_parts = []
    if not P1_r1_ok:
        bad_parts.append(f"r1={r1:.4f} (exponent e1={e1:.2f})")
    if not P1_r2_ok:
        bad_parts.append(f"r2={r2:.4f} (exponent e2={e2:.2f})")
    print(f"  F1 FIRED: ratio(s) outside [{RATIO_LO}, {RATIO_HI}]: {', '.join(bad_parts)}")
    print(f"    -> noise is NOT sqrt(t); walk autocorrelation structure matters;")
    print(f"       noise model needs a different form (exponents: e1={e1:.2f}, e2={e2:.2f}).")

if not F2_fired:
    print(f"  F2 did not fire (P2 PASS — SNR(24000)={snr_24000:.4f} < 1.0; no forecast horizon yet).")
else:
    print(f"  F2 FIRED: SNR(24000)={snr_24000:.4f} >= 1.0 -> a usable forecast horizon EXISTS;")
    print(f"    Exp 83's 'may never arrive' is corrected to a measured horizon estimate.")

print()

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

if not F1_fired and not F2_fired:
    print("EXP84: NOISE MODEL CONFIRMED (sqrt-t; no forecast horizon)")
elif F1_fired and not F2_fired:
    print(f"EXP84: F1 — exponent(s) {e1:.2f}/{e2:.2f}, not sqrt-t")
elif not F1_fired and F2_fired:
    print(f"EXP84: F2 — forecast horizon exists (SNR(24000)={snr_24000:.2f})")
else:
    print(f"EXP84: F1+F2 — exponent(s) {e1:.2f}/{e2:.2f}, not sqrt-t; AND forecast horizon exists (SNR(24000)={snr_24000:.2f})")
