"""Exp 83 — error bars for the accrual law: the drift distribution under counterfactual walks.

Exp 82 resolved the 3x anomaly as visit noise but estimated the noise scale from a single
draw. This experiment measures the per-epoch drift distribution properly: from the exact
committed state where the false alarm occurred (vela@12750, git-recovered as in Exp 82),
run 20 counterfactual 6000-step epochs that differ ONLY in the action stream (live()'s
documented explicit-seed override, seeds 3000..3019), and measure the realized gap drift
D = (value_counts[2]-value_counts[0])_end - start for each branch.

Predeclared predictions:
  P1 (the law is the expectation): the counterfactual mean of D is within 2 standard
     errors of the analytic forecast -29.27 (frozen gates, uniform visits).
  P2 (noise scale, Exp 82's reading made precise): the sample standard deviation lies in
     [25, 90] — comparable to the mean's magnitude.
  P3 (the false alarm settles): the actually-lived epoch's D = -83.72 lies within
     mean +- 2.5 sigma of the counterfactual distribution (an ordinary draw).
Falsifiers:
  F1 = P1 fails -> the law is BIASED as an expectation (a worse defect than variance);
     re-open the law thread.
  F2 = sigma < 25 -> the lived -83.72 was a genuine outlier and the noise explanation
     fails (re-open Exp 82's diagnosis); sigma > 90 -> per-epoch forecasting at this
     horizon is uninformative (the law is expectation-only at much longer horizons).
  F3 = P3 fails with P1+P2 passing -> the lived draw is anomalous relative to its own
     counterfactual ensemble — seed-specific structure; diagnose.
Notes: counterfactual branches run on in-memory deepcopies of the recovered snapshot
(no fork events, no saves); neither committed line is touched. The 20 branches are
deterministic given their explicit seeds (all reported). The scratch snapshot is
untracked and derivable (git show e7220c1~1:creature/state/vela/...).
"""
from __future__ import annotations

import copy
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

ANALYTIC = -29.2669               # frozen gates, uniform visits (Exp 81 forecast)
LIVED = -83.7173                  # actually-lived epoch's D (Exp 81 / Exp 82 replay)
N = 20
SEEDS = range(3000, 3020)
EPOCH = 6000

# ---------------------------------------------------------------------------
# Step 1: recover pre-epoch snapshot from git history (same pattern as Exp 82)
# ---------------------------------------------------------------------------

scratch = Path("experiments/outputs/exp83_pre_snapshot")
scratch.mkdir(parents=True, exist_ok=True)

print("Exp 83 — error bars for the accrual law")
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
# Step 2: run 20 counterfactual branches
# ---------------------------------------------------------------------------

print(f"Running {N} counterfactual branches (seeds {SEEDS.start}..{SEEDS.stop - 1}, "
      f"{EPOCH} steps each)...")
print()

D_values: list[float] = []
end_favorites: list[int] = []

for k in SEEDS:
    b = copy.deepcopy(v)
    d0 = float(b.value_counts[2] - b.value_counts[0])
    b.live(EPOCH, seed=k)
    d_end = float(b.value_counts[2] - b.value_counts[0])
    D_k = d_end - d0
    fav_k = b.favorite()
    D_values.append(D_k)
    end_favorites.append(fav_k)
    print(f"  seed={k}  D={D_k:+.4f}  end_favorite={fav_k}")

print()

# ---------------------------------------------------------------------------
# Step 3: statistics
# ---------------------------------------------------------------------------

D_arr = np.array(D_values)
mean_D = float(np.mean(D_arr))
std_D = float(np.std(D_arr, ddof=1))       # sample std
SE = std_D / np.sqrt(N)

flip_count = sum(1 for f in end_favorites if f == 0)

print("--- STATISTICS ---")
print(f"  N                  = {N}")
print(f"  mean(D)            = {mean_D:+.4f}")
print(f"  std(D) [ddof=1]    = {std_D:.4f}")
print(f"  SE = std/sqrt(N)   = {SE:.4f}")
print(f"  analytic forecast  = {ANALYTIC:+.4f}")
print(f"  |mean - analytic|  = {abs(mean_D - ANALYTIC):.4f}  (2*SE = {2*SE:.4f})")
print(f"  lived draw D       = {LIVED:+.4f}")
print(f"  |lived - mean|     = {abs(LIVED - mean_D):.4f}  (2.5*std = {2.5*std_D:.4f})")
print(f"  branches flipped to favorite=0: {flip_count}/{N}")
print()

# ---------------------------------------------------------------------------
# Step 4: text histogram of sorted D values
# ---------------------------------------------------------------------------

D_sorted = sorted(D_values)
print("--- SORTED D VALUES (text histogram) ---")
d_min = D_sorted[0]
d_max = D_sorted[-1]
bar_width = 40
d_range = d_max - d_min if d_max != d_min else 1.0
for d_val in D_sorted:
    bar_len = int(round((d_val - d_min) / d_range * bar_width))
    bar = "#" * bar_len
    print(f"  {d_val:+8.2f} |{bar}")
print()

# ---------------------------------------------------------------------------
# Step 5: property checks
# ---------------------------------------------------------------------------

P1_ok = abs(mean_D - ANALYTIC) <= 2 * SE
P2_ok = 25.0 <= std_D <= 90.0
P3_ok = abs(LIVED - mean_D) <= 2.5 * std_D

checks = [
    ("P1", P1_ok,
     f"|mean-analytic|={abs(mean_D - ANALYTIC):.4f} <= 2*SE={2*SE:.4f} -> {'PASS' if P1_ok else 'FAIL'}"),
    ("P2", P2_ok,
     f"std={std_D:.4f} in [25, 90] -> {'PASS' if P2_ok else 'FAIL'}"),
    ("P3", P3_ok,
     f"|lived-mean|={abs(LIVED - mean_D):.4f} <= 2.5*std={2.5*std_D:.4f} -> {'PASS' if P3_ok else 'FAIL'}"),
]

print("--- PROPERTY CHECKS ---")
for name, passed, detail in checks:
    verdict = "PASS" if passed else "FAIL"
    print(f"  {verdict}  {name}: {detail}")
print()

# ---------------------------------------------------------------------------
# Step 6: falsifier map
# ---------------------------------------------------------------------------

print("--- FALSIFIER MAP ---")
if not P1_ok:
    print("  F1 FIRED: mean deviates from analytic by > 2 SE -> law is BIASED as an expectation; re-open law thread.")
else:
    print("  F1 did not fire (P1 PASS — mean within 2 SE of analytic).")

if not P2_ok:
    if std_D < 25.0:
        print(f"  F2 FIRED (low): std={std_D:.2f} < 25 -> lived -83.72 was a genuine outlier; noise explanation fails; re-open Exp 82 diagnosis.")
    else:
        print(f"  F2 FIRED (high): std={std_D:.2f} > 90 -> per-epoch forecasting uninformative at this horizon; law expectation-only at much longer horizons.")
else:
    print(f"  F2 did not fire (P2 PASS — std={std_D:.2f} in [25, 90]).")

if not P3_ok and P1_ok and P2_ok:
    print("  F3 FIRED: lived draw anomalous relative to counterfactual ensemble despite valid P1+P2; seed-specific structure; diagnose.")
elif not P3_ok:
    print("  F3 not evaluated independently (P1 or P2 also failed).")
else:
    print(f"  F3 did not fire (P3 PASS — lived draw is an ordinary draw from the counterfactual ensemble).")

print()

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

all_pass = P1_ok and P2_ok and P3_ok

if all_pass:
    print(f"EXP83: LAW CALIBRATED (mean ok, sigma={std_D:.1f}, lived draw ordinary)")
else:
    fired = []
    if not P1_ok:
        fired.append(f"F1(law biased: |mean-analytic|={abs(mean_D - ANALYTIC):.2f} > 2*SE={2*SE:.2f})")
    if not P2_ok:
        if std_D < 25.0:
            fired.append(f"F2-low(std={std_D:.1f} < 25, lived draw is genuine outlier)")
        else:
            fired.append(f"F2-high(std={std_D:.1f} > 90, law expectation-only at longer horizons)")
    if not P3_ok and P1_ok and P2_ok:
        fired.append(f"F3(lived draw anomalous: |lived-mean|={abs(LIVED - mean_D):.2f} > 2.5*sigma={2.5*std_D:.2f})")
    print("EXP83: FALSIFIER(S) FIRED — " + "; ".join(fired))
