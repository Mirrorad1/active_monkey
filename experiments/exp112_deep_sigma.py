"""Exp 112 — the F3 diagnosis: re-estimating sigma at the deep state the lurch launched
from.

Exp 111's falsifier: vela swung -261.8 -> +192.6 in one epoch (D_lived = +454.5), a
3.05-sigma event under sigma = 121 — which was measured ONCE (Exp 83, vela@12750, a
shallow state). Exp 84's superdiffusive exponents predict deep states swing harder. The
mandated diagnosis: recover vela@66750 (git show 25786b0~1:creature/state/vela/...,
the exact pre-lurch state), run 20 counterfactual 6000-step epochs (explicit action
seeds 5000-5019), and measure the drift distribution AT THIS STATE.

Predeclared:
  P1 (state-dependent sigma): sigma_deep >= 180 (1.5x the shallow 121) -> F3 resolves as
     CALIBRATION (state-dependent error bars, physics intact); sigma_deep <= 150 -> the
     lurch stays anomalous and a seed-specific diagnosis is required. 150-180 =
     borderline, reported. Predicted: sigma_deep in [150, 350].
  P2 (the lived draw is ordinary in its own ensemble): |D_lived - mean(D)| <= 2.5 *
     sigma_deep.
  P3 (the law stays unbiased): |mean(D) - D_analytic| <= 2 SE, where D_analytic =
     (R(2)-R(0))_frozen * 6000 from the recovered state's gates.
  RECORDED (not checked): the counterfactual flip fraction from -261.8 — a direct
  measurement of deep-bin flip probability, feeding the registered test's evaluation.
Falsifiers: F1 = P2 fails (the lived draw is an outlier even in its own ensemble —
seed-specific structure, halt for deeper diagnosis). F2 = P3 fails (the law is biased at
deep states — re-open the law).
Notes: in-memory deepcopies only; neither committed line touched; scratch untracked and
derivable. D = (value_counts[2]-value_counts[0])_end - start per branch.
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

COMMIT_REF = "25786b0~1"          # parent of the Exp 111 commit = pre-lurch snapshot
EXPECT_START_AGE = 66750

D_LIVED = 192.6144 - (-261.8498)  # = +454.4642, the lived lurch (Exp 111)
SIGMA_SHALLOW = 121.0             # Exp 83's single-state estimate (vela@12750)
N = 20
SEEDS = range(5000, 5020)
EPOCH = 6000

P1_RESOLVE = 180.0                # sigma_deep >= this -> F3 resolves as CALIBRATION
P1_ANOMALOUS = 150.0              # sigma_deep <= this -> lurch stays anomalous
PRED_BAND = (150.0, 350.0)        # predicted sigma_deep band (reported, not checked)


# ---------------------------------------------------------------------------
# Analytic gate-rate helper (Exp 80)
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
    for color in range(n_colors):
        mask = cmap == color
        w_c = w_cells[mask]
        n_c = int(mask.sum())
        R_c = float(w_c.sum()) / n_cells
        result[f"R{color}"] = R_c
        result[f"mean_gate_{color}"] = float(w_c.mean()) if n_c > 0 else 0.0

    return result


# ---------------------------------------------------------------------------
# Step 1: recover the pre-lurch snapshot from git history (same pattern as Exp 83)
# ---------------------------------------------------------------------------

scratch = Path("experiments/outputs/exp112_pre_snapshot")
scratch.mkdir(parents=True, exist_ok=True)

print("Exp 112 — the F3 diagnosis: sigma at the deep state")
print()
print(f"Recovering pre-lurch vela snapshot from {COMMIT_REF}...")
print(f"  (derivable via: git show {COMMIT_REF}:creature/state/vela/{{manifest.json,arrays.npz}})")
print(f"  scratch dir: {scratch}  [untracked; not committed — derivable from git history]")
print()

for fname in ["manifest.json", "arrays.npz"]:
    result = subprocess.run(
        ["git", "show", f"{COMMIT_REF}:creature/state/vela/{fname}"],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"ERROR: git show failed for {fname}: {result.stderr.decode()}")
        sys.exit(1)
    (scratch / fname).write_bytes(result.stdout)

# Load and verify snapshot
v = Creature.load(scratch)
print(f"Loaded vela: age={v.age_steps}  hash={v._state_hash()[:12]}")

if v.age_steps != EXPECT_START_AGE:
    print(f"SNAPSHOT MISMATCH: expected age={EXPECT_START_AGE}, got age={v.age_steps}")
    sys.exit(1)

start_gap = float(v.value_counts[2] - v.value_counts[0])
start_fav = v.favorite()
print(f"  start gap = {start_gap:+.4f}  start favorite = {start_fav}")
print()

# ---------------------------------------------------------------------------
# Step 2: analytic forecast from the recovered state's frozen gates
# ---------------------------------------------------------------------------

r = rates(v)
D_ANALYTIC = (r["R2"] - r["R0"]) * EPOCH

print("--- ANALYTIC FORECAST (frozen gates at age 66750) ---")
print(f"  R(0) = {r['R0']:.6f}   R(1) = {r['R1']:.6f}   R(2) = {r['R2']:.6f}")
print(f"  D_analytic = (R(2)-R(0)) * {EPOCH} = {D_ANALYTIC:+.4f}")
print()

# ---------------------------------------------------------------------------
# Step 3: run 20 counterfactual branches
# ---------------------------------------------------------------------------

print(f"Running {N} counterfactual branches (seeds {SEEDS.start}..{SEEDS.stop - 1}, "
      f"{EPOCH} steps each)...")
print()

D_values: list[float] = []
flipped: list[bool] = []

for k in SEEDS:
    b = copy.deepcopy(v)
    d0 = float(b.value_counts[2] - b.value_counts[0])
    b.live(EPOCH, seed=k)
    d_end = float(b.value_counts[2] - b.value_counts[0])
    D_k = d_end - d0
    fav_k = b.favorite()
    flipped_k = fav_k == 2          # vela starts favorite 0 at this state
    D_values.append(D_k)
    flipped.append(flipped_k)
    print(f"  seed={k}  D={D_k:+.4f}  end_favorite={fav_k}  flipped={flipped_k}")

print()

# ---------------------------------------------------------------------------
# Step 4: statistics
# ---------------------------------------------------------------------------

D_arr = np.array(D_values)
mean_D = float(np.mean(D_arr))
std_D = float(np.std(D_arr, ddof=1))       # sample std = sigma_deep
SE = std_D / np.sqrt(N)

flip_count = sum(flipped)
flip_frac = flip_count / N
z_lived = abs(D_LIVED - mean_D) / std_D if std_D > 0 else float("inf")

print("--- STATISTICS ---")
print(f"  N                    = {N}")
print(f"  mean(D)              = {mean_D:+.4f}")
print(f"  sigma_deep [ddof=1]  = {std_D:.4f}")
print(f"  SE = std/sqrt(N)     = {SE:.4f}")
print(f"  sigma_shallow        = {SIGMA_SHALLOW:.4f}  (Exp 83, vela@12750)")
print(f"  ratio deep/shallow   = {std_D / SIGMA_SHALLOW:.4f}")
print(f"  D_analytic           = {D_ANALYTIC:+.4f}")
print(f"  |mean - analytic|    = {abs(mean_D - D_ANALYTIC):.4f}  (2*SE = {2*SE:.4f})")
print(f"  lived draw D         = {D_LIVED:+.4f}")
print(f"  |lived - mean|       = {abs(D_LIVED - mean_D):.4f}  (2.5*sigma = {2.5*std_D:.4f})")
print(f"  z of lived draw      = {z_lived:.4f}")
print(f"  RECORDED flip fraction (favorite 0 -> 2 from gap -261.8): "
      f"{flip_count}/{N} = {flip_frac:.2f}")
print()

# ---------------------------------------------------------------------------
# Step 5: text histogram of sorted D values
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
# Step 6: property checks
# ---------------------------------------------------------------------------

# P1: three-branch read of sigma_deep
P1_resolved = std_D >= P1_RESOLVE
P1_anomalous = std_D <= P1_ANOMALOUS
P1_borderline = not P1_resolved and not P1_anomalous
in_band = PRED_BAND[0] <= std_D <= PRED_BAND[1]

if P1_resolved:
    p1_detail = (f"sigma_deep={std_D:.4f} >= {P1_RESOLVE:.0f} -> F3 resolves as "
                 f"CALIBRATION (state-dependent error bars, physics intact)")
elif P1_anomalous:
    p1_detail = (f"sigma_deep={std_D:.4f} <= {P1_ANOMALOUS:.0f} -> lurch stays "
                 f"anomalous; seed-specific diagnosis required")
else:
    p1_detail = (f"sigma_deep={std_D:.4f} in ({P1_ANOMALOUS:.0f}, {P1_RESOLVE:.0f}) -> "
                 f"BORDERLINE, reported")

P2_ok = abs(D_LIVED - mean_D) <= 2.5 * std_D
P3_ok = abs(mean_D - D_ANALYTIC) <= 2 * SE

print("--- PROPERTY CHECKS ---")
print(f"  P1: {p1_detail}")
print(f"      predicted band [{PRED_BAND[0]:.0f}, {PRED_BAND[1]:.0f}]: "
      f"{'INSIDE' if in_band else 'OUTSIDE'} (sigma_deep={std_D:.4f})")
print(f"  {'PASS' if P2_ok else 'FAIL'}  P2: |lived-mean|={abs(D_LIVED - mean_D):.4f} "
      f"<= 2.5*sigma={2.5*std_D:.4f} -> {'PASS' if P2_ok else 'FAIL'}")
print(f"  {'PASS' if P3_ok else 'FAIL'}  P3: |mean-analytic|={abs(mean_D - D_ANALYTIC):.4f} "
      f"<= 2*SE={2*SE:.4f} -> {'PASS' if P3_ok else 'FAIL'}")
print(f"  RECORDED: flip fraction {flip_count}/{N} = {flip_frac:.2f} "
      f"(deep-bin flip probability; feeds the registered test's evaluation)")
print()

# ---------------------------------------------------------------------------
# Step 7: falsifier map
# ---------------------------------------------------------------------------

print("--- FALSIFIER MAP ---")
if not P2_ok:
    print(f"  F1 FIRED: |lived-mean|={abs(D_LIVED - mean_D):.2f} > "
          f"2.5*sigma={2.5*std_D:.2f} -> the lived draw is an outlier even in its own "
          f"ensemble — seed-specific structure; halt for deeper diagnosis.")
else:
    print(f"  F1 did not fire (P2 PASS — lived draw ordinary in its own ensemble, "
          f"z={z_lived:.2f}).")

if not P3_ok:
    print(f"  F2 FIRED: |mean-analytic|={abs(mean_D - D_ANALYTIC):.2f} > "
          f"2*SE={2*SE:.2f} -> the law is biased at deep states; re-open the law.")
else:
    print(f"  F2 did not fire (P3 PASS — law unbiased at this state).")

print()

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

if not P2_ok:
    print(f"EXP112: F1 FIRED — lived draw is an outlier even in its own ensemble "
          f"(z={z_lived:.2f} > 2.5); seed-specific structure; halt for deeper diagnosis")
elif not P3_ok:
    print(f"EXP112: F2 FIRED — law biased at deep states "
          f"(|mean-analytic|={abs(mean_D - D_ANALYTIC):.2f} > 2*SE={2*SE:.2f}); "
          f"re-open the law")
elif P1_resolved:
    print(f"EXP112: F3 RESOLVED — state-dependent sigma ({std_D:.0f} at deep vs 121 "
          f"shallow); lurch ordinary ({z_lived:.2f} sigma in own ensemble)")
elif P1_anomalous:
    print(f"EXP112: LURCH STAYS ANOMALOUS — sigma_deep={std_D:.0f} <= 150 (no "
          f"state-dependent widening); seed-specific diagnosis required")
else:
    print(f"EXP112: BORDERLINE — sigma_deep={std_D:.0f} in (150, 180); partial "
          f"widening over shallow 121; lived draw z={z_lived:.2f}; judgment deferred "
          f"to the entry")
