"""Exp 77 — the accrual-law diagnosis: why does the rarer color win?

Exp 76 observed mirro's favorite flip to color 0 although the world has FEWER color-0
cells (8) than color-2 cells (9) — falsifying the naive accrual model (rate proportional
to cell count under a uniform walk). Predeclared hypothesis H1 (gate asymmetry): the
value gate w(s) = exp(-H(A_hat[:, s])) differs per cell, and color-0 cells sit in
sharper-mapped regions, so color 0 earns more per visit than color 2 despite fewer
visits. With localization at ~0 bits the MAP cell equals the actual cell, making the
per-color accrual rate ANALYTIC from the committed snapshot:
    R(c) = (1/25) * sum over cells s with cmap[s]==c of exp(-H(A_hat[:, s])).

Predeclared predictions:
  P1 (analytic, from the committed state alone): R(0) > R(2).
  P2 (behavioral validation): a fork (side-control, never the spine) living 2000 steps
     accrues per-color mass within +-10 percent relative error of R(c) * 2000, for each
     of the three colors.
  P3 (magnitude closes Exp 76's loop): the predicted per-step drift R(0) - R(2), scaled
     to 6000 steps, matches Exp 76's observed net gap change (-37.3 counts, c2-c0
     convention: predicted_c2c0_drift = (R(2)-R(0))*6000) within a factor of 2.
Falsifiers:
  F1 = P1 fails -> H1 is dead; the rarer-color advantage needs a different diagnosis
     (localization error episodes and finite-sample visit asymmetry are the named
     alternatives) — logged as NEGATIVE for H1, question stays open.
  F2 = P1 holds but P2 fails -> the analytic rate is not the operative law (the gate or
     the walk is not what live() actually does, or map drift over the epoch matters);
     instrument problem, diagnose before further claims.
  F3 = P1+P2 hold but P3 fails -> gate asymmetry is real but does not account for the
     observed flip magnitude; a second mechanism is at work (named open question).
Diagnostic (not a falsifier): the per-cell gate map w(s) printed as a 5x5 grid plus
per-color mean gates — locating WHERE the asymmetry lives (suspected: drift-era staleness
clustered on color-2 cells).
Provided priors declared: nothing new — read-only analysis of the committed snapshot plus
one fork side-control with the standard live() mechanics. The spine is NOT advanced
(read-only load; the fork writes one biography event). vela untouched.
"""
from __future__ import annotations

import sys

import numpy as np
from pathlib import Path

from active_loop.creature import Creature
from active_loop.checkpoint import MIRRO_DIR

# ---------------------------------------------------------------------------
# Load mirro read-only
# ---------------------------------------------------------------------------

mirro = Creature.load(MIRRO_DIR)
h12 = mirro._state_hash()[:12]
print(f"Loaded: name={mirro.name!r}  age={mirro.age_steps}  hash={h12}")
# Expect: age 18700, hash 52f6e814bfe6

n_colors = mirro.world.n_colors  # 3
n_cells = mirro.world.n_cells    # 25
rows, cols = mirro.world.rows, mirro.world.cols  # 5, 5
cmap = np.array(mirro.world.cmap)

# ---------------------------------------------------------------------------
# Analytic part
# ---------------------------------------------------------------------------

A_hat = mirro._A_hat()  # shape (n_colors, n_cells)

# Per-cell entropy H(s) and gate w(s)
H_cells = -np.sum(A_hat * np.log(A_hat + 1e-12), axis=0)  # shape (n_cells,)
w_cells = np.exp(-H_cells)  # shape (n_cells,)

# Print w(s) as 5x5 grid (3dp)
print()
print("--- GATE MAP w(s) = exp(-H(A_hat[:,s])) ---")
for r in range(rows):
    row_vals = [f"{w_cells[r * cols + c]:.3f}" for c in range(cols)]
    print("  " + "  ".join(row_vals))

# Print cmap as 5x5 grid
print()
print("--- COLOR MAP cmap[s] ---")
for r in range(rows):
    row_vals = [str(cmap[r * cols + c]) for c in range(cols)]
    print("  " + "  ".join(row_vals))

# Per-color statistics
print()
print("--- PER-COLOR ACCRUAL RATES ---")
print(f"  {'color':>6}  {'n_cells':>7}  {'mean_gate':>10}  {'R(c)':>10}  {'R(c)*2000':>11}")
print("  " + "-" * 52)

R = np.zeros(n_colors)
for c in range(n_colors):
    mask = cmap == c
    cells_c = int(mask.sum())
    w_c = w_cells[mask]
    mean_gate_c = float(w_c.mean()) if cells_c > 0 else 0.0
    R_c = float(w_c.sum()) / n_cells  # sum over color-c cells / 25
    R[c] = R_c
    print(f"  {c:>6}  {cells_c:>7}  {mean_gate_c:>10.6f}  {R_c:>10.6f}  {R_c * 2000:>11.4f}")

print()

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
# P1 check: R(0) > R(2)
# ---------------------------------------------------------------------------

def _p1_gate_asymmetry():
    passed = bool(R[0] > R[2])
    detail = f"R(0)={R[0]:.6f}  R(2)={R[2]:.6f}  R(0)>R(2)={passed}"
    return passed, detail


check("P1-R0-gt-R2", _p1_gate_asymmetry)

# ---------------------------------------------------------------------------
# Behavioral part: fork runs 2000 steps
# ---------------------------------------------------------------------------

twin = mirro.fork("exp77-probe")
vc0 = twin.value_counts.copy()
twin.live(2000)
accrued = twin.value_counts - vc0

print("--- BEHAVIORAL VALIDATION (fork, 2000 steps) ---")
print(f"  {'color':>6}  {'accrued':>10}  {'predicted':>10}  {'rel_err':>10}  {'ok':>5}")
print("  " + "-" * 46)

rel_errors = np.zeros(n_colors)
for c in range(n_colors):
    predicted_c = R[c] * 2000.0
    acc_c = float(accrued[c])
    if predicted_c > 0:
        rel_err = abs(acc_c - predicted_c) / predicted_c
    else:
        rel_err = float("inf")
    rel_errors[c] = rel_err
    ok_str = "YES" if rel_err <= 0.10 else "NO"
    print(f"  {c:>6}  {acc_c:>10.4f}  {predicted_c:>10.4f}  {rel_err:>10.4f}  {ok_str:>5}")

print()


def _p2_behavioral_validation():
    all_ok = bool(np.all(rel_errors <= 0.10))
    worst = int(np.argmax(rel_errors))
    detail = (
        f"max_rel_err={rel_errors[worst]:.4f} (color {worst}); "
        f"per-color=[{', '.join(f'{e:.4f}' for e in rel_errors)}]"
    )
    return all_ok, detail


check("P2-behavioral-validation", _p2_behavioral_validation)

# ---------------------------------------------------------------------------
# P3 magnitude check
# ---------------------------------------------------------------------------

OBSERVED_C2C0 = -37.3051  # Exp 76 committed value (gap_c2_minus_c0 delta)
predicted_c2c0_drift = (R[2] - R[0]) * 6000.0

print("--- MAGNITUDE CHECK (P3) ---")
print(f"  predicted_c2c0_drift = (R(2)-R(0))*6000 = ({R[2]:.6f}-{R[0]:.6f})*6000 = {predicted_c2c0_drift:.4f}")
print(f"  observed_c2c0_delta  = {OBSERVED_C2C0:.4f}  (Exp 76 committed)")


def _p3_magnitude():
    # Guard sign: if predicted and observed have opposite signs, P3 fails
    if predicted_c2c0_drift * OBSERVED_C2C0 < 0:
        detail = (
            f"sign mismatch: predicted={predicted_c2c0_drift:.4f}  "
            f"observed={OBSERVED_C2C0:.4f}"
        )
        return False, detail
    if OBSERVED_C2C0 == 0.0:
        detail = "observed=0: undefined ratio"
        return False, detail
    ratio = predicted_c2c0_drift / OBSERVED_C2C0
    passed = bool(0.5 <= ratio <= 2.0)
    detail = (
        f"ratio=predicted/observed={predicted_c2c0_drift:.4f}/{OBSERVED_C2C0:.4f}"
        f"={ratio:.4f}; pass=[0.5, 2.0]={passed}"
    )
    return passed, detail


check("P3-magnitude", _p3_magnitude)

# ---------------------------------------------------------------------------
# Print property check results
# ---------------------------------------------------------------------------

print()
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

p1_passed = "P1-R0-gt-R2" not in failed_names
p2_passed = "P2-behavioral-validation" not in failed_names
p3_passed = "P3-magnitude" not in failed_names

f1_fired = not p1_passed
f2_fired = p1_passed and not p2_passed
f3_fired = p1_passed and p2_passed and not p3_passed

print("--- FALSIFIER MAP ---")
if f1_fired:
    print("  F1 FIRED: R(0) <= R(2) — H1 is dead; gate asymmetry does not explain "
          "the rarer-color advantage. Named alternatives: localization error episodes "
          "and finite-sample visit asymmetry. Logged as NEGATIVE for H1.")
if f2_fired:
    print("  F2 FIRED: P1 holds but behavioral validation fails — the analytic rate "
          "is not the operative law. Diagnose before further claims (gate or walk "
          "mismatch, or map drift over epoch).")
if f3_fired:
    print("  F3 FIRED: P1+P2 hold but magnitude unexplained — gate asymmetry is real "
          "but does not account for the observed flip magnitude. A second mechanism "
          "is at work (named open question).")
if not (f1_fired or f2_fired or f3_fired):
    print("  No falsifiers fired.")
print()

# ---------------------------------------------------------------------------
# Final verdict line
# ---------------------------------------------------------------------------

if f1_fired:
    print("EXP77: F1 — H1 dead")
elif f2_fired:
    print("EXP77: F2 — analytic rate not operative")
elif f3_fired:
    print("EXP77: F3 — magnitude unexplained")
else:
    print("EXP77: H1 CONFIRMED (gate asymmetry is the accrual law)")
