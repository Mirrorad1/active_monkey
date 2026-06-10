"""Exp 82 — exact accounting of vela's 3x drift anomaly (the Exp 81 halt-mandated diagnosis).

Exp 81: the analytic accrual law forecast vela's preference drift at -29.3/6000 steps;
the realized drift was ~-84 (2.9x) and the favorite flipped. This experiment locates the
3x EXACTLY, exploiting determinism: vela@12750 is recovered from git history
(git show e7220c1~1:creature/state/vela/...), its 6000-step epoch is REPLAYED with
per-step instrumentation replicating live() bit-for-bit, and the replay is valid only if
it reproduces the committed end-state hash (0cd2d991cf1b) exactly.

Decomposition (exact by construction; s(t) = +1 if obs(t)==color2, -1 if color0, else 0):
  D_realized        = sum_t s(t) * w_realized(t)      [gate at CURRENT A_hat, MAP cell — what live() does]
  D_frozen_analytic = 6000 * (R2 - R0) from the frozen age-12750 gates   [= -29.27, Exp 81's forecast]
  D_frozen_visits   = sum_t s(t) * w_frozen(true_pos(t))                 [frozen gates, actual visits]
  D_current_true    = sum_t s(t) * w_current(t, true_pos(t))             [current gates, true cell]
  Delta_visit  = D_frozen_visits - D_frozen_analytic   [finite-sample visit/observation noise]
  Delta_rate   = D_current_true  - D_frozen_visits     [gate evolution during the epoch (healing)]
  Delta_misloc = D_realized      - D_current_true      [MAP cell != true cell gate misattribution]
  Identity: D_realized = D_frozen_analytic + Delta_visit + Delta_rate + Delta_misloc.

Predeclared:
  P0 (replay exactness, validity gate): the replayed end state's hash equals the committed
     0cd2d991cf1b (and the replayed D_realized equals Exp 81's observed gap change
     -21.68 - 62.03 = -83.72 within 0.01). Failure -> accounting INVALID, halt.
  P1 (bookkeeping closes): the four terms satisfy the identity within 1 percent of
     |D_realized| (exact by construction; a violation means an implementation error).
  P2 (the diagnosis): the dominant |Delta| term is identified; predicted (LOW confidence)
     Delta_rate (the healing immigrant's gates rose ~13 percent during the epoch).
Falsifiers: F0 = P0 fails (replay not exact; live() not faithfully replicated). The
diagnosis itself cannot fail — whichever term dominates IS the answer; the prediction
being wrong is reported as such.
Also reported: MAP-cell correctness rate over the epoch (the mislocalization exposure),
and the per-2000-step evolution of (R0 - R2) on the current map (the rate trajectory).
Provided priors declared: nothing new — a git-recovered snapshot, an instrumented exact
replay, read-only arithmetic. Neither committed line is advanced or saved (the replay
works on an in-memory copy; vela's committed state is untouched).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from active_loop.creature import Creature

# ---------------------------------------------------------------------------
# Constants from Exp 81 output
# ---------------------------------------------------------------------------

COMMIT_BEFORE = "e7220c1~1"   # parent of the Exp 81 commit = the pre-epoch snapshot
EXPECT_START_AGE = 12750
EXPECT_START_HASH = "875ac30d715a"
EXPECT_END_HASH = "0cd2d991cf1b"
EPOCH = 6000

# Exact observed numbers from Exp 81 output
START_GAP = 62.0337         # gap_c2_minus_c0 at age 12750
END_GAP = -21.6836          # gap_c2_minus_c0 at age 18750
D_REALIZED_EXPECTED = END_GAP - START_GAP   # = -83.7173

# ---------------------------------------------------------------------------
# Step 1: recover pre-epoch snapshot from git history
# ---------------------------------------------------------------------------

scratch = Path("experiments/outputs/exp82_pre_snapshot")
scratch.mkdir(parents=True, exist_ok=True)

print("Exp 82 — exact accounting of vela's 3x drift anomaly")
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

# Load — Creature.load() verifies the stored hash 875ac30d715a internally
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
# Step 2: frozen quantities (from the age-12750 snapshot)
# ---------------------------------------------------------------------------

A_frozen = v._A_hat().copy()   # shape (n_colors, n_cells)
cmap = np.array(v.world.cmap)  # shape (n_cells,)
n_cells = v.world.n_cells      # 25
n_colors = v.world.n_colors    # 3

# Frozen per-cell entropies and weights
H_frozen = -np.sum(A_frozen * np.log(A_frozen + 1e-12), axis=0)   # (n_cells,)
w_frozen_cells = np.exp(-H_frozen)                                  # (n_cells,)

# Frozen per-color rates: R(c) = sum of w(s) for cells of color c / n_cells
R_frozen = np.zeros(n_colors)
for color in range(n_colors):
    mask = cmap == color
    R_frozen[color] = float(w_frozen_cells[mask].sum()) / n_cells

D_frozen_analytic = float((R_frozen[2] - R_frozen[0]) * EPOCH)

print("--- FROZEN GATES (age 12750) ---")
print(f"  R_frozen(0)       = {R_frozen[0]:.6f}")
print(f"  R_frozen(1)       = {R_frozen[1]:.6f}")
print(f"  R_frozen(2)       = {R_frozen[2]:.6f}")
print(f"  R(2)-R(0)         = {R_frozen[2] - R_frozen[0]:.6f}")
print(f"  D_frozen_analytic = {D_frozen_analytic:+.4f}  [Exp 81 forecast]")
print()

# ---------------------------------------------------------------------------
# Step 3: exact replay of live(6000) with per-step instrumentation
# ---------------------------------------------------------------------------
# Replicates live() VERBATIM including its RNG derivation.

rng = np.random.default_rng(
    (v._seed * 1_000_003 + v.rng_counter) & 0xFFFFFFFFFFFFFFFF
)

B = v.world.transition_matrix()   # (n_cells, n_cells, 4)
n_actions = 4

# Accumulator sums — three flavors of D
D_realized = 0.0
D_frozen_visits = 0.0
D_current_true = 0.0

# Tracking
map_correct = 0

# Per-2000-step R0-R2 snapshots on the CURRENT A_hat (indices: step 0, 2000, 4000, 6000)
r0_r2_trajectory: list[float] = []


def _current_r0_r2(pA: np.ndarray) -> float:
    """Compute (R0 - R2) on the current pA snapshot."""
    A = pA.copy()
    col_sums = A.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    A /= col_sums
    H = -np.sum(A * np.log(A + 1e-12), axis=0)
    w = np.exp(-H)
    R = np.array([
        float(w[cmap == c].sum()) / n_cells
        for c in range(n_colors)
    ])
    return float(R[0] - R[2])


# Record step-0 snapshot
r0_r2_trajectory.append(_current_r0_r2(v.pA))

for step in range(EPOCH):
    # --- replicate live() VERBATIM ---
    A_hat = v._A_hat()                      # (n_colors, n_cells) — CURRENT at step start

    # observe
    obs = int(v.world.cmap[v.true_pos])

    # belief update: qs ∝ likelihood(obs) * prior(qs)
    likelihood = A_hat[obs, :]
    qs_updated = likelihood * v.qs
    denom = qs_updated.sum()
    if denom > 0:
        qs_updated = qs_updated / denom
    else:
        qs_updated = np.ones(n_cells) / n_cells

    # Dirichlet count learning
    v.pA[obs, :] += qs_updated

    # value accumulation — MAP cell
    map_cell = int(np.argmax(qs_updated))
    predicted_obs_dist = A_hat[:, map_cell]
    h_predicted = -np.sum(predicted_obs_dist * np.log(predicted_obs_dist + 1e-12))
    w_realized = float(np.exp(-h_predicted))
    v.value_counts[obs] += w_realized

    # --- instrumentation (uses quantities already computed above) ---
    # sign: +1 if obs==color2, -1 if obs==color0, else 0
    s_t = 1 if obs == 2 else (-1 if obs == 0 else 0)

    # D_realized accumulates exactly as live() does
    D_realized += s_t * w_realized

    # frozen gate at TRUE position
    w_froz_true = float(w_frozen_cells[v.true_pos])
    D_frozen_visits += s_t * w_froz_true

    # current gate at TRUE position (from A_hat at step start, same matrix used above)
    h_current_true = -np.sum(A_hat[:, v.true_pos] * np.log(A_hat[:, v.true_pos] + 1e-12))
    w_curr_true = float(np.exp(-h_current_true))
    D_current_true += s_t * w_curr_true

    # map correctness
    if map_cell == v.true_pos:
        map_correct += 1

    # random action, move
    action = int(rng.integers(0, n_actions))
    v.true_pos = v.world.move(v.true_pos, action)

    # advance belief through movement model
    v.qs = B[:, :, action] @ qs_updated

    # record trajectory snapshot every 2000 steps
    if (step + 1) % 2000 == 0:
        r0_r2_trajectory.append(_current_r0_r2(v.pA))

# Advance state counters exactly as live() does
v.age_steps += EPOCH
v.rng_counter += 1

# ---------------------------------------------------------------------------
# P0 — validity gate: replay exactness
# ---------------------------------------------------------------------------

replayed_hash = v._state_hash()[:12]
replayed_d_realized_check = abs(D_realized - D_REALIZED_EXPECTED)

print("--- P0 VALIDITY GATE ---")
print(f"  replayed end-state hash : {replayed_hash}")
print(f"  expected end-state hash : {EXPECT_END_HASH}")
print(f"  D_realized (replayed)   : {D_realized:+.4f}")
print(f"  D_realized (expected)   : {D_REALIZED_EXPECTED:+.4f}")
print(f"  |diff|                  : {replayed_d_realized_check:.6f}")
print()

hash_ok = replayed_hash == EXPECT_END_HASH
d_ok = replayed_d_realized_check <= 0.01

if not (hash_ok and d_ok):
    problems = []
    if not hash_ok:
        problems.append(f"hash mismatch (got {replayed_hash}, expected {EXPECT_END_HASH})")
    if not d_ok:
        problems.append(f"D_realized diff {replayed_d_realized_check:.6f} > 0.01")
    print("F0 — replay inexact, accounting INVALID: " + "; ".join(problems))
    sys.exit(1)

print("P0 PASS — replay exact: hash matches and D_realized within 0.01")
print()

# ---------------------------------------------------------------------------
# Step 4: decompose the 3x anomaly
# ---------------------------------------------------------------------------

Delta_visit = D_frozen_visits - D_frozen_analytic
Delta_rate = D_current_true - D_frozen_visits
Delta_misloc = D_realized - D_current_true

# Identity check
identity_sum = D_frozen_analytic + Delta_visit + Delta_rate + Delta_misloc
residual = D_realized - identity_sum

map_correctness_rate = map_correct / EPOCH

print("--- DRIFT DECOMPOSITION ---")
print(f"  D_frozen_analytic  = {D_frozen_analytic:+.4f}  [Exp 81 forecast: frozen gates, uniform visits]")
print(f"  Delta_visit        = {Delta_visit:+.4f}  [frozen gates, actual visits vs uniform]")
print(f"  Delta_rate         = {Delta_rate:+.4f}  [gate evolution during epoch (healing)]")
print(f"  Delta_misloc       = {Delta_misloc:+.4f}  [MAP cell != true cell misattribution]")
print(f"  ─────────────────────────────────────────────")
print(f"  Sum of four terms  = {identity_sum:+.4f}")
print(f"  D_realized         = {D_realized:+.4f}  [what live() actually accumulated]")
print(f"  residual           = {residual:+.6f}  [should be ~0 by construction]")
print()

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------

checks: list[tuple[str, bool, str]] = []


def check(name: str, predicate_fn) -> None:
    try:
        passed, detail = predicate_fn()
        checks.append((name, passed, detail))
    except Exception as exc:
        checks.append((name, False, f"exception: {exc}"))


def _p1_identity():
    threshold = 0.01 * abs(D_realized)
    ok = abs(residual) <= threshold
    return ok, (
        f"residual={residual:+.6f}  threshold=0.01*|D_realized|={threshold:.6f}  "
        f"closes={'YES' if ok else 'NO'}"
    )


def _p2_diagnosis():
    deltas = {
        "Delta_visit": Delta_visit,
        "Delta_rate": Delta_rate,
        "Delta_misloc": Delta_misloc,
    }
    dominant_name = max(deltas, key=lambda k: abs(deltas[k]))
    dominant_val = deltas[dominant_name]
    prediction_correct = dominant_name == "Delta_rate"
    return True, (  # cannot fail — whichever dominates IS the answer
        f"dominant={dominant_name}  value={dominant_val:+.4f}  "
        f"prediction(Delta_rate)={'RIGHT' if prediction_correct else 'WRONG'}"
    )


check("P0-replay-exact", lambda: (True, f"hash={replayed_hash} D_diff={replayed_d_realized_check:.6f}"))
check("P1-identity-closes", _p1_identity)
check("P2-diagnosis", _p2_diagnosis)

print("--- PROPERTY CHECKS ---")
failed_names: list[str] = []
for name, passed, detail in checks:
    verdict = "PASS" if passed else "FAIL"
    print(f"  {verdict}  {name}: {detail}")
    if not passed:
        failed_names.append(name)

print()

# ---------------------------------------------------------------------------
# Supporting diagnostics
# ---------------------------------------------------------------------------

print("--- SUPPORTING DIAGNOSTICS ---")
print(f"  MAP-cell correctness rate over epoch: {map_correctness_rate:.4f} ({map_correct}/{EPOCH} steps)")
print()
print("  (R0-R2) on current A_hat trajectory during epoch:")
labels = [0, 2000, 4000, 6000]
for label, val in zip(labels, r0_r2_trajectory):
    print(f"    step {label:5d}: R0-R2 = {val:.6f}")
print()

# ---------------------------------------------------------------------------
# Falsifier map
# ---------------------------------------------------------------------------

print("--- FALSIFIER MAP ---")
# F0 already handled above (sys.exit(1)) — if we're here, P0 passed
print("  F0 did not fire (P0 PASS — replay exact).")
if "P1-identity-closes" in failed_names:
    print("  P1 FAILED — bookkeeping does not close; implementation error in decomposition.")
else:
    print("  P1 PASS — identity closes within 1% of |D_realized|.")
print()

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

# Re-derive dominant for final line
deltas_final = {
    "Delta_visit": Delta_visit,
    "Delta_rate": Delta_rate,
    "Delta_misloc": Delta_misloc,
}
dominant_final = max(deltas_final, key=lambda k: abs(deltas_final[k]))
dominant_val_final = deltas_final[dominant_final]
prediction_right = dominant_final == "Delta_rate"

print(
    f"EXP82: DIAGNOSIS — dominant term = {dominant_final} ({dominant_val_final:+.1f} of "
    f"{D_realized:+.1f}); prediction {'right' if prediction_right else 'wrong'}"
)
