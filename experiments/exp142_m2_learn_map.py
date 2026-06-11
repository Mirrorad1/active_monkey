"""
Exp 142 — continuous-creature rung M2: NIW-learned emission map under the motor
anchor (non-aliased; the wall-information asymmetry predeclared).

Card loop/directions/continuous-creature.md; guardrail: any failed predeclared
property HALTS the migration for explicit human input.

Hypothesis: with emissions UNLEARNED and no provided position anchor, the walls are
the only symmetry-breaker for dead reckoning. M1's naive mean-clamp DISCARDS the
wall information that the tabular B preserves (clamping concentrates probability at
walls; a mean-clamped Gaussian never shrinks variance there). Therefore: a
moment-matched clamp (analytic truncated-Gaussian moments, per-axis, diagonal
Sigma) bootstraps the emission map from a broad prior, while the naive clamp
learns markedly worse; the Dirichlet twin (which keeps the wall information by
construction) forms its map as in Exp 21.

Setup: 4x4 grid in R^2, 16 distinct colors (non-aliased); TRUE dynamics as
World.move; T = 3000 uniform-random steps; seeds 0..7; three agents on IDENTICAL
streams: (a) tabular twin — the creature's own equations (uniform qs birth,
Dirichlet pA birth 0.1 + 0.01*jitter, soft-count learning, exact B); (b)
moment-matched continuous: place belief prior N(arena center, 4I) DIAGONAL,
predict_clamped_moments(delta, Q=0.05^2 I), per-color NIW (m0 = arena center,
kappa0 = 1, nu0 = 4, S0 = 0.35^2*(nu0-d-1)*I), observation update using current
expected (m_k, E[Sigma_k]), then NIW.update_moments(place posterior); (c) naive
continuous: identical except M1's mean-clamp predict. Per-step order as Exp 141
(observe -> place update -> NIW update -> act -> move -> predict).

Metrics: final map error per color ||m_k - center_k||; map-formed count =
colors with error <= 0.5; final-500-step median localization error; tabular
map_accuracy (argmax cell-color correctness, the creature's own metric).

Predictions (TRUE iff all):
- P1 twin sanity: tabular map_accuracy >= 14/16 cells in >= 7/8 seeds.
- P2 bootstrap: moment-matched forms the map — >= 14/16 colors within 0.5 in
  >= 6/8 seeds, AND final-500 median localization <= 0.5 in those seeds.
- P3 asymmetry: naive-clamp final mean map error >= 1.5x the moment-matched
  mean map error (cell means over seeds; the wall-information account).

Falsifier: P2 fails in >= 3/8 seeds (continuous map learning does not bootstrap
in the creature's world — MIGRATION HALT for human input), OR P1 fails (the
in-situ twin baseline itself broken — instrument problem, halt and inspect), OR
P3 fails with naive ~ moment-matched (within 1.2x — the wall-information account
is wrong; log it as the finding, NOT a halt: P2 carrying is what the migration
needs). Three-way rule per PROTOCOL step 3; MIXED for the P3-only miss.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from active_loop.continuous import NIW
from active_loop.creature_continuous import ContinuousPlace

# ---------------------------------------------------------------------------
# World constants  (mirrors Exp 141 / exp141_m1_perceive.py)
# ---------------------------------------------------------------------------

ROWS, COLS = 4, 4
N_CELLS = ROWS * COLS   # 16
N_COLORS = 16            # non-aliased: cell i -> color i

# Cell (r, c) -> (x=c, y=r); index cell = r*COLS + c
CELL_CENTERS = np.array(
    [[float(c), float(r)] for r in range(ROWS) for c in range(COLS)]
)  # shape (16, 2)

# cmap is identity for non-aliased
CMAP = np.arange(N_CELLS, dtype=int)  # cmap[cell] = cell

ARENA = (0.0, float(COLS - 1), 0.0, float(ROWS - 1))  # (0,3,0,3)
ARENA_CENTER = np.array([1.5, 1.5])

# Action -> (dx, dy): 0=up (y-1), 1=down (y+1), 2=left (x-1), 3=right (x+1)
ACTION_DELTA = {
    0: np.array([0.0, -1.0]),
    1: np.array([0.0, +1.0]),
    2: np.array([-1.0, 0.0]),
    3: np.array([+1.0, 0.0]),
}

Q_SCALE = 0.05
Q = Q_SCALE ** 2 * np.eye(2)

# NIW prior parameters
D = 2
KAPPA0 = 1.0
NU0 = 4.0   # >= d+2 = 4, so E[Sigma] = S0/(nu0-d-1) is finite
# S0 chosen so E[Sigma] = 0.35^2 * I: S0 = 0.35^2 * (nu0-d-1) * I = 0.35^2 * 1 * I
S0_SCALE = 0.35 ** 2 * (NU0 - D - 1)  # = 0.35^2 * 1.0
S0 = S0_SCALE * np.eye(D)

# Number of steps
T = 3000
SEEDS = list(range(8))
FINAL_WINDOW = 500  # last T steps for localization metric


# ---------------------------------------------------------------------------
# Wall-clamped move (exact mirror of World.move)
# ---------------------------------------------------------------------------

def move(cell: int, action: int) -> int:
    r, c = divmod(cell, COLS)
    if action == 0:
        r = max(0, r - 1)
    elif action == 1:
        r = min(ROWS - 1, r + 1)
    elif action == 2:
        c = max(0, c - 1)
    else:
        c = min(COLS - 1, c + 1)
    return r * COLS + c


# ---------------------------------------------------------------------------
# Tabular twin: creature's exact equations
# ---------------------------------------------------------------------------

def run_tabular(actions: np.ndarray, rng_pA: np.random.Generator) -> dict:
    """Run the creature's own update equations (linear space, 16 cells).

    Compute in linear space (16 cells, well-conditioned; the log-space rule
    applies to large-separation products — not needed here with 16 cells where
    the minimum pA-based likelihood is ~0.1/1.6 ≈ 0.06, far from underflow).

    map_accuracy: fraction of cells where argmax_color A_hat[:, cell] == cmap[cell].
    """
    # pA init: 0.1 uniform + 0.01*jitter (the creature's birth state)
    pA = np.full((N_COLORS, N_CELLS), 0.1) + 0.01 * rng_pA.random((N_COLORS, N_CELLS))

    # B transition matrix: B[s', s, a] = 1 iff move(s, a) == s'
    B = np.zeros((N_CELLS, N_CELLS, 4))
    for s in range(N_CELLS):
        for a in range(4):
            s2 = move(s, a)
            B[s2, s, a] = 1.0

    # Uniform place belief
    qs = np.ones(N_CELLS) / N_CELLS

    true_cell = 0  # start cell 0

    for t in range(T):
        obs = int(CMAP[true_cell])  # non-aliased: obs = cell

        # A_hat: column-normalized pA
        A_hat = pA.copy()
        col_sums = A_hat.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        A_hat = A_hat / col_sums

        # Belief update: qs_updated ∝ A_hat[obs, :] * qs
        qs_updated = A_hat[obs, :] * qs
        denom = qs_updated.sum()
        if denom > 0:
            qs_updated /= denom
        else:
            qs_updated = np.ones(N_CELLS) / N_CELLS

        # Dirichlet count learning: pA[obs, :] += qs_updated
        pA[obs, :] += qs_updated

        # Act and move
        action = int(actions[t])
        true_cell = move(true_cell, action)

        # Advance belief through B (deterministic)
        qs = B[:, :, action] @ qs_updated

    # Final map accuracy: argmax_color A_hat[:, cell] == cmap[cell]
    A_hat_final = pA.copy()
    col_sums = A_hat_final.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    A_hat_final = A_hat_final / col_sums
    predicted_colors = np.argmax(A_hat_final, axis=0)  # shape (N_CELLS,)
    map_acc = float(np.mean(predicted_colors == CMAP))

    return {"map_accuracy": map_acc}


# ---------------------------------------------------------------------------
# Continuous agents: moment-matched and naive
# ---------------------------------------------------------------------------

def run_continuous(
    actions: np.ndarray,
    use_clamped_moments: bool,
) -> dict:
    """Run a continuous agent (moment-matched or naive clamp).

    Per-step order (as Exp 141): observe -> place update -> NIW update -> act -> move -> predict.

    NIW emission map: per-color NIW prior (m0=arena_center, kappa0=1, nu0=4, S0).
    Observation update: ContinuousPlace.update(E[mu_k], E[Sigma_k] + 1e-6*I).
    NIW soft update: niw_k = niw_k.update_moments(place.mu, place.Sigma) after obs update.

    Place Sigma is kept diagonal throughout:
    - Prior: 4*I (diagonal)
    - predict_clamped_moments: sets off-diagonals to 0 explicitly
    - naive predict: adds diagonal Q to diagonal Sigma (preserves diagonal)
    - Conjugate update: with isotropic emission Sigma_k, the update is:
        Sigma_new = (Sigma^{-1} + Sigma_k^{-1})^{-1}
      For diagonal Sigma and isotropic Sigma_k, Sigma_new is diagonal — assert.

    Parameters
    ----------
    actions : pre-generated action array, shape (T,)
    use_clamped_moments : if True, use predict_clamped_moments; else use predict (naive).
    """
    # Place prior: N(arena_center, 4*I) — diagonal
    mu0 = ARENA_CENTER.copy()
    Sigma0 = 4.0 * np.eye(D)
    cp = ContinuousPlace(mu0, Sigma0, ARENA)

    # Per-color NIW priors (list of 16 NIW instances)
    niws = [NIW(m=ARENA_CENTER.copy(), kappa=KAPPA0, nu=NU0, S=S0.copy())
            for _ in range(N_COLORS)]

    true_cell = 0
    loc_errors = np.empty(T)

    for t in range(T):
        obs = int(CMAP[true_cell])  # non-aliased

        # --- Observe: get expected emission params for color obs ---
        mu_k = niws[obs].expected_mu()
        # Use only the diagonal of E[Sigma_k] to preserve place-Sigma diagonality.
        # After the first NIW update, E[Sigma_k] gains off-diagonal entries from
        # the outer-product scatter term; using the full matrix would couple the
        # x and y axes in the Gaussian product and break diagonality.
        # Projecting to diagonal is equivalent to treating the emission as
        # axis-independent, consistent with the per-axis clamped-moments approximation.
        Sigma_k_full = niws[obs].expected_Sigma()
        Sigma_k = np.diag(np.diag(Sigma_k_full)) + 1e-6 * np.eye(D)  # guard, diagonal only

        # --- Place update (Gaussian product) ---
        cp.update(mu_k, Sigma_k)

        # Assert diagonal is maintained after update
        # (diagonal emission + diagonal prior -> diagonal posterior, exact)
        off_diag_post = abs(cp.Sigma[0, 1])
        assert off_diag_post < 1e-9, (
            f"t={t}: Sigma off-diagonal after update = {off_diag_post:.3e} >= 1e-9"
        )

        # --- NIW soft update for observed color ---
        niws[obs] = niws[obs].update_moments(cp.mu, cp.Sigma)

        # --- Post-update localization error ---
        true_xy = CELL_CENTERS[true_cell]
        loc_errors[t] = float(np.linalg.norm(cp.mu - true_xy))

        # --- Act and move ---
        action = int(actions[t])
        true_cell = move(true_cell, action)

        # --- Predict (dynamics update) ---
        delta = ACTION_DELTA[action]
        if use_clamped_moments:
            cp.predict_clamped_moments(delta, Q)
        else:
            # Naive predict: clamp mean, no variance adjustment for wall
            cp.predict(delta, Q)
            # Ensure diagonal is preserved (predict only adds diagonal Q to diagonal Sigma)
            off_diag_pred = abs(cp.Sigma[0, 1])
            assert off_diag_pred < 1e-9, (
                f"t={t}: Sigma off-diagonal after naive predict = {off_diag_pred:.3e} >= 1e-9"
            )

    # --- Final map: per-color error ||E[mu_k] - center_k|| ---
    map_errors = np.array([
        float(np.linalg.norm(niws[k].expected_mu() - CELL_CENTERS[k]))
        for k in range(N_COLORS)
    ])
    map_formed_count = int(np.sum(map_errors <= 0.5))
    mean_map_error = float(np.mean(map_errors))

    # Final-500-step median localization error
    final_loc_errors = loc_errors[max(0, T - FINAL_WINDOW):]
    final_loc_median = float(np.median(final_loc_errors))

    return {
        "map_errors": map_errors,
        "map_formed_count": map_formed_count,
        "mean_map_error": mean_map_error,
        "final_loc_median": final_loc_median,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rows = []

    # Per-seed results
    tab_results = []
    mm_results = []
    naive_results = []

    print("=" * 80)
    print("Exp 142 — continuous-creature rung M2: NIW emission map learning")
    print("=" * 80)
    print()
    print(f"{'Seed':>4}  "
          f"{'TabMapAcc':>9}  "
          f"{'MM_Formed':>9}  "
          f"{'MM_MapErr':>9}  "
          f"{'MM_LocMed':>9}  "
          f"{'Na_Formed':>9}  "
          f"{'Na_MapErr':>9}  "
          f"{'ErrRatio':>8}")
    print("-" * 80)

    for seed in SEEDS:
        # rng for actions (seed convention per spec: 1000 + seed)
        rng = np.random.default_rng(1000 + seed)
        actions = rng.integers(0, 4, size=T)

        # rng for tabular pA jitter (same rng, next draw — fresh state per seed)
        rng_pA = np.random.default_rng(2000 + seed)

        # --- Tabular twin ---
        tab = run_tabular(actions, rng_pA)
        tab_results.append(tab)

        # --- Moment-matched continuous ---
        mm = run_continuous(actions, use_clamped_moments=True)
        mm_results.append(mm)

        # --- Naive continuous ---
        naive = run_continuous(actions, use_clamped_moments=False)
        naive_results.append(naive)

        err_ratio = (naive["mean_map_error"] / mm["mean_map_error"]
                     if mm["mean_map_error"] > 1e-9 else float("nan"))

        print(f"{seed:>4}  "
              f"{tab['map_accuracy']:>9.4f}  "
              f"{mm['map_formed_count']:>9d}  "
              f"{mm['mean_map_error']:>9.4f}  "
              f"{mm['final_loc_median']:>9.4f}  "
              f"{naive['map_formed_count']:>9d}  "
              f"{naive['mean_map_error']:>9.4f}  "
              f"{err_ratio:>8.3f}")

        rows.append({
            "exp": 142,
            "rung": "M2",
            "seed": seed,
            "tab_map_accuracy": tab["map_accuracy"],
            "mm_map_formed": mm["map_formed_count"],
            "mm_mean_map_error": mm["mean_map_error"],
            "mm_final_loc_median": mm["final_loc_median"],
            "naive_map_formed": naive["map_formed_count"],
            "naive_mean_map_error": naive["mean_map_error"],
            "naive_mm_err_ratio": err_ratio,
        })

    # ---------------------------------------------------------------------------
    # Tallies and predicate evaluation
    # ---------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("TALLIES")
    print("=" * 80)

    # P1: tabular map_accuracy >= 14/16 in >= 7/8 seeds
    p1_per_seed = [r["map_accuracy"] >= (14 / 16) for r in tab_results]
    p1_count = sum(p1_per_seed)
    p1_holds = p1_count >= 7
    print(f"\nP1 twin sanity:")
    print(f"  Seeds with tabular map_accuracy >= 14/16: {p1_count}/8  "
          f"(need >= 7/8)  {'PASS' if p1_holds else 'FAIL'}")
    for s, r in zip(SEEDS, tab_results):
        print(f"    seed={s}: map_accuracy={r['map_accuracy']:.4f}  "
              f"({'pass' if p1_per_seed[s] else 'FAIL'})")

    # P2: moment-matched >= 14/16 colors formed AND final-500 median loc <= 0.5
    #     in >= 6/8 seeds
    p2_per_seed = [
        (mm["map_formed_count"] >= 14) and (mm["final_loc_median"] <= 0.5)
        for mm in mm_results
    ]
    p2_count = sum(p2_per_seed)
    p2_holds = p2_count >= 6
    print(f"\nP2 bootstrap (moment-matched):")
    print(f"  Seeds with map_formed >= 14 AND final_loc_med <= 0.5: {p2_count}/8  "
          f"(need >= 6/8)  {'PASS' if p2_holds else 'FAIL'}")
    for s, mm in zip(SEEDS, mm_results):
        print(f"    seed={s}: map_formed={mm['map_formed_count']}/16  "
              f"final_loc_med={mm['final_loc_median']:.4f}  "
              f"({'pass' if p2_per_seed[s] else 'FAIL'})")

    # P3: naive mean map error >= 1.5x moment-matched mean map error (across seeds)
    mm_mean_errors_all = np.array([r["mean_map_error"] for r in mm_results])
    naive_mean_errors_all = np.array([r["mean_map_error"] for r in naive_results])
    mean_mm_err = float(np.mean(mm_mean_errors_all))
    mean_naive_err = float(np.mean(naive_mean_errors_all))
    p3_ratio = mean_naive_err / mean_mm_err if mean_mm_err > 1e-9 else float("nan")
    p3_holds = p3_ratio >= 1.5
    p3_near_equal = p3_ratio < 1.2
    print(f"\nP3 wall-information asymmetry:")
    print(f"  Mean map error — moment-matched: {mean_mm_err:.4f}  naive: {mean_naive_err:.4f}")
    print(f"  Ratio naive/mm: {p3_ratio:.3f}  (need >= 1.5x for PASS; within 1.2x is finding)  "
          f"{'PASS' if p3_holds else ('FINDING: wall-info account is wrong' if p3_near_equal else 'FAIL (between 1.2x and 1.5x)')}")

    # ---------------------------------------------------------------------------
    # Verdict (three-way rule)
    # ---------------------------------------------------------------------------
    print()
    print("=" * 80)

    # Falsifiers:
    # - P1 fails -> NEGATIVE + MIGRATION HALT (instrument problem)
    # - P2 fails (< 6/8 seeds pass) -> NEGATIVE + MIGRATION HALT
    # - P3 fails (within 1.2x, p3_near_equal) -> finding logged, NOT a halt
    # - P3 fails (1.2x <= ratio < 1.5x) -> FAIL (partial miss, MIXED)

    p2_falsifier = not p2_holds  # < 6/8 seeds pass
    p1_falsifier = not p1_holds

    halt_triggers = []
    if p1_falsifier:
        halt_triggers.append("P1 FAILED — in-situ twin baseline broken (instrument problem)")
    if p2_falsifier:
        halt_triggers.append("P2 FAILED — continuous map learning does not bootstrap")

    if halt_triggers:
        verdict = "NEGATIVE"
        print(f"VERDICT: {verdict}")
        print("MIGRATION HALT")
        for t in halt_triggers:
            print(f"  Falsifier triggered: {t}")
    elif p1_holds and p2_holds and p3_holds:
        verdict = "POSITIVE"
        print(f"VERDICT: {verdict}")
        print("All predeclared properties satisfied. Migration thread may advance.")
    elif p1_holds and p2_holds and p3_near_equal:
        verdict = "MIXED"
        print(f"VERDICT: {verdict}")
        print("P1+P2 carry; P3 miss: naive ~ moment-matched (ratio {:.3f} < 1.2x).".format(p3_ratio))
        print("Finding: wall-information account is wrong at this scale; "
              "the moment-matched clamp does not produce the predicted asymmetry.")
        print("Not a halt (P2 carrying is the migration criterion).")
    elif p1_holds and p2_holds and not p3_holds:
        verdict = "MIXED"
        print(f"VERDICT: {verdict}")
        print("P1+P2 carry; P3 partial miss: ratio {:.3f} (1.2x <= ratio < 1.5x).".format(p3_ratio))
        print("Wall-information asymmetry weaker than predicted.")
    else:
        verdict = "MIXED"
        print(f"VERDICT: {verdict}")
        print("Unexpected combination — inspect tallies above.")

    print("=" * 80)

    # ---------------------------------------------------------------------------
    # JSON output
    # ---------------------------------------------------------------------------
    rows.append({
        "exp": 142,
        "rung": "M2",
        "seed": -1,
        "summary": True,
        "p1_count": p1_count,
        "p1_holds": p1_holds,
        "p2_count": p2_count,
        "p2_holds": p2_holds,
        "mean_mm_map_error": mean_mm_err,
        "mean_naive_map_error": mean_naive_err,
        "p3_ratio": p3_ratio,
        "p3_holds": p3_holds,
        "verdict": verdict,
    })

    out_path = Path(__file__).parent / "outputs" / "exp142_rows.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"\nJSON rows written to {out_path}")


if __name__ == "__main__":
    main()
