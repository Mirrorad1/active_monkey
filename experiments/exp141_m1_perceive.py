"""
Exp 141 — continuous-creature rung M1: perceive under the motor anchor (non-aliased
world; the wall-clamp approximation declared and costed).

Migration directive "A" (loop/IDEAS.md, 2026-06-10); card
loop/directions/continuous-creature.md. Guardrail: ANY failed predeclared property
HALTS the migration thread for explicit human input.

Hypothesis: a Gaussian place belief with the known motor anchor (delta(a), clamped
mean + small process noise) and known per-color emission Gaussians localizes and
stays calibrated on the creature's grid world (non-aliased arm), with predictive
cost within a declared band of the exact tabular twin on identical streams.

Setup: 4x4 grid embedded in R^2 (cell (r,c) at (x=c, y=r)); 16 DISTINCT colors
(non-aliased); emissions: color k -> N(center_k, 0.35^2 I) (adjacent Mahalanobis
separation 2.86, the chapter's standard); dynamics: 0=up,1=down,2=left,3=right,
delta = (0,-1),(0,+1),(-1,0),(+1,0) in (x,y), TRUE position wall-clamped exactly as
World.move; belief predict: mu clamped to [0,3]^2, Q = 0.05^2 I (the DECLARED
approximation); prior N(center of arena, 4I). Tabular twin: exact known table
A[color, cell] (1 iff cmap[cell]==color), deterministic B, the creature's own update
equations in log space, identical action/observation streams. Arms: (i) WANDER
T=300, uniform random actions; (ii) WALL-STRESS T=120: 30 consecutive of each action
(left, up, right, down) — pinned to walls most of the run. Seeds 0..7.

Predictions (TRUE iff all):
- P1 localization: median final ||mu - true_pos|| <= 0.35 in >= 7/8 wander seeds;
  twin MAP cell == true cell at >= 95% of wander steps (exactness check).
- P2 clamp soundness: 95%-ellipse coverage of the true position at >= 85% of steps
  in >= 7/8 wander seeds, AND >= 75% in >= 7/8 wall-stress seeds, AND the posterior
  mean NEVER exits the arena by more than 0.5 in any run (the card's
  posterior-escapes falsifier; with mean-clamping this is structural — its check is
  an instrument sanity).
- P3 predictive parity: mean per-step pre-update observation NLL (continuous,
  predictive_word_logprobs at the current belief over the 16 color Gaussians) minus
  the twin's (exact, log-space) <= 0.5 nats, wander arm, every seed band: cell mean
  per seed, >= 7/8 seeds.

Falsifier (any triggers NEGATIVE and HALTS the migration): P1 localization fails, OR
coverage below band (the clamp approximation is unsound), OR mean escapes the arena,
OR NLL gap > 0.5 nats (the footprint approximation is unacceptably lossy in the
creature's world). Three-way rule per PROTOCOL step 3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp  # type: ignore

from active_loop.creature_continuous import ContinuousPlace
from active_loop.continuous import predictive_word_logprobs

# ---------------------------------------------------------------------------
# World constants
# ---------------------------------------------------------------------------

ROWS, COLS = 4, 4
N_CELLS = ROWS * COLS          # 16
N_COLORS = 16                   # non-aliased: cell i -> color i

# Cell (r, c) maps to position (x=c, y=r) in continuous space.
# Index: cell = r * COLS + c.
CELL_CENTERS = np.array(
    [[float(c), float(r)] for r in range(ROWS) for c in range(COLS)]
)  # shape (16, 2); CELL_CENTERS[cell] = (x, y)

# Arena bounds
ARENA = (0.0, float(COLS - 1), 0.0, float(ROWS - 1))  # (xmin, xmax, ymin, ymax)

# Action -> (dx, dy): 0=up means r-1 -> y-1, 1=down -> y+1, 2=left -> x-1, 3=right -> x+1
ACTION_DELTA = {
    0: np.array([0.0, -1.0]),   # up: row-1, y decreases
    1: np.array([0.0, +1.0]),   # down: row+1, y increases
    2: np.array([-1.0, 0.0]),   # left: col-1
    3: np.array([+1.0, 0.0]),   # right: col+1
}

# Process noise
Q_SCALE = 0.05
Q = Q_SCALE ** 2 * np.eye(2)

# Emission parameters
SIGMA_COLOR_SCALE = 0.35
Sigma_color = SIGMA_COLOR_SCALE ** 2 * np.eye(2)  # same for all 16 colors

# Tabular: A[color, cell] = 1.0 iff cmap[cell] == color; identity map
A_tabular = np.eye(N_COLORS)  # shape (16, 16): A[color, cell]

# Transition matrix B for tabular twin
# B[s', s, a] = 1 if move(s, a) == s'
def _build_B() -> np.ndarray:
    B = np.zeros((N_CELLS, N_CELLS, 4))
    for s in range(N_CELLS):
        r, c = divmod(s, COLS)
        for a in range(4):
            r2, c2 = r, c
            if a == 0:
                r2 = max(0, r - 1)
            elif a == 1:
                r2 = min(ROWS - 1, r + 1)
            elif a == 2:
                c2 = max(0, c - 1)
            else:
                c2 = min(COLS - 1, c + 1)
            s2 = r2 * COLS + c2
            B[s2, s, a] = 1.0
    return B


B_TABULAR = _build_B()  # (16, 16, 4)

SEEDS = list(range(8))


# ---------------------------------------------------------------------------
# Wall-clamped move (mirrors World.move exactly)
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
# Arm generators
# ---------------------------------------------------------------------------

def _wander_actions(rng: np.random.Generator, T: int = 300) -> np.ndarray:
    return rng.integers(0, 4, size=T)


def _wall_stress_actions(T_per_action: int = 30) -> np.ndarray:
    # 30 x left, 30 x up, 30 x right, 30 x down
    return np.concatenate([
        np.full(T_per_action, 2, dtype=int),  # left
        np.full(T_per_action, 0, dtype=int),  # up
        np.full(T_per_action, 3, dtype=int),  # right
        np.full(T_per_action, 1, dtype=int),  # down
    ])


# ---------------------------------------------------------------------------
# Continuous predictive NLL (pre-update, observation obs)
# ---------------------------------------------------------------------------

def _continuous_nll(mu: np.ndarray, Sigma: np.ndarray, obs: int) -> float:
    """Pre-update NLL for a single observation under the current Gaussian belief."""
    word_mus = [CELL_CENTERS[k] for k in range(N_COLORS)]
    word_Sigmas = [Sigma_color for _ in range(N_COLORS)]
    log_probs = predictive_word_logprobs(mu, Sigma, word_mus, word_Sigmas)
    return float(-log_probs[obs])


# ---------------------------------------------------------------------------
# Tabular twin log-space update and NLL
# ---------------------------------------------------------------------------

def _tabular_nll(logq: np.ndarray, obs: int) -> float:
    """Pre-update NLL for tabular twin.

    A is identity (non-aliased): p(obs=k | s=j) = 1 iff cmap[j]==k = 1 iff j==k.
    So p(obs=k) = exp(logq[k]).
    """
    # logq is normalized; p(obs) = sum_s exp(logq[s]) * A[obs, s] = exp(logq[obs])
    # because A is identity
    log_p_obs = logq[obs]  # already in log space, normalized
    return float(-log_p_obs)


def _tabular_obs_update(logq: np.ndarray, obs: int) -> np.ndarray:
    """Bayesian log-space observation update for tabular twin.

    logq_new[s] = logq[s] + log A[obs, s], then renormalize.
    A is identity, so log A[obs, s] = 0 if s==obs, -inf otherwise.
    This collapses to: logq_new = one-hot at obs.
    But we implement it generally for correctness in the non-aliased case.
    """
    log_A_obs = np.full(N_CELLS, -np.inf)
    # A[obs, s] = 1 iff cmap[s] == obs; cmap[s] = s (identity map)
    log_A_obs[obs] = 0.0  # log(1.0)
    logq_new = logq + log_A_obs
    logq_new = logq_new - float(logsumexp(logq_new))
    return logq_new


def _tabular_action_update(logq: np.ndarray, action: int) -> np.ndarray:
    """Advance tabular belief through deterministic B.

    B[:, s, a] is a one-hot: B[s', s, a] = 1 iff move(s, a) == s'.
    In log space: logq_new[s'] = logsumexp over {s : move(s,a) == s'} of logq[s].
    Since B is deterministic, each s' has exactly one source (possibly itself if wall).
    """
    logq_new = np.full(N_CELLS, -np.inf)
    for s2 in range(N_CELLS):
        # Find all sources mapping to s2 under action
        sources = [s for s in range(N_CELLS) if move(s, action) == s2]
        if sources:
            logq_new[s2] = float(logsumexp([logq[s] for s in sources]))
    logq_new = logq_new - float(logsumexp(logq_new))
    return logq_new


# ---------------------------------------------------------------------------
# Single-arm, single-seed run
# ---------------------------------------------------------------------------

def run_arm(seed: int, actions: np.ndarray) -> dict:
    """Run continuous + tabular twin on the given action sequence.

    Returns per-step records and summary statistics.
    """
    rng = np.random.default_rng(seed)  # not used for obs (deterministic world), kept for interface

    T = len(actions)
    true_cell = 0  # start at cell 0: (r=0, c=0), (x=0, y=0)

    # Continuous prior: N(center of arena, 4*I)
    mu0 = np.array([1.5, 1.5])  # center of [0,3]^2
    Sigma0 = 4.0 * np.eye(2)
    cp = ContinuousPlace(mu0, Sigma0, ARENA)

    # Tabular prior: uniform
    logq = np.full(N_CELLS, -np.log(N_CELLS))

    # Per-step tracking
    loc_errors = np.empty(T)
    coverage = np.empty(T, dtype=bool)
    cont_nlls = np.empty(T)
    tab_nlls = np.empty(T)
    map_correct = np.empty(T, dtype=bool)
    mu_arr = np.empty((T, 2))

    xmin, xmax, ymin, ymax = ARENA

    for t in range(T):
        true_xy = CELL_CENTERS[true_cell]  # (x, y)
        obs = true_cell  # cmap[cell] = cell (identity, non-aliased)

        # --- Pre-update predictive NLL ---
        cont_nlls[t] = _continuous_nll(cp.mu, cp.Sigma, obs)
        tab_nlls[t] = _tabular_nll(logq, obs)

        # --- Continuous: observation update ---
        cp.update(CELL_CENTERS[obs], Sigma_color)

        # --- Tabular: observation update ---
        logq = _tabular_obs_update(logq, obs)

        # --- Post-update metrics ---
        mu_cur = cp.mu
        mu_arr[t] = mu_cur
        loc_errors[t] = float(np.linalg.norm(mu_cur - true_xy))
        coverage[t] = cp.coverage_95(true_xy)

        # Check MAP cell of tabular twin
        map_cell = int(np.argmax(logq))
        map_correct[t] = (map_cell == true_cell)

        # --- Choose action, move ---
        action = int(actions[t])
        true_cell = move(true_cell, action)

        # --- Dynamics update ---
        delta = ACTION_DELTA[action]
        cp.predict(delta, Q)
        logq = _tabular_action_update(logq, action)

    # --- Arena escape check ---
    max_x_over = float(np.max(np.maximum(0.0, mu_arr[:, 0] - xmax)))
    max_x_under = float(np.max(np.maximum(0.0, xmin - mu_arr[:, 0])))
    max_y_over = float(np.max(np.maximum(0.0, mu_arr[:, 1] - ymax)))
    max_y_under = float(np.max(np.maximum(0.0, ymin - mu_arr[:, 1])))
    max_escape = max(max_x_over, max_x_under, max_y_over, max_y_under)

    # --- Summary statistics ---
    last50 = slice(max(0, T - 50), T)
    final_loc = float(np.median(loc_errors[last50]))
    cov_frac = float(np.mean(coverage))
    cont_nll_mean = float(np.mean(cont_nlls))
    tab_nll_mean = float(np.mean(tab_nlls))
    nll_gap = cont_nll_mean - tab_nll_mean
    map_correct_frac = float(np.mean(map_correct))

    return {
        "final_loc": final_loc,
        "cov_frac": cov_frac,
        "cont_nll_mean": cont_nll_mean,
        "tab_nll_mean": tab_nll_mean,
        "nll_gap": nll_gap,
        "map_correct_frac": map_correct_frac,
        "max_escape": max_escape,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results_wander = []
    results_wall = []

    # --- Wander arm ---
    print("=" * 70)
    print("ARM: WANDER (T=300, uniform random actions, seeds 0..7)")
    print("=" * 70)
    print(f"{'Seed':>4}  {'FinalLoc':>8}  {'CovFrac':>7}  {'ContNLL':>7}  "
          f"{'TabNLL':>6}  {'NLLGap':>7}  {'MAPcor':>6}  {'MaxEsc':>7}")
    print("-" * 70)

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        actions = _wander_actions(rng, T=300)
        r = run_arm(seed, actions)
        results_wander.append(r)
        print(f"{seed:>4}  {r['final_loc']:>8.4f}  {r['cov_frac']:>7.3f}  "
              f"{r['cont_nll_mean']:>7.4f}  {r['tab_nll_mean']:>6.4f}  "
              f"{r['nll_gap']:>7.4f}  {r['map_correct_frac']:>6.3f}  "
              f"{r['max_escape']:>7.4f}")

    # --- Wall-stress arm ---
    print()
    print("=" * 70)
    print("ARM: WALL-STRESS (T=120, 30x each: left/up/right/down)")
    print("=" * 70)
    print(f"{'Seed':>4}  {'FinalLoc':>8}  {'CovFrac':>7}  {'ContNLL':>7}  "
          f"{'TabNLL':>6}  {'NLLGap':>7}  {'MAPcor':>6}  {'MaxEsc':>7}")
    print("-" * 70)

    for seed in SEEDS:
        actions = _wall_stress_actions(T_per_action=30)
        r = run_arm(seed, actions)
        results_wall.append(r)
        print(f"{seed:>4}  {r['final_loc']:>8.4f}  {r['cov_frac']:>7.3f}  "
              f"{r['cont_nll_mean']:>7.4f}  {r['tab_nll_mean']:>6.4f}  "
              f"{r['nll_gap']:>7.4f}  {r['map_correct_frac']:>6.3f}  "
              f"{r['max_escape']:>7.4f}")

    # ---------------------------------------------------------------------------
    # Tallies and predicate evaluation
    # ---------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("TALLIES")
    print("=" * 70)

    # P1: localization
    # Wander: median final_loc <= 0.35 in >= 7/8 seeds
    p1_loc_pass = [r["final_loc"] <= 0.35 for r in results_wander]
    p1_loc_count = sum(p1_loc_pass)
    # Tabular twin MAP-correct >= 95% of wander steps in all seeds
    p1_map_pass = [r["map_correct_frac"] >= 0.95 for r in results_wander]
    p1_map_count = sum(p1_map_pass)
    p1_holds = (p1_loc_count >= 7) and (p1_map_count == 8)
    print(f"P1 localization:")
    print(f"  Wander seeds with median final loc <= 0.35: {p1_loc_count}/8  (need >= 7/8)  {'PASS' if p1_loc_count >= 7 else 'FAIL'}")
    print(f"  Wander seeds with MAP-correct >= 95%:       {p1_map_count}/8  (need 8/8)     {'PASS' if p1_map_count == 8 else 'FAIL'}")
    print(f"  P1 holds: {p1_holds}")

    # P2: clamp soundness
    # Wander: cov_frac >= 0.85 in >= 7/8 seeds
    p2_wander_cov_pass = [r["cov_frac"] >= 0.85 for r in results_wander]
    p2_wander_cov_count = sum(p2_wander_cov_pass)
    # Wall-stress: cov_frac >= 0.75 in >= 7/8 seeds
    p2_wall_cov_pass = [r["cov_frac"] >= 0.75 for r in results_wall]
    p2_wall_cov_count = sum(p2_wall_cov_pass)
    # Mean never exits arena by more than 0.5 in ANY run
    all_escapes = [r["max_escape"] for r in results_wander + results_wall]
    p2_no_escape = all(e <= 0.5 for e in all_escapes)
    max_escape_any = max(all_escapes)
    p2_holds = (p2_wander_cov_count >= 7) and (p2_wall_cov_count >= 7) and p2_no_escape
    print(f"\nP2 clamp soundness:")
    print(f"  Wander seeds with cov_frac >= 0.85:     {p2_wander_cov_count}/8  (need >= 7/8)  {'PASS' if p2_wander_cov_count >= 7 else 'FAIL'}")
    print(f"  Wall-stress seeds with cov_frac >= 0.75: {p2_wall_cov_count}/8  (need >= 7/8)  {'PASS' if p2_wall_cov_count >= 7 else 'FAIL'}")
    print(f"  Mean never escapes by > 0.5: max_escape={max_escape_any:.6f}  {'PASS (structural)' if p2_no_escape else 'FAIL'}")
    print(f"  P2 holds: {p2_holds}")

    # P3: predictive parity
    # Wander: nll_gap <= 0.5 in >= 7/8 seeds
    p3_gap_pass = [r["nll_gap"] <= 0.5 for r in results_wander]
    p3_gap_count = sum(p3_gap_pass)
    p3_holds = (p3_gap_count >= 7)
    print(f"\nP3 predictive parity (wander):")
    gaps_str = " ".join(f"{r['nll_gap']:.4f}" for r in results_wander)
    print(f"  NLL gaps per seed: [{gaps_str}]")
    print(f"  Seeds with NLL gap <= 0.5 nats: {p3_gap_count}/8  (need >= 7/8)  {'PASS' if p3_gap_count >= 7 else 'FAIL'}")
    print(f"  P3 holds: {p3_holds}")

    # ---------------------------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------------------------

    # Falsifiers (any -> NEGATIVE + HALT):
    falsifiers = []
    if not p1_holds:
        falsifiers.append("P1 localization failed")
    if not p2_holds:
        if p2_wander_cov_count < 7:
            falsifiers.append("P2 wander coverage below 85% band")
        if p2_wall_cov_count < 7:
            falsifiers.append("P2 wall-stress coverage below 75% band")
        if not p2_no_escape:
            falsifiers.append(f"P2 mean escaped arena (max_escape={max_escape_any:.4f} > 0.5)")
    if not p3_holds:
        falsifiers.append("P3 NLL gap > 0.5 nats in > 1/8 seeds")

    all_pass = p1_holds and p2_holds and p3_holds

    print()
    print("=" * 70)
    if falsifiers:
        verdict = "NEGATIVE"
        print(f"VERDICT: {verdict}")
        print("MIGRATION HALT")
        for f in falsifiers:
            print(f"  Falsifier triggered: {f}")
    elif all_pass:
        verdict = "POSITIVE"
        print(f"VERDICT: {verdict}")
        print("All predeclared properties satisfied. Migration thread may advance.")
    else:
        verdict = "MIXED"
        print(f"VERDICT: {verdict}")
        print("Some properties held, some failed. Review above.")
    print("=" * 70)

    # ---------------------------------------------------------------------------
    # JSON output
    # ---------------------------------------------------------------------------

    rows = []
    for seed_idx, seed in enumerate(SEEDS):
        for arm_name, results in [("wander", results_wander), ("wall_stress", results_wall)]:
            r = results[seed_idx]
            rows.append({
                "exp": 141,
                "rung": "M1",
                "arm": arm_name,
                "seed": seed,
                "final_loc_median": r["final_loc"],
                "cov_frac": r["cov_frac"],
                "cont_nll_mean": r["cont_nll_mean"],
                "tab_nll_mean": r["tab_nll_mean"],
                "nll_gap": r["nll_gap"],
                "map_correct_frac": r["map_correct_frac"],
                "max_escape": r["max_escape"],
            })

    rows.append({
        "exp": 141,
        "rung": "M1",
        "arm": "summary",
        "seed": -1,
        "p1_loc_count": p1_loc_count,
        "p1_map_count": p1_map_count,
        "p1_holds": p1_holds,
        "p2_wander_cov_count": p2_wander_cov_count,
        "p2_wall_cov_count": p2_wall_cov_count,
        "p2_no_escape": p2_no_escape,
        "p2_max_escape": max_escape_any,
        "p2_holds": p2_holds,
        "p3_gap_count": p3_gap_count,
        "p3_holds": p3_holds,
        "verdict": verdict,
        "falsifiers": falsifiers,
    })

    out_path = Path(__file__).parent / "outputs" / "exp141_rows.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"\nJSON rows written to {out_path}")


if __name__ == "__main__":
    main()
