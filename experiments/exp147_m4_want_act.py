"""
Exp 147 — continuous-creature rung M4: want + act (the Exp 26 grounded-valence
mechanism and an Exp 30-style value-seeking policy, on the continuous substrate;
the human's word "continue with m4", 2026-06-10 — the growth wall rides along,
parked and documented).

Card loop/directions/continuous-creature.md (M4). Guardrail: any failed
predeclared property HALTS the migration for explicit human input.

Hypothesis: the RECIPE's want-and-act links survive the substrate port. (i) WANT:
accumulating value by predictability (value[obs] += exp(-H(predictive color
distribution at the current belief)) — the creature's own Exp 26 rule with the
continuous mixture predictive in place of the tabular A-column) concentrates value
on the colors of the RELIABLE half of a half-noisy world, and the tabular twin
ranks colors the same way on identical streams. (ii) ACT: a one-step lookahead
policy on the continuous value field V(s) = sum_k value_share_k * p(o=k|s)
(softmax over the 4 actions' predicted-position values, epsilon=0.1) carries the
creature into its preferred region far above chance. (iii) The Exp 26 signature:
the same architecture raised in the MIRRORED world (noise on the other half)
forms its favorite on the other side — history, not architecture, sets the want.

Setup: 4x4 grid, 16 DISTINCT colors (non-aliased — the parked M3 prediction
limitation does not contaminate this rung); NOISY LEFT HALF: cells with column
c <= 1 emit their true color with prob 0.6, else a uniform random color; right
half deterministic. Phase A T=4000: uniform-random wander, M2 map learning
(moment-matched clamp, per-color NIW), valence accumulation each step using the
predictive color distribution at the pre-update belief (the same distribution
already computed for surprise). Tabular twin on identical streams: the creature's
own equations + its own valence rule (predictability weight from A_hat at the MAP
cell). Phase B T=2000 (continuous agent only — the tabular creature has no nav
policy in creature.py; chance baselines serve as its stand-in, declared): action
= softmax over a in {0..3} of V(clamp(mu + delta_a)) / tau with tau=0.05,
epsilon=0.1 random; map learning and valence CONTINUE (registered experience).
MIRROR ARM: an independent creature per seed lives the same protocol in the
mirrored world (noise on column c >= 2). Seeds 0..7.

Definitions: value_share_k = value_counts_k / sum(value_counts); reliable set =
the 8 colors of the deterministic half; favorite = argmax value_share; preferred
region = the favorite color's true cell; occupancy measured over phase B.

Predictions (TRUE iff all):
- P1 want grounds: end-of-phase-A reliable-set value share > 0.75 (chance 0.5)
  in >= 7/8 seeds, AND Spearman rank correlation between continuous and tabular
  per-color value counts > 0.6 in >= 7/8 seeds.
- P2 act works: phase-B fraction of steps with true position in the favorite's
  cell > 4x the uniform baseline (1/16), AND > 2x that creature's own phase-A
  occupancy of the same cell, in >= 6/8 seeds.
- P3 history sets the want: the mirrored creature's favorite lies in ITS reliable
  half (the left) while the primary's lies in the right, in >= 6/8 seed-pairs.

Falsifiers (any HALTS the migration): P1 fails in >= 3/8 (valence does not ground
in predictability on this substrate), P2 fails in >= 3/8 (values are not
actionable), P3 fails in >= 3/8 (history does not differentiate want). Three-way
rule per PROTOCOL step 3.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr  # type: ignore

from active_loop.continuous import NIW, predictive_word_logprobs
from active_loop.creature_continuous import ContinuousPlace

# ---------------------------------------------------------------------------
# World constants  (copied from exp142)
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

# NIW prior parameters  (copied from exp142)
D = 2
KAPPA0 = 1.0
NU0 = 4.0
S0_SCALE = 0.35 ** 2 * (NU0 - D - 1)
S0 = S0_SCALE * np.eye(D)

# Policy parameters
TAU = 0.05       # softmax temperature for value policy
EPSILON = 0.1    # epsilon-greedy exploration fraction
SIGMA_EVAL = 0.01 * np.eye(D)  # declared: policy evaluates at candidate positions

# Phase lengths
T_A = 4000
T_B = 2000
SEEDS = list(range(8))

# Noise model
NOISE_PROB = 0.6  # reliable observation probability when in noisy half


# ---------------------------------------------------------------------------
# World helpers  (copied from exp142)
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


def cell_col(cell: int) -> int:
    """Return the column index (0..3) of a cell."""
    return cell % COLS


def sample_obs(true_cell: int, rng: np.random.Generator, noisy_left: bool) -> int:
    """Draw an observation from the noisy world.

    noisy_left=True  -> cols 0,1 are noisy (NOISE_PROB reliable); cols 2,3 deterministic
    noisy_left=False -> cols 2,3 are noisy (MIRROR arm); cols 0,1 deterministic
    """
    true_color = int(CMAP[true_cell])
    col = cell_col(true_cell)
    if noisy_left:
        is_noisy = col <= 1
    else:
        is_noisy = col >= 2
    if is_noisy:
        if rng.random() < NOISE_PROB:
            return true_color
        else:
            return int(rng.integers(0, N_COLORS))
    else:
        return true_color


def reliable_colors(noisy_left: bool) -> set:
    """Return the set of colors belonging to the deterministic (reliable) half."""
    result = set()
    for cell in range(N_CELLS):
        col = cell_col(cell)
        if noisy_left:
            # reliable = right half cols 2,3
            if col >= 2:
                result.add(int(CMAP[cell]))
        else:
            # reliable = left half cols 0,1
            if col <= 1:
                result.add(int(CMAP[cell]))
    return result


# ---------------------------------------------------------------------------
# Continuous agent
# ---------------------------------------------------------------------------

def _init_niws() -> list:
    """Initialize per-color NIW priors (copied from exp142)."""
    return [NIW(m=ARENA_CENTER.copy(), kappa=KAPPA0, nu=NU0, S=S0.copy())
            for _ in range(N_COLORS)]


def _init_cp() -> ContinuousPlace:
    """Initialize continuous place belief."""
    mu0 = ARENA_CENTER.copy()
    Sigma0 = 4.0 * np.eye(D)
    return ContinuousPlace(mu0, Sigma0, ARENA)


def _predictive_color_logprobs(cp_mu: np.ndarray, cp_Sigma: np.ndarray,
                                niws: list) -> np.ndarray:
    """Compute predictive color log-probabilities at the current belief.

    Uses predictive_word_logprobs with:
      word_mus    = [niw_k.expected_mu() for k in range(N_COLORS)]
      word_Sigmas = [diag(diag(E[Sigma_k])) + 1e-6*I for k in range(N_COLORS)]

    Mirrors how exp142 uses the emission params: diagonal-only for consistency
    with the place belief's diagonal structure.
    """
    word_mus = [niws[k].expected_mu() for k in range(N_COLORS)]
    word_Sigmas = []
    for k in range(N_COLORS):
        Sk_full = niws[k].expected_Sigma()
        Sk_diag = np.diag(np.diag(Sk_full)) + 1e-6 * np.eye(D)
        word_Sigmas.append(Sk_diag)
    return predictive_word_logprobs(cp_mu, cp_Sigma, word_mus, word_Sigmas)


def _value_field(cp_mu: np.ndarray, niws: list,
                 value_share: np.ndarray) -> float:
    """V(s) = sum_k value_share_k * p(o=k | s).

    Evaluates the field at position s=cp_mu using a small fixed emission
    covariance SIGMA_EVAL (declared: policy evaluates at candidate positions).
    """
    word_mus = [niws[k].expected_mu() for k in range(N_COLORS)]
    word_Sigmas = []
    for k in range(N_COLORS):
        Sk_full = niws[k].expected_Sigma()
        Sk_diag = np.diag(np.diag(Sk_full)) + 1e-6 * np.eye(D)
        word_Sigmas.append(Sk_diag)
    # p(o=k | s) via predictive at (s, SIGMA_EVAL)
    log_probs = predictive_word_logprobs(cp_mu, SIGMA_EVAL, word_mus, word_Sigmas)
    probs = np.exp(log_probs)
    return float(np.dot(value_share, probs))


def _clamp_pos(pos: np.ndarray) -> np.ndarray:
    """Clamp a 2-vector to the arena."""
    xmin, xmax, ymin, ymax = ARENA
    out = pos.copy()
    out[0] = float(np.clip(out[0], xmin, xmax))
    out[1] = float(np.clip(out[1], ymin, ymax))
    return out


def run_continuous(
    actions_a: np.ndarray,
    obs_a: np.ndarray,
    obs_b_fn,          # callable(cell, rng) -> obs  (live, phase B only)
    rng_b: np.random.Generator,
    noisy_left: bool,
) -> dict:
    """Run the continuous creature through phases A and B.

    Phase A (T_A steps): uniform-random actions (pre-generated), pre-generated obs.
    Phase B (T_B steps): value-seeking policy (eps-greedy softmax), live obs from
    the creature's own trajectory.

    Returns a dict with:
      - value_counts: shape (N_COLORS,)
      - niws: final list of NIW (for map inspection)
      - fav_cell_occ_A: phase-A occupancy fraction of favorite cell
      - fav_cell_occ_B: phase-B occupancy fraction of favorite cell
      - favorite: color index (argmax value_share after phase A)
      - fav_true_cell: the cell whose CMAP color equals favorite
    """
    cp = _init_cp()
    niws = _init_niws()
    value_counts = np.zeros(N_COLORS)

    true_cell = 0
    fav_cell_counts_a = np.zeros(N_CELLS)  # we fill fav cell after phase A ends

    # -------- Phase A --------
    cell_visits_a = np.zeros(N_CELLS, dtype=int)
    for t in range(T_A):
        obs = int(obs_a[t])

        # Pre-update predictive for valence (at current belief BEFORE place update)
        pre_mu = cp.mu
        pre_Sigma = cp.Sigma
        log_pred = _predictive_color_logprobs(pre_mu, pre_Sigma, niws)
        H = float(-np.sum(np.exp(log_pred) * log_pred))  # entropy in nats
        weight = math.exp(-H)
        value_counts[obs] += weight

        # Place update
        mu_k = niws[obs].expected_mu()
        Sk_full = niws[obs].expected_Sigma()
        Sigma_k = np.diag(np.diag(Sk_full)) + 1e-6 * np.eye(D)
        cp.update(mu_k, Sigma_k)

        # NIW soft update
        niws[obs] = niws[obs].update_moments(cp.mu, cp.Sigma)

        # Track cell visits
        cell_visits_a[true_cell] += 1

        # Act (pre-generated random)
        action = int(actions_a[t])
        true_cell = move(true_cell, action)

        # Predict (moment-matched clamp)
        delta = ACTION_DELTA[action]
        cp.predict_clamped_moments(delta, Q)

    # After phase A: compute favorite and phase-A occupancy
    value_share_a = value_counts / (value_counts.sum() + 1e-30)
    favorite = int(np.argmax(value_share_a))
    fav_true_cell = favorite  # identity cmap: color k lives at cell k

    occ_a = cell_visits_a[fav_true_cell] / T_A

    # -------- Phase B --------
    cell_visits_b = np.zeros(N_CELLS, dtype=int)

    for t in range(T_B):
        # Live observation from current cell
        obs = obs_b_fn(true_cell, rng_b)

        # Pre-update predictive for valence
        pre_mu = cp.mu
        pre_Sigma = cp.Sigma
        log_pred = _predictive_color_logprobs(pre_mu, pre_Sigma, niws)
        H = float(-np.sum(np.exp(log_pred) * log_pred))
        weight = math.exp(-H)
        value_counts[obs] += weight

        # Place update
        mu_k = niws[obs].expected_mu()
        Sk_full = niws[obs].expected_Sigma()
        Sigma_k = np.diag(np.diag(Sk_full)) + 1e-6 * np.eye(D)
        cp.update(mu_k, Sigma_k)

        # NIW soft update
        niws[obs] = niws[obs].update_moments(cp.mu, cp.Sigma)

        # Track cell visits
        cell_visits_b[true_cell] += 1

        # Policy: value-seeking with eps-greedy softmax
        cur_value_share = value_counts / (value_counts.sum() + 1e-30)

        cur_mu = cp.mu
        v_actions = np.empty(4)
        for a in range(4):
            candidate = _clamp_pos(cur_mu + ACTION_DELTA[a])
            v_actions[a] = _value_field(candidate, niws, cur_value_share)

        if rng_b.random() < EPSILON:
            action = int(rng_b.integers(0, 4))
        else:
            # Softmax over V / tau
            v_scaled = v_actions / TAU
            v_scaled -= v_scaled.max()  # numerical stability
            probs = np.exp(v_scaled)
            probs /= probs.sum()
            action = int(rng_b.choice(4, p=probs))

        true_cell = move(true_cell, action)

        # Predict (moment-matched clamp)
        delta = ACTION_DELTA[action]
        cp.predict_clamped_moments(delta, Q)

    occ_b = cell_visits_b[fav_true_cell] / T_B

    return {
        "value_counts": value_counts.copy(),
        "value_share_a": value_share_a,
        "favorite": favorite,
        "fav_true_cell": fav_true_cell,
        "occ_a": occ_a,
        "occ_b": occ_b,
    }


# ---------------------------------------------------------------------------
# Tabular twin  (copied from exp142, extended with valence)
# ---------------------------------------------------------------------------

def run_tabular(
    actions_a: np.ndarray,
    obs_a: np.ndarray,
    rng_pA: np.random.Generator,
) -> dict:
    """Run the tabular twin through phase A only.

    Valence rule: A_hat column at MAP cell, H of that column, weight = exp(-H).
    value_counts[obs] += weight  (the creature's exact rule from creature.py).

    Returns value_counts (N_COLORS,) and value_share (N_COLORS,).
    """
    # pA init: 0.1 uniform + 0.01*jitter
    pA = np.full((N_COLORS, N_CELLS), 0.1) + 0.01 * rng_pA.random((N_COLORS, N_CELLS))

    # B transition matrix
    B = np.zeros((N_CELLS, N_CELLS, 4))
    for s in range(N_CELLS):
        for a in range(4):
            s2 = move(s, a)
            B[s2, s, a] = 1.0

    qs = np.ones(N_CELLS) / N_CELLS
    true_cell = 0
    value_counts = np.zeros(N_COLORS)

    for t in range(T_A):
        obs = int(obs_a[t])  # same drawn observations as continuous agent

        # A_hat: column-normalized pA
        A_hat = pA.copy()
        col_sums = A_hat.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        A_hat = A_hat / col_sums

        # Value accumulation BEFORE belief update (pre-update A_hat at MAP cell)
        map_cell = int(np.argmax(qs))
        predicted_obs_dist = A_hat[:, map_cell]  # P(obs | map_cell)
        h_predicted = float(-np.sum(predicted_obs_dist * np.log(predicted_obs_dist + 1e-12)))
        weight = math.exp(-h_predicted)
        value_counts[obs] += weight

        # Belief update: qs_updated ∝ A_hat[obs, :] * qs
        qs_updated = A_hat[obs, :] * qs
        denom = qs_updated.sum()
        if denom > 0:
            qs_updated /= denom
        else:
            qs_updated = np.ones(N_CELLS) / N_CELLS

        # Dirichlet count learning
        pA[obs, :] += qs_updated

        # Act and move (pre-generated)
        action = int(actions_a[t])
        true_cell = move(true_cell, action)

        # Advance belief through B
        qs = B[:, :, action] @ qs_updated

    value_share = value_counts / (value_counts.sum() + 1e-30)
    return {
        "value_counts": value_counts.copy(),
        "value_share": value_share,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rows = []

    print("=" * 80)
    print("Exp 147 — continuous-creature rung M4: want + act")
    print("=" * 80)
    print()

    # Per-seed results
    results_primary = []    # continuous, noisy_left=True
    results_mirror = []     # continuous, noisy_left=False
    results_tab = []        # tabular, noisy_left=True (phase A only)
    results_tab_mirror = [] # tabular, noisy_left=False (phase A only)

    RELIABLE_PRIMARY = reliable_colors(noisy_left=True)   # cols 2,3
    RELIABLE_MIRROR = reliable_colors(noisy_left=False)    # cols 0,1

    for seed in SEEDS:
        rng_a = np.random.default_rng(1000 + seed)
        rng_obs_a_primary = np.random.default_rng(3000 + seed)
        rng_obs_a_mirror = np.random.default_rng(4000 + seed)

        # Phase A: pre-generate actions and observations
        actions_a = rng_a.integers(0, 4, size=T_A)

        # Pre-generate phase-A observations on the primary creature's trajectory
        # (both continuous and tabular see the SAME drawn observations)
        obs_a_primary = np.empty(T_A, dtype=int)
        obs_a_mirror = np.empty(T_A, dtype=int)

        # Walk the trajectory once to get true cells
        cell_traj_a = np.empty(T_A, dtype=int)
        c = 0
        for t in range(T_A):
            cell_traj_a[t] = c
            c = move(c, int(actions_a[t]))

        for t in range(T_A):
            tc = int(cell_traj_a[t])
            obs_a_primary[t] = sample_obs(tc, rng_obs_a_primary, noisy_left=True)
            obs_a_mirror[t] = sample_obs(tc, rng_obs_a_mirror, noisy_left=False)

        # Phase B rng (live observations from creature's own trajectory)
        rng_b_primary = np.random.default_rng(5000 + seed)
        rng_b_mirror = np.random.default_rng(6000 + seed)

        # Tabular jitter rng
        rng_pA_primary = np.random.default_rng(2000 + seed)
        rng_pA_mirror = np.random.default_rng(7000 + seed)

        # --- Primary continuous ---
        def obs_b_fn_primary(cell, rng):
            return sample_obs(cell, rng, noisy_left=True)

        res_primary = run_continuous(
            actions_a, obs_a_primary,
            obs_b_fn_primary, rng_b_primary,
            noisy_left=True,
        )
        results_primary.append(res_primary)

        # --- Mirror continuous ---
        def obs_b_fn_mirror(cell, rng):
            return sample_obs(cell, rng, noisy_left=False)

        res_mirror = run_continuous(
            actions_a, obs_a_mirror,
            obs_b_fn_mirror, rng_b_mirror,
            noisy_left=False,
        )
        results_mirror.append(res_mirror)

        # --- Tabular twin (primary world) ---
        res_tab = run_tabular(actions_a, obs_a_primary, rng_pA_primary)
        results_tab.append(res_tab)

        # --- Tabular twin (mirror world) ---
        res_tab_mirror = run_tabular(actions_a, obs_a_mirror, rng_pA_mirror)
        results_tab_mirror.append(res_tab_mirror)

    # ---------------------------------------------------------------------------
    # Compute per-seed metrics
    # ---------------------------------------------------------------------------

    print(f"{'Seed':>4}  "
          f"{'RelShare_C':>10}  "
          f"{'Spearman':>8}  "
          f"{'Fav_C':>5}  "
          f"{'FavHalf':>7}  "
          f"{'OccA_C':>7}  "
          f"{'OccB_C':>7}  "
          f"{'Fav_M':>5}  "
          f"{'FavHalf_M':>9}  "
          f"{'Spearman_M':>10}")
    print("-" * 90)

    p1_rel_share_pass = []
    p1_spearman_pass = []
    p2_pass = []
    p3_pass = []

    per_seed_data = []

    for seed in SEEDS:
        rp = results_primary[seed]
        rm = results_mirror[seed]
        rt = results_tab[seed]
        rtm = results_tab_mirror[seed]

        # P1a: reliable-set value share (primary, end of phase A)
        vs_a = rp["value_share_a"]
        rel_share = float(sum(vs_a[k] for k in RELIABLE_PRIMARY))
        p1a = rel_share > 0.75

        # P1b: Spearman rank correlation continuous vs tabular value counts (primary)
        vc_c = rp["value_counts"]
        vc_t = rt["value_counts"]
        sp_val, _ = spearmanr(vc_c, vc_t)
        p1b = float(sp_val) > 0.6

        p1_rel_share_pass.append(p1a)
        p1_spearman_pass.append(p1b)

        # P2: phase-B occupancy of favorite cell
        fav = rp["favorite"]
        occ_a = rp["occ_a"]
        occ_b = rp["occ_b"]
        baseline = 1.0 / N_CELLS  # = 1/16
        p2_seed = (occ_b > 4.0 * baseline) and (occ_b > 2.0 * occ_a)
        p2_pass.append(p2_seed)

        # Determine which half the favorite is in
        fav_cell = rp["fav_true_cell"]
        fav_col = cell_col(fav_cell)
        fav_in_right = fav_col >= 2  # primary's reliable half

        # P3: primary favorite in right (reliable), mirror favorite in left (reliable)
        fav_m = rm["favorite"]
        fav_m_cell = rm["fav_true_cell"]
        fav_m_col = cell_col(fav_m_cell)
        fav_m_in_left = fav_m_col <= 1  # mirror's reliable half

        p3_seed = fav_in_right and fav_m_in_left
        p3_pass.append(p3_seed)

        # Spearman for mirror
        vc_cm = rm["value_counts"]
        vc_tm = rtm["value_counts"]
        sp_m, _ = spearmanr(vc_cm, vc_tm)

        print(f"{seed:>4}  "
              f"{rel_share:>10.4f}  "
              f"{float(sp_val):>8.4f}  "
              f"{fav:>5d}  "
              f"{'right' if fav_in_right else 'left':>7}  "
              f"{occ_a:>7.4f}  "
              f"{occ_b:>7.4f}  "
              f"{fav_m:>5d}  "
              f"{'left' if fav_m_in_left else 'right':>9}  "
              f"{float(sp_m):>10.4f}")

        per_seed_data.append({
            "seed": seed,
            "rel_share": rel_share,
            "spearman": float(sp_val),
            "favorite": fav,
            "fav_col": fav_col,
            "fav_in_right": bool(fav_in_right),
            "occ_a": float(occ_a),
            "occ_b": float(occ_b),
            "p1a": bool(p1a),
            "p1b": bool(p1b),
            "p2": bool(p2_seed),
            "fav_m": int(fav_m),
            "fav_m_col": int(fav_m_col),
            "fav_m_in_left": bool(fav_m_in_left),
            "p3": bool(p3_seed),
            "spearman_m": float(sp_m),
        })

    # ---------------------------------------------------------------------------
    # Tallies
    # ---------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("TALLIES")
    print("=" * 80)

    p1a_count = sum(p1_rel_share_pass)
    p1b_count = sum(p1_spearman_pass)
    p1_per_seed = [a and b for a, b in zip(p1_rel_share_pass, p1_spearman_pass)]
    p1_count = sum(p1_per_seed)
    p1_holds = (p1a_count >= 7) and (p1b_count >= 7)

    print(f"\nP1 want grounds:")
    print(f"  P1a reliable-set value share > 0.75: {p1a_count}/8 seeds  (need >= 7/8)")
    print(f"  P1b Spearman > 0.6 (cont vs tab):    {p1b_count}/8 seeds  (need >= 7/8)")
    print(f"  P1 HOLDS: {p1_holds}")
    for d in per_seed_data:
        print(f"    seed={d['seed']}: rel_share={d['rel_share']:.4f} ({'pass' if d['p1a'] else 'FAIL'})  "
              f"spearman={d['spearman']:.4f} ({'pass' if d['p1b'] else 'FAIL'})")

    p2_count = sum(p2_pass)
    p2_holds = p2_count >= 6
    baseline = 1.0 / N_CELLS
    print(f"\nP2 act works (occ_B > 4x baseline={baseline:.4f} AND > 2x occ_A):")
    print(f"  Seeds pass: {p2_count}/8  (need >= 6/8)  {'PASS' if p2_holds else 'FAIL'}")
    for d in per_seed_data:
        print(f"    seed={d['seed']}: occ_A={d['occ_a']:.4f}  occ_B={d['occ_b']:.4f}  "
              f"4x_baseline={4*baseline:.4f}  2x_occA={2*d['occ_a']:.4f}  "
              f"({'pass' if d['p2'] else 'FAIL'})")

    p3_count = sum(p3_pass)
    p3_holds = p3_count >= 6
    print(f"\nP3 history sets the want (primary fav in right, mirror fav in left):")
    print(f"  Seeds pass: {p3_count}/8  (need >= 6/8)  {'PASS' if p3_holds else 'FAIL'}")
    for d in per_seed_data:
        print(f"    seed={d['seed']}: primary_fav_col={d['fav_col']} ({'right' if d['fav_in_right'] else 'left'})  "
              f"mirror_fav_col={d['fav_m_col']} ({'left' if d['fav_m_in_left'] else 'right'})  "
              f"({'pass' if d['p3'] else 'FAIL'})")

    # ---------------------------------------------------------------------------
    # Verdict (three-way rule)
    # ---------------------------------------------------------------------------
    print()
    print("=" * 80)

    # Falsifiers per spec:
    # P1 fails in >= 3/8 -> MIGRATION HALT
    # P2 fails in >= 3/8 -> MIGRATION HALT
    # P3 fails in >= 3/8 -> MIGRATION HALT

    p1_failures = 8 - p1_count
    p2_failures = 8 - p2_count
    p3_failures = 8 - p3_count

    halt_triggers = []
    if p1_failures >= 3:
        halt_triggers.append(
            f"P1 FAILED in {p1_failures}/8 seeds — valence does not ground in "
            f"predictability on this substrate"
        )
    if p2_failures >= 3:
        halt_triggers.append(
            f"P2 FAILED in {p2_failures}/8 seeds — values are not actionable"
        )
    if p3_failures >= 3:
        halt_triggers.append(
            f"P3 FAILED in {p3_failures}/8 seeds — history does not differentiate want"
        )

    if halt_triggers:
        verdict = "NEGATIVE"
        print(f"VERDICT: {verdict}")
        print("MIGRATION HALT")
        for ht in halt_triggers:
            print(f"  Falsifier triggered: {ht}")
    elif p1_holds and p2_holds and p3_holds:
        verdict = "POSITIVE"
        print(f"VERDICT: {verdict}")
        print("All predeclared properties satisfied. Migration thread advances to M5.")
    else:
        verdict = "MIXED"
        print(f"VERDICT: {verdict}")
        print("Partial result — inspect tallies.")
        if not p1_holds:
            print(f"  P1 PARTIAL: P1a={p1a_count}/8  P1b={p1b_count}/8")
        if not p2_holds:
            print(f"  P2 PARTIAL: {p2_count}/8")
        if not p3_holds:
            print(f"  P3 PARTIAL: {p3_count}/8")

    print("=" * 80)

    # ---------------------------------------------------------------------------
    # JSON output
    # ---------------------------------------------------------------------------
    rows = []
    for d in per_seed_data:
        rows.append({
            "exp": 147,
            "rung": "M4",
            **d,
        })
    rows.append({
        "exp": 147,
        "rung": "M4",
        "seed": -1,
        "summary": True,
        "p1a_count": p1a_count,
        "p1b_count": p1b_count,
        "p1_holds": p1_holds,
        "p2_count": p2_count,
        "p2_holds": p2_holds,
        "p3_count": p3_count,
        "p3_holds": p3_holds,
        "verdict": verdict,
    })

    out_path = Path(__file__).parent / "outputs" / "exp147_rows.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return super().default(obj)

    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, cls=_NpEncoder) + "\n")
    print(f"\nJSON rows written to {out_path}")


if __name__ == "__main__":
    main()
