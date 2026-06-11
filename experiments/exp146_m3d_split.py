"""
Exp 146 — continuous-creature rung M3d: the SPLIT operator under the validated
live-probation test (the human's chosen resumption, "yes m3d", 2026-06-10).

Card loop/directions/continuous-creature.md (M3d). Guardrail: a third failure of
the surprise arm HALTS the migration with the park-and-continue-at-M4
recommendation (three mechanistically distinct failures = a documented wall).

Hypothesis: Exp 145 pinned the valley — ADD creates a coverage hole (a new narrow
component steals weight from the broad one covering all cells: helps ~1/4, hurts
~3/4, so every honest short-horizon test must reject it). SPLIT is the
coverage-preserving move: divide the alarmed color's WIDEST component along its
leading eigendirection — children at m +- sqrt(lambda1) * v, each inheriting half
the parent's weight and kappa, leading eigenvalue quartered (the children jointly
span the parent's footprint) — so predictive density is approximately maintained
everywhere at the moment of the move, and burn-in EM then tightens the children
onto the true sub-clusters. Prediction: probation surprise does NOT surge (the
anti-valley signature), splits pass the honest probation test, surprise descends
below the detector threshold, and the alarm finally quiets.

Setup: identical to Exp 145 except the move — three aliased layouts (rng 7/11/13),
seeds 0..7, phase 1 T=2000 unimodal, phase 2 T=8000 with per-color alarms (mean
of last 50 observations >= 0.7), round-robin scheduling over alarmed colors with
budget (4 components/color), and live-probation acceptance (install provisionally
after burn-in EM; keep iff the color's probation mean (its observations during
400 steps) <= pre-spawn mean (its last 100 observations) - 0.1; revert restores
the color's snapshot). SPLIT details: target = the alarmed color's component with
the largest tr of expected covariance; eigendecompose the FULL expected
covariance (2x2, off-diagonals included; the diag projection remains only inside
the place update, as before — declared); children means m +- sqrt(lambda1)*v;
child expected covariance = parent's with lambda1 -> lambda1/4 (other eigenvalue
kept); child weight = parent/2; child kappa = max(parent/2, 0.5), nu = kappa+3,
S = childCov*(nu-3). Then Exp 144's burn-in EM (10 iterations, that color's
replay pairs) over the color's FULL post-split component set, then probation.
Eval = final 1000 steps. Tabular twin sanity as before.

Predictions (TRUE iff all):
- P1 the alarm answered (the bar, third attempt): final-1000 mean predictive
  surprise <= phase-1 plateau - 0.4 nats AND zero ceiling events in the final
  1000 AND components >= 3 for >= 3/4 colors — each in >= 6/8 seeds per layout,
  all three layouts.
- P2 the anti-valley signature: mean over ALL split attempts of (probation mean -
  pre-spawn mean) <= +0.1 nats, per layout (vs Exp 145's surges of +0.6 to +3.5)
  — the coverage-preserving move is benign even when not kept.
- P3 no regression: final-500 median localization <= 0.5 in >= 7/8 seeds per
  layout.
Diagnostics (logged, not conjuncts): keep-rate among split attempts; per-color
final component counts vs true multiplicity; the demoted replay vote contingency.

Falsifiers: HALT = P1's surprise arm fails in >= 3/8 seeds in >= 2/3 layouts —
THIRD growth design fails; MIGRATION HALT with the park recommendation. P2
failing while P1 carries = the move works for the wrong reason — MIXED, log it.
P3 failing = HALT (regression). Three-way rule per PROTOCOL step 3.
"""
from __future__ import annotations

# NOTE: Mixture machinery copied from exp145 (which copied from exp144/exp143).
# ONLY behavioral changes from exp145:
#   1. The spawn-candidate construction is replaced by the SPLIT construction:
#      - Target = the alarmed color's component with largest tr(expected_covariance)
#      - Eigendecompose the FULL 2x2 expected covariance (np.linalg.eigh)
#      - lambda1 = larger eigenvalue, v = its eigenvector
#      - Children means: m +- sqrt(lambda1) * v
#      - Child kappa = max(parent_kappa/2, 0.5), nu = kappa+3
#      - Child expected covariance = parent's, but leading eigenvalue quartered
#      - Child weight = parent_weight / 2
#      - Parent component is REMOVED from the color's component list; two children added
#   2. Per-attempt record adds: parent_tr_cov, separation (2*sqrt(lambda1))
#   3. P2 is the anti-valley signature: mean(probation_mean - pre_spawn_mean) <= +0.1
#      instead of probation honesty (no "sustained benefit" framing)
#   4. HALT text updated: third growth design / MIGRATION HALT + park recommendation
#   5. JSON rung = "M3d"; output file = exp146_rows.json
#
# All other logic (alarms, round-robin, probation, snapshot/revert, replay-vote
# diagnostic, EM burn-in, tabular twin) is identical to exp145.

import copy
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from active_loop.continuous import NIW
from active_loop.creature_continuous import ContinuousPlace

# ---------------------------------------------------------------------------
# Detector constants — copied from exp145 (creature.py source-of-truth)
# ---------------------------------------------------------------------------
CEILING_MEAN_THRESH: float = 0.7    # creature.py line 56
CEILING_SLOPE_THRESH: float = 5e-4  # creature.py line 59
SURPRISE_WINDOW: int = 200          # creature.py line 52

# ---------------------------------------------------------------------------
# World constants — copied from exp145
# ---------------------------------------------------------------------------

ROWS, COLS = 4, 4
N_CELLS = ROWS * COLS   # 16
N_COLORS = 4            # ALIASED: 4 colors, 4 cells each
CELLS_PER_COLOR = 4

# Cell centers: (x=c, y=r)
CELL_CENTERS = np.array(
    [[float(c), float(r)] for r in range(ROWS) for c in range(COLS)]
)  # shape (16, 2)

ARENA = (0.0, float(COLS - 1), 0.0, float(ROWS - 1))  # (0, 3, 0, 3)
ARENA_CENTER = np.array([1.5, 1.5])

# Action -> (dx, dy)
ACTION_DELTA = {
    0: np.array([0.0, -1.0]),  # up
    1: np.array([0.0, +1.0]),  # down
    2: np.array([-1.0, 0.0]),  # left
    3: np.array([+1.0, 0.0]),  # right
}

Q_SCALE = 0.05
Q_diag = np.array([Q_SCALE ** 2, Q_SCALE ** 2])
Q = np.diag(Q_diag)

D = 2
KAPPA0 = 1.0
NU0 = 4.0
S0_SCALE = 0.35 ** 2 * (NU0 - D - 1)  # = 0.35^2 * 1.0
S0 = S0_SCALE * np.eye(D)

T_PHASE1 = 2000
T_PHASE2 = 8000           # same as M3c
T_TOTAL = T_PHASE1 + T_PHASE2

SPAWN_BUDGET = 4          # max components per color
SPAWN_INTERVAL = 200      # check every N steps during phase 2 if no probation running
REPLAY_WINDOW = 400       # replay pairs for burn-in and demoted NLL scoring
FINAL_EVAL_WINDOW = 1000  # final steps of phase 2 for grading
P1_PLATEAU_WINDOW = 500   # last N steps of phase 1 for plateau

BURN_IN_ITERS = 10        # EM iterations during burn-in
BURN_IN_COV_FLOOR = 1e-4  # floor for Cov_j diagonal entries

# M3c/M3d constants (identical to exp145)
COLOR_SURPRISE_WINDOW = 50    # per-color deque size for alarm check
ALARM_THRESH = 0.7            # mean per-color surprise threshold for alarm
PRE_SPAWN_WINDOW = 100        # last N color observations for pre-spawn mean
PROBATION_STEPS = 400         # live steps before keep/revert decision

SEEDS = list(range(8))

# Three layout seeds as specified
LAYOUT_SEEDS = [7, 11, 13]


# ---------------------------------------------------------------------------
# Build a CMAP from a given rng seed — copied from exp145
# ---------------------------------------------------------------------------

def build_cmap(layout_rng_seed: int) -> np.ndarray:
    """Build aliased CMAP: 4 colors x 4 cells, shuffled by rng(layout_rng_seed)."""
    rng = np.random.default_rng(layout_rng_seed)
    perm = rng.permutation(N_CELLS)
    cmap = np.empty(N_CELLS, dtype=int)
    for color_idx in range(N_COLORS):
        for slot in range(CELLS_PER_COLOR):
            cmap[perm[color_idx * CELLS_PER_COLOR + slot]] = color_idx
    return cmap


# ---------------------------------------------------------------------------
# Compute true within-color scatter traces — copied from exp145
# ---------------------------------------------------------------------------

def true_scatter_traces(cmap: np.ndarray) -> np.ndarray:
    """For each color k, compute tr(cov of its 4 cell centers, ddof=0).

    Returns array shape (N_COLORS,).
    """
    traces = np.empty(N_COLORS, dtype=float)
    for k in range(N_COLORS):
        cells_k = [i for i in range(N_CELLS) if cmap[i] == k]
        centers_k = CELL_CENTERS[cells_k]  # shape (4, 2)
        cov_k = np.cov(centers_k.T, ddof=0)  # shape (2, 2)
        traces[k] = float(np.trace(cov_k))
    return traces


# ---------------------------------------------------------------------------
# Grid world move (wall-clamped) — copied from exp145
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
# Diagonal 2x2 Gaussian utilities — copied from exp145
# ---------------------------------------------------------------------------

def _diag_log_integral(
    mu_p: np.ndarray, Sigma_p_diag: np.ndarray,
    m_k: np.ndarray, ESigma_k_diag: np.ndarray,
) -> float:
    """Log integral of N(s; m_k, ESigma_k) * N(s; mu_p, Sigma_p) over R^2.

    For diagonal 2x2:
        log I = 0.5*(logdet(Sk) - logdet(Sk + Sp)) - 0.5*maha
    where Sk = diag(ESigma_k_diag), Sp = diag(Sigma_p_diag).
    All done with scalars; no linalg.

    Declared: diagonal 2x2, scalar operations only.
    """
    sk0, sk1 = ESigma_k_diag[0], ESigma_k_diag[1]
    sp0, sp1 = Sigma_p_diag[0], Sigma_p_diag[1]
    logdet_Sk = math.log(sk0) + math.log(sk1)
    c0 = sk0 + sp0
    c1 = sk1 + sp1
    logdet_C = math.log(c0) + math.log(c1)
    d0 = mu_p[0] - m_k[0]
    d1 = mu_p[1] - m_k[1]
    maha = d0 * d0 / c0 + d1 * d1 / c1
    return 0.5 * (logdet_Sk - logdet_C) - 0.5 * maha


def _diag_gaussian_product(
    mu_a: np.ndarray, Sigma_a_diag: np.ndarray,
    mu_b: np.ndarray, Sigma_b_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Product of two diagonal Gaussians -> (mu, Sigma_diag).

    Scalar arithmetic per axis.  Returns (mu_out, Sigma_diag_out).
    """
    lam_a0 = 1.0 / Sigma_a_diag[0]
    lam_a1 = 1.0 / Sigma_a_diag[1]
    lam_b0 = 1.0 / Sigma_b_diag[0]
    lam_b1 = 1.0 / Sigma_b_diag[1]

    lam0 = lam_a0 + lam_b0
    lam1 = lam_a1 + lam_b1

    sig_out = np.array([1.0 / lam0, 1.0 / lam1])
    mu_out = np.array([
        (lam_a0 * mu_a[0] + lam_b0 * mu_b[0]) / lam0,
        (lam_a1 * mu_a[1] + lam_b1 * mu_b[1]) / lam1,
    ])
    return mu_out, sig_out


# ---------------------------------------------------------------------------
# Mixture emission: compute predictive log p(color) and responsibilities
# — copied from exp145
# ---------------------------------------------------------------------------

def mixture_predictive_logprobs(
    mu_p: np.ndarray,
    Sigma_p_diag: np.ndarray,
    components: list[list[tuple[float, NIW]]],  # components[color] = [(w, niw), ...]
) -> np.ndarray:
    """Compute normalized log p(color k) under the pre-update mixture predictive.

    For color k with components (w_kj, NIW_kj):
        logI_kj = _diag_log_integral(mu_p, Sigma_p_diag, m_kj, E[Sigma_kj]_diag)
        log p(k) = logsumexp_j(log w_kj + logI_kj)   [unnormalized over k]
    Normalize over k via logsumexp.

    Returns log_probs shape (N_COLORS,), logsumexp == 0.
    """
    log_unnorm = np.empty(N_COLORS, dtype=float)
    for k in range(N_COLORS):
        comps = components[k]
        log_terms = []
        for (w_kj, niw_kj) in comps:
            m_kj = niw_kj.m
            ESigma_kj = niw_kj.expected_Sigma()
            ESigma_kj_diag = np.array([ESigma_kj[0, 0], ESigma_kj[1, 1]])
            ESigma_kj_diag = np.maximum(ESigma_kj_diag, 1e-9)
            logI = _diag_log_integral(mu_p, Sigma_p_diag, m_kj, ESigma_kj_diag)
            log_terms.append(math.log(max(w_kj, 1e-300)) + logI)
        log_terms_arr = np.array(log_terms)
        log_unnorm[k] = float(np.logaddexp.reduce(log_terms_arr))
    log_sum = float(np.logaddexp.reduce(log_unnorm))
    return log_unnorm - log_sum


def mixture_emission_moments(
    mu_p: np.ndarray,
    Sigma_p_diag: np.ndarray,
    k: int,
    components: list[list[tuple[float, NIW]]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Moment-match the mixture emission for observed color k to a single Gaussian.

    Returns (mu_mix, Sigma_mix_diag, hard_idx) where hard_idx is the argmax
    responsibility component index for the NIW update.

    Declared approximation: moment-matching to diagonal Gaussian.

    Responsibilities r_j propto w_kj * exp(logI_kj).
    Product per component: diagonal Gaussian product of (mu_p, Sigma_p_diag)
    with (m_kj, E[Sigma_kj]_diag).
    Moment-match: mu = sum r_j mu_j; Sigma_ii = sum r_j (Sigma_j_ii + mu_j_i^2) - mu_i^2.
    """
    comps = components[k]
    n = len(comps)
    log_r = np.empty(n, dtype=float)

    prod_mus = []
    prod_sigs = []

    for j, (w_kj, niw_kj) in enumerate(comps):
        m_kj = niw_kj.m
        ESigma_kj = niw_kj.expected_Sigma()
        ESigma_kj_diag = np.maximum(
            np.array([ESigma_kj[0, 0], ESigma_kj[1, 1]]), 1e-9
        )
        logI = _diag_log_integral(mu_p, Sigma_p_diag, m_kj, ESigma_kj_diag)
        log_r[j] = math.log(max(w_kj, 1e-300)) + logI

        mu_j, sig_j_diag = _diag_gaussian_product(mu_p, Sigma_p_diag, m_kj, ESigma_kj_diag)
        prod_mus.append(mu_j)
        prod_sigs.append(sig_j_diag)

    log_r_sum = float(np.logaddexp.reduce(log_r))
    r = np.exp(log_r - log_r_sum)  # shape (n,)

    mu_mix = np.zeros(2)
    for j in range(n):
        mu_mix += r[j] * prod_mus[j]

    Sigma_mix_diag = np.zeros(2)
    for i in range(2):
        e2 = sum(r[j] * (prod_sigs[j][i] + prod_mus[j][i] ** 2) for j in range(n))
        Sigma_mix_diag[i] = e2 - mu_mix[i] ** 2

    Sigma_mix_diag = np.maximum(Sigma_mix_diag, 1e-9)

    hard_idx = int(np.argmax(r))
    return mu_mix, Sigma_mix_diag, hard_idx


# ---------------------------------------------------------------------------
# Surprise-ceiling detector — copied from exp145
# ---------------------------------------------------------------------------

def check_ceiling(surprise_buf: deque) -> bool:
    """Return True if the ceiling conjunction fires.

    Conditions: len(buf) == SURPRISE_WINDOW AND
        mean(buf) > CEILING_MEAN_THRESH AND |slope| < CEILING_SLOPE_THRESH.
    Slope via polyfit degree 1.
    """
    if len(surprise_buf) < SURPRISE_WINDOW:
        return False
    arr = np.array(surprise_buf)
    mean_s = float(np.mean(arr))
    if mean_s <= CEILING_MEAN_THRESH:
        return False
    xs = np.arange(len(arr), dtype=float)
    slope = float(np.polyfit(xs, arr, 1)[0])
    return abs(slope) < CEILING_SLOPE_THRESH


# ---------------------------------------------------------------------------
# Replay NLL scorer — copied from exp145 (demoted to diagnostic only)
# ---------------------------------------------------------------------------

def replay_nll(
    replay_buf: deque,  # deque of (obs_color, mu_stored, Sigma_diag_stored)
    components: list[list[tuple[float, NIW]]],
) -> float:
    """Mean -log p(obs | mu_stored, Sigma_diag_stored) over replay buffer."""
    if not replay_buf:
        return float("inf")
    total = 0.0
    for (obs_k, mu_s, Sigma_s_diag) in replay_buf:
        log_probs = mixture_predictive_logprobs(mu_s, Sigma_s_diag, components)
        total += -log_probs[obs_k]
    return total / len(replay_buf)


# ---------------------------------------------------------------------------
# Deep-copy utilities — copied from exp145
# ---------------------------------------------------------------------------

def _copy_components(
    components: list[list[tuple[float, NIW]]],
) -> list[list[tuple[float, NIW]]]:
    """Return a deep copy of the components structure."""
    result = []
    for color_comps in components:
        result.append([(w, copy.deepcopy(niw)) for (w, niw) in color_comps])
    return result


def _copy_color_components(
    color_comps: list[tuple[float, NIW]],
) -> list[tuple[float, NIW]]:
    """Return a deep copy of a single color's component list."""
    return [(w, copy.deepcopy(niw)) for (w, niw) in color_comps]


def _copy_counts(counts: list[list[int]]) -> list[list[int]]:
    return [list(c) for c in counts]


# ---------------------------------------------------------------------------
# Burn-in EM for a single color's mixture — copied from exp145
# ---------------------------------------------------------------------------

def burnin_em_color(
    color_comps: list[tuple[float, NIW]],
    color_replay: list[tuple[np.ndarray, np.ndarray]],  # list of (mean_i, Sigma_diag_i)
    n_iters: int = BURN_IN_ITERS,
) -> list[tuple[float, NIW]]:
    """Run burn-in EM on one color's component set using that color's replay pairs.

    Each replay pair (mean_i, Sigma_diag_i) represents a posterior uncertainty:
    the true location is modeled as N(mean_i, diag(Sigma_diag_i)).

    E-step responsibilities:
        r_ij propto w_j * N(mean_i; m_j, Cov_j + diag(Sigma_i))
    where N is the diagonal Gaussian density.

    M-step:
        w_j = sum_i r_ij / n
        m_j = weighted mean
        Cov_j = weighted scatter + weighted mean of Sigma_i
                floored at BURN_IN_COV_FLOOR * I on diagonal

    Returns new component list as (weight, NIW) using NIW write-back convention:
        kappa_j = max(n_j, 0.5),  nu_j = kappa_j + 3
        m_j = weighted mean,  S_j = Cov_j * (nu_j - 3)
        (nu_j - 3 = kappa_j, so S_j = Cov_j * kappa_j)
    """
    n = len(color_replay)
    if n == 0 or len(color_comps) == 0:
        return color_comps  # nothing to do

    n_comp = len(color_comps)

    # Extract initial weights, means, covariances from NIW
    weights = np.array([w for (w, _) in color_comps])
    weights = np.maximum(weights, 1e-300)
    weights /= weights.sum()

    # Get initial component means and expected covariances (diagonal)
    comp_means = np.array([niw.m for (_, niw) in color_comps])  # (n_comp, 2)
    comp_covs = np.array([
        np.diag(niw.expected_Sigma())  # diagonal entries only (2,)
        for (_, niw) in color_comps
    ])  # (n_comp, 2)
    comp_covs = np.maximum(comp_covs, BURN_IN_COV_FLOOR)

    # Replay observations: means and diag covariances
    obs_means = np.array([mu for (mu, _) in color_replay])   # (n, 2)
    obs_sigmas = np.array([sig for (_, sig) in color_replay])  # (n, 2)

    for _ in range(n_iters):
        # E-step: compute log responsibilities for each (obs_i, comp_j) pair
        log_r = np.empty((n, n_comp), dtype=float)
        for j in range(n_comp):
            mj = comp_means[j]
            cj = comp_covs[j]  # diagonal (2,)
            for i in range(n):
                # Effective covariance: Cov_j + Sigma_i
                eff_cov = cj + obs_sigmas[i]  # (2,)
                eff_cov = np.maximum(eff_cov, 1e-9)
                # log N(obs_means[i]; mj, diag(eff_cov))
                d0 = obs_means[i, 0] - mj[0]
                d1 = obs_means[i, 1] - mj[1]
                log_det = math.log(eff_cov[0]) + math.log(eff_cov[1])
                maha = d0 * d0 / eff_cov[0] + d1 * d1 / eff_cov[1]
                log_lik = -0.5 * (2 * math.log(2.0 * math.pi) + log_det + maha)
                log_r[i, j] = math.log(max(weights[j], 1e-300)) + log_lik

        # Normalize responsibilities row-wise
        log_r_max = log_r.max(axis=1, keepdims=True)
        r = np.exp(log_r - log_r_max)  # relative; will normalize
        r_sum = r.sum(axis=1, keepdims=True)
        r_sum = np.where(r_sum == 0, 1.0, r_sum)
        r = r / r_sum  # shape (n, n_comp), rows sum to 1

        # M-step
        n_j = r.sum(axis=0)  # shape (n_comp,); effective counts
        n_j_safe = np.maximum(n_j, 1e-9)

        new_weights = n_j / n
        new_weights = np.maximum(new_weights, 1e-300)
        new_weights /= new_weights.sum()

        new_means = (r.T @ obs_means) / n_j_safe[:, None]  # (n_comp, 2)

        new_covs = np.zeros((n_comp, 2), dtype=float)
        for j in range(n_comp):
            diff = obs_means - new_means[j]  # (n, 2)
            # Weighted scatter: sum_i r_ij * (diff_i * diff_i + Sigma_i)
            scatter = (r[:, j, None] * (diff ** 2 + obs_sigmas)).sum(axis=0)
            new_covs[j] = scatter / n_j_safe[j]
        new_covs = np.maximum(new_covs, BURN_IN_COV_FLOOR)

        weights = new_weights
        comp_means = new_means
        comp_covs = new_covs

    # Write back as NIW components using declared convention:
    #   kappa_j = max(n_j, 0.5),  nu_j = kappa_j + 3
    #   m_j = comp_means[j],  S_j = Cov_j_full * (nu_j - 3) = Cov_j_full * kappa_j
    # Cov_j is diagonal so S_j = diag(comp_covs[j]) * kappa_j
    result = []
    for j in range(n_comp):
        kappa_j = float(max(n_j[j], 0.5))
        nu_j = kappa_j + 3.0
        m_j = comp_means[j].copy()
        # S_j = Cov_j * kappa_j (diagonal)
        S_j = np.diag(comp_covs[j] * kappa_j)
        # Enforce NIW nu >= d+2 = 4
        if nu_j < D + 2:
            nu_j = float(D + 2)
            kappa_j = nu_j - 3.0
        new_niw = NIW(m=m_j, kappa=kappa_j, nu=nu_j, S=S_j)
        result.append((float(weights[j]), new_niw))

    return result


# ---------------------------------------------------------------------------
# SPLIT operator — M3d: replaces the ADD operator from exp145
# ---------------------------------------------------------------------------

def split_component(
    color_comps: list[tuple[float, NIW]],
) -> tuple[list[tuple[float, NIW]], int, float, float]:
    """Split the widest component (largest tr(expected_Sigma)) of color_comps.

    SPLIT details (per docstring spec):
      - Target: component j* with largest tr(E[Sigma_j]) — the widest one
      - Eigendecompose the FULL 2x2 expected covariance via np.linalg.eigh
        (off-diagonals included; declared: full 2x2 eig for split geometry,
         diagonal projection used only inside the place update as before)
      - lambda1 = larger eigenvalue, v = its unit eigenvector
      - Children means: m_parent +- sqrt(lambda1) * v
      - Child kappa = max(parent_kappa / 2, 0.5)
      - Child nu = child_kappa + 3 (enforced >= D+2=4)
      - Child expected covariance = parent's, with lambda1 -> lambda1/4 (other
        eigenvalue kept), expressed back as S = child_Cov * (child_nu - 3)
        where child_Cov is reconstructed from modified eigenvalues in eigen-basis
      - Child weight = parent_weight / 2
      - Parent is REMOVED; two children replace it in the component list

    Returns:
      (new_comps, target_idx, parent_tr_cov, separation)
      where separation = 2 * sqrt(lambda1) (distance between child means)

    If only one component: splits it (no guard needed — alarm fires only when
    >= 1 component exists and budget > 0, which allows up to SPAWN_BUDGET-1
    further adds; since we start with 1, after a keep we have 2, budget=3, etc.)
    """
    # Find target: widest component by tr(expected_Sigma)
    target_idx = 0
    best_tr = -1.0
    for j, (w_j, niw_j) in enumerate(color_comps):
        ES = niw_j.expected_Sigma()
        tr_j = float(np.trace(ES))
        if tr_j > best_tr:
            best_tr = tr_j
            target_idx = j

    parent_w, parent_niw = color_comps[target_idx]
    parent_ES = parent_niw.expected_Sigma()  # full 2x2
    parent_tr_cov = float(np.trace(parent_ES))

    # Eigendecompose FULL 2x2 expected covariance
    # np.linalg.eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(parent_ES)
    # lambda1 = larger eigenvalue (index 1), v1 = its column
    lambda1 = float(eigenvalues[1])
    v1 = eigenvectors[:, 1].copy()  # unit eigenvector for lambda1
    lambda2 = float(eigenvalues[0])

    # Guard: lambda1 must be positive; floor to avoid sqrt(0)
    lambda1 = max(lambda1, 1e-9)
    separation = 2.0 * math.sqrt(lambda1)

    # Child means
    offset = math.sqrt(lambda1) * v1
    m_child_A = parent_niw.m + offset
    m_child_B = parent_niw.m - offset

    # Child kappa, nu
    child_kappa = float(max(parent_niw.kappa / 2.0, 0.5))
    child_nu = child_kappa + 3.0
    if child_nu < D + 2:
        child_nu = float(D + 2)
        child_kappa = child_nu - 3.0

    # Child expected covariance: parent's eigenvectors, lambda1 quartered, lambda2 kept
    # Reconstruct: child_Cov = V * diag(lambda1/4, lambda2) * V^T
    child_lambda1 = lambda1 / 4.0
    child_lambda2 = lambda2
    # child_Cov = v1 * child_lambda1 * v1^T + v2 * child_lambda2 * v2^T
    v2 = eigenvectors[:, 0].copy()
    child_Cov = (child_lambda1 * np.outer(v1, v1)
                 + child_lambda2 * np.outer(v2, v2))
    # Enforce positive-definiteness floor on diagonal
    child_Cov[0, 0] = max(child_Cov[0, 0], BURN_IN_COV_FLOOR)
    child_Cov[1, 1] = max(child_Cov[1, 1], BURN_IN_COV_FLOOR)

    # S = child_Cov * (nu - 3) = child_Cov * child_kappa
    child_S = child_Cov * (child_nu - 3.0)

    # Child weight
    child_w = parent_w / 2.0

    # Build children NIW
    niw_A = NIW(m=m_child_A.copy(), kappa=child_kappa, nu=child_nu, S=child_S.copy())
    niw_B = NIW(m=m_child_B.copy(), kappa=child_kappa, nu=child_nu, S=child_S.copy())

    # Build new component list: remove parent, insert children
    new_comps = [
        (w, copy.deepcopy(niw))
        for j, (w, niw) in enumerate(color_comps)
        if j != target_idx
    ]
    new_comps.append((child_w, niw_A))
    new_comps.append((child_w, niw_B))

    # Re-normalize weights (parent is removed, children together have parent_w)
    total_w = sum(w for (w, _) in new_comps)
    if total_w > 1e-300:
        new_comps = [(w / total_w, niw) for (w, niw) in new_comps]

    return new_comps, target_idx, parent_tr_cov, separation


# ---------------------------------------------------------------------------
# Tabular twin — copied from exp145 (sanity only)
# ---------------------------------------------------------------------------

def run_tabular(actions: np.ndarray, cmap: np.ndarray) -> dict:
    """Run tabular twin (creature's own equations) on aliased world.

    Declared: linear-space per exp142 conventions.

    Returns final_500_map_frac: fraction of final-500-step timesteps where
    MAP cell == true cell.
    """
    rng_pA = np.random.default_rng(9999)
    pA = np.full((N_COLORS, N_CELLS), 0.1) + 0.01 * rng_pA.random((N_COLORS, N_CELLS))

    B = np.zeros((N_CELLS, N_CELLS, 4))
    for s in range(N_CELLS):
        for a in range(4):
            s2 = move(s, a)
            B[s2, s, a] = 1.0

    qs = np.ones(N_CELLS) / N_CELLS
    true_cell = 0

    map_correct = np.empty(T_TOTAL, dtype=bool)

    for t in range(T_TOTAL):
        obs = int(cmap[true_cell])

        A_hat = pA.copy()
        col_sums = A_hat.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        A_hat = A_hat / col_sums

        qs_updated = A_hat[obs, :] * qs
        denom = qs_updated.sum()
        if denom > 0:
            qs_updated /= denom
        else:
            qs_updated = np.ones(N_CELLS) / N_CELLS

        pA[obs, :] += qs_updated

        map_cell = int(np.argmax(qs_updated))
        map_correct[t] = (map_cell == true_cell)

        action = int(actions[t])
        true_cell = move(true_cell, action)
        qs = B[:, :, action] @ qs_updated

    final_500 = map_correct[T_TOTAL - 500:]
    return {"final_500_map_frac": float(np.mean(final_500))}


# ---------------------------------------------------------------------------
# Continuous agent with mixture + detector + M3d live-probation SPLIT growth
# ---------------------------------------------------------------------------

def run_continuous(actions: np.ndarray, cmap: np.ndarray) -> dict:
    """Run the continuous agent with M3d live-probation EM split.

    M3d changes from M3c (exp145):
      - SPLIT operator replaces ADD: the widest component of the alarmed color
        is split along its leading eigendirection; no new point is inserted.
      - Per-attempt record adds: parent_tr_cov, separation (2*sqrt(lambda1))
      - Budget consumption: a KEPT split consumes 1 from spawn_budget[color]
        (net +1 component after split = 2 children - 1 parent)
      - All other logic (alarms, round-robin, probation, snapshot/revert,
        replay NLL diagnostic, burn-in EM over full post-split component set)
        identical to exp145.

    Returns a dict with all per-seed metrics needed for P1/P2/P3 grading.
    """
    # --- Place prior: diagonal N(arena_center, 4I) ---
    mu0 = ARENA_CENTER.copy()
    Sigma0_diag = np.array([4.0, 4.0])
    cp = ContinuousPlace(mu0, np.diag(Sigma0_diag), ARENA)

    # --- Per-color mixture: 1 component per color initially ---
    components: list[list[tuple[float, NIW]]] = []
    counts: list[list[int]] = []
    for k in range(N_COLORS):
        niw0 = NIW(m=ARENA_CENTER.copy(), kappa=KAPPA0, nu=NU0, S=S0.copy())
        components.append([(1.0, niw0)])
        counts.append([1])

    # Spawn budget tracker
    spawn_budget = [SPAWN_BUDGET] * N_COLORS

    # Global surprise buffer (rolling deque) — for global detector / quiet check
    surprise_buf: deque = deque(maxlen=SURPRISE_WINDOW)

    # Replay buffer: (obs_color, mu_at_predict, Sigma_diag_at_predict)
    replay_buf: deque = deque(maxlen=REPLAY_WINDOW)

    # Per-color surprise deque(50) for alarm checking (M3c/M3d)
    color_surprise_bufs: list[deque] = [deque(maxlen=COLOR_SURPRISE_WINDOW) for _ in range(N_COLORS)]

    # Per-color larger deque(PRE_SPAWN_WINDOW) for pre-spawn mean (M3c/M3d)
    color_pre_spawn_bufs: list[deque] = [deque(maxlen=PRE_SPAWN_WINDOW) for _ in range(N_COLORS)]

    # Round-robin scheduling — last attempt step per color (M3c/M3d)
    last_attempt_step: list[int] = [-1] * N_COLORS
    color_attempt_counts: list[int] = [0] * N_COLORS

    # Probation state (M3c/M3d — identical structure to exp145)
    probation_color: int = -1
    probation_start_phase2_t: int = -1
    probation_pre_spawn_mean: float = float("inf")
    probation_color_snap: list[tuple[float, NIW]] = []
    probation_color_counts_snap: list[int] = []
    probation_observations: list[float] = []

    # M3d: per-spawn record for contingency + P2 anti-valley analysis
    # Each entry: dict with color, parent_tr_cov, separation, pre_spawn_mean,
    #             probation_mean, delta, kept (bool), replay_vote_keep (bool)
    spawn_records: list[dict] = []

    # --- Phase 1 tracking ---
    phase1_loc_errors = np.empty(T_PHASE1)
    phase1_ceiling_events = 0
    phase1_surprise_vals = []

    # --- Phase 2 tracking ---
    phase2_loc_errors = np.empty(T_PHASE2)
    phase2_ceiling_events_total = 0
    phase2_final_ceiling_events = 0
    spawns_kept = 0
    spawns_reverted = 0
    phase2_surprise_vals = []

    true_cell = 0

    for t in range(T_TOTAL):
        obs_k = int(cmap[true_cell])

        # Current place diagonal
        Sigma_p_diag = np.array([cp.Sigma[0, 0], cp.Sigma[1, 1]])
        Sigma_p_diag = np.maximum(Sigma_p_diag, 1e-9)
        mu_p = cp.mu

        # --- Record in replay buffer BEFORE update ---
        replay_buf.append((obs_k, mu_p.copy(), Sigma_p_diag.copy()))

        # --- Per-step surprise: -log p(obs_k) under pre-update mixture ---
        log_probs = mixture_predictive_logprobs(mu_p, Sigma_p_diag, components)
        surprise_t = float(-log_probs[obs_k])
        surprise_buf.append(surprise_t)

        is_phase1 = t < T_PHASE1
        phase2_t = t - T_PHASE1

        if is_phase1:
            phase1_loc_errors[t] = float(np.linalg.norm(mu_p - CELL_CENTERS[true_cell]))
            phase1_surprise_vals.append(surprise_t)

            # Check detector (armed in phase 1)
            if check_ceiling(surprise_buf):
                phase1_ceiling_events += 1

        else:
            # --- Phase 2 ---
            phase2_loc_errors[phase2_t] = float(np.linalg.norm(mu_p - CELL_CENTERS[true_cell]))
            phase2_surprise_vals.append(surprise_t)

            # Update per-color surprise tracking
            color_surprise_bufs[obs_k].append(surprise_t)
            color_pre_spawn_bufs[obs_k].append(surprise_t)

            # Collect probation observation if that color is on probation
            if probation_color == obs_k:
                probation_observations.append(surprise_t)

            # Global ceiling check
            is_ceiling = check_ceiling(surprise_buf)
            if is_ceiling:
                phase2_ceiling_events_total += 1
                if phase2_t >= T_PHASE2 - FINAL_EVAL_WINDOW:
                    phase2_final_ceiling_events += 1

            # --- Probation resolution: check if probation window expired ---
            if probation_color >= 0:
                elapsed = phase2_t - probation_start_phase2_t
                if elapsed >= PROBATION_STEPS:
                    pc = probation_color

                    # Compute probation mean (observations of pc color during probation)
                    if probation_observations:
                        prob_mean = float(np.mean(probation_observations))
                    else:
                        prob_mean = float("inf")

                    keep = (prob_mean <= probation_pre_spawn_mean - 0.1)

                    # Update last spawn record with probation result
                    if spawn_records:
                        spawn_records[-1]["probation_mean"] = prob_mean
                        spawn_records[-1]["delta"] = probation_pre_spawn_mean - prob_mean
                        spawn_records[-1]["kept"] = keep

                    if keep:
                        spawns_kept += 1
                        spawn_budget[pc] -= 1
                        # Other colors keep their probation-period learning
                    else:
                        spawns_reverted += 1
                        # Restore ONLY that color's snapshot
                        components[pc] = probation_color_snap
                        counts[pc] = probation_color_counts_snap

                    # Clear probation state
                    probation_color = -1
                    probation_start_phase2_t = -1
                    probation_pre_spawn_mean = float("inf")
                    probation_color_snap = []
                    probation_color_counts_snap = []
                    probation_observations = []

            # --- Split attempt every SPAWN_INTERVAL steps, only if no probation running ---
            if (probation_color < 0
                    and phase2_t > 0
                    and phase2_t % SPAWN_INTERVAL == 0):

                # Compute per-color alarm: mean >= ALARM_THRESH AND budget > 0
                alarmed_colors = []
                for k in range(N_COLORS):
                    if spawn_budget[k] > 0 and len(color_surprise_bufs[k]) > 0:
                        mean_k = float(np.mean(color_surprise_bufs[k]))
                        if mean_k >= ALARM_THRESH:
                            alarmed_colors.append(k)

                if alarmed_colors:
                    # Round-robin: pick the alarmed color with min last_attempt_step
                    spawn_color = min(alarmed_colors, key=lambda k: (last_attempt_step[k], k))
                    last_attempt_step[spawn_color] = phase2_t
                    color_attempt_counts[spawn_color] += 1

                    # Pre-spawn mean: mean of color's last PRE_SPAWN_WINDOW observations
                    pre_spawn_buf = list(color_pre_spawn_bufs[spawn_color])
                    if pre_spawn_buf:
                        pre_spawn_mean = float(np.mean(pre_spawn_buf))
                    else:
                        pre_spawn_mean = float("inf")

                    # Snapshot this color's mixture BEFORE splitting
                    snap_comps = _copy_color_components(components[spawn_color])
                    snap_counts = list(counts[spawn_color])

                    # Save pre-spawn full-replay NLL for demoted vote
                    pre_spawn_components_copy = _copy_components(components)
                    nll_old = replay_nll(replay_buf, pre_spawn_components_copy)

                    # --- M3d: SPLIT the widest component ---
                    (split_comps,
                     target_idx,
                     parent_tr_cov,
                     separation) = split_component(components[spawn_color])
                    components[spawn_color] = split_comps

                    # Burn-in EM on this color's full post-split component set
                    # using that color's replay pairs
                    color_replay_pairs = [
                        (mu_s.copy(), sig_s.copy())
                        for (obs_c, mu_s, sig_s) in replay_buf
                        if obs_c == spawn_color
                    ]
                    if color_replay_pairs:
                        components[spawn_color] = burnin_em_color(
                            components[spawn_color],
                            color_replay_pairs,
                        )

                    # Sync counts to match new component count
                    n_new = len(components[spawn_color])
                    while len(counts[spawn_color]) < n_new:
                        counts[spawn_color].append(1)
                    # Trim if somehow reduced (shouldn't happen, but guard)
                    counts[spawn_color] = counts[spawn_color][:n_new]

                    # Demoted replay vote (diagnostic — what exp144 would have decided)
                    nll_new = replay_nll(replay_buf, components)
                    replay_vote_keep = (nll_new < nll_old - 1e-6)

                    # INSTALL PROVISIONALLY — start probation
                    probation_color = spawn_color
                    probation_start_phase2_t = phase2_t
                    probation_pre_spawn_mean = pre_spawn_mean
                    probation_color_snap = snap_comps
                    probation_color_counts_snap = snap_counts
                    probation_observations = []

                    # Record split attempt
                    spawn_records.append({
                        "color": spawn_color,
                        "parent_tr_cov": parent_tr_cov,
                        "separation": separation,
                        "pre_spawn_mean": pre_spawn_mean,
                        "probation_mean": float("nan"),
                        "delta": float("nan"),
                        "kept": None,
                        "replay_vote_keep": replay_vote_keep,
                        "nll_old": nll_old,
                        "nll_new": nll_new,
                    })

        # --- Place update: moment-match mixture emission for obs_k ---
        mu_mix, Sigma_mix_diag, hard_idx = mixture_emission_moments(
            mu_p, Sigma_p_diag, obs_k, components
        )
        cp.update(mu_mix, np.diag(Sigma_mix_diag))

        # --- NIW hard-assignment update ---
        post_mu = cp.mu
        post_Sigma_diag = np.array([cp.Sigma[0, 0], cp.Sigma[1, 1]])
        post_Sigma_diag = np.maximum(post_Sigma_diag, 1e-9)

        old_niw = components[obs_k][hard_idx][1]
        new_niw = old_niw.update_moments(post_mu, np.diag(post_Sigma_diag))
        components[obs_k][hard_idx] = (components[obs_k][hard_idx][0], new_niw)

        # Update weights from running counts
        counts[obs_k][hard_idx] += 1
        total_k = sum(counts[obs_k])
        components[obs_k] = [(counts[obs_k][j] / total_k, niw)
                             for j, (_, niw) in enumerate(components[obs_k])]

        # --- Act and move ---
        action = int(actions[t])
        true_cell = move(true_cell, action)

        # --- Predict: moment-matched clamped ---
        cp.predict_clamped_moments(ACTION_DELTA[action], Q)

    # --- Handle any in-flight probation at end of phase 2 (treat as not resolved — revert) ---
    if probation_color >= 0:
        pc = probation_color
        components[pc] = probation_color_snap
        counts[pc] = probation_color_counts_snap
        spawns_reverted += 1
        if spawn_records:
            spawn_records[-1]["kept"] = False
            spawn_records[-1]["probation_mean"] = (
                float(np.mean(probation_observations)) if probation_observations else float("nan")
            )
        probation_color = -1

    # --- Metrics ---
    final_comps_per_color = [len(components[k]) for k in range(N_COLORS)]

    p1_final500_loc_median = float(np.median(phase1_loc_errors[T_PHASE1 - 500:]))
    p2_final500_loc_median = float(np.median(phase2_loc_errors[T_PHASE2 - 500:]))

    # Phase-1 plateau: mean over last P1_PLATEAU_WINDOW steps
    phase1_plateau = float(np.mean(phase1_surprise_vals[-P1_PLATEAU_WINDOW:]))

    # Final surprise: mean over last FINAL_EVAL_WINDOW steps of phase 2
    final_surprise = float(np.mean(phase2_surprise_vals[-FINAL_EVAL_WINDOW:]))

    drop = phase1_plateau - final_surprise

    # P2 (M3d): anti-valley signature — mean(probation_mean - pre_spawn_mean) per layout
    # Computed at the layout level in main(); here we return all decided records
    # so main() can pool them.

    # P2 (carried from M3c for reference): sustained benefit among kept spawns
    kept_records = [r for r in spawn_records if r.get("kept") is True]
    sustained_benefit_kept = 0
    for r in kept_records:
        k = r["color"]
        if len(color_surprise_bufs[k]) > 0:
            final_color_mean = float(np.mean(color_surprise_bufs[k]))
        else:
            final_color_mean = float("inf")
        if final_color_mean < r["pre_spawn_mean"]:
            sustained_benefit_kept += 1

    sustained_benefit_frac = (
        sustained_benefit_kept / len(kept_records)
        if kept_records
        else float("nan")
    )

    return {
        "plateau": phase1_plateau,
        "final_surprise": final_surprise,
        "drop": drop,
        "p1_final500_loc_median": p1_final500_loc_median,
        "p1_ceiling_events": phase1_ceiling_events,
        "phase2_final_ceiling_events": phase2_final_ceiling_events,
        "final_comps_per_color": final_comps_per_color,
        "spawns_kept": spawns_kept,
        "spawns_reverted": spawns_reverted,
        "p2_final500_loc_median": p2_final500_loc_median,
        "sustained_benefit_frac": sustained_benefit_frac,
        "sustained_benefit_kept": sustained_benefit_kept,
        "total_kept": len(kept_records),
        "spawn_records": spawn_records,
        "color_attempt_counts": color_attempt_counts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("Exp 146 — continuous-creature rung M3d: SPLIT growth under live-probation")
    print("=" * 80)
    print()

    # Precompute cmaps for all three layouts
    cmaps = [build_cmap(s) for s in LAYOUT_SEEDS]

    all_rows_json = []

    # Per-layout results
    layout_verdicts = []

    for layout_idx, (layout_seed, cmap) in enumerate(zip(LAYOUT_SEEDS, cmaps)):
        print()
        print(f"{'=' * 80}")
        print(f"LAYOUT {layout_idx + 1}  (rng({layout_seed}))")
        print(f"{'=' * 80}")
        print()
        print(f"  ALIASED CMAP (4 colors x 4 cells):")
        for r in range(ROWS):
            row_colors = [int(cmap[r * COLS + c]) for c in range(COLS)]
            print(f"    row {r}: {row_colors}")
        print(f"  CMAP array: {cmap.tolist()}")
        print()
        print(f"  Color -> cells and centers:")
        for k in range(N_COLORS):
            cells_k = [i for i in range(N_CELLS) if cmap[i] == k]
            centers_k = [CELL_CENTERS[i].tolist() for i in cells_k]
            print(f"    color {k}: cells {cells_k}  centers {centers_k}")
        print()

        cont_results = []
        tab_results = []

        print(f"{'Seed':>4}  "
              f"{'Plateau':>7}  "
              f"{'Final':>7}  "
              f"{'Drop':>6}  "
              f"{'FinCeil':>7}  "
              f"{'Comps':>12}  "
              f"{'Kept':>4}  "
              f"{'Revt':>4}  "
              f"{'LocMed':>6}  "
              f"{'MnDelta':>8}  "
              f"{'TwinF500':>8}")
        print("-" * 105)

        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed)
            actions = rng.integers(0, 4, size=T_TOTAL)

            tab = run_tabular(actions, cmap)
            cont = run_continuous(actions, cmap)

            tab_results.append(tab)
            cont_results.append(cont)

            comps_str = str(cont["final_comps_per_color"])

            # Mean attempt delta for this seed (P2 anti-valley metric)
            decided_recs = [rec for rec in cont["spawn_records"]
                            if rec.get("kept") is not None
                            and not math.isnan(rec.get("delta", float("nan")))]
            if decided_recs:
                # delta = pre_spawn_mean - probation_mean (positive = improvement)
                # P2 asks: mean(probation_mean - pre_spawn_mean) = mean(-delta)
                mean_attempt_neg_delta = float(np.mean([-rec["delta"] for rec in decided_recs]))
            else:
                mean_attempt_neg_delta = float("nan")

            mn_delta_str = (f"{mean_attempt_neg_delta:+.4f}"
                            if not math.isnan(mean_attempt_neg_delta) else "   N/A")

            print(f"{seed:>4}  "
                  f"{cont['plateau']:>7.4f}  "
                  f"{cont['final_surprise']:>7.4f}  "
                  f"{cont['drop']:>6.3f}  "
                  f"{cont['phase2_final_ceiling_events']:>7d}  "
                  f"{comps_str:>12}  "
                  f"{cont['spawns_kept']:>4d}  "
                  f"{cont['spawns_reverted']:>4d}  "
                  f"{cont['p2_final500_loc_median']:>6.4f}  "
                  f"{mn_delta_str:>8}  "
                  f"{tab['final_500_map_frac']:>8.4f}")

            all_rows_json.append({
                "exp": 146,
                "rung": "M3d",
                "layout_seed": layout_seed,
                "layout_idx": layout_idx + 1,
                "seed": seed,
                "plateau": cont["plateau"],
                "final_surprise": cont["final_surprise"],
                "drop": cont["drop"],
                "p1_ceiling_events": cont["p1_ceiling_events"],
                "phase2_final_ceiling_events": cont["phase2_final_ceiling_events"],
                "final_comps_per_color": cont["final_comps_per_color"],
                "spawns_kept": cont["spawns_kept"],
                "spawns_reverted": cont["spawns_reverted"],
                "p1_final500_loc_median": cont["p1_final500_loc_median"],
                "p2_final500_loc_median": cont["p2_final500_loc_median"],
                "sustained_benefit_frac": (
                    cont["sustained_benefit_frac"]
                    if not math.isnan(cont["sustained_benefit_frac"])
                    else None
                ),
                "total_kept": cont["total_kept"],
                "color_attempt_counts": cont["color_attempt_counts"],
                "mean_attempt_neg_delta": (
                    mean_attempt_neg_delta
                    if not math.isnan(mean_attempt_neg_delta) else None
                ),
                "twin_final500_map_frac": tab["final_500_map_frac"],
            })

        # ---------------------------------------------------------------
        # P2 anti-valley summary — pooled across seeds for this layout
        # ---------------------------------------------------------------
        print()
        print(f"  P2 Anti-Valley Summary (Layout {layout_idx + 1}):")
        print(f"  (mean(probation_mean - pre_spawn_mean) per attempt, vs Exp 145 surges +0.6..+3.5)")
        all_neg_deltas = []
        total_attempts = 0
        total_kept_layout_p2 = 0
        total_reverted_layout_p2 = 0
        for cont in cont_results:
            for rec in cont["spawn_records"]:
                if rec.get("kept") is not None and not math.isnan(rec.get("delta", float("nan"))):
                    all_neg_deltas.append(-rec["delta"])  # probation_mean - pre_spawn_mean
                    total_attempts += 1
                    if rec["kept"]:
                        total_kept_layout_p2 += 1
                    else:
                        total_reverted_layout_p2 += 1

        if all_neg_deltas:
            p2_mean_neg_delta = float(np.mean(all_neg_deltas))
            p2_max_neg_delta = float(np.max(all_neg_deltas))
            p2_min_neg_delta = float(np.min(all_neg_deltas))
            p2_holds_layout_p2 = p2_mean_neg_delta <= 0.1
        else:
            p2_mean_neg_delta = float("nan")
            p2_max_neg_delta = float("nan")
            p2_min_neg_delta = float("nan")
            p2_holds_layout_p2 = False

        keep_rate = (total_kept_layout_p2 / total_attempts * 100
                     if total_attempts > 0 else float("nan"))

        if not math.isnan(p2_mean_neg_delta):
            print(f"    Total attempts: {total_attempts}  Kept: {total_kept_layout_p2}  "
                  f"Reverted: {total_reverted_layout_p2}  "
                  f"Keep-rate: {keep_rate:.1f}%")
            print(f"    Mean(prob-pre): {p2_mean_neg_delta:+.4f}  "
                  f"Max: {p2_max_neg_delta:+.4f}  Min: {p2_min_neg_delta:+.4f}  "
                  f"(P2 threshold <= +0.1)  "
                  f"{'PASS' if p2_holds_layout_p2 else 'FAIL'}")
        else:
            print(f"    No decided attempts — P2 N/A")
            p2_holds_layout_p2 = False  # no evidence = not pass

        print()

        # ---------------------------------------------------------------
        # Keep/replay-vote contingency table — pooled across seeds
        # ---------------------------------------------------------------
        print(f"  Keep vs Demoted-Replay-Vote Contingency (Layout {layout_idx + 1}):")
        both_keep = 0
        live_keep_replay_revert = 0
        live_revert_replay_keep = 0
        both_revert = 0

        for cont in cont_results:
            for rec in cont["spawn_records"]:
                kept = rec.get("kept")
                if kept is None:
                    continue
                replay_k = rec["replay_vote_keep"]
                if kept and replay_k:
                    both_keep += 1
                elif kept and not replay_k:
                    live_keep_replay_revert += 1
                elif not kept and replay_k:
                    live_revert_replay_keep += 1
                else:
                    both_revert += 1

        total_decided = both_keep + live_keep_replay_revert + live_revert_replay_keep + both_revert
        print(f"    (Live decision vs what Exp144 would have decided)")
        print(f"                        Replay=KEEP  Replay=REVERT")
        print(f"    Live=KEEP            {both_keep:>10}  {live_keep_replay_revert:>12}")
        print(f"    Live=REVERT          {live_revert_replay_keep:>10}  {both_revert:>12}")
        print(f"    Total decided: {total_decided}")
        if total_decided > 0:
            agree = both_keep + both_revert
            disagree = live_keep_replay_revert + live_revert_replay_keep
            print(f"    Agree: {agree}  Disagree: {disagree}  "
                  f"Agreement rate: {agree / total_decided * 100:.1f}%")
        else:
            print(f"    No decided spawns.")
        print()

        # Per-seed split detail
        print(f"  Per-split detail (Layout {layout_idx + 1}):")
        for seed_idx, cont in enumerate(cont_results):
            recs = cont["spawn_records"]
            if not recs:
                print(f"    seed={SEEDS[seed_idx]}: no split attempts")
                continue
            for i, rec in enumerate(recs):
                kept_str = "KEPT" if rec.get("kept") else "REVT"
                replay_str = "RV-K" if rec["replay_vote_keep"] else "RV-R"
                prob_mean = rec.get("probation_mean", float("nan"))
                delta = rec.get("delta", float("nan"))
                prob_str = f"{prob_mean:.4f}" if not math.isnan(prob_mean) else "  N/A"
                delta_str = f"{delta:+.4f}" if not math.isnan(delta) else "  N/A"
                sep_str = f"{rec.get('separation', float('nan')):.4f}"
                tr_str = f"{rec.get('parent_tr_cov', float('nan')):.4f}"
                print(f"    seed={SEEDS[seed_idx]} split#{i+1}: "
                      f"color={rec['color']}  "
                      f"parent_tr={tr_str}  sep={sep_str}  "
                      f"pre={rec['pre_spawn_mean']:.4f}  "
                      f"prob={prob_str}  delta={delta_str}  "
                      f"{kept_str}  {replay_str}  "
                      f"(nll_old={rec['nll_old']:.4f} nll_new={rec['nll_new']:.4f})")

        # ---------------------------------------------------------------
        # Tallies for this layout
        # ---------------------------------------------------------------
        print()
        print(f"  TALLIES — Layout {layout_idx + 1}")
        print()

        # --- P1: the alarm answered ---
        # Arm 1: drop >= 0.4 nats
        p1_drop_per_seed = [cont["drop"] >= 0.4 for cont in cont_results]
        p1_drop_count = sum(p1_drop_per_seed)
        p1_drop_holds = p1_drop_count >= 6

        # Arm 2: zero ceiling events in final FINAL_EVAL_WINDOW
        p1_quiet_per_seed = [cont["phase2_final_ceiling_events"] == 0 for cont in cont_results]
        p1_quiet_count = sum(p1_quiet_per_seed)
        p1_quiet_holds = p1_quiet_count >= 6

        # Arm 3: grown components >= 3 for >= 3/4 colors
        p1_comps_per_seed = []
        for cont in cont_results:
            comps = cont["final_comps_per_color"]
            colors_grown = sum(1 for c in comps if c >= 3)
            p1_comps_per_seed.append(colors_grown >= 3)
        p1_comps_count = sum(p1_comps_per_seed)
        p1_comps_holds = p1_comps_count >= 6

        p1_holds_layout = p1_drop_holds and p1_quiet_holds and p1_comps_holds

        print(f"  P1 alarm answered:")
        print(f"    Arm1 drop >= 0.4 nats: {p1_drop_count}/8  "
              f"(need >= 6/8)  {'PASS' if p1_drop_holds else 'FAIL'}")
        print(f"    Arm2 zero final ceil: {p1_quiet_count}/8  "
              f"(need >= 6/8)  {'PASS' if p1_quiet_holds else 'FAIL'}")
        print(f"    Arm3 comps>=3 for 3/4 colors: {p1_comps_count}/8  "
              f"(need >= 6/8)  {'PASS' if p1_comps_holds else 'FAIL'}")
        print(f"  P1 layout {layout_idx + 1}: {'PASS' if p1_holds_layout else 'FAIL'}")
        for s_idx, cont in enumerate(cont_results):
            comps = cont["final_comps_per_color"]
            colors_grown = sum(1 for c in comps if c >= 3)
            print(f"    seed={SEEDS[s_idx]}: plateau={cont['plateau']:.4f}  "
                  f"final={cont['final_surprise']:.4f}  drop={cont['drop']:.3f}  "
                  f"finCeil={cont['phase2_final_ceiling_events']}  "
                  f"comps={comps}({colors_grown}/4>=3)  "
                  f"drop={'pass' if p1_drop_per_seed[s_idx] else 'FAIL'}  "
                  f"quiet={'pass' if p1_quiet_per_seed[s_idx] else 'FAIL'}  "
                  f"comps={'pass' if p1_comps_per_seed[s_idx] else 'FAIL'}")

        # --- P2: anti-valley signature (already computed above) ---
        print(f"\n  P2 anti-valley (mean(prob-pre) <= +0.1, all attempts per layout):")
        if not math.isnan(p2_mean_neg_delta):
            print(f"    Mean(prob-pre): {p2_mean_neg_delta:+.4f}  "
                  f"{'PASS' if p2_holds_layout_p2 else 'FAIL'}")
        else:
            print(f"    No decided attempts — P2 N/A (treat as FAIL for conservatism)")
            p2_holds_layout_p2 = False

        # --- P3: no regression ---
        p3_per_seed = [cont["p2_final500_loc_median"] <= 0.5 for cont in cont_results]
        p3_count = sum(p3_per_seed)
        p3_holds_layout = p3_count >= 7

        print(f"\n  P3 no regression (final-500 loc_median <= 0.5):")
        print(f"    Seeds passing: {p3_count}/8  (need >= 7/8)  "
              f"{'PASS' if p3_holds_layout else 'FAIL'}")
        for s_idx, cont in enumerate(cont_results):
            print(f"    seed={SEEDS[s_idx]}: loc_med={cont['p2_final500_loc_median']:.4f}  "
                  f"{'pass' if p3_per_seed[s_idx] else 'FAIL'}")

        # --- Twin sanity ---
        twin_per_seed = [tab["final_500_map_frac"] >= 0.80 for tab in tab_results]
        twin_count = sum(twin_per_seed)
        twin_holds_layout = twin_count >= 7
        print(f"\n  Twin sanity (MAP >= 80% in final 500):")
        print(f"    Seeds passing: {twin_count}/8  (need >= 7/8)  "
              f"{'PASS' if twin_holds_layout else 'FAIL'}")

        layout_verdicts.append({
            "layout_idx": layout_idx + 1,
            "layout_seed": layout_seed,
            "p1_holds": p1_holds_layout,
            "p1_drop_count": p1_drop_count,
            "p1_quiet_count": p1_quiet_count,
            "p1_comps_count": p1_comps_count,
            "p2_holds": p2_holds_layout_p2,
            "p2_mean_neg_delta": p2_mean_neg_delta,
            "p2_max_neg_delta": p2_max_neg_delta,
            "p2_min_neg_delta": p2_min_neg_delta,
            "total_attempts_layout": total_attempts,
            "total_kept_layout": total_kept_layout_p2,
            "total_reverted_layout": total_reverted_layout_p2,
            "keep_rate": keep_rate,
            "p3_holds": p3_holds_layout,
            "p3_count": p3_count,
            "twin_holds": twin_holds_layout,
            "twin_count": twin_count,
            # For HALT evaluation
            "p1_drop_fails": 8 - p1_drop_count,  # seeds where drop < 0.4
        })

    # -----------------------------------------------------------------------
    # Global conjunct evaluation across all three layouts
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("GLOBAL CONJUNCT TALLIES (across all 3 layouts)")
    print("=" * 80)

    # P1: each arm in >= 6/8 seeds for ALL THREE layouts
    p1_all_layouts = all(v["p1_holds"] for v in layout_verdicts)
    print(f"\nP1 alarm answered (all 3 layouts):")
    for v in layout_verdicts:
        print(f"  Layout {v['layout_idx']} (rng({v['layout_seed']})): "
              f"drop={v['p1_drop_count']}/8  quiet={v['p1_quiet_count']}/8  "
              f"comps={v['p1_comps_count']}/8  "
              f"{'PASS' if v['p1_holds'] else 'FAIL'}")
    print(f"  P1 GLOBAL: {'PASS' if p1_all_layouts else 'FAIL'}")

    # P2: anti-valley signature per layout
    p2_all_layouts = all(v["p2_holds"] for v in layout_verdicts)
    print(f"\nP2 anti-valley signature (mean(prob-pre) <= +0.1, all 3 layouts):")
    for v in layout_verdicts:
        if not math.isnan(v["p2_mean_neg_delta"]):
            nd_str = f"{v['p2_mean_neg_delta']:+.4f}"
        else:
            nd_str = "  N/A"
        print(f"  Layout {v['layout_idx']}: attempts={v['total_attempts_layout']}  "
              f"kept={v['total_kept_layout']}  keep_rate={v['keep_rate']:.1f}%  "
              f"mean(prob-pre)={nd_str}  "
              f"{'PASS' if v['p2_holds'] else 'FAIL'}")
    print(f"  P2 GLOBAL: {'PASS' if p2_all_layouts else 'FAIL (P1 carries if P2 alone fails)'}")

    # P3: >= 7/8 seeds all three layouts
    p3_all_layouts = all(v["p3_holds"] for v in layout_verdicts)
    print(f"\nP3 no regression (>= 7/8 seeds, all 3 layouts):")
    for v in layout_verdicts:
        print(f"  Layout {v['layout_idx']}: {v['p3_count']}/8  "
              f"{'PASS' if v['p3_holds'] else 'FAIL'}")
    print(f"  P3 GLOBAL: {'PASS' if p3_all_layouts else 'FAIL (HALT — regression)'}")

    # Twin sanity
    twin_all = all(v["twin_holds"] for v in layout_verdicts)
    print(f"\nTwin sanity (>= 7/8, all 3 layouts):")
    for v in layout_verdicts:
        print(f"  Layout {v['layout_idx']}: {v['twin_count']}/8  "
              f"{'PASS' if v['twin_holds'] else 'FAIL'}")

    # -----------------------------------------------------------------------
    # Halt-arm evaluation
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("HALT-ARM EVALUATION")
    print("=" * 80)

    # HALT arm 1 (P1 surprise): drop < 0.4 nats in >= 3/8 seeds in >= 2/3 layouts
    halt_p1_layouts = sum(1 for v in layout_verdicts if v["p1_drop_fails"] >= 3)
    halt_migration_p1 = halt_p1_layouts >= 2

    # HALT arm 2: P3 failure
    halt_regression = not p3_all_layouts

    halt_triggers = []
    if halt_migration_p1:
        halt_triggers.append(
            f"P1 SURPRISE HALT: drop < 0.4 nats in >= 3/8 seeds for "
            f"{halt_p1_layouts}/3 layouts (>= 2) — "
            f"THIRD growth design fails (ADD and SPLIT both exhausted); "
            f"MIGRATION HALT — park at M3d and continue at M4 "
            f"(three mechanistically distinct failures = a documented wall)"
        )
    if halt_regression:
        halt_triggers.append(
            f"P3 REGRESSION HALT: final-500 loc_median > 0.5 in >= 2/8 seeds "
            f"in at least one layout — growth broke localization (HALT)"
        )

    print()
    if halt_triggers:
        for ht in halt_triggers:
            print(f"  Falsifier: {ht}")
    else:
        print("  No HALT arms triggered.")

    if not p2_all_layouts and not halt_migration_p1 and not halt_regression:
        print(f"  NOTE P2 FAIL: anti-valley not confirmed — the move works for the "
              f"wrong reason — MIXED, logged.")

    # -----------------------------------------------------------------------
    # Three-way VERDICT
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)

    if halt_triggers:
        verdict = "NEGATIVE"
        print(f"VERDICT: {verdict}")
        print("MIGRATION HALT")
        for ht in halt_triggers:
            print(f"  {ht}")
        if halt_migration_p1:
            print("  Park recommendation: park M3d, continue migration at M4 "
                  "(three mechanistically distinct growth designs exhausted; "
                  "the alarm ceiling may require a fundamentally different approach "
                  "at the next rung rather than further growth operator variants).")
    elif p1_all_layouts and p2_all_layouts and p3_all_layouts:
        verdict = "POSITIVE"
        print(f"VERDICT: {verdict}")
        print(
            "All predeclared properties satisfied (P1 + P2 + P3 across all 3 layouts). "
            "SPLIT is coverage-preserving: the anti-valley signature holds, "
            "probation keeps beneficial splits, surprise descends below the ceiling "
            "threshold, and no localization regression. "
            "Migration thread may advance to rung M4."
        )
    elif p1_all_layouts and p3_all_layouts and not p2_all_layouts:
        verdict = "MIXED"
        print(f"VERDICT: {verdict}")
        print(
            "P1 (alarm answered) and P3 (no regression) pass across all 3 layouts. "
            "P2 (anti-valley) fails — the SPLIT move works but may not preserve "
            "predictive density as cleanly as hypothesized. MIXED, logged. "
            "No HALT (P1 carries)."
        )
    elif not p1_all_layouts and not halt_triggers:
        p1_pass_layouts = sum(1 for v in layout_verdicts if v["p1_holds"])
        verdict = "MIXED"
        print(f"VERDICT: {verdict}")
        print(
            f"P1 holds in {p1_pass_layouts}/3 layouts (need all 3). "
            f"P2={'PASS' if p2_all_layouts else 'FAIL'}. "
            f"P3={'PASS' if p3_all_layouts else 'FAIL'}. "
            f"Partial progress — HALT not triggered. Inspect split in failing layout."
        )
    else:
        verdict = "MIXED"
        print(f"VERDICT: {verdict}")
        print(
            f"Partial: P1={'PASS' if p1_all_layouts else 'FAIL'}, "
            f"P2={'PASS' if p2_all_layouts else 'FAIL'}, "
            f"P3={'PASS' if p3_all_layouts else 'FAIL'}. "
            f"Inspect tallies above."
        )

    print("=" * 80)

    # -----------------------------------------------------------------------
    # JSON output
    # -----------------------------------------------------------------------
    for v in layout_verdicts:
        all_rows_json.append({
            "exp": 146,
            "rung": "M3d",
            "layout_seed": v["layout_seed"],
            "layout_idx": v["layout_idx"],
            "seed": -1,
            "summary": True,
            "p1_holds": v["p1_holds"],
            "p1_drop_count": v["p1_drop_count"],
            "p1_quiet_count": v["p1_quiet_count"],
            "p1_comps_count": v["p1_comps_count"],
            "p2_holds": v["p2_holds"],
            "p2_mean_neg_delta": (
                v["p2_mean_neg_delta"]
                if not math.isnan(v["p2_mean_neg_delta"]) else None
            ),
            "p2_max_neg_delta": (
                v["p2_max_neg_delta"]
                if not math.isnan(v["p2_max_neg_delta"]) else None
            ),
            "p2_min_neg_delta": (
                v["p2_min_neg_delta"]
                if not math.isnan(v["p2_min_neg_delta"]) else None
            ),
            "total_attempts_layout": v["total_attempts_layout"],
            "total_kept_layout": v["total_kept_layout"],
            "keep_rate": v["keep_rate"] if not math.isnan(v["keep_rate"]) else None,
            "p3_holds": v["p3_holds"],
            "p3_count": v["p3_count"],
            "twin_holds": v["twin_holds"],
            "twin_count": v["twin_count"],
        })
    all_rows_json.append({
        "exp": 146,
        "rung": "M3d",
        "seed": -2,
        "global_summary": True,
        "p1_global": p1_all_layouts,
        "p2_global": p2_all_layouts,
        "p3_global": p3_all_layouts,
        "halt_migration_p1": halt_migration_p1,
        "halt_regression": halt_regression,
        "verdict": verdict,
    })

    out_path = Path(__file__).parent / "outputs" / "exp146_rows.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for row in all_rows_json:
            fh.write(json.dumps(row) + "\n")
    print(f"\nJSON rows written to {out_path}")


if __name__ == "__main__":
    main()
