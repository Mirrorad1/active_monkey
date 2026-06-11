"""
Exp 152 — continuous-creature rung M3e: BATCH-JUMP (the first of the two
human-authorized growth cracks, "address the cracks we want to target growth",
2026-06-11; if this design's surprise arm fails, the loop proceeds DIRECTLY to
background-floor per the pre-authorized sequence — no fresh consult).

Card loop/directions/continuous-creature.md (re-opened). The wall (Exp 144-146):
under log-loss an INCOMPLETE tight cover loses to a complete loose cover, so
LOCAL moves (add one, split one) are anti-productive under any honest
short-horizon test. Batch-jump attacks the wall's own premise: fit the COMPLETE
per-color mixture in one move — EM on that color's replay pairs with K chosen
from {2,3,4} by penalized replay NLL (penalty 0.05*K, declared constant) — and
probate the complete cover as ONE move under the Exp 145-validated live test.
Mechanism prediction: completeness is what the honest test was waiting for —
accepted jumps IMPROVE live surprise (negative probation deltas), most jumps are
accepted, and the alarm finally quiets.

Setup: identical to Exp 145 except the move — three aliased layouts (rng 7/11/
13), seeds 0..7, phase 1 T=2000 unimodal (detector armed), phase 2 T=6000:
every 200 steps, if no probation is running and a color is alarmed (per-color
mean of last 50 observations >= 0.7) with budget (one jump per color per 1200
steps; re-jumps allowed), pick the least-recently-attempted alarmed color
(round-robin); JUMP: fit EM mixtures with K=2,3,4 on that color's replay pairs
(stored posterior moments as per-point uncertainty, 10 EM iterations each from
k-means++ style seeding with the run's rng), pick K minimizing (replay NLL +
0.05*K); REPLACE the color's entire mixture with the fitted one (snapshot
first; NIW write-back as Exp 144's convention; weights from EM); probation 400
live steps: KEEP iff the color's probation mean <= pre-jump mean (last 100
observations) - 0.1, else restore the snapshot. Eval = final 1000 steps.
Tabular twin sanity as before.

Predictions (TRUE iff all):
- P1 the alarm answered (the unchanged bar, fourth design): final-1000 mean
  predictive surprise <= phase-1 plateau - 0.4 nats AND zero ceiling events in
  the final 1000 AND components >= 3 for >= 3/4 colors — each in >= 6/8 seeds
  per layout, all three layouts.
- P2 the completeness mechanism: mean probation delta among ACCEPTED jumps
  <= -0.2 nats per layout, AND >= 50% of attempted jumps accepted (pooled per
  layout).
- P3 no regression: final-500 median localization <= 0.5 in >= 7/8 seeds per
  layout.

Falsifiers: P1's surprise arm fails in >= 3/8 seeds in >= 2/3 layouts — the
batch-jump crack fails; per the pre-authorized sequence the loop PROCEEDS to
background-floor (Exp 153) and logs this as NEGATIVE (no halt-for-word). P3
fails = HALT (regression). P2 failing while P1 carries = MIXED (works for a
different reason — log it). Three-way rule per PROTOCOL step 3; the blinded
verifier (step 4.5) checks the verdict before logging.
"""
from __future__ import annotations

# NOTE: Mixture machinery copied from exp145 (which copied from exp144/exp143).
# ONLY behavioral changes from exp145:
#   1. Phase 2 T=6000 (from 8000)
#   2. BATCH-JUMP replaces local spawn: EM fit of full K-component mixture
#      on the color's replay pairs, K in {2,3,4} chosen by penalized replay NLL
#   3. Budget: time-based cooldown (1200 steps per color) instead of component count cap
#   4. REPLACE semantics: entire color mixture replaced (not append)
#   5. K-selection: farthest-point init + 10 EM iterations each K, score = replay_NLL + 0.05*K
#   6. P2 metric: mean accepted probation delta + accepted fraction
#   7. Attempt tracking: color, K chosen, NLL by K, pre-jump mean, probation mean, delta, kept

import copy
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from active_loop.continuous import NIW
from active_loop.creature_continuous import ContinuousPlace

# ---------------------------------------------------------------------------
# Detector constants — copied from exp145
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
T_PHASE2 = 6000           # M3e: 6000 (from exp145's 8000)
T_TOTAL = T_PHASE1 + T_PHASE2

# M3e new constants
JUMP_INTERVAL = 200       # check every N steps during phase 2 if no probation running
JUMP_COOLDOWN = 1200      # min steps between jumps per color (time-based budget)
REPLAY_WINDOW = 400       # replay pairs for EM fitting and NLL scoring
FINAL_EVAL_WINDOW = 1000  # final steps of phase 2 for grading
P1_PLATEAU_WINDOW = 500   # last N steps of phase 1 for plateau

EM_ITERS = 10             # EM iterations per K candidate
EM_COV_FLOOR = 1e-4       # floor for Cov_j diagonal entries
K_CANDIDATES = [2, 3, 4]  # K options for batch-jump
K_PENALTY = 0.05          # penalty per component (declared constant)
MIN_REPLAY_PAIRS = 30     # minimum color replay pairs required for jump

COLOR_SURPRISE_WINDOW = 50    # per-color deque size for alarm check
ALARM_THRESH = 0.7            # mean per-color surprise threshold for alarm
PRE_JUMP_WINDOW = 100         # last N color observations for pre-jump mean
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
# Replay NLL scorer for a color subset — used for K-selection scoring
# ---------------------------------------------------------------------------

def color_replay_nll(
    color_replay_pairs: list[tuple[np.ndarray, np.ndarray]],
    comp_means: np.ndarray,   # (K, 2)
    comp_covs: np.ndarray,    # (K, 2) diagonal entries
    weights: np.ndarray,      # (K,)
) -> float:
    """Mean -log sum_j w_j N(mean_i; m_j, Cov_j + Sigma_i) over color replay pairs.

    Used for K-selection: score = color_replay_nll + K_PENALTY * K.
    """
    n = len(color_replay_pairs)
    if n == 0:
        return float("inf")
    K = len(weights)
    total = 0.0
    for (mu_i, Sigma_i_diag) in color_replay_pairs:
        log_terms = np.empty(K, dtype=float)
        for j in range(K):
            eff_cov = np.maximum(comp_covs[j] + Sigma_i_diag, 1e-9)
            d0 = mu_i[0] - comp_means[j, 0]
            d1 = mu_i[1] - comp_means[j, 1]
            log_det = math.log(eff_cov[0]) + math.log(eff_cov[1])
            maha = d0 * d0 / eff_cov[0] + d1 * d1 / eff_cov[1]
            log_lik = -0.5 * (2.0 * math.log(2.0 * math.pi) + log_det + maha)
            log_terms[j] = math.log(max(weights[j], 1e-300)) + log_lik
        lse = float(np.logaddexp.reduce(log_terms))
        total += -lse
    return total / n


# ---------------------------------------------------------------------------
# Full replay NLL for global mixture (for diagnostic printing) — copied from exp145
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
# Farthest-point seeding (k-means++ style) for EM initialization
# ---------------------------------------------------------------------------

def _farthest_point_init(
    obs_means: np.ndarray,  # (n, 2)
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Initialize K cluster means by farthest-point (greedy) selection.

    First center: uniformly random from obs_means.
    Each subsequent center: the point that maximizes min-distance to existing centers.
    Returns array of shape (K, 2).
    """
    n = len(obs_means)
    first_idx = int(rng.integers(0, n))
    centers = [obs_means[first_idx].copy()]
    for _ in range(1, K):
        # Compute min squared distance to nearest existing center for each point
        min_sq_dists = np.full(n, np.inf)
        for c in centers:
            diffs = obs_means - c  # (n, 2)
            sq_dists = (diffs ** 2).sum(axis=1)  # (n,)
            min_sq_dists = np.minimum(min_sq_dists, sq_dists)
        # Pick the point with the maximum min-distance
        next_idx = int(np.argmax(min_sq_dists))
        centers.append(obs_means[next_idx].copy())
    return np.array(centers)  # (K, 2)


# ---------------------------------------------------------------------------
# Batch-jump EM: fit K-component mixture on color's replay pairs
# ---------------------------------------------------------------------------

def batch_jump_em(
    color_replay_pairs: list[tuple[np.ndarray, np.ndarray]],
    K: int,
    rng: np.random.Generator,
    n_iters: int = EM_ITERS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a K-component diagonal-Gaussian mixture on color replay pairs via EM.

    Each pair (mean_i, Sigma_diag_i) represents per-point uncertainty:
    true location modeled as N(mean_i, diag(Sigma_diag_i)).

    E-step responsibilities:
        r_ij propto w_j * N(mean_i; m_j, Cov_j + diag(Sigma_i))
    M-step:
        w_j = sum_i r_ij / n
        m_j = weighted mean
        Cov_j = weighted scatter + weighted mean of Sigma_i, floored at EM_COV_FLOOR * I

    Init: farthest-point seeding on obs_means, uniform weights, Cov = 0.5*I.

    Returns (weights, comp_means, comp_covs) each shape (K,), (K,2), (K,2).
    """
    n = len(color_replay_pairs)
    obs_means = np.array([mu for (mu, _) in color_replay_pairs])    # (n, 2)
    obs_sigmas = np.array([sig for (_, sig) in color_replay_pairs])  # (n, 2)

    # Initialize
    centers = _farthest_point_init(obs_means, K, rng)  # (K, 2)
    weights = np.full(K, 1.0 / K)
    comp_means = centers.copy()
    comp_covs = np.full((K, 2), 0.5)  # initial diagonal covariance 0.5 per axis

    for _ in range(n_iters):
        # E-step
        log_r = np.empty((n, K), dtype=float)
        for j in range(K):
            mj = comp_means[j]
            cj = comp_covs[j]
            for i in range(n):
                eff_cov = np.maximum(cj + obs_sigmas[i], 1e-9)
                d0 = obs_means[i, 0] - mj[0]
                d1 = obs_means[i, 1] - mj[1]
                log_det = math.log(eff_cov[0]) + math.log(eff_cov[1])
                maha = d0 * d0 / eff_cov[0] + d1 * d1 / eff_cov[1]
                log_lik = -0.5 * (2.0 * math.log(2.0 * math.pi) + log_det + maha)
                log_r[i, j] = math.log(max(weights[j], 1e-300)) + log_lik

        # Normalize responsibilities row-wise
        log_r_max = log_r.max(axis=1, keepdims=True)
        r = np.exp(log_r - log_r_max)
        r_sum = r.sum(axis=1, keepdims=True)
        r_sum = np.where(r_sum == 0, 1.0, r_sum)
        r = r / r_sum  # (n, K)

        # M-step
        n_j = r.sum(axis=0)  # (K,)
        n_j_safe = np.maximum(n_j, 1e-9)

        new_weights = n_j / n
        new_weights = np.maximum(new_weights, 1e-300)
        new_weights /= new_weights.sum()

        new_means = (r.T @ obs_means) / n_j_safe[:, None]  # (K, 2)

        new_covs = np.zeros((K, 2), dtype=float)
        for j in range(K):
            diff = obs_means - new_means[j]  # (n, 2)
            scatter = (r[:, j, None] * (diff ** 2 + obs_sigmas)).sum(axis=0)
            new_covs[j] = scatter / n_j_safe[j]
        new_covs = np.maximum(new_covs, EM_COV_FLOOR)

        weights = new_weights
        comp_means = new_means
        comp_covs = new_covs

    return weights, comp_means, comp_covs


# ---------------------------------------------------------------------------
# Build NIW component list from EM result (REPLACE semantics)
# NIW write-back convention: kappa_j = max(n_j, 0.5), nu_j = kappa_j + 3
# m_j = comp_means[j], S_j = diag(comp_covs[j]) * kappa_j
# ---------------------------------------------------------------------------

def em_result_to_components(
    weights: np.ndarray,       # (K,)
    comp_means: np.ndarray,    # (K, 2)
    comp_covs: np.ndarray,     # (K, 2) diagonal entries
    n_eff: np.ndarray,         # (K,) effective counts (r.sum per component, from last EM iter)
) -> list[tuple[float, NIW]]:
    """Convert EM result to NIW component list using declared write-back convention.

    n_eff[j] = effective count for component j (from final EM responsibilities).
    """
    K = len(weights)
    result = []
    for j in range(K):
        kappa_j = float(max(n_eff[j], 0.5))
        nu_j = kappa_j + 3.0
        m_j = comp_means[j].copy()
        S_j = np.diag(comp_covs[j] * kappa_j)
        if nu_j < D + 2:
            nu_j = float(D + 2)
            kappa_j = nu_j - 3.0
        new_niw = NIW(m=m_j, kappa=kappa_j, nu=nu_j, S=S_j)
        result.append((float(weights[j]), new_niw))
    return result


# ---------------------------------------------------------------------------
# Select best K via penalized replay NLL
# ---------------------------------------------------------------------------

def select_best_k(
    color_replay_pairs: list[tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    print_debug: bool = False,
    attempt_num: int = 0,
    color_idx: int = 0,
) -> tuple[int, list[tuple[float, NIW]], np.ndarray, dict]:
    """Fit EM for K in {2,3,4}, score = replay_NLL + 0.05*K, pick argmin.

    Returns (best_K, best_components, best_n_eff, nll_by_k_dict).
    nll_by_k_dict: {K: (nll, penalized_score)} for diagnostics.
    """
    best_K = -1
    best_score = float("inf")
    best_components = None
    best_n_eff = None
    nll_by_k: dict[int, tuple[float, float]] = {}

    for K in K_CANDIDATES:
        weights, comp_means, comp_covs = batch_jump_em(color_replay_pairs, K, rng)
        nll = color_replay_nll(color_replay_pairs, comp_means, comp_covs, weights)
        score = nll + K_PENALTY * K
        nll_by_k[K] = (nll, score)

        # Recompute n_eff for NIW write-back
        n = len(color_replay_pairs)
        obs_means = np.array([mu for (mu, _) in color_replay_pairs])
        obs_sigmas = np.array([sig for (_, sig) in color_replay_pairs])
        log_r = np.empty((n, K), dtype=float)
        for j in range(K):
            mj = comp_means[j]
            cj = comp_covs[j]
            for i in range(n):
                eff_cov = np.maximum(cj + obs_sigmas[i], 1e-9)
                d0 = obs_means[i, 0] - mj[0]
                d1 = obs_means[i, 1] - mj[1]
                log_det = math.log(eff_cov[0]) + math.log(eff_cov[1])
                maha = d0 * d0 / eff_cov[0] + d1 * d1 / eff_cov[1]
                log_lik = -0.5 * (2.0 * math.log(2.0 * math.pi) + log_det + maha)
                log_r[i, j] = math.log(max(weights[j], 1e-300)) + log_lik
        log_r_max = log_r.max(axis=1, keepdims=True)
        r = np.exp(log_r - log_r_max)
        r_sum = r.sum(axis=1, keepdims=True)
        r_sum = np.where(r_sum == 0, 1.0, r_sum)
        r = r / r_sum
        n_eff = r.sum(axis=0)

        if score < best_score:
            best_score = score
            best_K = K
            best_components = em_result_to_components(weights, comp_means, comp_covs, n_eff)
            best_n_eff = n_eff.copy()

    if print_debug:
        print(f"      [K-select] attempt#{attempt_num} color={color_idx} "
              f"pairs={len(color_replay_pairs)}")
        for K, (nll, score) in sorted(nll_by_k.items()):
            marker = " <-- CHOSEN" if K == best_K else ""
            print(f"        K={K}: nll={nll:.4f}  score={score:.4f}{marker}")

    return best_K, best_components, best_n_eff, nll_by_k


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
# Continuous agent with mixture + detector + M3e batch-jump growth
# ---------------------------------------------------------------------------

def run_continuous(
    actions: np.ndarray,
    cmap: np.ndarray,
    run_rng: np.random.Generator,
    seed: int = 0,
    layout_idx: int = 1,
    print_k_debug: bool = False,
) -> dict:
    """Run the continuous agent with M3e batch-jump EM REPLACE move.

    M3e changes from M3c (exp145):
      - Phase 2 T=6000 (from 8000)
      - BATCH-JUMP: fit complete K-component mixture (K in {2,3,4}) on color's
        replay pairs via EM; replace entire color mixture with fitted one
      - Budget: time-based cooldown (1200 steps between jumps per color, re-jumps
        allowed) instead of component count cap
      - K-selection: penalized replay NLL (score = nll + 0.05*K), pick argmin
      - Probation: 400 live steps, KEEP iff probation mean <= pre-jump mean - 0.1
      - Snapshot/restore is per-color (other colors keep probation-period learning)
      - P2 metric: mean accepted delta + accepted fraction (replaced P2 of exp145)

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

    # Global surprise buffer (rolling deque) — for global detector / quiet check
    surprise_buf: deque = deque(maxlen=SURPRISE_WINDOW)

    # Replay buffer: (obs_color, mu_at_predict, Sigma_diag_at_predict)
    replay_buf: deque = deque(maxlen=REPLAY_WINDOW)

    # Per-color surprise deque(50) for alarm checking
    color_surprise_bufs: list[deque] = [deque(maxlen=COLOR_SURPRISE_WINDOW) for _ in range(N_COLORS)]

    # Per-color larger deque(PRE_JUMP_WINDOW) for pre-jump mean
    color_pre_jump_bufs: list[deque] = [deque(maxlen=PRE_JUMP_WINDOW) for _ in range(N_COLORS)]

    # M3e: Round-robin scheduling — last attempt step per color (init to -inf equiv)
    last_attempt_step: list[int] = [-JUMP_COOLDOWN - 1] * N_COLORS

    # M3e: Probation state
    probation_color: int = -1
    probation_start_phase2_t: int = -1
    probation_pre_jump_mean: float = float("inf")
    probation_color_snap: list[tuple[float, NIW]] = []
    probation_color_counts_snap: list[int] = []
    probation_observations: list[float] = []

    # M3e: per-attempt record
    # Each entry: color, K_chosen, nll_by_k, pre_jump_mean, probation_mean,
    #             delta, kept (bool)
    attempt_records: list[dict] = []
    attempt_counter: list[int] = [0]  # mutable for closure

    # --- Phase 1 tracking ---
    phase1_loc_errors = np.empty(T_PHASE1)
    phase1_ceiling_events = 0
    phase1_surprise_vals = []

    # --- Phase 2 tracking ---
    phase2_loc_errors = np.empty(T_PHASE2)
    phase2_ceiling_events_total = 0
    phase2_final_ceiling_events = 0
    jumps_kept = 0
    jumps_reverted = 0
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
            color_pre_jump_bufs[obs_k].append(surprise_t)

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

                    if probation_observations:
                        prob_mean = float(np.mean(probation_observations))
                    else:
                        prob_mean = float("inf")

                    keep = (prob_mean <= probation_pre_jump_mean - 0.1)

                    if attempt_records:
                        attempt_records[-1]["probation_mean"] = prob_mean
                        attempt_records[-1]["delta"] = probation_pre_jump_mean - prob_mean
                        attempt_records[-1]["kept"] = keep

                    if keep:
                        jumps_kept += 1
                        # Mixture already installed; last_attempt_step already recorded
                    else:
                        jumps_reverted += 1
                        # Restore ONLY that color's snapshot
                        components[pc] = probation_color_snap
                        counts[pc] = probation_color_counts_snap

                    # Clear probation state
                    probation_color = -1
                    probation_start_phase2_t = -1
                    probation_pre_jump_mean = float("inf")
                    probation_color_snap = []
                    probation_color_counts_snap = []
                    probation_observations = []

            # --- Jump attempt every JUMP_INTERVAL steps, only if no probation running ---
            if (probation_color < 0
                    and phase2_t > 0
                    and phase2_t % JUMP_INTERVAL == 0):

                # Compute per-color alarm and budget:
                # alarmed = mean >= ALARM_THRESH
                # eligible = alarmed AND (last_attempt_step[k] + JUMP_COOLDOWN <= phase2_t)
                eligible_colors = []
                for k in range(N_COLORS):
                    if len(color_surprise_bufs[k]) > 0:
                        mean_k = float(np.mean(color_surprise_bufs[k]))
                        if mean_k >= ALARM_THRESH:
                            cooldown_elapsed = (phase2_t - last_attempt_step[k]) >= JUMP_COOLDOWN
                            if cooldown_elapsed:
                                eligible_colors.append(k)

                if eligible_colors:
                    # Round-robin: pick the eligible color with min last_attempt_step
                    jump_color = min(eligible_colors, key=lambda k: (last_attempt_step[k], k))
                    last_attempt_step[jump_color] = phase2_t

                    # Gather this color's replay pairs
                    color_replay_pairs = [
                        (mu_s.copy(), sig_s.copy())
                        for (obs_c, mu_s, sig_s) in replay_buf
                        if obs_c == jump_color
                    ]

                    if len(color_replay_pairs) < MIN_REPLAY_PAIRS:
                        # Not enough data — skip this attempt (no record added)
                        pass
                    else:
                        attempt_num = attempt_counter[0]
                        attempt_counter[0] += 1

                        # Pre-jump mean: mean of color's last PRE_JUMP_WINDOW observations
                        pre_jump_buf = list(color_pre_jump_bufs[jump_color])
                        if pre_jump_buf:
                            pre_jump_mean = float(np.mean(pre_jump_buf))
                        else:
                            pre_jump_mean = float("inf")

                        # Snapshot this color's mixture BEFORE replacing
                        snap_comps = _copy_color_components(components[jump_color])
                        snap_counts = list(counts[jump_color])

                        # K-selection: fit EM for K in {2,3,4}, pick best by penalized NLL
                        do_print = print_k_debug and (seed == 0) and (layout_idx == 1)
                        best_K, best_comps, best_n_eff, nll_by_k = select_best_k(
                            color_replay_pairs,
                            run_rng,
                            print_debug=do_print,
                            attempt_num=attempt_num,
                            color_idx=jump_color,
                        )

                        # REPLACE: install complete fitted mixture
                        components[jump_color] = best_comps

                        # Sync counts: set to match EM effective counts
                        # Use n_eff rescaled so sum ~ len(color_replay_pairs)
                        n_total = len(color_replay_pairs)
                        new_counts = [max(1, int(round(float(best_n_eff[j]) * n_total / sum(best_n_eff))))
                                      for j in range(best_K)]
                        counts[jump_color] = new_counts

                        # INSTALL PROVISIONALLY — start probation
                        probation_color = jump_color
                        probation_start_phase2_t = phase2_t
                        probation_pre_jump_mean = pre_jump_mean
                        probation_color_snap = snap_comps
                        probation_color_counts_snap = snap_counts
                        probation_observations = []

                        # Record attempt (probation_mean/delta/kept filled on resolution)
                        nll_by_k_serializable = {
                            str(K): {"nll": nll_by_k[K][0], "score": nll_by_k[K][1]}
                            for K in K_CANDIDATES
                        }
                        attempt_records.append({
                            "color": jump_color,
                            "K_chosen": best_K,
                            "nll_by_k": nll_by_k_serializable,
                            "pre_jump_mean": pre_jump_mean,
                            "probation_mean": float("nan"),
                            "delta": float("nan"),
                            "kept": None,
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
        jumps_reverted += 1
        if attempt_records:
            attempt_records[-1]["kept"] = False
            attempt_records[-1]["probation_mean"] = (
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

    # P2 completeness mechanism:
    # (a) mean probation delta among ACCEPTED jumps
    # (b) fraction of attempted jumps accepted
    decided_records = [r for r in attempt_records if r.get("kept") is not None]
    kept_records = [r for r in decided_records if r.get("kept") is True]

    accepted_deltas = [r["delta"] for r in kept_records if not math.isnan(r.get("delta", float("nan")))]
    mean_accepted_delta = float(np.mean(accepted_deltas)) if accepted_deltas else float("nan")

    total_attempted = len(decided_records)
    total_accepted = len(kept_records)
    accepted_frac = total_accepted / total_attempted if total_attempted > 0 else float("nan")

    return {
        "plateau": phase1_plateau,
        "final_surprise": final_surprise,
        "drop": drop,
        "p1_final500_loc_median": p1_final500_loc_median,
        "p1_ceiling_events": phase1_ceiling_events,
        "phase2_final_ceiling_events": phase2_final_ceiling_events,
        "final_comps_per_color": final_comps_per_color,
        "jumps_kept": jumps_kept,
        "jumps_reverted": jumps_reverted,
        "p2_final500_loc_median": p2_final500_loc_median,
        "mean_accepted_delta": mean_accepted_delta,
        "accepted_frac": accepted_frac,
        "total_attempted": total_attempted,
        "total_accepted": total_accepted,
        "attempt_records": attempt_records,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("Exp 152 — continuous-creature rung M3e: BATCH-JUMP")
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
              f"{'AccFrac':>7}  "
              f"{'MnDelta':>8}  "
              f"{'TwinF500':>8}")
        print("-" * 110)

        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed)
            actions = rng.integers(0, 4, size=T_TOTAL)
            run_rng = np.random.default_rng(2000 + seed)

            tab = run_tabular(actions, cmap)
            # Print K-selection debug for seed=0 layout=1 only
            cont = run_continuous(
                actions, cmap, run_rng,
                seed=seed, layout_idx=layout_idx + 1,
                print_k_debug=(seed == 0 and layout_idx == 0),
            )

            tab_results.append(tab)
            cont_results.append(cont)

            comps_str = str(cont["final_comps_per_color"])
            acc_frac = cont["accepted_frac"]
            acc_str = f"{acc_frac * 100:.0f}%" if not math.isnan(acc_frac) else " N/A"
            mn_delta = cont["mean_accepted_delta"]
            mn_str = f"{mn_delta:+.4f}" if not math.isnan(mn_delta) else "    N/A"
            print(f"{seed:>4}  "
                  f"{cont['plateau']:>7.4f}  "
                  f"{cont['final_surprise']:>7.4f}  "
                  f"{cont['drop']:>6.3f}  "
                  f"{cont['phase2_final_ceiling_events']:>7d}  "
                  f"{comps_str:>12}  "
                  f"{cont['jumps_kept']:>4d}  "
                  f"{cont['jumps_reverted']:>4d}  "
                  f"{cont['p2_final500_loc_median']:>6.4f}  "
                  f"{acc_str:>7}  "
                  f"{mn_str:>8}  "
                  f"{tab['final_500_map_frac']:>8.4f}")

            all_rows_json.append({
                "exp": 152,
                "rung": "M3e",
                "layout_seed": layout_seed,
                "layout_idx": layout_idx + 1,
                "seed": seed,
                "plateau": cont["plateau"],
                "final_surprise": cont["final_surprise"],
                "drop": cont["drop"],
                "p1_ceiling_events": cont["p1_ceiling_events"],
                "phase2_final_ceiling_events": cont["phase2_final_ceiling_events"],
                "final_comps_per_color": cont["final_comps_per_color"],
                "jumps_kept": cont["jumps_kept"],
                "jumps_reverted": cont["jumps_reverted"],
                "p1_final500_loc_median": cont["p1_final500_loc_median"],
                "p2_final500_loc_median": cont["p2_final500_loc_median"],
                "accepted_frac": (
                    cont["accepted_frac"] if not math.isnan(cont["accepted_frac"]) else None
                ),
                "mean_accepted_delta": (
                    cont["mean_accepted_delta"]
                    if not math.isnan(cont["mean_accepted_delta"]) else None
                ),
                "total_attempted": cont["total_attempted"],
                "total_accepted": cont["total_accepted"],
                "twin_final500_map_frac": tab["final_500_map_frac"],
            })

        # ---------------------------------------------------------------
        # Per-attempt detail summary
        # ---------------------------------------------------------------
        print()
        print(f"  Per-jump attempt detail (Layout {layout_idx + 1}):")
        for seed_idx, cont in enumerate(cont_results):
            recs = cont["attempt_records"]
            if not recs:
                print(f"    seed={SEEDS[seed_idx]}: no jump attempts")
                continue
            for i, rec in enumerate(recs):
                kept_str = "KEPT" if rec.get("kept") else "REVT"
                prob_mean = rec.get("probation_mean", float("nan"))
                delta = rec.get("delta", float("nan"))
                prob_str = f"{prob_mean:.4f}" if not math.isnan(prob_mean) else "  N/A"
                delta_str = f"{delta:+.4f}" if not math.isnan(delta) else "  N/A"
                # summarize nll_by_k
                nll_str_parts = []
                for K in K_CANDIDATES:
                    kstr = str(K)
                    if kstr in rec.get("nll_by_k", {}):
                        nll_str_parts.append(f"K{K}:{rec['nll_by_k'][kstr]['score']:.3f}")
                nll_summary = " ".join(nll_str_parts)
                print(f"    seed={SEEDS[seed_idx]} jump#{i+1}: "
                      f"color={rec['color']}  K={rec['K_chosen']}  "
                      f"pre={rec['pre_jump_mean']:.4f}  "
                      f"prob={prob_str}  delta={delta_str}  "
                      f"{kept_str}  [{nll_summary}]")

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

        # --- P2: completeness mechanism ---
        # (a) mean probation delta among accepted jumps <= -0.2 nats
        # (b) accepted fraction >= 50%
        # Both computed pooled per layout

        # Collect all decided records across seeds for this layout
        all_decided = []
        all_kept = []
        for cont in cont_results:
            for rec in cont["attempt_records"]:
                if rec.get("kept") is not None:
                    all_decided.append(rec)
                    if rec.get("kept"):
                        all_kept.append(rec)

        accepted_deltas_layout = [
            r["delta"] for r in all_kept if not math.isnan(r.get("delta", float("nan")))
        ]
        mean_delta_layout = (
            float(np.mean(accepted_deltas_layout)) if accepted_deltas_layout else float("nan")
        )
        accepted_frac_layout = (
            len(all_kept) / len(all_decided) if all_decided else float("nan")
        )

        p2_delta_holds = (
            (mean_delta_layout <= -0.2) if not math.isnan(mean_delta_layout) else False
        )
        p2_frac_holds = (
            (accepted_frac_layout >= 0.5) if not math.isnan(accepted_frac_layout) else False
        )
        p2_holds_layout = p2_delta_holds and p2_frac_holds

        print(f"\n  P2 completeness mechanism:")
        mean_delta_str = f"{mean_delta_layout:+.4f}" if not math.isnan(mean_delta_layout) else "N/A"
        acc_frac_str = f"{accepted_frac_layout * 100:.1f}%" if not math.isnan(accepted_frac_layout) else "N/A"
        print(f"    Attempted: {len(all_decided)}  Accepted: {len(all_kept)}  "
              f"AccFrac: {acc_frac_str}  (need >= 50%)  "
              f"{'PASS' if p2_frac_holds else 'FAIL'}")
        print(f"    Mean accepted delta: {mean_delta_str}  (need <= -0.2)  "
              f"{'PASS' if p2_delta_holds else 'FAIL'}")
        print(f"  P2 layout {layout_idx + 1}: {'PASS' if p2_holds_layout else 'FAIL (not a halt if P1 carries)'}")

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
            "p1_drop_fails": 8 - p1_drop_count,
            "p2_holds": p2_holds_layout,
            "p2_delta_holds": p2_delta_holds,
            "p2_frac_holds": p2_frac_holds,
            "mean_delta_layout": mean_delta_layout,
            "accepted_frac_layout": accepted_frac_layout,
            "total_attempted_layout": len(all_decided),
            "total_accepted_layout": len(all_kept),
            "p3_holds": p3_holds_layout,
            "p3_count": p3_count,
            "twin_holds": twin_holds_layout,
            "twin_count": twin_count,
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

    # P2: completeness mechanism — both arms per layout
    p2_all_layouts = all(v["p2_holds"] for v in layout_verdicts)
    print(f"\nP2 completeness mechanism (both arms, all 3 layouts):")
    for v in layout_verdicts:
        mean_str = (f"{v['mean_delta_layout']:+.4f}"
                    if not math.isnan(v["mean_delta_layout"]) else "N/A")
        frac_str = (f"{v['accepted_frac_layout'] * 100:.1f}%"
                    if not math.isnan(v["accepted_frac_layout"]) else "N/A")
        print(f"  Layout {v['layout_idx']}: attempted={v['total_attempted_layout']}  "
              f"accepted={v['total_accepted_layout']}  "
              f"acc_frac={frac_str}  mean_delta={mean_str}  "
              f"delta_arm={'PASS' if v['p2_delta_holds'] else 'FAIL'}  "
              f"frac_arm={'PASS' if v['p2_frac_holds'] else 'FAIL'}  "
              f"{'PASS' if v['p2_holds'] else 'FAIL (not halt if P1 carries)'}")
    print(f"  P2 GLOBAL: {'PASS' if p2_all_layouts else 'FAIL (not halt if P1 carries)'}")

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
            f"P1 SURPRISE FAILURE: drop < 0.4 nats in >= 3/8 seeds for "
            f"{halt_p1_layouts}/3 layouts (>= 2) — batch-jump crack fails"
        )
    if halt_regression:
        halt_triggers.append(
            f"P3 REGRESSION: final-500 loc_median > 0.5 in >= 2/8 seeds "
            f"in at least one layout — growth broke localization (HALT)"
        )

    print()
    if halt_triggers:
        for ht in halt_triggers:
            print(f"  Falsifier: {ht}")
    else:
        print("  No HALT arms triggered.")

    if not p2_all_layouts and not halt_migration_p1 and not halt_regression:
        print(f"  NOTE P2 FAIL: mechanism claim not supported — "
              f"MIXED (works for a different reason — log it).")

    # -----------------------------------------------------------------------
    # Three-way VERDICT
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)

    if halt_regression:
        verdict = "NEGATIVE"
        print(f"VERDICT: {verdict}")
        print("MIGRATION HALT")
        for ht in halt_triggers:
            print(f"  {ht}")
    elif halt_migration_p1:
        verdict = "NEGATIVE"
        print(f"VERDICT: {verdict}")
        print("PROCEED TO BACKGROUND-FLOOR (pre-authorized)")
        for ht in halt_triggers:
            print(f"  {ht}")
    elif p1_all_layouts and p2_all_layouts and p3_all_layouts:
        verdict = "POSITIVE"
        print(f"VERDICT: {verdict}")
        print(
            "All predeclared properties satisfied (P1 + P2 + P3 across all 3 layouts). "
            "Batch-jump EM replacement enables growth to reduce predictive surprise; "
            "completeness is the operative mechanism (accepted deltas negative, high "
            "acceptance rate); no localization regression. "
            "Migration thread may advance per PROTOCOL."
        )
    elif p1_all_layouts and p3_all_layouts and not p2_all_layouts:
        verdict = "MIXED"
        print(f"VERDICT: {verdict}")
        print(
            "P1 (alarm answered) and P3 (no regression) pass across all 3 layouts. "
            "P2 (completeness mechanism) fails — batch-jump works but the mean accepted "
            "delta or acceptance fraction doesn't meet the mechanistic bar. "
            "MIXED (works for a different reason — log it). No HALT (P1 carries)."
        )
    elif p1_all_layouts and not p3_all_layouts:
        # Already caught by halt_regression
        verdict = "NEGATIVE"
        print(f"VERDICT: {verdict}")
        print("MIGRATION HALT — P3 regression detected.")
    elif not p1_all_layouts and not halt_triggers:
        p1_pass_layouts = sum(1 for v in layout_verdicts if v["p1_holds"])
        verdict = "MIXED"
        print(f"VERDICT: {verdict}")
        print(
            f"P1 holds in {p1_pass_layouts}/3 layouts (need all 3). "
            f"P2={'PASS' if p2_all_layouts else 'FAIL'}. "
            f"P3={'PASS' if p3_all_layouts else 'FAIL'}. "
            f"Partial progress — HALT not triggered. Inspect attempts in failing layout."
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
            "exp": 152,
            "rung": "M3e",
            "layout_seed": v["layout_seed"],
            "layout_idx": v["layout_idx"],
            "seed": -1,
            "summary": True,
            "p1_holds": v["p1_holds"],
            "p1_drop_count": v["p1_drop_count"],
            "p1_quiet_count": v["p1_quiet_count"],
            "p1_comps_count": v["p1_comps_count"],
            "p2_holds": v["p2_holds"],
            "p2_delta_holds": v["p2_delta_holds"],
            "p2_frac_holds": v["p2_frac_holds"],
            "mean_delta_layout": (
                v["mean_delta_layout"] if not math.isnan(v["mean_delta_layout"]) else None
            ),
            "accepted_frac_layout": (
                v["accepted_frac_layout"]
                if not math.isnan(v["accepted_frac_layout"]) else None
            ),
            "p3_holds": v["p3_holds"],
            "p3_count": v["p3_count"],
            "twin_holds": v["twin_holds"],
            "twin_count": v["twin_count"],
        })
    all_rows_json.append({
        "exp": 152,
        "rung": "M3e",
        "seed": -2,
        "global_summary": True,
        "p1_global": p1_all_layouts,
        "p2_global": p2_all_layouts,
        "p3_global": p3_all_layouts,
        "halt_migration_p1": halt_migration_p1,
        "halt_regression": halt_regression,
        "verdict": verdict,
    })

    out_path = Path(__file__).parent / "outputs" / "exp152_rows.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for row in all_rows_json:
            fh.write(json.dumps(row) + "\n")
    print(f"\nJSON rows written to {out_path}")


if __name__ == "__main__":
    main()
