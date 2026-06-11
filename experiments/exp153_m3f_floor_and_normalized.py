"""
Exp 153 — continuous-creature rung M3f: the BACKGROUND-FLOOR crack (pre-authorized
crack 2) and the NORMALIZED-DENSITY diagnostic of Exp 152's dilution hypothesis,
on the same worlds and seeds as every failed design.

Card loop/directions/continuous-creature.md (re-opened). Context: four growth
designs (add, add+probation, split, batch-jump) have failed the same
alarm-answered bar; Exp 152's autopsy proposed the DILUTION mechanism — under the
rung-1 UNNORMALIZED-FOOTPRINT convention, footprint values are capped at 1
regardless of tightness, so K tight components speak at ~1/K the volume of one
broad component and sharpness buys no density reward. If true, the wall is
substantially the CONVENTION's, not growth's.

Key structural point (declared): the conjugate place UPDATE keeps the footprint
form (conjugacy untouched); only the PREDICTIVE evaluation (surprise, the
detector, probation scoring) changes in the normalized arm — evaluation is not
an update, so nothing about inference or learning is altered.

ARM A — background-floor (the authorized crack, standard convention): the
color's original broad component becomes a PERMANENT floor (within-color weight
0.3, never modified, never removed); on an alarm + budget, batch-fit tight
components (Exp 152's EM, K in {2,3,4} by penalized replay NLL) are installed ON
TOP at combined weight 0.7 (w_j renormalized within); probation as validated
(keep iff the color's live mean drops >= 0.1 vs pre-install; revert restores).
Mechanism prediction: harmless but insufficient — no surge (the floor prevents
holes) but no 0.4-nat drop either (dilution caps the tight components' voice).

ARM B — normalized-density batch-jump (the diagnostic): Exp 152's batch-jump
verbatim, except every PREDICTIVE evaluation uses the normalized mixture
integral log p(o=k | belief) propto logsumexp_j [ log w_kj + log N(mu_belief;
m_kj, Cov_kj + Sigma_belief) ] (the true Gaussian-mixture marginal, with the
1/(2*pi*|C|^(1/2)) factor — sharp components are finally allowed to be loud),
normalized over colors. The place update is UNCHANGED (footprint conjugate).
Phase-1 plateau and the detector run on the same normalized predictive within
this arm (the bar is internal to the arm).

Setup: three layouts (rng 7/11/13) x seeds 0..7 x two arms on identical streams;
phase 1 T=2000 unimodal, phase 2 T=6000, jump/install cadence and budgets as
Exp 152; eval = final 1000.

Predictions (TRUE iff all):
- P1 the dilution mechanism (Arm B): the alarm-answered bar — final-1000 mean
  surprise <= plateau - 0.4 nats AND zero ceiling events in the final 1000 AND
  components >= 3 for >= 3/4 colors — in >= 6/8 seeds per layout, >= 2/3
  layouts.
- P2 the floor under the old convention (Arm A): NO SURGE (per-layout mean
  probation delta across attempts within (-0.3, +0.3)) AND the drop arm FAILS
  (< 6/8 seeds reach the 0.4 drop) — the harmless-but-insufficient signature.
- P3 no regression: final-500 median localization <= 0.5 in >= 7/8 seeds per
  layout, BOTH arms.

Named outcome map (predeclared): B passes & A insufficient -> the growth wall is
RE-BOUNDED to the unnormalized-footprint convention (major finding; the wall doc
gets amended). Both arms fail their bars -> the wall stands with both authorized
cracks AND the convention hypothesis spent -> MIGRATION HALT consult. A passes
its alarm bar -> the floor crack works even under the convention (log it; growth
unlocked). P3 failing anywhere -> HALT (regression). Three-way rule per PROTOCOL
step 3; blinded verification (step 4.5) before logging.
"""
from __future__ import annotations

# NOTE: copied from exp152 (which copied from exp145/exp144/exp143).
# Arm A changes from exp152:
#   - Phase 1: broad initial component becomes a permanent floor (weight 0.3,
#     never updated after phase 1 ends).
#   - Phase 2 install: tight EM components installed at combined weight 0.7,
#     renormalized within; floor is frozen from that point.
#   - Hard-assignment updates during phase 2 go to non-floor components only
#     when their responsibility > floor's responsibility.
#   - Predictive evaluation: UNNORMALIZED convention (standard, as in exp152).
# Arm B changes from exp152:
#   - Single predictive function swap: normalized Gaussian-mixture marginal
#     with full normalization constant (2pi|C|^{1/2} terms included).
#   - Applies to: per-step surprise, per-color alarm stats, detector window,
#     probation pre/post means, plateau/final metrics.
#   - Place update: UNCHANGED (footprint conjugate).

import copy
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from active_loop.continuous import NIW
from active_loop.creature_continuous import ContinuousPlace

# ---------------------------------------------------------------------------
# Detector constants — copied from exp145/exp152
# ---------------------------------------------------------------------------
CEILING_MEAN_THRESH: float = 0.7    # creature.py line 56
CEILING_SLOPE_THRESH: float = 5e-4  # creature.py line 59
SURPRISE_WINDOW: int = 200          # creature.py line 52

# ---------------------------------------------------------------------------
# World constants — copied from exp145/exp152
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
T_PHASE2 = 6000           # M3e/M3f: 6000
T_TOTAL = T_PHASE1 + T_PHASE2

# Jump / probation constants — identical to exp152
JUMP_INTERVAL = 200
JUMP_COOLDOWN = 1200
REPLAY_WINDOW = 400
FINAL_EVAL_WINDOW = 1000
P1_PLATEAU_WINDOW = 500

EM_ITERS = 10
EM_COV_FLOOR = 1e-4
K_CANDIDATES = [2, 3, 4]
K_PENALTY = 0.05
MIN_REPLAY_PAIRS = 30

COLOR_SURPRISE_WINDOW = 50
ALARM_THRESH = 0.7
PRE_JUMP_WINDOW = 100
PROBATION_STEPS = 400

# Arm A specific constants
FLOOR_WEIGHT = 0.3       # frozen floor within-color weight
TIGHT_WEIGHT = 0.7       # combined weight for batch-fitted tight components

SEEDS = list(range(8))
LAYOUT_SEEDS = [7, 11, 13]

LOG_2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Build a CMAP from a given rng seed — copied from exp152
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
# Compute true within-color scatter traces — copied from exp152
# ---------------------------------------------------------------------------

def true_scatter_traces(cmap: np.ndarray) -> np.ndarray:
    """For each color k, compute tr(cov of its 4 cell centers, ddof=0)."""
    traces = np.empty(N_COLORS, dtype=float)
    for k in range(N_COLORS):
        cells_k = [i for i in range(N_CELLS) if cmap[i] == k]
        centers_k = CELL_CENTERS[cells_k]
        cov_k = np.cov(centers_k.T, ddof=0)
        traces[k] = float(np.trace(cov_k))
    return traces


# ---------------------------------------------------------------------------
# Grid world move (wall-clamped) — copied from exp152
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
# Diagonal 2x2 Gaussian utilities — copied from exp152
# ---------------------------------------------------------------------------

def _diag_log_integral(
    mu_p: np.ndarray, Sigma_p_diag: np.ndarray,
    m_k: np.ndarray, ESigma_k_diag: np.ndarray,
) -> float:
    """Log integral of N(s; m_k, ESigma_k) * N(s; mu_p, Sigma_p) over R^2.

    For diagonal 2x2:
        log I = 0.5*(logdet(Sk) - logdet(Sk + Sp)) - 0.5*maha
    where Sk = diag(ESigma_k_diag), Sp = diag(Sigma_p_diag).

    NOTE: this is the UNNORMALIZED footprint integral (no 1/(2pi|C|^{1/2})
    normalization). Used for the standard (Arm A) convention.
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


def _diag_log_gaussian_eval(
    mu_p: np.ndarray, Sigma_p_diag: np.ndarray,
    m_k: np.ndarray, ESigma_k_diag: np.ndarray,
) -> float:
    """Log N(mu_p; m_k, Cov_k + Sigma_p) — the NORMALIZED Gaussian density.

    This is the true Gaussian-mixture marginal term for Arm B:
        log p(o | belief, component j) = log N(mu_belief; m_j, Cov_j + Sigma_belief)
        = -D/2*log(2*pi) - 0.5*logdet(C) - 0.5*maha
    where C = diag(ESigma_k_diag + Sigma_p_diag).

    Sharp components have smaller |C|, so they are louder (density reward).
    """
    c0 = ESigma_k_diag[0] + Sigma_p_diag[0]
    c1 = ESigma_k_diag[1] + Sigma_p_diag[1]
    logdet_C = math.log(c0) + math.log(c1)
    d0 = mu_p[0] - m_k[0]
    d1 = mu_p[1] - m_k[1]
    maha = d0 * d0 / c0 + d1 * d1 / c1
    return -D * 0.5 * LOG_2PI - 0.5 * logdet_C - 0.5 * maha


def _diag_gaussian_product(
    mu_a: np.ndarray, Sigma_a_diag: np.ndarray,
    mu_b: np.ndarray, Sigma_b_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Product of two diagonal Gaussians -> (mu, Sigma_diag)."""
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
# UNNORMALIZED predictive (standard convention, Arm A) — copied from exp152
# ---------------------------------------------------------------------------

def mixture_predictive_logprobs(
    mu_p: np.ndarray,
    Sigma_p_diag: np.ndarray,
    components: list[list[tuple[float, NIW]]],
) -> np.ndarray:
    """Compute normalized log p(color k) under the pre-update mixture predictive.

    Uses the UNNORMALIZED footprint integral (rung-1 convention).
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


# ---------------------------------------------------------------------------
# NORMALIZED predictive (Arm B) — new for Exp 153
# ---------------------------------------------------------------------------

def mixture_predictive_logprobs_normalized(
    mu_p: np.ndarray,
    Sigma_p_diag: np.ndarray,
    components: list[list[tuple[float, NIW]]],
) -> np.ndarray:
    """Compute normalized log p(color k) under the NORMALIZED Gaussian-mixture marginal.

    For color k with components (w_kj, NIW_kj):
        log_term_kj = log w_kj + log N(mu_p; m_kj, Cov_kj + Sigma_p)
    where log N includes the full 1/(2*pi*|C|^(1/2)) normalization constant,
    so sharp (small Cov_kj) components are louder.

    log p(k) = logsumexp_j(log_term_kj)   [unnormalized over k]
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
            log_gauss = _diag_log_gaussian_eval(
                mu_p, Sigma_p_diag, m_kj, ESigma_kj_diag
            )
            log_terms.append(math.log(max(w_kj, 1e-300)) + log_gauss)
        log_terms_arr = np.array(log_terms)
        log_unnorm[k] = float(np.logaddexp.reduce(log_terms_arr))
    log_sum = float(np.logaddexp.reduce(log_unnorm))
    return log_unnorm - log_sum


# ---------------------------------------------------------------------------
# Mixture emission moments — copied from exp152 (shared by both arms)
# ---------------------------------------------------------------------------

def mixture_emission_moments(
    mu_p: np.ndarray,
    Sigma_p_diag: np.ndarray,
    k: int,
    components: list[list[tuple[float, NIW]]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Moment-match the mixture emission for observed color k to a single Gaussian.

    Uses the UNNORMALIZED integral for responsibilities (shared by both arms —
    the place update is UNCHANGED). Returns (mu_mix, Sigma_mix_diag, hard_idx).

    Declared approximation: moment-matching to diagonal Gaussian.
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
    r = np.exp(log_r - log_r_sum)

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
# Surprise-ceiling detector — copied from exp152
# ---------------------------------------------------------------------------

def check_ceiling(surprise_buf: deque) -> bool:
    """Return True if the ceiling conjunction fires."""
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
# Replay NLL scorer for a color subset — copied from exp152
# ---------------------------------------------------------------------------

def color_replay_nll(
    color_replay_pairs: list[tuple[np.ndarray, np.ndarray]],
    comp_means: np.ndarray,
    comp_covs: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Mean -log sum_j w_j N(mean_i; m_j, Cov_j + Sigma_i) over color replay pairs."""
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
# Deep-copy utilities — copied from exp152
# ---------------------------------------------------------------------------

def _copy_components(
    components: list[list[tuple[float, NIW]]],
) -> list[list[tuple[float, NIW]]]:
    result = []
    for color_comps in components:
        result.append([(w, copy.deepcopy(niw)) for (w, niw) in color_comps])
    return result


def _copy_color_components(
    color_comps: list[tuple[float, NIW]],
) -> list[tuple[float, NIW]]:
    return [(w, copy.deepcopy(niw)) for (w, niw) in color_comps]


def _copy_counts(counts: list[list[int]]) -> list[list[int]]:
    return [list(c) for c in counts]


# ---------------------------------------------------------------------------
# Farthest-point seeding — copied from exp152
# ---------------------------------------------------------------------------

def _farthest_point_init(
    obs_means: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(obs_means)
    first_idx = int(rng.integers(0, n))
    centers = [obs_means[first_idx].copy()]
    for _ in range(1, K):
        min_sq_dists = np.full(n, np.inf)
        for c in centers:
            diffs = obs_means - c
            sq_dists = (diffs ** 2).sum(axis=1)
            min_sq_dists = np.minimum(min_sq_dists, sq_dists)
        next_idx = int(np.argmax(min_sq_dists))
        centers.append(obs_means[next_idx].copy())
    return np.array(centers)


# ---------------------------------------------------------------------------
# Batch-jump EM — copied from exp152
# ---------------------------------------------------------------------------

def batch_jump_em(
    color_replay_pairs: list[tuple[np.ndarray, np.ndarray]],
    K: int,
    rng: np.random.Generator,
    n_iters: int = EM_ITERS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a K-component diagonal-Gaussian mixture on color replay pairs via EM."""
    n = len(color_replay_pairs)
    obs_means = np.array([mu for (mu, _) in color_replay_pairs])
    obs_sigmas = np.array([sig for (_, sig) in color_replay_pairs])

    centers = _farthest_point_init(obs_means, K, rng)
    weights = np.full(K, 1.0 / K)
    comp_means = centers.copy()
    comp_covs = np.full((K, 2), 0.5)

    for _ in range(n_iters):
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

        n_j = r.sum(axis=0)
        n_j_safe = np.maximum(n_j, 1e-9)

        new_weights = n_j / n
        new_weights = np.maximum(new_weights, 1e-300)
        new_weights /= new_weights.sum()

        new_means = (r.T @ obs_means) / n_j_safe[:, None]

        new_covs = np.zeros((K, 2), dtype=float)
        for j in range(K):
            diff = obs_means - new_means[j]
            scatter = (r[:, j, None] * (diff ** 2 + obs_sigmas)).sum(axis=0)
            new_covs[j] = scatter / n_j_safe[j]
        new_covs = np.maximum(new_covs, EM_COV_FLOOR)

        weights = new_weights
        comp_means = new_means
        comp_covs = new_covs

    return weights, comp_means, comp_covs


# ---------------------------------------------------------------------------
# Build NIW component list from EM result — copied from exp152
# ---------------------------------------------------------------------------

def em_result_to_components(
    weights: np.ndarray,
    comp_means: np.ndarray,
    comp_covs: np.ndarray,
    n_eff: np.ndarray,
) -> list[tuple[float, NIW]]:
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
# Select best K via penalized replay NLL — copied from exp152
# ---------------------------------------------------------------------------

def select_best_k(
    color_replay_pairs: list[tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    print_debug: bool = False,
    attempt_num: int = 0,
    color_idx: int = 0,
) -> tuple[int, list[tuple[float, NIW]], np.ndarray, dict]:
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
# Tabular twin — copied from exp152 (sanity only)
# ---------------------------------------------------------------------------

def run_tabular(actions: np.ndarray, cmap: np.ndarray) -> dict:
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
# ARM B: batch-jump with NORMALIZED predictive
# (exp152 verbatim except predictive function swapped)
# ---------------------------------------------------------------------------

def run_arm_b_normalized(
    actions: np.ndarray,
    cmap: np.ndarray,
    run_rng: np.random.Generator,
    seed: int = 0,
    layout_idx: int = 1,
    print_k_debug: bool = False,
) -> dict:
    """Run Arm B: Exp 152 batch-jump with NORMALIZED Gaussian-mixture predictive.

    The place update is UNCHANGED (footprint conjugate, unnormalized integral).
    The normalized predictive is used ONLY for:
      - per-step surprise
      - per-color alarm stats
      - detector window
      - probation pre/post means
      - plateau/final metrics

    Everything else (EM fitting, NIW updates, responsibilities for place update)
    is identical to exp152.
    """
    mu0 = ARENA_CENTER.copy()
    Sigma0_diag = np.array([4.0, 4.0])
    cp = ContinuousPlace(mu0, np.diag(Sigma0_diag), ARENA)

    components: list[list[tuple[float, NIW]]] = []
    counts: list[list[int]] = []
    for k in range(N_COLORS):
        niw0 = NIW(m=ARENA_CENTER.copy(), kappa=KAPPA0, nu=NU0, S=S0.copy())
        components.append([(1.0, niw0)])
        counts.append([1])

    surprise_buf: deque = deque(maxlen=SURPRISE_WINDOW)
    replay_buf: deque = deque(maxlen=REPLAY_WINDOW)
    color_surprise_bufs: list[deque] = [deque(maxlen=COLOR_SURPRISE_WINDOW) for _ in range(N_COLORS)]
    color_pre_jump_bufs: list[deque] = [deque(maxlen=PRE_JUMP_WINDOW) for _ in range(N_COLORS)]

    last_attempt_step: list[int] = [-JUMP_COOLDOWN - 1] * N_COLORS

    probation_color: int = -1
    probation_start_phase2_t: int = -1
    probation_pre_jump_mean: float = float("inf")
    probation_color_snap: list[tuple[float, NIW]] = []
    probation_color_counts_snap: list[int] = []
    probation_observations: list[float] = []

    attempt_records: list[dict] = []
    attempt_counter: list[int] = [0]

    phase1_loc_errors = np.empty(T_PHASE1)
    phase1_ceiling_events = 0
    phase1_surprise_vals = []

    phase2_loc_errors = np.empty(T_PHASE2)
    phase2_ceiling_events_total = 0
    phase2_final_ceiling_events = 0
    jumps_kept = 0
    jumps_reverted = 0
    phase2_surprise_vals = []

    true_cell = 0

    for t in range(T_TOTAL):
        obs_k = int(cmap[true_cell])

        Sigma_p_diag = np.array([cp.Sigma[0, 0], cp.Sigma[1, 1]])
        Sigma_p_diag = np.maximum(Sigma_p_diag, 1e-9)
        mu_p = cp.mu

        replay_buf.append((obs_k, mu_p.copy(), Sigma_p_diag.copy()))

        # ARM B: use NORMALIZED predictive for surprise
        log_probs = mixture_predictive_logprobs_normalized(mu_p, Sigma_p_diag, components)
        surprise_t = float(-log_probs[obs_k])
        surprise_buf.append(surprise_t)

        is_phase1 = t < T_PHASE1
        phase2_t = t - T_PHASE1

        if is_phase1:
            phase1_loc_errors[t] = float(np.linalg.norm(mu_p - CELL_CENTERS[true_cell]))
            phase1_surprise_vals.append(surprise_t)
            if check_ceiling(surprise_buf):
                phase1_ceiling_events += 1
        else:
            phase2_loc_errors[phase2_t] = float(np.linalg.norm(mu_p - CELL_CENTERS[true_cell]))
            phase2_surprise_vals.append(surprise_t)

            color_surprise_bufs[obs_k].append(surprise_t)
            color_pre_jump_bufs[obs_k].append(surprise_t)

            if probation_color == obs_k:
                probation_observations.append(surprise_t)

            is_ceiling = check_ceiling(surprise_buf)
            if is_ceiling:
                phase2_ceiling_events_total += 1
                if phase2_t >= T_PHASE2 - FINAL_EVAL_WINDOW:
                    phase2_final_ceiling_events += 1

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
                    else:
                        jumps_reverted += 1
                        components[pc] = probation_color_snap
                        counts[pc] = probation_color_counts_snap

                    probation_color = -1
                    probation_start_phase2_t = -1
                    probation_pre_jump_mean = float("inf")
                    probation_color_snap = []
                    probation_color_counts_snap = []
                    probation_observations = []

            if (probation_color < 0
                    and phase2_t > 0
                    and phase2_t % JUMP_INTERVAL == 0):

                eligible_colors = []
                for k in range(N_COLORS):
                    if len(color_surprise_bufs[k]) > 0:
                        mean_k = float(np.mean(color_surprise_bufs[k]))
                        if mean_k >= ALARM_THRESH:
                            cooldown_elapsed = (phase2_t - last_attempt_step[k]) >= JUMP_COOLDOWN
                            if cooldown_elapsed:
                                eligible_colors.append(k)

                if eligible_colors:
                    jump_color = min(eligible_colors, key=lambda k: (last_attempt_step[k], k))
                    last_attempt_step[jump_color] = phase2_t

                    color_replay_pairs = [
                        (mu_s.copy(), sig_s.copy())
                        for (obs_c, mu_s, sig_s) in replay_buf
                        if obs_c == jump_color
                    ]

                    if len(color_replay_pairs) >= MIN_REPLAY_PAIRS:
                        attempt_num = attempt_counter[0]
                        attempt_counter[0] += 1

                        pre_jump_buf = list(color_pre_jump_bufs[jump_color])
                        if pre_jump_buf:
                            pre_jump_mean = float(np.mean(pre_jump_buf))
                        else:
                            pre_jump_mean = float("inf")

                        snap_comps = _copy_color_components(components[jump_color])
                        snap_counts = list(counts[jump_color])

                        do_print = print_k_debug and (seed == 0) and (layout_idx == 1)
                        best_K, best_comps, best_n_eff, nll_by_k = select_best_k(
                            color_replay_pairs,
                            run_rng,
                            print_debug=do_print,
                            attempt_num=attempt_num,
                            color_idx=jump_color,
                        )

                        components[jump_color] = best_comps

                        n_total = len(color_replay_pairs)
                        new_counts = [max(1, int(round(float(best_n_eff[j]) * n_total / sum(best_n_eff))))
                                      for j in range(best_K)]
                        counts[jump_color] = new_counts

                        probation_color = jump_color
                        probation_start_phase2_t = phase2_t
                        probation_pre_jump_mean = pre_jump_mean
                        probation_color_snap = snap_comps
                        probation_color_counts_snap = snap_counts
                        probation_observations = []

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

        # Place update: UNCHANGED (unnormalized integral for responsibilities)
        mu_mix, Sigma_mix_diag, hard_idx = mixture_emission_moments(
            mu_p, Sigma_p_diag, obs_k, components
        )
        cp.update(mu_mix, np.diag(Sigma_mix_diag))

        post_mu = cp.mu
        post_Sigma_diag = np.array([cp.Sigma[0, 0], cp.Sigma[1, 1]])
        post_Sigma_diag = np.maximum(post_Sigma_diag, 1e-9)

        old_niw = components[obs_k][hard_idx][1]
        new_niw = old_niw.update_moments(post_mu, np.diag(post_Sigma_diag))
        components[obs_k][hard_idx] = (components[obs_k][hard_idx][0], new_niw)

        counts[obs_k][hard_idx] += 1
        total_k = sum(counts[obs_k])
        components[obs_k] = [(counts[obs_k][j] / total_k, niw)
                             for j, (_, niw) in enumerate(components[obs_k])]

        action = int(actions[t])
        true_cell = move(true_cell, action)
        cp.predict_clamped_moments(ACTION_DELTA[action], Q)

    # Handle in-flight probation
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

    final_comps_per_color = [len(components[k]) for k in range(N_COLORS)]

    p1_final500_loc_median = float(np.median(phase1_loc_errors[T_PHASE1 - 500:]))
    p2_final500_loc_median = float(np.median(phase2_loc_errors[T_PHASE2 - 500:]))

    phase1_plateau = float(np.mean(phase1_surprise_vals[-P1_PLATEAU_WINDOW:]))
    final_surprise = float(np.mean(phase2_surprise_vals[-FINAL_EVAL_WINDOW:]))
    drop = phase1_plateau - final_surprise

    decided_records = [r for r in attempt_records if r.get("kept") is not None]
    kept_records = [r for r in decided_records if r.get("kept") is True]

    accepted_deltas = [r["delta"] for r in kept_records if not math.isnan(r.get("delta", float("nan")))]
    mean_accepted_delta = float(np.mean(accepted_deltas)) if accepted_deltas else float("nan")

    total_attempted = len(decided_records)
    total_accepted = len(kept_records)
    accepted_frac = total_accepted / total_attempted if total_attempted > 0 else float("nan")

    return {
        "arm": "normalized",
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
# ARM A: background-floor (standard/unnormalized convention)
# ---------------------------------------------------------------------------

def run_arm_a_floor(
    actions: np.ndarray,
    cmap: np.ndarray,
    run_rng: np.random.Generator,
    seed: int = 0,
    layout_idx: int = 1,
    print_k_debug: bool = False,
) -> dict:
    """Run Arm A: background-floor install under the standard unnormalized convention.

    Phase 1: runs normally (unimodal mixture); after phase 1 ends, each color's
    single initial component is frozen as the FLOOR at within-color weight 0.3.
    The floor is never updated further (declared; hard-assignment updates during
    phase 2 go to non-floor components only when their responsibility >= floor's).

    Phase 2 install: on alarm+budget, fit tight EM components (K in {2,3,4} by
    penalized replay NLL); install them at combined weight 0.7 on top of the
    frozen floor (w_j of tight components = 0.7 * w_j_em, renormalized within
    the 0.7 budget); probation as exp152 (keep iff color mean drops >= 0.1).

    The predictive evaluation uses the UNNORMALIZED convention (same as exp152).
    """
    mu0 = ARENA_CENTER.copy()
    Sigma0_diag = np.array([4.0, 4.0])
    cp = ContinuousPlace(mu0, np.diag(Sigma0_diag), ARENA)

    components: list[list[tuple[float, NIW]]] = []
    counts: list[list[int]] = []
    for k in range(N_COLORS):
        niw0 = NIW(m=ARENA_CENTER.copy(), kappa=KAPPA0, nu=NU0, S=S0.copy())
        components.append([(1.0, niw0)])
        counts.append([1])

    # floor_niw[k]: the frozen broad component for each color; set at end of phase 1
    floor_niw: list[NIW | None] = [None] * N_COLORS
    floor_installed: bool = False  # True after phase 1 ends

    surprise_buf: deque = deque(maxlen=SURPRISE_WINDOW)
    replay_buf: deque = deque(maxlen=REPLAY_WINDOW)
    color_surprise_bufs: list[deque] = [deque(maxlen=COLOR_SURPRISE_WINDOW) for _ in range(N_COLORS)]
    color_pre_jump_bufs: list[deque] = [deque(maxlen=PRE_JUMP_WINDOW) for _ in range(N_COLORS)]

    last_attempt_step: list[int] = [-JUMP_COOLDOWN - 1] * N_COLORS

    probation_color: int = -1
    probation_start_phase2_t: int = -1
    probation_pre_jump_mean: float = float("inf")
    probation_color_snap: list[tuple[float, NIW]] = []
    probation_color_counts_snap: list[int] = []
    probation_observations: list[float] = []

    attempt_records: list[dict] = []
    attempt_counter: list[int] = [0]

    phase1_loc_errors = np.empty(T_PHASE1)
    phase1_ceiling_events = 0
    phase1_surprise_vals = []

    phase2_loc_errors = np.empty(T_PHASE2)
    phase2_ceiling_events_total = 0
    phase2_final_ceiling_events = 0
    jumps_kept = 0
    jumps_reverted = 0
    phase2_surprise_vals = []

    true_cell = 0

    for t in range(T_TOTAL):
        obs_k = int(cmap[true_cell])

        Sigma_p_diag = np.array([cp.Sigma[0, 0], cp.Sigma[1, 1]])
        Sigma_p_diag = np.maximum(Sigma_p_diag, 1e-9)
        mu_p = cp.mu

        replay_buf.append((obs_k, mu_p.copy(), Sigma_p_diag.copy()))

        # ARM A: UNNORMALIZED predictive (standard convention)
        log_probs = mixture_predictive_logprobs(mu_p, Sigma_p_diag, components)
        surprise_t = float(-log_probs[obs_k])
        surprise_buf.append(surprise_t)

        is_phase1 = t < T_PHASE1
        phase2_t = t - T_PHASE1

        if is_phase1:
            phase1_loc_errors[t] = float(np.linalg.norm(mu_p - CELL_CENTERS[true_cell]))
            phase1_surprise_vals.append(surprise_t)
            if check_ceiling(surprise_buf):
                phase1_ceiling_events += 1
        else:
            # --- At very first phase-2 step: install floor ---
            if not floor_installed:
                floor_installed = True
                for k in range(N_COLORS):
                    # The current single component becomes the floor
                    # Deep-copy the NIW so it's frozen from this point
                    assert len(components[k]) == 1, (
                        f"Floor install: expected 1 component for color {k}, "
                        f"got {len(components[k])}"
                    )
                    floor_niw[k] = copy.deepcopy(components[k][0][1])
                    # Re-weight: floor at FLOOR_WEIGHT (0.3), same component
                    components[k] = [(FLOOR_WEIGHT, floor_niw[k])]
                    counts[k] = [1]  # counts reset; weight is now explicit floor

            phase2_loc_errors[phase2_t] = float(np.linalg.norm(mu_p - CELL_CENTERS[true_cell]))
            phase2_surprise_vals.append(surprise_t)

            color_surprise_bufs[obs_k].append(surprise_t)
            color_pre_jump_bufs[obs_k].append(surprise_t)

            if probation_color == obs_k:
                probation_observations.append(surprise_t)

            is_ceiling = check_ceiling(surprise_buf)
            if is_ceiling:
                phase2_ceiling_events_total += 1
                if phase2_t >= T_PHASE2 - FINAL_EVAL_WINDOW:
                    phase2_final_ceiling_events += 1

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
                    else:
                        jumps_reverted += 1
                        # Restore snapshot (which includes the floor + any tight comps)
                        components[pc] = probation_color_snap
                        counts[pc] = probation_color_counts_snap

                    probation_color = -1
                    probation_start_phase2_t = -1
                    probation_pre_jump_mean = float("inf")
                    probation_color_snap = []
                    probation_color_counts_snap = []
                    probation_observations = []

            if (probation_color < 0
                    and phase2_t > 0
                    and phase2_t % JUMP_INTERVAL == 0):

                eligible_colors = []
                for k in range(N_COLORS):
                    if len(color_surprise_bufs[k]) > 0:
                        mean_k = float(np.mean(color_surprise_bufs[k]))
                        if mean_k >= ALARM_THRESH:
                            cooldown_elapsed = (phase2_t - last_attempt_step[k]) >= JUMP_COOLDOWN
                            if cooldown_elapsed:
                                eligible_colors.append(k)

                if eligible_colors:
                    jump_color = min(eligible_colors, key=lambda k: (last_attempt_step[k], k))
                    last_attempt_step[jump_color] = phase2_t

                    color_replay_pairs = [
                        (mu_s.copy(), sig_s.copy())
                        for (obs_c, mu_s, sig_s) in replay_buf
                        if obs_c == jump_color
                    ]

                    if len(color_replay_pairs) >= MIN_REPLAY_PAIRS:
                        attempt_num = attempt_counter[0]
                        attempt_counter[0] += 1

                        pre_jump_buf = list(color_pre_jump_bufs[jump_color])
                        if pre_jump_buf:
                            pre_jump_mean = float(np.mean(pre_jump_buf))
                        else:
                            pre_jump_mean = float("inf")

                        # Snapshot current mixture (floor + any existing tight comps)
                        snap_comps = _copy_color_components(components[jump_color])
                        snap_counts = list(counts[jump_color])

                        do_print = print_k_debug and (seed == 0) and (layout_idx == 1)
                        best_K, best_comps_raw, best_n_eff, nll_by_k = select_best_k(
                            color_replay_pairs,
                            run_rng,
                            print_debug=do_print,
                            attempt_num=attempt_num,
                            color_idx=jump_color,
                        )

                        # ARM A: compose floor + tight components
                        # Tight EM weights renormalized within the 0.7 budget
                        tight_weights_raw = np.array([w for (w, _) in best_comps_raw])
                        tight_weights = tight_weights_raw / tight_weights_raw.sum() * TIGHT_WEIGHT

                        new_components_k: list[tuple[float, NIW]] = []
                        # Floor first (frozen, weight FLOOR_WEIGHT)
                        new_components_k.append((FLOOR_WEIGHT, copy.deepcopy(floor_niw[jump_color])))
                        # Tight components at combined weight TIGHT_WEIGHT
                        for j, (_, niw_j) in enumerate(best_comps_raw):
                            new_components_k.append((float(tight_weights[j]), niw_j))

                        components[jump_color] = new_components_k

                        # Counts: floor = 1 (frozen placeholder), tight = proportional
                        n_total = len(color_replay_pairs)
                        tight_counts = [
                            max(1, int(round(float(best_n_eff[j]) * n_total / sum(best_n_eff))))
                            for j in range(best_K)
                        ]
                        # Floor count = 1; total managed via explicit weights not counts
                        # (we don't use counts to re-weight in phase 2 for Arm A —
                        # weights are set explicitly; counts just guard hard_idx updates)
                        counts[jump_color] = [1] + tight_counts

                        probation_color = jump_color
                        probation_start_phase2_t = phase2_t
                        probation_pre_jump_mean = pre_jump_mean
                        probation_color_snap = snap_comps
                        probation_color_counts_snap = snap_counts
                        probation_observations = []

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

        # --- Place update (UNCHANGED) ---
        mu_mix, Sigma_mix_diag, hard_idx = mixture_emission_moments(
            mu_p, Sigma_p_diag, obs_k, components
        )
        cp.update(mu_mix, np.diag(Sigma_mix_diag))

        post_mu = cp.mu
        post_Sigma_diag = np.array([cp.Sigma[0, 0], cp.Sigma[1, 1]])
        post_Sigma_diag = np.maximum(post_Sigma_diag, 1e-9)

        # ARM A: NIW hard-assignment.
        # In phase 2, skip update to the floor component (index 0) if there's
        # a non-floor component that wins hard responsibility. The floor is frozen.
        if floor_installed and len(components[obs_k]) > 1:
            # If hard_idx == 0 (floor), redirect to the closest non-floor component
            # by finding the component with highest non-floor responsibility.
            comps_k = components[obs_k]
            if hard_idx == 0:
                # Find non-floor component with highest unnormalized responsibility
                n_c = len(comps_k)
                log_r_list = []
                for j in range(n_c):
                    w_kj, niw_kj = comps_k[j]
                    m_kj = niw_kj.m
                    ESigma_kj = niw_kj.expected_Sigma()
                    ESigma_kj_diag = np.maximum(
                        np.array([ESigma_kj[0, 0], ESigma_kj[1, 1]]), 1e-9
                    )
                    logI = _diag_log_integral(mu_p, Sigma_p_diag, m_kj, ESigma_kj_diag)
                    log_r_list.append(math.log(max(w_kj, 1e-300)) + logI)
                # Among non-floor (j >= 1), pick argmax
                non_floor_log_r = log_r_list[1:]
                if non_floor_log_r:
                    hard_idx = 1 + int(np.argmax(non_floor_log_r))
                # If only floor, hard_idx stays 0 but we skip the NIW update below
                # (floor is frozen)
                if hard_idx == 0:
                    # Only floor exists for this color — skip NIW update
                    action = int(actions[t])
                    true_cell = move(true_cell, action)
                    cp.predict_clamped_moments(ACTION_DELTA[action], Q)
                    continue

            # Update the non-floor component
            old_niw = components[obs_k][hard_idx][1]
            new_niw = old_niw.update_moments(post_mu, np.diag(post_Sigma_diag))
            components[obs_k][hard_idx] = (components[obs_k][hard_idx][0], new_niw)

            # ARM A: do NOT re-weight by counts — weights are managed explicitly
            # (floor at FLOOR_WEIGHT, tight components sum to TIGHT_WEIGHT).
            # We only update counts for non-floor components for diagnostics.
            if hard_idx > 0:
                counts[obs_k][hard_idx] += 1
            # No weight re-normalization from counts for Arm A in phase 2.

        else:
            # Phase 1 (single component) or single-component color: standard update
            old_niw = components[obs_k][hard_idx][1]
            new_niw = old_niw.update_moments(post_mu, np.diag(post_Sigma_diag))
            components[obs_k][hard_idx] = (components[obs_k][hard_idx][0], new_niw)

            if not floor_installed:
                # Phase 1: update weights from counts as usual
                counts[obs_k][hard_idx] += 1
                total_k = sum(counts[obs_k])
                components[obs_k] = [(counts[obs_k][j] / total_k, niw)
                                     for j, (_, niw) in enumerate(components[obs_k])]

        action = int(actions[t])
        true_cell = move(true_cell, action)
        cp.predict_clamped_moments(ACTION_DELTA[action], Q)

    # Handle in-flight probation
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

    final_comps_per_color = [len(components[k]) for k in range(N_COLORS)]

    p1_final500_loc_median = float(np.median(phase1_loc_errors[T_PHASE1 - 500:]))
    p2_final500_loc_median = float(np.median(phase2_loc_errors[T_PHASE2 - 500:]))

    phase1_plateau = float(np.mean(phase1_surprise_vals[-P1_PLATEAU_WINDOW:]))
    final_surprise = float(np.mean(phase2_surprise_vals[-FINAL_EVAL_WINDOW:]))
    drop = phase1_plateau - final_surprise

    decided_records = [r for r in attempt_records if r.get("kept") is not None]
    kept_records = [r for r in decided_records if r.get("kept") is True]

    accepted_deltas_list = [r["delta"] for r in kept_records if not math.isnan(r.get("delta", float("nan")))]
    mean_accepted_delta = float(np.mean(accepted_deltas_list)) if accepted_deltas_list else float("nan")

    total_attempted = len(decided_records)
    total_accepted = len(kept_records)
    accepted_frac = total_accepted / total_attempted if total_attempted > 0 else float("nan")

    # Arm A specific: mean attempt delta (accepted + rejected) for no-surge check
    all_deltas = [r["delta"] for r in decided_records if not math.isnan(r.get("delta", float("nan")))]
    mean_attempt_delta = float(np.mean(all_deltas)) if all_deltas else float("nan")

    return {
        "arm": "floor",
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
        "mean_attempt_delta": mean_attempt_delta,  # Arm A specific: all attempts
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("Exp 153 — M3f: BACKGROUND-FLOOR (Arm A) + NORMALIZED-DENSITY (Arm B)")
    print("=" * 80)
    print()

    cmaps = [build_cmap(s) for s in LAYOUT_SEEDS]

    all_rows_json = []

    # Per-layout verdicts per arm
    layout_verdicts_a: list[dict] = []
    layout_verdicts_b: list[dict] = []

    for layout_idx, (layout_seed, cmap) in enumerate(zip(LAYOUT_SEEDS, cmaps)):
        print()
        print(f"{'=' * 80}")
        print(f"LAYOUT {layout_idx + 1}  (rng({layout_seed}))")
        print(f"{'=' * 80}")
        if layout_idx == 0:
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

        results_a = []
        results_b = []
        tab_results = []

        # --- ARM A ---
        print(f"  --- ARM A: background-floor (unnormalized convention) ---")
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
              f"{'MnAllD':>8}  "
              f"{'TwinF500':>8}")
        print("-" * 125)

        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed)
            actions = rng.integers(0, 4, size=T_TOTAL)
            run_rng_a = np.random.default_rng(2000 + seed)
            run_rng_b = np.random.default_rng(2000 + seed)

            tab = run_tabular(actions, cmap)
            cont_a = run_arm_a_floor(
                actions, cmap, run_rng_a,
                seed=seed, layout_idx=layout_idx + 1,
                print_k_debug=(seed == 0 and layout_idx == 0),
            )

            tab_results.append(tab)
            results_a.append(cont_a)

            comps_str = str(cont_a["final_comps_per_color"])
            acc_frac = cont_a["accepted_frac"]
            acc_str = f"{acc_frac * 100:.0f}%" if not math.isnan(acc_frac) else " N/A"
            mn_delta = cont_a["mean_accepted_delta"]
            mn_str = f"{mn_delta:+.4f}" if not math.isnan(mn_delta) else "    N/A"
            mn_all_delta = cont_a["mean_attempt_delta"]
            mn_all_str = f"{mn_all_delta:+.4f}" if not math.isnan(mn_all_delta) else "    N/A"
            print(f"{seed:>4}  "
                  f"{cont_a['plateau']:>7.4f}  "
                  f"{cont_a['final_surprise']:>7.4f}  "
                  f"{cont_a['drop']:>6.3f}  "
                  f"{cont_a['phase2_final_ceiling_events']:>7d}  "
                  f"{comps_str:>12}  "
                  f"{cont_a['jumps_kept']:>4d}  "
                  f"{cont_a['jumps_reverted']:>4d}  "
                  f"{cont_a['p2_final500_loc_median']:>6.4f}  "
                  f"{acc_str:>7}  "
                  f"{mn_str:>8}  "
                  f"{mn_all_str:>8}  "
                  f"{tab['final_500_map_frac']:>8.4f}")

            all_rows_json.append({
                "exp": 153,
                "rung": "M3f",
                "arm": "floor",
                "layout_seed": layout_seed,
                "layout_idx": layout_idx + 1,
                "seed": seed,
                "plateau": cont_a["plateau"],
                "final_surprise": cont_a["final_surprise"],
                "drop": cont_a["drop"],
                "p1_ceiling_events": cont_a["p1_ceiling_events"],
                "phase2_final_ceiling_events": cont_a["phase2_final_ceiling_events"],
                "final_comps_per_color": cont_a["final_comps_per_color"],
                "jumps_kept": cont_a["jumps_kept"],
                "jumps_reverted": cont_a["jumps_reverted"],
                "p1_final500_loc_median": cont_a["p1_final500_loc_median"],
                "p2_final500_loc_median": cont_a["p2_final500_loc_median"],
                "accepted_frac": (
                    cont_a["accepted_frac"] if not math.isnan(cont_a["accepted_frac"]) else None
                ),
                "mean_accepted_delta": (
                    cont_a["mean_accepted_delta"]
                    if not math.isnan(cont_a["mean_accepted_delta"]) else None
                ),
                "mean_attempt_delta": (
                    cont_a["mean_attempt_delta"]
                    if not math.isnan(cont_a["mean_attempt_delta"]) else None
                ),
                "total_attempted": cont_a["total_attempted"],
                "total_accepted": cont_a["total_accepted"],
                "twin_final500_map_frac": tab["final_500_map_frac"],
            })

        # --- ARM B ---
        print()
        print(f"  --- ARM B: normalized-density batch-jump ---")
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
            run_rng_b = np.random.default_rng(2000 + seed)

            tab = tab_results[seed]  # reuse tab from Arm A loop (same actions/cmap)
            cont_b = run_arm_b_normalized(
                actions, cmap, run_rng_b,
                seed=seed, layout_idx=layout_idx + 1,
                print_k_debug=False,
            )

            results_b.append(cont_b)

            comps_str = str(cont_b["final_comps_per_color"])
            acc_frac = cont_b["accepted_frac"]
            acc_str = f"{acc_frac * 100:.0f}%" if not math.isnan(acc_frac) else " N/A"
            mn_delta = cont_b["mean_accepted_delta"]
            mn_str = f"{mn_delta:+.4f}" if not math.isnan(mn_delta) else "    N/A"
            print(f"{seed:>4}  "
                  f"{cont_b['plateau']:>7.4f}  "
                  f"{cont_b['final_surprise']:>7.4f}  "
                  f"{cont_b['drop']:>6.3f}  "
                  f"{cont_b['phase2_final_ceiling_events']:>7d}  "
                  f"{comps_str:>12}  "
                  f"{cont_b['jumps_kept']:>4d}  "
                  f"{cont_b['jumps_reverted']:>4d}  "
                  f"{cont_b['p2_final500_loc_median']:>6.4f}  "
                  f"{acc_str:>7}  "
                  f"{mn_str:>8}  "
                  f"{tab['final_500_map_frac']:>8.4f}")

            all_rows_json.append({
                "exp": 153,
                "rung": "M3f",
                "arm": "normalized",
                "layout_seed": layout_seed,
                "layout_idx": layout_idx + 1,
                "seed": seed,
                "plateau": cont_b["plateau"],
                "final_surprise": cont_b["final_surprise"],
                "drop": cont_b["drop"],
                "p1_ceiling_events": cont_b["p1_ceiling_events"],
                "phase2_final_ceiling_events": cont_b["phase2_final_ceiling_events"],
                "final_comps_per_color": cont_b["final_comps_per_color"],
                "jumps_kept": cont_b["jumps_kept"],
                "jumps_reverted": cont_b["jumps_reverted"],
                "p1_final500_loc_median": cont_b["p1_final500_loc_median"],
                "p2_final500_loc_median": cont_b["p2_final500_loc_median"],
                "accepted_frac": (
                    cont_b["accepted_frac"] if not math.isnan(cont_b["accepted_frac"]) else None
                ),
                "mean_accepted_delta": (
                    cont_b["mean_accepted_delta"]
                    if not math.isnan(cont_b["mean_accepted_delta"]) else None
                ),
                "total_attempted": cont_b["total_attempted"],
                "total_accepted": cont_b["total_accepted"],
                "twin_final500_map_frac": tab["final_500_map_frac"],
            })

        # ---------------------------------------------------------------
        # Per-attempt detail summary (Layout 1 full, others summarized)
        # ---------------------------------------------------------------
        if layout_idx == 0:
            for arm_label, arm_results in [("A floor", results_a), ("B normalized", results_b)]:
                print()
                print(f"  Per-jump attempt detail (Layout 1, Arm {arm_label}):")
                for seed_idx, cont in enumerate(arm_results):
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
        else:
            # Summarized for layouts 2 and 3
            for arm_label, arm_results in [("A floor", results_a), ("B normalized", results_b)]:
                total_attempts = sum(len(c["attempt_records"]) for c in arm_results)
                total_kept = sum(c["jumps_kept"] for c in arm_results)
                print(f"  Arm {arm_label} Layout {layout_idx + 1}: "
                      f"{total_attempts} attempts, {total_kept} kept")

        # ---------------------------------------------------------------
        # Tallies — ARM A
        # ---------------------------------------------------------------
        print()
        print(f"  TALLIES — Layout {layout_idx + 1} — ARM A (floor)")
        print()

        # P2 for Arm A: no-surge check (mean attempt delta in (-0.3, +0.3))
        # and drop arm fails (< 6/8 seeds reach 0.4 drop)
        p2a_drop_per_seed = [cont["drop"] >= 0.4 for cont in results_a]
        p2a_drop_count = sum(p2a_drop_per_seed)
        p2a_drop_fails = p2a_drop_count < 6  # "fails" = passes the P2 prediction

        all_decided_a = []
        for cont in results_a:
            for rec in cont["attempt_records"]:
                if rec.get("kept") is not None:
                    all_decided_a.append(rec)
        all_deltas_a = [r["delta"] for r in all_decided_a if not math.isnan(r.get("delta", float("nan")))]
        mean_attempt_delta_a = float(np.mean(all_deltas_a)) if all_deltas_a else float("nan")
        no_surge = (
            (-0.3 < mean_attempt_delta_a < 0.3)
            if not math.isnan(mean_attempt_delta_a) else False
        )

        p2a_holds = no_surge and p2a_drop_fails

        # P3 Arm A
        p3a_per_seed = [cont["p2_final500_loc_median"] <= 0.5 for cont in results_a]
        p3a_count = sum(p3a_per_seed)
        p3a_holds = p3a_count >= 7

        mn_attempt_str = (f"{mean_attempt_delta_a:+.4f}"
                          if not math.isnan(mean_attempt_delta_a) else "N/A")
        print(f"  P2 Arm A — no-surge check:")
        print(f"    Mean attempt delta (all): {mn_attempt_str}  "
              f"no-surge=(-0.3,+0.3): {'YES' if no_surge else 'NO'}")
        print(f"    Drop >= 0.4 count: {p2a_drop_count}/8  "
              f"(P2 predicts < 6/8): {'PASS P2' if p2a_drop_fails else 'FAIL P2 (A passed drop arm)'}")
        print(f"  P2 Arm A layout {layout_idx + 1}: {'PASS' if p2a_holds else 'FAIL'}")

        print(f"\n  P3 Arm A (loc_med <= 0.5, >= 7/8):")
        print(f"    {p3a_count}/8  {'PASS' if p3a_holds else 'FAIL (HALT)'}")
        for s_idx, cont in enumerate(results_a):
            print(f"    seed={SEEDS[s_idx]}: loc_med={cont['p2_final500_loc_median']:.4f}  "
                  f"drop={cont['drop']:.3f}  "
                  f"{'pass' if p3a_per_seed[s_idx] else 'FAIL'}")

        # ---------------------------------------------------------------
        # Tallies — ARM B
        # ---------------------------------------------------------------
        print()
        print(f"  TALLIES — Layout {layout_idx + 1} — ARM B (normalized)")
        print()

        # P1 for Arm B: the alarm-answered bar
        p1b_drop_per_seed = [cont["drop"] >= 0.4 for cont in results_b]
        p1b_drop_count = sum(p1b_drop_per_seed)
        p1b_drop_holds = p1b_drop_count >= 6

        p1b_quiet_per_seed = [cont["phase2_final_ceiling_events"] == 0 for cont in results_b]
        p1b_quiet_count = sum(p1b_quiet_per_seed)
        p1b_quiet_holds = p1b_quiet_count >= 6

        p1b_comps_per_seed = []
        for cont in results_b:
            comps = cont["final_comps_per_color"]
            colors_grown = sum(1 for c in comps if c >= 3)
            p1b_comps_per_seed.append(colors_grown >= 3)
        p1b_comps_count = sum(p1b_comps_per_seed)
        p1b_comps_holds = p1b_comps_count >= 6

        p1b_holds_layout = p1b_drop_holds and p1b_quiet_holds and p1b_comps_holds

        print(f"  P1 Arm B — alarm answered:")
        print(f"    Arm1 drop >= 0.4 nats: {p1b_drop_count}/8  "
              f"(need >= 6/8)  {'PASS' if p1b_drop_holds else 'FAIL'}")
        print(f"    Arm2 zero final ceil: {p1b_quiet_count}/8  "
              f"(need >= 6/8)  {'PASS' if p1b_quiet_holds else 'FAIL'}")
        print(f"    Arm3 comps>=3 for 3/4 colors: {p1b_comps_count}/8  "
              f"(need >= 6/8)  {'PASS' if p1b_comps_holds else 'FAIL'}")
        print(f"  P1 Arm B layout {layout_idx + 1}: {'PASS' if p1b_holds_layout else 'FAIL'}")
        for s_idx, cont in enumerate(results_b):
            comps = cont["final_comps_per_color"]
            colors_grown = sum(1 for c in comps if c >= 3)
            print(f"    seed={SEEDS[s_idx]}: plateau={cont['plateau']:.4f}  "
                  f"final={cont['final_surprise']:.4f}  drop={cont['drop']:.3f}  "
                  f"finCeil={cont['phase2_final_ceiling_events']}  "
                  f"comps={comps}({colors_grown}/4>=3)  "
                  f"drop={'pass' if p1b_drop_per_seed[s_idx] else 'FAIL'}  "
                  f"quiet={'pass' if p1b_quiet_per_seed[s_idx] else 'FAIL'}  "
                  f"comps={'pass' if p1b_comps_per_seed[s_idx] else 'FAIL'}")

        # P3 Arm B
        p3b_per_seed = [cont["p2_final500_loc_median"] <= 0.5 for cont in results_b]
        p3b_count = sum(p3b_per_seed)
        p3b_holds = p3b_count >= 7

        print(f"\n  P3 Arm B (loc_med <= 0.5, >= 7/8):")
        print(f"    {p3b_count}/8  {'PASS' if p3b_holds else 'FAIL (HALT)'}")
        for s_idx, cont in enumerate(results_b):
            print(f"    seed={SEEDS[s_idx]}: loc_med={cont['p2_final500_loc_median']:.4f}  "
                  f"drop={cont['drop']:.3f}  "
                  f"{'pass' if p3b_per_seed[s_idx] else 'FAIL'}")

        layout_verdicts_a.append({
            "layout_idx": layout_idx + 1,
            "layout_seed": layout_seed,
            "p2a_holds": p2a_holds,
            "no_surge": no_surge,
            "p2a_drop_fails": p2a_drop_fails,
            "p2a_drop_count": p2a_drop_count,
            "mean_attempt_delta_a": mean_attempt_delta_a,
            "p3a_holds": p3a_holds,
            "p3a_count": p3a_count,
        })
        layout_verdicts_b.append({
            "layout_idx": layout_idx + 1,
            "layout_seed": layout_seed,
            "p1b_holds": p1b_holds_layout,
            "p1b_drop_count": p1b_drop_count,
            "p1b_quiet_count": p1b_quiet_count,
            "p1b_comps_count": p1b_comps_count,
            "p1b_drop_fails": 8 - p1b_drop_count,
            "p3b_holds": p3b_holds,
            "p3b_count": p3b_count,
        })

        # JSON summary rows
        all_rows_json.append({
            "exp": 153,
            "rung": "M3f",
            "arm": "floor",
            "layout_seed": layout_seed,
            "layout_idx": layout_idx + 1,
            "seed": -1,
            "summary": True,
            "p2a_holds": p2a_holds,
            "no_surge": no_surge,
            "p2a_drop_fails": p2a_drop_fails,
            "p2a_drop_count": p2a_drop_count,
            "mean_attempt_delta_a": (
                mean_attempt_delta_a if not math.isnan(mean_attempt_delta_a) else None
            ),
            "p3a_holds": p3a_holds,
            "p3a_count": p3a_count,
        })
        all_rows_json.append({
            "exp": 153,
            "rung": "M3f",
            "arm": "normalized",
            "layout_seed": layout_seed,
            "layout_idx": layout_idx + 1,
            "seed": -1,
            "summary": True,
            "p1b_holds": p1b_holds_layout,
            "p1b_drop_count": p1b_drop_count,
            "p1b_quiet_count": p1b_quiet_count,
            "p1b_comps_count": p1b_comps_count,
            "p3b_holds": p3b_holds,
            "p3b_count": p3b_count,
        })

    # -----------------------------------------------------------------------
    # Global conjunct evaluation
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("GLOBAL CONJUNCT TALLIES (across all 3 layouts)")
    print("=" * 80)

    # P1 (Arm B): >= 6/8 seeds in >= 2/3 layouts
    p1b_layouts_passing = sum(1 for v in layout_verdicts_b if v["p1b_holds"])
    p1b_global = p1b_layouts_passing >= 2

    print(f"\nP1 Arm B — normalized arm alarm bar (>= 6/8 in >= 2/3 layouts):")
    for v in layout_verdicts_b:
        print(f"  Layout {v['layout_idx']} (rng({v['layout_seed']})): "
              f"drop={v['p1b_drop_count']}/8  quiet={v['p1b_quiet_count']}/8  "
              f"comps={v['p1b_comps_count']}/8  "
              f"{'PASS' if v['p1b_holds'] else 'FAIL'}")
    print(f"  Layouts passing: {p1b_layouts_passing}/3  (need >= 2)  "
          f"P1 Arm B GLOBAL: {'PASS' if p1b_global else 'FAIL'}")

    # P2 (Arm A): no-surge AND drop fails — both all 3 layouts
    p2a_global_no_surge = all(v["no_surge"] for v in layout_verdicts_a)
    p2a_global_drop_fails = all(v["p2a_drop_fails"] for v in layout_verdicts_a)
    p2a_global = p2a_global_no_surge and p2a_global_drop_fails

    print(f"\nP2 Arm A — no-surge AND drop-fails (all 3 layouts):")
    for v in layout_verdicts_a:
        mn_str = (f"{v['mean_attempt_delta_a']:+.4f}"
                  if not math.isnan(v["mean_attempt_delta_a"]) else "N/A")
        print(f"  Layout {v['layout_idx']}: mean_all_delta={mn_str}  "
              f"no_surge={'YES' if v['no_surge'] else 'NO'}  "
              f"drop_count={v['p2a_drop_count']}/8  "
              f"drop_fails={'YES' if v['p2a_drop_fails'] else 'NO (A passed drop arm)'}  "
              f"P2A={'PASS' if v['p2a_holds'] else 'FAIL'}")
    print(f"  P2 Arm A GLOBAL: {'PASS' if p2a_global else 'FAIL'}")

    # P3: >= 7/8 seeds in all 3 layouts, BOTH arms
    p3a_global = all(v["p3a_holds"] for v in layout_verdicts_a)
    p3b_global = all(v["p3b_holds"] for v in layout_verdicts_b)
    p3_global = p3a_global and p3b_global

    print(f"\nP3 no regression (>= 7/8 seeds, all 3 layouts, BOTH arms):")
    print(f"  Arm A:")
    for v in layout_verdicts_a:
        print(f"    Layout {v['layout_idx']}: {v['p3a_count']}/8  "
              f"{'PASS' if v['p3a_holds'] else 'FAIL'}")
    print(f"  Arm B:")
    for v in layout_verdicts_b:
        print(f"    Layout {v['layout_idx']}: {v['p3b_count']}/8  "
              f"{'PASS' if v['p3b_holds'] else 'FAIL'}")
    print(f"  P3 Arm A GLOBAL: {'PASS' if p3a_global else 'FAIL (HALT)'}")
    print(f"  P3 Arm B GLOBAL: {'PASS' if p3b_global else 'FAIL (HALT)'}")
    print(f"  P3 GLOBAL (both): {'PASS' if p3_global else 'FAIL (HALT)'}")

    # -----------------------------------------------------------------------
    # Named outcome determination
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("NAMED OUTCOME + VERDICT")
    print("=" * 80)
    print()

    # P3 regression is a hard halt
    halt_regression = not p3_global

    if halt_regression:
        verdict = "NEGATIVE"
        print(f"VERDICT: {verdict}")
        print("HALT — P3 REGRESSION DETECTED")
        if not p3a_global:
            print("  Arm A P3 failed.")
        if not p3b_global:
            print("  Arm B P3 failed.")
    elif p1b_global and p2a_global:
        # Named outcome: B passes & A insufficient -> wall is RE-BOUNDED to convention
        verdict = "POSITIVE"
        print(f"VERDICT: {verdict}")
        print(
            "Named outcome: B passes & A insufficient -> the growth wall is RE-BOUNDED "
            "to the unnormalized-footprint convention (major finding; the wall doc gets "
            "amended). Arm B (normalized predictive) passed the alarm-answered bar "
            f"({p1b_layouts_passing}/3 layouts). Arm A (floor, standard convention) "
            "showed the harmless-but-insufficient signature (no surge, drop arm fails). "
            "P3 (no regression) holds for both arms. "
            "Convention amendment warranted."
        )
    elif p1b_global and not p2a_global:
        # Arm B passes but Arm A also passed its drop arm (unexpected)
        verdict = "POSITIVE"
        print(f"VERDICT: {verdict}")
        print(
            f"Arm B (normalized) PASSED ({p1b_layouts_passing}/3 layouts). "
            f"Arm A: P2 check = {'PASS' if p2a_global else 'FAIL'} "
            f"(no_surge={'PASS' if p2a_global_no_surge else 'FAIL'}, "
            f"drop_fails={'PASS' if p2a_global_drop_fails else 'FAIL'}) — "
            f"Arm A may have also passed its drop arm (log; growth unlocked). "
            "P3 holds."
        )
    elif not p1b_global and not p2a_global:
        # Both arms fail their bars -> wall stands, convention hypothesis spent
        verdict = "NEGATIVE"
        print(f"VERDICT: {verdict}")
        print(
            "Named outcome: BOTH arms fail their bars -> the wall stands with both "
            "authorized cracks AND the convention hypothesis spent -> MIGRATION HALT "
            "consult required. "
            f"Arm B: {p1b_layouts_passing}/3 layouts (need >= 2). "
            f"Arm A: P2_global={'PASS' if p2a_global else 'FAIL'}. "
            "P3 holds."
        )
    elif not p1b_global and p2a_global:
        # Arm B fails, Arm A is insufficient (the expected neg result)
        verdict = "NEGATIVE"
        print(f"VERDICT: {verdict}")
        print(
            f"Arm B (normalized) FAILED ({p1b_layouts_passing}/3 layouts, need >= 2). "
            "Arm A showed harmless-but-insufficient signature. "
            "Convention hypothesis not confirmed by Arm B; batch-jump wall stands "
            "with both authorized cracks spent. MIGRATION HALT consult."
        )
    elif p2a_global and not p1b_global:
        verdict = "NEGATIVE"
        print(f"VERDICT: {verdict}")
        print(
            f"Arm B normalized failed ({p1b_layouts_passing}/3 layouts). "
            "Arm A floor: harmless-but-insufficient confirmed. "
            "MIGRATION HALT consult."
        )
    else:
        verdict = "MIXED"
        print(f"VERDICT: {verdict}")
        print(
            f"Partial result: P1_B={'PASS' if p1b_global else 'FAIL'} "
            f"({p1b_layouts_passing}/3 layouts), "
            f"P2_A={'PASS' if p2a_global else 'FAIL'}, "
            f"P3={'PASS' if p3_global else 'FAIL'}. "
            "Inspect tallies above."
        )

    print("=" * 80)

    # -----------------------------------------------------------------------
    # JSON output
    # -----------------------------------------------------------------------
    all_rows_json.append({
        "exp": 153,
        "rung": "M3f",
        "seed": -2,
        "global_summary": True,
        "p1b_global": p1b_global,
        "p1b_layouts_passing": p1b_layouts_passing,
        "p2a_global": p2a_global,
        "p2a_global_no_surge": p2a_global_no_surge,
        "p2a_global_drop_fails": p2a_global_drop_fails,
        "p3a_global": p3a_global,
        "p3b_global": p3b_global,
        "p3_global": p3_global,
        "halt_regression": halt_regression,
        "verdict": verdict,
    })

    out_path = Path(__file__).parent / "outputs" / "exp153_rows.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for row in all_rows_json:
            fh.write(json.dumps(row) + "\n")
    print(f"\nJSON rows written to {out_path}")


if __name__ == "__main__":
    main()
