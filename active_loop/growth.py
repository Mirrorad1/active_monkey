"""
active_loop/growth.py — extracted growth machinery for the continuous-creature lab.

Lifted verbatim (arithmetic preserved) from the validated experiment scripts:
  - Exp 145 (exp145_m3c_live_probation.py): per-color + global surprise-ceiling
    detection, LiveProbation, round-robin alarm scheduling, burnin_em_color, and
    the unnormalized-footprint mixture predictive convention.
  - Exp 152/153 (exp153_m3f_floor_and_normalized.py): normalized-density convention
    for the mixture predictive, batch-jump EM, penalized K-selection, and
    color_replay_nll. Both conventions are exposed via an explicit ``convention=``
    argument — never silently.

ADDITIVE MODULE. No existing experiment script is modified.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from typing import Literal

import numpy as np

from active_loop.continuous import NIW

# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------

ConventionT = Literal["unnormalized", "normalized"]

# ---------------------------------------------------------------------------
# Constants — with provenance comments
# ---------------------------------------------------------------------------

# --- Ceiling / surprise detector ---
# Validated in Exp 145 (rung M3c, 4x4 aliased worlds, 4 colors x 4 cells,
# phases 1+2 at 2000+8000 steps). Copied from creature.py line annotations
# preserved by all M3* experiments.
CEILING_MEAN_THRESH: float = 0.7    # mean(buf) >= this triggers ceiling; Exp 145 line 78
CEILING_SLOPE_THRESH: float = 5e-4  # |slope| < this (plateau condition); Exp 145 line 79
SURPRISE_WINDOW: int = 200          # rolling window for global ceiling; Exp 145 line 80

# --- Per-color alarm ---
# Validated in Exp 145 (rung M3c). The color-level detector watches a shorter
# window than the global detector so it responds before the global fires.
COLOR_SURPRISE_WINDOW: int = 50     # per-color deque length for alarm check; Exp 145 line 131
ALARM_THRESH: float = 0.7           # per-color mean >= this = ALARMED; Exp 145 line 132
PRE_SPAWN_WINDOW: int = 100         # per-color deque for pre-spawn baseline; Exp 145 line 133

# --- Probation ---
# Validated in Exp 145. "Power note": with ~uniform color visitation, 400 live
# steps yield ~100 observations of the probated color (4 colors), matching the
# 50-obs alarm window with 2x observations for stability.
PROBATION_STEPS: int = 400          # live-step window for keep/revert; Exp 145 line 134
KEEP_MARGIN: float = 0.1            # keep iff probation_mean <= pre_spawn_mean - KEEP_MARGIN; Exp 145

# --- Spawn scheduling ---
SPAWN_INTERVAL: int = 200           # check for spawn every N phase-2 steps; Exp 145 line 122
SPAWN_BUDGET: int = 4               # max components per color; Exp 145 line 121
JUMP_COOLDOWN: int = 1200           # cooldown steps between jump attempts (Exp 152/153)

# --- Burn-in EM ---
# Validated in Exp 145 (copied from Exp 144). These values were fixed throughout
# all M3* designs.
BURN_IN_ITERS: int = 10             # EM iterations during burn-in; Exp 145 line 127
BURN_IN_COV_FLOOR: float = 1e-4     # diagonal floor for covariance; Exp 145 line 128

# --- Batch-jump EM (Exp 152/153) ---
EM_ITERS: int = 10                  # EM iters for batch-jump; Exp 153 line 143
EM_COV_FLOOR: float = 1e-4          # covariance floor; Exp 153 line 144
K_CANDIDATES: list[int] = [2, 3, 4] # component counts tried by K-selection; Exp 153 line 145
K_PENALTY: float = 0.05             # BIC-style penalty per extra component; Exp 153 line 146
MIN_REPLAY_PAIRS: int = 30          # minimum pairs before a jump is attempted; Exp 153 line 147

# Used internally for normalized convention
_LOG_2PI: float = math.log(2.0 * math.pi)

# Space dimensionality (fixed throughout the lab)
_D: int = 2

# ---------------------------------------------------------------------------
# Low-level Gaussian utilities
# ---------------------------------------------------------------------------


def _diag_log_integral(
    mu_p: np.ndarray,
    Sigma_p_diag: np.ndarray,
    m_k: np.ndarray,
    ESigma_k_diag: np.ndarray,
) -> float:
    """Log integral of N(s; m_k, ESigma_k) * N(s; mu_p, Sigma_p) over R^2.

    UNNORMALIZED-FOOTPRINT convention: no 1/(2*pi*|C|^{1/2}) term.
    Formula: log I = 0.5*(logdet(Sk) - logdet(Sk+Sp)) - 0.5*maha
    where Sk = diag(ESigma_k_diag), Sp = diag(Sigma_p_diag).
    All operations scalar; no linalg.

    Validated: Exp 145 (lines 196-218), Exp 153 (lines 214-236).
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
    mu_p: np.ndarray,
    Sigma_p_diag: np.ndarray,
    m_k: np.ndarray,
    ESigma_k_diag: np.ndarray,
) -> float:
    """Log N(mu_p; m_k, Cov_k + Sigma_p) — NORMALIZED Gaussian density.

    NORMALIZED-DENSITY convention: includes -D/2*log(2*pi) - 0.5*logdet(C).
    Sharp (small Cov_k) components have smaller |C| and are therefore louder.
    This is the true Gaussian-mixture marginal density term.

    Validated: Exp 153 Arm B (lines 239-258); introduced to diagnose the
    dilution hypothesis (Exp 152 autopsy).
    """
    c0 = ESigma_k_diag[0] + Sigma_p_diag[0]
    c1 = ESigma_k_diag[1] + Sigma_p_diag[1]
    logdet_C = math.log(c0) + math.log(c1)
    d0 = mu_p[0] - m_k[0]
    d1 = mu_p[1] - m_k[1]
    maha = d0 * d0 / c0 + d1 * d1 / c1
    return -_D * 0.5 * _LOG_2PI - 0.5 * logdet_C - 0.5 * maha


def _diag_gaussian_product(
    mu_a: np.ndarray,
    Sigma_a_diag: np.ndarray,
    mu_b: np.ndarray,
    Sigma_b_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Product of two diagonal 2-D Gaussians -> (mu_out, Sigma_diag_out).

    Precision-weighted mean; per-axis scalar arithmetic.
    Validated: Exp 145 (lines 221-242), Exp 153 (lines 261-279).
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
# Mixture predictive log-probabilities — two conventions
# ---------------------------------------------------------------------------


def mixture_predictive_logprobs(
    mu_p: np.ndarray,
    Sigma_p_diag: np.ndarray,
    components: list[list[tuple[float, NIW]]],
    *,
    convention: ConventionT,
) -> np.ndarray:
    """Compute normalized log p(color k) under the mixture predictive.

    Parameters
    ----------
    mu_p, Sigma_p_diag:
        Current place belief mean and diagonal covariance.
    components:
        components[k] = [(w_kj, NIW_kj), ...] for color k.
    convention:
        ``"unnormalized"`` — uses the footprint integral (historical default,
        rung-1 convention, Exp 145). Tight components do NOT get a density
        reward; K tight components speak at ~1/K the volume of one broad
        component (the dilution mechanism diagnosed in Exp 152).

        ``"normalized"`` — uses the full Gaussian density (Exp 153 Arm B).
        Sharp components are louder; this is the true Gaussian-mixture
        marginal over spatial position. Introduced to test the dilution
        hypothesis.

    Returns
    -------
    log_probs : np.ndarray, shape (N_COLORS,)
        Normalized log-probabilities (logsumexp == 0).

    Validated: unnormalized in Exp 145 (lines 250-278); normalized in
    Exp 153 Arm B (lines 317-350).
    """
    if convention not in ("unnormalized", "normalized"):
        raise ValueError(
            f"convention must be 'unnormalized' or 'normalized', got {convention!r}"
        )

    n_colors = len(components)
    log_unnorm = np.empty(n_colors, dtype=float)

    if convention == "unnormalized":
        _log_eval = _diag_log_integral
    else:
        _log_eval = _diag_log_gaussian_eval

    for k in range(n_colors):
        comps = components[k]
        log_terms = []
        for w_kj, niw_kj in comps:
            m_kj = niw_kj.m
            ESigma_kj = niw_kj.expected_Sigma()
            ESigma_kj_diag = np.maximum(
                np.array([ESigma_kj[0, 0], ESigma_kj[1, 1]]), 1e-9
            )
            log_val = _log_eval(mu_p, Sigma_p_diag, m_kj, ESigma_kj_diag)
            log_terms.append(math.log(max(w_kj, 1e-300)) + log_val)
        log_unnorm[k] = float(np.logaddexp.reduce(np.array(log_terms)))

    log_sum = float(np.logaddexp.reduce(log_unnorm))
    return log_unnorm - log_sum


def mixture_emission_moments(
    mu_p: np.ndarray,
    Sigma_p_diag: np.ndarray,
    k: int,
    components: list[list[tuple[float, NIW]]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Moment-match the mixture emission for observed color k to a single Gaussian.

    The place update always uses the UNNORMALIZED integral for responsibilities
    regardless of which convention is used for the predictive. This is
    intentional: the update is conjugate (Exp 153 design note, lines 63-76).

    Returns (mu_mix, Sigma_mix_diag, hard_idx).
    hard_idx: index of the highest-responsibility component for NIW hard update.

    Declared approximation: moment-matching to a diagonal Gaussian.
    Validated: Exp 145 (lines 281-334), Exp 153 (lines 357-405).
    """
    comps = components[k]
    n = len(comps)
    log_r = np.empty(n, dtype=float)
    prod_mus: list[np.ndarray] = []
    prod_sigs: list[np.ndarray] = []

    for j, (w_kj, niw_kj) in enumerate(comps):
        m_kj = niw_kj.m
        ESigma_kj = niw_kj.expected_Sigma()
        ESigma_kj_diag = np.maximum(
            np.array([ESigma_kj[0, 0], ESigma_kj[1, 1]]), 1e-9
        )
        logI = _diag_log_integral(mu_p, Sigma_p_diag, m_kj, ESigma_kj_diag)
        log_r[j] = math.log(max(w_kj, 1e-300)) + logI

        mu_j, sig_j_diag = _diag_gaussian_product(
            mu_p, Sigma_p_diag, m_kj, ESigma_kj_diag
        )
        prod_mus.append(mu_j)
        prod_sigs.append(sig_j_diag)

    log_r_sum = float(np.logaddexp.reduce(log_r))
    r = np.exp(log_r - log_r_sum)  # shape (n,)

    mu_mix = np.zeros(2)
    for j in range(n):
        mu_mix += r[j] * prod_mus[j]

    Sigma_mix_diag = np.zeros(2)
    for i in range(2):
        e2 = sum(
            r[j] * (prod_sigs[j][i] + prod_mus[j][i] ** 2) for j in range(n)
        )
        Sigma_mix_diag[i] = e2 - mu_mix[i] ** 2

    Sigma_mix_diag = np.maximum(Sigma_mix_diag, 1e-9)
    hard_idx = int(np.argmax(r))
    return mu_mix, Sigma_mix_diag, hard_idx


# ---------------------------------------------------------------------------
# Surprise-ceiling detector
# ---------------------------------------------------------------------------


def check_ceiling(
    surprise_buf: deque,
    *,
    mean_thresh: float = CEILING_MEAN_THRESH,
    slope_thresh: float = CEILING_SLOPE_THRESH,
    window: int = SURPRISE_WINDOW,
) -> bool:
    """Return True when the mean/slope conjunction fires on ``surprise_buf``.

    Conditions (all must hold):
      1. len(buf) >= window (buffer is full)
      2. mean(buf) > mean_thresh
      3. |slope| < slope_thresh  (plateau — not still descending)

    This is the per-color OR global ceiling check; callers pass the appropriate
    deque (global: maxlen=SURPRISE_WINDOW; per-color: maxlen=COLOR_SURPRISE_WINDOW).
    For per-color alarm use the shorter window/threshold via the keyword args.

    Validated: Exp 145 (lines 341-356), Exp 153 (lines 412-422).
    """
    if len(surprise_buf) < window:
        return False
    arr = np.array(surprise_buf)
    mean_s = float(np.mean(arr))
    if mean_s <= mean_thresh:
        return False
    xs = np.arange(len(arr), dtype=float)
    slope = float(np.polyfit(xs, arr, 1)[0])
    return abs(slope) < slope_thresh


def check_color_alarm(
    color_surprise_buf: deque,
    *,
    thresh: float = ALARM_THRESH,
) -> bool:
    """Return True if the per-color alarm fires.

    Condition: len(buf) > 0 AND mean(buf) >= thresh.
    Note: the alarm is a simpler mean-only check (no slope); it detects that the
    color is currently high-surprise, not necessarily plateaued.

    Validated: Exp 145 (lines 755-760).
    """
    if len(color_surprise_buf) == 0:
        return False
    return float(np.mean(color_surprise_buf)) >= thresh


# ---------------------------------------------------------------------------
# Deep-copy utilities
# ---------------------------------------------------------------------------


def _copy_components(
    components: list[list[tuple[float, NIW]]],
) -> list[list[tuple[float, NIW]]]:
    """Deep copy the full components structure (all colors).

    Validated: Exp 145 (lines 381-388), Exp 153 (lines 460-466).
    """
    result = []
    for color_comps in components:
        result.append([(w, copy.deepcopy(niw)) for (w, niw) in color_comps])
    return result


def _copy_color_components(
    color_comps: list[tuple[float, NIW]],
) -> list[tuple[float, NIW]]:
    """Deep copy a single color's component list.

    Validated: Exp 145 (lines 391-395), Exp 153 (lines 469-472).
    """
    return [(w, copy.deepcopy(niw)) for (w, niw) in color_comps]


# ---------------------------------------------------------------------------
# Burn-in EM (Exp 145)
# ---------------------------------------------------------------------------


def burnin_em_color(
    color_comps: list[tuple[float, NIW]],
    color_replay: list[tuple[np.ndarray, np.ndarray]],
    n_iters: int = BURN_IN_ITERS,
) -> list[tuple[float, NIW]]:
    """Run burn-in EM on one color's component set using that color's replay pairs.

    Each replay pair ``(mean_i, Sigma_diag_i)`` represents posterior uncertainty:
    the true location is modeled as N(mean_i, diag(Sigma_diag_i)).

    E-step responsibilities:
        r_ij ∝ w_j * N(mean_i; m_j, Cov_j + diag(Sigma_i))
    M-step:
        w_j = sum_i r_ij / n
        m_j = precision-weighted mean
        Cov_j = weighted scatter + weighted mean of Sigma_i,
                floored at BURN_IN_COV_FLOOR * I on diagonal

    NIW write-back convention (declared):
        kappa_j = max(n_j, 0.5),  nu_j = kappa_j + 3
        m_j = weighted mean,  S_j = Cov_j * kappa_j

    Validated: Exp 145 (lines 406-519). All arithmetic copied verbatim.
    """
    n = len(color_replay)
    if n == 0 or len(color_comps) == 0:
        return color_comps

    n_comp = len(color_comps)

    weights = np.array([w for (w, _) in color_comps])
    weights = np.maximum(weights, 1e-300)
    weights /= weights.sum()

    comp_means = np.array([niw.m for (_, niw) in color_comps])  # (n_comp, 2)
    comp_covs = np.array([
        np.diag(niw.expected_Sigma())
        for (_, niw) in color_comps
    ])  # (n_comp, 2)
    comp_covs = np.maximum(comp_covs, BURN_IN_COV_FLOOR)

    obs_means = np.array([mu for (mu, _) in color_replay])    # (n, 2)
    obs_sigmas = np.array([sig for (_, sig) in color_replay]) # (n, 2)

    for _ in range(n_iters):
        # E-step
        log_r = np.empty((n, n_comp), dtype=float)
        for j in range(n_comp):
            mj = comp_means[j]
            cj = comp_covs[j]
            for i in range(n):
                eff_cov = np.maximum(cj + obs_sigmas[i], 1e-9)
                d0 = obs_means[i, 0] - mj[0]
                d1 = obs_means[i, 1] - mj[1]
                log_det = math.log(eff_cov[0]) + math.log(eff_cov[1])
                maha = d0 * d0 / eff_cov[0] + d1 * d1 / eff_cov[1]
                log_lik = -0.5 * (2 * math.log(2.0 * math.pi) + log_det + maha)
                log_r[i, j] = math.log(max(weights[j], 1e-300)) + log_lik

        log_r_max = log_r.max(axis=1, keepdims=True)
        r = np.exp(log_r - log_r_max)
        r_sum = r.sum(axis=1, keepdims=True)
        r_sum = np.where(r_sum == 0, 1.0, r_sum)
        r = r / r_sum  # (n, n_comp), rows sum to 1

        # M-step
        n_j = r.sum(axis=0)  # (n_comp,)
        n_j_safe = np.maximum(n_j, 1e-9)

        new_weights = n_j / n
        new_weights = np.maximum(new_weights, 1e-300)
        new_weights /= new_weights.sum()

        new_means = (r.T @ obs_means) / n_j_safe[:, None]

        new_covs = np.zeros((n_comp, 2), dtype=float)
        for j in range(n_comp):
            diff = obs_means - new_means[j]
            scatter = (r[:, j, None] * (diff ** 2 + obs_sigmas)).sum(axis=0)
            new_covs[j] = scatter / n_j_safe[j]
        new_covs = np.maximum(new_covs, BURN_IN_COV_FLOOR)

        weights = new_weights
        comp_means = new_means
        comp_covs = new_covs

    # NIW write-back
    result: list[tuple[float, NIW]] = []
    for j in range(n_comp):
        kappa_j = float(max(n_j[j], 0.5))
        nu_j = kappa_j + 3.0
        m_j = comp_means[j].copy()
        S_j = np.diag(comp_covs[j] * kappa_j)
        if nu_j < _D + 2:
            nu_j = float(_D + 2)
            kappa_j = nu_j - 3.0
        new_niw = NIW(m=m_j, kappa=kappa_j, nu=nu_j, S=S_j)
        result.append((float(weights[j]), new_niw))

    return result


# ---------------------------------------------------------------------------
# Batch-jump EM and helpers (Exp 152/153)
# ---------------------------------------------------------------------------


def _farthest_point_init(
    obs_means: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Farthest-point seeding for K initial cluster centers.

    Validated: Exp 153 (lines 483-499).
    """
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


def color_replay_nll(
    color_replay_pairs: list[tuple[np.ndarray, np.ndarray]],
    comp_means: np.ndarray,
    comp_covs: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Mean -log sum_j w_j N(mean_i; m_j, Cov_j + Sigma_i) over color replay pairs.

    Used for penalized K-selection in batch-jump EM.
    Validated: Exp 153 (lines 429-453).
    """
    n = len(color_replay_pairs)
    if n == 0:
        return float("inf")
    K = len(weights)
    total = 0.0
    for mu_i, Sigma_i_diag in color_replay_pairs:
        log_terms = np.empty(K, dtype=float)
        for j in range(K):
            eff_cov = np.maximum(comp_covs[j] + Sigma_i_diag, 1e-9)
            d0 = mu_i[0] - comp_means[j, 0]
            d1 = mu_i[1] - comp_means[j, 1]
            log_det = math.log(eff_cov[0]) + math.log(eff_cov[1])
            maha = d0 * d0 / eff_cov[0] + d1 * d1 / eff_cov[1]
            log_lik = -0.5 * (2.0 * math.log(2.0 * math.pi) + log_det + maha)
            log_terms[j] = math.log(max(weights[j], 1e-300)) + log_lik
        total += -float(np.logaddexp.reduce(log_terms))
    return total / n


def batch_jump_em(
    color_replay_pairs: list[tuple[np.ndarray, np.ndarray]],
    K: int,
    rng: np.random.Generator,
    n_iters: int = EM_ITERS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a K-component diagonal-Gaussian mixture on color replay pairs via EM.

    Returns (weights, comp_means, comp_covs) as numpy arrays.
    Seeding: farthest-point init on the obs_means.

    Validated: Exp 153 (lines 506-562).
    """
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


def em_result_to_components(
    weights: np.ndarray,
    comp_means: np.ndarray,
    comp_covs: np.ndarray,
    n_eff: np.ndarray,
) -> list[tuple[float, NIW]]:
    """Convert raw EM output arrays to [(weight, NIW), ...] using the standard write-back.

    NIW write-back: kappa_j = max(n_eff_j, 0.5), nu_j = kappa_j + 3,
    S_j = diag(comp_covs[j]) * kappa_j.

    Validated: Exp 153 (lines 569-587).
    """
    K = len(weights)
    result: list[tuple[float, NIW]] = []
    for j in range(K):
        kappa_j = float(max(n_eff[j], 0.5))
        nu_j = kappa_j + 3.0
        m_j = comp_means[j].copy()
        S_j = np.diag(comp_covs[j] * kappa_j)
        if nu_j < _D + 2:
            nu_j = float(_D + 2)
            kappa_j = nu_j - 3.0
        new_niw = NIW(m=m_j, kappa=kappa_j, nu=nu_j, S=S_j)
        result.append((float(weights[j]), new_niw))
    return result


def select_best_k(
    color_replay_pairs: list[tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    print_debug: bool = False,
    attempt_num: int = 0,
    color_idx: int = 0,
) -> tuple[int, list[tuple[float, NIW]], np.ndarray, dict]:
    """Select best K in K_CANDIDATES by penalized replay NLL (BIC-style).

    Score = NLL + K_PENALTY * K. Returns (best_K, components, n_eff, nll_by_k).

    Validated: Exp 153 (lines 594-648).
    """
    best_K = -1
    best_score = float("inf")
    best_components: list[tuple[float, NIW]] | None = None
    best_n_eff: np.ndarray | None = None
    nll_by_k: dict[int, tuple[float, float]] = {}

    for K in K_CANDIDATES:
        weights, comp_means, comp_covs = batch_jump_em(color_replay_pairs, K, rng)
        nll = color_replay_nll(color_replay_pairs, comp_means, comp_covs, weights)
        score = nll + K_PENALTY * K
        nll_by_k[K] = (nll, score)

        # Recompute responsibilities for n_eff
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
            best_components = em_result_to_components(
                weights, comp_means, comp_covs, n_eff
            )
            best_n_eff = n_eff.copy()

    if print_debug:
        print(
            f"      [K-select] attempt#{attempt_num} color={color_idx} "
            f"pairs={len(color_replay_pairs)}"
        )
        for K, (nll, score) in sorted(nll_by_k.items()):
            marker = " <-- CHOSEN" if K == best_K else ""
            print(f"        K={K}: nll={nll:.4f}  score={score:.4f}{marker}")

    assert best_components is not None and best_n_eff is not None
    return best_K, best_components, best_n_eff, nll_by_k


# ---------------------------------------------------------------------------
# Replay NLL scorer (Exp 145 — demoted to diagnostic in M3c)
# ---------------------------------------------------------------------------


def replay_nll(
    replay_buf: deque,
    components: list[list[tuple[float, NIW]]],
) -> float:
    """Mean -log p(obs | mu_stored, Sigma_diag_stored) over the replay buffer.

    Uses the UNNORMALIZED convention (the historical default for all M3*
    experiments). Demoted to a diagnostic-only role in Exp 145 (M3c) — it
    is computed and printed but NOT used for the keep/revert decision.

    Validated: Exp 145 (lines 363-374).
    """
    if not replay_buf:
        return float("inf")
    total = 0.0
    for obs_k, mu_s, Sigma_s_diag in replay_buf:
        log_probs = mixture_predictive_logprobs(
            mu_s, Sigma_s_diag, components, convention="unnormalized"
        )
        total += -log_probs[obs_k]
    return total / len(replay_buf)


# ---------------------------------------------------------------------------
# LiveProbation — snapshot → provisional install → keep/revert
# ---------------------------------------------------------------------------


class LiveProbation:
    """Per-color live-probation state for growth machinery.

    Protocol (Exp 145, rung M3c):
      1. Before installing a candidate, call ``snapshot(color, components, counts)``.
      2. Install the candidate into ``components[color]`` (caller's responsibility).
      3. Call ``start(color, pre_spawn_mean, phase2_t)`` to begin the probation window.
      4. At each phase-2 step for the probated color, call ``observe(surprise)``.
      5. When ``elapsed(phase2_t) >= PROBATION_STEPS``, call ``resolve(components,
         counts)`` which keeps or reverts and resets state.

    Revert restores ONLY the probated color's snapshot; other colors keep their
    probation-period learning. This is the declared policy from Exp 145.

    Validated: Exp 145 (lines 625-878).
    """

    def __init__(self) -> None:
        self._color: int = -1
        self._start_phase2_t: int = -1
        self._pre_spawn_mean: float = float("inf")
        self._snap_comps: list[tuple[float, NIW]] = []
        self._snap_counts: list[int] = []
        self._observations: list[float] = []

    @property
    def active(self) -> bool:
        """True when a probation is running."""
        return self._color >= 0

    @property
    def color(self) -> int:
        """The color currently on probation (-1 if none)."""
        return self._color

    def snapshot(
        self,
        color: int,
        components: list[list[tuple[float, NIW]]],
        counts: list[list[int]],
    ) -> None:
        """Deep-copy the color's current mixture BEFORE installing the candidate.

        Must be called before ``start()``.
        Validated: Exp 145 (lines 777-778).
        """
        self._snap_comps = _copy_color_components(components[color])
        self._snap_counts = list(counts[color])

    def start(
        self,
        color: int,
        pre_spawn_mean: float,
        phase2_t: int,
    ) -> None:
        """Begin probation for ``color`` at phase-2 step ``phase2_t``.

        Validated: Exp 145 (lines 819-823).
        """
        self._color = color
        self._start_phase2_t = phase2_t
        self._pre_spawn_mean = pre_spawn_mean
        self._observations = []

    def observe(self, surprise: float) -> None:
        """Record a per-step surprise for the probated color during probation.

        Only the probated color's observations count; callers check
        ``self.color == obs_k`` before calling.
        Validated: Exp 145 (lines 696-697).
        """
        self._observations.append(surprise)

    def elapsed(self, phase2_t: int) -> int:
        """Steps elapsed since probation started."""
        return phase2_t - self._start_phase2_t

    def resolve(
        self,
        components: list[list[tuple[float, NIW]]],
        counts: list[list[int]],
        spawn_budget: list[int],
    ) -> dict:
        """Resolve keep/revert when the probation window has expired.

        Keep condition: probation_mean <= pre_spawn_mean - KEEP_MARGIN
        (i.e., the live mean dropped by at least 0.1 nats during probation).

        On keep: decrement spawn_budget[color]; other colors' learning is retained.
        On revert: restore snapshot to components[color] and counts[color] only.

        Returns a dict with keys: color, pre_spawn_mean, probation_mean, delta,
        kept (bool).

        Validated: Exp 145 (lines 707-748).
        """
        pc = self._color
        if self._observations:
            prob_mean = float(np.mean(self._observations))
        else:
            prob_mean = float("inf")

        keep = prob_mean <= self._pre_spawn_mean - KEEP_MARGIN

        result = {
            "color": pc,
            "pre_spawn_mean": self._pre_spawn_mean,
            "probation_mean": prob_mean,
            "delta": self._pre_spawn_mean - prob_mean,
            "kept": keep,
        }

        if keep:
            spawn_budget[pc] -= 1
        else:
            components[pc] = self._snap_comps
            counts[pc] = self._snap_counts

        # Reset
        self._color = -1
        self._start_phase2_t = -1
        self._pre_spawn_mean = float("inf")
        self._snap_comps = []
        self._snap_counts = []
        self._observations = []

        return result

    def revert_incomplete(
        self,
        components: list[list[tuple[float, NIW]]],
        counts: list[list[int]],
    ) -> None:
        """Revert an in-flight probation at end of run (window did not complete).

        Validated: Exp 145 (lines 865-878), Exp 153 (lines 930-942).
        """
        if not self.active:
            return
        pc = self._color
        components[pc] = self._snap_comps
        counts[pc] = self._snap_counts
        self._color = -1


# ---------------------------------------------------------------------------
# Round-robin alarm scheduling
# ---------------------------------------------------------------------------


def pick_round_robin_color(
    alarmed_colors: list[int],
    last_attempt_step: list[int],
) -> int:
    """Pick the alarmed color with the smallest ``last_attempt_step`` (round-robin).

    Ties broken by color index (Python ``min`` is stable on the primary key).
    This ensures no persistently-alarmed color is perpetually starved in favor of
    the loudest one.

    Validated: Exp 145 (lines 762-766).
    """
    return min(alarmed_colors, key=lambda k: (last_attempt_step[k], k))


def alarmed_colors_with_budget(
    color_surprise_bufs: list[deque],
    spawn_budget: list[int],
    *,
    thresh: float = ALARM_THRESH,
) -> list[int]:
    """Return list of colors that are ALARMED and have remaining spawn budget.

    A color is alarmed when its per-color mean surprise >= thresh AND its deque
    is non-empty.  Budget check: spawn_budget[k] > 0.

    Validated: Exp 145 (lines 755-760).
    """
    result = []
    for k in range(len(color_surprise_bufs)):
        if spawn_budget[k] > 0 and len(color_surprise_bufs[k]) > 0:
            mean_k = float(np.mean(color_surprise_bufs[k]))
            if mean_k >= thresh:
                result.append(k)
    return result
