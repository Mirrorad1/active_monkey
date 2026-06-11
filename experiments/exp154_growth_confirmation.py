"""
Exp 154 — the growth-wall confirmation increment (the human's explicit word "154",
2026-06-11): normalized-density batch-jump on FRESH seeds under the honest
alarm-answered criterion — no structure-count proxy.

Context: five growth designs failed under the unnormalized-footprint convention;
Exp 152's dilution autopsy predicted and Exp 153's diagnostic arm showed that
under NORMALIZED predictive evaluation (the conjugate place update untouched),
batch-fitted complete covers are accepted at 100% and the structural-inadequacy
alarm goes fully silent — but Exp 153 failed its letter-bar on a structure-count
conjunct its own success made obsolete (un-grown colors never re-alarm because
their neighbors' tightness makes them predictable). Per L7 this confirmation
re-predeclares the criterion honestly on seeds never run on this protocol.

Hypothesis: the growth wall is the CONVENTION's, not growth's. Under normalized
evaluation, growth answers the alarm: surprise drops far below the detector
threshold globally AND per-color — growth where needed, none where not.

Setup: Exp 153 Arm B verbatim (three layouts rng 7/11/13; phase 1 T=2000
unimodal with the detector on the arm-internal normalized predictive; phase 2
T=6000 with batch-jump growth: EM complete-cover fit, K in {2,3,4} by penalized
replay NLL, replace + 400-step live probation, per-color alarms, round-robin,
jump budget one per color per 1200 steps) — EXCEPT seeds 8..15 (never run).
Per-color bookkeeping: for each color, whether it was ever alarmed in phase 2,
whether it received a kept jump, and its final-1000 per-color mean surprise.

Predictions (TRUE iff all):
- P1 the alarm answered, honestly stated: in >= 6/8 seeds per layout, >= 2/3
  layouts: (a) final-1000 global mean surprise <= phase-1 plateau - 0.4 nats;
  (b) zero detector ceiling events in the final 1000; (c) per-color final-1000
  mean surprise < 0.7 nats for ALL 4 colors (each color is quiet — grown if it
  needed growing, predictable anyway if not; the grown-vs-never-re-alarmed
  status is LOGGED per color, not gated).
- P2 the mechanism out of sample: pooled jump acceptance >= 50% per layout.
- P3 no regression: final-500 median localization <= 0.5 in >= 7/8 seeds per
  layout.

Falsifiers (final, per the human's word): P1 fails in >= 2/3 layouts — the
confirmation FAILS; the convention finding is NOT confirmed; the growth wall is
RE-CONFIRMED with both cracks and the convention hypothesis spent (MIGRATION
HALT). P3 fails = HALT (regression). P2 failing while P1 carries = MIXED (the
alarm answered by another route — log it). If P1 passes: the growth wall is
RE-BOUNDED to the unnormalized-footprint convention; the wall documentation is
amended and the normalized-predictive switch (evaluation only) is proposed for
the creature. Three-way rule per PROTOCOL step 3; blinded verification (step
4.5) before logging.
"""
from __future__ import annotations

# NOTE: copied from exp153 Arm B

import copy
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from active_loop.continuous import NIW
from active_loop.creature_continuous import ContinuousPlace

# ---------------------------------------------------------------------------
# Detector constants — copied from exp145/exp152/exp153
# ---------------------------------------------------------------------------
CEILING_MEAN_THRESH: float = 0.7    # creature.py line 56
CEILING_SLOPE_THRESH: float = 5e-4  # creature.py line 59
SURPRISE_WINDOW: int = 200          # creature.py line 52

# ---------------------------------------------------------------------------
# World constants — copied from exp145/exp152/exp153
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
T_PHASE2 = 6000           # M3e/M3f/M3g: 6000
T_TOTAL = T_PHASE1 + T_PHASE2

# Jump / probation constants — identical to exp152/exp153
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

# Exp 154: seeds 8..15 (never run on this protocol)
SEEDS = list(range(8, 16))
LAYOUT_SEEDS = [7, 11, 13]

LOG_2PI = math.log(2.0 * math.pi)

# P1 per-color quiet threshold
PER_COLOR_QUIET_THRESH = 0.7   # final-1000 per-color mean surprise < 0.7 nats


# ---------------------------------------------------------------------------
# Build a CMAP from a given rng seed — copied from exp153
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
# Compute true within-color scatter traces — copied from exp153
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
# Grid world move (wall-clamped) — copied from exp153
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
# Diagonal 2x2 Gaussian utilities — copied from exp153
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
    normalization). Used for the place update responsibility (unchanged).
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

    This is the true Gaussian-mixture marginal term for the normalized arm:
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
# NORMALIZED predictive (Arm B / exp153 verbatim) — copied from exp153
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
# Mixture emission moments — copied from exp153 (place update UNCHANGED)
# ---------------------------------------------------------------------------

def mixture_emission_moments(
    mu_p: np.ndarray,
    Sigma_p_diag: np.ndarray,
    k: int,
    components: list[list[tuple[float, NIW]]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Moment-match the mixture emission for observed color k to a single Gaussian.

    Uses the UNNORMALIZED integral for responsibilities (place update unchanged).
    Returns (mu_mix, Sigma_mix_diag, hard_idx).

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
# Surprise-ceiling detector — copied from exp153
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
# Replay NLL scorer for a color subset — copied from exp153
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
# Deep-copy utilities — copied from exp153
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
# Farthest-point seeding — copied from exp153
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
# Batch-jump EM — copied from exp153
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
# Build NIW component list from EM result — copied from exp153
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
# Select best K via penalized replay NLL — copied from exp153
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
# ARM B (Exp 154 sole arm): batch-jump with NORMALIZED predictive
# Extended vs exp153 with per-color bookkeeping for Exp 154's honest criterion.
# (copied from exp153 Arm B, extended with per-color tracking)
# ---------------------------------------------------------------------------

def run_arm_b_normalized(
    actions: np.ndarray,
    cmap: np.ndarray,
    run_rng: np.random.Generator,
    seed: int = 0,
    layout_idx: int = 1,
    print_k_debug: bool = False,
) -> dict:
    """Run Arm B: Exp 153 batch-jump with NORMALIZED Gaussian-mixture predictive.

    Identical to exp153 Arm B, plus per-color bookkeeping:
      - color_ever_alarmed[k]: True if color k triggered an alarm attempt in phase 2
      - color_kept_jump[k]: True if color k received at least one kept jump
      - color_final_mean_surprise[k]: mean surprise for color k over final 1000 steps

    The place update is UNCHANGED (footprint conjugate, unnormalized integral).
    The normalized predictive is used ONLY for:
      - per-step surprise
      - per-color alarm stats
      - detector window
      - probation pre/post means
      - plateau/final metrics
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

    # Per-color bookkeeping (Exp 154 extension)
    color_ever_alarmed: list[bool] = [False] * N_COLORS
    color_kept_jump: list[bool] = [False] * N_COLORS
    # Per-step (color, surprise) pairs for last 1000 steps — track the last T_TOTAL steps
    # We keep ALL phase2 (color, surprise) pairs then slice the last FINAL_EVAL_WINDOW
    phase2_color_surprise_pairs: list[tuple[int, float]] = []

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

        # NORMALIZED predictive for surprise (Arm B convention)
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
            phase2_color_surprise_pairs.append((obs_k, surprise_t))

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
                        color_kept_jump[pc] = True
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
                    color_ever_alarmed[jump_color] = True  # Exp 154 bookkeeping

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

                        do_print = print_k_debug and (seed == 8) and (layout_idx == 1)
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

    # Per-color bookkeeping (Exp 154 extension)
    # Compute per-color final-1000 mean surprise from phase2_color_surprise_pairs
    final_window_pairs = phase2_color_surprise_pairs[-FINAL_EVAL_WINDOW:]
    color_final_mean_surprise: list[float] = []
    for k in range(N_COLORS):
        vals_k = [s for (c, s) in final_window_pairs if c == k]
        if vals_k:
            color_final_mean_surprise.append(float(np.mean(vals_k)))
        else:
            color_final_mean_surprise.append(float("nan"))

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
        # Exp 154 per-color bookkeeping
        "color_ever_alarmed": color_ever_alarmed,
        "color_kept_jump": color_kept_jump,
        "color_final_mean_surprise": color_final_mean_surprise,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("Exp 154 — M3g: GROWTH-WALL CONFIRMATION (normalized-density, fresh seeds 8-15)")
    print("=" * 80)
    print()

    cmaps = [build_cmap(s) for s in LAYOUT_SEEDS]

    all_rows_json = []

    layout_verdicts: list[dict] = []

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

        results = []

        print(f"  --- ARM B: normalized-density batch-jump (seeds {SEEDS[0]}..{SEEDS[-1]}) ---")
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
              f"{'ColMnS':>28}")
        print("-" * 145)

        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed)
            actions = rng.integers(0, 4, size=T_TOTAL)
            run_rng = np.random.default_rng(2000 + seed)

            cont = run_arm_b_normalized(
                actions, cmap, run_rng,
                seed=seed, layout_idx=layout_idx + 1,
                print_k_debug=False,
            )
            results.append(cont)

            comps_str = str(cont["final_comps_per_color"])
            acc_frac = cont["accepted_frac"]
            acc_str = f"{acc_frac * 100:.0f}%" if not math.isnan(acc_frac) else " N/A"
            mn_delta = cont["mean_accepted_delta"]
            mn_str = f"{mn_delta:+.4f}" if not math.isnan(mn_delta) else "    N/A"
            # Per-color mean surprises compact string
            col_means = cont["color_final_mean_surprise"]
            col_str = "[" + " ".join(
                f"{v:.3f}" if not math.isnan(v) else " nan" for v in col_means
            ) + "]"
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
                  f"{col_str:>28}")

            # JSON row
            all_rows_json.append({
                "exp": 154,
                "rung": "M3g",
                "arm": "normalized",
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
                "color_ever_alarmed": cont["color_ever_alarmed"],
                "color_kept_jump": cont["color_kept_jump"],
                "color_final_mean_surprise": [
                    v if not math.isnan(v) else None
                    for v in cont["color_final_mean_surprise"]
                ],
            })

        # ---------------------------------------------------------------
        # Per-seed detail for Layout 1; summary for others
        # ---------------------------------------------------------------
        if layout_idx == 0:
            print()
            print(f"  Per-color detail (Layout 1):")
            for s_idx, cont in enumerate(results):
                seed = SEEDS[s_idx]
                alarmed = cont["color_ever_alarmed"]
                kept = cont["color_kept_jump"]
                col_means = cont["color_final_mean_surprise"]
                status_parts = []
                for k in range(N_COLORS):
                    a_str = "ALARMD" if alarmed[k] else "quiet"
                    k_str = "KEPT" if kept[k] else "no-jump"
                    m_str = f"{col_means[k]:.3f}" if not math.isnan(col_means[k]) else "nan"
                    status_parts.append(f"c{k}:{a_str}/{k_str}/{m_str}")
                print(f"    seed={seed}: " + "  ".join(status_parts))

            print()
            print(f"  Per-jump attempt detail (Layout 1):")
            for s_idx, cont in enumerate(results):
                seed = SEEDS[s_idx]
                recs = cont["attempt_records"]
                if not recs:
                    print(f"    seed={seed}: no jump attempts")
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
                    print(f"    seed={seed} jump#{i+1}: "
                          f"color={rec['color']}  K={rec['K_chosen']}  "
                          f"pre={rec['pre_jump_mean']:.4f}  "
                          f"prob={prob_str}  delta={delta_str}  "
                          f"{kept_str}  [{nll_summary}]")
        else:
            total_attempts = sum(len(c["attempt_records"]) for c in results)
            total_kept = sum(c["jumps_kept"] for c in results)
            print(f"  Layout {layout_idx + 1}: {total_attempts} attempts, {total_kept} kept")

        # ---------------------------------------------------------------
        # Tallies — P1 (honest criterion, no structure-count conjunct)
        # ---------------------------------------------------------------
        print()
        print(f"  TALLIES — Layout {layout_idx + 1}")
        print()

        # P1(a): drop >= 0.4 nats
        p1a_per_seed = [cont["drop"] >= 0.4 for cont in results]
        p1a_count = sum(p1a_per_seed)

        # P1(b): zero final ceiling events
        p1b_per_seed = [cont["phase2_final_ceiling_events"] == 0 for cont in results]
        p1b_count = sum(p1b_per_seed)

        # P1(c): ALL 4 colors have final-1000 per-color mean surprise < 0.7 nats
        p1c_per_seed = []
        for cont in results:
            col_means = cont["color_final_mean_surprise"]
            all_quiet = all(
                (not math.isnan(v)) and v < PER_COLOR_QUIET_THRESH
                for v in col_means
            )
            p1c_per_seed.append(all_quiet)
        p1c_count = sum(p1c_per_seed)

        # P1 conjunct: all three conditions pass for this seed
        p1_per_seed = [
            p1a_per_seed[i] and p1b_per_seed[i] and p1c_per_seed[i]
            for i in range(len(results))
        ]
        p1_count = sum(p1_per_seed)
        p1_holds_layout = p1_count >= 6

        print(f"  P1(a) drop >= 0.4 nats: {p1a_count}/8  (need >= 6/8)  "
              f"{'PASS' if p1a_count >= 6 else 'FAIL'}")
        print(f"  P1(b) zero final ceil:  {p1b_count}/8  (need >= 6/8)  "
              f"{'PASS' if p1b_count >= 6 else 'FAIL'}")
        print(f"  P1(c) all-4 col<0.7:    {p1c_count}/8  (need >= 6/8)  "
              f"{'PASS' if p1c_count >= 6 else 'FAIL'}")
        print(f"  P1 conjunct (all three): {p1_count}/8  (need >= 6/8)  "
              f"{'PASS' if p1_holds_layout else 'FAIL'}")
        print()
        for s_idx, cont in enumerate(results):
            seed = SEEDS[s_idx]
            col_means = cont["color_final_mean_surprise"]
            col_str = "[" + " ".join(
                f"{v:.3f}" if not math.isnan(v) else " nan" for v in col_means
            ) + "]"
            alarmed_str = str([int(x) for x in cont["color_ever_alarmed"]])
            kept_str_per = str([int(x) for x in cont["color_kept_jump"]])
            print(f"    seed={seed}: "
                  f"drop={cont['drop']:.3f}({'pass' if p1a_per_seed[s_idx] else 'FAIL'})  "
                  f"ceil={cont['phase2_final_ceiling_events']}({'pass' if p1b_per_seed[s_idx] else 'FAIL'})  "
                  f"col{col_str}({'pass' if p1c_per_seed[s_idx] else 'FAIL'})  "
                  f"alarmed={alarmed_str}  kept={kept_str_per}  "
                  f"P1={'pass' if p1_per_seed[s_idx] else 'FAIL'}")

        # P2: pooled jump acceptance >= 50%
        total_attempted_layout = sum(c["total_attempted"] for c in results)
        total_accepted_layout = sum(c["total_accepted"] for c in results)
        pooled_acc_frac = (
            total_accepted_layout / total_attempted_layout
            if total_attempted_layout > 0 else float("nan")
        )
        p2_holds_layout = (
            (not math.isnan(pooled_acc_frac)) and pooled_acc_frac >= 0.5
        )
        acc_str = f"{pooled_acc_frac * 100:.1f}%" if not math.isnan(pooled_acc_frac) else "N/A"
        print(f"\n  P2 pooled acceptance >= 50%: {total_accepted_layout}/{total_attempted_layout} "
              f"= {acc_str}  {'PASS' if p2_holds_layout else 'FAIL'}")

        # P3: final-500 median localization <= 0.5, >= 7/8 seeds
        p3_per_seed = [cont["p2_final500_loc_median"] <= 0.5 for cont in results]
        p3_count = sum(p3_per_seed)
        p3_holds_layout = p3_count >= 7
        print(f"\n  P3 no regression (loc_med <= 0.5, >= 7/8): {p3_count}/8  "
              f"{'PASS' if p3_holds_layout else 'FAIL (HALT)'}")
        for s_idx, cont in enumerate(results):
            seed = SEEDS[s_idx]
            print(f"    seed={seed}: loc_med={cont['p2_final500_loc_median']:.4f}  "
                  f"{'pass' if p3_per_seed[s_idx] else 'FAIL'}")

        layout_verdicts.append({
            "layout_idx": layout_idx + 1,
            "layout_seed": layout_seed,
            "p1_holds": p1_holds_layout,
            "p1_count": p1_count,
            "p1a_count": p1a_count,
            "p1b_count": p1b_count,
            "p1c_count": p1c_count,
            "p2_holds": p2_holds_layout,
            "pooled_acc_frac": pooled_acc_frac,
            "total_attempted": total_attempted_layout,
            "total_accepted": total_accepted_layout,
            "p3_holds": p3_holds_layout,
            "p3_count": p3_count,
        })

        # JSON summary row
        all_rows_json.append({
            "exp": 154,
            "rung": "M3g",
            "arm": "normalized",
            "layout_seed": layout_seed,
            "layout_idx": layout_idx + 1,
            "seed": -1,
            "summary": True,
            "p1_holds": p1_holds_layout,
            "p1_count": p1_count,
            "p1a_count": p1a_count,
            "p1b_count": p1b_count,
            "p1c_count": p1c_count,
            "p2_holds": p2_holds_layout,
            "pooled_acc_frac": (
                pooled_acc_frac if not math.isnan(pooled_acc_frac) else None
            ),
            "total_attempted": total_attempted_layout,
            "total_accepted": total_accepted_layout,
            "p3_holds": p3_holds_layout,
            "p3_count": p3_count,
        })

    # -----------------------------------------------------------------------
    # Global conjunct evaluation
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("GLOBAL CONJUNCT TALLIES (across all 3 layouts)")
    print("=" * 80)

    # P1: >= 6/8 seeds in >= 2/3 layouts
    p1_layouts_passing = sum(1 for v in layout_verdicts if v["p1_holds"])
    p1_global = p1_layouts_passing >= 2

    print(f"\nP1 alarm answered (>= 6/8 seeds in >= 2/3 layouts):")
    for v in layout_verdicts:
        print(f"  Layout {v['layout_idx']} (rng({v['layout_seed']})): "
              f"conjunct={v['p1_count']}/8  "
              f"[a(drop)={v['p1a_count']}/8  b(ceil)={v['p1b_count']}/8  c(per-col)={v['p1c_count']}/8]  "
              f"{'PASS' if v['p1_holds'] else 'FAIL'}")
    print(f"  Layouts passing: {p1_layouts_passing}/3  (need >= 2)  "
          f"P1 GLOBAL: {'PASS' if p1_global else 'FAIL'}")

    # P2: >= 50% acceptance per layout
    p2_layouts_passing = sum(1 for v in layout_verdicts if v["p2_holds"])
    p2_global = p2_layouts_passing == 3  # all layouts

    print(f"\nP2 pooled acceptance >= 50% (per layout):")
    for v in layout_verdicts:
        acc_str = (f"{v['pooled_acc_frac'] * 100:.1f}%"
                   if not math.isnan(v["pooled_acc_frac"]) else "N/A")
        print(f"  Layout {v['layout_idx']}: {v['total_accepted']}/{v['total_attempted']} "
              f"= {acc_str}  {'PASS' if v['p2_holds'] else 'FAIL'}")
    print(f"  P2 GLOBAL (all 3 layouts): {'PASS' if p2_global else 'FAIL'}")

    # P3: >= 7/8 in all 3 layouts
    p3_global = all(v["p3_holds"] for v in layout_verdicts)

    print(f"\nP3 no regression (>= 7/8 seeds, all 3 layouts):")
    for v in layout_verdicts:
        print(f"  Layout {v['layout_idx']}: {v['p3_count']}/8  "
              f"{'PASS' if v['p3_holds'] else 'FAIL'}")
    print(f"  P3 GLOBAL: {'PASS' if p3_global else 'FAIL (HALT)'}")

    # -----------------------------------------------------------------------
    # Named outcome determination (three-way rule per PROTOCOL step 3)
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("NAMED OUTCOME + VERDICT")
    print("=" * 80)
    print()

    halt_regression = not p3_global

    if halt_regression:
        verdict = "NEGATIVE"
        print(f"VERDICT: {verdict}")
        print("HALT — P3 REGRESSION DETECTED")
        for v in layout_verdicts:
            if not v["p3_holds"]:
                print(f"  Layout {v['layout_idx']} P3 failed: {v['p3_count']}/8 seeds pass.")
    elif not p1_global:
        # P1 fails in >= 2/3 layouts
        verdict = "NEGATIVE"
        print(f"VERDICT: {verdict}")
        print(
            f"P1 FAILS ({p1_layouts_passing}/3 layouts pass, need >= 2). "
            "The confirmation FAILS. The convention finding is NOT confirmed. "
            "The growth wall is RE-CONFIRMED with both cracks and the convention "
            "hypothesis spent. MIGRATION HALT."
        )
    elif p1_global and not p2_global:
        verdict = "MIXED"
        print(f"VERDICT: {verdict}")
        print(
            f"P1 PASSES ({p1_layouts_passing}/3 layouts). "
            f"P2 FAILS ({p2_layouts_passing}/3 layouts >= 50% acceptance). "
            "Alarm answered by another route — the growth wall is RE-BOUNDED to the "
            "unnormalized-footprint convention; wall documentation amended; normalized-"
            "predictive switch (evaluation only) proposed for creature. "
            "P2 failing while P1 carries = MIXED (mechanism not confirmed out-of-sample "
            "at this acceptance threshold; log it). P3 holds. "
            "NOTE: P2 failure here means acceptance < 50%, not that growth failed — "
            "alarm answered by pre-emptive tightness or another route."
        )
    elif p1_global and p2_global:
        verdict = "POSITIVE"
        print(f"VERDICT: {verdict}")
        print(
            f"P1 PASSES ({p1_layouts_passing}/3 layouts). "
            f"P2 PASSES ({p2_layouts_passing}/3 layouts >= 50% acceptance). "
            "P3 holds. "
            "The growth wall is RE-BOUNDED to the unnormalized-footprint convention. "
            "Wall documentation is amended. The normalized-predictive switch (evaluation "
            "only) is proposed for the creature. Under normalized evaluation, growth "
            "answers the alarm: surprise drops far below the detector threshold globally "
            "AND per-color — grown where needed, predictable anyway where not."
        )
    else:
        verdict = "MIXED"
        print(f"VERDICT: {verdict}")
        print(
            f"P1={'PASS' if p1_global else 'FAIL'} ({p1_layouts_passing}/3 layouts), "
            f"P2={'PASS' if p2_global else 'FAIL'}, "
            f"P3={'PASS' if p3_global else 'FAIL'}. "
            "Inspect tallies above."
        )

    print("=" * 80)

    # -----------------------------------------------------------------------
    # JSON output
    # -----------------------------------------------------------------------
    all_rows_json.append({
        "exp": 154,
        "rung": "M3g",
        "seed": -2,
        "global_summary": True,
        "p1_global": p1_global,
        "p1_layouts_passing": p1_layouts_passing,
        "p2_global": p2_global,
        "p2_layouts_passing": p2_layouts_passing,
        "p3_global": p3_global,
        "halt_regression": halt_regression,
        "verdict": verdict,
    })

    out_path = Path(__file__).parent / "outputs" / "exp154_rows.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for row in all_rows_json:
            fh.write(json.dumps(row) + "\n")
    print(f"\nJSON rows written to {out_path}")


if __name__ == "__main__":
    main()
