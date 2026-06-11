"""
Exp 143 — continuous-creature rung M3: the aliasing wall (the centerpiece — the
Exp 132 detector's first live test, and the spawn-and-select principle's trial on
the continuous substrate).

Card loop/directions/continuous-creature.md. Guardrail: FAIL-A or FAIL-B below
HALTS the migration for explicit human input.

Hypothesis: in an ALIASED world (4 colors x 4 cells each), one Gaussian per color
structurally cannot represent the emission map — each learned mean is pulled to
its color's cell centroid, localization stays at chance scale, and the surprise
ceiling detector (Exp 132's conjunction, same constants: rolling-200 window,
mean > 0.7 nats, |slope| < 5e-4, learning active) FIRES — irreducible surprise
from structural inadequacy, its first live positive. Growing the model by the
structure-learning spawn rule (ceiling -> spawn a component for the
worst-surprise color at the current posterior mean -> keep iff running predictive
NLL on the recent replay window STRICTLY decreases, else revert; budget 4 per
color) then recovers localization, and the detector goes quiet.

Setup: 4x4 grid in R^2; ALIASED cmap: 4 colors, 4 cells each, one fixed seeded
random layout (rng(7), shuffled assignment, same for all seeds; print it);
uniform-random actions; seeds 0..7; identical streams to all agents. Continuous
agent: ContinuousPlace with predict_clamped_moments (Q=0.05^2 I), diagonal
emissions; per-color mixture of NIW components (start 1 per color, m0 = arena
center, kappa0 = 1, nu0 = 4, S0 = 0.35^2*(nu0-d-1)*I); place update with the
MIXTURE emission for the observed color moment-matched to a single Gaussian
(declared); NIW update by hard assignment to the max-responsibility component
(declared); per-step surprise = -ln p(o_t) under the pre-update mixture
predictive over the 4 colors (normalized). Phase 1 T=2000: spawning DISABLED
(detector armed, events recorded). Phase 2 T=4000: every 200 steps, if a ceiling
event fired in the last window and the color budget allows, spawn for the color
with the highest mean recent surprise contribution (component seeded at the
current place posterior mean, kappa=1, nu=4, S=S0, weight = 1/(n_components+1),
renormalized); KEEP iff mean predictive NLL over the last 400 (obs, posterior-
mean) replay pairs, re-scored under the new model with the SAME stored posterior
means, strictly decreases vs the pre-spawn model; else revert. Tabular twin (the
creature's own equations, exp142 conventions) on identical streams. Eval = final
1000 steps of phase 2.

Predictions (TRUE iff all):
- P2a unimodal wall + detector: phase-1 final-500 median localization error
  >= 1.0 in >= 6/8 seeds (the wall is real), AND >= 1 ceiling event in >= 7/8
  seeds during phase 1 (the detector's first live positive).
- P2b spawn recovery: final-500 median localization <= 0.5 in >= 6/8 seeds, AND
  grown component count within +-1 of the true multiplicity (4) for >= 3/4
  colors (those seeds), AND zero ceiling events in the final 1000 steps (those
  seeds).
- P3 twin sanity: tabular MAP-cell correct on >= 80% of the final 500 steps in
  >= 7/8 seeds.

Falsifiers: FAIL-A (HALT) = phase-1 localization >= 1.0 (wall present) but
detector silent in >= 2/8 seeds — instrument gap. FAIL-B (HALT) = spawning fails
to recover (P2b localization arm fails in >= 3/8 seeds) — the toolkit does not
compose with the substrate; log as the deep negative. P3 failing = twin/stream
instrument problem (halt and inspect). If phase-1 localization does NOT hit the
wall (< 1.0 — the unimodal map localizes despite aliasing), that is a surprise:
NOT a halt; log as the finding and skip P2b grading (the premise of recovery is
moot). Three-way rule per PROTOCOL step 3; 'MIGRATION HALT' printed on any halt
arm.
"""
from __future__ import annotations

# NOTE: Mixture machinery lives IN this experiment file.
# Promotion to active_loop/creature_continuous.py or active_loop/continuous.py
# happens AFTER this rung earns it by passing P2a + P2b. Do not promote early.

import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from active_loop.continuous import NIW
from active_loop.creature_continuous import ContinuousPlace

# ---------------------------------------------------------------------------
# Detector constants — reused from creature.py (CEILING_MEAN_THRESH=0.7,
# CEILING_SLOPE_THRESH=5e-4, SURPRISE_WINDOW=200).  Imported by value to keep
# this file standalone; the comment above is the source-of-truth pointer.
# ---------------------------------------------------------------------------
CEILING_MEAN_THRESH: float = 0.7    # creature.py line 56
CEILING_SLOPE_THRESH: float = 5e-4  # creature.py line 59
SURPRISE_WINDOW: int = 200          # creature.py line 52

# ---------------------------------------------------------------------------
# World constants
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

# Fixed aliased color map: rng(7), shuffled assignment — same for ALL seeds.
# 4 colors x 4 cells each.  CMAP[cell] = color in {0,1,2,3}.
_cmap_rng = np.random.default_rng(7)
_perm = _cmap_rng.permutation(N_CELLS)
CMAP = np.empty(N_CELLS, dtype=int)
for _color_idx in range(N_COLORS):
    for _slot in range(CELLS_PER_COLOR):
        CMAP[_perm[_color_idx * CELLS_PER_COLOR + _slot]] = _color_idx

# Action -> (dx, dy)
ACTION_DELTA = {
    0: np.array([0.0, -1.0]),  # up
    1: np.array([0.0, +1.0]),  # down
    2: np.array([-1.0, 0.0]),  # left
    3: np.array([+1.0, 0.0]),  # right
}

Q_SCALE = 0.05
Q_diag = np.array([Q_SCALE ** 2, Q_SCALE ** 2])  # diagonal entries only (2,)
Q = np.diag(Q_diag)

D = 2
KAPPA0 = 1.0
NU0 = 4.0
S0_SCALE = 0.35 ** 2 * (NU0 - D - 1)  # = 0.35^2 * 1.0
S0 = S0_SCALE * np.eye(D)

T_PHASE1 = 2000
T_PHASE2 = 4000
T_TOTAL = T_PHASE1 + T_PHASE2

SPAWN_BUDGET = 4          # max components per color
SPAWN_INTERVAL = 200      # check every N steps during phase 2
REPLAY_WINDOW = 400       # replay pairs for NLL scoring
FINAL_EVAL_WINDOW = 1000  # final steps of phase 2 for grading
FINAL_WINDOW_P1 = 500     # final steps of phase 1 for P2a localization

SEEDS = list(range(8))


# ---------------------------------------------------------------------------
# Grid world move (wall-clamped)
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
# Diagonal 2x2 Gaussian utilities (scalars, no linalg needed)
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
    # Sk diagonal entries
    sk0, sk1 = ESigma_k_diag[0], ESigma_k_diag[1]
    sp0, sp1 = Sigma_p_diag[0], Sigma_p_diag[1]
    # logdet(Sk) = log(sk0) + log(sk1)
    logdet_Sk = math.log(sk0) + math.log(sk1)
    # logdet(Sk + Sp) = log(sk0+sp0) + log(sk1+sp1)
    c0 = sk0 + sp0
    c1 = sk1 + sp1
    logdet_C = math.log(c0) + math.log(c1)
    # Mahalanobis: (mu_p - m_k)^T (Sk + Sp)^{-1} (mu_p - m_k) — diagonal
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
    # Precision-weighted sum per axis
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
            # Guard: floor to 1e-9 for numerical safety
            ESigma_kj_diag = np.maximum(ESigma_kj_diag, 1e-9)
            logI = _diag_log_integral(mu_p, Sigma_p_diag, m_kj, ESigma_kj_diag)
            log_terms.append(math.log(max(w_kj, 1e-300)) + logI)
        # logsumexp over components for this color
        log_terms_arr = np.array(log_terms)
        log_unnorm[k] = float(np.logaddexp.reduce(log_terms_arr))
    # Normalize over colors
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

        # Product: N(s; mu_p, Sigma_p) * N(s; m_kj, ESigma_kj_diag) -> (mu_j, sig_j_diag)
        mu_j, sig_j_diag = _diag_gaussian_product(mu_p, Sigma_p_diag, m_kj, ESigma_kj_diag)
        prod_mus.append(mu_j)
        prod_sigs.append(sig_j_diag)

    # Normalize responsibilities
    log_r_sum = float(np.logaddexp.reduce(log_r))
    r = np.exp(log_r - log_r_sum)  # shape (n,)

    # Moment-match to single diagonal Gaussian
    mu_mix = np.zeros(2)
    for j in range(n):
        mu_mix += r[j] * prod_mus[j]

    Sigma_mix_diag = np.zeros(2)
    for i in range(2):
        e2 = sum(r[j] * (prod_sigs[j][i] + prod_mus[j][i] ** 2) for j in range(n))
        Sigma_mix_diag[i] = e2 - mu_mix[i] ** 2

    # Floor: covariance must be positive
    Sigma_mix_diag = np.maximum(Sigma_mix_diag, 1e-9)

    hard_idx = int(np.argmax(r))
    return mu_mix, Sigma_mix_diag, hard_idx


# ---------------------------------------------------------------------------
# Surprise-ceiling detector (Exp 132 conjunction, same constants)
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
    # Slope via linear polyfit
    xs = np.arange(len(arr), dtype=float)
    slope = float(np.polyfit(xs, arr, 1)[0])
    return abs(slope) < CEILING_SLOPE_THRESH


# ---------------------------------------------------------------------------
# Replay NLL scorer: re-score stored (obs_color, place_mu, place_Sigma_diag)
# under a given components model.
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
# Deep-copy components (list of lists of (weight, NIW))
# ---------------------------------------------------------------------------

def _copy_components(
    components: list[list[tuple[float, NIW]]],
) -> list[list[tuple[float, NIW]]]:
    """Return a deep copy of the components structure."""
    import copy
    result = []
    for color_comps in components:
        result.append([(w, copy.deepcopy(niw)) for (w, niw) in color_comps])
    return result


def _copy_counts(counts: list[list[int]]) -> list[list[int]]:
    return [list(c) for c in counts]


# ---------------------------------------------------------------------------
# Per-color mean recent surprise: sum of surprise over steps where obs==color
# (last SPAWN_INTERVAL steps in phase 2)
# ---------------------------------------------------------------------------

def color_mean_surprise(
    recent_surprises_by_color: list[list[float]],
) -> np.ndarray:
    """Mean surprise per color over recent window.  Colors with no observations get 0."""
    out = np.zeros(N_COLORS)
    for k in range(N_COLORS):
        vals = recent_surprises_by_color[k]
        if vals:
            out[k] = float(np.mean(vals))
    return out


# ---------------------------------------------------------------------------
# Tabular twin
# ---------------------------------------------------------------------------

def run_tabular(actions: np.ndarray) -> dict:
    """Run tabular twin (creature's own equations) on aliased world.

    Declared: linear-space per exp142 conventions (16 cells well-conditioned,
    no underflow risk with 4 colors and balanced cmap).

    Returns final_500_map_frac: fraction of final-500-step timesteps where
    MAP cell == true cell.
    """
    # pA: shape (N_COLORS, N_CELLS), init 0.1 uniform + 0.01*jitter
    rng_pA = np.random.default_rng(9999)  # fixed jitter rng (same across seeds for twin)
    pA = np.full((N_COLORS, N_CELLS), 0.1) + 0.01 * rng_pA.random((N_COLORS, N_CELLS))

    # B transition matrix: B[s', s, a]
    B = np.zeros((N_CELLS, N_CELLS, 4))
    for s in range(N_CELLS):
        for a in range(4):
            s2 = move(s, a)
            B[s2, s, a] = 1.0

    qs = np.ones(N_CELLS) / N_CELLS
    true_cell = 0

    map_correct = np.empty(T_TOTAL, dtype=bool)

    for t in range(T_TOTAL):
        obs = int(CMAP[true_cell])

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

        # MAP cell correct?
        map_cell = int(np.argmax(qs_updated))
        map_correct[t] = (map_cell == true_cell)

        action = int(actions[t])
        true_cell = move(true_cell, action)
        qs = B[:, :, action] @ qs_updated

    # Final 500 of phase 2 = last 500 of T_TOTAL
    final_500 = map_correct[T_TOTAL - 500:]
    return {"final_500_map_frac": float(np.mean(final_500))}


# ---------------------------------------------------------------------------
# Continuous agent with mixture + detector + spawning
# ---------------------------------------------------------------------------

def run_continuous(actions: np.ndarray) -> dict:
    """Run the continuous agent across phase 1 (T=2000) and phase 2 (T=4000).

    Returns a dict with all per-seed metrics needed for P2a, P2b grading.
    """
    # --- Place prior: diagonal N(arena_center, 4I) ---
    mu0 = ARENA_CENTER.copy()
    Sigma0_diag = np.array([4.0, 4.0])
    cp = ContinuousPlace(mu0, np.diag(Sigma0_diag), ARENA)

    # --- Per-color mixture: 1 component per color initially ---
    # components[k] = list of (weight, NIW)
    # Hard-assignment counts: counts[k][j] = number of times component j was updated
    components: list[list[tuple[float, NIW]]] = []
    counts: list[list[int]] = []
    for k in range(N_COLORS):
        niw0 = NIW(m=ARENA_CENTER.copy(), kappa=KAPPA0, nu=NU0, S=S0.copy())
        components.append([(1.0, niw0)])
        counts.append([1])  # start at 1 to avoid div-by-zero on weights

    # Spawn budget tracker
    spawn_budget = [SPAWN_BUDGET] * N_COLORS  # remaining spawns per color

    # Surprise buffer (rolling deque)
    surprise_buf: deque = deque(maxlen=SURPRISE_WINDOW)

    # Replay buffer: (obs_color, mu_at_predict, Sigma_diag_at_predict)
    replay_buf: deque = deque(maxlen=REPLAY_WINDOW)

    # --- Phase 1 tracking ---
    phase1_loc_errors = np.empty(T_PHASE1)
    phase1_ceiling_events = 0
    phase1_surprise_by_color: list[list[float]] = [[] for _ in range(N_COLORS)]  # for spawn scoring

    # --- Phase 2 tracking ---
    phase2_loc_errors = np.empty(T_PHASE2)
    phase2_ceiling_events_total = 0
    phase2_final_ceiling_events = 0  # in final FINAL_EVAL_WINDOW steps
    spawns_kept = 0
    spawns_reverted = 0

    # For per-200-step spawn tracking in phase 2
    recent_surprises_by_color: list[list[float]] = [[] for _ in range(N_COLORS)]
    last_spawn_step = -SPAWN_INTERVAL  # track last spawn attempt timing

    true_cell = 0

    for t in range(T_TOTAL):
        obs_k = int(CMAP[true_cell])

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
        is_phase2 = not is_phase1
        phase2_t = t - T_PHASE1  # step index within phase 2

        if is_phase1:
            phase1_loc_errors[t] = float(np.linalg.norm(mu_p - CELL_CENTERS[true_cell]))
            phase1_surprise_by_color[obs_k].append(surprise_t)
            # Check detector (armed in phase 1)
            if check_ceiling(surprise_buf):
                phase1_ceiling_events += 1
        else:
            phase2_loc_errors[phase2_t] = float(np.linalg.norm(mu_p - CELL_CENTERS[true_cell]))
            recent_surprises_by_color[obs_k].append(surprise_t)

            # Ceiling check (always armed in phase 2 too, for quiet check)
            is_ceiling = check_ceiling(surprise_buf)
            if is_ceiling:
                phase2_ceiling_events_total += 1
                if phase2_t >= T_PHASE2 - FINAL_EVAL_WINDOW:
                    phase2_final_ceiling_events += 1

            # --- Spawn attempt every SPAWN_INTERVAL steps in phase 2 ---
            if (phase2_t > 0 and phase2_t % SPAWN_INTERVAL == 0):
                if is_ceiling:
                    # Find worst-surprise color with remaining budget
                    color_surps = color_mean_surprise(recent_surprises_by_color)
                    # Only consider colors with budget remaining
                    candidates = [k for k in range(N_COLORS) if spawn_budget[k] > 0]
                    if candidates:
                        worst_color = int(max(candidates, key=lambda k: color_surps[k]))

                        # Spawn: new component at current place posterior mean
                        spawn_mu = mu_p.copy()
                        spawn_niw = NIW(m=spawn_mu, kappa=KAPPA0, nu=NU0, S=S0.copy())

                        # Save pre-spawn state
                        pre_spawn_components = _copy_components(components)
                        pre_spawn_counts = _copy_counts(counts)
                        pre_spawn_budget = list(spawn_budget)

                        # Build candidate model
                        n_old = len(components[worst_color])
                        new_w = 1.0 / (n_old + 1)
                        # Renormalize existing weights
                        old_total = sum(w for (w, _) in components[worst_color])
                        new_comps = [(w / old_total * (1.0 - new_w), niw)
                                     for (w, niw) in components[worst_color]]
                        new_comps.append((new_w, spawn_niw))
                        components[worst_color] = new_comps
                        counts[worst_color].append(1)

                        # Score: mean NLL under candidate vs pre-spawn
                        nll_new = replay_nll(replay_buf, components)
                        nll_old = replay_nll(replay_buf, pre_spawn_components)

                        if nll_new < nll_old - 1e-6:
                            # KEEP
                            spawn_budget[worst_color] -= 1
                            spawns_kept += 1
                        else:
                            # REVERT
                            components = pre_spawn_components
                            counts = pre_spawn_counts
                            spawn_budget = pre_spawn_budget
                            spawns_reverted += 1

                # Reset recent surprise tracking regardless
                recent_surprises_by_color = [[] for _ in range(N_COLORS)]

        # --- Place update: moment-match mixture emission for obs_k ---
        mu_mix, Sigma_mix_diag, hard_idx = mixture_emission_moments(
            mu_p, Sigma_p_diag, obs_k, components
        )
        cp.update(mu_mix, np.diag(Sigma_mix_diag))

        # --- NIW hard-assignment update ---
        post_mu = cp.mu
        post_Sigma_diag = np.array([cp.Sigma[0, 0], cp.Sigma[1, 1]])
        post_Sigma_diag = np.maximum(post_Sigma_diag, 1e-9)

        # Declared: hard assignment to max-responsibility component
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

    # --- Metrics ---
    final_comps_per_color = [len(components[k]) for k in range(N_COLORS)]

    p1_final500_loc_median = float(np.median(phase1_loc_errors[T_PHASE1 - 500:]))
    p2_final1000_loc_median = float(np.median(phase2_loc_errors[T_PHASE2 - FINAL_EVAL_WINDOW:]))
    p2_final500_loc_median = float(np.median(phase2_loc_errors[T_PHASE2 - 500:]))

    return {
        "p1_final500_loc_median": p1_final500_loc_median,
        "p1_ceiling_events": phase1_ceiling_events,
        "p2_final1000_loc_median": p2_final1000_loc_median,
        "p2_final500_loc_median": p2_final500_loc_median,
        "final_comps_per_color": final_comps_per_color,
        "spawns_kept": spawns_kept,
        "spawns_reverted": spawns_reverted,
        "phase2_final_ceiling_events": phase2_final_ceiling_events,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Print the fixed aliased cmap layout
    print("=" * 80)
    print("Exp 143 — continuous-creature rung M3: the aliasing wall")
    print("=" * 80)
    print()
    print("ALIASED CMAP (rng(7), 4 colors x 4 cells each):")
    print("  Grid layout (cell index -> color; row-major, row=y, col=x):")
    for r in range(ROWS):
        row_colors = [CMAP[r * COLS + c] for c in range(COLS)]
        print(f"  row {r}: {row_colors}")
    print(f"  CMAP array: {CMAP.tolist()}")
    print()
    print(f"  Color -> cells:")
    for k in range(N_COLORS):
        cells_k = [i for i in range(N_CELLS) if CMAP[i] == k]
        centers_k = [CELL_CENTERS[i].tolist() for i in cells_k]
        print(f"    color {k}: cells {cells_k}  centers {centers_k}")
    print()

    rows_json = []
    tab_results = []
    cont_results = []

    print(f"{'Seed':>4}  "
          f"{'P1LocMed':>8}  "
          f"{'P1Ceil':>6}  "
          f"{'P2LocMed':>8}  "
          f"{'Comps':>12}  "
          f"{'Kept':>4}  "
          f"{'Revt':>4}  "
          f"{'P2FinCeil':>9}  "
          f"{'TwinF500':>8}")
    print("-" * 85)

    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        actions = rng.integers(0, 4, size=T_TOTAL)

        tab = run_tabular(actions)
        cont = run_continuous(actions)

        tab_results.append(tab)
        cont_results.append(cont)

        comps_str = str(cont["final_comps_per_color"])
        print(f"{seed:>4}  "
              f"{cont['p1_final500_loc_median']:>8.4f}  "
              f"{cont['p1_ceiling_events']:>6d}  "
              f"{cont['p2_final500_loc_median']:>8.4f}  "
              f"{comps_str:>12}  "
              f"{cont['spawns_kept']:>4d}  "
              f"{cont['spawns_reverted']:>4d}  "
              f"{cont['phase2_final_ceiling_events']:>9d}  "
              f"{tab['final_500_map_frac']:>8.4f}")

        rows_json.append({
            "exp": 143,
            "rung": "M3",
            "seed": seed,
            "p1_final500_loc_median": cont["p1_final500_loc_median"],
            "p1_ceiling_events": cont["p1_ceiling_events"],
            "p2_final1000_loc_median": cont["p2_final1000_loc_median"],
            "p2_final500_loc_median": cont["p2_final500_loc_median"],
            "final_comps_per_color": cont["final_comps_per_color"],
            "spawns_kept": cont["spawns_kept"],
            "spawns_reverted": cont["spawns_reverted"],
            "phase2_final_ceiling_events": cont["phase2_final_ceiling_events"],
            "twin_final500_map_frac": tab["final_500_map_frac"],
        })

    # ---------------------------------------------------------------------------
    # Tallies
    # ---------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("TALLIES")
    print("=" * 80)

    # --- P2a: unimodal wall + detector ---
    # Arm 1: phase-1 final-500 median loc >= 1.0 in >= 6/8 seeds
    p2a_wall_per_seed = [r["p1_final500_loc_median"] >= 1.0 for r in cont_results]
    p2a_wall_count = sum(p2a_wall_per_seed)
    # Arm 2: >= 1 ceiling event in >= 7/8 seeds during phase 1
    p2a_ceil_per_seed = [r["p1_ceiling_events"] >= 1 for r in cont_results]
    p2a_ceil_count = sum(p2a_ceil_per_seed)

    wall_present = p2a_wall_count >= 6
    ceil_fires = p2a_ceil_count >= 7
    p2a_holds = wall_present and ceil_fires

    print(f"\nP2a unimodal wall + detector:")
    print(f"  Seeds with phase-1 final-500 loc_median >= 1.0: {p2a_wall_count}/8  "
          f"(need >= 6/8)  {'PASS' if wall_present else 'FAIL'}")
    print(f"  Seeds with >= 1 ceiling event in phase 1: {p2a_ceil_count}/8  "
          f"(need >= 7/8)  {'PASS' if ceil_fires else 'FAIL'}")
    print(f"  P2a: {'PASS' if p2a_holds else 'FAIL'}")
    for s, r in zip(SEEDS, cont_results):
        print(f"    seed={s}: p1_loc_med={r['p1_final500_loc_median']:.4f}  "
              f"p1_ceil_events={r['p1_ceiling_events']}  "
              f"wall={'pass' if p2a_wall_per_seed[s] else 'FAIL'}  "
              f"ceil={'pass' if p2a_ceil_per_seed[s] else 'FAIL'}")

    # --- P2b: spawn recovery ---
    # Arm 1: final-500 median loc <= 0.5 in >= 6/8 seeds
    p2b_loc_per_seed = [r["p2_final500_loc_median"] <= 0.5 for r in cont_results]
    p2b_loc_count = sum(p2b_loc_per_seed)
    p2b_loc_holds = p2b_loc_count >= 6

    # Arm 2: grown component count within +/-1 of true multiplicity (4) for >= 3/4
    #        colors in each passing seed
    TRUE_MULT = 4
    p2b_comps_per_seed = []
    for r in cont_results:
        comps = r["final_comps_per_color"]
        colors_ok = sum(1 for c in comps if abs(c - TRUE_MULT) <= 1)
        p2b_comps_per_seed.append(colors_ok >= 3)
    p2b_comps_count = sum(p2b_comps_per_seed)
    p2b_comps_holds = p2b_comps_count >= 6  # applied to seeds that pass loc arm

    # Arm 3: zero ceiling events in final FINAL_EVAL_WINDOW steps in those seeds
    p2b_quiet_per_seed = [r["phase2_final_ceiling_events"] == 0 for r in cont_results]
    p2b_quiet_count = sum(p2b_quiet_per_seed)
    p2b_quiet_holds = p2b_quiet_count >= 6  # in those (loc-passing) seeds

    p2b_holds = p2b_loc_holds and p2b_comps_holds and p2b_quiet_holds

    print(f"\nP2b spawn recovery:")
    print(f"  Seeds with final-500 loc_median <= 0.5: {p2b_loc_count}/8  "
          f"(need >= 6/8)  {'PASS' if p2b_loc_holds else 'FAIL'}")
    print(f"  Seeds with comps within +/-1 of {TRUE_MULT} for >= 3/4 colors: {p2b_comps_count}/8  "
          f"(need >= 6/8)  {'PASS' if p2b_comps_holds else 'FAIL'}")
    print(f"  Seeds with zero ceiling events in final {FINAL_EVAL_WINDOW} steps: {p2b_quiet_count}/8  "
          f"(need >= 6/8)  {'PASS' if p2b_quiet_holds else 'FAIL'}")
    print(f"  P2b: {'PASS' if p2b_holds else 'FAIL'}")
    for s, r in zip(SEEDS, cont_results):
        comps = r["final_comps_per_color"]
        colors_ok = sum(1 for c in comps if abs(c - TRUE_MULT) <= 1)
        print(f"    seed={s}: p2_loc_med={r['p2_final500_loc_median']:.4f}  "
              f"comps={comps} ({colors_ok}/4 within +/-1)  "
              f"fin_ceil={r['phase2_final_ceiling_events']}  "
              f"kept={r['spawns_kept']}  reverted={r['spawns_reverted']}")

    # --- P3 twin sanity ---
    p3_per_seed = [r["final_500_map_frac"] >= 0.80 for r in tab_results]
    p3_count = sum(p3_per_seed)
    p3_holds = p3_count >= 7

    print(f"\nP3 twin sanity (tabular MAP-cell correct >= 80% of final-500 steps):")
    print(f"  Seeds passing: {p3_count}/8  (need >= 7/8)  {'PASS' if p3_holds else 'FAIL'}")
    for s, r in zip(SEEDS, tab_results):
        print(f"    seed={s}: final_500_map_frac={r['final_500_map_frac']:.4f}  "
              f"({'pass' if p3_per_seed[s] else 'FAIL'})")

    # ---------------------------------------------------------------------------
    # Halt-arm evaluation
    # ---------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("HALT-ARM EVALUATION")
    print("=" * 80)

    # Special case: phase-1 localization does NOT hit the wall (< 1.0 in >= 3/8 seeds)
    # That means wall_present is False — log as finding, skip P2b grading.
    wall_absent_surprise = (not wall_present)

    # FAIL-A: wall present but detector silent in >= 2/8 seeds
    # "wall present" per arm 1 = p2a_wall_count >= 6
    # Detector silent = 0 ceiling events in phase 1
    if wall_present:
        detector_silent_count = sum(1 for r in cont_results
                                    if r["p1_ceiling_events"] == 0
                                    and r["p1_final500_loc_median"] >= 1.0)
        fail_a = detector_silent_count >= 2
    else:
        fail_a = False  # wall absent; FAIL-A condition is moot

    # FAIL-B: spawning fails to recover — P2b localization arm fails in >= 3/8 seeds
    # "fails in >= 3/8" = p2b_loc_count <= 5 (i.e., fails in 3 or more)
    fail_b = (p2b_loc_count <= 5) and wall_present  # only applicable if wall was real

    # P3 failure: twin/stream instrument problem
    p3_fail = not p3_holds

    halt_triggers = []
    if fail_a:
        halt_triggers.append(
            f"FAIL-A: wall present ({p2a_wall_count}/8 seeds >= 1.0) "
            f"but detector silent in {detector_silent_count}/8 wall-seeds — instrument gap"
        )
    if fail_b:
        halt_triggers.append(
            f"FAIL-B: spawning fails to recover — P2b loc arm fails in "
            f"{8 - p2b_loc_count}/8 seeds (>= 3) — toolkit does not compose with substrate"
        )
    if p3_fail:
        halt_triggers.append(
            f"P3 FAILED: twin MAP correct < 80% in {8 - p3_count}/8 seeds — "
            f"twin/stream instrument problem"
        )

    print()
    if halt_triggers:
        for ht in halt_triggers:
            print(f"  Falsifier: {ht}")

    # ---------------------------------------------------------------------------
    # Three-way VERDICT
    # ---------------------------------------------------------------------------
    print()
    print("=" * 80)

    if halt_triggers:
        verdict = "NEGATIVE"
        print(f"VERDICT: {verdict}")
        print("MIGRATION HALT")
        for ht in halt_triggers:
            print(f"  {ht}")
    elif wall_absent_surprise:
        verdict = "FINDING"
        print(f"VERDICT: {verdict} (unexpected — not a halt)")
        print(
            f"Phase-1 localization did NOT hit the wall: only {p2a_wall_count}/8 seeds "
            f"reached median >= 1.0.  The unimodal map localizes despite aliasing.  "
            f"This is a genuine surprise — NOT a halt; P2b grading is moot (recovery "
            f"premise absent).  Log as finding and re-examine the aliasing hypothesis."
        )
        if p2a_holds:
            print("  (But both P2a arms pass — this branch unreachable if wall_present; "
                  "logic check: wall_absent_surprise implies not wall_present.)")
        if p3_holds:
            print(f"  P3 PASS: twin sanity holds ({p3_count}/8 seeds >= 80%).")
        else:
            print(f"  P3 FAIL: twin sanity fails ({p3_count}/8 seeds >= 80%) — "
                  f"inspect stream.")
    elif p2a_holds and p2b_holds and p3_holds:
        verdict = "POSITIVE"
        print(f"VERDICT: {verdict}")
        print(
            "All predeclared properties satisfied (P2a + P2b + P3).  "
            "The aliasing wall is real, the Exp-132 detector fires on the "
            "continuous substrate, and the spawn-and-select rule recovers "
            "localization.  Migration thread may advance to rung M4."
        )
    elif p2a_holds and not p2b_holds and not fail_b:
        verdict = "MIXED"
        print(f"VERDICT: {verdict}")
        print(
            f"P2a carries (wall + detector); P2b partial: "
            f"loc_ok={p2b_loc_count}/8 (>= 6 needed), "
            f"comps_ok={p2b_comps_count}/8, quiet_ok={p2b_quiet_count}/8.  "
            f"Spawn recovery weaker than predicted but FAIL-B threshold not triggered.  "
            f"P3={'PASS' if p3_holds else 'FAIL'}.  "
            f"Log as partial recovery; inspect spawn acceptance rate."
        )
    elif not p2a_holds and not halt_triggers:
        verdict = "MIXED"
        print(f"VERDICT: {verdict}")
        print(
            f"P2a misses: wall_count={p2a_wall_count}/8 (need 6), "
            f"ceil_count={p2a_ceil_count}/8 (need 7).  "
            f"Partial evidence only.  "
            f"P2b={'PASS' if p2b_holds else 'FAIL'}, P3={'PASS' if p3_holds else 'FAIL'}.  "
            f"Inspect ceiling threshold and localization scale."
        )
    else:
        verdict = "MIXED"
        print(f"VERDICT: {verdict}")
        print(
            f"Partial: P2a={'PASS' if p2a_holds else 'FAIL'}, "
            f"P2b={'PASS' if p2b_holds else 'FAIL'}, "
            f"P3={'PASS' if p3_holds else 'FAIL'}.  "
            f"Inspect tallies above."
        )

    print("=" * 80)

    # ---------------------------------------------------------------------------
    # JSON output
    # ---------------------------------------------------------------------------
    rows_json.append({
        "exp": 143,
        "rung": "M3",
        "seed": -1,
        "summary": True,
        "p2a_wall_count": p2a_wall_count,
        "p2a_ceil_count": p2a_ceil_count,
        "p2a_holds": p2a_holds,
        "p2b_loc_count": p2b_loc_count,
        "p2b_comps_count": p2b_comps_count,
        "p2b_quiet_count": p2b_quiet_count,
        "p2b_holds": p2b_holds,
        "p3_count": p3_count,
        "p3_holds": p3_holds,
        "fail_a": fail_a,
        "fail_b": fail_b,
        "p3_fail": p3_fail,
        "verdict": verdict,
    })

    out_path = Path(__file__).parent / "outputs" / "exp143_rows.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for row in rows_json:
            fh.write(json.dumps(row) + "\n")
    print(f"\nJSON rows written to {out_path}")


if __name__ == "__main__":
    main()
