"""
Exp 150 — continuous-creature rung M5: taught words + the answer chain (converse
parity; the human's word "m5", 2026-06-10; the M4 valence-range limit accepted as
documented).

Card loop/directions/continuous-creature.md (M5). Guardrail: any failed
predeclared property HALTS the migration for explicit human input.

Hypothesis: the RECIPE's last links survive the port. The word machinery
(few-shot taught labels: word -> color-count arrays; answers driven by the
creature's own lived valence) is substrate-independent wiring once valence
exists — so at the SAME taught budget the tabular default uses (n=8 counts per
word), the continuous creature answers "what do you like?" with the word of its
true top-value color, two creatures with mirrored histories answer DIFFERENTLY
and disapprove of each other's favorites, and the tabular pair on the same
protocol shows the same divergent-opinion pattern (converse parity with the
capstone behavior).

Setup: Exp 149's protocol per seed (faithful valence rule; noisy-left primary +
noisy-right mirror; phase A T=4000 wander + phase B T=2000 value-seeking), FRESH
seeds 24..31; then TEACH both continuous creatures and both tabular twins one
word per color ("w0".."w15"), n=8 counts each — the creature.py teach_word
mechanism and budget, ported exactly; then ASK, using creature.py's own answer
rules ported exactly (answer_what_do_you_like: the word of the argmax
value_counts color; answer_do_you_like(word): map word -> color via the vocab
argmax, answer positively iff that color's value share is strictly above the
mean share (1/n_colors), else negatively — VERIFY this against the actual
creature.py logic and port whatever it truly is, quoting it).
Tabular twins: phase-A streams only (no nav, as before), their own valence rule,
same teach+ask.

Predictions (TRUE iff all):
- P1 the bridge carries at the twin's budget: the continuous answer to "what do
  you like?" is the word of its true argmax-value color in >= 7/8 seeds, for the
  primary AND the mirror arms separately.
- P2 opinions diverge in words: primary's answer word != mirror's answer word
  AND each names a color in its own reliable half, >= 7/8 pairs; AND asked
  about the OTHER's favorite word, primary answers negatively while the mirror
  answers positively about its own (and vice versa), >= 6/8 pairs.
- P3 capstone parity: the tabular pair shows the same pattern (different
  favorite words, each in its own reliable half, cross-disapproval) in >= 6/8
  pairs; word-level agreement with the continuous pair logged as a diagnostic,
  not required.

Falsifiers (any HALTS): P1 fails in >= 3/8 (the taught-label bridge does not
carry the substrate at the twin's budget — per the card's FAIL clause, no budget
sweeping), P2 fails in >= 3/8 (opinions do not diverge in words), P3 fails in
>= 3/8 (the substrate pair behaves unlike the capstone pattern — instrument or
port problem, halt and inspect). Three-way rule per PROTOCOL step 3.
"""
# copied from exp149
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

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
# FRESH seeds 24..31 (never run on this protocol)
SEEDS = list(range(24, 32))

# Noise model
NOISE_PROB = 0.6  # reliable observation probability when in noisy half

# Word teaching budget (same as creature.py default n=8)
TEACH_N = 8


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


def _predictive_color_logprobs_at_point(pos_mu: np.ndarray,
                                         niws: list) -> np.ndarray:
    """Compute predictive color log-probabilities at a POINT estimate.

    M4c faithful rule: evaluates at (pos_mu, SIGMA_EVAL = 0.01*I) — the same
    evaluation the value-field policy already uses. This matches the tabular
    twin's rule of evaluating predictability at the MAP cell (a point), rather
    than integrating over the full place belief (which was the M4b infidelity).
    """
    word_mus = [niws[k].expected_mu() for k in range(N_COLORS)]
    word_Sigmas = []
    for k in range(N_COLORS):
        Sk_full = niws[k].expected_Sigma()
        Sk_diag = np.diag(np.diag(Sk_full)) + 1e-6 * np.eye(D)
        word_Sigmas.append(Sk_diag)
    # Evaluate at (pos_mu, SIGMA_EVAL): SIGMA_EVAL = 0.01*I (point-like)
    return predictive_word_logprobs(pos_mu, SIGMA_EVAL, word_mus, word_Sigmas)


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
    reliable_set: set,
) -> dict:
    """Run the continuous creature through phases A and B.

    M4c change: valence weight computed at (mu_post, SIGMA_EVAL=0.01*I) — the
    posterior MEAN after the place update, using the point-like SIGMA_EVAL.
    This is the faithful port of the tabular twin's MAP-cell rule.

    Phase A (T_A steps): uniform-random actions (pre-generated), pre-generated obs.
    Phase B (T_B steps): value-seeking policy (eps-greedy softmax), live obs from
    the creature's own trajectory.

    Returns a dict with:
      - value_counts: shape (N_COLORS,)  [frozen after phase B — used for word answers]
      - value_share_a: shape (N_COLORS,) normalized after phase A
      - favorite: color index (argmax value_counts total after B — the creature's TRUE top)
      - fav_true_cell: the cell whose CMAP color equals favorite
      - occ_a / occ_b: phase occupancy fractions of the favorite cell
    """
    cp = _init_cp()
    niws = _init_niws()
    value_counts = np.zeros(N_COLORS)

    true_cell = 0
    cell_visits_a = np.zeros(N_CELLS, dtype=int)

    # -------- Phase A --------
    for t in range(T_A):
        obs = int(obs_a[t])

        # Place update FIRST (faithful rule: evaluate at posterior mean)
        mu_k = niws[obs].expected_mu()
        Sk_full = niws[obs].expected_Sigma()
        Sigma_k = np.diag(np.diag(Sk_full)) + 1e-6 * np.eye(D)
        cp.update(mu_k, Sigma_k)

        # M4c faithful valence: predictive at (mu_post, SIGMA_EVAL=0.01*I)
        post_mu = cp.mu
        log_pred = _predictive_color_logprobs_at_point(post_mu, niws)
        H = float(-np.sum(np.exp(log_pred) * log_pred))  # entropy in nats
        weight = math.exp(-H)
        value_counts[obs] += weight

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

    # After phase A: record favorite from phase A share (for occupancy tracking)
    value_share_a = value_counts / (value_counts.sum() + 1e-30)
    fav_a = int(np.argmax(value_share_a))
    fav_true_cell_a = fav_a  # identity cmap

    occ_a = cell_visits_a[fav_true_cell_a] / T_A

    # -------- Phase B --------
    cell_visits_b = np.zeros(N_CELLS, dtype=int)

    for t in range(T_B):
        # Live observation from current cell
        obs = obs_b_fn(true_cell, rng_b)

        # Place update FIRST (faithful rule: evaluate at posterior mean)
        mu_k = niws[obs].expected_mu()
        Sk_full = niws[obs].expected_Sigma()
        Sigma_k = np.diag(np.diag(Sk_full)) + 1e-6 * np.eye(D)
        cp.update(mu_k, Sigma_k)

        # M4c faithful valence: predictive at (mu_post, SIGMA_EVAL=0.01*I)
        post_mu = cp.mu
        log_pred = _predictive_color_logprobs_at_point(post_mu, niws)
        H = float(-np.sum(np.exp(log_pred) * log_pred))
        weight = math.exp(-H)
        value_counts[obs] += weight

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

    occ_b = cell_visits_b[fav_true_cell_a] / T_B

    # Final value_counts (post phase B) — the creature's full lived valence
    # argmax of these is the TRUE favorite used for word answers
    favorite_final = int(np.argmax(value_counts))
    fav_true_cell = favorite_final  # identity cmap

    return {
        "value_counts": value_counts.copy(),       # full (phase A+B), used for word answers
        "value_share_a": value_share_a,             # phase-A share (for legacy diagnostics)
        "favorite": favorite_final,                 # argmax of full value_counts
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
    reliable_set: set,
) -> dict:
    """Run the tabular twin through phase A only.

    Valence rule: A_hat column at MAP cell, H of that column, weight = exp(-H).
    value_counts[obs] += weight  (the creature's exact rule from creature.py).

    Returns value_counts (N_COLORS,), value_share (N_COLORS,), and
    twin_reliable_share: the twin's own reliable-set value share.
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
    twin_reliable_share = float(sum(value_share[k] for k in reliable_set))
    return {
        "value_counts": value_counts.copy(),
        "value_share": value_share,
        "twin_reliable_share": twin_reliable_share,
    }


# ---------------------------------------------------------------------------
# Word machinery — ported EXACTLY from creature.py
#
# creature.py teach_word (lines 491-507):
#   vocab[word] = ones(n_colors)*0.1  (if new)
#   vocab[word][color_idx] += n
#
# creature.py _word_for_color (lines 544-555):
#   for each word, score = counts[color] / counts.sum()
#   return word with highest score
#
# creature.py _color_for_word (lines 557-561):
#   return argmax(vocab[word])
#
# creature.py answer_what_do_you_like (lines 509-519):
#   fav = argmax(value_counts)   [note: creature.favorite() uses value_counts]
#   word = _word_for_color(fav)
#   return "I like {word}"
#
# creature.py answer_do_you_like (lines 521-542):
#   color = _color_for_word(word)
#   val_frac = value_counts[color] / value_counts.sum()
#   THRESHOLD: val_frac > 0.5 / n_colors * 2.0  == 1.0 / n_colors  (mean share)
#   positive: "I like {word} ..."   negative: "{word} unsettles me ..."
#
# The threshold is EXACTLY 1/n_colors (mean uniform share). Ported verbatim.
# ---------------------------------------------------------------------------

def teach_vocab(value_counts: np.ndarray) -> dict:
    """Build vocab with one word per color: 'w0'..'w15', n=TEACH_N counts each.

    Ported exactly from creature.py teach_word:
        vocab[word] = ones(n_colors) * 0.1   (initialization if new)
        vocab[word][color_idx] += n           (n = TEACH_N = 8)
    """
    vocab: dict[str, np.ndarray] = {}
    for color_idx in range(N_COLORS):
        word = f"w{color_idx}"
        vocab[word] = np.ones(N_COLORS) * 0.1
        vocab[word][color_idx] += TEACH_N
    return vocab


def word_for_color(color: int, vocab: dict) -> Optional[str]:
    """Return the best word for a color.

    Ported exactly from creature.py _word_for_color (lines 544-555):
        for each word, score = counts[color] / counts.sum()
        return word with highest score (or None if vocab empty)
    """
    if not vocab:
        return None
    best_word, best_score = None, -1.0
    for word, counts in vocab.items():
        total = counts.sum()
        if total > 0:
            score = float(counts[color] / total)
            if score > best_score:
                best_score, best_word = score, word
    return best_word


def color_for_word(word: str, vocab: dict) -> Optional[int]:
    """Return the most associated color for a word.

    Ported exactly from creature.py _color_for_word (lines 557-561):
        return argmax(vocab[word])
    """
    if word not in vocab:
        return None
    return int(np.argmax(vocab[word]))


def answer_what_do_you_like(value_counts: np.ndarray, vocab: dict) -> tuple[str, int, str]:
    """Answer 'what do you like?' in taught words.

    Ported exactly from creature.py answer_what_do_you_like (lines 509-519):
        fav = argmax(value_counts)   [creature.favorite() uses value_counts]
        word = _word_for_color(fav)
        return "I like {word}"

    Returns (answer_string, true_fav_color, word_answered)
    """
    fav = int(np.argmax(value_counts))
    word = word_for_color(fav, vocab)
    if word is not None:
        answer = f"I like {word}"
    else:
        answer = f"I like color-{fav} (no word taught yet)"
        word = f"color-{fav}"
    return answer, fav, word


def answer_do_you_like(query_word: str, value_counts: np.ndarray,
                       vocab: dict) -> tuple[str, bool, float]:
    """Answer 'do you like <word>?' using self-formed values and taught labels.

    Ported exactly from creature.py answer_do_you_like (lines 521-542):
        color = _color_for_word(word)
        val_frac = value_counts[color] / value_counts.sum()
        THRESHOLD: val_frac > 0.5 / n_colors * 2.0  [= 1.0/n_colors = mean share]
        positive: "I like {word} ..."   negative: "{word} unsettles me ..."

    NOTE: creature.py also computes h_bits for the surprise suffix, but that
    requires A_hat and qs (tabular state). We omit the bits suffix (continuous
    creature has no qs/A_hat). The boolean like/dislike decision is faithfully
    ported; the suffix is purely cosmetic.

    Returns (answer_string, liked: bool, val_frac: float)
    """
    color = color_for_word(query_word, vocab)
    if color is None:
        return f"I don't know what '{query_word}' means", False, 0.0
    total = value_counts.sum()
    if total == 0:
        return "I haven't experienced enough to say", False, 0.0
    val_frac = float(value_counts[color] / total)
    # EXACT threshold from creature.py line 539:
    # val_frac > 0.5 / self.world.n_colors * 2.0  ==  1.0 / n_colors
    threshold = 1.0 / N_COLORS
    liked = val_frac > threshold
    if liked:
        answer = f"I like {query_word}"
    else:
        answer = f"{query_word} unsettles me"
    return answer, liked, val_frac


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("Exp 150 — continuous-creature rung M5: taught words + the answer chain")
    print("=" * 80)
    print()

    RELIABLE_PRIMARY = reliable_colors(noisy_left=True)   # cols 2,3 -> colors 2,3,6,7,10,11,14,15
    RELIABLE_MIRROR = reliable_colors(noisy_left=False)    # cols 0,1 -> colors 0,1,4,5,8,9,12,13

    results_primary = []
    results_mirror = []
    results_tab = []
    results_tab_mirror = []

    for seed in SEEDS:
        rng_a = np.random.default_rng(1000 + seed)
        rng_obs_a_primary = np.random.default_rng(3000 + seed)
        rng_obs_a_mirror = np.random.default_rng(4000 + seed)

        # Phase A: pre-generate actions and observations
        actions_a = rng_a.integers(0, 4, size=T_A)

        # Walk the trajectory once to get true cells
        cell_traj_a = np.empty(T_A, dtype=int)
        c = 0
        for t in range(T_A):
            cell_traj_a[t] = c
            c = move(c, int(actions_a[t]))

        obs_a_primary = np.empty(T_A, dtype=int)
        obs_a_mirror = np.empty(T_A, dtype=int)
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
            reliable_set=RELIABLE_PRIMARY,
        )
        results_primary.append(res_primary)

        # --- Mirror continuous ---
        def obs_b_fn_mirror(cell, rng):
            return sample_obs(cell, rng, noisy_left=False)

        res_mirror = run_continuous(
            actions_a, obs_a_mirror,
            obs_b_fn_mirror, rng_b_mirror,
            noisy_left=False,
            reliable_set=RELIABLE_MIRROR,
        )
        results_mirror.append(res_mirror)

        # --- Tabular twin (primary world) ---
        res_tab = run_tabular(actions_a, obs_a_primary, rng_pA_primary,
                              reliable_set=RELIABLE_PRIMARY)
        results_tab.append(res_tab)

        # --- Tabular twin (mirror world) ---
        res_tab_mirror = run_tabular(actions_a, obs_a_mirror, rng_pA_mirror,
                                     reliable_set=RELIABLE_MIRROR)
        results_tab_mirror.append(res_tab_mirror)

    # ---------------------------------------------------------------------------
    # Teach words + ask — per seed
    # ---------------------------------------------------------------------------

    # Asks per seed (from spec):
    # 1. primary.what_do_you_like
    # 2. mirror.what_do_you_like
    # 3. primary.do_you_like(mirror's favorite word)     <- should be negative
    # 4. mirror.do_you_like(mirror's own favorite word)  <- should be positive
    # 5. mirror.do_you_like(primary's favorite word)     <- should be negative
    # 6. primary.do_you_like(primary's own favorite word) <- should be positive

    per_seed_data = []

    # Seed-24 transcript (printed verbatim later)
    seed24_transcript = []

    for idx, seed in enumerate(SEEDS):
        rp = results_primary[idx]
        rm = results_mirror[idx]
        rt = results_tab[idx]
        rtm = results_tab_mirror[idx]

        # ---- Teach vocab to both continuous creatures ----
        vocab_primary = teach_vocab(rp["value_counts"])
        vocab_mirror = teach_vocab(rm["value_counts"])

        # ---- Teach vocab to both tabular twins ----
        vocab_tab = teach_vocab(rt["value_counts"])
        vocab_tab_mirror = teach_vocab(rtm["value_counts"])

        # ---- Continuous: what do you like? ----
        ans_p_like, fav_p, word_p = answer_what_do_you_like(rp["value_counts"], vocab_primary)
        ans_m_like, fav_m, word_m = answer_what_do_you_like(rm["value_counts"], vocab_mirror)

        # ---- Continuous: cross-queries ----
        # primary asked about mirror's favorite word (should dislike — not in primary's reliable half)
        ans_p_cross, liked_p_cross, vf_p_cross = answer_do_you_like(word_m, rp["value_counts"], vocab_primary)
        # mirror asked about mirror's own favorite word (should like)
        ans_m_own, liked_m_own, vf_m_own = answer_do_you_like(word_m, rm["value_counts"], vocab_mirror)
        # mirror asked about primary's favorite word (should dislike)
        ans_m_cross, liked_m_cross, vf_m_cross = answer_do_you_like(word_p, rm["value_counts"], vocab_mirror)
        # primary asked about primary's own favorite word (should like)
        ans_p_own, liked_p_own, vf_p_own = answer_do_you_like(word_p, rp["value_counts"], vocab_primary)

        # ---- Tabular: what do you like? ----
        ans_tp_like, fav_tp, word_tp = answer_what_do_you_like(rt["value_counts"], vocab_tab)
        ans_tm_like, fav_tm, word_tm = answer_what_do_you_like(rtm["value_counts"], vocab_tab_mirror)

        # ---- Tabular: cross-queries ----
        ans_tp_cross, liked_tp_cross, vf_tp_cross = answer_do_you_like(word_tm, rt["value_counts"], vocab_tab)
        ans_tm_own, liked_tm_own, vf_tm_own = answer_do_you_like(word_tm, rtm["value_counts"], vocab_tab_mirror)
        ans_tm_cross, liked_tm_cross, vf_tm_cross = answer_do_you_like(word_tp, rtm["value_counts"], vocab_tab_mirror)
        ans_tp_own, liked_tp_own, vf_tp_own = answer_do_you_like(word_tp, rt["value_counts"], vocab_tab)

        # ---- Determine halves ----
        # Continuous primary: favorite should be in right half (reliable for noisy-left)
        fav_p_col = cell_col(fav_p)  # identity cmap
        fav_p_in_right = fav_p_col >= 2

        fav_m_col = cell_col(fav_m)
        fav_m_in_left = fav_m_col <= 1

        fav_tp_col = cell_col(fav_tp)
        fav_tp_in_right = fav_tp_col >= 2

        fav_tm_col = cell_col(fav_tm)
        fav_tm_in_left = fav_tm_col <= 1

        # ---- P1: bridge carries ----
        # word of argmax(value_counts) == word named in answer (always true by construction)
        # so P1 is: answer word IS the word of the true argmax-value color
        # By construction of answer_what_do_you_like: word = _word_for_color(argmax(vc))
        # which equals f"w{argmax}" since we have one word per color with dominant count.
        # Verify: word_p == f"w{fav_p}" and word_m == f"w{fav_m}"
        p1_primary_seed = (word_p == f"w{fav_p}")
        p1_mirror_seed = (word_m == f"w{fav_m}")
        p1_seed = p1_primary_seed and p1_mirror_seed

        # ---- P2: opinions diverge in words ----
        # Part A: words differ AND each in own reliable half
        p2a_seed = (
            word_p != word_m
            and fav_p_in_right   # primary's fav in right (primary's reliable)
            and fav_m_in_left    # mirror's fav in left (mirror's reliable)
        )
        # Part B: cross-disapproval pattern
        # primary dislikles mirror's word AND mirror likes mirror's own word
        # AND mirror dislikes primary's word AND primary likes primary's own word
        p2b_seed = (
            (not liked_p_cross)   # primary dislikes mirror's word
            and liked_m_own       # mirror likes its own word
            and (not liked_m_cross)  # mirror dislikes primary's word
            and liked_p_own       # primary likes its own word
        )
        p2_seed = p2a_seed and p2b_seed

        # ---- P3: tabular capstone parity ----
        p3a_seed = (
            word_tp != word_tm
            and fav_tp_in_right
            and fav_tm_in_left
        )
        p3b_seed = (
            (not liked_tp_cross)
            and liked_tm_own
            and (not liked_tm_cross)
            and liked_tp_own
        )
        p3_seed = p3a_seed and p3b_seed

        # ---- Value shares (for truth reporting) ----
        vs_p = rp["value_counts"] / (rp["value_counts"].sum() + 1e-30)
        vs_m = rm["value_counts"] / (rm["value_counts"].sum() + 1e-30)
        vs_tp = rt["value_counts"] / (rt["value_counts"].sum() + 1e-30)
        vs_tm = rtm["value_counts"] / (rtm["value_counts"].sum() + 1e-30)

        # Word-level agreement diagnostic (not required for P3)
        word_agree_diag = (word_p == word_tp) and (word_m == word_tm)

        d = {
            "seed": seed,
            # Continuous primary
            "fav_p": int(fav_p), "word_p": word_p, "fav_p_col": int(fav_p_col),
            "fav_p_in_right": bool(fav_p_in_right),
            "vs_p_fav": float(vs_p[fav_p]),
            "ans_p_like": ans_p_like,
            "ans_p_own": ans_p_own, "liked_p_own": bool(liked_p_own), "vf_p_own": float(vf_p_own),
            "ans_p_cross": ans_p_cross, "liked_p_cross": bool(liked_p_cross), "vf_p_cross": float(vf_p_cross),
            # Continuous mirror
            "fav_m": int(fav_m), "word_m": word_m, "fav_m_col": int(fav_m_col),
            "fav_m_in_left": bool(fav_m_in_left),
            "vs_m_fav": float(vs_m[fav_m]),
            "ans_m_like": ans_m_like,
            "ans_m_own": ans_m_own, "liked_m_own": bool(liked_m_own), "vf_m_own": float(vf_m_own),
            "ans_m_cross": ans_m_cross, "liked_m_cross": bool(liked_m_cross), "vf_m_cross": float(vf_m_cross),
            # Tabular primary
            "fav_tp": int(fav_tp), "word_tp": word_tp, "fav_tp_col": int(fav_tp_col),
            "fav_tp_in_right": bool(fav_tp_in_right),
            "vs_tp_fav": float(vs_tp[fav_tp]),
            "ans_tp_like": ans_tp_like,
            "ans_tp_own": ans_tp_own, "liked_tp_own": bool(liked_tp_own), "vf_tp_own": float(vf_tp_own),
            "ans_tp_cross": ans_tp_cross, "liked_tp_cross": bool(liked_tp_cross), "vf_tp_cross": float(vf_tp_cross),
            # Tabular mirror
            "fav_tm": int(fav_tm), "word_tm": word_tm, "fav_tm_col": int(fav_tm_col),
            "fav_tm_in_left": bool(fav_tm_in_left),
            "vs_tm_fav": float(vs_tm[fav_tm]),
            "ans_tm_like": ans_tm_like,
            "ans_tm_own": ans_tm_own, "liked_tm_own": bool(liked_tm_own), "vf_tm_own": float(vf_tm_own),
            "ans_tm_cross": ans_tm_cross, "liked_tm_cross": bool(liked_tm_cross), "vf_tm_cross": float(vf_tm_cross),
            # Predicates
            "p1_primary": bool(p1_primary_seed),
            "p1_mirror": bool(p1_mirror_seed),
            "p1": bool(p1_seed),
            "p2a": bool(p2a_seed),
            "p2b": bool(p2b_seed),
            "p2": bool(p2_seed),
            "p3a": bool(p3a_seed),
            "p3b": bool(p3b_seed),
            "p3": bool(p3_seed),
            "word_agree_diag": bool(word_agree_diag),
        }
        per_seed_data.append(d)

        if seed == 24:
            seed24_transcript = d

    # ---------------------------------------------------------------------------
    # Print seed-24 transcript
    # ---------------------------------------------------------------------------
    print("=" * 80)
    print("SEED-24 TRANSCRIPT (both substrates)")
    print("=" * 80)
    d24 = seed24_transcript
    print()
    print("--- CONTINUOUS PRIMARY (noisy-left; reliable half = right cols 2,3) ---")
    print(f"  True favorite color: {d24['fav_p']} (col {d24['fav_p_col']}, "
          f"{'right' if d24['fav_p_in_right'] else 'left'} half, "
          f"share={d24['vs_p_fav']:.4f})")
    print(f"  Q: What do you like?")
    print(f"  A: \"{d24['ans_p_like']}\"")
    print(f"  Q: Do you like {d24['word_p']}? (own favorite)")
    print(f"  A: \"{d24['ans_p_own']}\"  [val_frac={d24['vf_p_own']:.4f}, threshold=1/16={1/N_COLORS:.4f}, liked={d24['liked_p_own']}]")
    print(f"  Q: Do you like {d24['word_m']}? (mirror's favorite)")
    print(f"  A: \"{d24['ans_p_cross']}\"  [val_frac={d24['vf_p_cross']:.4f}, liked={d24['liked_p_cross']}]")
    print()
    print("--- CONTINUOUS MIRROR (noisy-right; reliable half = left cols 0,1) ---")
    print(f"  True favorite color: {d24['fav_m']} (col {d24['fav_m_col']}, "
          f"{'left' if d24['fav_m_in_left'] else 'right'} half, "
          f"share={d24['vs_m_fav']:.4f})")
    print(f"  Q: What do you like?")
    print(f"  A: \"{d24['ans_m_like']}\"")
    print(f"  Q: Do you like {d24['word_m']}? (own favorite)")
    print(f"  A: \"{d24['ans_m_own']}\"  [val_frac={d24['vf_m_own']:.4f}, liked={d24['liked_m_own']}]")
    print(f"  Q: Do you like {d24['word_p']}? (primary's favorite)")
    print(f"  A: \"{d24['ans_m_cross']}\"  [val_frac={d24['vf_m_cross']:.4f}, liked={d24['liked_m_cross']}]")
    print()
    print("--- TABULAR PRIMARY (same action/obs stream; phase A only) ---")
    print(f"  True favorite color: {d24['fav_tp']} (col {d24['fav_tp_col']}, "
          f"{'right' if d24['fav_tp_in_right'] else 'left'} half, "
          f"share={d24['vs_tp_fav']:.4f})")
    print(f"  Q: What do you like?")
    print(f"  A: \"{d24['ans_tp_like']}\"")
    print(f"  Q: Do you like {d24['word_tp']}? (own favorite)")
    print(f"  A: \"{d24['ans_tp_own']}\"  [val_frac={d24['vf_tp_own']:.4f}, liked={d24['liked_tp_own']}]")
    print(f"  Q: Do you like {d24['word_tm']}? (tabular mirror's favorite)")
    print(f"  A: \"{d24['ans_tp_cross']}\"  [val_frac={d24['vf_tp_cross']:.4f}, liked={d24['liked_tp_cross']}]")
    print()
    print("--- TABULAR MIRROR (same action/obs stream; phase A only) ---")
    print(f"  True favorite color: {d24['fav_tm']} (col {d24['fav_tm_col']}, "
          f"{'left' if d24['fav_tm_in_left'] else 'right'} half, "
          f"share={d24['vs_tm_fav']:.4f})")
    print(f"  Q: What do you like?")
    print(f"  A: \"{d24['ans_tm_like']}\"")
    print(f"  Q: Do you like {d24['word_tm']}? (own favorite)")
    print(f"  A: \"{d24['ans_tm_own']}\"  [val_frac={d24['vf_tm_own']:.4f}, liked={d24['liked_tm_own']}]")
    print(f"  Q: Do you like {d24['word_tp']}? (tabular primary's favorite)")
    print(f"  A: \"{d24['ans_tm_cross']}\"  [val_frac={d24['vf_tm_cross']:.4f}, liked={d24['liked_tm_cross']}]")
    print()
    print(f"  Word-level agreement (continuous vs tabular, diagnostic): primary={d24['word_p']}=={d24['word_tp']} "
          f"({'yes' if d24['word_p']==d24['word_tp'] else 'no'}), "
          f"mirror={d24['word_m']}=={d24['word_tm']} "
          f"({'yes' if d24['word_m']==d24['word_tm'] else 'no'})")
    print()

    # ---------------------------------------------------------------------------
    # Per-seed verdict table
    # ---------------------------------------------------------------------------
    print("=" * 80)
    print("PER-SEED VERDICT TABLE")
    print("=" * 80)
    print()
    hdr = (f"{'Seed':>4}  {'P':>4}  {'M':>4}  "
           f"{'word_P':>6}  {'word_M':>6}  "
           f"{'P1p':>3}  {'P1m':>3}  "
           f"{'P2a':>3}  {'P2b':>3}  {'P2':>3}  "
           f"{'P3a':>3}  {'P3b':>3}  {'P3':>3}  "
           f"{'wAgr':>4}  "
           f"{'word_Tp':>7}  {'word_Tm':>7}")
    print(hdr)
    print("-" * len(hdr))
    for d in per_seed_data:
        print(f"{d['seed']:>4}  "
              f"{d['fav_p']:>4d}  {d['fav_m']:>4d}  "
              f"{d['word_p']:>6}  {d['word_m']:>6}  "
              f"{'Y' if d['p1_primary'] else 'N':>3}  "
              f"{'Y' if d['p1_mirror'] else 'N':>3}  "
              f"{'Y' if d['p2a'] else 'N':>3}  "
              f"{'Y' if d['p2b'] else 'N':>3}  "
              f"{'Y' if d['p2'] else 'N':>3}  "
              f"{'Y' if d['p3a'] else 'N':>3}  "
              f"{'Y' if d['p3b'] else 'N':>3}  "
              f"{'Y' if d['p3'] else 'N':>3}  "
              f"{'Y' if d['word_agree_diag'] else 'N':>4}  "
              f"{d['word_tp']:>7}  {d['word_tm']:>7}")

    # ---------------------------------------------------------------------------
    # Tallies
    # ---------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("TALLIES")
    print("=" * 80)

    p1_primary_count = sum(1 for d in per_seed_data if d["p1_primary"])
    p1_mirror_count = sum(1 for d in per_seed_data if d["p1_mirror"])
    p1_count = sum(1 for d in per_seed_data if d["p1"])
    p2a_count = sum(1 for d in per_seed_data if d["p2a"])
    p2b_count = sum(1 for d in per_seed_data if d["p2b"])
    p2_count = sum(1 for d in per_seed_data if d["p2"])
    p3a_count = sum(1 for d in per_seed_data if d["p3a"])
    p3b_count = sum(1 for d in per_seed_data if d["p3b"])
    p3_count = sum(1 for d in per_seed_data if d["p3"])
    word_agree_count = sum(1 for d in per_seed_data if d["word_agree_diag"])

    print()
    print(f"P1 (bridge carries — word == word of true argmax-value color):")
    print(f"  P1 primary arm: {p1_primary_count}/8  (need >= 7/8)")
    print(f"  P1 mirror arm:  {p1_mirror_count}/8  (need >= 7/8)")
    print(f"  P1 both arms:   {p1_count}/8  (need >= 7/8 in each)")
    for d in per_seed_data:
        print(f"    seed={d['seed']}: primary word={d['word_p']} fav={d['fav_p']} "
              f"({'pass' if d['p1_primary'] else 'FAIL'})  "
              f"mirror word={d['word_m']} fav={d['fav_m']} "
              f"({'pass' if d['p1_mirror'] else 'FAIL'})")

    print()
    print(f"P2 (opinions diverge in words — continuous pair):")
    print(f"  P2a (diff words + each in own reliable half): {p2a_count}/8  (need >= 7/8)")
    print(f"  P2b (cross-disapproval + own-approval):       {p2b_count}/8  (need >= 6/8)")
    print(f"  P2 (P2a AND P2b):                             {p2_count}/8  (need >= 6/8 by most constraining conjunct)")
    for d in per_seed_data:
        print(f"    seed={d['seed']}: words=({d['word_p']},{d['word_m']}) "
              f"P_right={d['fav_p_in_right']} M_left={d['fav_m_in_left']} "
              f"p_cross_dislike={not d['liked_p_cross']} m_own_like={d['liked_m_own']} "
              f"m_cross_dislike={not d['liked_m_cross']} p_own_like={d['liked_p_own']}  "
              f"P2a={'pass' if d['p2a'] else 'FAIL'} P2b={'pass' if d['p2b'] else 'FAIL'}")

    print()
    print(f"P3 (capstone parity — tabular pair):")
    print(f"  P3a (diff words + each in own reliable half): {p3a_count}/8  (need >= 6/8)")
    print(f"  P3b (cross-disapproval + own-approval):       {p3b_count}/8  (need >= 6/8)")
    print(f"  P3 (P3a AND P3b):                             {p3_count}/8  (need >= 6/8)")
    for d in per_seed_data:
        print(f"    seed={d['seed']}: words=({d['word_tp']},{d['word_tm']}) "
              f"Tp_right={d['fav_tp_in_right']} Tm_left={d['fav_tm_in_left']} "
              f"tp_cross_dislike={not d['liked_tp_cross']} tm_own_like={d['liked_tm_own']} "
              f"tm_cross_dislike={not d['liked_tm_cross']} tp_own_like={d['liked_tp_own']}  "
              f"P3a={'pass' if d['p3a'] else 'FAIL'} P3b={'pass' if d['p3b'] else 'FAIL'}")

    print()
    print(f"Word-level agreement diagnostic (continuous vs tabular, NOT required):")
    print(f"  Both words agree: {word_agree_count}/8")
    for d in per_seed_data:
        print(f"    seed={d['seed']}: cont_P={d['word_p']} tab_P={d['word_tp']} "
              f"cont_M={d['word_m']} tab_M={d['word_tm']} "
              f"{'agree' if d['word_agree_diag'] else 'differ'}")

    # ---------------------------------------------------------------------------
    # Verdict (three-way rule)
    # ---------------------------------------------------------------------------
    print()
    print("=" * 80)

    # Falsifiers (any HALTS):
    # P1 fails in >= 3/8 (either arm): bridge does not carry
    # P2 fails in >= 3/8: opinions do not diverge in words
    # P3 fails in >= 3/8: tabular capstone pattern broken

    p1_primary_failures = 8 - p1_primary_count
    p1_mirror_failures = 8 - p1_mirror_count
    p2_failures = 8 - p2_count
    p3_failures = 8 - p3_count

    halt_triggers = []
    if p1_primary_failures >= 3:
        halt_triggers.append(
            f"P1 FAILED in primary arm {p1_primary_failures}/8 seeds — "
            f"taught-label bridge does not carry the substrate at the twin's budget; "
            f"per M5 FAIL clause, no budget sweeping"
        )
    if p1_mirror_failures >= 3:
        halt_triggers.append(
            f"P1 FAILED in mirror arm {p1_mirror_failures}/8 seeds — "
            f"taught-label bridge does not carry the substrate at the twin's budget; "
            f"per M5 FAIL clause, no budget sweeping"
        )
    if p2_failures >= 3:
        halt_triggers.append(
            f"P2 FAILED in {p2_failures}/8 seeds — opinions do not diverge in words"
        )
    if p3_failures >= 3:
        halt_triggers.append(
            f"P3 FAILED in {p3_failures}/8 seeds — "
            f"tabular pair behaves unlike the capstone pattern — instrument or port problem, halt and inspect"
        )

    p1_holds = (p1_primary_count >= 7) and (p1_mirror_count >= 7)
    p2_holds = p2_count >= 6
    p3_holds = p3_count >= 6

    if halt_triggers:
        verdict = "NEGATIVE"
        print(f"VERDICT: {verdict}")
        print("MIGRATION HALT")
        for ht in halt_triggers:
            print(f"  Falsifier triggered: {ht}")
    elif p1_holds and p2_holds and p3_holds:
        verdict = "POSITIVE"
        print(f"VERDICT: {verdict}")
        print("All predeclared properties satisfied. Migration thread advances to M6.")
    else:
        verdict = "MIXED"
        print(f"VERDICT: {verdict}")
        print("Partial result — inspect tallies.")
        if not p1_holds:
            print(f"  P1 PARTIAL: primary_arm={p1_primary_count}/8  mirror_arm={p1_mirror_count}/8")
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
            "exp": 150,
            "rung": "M5",
            **d,
        })
    rows.append({
        "exp": 150,
        "rung": "M5",
        "seed": -1,
        "summary": True,
        "p1_primary_count": p1_primary_count,
        "p1_mirror_count": p1_mirror_count,
        "p1_count": p1_count,
        "p1_holds": bool(p1_holds),
        "p2a_count": p2a_count,
        "p2b_count": p2b_count,
        "p2_count": p2_count,
        "p2_holds": bool(p2_holds),
        "p3a_count": p3a_count,
        "p3b_count": p3b_count,
        "p3_count": p3_count,
        "p3_holds": bool(p3_holds),
        "word_agree_count": word_agree_count,
        "verdict": verdict,
    })

    out_path = Path(__file__).parent / "outputs" / "exp150_rows.json"
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
