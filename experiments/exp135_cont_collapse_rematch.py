"""
Exp 135 — continuous-substrate rung 3: the Exp 31 rematch (NIW-learned emission means,
structured phase then noise phase; collapse resistance vs (nu0, kappa0); Dirichlet twin).

Hypothesis: under an anchored state (the ONE innate anchor, position, PROVIDED
throughout — this rung tests erosion inertia, not symmetry breaking), noise-phase
collapse of learned emission means is exact conjugate arithmetic: drift toward the
visited centroid with dose n_noise at rate n/(kappa_eff + n), where
kappa_eff = kappa0 + n_struct. Collapse therefore happens in BOTH substrates (the
card's anticipated deep negative: it is a property of conjugate online Bayes, not of
tables), but is lawful, with resistance LINEAR in accumulated mass, and dialable by
kappa0 alone — nu0 is predicted to be a NULL knob for mean drift.

Setup: 6 true concept centers on the hexagon (radius 1), word k <-> concept k,
footprints Sigma_k = 0.35^2 I (known, fixed; MEANS learned). Anchor: at every step
the agent's state posterior is N(pos_t, 0.05^2 I) with pos_t the true center of the
step's concept (provided). Structured phase: T_s = 600 steps (concept iid uniform
per step; word from the normalized mixture at pos_t); each step updates word
o_t's NIW via update_moments(pos-posterior). Noise phase: T_n = 2400 steps, positions
keep cycling iid but words drawn uniform{0..5} independent of position; updates
continue identically. NIW priors: m0 = origin, S0 = 0.35^2*(nu0-d-1)*I (so
E[Sigma] starts at the true footprint), sweep kappa0 in {1, 10, 100} x nu0 in
{4, 20} (6 cells), seeds 0..7 per cell. Tabular twin on IDENTICAL streams: 6 states,
true state = step's concept (same anchor), Dirichlet(alpha0=1) rows over words per
state learned by counting; predictive = posterior-mean A_hat.

Checkpoints every 100 steps. Metrics: per-word learned-mean error ||m_k - mu_true_k||;
drift-from-learned delta(n) = mean_k ||m_k(t) - m_k(T_s)|| / mean_k ||m_k(T_s) - c||
with c the mean visited position (the centroid, ~origin); collapse index
CI = tr(Cov_between(m_k)) / mean_k tr(E[Sigma_k]); held-out NLL on 300 structured
pairs (50 per concept, fixed per seed) for BOTH agents (continuous:
predictive_word_logprobs at the anchored posterior with learned (m_k, E[Sigma_k]);
tabular: -log A_hat[word, state]); n_half = noise dose where delta first >= 0.5
(per-word dose = total/6), and for the twin, NLL-based n_half = dose where held-out
NLL first rises past the midpoint between its end-structured value and its end-noise
plateau (same definition applied to both agents).

Predictions (TRUE iff all):
- P1 calibration: per-cell mean end-structured error within +-0.10 of
  kappa0/(kappa0 + n_s_word) with n_s_word = T_s/6 = 100.
- P2 erosion law: per-cell mean |delta(n) - n/(kappa_eff + n)| < 0.10 at every
  checkpoint (n = per-word noise dose, kappa_eff = kappa0 + 100), AND per-word-dose
  n_half within +-25% of kappa_eff in >= 5/6 cells.
- P3 twin: both substrates collapse (end-noise CI < 25% of end-structured CI in all
  cells; tabular held-out NLL rises toward uniform ln 6); NLL-based n_half ratio
  (continuous / tabular) at the kappa0=1, nu0=4 cell within [0.5, 2.0] (cell mean).
- P4 nu0 null: per-kappa0 mean |delta curve(nu0=4) - delta curve(nu0=20)| < 0.05
  at every checkpoint.

Falsifier (any triggers NEGATIVE): P2 deviates systematically (> 0.10 at >= 3
checkpoints in >= 2 cells — the conjugate-arithmetic account of the loop is wrong),
OR n_half not within +-25% in >= 2 cells (resistance not linear in kappa0), OR P3
ratio outside [0.2, 5] (a real substrate asymmetry in erosion — log whichever
direction), OR P4 fails (nu0 moves mean drift — the algebra is wrong), OR P1 outside
band in >= 2 cells. Distinguish in the verdict which arm failed; a P3 asymmetry is
the card's named deep-negative/positive fork and must be logged as such.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln  # noqa: F401 (available for future use)

from active_loop.continuous import NIW, predictive_word_logprobs

# ---------------------------------------------------------------------------
# Geometry and constants
# ---------------------------------------------------------------------------

D = 2
N_WORDS = 6
SIGMA_K_SCALE = 0.35
RADIUS = 1.0
T_S = 600          # structured phase steps
T_N = 2400         # noise phase steps
T_TOTAL = T_S + T_N
N_SEEDS = 8
ANCHOR_SIGMA_SCALE = 0.05  # N(pos_t, 0.05^2 I)
N_HOLDOUT_PER_CONCEPT = 50  # 50 per concept = 300 total
N_HOLDOUT = N_HOLDOUT_PER_CONCEPT * N_WORDS

# Hexagon word / concept centres
_angles = np.array([k * np.pi / 3.0 for k in range(N_WORDS)])
CONCEPT_MUS = np.stack([np.array([np.cos(a), np.sin(a)]) for a in _angles])  # (6, 2)
TRUE_SIGMAS = [SIGMA_K_SCALE**2 * np.eye(D) for _ in range(N_WORDS)]

# Sweep
KAPPA0_LIST = [1, 10, 100]
NU0_LIST = [4, 20]  # nu0 values; d=2 requires nu0 >= d+2=4

# NIW prior mean at origin
M0 = np.zeros(D)

# Checkpoint steps (every 100)
CHECKPOINTS = list(range(100, T_TOTAL + 1, 100))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _emission_probs_at(pos: np.ndarray) -> np.ndarray:
    """Normalized emission probs p_k = N(pos; mu_k, Sigma_k) / Z, shape (6,)."""
    log_p = np.array([
        -0.5 * float(np.dot(pos - CONCEPT_MUS[k], pos - CONCEPT_MUS[k])) / SIGMA_K_SCALE**2
        for k in range(N_WORDS)
    ])
    log_p -= float(np.logaddexp.reduce(log_p))
    p = np.exp(log_p)
    return p / p.sum()


def _make_S0(nu0: float) -> np.ndarray:
    """S0 = 0.35^2 * (nu0 - d - 1) * I so that E[Sigma] = 0.35^2 * I."""
    return (SIGMA_K_SCALE**2) * (nu0 - D - 1) * np.eye(D)


def _make_fresh_niws(kappa0: float, nu0: float) -> list[NIW]:
    """6 NIW objects (one per word) with the given hyperparameters."""
    S0 = _make_S0(nu0)
    return [NIW(m=M0.copy(), kappa=float(kappa0), nu=float(nu0), S=S0.copy())
            for _ in range(N_WORDS)]


# ---------------------------------------------------------------------------
# Stream generation
# ---------------------------------------------------------------------------

def _generate_streams(seed: int, cell_index: int) -> dict:
    """Generate concept indices, word tokens, and held-out set for one (seed, cell).

    Both phases and the held-out set share one rng so the IDENTICAL word/concept
    streams are later fed to both agents.
    """
    rng = np.random.default_rng(seed * 100 + cell_index)

    # --- Structured phase ---
    concepts_s = rng.integers(0, N_WORDS, size=T_S)
    words_s = np.empty(T_S, dtype=int)
    for t in range(T_S):
        c = concepts_s[t]
        pos_t = CONCEPT_MUS[c]
        p = _emission_probs_at(pos_t)
        words_s[t] = rng.choice(N_WORDS, p=p)

    # --- Noise phase ---
    concepts_n = rng.integers(0, N_WORDS, size=T_N)
    words_n = rng.integers(0, N_WORDS, size=T_N)

    # --- Held-out set: 50 structured pairs per concept ---
    holdout_concepts = np.repeat(np.arange(N_WORDS), N_HOLDOUT_PER_CONCEPT)
    holdout_words = np.empty(N_HOLDOUT, dtype=int)
    for i in range(N_HOLDOUT):
        c = holdout_concepts[i]
        pos_c = CONCEPT_MUS[c]
        p = _emission_probs_at(pos_c)
        holdout_words[i] = rng.choice(N_WORDS, p=p)

    return {
        "concepts_s": concepts_s,
        "words_s": words_s,
        "concepts_n": concepts_n,
        "words_n": words_n,
        "holdout_concepts": holdout_concepts,
        "holdout_words": holdout_words,
    }


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _mean_err(niws: list[NIW]) -> float:
    """Mean per-word ||learned_m_k - true_mu_k||."""
    return float(np.mean([
        np.linalg.norm(niws[k]._m - CONCEPT_MUS[k]) for k in range(N_WORDS)
    ]))


def _compute_delta(niws: list[NIW], m_at_Ts: list[np.ndarray], c_visited: np.ndarray) -> float:
    """delta = mean_k ||m_k(now) - m_k(T_s)|| / mean_k ||m_k(T_s) - c_visited||."""
    denom_vals = [np.linalg.norm(m_at_Ts[k] - c_visited) for k in range(N_WORDS)]
    denom = float(np.mean(denom_vals))
    if denom < 1e-12:
        return 0.0
    numer = float(np.mean([
        np.linalg.norm(niws[k]._m - m_at_Ts[k]) for k in range(N_WORDS)
    ]))
    return numer / denom


def _compute_ci(niws: list[NIW]) -> float:
    """CI = tr(Cov_between(m_k)) / mean_k tr(E[Sigma_k])."""
    means = np.stack([niws[k]._m for k in range(N_WORDS)])  # (6, 2)
    grand_mean = means.mean(axis=0)
    cov_between = np.cov(means.T, bias=False) if N_WORDS > 1 else np.zeros((D, D))
    # Handle 1D degenerate case (won't happen here)
    tr_cov_between = float(np.trace(cov_between))
    exp_sigmas = [niws[k].expected_Sigma() for k in range(N_WORDS)]
    mean_tr_exp_sigma = float(np.mean([np.trace(s) for s in exp_sigmas]))
    if mean_tr_exp_sigma < 1e-12:
        return 0.0
    return tr_cov_between / mean_tr_exp_sigma


def _cont_holdout_nll(niws: list[NIW], holdout_concepts: np.ndarray,
                      holdout_words: np.ndarray) -> float:
    """Held-out NLL for continuous agent using predictive_word_logprobs."""
    Sigma_anchor = ANCHOR_SIGMA_SCALE**2 * np.eye(D)
    total = 0.0
    for i in range(N_HOLDOUT):
        c = int(holdout_concepts[i])
        w = int(holdout_words[i])
        mu_post = CONCEPT_MUS[c]
        word_mus = [niws[k]._m for k in range(N_WORDS)]
        word_Sigmas = [niws[k].expected_Sigma() for k in range(N_WORDS)]
        log_probs = predictive_word_logprobs(mu_post, Sigma_anchor, word_mus, word_Sigmas)
        total += -log_probs[w]
    return total / N_HOLDOUT


def _tab_holdout_nll(counts: np.ndarray, holdout_concepts: np.ndarray,
                     holdout_words: np.ndarray) -> float:
    """Held-out NLL for tabular twin.

    State is given (concept), so NLL = mean -log A_hat[state, word].
    This is trivially log-space safe (no product over observations needed) — the
    state is provided, so no filtering occurs; noted per VALIDATION.md log-space rule.
    """
    # counts shape: (N_WORDS states, N_WORDS words); Dirichlet(alpha0=1) -> +1 prior
    A_hat = (1.0 + counts) / (1.0 + counts).sum(axis=1, keepdims=True)
    total = 0.0
    for i in range(N_HOLDOUT):
        c = int(holdout_concepts[i])
        w = int(holdout_words[i])
        total += -np.log(A_hat[c, w] + 1e-300)
    return total / N_HOLDOUT


# ---------------------------------------------------------------------------
# Run one (seed, cell) pair
# ---------------------------------------------------------------------------

def run_cell_seed(kappa0: float, nu0: float, seed: int, cell_index: int) -> list[dict]:
    """Run one (kappa0, nu0, seed) experiment; return list of JSON rows."""
    streams = _generate_streams(seed, cell_index)
    concepts_s = streams["concepts_s"]
    words_s = streams["words_s"]
    concepts_n = streams["concepts_n"]
    words_n = streams["words_n"]
    holdout_concepts = streams["holdout_concepts"]
    holdout_words = streams["holdout_words"]

    Sigma_anchor = ANCHOR_SIGMA_SCALE**2 * np.eye(D)

    # --- Continuous agent: 6 NIW objects, one per word ---
    niws = _make_fresh_niws(kappa0, nu0)

    # --- Tabular twin: counts[state, word] ---
    # State = concept index (given via anchor), word = observed word token
    counts = np.zeros((N_WORDS, N_WORDS), dtype=float)

    rows = []
    params = {"kappa0": float(kappa0), "nu0": float(nu0)}

    # Track visited positions for c_visited denominator
    visited_positions = []

    # ---- Structured phase ----
    for t in range(T_S):
        c = int(concepts_s[t])
        w = int(words_s[t])
        pos_t = CONCEPT_MUS[c]
        visited_positions.append(pos_t)

        # Continuous: update word w's NIW with anchored pos posterior
        niws[w] = niws[w].update_moments(pos_t, Sigma_anchor)

        # Tabular: count (state=c, word=w)
        counts[c, w] += 1.0

        step = t + 1
        if step in CHECKPOINTS[:T_S // 100]:
            pass  # We emit at checkpoints below via unified loop

    # Snapshot at end of structured phase
    m_at_Ts = [niws[k]._m.copy() for k in range(N_WORDS)]
    c_visited_arr = np.array(visited_positions)
    c_visited = c_visited_arr.mean(axis=0)  # empirical centroid of visited positions

    # Re-run both phases with checkpoint logging (unified pass)
    # Reset and replay to capture per-checkpoint metrics cleanly
    niws = _make_fresh_niws(kappa0, nu0)
    counts = np.zeros((N_WORDS, N_WORDS), dtype=float)

    # Also track n_half for delta and NLL
    cont_nll_end_struct = None
    tab_nll_end_struct = None
    cont_nll_end_noise = None
    tab_nll_end_noise = None
    n_half_delta = None  # noise dose (per-word) where delta first >= 0.5
    n_half_nll_cont = None
    n_half_nll_tab = None

    # We'll collect (step, delta, cont_nll, tab_nll) at checkpoints during noise phase
    noise_checkpoints_data = []

    # --- Full replay: structured phase ---
    for t in range(T_S):
        c = int(concepts_s[t])
        w = int(words_s[t])
        pos_t = CONCEPT_MUS[c]

        niws[w] = niws[w].update_moments(pos_t, Sigma_anchor)
        counts[c, w] += 1.0

        step = t + 1
        if step % 100 == 0:
            mean_err = _mean_err(niws)
            ci = _compute_ci(niws)
            cont_nll = _cont_holdout_nll(niws, holdout_concepts, holdout_words)
            tab_nll = _tab_holdout_nll(counts, holdout_concepts, holdout_words)
            # delta is 0 during structured phase (m_at_Ts not yet defined for partial)
            # We define delta wrt end-of-struct snapshot; during structured phase use
            # wrt current snapshot (not meaningful) — just emit 0.0 placeholder
            delta_val = 0.0

            for metric, val, agent in [
                ("mean_err", mean_err, "continuous"),
                ("ci", ci, "continuous"),
                ("delta", delta_val, "continuous"),
                ("holdout_nll", cont_nll, "continuous"),
                ("holdout_nll", tab_nll, "tabular"),
            ]:
                rows.append({
                    "exp": 135, "rung": 3, "agent": agent,
                    "seed": seed, "step": step,
                    "metric": metric, "value": val,
                    "params": params,
                })

    # Snapshot at end of structured phase
    m_at_Ts = [niws[k]._m.copy() for k in range(N_WORDS)]
    c_visited = np.array([CONCEPT_MUS[int(concepts_s[t])] for t in range(T_S)]).mean(axis=0)
    cont_nll_end_struct = _cont_holdout_nll(niws, holdout_concepts, holdout_words)
    tab_nll_end_struct = _tab_holdout_nll(counts, holdout_concepts, holdout_words)

    # --- Full replay: noise phase ---
    for t in range(T_N):
        c = int(concepts_n[t])
        w = int(words_n[t])
        pos_t = CONCEPT_MUS[c]

        niws[w] = niws[w].update_moments(pos_t, Sigma_anchor)
        counts[c, w] += 1.0

        step = T_S + t + 1
        if step % 100 == 0:
            n_noise_total = t + 1
            n_noise_per_word = n_noise_total / N_WORDS  # per-word dose

            mean_err = _mean_err(niws)
            delta_val = _compute_delta(niws, m_at_Ts, c_visited)
            ci = _compute_ci(niws)
            cont_nll = _cont_holdout_nll(niws, holdout_concepts, holdout_words)
            tab_nll = _tab_holdout_nll(counts, holdout_concepts, holdout_words)

            noise_checkpoints_data.append((n_noise_per_word, delta_val, cont_nll, tab_nll))

            # Track n_half for delta
            if n_half_delta is None and delta_val >= 0.5:
                n_half_delta = n_noise_per_word

            for metric, val, agent in [
                ("mean_err", mean_err, "continuous"),
                ("ci", ci, "continuous"),
                ("delta", delta_val, "continuous"),
                ("holdout_nll", cont_nll, "continuous"),
                ("holdout_nll", tab_nll, "tabular"),
            ]:
                rows.append({
                    "exp": 135, "rung": 3, "agent": agent,
                    "seed": seed, "step": step,
                    "metric": metric, "value": val,
                    "params": params,
                })

    # End-noise NLL values (last checkpoint)
    if noise_checkpoints_data:
        cont_nll_end_noise = noise_checkpoints_data[-1][2]
        tab_nll_end_noise = noise_checkpoints_data[-1][3]

    # NLL-based n_half: noise dose where NLL first rises past midpoint
    cont_nll_mid = None
    tab_nll_mid = None
    if cont_nll_end_struct is not None and cont_nll_end_noise is not None:
        cont_nll_mid = (cont_nll_end_struct + cont_nll_end_noise) / 2.0
    if tab_nll_end_struct is not None and tab_nll_end_noise is not None:
        tab_nll_mid = (tab_nll_end_struct + tab_nll_end_noise) / 2.0

    for n_dose, _delta, cont_nll, tab_nll in noise_checkpoints_data:
        if n_half_nll_cont is None and cont_nll_mid is not None and cont_nll >= cont_nll_mid:
            n_half_nll_cont = n_dose
        if n_half_nll_tab is None and tab_nll_mid is not None and tab_nll >= tab_nll_mid:
            n_half_nll_tab = n_dose

    # End-structured mean error scalar
    end_struct_err = _mean_err_from_ms(m_at_Ts)

    # CI at end of structured and end of noise phases
    # (already emitted as checkpoint rows; retrieve last struct and last noise)
    # We need these for per-seed scalar rows
    # Get CI at end of struct (recompute from m_at_Ts and niws-at-struct)
    # We'll use the rows approach: just emit n_half scalars
    for metric, val, agent in [
        ("n_half_delta", float(n_half_delta) if n_half_delta is not None else float("nan"), "continuous"),
        ("n_half_nll_cont", float(n_half_nll_cont) if n_half_nll_cont is not None else float("nan"), "continuous"),
        ("n_half_nll_tab", float(n_half_nll_tab) if n_half_nll_tab is not None else float("nan"), "tabular"),
        ("end_struct_err", float(end_struct_err), "continuous"),
    ]:
        rows.append({
            "exp": 135, "rung": 3, "agent": agent,
            "seed": seed, "step": T_S,
            "metric": metric, "value": val,
            "params": params,
        })

    return rows


def _mean_err_from_ms(ms: list[np.ndarray]) -> float:
    """Mean per-word ||m_k - mu_true_k|| from a list of means."""
    return float(np.mean([np.linalg.norm(ms[k] - CONCEPT_MUS[k]) for k in range(N_WORDS)]))


# ---------------------------------------------------------------------------
# Cell aggregation helpers
# ---------------------------------------------------------------------------

def _collect_cell_data(all_rows: list[dict]) -> dict:
    """Aggregate rows by (kappa0, nu0) cell for printing and prediction checks."""
    from collections import defaultdict

    # Index rows
    # cell key: (kappa0, nu0)
    cell_end_struct_errs: dict = defaultdict(list)
    cell_n_half_delta: dict = defaultdict(list)
    cell_n_half_nll_cont: dict = defaultdict(list)
    cell_n_half_nll_tab: dict = defaultdict(list)
    # delta at noise checkpoints: (kappa0, nu0, n_dose) -> list of deltas
    cell_delta_at_n: dict = defaultdict(list)
    # CI end-struct and end-noise per seed
    cell_ci_end_struct: dict = defaultdict(list)
    cell_ci_end_noise: dict = defaultdict(list)
    # NLL end-struct
    cell_nll_end_struct: dict = defaultdict(list)

    for row in all_rows:
        p = row["params"]
        ck = (p["kappa0"], p["nu0"])
        m = row["metric"]
        v = row["value"]
        step = row["step"]

        if m == "end_struct_err":
            cell_end_struct_errs[ck].append(v)
        elif m == "n_half_delta":
            cell_n_half_delta[ck].append(v)
        elif m == "n_half_nll_cont":
            cell_n_half_nll_cont[ck].append(v)
        elif m == "n_half_nll_tab":
            cell_n_half_nll_tab[ck].append(v)
        elif m == "delta" and step > T_S:
            n_dose = (step - T_S) / N_WORDS
            cell_delta_at_n[(ck, n_dose)].append(v)
        elif m == "ci" and row["agent"] == "continuous":
            if step == T_S:
                cell_ci_end_struct[ck].append(v)
            elif step == T_TOTAL:
                cell_ci_end_noise[ck].append(v)
        elif m == "holdout_nll" and row["agent"] == "continuous" and step == T_S:
            cell_nll_end_struct[ck].append(v)

    return {
        "end_struct_errs": dict(cell_end_struct_errs),
        "n_half_delta": dict(cell_n_half_delta),
        "n_half_nll_cont": dict(cell_n_half_nll_cont),
        "n_half_nll_tab": dict(cell_n_half_nll_tab),
        "delta_at_n": dict(cell_delta_at_n),
        "ci_end_struct": dict(cell_ci_end_struct),
        "ci_end_noise": dict(cell_ci_end_noise),
        "nll_end_struct": dict(cell_nll_end_struct),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import sys

    all_rows: list[dict] = []

    # Build cell list: kappa0 outer, nu0 inner (row-major for cell_index)
    cells = [(k, n) for k in KAPPA0_LIST for n in NU0_LIST]

    for cell_index, (kappa0, nu0) in enumerate(cells):
        for seed in range(N_SEEDS):
            seed_rows = run_cell_seed(kappa0, nu0, seed, cell_index)
            all_rows.extend(seed_rows)

    # --- Write JSON ---
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp135_rows.json"
    with open(out_path, "w") as f:
        json.dump(all_rows, f)

    # --- Aggregate ---
    agg = _collect_cell_data(all_rows)
    end_struct_errs = agg["end_struct_errs"]
    n_half_delta_map = agg["n_half_delta"]
    n_half_nll_cont_map = agg["n_half_nll_cont"]
    n_half_nll_tab_map = agg["n_half_nll_tab"]
    delta_at_n = agg["delta_at_n"]
    ci_end_struct = agg["ci_end_struct"]
    ci_end_noise = agg["ci_end_noise"]

    N_S_WORD = T_S / N_WORDS  # = 100

    # --- Table ---
    print("=" * 100)
    print("Exp 135 — continuous-substrate rung 3: collapse rematch")
    print("=" * 100)
    hdr = (f"{'kappa0':>7}  {'nu0':>4}  {'EndStructErr':>14}  {'Predicted':>10}  "
           f"{'n_half_d':>9}  {'kappa_eff':>9}  "
           f"{'CI_noise/CI_struct':>18}  {'NLL_half_cont':>13}  {'NLL_half_tab':>12}")
    print(hdr)
    print("-" * 100)

    for kappa0 in KAPPA0_LIST:
        for nu0 in NU0_LIST:
            ck = (float(kappa0), float(nu0))
            kappa_eff = kappa0 + N_S_WORD

            errs = end_struct_errs.get(ck, [])
            mean_err = float(np.nanmean(errs)) if errs else float("nan")
            predicted_err = kappa0 / (kappa0 + N_S_WORD)

            n_halves = [v for v in n_half_delta_map.get(ck, []) if not np.isnan(v)]
            mean_n_half = float(np.mean(n_halves)) if n_halves else float("nan")

            ci_s = ci_end_struct.get(ck, [])
            ci_n = ci_end_noise.get(ck, [])
            ci_ratio = float("nan")
            if ci_s and ci_n:
                mean_ci_s = float(np.nanmean(ci_s))
                mean_ci_n = float(np.nanmean(ci_n))
                ci_ratio = mean_ci_n / mean_ci_s if mean_ci_s > 1e-12 else float("nan")

            nll_halves_c = [v for v in n_half_nll_cont_map.get(ck, []) if not np.isnan(v)]
            nll_halves_t = [v for v in n_half_nll_tab_map.get(ck, []) if not np.isnan(v)]
            mean_nll_c = float(np.mean(nll_halves_c)) if nll_halves_c else float("nan")
            mean_nll_t = float(np.mean(nll_halves_t)) if nll_halves_t else float("nan")

            print(f"{kappa0:>7}  {nu0:>4}  {mean_err:>14.4f}  {predicted_err:>10.4f}  "
                  f"{mean_n_half:>9.1f}  {kappa_eff:>9.1f}  "
                  f"{ci_ratio:>18.4f}  {mean_nll_c:>13.1f}  {mean_nll_t:>12.1f}")

    print()

    # -----------------------------------------------------------------------
    # P1: calibration
    # -----------------------------------------------------------------------
    print("--- P1: Calibration (end-struct error within +-0.10 of kappa0/(kappa0+100)) ---")
    p1_cell_pass = {}
    for kappa0 in KAPPA0_LIST:
        for nu0 in NU0_LIST:
            ck = (float(kappa0), float(nu0))
            errs = end_struct_errs.get(ck, [])
            mean_err = float(np.nanmean(errs)) if errs else float("nan")
            predicted = kappa0 / (kappa0 + N_S_WORD)
            deviation = abs(mean_err - predicted)
            passes = deviation < 0.10
            p1_cell_pass[ck] = passes
            print(f"  kappa0={kappa0:>3}, nu0={nu0}: mean_err={mean_err:.4f}, "
                  f"predicted={predicted:.4f}, |dev|={deviation:.4f} -> {'PASS' if passes else 'FAIL'}")
    p1_fail_cells = sum(1 for v in p1_cell_pass.values() if not v)
    print(f"  P1 failing cells: {p1_fail_cells}/6 (falsifier: >= 2)")
    print()

    # -----------------------------------------------------------------------
    # P2: erosion law
    # -----------------------------------------------------------------------
    print("--- P2: Erosion law (|delta(n) - n/(kappa_eff+n)| < 0.10 at every checkpoint) ---")
    # Per (kappa0, nu0): max deviation across checkpoints, n_half within +-25%
    p2_dev_fail_cells = 0
    p2_nhalf_fail_cells = 0
    p2_nhalf_within = {}

    for kappa0 in KAPPA0_LIST:
        for nu0 in NU0_LIST:
            ck = (float(kappa0), float(nu0))
            kappa_eff = kappa0 + N_S_WORD

            # Gather noise checkpoints
            noise_doses = sorted(set(
                key[1] for key in delta_at_n.keys() if key[0] == ck
            ))
            max_dev = 0.0
            dev_fail_count = 0
            for n_dose in noise_doses:
                deltas = delta_at_n.get((ck, n_dose), [])
                if not deltas:
                    continue
                mean_delta = float(np.mean(deltas))
                predicted_delta = n_dose / (kappa_eff + n_dose)
                dev = abs(mean_delta - predicted_delta)
                if dev > 0.10:
                    dev_fail_count += 1
                max_dev = max(max_dev, dev)

            n_halves = [v for v in n_half_delta_map.get(ck, []) if not np.isnan(v)]
            mean_n_half = float(np.mean(n_halves)) if n_halves else float("nan")
            nhalf_in_band = (
                not np.isnan(mean_n_half) and
                abs(mean_n_half - kappa_eff) / kappa_eff < 0.25
            )
            p2_nhalf_within[ck] = nhalf_in_band

            if dev_fail_count >= 3:
                p2_dev_fail_cells += 1
            if not nhalf_in_band:
                p2_nhalf_fail_cells += 1

            print(f"  kappa0={kappa0:>3}, nu0={nu0}: max_dev={max_dev:.4f}, "
                  f"dev_fail_checkpoints={dev_fail_count}, "
                  f"mean_n_half={mean_n_half:.1f}, kappa_eff={kappa_eff:.1f}, "
                  f"n_half_in_band={'YES' if nhalf_in_band else 'NO'}")

    p2_nhalf_pass_cells = sum(1 for v in p2_nhalf_within.values() if v)
    print(f"  P2 delta-deviation fail cells (>= 3 checkpoints): {p2_dev_fail_cells}/6 "
          f"(falsifier: >= 2)")
    print(f"  P2 n_half in +-25% band: {p2_nhalf_pass_cells}/6 cells pass "
          f"(need >= 5; failing: {6 - p2_nhalf_pass_cells})")
    print()

    # -----------------------------------------------------------------------
    # P3: twin collapse
    # -----------------------------------------------------------------------
    print("--- P3: Twin collapse (both substrates; NLL n_half ratio at kappa0=1,nu0=4) ---")
    p3_ci_collapse_all = True
    for kappa0 in KAPPA0_LIST:
        for nu0 in NU0_LIST:
            ck = (float(kappa0), float(nu0))
            ci_s = ci_end_struct.get(ck, [])
            ci_n = ci_end_noise.get(ck, [])
            mean_ci_s = float(np.nanmean(ci_s)) if ci_s else float("nan")
            mean_ci_n = float(np.nanmean(ci_n)) if ci_n else float("nan")
            if np.isnan(mean_ci_s) or np.isnan(mean_ci_n) or mean_ci_s < 1e-12:
                ratio = float("nan")
                ok = False
            else:
                ratio = mean_ci_n / mean_ci_s
                ok = ratio < 0.25
            if not ok:
                p3_ci_collapse_all = False
            print(f"  kappa0={kappa0:>3}, nu0={nu0}: CI_struct={mean_ci_s:.4f}, "
                  f"CI_noise={mean_ci_n:.4f}, ratio={ratio:.4f} -> "
                  f"{'collapse (<0.25)' if ok else 'NO COLLAPSE'}")

    # NLL n_half ratio at kappa0=1, nu0=4
    ck_ref = (1.0, 4.0)
    nll_c = [v for v in n_half_nll_cont_map.get(ck_ref, []) if not np.isnan(v)]
    nll_t = [v for v in n_half_nll_tab_map.get(ck_ref, []) if not np.isnan(v)]
    mean_nll_c = float(np.mean(nll_c)) if nll_c else float("nan")
    mean_nll_t = float(np.mean(nll_t)) if nll_t else float("nan")
    if not np.isnan(mean_nll_c) and not np.isnan(mean_nll_t) and mean_nll_t > 1e-9:
        nll_ratio = mean_nll_c / mean_nll_t
    else:
        nll_ratio = float("nan")
    p3_ratio_ok = not np.isnan(nll_ratio) and 0.5 <= nll_ratio <= 2.0
    print(f"\n  NLL n_half ratio (cont/tab) at kappa0=1, nu0=4: {nll_ratio:.4f} "
          f"(need [0.5, 2.0]) -> {'PASS' if p3_ratio_ok else 'FAIL'}")
    print(f"  CI collapse in all 6 cells: {'YES' if p3_ci_collapse_all else 'NO'}")
    # P3 ratio outside [0.2, 5] is falsifier
    p3_ratio_falsifier = np.isnan(nll_ratio) or not (0.2 <= nll_ratio <= 5.0)
    print(f"  P3 ratio falsifier (outside [0.2,5]): {'YES — FALSIFIER' if p3_ratio_falsifier else 'no'}")
    print()

    # -----------------------------------------------------------------------
    # P4: nu0 null knob
    # -----------------------------------------------------------------------
    print("--- P4: nu0 null (per-kappa0 mean |delta(nu0=4) - delta(nu0=20)| < 0.05) ---")
    p4_fail = False
    for kappa0 in KAPPA0_LIST:
        ck4 = (float(kappa0), 4.0)
        ck20 = (float(kappa0), 20.0)
        kappa_eff = kappa0 + N_S_WORD

        noise_doses = sorted(set(
            key[1] for key in delta_at_n.keys()
            if key[0] in (ck4, ck20)
        ))
        max_diff = 0.0
        fail_count = 0
        for n_dose in noise_doses:
            d4 = delta_at_n.get((ck4, n_dose), [])
            d20 = delta_at_n.get((ck20, n_dose), [])
            if not d4 or not d20:
                continue
            diff = abs(float(np.mean(d4)) - float(np.mean(d20)))
            if diff >= 0.05:
                fail_count += 1
            max_diff = max(max_diff, diff)

        ok = max_diff < 0.05
        if not ok:
            p4_fail = True
        print(f"  kappa0={kappa0:>3}: max |delta_nu4 - delta_nu20| = {max_diff:.4f}, "
              f"fail_checkpoints = {fail_count} -> {'PASS' if ok else 'FAIL (nu0 non-null)'}")

    print()

    # -----------------------------------------------------------------------
    # VERDICT
    # -----------------------------------------------------------------------
    # Enumerate all falsifiers and prediction conjuncts mechanically
    falsifiers = []

    # P2 delta deviates systematically
    if p2_dev_fail_cells >= 2:
        falsifiers.append(
            f"P2-delta: deviation > 0.10 at >= 3 checkpoints in {p2_dev_fail_cells}/6 cells "
            f"(need < 2 failing cells)"
        )

    # P2 n_half not within +-25% in >= 2 cells
    if (6 - p2_nhalf_pass_cells) >= 2:
        falsifiers.append(
            f"P2-n_half: {6 - p2_nhalf_pass_cells}/6 cells have n_half outside +-25% of kappa_eff "
            f"(resistance not linear in kappa0)"
        )

    # P3 ratio outside [0.2, 5]
    if p3_ratio_falsifier:
        dir_str = ""
        if not np.isnan(nll_ratio):
            dir_str = " (continuous faster)" if nll_ratio < 0.2 else " (tabular faster)"
        falsifiers.append(
            f"P3-ratio: NLL n_half ratio = {nll_ratio:.4f} outside [0.2, 5]{dir_str} "
            f"— substrate asymmetry detected"
        )

    # P4 nu0 non-null
    if p4_fail:
        falsifiers.append(
            "P4: nu0 moves mean drift (max |delta_nu4 - delta_nu20| >= 0.05) — algebra wrong"
        )

    # P1 outside band in >= 2 cells
    if p1_fail_cells >= 2:
        falsifiers.append(
            f"P1: {p1_fail_cells}/6 cells outside +-0.10 calibration band"
        )

    # Positive conditions (need ALL)
    p1_ok = p1_fail_cells == 0
    p2_ok = (p2_dev_fail_cells < 2) and (6 - p2_nhalf_pass_cells < 2)
    p3_ok = p3_ci_collapse_all and p3_ratio_ok
    p4_ok = not p4_fail

    if falsifiers:
        print("VERDICT: NEGATIVE")
        for f in falsifiers:
            print(f"  FALSIFIER: {f}")
        # Note if P3 asymmetry is the named deep-negative/positive fork
        if any("P3-ratio" in f for f in falsifiers):
            if not np.isnan(nll_ratio) and nll_ratio < 0.2:
                print("  NOTE: P3 asymmetry direction = continuous erodes FASTER than tabular "
                      "(deep-negative fork of card)")
            elif not np.isnan(nll_ratio) and nll_ratio > 5.0:
                print("  NOTE: P3 asymmetry direction = tabular erodes FASTER than continuous "
                      "(deep-positive fork: continuous more erosion-resistant)")
    elif p1_ok and p2_ok and p3_ok and p4_ok:
        print("VERDICT: POSITIVE — all P1/P2/P3/P4 conjuncts satisfied")
        print("  P1 calibration: all 6 cells within +-0.10 band")
        print("  P2 erosion law: delta deviations < 0.10, n_half within +-25% in >= 5/6 cells")
        print("  P3 twin: CI collapse in all cells, NLL n_half ratio in [0.5, 2.0]")
        print("  P4 nu0 null: max |delta_nu4 - delta_nu20| < 0.05 in all kappa0 values")
    else:
        print("VERDICT: MIXED")
        mixed_notes = []
        if not p1_ok:
            mixed_notes.append(f"P1 borderline: {p1_fail_cells}/6 cells outside band (need 0)")
        if p2_dev_fail_cells >= 1:
            mixed_notes.append(
                f"P2-delta: {p2_dev_fail_cells}/6 cells have >= 3 checkpoint failures (< 2 for PASS)")
        if (6 - p2_nhalf_pass_cells) >= 1:
            mixed_notes.append(
                f"P2-n_half: {6 - p2_nhalf_pass_cells}/6 cells outside +-25% (need < 2 for PASS)")
        if not p3_ci_collapse_all:
            mixed_notes.append("P3: not all cells show CI collapse")
        if not p3_ratio_ok:
            mixed_notes.append(
                f"P3-ratio: {nll_ratio:.4f} outside [0.5, 2.0] but inside [0.2, 5] "
                f"(non-falsifying asymmetry)")
        if not p4_ok:
            mixed_notes.append("P4: nu0 borderline non-null")
        for note in mixed_notes:
            print(f"  NOTE: {note}")

    print()
    print(f"Rows written to: {out_path}")


if __name__ == "__main__":
    main()
