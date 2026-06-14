"""
ecology.evolvability.metrics — pure numeric helpers for evolvability analysis.

All functions return float('nan') gracefully on degenerate input and never raise.

Formulas
--------
logit(p)
    log(p / (1 - p)), clipped to avoid log(0).

selection_coefficient(w_mut, w_res)
    s = log(W_mut / W_res)
    s > 0 ⇒ mutant favoured; s < 0 ⇒ mutant disfavoured.

selection_coefficient_freq(f0, f1, elapsed)
    (logit(f1) - logit(f0)) / elapsed
    Preferred when only allele frequencies and elapsed time are observed (robust to
    fixation, avoids tracking individual fitnesses).

cross_partial(b_ll, b_hl, b_lh, b_hh)
    b_hh - b_lh - b_hl + b_ll
    Positive ⇒ trait (h) and controller (theta) are synergistic: their joint value
    exceeds the sum of their independent effects.

corner_effects(b_ll, b_hl, b_lh, b_hh)
    Dictionary of marginal and interaction effects at the four corners of the
    (h, theta) grid.

monomorphic_optimum(curve)
    argmax over (h, value) pairs in curve; returns (h_star, value_star).

optimum_above_resident(curve, resident)
    True iff the monomorphic optimum h* > resident.

count_wins(pairs, eps)
    Returns (wins, n_valid) over (mutant_metric, resident_metric) pairs.

mean_diff(pairs)
    Mean of (mutant - resident) over valid (non-nan) pairs.

summarize(values)
    {"mean", "std", "n", "n_valid"} using numpy nan-aware functions.

is_population_valid(final_pop, extinct, exploded, min_pop)
    Guard that a simulation run produced a usable population.

default_thresholds(n)
    (win_threshold, lose_threshold) = (ceil(7n/8), floor(3n/8)).
    For n=8 ⇒ (7, 3).
    NEGATIVE (lose) is intentionally easier to reach than POSITIVE (win) so that
    the framework does not over-declare positive local gradients.
"""

import math
import warnings
from typing import Union

import numpy as np

NaN = float("nan")


# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------

def logit(p: float, eps: float = 1e-6) -> float:
    """log(p/(1-p)) with p clipped to [eps, 1-eps].

    Returns nan if p is nan.
    """
    if math.isnan(p):
        return NaN
    p_clipped = max(eps, min(1.0 - eps, p))
    return math.log(p_clipped / (1.0 - p_clipped))


# ---------------------------------------------------------------------------
# Selection coefficients
# ---------------------------------------------------------------------------

def selection_coefficient(w_mut: float, w_res: float) -> float:
    """s = log(W_mut / W_res); s > 0 ⇒ mutant favoured.

    Returns nan if either fitness is non-positive or nan.
    """
    if math.isnan(w_mut) or math.isnan(w_res):
        return NaN
    if w_mut <= 0.0 or w_res <= 0.0:
        return NaN
    return math.log(w_mut / w_res)


def selection_coefficient_freq(f0: float, f1: float, elapsed: float) -> float:
    """(logit(f1) - logit(f0)) / elapsed — frequency-change estimator of s.

    Preferred over the fitness-ratio form when only allele frequencies and
    elapsed time are available; robust to fixation events that make individual
    fitness tracking infeasible.

    Returns nan if elapsed <= 0 or if either frequency is nan.
    """
    if math.isnan(f0) or math.isnan(f1) or math.isnan(elapsed):
        return NaN
    if elapsed <= 0.0:
        return NaN
    return (logit(f1) - logit(f0)) / elapsed


# ---------------------------------------------------------------------------
# Corner-effect helpers (2×2 grid over h and theta)
# ---------------------------------------------------------------------------

def cross_partial(
    b_ll: float,
    b_hl: float,
    b_lh: float,
    b_hh: float,
) -> float:
    """b_hh - b_lh - b_hl + b_ll — the discrete cross-partial derivative.

    Notation: b_<h_level><theta_level>
        b_ll = low-h / low-theta
        b_hl = high-h / low-theta
        b_lh = low-h / high-theta
        b_hh = high-h / high-theta

    Positive ⇒ trait (h) and controller (theta) are synergistic: their joint
    value exceeds the sum of their independent effects.
    Negative ⇒ antagonistic.
    """
    return b_hh - b_lh - b_hl + b_ll


def corner_effects(
    b_ll: float,
    b_hl: float,
    b_lh: float,
    b_hh: float,
) -> dict:
    """Five-key dictionary of marginal and interaction effects.

    Keys
    ----
    cross_partial       : b_hh - b_lh - b_hl + b_ll
    dB_dh_lo_theta      : b_hl - b_ll  (marginal effect of h at low theta)
    dB_dh_hi_theta      : b_hh - b_lh  (marginal effect of h at high theta)
    dB_dtheta_lo_h      : b_lh - b_ll  (marginal effect of theta at low h)
    dB_dtheta_hi_h      : b_hh - b_hl  (marginal effect of theta at high h)
    """
    return {
        "cross_partial":   cross_partial(b_ll, b_hl, b_lh, b_hh),
        "dB_dh_lo_theta":  b_hl - b_ll,
        "dB_dh_hi_theta":  b_hh - b_lh,
        "dB_dtheta_lo_h":  b_lh - b_ll,
        "dB_dtheta_hi_h":  b_hh - b_hl,
    }


# ---------------------------------------------------------------------------
# Monomorphic optimum
# ---------------------------------------------------------------------------

def monomorphic_optimum(curve: dict) -> tuple:
    """argmax over (h, value) pairs in curve where value is not nan.

    Parameters
    ----------
    curve : dict mapping float h -> float value

    Returns
    -------
    (h_star, value_star) or (nan, nan) if no valid entries exist.
    """
    best_h = NaN
    best_v = NaN
    for h, v in curve.items():
        if math.isnan(v):
            continue
        if math.isnan(best_v) or v > best_v:
            best_h = float(h)
            best_v = float(v)
    return (best_h, best_v)


def optimum_above_resident(curve: dict, resident: float) -> bool:
    """True iff the monomorphic optimum h* > resident and h* is not nan."""
    h_star, _ = monomorphic_optimum(curve)
    if math.isnan(h_star):
        return False
    return h_star > resident


# ---------------------------------------------------------------------------
# Win-count helpers
# ---------------------------------------------------------------------------

def count_wins(
    pairs: list,
    eps: float = 0.0,
) -> tuple:
    """Count wins in (mutant_metric, resident_metric) pairs.

    A pair is valid if neither element is nan.
    A win is a valid pair where mutant > resident + eps.

    Parameters
    ----------
    pairs : list of (mutant_metric, resident_metric)
    eps   : margin; default 0.0 (strict >)

    Returns
    -------
    (wins, n_valid)
    """
    wins = 0
    n_valid = 0
    for mutant, resident in pairs:
        if math.isnan(mutant) or math.isnan(resident):
            continue
        n_valid += 1
        if mutant > resident + eps:
            wins += 1
    return (wins, n_valid)


def mean_diff(pairs: list) -> float:
    """Mean of (mutant - resident) over pairs where neither element is nan.

    Returns nan if there are no valid pairs.
    """
    diffs = []
    for mutant, resident in pairs:
        if math.isnan(mutant) or math.isnan(resident):
            continue
        diffs.append(mutant - resident)
    if not diffs:
        return NaN
    return sum(diffs) / len(diffs)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize(values: list) -> dict:
    """Summary statistics for a list of floats using numpy nan-aware functions.

    Returns
    -------
    {"mean": float, "std": float, "n": int, "n_valid": int}
    mean and std are nan if n_valid == 0.
    """
    n = len(values)
    arr = np.array(values, dtype=float)
    n_valid = int(np.sum(~np.isnan(arr)))
    if n_valid == 0:
        return {"mean": NaN, "std": NaN, "n": n, "n_valid": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_val = float(np.nanmean(arr))
        std_val = float(np.nanstd(arr))
    return {"mean": mean_val, "std": std_val, "n": n, "n_valid": n_valid}


# ---------------------------------------------------------------------------
# Population validity
# ---------------------------------------------------------------------------

def is_population_valid(
    final_pop: int,
    extinct: bool,
    exploded: bool,
    min_pop: int,
) -> bool:
    """True iff the population is usable for analysis.

    A population is invalid if it went extinct, exploded (uncontrolled growth),
    or ended below the minimum viable size.
    """
    if extinct:
        return False
    if exploded:
        return False
    return final_pop >= min_pop


# ---------------------------------------------------------------------------
# Threshold convention
# ---------------------------------------------------------------------------

def default_thresholds(n: int) -> tuple:
    """(win_threshold, lose_threshold) = (ceil(7n/8), floor(3n/8)).

    For n=8 ⇒ (7, 3).

    Convention
    ----------
    This is the repo's 7/8-strict convention.  NEGATIVE (lose) is intentionally
    easier to reach than POSITIVE (win): lose_threshold = floor(3n/8) is below
    the midpoint while win_threshold = ceil(7n/8) is well above it.  This
    asymmetry ensures the framework does not over-declare positive local
    gradients — a binding design requirement.
    """
    win_threshold = math.ceil(7 * n / 8)
    lose_threshold = math.floor(3 * n / 8)
    return (win_threshold, lose_threshold)
