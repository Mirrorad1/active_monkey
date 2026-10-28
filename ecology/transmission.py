"""Read-only Price accounting and the exact mean drift of bounded mutation.

Survivors are their own descendants; each new ID has exactly one recorded
parent in the pre-step population, even if that parent subsequently dies.
The selection term is a realized association, not a causal trait effect.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping
from math import fsum, isfinite, sqrt, pi

import numpy as np
from scipy.special import ndtr


def _bounds(lo: float, hi: float) -> None:
    if not (isfinite(lo) and isfinite(hi) and lo < hi):
        raise ValueError("bounds must be finite with lo < hi")


def clipped_gaussian_drift(x, sigma: float, lo: float = 0., hi: float = 1.):
    """Return E[clip(x + Normal(0, sigma), lo, hi) - x], elementwise.

    ``x`` must already be inside the finite bounds; sigma is a finite,
    nonnegative scalar. A scalar input returns a float. No RNG is used.
    """
    _bounds(lo, hi)
    if not isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    values = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(values)) or np.any((values < lo) | (values > hi)):
        raise ValueError("trait values must be finite and within bounds")
    if sigma == 0:
        drift = np.zeros_like(values)
    else:
        with np.errstate(over="ignore", under="ignore"):
            a, b = (lo - values) / sigma, (hi - values) / sigma
            phi_a = np.exp(-.5 * a * a) / sqrt(2 * pi)
            phi_b = np.exp(-.5 * b * b) / sqrt(2 * pi)
        drift = sigma * (phi_a - phi_b) + (lo - values) * ndtr(a) + (hi - values) * ndtr(-b)
    return float(drift) if values.ndim == 0 else drift


def price_step(
    before: Mapping[Hashable, float],
    after: Mapping[Hashable, float],
    newborn_parents: Mapping[Hashable, Hashable],
    *, lo: float = 0., hi: float = 1.,
) -> dict:
    """Decompose the mean-trait change for one population transition.

    S = mean(ancestor trait over descendants) - mean(before).
    T = mean(descendant trait - ancestor trait), including self-descendants.
    For an empty population, its mean and the decomposition are undefined
    (None); extinction itself is not malformed lineage. Input mappings ensure
    a single value/ancestor per ID, and ancestry must cover *exactly* new IDs.
    """
    _bounds(lo, hi)
    for cohort in (before, after):
        if any(not isfinite(z) or not lo <= z <= hi for z in cohort.values()):
            raise ValueError("traits must be finite and within bounds")
    newborn_ids = after.keys() - before.keys()
    if newborn_parents.keys() != newborn_ids:
        raise ValueError("newborn ancestry must cover exactly the new descendant IDs")
    if any(parent not in before for parent in newborn_parents.values()):
        raise ValueError("every newborn parent must belong to the pre-step population")
    n_before, n_after = len(before), len(after)
    result = {
        "n_before": n_before,
        "n_after": n_after,
        "births": len(newborn_ids),
        "deaths": len(before.keys() - after.keys()),
        "mean_before": fsum(before.values()) / n_before if n_before else None,
        "mean_after": fsum(after.values()) / n_after if n_after else None,
        "mean_ancestor": None,
        "selection": None,
        "transmission": None,
        "delta": None,
        "residual": None,
    }
    if not n_before or not n_after:
        return result
    ancestors = {
        child: before[child] if child in before else before[newborn_parents[child]]
        for child in after
    }
    mean_ancestor = fsum(ancestors.values()) / n_after
    selection = mean_ancestor - result["mean_before"]
    transmission = fsum(after[child] - ancestors[child] for child in after) / n_after
    delta = result["mean_after"] - result["mean_before"]
    result.update(
        mean_ancestor=mean_ancestor, selection=selection,
        transmission=transmission, delta=delta,
        residual=delta - selection - transmission,
    )
    return result
