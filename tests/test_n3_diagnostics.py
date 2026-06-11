"""Unit tests for active_loop/n3_diagnostics.py (N3 shadow diagnostic)."""
from __future__ import annotations

import numpy as np

from active_loop.n3_diagnostics import (
    REGIMES_4,
    RunSignals,
    classify_regime,
    structural_gain,
)


# ---------------------------------------------------------------------------
# classify_regime: one synthetic RunSignals per regime
# ---------------------------------------------------------------------------


def test_classify_stable_known():
    sig = RunSignals(loud_mean=0.3, ceiling_fired=False, early_mean=0.3,
                     late_mean=0.3, structural=False)
    assert classify_regime(sig) == "stable_known"


def test_classify_nonstationarity_low_then_jump():
    # Low early, high late by a clear margin -> world changed.
    sig = RunSignals(loud_mean=1.1, ceiling_fired=True, early_mean=0.25,
                     late_mean=1.0, structural=True)  # structural True must NOT win
    assert classify_regime(sig) == "nonstationarity"


def test_classify_structural_inadequacy():
    # High from the start (no jump), split test fired.
    sig = RunSignals(loud_mean=1.2, ceiling_fired=True, early_mean=1.1,
                     late_mean=1.2, structural=True)
    assert classify_regime(sig) == "structural_inadequacy"


def test_classify_irreducible_noise():
    # High from the start, split test did NOT fire.
    sig = RunSignals(loud_mean=0.9, ceiling_fired=True, early_mean=0.9,
                     late_mean=0.9, structural=False)
    assert classify_regime(sig) == "irreducible_noise"


def test_high_via_loud_mean_without_ceiling():
    sig = RunSignals(loud_mean=0.85, ceiling_fired=False, early_mean=0.85,
                     late_mean=0.85, structural=False)
    assert classify_regime(sig) == "irreducible_noise"


def test_classify_returns_known_label():
    sig = RunSignals(loud_mean=0.5, ceiling_fired=False, early_mean=0.5,
                     late_mean=0.5, structural=False)
    assert classify_regime(sig) in REGIMES_4


def test_small_jitter_near_threshold_is_not_nonstationarity():
    # early just below thresh, late just above but jump < JUMP_MARGIN -> not a change.
    sig = RunSignals(loud_mean=0.75, ceiling_fired=True, early_mean=0.65,
                     late_mean=0.75, structural=False)
    assert classify_regime(sig) == "irreducible_noise"


# ---------------------------------------------------------------------------
# structural_gain: faithful held-out-predictive split test (needs the full mixture)
# ---------------------------------------------------------------------------

from active_loop.continuous import NIW  # noqa: E402


def _broad_components(n_colors: int, center=(1.5, 1.5), var: float = 4.0):
    """n_colors broad, mutually-overlapping single Gaussians (the aliasing setup).

    With nu=4, D=2 the NIW expected covariance is S/(nu-D-1) = S, so S=var*I gives
    expected variance ``var``. Identical broad components mean a single loud-color
    Gaussian wins nothing — only a tight SPLIT concentrates density at true cells.
    """
    S = var * np.eye(2)
    return [
        [(1.0, NIW(m=np.array(center, float), kappa=1.0, nu=4.0, S=S.copy()))]
        for _ in range(n_colors)
    ]


def _pairs_at(points, reps: int, sigma: float = 0.05):
    sig = np.array([sigma, sigma])
    out = []
    for _ in range(reps):
        for p in points:
            out.append((np.array(p, float), sig.copy()))
    return out


def test_structural_gain_true_when_split_concentrates_density():
    # Loud color truly at 4 separated corners; competitors broad & overlapping.
    comps = _broad_components(4)
    corners = [(0.0, 0.0), (0.0, 3.0), (3.0, 0.0), (3.0, 3.0)]
    loud_pairs = _pairs_at(corners, reps=12)  # 48 pairs -> train 34 / test 14
    structural, sc = structural_gain(comps, 0, loud_pairs, np.random.default_rng(0))
    assert structural is True
    assert sc["heldout_surprise_bestK"] < 0.7
    assert sc["heldout_surprise_bestK"] <= sc["heldout_surprise_k1"] - 0.1


def test_structural_gain_false_under_spatial_noise():
    # Loud color positions uniformly scattered (no concentration) -> floor, not split-able.
    rng = np.random.default_rng(1)
    pts = rng.uniform(0.0, 3.0, size=(48, 2))
    loud_pairs = [(p, np.array([0.05, 0.05])) for p in pts]
    comps = _broad_components(4)
    structural, sc = structural_gain(comps, 0, loud_pairs, np.random.default_rng(2))
    assert structural is False
    assert sc["heldout_surprise_bestK"] >= 0.7


def test_structural_gain_insufficient_pairs():
    comps = _broad_components(4)
    loud_pairs = _pairs_at([(0.0, 0.0)], reps=10)  # 10 < MIN_REPLAY_PAIRS
    structural, sc = structural_gain(comps, 0, loud_pairs, np.random.default_rng(3))
    assert structural is False
    assert sc["reason"] == "insufficient_pairs"
