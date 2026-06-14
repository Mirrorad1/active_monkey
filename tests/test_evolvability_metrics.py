"""
Tests for ecology.evolvability.metrics.

All contractual acceptance cases are marked with # ACCEPTANCE comment.
"""

import math
import pytest

from ecology.evolvability.metrics import (
    logit,
    selection_coefficient,
    selection_coefficient_freq,
    cross_partial,
    corner_effects,
    monomorphic_optimum,
    optimum_above_resident,
    count_wins,
    mean_diff,
    summarize,
    is_population_valid,
    default_thresholds,
)

NaN = float("nan")


# ---------------------------------------------------------------------------
# logit
# ---------------------------------------------------------------------------

class TestLogit:
    def test_midpoint(self):
        assert logit(0.5) == pytest.approx(0.0, abs=1e-12)

    def test_above_half_positive(self):
        assert logit(0.75) > 0

    def test_below_half_negative(self):
        assert logit(0.25) < 0

    def test_nan_passthrough(self):
        assert math.isnan(logit(NaN))

    def test_clips_near_zero(self):
        # Should not raise; clipping handles p ≈ 0
        result = logit(0.0)
        assert math.isfinite(result)

    def test_clips_near_one(self):
        result = logit(1.0)
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# selection_coefficient
# ---------------------------------------------------------------------------

class TestSelectionCoefficient:
    def test_mutant_favoured(self):  # ACCEPTANCE
        assert selection_coefficient(2.0, 1.0) == pytest.approx(math.log(2))

    def test_zero_resident_is_nan(self):  # ACCEPTANCE
        assert math.isnan(selection_coefficient(1.0, 0.0))

    def test_zero_mutant_is_nan(self):  # ACCEPTANCE
        assert math.isnan(selection_coefficient(0.0, 1.0))

    def test_negative_resident_is_nan(self):
        assert math.isnan(selection_coefficient(1.0, -1.0))

    def test_equal_fitnesses_is_zero(self):
        assert selection_coefficient(1.5, 1.5) == pytest.approx(0.0)

    def test_mutant_disfavoured_negative(self):
        s = selection_coefficient(0.5, 1.0)
        assert s < 0
        assert s == pytest.approx(math.log(0.5))


# ---------------------------------------------------------------------------
# selection_coefficient_freq
# ---------------------------------------------------------------------------

class TestSelectionCoefficientFreq:
    def test_increasing_frequency_positive(self):  # ACCEPTANCE
        s = selection_coefficient_freq(0.5, 0.6, 10)
        assert s > 0

    def test_zero_elapsed_is_nan(self):  # ACCEPTANCE
        assert math.isnan(selection_coefficient_freq(0.5, 0.6, 0))

    def test_negative_elapsed_is_nan(self):
        assert math.isnan(selection_coefficient_freq(0.5, 0.6, -1))

    def test_decreasing_frequency_negative(self):
        s = selection_coefficient_freq(0.6, 0.5, 10)
        assert s < 0

    def test_unchanged_frequency_zero(self):
        s = selection_coefficient_freq(0.5, 0.5, 10)
        assert s == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# cross_partial
# ---------------------------------------------------------------------------

class TestCrossPartial:
    def test_synergistic(self):  # ACCEPTANCE
        # b_ll=1.0, b_hl=1.0, b_lh=1.0, b_hh=2.0  =>  2-1-1+1 = 1
        result = cross_partial(b_ll=1.0, b_hl=1.0, b_lh=1.0, b_hh=2.0)
        assert result == pytest.approx(1.0)
        assert result > 0

    def test_independent_effects_zero(self):
        # b_hh = b_hl + b_lh - b_ll => cross_partial = 0
        result = cross_partial(b_ll=1.0, b_hl=2.0, b_lh=1.5, b_hh=2.5)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_antagonistic_negative(self):
        result = cross_partial(b_ll=2.0, b_hl=3.0, b_lh=3.0, b_hh=3.0)
        assert result < 0


# ---------------------------------------------------------------------------
# corner_effects
# ---------------------------------------------------------------------------

class TestCornerEffects:
    def test_five_keys_present(self):
        eff = corner_effects(b_ll=1.0, b_hl=2.0, b_lh=1.5, b_hh=3.0)
        assert set(eff.keys()) == {
            "cross_partial",
            "dB_dh_lo_theta",
            "dB_dh_hi_theta",
            "dB_dtheta_lo_h",
            "dB_dtheta_hi_h",
        }

    def test_correct_arithmetic(self):  # ACCEPTANCE
        # b_ll=1.0, b_hl=2.0, b_lh=1.5, b_hh=3.0
        eff = corner_effects(b_ll=1.0, b_hl=2.0, b_lh=1.5, b_hh=3.0)
        assert eff["cross_partial"]   == pytest.approx(3.0 - 1.5 - 2.0 + 1.0)   # 0.5
        assert eff["dB_dh_lo_theta"]  == pytest.approx(2.0 - 1.0)                # 1.0
        assert eff["dB_dh_hi_theta"]  == pytest.approx(3.0 - 1.5)                # 1.5
        assert eff["dB_dtheta_lo_h"]  == pytest.approx(1.5 - 1.0)                # 0.5
        assert eff["dB_dtheta_hi_h"]  == pytest.approx(3.0 - 2.0)                # 1.0


# ---------------------------------------------------------------------------
# monomorphic_optimum and optimum_above_resident
# ---------------------------------------------------------------------------

class TestMonomorphicOptimum:
    CURVE = {0.0: 1.0, 0.1: 2.0, 0.6: 3.0}

    def test_argmax(self):  # ACCEPTANCE
        h_star, v_star = monomorphic_optimum(self.CURVE)
        assert h_star == pytest.approx(0.6)
        assert v_star == pytest.approx(3.0)

    def test_optimum_above_resident(self):  # ACCEPTANCE
        assert optimum_above_resident(self.CURVE, 0.10) is True

    def test_optimum_not_above_resident_at_top(self):
        assert optimum_above_resident(self.CURVE, 0.6) is False

    def test_all_nan_returns_nan_nan(self):  # ACCEPTANCE
        curve = {0.0: NaN, 0.5: NaN}
        h_star, v_star = monomorphic_optimum(curve)
        assert math.isnan(h_star)
        assert math.isnan(v_star)

    def test_all_nan_optimum_above_resident_false(self):
        curve = {0.0: NaN, 0.5: NaN}
        assert optimum_above_resident(curve, 0.0) is False

    def test_empty_curve_returns_nan_nan(self):
        h_star, v_star = monomorphic_optimum({})
        assert math.isnan(h_star)
        assert math.isnan(v_star)


# ---------------------------------------------------------------------------
# count_wins
# ---------------------------------------------------------------------------

class TestCountWins:
    def test_7_wins_out_of_8(self):  # ACCEPTANCE
        pairs = [(2.0, 1.0)] * 7 + [(1.0, 2.0)]
        wins, n_valid = count_wins(pairs)
        assert wins == 7
        assert n_valid == 8

    def test_3_wins_out_of_8(self):  # ACCEPTANCE
        pairs = [(2.0, 1.0)] * 3 + [(1.0, 2.0)] * 5
        wins, n_valid = count_wins(pairs)
        assert wins == 3
        assert n_valid == 8

    def test_nan_excluded_from_n_valid(self):  # ACCEPTANCE
        pairs = [(2.0, 1.0), (NaN, 1.0), (1.0, NaN)]
        wins, n_valid = count_wins(pairs)
        assert n_valid == 1
        assert wins == 1

    def test_eps_margin(self):
        pairs = [(1.05, 1.0)]
        wins, n_valid = count_wins(pairs, eps=0.1)
        assert wins == 0
        assert n_valid == 1

    def test_all_nan(self):
        pairs = [(NaN, 1.0), (NaN, 2.0)]
        wins, n_valid = count_wins(pairs)
        assert n_valid == 0
        assert wins == 0


# ---------------------------------------------------------------------------
# mean_diff
# ---------------------------------------------------------------------------

class TestMeanDiff:
    def test_basic(self):
        pairs = [(2.0, 1.0), (3.0, 1.0)]
        assert mean_diff(pairs) == pytest.approx(1.5)

    def test_nan_excluded(self):
        pairs = [(2.0, 1.0), (NaN, 1.0)]
        assert mean_diff(pairs) == pytest.approx(1.0)

    def test_all_nan_is_nan(self):
        assert math.isnan(mean_diff([(NaN, 1.0)]))

    def test_empty_is_nan(self):
        assert math.isnan(mean_diff([]))


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

class TestSummarize:
    def test_basic(self):
        s = summarize([1.0, 2.0, 3.0])
        assert s["n"] == 3
        assert s["n_valid"] == 3
        assert s["mean"] == pytest.approx(2.0)

    def test_nan_excluded_from_stats(self):
        s = summarize([1.0, NaN, 3.0])
        assert s["n"] == 3
        assert s["n_valid"] == 2
        assert s["mean"] == pytest.approx(2.0)

    def test_all_nan(self):
        s = summarize([NaN, NaN])
        assert s["n"] == 2
        assert s["n_valid"] == 0
        assert math.isnan(s["mean"])
        assert math.isnan(s["std"])

    def test_empty(self):
        s = summarize([])
        assert s["n"] == 0
        assert s["n_valid"] == 0
        assert math.isnan(s["mean"])


# ---------------------------------------------------------------------------
# is_population_valid
# ---------------------------------------------------------------------------

class TestIsPopulationValid:
    def test_extinct_is_invalid(self):  # ACCEPTANCE
        assert is_population_valid(0, extinct=True, exploded=False, min_pop=10) is False

    def test_healthy_large_pop_valid(self):  # ACCEPTANCE
        assert is_population_valid(50, extinct=False, exploded=False, min_pop=10) is True

    def test_exploded_is_invalid(self):
        assert is_population_valid(1000, extinct=False, exploded=True, min_pop=10) is False

    def test_below_min_pop_invalid(self):
        assert is_population_valid(5, extinct=False, exploded=False, min_pop=10) is False

    def test_exactly_min_pop_valid(self):
        assert is_population_valid(10, extinct=False, exploded=False, min_pop=10) is True


# ---------------------------------------------------------------------------
# default_thresholds
# ---------------------------------------------------------------------------

class TestDefaultThresholds:
    def test_n8(self):  # ACCEPTANCE
        win_t, lose_t = default_thresholds(8)
        assert win_t == 7
        assert lose_t == 3

    def test_asymmetry(self):
        # NEGATIVE is intentionally easier: lose_threshold < n/2 < win_threshold
        for n in [4, 8, 16, 32]:
            win_t, lose_t = default_thresholds(n)
            assert win_t > n / 2, f"win_threshold {win_t} not > n/2={n/2} for n={n}"
            assert lose_t < n / 2, f"lose_threshold {lose_t} not < n/2={n/2} for n={n}"
            assert win_t > lose_t

    def test_n4(self):
        win_t, lose_t = default_thresholds(4)
        assert win_t == math.ceil(7 * 4 / 8)   # ceil(3.5) = 4
        assert lose_t == math.floor(3 * 4 / 8)  # floor(1.5) = 1
