"""
Tests for ecology.evolvability.verdicts.

All contractual acceptance cases are marked with # ACCEPTANCE comment.
"""

import math
import pytest

from ecology.evolvability.metrics import corner_effects
from ecology.evolvability.verdicts import (
    BenefitVerdict,
    GradientVerdict,
    InvasionVerdict,
    CrossPartialVerdict,
    GuardStatus,
    AggregateVerdict,
    gradient_verdict,
    benefit_verdict,
    invasion_verdict,
    crosspartial_verdict,
    aggregate_verdict,
)

NaN = float("nan")


# ---------------------------------------------------------------------------
# Enum sanity: str mixin ensures value == str representation
# ---------------------------------------------------------------------------

class TestEnumStrMixin:
    def test_gradient_value_is_string(self):
        v = GradientVerdict.POSITIVE_LOCAL_GRADIENT
        assert isinstance(v.value, str)
        assert v.value == "POSITIVE_LOCAL_GRADIENT"

    def test_aggregate_value_is_string(self):
        v = AggregateVerdict.PASS_LOCAL_GRADIENT
        assert isinstance(v.value, str)

    def test_guard_status_pass(self):
        assert GuardStatus.PASS.value == "PASS"


# ---------------------------------------------------------------------------
# gradient_verdict
# ---------------------------------------------------------------------------

class TestGradientVerdict:
    # Shared kwargs for the standard 8-seed setup
    THRESH = dict(win_threshold=7, lose_threshold=3, min_valid=4)

    def test_positive_local_gradient(self):  # ACCEPTANCE — 7/8 acceptance test
        verdict, reason = gradient_verdict(
            wins=7, n_valid=8, mean_effect=0.05, **self.THRESH
        )
        assert verdict == GradientVerdict.POSITIVE_LOCAL_GRADIENT

    def test_negative_local_gradient(self):  # ACCEPTANCE — ≤3/8 acceptance test
        verdict, reason = gradient_verdict(
            wins=3, n_valid=8, mean_effect=-0.01, **self.THRESH
        )
        assert verdict == GradientVerdict.NEGATIVE_LOCAL_GRADIENT

    def test_flat_or_noisy(self):  # ACCEPTANCE
        verdict, reason = gradient_verdict(
            wins=5, n_valid=8, mean_effect=0.001, **self.THRESH
        )
        assert verdict == GradientVerdict.FLAT_OR_NOISY

    def test_no_verdict_collapsed_pop(self):  # ACCEPTANCE — collapsed-pop acceptance
        verdict, reason = gradient_verdict(
            wins=3, n_valid=2, mean_effect=0.0, **self.THRESH
        )
        assert verdict == GradientVerdict.NO_VERDICT

    def test_high_wins_nan_mean_effect_not_positive(self):  # ACCEPTANCE
        # Proof: win-count alone cannot force POSITIVE
        verdict, reason = gradient_verdict(
            wins=8, n_valid=8, mean_effect=NaN, **self.THRESH
        )
        assert verdict != GradientVerdict.POSITIVE_LOCAL_GRADIENT

    def test_high_wins_nan_mean_effect_falls_to_flat(self):
        # wins=8 > lose_threshold=3 so it should be FLAT_OR_NOISY (not NEGATIVE)
        verdict, reason = gradient_verdict(
            wins=8, n_valid=8, mean_effect=NaN, **self.THRESH
        )
        assert verdict == GradientVerdict.FLAT_OR_NOISY

    def test_high_wins_zero_mean_effect_not_positive(self):
        # mean_effect=0.0 <= flat_eps (1e-9) => not POSITIVE
        verdict, reason = gradient_verdict(
            wins=8, n_valid=8, mean_effect=0.0, **self.THRESH
        )
        assert verdict != GradientVerdict.POSITIVE_LOCAL_GRADIENT

    def test_reason_is_string(self):
        _, reason = gradient_verdict(
            wins=7, n_valid=8, mean_effect=0.05, **self.THRESH
        )
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_exactly_win_threshold(self):
        # wins == win_threshold with positive mean_effect ⇒ POSITIVE
        verdict, _ = gradient_verdict(
            wins=7, n_valid=8, mean_effect=1e-8, **self.THRESH
        )
        assert verdict == GradientVerdict.POSITIVE_LOCAL_GRADIENT

    def test_one_above_lose_threshold_flat(self):
        # wins = lose_threshold + 1 = 4 => FLAT
        verdict, _ = gradient_verdict(
            wins=4, n_valid=8, mean_effect=0.1, **self.THRESH
        )
        assert verdict == GradientVerdict.FLAT_OR_NOISY


# ---------------------------------------------------------------------------
# benefit_verdict
# ---------------------------------------------------------------------------

class TestBenefitVerdict:
    def test_benefit(self):
        verdict, _ = benefit_verdict(delta=0.5)
        assert verdict == BenefitVerdict.BENEFIT

    def test_no_benefit(self):
        verdict, _ = benefit_verdict(delta=-0.5)
        assert verdict == BenefitVerdict.NO_BENEFIT

    def test_ambiguous_near_zero(self):
        verdict, _ = benefit_verdict(delta=0.0)
        assert verdict == BenefitVerdict.AMBIGUOUS

    def test_nan_is_ambiguous(self):
        verdict, _ = benefit_verdict(delta=NaN)
        assert verdict == BenefitVerdict.AMBIGUOUS

    def test_just_above_eps(self):
        verdict, _ = benefit_verdict(delta=2e-6, eps=1e-6)
        assert verdict == BenefitVerdict.BENEFIT

    def test_just_below_eps(self):
        verdict, _ = benefit_verdict(delta=5e-7, eps=1e-6)
        assert verdict == BenefitVerdict.AMBIGUOUS


# ---------------------------------------------------------------------------
# invasion_verdict
# ---------------------------------------------------------------------------

class TestInvasionVerdict:
    THRESH = dict(win_threshold=7, lose_threshold=3, min_valid=4)

    def test_invades(self):
        verdict, _ = invasion_verdict(increase_count=7, n_valid=8, **self.THRESH)
        assert verdict == InvasionVerdict.INVADES

    def test_does_not_invade(self):
        verdict, _ = invasion_verdict(increase_count=2, n_valid=8, **self.THRESH)
        assert verdict == InvasionVerdict.DOES_NOT_INVADE

    def test_flat_or_noisy(self):
        verdict, _ = invasion_verdict(increase_count=5, n_valid=8, **self.THRESH)
        assert verdict == InvasionVerdict.FLAT_OR_NOISY

    def test_no_verdict_insufficient_valid(self):
        verdict, _ = invasion_verdict(increase_count=5, n_valid=3, **self.THRESH)
        assert verdict == InvasionVerdict.NO_VERDICT


# ---------------------------------------------------------------------------
# crosspartial_verdict
# ---------------------------------------------------------------------------

class TestCrosspartialVerdict:
    def test_controller_pays_alone(self):  # ACCEPTANCE
        # h adds nothing at either theta; theta adds 0.5 at low h
        eff = corner_effects(b_ll=1.0, b_hl=1.0, b_lh=1.5, b_hh=1.5)
        verdict, reason = crosspartial_verdict(eff)
        assert verdict == CrossPartialVerdict.CONTROLLER_PAYS_ALONE

    def test_joint_valley_plausible(self):  # ACCEPTANCE
        # Neither pays alone but combined pays: cross_partial = +1
        eff = corner_effects(b_ll=1.0, b_hl=1.0, b_lh=1.0, b_hh=2.0)
        verdict, reason = crosspartial_verdict(eff)
        assert verdict == CrossPartialVerdict.JOINT_VALLEY_PLAUSIBLE

    def test_trait_pays_alone(self):  # ACCEPTANCE
        # h improves birth at low theta; theta adds nothing extra
        eff = corner_effects(b_ll=1.0, b_hl=1.5, b_lh=1.0, b_hh=1.5)
        verdict, reason = crosspartial_verdict(eff)
        assert verdict == CrossPartialVerdict.TRAIT_PAYS_ALONE

    def test_antagonistic(self):
        # Destructive INTERACTION: cross_partial < 0 (h gets more harmful as theta rises).
        # cross = 0.5 - 1.5 - 1.5 + 1.0 = -1.5 < 0.
        eff = corner_effects(b_ll=1.0, b_hl=1.5, b_lh=1.5, b_hh=0.5)
        verdict, reason = crosspartial_verdict(eff)
        assert verdict == CrossPartialVerdict.ANTAGONISTIC

    def test_controller_pays_alone_with_h_cost_exp207(self):
        # REGRESSION (Exp 207 niche regime): theta pays alone (+0.147), h is a UNIFORM
        # cost at both theta (dB/dh = -0.046 and -0.041), cross_partial ~ +0.005 (NOT a
        # destructive interaction). Must be CONTROLLER_PAYS_ALONE, NOT ANTAGONISTIC —
        # "h is a cost at high theta" alone must not trip the antagonistic rule.
        eff = corner_effects(b_ll=0.1134, b_hl=0.0678, b_lh=0.2606, b_hh=0.2196)
        verdict, reason = crosspartial_verdict(eff)
        assert verdict == CrossPartialVerdict.CONTROLLER_PAYS_ALONE

    def test_no_interaction_flat(self):
        # Everything flat => cross_partial == 0
        eff = corner_effects(b_ll=1.0, b_hl=1.0, b_lh=1.0, b_hh=1.0)
        verdict, reason = crosspartial_verdict(eff)
        assert verdict == CrossPartialVerdict.NO_INTERACTION

    def test_nan_key_gives_no_verdict(self):
        eff = {
            "cross_partial":  NaN,
            "dB_dh_lo_theta": 0.0,
            "dB_dh_hi_theta": 0.0,
            "dB_dtheta_lo_h": 0.0,
            "dB_dtheta_hi_h": 0.0,
        }
        verdict, reason = crosspartial_verdict(eff)
        assert verdict == CrossPartialVerdict.NO_VERDICT

    def test_reason_is_string(self):
        eff = corner_effects(b_ll=1.0, b_hl=1.0, b_lh=1.5, b_hh=1.5)
        _, reason = crosspartial_verdict(eff)
        assert isinstance(reason, str)


# ---------------------------------------------------------------------------
# aggregate_verdict
# ---------------------------------------------------------------------------

class TestAggregateVerdict:
    def test_pass_local_gradient(self):  # ACCEPTANCE
        verdict, reason = aggregate_verdict(
            gradient=GradientVerdict.POSITIVE_LOCAL_GRADIENT,
            guards_all_pass=True,
        )
        assert verdict == AggregateVerdict.PASS_LOCAL_GRADIENT

    def test_positive_gradient_guard_fail_gives_no_verdict(self):  # ACCEPTANCE
        verdict, reason = aggregate_verdict(
            gradient=GradientVerdict.POSITIVE_LOCAL_GRADIENT,
            guards_all_pass=False,
        )
        assert verdict == AggregateVerdict.NO_VERDICT

    def test_no_verdict_from_no_gradient_verdict(self):  # ACCEPTANCE — collapsed pop
        verdict, reason = aggregate_verdict(
            gradient=GradientVerdict.NO_VERDICT,
            guards_all_pass=True,
        )
        assert verdict == AggregateVerdict.NO_VERDICT

    def test_global_benefit_only_monomorphic(self):  # ACCEPTANCE
        # Monomorphic optimum above resident + survivable, but gradient negative
        verdict, reason = aggregate_verdict(
            gradient=GradientVerdict.NEGATIVE_LOCAL_GRADIENT,
            monomorphic_above_resident=True,
            monomorphic_survivable=True,
            guards_all_pass=True,
        )
        assert verdict == AggregateVerdict.GLOBAL_BENEFIT_ONLY
        # MUST NOT be PASS
        assert verdict != AggregateVerdict.PASS_LOCAL_GRADIENT

    def test_controller_pays_alone(self):  # ACCEPTANCE
        verdict, reason = aggregate_verdict(
            gradient=GradientVerdict.NEGATIVE_LOCAL_GRADIENT,
            crosspartial=CrossPartialVerdict.CONTROLLER_PAYS_ALONE,
            guards_all_pass=True,
        )
        assert verdict == AggregateVerdict.CONTROLLER_PAYS_ALONE

    def test_gradient_none_controller_pays_alone(self):
        # Gate-H-only config (binding local gate NOT run, gradient=None): a
        # CONTROLLER_PAYS_ALONE cross-partial must SURFACE, not be masked as NO_VERDICT.
        verdict, reason = aggregate_verdict(
            gradient=None,
            crosspartial=CrossPartialVerdict.CONTROLLER_PAYS_ALONE,
            guards_all_pass=True,
        )
        assert verdict == AggregateVerdict.CONTROLLER_PAYS_ALONE

    def test_gradient_none_no_signal_is_no_verdict(self):
        # Gate not run and nothing else conclusive ⇒ NO_VERDICT (can't pass/fail without it).
        verdict, reason = aggregate_verdict(gradient=None, guards_all_pass=True)
        assert verdict == AggregateVerdict.NO_VERDICT

    def test_global_benefit_only_from_benefit_verdict(self):
        verdict, reason = aggregate_verdict(
            gradient=GradientVerdict.FLAT_OR_NOISY,
            benefit=BenefitVerdict.BENEFIT,
            guards_all_pass=True,
        )
        assert verdict == AggregateVerdict.GLOBAL_BENEFIT_ONLY

    def test_no_effect(self):
        verdict, reason = aggregate_verdict(
            gradient=GradientVerdict.NEGATIVE_LOCAL_GRADIENT,
            benefit=BenefitVerdict.NO_BENEFIT,
            monomorphic_above_resident=False,
            guards_all_pass=True,
        )
        assert verdict == AggregateVerdict.NO_EFFECT

    def test_fail_local_gradient_residual(self):
        verdict, reason = aggregate_verdict(
            gradient=GradientVerdict.NEGATIVE_LOCAL_GRADIENT,
            guards_all_pass=True,
        )
        assert verdict == AggregateVerdict.FAIL_LOCAL_GRADIENT

    def test_none_args_do_not_satisfy_conditions(self):
        # monomorphic_above_resident=None should not trigger GLOBAL_BENEFIT_ONLY
        verdict, reason = aggregate_verdict(
            gradient=GradientVerdict.NEGATIVE_LOCAL_GRADIENT,
            monomorphic_above_resident=None,
            monomorphic_survivable=None,
            guards_all_pass=True,
        )
        # Should fall to FAIL, not GLOBAL_BENEFIT_ONLY
        assert verdict != AggregateVerdict.GLOBAL_BENEFIT_ONLY

    def test_none_monomorphic_with_no_benefit_not_no_effect(self):
        # None must NOT be treated as False — rule 3d requires explicit False
        verdict, reason = aggregate_verdict(
            gradient=GradientVerdict.NEGATIVE_LOCAL_GRADIENT,
            benefit=BenefitVerdict.NO_BENEFIT,
            monomorphic_above_resident=None,
            guards_all_pass=True,
        )
        assert verdict != AggregateVerdict.NO_EFFECT

    def test_flat_gradient_guard_pass_does_not_pass(self):
        # FLAT gradient can never become PASS even with guards all passing
        verdict, reason = aggregate_verdict(
            gradient=GradientVerdict.FLAT_OR_NOISY,
            guards_all_pass=True,
        )
        assert verdict != AggregateVerdict.PASS_LOCAL_GRADIENT

    def test_reason_is_string(self):
        _, reason = aggregate_verdict(
            gradient=GradientVerdict.POSITIVE_LOCAL_GRADIENT,
            guards_all_pass=True,
        )
        assert isinstance(reason, str)
        assert len(reason) > 0
