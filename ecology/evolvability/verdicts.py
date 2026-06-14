"""
ecology.evolvability.verdicts — verdict enums and decision functions for the
Evolvability Preflight framework.

Each verdict function returns (verdict_enum, reason: str) where reason is a
one-line human explanation citing the key numbers.

Module-level threshold defaults
--------------------------------
DEFAULT_FLAT_EPS    = 1e-9   mean-effect magnitude below which a win-count result
                              is treated as flat rather than directional.
DEFAULT_CROSS_EPS   = 1e-6   margin for cross-partial interaction tests.
DEFAULT_BENEFIT_EPS = 1e-6   margin for benefit-delta comparisons.

All thresholds are function parameters with these defaults so callers can
override them per experiment.

Precedence rules are documented in each function's docstring.
"""

import enum
import math
from typing import Optional

DEFAULT_FLAT_EPS: float = 1e-9
DEFAULT_CROSS_EPS: float = 1e-6
DEFAULT_BENEFIT_EPS: float = 1e-6

NaN = float("nan")


# ---------------------------------------------------------------------------
# Verdict enums  (str mixin so .value is JSON-friendly and str() round-trips)
# ---------------------------------------------------------------------------

class BenefitVerdict(str, enum.Enum):
    """Verdict on whether the trait produces a net benefit."""
    BENEFIT    = "BENEFIT"
    NO_BENEFIT = "NO_BENEFIT"
    AMBIGUOUS  = "AMBIGUOUS"


class GradientVerdict(str, enum.Enum):
    """Verdict on the local evolutionary gradient direction."""
    POSITIVE_LOCAL_GRADIENT = "POSITIVE_LOCAL_GRADIENT"
    NEGATIVE_LOCAL_GRADIENT = "NEGATIVE_LOCAL_GRADIENT"
    FLAT_OR_NOISY           = "FLAT_OR_NOISY"
    NO_VERDICT              = "NO_VERDICT"


class InvasionVerdict(str, enum.Enum):
    """Verdict on whether a mutant can invade a resident population."""
    INVADES          = "INVADES"
    DOES_NOT_INVADE  = "DOES_NOT_INVADE"
    FLAT_OR_NOISY    = "FLAT_OR_NOISY"
    NO_VERDICT       = "NO_VERDICT"


class CrossPartialVerdict(str, enum.Enum):
    """Verdict on the interaction structure between trait h and controller theta."""
    JOINT_VALLEY_PLAUSIBLE = "JOINT_VALLEY_PLAUSIBLE"
    CONTROLLER_PAYS_ALONE  = "CONTROLLER_PAYS_ALONE"
    TRAIT_PAYS_ALONE       = "TRAIT_PAYS_ALONE"
    NO_INTERACTION         = "NO_INTERACTION"
    ANTAGONISTIC           = "ANTAGONISTIC"
    NO_VERDICT             = "NO_VERDICT"


class GuardStatus(str, enum.Enum):
    """Status of a null/cheat guard check."""
    PASS            = "PASS"
    FAIL            = "FAIL"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    NA              = "NA"


class AggregateVerdict(str, enum.Enum):
    """Overall aggregate verdict for a preflight run."""
    PASS_LOCAL_GRADIENT  = "PASS_LOCAL_GRADIENT"
    FAIL_LOCAL_GRADIENT  = "FAIL_LOCAL_GRADIENT"
    GLOBAL_BENEFIT_ONLY  = "GLOBAL_BENEFIT_ONLY"
    CONTROLLER_PAYS_ALONE = "CONTROLLER_PAYS_ALONE"
    NO_EFFECT            = "NO_EFFECT"
    NO_VERDICT           = "NO_VERDICT"


# ---------------------------------------------------------------------------
# Verdict functions
# ---------------------------------------------------------------------------

def gradient_verdict(
    wins: int,
    n_valid: int,
    mean_effect: float,
    *,
    win_threshold: int,
    lose_threshold: int,
    min_valid: int,
    flat_eps: float = DEFAULT_FLAT_EPS,
) -> tuple:
    """Decide the local evolutionary gradient verdict.

    Precedence
    ----------
    1. n_valid < min_valid
       → NO_VERDICT  (insufficient valid seeds; population collapse or too few replicates)
    2. wins >= win_threshold AND mean_effect is not nan AND mean_effect > flat_eps
       → POSITIVE_LOCAL_GRADIENT
       NOTE: a high win-count paired with nan or non-positive mean_effect does NOT
       produce a positive verdict — it falls through to rule 3 or 4.
    3. wins <= lose_threshold
       → NEGATIVE_LOCAL_GRADIENT
    4. otherwise
       → FLAT_OR_NOISY

    Parameters
    ----------
    wins          : number of seeds where mutant outperformed resident
    n_valid       : number of seeds with non-nan metrics
    mean_effect   : mean(mutant - resident) over valid seeds
    win_threshold : minimum wins for a POSITIVE verdict (from default_thresholds)
    lose_threshold: maximum wins consistent with a NEGATIVE verdict
    min_valid     : minimum n_valid required to issue any verdict
    flat_eps      : mean-effect magnitude below which a win-count is treated as flat
    """
    if n_valid < min_valid:
        return (
            GradientVerdict.NO_VERDICT,
            f"insufficient valid seeds: n_valid={n_valid} < min_valid={min_valid}",
        )
    me_positive = (not math.isnan(mean_effect)) and (mean_effect > flat_eps)
    if wins >= win_threshold and me_positive:
        return (
            GradientVerdict.POSITIVE_LOCAL_GRADIENT,
            f"wins={wins}/{n_valid} >= win_threshold={win_threshold} "
            f"and mean_effect={mean_effect:.4g} > flat_eps={flat_eps:.2g}",
        )
    if wins <= lose_threshold:
        return (
            GradientVerdict.NEGATIVE_LOCAL_GRADIENT,
            f"wins={wins}/{n_valid} <= lose_threshold={lose_threshold}",
        )
    return (
        GradientVerdict.FLAT_OR_NOISY,
        f"wins={wins}/{n_valid} in ambiguous range "
        f"[{lose_threshold+1}, {win_threshold-1}] or mean_effect not positive "
        f"(mean_effect={mean_effect if not math.isnan(mean_effect) else 'nan'})",
    )


def benefit_verdict(
    delta: float,
    *,
    eps: float = DEFAULT_BENEFIT_EPS,
) -> tuple:
    """Decide whether the trait produces a net benefit.

    delta = B_high - B_low (e.g. mean birth rate or fitness at high vs low trait value).

    Precedence
    ----------
    nan or |delta| <= eps → AMBIGUOUS
    delta > eps           → BENEFIT
    delta < -eps          → NO_BENEFIT

    Parameters
    ----------
    delta : B_high - B_low; positive means the high trait value outperforms low
    eps   : margin below which the difference is treated as ambiguous
    """
    if math.isnan(delta):
        return (
            BenefitVerdict.AMBIGUOUS,
            f"delta=nan; cannot determine benefit direction",
        )
    if delta > eps:
        return (
            BenefitVerdict.BENEFIT,
            f"delta={delta:.4g} > eps={eps:.2g}: high trait outperforms low",
        )
    if delta < -eps:
        return (
            BenefitVerdict.NO_BENEFIT,
            f"delta={delta:.4g} < -eps={-eps:.2g}: high trait underperforms low",
        )
    return (
        BenefitVerdict.AMBIGUOUS,
        f"delta={delta:.4g} within ±eps={eps:.2g}: ambiguous",
    )


def invasion_verdict(
    increase_count: int,
    n_valid: int,
    *,
    win_threshold: int,
    lose_threshold: int,
    min_valid: int,
) -> tuple:
    """Decide whether a mutant can invade a resident population.

    Precedence (same shape as gradient_verdict, without the mean-effect guard)
    ----------
    1. n_valid < min_valid        → NO_VERDICT
    2. increase_count >= win_threshold → INVADES
    3. increase_count <= lose_threshold → DOES_NOT_INVADE
    4. otherwise                  → FLAT_OR_NOISY

    Parameters
    ----------
    increase_count : number of seeds where mutant frequency increased
    n_valid        : number of valid seeds
    win_threshold  : from default_thresholds
    lose_threshold : from default_thresholds
    min_valid      : minimum n_valid required to issue any verdict
    """
    if n_valid < min_valid:
        return (
            InvasionVerdict.NO_VERDICT,
            f"insufficient valid seeds: n_valid={n_valid} < min_valid={min_valid}",
        )
    if increase_count >= win_threshold:
        return (
            InvasionVerdict.INVADES,
            f"increase_count={increase_count}/{n_valid} >= win_threshold={win_threshold}",
        )
    if increase_count <= lose_threshold:
        return (
            InvasionVerdict.DOES_NOT_INVADE,
            f"increase_count={increase_count}/{n_valid} <= lose_threshold={lose_threshold}",
        )
    return (
        InvasionVerdict.FLAT_OR_NOISY,
        f"increase_count={increase_count}/{n_valid} in ambiguous range "
        f"[{lose_threshold+1}, {win_threshold-1}]",
    )


def crosspartial_verdict(
    eff: dict,
    *,
    eps: float = DEFAULT_CROSS_EPS,
) -> tuple:
    """Decide the interaction structure between trait h and controller theta.

    eff is the dict returned by metrics.corner_effects.

    Precedence (applied in order; first matching rule wins)
    ----------
    1. Any required key is nan
       → NO_VERDICT
    2. dB_dh_hi_theta < -eps
       → ANTAGONISTIC  (sharper trait is WORSE when controller is engaged)
    3. cross_partial > eps AND dB_dh_lo_theta <= eps AND dB_dtheta_lo_h <= eps
       → JOINT_VALLEY_PLAUSIBLE  (neither pays alone; combined pays)
    4. dB_dtheta_lo_h > eps AND dB_dh_lo_theta <= eps AND dB_dh_hi_theta <= eps
       → CONTROLLER_PAYS_ALONE
    5. dB_dh_lo_theta > eps
       → TRAIT_PAYS_ALONE
    6. |cross_partial| <= eps
       → NO_INTERACTION
    7. residual
       → NO_INTERACTION  (with numbers in reason)

    Parameters
    ----------
    eff : dict with keys cross_partial, dB_dh_lo_theta, dB_dh_hi_theta,
          dB_dtheta_lo_h, dB_dtheta_hi_h
    eps : margin for interaction tests
    """
    required_keys = (
        "cross_partial",
        "dB_dh_lo_theta",
        "dB_dh_hi_theta",
        "dB_dtheta_lo_h",
        "dB_dtheta_hi_h",
    )
    # Rule 1: any nan → no verdict
    for k in required_keys:
        v = eff.get(k, NaN)
        if math.isnan(v):
            return (
                CrossPartialVerdict.NO_VERDICT,
                f"key '{k}' is nan; cannot classify interaction",
            )

    cp   = eff["cross_partial"]
    dh_lo = eff["dB_dh_lo_theta"]
    dh_hi = eff["dB_dh_hi_theta"]
    dt_lo = eff["dB_dtheta_lo_h"]

    # Rule 2: antagonistic
    if dh_hi < -eps:
        return (
            CrossPartialVerdict.ANTAGONISTIC,
            f"dB_dh_hi_theta={dh_hi:.4g} < -eps={-eps:.2g}: trait is WORSE when controller engaged",
        )
    # Rule 3: joint valley plausible
    if cp > eps and dh_lo <= eps and dt_lo <= eps:
        return (
            CrossPartialVerdict.JOINT_VALLEY_PLAUSIBLE,
            f"cross_partial={cp:.4g} > eps but dB_dh_lo_theta={dh_lo:.4g} and "
            f"dB_dtheta_lo_h={dt_lo:.4g} both <= eps: neither pays alone, combined pays",
        )
    # Rule 4: controller pays alone
    if dt_lo > eps and dh_lo <= eps and dh_hi <= eps:
        return (
            CrossPartialVerdict.CONTROLLER_PAYS_ALONE,
            f"dB_dtheta_lo_h={dt_lo:.4g} > eps, dB_dh_lo_theta={dh_lo:.4g} <= eps, "
            f"dB_dh_hi_theta={dh_hi:.4g} <= eps: controller pays; trait does not",
        )
    # Rule 5: trait pays alone
    if dh_lo > eps:
        return (
            CrossPartialVerdict.TRAIT_PAYS_ALONE,
            f"dB_dh_lo_theta={dh_lo:.4g} > eps: trait improves birth even at low theta",
        )
    # Rule 6: no interaction
    if abs(cp) <= eps:
        return (
            CrossPartialVerdict.NO_INTERACTION,
            f"cross_partial={cp:.4g}: |cp| <= eps={eps:.2g}: no meaningful interaction",
        )
    # Rule 7: residual
    return (
        CrossPartialVerdict.NO_INTERACTION,
        f"residual: cross_partial={cp:.4g}, dB_dh_lo_theta={dh_lo:.4g}, "
        f"dB_dh_hi_theta={dh_hi:.4g}, dB_dtheta_lo_h={dt_lo:.4g}",
    )


def aggregate_verdict(
    *,
    gradient: GradientVerdict,
    benefit: Optional[BenefitVerdict] = None,
    monomorphic_above_resident: Optional[bool] = None,
    monomorphic_survivable: Optional[bool] = None,
    crosspartial: Optional[CrossPartialVerdict] = None,
    guards_all_pass: bool,
) -> tuple:
    """Aggregate all sub-verdicts into a single experiment verdict.

    Precedence (applied in order; first matching rule wins)
    ----------
    1. gradient == NO_VERDICT
       → NO_VERDICT  (population invalid or insufficient data; no pass/fail)

    2. gradient == POSITIVE_LOCAL_GRADIENT:
       a. not guards_all_pass
          → NO_VERDICT  (suspected artifact; a null/cheat guard FAILED)
       b. guards_all_pass
          → PASS_LOCAL_GRADIENT

    3. gradient == NEGATIVE or FLAT (remaining cases):
       a. crosspartial == CONTROLLER_PAYS_ALONE
          → CONTROLLER_PAYS_ALONE
       b. monomorphic_above_resident is True AND monomorphic_survivable is True
          → GLOBAL_BENEFIT_ONLY  (optimum is above resident but gradient is locally negative)
       c. benefit == BENEFIT
          → GLOBAL_BENEFIT_ONLY
       d. benefit == NO_BENEFIT AND not monomorphic_above_resident
          → NO_EFFECT
       e. residual
          → FAIL_LOCAL_GRADIENT

    None arguments are treated as missing and never satisfy a positive condition.

    Parameters
    ----------
    gradient                 : from gradient_verdict()
    benefit                  : from benefit_verdict(); optional
    monomorphic_above_resident: True iff monomorphic optimum h* > resident h
    monomorphic_survivable    : True iff the monomorphic optimum is a viable population
    crosspartial             : from crosspartial_verdict(); optional
    guards_all_pass          : True iff all null/cheat guards passed
    """
    # Rule 1
    if gradient == GradientVerdict.NO_VERDICT:
        return (
            AggregateVerdict.NO_VERDICT,
            "local gradient gate produced no verdict (population invalid / insufficient data)",
        )

    # Rule 2
    if gradient == GradientVerdict.POSITIVE_LOCAL_GRADIENT:
        if not guards_all_pass:
            return (
                AggregateVerdict.NO_VERDICT,
                "positive local gradient but a null/cheat guard FAILED — "
                "suspected artifact, not a pass",
            )
        return (
            AggregateVerdict.PASS_LOCAL_GRADIENT,
            f"positive local gradient confirmed and all guards passed",
        )

    # Rule 3 (gradient is NEGATIVE or FLAT)
    # 3a
    if crosspartial == CrossPartialVerdict.CONTROLLER_PAYS_ALONE:
        return (
            AggregateVerdict.CONTROLLER_PAYS_ALONE,
            "gradient is not positive but controller alone drives the benefit — "
            "trait h is not the causal agent",
        )
    # 3b
    if monomorphic_above_resident is True and monomorphic_survivable is True:
        return (
            AggregateVerdict.GLOBAL_BENEFIT_ONLY,
            "monomorphic optimum is above resident and survivable, but local gradient "
            "is not positive — global benefit without local invasion advantage",
        )
    # 3c
    if benefit == BenefitVerdict.BENEFIT:
        return (
            AggregateVerdict.GLOBAL_BENEFIT_ONLY,
            "global benefit (B_high > B_low) detected but local gradient is not positive",
        )
    # 3d — None is treated as missing (not False), so must be explicit
    if benefit == BenefitVerdict.NO_BENEFIT and monomorphic_above_resident is False:
        return (
            AggregateVerdict.NO_EFFECT,
            "no benefit detected and monomorphic optimum is not above resident — "
            "trait appears to have no evolutionary effect",
        )
    # 3e residual
    return (
        AggregateVerdict.FAIL_LOCAL_GRADIENT,
        f"local gradient is {gradient.value}; no compensating signal found",
    )
