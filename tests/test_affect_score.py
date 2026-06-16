"""Guard tests for eval.affect_score (M4a increment 1e).

Fast tests (no slow marker): run in CI with every push.
Slow tests (pytest.mark.slow): skipped by default; the conductor runs them.
"""
from __future__ import annotations

import pytest

from eval.affect_score import (
    AffectScoreReport,
    score_affect,
    _constant_factory,
    CEIL,
    GENUINE_FLOOR,
)


def test_affect_score_imports_and_shape():
    """FAST: one short seed — checks shape and types without asserting verdict."""
    report = score_affect(seeds=(20,), turns=40)
    assert isinstance(report, AffectScoreReport)
    assert isinstance(report.metric, float)
    assert isinstance(report.mean_first, float)
    assert isinstance(report.mean_last, float)
    assert isinstance(report.improvement, float)
    assert isinstance(report.genuine_fraction, float)
    assert isinstance(report.ask_rate, float)
    assert isinstance(report.n_seeds, int)
    assert isinstance(report.guardrails, dict)
    assert isinstance(report.verdict, bool)
    assert report.n_seeds == 1
    assert set(report.guardrails) == {
        "realized_above_ceiling",
        "learned_improvement",
        "genuine_reliable",
    }
    assert 0.0 <= report.genuine_fraction <= 1.0
    assert 0.0 <= report.ask_rate <= 1.0


def test_constant_control_fails_verdict():
    """ANTI-HACK GUARD (load-bearing): a constant-response policy CANNOT fake the metric.

    A constant policy that always emits response 0 cannot reach correct_select >= 0.5
    (it maps only 2/6 codes correctly) and therefore genuine_fraction must stay below 0.5.
    verdict must be False.
    """
    report = score_affect(
        seeds=tuple(range(20, 24)),
        turns=120,
        agent_factory=_constant_factory(0),
    )
    assert report.verdict is False, (
        f"ANTI-HACK GUARD FAILED: a constant-0 policy returned verdict=True. "
        f"genuine_fraction={report.genuine_fraction:.3f} mean_last={report.mean_last:.3f}"
    )
    assert report.genuine_fraction < 0.5, (
        f"Constant policy should have genuine_fraction < 0.5; got {report.genuine_fraction:.3f}"
    )


@pytest.mark.slow
def test_genuine_learner_passes_at_default():
    """SLOW (conductor runs this): full default 300t×8 seeds must pass the verdict."""
    report = score_affect()
    assert report.verdict is True, (
        f"Expected verdict=True at full defaults; got guardrails={report.guardrails} "
        f"mean_last={report.mean_last:.3f} genuine_fraction={report.genuine_fraction:.3f}"
    )
    assert report.mean_last > 1 / 3, (
        f"mean_last should exceed ceiling 1/3; got {report.mean_last:.3f}"
    )
    assert report.genuine_fraction >= 0.5, (
        f"genuine_fraction should be >= 0.5; got {report.genuine_fraction:.3f}"
    )
