import numpy as np
import pytest

from eval.score import score_suite, ScoreReport


@pytest.mark.slow
def test_score_returns_finite_metric_and_verdict():
    report = score_suite()
    assert isinstance(report, ScoreReport)
    assert np.isfinite(report.metric)
    assert isinstance(report.verdict, bool)
    assert set(report.guardrails) == {"success_floor", "ask_rate_band"}


@pytest.mark.slow
def test_score_is_deterministic():
    a = score_suite()
    b = score_suite()
    assert a.metric == b.metric
    assert a.guardrails == b.guardrails
