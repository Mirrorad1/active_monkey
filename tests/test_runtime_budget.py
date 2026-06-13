"""Guards for the runtime / complexity pre-flight (L25).

The pre-flight must (i) pass a normal bounded ecology run, (ii) be LOGISTIC-AWARE — not cry
"explosion" on a healthy exponential-then-plateau run, and (iii) actually FIRE when a config's
projected population would approach the runaway guard. These pin the guard so it stays useful
(a guard that always flags, or never flags, is worthless).
"""
from __future__ import annotations

import dataclasses as D

from ecology import runtime_budget as RB
from ecology.scenarios import SCENARIOS


def _balanced(horizon=2000, max_pop=20000):
    return D.replace(SCENARIOS["balanced"], horizon=horizon, max_population=max_pop)


def test_normal_run_is_safe_and_decelerating():
    """A standard balanced regime plateaus (logistic) -> SAFE, decelerating, not flagged."""
    cfg = _balanced()
    rep = RB.preflight([("balanced", cfg, 0)], horizon=2000, n_jobs=10, max_workers=5,
                       probe_steps=600, time_budget_s=3600)
    assert rep["safe"], rep["flags"]
    assert rep["configs"][0]["decelerating"] is True
    # projected population is well under the (generous) guard
    assert rep["configs"][0]["proj_pop"] < cfg.max_population


def test_explosion_flag_fires_when_guard_is_near_the_plateau():
    """If max_population is set BELOW the regime's natural carrying capacity, the projected pop
    approaches the guard -> EXPLOSION flag fires (the guard is not a no-op)."""
    cfg = _balanced(max_pop=40)        # far below the balanced carrying capacity
    rep = RB.preflight([("tiny_guard", cfg, 0)], horizon=2000, n_jobs=10, max_workers=5,
                       probe_steps=600, time_budget_s=3600)
    assert not rep["safe"]
    assert any("EXPLOSION" in f for f in rep["flags"])


def test_require_safe_raises_on_flag():
    cfg = _balanced(max_pop=40)
    try:
        RB.preflight([("tiny_guard", cfg, 0)], horizon=2000, n_jobs=10, max_workers=5,
                     probe_steps=600, time_budget_s=3600, require_safe=True)
    except AssertionError as e:
        assert "pre-flight FAILED" in str(e)
    else:
        raise AssertionError("require_safe=True should have raised on the EXPLOSION flag")


def test_over_budget_flag_fires_on_tiny_budget():
    cfg = _balanced()
    rep = RB.preflight([("balanced", cfg, 0)], horizon=2000, n_jobs=1000, max_workers=1,
                       probe_steps=400, time_budget_s=1.0)     # absurdly tight budget
    assert not rep["safe"]
    assert any("OVER_BUDGET" in f for f in rep["flags"])
