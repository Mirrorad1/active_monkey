"""Tests for BeliefBench v0 (hidden partner-type inference). Fast + deterministic."""
from __future__ import annotations

from active_loop.benchmarks.beliefbench import (
    run_beliefbench, BeliefBenchReport, _emission, _filter_update, Z, O,
)
import numpy as np


def test_report_shape_and_verdict_domain():
    rep = run_beliefbench(seeds=(0, 1, 2), turns=120)
    assert isinstance(rep, BeliefBenchReport)
    assert rep.verdict in {"PASS", "FAIL", "INCONCLUSIVE"}
    assert set(rep.checks) == {
        "evidence_update", "action_relevance", "scramble_control", "held_out_transfer",
    }
    assert rep.n_seeds == 3


def test_belief_updates_and_drives_policy():
    """Evidence-update and action-relevance must hold (the load-bearing minimum)."""
    rep = run_beliefbench(seeds=(0, 1, 2, 3), turns=180)
    assert rep.belief_update_magnitude > 0.0
    assert rep.checks["evidence_update"] is True
    assert rep.checks["action_relevance"] is True


def test_scramble_hurts_and_beats_baselines():
    rep = run_beliefbench(seeds=tuple(range(5)), turns=200)
    # scrambling the posterior should reduce reward
    assert rep.reward_normal > rep.reward_scrambled
    # belief policy should not do worse than constant/random; oracle is the ceiling
    assert rep.reward_normal >= rep.reward_constant
    assert rep.reward_oracle >= rep.reward_normal


def test_deterministic():
    a = run_beliefbench(seeds=(0, 1), turns=120)
    b = run_beliefbench(seeds=(0, 1), turns=120)
    assert a.reward_normal == b.reward_normal
    assert a.verdict == b.verdict


def test_filter_update_is_a_distribution():
    emission = _emission(0.4)
    q = np.full(Z, 1.0 / Z)
    for o in range(O):
        q = _filter_update(q, o, emission, sharper=False)
        assert abs(q.sum() - 1.0) < 1e-9
        assert (q >= 0).all()
