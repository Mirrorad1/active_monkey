"""Behavioral tests for the Exp 278 moving-resource posability harness."""
from __future__ import annotations

from dataclasses import replace

import pytest

from active_loop.benchmarks.emergent_comm_posability import (
    PosabilityConfig,
    grade_posability,
    resource_path,
    run_batch,
    run_seed,
)


def test_static_control_calls_the_shared_trail_policy(monkeypatch):
    import active_loop.benchmarks.emergent_comm_posability as harness

    cfg = PosabilityConfig(n_sites=8, rounds=40, signal_cost=0.05)
    monkeypatch.setattr(
        harness,
        "_trail_predictions",
        lambda path, rng: [0] * len(path),
    )
    row = harness.run_seed(401, cfg)
    assert row.static_trail_success < 0.95


def test_oracle_score_uses_gifted_token_decode_and_costed_payoff(monkeypatch):
    import active_loop.benchmarks.emergent_comm_posability as harness

    cfg = PosabilityConfig(n_sites=8, rounds=40, signal_cost=0.05)
    monkeypatch.setattr(
        harness,
        "_decode_token",
        lambda token, token_to_site: (token_to_site[token] + 1) % cfg.n_sites,
    )
    row = harness.run_seed(401, cfg)
    assert row.oracle_success < 1.0
    assert row.oracle_net_reward == row.oracle_success - cfg.signal_cost


def test_moving_resource_never_repeats_and_is_deterministic():
    cfg = PosabilityConfig(n_sites=4, rounds=40, signal_cost=0.05)
    a = resource_path(17, cfg, static=False)
    b = resource_path(17, cfg, static=False)
    assert a == b
    assert len(a) == 40
    assert all(left != right for left, right in zip(a, a[1:]))


def test_oracle_is_perfect_but_pays_the_declared_cost():
    cfg = PosabilityConfig(n_sites=8, rounds=200, signal_cost=0.05)
    row = run_seed(401, cfg)
    assert row.oracle_success == 1.0
    assert row.oracle_net_reward == 0.95


def test_static_positive_control_proves_trail_policy_is_live():
    cfg = PosabilityConfig(n_sites=8, rounds=200, signal_cost=0.05)
    row = run_seed(401, cfg)
    assert row.static_trail_success >= 199 / 200


def test_predeclared_gate_accepts_clear_posability_rows():
    cfg = PosabilityConfig(n_sites=8, rounds=2000, signal_cost=0.05)
    rows = tuple(run_seed(seed, cfg) for seed in range(401, 409))
    gate = grade_posability(rows, cfg)
    assert gate.baselines_fail
    assert gate.oracle_advantage
    assert gate.costed_oracle_viable
    assert gate.static_control_live
    assert gate.verdict == "POSABLE"


def test_parallel_batch_is_identical_to_serial():
    cfg = PosabilityConfig(n_sites=8, rounds=200, signal_cost=0.05)
    seeds = (401, 402, 403)
    assert run_batch(seeds, cfg, parallel=False) == run_batch(seeds, cfg, parallel=True, max_workers=2)


def test_gate_fails_closed_when_one_row_violates_a_conjunct():
    cfg = PosabilityConfig(n_sites=8, rounds=2000, signal_cost=0.05)
    rows = tuple(run_seed(seed, cfg) for seed in range(401, 409))
    damaged = replace(rows[0], static_trail_success=0.0)
    gate = grade_posability((damaged, *rows[1:]), cfg)
    assert not gate.static_control_live
    assert gate.verdict == "CANT_POSE"


def test_zero_signal_cost_is_rejected():
    with pytest.raises(ValueError, match="signal_cost"):
        PosabilityConfig(signal_cost=0.0)
