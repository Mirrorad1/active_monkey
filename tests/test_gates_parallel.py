"""Parallel gate seed-loop must be bit-identical to serial execution.

Mirror of tests/test_ecology_batch.py: parallel == serial for both GENERIC
Evolvability Preflight gates that support max_workers.
"""
from __future__ import annotations

import dataclasses as D

from ecology.evolvability import gates as G
from ecology.evolvability.trait_axis import make_axis
from ecology.scenarios import SCENARIOS

SEEDS = [0, 1, 2, 3]


def _base():
    return D.replace(
        SCENARIOS["balanced"],
        horizon=150,
        enable_hidden_mode=True,
        enable_active_sensing=True,
        mode_wrong_regen_factor=1.0,
        mode_hazard_scale=0.6,
        capacity=30.0,
        regen_rate=2.0,
        initial_resource=0.7,
        max_population=30000,
        mode_switch_prob=0.05,
        cue_noise=1.0,
        memory_cost_slope=0.005,
        memory_upkeep_floor=0.0,
        probe_cost=0.01,
        probe_n_samples=4,
        shuffle_creature_order=True,
    )


def test_pairwise_gradient_parallel_equals_serial():
    """Parallel run_local_pairwise_gradient must be bit-identical to serial."""
    base = _base()
    axis = make_axis("information_sampling_rate")

    outcome_serial = G.run_local_pairwise_gradient(
        base, axis, SEEDS,
        win_threshold=3, lose_threshold=1, min_valid=1,
        window=(20, 140), min_pop=1,
        max_workers=1,
    )
    outcome_parallel = G.run_local_pairwise_gradient(
        base, axis, SEEDS,
        win_threshold=3, lose_threshold=1, min_valid=1,
        window=(20, 140), min_pop=1,
        max_workers=2,
    )

    assert outcome_serial.verdict == outcome_parallel.verdict
    assert outcome_serial.aggregate == outcome_parallel.aggregate
    assert outcome_serial.raw_rows == outcome_parallel.raw_rows


def test_invasion_parallel_equals_serial():
    """Parallel run_invasion_from_rarity must be bit-identical to serial."""
    base = _base()
    axis = make_axis("information_sampling_rate")

    outcome_serial = G.run_invasion_from_rarity(
        base, axis, SEEDS,
        win_threshold=3, lose_threshold=1, min_valid=1,
        window=(20, 140), min_pop=1,
        max_workers=1,
    )
    outcome_parallel = G.run_invasion_from_rarity(
        base, axis, SEEDS,
        win_threshold=3, lose_threshold=1, min_valid=1,
        window=(20, 140), min_pop=1,
        max_workers=2,
    )

    assert outcome_serial.verdict == outcome_parallel.verdict
    assert outcome_serial.aggregate == outcome_parallel.aggregate
    assert outcome_serial.raw_rows == outcome_parallel.raw_rows
