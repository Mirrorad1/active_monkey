"""
tests/test_evolvability_autosize.py — verifies auto-parallelism sizing for Evolvability Preflights.

Tests:
  1. serial fallback for small batches and max_workers=1
  2. worker count clamped by seeds and memory
  3. user ceiling cap respected
  4. PreflightConfig.max_workers round-trips through to_dict/from_dict/config_hash
  5. runner records repro["workers"] as an int >= 1
"""
from __future__ import annotations

import dataclasses as D

import pytest

from ecology.evolvability.config import PreflightConfig
from ecology.evolvability.runner import run_preflight
from ecology.evolvability.trait_axis import make_axis
from ecology.runtime_budget import recommended_workers_for
from ecology.scenarios import SCENARIOS


# ---------------------------------------------------------------------------
# Shared tiny base config (active sensing enabled, small cap)
# ---------------------------------------------------------------------------

def _tiny_base():
    return D.replace(
        SCENARIOS["balanced"],
        horizon=120,
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


# ---------------------------------------------------------------------------
# test_serial_for_small_batch
# ---------------------------------------------------------------------------

def test_serial_for_small_batch():
    """Fewer seeds than min_parallel_jobs (default 4) must return 1 (serial)."""
    base = _tiny_base()
    assert recommended_workers_for(base, 2, horizon=120) == 1


def test_serial_when_max_workers_is_1():
    """Explicit max_workers=1 ceiling must always return 1 regardless of batch size."""
    base = _tiny_base()
    assert recommended_workers_for(base, 8, horizon=120, max_workers=1) == 1


# ---------------------------------------------------------------------------
# test_capped_by_seeds_and_memory
# ---------------------------------------------------------------------------

def test_capped_by_seeds_and_memory():
    """Worker count must be in [1, n_seeds] regardless of machine RAM."""
    base = _tiny_base()
    w = recommended_workers_for(base, 6, horizon=120)
    assert 1 <= w <= 6


# ---------------------------------------------------------------------------
# test_user_ceiling_caps
# ---------------------------------------------------------------------------

def test_user_ceiling_caps():
    """User-supplied max_workers ceiling must be respected even if RAM would allow more."""
    base = _tiny_base()
    w = recommended_workers_for(base, 16, horizon=120, max_workers=2)
    assert w <= 2


# ---------------------------------------------------------------------------
# test_config_roundtrips_max_workers
# ---------------------------------------------------------------------------

def test_config_roundtrips_max_workers():
    """max_workers=3 survives to_dict / from_dict and is included in config_hash."""
    cfg = PreflightConfig(slug="mw-test", max_workers=3)
    assert cfg.max_workers == 3

    d = cfg.to_dict()
    assert d["max_workers"] == 3

    cfg2 = PreflightConfig.from_dict(d)
    assert cfg2.max_workers == 3

    # config_hash must be deterministic and encode max_workers
    assert cfg.config_hash() == cfg.config_hash()
    cfg_none = PreflightConfig(slug="mw-test", max_workers=None)
    assert cfg.config_hash() != cfg_none.config_hash()


# ---------------------------------------------------------------------------
# test_runner_records_workers
# ---------------------------------------------------------------------------

def test_runner_records_workers(tmp_path):
    """run_preflight must record repro['workers'] as an int >= 1, and complete successfully."""
    axis = make_axis("information_sampling_rate")

    cfg = PreflightConfig(
        slug="autosize-runner-test",
        base_scenario="balanced",
        base_overrides={
            "enable_hidden_mode": True,
            "enable_active_sensing": True,
            "mode_wrong_regen_factor": 1.0,
            "mode_hazard_scale": 0.6,
            "capacity": 30.0,
            "regen_rate": 2.0,
            "initial_resource": 0.7,
            "max_population": 30000,
            "mode_switch_prob": 0.05,
            "cue_noise": 1.0,
            "memory_cost_slope": 0.005,
            "memory_upkeep_floor": 0.0,
            "probe_cost": 0.01,
            "probe_n_samples": 4,
            "shuffle_creature_order": True,
        },
        trait=axis,
        seeds=(0, 1, 2, 3),
        horizon=120,
        measurement_window=(20, 100),
        gates=("local_pairwise_gradient", "null_guards"),
        min_population=1,
        min_valid_seeds=1,
        output_dir=str(tmp_path),
    )

    result = run_preflight(cfg)

    assert isinstance(result.repro["workers"], int)
    assert result.repro["workers"] >= 1
    assert result.aggregate_verdict != ""
