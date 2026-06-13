"""ecology/batch.py — the parallel batch runner must be bit-identical to sequential.

The whole correctness argument for parallelising experiment runs is that each
Ecology(cfg, seed) is independent and deterministic, so concurrency cannot change
any result.  These guards pin that: parallel == sequential events_hash, and the
batch runner's per-run summary matches a direct Ecology run.
"""
from __future__ import annotations

import dataclasses as D

from ecology.batch import RunSpec, run_batch, run_one, default_workers
from ecology.engine import Ecology
from ecology.scenarios import SCENARIOS, FOUNDER


def _cfg(horizon=400):
    f = D.replace(FOUNDER, thermosense_intensity=0.10, thermosense_inefficiency=0.2,
                  temperature_tolerance=0.10)
    return D.replace(
        SCENARIOS["balanced"], horizon=horizon, max_population=5000, founder=f,
        enable_thermosense=True, enable_temperature=True, temperature_stress_scale=0.0,
        thermosense_upkeep_floor=0.0, thermosense_active_threshold=0.05,
        thermosense_noise_base=0.5, thermal_avoidance_weight=4.0,
        food_optimal_base=0.5, food_optimal_amplitude=0.3, food_concentration=8.0,
        enable_food_coupling=True, thermosense_forage_mode=True,
        enable_band_staleness=True, band_responsiveness=1.0,
    )


def _specs():
    cfg = _cfg()
    return [RunSpec(key=("A", s), cfg=cfg, seed=s, window_start=0, checkpoint_stride=200,
                    trait_means=("thermosense_intensity", "learning_rate"))
            for s in (1, 2, 3, 4, 5)]


def test_default_workers_sane():
    assert default_workers() >= 1


def test_parallel_equals_sequential_hashes():
    """The load-bearing guarantee: concurrency does not change any events_hash."""
    specs = _specs()
    seq = run_batch(specs, sequential=True)
    par = run_batch(specs)
    assert set(seq) == set(par)
    for k in seq:
        assert seq[k]["events_hash"] == par[k]["events_hash"], f"parallel != sequential at {k}"
        assert seq[k]["end_means"] == par[k]["end_means"]


def test_run_one_matches_direct_ecology():
    """run_one's events_hash must equal a plain Ecology(cfg, seed).run()."""
    cfg = _cfg()
    direct = Ecology(cfg, seed=7)
    direct.run()
    spec = RunSpec(key="x", cfg=cfg, seed=7, window_start=0, checkpoint_stride=200)
    got = run_one(spec)
    assert got["events_hash"] == direct.events_hash()


def test_run_one_reports_requested_traits():
    spec = RunSpec(key="x", cfg=_cfg(), seed=3, window_start=0, checkpoint_stride=200,
                   trait_means=("thermosense_intensity", "learning_rate"))
    r = run_one(spec)
    assert set(r["end_means"]) == {"thermosense_intensity", "learning_rate"}
    assert "events_hash" in r and r["steps_run"] > 0
