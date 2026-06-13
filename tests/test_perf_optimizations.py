"""Guards for the ecology hot-path / memory optimizations (perf/ecology-hotpath).

These pin the BEHAVIOR-PRESERVING properties the optimizations rely on:
  - the precomputed neighbor table returns exactly the old per-call values/order;
  - _alive() returns the alive creatures in ascending-id order WITHOUT a per-step sort
    (the list is sorted by construction — the optimization's correctness precondition);
  - dead creatures' belief maps (m, visit_t) are freed on death (the memory win) while the
    policy object + band_estimate survive for post-hoc inspection;
  - none of it changes events_hash (byte-identical — the historical-hash tests cover this too).
"""
from __future__ import annotations

import dataclasses as D

import numpy as np

from ecology.engine import Ecology
from ecology.world import GridWorld
from ecology.scenarios import SCENARIOS, FOUNDER

_BASE = dict(
    enable_thermosense=True, enable_temperature=True, temperature_stress_scale=0.0,
    thermosense_upkeep_floor=0.0, thermosense_active_threshold=0.05, thermosense_noise_base=0.5,
    thermal_avoidance_weight=4.0, food_optimal_base=0.5, food_optimal_amplitude=0.3,
    food_optimal_period=1500.0, food_concentration=8.0, food_band_width=0.15,
    enable_food_coupling=True, thermosense_forage_mode=True,
)


def _cfg(horizon=1500, regen=0.20, **kw):
    f = D.replace(FOUNDER, thermosense_intensity=0.10, thermosense_inefficiency=0.2, temperature_tolerance=0.10)
    return D.replace(SCENARIOS["balanced"], horizon=horizon, max_population=20000, founder=f,
                     regen_rate=regen, **_BASE, **kw)


def _slow_neighbors(world: GridWorld, pos: int) -> list[int]:
    """The pre-optimization per-call neighbor computation (the reference)."""
    r, c = divmod(pos, world.cols)
    out = []
    if r > 0: out.append((r - 1) * world.cols + c)
    if r < world.rows - 1: out.append((r + 1) * world.cols + c)
    if c > 0: out.append(r * world.cols + (c - 1))
    if c < world.cols - 1: out.append(r * world.cols + (c + 1))
    return out


def test_neighbor_table_matches_old_per_call():
    """The cached neighbor table is byte-identical (values AND order) to the old computation."""
    res = np.full(144, 5.0)
    w = GridWorld(rows=12, cols=12, resource=res, capacity=10.0, regen_rate=0.1)
    for pos in range(144):
        assert w.neighbors(pos) == _slow_neighbors(w, pos)


def test_alive_list_stays_sorted_by_construction():
    """_alive() drops the per-step sort because _alive_list is ascending by construction.
    Verify the precondition holds across a high-turnover run (shuffle ON, the stress case)."""
    eco = Ecology(_cfg(horizon=1200, regen=0.08, shuffle_creature_order=True), 38)
    eco.run()
    ids = [c.creature_id for c in eco._alive_list]
    assert ids == sorted(ids)
    assert [c.creature_id for c in eco._alive()] == ids       # _alive() returns that order


def test_dead_creatures_release_belief_maps_keep_band_estimate():
    """On death the two n_cells maps are freed (memory); policy object + band_estimate survive."""
    cfg = D.replace(_cfg(horizon=600, enable_band_staleness=True), food_optimal_period=80.0)
    eco = Ecology(cfg, 7)
    eco.run()
    dead = [c for c in eco._creatures if not c.phenotype.alive]
    alive = eco._alive()
    assert dead, "no deaths in the run — test regime invalid"
    for c in dead:
        assert c.policy is not None                            # object kept (post-hoc inspectable)
        assert c.policy.m is None and c.policy.visit_t is None  # the big arrays freed
    for c in alive:
        assert c.policy.m is not None                          # the living keep their maps
    # band_estimate (a float) survives on the dead for post-hoc inspection (the exp201 guard)
    assert any(c.policy.band_estimate is not None for c in eco._creatures)


def test_optimizations_preserve_events_hash():
    """Determinism is untouched: same seed -> same hash (the historical-hash tests pin the
    actual VALUES against the pre-optimization baseline; this pins reproducibility)."""
    a = Ecology(_cfg(horizon=1200), 50); a.run()
    b = Ecology(_cfg(horizon=1200), 50); b.run()
    assert a.events_hash() == b.events_hash()
