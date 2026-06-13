"""Guards for the ecology hot-path / memory optimizations (perf/ecology-hotpath).

These pin the BEHAVIOR-PRESERVING properties the optimizations rely on:
  - the precomputed neighbor table returns exactly the old per-call values/order;
  - _alive() returns the alive creatures in ascending-id order WITHOUT a per-step sort
    (the list is sorted by construction — the optimization's correctness precondition);
  - the explicit accessors (alive_count/has_alive/alive_snapshot) agree with _alive() and the
    snapshot is an independent copy (mutating it cannot corrupt the engine's _alive_list);
  - with shuffle OFF, step() iterates the maintained _alive_list directly (no per-step copy)
    and that aliasing neither corrupts the list nor changes events_hash;
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


def test_alive_accessors_agree_with_alive_midrun():
    """alive_count()/has_alive()/alive_snapshot() agree with _alive() at every step of a run —
    they are pure no-allocation (count/emptiness) or copy (snapshot) equivalents, never a
    different value."""
    eco = Ecology(_cfg(horizon=400, regen=0.10), 11)
    while eco.t < 400 and eco.has_alive():
        eco.step()
        ref = eco._alive()
        assert eco.alive_count() == len(ref)
        assert eco.has_alive() == bool(ref)
        assert [c.creature_id for c in eco.alive_snapshot()] == [c.creature_id for c in ref]


def test_alive_snapshot_and_legacy_alive_are_independent_copies():
    """alive_snapshot() and _alive() each return a FRESH list (never the raw _alive_list);
    mutating the result cannot corrupt the engine's bookkeeping. _alive() delegates to
    alive_snapshot() (a private copy)."""
    eco = Ecology(_cfg(horizon=200, regen=0.20), 3)
    for _ in range(80):
        if not eco.has_alive():
            break
        eco.step()
    before = list(eco._alive_list)
    assert before, "regime produced no living creatures — test invalid"
    snap = eco.alive_snapshot()
    legacy = eco._alive()
    assert snap is not eco._alive_list and legacy is not eco._alive_list   # both are copies
    assert snap is not legacy                                              # a distinct copy per call
    assert snap == before and legacy == before                            # same contents
    snap.clear(); legacy.reverse()                                        # mutate the copies
    assert eco._alive_list == before                                      # engine state untouched


def test_step_off_path_no_copy_is_byte_identical_and_safe():
    """Item 3: with shuffle OFF, step() iterates the maintained _alive_list DIRECTLY (no copy).
    Guard the two properties that make that aliasing correct over a high-turnover run:
      (a) the run still REPRODUCES its events_hash (aliasing did not corrupt the stream), and
      (b) _alive_list is left sorted-by-id + all-alive (the maintained invariant survives)."""
    cfg = _cfg(horizon=1000, regen=0.08, shuffle_creature_order=False)
    a = Ecology(cfg, 21); sa = a.run()
    b = Ecology(cfg, 21); sb = b.run()
    assert sa["events_hash"] == sb["events_hash"]             # deterministic / no aliasing corruption
    ids = [c.creature_id for c in a._alive_list]
    assert ids == sorted(ids)                                 # left sorted by construction
    assert all(c.phenotype.alive for c in a._alive_list)      # only-alive invariant preserved
    assert a.alive_count() == len(a._alive_list)


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
