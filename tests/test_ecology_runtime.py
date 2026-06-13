"""ecology/runtime.py — fork/snapshot/restore/replay/distill first-class primitives.

The load-bearing guard is LOSSLESSNESS: pausing (snapshot) and resuming (restore) a run is
BIT-IDENTICAL to never having paused. The rest pin fork divergence/faithfulness, replay bit-match,
distil validity, and pickle round-trip (pause-to-disk).
"""
from __future__ import annotations

import copy
import dataclasses as D
import pickle

import numpy as np
import pytest

from ecology.engine import Ecology
from ecology.scenarios import SCENARIOS, FOUNDER
from ecology.genotype import is_valid
from ecology.runtime import (
    snapshot, restore, fork, replay, distill, run_to, fork_run_compare,
)

MID, HORIZON = 150, 400


def _cfg():
    # A config with real dynamics (births/deaths, thermosense, shuffle) so the captured state is
    # non-trivial and the rng is exercised on the hot path.
    f = D.replace(FOUNDER, thermosense_intensity=0.10, thermosense_inefficiency=0.2, temperature_tolerance=0.10)
    return D.replace(
        SCENARIOS["balanced"], horizon=HORIZON, max_population=6000, founder=f,
        regen_rate=0.08, enable_thermosense=True, enable_temperature=True, temperature_stress_scale=0.0,
        thermosense_active_threshold=0.05, thermosense_noise_base=0.5, thermal_avoidance_weight=4.0,
        food_optimal_base=0.5, food_optimal_amplitude=0.3, food_optimal_period=1500.0,
        food_concentration=14.0, food_band_width=0.08,
        enable_food_coupling=True, thermosense_forage_mode=True,
        shuffle_creature_order=True, track_band_strip=True,
    )


def _straight_through():
    eco = Ecology(_cfg(), seed=7)
    while eco.t < HORIZON and eco._alive_list:
        eco.step()
    return eco.events_hash()


def test_pause_resume_is_bit_identical():
    """THE losslessness invariant: snapshot at MID, restore, continue -> identical to straight-through."""
    straight = _straight_through()
    eco = Ecology(_cfg(), seed=7)
    while eco.t < MID:
        eco.step()
    snap = snapshot(eco)
    resumed = restore(snap)
    while resumed.t < HORIZON and resumed._alive_list:
        resumed.step()
    assert resumed.events_hash() == straight


def test_snapshot_does_not_perturb_source():
    """Taking a snapshot must not alter the source run's future."""
    a = Ecology(_cfg(), seed=7)
    while a.t < MID:
        a.step()
    _ = snapshot(a)                       # snapshot the source
    while a.t < HORIZON and a._alive_list:
        a.step()
    assert a.events_hash() == _straight_through()


def test_fork_faithful_twin_is_identical():
    """fork(seed=None) continues identically to the source."""
    base = Ecology(_cfg(), seed=7)
    while base.t < MID:
        base.step()
    twin = fork(base)                     # faithful
    base_cont = restore(snapshot(base))
    for eco in (twin, base_cont):
        while eco.t < HORIZON and eco._alive_list:
            eco.step()
    assert twin.events_hash() == base_cont.events_hash()


def test_fork_reseeded_diverges_but_shares_past():
    base = Ecology(_cfg(), seed=7)
    while base.t < MID:
        base.step()
    pre_hash = base.events_hash()         # history up to the fork point
    twin = fork(base, seed=999)
    # the twin still carries the identical pre-fork history...
    assert twin.events_hash() == pre_hash
    # ...but a re-seeded future diverges from a faithful continuation.
    faithful = fork(base)
    for eco in (twin, faithful):
        while eco.t < HORIZON and eco._alive_list:
            eco.step()
    assert twin.events_hash() != faithful.events_hash()


def test_replay_bit_match_gate():
    base = Ecology(_cfg(), seed=7)
    while base.t < MID:
        base.step()
    snap = snapshot(base)
    eco = replay(snap, HORIZON)
    good = eco.events_hash()
    # replay reproduces; expect_hash passes when correct, raises when wrong.
    assert replay(snap, HORIZON, expect_hash=good).events_hash() == good
    with pytest.raises(AssertionError):
        replay(snap, HORIZON, expect_hash="deadbeef" * 8)


def test_snapshot_pickles_round_trip():
    """Pause-to-disk: a Snapshot pickles and the restored run continues identically."""
    base = Ecology(_cfg(), seed=7)
    while base.t < MID:
        base.step()
    snap = snapshot(base)
    reloaded = pickle.loads(pickle.dumps(snap))
    a, b = restore(snap), restore(reloaded)
    for eco in (a, b):
        while eco.t < HORIZON and eco._alive_list:
            eco.step()
    assert a.events_hash() == b.events_hash() == _straight_through()


def test_distill_survivor_mean_valid():
    eco = run_to((_cfg(), 7), HORIZON)
    g = distill([eco], strategy="survivor_mean")
    assert is_valid(g)
    # the distilled intensity lies within the survivors' range (a real blend, not invented)
    survivors = [c.genotype.thermosense_intensity for c in eco._alive_list]
    assert min(survivors) - 1e-9 <= g.thermosense_intensity <= max(survivors) + 1e-9


def test_distill_top_reproducer_is_the_max():
    eco = run_to((_cfg(), 7), HORIZON)
    g = distill([eco], strategy="top_reproducer")
    assert is_valid(g)
    best = max(eco._creatures, key=lambda c: c.phenotype.offspring_count)
    assert g is best.genotype


def test_fork_run_compare_pipeline():
    base = run_to((_cfg(), 7), MID)
    out = fork_run_compare(base, HORIZON, treatment_seed=123, baseline_seed=None)
    assert out["fork_t"] == MID
    assert out["diverged"] is True                       # re-seeded treatment != baseline
    assert set(out["treatment"]) >= {"alive", "mean_intensity", "events_hash"}
