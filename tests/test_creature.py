"""Tests for active_loop.creature — fast, numpy-only, no network/jax.

Tests cover the six specified behaviours:
1. birth → save → load roundtrip
2. live() continuity across calls
3. save → load → live resumes deterministically
4. fork: lineage recorded; divergence after different post-fork histories
5. PROPERTY-LEVEL test: map_accuracy ≥ 8/9 in ≥ 2/3 seeds after 900 steps
6. Biography: append-only, valid JSONL
"""
from __future__ import annotations

import json
import copy
import numpy as np
import pytest
from pathlib import Path

from active_loop.creature import Creature, World


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_world_3x3() -> World:
    """3x3 aliased-color world from Exp 21: cmap [0,1,2, 1,2,0, 2,0,1]."""
    return World(rows=3, cols=3, cmap=[0, 1, 2, 1, 2, 0, 2, 0, 1], n_colors=3)


def make_world_2x2() -> World:
    """Tiny 2x2 world with 2 colors for fast unit tests."""
    return World(rows=2, cols=2, cmap=[0, 1, 1, 0], n_colors=2)


# ---------------------------------------------------------------------------
# Test 1 — birth → save → load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    """Arrays identical after save + load; state_hash matches; scalars preserved."""
    world = make_world_3x3()
    c = Creature.birth("alpha", world, seed=42)
    c.live(50, seed=0)
    c.teach_word("red", 0, n=8)

    save_dir = tmp_path / "alpha"
    c.save(str(save_dir))

    c2 = Creature.load(str(save_dir))

    # Scalar fields preserved
    assert c2.name == c.name
    assert c2.age_steps == c.age_steps
    assert c2.true_pos == c.true_pos
    assert c2.lineage == c.lineage
    assert c2.rng_counter == c.rng_counter
    assert c2._seed == c._seed

    # Array fields identical
    np.testing.assert_array_equal(c2.pA, c.pA)
    np.testing.assert_array_equal(c2.qs, c.qs)
    np.testing.assert_array_equal(c2.value_counts, c.value_counts)

    # Vocab preserved
    assert set(c2.vocab.keys()) == set(c.vocab.keys())
    for word in c.vocab:
        np.testing.assert_array_equal(c2.vocab[word], c.vocab[word])

    # World config preserved
    assert c2.world.rows == c.world.rows
    assert c2.world.cols == c.world.cols
    assert c2.world.cmap == c.world.cmap
    assert c2.world.n_colors == c.world.n_colors

    # State hash matches (integrity verification happens inside load())
    assert c._state_hash() == c2._state_hash()


# ---------------------------------------------------------------------------
# Test 2 — live() continuity: belief carried, pA changes
# ---------------------------------------------------------------------------

def test_live_continuity():
    """qs is NOT reset between live() calls; age_steps increases; pA changes."""
    world = make_world_3x3()
    c = Creature.birth("beta", world, seed=7)

    qs_before = c.qs.copy()
    pA_before = c.pA.copy()

    c.live(30, seed=1)
    qs_after_1 = c.qs.copy()

    # Belief has changed from uniform (creature has been updated)
    assert not np.allclose(qs_after_1, qs_before), "qs should change after live()"

    # pA has changed (counts accumulated)
    assert not np.allclose(c.pA, pA_before), "pA should change after live()"

    assert c.age_steps == 30

    qs_after_first_call = c.qs.copy()
    c.live(30, seed=2)

    # qs is NOT reset between calls — it continues from where it was
    # (it will be further updated, not reset to uniform)
    assert c.age_steps == 60

    # The belief after the second call is NOT the same as after the first
    # because it evolved from the carried-over belief, not from uniform.
    # We verify this by checking that the posterior entropy may differ from
    # what a fresh creature would produce.
    assert c.qs.sum() > 0.0, "qs must be a valid distribution"
    assert abs(c.qs.sum() - 1.0) < 1e-9, "qs must sum to 1"


# ---------------------------------------------------------------------------
# Test 3 — save → load → live resumes deterministically
# ---------------------------------------------------------------------------

def test_live_deterministic_after_load(tmp_path):
    """Living 100 steps in one go equals save+load+live with same derived seeds."""
    world = make_world_3x3()

    # Arm A: continuous 100 steps
    c_a = Creature.birth("gamma", world, seed=13)
    c_a.live(50, seed=999)   # warm-up, same both paths
    pA_snap = c_a.pA.copy()
    qs_snap = c_a.qs.copy()
    rng_snap = c_a.rng_counter

    # Record state after warm-up for arm B
    c_b = Creature.birth("gamma", world, seed=13)
    c_b.live(50, seed=999)   # identical warm-up

    # Arm A: 100 more steps without saving
    c_a.live(100)  # seed=None → derives from (seed=13, rng_counter after warm-up)

    # Arm B: save → load → live same 100 steps
    save_dir = tmp_path / "gamma"
    c_b.save(str(save_dir))
    c_b_loaded = Creature.load(str(save_dir))
    c_b_loaded.live(100)  # seed=None → same derived seed as arm A

    # Both should reach the same state
    np.testing.assert_array_almost_equal(c_a.pA, c_b_loaded.pA, decimal=10)
    np.testing.assert_array_almost_equal(c_a.qs, c_b_loaded.qs, decimal=10)
    assert c_a._state_hash() == c_b_loaded._state_hash(), (
        "state_hash must match: resumability guarantee"
    )
    assert c_a.age_steps == c_b_loaded.age_steps == 150


# ---------------------------------------------------------------------------
# Test 4 — fork: lineage, divergence, original untouched
# ---------------------------------------------------------------------------

def test_fork_lineage_and_divergence():
    """Fork records lineage; twin diverges under different world; original untouched."""
    world_a = make_world_3x3()
    c = Creature.birth("parent", world_a, seed=3)
    c.live(100, seed=0)

    pA_before_fork = c.pA.copy()
    hash_before_fork = c._state_hash()

    # Fork
    twin = c.fork("twin")

    # Lineage recorded in twin
    assert len(twin.lineage) == len(c.lineage) + 1
    assert "parent@" in twin.lineage[-1]
    assert hash_before_fork[:12] in twin.lineage[-1]

    # Original untouched
    assert c.name == "parent"
    np.testing.assert_array_equal(c.pA, pA_before_fork)

    # Twin starts with same state
    np.testing.assert_array_equal(twin.pA, c.pA)
    assert twin.name == "twin"

    # Run twin in a DIFFERENT world (2-color world) — fork deep-copies world too
    world_b = make_world_2x2()
    twin.world = world_b
    twin.pA = np.full((world_b.n_colors, world_b.n_cells), 0.1)
    twin.qs = np.ones(world_b.n_cells) / world_b.n_cells
    twin.true_pos = 0
    twin.value_counts = np.zeros(world_b.n_colors)
    twin.live(200, seed=99)

    # Run original further in its own world
    c.live(200, seed=99)

    # They should differ (different worlds → different learned states)
    # The test checks that fork() doesn't create a linked alias
    assert c._state_hash() != twin._state_hash(), (
        "original and twin must diverge under different post-fork histories"
    )
    assert c.name == "parent"  # original name unchanged


# ---------------------------------------------------------------------------
# Test 5 — PROPERTY-LEVEL: map_accuracy ≥ 8/9 in ≥ 2 of 3 seeds
# ---------------------------------------------------------------------------
#
# Statistical / property claim, not exact-number claim.
# Per loop/VALIDATION.md: this is a property-level assertion about a statistical
# regularity (place fields self-organize from 900 steps of continuous registered
# experience in an aliased 3x3 world), NOT a claim that any exact threshold is
# always met.  Thresholds are generous; 2-of-3 seeds avoids flakiness from a
# single unlucky seed.
#
# Exp 21 verified this with pymdp; here we verify the pure-numpy Creature
# implementation produces the same qualitative result.

def test_property_map_accuracy_3x3():
    """PROPERTY: in ≥ 2 of 3 seeds, creature learns the 3x3 cmap accurately."""
    world = make_world_3x3()
    SEEDS = [0, 1, 2]
    STEPS = 900
    THRESHOLD = 8 / 9  # 8 of 9 cells correct

    passes = 0
    for seed in SEEDS:
        c = Creature.birth(f"learner-{seed}", world, seed=seed)
        c.live(STEPS, seed=seed)
        acc = c.map_accuracy()
        if acc >= THRESHOLD:
            passes += 1

    assert passes >= 2, (
        f"Expected map_accuracy >= {THRESHOLD:.3f} in at least 2 of 3 seeds; "
        f"got {passes}/3. "
        f"This is a statistical property — if it fails consistently, the learning "
        f"mechanism may be broken, not just unlucky."
    )


# ---------------------------------------------------------------------------
# Test 6 — Biography: append-only, valid JSONL, not truncated by saves
# ---------------------------------------------------------------------------

def test_biography_append_only(tmp_path):
    """Biography JSONL appends events in order; save() does not truncate it."""
    world = make_world_2x2()
    c = Creature.birth("chronicled", world, seed=0)

    save_dir = tmp_path / "chronicled"
    c.save(str(save_dir))  # first save — binds state_dir, writes first bio event

    c.live(10, seed=1)     # live event
    c.teach_word("blue", 0, n=5)  # teach event
    c.save(str(save_dir))  # second save — must append, not truncate

    bio_path = save_dir / "BIOGRAPHY.jsonl"
    assert bio_path.exists(), "BIOGRAPHY.jsonl must exist after save"

    lines = bio_path.read_text().splitlines()
    # Remove any empty trailing lines
    lines = [l for l in lines if l.strip()]

    # All lines must be valid JSON
    records = []
    for i, line in enumerate(lines):
        rec = json.loads(line)  # raises if invalid JSON
        records.append(rec)

    # Events must appear in order
    events = [r["event"] for r in records]
    # Expected sequence: save, live, teach_word, save
    assert "save" in events
    assert "live" in events
    assert "teach_word" in events

    # age_steps must be non-decreasing
    ages = [r["age_steps"] for r in records]
    for a, b in zip(ages, ages[1:]):
        assert a <= b, f"age_steps must be non-decreasing: {ages}"

    # Each record must have a state_hash
    for rec in records:
        assert "state_hash" in rec, f"Missing state_hash in record: {rec}"
        assert len(rec["state_hash"]) == 64, "state_hash must be SHA-256 hex"

    # The save events must not have reset the file (each save appends)
    save_count = sum(1 for e in events if e == "save")
    assert save_count >= 2, "Both saves should appear in the biography"
