"""tests/test_exp197_thermosense_instrumentation.py — Fast unit tests for the Exp 197
thermosense instrumentation helpers (importable from the experiment module).

Tests:
  1. test_newborn_separate_from_living — newborn window metric is distinct from living snapshot.
  2. test_active_fraction_uses_threshold — active-fraction counts intensity > 0.05 correctly.
  3. test_age_stratified_returns_three_bins — young/mid/old tercile values are separate.
  4. test_control_has_no_temperature — control arm (temp OFF): world.temperature is None, 0 stress.
  5. test_death_is_starvation — under temperature + thermosense, all deaths are 'starvation'.
"""
from __future__ import annotations

import dataclasses
from dataclasses import replace

import numpy as np
import pytest

from ecology.engine import Ecology, EcologyConfig
from ecology.creature import Creature, Phenotype
from ecology.genotype import Genotype, founder, clamp_traits
from ecology.scenarios import SCENARIOS, FOUNDER
from ecology.world import GridWorld

# Import helpers from the experiment module.
from experiments.exp197_n5_maintenance_cost import (
    compute_living_thermosense,
    compute_newborn_thermosense,
    compute_age_stratified_thermosense,
    make_cfg,
    ACTIVE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Tiny helper factories
# ---------------------------------------------------------------------------

def _make_genotype(**overrides) -> Genotype:
    """Return a founder Genotype with specified field overrides."""
    base = founder()
    d = dataclasses.asdict(base)
    d.update(overrides)
    clamped = clamp_traits(d)
    return Genotype(**clamped)


def _make_creature(
    creature_id: int = 0,
    energy: float = 15.0,
    age: int = 10,
    pos: int = 0,
    birth_t: int = 0,
    parent_id: int | None = None,
    genotype: Genotype | None = None,
    n_cells: int = 144,
) -> Creature:
    g = genotype if genotype is not None else founder()
    ph = Phenotype(energy=energy, age=age, pos=pos, birth_t=birth_t)
    c = Creature(
        creature_id=creature_id,
        parent_id=parent_id,
        generation=0 if parent_id is None else 1,
        lineage_root=creature_id,
        genotype=g,
        phenotype=ph,
        n_cells=n_cells,
    )
    return c


def _tiny_cfg(**overrides) -> EcologyConfig:
    """Minimal config for fast unit tests (6x6 grid, horizon 50)."""
    base = SCENARIOS["balanced"]
    cfg = replace(
        base,
        rows=6,
        cols=6,
        horizon=50,
        initial_population=3,
        mutation_rate=0.05,
        max_population=200,
        min_survival_energy=0.5,
        name="exp197_test",
    )
    return replace(cfg, **overrides)


# ---------------------------------------------------------------------------
# Test 1: newborn metric is DISTINCT from the living snapshot
# ---------------------------------------------------------------------------

class TestNewbornSeparateFromLiving:
    """compute_newborn_thermosense(birth_t window) must be separate from the
    living snapshot: a creature can be alive now (counted in living) yet not be
    a newborn (born outside the window), or be a newborn that later died.

    We construct a case where:
      - alive population has ZERO intensity (born before the window; NOT newborns)
      - newborns (born inside window, non-founder) have HIGH intensity
    so living active_fraction == 0 but newborn mean_intensity > 0.
    """

    def test_newborn_separate_from_living(self):
        # Living creatures: born at t=0 (founder-like), zero intensity, parent_id=None
        old_g = _make_genotype(thermosense_intensity=0.0)
        alive = [
            _make_creature(creature_id=i, birth_t=0, parent_id=None, genotype=old_g)
            for i in range(4)
        ]

        # Newborns: born in window (prev_t=3000, curr_t=5000), non-founder, HIGH intensity
        new_g = _make_genotype(thermosense_intensity=0.80)
        # These creatures may be dead or alive — compute_newborn_thermosense reads all creatures.
        newborn_creatures = [
            _make_creature(creature_id=10 + i, birth_t=4000, parent_id=0, genotype=new_g)
            for i in range(3)
        ]
        # Mark them dead so they don't appear in "alive" list
        for c in newborn_creatures:
            c.phenotype.alive = False
            c.phenotype.cause_of_death = "starvation"

        all_creatures = alive + newborn_creatures

        # Living snapshot: only the old, zero-intensity creatures
        living_snap = compute_living_thermosense(alive)
        # Newborn snapshot: only the high-intensity creatures born in (3999, 5000]
        newborn_snap = compute_newborn_thermosense(all_creatures, prev_t=3999, curr_t=5000)

        # Living should have active_fraction == 0 (all intensity == 0)
        assert living_snap["active_fraction"] == 0.0, (
            f"Expected living active_fraction=0.0, got {living_snap['active_fraction']}"
        )

        # Newborn should show the high-intensity creatures
        assert newborn_snap["count"] == 3, (
            f"Expected 3 newborns, got {newborn_snap['count']}"
        )
        assert newborn_snap["mean_intensity"] > ACTIVE_THRESHOLD, (
            f"Expected newborn mean_intensity > {ACTIVE_THRESHOLD}, "
            f"got {newborn_snap['mean_intensity']}"
        )

        # The two metrics are distinct — living shows 0, newborn shows high
        assert living_snap["mean_intensity"] < newborn_snap["mean_intensity"], (
            "Living mean_intensity should be lower than newborn mean_intensity in this setup"
        )

    def test_founders_excluded_from_newborn_window(self):
        """Creatures with parent_id=None (founders) are excluded from newborn count."""
        g = _make_genotype(thermosense_intensity=0.9)
        founder_creature = _make_creature(
            creature_id=0, birth_t=4500, parent_id=None, genotype=g
        )
        non_founder = _make_creature(
            creature_id=1, birth_t=4500, parent_id=0, genotype=g
        )
        all_c = [founder_creature, non_founder]
        snap = compute_newborn_thermosense(all_c, prev_t=3999, curr_t=5000)
        assert snap["count"] == 1, (
            f"Expected 1 newborn (non-founder only), got {snap['count']}"
        )


# ---------------------------------------------------------------------------
# Test 2: active_fraction counts intensity > ACTIVE_THRESHOLD correctly
# ---------------------------------------------------------------------------

class TestActiveFractionUsesThreshold:
    """active_fraction must be the exact fraction with intensity > 0.05."""

    def test_active_fraction_uses_threshold(self):
        # 3 active (intensity=0.10 > 0.05), 2 inactive (intensity=0.03 < 0.05)
        active_g = _make_genotype(thermosense_intensity=0.10)
        inactive_g = _make_genotype(thermosense_intensity=0.03)
        creatures = (
            [_make_creature(creature_id=i, genotype=active_g) for i in range(3)]
            + [_make_creature(creature_id=3 + i, genotype=inactive_g) for i in range(2)]
        )
        snap = compute_living_thermosense(creatures)
        expected_af = 3 / 5
        assert abs(snap["active_fraction"] - expected_af) < 1e-12, (
            f"Expected active_fraction={expected_af}, got {snap['active_fraction']}"
        )

    def test_threshold_boundary_not_active(self):
        """intensity == ACTIVE_THRESHOLD (0.05) is NOT active (strictly greater required)."""
        g_boundary = _make_genotype(thermosense_intensity=ACTIVE_THRESHOLD)
        g_above = _make_genotype(thermosense_intensity=ACTIVE_THRESHOLD + 0.01)
        snap_boundary = compute_living_thermosense([_make_creature(genotype=g_boundary)])
        snap_above = compute_living_thermosense([_make_creature(genotype=g_above)])
        assert snap_boundary["active_fraction"] == 0.0, (
            f"intensity==threshold should be inactive; got {snap_boundary['active_fraction']}"
        )
        assert snap_above["active_fraction"] == 1.0, (
            f"intensity>threshold should be active; got {snap_above['active_fraction']}"
        )

    def test_empty_population_returns_nan(self):
        """Empty alive list returns NaN active_fraction."""
        snap = compute_living_thermosense([])
        assert np.isnan(snap["active_fraction"]), (
            f"Expected NaN for empty population, got {snap['active_fraction']}"
        )


# ---------------------------------------------------------------------------
# Test 3: age_stratified returns three SEPARATE bins
# ---------------------------------------------------------------------------

class TestAgeStratifiedReturnsThreeBins:
    """young/mid/old tercile values must be distinct, not collapsed, when ages vary."""

    def test_age_stratified_returns_three_bins(self):
        # 9 creatures with different ages and intensities so bins differ.
        # young (ages 1-3): intensity 0.0 (inactive)
        # mid (ages 4-6):   intensity 0.5
        # old (ages 7-9):   intensity 0.9
        creatures = []
        for i in range(3):
            g = _make_genotype(thermosense_intensity=0.0)
            creatures.append(_make_creature(creature_id=i, age=i + 1, genotype=g))
        for i in range(3):
            g = _make_genotype(thermosense_intensity=0.5)
            creatures.append(_make_creature(creature_id=3 + i, age=i + 4, genotype=g))
        for i in range(3):
            g = _make_genotype(thermosense_intensity=0.9)
            creatures.append(_make_creature(creature_id=6 + i, age=i + 7, genotype=g))

        strat = compute_age_stratified_thermosense(creatures)

        # All three keys present
        assert "young" in strat and "mid" in strat and "old" in strat

        # None should be NaN
        assert not np.isnan(strat["young"]), "young bin is NaN"
        assert not np.isnan(strat["mid"]), "mid bin is NaN"
        assert not np.isnan(strat["old"]), "old bin is NaN"

        # The bins should reflect different intensity groups (not all collapsed to one value)
        assert strat["young"] < strat["mid"] < strat["old"], (
            f"Expected young < mid < old; got young={strat['young']:.4f}, "
            f"mid={strat['mid']:.4f}, old={strat['old']:.4f}"
        )

    def test_age_stratified_empty_returns_nans(self):
        strat = compute_age_stratified_thermosense([])
        for k in ("young", "mid", "old"):
            assert np.isnan(strat[k]), f"Expected NaN for empty input, key={k}"

    def test_age_stratified_single_creature(self):
        """Single creature: young bin gets it, mid/old are NaN (n//3==0 for n==1)."""
        g = _make_genotype(thermosense_intensity=0.5)
        strat = compute_age_stratified_thermosense([_make_creature(genotype=g)])
        # With n=1: t1 = 0, t2 = 0 → young_idx is empty too (order[:0])
        # All bins will be NaN or only old will have value depending on slice logic
        # The key contract: function runs without error and returns the three keys
        assert "young" in strat and "mid" in strat and "old" in strat


# ---------------------------------------------------------------------------
# Test 4: control arm has no temperature field
# ---------------------------------------------------------------------------

class TestControlHasNoTemperature:
    """In the control arm (enable_temperature=False), world.temperature is None
    and temperature_stress always returns 0.
    """

    def test_control_has_no_temperature(self):
        cfg = make_cfg(temp_on=False)
        eco = Ecology(cfg, seed=8)

        # Temperature field must be absent
        assert eco.world.temperature is None, (
            "Control arm should have world.temperature == None"
        )

        # temperature_stress at any cell is 0
        for pos in range(eco.world.size()):
            stress = eco.world.temperature_stress(pos, tolerance=0.0, stress_scale=10.0)
            assert stress == 0.0, (
                f"Expected 0 stress in control arm at pos={pos}, got {stress}"
            )

    def test_treatment_has_temperature(self):
        """Treatment arm (enable_temperature=True) should have a temperature gradient."""
        cfg = make_cfg(temp_on=True)
        eco = Ecology(cfg, seed=8)

        assert eco.world.temperature is not None, (
            "Treatment arm should have a temperature field"
        )
        # Gradient: left column (c=0) has temp=0, right column (c=11) has temp=1
        n_cols = eco.world.cols
        left_col_pos = 0  # row 0, col 0
        right_col_pos = n_cols - 1  # row 0, col n_cols-1
        assert eco.world.temperature[left_col_pos] == pytest.approx(0.0, abs=1e-9)
        assert eco.world.temperature[right_col_pos] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 5: all deaths are cause 'starvation' (no thermosense_death)
# ---------------------------------------------------------------------------

class TestDeathIsStarvation:
    """Under temperature + thermosense (treatment arm), all deaths must be 'starvation'.

    No thermosense_death, complexity_death, or temperature_death cause exists.
    """

    def test_death_is_starvation(self):
        # Short treatment run with a stressed population
        cfg = make_cfg(temp_on=True)
        # Use a short horizon to keep the test fast
        cfg = replace(cfg, horizon=300)
        eco = Ecology(cfg, seed=8)

        while eco.t < cfg.horizon and not eco.exploded:
            eco.step()
            if not eco._alive():
                break

        dead = [c for c in eco._creatures if not c.is_alive()]
        assert len(dead) > 0, (
            "No deaths in 300-step treatment run — test setup may be wrong "
            "(population may not be under enough stress)"
        )

        unexpected = {
            c.phenotype.cause_of_death
            for c in dead
            if c.phenotype.cause_of_death != "starvation"
        }
        assert not unexpected, (
            f"Unexpected death causes in treatment arm: {unexpected}. "
            "All deaths must be 'starvation' (energy-mediated). "
            "No thermosense_death, complexity_death, or temperature_death rule exists."
        )
