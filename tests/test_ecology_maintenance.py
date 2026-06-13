"""tests/test_ecology_maintenance.py — Fast deterministic tests for the
Exp 197 complexity-linked maintenance cost mechanism.

Mechanism summary:
  With cfg.complexity_cost_scale > 0, each tick an additional energy cost of
  complexity_cost_scale * genotype_complexity(g) is subtracted AFTER the normal
  baseline + aging cost.  Death still flows through energy <= 0 -> "starvation";
  there is NO direct complexity-death rule.

  With complexity_cost_scale = 0.0 (default) the gated path is never entered,
  so the engine is byte-identical to Exp 194 (L16 no-op guard).
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ecology.engine import Ecology, EcologyConfig
from ecology.creature import Phenotype
from ecology.creature import Creature
from ecology.genotype import Genotype, founder, complexity as genotype_complexity
from ecology.scenarios import SCENARIOS, FOUNDER

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXP194_SUMMARY = Path(__file__).parent.parent / (
    "experiments/outputs/exp194_n5_homeostatic_population/balanced_seed0/summary.json"
)
EXP194_HASH = "fc19d23fefede56aa3c751281db9e74da8520f449e4198bb2237910613304ae4"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tiny_cfg(complexity_cost_scale: float = 0.0) -> EcologyConfig:
    """Minimal config for fast unit tests (6x6 grid, horizon 30)."""
    base = SCENARIOS["balanced"]
    return replace(
        base,
        rows=6,
        cols=6,
        horizon=30,
        initial_population=3,
        mutation_rate=0.05,
        max_population=100,
        min_survival_energy=0.5,
        name="balanced_test",
        complexity_cost_scale=complexity_cost_scale,
    )


def _make_creature_with_complexity(
    creature_id: int,
    energy: float,
    age: int,
    pos: int,
    energy_capacity: float,
    sensor_precision: float,
    memory_length: float,
    n_cells: int = 36,
) -> Creature:
    """Build a Creature whose complexity is determined by the three given traits."""
    g = replace(
        founder(),
        energy_capacity=energy_capacity,
        sensor_precision=sensor_precision,
        memory_length=memory_length,
        # Ensure reproduction threshold doesn't exceed capacity
        reproduction_energy_threshold=min(founder().reproduction_energy_threshold, energy_capacity * 0.85),
    )
    ph = Phenotype(energy=energy, age=age, pos=pos)
    return Creature(
        creature_id=creature_id,
        parent_id=None,
        generation=0,
        lineage_root=creature_id,
        genotype=g,
        phenotype=ph,
        n_cells=n_cells,
    )


# ---------------------------------------------------------------------------
# Test 1: Upkeep is monotonic in complexity
# ---------------------------------------------------------------------------
class TestUpkeepMonotonicInComplexity:
    """With scale > 0, a higher-complexity creature loses more energy per tick."""

    def test_upkeep_monotonic_in_complexity(self):
        """Higher-complexity creature drops MORE energy per tick (strict inequality)."""
        scale = 1.0
        eco = Ecology(_tiny_cfg(complexity_cost_scale=scale), seed=42)

        # LOW complexity: minimal energy_capacity, sensor_precision, memory_length
        low_c = _make_creature_with_complexity(
            creature_id=100,
            energy=50.0,          # plenty of energy so no death risk
            age=0,
            pos=0,
            energy_capacity=50.0,
            sensor_precision=0.5,  # minimum
            memory_length=1.0,     # minimum -> complexity ~= 0.0
            n_cells=eco.world.size(),
        )

        # HIGH complexity: large energy_capacity, high sensor_precision, memory_length
        high_c = _make_creature_with_complexity(
            creature_id=101,
            energy=50.0,
            age=0,
            pos=0,
            energy_capacity=50.0,
            sensor_precision=1.0,  # maximum
            memory_length=20.0,    # maximum -> complexity ~= (1 + 1 + 1)/3 = 1.0
            n_cells=eco.world.size(),
        )

        # Drain the cell so eating contributes 0 (isolate metabolic cost)
        eco.world.resource[0] = 0.0
        for nbr in eco.world.neighbors(0):
            eco.world.resource[nbr] = 0.0

        low_energy_before = low_c.phenotype.energy
        high_energy_before = high_c.phenotype.energy

        pending: list = []
        eco._step_one_creature(low_c, pending)
        eco._step_one_creature(high_c, pending)

        low_drop = low_energy_before - low_c.phenotype.energy
        high_drop = high_energy_before - high_c.phenotype.energy

        assert high_drop > low_drop, (
            f"Expected high-complexity creature to lose more energy per tick; "
            f"low_drop={low_drop:.6f}, high_drop={high_drop:.6f}"
        )

        # Quantitative sanity: the gap should be approximately
        # scale * (complexity_high - complexity_low)
        c_low = genotype_complexity(low_c.genotype)
        c_high = genotype_complexity(high_c.genotype)
        expected_gap = scale * (c_high - c_low)
        actual_gap = high_drop - low_drop
        assert abs(actual_gap - expected_gap) < 0.01, (
            f"Complexity gap mismatch: expected ~{expected_gap:.4f}, got {actual_gap:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 2: Zero scale — no complexity dependence
# ---------------------------------------------------------------------------
class TestZeroScaleNoComplexityDependence:
    """With complexity_cost_scale = 0.0, complexity does NOT affect energy cost."""

    def test_zero_scale_no_complexity_dependence(self):
        """Two creatures of different complexity lose the SAME metabolic energy per tick."""
        eco = Ecology(_tiny_cfg(complexity_cost_scale=0.0), seed=42)

        # Build two creatures that are identical EXCEPT for complexity-driving traits,
        # and critically, have the SAME baseline_metabolic_cost and aging_cost and age.
        base_g = founder()

        low_g = replace(
            base_g,
            energy_capacity=50.0,
            sensor_precision=0.5,
            memory_length=1.0,
            reproduction_energy_threshold=min(base_g.reproduction_energy_threshold, 42.5),
        )
        high_g = replace(
            base_g,
            energy_capacity=50.0,
            sensor_precision=1.0,
            memory_length=20.0,
            reproduction_energy_threshold=min(base_g.reproduction_energy_threshold, 42.5),
        )

        # Both at same energy and age so baseline + aging costs are identical
        start_energy = 50.0
        age = 0

        low_c = Creature(
            creature_id=200,
            parent_id=None, generation=0, lineage_root=200,
            genotype=low_g,
            phenotype=Phenotype(energy=start_energy, age=age, pos=0),
            n_cells=eco.world.size(),
        )
        high_c = Creature(
            creature_id=201,
            parent_id=None, generation=0, lineage_root=201,
            genotype=high_g,
            phenotype=Phenotype(energy=start_energy, age=age, pos=0),
            n_cells=eco.world.size(),
        )

        # Drain cell so eating contributes 0
        eco.world.resource[0] = 0.0
        for nbr in eco.world.neighbors(0):
            eco.world.resource[nbr] = 0.0

        pending: list = []
        eco._step_one_creature(low_c, pending)
        eco._step_one_creature(high_c, pending)

        low_drop = start_energy - low_c.phenotype.energy
        high_drop = start_energy - high_c.phenotype.energy

        # The ONLY difference can come from movement (policy uses rng which diverges
        # when genotypes differ in exploration_bias / learning_rate). Both genotypes
        # have the same movement_cost and exploration_bias (inherited from base_g).
        # The complexity cost gap must be exactly 0.0.
        complexity_gap = abs(high_drop - low_drop)
        # Allow for movement difference (at most one movement_cost difference)
        movement_cost = base_g.movement_cost
        assert complexity_gap <= movement_cost + 1e-9, (
            f"With scale=0.0 the complexity-cost gap should be <= movement_cost; "
            f"gap={complexity_gap:.6f}, movement_cost={movement_cost}"
        )

        # Stronger: verify no complexity term was subtracted by checking
        # genotype_complexity differs but energy drops are within movement_cost tolerance
        c_low = genotype_complexity(low_c.genotype)
        c_high = genotype_complexity(high_c.genotype)
        assert c_high > c_low + 0.3, (
            f"Test setup: complexities should differ significantly; "
            f"c_low={c_low:.4f}, c_high={c_high:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Death from maintenance is energy-mediated ("starvation")
# ---------------------------------------------------------------------------
class TestMaintenanceDeathIsEnergyMediated:
    """A creature driven to energy<=0 by maintenance cost dies of 'starvation'."""

    def test_maintenance_death_is_energy_mediated(self):
        """cause_of_death == 'starvation', never 'complexity_death' or 'senescence'."""
        scale = 5.0   # large enough to guarantee death in one step
        eco = Ecology(_tiny_cfg(complexity_cost_scale=scale), seed=77)

        # High-complexity creature with nearly zero energy, world drained
        high_c = _make_creature_with_complexity(
            creature_id=300,
            energy=0.01,           # barely alive — will die after metabolic + maintenance
            age=0,
            pos=0,
            energy_capacity=50.0,
            sensor_precision=1.0,
            memory_length=20.0,    # maximum complexity -> large maintenance cost
            n_cells=eco.world.size(),
        )
        # Drain world so eating cannot rescue it
        eco.world.resource[:] = 0.0

        pending: list = []
        eco._step_one_creature(high_c, pending)

        assert not high_c.phenotype.alive, "High-complexity creature should have died"
        cause = high_c.phenotype.cause_of_death
        assert cause == "starvation", (
            f"Expected cause_of_death='starvation', got '{cause}'. "
            "There must be NO direct complexity_death rule."
        )


# ---------------------------------------------------------------------------
# Test 4: L16 no-op guard — complexity_cost_scale=0.0 reproduces Exp 194
# ---------------------------------------------------------------------------
class TestMaintenanceOffReproducesExp194:
    """CRITICAL regression guard: OFF (scale=0.0) is byte-identical to Exp 194."""

    def test_maintenance_off_reproduces_exp194(self):
        """complexity_cost_scale=0.0 must give fp=170, births=628, deaths=458, hash match."""
        with open(EXP194_SUMMARY) as f:
            exp194 = json.load(f)
        committed_hash = exp194["events_hash"]
        assert committed_hash == EXP194_HASH, (
            f"Committed hash in Exp 194 summary changed! "
            f"Got {committed_hash}, expected {EXP194_HASH}"
        )

        # Run with scale=0.0 (default) — must be byte-identical
        cfg = replace(
            SCENARIOS["balanced"],
            complexity_cost_scale=0.0,  # explicitly OFF
        )
        eco = Ecology(cfg, seed=0)
        summary = eco.run()

        assert summary["final_pop"] == 170, (
            f"L16 regression: final_pop={summary['final_pop']} != 170"
        )
        assert summary["births"] == 628, (
            f"L16 regression: births={summary['births']} != 628"
        )
        assert summary["deaths"] == 458, (
            f"L16 regression: deaths={summary['deaths']} != 458"
        )
        assert summary["events_hash"] == committed_hash, (
            f"L16 regression: events_hash mismatch!\n"
            f"  Got:      {summary['events_hash']}\n"
            f"  Expected: {committed_hash}"
        )


# ---------------------------------------------------------------------------
# Test 5: birth_t is set correctly on founders and children
# ---------------------------------------------------------------------------
class TestBirthTSet:
    """Founders have birth_t==0; a child born at step N has birth_t==N."""

    def test_birth_t_set(self):
        """Founders have birth_t==0; a child born at engine t==N has birth_t==N.

        Timing: self.t is incremented at the END of step().  Children born during
        a step() call in which eco.t == N get birth_t == N.  Founders created in
        __init__ (before any step) also get birth_t == 0 (default), which is
        correct since self.t == 0 at that moment too.

        We verify:
          (a) All founders have birth_t == 0.
          (b) A child born when eco.t == N before increment has birth_t == N.
          (c) 'birth_t' never appears in any event dict (events_hash safety).
        """
        cfg = _tiny_cfg(complexity_cost_scale=0.0)
        eco = Ecology(cfg, seed=5)

        # (a) All initial founders were created at t=0 in __init__
        for c in eco._creatures:
            assert c.phenotype.birth_t == 0, (
                f"Founder creature_id={c.creature_id} should have birth_t=0, "
                f"got {c.phenotype.birth_t}"
            )

        initial_ids = {c.creature_id for c in eco._creatures}

        # Force reproduction: make all founders mature and energy-rich
        for c in eco._creatures:
            c.phenotype.energy = c.genotype.energy_capacity
            c.phenotype.age = c.genotype.maturity_age + 10

        # Step once (t=0 -> t=1) — children born here have birth_t == 0
        eco.step()  # eco.t is 0 during reproduction, becomes 1 after

        # Now step again several times, looking for children born when t > 0
        # We want children whose birth_t > 0 to confirm they record the right timestep.
        # Force energy refresh on each step to keep reproduction happening.
        found_child_with_nonzero_birth_t = False
        for _ in range(15):
            ids_before = {c.creature_id for c in eco._creatures}
            # Refresh energy on alive creatures
            for c in eco._creatures:
                if c.is_alive():
                    c.phenotype.energy = c.genotype.energy_capacity
                    if c.phenotype.age < c.genotype.maturity_age:
                        c.phenotype.age = c.genotype.maturity_age + 1
            t_before_step = eco.t  # this is what birth_t should be set to
            eco.step()
            new_children = [
                c for c in eco._creatures
                if c.creature_id not in ids_before
            ]
            for child in new_children:
                assert child.phenotype.birth_t == t_before_step, (
                    f"Child creature_id={child.creature_id} born at t={t_before_step} "
                    f"has birth_t={child.phenotype.birth_t} (expected {t_before_step})"
                )
                if t_before_step > 0:
                    found_child_with_nonzero_birth_t = True

        assert found_child_with_nonzero_birth_t, (
            "No children were born at t > 0 in 15 steps — test setup issue. "
            "Cannot verify birth_t > 0 case."
        )

        # (c) Verify birth_t does NOT appear in any event dict (events_hash safety)
        for event in eco.events:
            for key in event:
                assert key != "birth_t", (
                    f"'birth_t' found as top-level key in event dict "
                    f"(event_type={event.get('event_type')}); "
                    "this would corrupt the events_hash L16 guard."
                )
            details = event.get("details", {})
            assert "birth_t" not in details, (
                f"'birth_t' found in event details "
                f"(event_type={event.get('event_type')})"
            )
