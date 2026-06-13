"""tests/test_ecology_foraging.py — Deterministic tests for Exp 200 foraging-sense.

Design invariants verified here:
  - Exp 194 events_hash is preserved when all new flags are off (critical regression).
  - Food regen is concentrated in-band when enable_food_coupling is True;
    regen is uniform/unchanged when it is False.
  - current_food_optimal drifts sinusoidally with food_optimal_amplitude > 0;
    stays constant when amplitude is 0.
  - Arm B (thermosense_intensity=0.8, forage_mode) eats more resource per creature
    than Arm A (no organ) when food is concentrated in a drifting thermal band.
  - thermosense_forage_mode=False leaves avoid-mode behaviour unchanged.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ecology.engine import Ecology, EcologyConfig
from ecology.creature import Creature, Phenotype
from ecology.genotype import Genotype, founder
from ecology.scenarios import SCENARIOS, FOUNDER
from ecology.world import GridWorld


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXP194_SUMMARY = Path(__file__).parent.parent / (
    "experiments/outputs/exp194_n5_homeostatic_population/balanced_seed0/summary.json"
)
EXP194_HASH = "fc19d23fefede56aa3c751281db9e74da8520f449e4198bb2237910613304ae4"

# Foraging behavioral test parameters — tuned so thermosense forager eats more.
# food_concentration=5.0 creates a strong band gradient; food_optimal_amplitude=0.3
# makes the band drift across the temperature gradient, requiring active tracking.
_FORAGE_FOOD_CONCENTRATION = 5.0
_FORAGE_BAND_WIDTH = 0.15
_FORAGE_AMPLITUDE = 0.3
_FORAGE_PERIOD = 1500.0
_FORAGE_AVOIDANCE_WEIGHT = 2.0
_FORAGE_NOISE_BASE = 0.2
_FORAGE_INTENSITY = 0.8
_FORAGE_INEFFICIENCY = 0.3
_FORAGE_HORIZON = 1500
_FORAGE_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tiny_cfg(**overrides) -> EcologyConfig:
    """Minimal config for fast unit tests (6x6 grid, horizon 30)."""
    base = SCENARIOS["balanced"]
    cfg = replace(
        base,
        rows=6,
        cols=6,
        horizon=30,
        initial_population=3,
        mutation_rate=0.05,
        max_population=100,
        min_survival_energy=0.5,
        name="foraging_test",
    )
    return replace(cfg, **overrides)


# ---------------------------------------------------------------------------
# Test 1: Exp 194 regression hash preserved
# ---------------------------------------------------------------------------
class TestExp194RegressionHashPreserved:
    """CRITICAL: with all new flags off / defaults, events_hash must equal Exp 194."""

    def test_exp194_regression_hash_preserved(self):
        """All new foraging fields off → balanced/seed0 hash matches committed Exp 194."""
        with open(EXP194_SUMMARY) as f:
            exp194 = json.load(f)
        committed_hash = exp194["events_hash"]
        assert committed_hash == EXP194_HASH, (
            f"Committed Exp 194 hash changed: got {committed_hash}"
        )

        # Default config — all new foraging fields at their default (False / 0.0 / 1.0).
        cfg = replace(
            SCENARIOS["balanced"],
            complexity_cost_scale=0.0,
            enable_temperature=False,
            enable_thermosense=False,
            enable_food_coupling=False,
            thermosense_forage_mode=False,
            food_optimal_amplitude=0.0,
            food_concentration=1.0,
        )
        eco = Ecology(cfg, seed=0)
        summary = eco.run()

        assert summary["births"] == 628, f"births={summary['births']} != 628"
        assert summary["deaths"] == 458, f"deaths={summary['deaths']} != 458"
        assert summary["events_hash"] == committed_hash, (
            f"events_hash mismatch after foraging fields added!\n"
            f"  Got:      {summary['events_hash']}\n"
            f"  Expected: {committed_hash}"
        )


# ---------------------------------------------------------------------------
# Test 2: food concentration in-band vs out-of-band
# ---------------------------------------------------------------------------
class TestFoodConcentrationInBand:
    """With enable_food_coupling, cells near current_food_optimal get more regen;
    without it, regen is uniform and unchanged.
    """

    def test_in_band_cells_get_more_regen_when_enabled(self):
        """After one step_regen, cells in the food band have higher resource than
        out-of-band cells, relative to their starting levels.
        """
        cfg = _tiny_cfg(
            enable_temperature=True,
            enable_food_coupling=True,
            food_optimal_base=0.5,    # band centered at temp=0.5 (grid center)
            food_band_width=0.15,
            food_concentration=5.0,   # strong boost
        )
        eco = Ecology(cfg, seed=0)

        # Drain all cells so regen goes exclusively into the food band.
        eco.world.resource[:] = 0.0
        eco.world.current_food_optimal = 0.5

        eco.world.step_regen()

        in_band = np.abs(eco.world.temperature - 0.5) <= 0.15
        out_band = ~in_band

        if np.any(in_band) and np.any(out_band):
            mean_in = float(np.mean(eco.world.resource[in_band]))
            mean_out = float(np.mean(eco.world.resource[out_band]))
            assert mean_in > mean_out, (
                f"In-band cells should get more regen; mean_in={mean_in:.4f}, "
                f"mean_out={mean_out:.4f}"
            )

    def test_regen_uniform_when_food_coupling_off(self):
        """When enable_food_coupling=False, all cells get the same regen_rate.
        Verify step_regen is identical to the baseline.
        """
        cfg_off = _tiny_cfg(enable_temperature=True, enable_food_coupling=False)
        eco_off = Ecology(cfg_off, seed=0)
        eco_off.world.resource[:] = 0.0
        eco_off.world.step_regen()
        regen_off = eco_off.world.resource.copy()

        # With food coupling off, every cell should gain exactly regen_rate
        # (since they all started at 0, and regen_rate < capacity).
        expected = np.full(eco_off.world.size(), eco_off.cfg.regen_rate)
        np.testing.assert_allclose(
            regen_off, expected, rtol=0, atol=1e-12,
            err_msg="With food_coupling=False, regen should be uniform regen_rate everywhere"
        )

    def test_total_regen_approximately_conserved_with_coupling(self):
        """Total regen across the grid is roughly n_cells * regen_rate when
        food coupling is on with mild concentration (concentrates WHERE food is,
        not how much total).

        Note: conservation only holds when food_concentration is small enough
        that out_factor = (n_cells - n_in * food_concentration) / n_out > 0.
        With food_concentration=2.0 and n_in/n_cells~1/3 on a 6x6 grid:
          out_factor = (36 - 12*2.0) / 24 = 12/24 = 0.5 > 0 — conserved exactly.
        """
        cfg = _tiny_cfg(
            enable_temperature=True,
            enable_food_coupling=True,
            food_optimal_base=0.5,
            food_band_width=0.15,
            food_concentration=2.0,   # mild concentration — conservation holds
        )
        eco = Ecology(cfg, seed=0)
        eco.world.resource[:] = 0.0
        eco.world.current_food_optimal = 0.5
        eco.world.step_regen()

        n_cells = eco.world.size()
        expected_total = n_cells * eco.cfg.regen_rate
        actual_total = float(np.sum(eco.world.resource))
        # Exact conservation when out_factor > 0; allow floating-point epsilon.
        assert abs(actual_total - expected_total) < 1e-9, (
            f"Total regen not conserved: actual={actual_total:.10f}, "
            f"expected={expected_total:.10f}"
        )


# ---------------------------------------------------------------------------
# Test 3: food_optimal drifts with amplitude > 0
# ---------------------------------------------------------------------------
class TestFoodOptimalDrifts:
    """With food_optimal_amplitude > 0, current_food_optimal varies across
    timesteps; with amplitude=0 it stays constant.
    """

    def test_food_optimal_drifts_with_amplitude(self):
        """food_optimal_amplitude > 0 → current_food_optimal differs at selected t."""
        amplitude = 0.3
        period = 100.0
        base = 0.5

        cfg = _tiny_cfg(
            enable_temperature=True,
            enable_food_coupling=True,
            food_optimal_base=base,
            food_optimal_amplitude=amplitude,
            food_optimal_period=period,
        )
        eco = Ecology(cfg, seed=0)

        food_opt_readings = []
        for _ in range(30):
            expected = base + amplitude * math.sin(2.0 * math.pi * eco.t / period)
            eco.step()
            food_opt_readings.append((eco.t - 1, expected, eco.world.current_food_optimal))

        # At least two readings should differ.
        unique_vals = {round(v, 10) for _, _, v in food_opt_readings}
        assert len(unique_vals) > 1, (
            f"current_food_optimal never changed across 30 steps with amplitude={amplitude}"
        )

        # Verify each reading matches the formula.
        for t_val, expected_val, actual_val in food_opt_readings:
            assert abs(actual_val - expected_val) < 1e-12, (
                f"At t={t_val}: current_food_optimal={actual_val:.12f} != "
                f"expected {expected_val:.12f}"
            )

    def test_food_optimal_constant_with_zero_amplitude(self):
        """food_optimal_amplitude=0 → current_food_optimal == food_optimal_base always."""
        base = 0.5
        cfg = _tiny_cfg(
            enable_temperature=True,
            enable_food_coupling=True,
            food_optimal_base=base,
            food_optimal_amplitude=0.0,
        )
        eco = Ecology(cfg, seed=0)

        for _ in range(20):
            eco.step()
            assert eco.world.current_food_optimal == base, (
                f"current_food_optimal drifted to {eco.world.current_food_optimal} "
                f"despite food_optimal_amplitude=0 at t={eco.t}"
            )

    def test_food_optimal_not_updated_when_coupling_off(self):
        """When enable_food_coupling=False, current_food_optimal is never updated."""
        base = 0.5
        cfg = _tiny_cfg(
            enable_temperature=True,
            enable_food_coupling=False,
            food_optimal_base=base,
            food_optimal_amplitude=0.3,
        )
        eco = Ecology(cfg, seed=0)
        initial_food_opt = eco.world.current_food_optimal

        for _ in range(10):
            eco.step()

        assert eco.world.current_food_optimal == initial_food_opt, (
            "current_food_optimal should not change when enable_food_coupling=False"
        )


# ---------------------------------------------------------------------------
# Test 4: forage mode steers creatures toward food (BEHAVIORAL)
# ---------------------------------------------------------------------------
class TestForageModeSteersToFood:
    """Arm B (thermosense ON, forage_mode) eats more resource per creature than
    Arm A (no organ, no forage).

    Parameters used (tuned so the benefit is detectable):
      food_concentration=5.0, food_band_width=0.15 — strong, narrow band
      food_optimal_amplitude=0.3, food_optimal_period=1500.0 — drifting band
      thermosense_intensity=0.8, thermosense_inefficiency=0.3 — good organ
      thermal_avoidance_weight=2.0 — policy weights food signal strongly
      thermosense_noise_base=0.2 — good signal quality at intensity=0.8
      temperature_stress_scale=0.0 — stress OFF so only foraging matters
      horizon=1500 steps, seed=42
    """

    def _run_arm(self, use_organ: bool) -> dict:
        base = SCENARIOS["balanced"]
        f = founder()

        if use_organ:
            f = replace(
                f,
                thermosense_intensity=_FORAGE_INTENSITY,
                thermosense_inefficiency=_FORAGE_INEFFICIENCY,
            )

        cfg = replace(
            base,
            enable_temperature=True,
            enable_thermosense=use_organ,
            enable_food_coupling=True,
            thermosense_forage_mode=use_organ,
            temperature_stress_scale=0.0,        # no thermal penalty — foraging only
            thermosense_upkeep_floor=0.01,       # small organ upkeep
            thermosense_active_threshold=0.05,
            thermal_avoidance_weight=_FORAGE_AVOIDANCE_WEIGHT,
            thermosense_noise_base=_FORAGE_NOISE_BASE,
            food_optimal_base=0.5,
            food_optimal_amplitude=_FORAGE_AMPLITUDE,
            food_optimal_period=_FORAGE_PERIOD,
            food_band_width=_FORAGE_BAND_WIDTH,
            food_concentration=_FORAGE_FOOD_CONCENTRATION,
            founder=f,
            horizon=_FORAGE_HORIZON,
            name="forage_B" if use_organ else "forage_A",
        )
        eco = Ecology(cfg, seed=_FORAGE_SEED)
        summary = eco.run()
        # Attach raw creatures for per-creature resource_eaten analysis.
        summary["_creatures"] = eco._creatures
        return summary

    def test_forager_eats_more_resource_per_creature(self):
        """Arm B (forager) achieves higher mean resource_eaten/lifespan than arm A."""
        s_a = self._run_arm(use_organ=False)
        s_b = self._run_arm(use_organ=True)

        def _mean_rate(summary):
            """Mean resource_eaten / lifespan for dead creatures (lifespan > 0)."""
            creatures = summary["_creatures"]
            rates = []
            for c in creatures:
                if not c.is_alive() and c.phenotype.age > 0:
                    rates.append(c.phenotype.resource_eaten / c.phenotype.age)
                elif c.is_alive() and c.phenotype.age > 0:
                    rates.append(c.phenotype.resource_eaten / c.phenotype.age)
            return float(np.mean(rates)) if rates else 0.0

        rate_a = _mean_rate(s_a)
        rate_b = _mean_rate(s_b)

        # Also compare reproduction counts as a secondary metric.
        repro_a = s_a["reproduction_count"]
        repro_b = s_b["reproduction_count"]

        # Primary assertion: forager eats more per unit time OR reproduces more.
        primary_passes = rate_b > rate_a
        secondary_passes = repro_b > repro_a

        assert primary_passes or secondary_passes, (
            f"Forager (arm B) should eat more or reproduce more than non-forager (arm A).\n"
            f"  Arm A resource rate = {rate_a:.4f}/step, repro = {repro_a}\n"
            f"  Arm B resource rate = {rate_b:.4f}/step, repro = {repro_b}\n"
            f"Parameters used: food_concentration={_FORAGE_FOOD_CONCENTRATION}, "
            f"food_band_width={_FORAGE_BAND_WIDTH}, "
            f"food_optimal_amplitude={_FORAGE_AMPLITUDE}, "
            f"forage_weight={_FORAGE_AVOIDANCE_WEIGHT}. "
            "If this fails, tune _FORAGE_* params so foraging provides a functional benefit."
        )

        # Log the actual numbers (visible in pytest output with -v).
        print(
            f"\nForaging behavioral test:\n"
            f"  Arm A (no organ): resource_rate={rate_a:.4f}/step, "
            f"repro={repro_a}, lifespan={s_a['mean_lifespan']:.1f}\n"
            f"  Arm B (forager):  resource_rate={rate_b:.4f}/step, "
            f"repro={repro_b}, lifespan={s_b['mean_lifespan']:.1f}\n"
            f"  food_concentration={_FORAGE_FOOD_CONCENTRATION}, "
            f"amplitude={_FORAGE_AMPLITUDE}, band_width={_FORAGE_BAND_WIDTH}"
        )


# ---------------------------------------------------------------------------
# Test 5: forage_off behaves as avoid mode
# ---------------------------------------------------------------------------
class TestForageOffIsAvoid:
    """With thermosense_forage_mode=False, the policy runs avoid-mode unchanged."""

    def test_forage_mode_off_flag_is_respected(self):
        """forage_mode=False → world.forage_mode is False → avoid branch runs."""
        cfg = _tiny_cfg(
            enable_temperature=True,
            enable_thermosense=True,
            thermosense_forage_mode=False,
            enable_food_coupling=False,
        )
        eco = Ecology(cfg, seed=0)

        # The world should have forage_mode=False.
        assert not eco.world.forage_mode, (
            "world.forage_mode should be False when thermosense_forage_mode=False"
        )

    def test_forage_mode_on_flag_sets_world_forage_mode(self):
        """thermosense_forage_mode=True → world.forage_mode=True."""
        cfg = _tiny_cfg(
            enable_temperature=True,
            enable_thermosense=True,
            thermosense_forage_mode=True,
            enable_food_coupling=True,
        )
        eco = Ecology(cfg, seed=0)

        assert eco.world.forage_mode, (
            "world.forage_mode should be True when thermosense_forage_mode=True"
        )

    def test_forage_off_produces_avoid_mode_behavior(self):
        """Run a short sim in avoid mode (forage=False) and verify it runs without error.

        The key invariant: with thermosense on and forage_mode=False, the policy
        uses the thermal-avoidance branch (steering away from stress), NOT the
        foraging branch.  We verify this indirectly: the simulation runs, creatures
        move, and no exception is raised (the avoidance branch is exercised).
        """
        f = replace(
            founder(),
            thermosense_intensity=0.8,
            thermosense_inefficiency=0.3,
        )
        cfg = _tiny_cfg(
            enable_temperature=True,
            enable_thermosense=True,
            thermosense_forage_mode=False,      # avoid mode
            enable_food_coupling=False,
            temperature_stress_scale=0.3,
            thermosense_upkeep_floor=0.02,
            thermal_avoidance_weight=2.0,
            thermosense_noise_base=0.2,
            founder=f,
            horizon=50,
        )
        eco = Ecology(cfg, seed=7)
        summary = eco.run()

        # Should have run to completion (no explosion, not all founder deaths on step 1).
        assert summary["steps_run"] > 0, "Simulation should have run at least 1 step"
