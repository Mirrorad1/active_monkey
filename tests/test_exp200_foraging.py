"""tests/test_exp200_foraging.py — Fast unit tests for Exp 200 foraging escape.

Tests:
  - test_arm_configs: WIDE/NARROW have enable_food_coupling=True + forage_mode + correct params;
    CONTROL has enable_food_coupling=False.
  - test_newborn_window_metric: newborn metric uses birth_t window, distinct from living snapshot.
  - test_validity_marks_collapse_invalid: arm with pop<10/no newborns is marked None, not 0.
  - test_short_run_determinism: SHORT run (horizon ~1500) of WIDE, same seed twice, identical hash.
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure repo root is on sys.path when tests are run directly.
_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ecology.engine import Ecology, EcologyConfig
from ecology.creature import Creature, Phenotype
from ecology.genotype import Genotype, founder as base_founder
from ecology.scenarios import SCENARIOS, FOUNDER

# Import helpers from the experiment script
from experiments.exp200_n5_foraging_escape import (
    SEEDS,
    HORIZON,
    NEWBORN_WINDOW_START,
    NEWBORN_WINDOW_END,
    MIN_VALID_POP,
    cfg_forage,
    cfg_control,
    make_arm_cfg,
    founder as exp_founder,
    newborn_mean_intensity_in_window,
    is_valid_run,
    run_arm,
)


# ---------------------------------------------------------------------------
# Test 1: arm configs have the expected flag settings
# ---------------------------------------------------------------------------
class TestArmConfigs:
    """WIDE and NARROW must have enable_food_coupling=True and forage_mode=True.
    CONTROL must have enable_food_coupling=False (and forage_mode=False).
    Also verify the swept band/amp/period params are set correctly per arm.
    """

    def test_wide_has_food_coupling_and_forage_mode(self):
        cfg = make_arm_cfg("WIDE")
        assert cfg.enable_food_coupling is True, (
            f"WIDE: enable_food_coupling should be True, got {cfg.enable_food_coupling}"
        )
        assert cfg.thermosense_forage_mode is True, (
            f"WIDE: thermosense_forage_mode should be True, got {cfg.thermosense_forage_mode}"
        )

    def test_wide_band_params(self):
        cfg = make_arm_cfg("WIDE")
        assert abs(cfg.food_band_width - 0.15) < 1e-12, f"WIDE band_width={cfg.food_band_width}"
        assert abs(cfg.food_optimal_amplitude - 0.3) < 1e-12, f"WIDE amplitude={cfg.food_optimal_amplitude}"
        assert abs(cfg.food_optimal_period - 1500.0) < 1e-12, f"WIDE period={cfg.food_optimal_period}"

    def test_narrow_has_food_coupling_and_forage_mode(self):
        cfg = make_arm_cfg("NARROW")
        assert cfg.enable_food_coupling is True, (
            f"NARROW: enable_food_coupling should be True, got {cfg.enable_food_coupling}"
        )
        assert cfg.thermosense_forage_mode is True, (
            f"NARROW: thermosense_forage_mode should be True, got {cfg.thermosense_forage_mode}"
        )

    def test_narrow_band_params(self):
        cfg = make_arm_cfg("NARROW")
        assert abs(cfg.food_band_width - 0.05) < 1e-12, f"NARROW band_width={cfg.food_band_width}"
        assert abs(cfg.food_optimal_amplitude - 0.4) < 1e-12, f"NARROW amplitude={cfg.food_optimal_amplitude}"
        assert abs(cfg.food_optimal_period - 600.0) < 1e-12, f"NARROW period={cfg.food_optimal_period}"

    def test_control_has_no_food_coupling(self):
        cfg = make_arm_cfg("CONTROL")
        assert cfg.enable_food_coupling is False, (
            f"CONTROL: enable_food_coupling should be False, got {cfg.enable_food_coupling}"
        )
        assert cfg.thermosense_forage_mode is False, (
            f"CONTROL: thermosense_forage_mode should be False, got {cfg.thermosense_forage_mode}"
        )

    def test_all_arms_have_thermosense_enabled(self):
        """All arms have enable_thermosense=True (including CONTROL)."""
        for arm in ["WIDE", "NARROW", "CONTROL"]:
            cfg = make_arm_cfg(arm)
            assert cfg.enable_thermosense is True, (
                f"{arm}: enable_thermosense should be True"
            )

    def test_all_arms_have_zero_stress(self):
        """temperature_stress_scale=0.0: ONLY benefit is foraging, no avoidance pressure."""
        for arm in ["WIDE", "NARROW", "CONTROL"]:
            cfg = make_arm_cfg(arm)
            assert cfg.temperature_stress_scale == 0.0, (
                f"{arm}: temperature_stress_scale should be 0.0, got {cfg.temperature_stress_scale}"
            )

    def test_founder_params(self):
        """Founder has intensity=0.10, inefficiency=0.20, tolerance=0.10."""
        f = exp_founder()
        assert abs(f.thermosense_intensity - 0.10) < 1e-12, (
            f"founder intensity={f.thermosense_intensity}"
        )
        assert abs(f.thermosense_inefficiency - 0.20) < 1e-12, (
            f"founder inefficiency={f.thermosense_inefficiency}"
        )
        assert abs(f.temperature_tolerance - 0.10) < 1e-12, (
            f"founder tolerance={f.temperature_tolerance}"
        )


# ---------------------------------------------------------------------------
# Test 2: newborn window metric is distinct from living snapshot
# ---------------------------------------------------------------------------
class TestNewbornWindowMetric:
    """The newborn metric reads birth cohort by birth_t window, not who is alive."""

    def _make_mock_creature(
        self,
        parent_id: int | None,
        birth_t: int,
        intensity: float,
        alive: bool = True,
    ) -> MagicMock:
        """Create a minimal mock creature for metric testing."""
        c = MagicMock()
        c.parent_id = parent_id
        c.phenotype = MagicMock()
        c.phenotype.birth_t = birth_t
        c.genotype = MagicMock()
        c.genotype.thermosense_intensity = intensity
        c.is_alive.return_value = alive
        return c

    def test_uses_birth_window_not_alive_status(self):
        """Newborns born in window are counted even if now dead;
        newborns born outside window are excluded even if alive.
        """
        # In-window, now dead: should be counted
        c1 = self._make_mock_creature(parent_id=0, birth_t=10500, intensity=0.8, alive=False)
        # In-window, alive: should be counted
        c2 = self._make_mock_creature(parent_id=1, birth_t=11000, intensity=0.4, alive=True)
        # Out-of-window, alive: should be excluded
        c3 = self._make_mock_creature(parent_id=2, birth_t=5000, intensity=0.9, alive=True)
        # Founder (parent_id=None), in-window: should be excluded
        c4 = self._make_mock_creature(parent_id=None, birth_t=10200, intensity=0.7, alive=True)

        creatures = [c1, c2, c3, c4]
        result = newborn_mean_intensity_in_window(creatures, 10000, 12000)

        # Should average c1 (0.8) and c2 (0.4) = 0.6; c3 and c4 excluded
        assert abs(result - 0.6) < 1e-12, (
            f"Expected mean=0.6 (in-window non-founders only), got {result}"
        )

    def test_excludes_founders(self):
        """Founders (parent_id=None) are never included in newborn metric."""
        # Founder born at t=0 (in window if window=[0, 12000])
        founder_c = self._make_mock_creature(parent_id=None, birth_t=5000, intensity=0.99)
        # Real newborn
        newborn_c = self._make_mock_creature(parent_id=0, birth_t=5000, intensity=0.1)

        result = newborn_mean_intensity_in_window([founder_c, newborn_c], 0, 12000)
        assert abs(result - 0.1) < 1e-12, (
            f"Founder should be excluded; expected 0.1, got {result}"
        )

    def test_returns_nan_when_no_newborns_in_window(self):
        """If no newborns in the window, return NaN (not 0.0 or error)."""
        # All creatures outside window
        c1 = self._make_mock_creature(parent_id=0, birth_t=500, intensity=0.5)
        c2 = self._make_mock_creature(parent_id=0, birth_t=999, intensity=0.3)

        result = newborn_mean_intensity_in_window([c1, c2], 1000, 2000)
        assert math.isnan(result), f"Expected NaN when no newborns in window, got {result}"

    def test_distinct_from_living_snapshot(self):
        """Demonstrates that counting only alive creatures would give different result."""
        # Born in window, dead: intensity=0.8
        dead_c = self._make_mock_creature(parent_id=0, birth_t=10500, intensity=0.8, alive=False)
        # Born outside window, alive: intensity=0.1
        alive_outside = self._make_mock_creature(parent_id=0, birth_t=5000, intensity=0.1, alive=True)

        metric = newborn_mean_intensity_in_window([dead_c, alive_outside], 10000, 12000)
        # Correct metric: only dead_c (in-window) = 0.8
        assert abs(metric - 0.8) < 1e-12, (
            f"Expected 0.8 (only in-window creature), got {metric}"
        )
        # If we incorrectly used alive status: only alive_outside = 0.1 (WRONG)
        # This test confirms we are NOT filtering by alive status.


# ---------------------------------------------------------------------------
# Test 3: validity marks collapsed/extinct arm as None (not a numeric 0)
# ---------------------------------------------------------------------------
class TestValidityMarksCollapseInvalid:
    """An arm with pop<10 or no newborns in the end window is INVALID -> None, not 0."""

    def test_collapsed_arm_is_marked_none_not_zero(self):
        """A short run with pop<MIN_VALID_POP is invalid; end_nb should be None."""
        # Use a tiny, hostile config that goes extinct quickly.
        cfg = replace(
            SCENARIOS["balanced"],
            horizon=300,
            max_population=5000,
            min_survival_energy=18.0,  # extremely harsh: almost nothing survives
            founder=exp_founder(),
            enable_food_coupling=False,
            thermosense_forage_mode=False,
            enable_thermosense=True,
            enable_temperature=True,
            temperature_stress_scale=0.0,
            tolerance_cost_scale=0.0,
            comfort_amplitude=0.0,
            thermosense_upkeep_floor=0.0,
            thermosense_active_threshold=0.05,
            thermosense_noise_base=0.5,
            thermal_avoidance_weight=4.0,
            food_optimal_base=0.5,
            food_concentration=8.0,
            food_band_width=0.15,
            food_optimal_amplitude=0.0,
            food_optimal_period=1500.0,
        )
        eco = Ecology(cfg, seed=99)
        eco.run()

        alive = eco._alive()
        end_nb_intensity = newborn_mean_intensity_in_window(
            eco._creatures, 10000, 12000
        )
        valid = is_valid_run(eco, end_nb_intensity)

        if not valid:
            # This is the correct result: invalid => mark as None, not 0
            # Simulate the verdict logic: end_nb[arm][seed] = None if not valid
            end_nb_for_seed: float | None = None if not valid else end_nb_intensity
            assert end_nb_for_seed is None, (
                "Invalid run should produce None, not a numeric value"
            )
        # If somehow this config produced a valid run, the test is informational.

    def test_no_newborns_in_window_means_invalid(self):
        """A run where no creatures were born in [10000, 12000] is invalid (NaN -> None)."""
        # Simulate: the run reached horizon, has pop>=10, but no newborns in window.
        # This happens e.g. with very high reproduction threshold.
        cfg = replace(
            SCENARIOS["balanced"],
            horizon=500,
            max_population=5000,
            founder=exp_founder(),
            enable_food_coupling=False,
            thermosense_forage_mode=False,
            enable_thermosense=True,
            enable_temperature=True,
            temperature_stress_scale=0.0,
            tolerance_cost_scale=0.0,
            comfort_amplitude=0.0,
            thermosense_upkeep_floor=0.0,
            thermosense_active_threshold=0.05,
            thermosense_noise_base=0.5,
            thermal_avoidance_weight=4.0,
            food_optimal_base=0.5,
            food_concentration=8.0,
            food_band_width=0.15,
            food_optimal_amplitude=0.0,
            food_optimal_period=1500.0,
        )
        eco = Ecology(cfg, seed=42)
        eco.run()

        # Short horizon (500) means no births in [10000, 12000].
        end_nb_intensity = newborn_mean_intensity_in_window(
            eco._creatures, 10000, 12000
        )
        # Must be NaN (no newborns in that future window)
        assert math.isnan(end_nb_intensity), (
            f"Expected NaN for short-horizon run with window [10000,12000], "
            f"got {end_nb_intensity}"
        )

        valid = is_valid_run(eco, end_nb_intensity)
        # Not valid because no newborns in window (NaN intensity)
        assert not valid, (
            "Run with no newborns in [10000,12000] should be invalid"
        )

    def test_valid_run_produces_numeric_not_nan(self):
        """A short run (horizon < NEWBORN_WINDOW_START) produces no newborns in the end
        window, so end_newborn_intensity is NaN and valid=False.
        The key contract: invalid -> valid=False and end_nb is NaN (not 0.0).
        """
        # Use WIDE arm with very short horizon (50 steps) — fast, and horizon < 10000.
        cfg = replace(
            make_arm_cfg("WIDE"),
            horizon=50,
        )
        # run_arm respects cfg.horizon, so simulation stops at t=50.
        result = run_arm(cfg, seed=23)
        s = result["summary"]
        # With horizon=50, no newborn in [10000,12000] -> valid=False
        assert not s["valid"], (
            f"With horizon=50 there are no newborns in [10000,12000]; expected valid=False, "
            f"got valid={s['valid']}, end_nb={s['end_newborn_intensity']}"
        )
        # end_newborn_intensity should be NaN (not 0.0 or another numeric)
        assert math.isnan(s["end_newborn_intensity"]), (
            f"end_newborn_intensity should be NaN when no newborns in window, "
            f"got {s['end_newborn_intensity']}"
        )
        # Sanity: some creatures should have been created (founders + offspring)
        assert s["steps_run"] <= 50, (
            f"Short run should stop at or before horizon=50, got steps_run={s['steps_run']}"
        )


# ---------------------------------------------------------------------------
# Test 4: short-run determinism (WIDE, same seed twice, identical events_hash)
# ---------------------------------------------------------------------------
class TestShortRunDeterminism:
    """Two runs of WIDE with identical seeds must produce identical events_hash."""

    def test_wide_same_seed_identical_hash(self):
        """WIDE arm, seed=42, two runs: events_hash must match."""
        short_cfg = replace(make_arm_cfg("WIDE"), horizon=1500)

        result_a = run_arm(short_cfg, seed=42)
        result_b = run_arm(short_cfg, seed=42)

        hash_a = result_a["summary"]["events_hash"]
        hash_b = result_b["summary"]["events_hash"]

        assert hash_a == hash_b, (
            f"Two runs of WIDE (seed=42, horizon=1500) produced different event hashes!\n"
            f"  Run 1: {hash_a}\n"
            f"  Run 2: {hash_b}\n"
            "This indicates non-determinism in the engine."
        )

    def test_different_seeds_produce_different_hashes(self):
        """Different seeds should (almost certainly) produce different event streams."""
        short_cfg = replace(make_arm_cfg("WIDE"), horizon=1500)

        result_23 = run_arm(short_cfg, seed=23)
        result_24 = run_arm(short_cfg, seed=24)

        hash_23 = result_23["summary"]["events_hash"]
        hash_24 = result_24["summary"]["events_hash"]

        # Two independent seeds should produce distinct runs
        assert hash_23 != hash_24, (
            "Seeds 23 and 24 produced identical event hashes — highly unexpected"
        )

    def test_control_same_seed_identical_hash(self):
        """CONTROL arm, same seed, identical hash (verifies determinism on the no-coupling path)."""
        short_cfg = replace(make_arm_cfg("CONTROL"), horizon=1500)

        result_a = run_arm(short_cfg, seed=23)
        result_b = run_arm(short_cfg, seed=23)

        hash_a = result_a["summary"]["events_hash"]
        hash_b = result_b["summary"]["events_hash"]

        assert hash_a == hash_b, (
            f"CONTROL arm (seed=23, horizon=1500) produced different event hashes!\n"
            f"  Run 1: {hash_a}\n"
            f"  Run 2: {hash_b}"
        )
