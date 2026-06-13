"""Tests for Exp 198 — N5: thermosense intensity ATTRACTOR + de-novo emergence.

Fast tests covering:
  - test_arm_configs_differ: A/B/C/D configs have the correct temperature and founder intensity.
  - test_newborn_window_metric: the newborn-intensity-in-window helper is distinct from a living
    snapshot and returns the mean over creatures with birth_t in the window.
  - test_short_run_determinism: a short run (horizon ~1500) of arms A and C, same seed twice,
    yields identical events_hash.
  - test_de_novo_starts_at_zero: arm A/C founder has thermosense_intensity == 0.0;
    arm B/D founder has thermosense_intensity == 0.20.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.exp198_n5_thermosense_attractor import (
    make_arm_A,
    make_arm_B,
    make_arm_C,
    make_arm_D,
    make_founder,
    newborn_mean_intensity_in_window,
    run_arm,
)
from ecology.engine import Ecology


# ---------------------------------------------------------------------------
# test_arm_configs_differ
# ---------------------------------------------------------------------------

def test_arm_configs_differ():
    """A/B/C/D differ correctly in enable_temperature + founder thermosense_intensity."""
    cfg_A = make_arm_A()
    cfg_B = make_arm_B()
    cfg_C = make_arm_C()
    cfg_D = make_arm_D()

    # Temperature flag
    assert cfg_A.enable_temperature is True,  "A: temperature should be ON"
    assert cfg_B.enable_temperature is True,  "B: temperature should be ON"
    assert cfg_C.enable_temperature is False, "C: temperature should be OFF"
    assert cfg_D.enable_temperature is False, "D: temperature should be OFF"

    # Founder intensity
    assert cfg_A.founder.thermosense_intensity == 0.0,  "A: de-novo, intensity must be 0.0"
    assert cfg_B.founder.thermosense_intensity == 0.20, "B: seeded, intensity must be 0.20"
    assert cfg_C.founder.thermosense_intensity == 0.0,  "C: de-novo, intensity must be 0.0"
    assert cfg_D.founder.thermosense_intensity == 0.20, "D: seeded, intensity must be 0.20"

    # All enable_thermosense (required by regime)
    assert cfg_A.enable_thermosense is True
    assert cfg_B.enable_thermosense is True
    assert cfg_C.enable_thermosense is True
    assert cfg_D.enable_thermosense is True


# ---------------------------------------------------------------------------
# test_de_novo_starts_at_zero
# ---------------------------------------------------------------------------

def test_de_novo_starts_at_zero():
    """Arm A/C founder has thermosense_intensity == 0.0; arm B/D == 0.20."""
    f_A = make_founder(0.0)
    f_B = make_founder(0.20)
    f_C = make_founder(0.0)
    f_D = make_founder(0.20)

    assert f_A.thermosense_intensity == 0.0
    assert f_C.thermosense_intensity == 0.0
    assert f_B.thermosense_intensity == 0.20
    assert f_D.thermosense_intensity == 0.20

    # Confirm tolerance is set to 0.10 for all founders
    for f in (f_A, f_B, f_C, f_D):
        assert f.temperature_tolerance == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# test_newborn_window_metric
# ---------------------------------------------------------------------------

def test_newborn_window_metric():
    """The newborn-intensity-in-window helper returns mean over creatures with
    birth_t in [window_start, window_end], distinct from a living snapshot."""
    # Build mock creatures
    def make_mock(birth_t, intensity, parent_id=1, alive=True):
        c = MagicMock()
        c.parent_id = parent_id
        c.phenotype = MagicMock()
        c.phenotype.birth_t = birth_t
        c.genotype = MagicMock()
        c.genotype.thermosense_intensity = intensity
        c.phenotype.alive = alive
        return c

    # Creatures with birth_t in and out of [10000, 12000]
    c1 = make_mock(birth_t=9999, intensity=0.5)   # OUTSIDE window (too early)
    c2 = make_mock(birth_t=10000, intensity=0.10)  # IN window
    c3 = make_mock(birth_t=11000, intensity=0.20)  # IN window
    c4 = make_mock(birth_t=12000, intensity=0.30)  # IN window (boundary inclusive)
    c5 = make_mock(birth_t=12001, intensity=0.9)   # OUTSIDE window (too late)
    c6 = make_mock(birth_t=11500, intensity=0.40, parent_id=None)  # IN time but founder -> excluded

    creatures = [c1, c2, c3, c4, c5, c6]

    result = newborn_mean_intensity_in_window(creatures, 10000, 12000)
    # c2, c3, c4 qualify: intensities 0.10, 0.20, 0.30 -> mean = 0.20
    assert result == pytest.approx(0.20, abs=1e-9)

    # Living snapshot would include c1 (alive=True, intensity 0.5) if we just averaged alive —
    # confirm the window metric gives a DIFFERENT result than a naive alive-average.
    naive_alive_mean = (0.5 + 0.10 + 0.20 + 0.30 + 0.9 + 0.40) / 6  # 0.4083
    assert result != pytest.approx(naive_alive_mean, abs=1e-9)

    # Empty window
    result_empty = newborn_mean_intensity_in_window(creatures, 20000, 25000)
    import math
    assert math.isnan(result_empty)


# ---------------------------------------------------------------------------
# test_short_run_determinism
# ---------------------------------------------------------------------------

def test_short_run_determinism():
    """Short run (horizon ~1500) of arm A and arm C, same seed twice, identical events_hash."""
    import dataclasses as D
    from ecology.scenarios import SCENARIOS

    SHORT_HORIZON = 1500
    SEED = 42

    # Patch A and C configs to short horizon
    cfg_A_short = D.replace(make_arm_A(), horizon=SHORT_HORIZON)
    cfg_C_short = D.replace(make_arm_C(), horizon=SHORT_HORIZON)

    # Run arm A twice
    eco_A1 = Ecology(cfg_A_short, seed=SEED)
    while eco_A1.t < SHORT_HORIZON and not eco_A1.exploded and eco_A1._alive():
        eco_A1.step()

    eco_A2 = Ecology(cfg_A_short, seed=SEED)
    while eco_A2.t < SHORT_HORIZON and not eco_A2.exploded and eco_A2._alive():
        eco_A2.step()

    assert eco_A1.events_hash() == eco_A2.events_hash(), (
        f"Arm A (temp-ON) not deterministic: {eco_A1.events_hash()} != {eco_A2.events_hash()}"
    )

    # Run arm C twice
    eco_C1 = Ecology(cfg_C_short, seed=SEED)
    while eco_C1.t < SHORT_HORIZON and not eco_C1.exploded and eco_C1._alive():
        eco_C1.step()

    eco_C2 = Ecology(cfg_C_short, seed=SEED)
    while eco_C2.t < SHORT_HORIZON and not eco_C2.exploded and eco_C2._alive():
        eco_C2.step()

    assert eco_C1.events_hash() == eco_C2.events_hash(), (
        f"Arm C (temp-OFF) not deterministic: {eco_C1.events_hash()} != {eco_C2.events_hash()}"
    )
