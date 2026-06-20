"""tests/test_logistic_prey_growth.py — Exp 249: gated logistic prey-growth.

TDD tests for the instantaneous density-dependent birth suppression mechanism
(enable_logistic_prey_growth / prey_carrying_capacity):

  1. test_off_byte_identical         — OFF config reproduces same events_hash across two runs.
  2. test_on_differs_from_off        — ON (K=40) produces a DIFFERENT hash than OFF.
  3. test_caps_prey_population       — ON at low K=40 keeps prey count below OFF at the horizon.
  4. test_on_deterministic           — ON run is bit-identical across two runs.

All tests use the VIABLE Exp-240/242 calibrated continuous founder (not raw founder() —
which collapses, making cap tests meaningless).
"""
from __future__ import annotations

import dataclasses as D

from ecology.engine import Ecology
from ecology.genotype import founder
from ecology.scenarios import SCENARIOS

# ---------------------------------------------------------------------------
# Exp-240/242 calibrated continuous founder (mirrored from test_continuous_viability.py)
# ---------------------------------------------------------------------------
_CAL = dict(
    baseline_metabolic_cost=0.05,
    movement_cost=0.03,
    energy_capacity=10.0,
    reproduction_energy_threshold=4.2,
    reproduction_energy_transfer_fraction=0.35,
    reproduction_cost_fraction=0.06,
    aging_cost=0.003,
)


def _cont_base(**over):
    base = dict(
        enable_continuous_locomotion=True,
        continuous_layout="bump",
        continuous_dt=1.0,
        speed_cost_floor=0.0,
        speed_cost_slope=0.0,
        enable_continuous_depletion_intake=True,
        continuous_logistic_regen=False,
        continuous_regen_rate=1.0,
        continuous_capacity=2.0,
        min_survival_energy=0.5,
        mutation_rate=0.0,
        max_population=4000,
        horizon=150,
    )
    base.update(over)
    return D.replace(SCENARIOS["balanced"], **base)


def _viable_founder():
    """Return the Exp-240 calibrated founder with locomotor_speed=1.0."""
    return D.replace(founder(), locomotor_speed=1.0, **_CAL)


def _lpg_cfg(*, enable: bool, k: float = 40.0, horizon: int = 150, seed: int = 0):
    """Build a viable continuous config with logistic prey growth optionally enabled."""
    g = _viable_founder()
    return _cont_base(
        founder=g,
        initial_population=21,
        horizon=horizon,
        enable_logistic_prey_growth=enable,
        prey_carrying_capacity=k,
    )


def _run_hash(cfg, seed: int = 0) -> str:
    return Ecology(cfg, seed=seed).run()["events_hash"]


def _final_prey(cfg, seed: int = 0) -> int:
    """Run to horizon, return count of alive creatures (all prey in prey-only sim)."""
    eco = Ecology(cfg, seed=seed)
    while eco.has_alive() and not eco.exploded and eco.t < cfg.horizon:
        eco.step()
    return eco.alive_count()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_off_byte_identical():
    """With enable_logistic_prey_growth=False, two runs produce the SAME events_hash."""
    cfg = _lpg_cfg(enable=False)
    h1 = _run_hash(cfg, seed=0)
    h2 = _run_hash(cfg, seed=0)
    assert h1 == h2, (
        f"OFF path is NOT deterministic!\n  run1: {h1}\n  run2: {h2}"
    )


def test_on_differs_from_off():
    """ON (K=40) must produce a DIFFERENT events_hash than OFF — mechanism is not a no-op."""
    cfg_off = _lpg_cfg(enable=False, k=40.0)
    cfg_on = _lpg_cfg(enable=True, k=40.0)
    h_off = _run_hash(cfg_off, seed=0)
    h_on = _run_hash(cfg_on, seed=0)
    assert h_off != h_on, (
        "ON and OFF produced the SAME events_hash — logistic prey growth is a silent no-op!\n"
        f"  hash: {h_off}"
    )


def test_caps_prey_population():
    """ON (K=40) keeps final prey count below OFF at the horizon.

    The logistic suppression should bind at low K, preventing the bloom that occurs
    in the OFF arm.  Assert ON_final < OFF_final.
    """
    cfg_off = _lpg_cfg(enable=False, k=40.0, horizon=150)
    cfg_on = _lpg_cfg(enable=True, k=40.0, horizon=150)

    n_off = _final_prey(cfg_off, seed=0)
    n_on = _final_prey(cfg_on, seed=0)

    assert n_on < n_off, (
        f"Logistic suppression did NOT cap prey population!\n"
        f"  ON_final={n_on}  OFF_final={n_off}\n"
        f"  Expected ON_final < OFF_final (K=40 should bind vs uncapped OFF)"
    )


def test_on_deterministic():
    """Two identical-seed ON runs produce identical events_hash (determinism guard)."""
    cfg = _lpg_cfg(enable=True, k=40.0)
    h1 = _run_hash(cfg, seed=0)
    h2 = _run_hash(cfg, seed=0)
    assert h1 == h2, (
        f"ON path is NOT deterministic!\n  run1: {h1}\n  run2: {h2}"
    )
