"""tests/test_decoupled_prey_birth.py — Exp 250: gated additive resource-independent
logistic prey birth.

TDD tests for the decoupled prey birth mechanism
(enable_decoupled_prey_birth / prey_birth_rate / prey_carrying_capacity):

  1. test_off_byte_identical          — OFF config reproduces same events_hash across two runs.
  2. test_on_differs_from_off         — ON (rate=0.3, K=80) produces a DIFFERENT hash than OFF.
  3. test_raises_low_density_productivity — starting from small prey pop well below K,
                                         ON prey count exceeds OFF prey count at an early horizon.
  4. test_on_deterministic            — ON run is bit-identical across two runs.

All tests use the VIABLE Exp-240/242 calibrated continuous founder (not raw founder() —
which collapses, making productivity tests meaningless).
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


def _dpb_cfg(*, enable: bool, rate: float = 0.3, k: float = 80.0,
             horizon: int = 150, initial_population: int = 21, seed: int = 0):
    """Build a viable continuous config with decoupled prey birth optionally enabled."""
    g = _viable_founder()
    return _cont_base(
        founder=g,
        initial_population=initial_population,
        horizon=horizon,
        enable_decoupled_prey_birth=enable,
        prey_birth_rate=rate,
        prey_carrying_capacity=k,
    )


def _run_hash(cfg, seed: int = 0) -> str:
    return Ecology(cfg, seed=seed).run()["events_hash"]


def _prey_count_at(cfg, t_target: int, seed: int = 0) -> int:
    """Run to t_target steps (or earlier if extinct/exploded), return alive count."""
    eco = Ecology(cfg, seed=seed)
    while eco.has_alive() and not eco.exploded and eco.t < t_target:
        eco.step()
    return eco.alive_count()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_off_byte_identical():
    """With enable_decoupled_prey_birth=False, two runs produce the SAME events_hash."""
    cfg = _dpb_cfg(enable=False)
    h1 = _run_hash(cfg, seed=0)
    h2 = _run_hash(cfg, seed=0)
    assert h1 == h2, (
        f"OFF path is NOT deterministic!\n  run1: {h1}\n  run2: {h2}"
    )


def test_on_differs_from_off():
    """ON (rate=0.3, K=80) must produce a DIFFERENT events_hash than OFF.

    This confirms the mechanism is not a no-op when enabled.
    """
    cfg_off = _dpb_cfg(enable=False, rate=0.3, k=80.0)
    cfg_on = _dpb_cfg(enable=True, rate=0.3, k=80.0)
    h_off = _run_hash(cfg_off, seed=0)
    h_on = _run_hash(cfg_on, seed=0)
    assert h_off != h_on, (
        "ON and OFF produced the SAME events_hash — decoupled prey birth is a silent no-op!\n"
        f"  hash: {h_off}"
    )


def test_raises_low_density_productivity():
    """Starting from a small prey pop (5) well below K=120, ON count exceeds OFF at t=40.

    The resource-independent births should accelerate low-density growth so that after
    40 steps the ON arm has more prey than the OFF arm.  The 5-prey start ensures we
    probe the low-density regime where the additive term is strongest.
    """
    T_HORIZON = 40
    K = 120.0
    INIT_POP = 5

    cfg_off = _dpb_cfg(enable=False, rate=0.3, k=K,
                       horizon=T_HORIZON, initial_population=INIT_POP)
    cfg_on = _dpb_cfg(enable=True, rate=0.3, k=K,
                      horizon=T_HORIZON, initial_population=INIT_POP)

    n_off = _prey_count_at(cfg_off, T_HORIZON, seed=0)
    n_on = _prey_count_at(cfg_on, T_HORIZON, seed=0)

    assert n_on > n_off, (
        f"Decoupled prey birth did NOT raise low-density productivity!\n"
        f"  ON_count={n_on}  OFF_count={n_off}  (t={T_HORIZON}, N0={INIT_POP}, K={K})\n"
        f"  Expected ON_count > OFF_count (resource-independent births accelerate low-density growth)"
    )


def test_on_deterministic():
    """Two identical-seed ON runs produce identical events_hash (determinism guard)."""
    cfg = _dpb_cfg(enable=True, rate=0.3, k=80.0)
    h1 = _run_hash(cfg, seed=0)
    h2 = _run_hash(cfg, seed=0)
    assert h1 == h2, (
        f"ON path is NOT deterministic!\n  run1: {h1}\n  run2: {h2}"
    )
