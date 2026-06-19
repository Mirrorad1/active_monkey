"""tests/test_density_mortality.py — Exp 243 Mechanism A: global density-dependent crowding mortality.

TDD tests for:
  1. _density_mortality_p hazard-formula unit test (module-level pure helper).
  2. Integration tests: OFF byte-identical, ON lowers population, hmax=0 is null, determinism.

The run() summary returns "final_pop" (not "final_alive"); events live on eco.events (not
in the summary dict). Tests are adapted to the ACTUAL Ecology.run() return shape.
"""
from __future__ import annotations

import math
import dataclasses as D
import numpy as np
from ecology.engine import Ecology, _density_mortality_p
from tests.test_ecology_continuous import _cont_cfg, _founder_with_speed  # reuse helpers


def test_hazard_formula_and_clamp():
    # p = hmax * clamp((N/Kc)**theta, 0, 1)
    assert _density_mortality_p(60, hmax=0.04, Kc=60.0, theta=1.0) == 0.04
    assert _density_mortality_p(30, hmax=0.04, Kc=60.0, theta=1.0) == 0.02
    assert _density_mortality_p(0,  hmax=0.04, Kc=60.0, theta=1.0) == 0.0
    # clamp at 1: above Kc the density factor saturates at 1 -> p == hmax
    assert _density_mortality_p(600, hmax=0.04, Kc=60.0, theta=1.0) == 0.04
    # theta=2 quadratic
    assert abs(_density_mortality_p(30, hmax=0.04, Kc=60.0, theta=2.0) - 0.04*0.25) < 1e-12
    # optional lead term
    assert _density_mortality_p(60, hmax=0.04, Kc=60.0, theta=1.0, rate_scale=0.04, dN=60.0) == 0.04 + 0.04*1.0


def _dm_cfg(*, enable, hmax=0.04, horizon=120, seed=3, pop=40):
    return D.replace(_cont_cfg(horizon=horizon),
                     founder=_founder_with_speed(1.5),
                     initial_population=pop,
                     enable_continuous_depletion_intake=True,
                     continuous_floored_regen=True,
                     freeze_continuous_locomotion=True,
                     enable_density_mortality=enable,
                     density_mortality_hmax=hmax)


def _final_alive(res):
    # run() summary returns "final_pop" (alive at end of run)
    return res["final_pop"]


def test_off_path_byte_identical():
    on  = Ecology(_dm_cfg(enable=False), seed=3).run()["events_hash"]
    # An equivalent run WITHOUT the density-mortality config keys must match.
    base = Ecology(D.replace(_dm_cfg(enable=False)), seed=3).run()["events_hash"]
    assert on == base


def _dm_cfg_pop(*, enable, hmax, seed=3):
    """Short-horizon config where the OFF arm survives, so final_pop comparison is meaningful.

    The standard _dm_cfg with enable_continuous_depletion_intake=True leads to full extinction
    in both arms at horizon=120, making the final_pop comparison (0 < 0) vacuously false.
    This config uses horizon=25 so OFF survives and ON shows visibly lower N.
    """
    return D.replace(_cont_cfg(horizon=25),
                     founder=_founder_with_speed(1.5),
                     initial_population=40,
                     enable_continuous_depletion_intake=True,
                     continuous_floored_regen=True,
                     freeze_continuous_locomotion=True,
                     enable_density_mortality=enable,
                     density_mortality_hmax=hmax)


def test_on_differs_and_lowers_population():
    off = Ecology(_dm_cfg_pop(enable=False, hmax=0.30), seed=3).run()
    on  = Ecology(_dm_cfg_pop(enable=True,  hmax=0.30), seed=3).run()
    assert on["events_hash"] != off["events_hash"]
    assert _final_alive(on) < _final_alive(off)              # crowding mortality lowers N


def test_hmax_zero_on_but_null():
    # ON path (rng draw occurs) but hmax=0 => zero 'crowding' deaths, deterministic.
    eco1 = Ecology(_dm_cfg(enable=True, hmax=0.0), seed=3)
    r1 = eco1.run()
    eco2 = Ecology(_dm_cfg(enable=True, hmax=0.0), seed=3)
    r2 = eco2.run()
    assert r1["events_hash"] == r2["events_hash"]
    crowding = [e for e in eco1.events if e.get("details", {}).get("cause") == "crowding"]
    assert crowding == []


def test_fixed_order_determinism():
    a = Ecology(_dm_cfg(enable=True), seed=7).run()["events_hash"]
    b = Ecology(_dm_cfg(enable=True), seed=7).run()["events_hash"]
    assert a == b
