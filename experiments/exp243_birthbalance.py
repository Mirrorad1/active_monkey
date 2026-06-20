"""Exp 243 — Birth-balance diagnostic.

Goal: measure the empirical per-capita BIRTH rate `b` so we can confirm Kc=60 places
the crowding equilibrium below the steep knee of the theta-logistic hazard.

Strategy:
  1. Run monomorphic speed=1.0 with density-mortality OFF (hmax=0.0) to measure UNTHROTTLED
     birth rate during the growth phase. If births = 0, this is itself the critical finding.
  2. Analyse the energy budget analytically to understand WHY births do/don't occur and
     compute the THEORETICAL b (or confirm it is zero at these params).
  3. Report implied Kc or a finding that the sweep defaults will produce ZERO births.

HONESTY NOTE: The default `SCENARIOS["balanced"]` founder has reproduction_energy_threshold=17.0
(85% of energy_capacity=20.0) and aging_cost=0.02/step. In the continuous world at the sweep
defaults (regen_rate=0.05, continuous_capacity=2.0), single-creature analysis shows the
maximum achievable energy is ~15.0 (start) + small increment before aging dominates.
If births never occur, the birth-balance argument for Kc=60 is not empirically grounded and
this must be reported clearly as a NO-GO finding for the preflight's human gate.

Run:
  uv run --python .venv python experiments/exp243_birthbalance.py
"""
from __future__ import annotations

import os
import sys
import time
import dataclasses as D
import math
from pathlib import Path

import numpy as np

# Ensure repo root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from ecology.engine import Ecology, EcologyConfig, _density_mortality_p
from ecology.scenarios import SCENARIOS
from ecology.genotype import founder as _base_founder
from ecology.evolvability.cert_run import _reconstruct_N_per_step, _founder_with_speed, _EXP242_BMC, _EXP242_MC, _EXP242_CAP, _EXP242_THRESHOLD, _EXP242_TRANSFER, _EXP242_COST_FRAC, _EXP242_AGING, _EXP242_CONTINUOUS_CAPACITY
from ecology.continuous_world import ContinuousWorld

# ---- Parameters (matching the cert_run sweep defaults) ----
SPEED = 1.0
HORIZON = 2000
# Use the same regen_rate that cert_run uses (thread via run_cert hmax=0.0)
# The Exp-242 viable regime used regen_rate in [0.5, 1.0, 2.0]; use 1.0 as the reference.
REGEN_RATE = 1.0
LAYOUT = "bump"
SEED = 42

HMAX_TARGET = 0.04
THETA_TARGET = 1.0
KC_DEFAULT = 60.0


def _build_birthbalance_config() -> EcologyConfig:
    """Build config on the Exp-242 viable continuous base; density-mortality OFF (hmax=0).

    Uses the Exp-242 viability-calibrated founder (threshold=4.2, cap=10.0, bmc=0.05,
    mc=0.03, aging=0.003) and substrate (continuous_capacity=2.0, min_survival_energy=0.5,
    max_population=4000, continuous_logistic_regen=False) rather than the SCENARIOS["balanced"]
    discrete founder (threshold=17.0, cap=20.0) which was NON-VIABLE on the continuous substrate.
    """
    base = SCENARIOS["balanced"]
    return D.replace(
        base,
        enable_continuous_locomotion=True,
        continuous_layout=LAYOUT,
        continuous_dt=1.0,
        speed_cost_floor=0.0,
        speed_cost_slope=0.6,
        continuous_regen_rate=REGEN_RATE,
        continuous_capacity=_EXP242_CONTINUOUS_CAPACITY,
        enable_continuous_depletion_intake=True,
        continuous_logistic_regen=False,
        continuous_floored_regen=True,
        # Density mortality OFF — we want unthrottled birth measurement
        enable_density_mortality=False,
        density_mortality_hmax=0.0,
        density_mortality_Kc=KC_DEFAULT,
        density_mortality_theta=THETA_TARGET,
        density_mortality_rate_scale=0.0,
        freeze_continuous_locomotion=True,
        # Use Exp-242 viable founder, NOT the discrete balanced founder
        founder=_founder_with_speed(SPEED),
        initial_population=21,
        max_population=4000,
        min_survival_energy=0.5,
        horizon=HORIZON,
        mutation_rate=0.0,
    )


def _measure_single_creature_max_intake() -> float:
    """Run a single Exp-242-viable creature with effectively infinite regen to measure max intake/step."""
    base = SCENARIOS["balanced"]
    cfg = D.replace(
        base,
        enable_continuous_locomotion=True,
        continuous_layout=LAYOUT,
        continuous_dt=1.0,
        speed_cost_floor=0.0,
        speed_cost_slope=0.6,
        # Very high regen to keep field full
        continuous_regen_rate=100.0,
        continuous_capacity=_EXP242_CONTINUOUS_CAPACITY,
        enable_continuous_depletion_intake=True,
        continuous_logistic_regen=False,
        continuous_floored_regen=True,
        enable_density_mortality=False,
        freeze_continuous_locomotion=True,
        # Use Exp-242 viable founder
        founder=_founder_with_speed(SPEED),
        initial_population=1,
        max_population=4000,
        min_survival_energy=0.5,
        horizon=1,
        mutation_rate=0.0,
    )
    eco = Ecology(cfg, seed=SEED)
    eco.run()
    alive = [c for c in eco._creatures if c.phenotype.alive]
    if alive:
        # start_energy = 0.75 * energy_capacity (Exp-242 cap=10.0 => start=7.5)
        start_energy = _EXP242_CAP * 0.75
        step_cost_age0 = _EXP242_MC + _EXP242_BMC + 0.6 * SPEED  # movement + bmc + speed_cost
        energy_after = alive[0].phenotype.energy
        intake_est = energy_after - start_energy + step_cost_age0
        return intake_est
    return 0.0


def _energy_budget_analysis() -> dict:
    """Analytical energy budget: can an Exp-242-viable creature reach the reproduction threshold?

    Uses the Exp-242 calibrated founder params (threshold=4.2, cap=10.0, bmc=0.05, mc=0.03,
    aging=0.003) which replaced the NON-VIABLE discrete founder (threshold=17, cap=20).
    Returns a dict with the analysis results.
    """
    start_energy = _EXP242_CAP * 0.75   # = 7.5
    repro_threshold = _EXP242_THRESHOLD  # = 4.2
    max_energy = _EXP242_CAP              # = 10.0
    speed_cost = 0.6 * SPEED              # = 0.6 at speed=1.0

    # Measure single-step intake at full field, single Exp-242 creature
    max_intake = _measure_single_creature_max_intake()

    # Per-step energy trajectory with constant max intake
    energy = start_energy
    max_achieved = start_energy
    repro_age = None

    for age in range(500):  # simulate up to 500 steps
        step_cost = _EXP242_MC + _EXP242_BMC + speed_cost + _EXP242_AGING * age
        energy = energy + max_intake - step_cost
        if energy > max_achieved:
            max_achieved = energy
        if energy > max_energy:
            energy = max_energy  # cap at capacity

        if energy >= repro_threshold and repro_age is None:
            repro_age = age  # first age where reproduction could happen

        if energy <= 0:
            break

    base_cost_age0 = _EXP242_MC + _EXP242_BMC + speed_cost  # = 0.03 + 0.05 + 0.6 = 0.68
    return {
        "start_energy": start_energy,
        "repro_threshold": repro_threshold,
        "max_achievable_energy": max_achieved,
        "max_intake_per_step": max_intake,
        "repro_age": repro_age,  # None if never reached
        "step_cost_age0": base_cost_age0,
        "aging_cost_per_step": _EXP242_AGING,
        "break_even_age": (max_intake - base_cost_age0) / _EXP242_AGING
        if _EXP242_AGING > 0 else float("inf"),
    }


def main():
    print("=" * 65)
    print("Exp 243 Birth-Balance Diagnostic (Exp-242 viable founder)")
    print(f"  speed={SPEED}, horizon={HORIZON}, regen_rate={REGEN_RATE}")
    print(f"  layout={LAYOUT}, seed={SEED}, hmax=0.0 (mortality OFF)")
    print(f"  founder: threshold={_EXP242_THRESHOLD}, cap={_EXP242_CAP}, "
          f"bmc={_EXP242_BMC}, mc={_EXP242_MC}, aging={_EXP242_AGING}")
    print("=" * 65)

    # Energy budget analysis
    print("\n--- Energy Budget Analysis (single-creature, full field) ---")
    budget = _energy_budget_analysis()
    print(f"  Start energy:              {budget['start_energy']:.1f}")
    print(f"  Reproduction threshold:    {budget['repro_threshold']:.1f}")
    print(f"  Max intake/step (full fld):{budget['max_intake_per_step']:.4f}")
    print(f"  Step cost at age 0:        {budget['step_cost_age0']:.3f}")
    print(f"  Aging cost/step:           {budget['aging_cost_per_step']:.4f}")
    print(f"  Break-even age:            {budget['break_even_age']:.1f} steps")
    print(f"  Max achievable energy:     {budget['max_achievable_energy']:.4f}")
    if budget['repro_age'] is not None:
        print(f"  Can reproduce at age:      {budget['repro_age']}")
    else:
        print(f"  Can reproduce:             NO — threshold NEVER reached")

    # Run the actual experiment
    print("\n--- Running birth-balance experiment (hmax=0.0, horizon=2000) ---")
    cfg = _build_birthbalance_config()
    t0 = time.perf_counter()
    eco = Ecology(cfg, seed=SEED)
    eco.run()
    elapsed = time.perf_counter() - t0
    print(f"  Run complete in {elapsed:.2f}s. exploded={eco.exploded}")

    # Reconstruct N(t)
    N_series = _reconstruct_N_per_step(eco.events, HORIZON)
    Narr = np.array(N_series, dtype=float)

    print(f"  N(0)={N_series[0]}, N_max={max(N_series)}, N_final={N_series[-1]}")

    # Count births
    births_total = sum(
        1 for e in eco.events
        if e.get("event_type") == "birth" and not e.get("details", {}).get("founder", False)
    )
    deaths_total = sum(1 for e in eco.events if e.get("event_type") == "death")
    death_causes = {}
    for e in eco.events:
        if e.get("event_type") == "death":
            cause = e.get("details", {}).get("cause", "unknown")
            death_causes[cause] = death_causes.get(cause, 0) + 1

    print(f"  Total births:    {births_total}")
    print(f"  Total deaths:    {deaths_total}  (causes: {death_causes})")

    # End-state resource availability (proxy)
    availability_mean = 0.0
    if eco.cont_world is not None:
        grid = eco.cont_world._resource
        cap = eco.cont_world.capacity
        n_cells_total = len(grid) * len(grid[0])
        total_avail = sum(
            float(grid[ri][ci]) / cap
            for ri in range(len(grid))
            for ci in range(len(grid[0]))
        )
        availability_mean = total_avail / n_cells_total if n_cells_total > 0 else 0.0
    print(f"  Field availability (end): {availability_mean:.3f}")

    # Determine b
    if births_total > 0:
        # Find a growth window where population was growing
        alive_period = sum(1 for n in N_series if n > 0)
        mean_N_alive = float(np.mean([n for n in N_series if n > 0])) if any(n > 0 for n in N_series) else 0.0
        b_percapita = births_total / max(1, alive_period) / max(1.0, mean_N_alive)
        b_measured = True
    else:
        b_percapita = 0.0
        b_measured = False

    # Birth-balance calculation
    if b_percapita > 0:
        n_eq_implied_kc60 = KC_DEFAULT * (b_percapita / HMAX_TARGET) ** (1.0 / THETA_TARGET)
        n_eq_guess = max(N_series) * 0.5
        kc_implied = n_eq_guess / (b_percapita / HMAX_TARGET) if b_percapita > 0 else float("nan")
    else:
        n_eq_implied_kc60 = float("nan")
        kc_implied = float("nan")

    # Recommendation
    if not b_measured or b_percapita == 0:
        recommendation = (
            "CRITICAL FINDING: b=0 — NO BIRTHS at default sweep params. "
            "Max achievable energy ({:.2f}) < repro threshold ({:.1f}). "
            "The sweep will produce EXTINCT populations; Kc calibration is moot. "
            "REQUIRES PARAMETER REDESIGN before Stage-1 sweep can produce valid data."
        ).format(budget["max_achievable_energy"], budget["repro_threshold"])
        can_calibrate_kc = False
    elif n_eq_implied_kc60 <= KC_DEFAULT * 0.8:
        recommendation = (
            f"KEEP Kc=60: implied N_eq={n_eq_implied_kc60:.1f} is well below Kc; "
            f"equilibrium is in the sub-saturation regime."
        )
        can_calibrate_kc = True
    elif n_eq_implied_kc60 <= KC_DEFAULT * 1.2:
        recommendation = (
            f"KEEP Kc=60 (marginal): implied N_eq={n_eq_implied_kc60:.1f} ≈ Kc; "
            f"hazard saturates near equilibrium."
        )
        can_calibrate_kc = True
    else:
        kc_adj = round(n_eq_implied_kc60 * 1.5)
        recommendation = (
            f"ADJUST Kc to ~{kc_adj}: implied N_eq={n_eq_implied_kc60:.1f} exceeds Kc={KC_DEFAULT}."
        )
        can_calibrate_kc = True

    # Print results
    print()
    print("=" * 65)
    print("BIRTH-BALANCE RESULTS")
    print("=" * 65)
    print(f"  Total births observed:  {births_total}")
    print(f"  Per-capita b:           {b_percapita:.5f} births/creature/step")
    print(f"  Field availability:     {availability_mean:.3f} (end-state proxy)")
    print()
    if not b_measured or b_percapita == 0:
        print(f"  Birth-balance calc: CANNOT PROCEED (b=0)")
        print(f"  Implied N_eq at Kc=60: nan")
        print()
        print(f"  ENERGY BUDGET SUMMARY:")
        print(f"    Max intake/step (full field, single creature): {budget['max_intake_per_step']:.4f}")
        print(f"    Step cost at age 0:    {budget['step_cost_age0']:.3f}")
        print(f"    Break-even age:        {budget['break_even_age']:.1f}  (intake=cost; after this, N drain)")
        print(f"    Max achievable energy: {budget['max_achievable_energy']:.4f}")
        print(f"    Repro threshold:       {budget['repro_threshold']:.1f}")
        print(f"    GAP:                   {budget['repro_threshold'] - budget['max_achievable_energy']:.4f}  (>0 means UNREACHABLE)")
    else:
        print(f"  Births found.")
        print(f"  With hmax={HMAX_TARGET}, theta={THETA_TARGET}, Kc={KC_DEFAULT}:")
        print(f"    => implied N_eq at balance: {n_eq_implied_kc60:.1f}")
        print(f"  Observed N_max: {max(N_series)}")
        print(f"  Implied Kc for N_eq=50%*N_max: {kc_implied:.1f}")
    print()
    print(f"  RECOMMENDATION: {recommendation}")
    print("=" * 65)

    # Write output file
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp243_birthbalance.txt"

    summary = f"""Exp 243 Birth-Balance Diagnostic
=================================
Parameters:
  speed={SPEED}, horizon={HORIZON}, regen_rate={REGEN_RATE}
  layout={LAYOUT}, seed={SEED}, hmax=0.0 (mortality OFF)
  hmax_target={HMAX_TARGET}, theta_target={THETA_TARGET}, Kc_default={KC_DEFAULT}

Energy Budget Analysis (single creature, full field):
  Start energy:              {budget['start_energy']:.1f}
  Reproduction threshold:    {budget['repro_threshold']:.1f}
  Max intake/step (full fld):{budget['max_intake_per_step']:.4f}
  Step cost at age 0:        {budget['step_cost_age0']:.3f}
  Aging cost/step:           {budget['aging_cost_per_step']:.4f}
  Break-even age:            {budget['break_even_age']:.1f} steps
  Max achievable energy:     {budget['max_achievable_energy']:.4f}
  Can reproduce:             {"YES at age " + str(budget['repro_age']) if budget['repro_age'] is not None else "NO — threshold NEVER reached"}

Empirical run results (hmax=0.0, horizon={HORIZON}, seed={SEED}):
  N(0)={N_series[0]}, N_max={max(N_series)}, N_final={N_series[-1]}
  Total births:    {births_total}
  Total deaths:    {deaths_total}  (causes: {death_causes})
  Field availability (end): {availability_mean:.3f}
  Per-capita b:    {b_percapita:.5f}

Birth-balance computation:
  hmax*(N_eq/Kc)^theta = b  =>  N_eq = Kc*(b/hmax)^(1/theta)
  With hmax={HMAX_TARGET}, theta={THETA_TARGET}, Kc={KC_DEFAULT}:
    b = {b_percapita:.5f}
    Implied N_eq at Kc=60: {n_eq_implied_kc60}
    Implied Kc for N_eq=50%*Nmax: {kc_implied}

RECOMMENDATION: {recommendation}

CRITICAL CONTEXT (FIX APPLIED):
  This diagnostic now uses the Exp-242 viability-calibrated continuous founder:
    threshold={_EXP242_THRESHOLD}, cap={_EXP242_CAP}, bmc={_EXP242_BMC}, mc={_EXP242_MC},
    aging={_EXP242_AGING}, transfer={_EXP242_TRANSFER}, cost_frac={_EXP242_COST_FRAC}
    regen_rate={REGEN_RATE}, continuous_capacity={_EXP242_CONTINUOUS_CAPACITY}
  The original NON-VIABLE discrete founder (threshold=17.0, cap=20.0, bmc=0.5, mc=0.3,
  aging=0.02) could not reproduce in the continuous substrate (~0.14/step intake vs
  ~0.8/step cost => starved in 20-50 steps). The Exp-242 calibrated founder uses
  a lower threshold (4.2 vs 17.0) and much lower costs (bmc=0.05 vs 0.5, mc=0.03 vs 0.3)
  that are commensurate with the continuous intake scale.
  Start energy: {budget['start_energy']:.1f} | threshold: {budget['repro_threshold']:.1f}
  Can reproduce: {"YES at age " + str(budget['repro_age']) if budget['repro_age'] is not None else "NO — threshold NEVER reached"}
"""
    out_path.write_text(summary)
    print(f"\nSummary written to: {out_path}")


if __name__ == "__main__":
    main()
