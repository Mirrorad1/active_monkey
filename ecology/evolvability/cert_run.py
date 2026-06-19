"""Exp 243: monomorphic certification run wrapper + telemetry.

`run_cert` builds an EcologyConfig for a MONOMORPHIC clamped-speed continuous run
with the Exp 243 mechanisms ON, runs it for `horizon` steps, and returns the
analysis-window telemetry dict consumed by `ecology.evolvability.stability.certify_run`.

Telemetry extraction strategy (byte-identity constraints preserved):
- N(t): RECONSTRUCTED from `eco.events` via event-replay (births/deaths per step).
  No engine recorder added; events_hash is untouched.
- births_per_step / crowding_per_step: from event stream over the analysis window.
- p_hazard_mean: computed analytically from N(t) and density-mortality params via
  `_density_mortality_p` (no engine change required).
- availability_mean: full-grid mean of `eco.cont_world._resource / capacity` at the
  END of the run (post-hoc read; a snapshot, not a per-step average).
  PROXY DISCLOSURE: this is an end-state snapshot, not a time-average over the window.
  It underestimates temporal variation but is the honest best available without engine
  changes that would alter events_hash.
- boundary_frac: fraction of ALIVE creatures whose `pos_cont` is within 1.0 arena unit
  of the arena edge, measured at the END of the run.
  PROXY DISCLOSURE: end-state snapshot only (per-step tracking would require engine
  instrumentation that touches the event stream). Documented as a proxy.
- interbump_flux: fraction of alive creatures NOT within 1.5-sigma of the nearest bump
  center, measured at the end of the run. A creature far from every bump center has
  moved through inter-bump space, suggesting population spread.
  PROXY DISCLOSURE: end-state spatial proxy; does not track trajectory history.
  True inter-bump mixing requires per-step position logging (engine change). This
  measure is non-zero iff some creatures are in inter-bump regions at run end.
- n_eq: median of N over the analysis window (via `stability.n_eq`).
- exploded: directly from `eco.exploded` (O(1), no engine change).
"""
from __future__ import annotations

import dataclasses as D
import math

import numpy as np

from ecology.engine import Ecology, EcologyConfig, _density_mortality_p
from ecology.scenarios import SCENARIOS
from ecology.genotype import founder as _base_founder
from ecology.evolvability.stability import n_eq as _n_eq
from ecology.continuous_world import ARENA_W, ARENA_H, _BUMP_CENTERS_BUMP


# ---------------------------------------------------------------------------
# Exp-242 viability-calibrated continuous founder parameters
# (extracted from experiments/exp242_regulated_ess.py, lines 74–80)
# These replace the discrete "balanced" founder (threshold=17, cap=20, bmc=0.5, mc=0.3)
# which was NON-VIABLE on the continuous intake scale (~0.14/step intake vs ~0.8/step cost).
# ---------------------------------------------------------------------------
_EXP242_BMC: float = 0.05        # baseline_metabolic_cost  (line 74)
_EXP242_MC: float = 0.03         # movement_cost            (line 75)
_EXP242_CAP: float = 10.0        # energy_capacity          (line 76)
_EXP242_THRESHOLD: float = 4.2   # reproduction_energy_threshold  (line 77)
_EXP242_TRANSFER: float = 0.35   # reproduction_energy_transfer_fraction (line 78)
_EXP242_COST_FRAC: float = 0.06  # reproduction_cost_fraction  (line 79)
_EXP242_AGING: float = 0.003     # aging_cost               (line 80)
_EXP242_CONTINUOUS_CAPACITY: float = 2.0   # continuous_capacity (line 87)


def _founder_with_speed(speed: float):
    """Return the Exp-242 viability-calibrated continuous founder with locomotor_speed=speed.

    Uses the Exp-242 viable continuous parameters (threshold=4.2, cap=10.0, bmc=0.05,
    mc=0.03, aging=0.003) rather than the discrete 'balanced' founder (threshold=17,
    cap=20, bmc=0.5, mc=0.3), which starved in ~20-50 steps on the continuous substrate.
    """
    return D.replace(
        _base_founder(),
        baseline_metabolic_cost=_EXP242_BMC,
        movement_cost=_EXP242_MC,
        energy_capacity=_EXP242_CAP,
        reproduction_energy_threshold=_EXP242_THRESHOLD,
        reproduction_energy_transfer_fraction=_EXP242_TRANSFER,
        reproduction_cost_fraction=_EXP242_COST_FRAC,
        aging_cost=_EXP242_AGING,
        locomotor_speed=speed,
    )


def _build_config(
    speed: float,
    *,
    hmax: float,
    Kc: float,
    theta: float,
    regen_rate: float,
    rate_scale: float,
    layout: str,
    horizon: int,
    speed_cost_slope: float = 0.6,
    bump_sigma: float = 1.5,
    bump_amplitude: float = 1.0,
    bump_centers: tuple | None = None,
    moving_patch: bool = False,
    patch_sigma: float = 0.8,
    patch_amplitude: float = 3.0,
    patch_orbit_radius: float = 3.0,
    patch_period: int = 300,
) -> EcologyConfig:
    """Build the Exp 243 monomorphic continuous config via dataclasses.replace.

    Uses the Exp-242 viability-calibrated continuous founder and substrate params
    (continuous_capacity=2.0, min_survival_energy=0.5, max_population=4000,
    initial_population=21, continuous_logistic_regen=False) that produced viable
    reproducing populations in Exp 242. Layers Exp 243 mechanisms A+B on top.
    """
    base = SCENARIOS["balanced"]
    return D.replace(
        base,
        # Locomotion + depletion intake (Exp 238 + 242)
        enable_continuous_locomotion=True,
        continuous_layout=layout,
        continuous_dt=1.0,
        speed_cost_floor=0.0,
        speed_cost_slope=speed_cost_slope,   # configurable; default=0.6 (Exp-242 baseline)
        continuous_regen_rate=regen_rate,
        continuous_capacity=_EXP242_CONTINUOUS_CAPACITY,
        # Depletion-aware intake (Exp 242) — fixes the density-feedback bug
        enable_continuous_depletion_intake=True,
        continuous_logistic_regen=False,   # Exp-242: NOT logistic; floored regen replaces
        # Floored regen (Exp 243 Mechanism B) — replaces Exp-242's logistic regen
        continuous_floored_regen=True,
        # Density mortality (Exp 243 Mechanism A)
        enable_density_mortality=True,
        density_mortality_hmax=hmax,
        density_mortality_Kc=Kc,
        density_mortality_theta=theta,
        density_mortality_rate_scale=rate_scale,
        # Freeze locomotion (monomorphic: all founders + offspring at speed)
        freeze_continuous_locomotion=True,
        # Viable founder with fixed speed (Exp-242 calibrated)
        founder=_founder_with_speed(speed),
        # Viable substrate params from Exp-242 (exp242_regulated_ess.py line 97-173)
        initial_population=21,
        max_population=4000,
        min_survival_energy=0.5,
        mutation_rate=0.0,  # monomorphic — no mutation
        # Run params
        horizon=horizon,
        # Exp 245: configurable bump geometry (defaults byte-identical to Exp 238-244).
        continuous_bump_sigma=bump_sigma,
        continuous_bump_amplitude=bump_amplitude,
        continuous_bump_centers=bump_centers,
        # Exp 246: moving-patch resource mode (defaults byte-identical to Exp 238-245).
        continuous_moving_patch=moving_patch,
        continuous_patch_sigma=patch_sigma,
        continuous_patch_amplitude=patch_amplitude,
        continuous_patch_orbit_radius=patch_orbit_radius,
        continuous_patch_period=patch_period,
    )


def _reconstruct_N_per_step(events: list, horizon: int) -> list[int]:
    """Reconstruct per-step alive count via event-replay.

    Engine founder/offspring representation (verified from ecology/engine.py):
    - FOUNDERS: emitted as "birth" events at t=0 (before any step() call) with
      details={"founder": True}. They are added in __init__ before the run loop.
    - OFFSPRING: emitted as "birth" events with details={"founder": False, "parent_id": ...}
      at whatever t the engine's self.t holds when _step_one_creature runs.
      Step-0 offspring have t=0 AND founder=False.
    - DEATHS: emitted as "death" events at the t when they occur in _step_one_creature.

    Strategy: seed alive count from FOUNDER-FLAGGED births (t=0, founder=True) before
    the per-step loop. Then in each step, count non-founder births at that step's t
    (offspring, including genuine step-0 offspring) as +1, and deaths at that t as -1.
    This correctly handles step-0 offspring that share t=0 with founders.

    Returns a list of length `horizon` where index i is the alive count AT THE END
    of step i (after births and deaths in step i are applied).

    NOTE: events_hash is NOT touched — this is purely a post-hoc read of eco.events.
    """
    # Separate founders from offspring births; build per-step deltas.
    n_founders: int = 0
    offspring_delta: dict[int, int] = {}
    death_delta: dict[int, int] = {}

    for ev in events:
        et = ev.get("event_type")
        t = ev.get("t", 0)
        if et == "birth":
            if ev.get("details", {}).get("founder", False):
                # Founder: added before the run loop; counts toward initial alive.
                n_founders += 1
            else:
                # Offspring (genuine birth during a step, including step 0).
                offspring_delta[t] = offspring_delta.get(t, 0) + 1
        elif et == "death":
            death_delta[t] = death_delta.get(t, 0) + 1

    # Walk steps 0..horizon-1. Initial alive = founder count (pre-step, from __init__).
    # At each step t, add offspring born during that step and subtract deaths.
    alive = n_founders
    N_series: list[int] = []
    for step_t in range(horizon):
        # Offspring born during this step (includes genuine step-0 offspring).
        alive += offspring_delta.get(step_t, 0)
        # Deaths during this step.
        alive -= death_delta.get(step_t, 0)
        alive = max(0, alive)
        N_series.append(alive)

    return N_series


def run_cert(
    speed: float,
    *,
    hmax: float,
    Kc: float,
    theta: float,
    regen_rate: float,
    rate_scale: float,
    layout: str,
    seed: int,
    horizon: int = 4000,
    burn_in: float = 0.6,
    speed_cost_slope: float = 0.6,
    bump_sigma: float = 1.5,
    bump_amplitude: float = 1.0,
    bump_centers: tuple | None = None,
    moving_patch: bool = False,
    patch_sigma: float = 0.8,
    patch_amplitude: float = 3.0,
    patch_orbit_radius: float = 3.0,
    patch_period: int = 300,
) -> dict:
    """Run a monomorphic certification run and return the analysis-window telemetry dict.

    Parameters
    ----------
    speed : float
        Fixed locomotor_speed for all creatures (monomorphic — freeze_continuous_locomotion=True).
    hmax, Kc, theta : float
        Density-mortality parameters for Mechanism A.
    regen_rate : float
        Per-step regen rate for the continuous resource sub-cells.
    rate_scale : float
        Rate-derivative brake for density mortality (0.0 = off).
    layout : str
        Continuous world layout ("bump", "flat", "neutral").
    seed : int
        RNG seed.
    horizon : int
        Number of steps to run.
    burn_in : float
        Fraction of horizon to discard as burn-in (default 0.6).
    speed_cost_slope : float
        Linear energy cost per unit speed (default=0.6, the Exp-242 baseline).
        Pass lower values (e.g. 0.05–0.2) to search for viable slow/medium-speed regimes.
    bump_sigma : float
        Gaussian bump width (default=1.5, byte-identical to Exp 238-244). Lower → sharper.
    bump_amplitude : float
        Bump peak density (default=1.0, byte-identical to Exp 238-244).
    bump_centers : tuple | None
        Explicit bump center tuple for the "bump" layout (None → legacy default).
    moving_patch : bool
        Exp 246: enable single drifting Gaussian resource mode (default=False, byte-identical).
    patch_sigma : float
        Exp 246: Gaussian half-width of the moving patch (default=0.8).
    patch_amplitude : float
        Exp 246: Peak density at the patch center (default=3.0).
    patch_orbit_radius : float
        Exp 246: Orbit radius around the arena center (default=3.0).
    patch_period : int
        Exp 246: Steps per full orbit (default=300).

    Returns
    -------
    dict with keys:
        N                  : np.ndarray — per-step alive count over the analysis window
        births_per_step    : float
        crowding_per_step  : float
        p_hazard_mean      : float
        exploded           : bool
        availability_mean  : float — PROXY (end-state snapshot; see module docstring)
        boundary_frac      : float — PROXY (end-state snapshot; see module docstring)
        interbump_flux     : float — PROXY (end-state spatial measure; see module docstring)
        n_eq               : float
    """
    cfg = _build_config(
        speed,
        hmax=hmax,
        Kc=Kc,
        theta=theta,
        regen_rate=regen_rate,
        rate_scale=rate_scale,
        layout=layout,
        horizon=horizon,
        speed_cost_slope=speed_cost_slope,
        bump_sigma=bump_sigma,
        bump_amplitude=bump_amplitude,
        bump_centers=bump_centers,
        moving_patch=moving_patch,
        patch_sigma=patch_sigma,
        patch_amplitude=patch_amplitude,
        patch_orbit_radius=patch_orbit_radius,
        patch_period=patch_period,
    )

    eco = Ecology(cfg, seed=seed)
    eco.run()

    # --- Analysis window ---
    win_start = int(horizon * burn_in)
    win_len = horizon - win_start  # = int(horizon * (1 - burn_in))

    # --- N(t) via event-replay (no engine change, events_hash untouched) ---
    N_full = _reconstruct_N_per_step(eco.events, horizon)
    # If the run stopped early (extinction or explosion), pad the remainder with 0
    if len(N_full) < horizon:
        N_full = N_full + [0] * (horizon - len(N_full))
    N_window = np.array(N_full[win_start : win_start + win_len], dtype=float)

    # --- births_per_step and crowding_per_step over analysis window ---
    births_in_window = 0
    crowding_deaths_in_window = 0
    for ev in eco.events:
        t = ev.get("t", 0)
        if t < win_start or t >= win_start + win_len:
            continue
        et = ev.get("event_type")
        if et == "birth" and not ev.get("details", {}).get("founder", False):
            births_in_window += 1
        elif et == "death":
            cause = ev.get("details", {}).get("cause", "")
            if cause == "crowding":
                crowding_deaths_in_window += 1

    births_per_step = births_in_window / win_len if win_len > 0 else 0.0
    crowding_per_step = crowding_deaths_in_window / win_len if win_len > 0 else 0.0

    # --- p_hazard_mean: mean realized crowding hazard over the analysis window ---
    # Computed analytically from N(t) using the canonical engine helper.
    # No engine change needed.
    p_hazard_vals = [
        _density_mortality_p(float(n), hmax, Kc, theta, rate_scale)  # NOTE: dN=0 proxy; underestimates p_hazard when rate_scale>0 (the lead term is omitted).
        for n in N_window
    ]
    p_hazard_mean = float(np.mean(p_hazard_vals)) if p_hazard_vals else 0.0

    # --- availability_mean: mean resource availability (END-STATE PROXY) ---
    # PROXY: measured at the end of the run, not time-averaged over the window.
    # A full-grid mean of (resource_cell / capacity) across all 24x24 sub-cells.
    # Honest approximation: captures the equilibrium depletion state but not temporal
    # variation. Documents as proxy per task specification.
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

    # --- boundary_frac: fraction of alive creatures near arena boundary (END-STATE PROXY) ---
    # PROXY: measured at run end only (per-step tracking would require engine instrumentation).
    # Boundary = within 1.0 arena unit of any edge (ARENA_W=12, ARENA_H=12).
    _BOUNDARY_MARGIN = 1.0
    alive_creatures = [c for c in eco._creatures if c.phenotype.alive]
    boundary_count = 0
    total_alive = len(alive_creatures)
    for c in alive_creatures:
        if c.phenotype.pos_cont is not None:
            x, y = c.phenotype.pos_cont
            if (x < _BOUNDARY_MARGIN or x > ARENA_W - _BOUNDARY_MARGIN or
                    y < _BOUNDARY_MARGIN or y > ARENA_H - _BOUNDARY_MARGIN):
                boundary_count += 1
    boundary_frac = boundary_count / total_alive if total_alive > 0 else 0.0

    # --- interbump_flux: end-state spatial spread proxy (END-STATE PROXY) ---
    # PROXY: fraction of alive creatures NOT within 1.5*_BUMP_SIGMA of the nearest
    # bump center (i.e. in inter-bump space). This is a spatial spread measure rather
    # than a true trajectory-history mixing measure. A non-zero value indicates some
    # creatures are in inter-bump regions at run end, suggesting population spread.
    # TRUE inter-bump mixing (fraction of creatures visiting >1 bump basin over time)
    # would require per-step position logging — an engine change that would alter the
    # events_hash. This proxy is honest and non-degenerate for the "static spatial
    # mosaic" non-degeneracy check (interbump_flux > 0 iff creatures are found off-bump).
    #
    # For "bump" layout, bump centers are in _BUMP_CENTERS_BUMP with sigma=1.5.
    # For other layouts, we use a conservative center of the arena as proxy.
    _INTERBUMP_RADIUS_SQ = (1.5 * bump_sigma) ** 2  # radius = 1.5*sigma, squared (Exp 245: uses configurable sigma)
    if layout == "bump":
        # Exp 245: use configured bump_centers if supplied, otherwise legacy default.
        interbump_centers = bump_centers if bump_centers is not None else _BUMP_CENTERS_BUMP
    elif layout == "neutral":
        from ecology.continuous_world import _BUMP_CENTERS_NEUTRAL
        interbump_centers = _BUMP_CENTERS_NEUTRAL
    else:
        # flat layout: no bumps; all creatures are "off-bump" — flux = 1.0 proxy
        interbump_centers = ()

    interbump_count = 0
    for c in alive_creatures:
        if c.phenotype.pos_cont is not None:
            x, y = c.phenotype.pos_cont
            if not interbump_centers:
                # flat layout: define interbump_flux=1.0 (no bump structure)
                interbump_count += 1
            else:
                min_dist_sq = min(
                    (x - cx) ** 2 + (y - cy) ** 2
                    for cx, cy in interbump_centers
                )
                if min_dist_sq > _INTERBUMP_RADIUS_SQ:
                    interbump_count += 1

    interbump_flux = interbump_count / total_alive if total_alive > 0 else 0.0

    # --- n_eq: median of N over the analysis window ---
    n_eq_val = float(_n_eq(N_window))

    return {
        "N": N_window,
        "births_per_step": births_per_step,
        "crowding_per_step": crowding_per_step,
        "p_hazard_mean": p_hazard_mean,
        "exploded": eco.exploded,
        "availability_mean": availability_mean,
        "boundary_frac": boundary_frac,
        "interbump_flux": interbump_flux,
        "n_eq": n_eq_val,
    }
