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
from ecology.genotype import founder
from ecology.evolvability.stability import n_eq as _n_eq
from ecology.continuous_world import ARENA_W, ARENA_H, _BUMP_CENTERS_BUMP


def _founder_with_speed(speed: float):
    """Return a base founder genotype with locomotor_speed set to `speed`."""
    return D.replace(founder(), locomotor_speed=speed)


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
) -> EcologyConfig:
    """Build the Exp 243 monomorphic continuous config via dataclasses.replace."""
    base = SCENARIOS["balanced"]
    return D.replace(
        base,
        # Locomotion + depletion intake (Exp 238 + 242)
        enable_continuous_locomotion=True,
        continuous_layout=layout,
        continuous_dt=1.0,
        speed_cost_floor=0.0,
        speed_cost_slope=0.0,
        continuous_regen_rate=regen_rate,
        # Depletion-aware intake (Exp 242)
        enable_continuous_depletion_intake=True,
        # Floored regen (Exp 243 Mechanism B)
        continuous_floored_regen=True,
        # Density mortality (Exp 243 Mechanism A)
        enable_density_mortality=True,
        density_mortality_hmax=hmax,
        density_mortality_Kc=Kc,
        density_mortality_theta=theta,
        density_mortality_rate_scale=rate_scale,
        # Freeze locomotion (monomorphic: all founders + offspring at speed)
        freeze_continuous_locomotion=True,
        # Founder with fixed speed
        founder=_founder_with_speed(speed),
        # Run params
        horizon=horizon,
        mutation_rate=0.0,  # monomorphic — no mutation
    )


def _reconstruct_N_per_step(events: list, horizon: int) -> list[int]:
    """Reconstruct per-step alive count via event-replay.

    Strategy: scan events in order. Birth events (event_type=="birth") add +1 to
    the running alive count at their timestep `t`. Death events (event_type=="death")
    subtract 1. The resource_tick and reproduction events do not change alive count.

    Returns a list of length `horizon` where index i is the alive count AT THE END
    of step i (after births and deaths in step i are applied).

    Birth events include both founder births (t=0, details.founder=True) and offspring
    births. The engine initialises founders in __init__ (t=0 events, before any step),
    so the initial count is derived from those t=0 births, then each step's births and
    deaths advance it.

    NOTE: events_hash is NOT touched — this is purely a post-hoc read of eco.events.
    """
    # Build a per-timestep delta: births = +1, deaths = -1
    birth_delta: dict[int, int] = {}
    death_delta: dict[int, int] = {}

    for ev in events:
        et = ev.get("event_type")
        t = ev.get("t", 0)
        if et == "birth":
            birth_delta[t] = birth_delta.get(t, 0) + 1
        elif et == "death":
            death_delta[t] = death_delta.get(t, 0) + 1

    # Walk through steps 0..horizon-1 accumulating alive count.
    # Founder births happen at t=0 (they are pre-step events in __init__).
    # The engine's step() increments self.t at the END of each step, so events at t=k
    # are from step k. After step k finishes, the alive count includes those t=k
    # births and deaths. We snapshot AFTER each step.
    #
    # The initial alive count before any step() calls = founder births at t=0.
    # After step 0 completes: add births at t=0, subtract deaths at t=0 from STEP
    # (but founders were added before step 0 — we need to separate them).
    #
    # Cleaner approach: replay cumulatively step by step.
    # At the end of step t (0-indexed), N[t] = sum of all births so far - sum of all deaths so far.
    alive = 0
    # Add founder births at t=0 (before the first step runs)
    alive += birth_delta.pop(0, 0)

    N_series: list[int] = []
    for step_t in range(horizon):
        # Births and deaths emitted during step `step_t` (engine uses self.t = step_t
        # at the start of the step, then increments at the end)
        # Non-founder births at t=step_t are offspring born during that step.
        alive += birth_delta.get(step_t, 0)
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
        _density_mortality_p(float(n), hmax, Kc, theta, rate_scale)
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
    _INTERBUMP_RADIUS = 1.5 * 1.5  # 1.5 * _BUMP_SIGMA; squared for efficiency
    if layout == "bump":
        bump_centers = _BUMP_CENTERS_BUMP
    elif layout == "neutral":
        from ecology.continuous_world import _BUMP_CENTERS_NEUTRAL
        bump_centers = _BUMP_CENTERS_NEUTRAL
    else:
        # flat layout: no bumps; all creatures are "off-bump" — flux = 1.0 proxy
        bump_centers = ()

    interbump_count = 0
    for c in alive_creatures:
        if c.phenotype.pos_cont is not None:
            x, y = c.phenotype.pos_cont
            if not bump_centers:
                # flat layout: define interbump_flux=1.0 (no bump structure)
                interbump_count += 1
            else:
                min_dist_sq = min(
                    (x - cx) ** 2 + (y - cy) ** 2
                    for cx, cy in bump_centers
                )
                if min_dist_sq > _INTERBUMP_RADIUS:
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
