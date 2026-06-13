"""ecology/sense_axis.py — a reusable SENSE-EVOLUTION diagnostic interface.

Phase 1 of the sense-evolution sub-arc (human steer 2026-06-13, pre-registered in
loop/directions/population-ecology.md). This module treats a *sense* as a generic
axis, not as thermosense-specific hardcoding, so the same machinery later maps to
sight, hearing, localization, and communication.

THE GENERIC SENSE AXIS
----------------------
A sense is the tuple
    h      sensory investment / precision of creature i        (a genotype trait)
    C_h(h) direct upkeep cost of maintaining/using the sensor  (floored, never free)
    z      hidden environmental state                          (band centre, niche, residue)
    s ~ P(s|z,h)  the observation (noise shrinks with h)
    theta  policy/controller ability to USE the sensor         (a genotype trait; reserved)
    a = pi(s, theta)  the action chosen from the observation
    rho    ecological state (density, depletion, residue, competitors, niche occupancy)
    w      realized reproductive fitness = B(a,z,h,theta,rho) - C_h(h) - C_theta(theta)

and the diagnostics it exposes are B(h) (installed benefit), C(h) (cost), the LOCAL
slope dB/dh - dC/dh near the resident, and the REALIZED selection gradient
    g(h) = dE[w|h] / dh
measured from a live population with cost ON — NOT the endpoint installed advantage.

THE META-PRINCIPLE
------------------
    A sense is evolvable when it exposes private actionable information whose
    marginal value remains positive across small heritable improvements:
        dB/dh > dC/dh + drift/noise/transmission-erosion   near the resident.

THERMOSENSE NOW -> OTHER SENSES LATER (the same fields; Phase 6)
    thermosense : z = fresh/depleted food, drifting band, residue, niche temperature
                  h = thermal precision; false positives = residue read as food;
                  affordance = fresh-food discrimination, niche access, less search waste.
    sight       : z = object identity / distance / occlusion / predator-prey-food class
                  h = spatial/angular resolution, field of view, channel discrimination;
                  false positives = decoys, shadows, occluded objects;
                  affordance = long-range planning, classification, obstacle avoidance.
    hearing     : z = source location, frequency signature, social signal, movement
                  h = frequency/temporal resolution, localization precision;
                  false positives = echo/noise read as signal;
                  affordance = localization, coordination, threat detection, mate finding.
    communication: z = agent intention, local resource info, danger, role, identity
                  h = channel bandwidth, symbol discrimination, signal reliability;
                  false positives = misdecoded messages;
                  affordance = coordination, division of labour, teaching, recruitment.

ANTI-CHEAT (binding; extends the no-hidden-evaluator invariant)
--------------------------------------------------------------
NOTHING here writes fitness or food as f(h). The clamp founders ONLY set a genotype
trait; survival/reproduction still read solely each creature's own state + local cell.
The cost is the ORDINARY thermosense upkeep (floor + inefficiency*h), charged by the
unmodified engine. `assert_no_direct_h_reward` documents the property; the realized
gradient is read from births the creatures actually achieve in the world.
"""
from __future__ import annotations

import dataclasses as D
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Hashable

import numpy as np

from ecology.engine import Ecology, EcologyConfig
from ecology.genotype import Genotype


# The clamp grid the audit walks (the human's pre-registered values).
CLAMP_GRID: tuple[float, ...] = (0.00, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60)

# The resident / founder value the local gradient is read AT.
RESIDENT_H: float = 0.10

# Neighbours used for the central-difference slope at each anchor (lo, hi) so the
# slope is (w[hi]-w[lo])/(hi-lo).  Anchors: 0.03, the 0.10 resident, 0.20.
SLOPE_ANCHORS: dict[str, tuple[float, float]] = {
    "h0.03": (0.00, 0.06),
    "h0.10": (0.06, 0.15),   # the resident slope — the decisive reading
    "h0.20": (0.15, 0.30),
}


@D.dataclass(frozen=True)
class SenseAxisSpec:
    """Which genotype fields play the (h, theta, inefficiency) roles for one sense.

    Thermosense is the only instance today; the fields exist so a future sense
    (sight/hearing/communication) is a different SenseAxisSpec, not a fork of the
    diagnostics.  `cost_floor`/`cost_slope` mirror the engine's thermosense upkeep
    (floor + inefficiency*h) so C(h) can be reported analytically alongside B(h).
    """
    name: str
    h_trait: str = "thermosense_intensity"
    inefficiency_trait: str = "thermosense_inefficiency"
    controller_trait: str | None = None        # theta — reserved (Exp 206)
    cost_floor: float = 0.0
    cost_inefficiency: float = 0.20

    def cost(self, h: float) -> float:
        """Analytic upkeep C_h(h) of a clamped sensor (matches engine when active)."""
        if h <= 0.05:        # below the active threshold the organ is off -> no upkeep
            return 0.0
        return self.cost_floor + self.cost_inefficiency * h


THERMOSENSE_AXIS = SenseAxisSpec(name="thermosense")


# ---------------------------------------------------------------------------
# Clamp founders + common-garden seeding (founder_mix builders)
# ---------------------------------------------------------------------------

def clamp_founder(base: Genotype, h: float, inefficiency: float = 0.20) -> Genotype:
    """A founder identical to `base` except the sensor is clamped to intensity h.

    Combined with cfg.freeze_thermosense=True the value BREEDS TRUE, so every
    descendant carries exactly this h — the unit of a realized-fitness-at-fixed-h
    measurement.  inefficiency fixes the upkeep slope (cost = floor + inefficiency*h).
    """
    return D.replace(base, thermosense_intensity=float(h), thermosense_inefficiency=float(inefficiency))


def founder_mix_equal(base: Genotype, grid: tuple[float, ...] = CLAMP_GRID,
                      per_clamp: int = 16, inefficiency: float = 0.20
                      ) -> tuple[tuple[Genotype, int], ...]:
    """Equal cohorts across the clamp grid in ONE shared world (the common garden).

    Founders are EXPANDED in interleaved order by the engine's spread placement, but
    we additionally INTERLEAVE here (round-robin) so no clamp value is concentrated in
    one grid region — neutralising any starting-position-quality x clamp correlation.
    """
    # Round-robin interleave: [(h0,1),(h1,1),...,(hk,1),(h0,1),...] flattened by the
    # engine into spread positions, so clamp identity is decorrelated from start pos.
    seq: list[tuple[Genotype, int]] = []
    for _ in range(per_clamp):
        for h in grid:
            seq.append((clamp_founder(base, h, inefficiency), 1))
    return tuple(seq)


def founder_mix_resident(base: Genotype, grid: tuple[float, ...] = CLAMP_GRID,
                         resident_h: float = RESIDENT_H, resident_count: int = 48,
                         invader_count: int = 6, inefficiency: float = 0.20
                         ) -> tuple[tuple[Genotype, int], ...]:
    """Resident-dominant common garden = a rare-invader INVASION assay.

    The resident (h=0.10) is the dominant background; every OTHER grid value enters
    as a rare invader cohort.  Measured early (resident still dominant) this reads the
    adaptive-dynamics invasion fitness near the resident — the gradient sign evolution
    actually faces (NOT the equal-cohort mixed background).  Interleaved placement.
    """
    seq: list[tuple[Genotype, int]] = []
    # interleave invaders, then sprinkle residents throughout
    invaders = [h for h in grid if abs(h - resident_h) > 1e-9]
    # round-robin: one resident then one invader-of-each, repeated, until counts met
    res_left = resident_count
    inv_left = {h: invader_count for h in invaders}
    while res_left > 0 or any(v > 0 for v in inv_left.values()):
        if res_left > 0:
            seq.append((clamp_founder(base, resident_h, inefficiency), 1)); res_left -= 1
        for h in invaders:
            if inv_left[h] > 0:
                seq.append((clamp_founder(base, h, inefficiency), 1)); inv_left[h] -= 1
    return tuple(seq)


def assert_no_direct_h_reward(cfg: EcologyConfig) -> None:
    """Anti-cheat guard: the audit must NOT smuggle a sensor reward.

    Verifies the only thing the clamp grid touches is the genotype trait + its
    ordinary upkeep — never a coupling that writes intake/fitness as f(h).  Concretely:
    food coupling must be the SAME engine path used by the (h-blind) evolution runs,
    and cost must be ON (enable_thermosense True) so the sensor is never free.
    """
    assert cfg.enable_thermosense, "audit verdict requires cost ON (enable_thermosense=True)"
    assert cfg.freeze_thermosense, "audit requires freeze_thermosense=True (clamp breeds true)"
    # founder_mix only seeds genotypes; survival/reproduction read own-state only.
    assert cfg.founder_mix is not None, "common garden needs an explicit founder_mix"


# ---------------------------------------------------------------------------
# The realized-gradient measurement
# ---------------------------------------------------------------------------

def _bucket_h(h: float, grid: tuple[float, ...]) -> float | None:
    """Snap an exact (breeds-true) intensity to its grid value, else None (founder drift guard)."""
    for g in grid:
        if abs(h - g) < 1e-6:
            return g
    return None


def run_gradient_audit(cfg: EcologyConfig, seed: int, *, grid: tuple[float, ...] = CLAMP_GRID,
                       window: tuple[int, int] = (100, 700),
                       checkpoint_stride: int = 50, min_clamp_pop: int = 8) -> dict[str, Any]:
    """Run ONE common-garden audit and return the per-clamp REALIZED fitness curve.

    PRIMARY readout — the Malthusian log-growth rate r(h) (fixation-robust): the OLS
    slope of ln(alive[h](t)) vs t over the EARLY window, while every clamp is still
    present.  r(h) IS the realized selection gradient's basis: the clamp whose
    sub-population grows fastest per capita is the one selection favours; the slope of
    r(h) at h=0.10 is g.  It survives the fixation the equal-cohort garden suffers
    (one clamp eventually sweeps) because it is read EARLY from the growth rate, not
    from post-fixation head-counts (the red-team fix).

    SECONDARY — per-capita reproduction rate over the window
        repro_rate[h] = (children born in-window carrying h) / (alive[h]-steps in-window).
    Births carry the parent's h exactly (freeze_thermosense breeds true).

    PER-CLAMP HEALTH GATE (red-team fix, kin of L21/L24): a clamp with fewer than
    `min_clamp_pop` alive at the window start is UNDER-REPRESENTED — its r(h)/repro_rate
    are marked NaN (NO_VERDICT for that clamp/seed), never a metric violation.  Whole-pop
    and per-clamp head-counts are returned so drift (high fitness only at collapse) is auditable.
    """
    assert_no_direct_h_reward(cfg)
    w_lo, w_hi = window
    # PLACEMENT DECORRELATION (red-team BLOCKER fix): the engine seeds founders at
    # deterministic stride positions in founder_mix order, so clamp identity correlates
    # with starting column -> temperature -> food-band proximity (clamps split into 3
    # mean-temperature groups by index mod 3 — which could fake a resident slope).
    # Shuffle the expanded founder list with a seed-derived rng so clamp<->position is
    # randomised FRESH per seed; averaging over seeds removes the positional bias.
    cfg = D.replace(cfg, founder_mix=_shuffle_founder_mix(cfg.founder_mix, seed))
    eco = Ecology(cfg, seed=seed)

    # Starting-temperature decorrelation diagnostic: per-clamp mean temperature of the
    # gen-0 founders (must be ~equal across clamps after the shuffle; reported, audited).
    start_temp: dict[float, list[float]] = {g: [] for g in grid}
    if eco.world.temperature is not None:
        for c in eco._creatures:               # gen-0 founders only exist at construction
            if c.generation == 0:
                b = _bucket_h(c.genotype.thermosense_intensity, grid)
                if b is not None:
                    start_temp[b].append(float(eco.world.temperature[c.phenotype.pos]))
    start_temp_mean = {g: (float(np.mean(v)) if v else float("nan")) for g, v in start_temp.items()}

    cps = list(range(checkpoint_stride, cfg.horizon + 1, checkpoint_stride))
    traj: list[dict[str, Any]] = []
    win_series: dict[float, list[tuple[int, int]]] = {g: [] for g in grid}  # (t, count) in window
    alive_steps: dict[float, float] = {g: 0.0 for g in grid}
    other_intensities = 0       # creatures whose h escaped the grid (must stay 0)

    while eco.t < cfg.horizon and not eco.exploded:
        eco.step()
        if eco.t in cps:
            alive = eco._alive()
            counts: dict[float, int] = {g: 0 for g in grid}
            for c in alive:
                b = _bucket_h(c.genotype.thermosense_intensity, grid)
                if b is None:
                    other_intensities += 1
                else:
                    counts[b] += 1
            if w_lo <= eco.t <= w_hi:
                for g in grid:
                    win_series[g].append((eco.t, counts[g]))
                    alive_steps[g] += counts[g] * checkpoint_stride
            traj.append({"t": eco.t, "pop": len(alive),
                         "counts": {f"{g:.2f}": counts[g] for g in grid}})
        if not eco._alive():
            break

    # births in-window per clamp (secondary readout) + the PER-INDIVIDUAL realized
    # fitness of window-born creatures (PRIMARY-2): E[offspring_count | h] over creatures
    # BORN IN THE WINDOW.  This is robust to the harsh founder cold-start mortality that
    # makes cohort growth-rate noisy — it is a per-individual mean over many births, and
    # the shared cold-start is differenced out by comparing to the resident clamp.  A
    # creature born in [w_lo,w_hi] has completed its life by the (much later) horizon.
    births_in_window: dict[float, int] = {g: 0 for g in grid}
    offspring_window: dict[float, list[int]] = {g: [] for g in grid}
    total_offspring: dict[float, list[int]] = {g: [] for g in grid}
    lifetime_food: dict[float, list[float]] = {g: [] for g in grid}
    age_at_death: dict[float, list[int]] = {g: [] for g in grid}
    for c in eco._creatures:
        b = _bucket_h(c.genotype.thermosense_intensity, grid)
        if b is None:
            continue
        if c.parent_id is not None and w_lo < c.phenotype.birth_t <= w_hi:
            births_in_window[b] += 1
            offspring_window[b].append(c.phenotype.offspring_count)
        total_offspring[b].append(c.phenotype.offspring_count)
        lifetime_food[b].append(c.phenotype.resource_eaten)
        if not c.phenotype.alive:
            age_at_death[b].append(c.phenotype.age)

    def _growth_rate(series: list[tuple[int, int]]) -> tuple[float, int]:
        """OLS slope of ln(count) vs t over checkpoints with count>=1; returns (r, start_count)."""
        start_count = series[0][1] if series else 0
        pts = [(t, c) for t, c in series if c >= 1]
        if len(pts) < 3:
            return float("nan"), start_count
        ts = np.array([t for t, _ in pts], dtype=float)
        ys = np.log(np.array([c for _, c in pts], dtype=float))
        A = np.vstack([ts, np.ones_like(ts)]).T
        slope = float(np.linalg.lstsq(A, ys, rcond=None)[0][0])
        return slope, start_count

    per_clamp: dict[float, dict[str, Any]] = {}
    for g in grid:
        r, start_count = _growth_rate(win_series[g])
        gated = start_count >= min_clamp_pop
        n_win = len(offspring_window[g])
        # the per-individual fitness is gated on a SAMPLE SIZE of window-born creatures
        win_ok = n_win >= min_clamp_pop
        a_steps = alive_steps[g]
        per_clamp[g] = {
            "w_offspring": float(np.mean(offspring_window[g])) if win_ok else float("nan"),  # PRIMARY
            "n_window": n_win,
            "win_ok": bool(win_ok),
            "r": r if gated else float("nan"),                 # SECONDARY (cohort log-growth)
            "repro_rate": (births_in_window[g] / a_steps) if (gated and a_steps > 0) else float("nan"),
            "window_start_count": start_count,
            "gated_ok": bool(gated),
            "births_in_window": births_in_window[g],
            "mean_offspring": float(np.mean(total_offspring[g])) if total_offspring[g] else float("nan"),
            "mean_lifetime_food": float(np.mean(lifetime_food[g])) if lifetime_food[g] else float("nan"),
            "mean_age_at_death": float(np.mean(age_at_death[g])) if age_at_death[g] else float("nan"),
            "n_ever": len(total_offspring[g]),
            "alive_end": traj[-1]["counts"][f"{g:.2f}"] if traj else 0,
        }

    alive_end = eco._alive()
    return {
        "seed": seed,
        "per_clamp": per_clamp,
        "start_temp_mean": start_temp_mean,        # decorrelation audit (≈equal across clamps)
        "trajectory": traj,
        "final_pop": len(alive_end),
        "extinct": len(alive_end) == 0,
        "exploded": eco.exploded,
        "steps_run": eco.t,
        "other_intensities": other_intensities,   # MUST be 0 (clamp integrity)
        "events_hash": eco.events_hash(),
    }


def _shuffle_founder_mix(founder_mix: tuple[tuple[Genotype, int], ...] | None, seed: int
                         ) -> tuple[tuple[Genotype, int], ...] | None:
    """Expand + shuffle a founder_mix to singletons with a seed-derived rng (placement
    decorrelation).  None passes through unchanged.  A SEPARATE rng (seed ^ const) is
    used so the engine's own per-seed stream is unaffected by this re-ordering choice."""
    if founder_mix is None:
        return None
    flat = [g for g, cnt in founder_mix for _ in range(int(cnt))]
    rng = np.random.default_rng(seed ^ 0x5E_17_AC)
    order = rng.permutation(len(flat))
    return tuple((flat[i], 1) for i in order)


def run_intrinsic_growth(cfg: EcologyConfig, h: float, seed: int, *,
                         window: tuple[int, int] = (100, 700), stride: int = 50,
                         inefficiency: float = 0.20) -> float:
    """Monomorphic intrinsic (density-independent) growth rate r(h) from a SMALL seed.

    The red-team's monomorphic-background control: seed a small monomorphic population at
    a clamped h (cost ON, breeds true), measure the early log-growth rate r before
    density-dependence saturates.  r_mono(0.15) − r_mono(0.10) is the density-INDEPENDENT
    gradient — it must agree in sign with the common-garden (density-DEPENDENT) reading
    for a clean gradient claim.  Uses the genuine ecology (food coupling etc.), so r is
    the realized fitness of h against a pure-h background, not a mixed one.
    """
    founder = clamp_founder(cfg.founder, h, inefficiency)
    cfg2 = D.replace(cfg, founder=founder, founder_mix=None,
                     initial_population=min(cfg.initial_population, 12),
                     freeze_thermosense=True)
    eco = Ecology(cfg2, seed=seed)
    w_lo, w_hi = window
    series: list[tuple[int, int]] = []
    cps = set(range(stride, cfg2.horizon + 1, stride))
    while eco.t < cfg2.horizon and not eco.exploded:
        eco.step()
        if eco.t in cps and w_lo <= eco.t <= w_hi:
            series.append((eco.t, len(eco._alive())))
        if not eco._alive():
            break
    pts = [(t, c) for t, c in series if c >= 1]
    if len(pts) < 3:
        return float("nan")
    ts = np.array([t for t, _ in pts], dtype=float)
    ys = np.log(np.array([c for _, c in pts], dtype=float))
    A = np.vstack([ts, np.ones_like(ts)]).T
    return float(np.linalg.lstsq(A, ys, rcond=None)[0][0])


# ---------------------------------------------------------------------------
# Picklable job dispatcher + parallel batch (embarrassingly parallel over jobs;
# each job is one independent Ecology(cfg, seed) — order-independent, deterministic).
# Top-level so ProcessPoolExecutor (spawn on macOS) can re-import it.
# ---------------------------------------------------------------------------

def audit_job(spec: dict[str, Any]) -> dict[str, Any]:
    """Run ONE audit job (kind ∈ capacity/pairwise/benefit/garden) and tag with its key."""
    kind, cfg, seed = spec["kind"], spec["cfg"], spec["seed"]
    ineff = spec.get("ineff", 0.20)
    if kind == "capacity":
        out = run_carrying_capacity(cfg, spec["h"], seed, inefficiency=ineff,
                                    late=spec.get("late", 1500))
    elif kind == "pairwise":
        out = run_pairwise_competition(cfg, spec["h_res"], spec["h_inv"], seed,
                                       count_each=spec.get("count_each", 50),
                                       window=spec.get("window", (200, 2500)), inefficiency=ineff)
    elif kind == "benefit":
        out = {"B": run_installed_benefit(cfg, spec["h"], seed, window=spec.get("bwindow", 1000))}
    elif kind == "growth":
        # density-independent intrinsic growth rate r(h) from a small monomorphic seed (the
        # proper installed benefit, NOT washed out by equilibrium/herding). cost ON or OFF is
        # set by the cfg's enable_thermosense (cost-off cfg ⇒ pure benefit B; cost-on ⇒ realized).
        out = {"r": run_intrinsic_growth(cfg, spec["h"], seed, window=spec.get("window", (100, 700)),
                                         inefficiency=ineff)}
    elif kind == "garden":
        out = run_gradient_audit(cfg, seed, window=spec.get("window", (50, 800)),
                                 checkpoint_stride=spec.get("stride", 50),
                                 min_clamp_pop=spec.get("min_clamp_pop", 8))
    else:
        raise ValueError(f"unknown audit job kind: {kind}")
    return {"key": spec["key"], **out}


def _audit_workers() -> int:
    n = os.cpu_count() or 4
    return max(1, min(n - 1, 16))


def run_audit_batch(specs: list[dict[str, Any]], *, max_workers: int | None = None,
                    sequential: bool = False) -> dict[Hashable, dict[str, Any]]:
    """Run all audit specs and return {key: result}.  Deterministic, order-independent."""
    if sequential or len(specs) <= 1:
        return {s["key"]: audit_job(s) for s in specs}
    results: dict[Hashable, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max_workers or _audit_workers()) as ex:
        futs = {ex.submit(audit_job, s): s["key"] for s in specs}
        for fut in as_completed(futs):
            r = fut.result()
            results[r["key"]] = r
    return results


# ---------------------------------------------------------------------------
# Slope / regression diagnostics (the selection gradient)
# ---------------------------------------------------------------------------

def local_slopes(curve: dict[float, float]) -> dict[str, float]:
    """Central-difference slope of w(h) at each SLOPE_ANCHORS point."""
    out: dict[str, float] = {}
    for name, (lo, hi) in SLOPE_ANCHORS.items():
        wl, wh = curve.get(lo, float("nan")), curve.get(hi, float("nan"))
        out[name] = (wh - wl) / (hi - lo) if not (math.isnan(wl) or math.isnan(wh)) else float("nan")
    return out


def optimum_h(curve: dict[float, float]) -> float:
    """argmax_h w(h) over the measured (non-nan) clamps — the estimated optimum h*."""
    pts = [(h, w) for h, w in curve.items() if not math.isnan(w)]
    if not pts:
        return float("nan")
    return max(pts, key=lambda hw: hw[1])[0]


def fit_selection_regression(rows: list[dict[str, float]]) -> dict[str, float]:
    """OLS fit w = alpha_seed + b1*h + b2*h^2 + b3*(h*density) + eps over (seed,clamp) rows.

    Per-seed intercepts (fixed effects) absorb seed-level differences so b1/b2/b3 are
    within-seed.  Pure numpy lstsq.  Returns b1 (linear), b2 (curvature/concavity:
    b2<0 = diminishing returns), b3 (density interaction), and R^2.
    rows: dicts with keys h, density, w, seed.
    """
    rows = [r for r in rows if not math.isnan(r["w"])]
    if len(rows) < 6:
        return {"b1": float("nan"), "b2": float("nan"), "b3": float("nan"), "r2": float("nan"), "n": len(rows)}
    seeds = sorted({r["seed"] for r in rows})
    seed_idx = {s: i for i, s in enumerate(seeds)}
    n, k = len(rows), len(seeds)
    X = np.zeros((n, k + 3))
    y = np.zeros(n)
    for i, r in enumerate(rows):
        X[i, seed_idx[r["seed"]]] = 1.0                  # per-seed intercept
        X[i, k + 0] = r["h"]
        X[i, k + 1] = r["h"] ** 2
        X[i, k + 2] = r["h"] * r["density"]
        y[i] = r["w"]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"b1": float(beta[k + 0]), "b2": float(beta[k + 1]), "b3": float(beta[k + 2]),
            "r2": r2, "n": n}


# ---------------------------------------------------------------------------
# B(h) installed-benefit overlay (DIAGNOSTIC ONLY — cost OFF; never the verdict)
# ---------------------------------------------------------------------------

def run_carrying_capacity(cfg: EcologyConfig, h: float, seed: int, *,
                          late: int = 1500, stride: int = 100, inefficiency: float = 0.20
                          ) -> dict[str, float]:
    """Monomorphic competitive-fitness readout at clamped h (cost ON, breeds true).

    Density-ROBUST and cold-start-FREE (read at equilibrium, not from a transient cohort):
      N_star  mean standing population late-window  — the carrying capacity supported by h.
      R_star  mean standing resource late-window    — R*-competition ability: the LOWER R*,
              the harder the strategy is to invade (Tilman R* rule: the strategy drawing the
              limiting resource lowest competitively excludes the others).  argmin_h R*(h) is
              the competitively dominant sensor; argmax_h N*(h) is its usual proxy.
      intake  mean per-capita resource_eaten/age (realized foraging WITH cost).
      repro_rate  late-window per-capita reproduction rate.
    This sidesteps the founder cold-start lottery that swamps cohort growth-rate readings.
    """
    founder = clamp_founder(cfg.founder, h, inefficiency)
    cfg2 = D.replace(cfg, founder=founder, founder_mix=None, freeze_thermosense=True)
    eco = Ecology(cfg2, seed=seed)
    pops: list[int] = []
    res: list[float] = []
    births = 0
    alive_steps = 0
    while eco.t < cfg2.horizon and not eco.exploded:
        n_before = len(eco._alive())
        eco.step()
        if eco.t > cfg2.horizon - late:
            alive_steps += n_before
            # count reproduction events this step via population bookkeeping is awkward;
            # use the standing pop + resource sampled on a stride (the equilibrium readout)
            if eco.t % stride == 0:
                pops.append(len(eco._alive()))
                res.append(float(np.mean(eco.world.resource)))
        if not eco._alive():
            break
    # per-capita reproduction over the late window from the event log (cheap final scan)
    repro = sum(1 for e in eco.events if e["event_type"] == "reproduction" and e["t"] > cfg2.horizon - late)
    intake = float(np.mean([c.phenotype.resource_eaten / max(1, c.phenotype.age) for c in eco._alive()])) \
        if eco._alive() else float("nan")
    return {
        "N_star": float(np.mean(pops)) if pops else 0.0,
        "R_star": float(np.mean(res)) if res else float("nan"),
        "intake_on": intake,
        "repro_rate": (repro / alive_steps) if alive_steps > 0 else float("nan"),
        "final_pop": len(eco._alive()),
        "extinct": len(eco._alive()) == 0,
    }


def run_pairwise_competition(cfg: EcologyConfig, h_res: float, h_inv: float, seed: int, *,
                             count_each: int = 50, window: tuple[int, int] = (50, 1500),
                             stride: int = 25, inefficiency: float = 0.20) -> dict[str, float]:
    """Direct selection signal for h_inv vs h_res from a head-to-head competition.

    Seed EQUAL founder counts of the resident (h_res) and one invader (h_inv); both breed
    true (freeze_thermosense), cost ON, shuffle ON, placement shuffled per seed.  Equal
    founders ⇒ both suffer the cold-start mortality EQUALLY, so it differences out.

    The competition can fixate fast under this engine's strong drift/bottleneck, so the
    robust per-seed signal is the INVADER FREQUENCY relative to its 0.5 start:
      inv_frac_auc  — time-averaged invader fraction over the window (0.5 = neutral,
                      >0.5 = invader favoured, <0.5 disfavoured).  Robust to fast fixation.
      inv_won       — 1 if inv_frac_final > 0.5 else 0 (the winner; averaged over seeds the
                      WIN-FRACTION cancels the founder lottery → the selection sign).
      s             — OLS slope of ln(N_inv/N_res) over coexistence points (NaN if <3 points).
    """
    res_f = clamp_founder(cfg.founder, h_res, inefficiency)
    inv_f = clamp_founder(cfg.founder, h_inv, inefficiency)
    fm = tuple([(res_f, 1), (inv_f, 1)] * count_each)
    cfg2 = D.replace(cfg, founder_mix=_shuffle_founder_mix(fm, seed), freeze_thermosense=True)
    eco = Ecology(cfg2, seed=seed)
    w_lo, w_hi = window
    series: list[tuple[int, float]] = []     # (t, ln(n_inv/n_res)) coexistence points
    fracs: list[float] = []                  # invader fraction over the window
    cps = set(range(stride, cfg2.horizon + 1, stride))
    while eco.t < cfg2.horizon and not eco.exploded:
        eco.step()
        if eco.t in cps and w_lo <= eco.t <= w_hi:
            n_res = sum(1 for c in eco._alive() if abs(c.genotype.thermosense_intensity - h_res) < 1e-6)
            n_inv = sum(1 for c in eco._alive() if abs(c.genotype.thermosense_intensity - h_inv) < 1e-6)
            tot = n_res + n_inv
            if tot >= 1:
                fracs.append(n_inv / tot)
            if n_res >= 1 and n_inv >= 1:
                series.append((eco.t, math.log(n_inv / n_res)))
        if not eco._alive():
            break
    alive = eco._alive()
    n_res_f = sum(1 for c in alive if abs(c.genotype.thermosense_intensity - h_res) < 1e-6)
    n_inv_f = sum(1 for c in alive if abs(c.genotype.thermosense_intensity - h_inv) < 1e-6)
    inv_frac_final = (n_inv_f / (n_res_f + n_inv_f)) if (n_res_f + n_inv_f) > 0 else float("nan")
    s = float("nan")
    if len(series) >= 3:
        ts = np.array([t for t, _ in series], dtype=float)
        ys = np.array([y for _, y in series], dtype=float)
        A = np.vstack([ts, np.ones_like(ts)]).T
        s = float(np.linalg.lstsq(A, ys, rcond=None)[0][0])
    return {"s": s, "n_points": len(series),
            "inv_frac_auc": float(np.mean(fracs)) if fracs else float("nan"),
            "inv_frac_final": inv_frac_final,
            "inv_won": (1 if (inv_frac_final == inv_frac_final and inv_frac_final > 0.5) else 0),
            "final_pop": len(alive), "extinct": len(alive) == 0}


def run_installed_benefit(cfg_costoff: EcologyConfig, h: float, seed: int, *,
                          window: int = 1000, stride: int = 100) -> float:
    """Gross per-capita intake B(h) at FROZEN h with cost OFF (the returns-probe pattern).

    enable_thermosense=False pins h for the whole monomorphic lineage AND leaves the
    organ active with NO upkeep charged, so intake(h) is the pure installed benefit.
    The contrast B(0.60)-B(0.00) >> 0 (gift real) vs the realized gradient g(0.10) <= 0
    is the whole point: a forced strong sensor helps; a small heritable step does not pay.
    """
    assert not cfg_costoff.enable_thermosense, "B(h) overlay must be cost-OFF (enable_thermosense=False)"
    founder = clamp_founder(cfg_costoff.founder, h)
    cfg = D.replace(cfg_costoff, founder=founder, founder_mix=None)
    eco = Ecology(cfg, seed=seed)
    samples: list[float] = []
    while eco.t < cfg.horizon and not eco.exploded:
        eco.step()
        if eco.t > cfg.horizon - window and eco.t % stride == 0:
            rates = [c.phenotype.resource_eaten / max(1, c.phenotype.age) for c in eco._alive()]
            if rates:
                samples.append(float(np.mean(rates)))
        if not eco._alive():
            break
    return float(np.mean(samples)) if samples else float("nan")
