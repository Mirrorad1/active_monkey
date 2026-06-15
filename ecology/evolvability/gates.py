"""ecology.evolvability.gates — engine-coupled gate runners for the Evolvability Preflight.

Each gate function takes explicit args (base_cfg, axis, seeds, ...) so this module
is decoupled from any config.py schema.  All gate functions return GateOutcome.

Backend coupling
----------------
For backend=="thermosense" every measurement delegates to sense_axis.py (the proven
Exp 203 instrument).  For any other backend, gates that require engine-specific
cost-off percept paths raise NotImplementedError with a documented reason:
    - run_gifted_benefit: requires cost-off percept which is engine-specific (thermosense only)
    - run_monomorphic_sweep: same reason (calls run_carrying_capacity from sense_axis)
    - run_density_independent_growth: delegates to run_intrinsic_growth (thermosense only)
For generic backends, run_local_pairwise_gradient uses _run_pairwise_generic (a generic
common-garden runner that reads/writes the trait via the axis); run_invasion_from_rarity,
run_null_guards, and run_controller_cross_partial use a generic in-module implementation
(axis.clamp_founder + axis.get + freeze via freeze_flag).

Anti-cheat / honesty
---------------------
- None of these functions write fitness or food as f(h).
- sense_axis.assert_no_direct_h_reward is called where applicable.
- The null_guards gate specifically checks for byte-identical events when the trait
  is disconnected (cost OFF + food coupling OFF), which is the key anti-cheat test.
"""
from __future__ import annotations

import dataclasses as D
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ecology.engine import Ecology, EcologyConfig
from ecology.genotype import Genotype
from ecology.scenarios import SCENARIOS
from ecology import sense_axis as _sa
from ecology.evolvability.trait_axis import TraitAxis
from ecology.evolvability import metrics as _m
from ecology.evolvability import verdicts as _v


# ---------------------------------------------------------------------------
# GateOutcome
# ---------------------------------------------------------------------------

@dataclass
class GateOutcome:
    """Result of one gate function call.

    Fields
    ------
    name            : gate identifier
    question        : plain-English question the gate tests
    metric          : what was measured
    raw_rows        : one row per (seed[, h/theta/cost]) — RAW observations carrying
                      enough to recompute the aggregate
    per_seed        : per-seed derived summary (optional; may duplicate raw)
    aggregate       : derived numbers (means, curve, wins, n_valid, etc.)
    verdict         : the gate verdict enum .value, or "" for non-verdict gates
    validity_flags  : e.g. {"n_valid": k, "n_seeds": n, "extinct_fraction": ...}
    interpretation  : one or two sentences; for gifted/monomorphic MUST state it does
                      NOT prove evolvability
    """
    name: str
    question: str
    metric: str
    raw_rows: list[dict]
    per_seed: list[dict]
    aggregate: dict
    verdict: str
    validity_flags: dict
    interpretation: str


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def build_base_cfg(scenario: str, horizon: int, overrides: dict) -> EcologyConfig:
    """Build a base EcologyConfig from a named scenario + overrides.

    Parameters
    ----------
    scenario  : key in SCENARIOS ("balanced", "scarce", "overabundant")
    horizon   : simulation steps
    overrides : dict of EcologyConfig field overrides (applied after horizon)
    """
    return D.replace(SCENARIOS[scenario], horizon=horizon, **overrides)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nanmean(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _freeze_kwargs(axis: TraitAxis) -> dict:
    """Return the freeze kwargs for a TraitAxis — freeze_flag if set, else mutation_rate=0."""
    if axis.freeze_flag:
        return {axis.freeze_flag: True}
    return {"mutation_rate": 0.0}


def _require_thermosense(axis: TraitAxis, gate_name: str) -> None:
    """Raise NotImplementedError if backend is not thermosense for a sense_axis-only gate."""
    if axis.backend != "thermosense":
        raise NotImplementedError(
            f"Gate '{gate_name}' requires backend=='thermosense' because it uses "
            f"sense_axis.py percept-path functions (run_installed_benefit, "
            f"run_carrying_capacity, run_pairwise_competition, run_intrinsic_growth) "
            f"which are hard-coded to the thermosense organ.  Got backend={axis.backend!r}.  "
            f"To support other backends, implement a generic cost-off percept runner "
            f"that is h-blind in the eat step."
        )


# ---------------------------------------------------------------------------
# A) Gifted benefit (cost-off installed benefit)
# ---------------------------------------------------------------------------

def run_gifted_benefit(
    base_cfg: EcologyConfig,
    axis: TraitAxis,
    seeds: list[int],
    *,
    eps: float = 1e-6,
) -> GateOutcome:
    """Gate A: does the trait provide any installed benefit when the cost is waived?

    Measures B(low) and B(high) with the trait cost disabled.  A positive delta
    shows the feature CAN be useful; it does NOT imply the trait is evolvable.

    ONLY backend=='thermosense' is supported (raises NotImplementedError otherwise)
    because the cost-off percept path (enable_thermosense=False while keeping the
    foraging policy active) is specific to the thermosense engine branch.
    """
    _require_thermosense(axis, "gifted_benefit")

    cost_off_cfg = D.replace(base_cfg, **{axis.enable_flag: False})
    h_low = axis.low_value if axis.low_value is not None else 0.0
    h_high = axis.high_value if axis.high_value is not None else 1.0

    raw_rows: list[dict] = []
    deltas: list[float] = []

    for seed in seeds:
        B_low = _sa.run_installed_benefit(cost_off_cfg, h_low, seed)
        B_high = _sa.run_installed_benefit(cost_off_cfg, h_high, seed)
        delta = B_high - B_low if not (math.isnan(B_low) or math.isnan(B_high)) else float("nan")
        deltas.append(delta)
        raw_rows.append({
            "gate": "gifted_benefit",
            "seed": seed,
            "h_low": h_low,
            "h_high": h_high,
            "B_low": B_low,
            "B_high": B_high,
            "delta": delta,
        })

    mean_delta = _nanmean(deltas)
    verdict_enum, verdict_reason = _v.benefit_verdict(mean_delta, eps=eps)

    n_valid = sum(1 for d in deltas if not math.isnan(d))

    return GateOutcome(
        name="gifted_benefit",
        question="Does the trait provide an installed benefit when the cost is waived?",
        metric="mean(B_high - B_low) with cost OFF",
        raw_rows=raw_rows,
        per_seed=[{"seed": r["seed"], "delta": r["delta"]} for r in raw_rows],
        aggregate={"mean_delta": mean_delta, "h_low": h_low, "h_high": h_high},
        verdict=verdict_enum.value,
        validity_flags={"n_seeds": len(seeds), "n_valid": n_valid},
        interpretation=(
            f"A gifted/cost-off benefit ({verdict_reason}) shows the feature CAN be "
            f"useful; it does NOT imply the trait is evolvable.  Evolvability requires "
            f"a positive LOCAL selection gradient near the resident value, not just a "
            f"global benefit at high trait values."
        ),
    )


# ---------------------------------------------------------------------------
# B) Monomorphic sweep
# ---------------------------------------------------------------------------

def run_monomorphic_sweep(
    base_cfg: EcologyConfig,
    axis: TraitAxis,
    seeds: list[int],
    grid: list[float],
    *,
    min_pop: int = 10,
) -> GateOutcome:
    """Gate B: monomorphic N* sweep over the trait grid.

    Measures the carrying capacity N*(h) for each h in the grid.  The global optimum
    h* = argmax N*(h) and its position relative to the resident reveals whether the
    monomorphic landscape peaks above or below the resident.  This is a GLOBAL
    (monomorphic) signal, NOT a local invasion gradient; it does NOT prove local
    evolvability.

    ONLY backend=='thermosense' is supported (raises NotImplementedError otherwise).
    """
    _require_thermosense(axis, "monomorphic_sweep")

    raw_rows: list[dict] = []
    # per-h accumulator: {h: {key: [values across seeds]}}
    by_h: dict[float, dict[str, list[float]]] = {
        h: {"N_star": [], "repro_rate": [], "final_pop": [], "extinct": []}
        for h in grid
    }

    for h in grid:
        for seed in seeds:
            r = _sa.run_carrying_capacity(base_cfg, h, seed, inefficiency=axis.inefficiency_value)
            raw_rows.append({
                "gate": "monomorphic_sweep",
                "seed": seed,
                "h": h,
                "N_star": r["N_star"],
                "R_star": r["R_star"],
                "intake_on": r["intake_on"],
                "repro_rate": r["repro_rate"],
                "final_pop": r["final_pop"],
                "extinct": r["extinct"],
            })
            by_h[h]["N_star"].append(r["N_star"])
            by_h[h]["repro_rate"].append(r["repro_rate"] if not r["extinct"] else float("nan"))
            by_h[h]["final_pop"].append(r["final_pop"])
            by_h[h]["extinct"].append(float(r["extinct"]))

    curve_Nstar = {h: _nanmean(by_h[h]["N_star"]) for h in grid}
    curve_repro = {h: _nanmean(by_h[h]["repro_rate"]) for h in grid}
    extinct_fraction = {h: _nanmean(by_h[h]["extinct"]) for h in grid}

    opt_h, opt_v = _m.monomorphic_optimum(curve_Nstar)
    above_resident = bool(_m.optimum_above_resident(curve_Nstar, axis.resident_value))

    # Survivable at optimum: mean final_pop >= min_pop and extinct_fraction < 0.5
    if not math.isnan(opt_h):
        # find the h in grid closest to opt_h (should be exact)
        opt_h_key = min(grid, key=lambda h: abs(h - opt_h))
        mean_fp_at_opt = _nanmean(by_h[opt_h_key]["final_pop"])
        ext_frac_at_opt = _nanmean(by_h[opt_h_key]["extinct"])
        survivable = bool(
            not math.isnan(mean_fp_at_opt) and
            mean_fp_at_opt >= min_pop and
            not math.isnan(ext_frac_at_opt) and
            ext_frac_at_opt < 0.5
        )
    else:
        survivable = False

    return GateOutcome(
        name="monomorphic_sweep",
        question="Where does the monomorphic N* landscape peak relative to the resident?",
        metric="mean N*(h) over seeds for each h in grid",
        raw_rows=raw_rows,
        per_seed=[],
        aggregate={
            "optimum_h": opt_h,
            "optimum_value": opt_v,
            "above_resident": above_resident,
            "survivable": survivable,
            "curve_Nstar": curve_Nstar,
            "curve_repro": curve_repro,
            "extinct_fraction": extinct_fraction,
        },
        verdict="",
        validity_flags={"n_seeds": len(seeds), "n_h": len(grid)},
        interpretation=(
            f"Monomorphic global optimum at h*={opt_h:.3g} with mean N*={opt_v:.3g}. "
            f"A global/monomorphic optimum above the resident does NOT prove LOCAL "
            f"evolvability: the resident might be separated from h* by a fitness valley "
            f"that rare mutants cannot cross.  Local invasion assays (gates C and D) are "
            f"required to establish a positive selection gradient near the resident."
        ),
    )


# ---------------------------------------------------------------------------
# Parallel-map helper (order-preserving, falls back to serial)
# ---------------------------------------------------------------------------

def _parallel_map(worker, arg_list, max_workers):
    """Map worker over arg_list. Serial when max_workers in (None, 0, 1); else a
    ProcessPoolExecutor with exactly max_workers procs. Results ALWAYS in input order
    (ex.map preserves order), so downstream aggregation is identical to serial."""
    if max_workers in (None, 0, 1):
        return [worker(a) for a in arg_list]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(worker, arg_list))


# ---------------------------------------------------------------------------
# C-helper) Generic equal-count common-garden runner (Gate C fallback)
# ---------------------------------------------------------------------------

def _run_pairwise_generic(
    base_cfg: "EcologyConfig",
    axis: TraitAxis,
    h_res: float,
    h_mut: float,
    seed: int,
    *,
    count_each: int = 50,
    window: tuple[int, int] = (50, 1500),
    stride: int = 25,
) -> dict:
    """Generic equal-count resident-vs-mutant common garden (Gate C fallback for any
    TraitAxis backend). Mirrors sense_axis.run_pairwise_competition but reads/writes the
    trait via the axis (axis.clamp_founder / axis.get) and freezes via _freeze_kwargs(axis)
    (freeze_flag if the engine supports it, else mutation_rate=0). Returns the SAME dict
    keys as run_pairwise_competition so the gate logic is backend-agnostic."""
    res_f = axis.clamp_founder(base_cfg.founder, h_res, axis.inefficiency_value)
    inv_f = axis.clamp_founder(base_cfg.founder, h_mut, axis.inefficiency_value)

    fm: tuple = tuple([(res_f, 1), (inv_f, 1)] * count_each)
    cfg = D.replace(
        base_cfg,
        founder_mix=_sa._shuffle_founder_mix(fm, seed),
        **_freeze_kwargs(axis),
    )

    eco = Ecology(cfg, seed=seed)
    w_lo, w_hi = window

    fracs: list[float] = []
    series: list[tuple[float, float]] = []  # (t, ln(n_inv/n_res))

    cps = set(range(stride, cfg.horizon + 1, stride))
    while eco.t < cfg.horizon and not eco.exploded:
        eco.step()
        if eco.t in cps and w_lo <= eco.t <= w_hi:
            alive_now = eco.alive_snapshot()
            n_res = sum(1 for c in alive_now if abs(axis.get(c.genotype) - h_res) < 1e-6)
            n_inv = sum(1 for c in alive_now if abs(axis.get(c.genotype) - h_mut) < 1e-6)
            total = n_res + n_inv
            if total >= 1:
                fracs.append(n_inv / total)
            if n_res >= 1 and n_inv >= 1:
                series.append((float(eco.t), math.log(n_inv / n_res)))
        if not eco.has_alive():
            break

    alive = eco.alive_snapshot()
    n_res_final = sum(1 for c in alive if abs(axis.get(c.genotype) - h_res) < 1e-6)
    n_inv_final = sum(1 for c in alive if abs(axis.get(c.genotype) - h_mut) < 1e-6)
    total_final = n_res_final + n_inv_final
    inv_frac_final = (n_inv_final / total_final) if total_final >= 1 else float("nan")

    if len(series) >= 3:
        ts = np.array([p[0] for p in series]).reshape(-1, 1)
        ys = np.array([p[1] for p in series])
        A = np.hstack([ts, np.ones((len(ts), 1))])
        coeffs, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        s = float(coeffs[0])
    else:
        s = float("nan")

    inv_won = 1 if (inv_frac_final == inv_frac_final and inv_frac_final > 0.5) else 0

    return {
        "s": s,
        "n_points": len(series),
        "inv_frac_auc": _nanmean(fracs) if fracs else float("nan"),
        "inv_frac_final": inv_frac_final,
        "inv_won": inv_won,
        "final_pop": len(alive),
        "extinct": len(alive) == 0,
    }


# ---------------------------------------------------------------------------
# C-worker) Top-level picklable per-seed worker for the pairwise gate
# ---------------------------------------------------------------------------

def _pairwise_one_seed(args):
    base_cfg, axis, h_res, h_mut, seed, count_each, window = args
    if axis.backend == "thermosense":
        return _sa.run_pairwise_competition(base_cfg, h_res, h_mut, seed,
                                            count_each=count_each, window=window,
                                            inefficiency=axis.inefficiency_value)
    return _run_pairwise_generic(base_cfg, axis, h_res, h_mut, seed,
                                 count_each=count_each, window=window)


# ---------------------------------------------------------------------------
# C) Local pairwise gradient
# ---------------------------------------------------------------------------

def run_local_pairwise_gradient(
    base_cfg: EcologyConfig,
    axis: TraitAxis,
    seeds: list[int],
    *,
    win_threshold: int,
    lose_threshold: int,
    min_valid: int,
    count_each: int = 50,
    window: tuple[int, int] = (50, 1500),
    min_pop: int = 10,
    max_workers: int | None = None,
) -> GateOutcome:
    """Gate C: local pairwise invasion gradient (resident vs single-step mutant).

    Runs a 50/50 common-garden competition between resident (h_res) and mutant (h_mut)
    for each seed.  The verdict is based on invader fraction relative to 0.5 (starting
    share): inv_frac_final > 0.5 means the invader (mutant=h_mut) beat the resident.

    For backend=='thermosense', delegates to sense_axis.run_pairwise_competition (the
    proven Exp 203 instrument).  For any other backend, uses _run_pairwise_generic, a
    generic common-garden runner that reads/writes the trait via the axis
    (axis.clamp_founder / axis.get) and returns the same dict keys.

    max_workers: None/1 = serial (default). When set (e.g. ecology.runtime_budget.preflight(...)'s
    memory-sized recommended_workers), the per-seed runs execute in parallel across that many
    processes; results are bit-identical to serial (seeds are independent).
    """
    h_res = axis.resident_value
    h_mut = axis.mutant_value

    raw_rows: list[dict] = []
    pairs: list[tuple[float, float]] = []
    s_values: list[float] = []
    inv_frac_finals: list[float] = []
    extinct_count = 0

    args = [(base_cfg, axis, h_res, h_mut, s, count_each, window) for s in seeds]
    d_list = _parallel_map(_pairwise_one_seed, args, max_workers)

    for seed, d in zip(seeds, d_list):
        valid = (
            _m.is_population_valid(d["final_pop"], d["extinct"], False, min_pop)
            and not math.isnan(d["inv_frac_final"])
        )
        if d["extinct"]:
            extinct_count += 1
        raw_rows.append({
            "gate": "local_pairwise_gradient",
            "seed": seed,
            "h_res": h_res,
            "h_mut": h_mut,
            "s": d["s"],
            "inv_frac_auc": d["inv_frac_auc"],
            "inv_frac_final": d["inv_frac_final"],
            "inv_won": d["inv_won"],
            "final_pop": d["final_pop"],
            "extinct": d["extinct"],
            "valid": valid,
        })
        if valid:
            pairs.append((d["inv_frac_final"], 0.5))
            s_values.append(d["s"])
            inv_frac_finals.append(d["inv_frac_final"])

    wins, n_valid = _m.count_wins(pairs)
    mean_effect = _m.mean_diff(pairs)
    mean_s = _nanmean(s_values)
    mean_inv_frac_final = _nanmean(inv_frac_finals)
    extinct_fraction = extinct_count / len(seeds) if seeds else float("nan")

    verdict_enum, verdict_reason = _v.gradient_verdict(
        wins, n_valid, mean_effect,
        win_threshold=win_threshold,
        lose_threshold=lose_threshold,
        min_valid=min_valid,
    )

    return GateOutcome(
        name="local_pairwise_gradient",
        question=f"Does the mutant h={h_mut} invade the resident h={h_res} in head-to-head competition?",
        metric="invader fraction at end relative to 0.5 neutral start",
        raw_rows=raw_rows,
        per_seed=[
            {"seed": r["seed"], "inv_frac_final": r["inv_frac_final"], "valid": r["valid"]}
            for r in raw_rows
        ],
        aggregate={
            "wins": wins,
            "n_valid": n_valid,
            "mean_effect": mean_effect,
            "mean_s": mean_s,
            "mean_inv_frac_final": mean_inv_frac_final,
        },
        verdict=verdict_enum.value,
        validity_flags={
            "n_seeds": len(seeds),
            "n_valid": n_valid,
            "extinct_fraction": extinct_fraction,
        },
        interpretation=(
            f"Local pairwise gradient verdict: {verdict_enum.value}.  {verdict_reason}.  "
            f"This tests whether a single-step mutant (h={h_mut}) can invade a resident "
            f"(h={h_res}) population — the direct test of local evolvability."
        ),
    )


# ---------------------------------------------------------------------------
# D) Invasion from rarity
# ---------------------------------------------------------------------------

def _run_invasion(
    base_cfg: EcologyConfig,
    axis: TraitAxis,
    h_res: float,
    h_mut: float,
    seed: int,
    *,
    resident_count: int,
    invader_count: int,
    window: tuple[int, int],
    stride: int,
) -> dict:
    """Generic rare-invasion runner using axis.clamp_founder + freeze_flag.

    Seeds (resident_count) residents + (invader_count) mutants, freezes the trait,
    and measures the mutant frequency at the start and end of the window.
    """
    freeze_kw = _freeze_kwargs(axis)

    res_founder = axis.clamp_founder(base_cfg.founder, h_res, axis.inefficiency_value)
    mut_founder = axis.clamp_founder(base_cfg.founder, h_mut, axis.inefficiency_value)

    founders: tuple[tuple[Genotype, int], ...] = (
        (res_founder, resident_count),
        (mut_founder, invader_count),
    )
    shuffled_mix = _sa._shuffle_founder_mix(founders, seed)
    cfg = D.replace(base_cfg, founder_mix=shuffled_mix, **freeze_kw)

    eco = Ecology(cfg, seed=seed)
    w_lo, w_hi = window

    f_initial = float("nan")
    f_final = float("nan")

    cps = set(range(stride, cfg.horizon + 1, stride))
    while eco.t < cfg.horizon and not eco.exploded:
        eco.step()
        if eco.t in cps and w_lo <= eco.t <= w_hi:
            alive_now = eco.alive_snapshot()
            n_res = sum(1 for c in alive_now if abs(axis.get(c.genotype) - h_res) < 1e-6)
            n_mut = sum(1 for c in alive_now if abs(axis.get(c.genotype) - h_mut) < 1e-6)
            tot = n_res + n_mut
            if tot >= 1:
                frac = n_mut / tot
                if math.isnan(f_initial):
                    f_initial = frac
                f_final = frac
        if not eco.has_alive():
            break

    alive = eco.alive_snapshot()
    final_pop = len(alive)
    extinct = final_pop == 0

    increased = (not math.isnan(f_initial) and not math.isnan(f_final) and f_final > f_initial)

    return {
        "f_initial": f_initial,
        "f_final": f_final,
        "increased": increased,
        "final_pop": final_pop,
        "extinct": extinct,
    }


def _invasion_one_seed(args):
    """Top-level picklable per-seed worker for the invasion gate."""
    base_cfg, axis, h_res, h_mut, seed, rc, ic, window, stride = args
    return _run_invasion(base_cfg, axis, h_res, h_mut, seed,
                         resident_count=rc, invader_count=ic,
                         window=window, stride=stride)


def run_invasion_from_rarity(
    base_cfg: EcologyConfig,
    axis: TraitAxis,
    seeds: list[int],
    *,
    win_threshold: int,
    lose_threshold: int,
    min_valid: int,
    mutant_fraction: float = 0.05,
    resident_count: int = 95,
    window: tuple[int, int] = (50, 1500),
    stride: int = 25,
    min_pop: int = 10,
    max_workers: int | None = None,
) -> GateOutcome:
    """Gate D: can a rare mutant increase in frequency when starting from rarity?

    Seeds a RARE mutant (≈mutant_fraction of total) into a resident background and
    measures whether the mutant frequency increases over the competition window.
    Uses a generic invasion runner that works for any backend.

    max_workers: None/1 = serial (default). When set (e.g. ecology.runtime_budget.preflight(...)'s
    memory-sized recommended_workers), the per-seed runs execute in parallel across that many
    processes; results are bit-identical to serial (seeds are independent).
    """
    h_res = axis.resident_value
    h_mut = axis.mutant_value
    invader_count = max(1, round(resident_count * mutant_fraction / (1.0 - mutant_fraction)))

    raw_rows: list[dict] = []
    increase_count = 0
    n_valid = 0
    extinct_count = 0

    args = [(base_cfg, axis, h_res, h_mut, s, resident_count, invader_count, window, stride)
            for s in seeds]
    r_list = _parallel_map(_invasion_one_seed, args, max_workers)

    for seed, r in zip(seeds, r_list):
        valid = _m.is_population_valid(r["final_pop"], r["extinct"], False, min_pop)
        if r["extinct"]:
            extinct_count += 1
        raw_rows.append({
            "gate": "invasion_from_rarity",
            "seed": seed,
            "h_res": h_res,
            "h_mut": h_mut,
            "mutant_fraction": mutant_fraction,
            "f_initial": r["f_initial"],
            "f_final": r["f_final"],
            "increased": r["increased"],
            "final_pop": r["final_pop"],
            "extinct": r["extinct"],
        })
        if valid:
            n_valid += 1
            if r["increased"]:
                increase_count += 1

    extinct_fraction = extinct_count / len(seeds) if seeds else float("nan")
    verdict_enum, verdict_reason = _v.invasion_verdict(
        increase_count, n_valid,
        win_threshold=win_threshold,
        lose_threshold=lose_threshold,
        min_valid=min_valid,
    )

    return GateOutcome(
        name="invasion_from_rarity",
        question=f"Can a rare mutant (h={h_mut}, ~{mutant_fraction:.0%}) increase in a resident background?",
        metric="fraction of seeds where mutant frequency increased over the window",
        raw_rows=raw_rows,
        per_seed=[
            {"seed": r["seed"], "f_initial": r["f_initial"], "f_final": r["f_final"],
             "increased": r["increased"]}
            for r in raw_rows
        ],
        aggregate={
            "increase_count": increase_count,
            "n_valid": n_valid,
            "invader_count": invader_count,
            "resident_count": resident_count,
        },
        verdict=verdict_enum.value,
        validity_flags={
            "n_seeds": len(seeds),
            "n_valid": n_valid,
            "extinct_fraction": extinct_fraction,
        },
        interpretation=(
            f"Invasion from rarity verdict: {verdict_enum.value}.  {verdict_reason}.  "
            f"Mutant started at ~{mutant_fraction:.0%} frequency (invader_count={invader_count}).  "
            f"This is a more stringent test than equal-count pairwise: the mutant must "
            f"increase when genuinely rare, as required by adaptive dynamics."
        ),
    )


# ---------------------------------------------------------------------------
# E) Density-independent growth
# ---------------------------------------------------------------------------

def run_density_independent_growth(
    base_cfg: EcologyConfig,
    axis: TraitAxis,
    seeds: list[int],
    *,
    eps: float = 1e-6,
) -> GateOutcome:
    """Gate E: intrinsic (density-independent) growth rate contrast r(high) - r(low).

    Measures r(h) at low and high trait values from a small monomorphic seed at low
    density.  A positive delta_r indicates the trait helps intrinsically (private/
    absolute benefit) rather than only beating crowded neighbours.

    ONLY backend=='thermosense' is supported (raises NotImplementedError otherwise).
    """
    _require_thermosense(axis, "density_independent_growth")

    h_low = axis.low_value if axis.low_value is not None else 0.0
    h_high = axis.high_value if axis.high_value is not None else 1.0

    raw_rows: list[dict] = []
    delta_rs: list[float] = []

    for seed in seeds:
        r_low = _sa.run_intrinsic_growth(base_cfg, h_low, seed, inefficiency=axis.inefficiency_value)
        r_high = _sa.run_intrinsic_growth(base_cfg, h_high, seed, inefficiency=axis.inefficiency_value)
        delta_r = r_high - r_low if not (math.isnan(r_low) or math.isnan(r_high)) else float("nan")
        delta_rs.append(delta_r)
        raw_rows.append({
            "gate": "density_independent_growth",
            "seed": seed,
            "h_low": h_low,
            "h_high": h_high,
            "r_low": r_low,
            "r_high": r_high,
            "delta_r": delta_r,
        })

    mean_delta_r = _nanmean(delta_rs)
    verdict_enum, verdict_reason = _v.benefit_verdict(mean_delta_r, eps=eps)
    n_valid = sum(1 for d in delta_rs if not math.isnan(d))

    return GateOutcome(
        name="density_independent_growth",
        question="Does the trait improve intrinsic growth rate at low population density?",
        metric="mean(r(h_high) - r(h_low)) from small monomorphic seeds",
        raw_rows=raw_rows,
        per_seed=[{"seed": r["seed"], "delta_r": r["delta_r"]} for r in raw_rows],
        aggregate={"mean_delta_r": mean_delta_r, "h_low": h_low, "h_high": h_high},
        verdict=verdict_enum.value,
        validity_flags={"n_seeds": len(seeds), "n_valid": n_valid},
        interpretation=(
            f"Density-independent growth verdict: {verdict_enum.value}.  {verdict_reason}.  "
            f"delta_r > 0 at LOW density means the trait helps intrinsically (private/absolute "
            f"benefit), not only by beating crowded neighbours.  This distinguishes frequency-"
            f"dependent and frequency-independent selection pressures."
        ),
    )


# ---------------------------------------------------------------------------
# F) Cost sensitivity
# ---------------------------------------------------------------------------

def run_cost_sensitivity(
    base_cfg: EcologyConfig,
    axis: TraitAxis,
    seeds: list[int],
    cost_values: list[float],
    *,
    win_threshold: int,
    lose_threshold: int,
    min_valid: int,
    count_each: int = 50,
    window: tuple[int, int] = (50, 1500),
    min_pop: int = 10,
) -> GateOutcome:
    """Gate F: how does the pairwise gradient change across a range of cost levels?

    For each inefficiency in cost_values, reruns the pairwise gradient between
    axis.resident_value and axis.mutant_value.  Reports per-cost gradient verdicts,
    sign_change_cost (first cost where the verdict flips POSITIVE → non-positive),
    and unaffordable_cost (first cost where extinction_fraction >= 0.5).

    ONLY backend=='thermosense' is supported (raises NotImplementedError otherwise).

    This gate is diagnostic (no pass/fail verdict) — it reveals the cost tolerance
    of the selection gradient.
    """
    _require_thermosense(axis, "cost_sensitivity")

    h_res = axis.resident_value
    h_mut = axis.mutant_value

    raw_rows: list[dict] = []
    per_cost: list[dict] = []

    sign_change_cost = None
    unaffordable_cost = None
    prev_positive = None

    for cost in cost_values:
        pairs: list[tuple[float, float]] = []
        ext_count = 0
        cost_raw: list[dict] = []

        for seed in seeds:
            d = _sa.run_pairwise_competition(
                base_cfg, h_res, h_mut, seed,
                count_each=count_each,
                window=window,
                inefficiency=cost,
            )
            valid = (
                _m.is_population_valid(d["final_pop"], d["extinct"], False, min_pop)
                and not math.isnan(d["inv_frac_final"])
            )
            if d["extinct"]:
                ext_count += 1
            row = {
                "gate": "cost_sensitivity",
                "seed": seed,
                "cost": cost,
                "h_res": h_res,
                "h_mut": h_mut,
                "s": d["s"],
                "inv_frac_final": d["inv_frac_final"],
                "inv_won": d["inv_won"],
                "final_pop": d["final_pop"],
                "extinct": d["extinct"],
                "valid": valid,
            }
            raw_rows.append(row)
            cost_raw.append(row)
            if valid:
                pairs.append((d["inv_frac_final"], 0.5))

        wins, n_valid_cost = _m.count_wins(pairs)
        mean_effect = _m.mean_diff(pairs)
        ext_frac = ext_count / len(seeds) if seeds else float("nan")
        v_enum, v_reason = _v.gradient_verdict(
            wins, n_valid_cost, mean_effect,
            win_threshold=win_threshold,
            lose_threshold=lose_threshold,
            min_valid=min_valid,
        )
        is_positive = (v_enum == _v.GradientVerdict.POSITIVE_LOCAL_GRADIENT)

        per_cost.append({
            "cost": cost,
            "wins": wins,
            "n_valid": n_valid_cost,
            "mean_effect": mean_effect,
            "verdict": v_enum.value,
            "extinct_fraction": ext_frac,
        })

        if unaffordable_cost is None and not math.isnan(ext_frac) and ext_frac >= 0.5:
            unaffordable_cost = cost
        if sign_change_cost is None:
            if prev_positive is True and not is_positive:
                sign_change_cost = cost
        prev_positive = is_positive

    return GateOutcome(
        name="cost_sensitivity",
        question="How does the pairwise gradient change across different cost (inefficiency) levels?",
        metric="per-cost pairwise gradient verdict and extinction fraction",
        raw_rows=raw_rows,
        per_seed=[],
        aggregate={
            "per_cost": per_cost,
            "sign_change_cost": sign_change_cost,
            "unaffordable_cost": unaffordable_cost,
        },
        verdict="",
        validity_flags={"n_seeds": len(seeds), "n_cost_levels": len(cost_values)},
        interpretation=(
            f"Cost sensitivity sweep over {len(cost_values)} inefficiency levels.  "
            f"sign_change_cost={sign_change_cost} (lowest cost where positive gradient "
            f"flips to non-positive), unaffordable_cost={unaffordable_cost} (lowest cost "
            f"where >50% of seeds go extinct).  This is a diagnostic gate — no pass/fail."
        ),
    )


# ---------------------------------------------------------------------------
# G) Null guards
# ---------------------------------------------------------------------------

def run_null_guards(
    base_cfg: EcologyConfig,
    axis: TraitAxis,
    seeds: list[int],
    *,
    min_pop: int = 10,
    pairwise_extinct_fraction: float | None = None,
) -> GateOutcome:
    """Gate G: null/anti-cheat guard battery.

    Guards implemented:
    1. cost_off_disconnected_byte_identical — events_hash must be equal when trait is
       disconnected (enable_flag=False + enable_food_coupling=False) regardless of h.
    2. no_direct_h_reward — for thermosense, assert_no_direct_h_reward must not raise.
    3. trait_disabled_null — per-capita intake must be ~identical when trait is disconnected.
    4. population_validity — pairwise_extinct_fraction < 0.5 (if provided).
    5. perfect_percept_null — NOT_IMPLEMENTED for most configs (see reason).
    6. frozen_memory_map — NOT_IMPLEMENTED (documented, not faked).
    7. shuffle_order — PASS if base_cfg.shuffle_creature_order, else NOT_IMPLEMENTED/NA.

    aggregate carries {"guards": [...], "all_pass": bool}.
    verdict = "" (diagnostic battery, not a single pass/fail gate).
    """
    GS = _v.GuardStatus
    guards: list[dict] = []

    h_low = axis.low_value if axis.low_value is not None else 0.0
    h_high = axis.high_value if axis.high_value is not None else 1.0
    seed0 = seeds[0] if seeds else 0
    freeze_kw = _freeze_kwargs(axis)

    # ------------------------------------------------------------------
    # Guard 1: cost_off_disconnected_byte_identical
    # ------------------------------------------------------------------
    # Guard 3 shares the same runs — computed here together.
    # Disconnect recipe: the trait must feed into NOTHING (cost AND every steering/
    # percept channel off). The axis declares this; fall back to cost-off only (which
    # is usually insufficient — documented on TraitAxis.disconnect_overrides).
    disconnect: dict = {axis.enable_flag: False}
    disconnect.update(axis.disconnect_overrides)
    try:
        disc_cfg_low = D.replace(
            base_cfg,
            **disconnect,
            founder=axis.clamp_founder(base_cfg.founder, h_low, axis.inefficiency_value),
            founder_mix=None,
            **freeze_kw,
        )
        disc_cfg_high = D.replace(
            base_cfg,
            **disconnect,
            founder=axis.clamp_founder(base_cfg.founder, h_high, axis.inefficiency_value),
            founder_mix=None,
            **freeze_kw,
        )
        eco_low = Ecology(disc_cfg_low, seed=seed0)
        eco_low.run()
        eco_high = Ecology(disc_cfg_high, seed=seed0)
        eco_high.run()

        hash_low = eco_low.events_hash()
        hash_high = eco_high.events_hash()
        if hash_low == hash_high:
            g1_status = GS.PASS
            g1_reason = f"events_hash identical for h={h_low} and h={h_high} when trait disconnected"
        else:
            g1_status = GS.FAIL
            g1_reason = (
                f"events_hash DIFFER ({hash_low[:8]}... vs {hash_high[:8]}...) "
                f"when trait is disconnected — suspected artifact"
            )

        # Guard 3: intake identical
        snap_low = eco_low.alive_snapshot()
        snap_high = eco_high.alive_snapshot()
        def _intake(snap: list) -> float:
            if not snap:
                return float("nan")
            return float(sum(c.phenotype.resource_eaten for c in snap))
        il = _intake(snap_low)
        ih = _intake(snap_high)
        if math.isnan(il) or math.isnan(ih):
            g3_status = GS.FAIL
            g3_reason = "intake is nan (population extinct) — cannot confirm null"
        elif abs(il - ih) < 1e-9:
            g3_status = GS.PASS
            g3_reason = f"per-capita intake identical: |{il:.6g} - {ih:.6g}| < 1e-9"
        else:
            g3_status = GS.FAIL
            g3_reason = f"intake differs: {il:.6g} vs {ih:.6g} (diff={abs(il-ih):.3g}) — unexpected"

    except Exception as e:
        g1_status = GS.FAIL
        g1_reason = f"exception during disconnected run: {e}"
        g3_status = GS.FAIL
        g3_reason = f"exception during disconnected run (shared with guard 1): {e}"

    guards.append({"name": "cost_off_disconnected_byte_identical",
                   "status": g1_status.value, "reason": g1_reason})

    # ------------------------------------------------------------------
    # Guard 2: no_direct_h_reward
    # ------------------------------------------------------------------
    if axis.backend == "thermosense" and axis.freeze_flag:
        try:
            founder_low = axis.clamp_founder(base_cfg.founder, axis.resident_value, axis.inefficiency_value)
            founder_high = axis.clamp_founder(base_cfg.founder, axis.mutant_value, axis.inefficiency_value)
            guard_cfg = D.replace(
                base_cfg,
                **{axis.enable_flag: True, axis.freeze_flag: True},
                founder_mix=((founder_low, 1), (founder_high, 1)),
            )
            _sa.assert_no_direct_h_reward(guard_cfg)
            g2_status = GS.PASS
            g2_reason = "assert_no_direct_h_reward passed: enable_thermosense=True, freeze=True, founder_mix set"
        except AssertionError as ae:
            g2_status = GS.FAIL
            g2_reason = f"assert_no_direct_h_reward FAILED: {ae}"
        except Exception as e:
            g2_status = GS.FAIL
            g2_reason = f"unexpected exception in no_direct_h_reward guard: {e}"
    else:
        g2_status = GS.NOT_IMPLEMENTED
        g2_reason = (
            f"no_direct_h_reward guard requires backend=='thermosense' and a freeze_flag. "
            f"Got backend={axis.backend!r}, freeze_flag={axis.freeze_flag!r}."
        )
    guards.append({"name": "no_direct_h_reward", "status": g2_status.value, "reason": g2_reason})

    # Guard 3 appended here (computed above alongside guard 1)
    guards.append({"name": "trait_disabled_null", "status": g3_status.value, "reason": g3_reason})

    # ------------------------------------------------------------------
    # Guard 4: population_validity
    # ------------------------------------------------------------------
    if pairwise_extinct_fraction is not None:
        if pairwise_extinct_fraction < 0.5:
            g4_status = GS.PASS
            g4_reason = f"pairwise extinct_fraction={pairwise_extinct_fraction:.3f} < 0.5"
        else:
            g4_status = GS.FAIL
            g4_reason = f"pairwise extinct_fraction={pairwise_extinct_fraction:.3f} >= 0.5"
    else:
        g4_status = GS.NA
        g4_reason = "pairwise_extinct_fraction not provided; cannot assess"
    guards.append({"name": "population_validity", "status": g4_status.value, "reason": g4_reason})

    # ------------------------------------------------------------------
    # Guard 5: perfect_percept_null
    # ------------------------------------------------------------------
    # Only meaningful if backend==thermosense AND there is a niche_confusion knob AND
    # enable_niche is active — otherwise NOT_IMPLEMENTED.
    if axis.backend == "thermosense" and getattr(base_cfg, "enable_niche", False):
        # A niche_confusion=0 percept makes h irrelevant to routing — but verifying that
        # honestly needs a matched non-null comparison run, which this guard does not
        # build. Left NOT_IMPLEMENTED rather than running a sim we would only discard
        # (and never faking a PASS). The byte-identity guard (1) already proves h leaks
        # nowhere when fully disconnected via the axis disconnect_overrides.
        g5_status = GS.NOT_IMPLEMENTED
        g5_reason = (
            "perfect_percept_null requires a matched (null vs non-null) comparison run "
            "that this guard does not construct; left NOT_IMPLEMENTED (no spurious PASS). "
            "Guard 1 (byte-identity when disconnected) covers the anti-cheat property here."
        )
    else:
        g5_status = GS.NOT_IMPLEMENTED
        g5_reason = (
            "perfect_percept_null requires backend=='thermosense' AND enable_niche=True "
            f"(a confusion knob).  Got backend={axis.backend!r}, "
            f"enable_niche={getattr(base_cfg, 'enable_niche', False)}.  "
            "Do NOT fake a pass here — this guard must be left NOT_IMPLEMENTED."
        )
    guards.append({"name": "perfect_percept_null", "status": g5_status.value, "reason": g5_reason})

    # ------------------------------------------------------------------
    # Guard 6: frozen_memory_map
    # ------------------------------------------------------------------
    g6_status = GS.NOT_IMPLEMENTED
    g6_reason = (
        "frozen_memory_map is a documented interface (the creature's learned resource "
        "map could substitute for sensor skill), but no freeze_learning_rate check or "
        "separate no-memory control run is implemented in this gate.  Left NOT_IMPLEMENTED; "
        "enable_band_staleness + freeze_learning_rate is the relevant engine knob."
    )
    guards.append({"name": "frozen_memory_map", "status": g6_status.value, "reason": g6_reason})

    # ------------------------------------------------------------------
    # Guard 7: shuffle_order
    # ------------------------------------------------------------------
    if getattr(base_cfg, "shuffle_creature_order", False):
        g7_status = GS.PASS
        g7_reason = "base_cfg.shuffle_creature_order=True: id-order eat-first confound is neutralised"
    else:
        g7_status = GS.NOT_IMPLEMENTED
        g7_reason = (
            "base_cfg.shuffle_creature_order=False: id-order eat-first confound is active. "
            "Set shuffle_creature_order=True in the base config to enable this guard."
        )
    guards.append({"name": "shuffle_order", "status": g7_status.value, "reason": g7_reason})

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    terminal_statuses = {GS.PASS.value, GS.FAIL.value}
    all_pass = all(
        g["status"] == GS.PASS.value
        for g in guards
        if g["status"] in terminal_statuses
    )
    pass_names = [g["name"] for g in guards if g["status"] == GS.PASS.value]
    not_impl_names = [g["name"] for g in guards if g["status"] in (GS.NOT_IMPLEMENTED.value, GS.NA.value)]

    return GateOutcome(
        name="null_guards",
        question="Do all null/anti-cheat guards pass?",
        metric="guard battery status (PASS/FAIL/NOT_IMPLEMENTED/NA)",
        raw_rows=[],
        per_seed=[],
        aggregate={"guards": guards, "all_pass": all_pass},
        verdict="",
        validity_flags={"n_guards": len(guards)},
        interpretation=(
            f"Guard battery: PASS={pass_names}; "
            f"NOT_IMPLEMENTED/NA={not_impl_names}.  "
            f"all_pass={all_pass} (counts only PASS/FAIL; NOT_IMPLEMENTED/NA do not fail).  "
            f"A positive gradient verdict should be treated as an artifact if any guard FAILs."
        ),
    )


# ---------------------------------------------------------------------------
# H) Controller cross-partial
# ---------------------------------------------------------------------------

def _corner_births(
    base_cfg: EcologyConfig,
    axis: TraitAxis,
    h: float,
    controller_field: str,
    theta: float,
    seed: int,
    window: int,
) -> float:
    """Births per step in the last `window` steps of a monomorphic run at (h, theta).

    The config is frozen (trait breeds true) and the controller field is set to theta.
    Returns births/window as a proxy for reproductive output at the corner.
    """
    freeze_kw = _freeze_kwargs(axis)
    cfg = D.replace(base_cfg, **{controller_field: theta}, **freeze_kw)
    cfg = D.replace(cfg, founder=axis.clamp_founder(base_cfg.founder, h, axis.inefficiency_value))
    eco = Ecology(cfg, seed=seed)
    while eco.t < cfg.horizon and not eco.exploded:
        eco.step()
        if not eco.has_alive():
            break
    cutoff = cfg.horizon - window
    births = sum(
        1 for e in eco.events
        if e.get("event_type") == "reproduction" and e.get("t", 0) > cutoff
    )
    return births / window if window > 0 else float("nan")


@dataclass(frozen=True)
class ControllerSpec:
    """Specification for a controller axis in the cross-partial gate.

    Fields
    ------
    config_field : EcologyConfig field name for the controller (e.g. "niche_weight")
    low_value    : low controller value (theta_lo)
    high_value   : high controller value (theta_hi)
    """
    config_field: str
    low_value: float
    high_value: float


def run_controller_cross_partial(
    base_cfg: EcologyConfig,
    axis: TraitAxis,
    controller: ControllerSpec,
    seeds: list[int],
    *,
    window: int = 1000,
) -> GateOutcome:
    """Gate H: 2x2 cross-partial of trait h and controller theta on birth rate.

    Measures the four corners:
        B_ll = births(h_lo, theta_lo)
        B_hl = births(h_hi, theta_lo)
        B_lh = births(h_lo, theta_hi)
        B_hh = births(h_hi, theta_hi)

    where h_lo=axis.low_value (or resident_value) and h_hi=axis.high_value.
    The cross-partial eff = corner_effects(B_ll, B_hl, B_lh, B_hh).

    IMPORTANT: This is a PREFLIGHT, not a full co-adaptation run.  The corners
    are measured in isolation; true co-evolutionary dynamics are not captured.

    The corner choice: h_lo=axis.low_value (if not None, else axis.resident_value),
    h_hi=axis.high_value.  This measures the full h range; if you want the local
    (near-resident) cross-partial, set axis.low_value=axis.resident_value.
    """
    h_lo = axis.low_value if axis.low_value is not None else axis.resident_value
    h_hi = axis.high_value if axis.high_value is not None else axis.resident_value
    theta_lo = controller.low_value
    theta_hi = controller.high_value
    ctrl_field = controller.config_field

    corners = {
        "ll": (h_lo, theta_lo),
        "hl": (h_hi, theta_lo),
        "lh": (h_lo, theta_hi),
        "hh": (h_hi, theta_hi),
    }

    raw_rows: list[dict] = []
    corner_vals: dict[str, list[float]] = {"ll": [], "hl": [], "lh": [], "hh": []}

    for seed in seeds:
        for corner, (h, theta) in corners.items():
            b = _corner_births(base_cfg, axis, h, ctrl_field, theta, seed, window)
            corner_vals[corner].append(b)
            raw_rows.append({
                "gate": "controller_cross_partial",
                "seed": seed,
                "corner": corner,
                "h": h,
                "theta": theta,
                "controller_field": ctrl_field,
                "births_per_step": b,
            })

    B_ll = _nanmean(corner_vals["ll"])
    B_hl = _nanmean(corner_vals["hl"])
    B_lh = _nanmean(corner_vals["lh"])
    B_hh = _nanmean(corner_vals["hh"])

    eff = _m.corner_effects(B_ll, B_hl, B_lh, B_hh)
    verdict_enum, verdict_reason = _v.crosspartial_verdict(eff)

    return GateOutcome(
        name="controller_cross_partial",
        question=(
            f"Do trait h and controller {ctrl_field} interact synergistically (cross-partial > 0)?"
        ),
        metric=f"births/step at 4 corners of ({axis.name} x {ctrl_field})",
        raw_rows=raw_rows,
        per_seed=[],
        aggregate={
            "B_ll": B_ll,
            "B_hl": B_hl,
            "B_lh": B_lh,
            "B_hh": B_hh,
            "h_lo": h_lo,
            "h_hi": h_hi,
            "theta_lo": theta_lo,
            "theta_hi": theta_hi,
            "corner_effects": eff,
        },
        verdict=verdict_enum.value,
        validity_flags={"n_seeds": len(seeds), "n_corners": 4},
        interpretation=(
            f"Cross-partial verdict: {verdict_enum.value}.  {verdict_reason}.  "
            f"This is a PREFLIGHT measurement, NOT a full co-adaptation run.  "
            f"The corners are measured in isolation; actual co-evolutionary dynamics "
            f"(e.g. the controller evolving to exploit the trait) are not captured here."
        ),
    )


# ---------------------------------------------------------------------------
# Gate registry
# ---------------------------------------------------------------------------

GATE_REGISTRY: dict[str, Any] = {
    "gifted_benefit": run_gifted_benefit,
    "monomorphic_sweep": run_monomorphic_sweep,
    "local_pairwise_gradient": run_local_pairwise_gradient,
    "invasion_from_rarity": run_invasion_from_rarity,
    "density_independent_growth": run_density_independent_growth,
    "cost_sensitivity": run_cost_sensitivity,
    "null_guards": run_null_guards,
    "controller_cross_partial": run_controller_cross_partial,
}
