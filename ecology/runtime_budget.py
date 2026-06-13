"""ecology/runtime_budget.py — a RUNTIME / ALGORITHMIC-COMPLEXITY pre-flight for ecology runs.

Why this exists (the standing rule, human request 2026-06-13; kin of the Exp 202 ABUNDANT-arm
explosion that was dropped for runtime + the engine's O(alive) per-step fix): BEFORE launching a
full experiment batch, do a cheap algorithmic-complexity / runtime check so a BUG (an unbounded
population, an accidental super-linear per-step cost, a no-scarcity regime that grows toward the
runaway cap) cannot quietly burn hours of compute. The check is mechanical and conservative — it
PROBES each distinct config at a short horizon, then PROJECTS population growth and wall-clock to
the full horizon and flags three failure modes:

  EXPLOSION    population is still growing geometrically at probe end -> projected pop approaches /
               exceeds max_population (the runaway guard), OR the probe already exploded.
  SUPERLINEAR  the per-creature-per-step cost RISES across the probe (second half slower per
               creature than first half) -> the inner loop is worse than O(alive) (e.g. an
               O(alive^2) or O(total-ever-born) regression) -> cost blows up as the pop grows.
  OVER_BUDGET  the projected total wall-clock (jobs / workers * per-job time) exceeds the budget.

It is ADVISORY by default (returns a report) and can be made BLOCKING (raise) via require_safe.
This is a heuristic guard, not a proof: it catches the gross explosions, not subtle constant
factors. Run it on the EXPERIMENT'S OWN configs, not a toy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ecology.engine import Ecology, EcologyConfig


@dataclass
class ProbeResult:
    label: str
    pops: list[int]                # population at each quarter [q1, q2, q3, q4]
    growth_first: float            # pop_q2 / pop_q1 (early growth factor per quarter)
    growth_last: float             # pop_q4 / pop_q3 (late growth factor per quarter)
    decelerating: bool             # growth_last < growth_first (logistic / carrying-capacity bound)
    per_creature_us_early: float   # microseconds / creature-step in the 1st half
    per_creature_us_late: float    # microseconds / creature-step in the 2nd half
    probe_steps: int
    probe_time_s: float
    exploded: bool

    @property
    def pop_end(self) -> int:
        return self.pops[-1] if self.pops else 0


def probe_config(label: str, cfg: EcologyConfig, seed: int, probe_steps: int = 800) -> ProbeResult:
    """Run ONE short probe; sample population per QUARTER (to see growth deceleration) and the
    per-creature-step cost in each half (to catch a worse-than-O(alive) inner loop)."""
    eco = Ecology(cfg, seed=seed)
    qs = [max(1, probe_steps * k // 4) for k in (1, 2, 3, 4)]
    pops = [0, 0, 0, 0]
    mid = qs[1]
    t_first = t_second = 0.0
    cs_first = cs_second = 0
    while eco.t < probe_steps and not eco.exploded:
        n = len(eco._alive())
        t0 = time.perf_counter()
        eco.step()
        dt = time.perf_counter() - t0
        if eco.t <= mid:
            t_first += dt; cs_first += n
        else:
            t_second += dt; cs_second += n
        for i, q in enumerate(qs):
            if eco.t == q:
                pops[i] = len(eco._alive())
        if not eco._alive():
            break
    # growth factor per quarter (guard against zero)
    def _gf(a: int, b: int) -> float:
        return (b / a) if a > 0 else (1.0 if b == 0 else float("inf"))
    growth_first = _gf(pops[0], pops[1])
    growth_last = _gf(pops[2], pops[3])
    decelerating = growth_last <= max(1.05, 0.85 * growth_first)
    us_early = (t_first / cs_first * 1e6) if cs_first else 0.0
    us_late = (t_second / cs_second * 1e6) if cs_second else 0.0
    return ProbeResult(label, pops, growth_first, growth_last, decelerating, us_early, us_late,
                       eco.t, t_first + t_second, eco.exploded)


def _project_pop(pr: ProbeResult, horizon: int, guard: int) -> tuple[float, bool]:
    """Project the equilibrium/peak population and whether it is a runaway.

    Logistic-aware: if growth is DECELERATING (carrying-capacity bound) the population plateaus —
    extrapolate the remaining quarters with the GEOMETRICALLY-SHRINKING growth factor (a bounded
    sum), NOT naive constant-geometric (which over-predicts the exponential phase to the guard).
    If growth is NOT decelerating (constant/accelerating), it is a true runaway -> project to the
    guard and flag.
    """
    remaining_quarters = max(0.0, (horizon - pr.probe_steps) / max(1, pr.probe_steps // 4))
    if pr.exploded:
        return float(guard), True
    if pr.decelerating:
        # remaining growth factors shrink each quarter by `decay` (the observed deceleration);
        # the cumulative multiplier converges (bounded sum), so the projection plateaus.
        g = max(1.0, pr.growth_last)
        decay = (pr.growth_last / pr.growth_first) if pr.growth_first > 1e-9 else 0.5
        decay = min(0.95, max(0.0, decay))
        mult, gk = 1.0, g
        for _ in range(int(min(remaining_quarters, 40))):
            mult *= gk
            gk = 1.0 + (gk - 1.0) * decay        # growth factor decays toward 1 (plateau)
            if gk <= 1.0001:
                break
        proj = pr.pop_end * mult
        return min(proj, float(guard)), proj >= 0.8 * guard
    # not decelerating -> runaway: constant-geometric to the guard
    proj = pr.pop_end * (max(1.0, pr.growth_last) ** remaining_quarters)
    return min(proj, float(guard)), proj >= 0.8 * guard


def preflight(configs: list[tuple[str, EcologyConfig, int]], *, horizon: int, n_jobs: int,
              max_workers: int, probe_steps: int = 800, time_budget_s: float = 2400.0,
              require_safe: bool = False) -> dict[str, Any]:
    """Pre-flight a planned batch. `configs` are REPRESENTATIVE (label, cfg, seed) — one per
    distinct regime (the cheapest / highest-regen, most explosion-prone arms).

    Returns a report dict with per-config projections + a `safe` bool + `flags`. With
    require_safe=True it raises AssertionError on any flag (BLOCKING pre-flight).
    """
    reports: list[dict[str, Any]] = []
    flags: list[str] = []
    worst_per_job_s = 0.0
    for label, cfg, seed in configs:
        pr = probe_config(label, cfg, seed, probe_steps)
        proj_pop, runaway = _project_pop(pr, horizon, cfg.max_population)
        # conservative per-job time: assume the run sits near its projected PLATEAU pop for the
        # full horizon (over-estimates slightly — the safe direction for a budget guard).
        avg_pop = max(pr.pop_end, proj_pop)
        per_job_s = (pr.per_creature_us_late / 1e6) * avg_pop * horizon
        worst_per_job_s = max(worst_per_job_s, per_job_s)
        superlinear = pr.per_creature_us_late > 2.0 * max(1e-9, pr.per_creature_us_early) and pr.pop_end > 50
        r = {"label": label, "pops": pr.pops, "growth_first": round(pr.growth_first, 3),
             "growth_last": round(pr.growth_last, 3), "decelerating": pr.decelerating,
             "proj_pop": int(proj_pop), "max_population": cfg.max_population,
             "us_per_creature_early": round(pr.per_creature_us_early, 3),
             "us_per_creature_late": round(pr.per_creature_us_late, 3),
             "superlinear": superlinear, "explosion_risk": runaway,
             "proj_per_job_s": round(per_job_s, 1), "exploded": pr.exploded}
        reports.append(r)
        if runaway:
            flags.append(f"EXPLOSION[{label}]: proj_pop {int(proj_pop)} -> guard {cfg.max_population} "
                         f"(growth/quarter {pr.growth_first:.2f}->{pr.growth_last:.2f}, "
                         f"{'NOT ' if not pr.decelerating else ''}decelerating)")
        if superlinear:
            flags.append(f"SUPERLINEAR[{label}]: per-creature cost {pr.per_creature_us_early:.2f}->"
                         f"{pr.per_creature_us_late:.2f} us (inner loop worse than O(alive)?)")
    proj_total_s = (n_jobs / max(1, max_workers)) * worst_per_job_s
    if proj_total_s > time_budget_s:
        flags.append(f"OVER_BUDGET: projected {proj_total_s/60:.1f} min > budget {time_budget_s/60:.1f} min")
    report = {"safe": len(flags) == 0, "flags": flags, "configs": reports,
              "n_jobs": n_jobs, "max_workers": max_workers, "horizon": horizon,
              "proj_total_min": round(proj_total_s / 60, 1)}
    if require_safe and flags:
        raise AssertionError("runtime pre-flight FAILED:\n  " + "\n  ".join(flags))
    return report


def format_report(report: dict[str, Any]) -> str:
    """One-screen human summary of a preflight() report."""
    L = [f"RUNTIME PRE-FLIGHT — {'SAFE' if report['safe'] else 'FLAGS RAISED'} "
         f"(jobs={report['n_jobs']}, workers={report['max_workers']}, horizon={report['horizon']}, "
         f"proj~{report['proj_total_min']} min)"]
    L.append(f"  {'regime':<14}{'pop_end':>8}{'g1->gN':>12}{'decel':>7}{'proj_pop':>9}"
             f"{'us/cr early':>12}{'us/cr late':>11}{'~per-job s':>11}")
    for r in report["configs"]:
        L.append(f"  {r['label']:<14}{r['pops'][-1]:>8}"
                 f"{(str(r['growth_first'])+'->'+str(r['growth_last'])):>12}{str(r['decelerating']):>7}"
                 f"{r['proj_pop']:>9}{r['us_per_creature_early']:>12}{r['us_per_creature_late']:>11}"
                 f"{r['proj_per_job_s']:>11}")
    if report["flags"]:
        L.append("  FLAGS:")
        L += [f"    - {f}" for f in report["flags"]]
    return "\n".join(L)
