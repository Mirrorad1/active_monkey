"""embodied.run_population — population entrypoint + Preflight stability certify.

Stability gates are applied using FROZEN threshold constants imported from
ecology.evolvability.stability (never hardcoded or loosened here):
    _PERSIST_FLOOR        : min(N(t)) >= 30  (persistence)
    _LEVEL_CV_SEED_MAX    : level_cv(N) <= 0.25  (per-run level stability)
    _DRIFT_MAX            : drift_slope <= 0.10  (no systematic trend)
    oscillation_verdict   : classification == "DAMPED"

The engine module from ecology is NOT imported here (import-boundary guard).

Coupled checks from ecology.evolvability.stability.certify_run that require
the engine module (_marginal_brake / return_map) or ecology-specific telemetry
(availability_mean, boundary_frac, interbump_flux, births_per_step/crowding_per_step)
are NOT applied — those are substrate-specific and the embodied substrate does not
produce them.  We apply the four portable gates above, which are what the Preflight
stability spec calls the "core" gates (persist, level-CV, drift, oscillation).

Usage:
    python -m embodied.run_population [--founders N] [--horizon T] [--seeds S [S ...]]
    active-monkey-embodied-pop [--founders N] [--horizon T] [--seeds S [S ...]]
"""
from __future__ import annotations

import argparse
import pathlib
import textwrap
from typing import Sequence

import numpy as np

from embodied.population import PopConfig, PopResult, run
from embodied.foodfield import FoodFieldConfig

# --- Import FROZEN threshold constants from the Preflight stability instrument ---
# These are the same constants that govern the ecology.evolvability Preflight's
# stability gate.  We import them verbatim; we never re-declare or relax them.
from ecology.evolvability.stability import (
    _PERSIST_FLOOR,         # 30  — min(N) must be >= this
    _LEVEL_CV_SEED_MAX,     # 0.25 — per-run level CV must be <= this
    _DRIFT_MAX,             # 0.10 — fractional drift must be <= this
    # helper functions (pure numpy — no engine dependency)
    persistence,
    level_cv,
    drift_slope,
    n_eq as _n_eq,
    oscillation_verdict,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def certify(result: PopResult) -> dict:
    """Apply the FROZEN Preflight stability gates to a PopResult.

    Applies four portable gates (the engine-independent subset of
    ecology.evolvability.stability.certify_run):
      1. persistence   >= _PERSIST_FLOOR (30)
      2. level_cv      <= _LEVEL_CV_SEED_MAX (0.25)
      3. drift_slope   <= _DRIFT_MAX (0.10)
      4. oscillation_verdict == "DAMPED"

    Also computes a density-dependence boolean: is per-capita intake
    negatively correlated with population size (within-run)?  A negative
    Pearson correlation between N(t) and per_capita_intake(t) is the expected
    density-dependence signal.

    Returns a dict with at least:
        stable (bool)     — True iff all four gates pass
        n_eq (float)      — median N across the run window
        level_cv (float)
        drift (float)
        persistence (float)
        oscillation (str) — "DAMPED" or "OSCILLATORY"
        density_dependent (bool)
        checks (dict)     — per-gate bool
    """
    N = np.asarray(result.n_series, dtype=float)
    pci = np.asarray(result.per_capita_intake, dtype=float)

    eq = _n_eq(N)
    lc = level_cv(N)
    ds = drift_slope(N, eq)
    ps = persistence(N)
    ov = oscillation_verdict(N)

    checks: dict[str, bool] = {
        "persistence": ps >= _PERSIST_FLOOR,
        "level_cv":    lc <= _LEVEL_CV_SEED_MAX,
        "drift":       ds <= _DRIFT_MAX,
        "oscillation": ov["classification"] == "DAMPED",
    }
    stable = all(checks.values())

    # Density-dependence: negative correlation between N(t) and per-capita intake(t).
    # np.corrcoef returns nan if std is 0; treat that as False (no signal).
    if N.std() > 0 and pci.std() > 0:
        corr = float(np.corrcoef(N, pci)[0, 1])
        density_dependent = corr < 0.0
    else:
        corr = float("nan")
        density_dependent = False

    return {
        "stable":           stable,
        "n_eq":             float(eq),
        "level_cv":         float(lc),
        "drift":            float(ds),
        "persistence":      float(ps),
        "oscillation":      ov["classification"],
        "density_dependent": density_dependent,
        "intake_vs_N_corr": corr,
        "checks":           checks,
        "thresholds": {
            "persist_floor":    _PERSIST_FLOOR,
            "level_cv_max":     _LEVEL_CV_SEED_MAX,
            "drift_max":        _DRIFT_MAX,
        },
        "oscillation_detail": ov,
    }


def _run_multi_seed(
    founders: int,
    horizon: int,
    seeds: Sequence[int],
    field: FoodFieldConfig | None = None,
) -> list[tuple[int, PopResult]]:
    if field is None:
        field = FoodFieldConfig(capacity=5.0, regen=0.2)
    results = []
    for s in seeds:
        cfg = PopConfig(
            n_founders=founders,
            horizon=horizon,
            bout_steps=6,
            seed=s,
            field=field,
        )
        results.append((s, run(cfg)))
    return results


def _write_report(
    path: pathlib.Path,
    runs: list[tuple[int, PopResult]],
    verdicts: list[dict],
    founders: int,
    horizon: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.append("=" * 72)
    lines.append("embodied population run — Preflight stability report")
    lines.append("=" * 72)
    lines.append(f"founders={founders}  horizon={horizon}  seeds={[s for s,_ in runs]}")
    lines.append("")

    for (seed, r), v in zip(runs, verdicts):
        lines.append(f"--- seed {seed} ---")
        lines.append(f"  events_hash        : {r.events_hash}")
        lines.append(f"  births / deaths    : {r.births} / {r.deaths}")
        lines.append(f"  final_alive        : {r.final_alive}")
        lines.append(f"  N(t) summary       : min={min(r.n_series)}  max={max(r.n_series)}"
                     f"  mean={np.mean(r.n_series):.1f}")
        lines.append(f"  per_capita_intake  : mean={np.mean(r.per_capita_intake):.4f}"
                     f"  std={np.std(r.per_capita_intake):.4f}")
        lines.append(f"  intake-vs-N corr   : {v['intake_vs_N_corr']:.3f}"
                     f"  density_dependent={v['density_dependent']}")
        lines.append("")
        lines.append("  Preflight stability gates (FROZEN thresholds):")
        for k, ok in v["checks"].items():
            lines.append(f"    {k:<20} {'PASS' if ok else 'FAIL'}")
        lines.append(f"  n_eq={v['n_eq']:.1f}  level_cv={v['level_cv']:.3f}"
                     f"  drift={v['drift']:.3f}  persistence={v['persistence']:.0f}")
        lines.append(f"  oscillation        : {v['oscillation']}")
        lines.append(f"  STABLE             : {v['stable']}")
        lines.append("")

    # Cross-seed summary (if more than one seed)
    if len(verdicts) > 1:
        n_stable = sum(1 for v in verdicts if v["stable"])
        n_eqs = [v["n_eq"] for v in verdicts]
        from ecology.evolvability.stability import seed_agreement, _SEED_AGREE_MAX
        agree = seed_agreement(n_eqs)
        lines.append("--- cross-seed summary ---")
        lines.append(f"  seeds stable       : {n_stable}/{len(verdicts)}")
        lines.append(f"  n_eq per seed      : {[f'{e:.1f}' for e in n_eqs]}")
        lines.append(f"  seed_agreement     : {agree:.3f}"
                     f"  (threshold <= {_SEED_AGREE_MAX})"
                     f"  {'PASS' if agree <= _SEED_AGREE_MAX else 'FAIL'}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")
    print(f"Report written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run embodied population loop + Preflight stability certify",
    )
    parser.add_argument("--founders", type=int, default=10,
                        help="Number of founding creatures (default: 10)")
    parser.add_argument("--horizon", type=int, default=200,
                        help="Steps per run (default: 200)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="RNG seeds to run (default: 0 1 2)")
    args = parser.parse_args()

    field = FoodFieldConfig(capacity=5.0, regen=0.2)
    print(f"Running embodied population: founders={args.founders} "
          f"horizon={args.horizon} seeds={args.seeds}")
    runs = _run_multi_seed(args.founders, args.horizon, args.seeds, field)
    verdicts = [certify(r) for _, r in runs]

    out = pathlib.Path(__file__).parent / "outputs" / "embodied_population.txt"
    _write_report(out, runs, verdicts, args.founders, args.horizon)

    # Print compact summary to stdout
    for (seed, r), v in zip(runs, verdicts):
        status = "STABLE" if v["stable"] else "UNSTABLE"
        print(f"  seed={seed}  n_eq={v['n_eq']:.1f}  {status}"
              f"  density_dep={v['density_dependent']}"
              f"  hash={r.events_hash}")


if __name__ == "__main__":
    main()
