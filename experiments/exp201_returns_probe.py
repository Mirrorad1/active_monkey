"""Exp 201 — RETURNS-CURVE PROBE (diagnostic; probe seeds {120,121}) — is the benefit CONVEX?

Pre-registered DIAGNOSTIC, run + committed BEFORE the verdict. It directly tests the load-bearing
claim of the increasing-returns escape: that the GROSS foraging benefit of tracking precision is
CONVEX (accelerating) in the FAST-band regime but flat in the SLOW regime — and, decisively, whether
that convexity persists ABOVE intensity ~0.25 (where evolution must push to reach functional 0.30).

Method (clean separation of benefit from cost, no engine change): for FROZEN intensity p (enable_
thermosense=False pins p for the whole lineage AND leaves the organ active with NO upkeep charged),
run a monomorphic population and measure GROSS per-capita foraging intake (mean resource_eaten/age over
alive creatures) at equilibrium.  The intensity-keyed band tracker (enable_band_staleness=True) is the
ONLY thing p affects, so intake(p) is the pure benefit curve.  The COST is the KNOWN linear upkeep
0.2*p/step (analytic).  Net fitness signal ∝ intake(p) - 0.2*p.  Increasing returns require
d(intake)/dp to RISE with p (convex) and the net to keep rising across 0.25 -> 0.30.

CONVEXITY test (pre-registered): report d(intake)/dp separately in [0.10,0.25] and [0.25,0.50].
Increasing returns => the upper-range marginal benefit is NOT smaller than the lower-range.
SPEED-GATING: FAST must be more convex / steeper than SLOW (a near-flat SLOW curve = the benefit is
the moving-target threshold, not an artifact).  Diagnostic only; the VERDICT is the evolving gene-pool
metric in exp201_n5_increasing_returns.py (L22: a favorable benefit curve does NOT prove evolvability).
Disclosed per L7: probe seeds {120,121} are NOT the verdict seeds {33-37}.
"""
from __future__ import annotations

import dataclasses as D
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ecology.engine import Ecology, EcologyConfig
from ecology.scenarios import SCENARIOS, FOUNDER

PROBE_SEEDS = [120, 121]
HORIZON = 6000
INTENSITIES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
FAST_PERIOD = 60.0
SLOW_PERIOD = 2400.0
BAND_WIDTH = 0.12
REGEN_RATE = 0.20

BASE = dict(
    enable_thermosense=False,   # FREEZE intensity (pinned per lineage); organ active; NO upkeep
    enable_temperature=True, temperature_stress_scale=0.0, tolerance_cost_scale=0.0,
    comfort_amplitude=0.0, thermosense_upkeep_floor=0.0, thermosense_active_threshold=0.05,
    thermosense_noise_base=0.5, thermal_avoidance_weight=4.0,
    food_optimal_base=0.5, food_optimal_amplitude=0.3, food_concentration=8.0,
    enable_food_coupling=True, thermosense_forage_mode=True, enable_band_staleness=True,
    band_responsiveness=1.0,
)


def cfg(p: float, period: float) -> EcologyConfig:
    f = D.replace(FOUNDER, thermosense_intensity=p, thermosense_inefficiency=0.2, temperature_tolerance=0.10)
    return D.replace(SCENARIOS["balanced"], horizon=HORIZON, max_population=5000, founder=f,
                     regen_rate=REGEN_RATE, food_band_width=BAND_WIDTH, food_optimal_period=period, **BASE)


def gross_intake(p: float, period: float, seed: int) -> float:
    """Mean per-capita foraging intake (resource_eaten/age) over alive creatures, late-window."""
    eco = Ecology(cfg(p, period), seed=seed)
    samples = []
    while eco.t < eco.cfg.horizon and not eco.exploded:
        eco.step()
        if eco.t > eco.cfg.horizon - 1000 and eco.t % 100 == 0:
            rates = [c.phenotype.resource_eaten / max(1, c.phenotype.age) for c in eco._alive()]
            if rates:
                samples.append(float(np.mean(rates)))
        if not eco._alive():
            break
    return float(np.mean(samples)) if samples else float("nan")


def main() -> None:
    t0 = time.time()
    print(f"Exp 201 RETURNS-CURVE PROBE — frozen intensity, gross intake(p), seeds {PROBE_SEEDS}\n")
    curves: dict[str, dict[float, float]] = {"FAST": {}, "SLOW": {}}
    for label, period in [("FAST", FAST_PERIOD), ("SLOW", SLOW_PERIOD)]:
        for p in INTENSITIES:
            vals = [gross_intake(p, period, s) for s in PROBE_SEEDS]
            curves[label][p] = float(np.nanmean(vals))
            print(f"  {label:<5} p={p:.2f}  intake={curves[label][p]:.4f}  net(intake-0.2p)={curves[label][p]-0.2*p:.4f}")

    def slopes(curve):
        lo = (curve[0.25 if 0.25 in curve else 0.30] - curve[0.10]) / (0.20)  # ~lower range
        return lo

    print("\n=== convexity (marginal d(intake)/dp by sub-range) ===")
    summary = {}
    for label in ("FAST", "SLOW"):
        c = curves[label]
        d_lo = (c[0.30] - c[0.10]) / 0.20   # marginal in [0.10,0.30]
        d_hi = (c[0.50] - c[0.30]) / 0.20   # marginal in [0.30,0.50]
        convex_above = d_hi >= d_lo
        summary[label] = {"d_lo_0.10_0.30": d_lo, "d_hi_0.30_0.50": d_hi, "convex_above_0.30": convex_above}
        print(f"  {label}: d(intake)/dp  [0.10-0.30]={d_lo:+.4f}  [0.30-0.50]={d_hi:+.4f}  "
              f"convex_above_0.30={convex_above}")
    print(f"\nruntime: {time.time()-t0:.0f}s")

    out_dir = _REPO / "experiments" / "outputs" / "exp201_n5_increasing_returns"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "returns_probe.json", "w") as f:
        json.dump({"probe_seeds": PROBE_SEEDS, "intensities": INTENSITIES,
                   "curves": {k: {str(p): v for p, v in d.items()} for k, d in curves.items()},
                   "convexity": summary}, f, indent=2)
    print(f"[saved {out_dir / 'returns_probe.json'}]")


if __name__ == "__main__":
    main()
