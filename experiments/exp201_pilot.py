"""Exp 201 — DISCLOSED PILOT v2 (seeds {110,111}) — full-horizon asymptote + scarcity levers.

Pilot v1 (horizon 4000) found a REAL speed-gated climb (fast band ~0.18 vs slow null ~0.096) that
plateaus in the MIXED zone, with strip~0 (interference competition not biting). This v2 asks the
decisive questions BEFORE fixing the regime: (1) does the climb keep rising to t=12000 (the real
horizon) toward functional 0.30, or plateau? (2) do scarcity/concentration/responsiveness levers
push the threshold up to give the functional line a fair shot? (3) does the SLOW null stay primitive
at full horizon? The chosen FAST-SCARCE regime + SLOW/MEDIUM ladder is fixed in the pre-registration
BEFORE the fresh-seed verdict.  Disclosed per L7: pilot seeds {110,111} are NOT the verdict seeds.
"""
from __future__ import annotations

import dataclasses as D
import math
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ecology.engine import Ecology, EcologyConfig
from ecology.scenarios import SCENARIOS, FOUNDER

PILOT_SEEDS = [110, 111]
HORIZON = 12000
MAX_POP = 5000
WINDOW = 2000  # newborn window = [HORIZON-WINDOW, HORIZON]


def _base(conc=8.0, resp=1.0):
    return dict(
        enable_thermosense=True, enable_temperature=True, temperature_stress_scale=0.0,
        tolerance_cost_scale=0.0, comfort_amplitude=0.0, thermosense_upkeep_floor=0.0,
        thermosense_active_threshold=0.05, thermosense_noise_base=0.5, thermal_avoidance_weight=4.0,
        food_optimal_base=0.5, food_optimal_amplitude=0.3, food_concentration=conc,
        enable_food_coupling=True, thermosense_forage_mode=True, enable_band_staleness=True,
        band_responsiveness=resp,
    )


def founder():
    return D.replace(FOUNDER, thermosense_intensity=0.10, thermosense_inefficiency=0.2,
                     temperature_tolerance=0.10)


def cfg(period, band, regen, conc=8.0, resp=1.0):
    return D.replace(SCENARIOS["balanced"], horizon=HORIZON, max_population=MAX_POP,
                     founder=founder(), regen_rate=regen, food_band_width=band,
                     food_optimal_period=period, **_base(conc, resp))


def newborn_mean(creatures, lo, hi):
    nb = [c for c in creatures if c.parent_id is not None and lo <= c.phenotype.birth_t <= hi]
    return float(np.mean([c.genotype.thermosense_intensity for c in nb])) if nb else float("nan")


def run(cfg_, seed):
    eco = Ecology(cfg_, seed=seed)
    traj = []
    strip = []
    while eco.t < cfg_.horizon and not eco.exploded:
        eco.step()
        if eco.t % 2000 == 0:
            traj.append((eco.t, newborn_mean(eco._creatures, eco.t - 2000, eco.t), len(eco._alive())))
        if eco.t > cfg_.horizon - 1000 and eco.t % 50 == 0:
            w = eco.world
            ib = np.abs(w.temperature - w.current_food_optimal) <= w.food_band_width
            if ib.any():
                strip.append(float(np.mean(w.resource[ib] < 0.5)))
        if not eco._alive():
            break
    alive = eco._alive()
    return {
        "pop": len(alive),
        "nb": newborn_mean(eco._creatures, cfg_.horizon - WINDOW, cfg_.horizon),
        "max_ever": max((c.genotype.thermosense_intensity for c in eco._creatures), default=float("nan")),
        "strip": float(np.mean(strip)) if strip else float("nan"),
        "extinct": len(alive) == 0,
        "traj": traj,
    }


REGIMES = {
    "A_base_p60_w12":      dict(period=60.0,  band=0.12, regen=0.20, conc=8.0,  resp=1.0),
    "G_scarce_conc":       dict(period=60.0,  band=0.12, regen=0.12, conc=14.0, resp=1.0),
    "H_lowresp_p60":       dict(period=60.0,  band=0.12, regen=0.20, conc=8.0,  resp=0.6),
    "I_scarce_narrow":     dict(period=60.0,  band=0.10, regen=0.12, conc=12.0, resp=1.0),
    "SLOW_null_p2400":     dict(period=2400.0,band=0.12, regen=0.20, conc=8.0,  resp=1.0),
}


def main():
    t0 = time.time()
    print(f"Exp 201 PILOT v2 — full-horizon {HORIZON}, seeds {PILOT_SEEDS}\n")
    print(f"{'regime':<20}{'seed':>5}{'pop':>7}{'nb':>9}{'max':>8}{'strip':>7}   trajectory(t:nb)")
    print("-" * 100)
    summ = {}
    for label, p in REGIMES.items():
        nbs = []
        for seed in PILOT_SEEDS:
            r = run(cfg(**p), seed)
            nbs.append(r["nb"])
            tj = " ".join(f"{t//1000}k:{(nb if not math.isnan(nb) else 0):.2f}" for t, nb, _ in r["traj"])
            nbstr = "EXT" if r["extinct"] else f"{r['nb']:.4f}"
            print(f"{label:<20}{seed:>5}{r['pop']:>7}{nbstr:>9}{r['max_ever']:>8.3f}{r['strip']:>7.2f}   {tj}")
        vals = [x for x in nbs if not math.isnan(x)]
        summ[label] = float(np.mean(vals)) if vals else float("nan")
    print("\n=== mean newborn intensity by regime (does it reach functional 0.30?) ===")
    for label, m in sorted(summ.items(), key=lambda kv: -(kv[1] if not math.isnan(kv[1]) else -1)):
        print(f"  {label:<20} {m:.4f}")
    print(f"\nruntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
