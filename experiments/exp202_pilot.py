"""Exp 202 — DISCLOSED PILOT (seeds {130,131}) — the STRIP-GATE go/no-go for the interference-
competition escape, + de-novo climb probe. Built on ecology/batch.py (parallel).

The Exp 201 audit named the last sensing escape: REAL positive frequency dependence — precision pays
MORE as rivals also sense — via interference competition at a GENUINELY DEPLETING band, with the
ascending-creature_id "eat-first" confound neutralised by a gated SHUFFLED processing order
(shuffle_creature_order; consume() unchanged). Both an independent Codex diagnosis and a Claude
design workflow converged on this mechanism AND on one guardrail: the band must be shown to actually
deplete within a step (strip>0, early AND late) BEFORE a verdict run is meaningful — else it collapses
to exp201 (strip~0) and we should PAUSE and accept the wall (the human's option b).

This pilot sweeps depleting-band regimes under shuffle and reports, per regime:
  strip_late_frac_pos (fraction of late steps where the band lost resource — want > ~0.5),
  occ_late_mean (mean in-band occupants — want > ~1.5: genuinely crowded),
  newborn intensity climb from the 0.10 founder.
GO if a regime strips AND crowds AND shows a climb; NO-GO (recommend pause) if strip stays ~0.
Disclosed per L7: pilot seeds {130,131} are NOT the verdict seeds.
"""
from __future__ import annotations

import dataclasses as D
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ecology.batch import RunSpec, run_batch
from ecology.scenarios import SCENARIOS, FOUNDER

PILOT_SEEDS = [130, 131]
HORIZON = 6000


def _base(conc, band, period, amp=0.3):
    return dict(
        enable_thermosense=True, enable_temperature=True, temperature_stress_scale=0.0,
        tolerance_cost_scale=0.0, comfort_amplitude=0.0, thermosense_upkeep_floor=0.0,
        thermosense_active_threshold=0.05, thermosense_noise_base=0.5, thermal_avoidance_weight=4.0,
        food_optimal_base=0.5, food_optimal_amplitude=amp, food_optimal_period=period,
        food_concentration=conc, food_band_width=band,
        enable_food_coupling=True, thermosense_forage_mode=True,
        shuffle_creature_order=True, track_band_strip=True,
    )


def founder():
    return D.replace(FOUNDER, thermosense_intensity=0.10, thermosense_inefficiency=0.2,
                     temperature_tolerance=0.10)


def cfg(conc, band, period, regen):
    return D.replace(SCENARIOS["balanced"], horizon=HORIZON, max_population=6000,
                     founder=founder(), regen_rate=regen, **_base(conc, band, period))


REGIMES = {
    # label: (conc, band_width, period, regen)
    "deplete_narrow":   (14.0, 0.08, 1500.0, 0.08),
    "deplete_v_narrow": (18.0, 0.05, 1500.0, 0.06),
    "deplete_mid":      (12.0, 0.10, 1500.0, 0.10),
    "deplete_static":   (16.0, 0.06,  1e9,   0.06),   # ~static band (amp still 0.3 but huge period)
    "deplete_fastdrift":(14.0, 0.08,  400.0, 0.08),
}


def main():
    t0 = time.time()
    specs = []
    for label, (conc, band, period, regen) in REGIMES.items():
        for s in PILOT_SEEDS:
            specs.append(RunSpec(key=(label, s), cfg=cfg(conc, band, period, regen), seed=s,
                                 window_start=HORIZON - 2000, checkpoint_stride=2000,
                                 trait_means=("thermosense_intensity",)))
    print(f"Exp 202 PILOT — strip-gate, {len(specs)} runs (parallel), horizon {HORIZON}\n")
    res = run_batch(specs)
    print(f"{'regime':<18}{'seed':>5}{'pop':>7}{'nb_int':>8}{'strip%':>8}{'occ':>7}{'strip_mean':>11}")
    print("-" * 72)
    summ = {}
    for label in REGIMES:
        rows = [res[(label, s)] for s in PILOT_SEEDS]
        for s, r in zip(PILOT_SEEDS, rows):
            nb = r["end_means"]["thermosense_intensity"]
            print(f"{label:<18}{s:>5}{r['final_pop']:>7}{nb:>8.3f}"
                  f"{r.get('strip_late_frac_pos',0)*100:>7.0f}%{r.get('occ_late_mean',0):>7.1f}"
                  f"{r.get('strip_late_mean',0):>11.3f}")
        nbs = [r["end_means"]["thermosense_intensity"] for r in rows]
        summ[label] = (float(np.nanmean(nbs)),
                       float(np.mean([r.get('strip_late_frac_pos', 0) for r in rows])),
                       float(np.mean([r.get('occ_late_mean', 0) for r in rows])))
    print("\n=== GO/NO-GO (want strip%>~50, occ>~1.5, nb climb >0.10) ===")
    for label, (nb, sfp, occ) in sorted(summ.items(), key=lambda kv: -kv[1][1]):
        go = "GO" if (sfp > 0.5 and occ > 1.5) else "no-go"
        print(f"  {label:<18} nb={nb:.3f}  strip_frac={sfp:.2f}  occ={occ:.1f}  -> {go}")
    print(f"\nruntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
