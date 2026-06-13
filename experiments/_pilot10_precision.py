"""SCRATCH PILOT10 (deleted) — does a PRECISION-demanding foraging regime cross the valley?"""
import dataclasses as D
import numpy as np
from ecology.engine import Ecology
from ecology.scenarios import SCENARIOS, FOUNDER

COMMON = dict(enable_thermosense=True, enable_temperature=True, thermosense_forage_mode=True,
              temperature_stress_scale=0.0, tolerance_cost_scale=0.0, enable_food_coupling=True,
              food_concentration=8.0, food_optimal_base=0.5, thermal_avoidance_weight=4.0,
              thermosense_upkeep_floor=0.0, thermosense_active_threshold=0.05, comfort_amplitude=0.0,
              thermosense_noise_base=0.5)  # high noise_base => intensity strongly improves tracking
def fnd(i): return D.replace(FOUNDER, thermosense_intensity=i, thermosense_inefficiency=0.2, temperature_tolerance=0.10)

# (label, food_band_width, food_optimal_amplitude, food_optimal_period)  -- narrower band + faster drift = more precision needed
REGIMES = [
    ("wide  band.15 amp.3 per1500", 0.15, 0.3, 1500.0),
    ("narrow band.05 amp.3 per1500", 0.05, 0.3, 1500.0),
    ("nar+fast band.05 amp.4 per600", 0.05, 0.4, 600.0),
    ("tiny+fast band.03 amp.4 per400", 0.03, 0.4, 400.0),
]
for label, bw, amp, per in REGIMES:
    for seed in (100, 101):
        cfg = D.replace(SCENARIOS["balanced"], horizon=12000, max_population=5000, founder=fnd(0.10),
                        food_band_width=bw, food_optimal_amplitude=amp, food_optimal_period=per, **COMMON)
        eco = Ecology(cfg, seed); out = []
        while eco.t < cfg.horizon and not eco.exploded:
            eco.step()
            if eco.t % 4000 == 0:
                alive = eco._alive()
                if not alive: out.append(f"t{eco.t}:EXT"); break
                I = np.array([c.genotype.thermosense_intensity for c in alive])
                out.append(f"t{eco.t}: meanI={I.mean():.3f} maxI={I.max():.3f} pop={len(alive)}")
        print(f"{label} seed{seed}: " + " | ".join(out))
