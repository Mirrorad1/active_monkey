"""SCRATCH PILOT9 (deleted) — does a FUNCTIONAL thermosense organ evolve under a foraging (non-saturating) benefit?"""
import dataclasses as D
import numpy as np
from ecology.engine import Ecology
from ecology.scenarios import SCENARIOS, FOUNDER

# Foraging regime: food concentrated in a drifting thermal band; thermosense used to FORAGE; NO avoidance stress.
COMMON = dict(enable_thermosense=True, enable_temperature=True, thermosense_forage_mode=True,
              temperature_stress_scale=0.0, tolerance_cost_scale=0.0,
              enable_food_coupling=True, food_concentration=5.0, food_band_width=0.15,
              food_optimal_base=0.5, food_optimal_amplitude=0.3, food_optimal_period=1500.0,
              thermal_avoidance_weight=2.0, thermosense_upkeep_floor=0.0, thermosense_active_threshold=0.05,
              comfort_amplitude=0.0)
def fnd(i): return D.replace(FOUNDER, thermosense_intensity=i, thermosense_inefficiency=0.2, temperature_tolerance=0.10)

for noise in (0.5, 0.2):
    for fi in (0.0, 0.10):
        cfg = D.replace(SCENARIOS["balanced"], horizon=12000, max_population=5000,
                        founder=fnd(fi), thermosense_noise_base=noise, **COMMON)
        eco = Ecology(cfg, 100); out = []
        while eco.t < cfg.horizon and not eco.exploded:
            eco.step()
            if eco.t % 3000 == 0:
                alive = eco._alive()
                if not alive: out.append(f"t{eco.t}:EXT"); break
                I = np.array([c.genotype.thermosense_intensity for c in alive])
                out.append(f"t{eco.t}: meanI={I.mean():.3f} maxI={I.max():.3f} pop={len(alive)}")
        print(f"noise={noise} founderI={fi}: " + " | ".join(out))
