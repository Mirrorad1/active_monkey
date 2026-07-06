"""experiments/exp257_pilot.py — PILOT (calibration only, not the verdict).

The patch-mosaic uses the wellmixed within-patch dynamics, which COEXIST by default (Exp 255).
The metapopulation question is only meaningful in a within-patch regime where a SINGLE/synchronized
arena BOOM-BUSTS (paradox of enrichment) — then we test whether patch asynchrony+migration rescue
GLOBAL persistence. This pilot finds such a collapse-prone regime and sanity-checks that:
  (i) a single patch (n_patches=1) collapses/boom-busts there, and
  (ii) an 8-patch mosaic with asynchrony + intermediate migration persists globally there.
RAW — calibration for the controller's Exp 257 sweep.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataclasses as D
import numpy as np
from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim

HORIZON = 800
SEEDS = [0, 1, 2]


def summarize(cfg, seed):
    r = PatchMosaicSim(cfg, seed).run()
    return r


def single_patch(attack_a, K_pred, hmax, seed):
    cfg = PatchMosaicConfig(n_patches=1, attack_a=attack_a, K_pred_local=K_pred,
                            pred_self_limit_hmax=hmax, migration_rate_prey=0.0,
                            migration_rate_pred=0.0, async_mode="synchronized",
                            async_amplitude=0.0, refuge_mode="none", horizon=HORIZON,
                            n_prey0_per_patch=40, n_pred0_per_patch=8)
    return summarize(cfg, seed)


def mosaic(attack_a, K_pred, hmax, seed, mig=0.05, amp=0.4):
    cfg = PatchMosaicConfig(n_patches=8, attack_a=attack_a, K_pred_local=K_pred,
                            pred_self_limit_hmax=hmax, migration_rate_prey=mig,
                            migration_rate_pred=mig, async_mode="rotating",
                            async_amplitude=amp, async_period=50.0, refuge_mode="per_patch",
                            refuge_predator_access=0.3, refuge_fraction=0.25, horizon=HORIZON,
                            n_prey0_per_patch=40, n_pred0_per_patch=8)
    return summarize(cfg, seed)


def main():
    print("=== PILOT: find a collapse-prone within-patch regime (single patch) ===")
    print(f"{'attack_a':>8} {'K_pred':>7} {'hmax':>5} | single-patch: persisted? prey_end/pred_end (mean over seeds)")
    regimes = []
    for attack_a in (0.02, 0.05, 0.1):
        for K_pred in (40, 120, 400):
            for hmax in (0.15, 0.05):
                rs = [single_patch(attack_a, K_pred, hmax, s) for s in SEEDS]
                persisted = sum(1 for r in rs if not r["global_extinct"] and r["t_end"] >= HORIZON)
                prey_end = np.mean([r["global_prey_series"][-1] for r in rs])
                pred_end = np.mean([r["global_pred_series"][-1] for r in rs])
                cvp = np.mean([r["cv_global_prey"] for r in rs if r["cv_global_prey"] == r["cv_global_prey"]]) if any(r["cv_global_prey"]==r["cv_global_prey"] for r in rs) else float('nan')
                tag = "COLLAPSE" if persisted < len(SEEDS) else ("BOOMBUST" if cvp > 0.5 else "stable")
                print(f"{attack_a:>8} {K_pred:>7} {hmax:>5} | persisted {persisted}/{len(SEEDS)} prey={prey_end:.0f} pred={pred_end:.0f} cv={cvp:.2f} -> {tag}")
                if tag in ("COLLAPSE", "BOOMBUST"):
                    regimes.append((attack_a, K_pred, hmax, tag, persisted))
    print()
    print("=== for the most collapse-prone regimes, does an 8-patch async+migration MOSAIC rescue? ===")
    # pick up to 3 collapse-prone regimes
    cand = sorted(regimes, key=lambda x: x[4])[:3]
    for attack_a, K_pred, hmax, tag, sp in cand:
        rs_single = [single_patch(attack_a, K_pred, hmax, s) for s in SEEDS]
        rs_mosaic = [mosaic(attack_a, K_pred, hmax, s) for s in SEEDS]
        sp_pers = sum(1 for r in rs_single if not r["global_extinct"] and r["t_end"] >= HORIZON)
        mo_pers = sum(1 for r in rs_mosaic if not r["global_extinct"] and r["t_end"] >= HORIZON)
        mo_localext = np.mean([r["local_extinction_events"] for r in rs_mosaic])
        mo_recol = np.mean([r["recolonization_events"] for r in rs_mosaic])
        sync = np.mean([PatchMosaicSim.cross_patch_synchrony(r["patch_prey_series"]) for r in rs_mosaic if len(r["patch_prey_series"]) > 1])
        print(f"  regime a={attack_a} K_pred={K_pred} hmax={hmax} ({tag}): "
              f"SINGLE persisted {sp_pers}/{len(SEEDS)} | MOSAIC persisted {mo_pers}/{len(SEEDS)} "
              f"(local_ext={mo_localext:.0f} recol={mo_recol:.0f} sync={sync:.2f})")
    print("\n[pilot done — controller picks the regime where SINGLE collapses but MOSAIC rescues]")


if __name__ == "__main__":
    main()
