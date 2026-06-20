"""experiments/exp257_recalibrate.py — recalibration (pilot seeds only; NOT the verdict).

The first Exp-257 run was NO_VERDICT: the regime was too mild — isolated 8-patch (mig=0, recol=0)
persisted 10/10, so global persistence came from mere replication + the within-patch seasonal pulse,
NOT cross-patch rescue. To make the metapopulation mechanism LOAD-BEARING we need a HARSHER
within-patch regime where:
  - a SINGLE patch (with seasonal modulation) reliably COLLAPSES,
  - 8 ISOLATED patches (mig=0) FAIL globally (no recolonization can save them),
  - but MEDIUM migration + asynchrony RESCUES global persistence (out-of-phase survivors recolonize).
Sweep predation harshness; calibrate on PILOT seeds {0..4} (verdict will use fresh 100-109).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim

HORIZON = 1200
SEEDS = [0, 1, 2, 3, 4]
AMP = 0.4


def persist(cfg_kwargs):
    base = dict(pred_self_limit_hmax=0.05, K_pred_local=40.0, horizon=HORIZON,
                n_prey0_per_patch=40, n_pred0_per_patch=8, async_amplitude=AMP, async_period=50.0)
    base.update(cfg_kwargs)
    cfg = PatchMosaicConfig(**base)
    p = 0
    rec = []
    for s in SEEDS:
        r = PatchMosaicSim(cfg, s).run()
        ok = (not r["global_extinct"]) and r["t_end"] >= HORIZON and r["global_prey_series"][-1] > 0 and r["global_pred_series"][-1] > 0
        p += int(ok); rec.append(r["recolonization_events"])
    return p, np.mean(rec)


def main():
    print(f"recalibration | horizon={HORIZON} seeds={SEEDS} | seeking: single FAIL, isolated FAIL, med-mig PASS")
    print(f"{'attack_a':>8} | {'1patch':>7} {'8p-iso(mig0)':>13} {'8p-medmig':>10}  (persist/5; recol for mosaic)")
    for attack_a in (0.05, 0.08, 0.12, 0.18, 0.25, 0.35):
        p1, _ = persist(dict(n_patches=1, attack_a=attack_a, async_mode="rotating",
                             migration_rate_prey=0.0, migration_rate_pred=0.0, refuge_mode="none"))
        pi, ri = persist(dict(n_patches=8, attack_a=attack_a, async_mode="rotating",
                              migration_rate_prey=0.0, migration_rate_pred=0.0,
                              refuge_mode="per_patch", refuge_predator_access=0.30, refuge_fraction=0.25))
        pm, rm = persist(dict(n_patches=8, attack_a=attack_a, async_mode="rotating",
                              migration_rate_prey=0.05, migration_rate_pred=0.05,
                              refuge_mode="per_patch", refuge_predator_access=0.30, refuge_fraction=0.25))
        flag = "  <-- CANDIDATE (iso fails, med-mig passes)" if (pi < 4 and pm >= 4) else ""
        print(f"{attack_a:>8} | {p1:>7}/5 {pi:>11}/5 {pm:>8}/5  (iso recol={ri:.0f}, mosaic recol={rm:.0f}){flag}")


if __name__ == "__main__":
    main()
