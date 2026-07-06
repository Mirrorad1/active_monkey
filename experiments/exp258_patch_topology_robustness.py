"""experiments/exp258_patch_topology_robustness.py — Exp 258: is the Exp-257 patch-mosaic
coexistence rescue ROBUST across topology and scale, or a ring-8 knife-edge?

Exp 257 (ring, 8 patches) posed discrete predator-prey coexistence via two load-bearing mechanisms:
REFUGE and MIGRATION+ASYNCHRONY (clean signal: MIG_ONLY_ASYNC persists, MIG_ONLY_SYNC collapses).
Rung-3 of the patch-mosaic-red-queen ladder: does that pattern HOLD across patch topology
{ring, grid2d, smallworld} x scale {8, 16, 32 patches}? FRESH seeds 200-209 (Exp 257 used 100-109).

PREDECLARED ROBUSTNESS VERDICT: ROBUST iff, at EVERY (topology, n_patches) cell:
  - NEITHER (no refuge, no migration) COLLAPSES (< PASS_SEEDS) [baseline failure preserved], AND
  - MIG_ONLY_SYNC COLLAPSES [synchrony kills the migration rescue everywhere], AND
  - the rescue holds: MIG_ONLY_ASYNC PASSES (>= PASS_SEEDS) [migration+asynchrony rescue is
    topology/scale-general] AND REFUGE_ONLY PASSES AND BOTH PASSES.
PREDECLARED FALSIFIER: if the rescue (MIG_ONLY_ASYNC / REFUGE_ONLY / BOTH) FAILS at any topology or
scale, OR the SYNC control PASSES somewhere, the Exp-257 result is topology/scale-specific
(knife-edge) -> robustness NOT supported (report which cells break).
PASS_SEEDS = 8/10. Collapse-prone within-patch regime as Exp 257 (attack_a=0.05, K_pred_local=40,
pred_self_limit_hmax=0.05). RAW NUMBERS — controller judges.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim

HORIZON = 1000
SEEDS = list(range(200, 210))
PASS_SEEDS = 8
TOPOS = [("ring", {}), ("grid2d", {"grid_cols": 4}), ("smallworld", {"smallworld_rewire": 0.2})]
NPATCHES = [8, 16, 32]
REGIME = dict(attack_a=0.05, K_pred_local=40.0, pred_self_limit_hmax=0.05, async_period=50.0,
              n_prey0_per_patch=40, n_pred0_per_patch=8, horizon=HORIZON)
AMP, MIG, REF_ACCESS, REF_FRAC = 0.4, 0.05, 0.30, 0.25

CELLS = {
 "NEITHER":        dict(async_mode="rotating", async_amplitude=AMP, migration_rate_prey=0.0, migration_rate_pred=0.0, refuge_mode="none"),
 "REFUGE_ONLY":    dict(async_mode="rotating", async_amplitude=AMP, migration_rate_prey=0.0, migration_rate_pred=0.0, refuge_mode="per_patch", refuge_predator_access=REF_ACCESS, refuge_fraction=REF_FRAC),
 "MIG_ONLY_ASYNC": dict(async_mode="rotating", async_amplitude=AMP, migration_rate_prey=MIG, migration_rate_pred=MIG, refuge_mode="none"),
 "MIG_ONLY_SYNC":  dict(async_mode="synchronized", async_amplitude=AMP, migration_rate_prey=MIG, migration_rate_pred=MIG, refuge_mode="none"),
 "BOTH":           dict(async_mode="rotating", async_amplitude=AMP, migration_rate_prey=MIG, migration_rate_pred=MIG, refuge_mode="per_patch", refuge_predator_access=REF_ACCESS, refuge_fraction=REF_FRAC),
}


def persist(topo, topo_extra, n, cell):
    kw = dict(REGIME); kw.update(topo_extra); kw.update(CELLS[cell])
    kw.update(topology=topo, n_patches=n)
    cfg = PatchMosaicConfig(**kw)
    p = 0
    for s in SEEDS:
        r = PatchMosaicSim(cfg, s).run()
        if (not r["global_extinct"]) and r["t_end"] >= HORIZON and r["global_prey_series"][-1] > 0 and r["global_pred_series"][-1] > 0:
            p += 1
    return p


def main():
    L = []
    L.append("=" * 100)
    L.append("Exp 258 — patch-mosaic ROBUSTNESS across topology x scale. RAW — controller judges.")
    L.append(f"regime attack_a=0.05 K_pred=40 hmax=0.05; FRESH seeds {SEEDS}; PASS={PASS_SEEDS}/10; grid_cols=4, smallworld_rewire=0.2")
    L.append("=" * 100)
    header = f"{'cell':>16} | " + " | ".join(f"{t}-{n}" for t, _ in TOPOS for n in NPATCHES)
    L.append(header); L.append("-" * len(header))
    grid = {}
    for cell in ["NEITHER", "MIG_ONLY_SYNC", "MIG_ONLY_ASYNC", "REFUGE_ONLY", "BOTH"]:
        row = []
        for t, extra in TOPOS:
            for n in NPATCHES:
                p = persist(t, extra, n, cell)
                grid[(cell, t, n)] = p
                row.append(f"{p:>{len(t)+3}}")
        L.append(f"{cell:>16} | " + " | ".join(row))
    L.append("")

    # robustness verdict
    breaks = []
    for t, _ in TOPOS:
        for n in NPATCHES:
            if grid[("NEITHER", t, n)] >= PASS_SEEDS:
                breaks.append(f"NEITHER passed at {t}-{n} (baseline did not collapse)")
            if grid[("MIG_ONLY_SYNC", t, n)] >= PASS_SEEDS:
                breaks.append(f"MIG_ONLY_SYNC passed at {t}-{n} (sync should kill the rescue)")
            for resc in ("MIG_ONLY_ASYNC", "REFUGE_ONLY", "BOTH"):
                if grid[(resc, t, n)] < PASS_SEEDS:
                    breaks.append(f"{resc} FAILED at {t}-{n} (rescue did not hold: {grid[(resc,t,n)]}/10)")
    L.append("ROBUSTNESS CHECKS (each rescue holds; baseline+sync collapse) across ALL topology x scale:")
    if not breaks:
        L.append("  ALL cells consistent with Exp 257: NEITHER+MIG_ONLY_SYNC collapse; MIG_ONLY_ASYNC+REFUGE_ONLY+BOTH rescue.")
        L.append("  VERDICT: ROBUST — the patch-mosaic coexistence rescue is topology- AND scale-general (NOT a ring-8 knife-edge).")
    else:
        L.append(f"  {len(breaks)} break(s) — robustness NOT clean:")
        for b in breaks:
            L.append(f"    - {b}")
        L.append("  VERDICT: PARTIAL/topology-or-scale-specific — controller adjudicates which mechanism is fragile.")
    out = "\n".join(L)
    print(out)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
              "experiments", "outputs", "exp258_patch_topology_robustness.txt"), "w") as f:
        f.write(out + "\n")
    print("\n[written to experiments/outputs/exp258_patch_topology_robustness.txt]")


if __name__ == "__main__":
    main()
