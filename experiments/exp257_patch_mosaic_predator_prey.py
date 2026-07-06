"""experiments/exp257_patch_mosaic_predator_prey.py — Exp 257: does a discrete patch mosaic POSE
global predator-prey coexistence where a single homogeneous arena cannot — and WHICH mechanism is
load-bearing?

CONTEXT: the homogeneous spatial-agent predator-prey chapter (Exp 248-254c) closed CAN'T-POSE via
substrate destabilization; the well-mixed Bazykin coexists (Exp 255). NEW HYPOTHESIS: a single
homogeneous arena is too globally synchronized; a discrete-agent PATCH MOSAIC (semi-isolated patches
+ local migration + per-patch asynchrony + refugia) may achieve GLOBAL bounded coexistence even when
LOCAL patches collapse, via metapopulation rescue / refugia.

DESIGN (mechanism-isolating factorial — refined after pilot+control scrutiny that ruled out a SCALE
confound and showed asynchrony ALONE does not rescue). Regime calibrated on PILOT seeds; this VERDICT
run uses FRESH seeds 100-109 (no tuning on test). Collapse-prone within-patch regime: attack_a=0.05,
K_pred_local=40, pred_self_limit_hmax=0.05 (a single patch over-exploits -> PREY_COLLAPSE).

PRE-REGISTERED CONTROLS + FALSIFIER:
  - BASELINE_1P (single patch, scaled to the mosaic's TOTAL population) must COLLAPSE (rules out: it's
    just patch-count/scale). Scale control already showed 1x..16x single patches all collapse.
  - NEITHER (8 patches, NO refuge, NO migration) must COLLAPSE (so a mosaic mechanism is NECESSARY).
  - A mechanism is LOAD-BEARING iff turning it ON (with the others off) flips COLLAPSE -> PERSIST.
  FALSIFIER: if NEITHER persists (mosaic mechanism not necessary) OR no single mechanism flips the
  outcome (nothing load-bearing) OR the single-patch baseline persists (scale, not structure) -> the
  patch mosaic does not cleanly pose coexistence here -> NO_VERDICT / FAIL.
PASS_SEEDS = 8/10 seeds with BOTH species alive at horizon, bounded (no explosion), with local
extinction + recolonization turnover (proves metapopulation dynamics, not a hidden fixed point).

VERDICT LABELS: CAN_POSE_GLOBAL_COHABITATION / PATCH_MOSAIC_RESCUE_SUPPORTED / SYNCHRONIZED_COLLAPSE /
PREDATOR_STARVATION / PREY_COLLAPSE / BOOM_BUST_UNBOUNDED / NO_VERDICT.
RAW NUMBERS — controller applies the binding verdict.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim

HORIZON = 1200
SEEDS = list(range(100, 110))
PASS_SEEDS = 8
REGIME = dict(attack_a=0.05, K_pred_local=40.0, pred_self_limit_hmax=0.05,
              async_period=50.0, horizon=HORIZON)
MIG_MED, MIG_HIGH, AMP = 0.05, 0.30, 0.4
REF_ACCESS, REF_FRAC = 0.30, 0.25

# cell -> config kwargs. Each isolates a mechanism. async on (rotating) unless noted.
CELLS = {
 "BASELINE_1P":  dict(n_patches=1, n_prey0_per_patch=320, n_pred0_per_patch=64, K_prey_local=2400.0,
                      K_pred_local=320.0, async_mode="rotating", async_amplitude=AMP,
                      migration_rate_prey=0.0, migration_rate_pred=0.0, refuge_mode="none"),
 "NEITHER":      dict(n_patches=8, async_mode="rotating", async_amplitude=AMP,
                      migration_rate_prey=0.0, migration_rate_pred=0.0, refuge_mode="none"),
 "REFUGE_ONLY":  dict(n_patches=8, async_mode="rotating", async_amplitude=AMP,
                      migration_rate_prey=0.0, migration_rate_pred=0.0,
                      refuge_mode="per_patch", refuge_predator_access=REF_ACCESS, refuge_fraction=REF_FRAC),
 "MIG_ONLY_ASYNC": dict(n_patches=8, async_mode="rotating", async_amplitude=AMP,
                      migration_rate_prey=MIG_MED, migration_rate_pred=MIG_MED, refuge_mode="none"),
 "MIG_ONLY_SYNC": dict(n_patches=8, async_mode="synchronized", async_amplitude=AMP,
                      migration_rate_prey=MIG_MED, migration_rate_pred=MIG_MED, refuge_mode="none"),
 "BOTH":         dict(n_patches=8, async_mode="rotating", async_amplitude=AMP,
                      migration_rate_prey=MIG_MED, migration_rate_pred=MIG_MED,
                      refuge_mode="per_patch", refuge_predator_access=REF_ACCESS, refuge_fraction=REF_FRAC),
 "HIGH_MIG":     dict(n_patches=8, async_mode="rotating", async_amplitude=AMP,
                      migration_rate_prey=MIG_HIGH, migration_rate_pred=MIG_HIGH, refuge_mode="none"),
 "STRONG_REFUGE": dict(n_patches=8, async_mode="rotating", async_amplitude=AMP,
                      migration_rate_prey=MIG_MED, migration_rate_pred=MIG_MED,
                      refuge_mode="per_patch", refuge_predator_access=0.05, refuge_fraction=0.50),
}
N0 = dict(n_prey0_per_patch=40, n_pred0_per_patch=8)  # default per-patch founders (BASELINE_1P overrides)


def run_cell(name):
    kw = dict(REGIME); kw.update(N0); kw.update(CELLS[name])
    cfg = PatchMosaicConfig(**kw)
    rows = []
    for s in SEEDS:
        r = PatchMosaicSim(cfg, s).run()
        pe, qe = r["global_prey_series"][-1], r["global_pred_series"][-1]
        both = (not r["global_extinct"]) and r["t_end"] >= HORIZON and pe > 0 and qe > 0
        sync = PatchMosaicSim.cross_patch_synchrony(r["patch_prey_series"]) if len(r["patch_prey_series"]) > 1 else float("nan")
        rows.append(dict(both=both, pe=pe, qe=qe, exploded=r["exploded"],
                         le=r["local_extinction_events"], rc=r["recolonization_events"],
                         sync=sync, cvp=r["cv_global_prey"], cvq=r["cv_global_pred"]))
    return dict(name=name, persist=sum(x["both"] for x in rows),
                prey_pers=sum(x["pe"] > 0 for x in rows), pred_pers=sum(x["qe"] > 0 for x in rows),
                exploded=any(x["exploded"] for x in rows),
                le=np.mean([x["le"] for x in rows]), rc=np.mean([x["rc"] for x in rows]),
                sync=np.nanmean([x["sync"] for x in rows]),
                cvp=np.nanmean([x["cvp"] for x in rows]), cvq=np.nanmean([x["cvq"] for x in rows]))


def main():
    L = []
    L.append("=" * 110)
    L.append("Exp 257 — PATCH-MOSAIC predator-prey: mechanism-isolating factorial. RAW — controller judges.")
    L.append(f"collapse regime attack_a=0.05 K_pred=40 hmax=0.05; FRESH seeds {SEEDS}; PASS = both persist >= {PASS_SEEDS}/10 + bounded + local turnover")
    L.append("=" * 110)
    L.append(f"{'cell':>16} {'persist':>8} {'prey/10':>7} {'pred/10':>7} {'localext':>8} {'recol':>6} {'sync':>6} {'cvP':>5} {'cvQ':>5} {'exploded':>8}")
    L.append("-" * 110)
    res = {}
    order = ["BASELINE_1P", "NEITHER", "REFUGE_ONLY", "MIG_ONLY_ASYNC", "MIG_ONLY_SYNC", "BOTH", "HIGH_MIG", "STRONG_REFUGE"]
    for name in order:
        r = run_cell(name); res[name] = r
        L.append(f"{name:>16} {r['persist']:>5}/{len(SEEDS)} {r['prey_pers']:>7} {r['pred_pers']:>7} "
                 f"{r['le']:>8.0f} {r['rc']:>6.0f} {r['sync']:>6.2f} {r['cvp']:>5.2f} {r['cvq']:>5.2f} {str(r['exploded']):>8}")
    L.append("")

    def passes(n):
        r = res[n]; return r["persist"] >= PASS_SEEDS and not r["exploded"]
    def turns_over(n):
        r = res[n]; return r["le"] > 0 and r["rc"] > 0

    L.append("PREDECLARED CONTROL CHECKS:")
    L.append(f"  BASELINE_1P (single patch, mosaic's TOTAL pop) COLLAPSES: {not passes('BASELINE_1P')} (persist={res['BASELINE_1P']['persist']}/10) [rules out scale]")
    L.append(f"  NEITHER (8p, no refuge, no migration) COLLAPSES: {not passes('NEITHER')} (persist={res['NEITHER']['persist']}/10) [a mosaic mechanism is NECESSARY]")
    L.append(f"  REFUGE load-bearing (REFUGE_ONLY flips NEITHER): {passes('REFUGE_ONLY') and not passes('NEITHER')} (refuge_only={res['REFUGE_ONLY']['persist']}/10)")
    L.append(f"  MIGRATION load-bearing (MIG_ONLY_ASYNC flips NEITHER): {passes('MIG_ONLY_ASYNC') and not passes('NEITHER')} (mig_only={res['MIG_ONLY_ASYNC']['persist']}/10)")
    L.append(f"  asynchrony matters for migration-rescue (ASYNC vs SYNC mig-only): mig_async={res['MIG_ONLY_ASYNC']['persist']}/10 vs mig_sync={res['MIG_ONLY_SYNC']['persist']}/10")
    L.append(f"  HIGH migration re-synchronizes/degrades vs MED: high={res['HIGH_MIG']['persist']}/10 (sync={res['HIGH_MIG']['sync']:.2f}) vs mig_async med={res['MIG_ONLY_ASYNC']['persist']}/10 (sync={res['MIG_ONLY_ASYNC']['sync']:.2f})")
    L.append(f"  too-strong refuge STARVES predator: pred_pers(STRONG_REFUGE)={res['STRONG_REFUGE']['pred_pers']}/10 vs BOTH={res['BOTH']['pred_pers']}/10")
    L.append(f"  BOTH (refuge+migration+async) persists with local turnover: {passes('BOTH') and turns_over('BOTH')} (persist={res['BOTH']['persist']}/10, localext={res['BOTH']['le']:.0f}, recol={res['BOTH']['rc']:.0f})")
    L.append("")

    baseline_collapses = (not passes("BASELINE_1P")) and (not passes("NEITHER"))
    refuge_lb = passes("REFUGE_ONLY") and not passes("NEITHER")
    mig_lb = passes("MIG_ONLY_ASYNC") and not passes("NEITHER")
    any_mechanism = refuge_lb or mig_lb
    both_ok = passes("BOTH") and turns_over("BOTH")
    if baseline_collapses and any_mechanism and both_ok:
        mechs = " + ".join([m for m, ok in [("refuge", refuge_lb), ("migration", mig_lb)] if ok])
        verdict = f"CAN_POSE_GLOBAL_COHABITATION + PATCH_MOSAIC_RESCUE_SUPPORTED (load-bearing: {mechs})"
    elif not baseline_collapses:
        verdict = "NO_VERDICT — baseline did not collapse (mosaic mechanism not shown necessary / scale confound)"
    elif not any_mechanism:
        verdict = "NO_VERDICT/FAIL — no single mechanism flips collapse->persist (nothing load-bearing)"
    else:
        verdict = "PARTIAL — see raw numbers (controller adjudicates)"
    L.append(f"PRELIMINARY VERDICT (controller confirms against raw numbers): {verdict}")
    out = "\n".join(L)
    print(out)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
              "experiments", "outputs", "exp257_patch_mosaic_predator_prey.txt"), "w") as f:
        f.write(out + "\n")
    print("\n[written to experiments/outputs/exp257_patch_mosaic_predator_prey.txt]")


if __name__ == "__main__":
    main()
