"""experiments/exp267_decoupled_roles.py — Exp 267 (R3b): do predator/prey ROLES emerge once
OFFENSE and DEFENSE are DECOUPLED?

R3 (Exp 266) found roles do NOT emerge from single-trait cannibalism: a lone predatoriness trait
couples offense (eat others) and invulnerability (aggr=1.0 unkillable), so aggr runs away to a
monomorphic max-predator — no stable prey niche. R3b decouples them: defense is now a SEPARATE
evolving, costed trait (the prey's ESCAPE; cannibalism vulnerability = 1/(1+escape_k*escape), gated by
cannibal_defense_by_escape), independent of offense (aggr). A predator can be high-aggr/low-escape; a
prey can be low-aggr/high-escape. Does the single population now DIVERGE into distinct roles?

HYPOTHESIS / PREDICTION: with offense (aggr, forage cost) and defense (escape, fecundity cost) decoupled,
frequency-dependent selection can sustain TWO coexisting strategy clusters in (aggr, escape) space —
a PREDATOR clade (high aggr, low escape) and a PREY/forager clade (low aggr, high escape) — i.e. emergent
roles, with aggr-escape NEGATIVELY correlated. PREDECLARED FALSIFIER: if the population stays unimodal in
aggr (one clade fixes — all-predator OR all-prey/forager) or aggr & escape are not anti-correlated, roles
still do NOT differentiate even with decoupling; report it (and whether decoupling at least breaks R3's
runaway-to-max, e.g. by letting a defended prey clade survive).

Single population (n_pred0=0, step()-driven), enable_cannibalism + cannibal_defense_by_escape, aggr AND
escape free-evolving + costed. Adjudicated on the JOINT distribution (clade fractions, within-clade escape,
aggr-escape correlation, aggr percentiles — NOT a boundary-fooled bimodality coefficient; see
mean-of-opposites-guard v1.2). RAW — controller adjudicates. FRESH seeds 750-759. multiprocessing.
"""
import sys
import os
import math
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim

SEEDS = list(range(750, 760))
HORIZON = 3000
POP_CAP = 100_000


def cfg(aggr0, escape_cost, cannibal_cost, horizon=HORIZON):
    return PatchMosaicConfig(
        n_patches=8, topology="ring", K_prey_local=300.0, escape_k=1.0,
        migration_rate_prey=0.05, migration_rate_pred=0.0,
        async_mode="rotating", async_amplitude=0.4, async_period=50.0,
        refuge_mode="none",
        enable_trait_evolution=True, enable_contest=False,
        enable_cannibalism=True, cannibal_defense_by_escape=True,
        mutation_rate=0.15, mutation_sd=0.06, aggr_mutation_sd=0.05,
        escape_cost=escape_cost, escape_baseline=1.0, prey_escape=1.0, pred_attack=1.0, aggr0=aggr0,
        cannibal_cost=cannibal_cost, cannibal_gain=0.5,
        freeze_prey_trait=False, freeze_predator_trait=True,   # BOTH aggr and escape evolve
        trait_min=0.0, trait_max=4.0, track_lineages=True,
        horizon=horizon, n_prey0_per_patch=40, n_pred0_per_patch=0)


def _cell(args):
    aggr0, escape_cost, cannibal_cost, seed = args
    c = cfg(aggr0, escape_cost, cannibal_cost)
    sim = PatchMosaicSim(c, seed)
    exploded = False
    for _ in range(HORIZON):
        if sum(len(p.prey) for p in sim.patches) == 0:
            break
        sim.step()
        if sum(len(p.prey) for p in sim.patches) > POP_CAP:
            exploded = True
            break
    aggr = np.array([cr.aggr for p in sim.patches for cr in p.prey])
    esc = np.array([cr.trait for p in sim.patches for cr in p.prey])
    n = len(aggr)
    if n == 0:
        return dict(args=args, n=0, extinct=True, exploded=exploded)
    pred_mask = aggr > 0.5
    prey_mask = aggr < 0.2
    corr = float(np.corrcoef(aggr, esc)[0, 1]) if (aggr.std() > 0 and esc.std() > 0) else float("nan")
    return dict(args=args, n=n, extinct=False, exploded=exploded,
                aggr_mean=float(aggr.mean()), aggr_p10=float(np.percentile(aggr, 10)),
                aggr_p50=float(np.percentile(aggr, 50)), aggr_p90=float(np.percentile(aggr, 90)),
                esc_mean=float(esc.mean()),
                frac_pred=float(pred_mask.mean()), frac_prey=float(prey_mask.mean()),
                esc_of_pred=float(esc[pred_mask].mean()) if pred_mask.any() else float("nan"),
                esc_of_prey=float(esc[prey_mask].mean()) if prey_mask.any() else float("nan"),
                aggr_of_pred=float(aggr[pred_mask].mean()) if pred_mask.any() else float("nan"),
                corr=corr)


def main():
    smoke = "--smoke" in sys.argv
    seeds = SEEDS[:2] if smoke else SEEDS
    # offense cost fixed; sweep DEFENSE cost (escape_cost) — cheap defense should favour a prey clade
    cost_sweep = [(0.15, 0.10)] if smoke else [(0.05, 0.10), (0.15, 0.10), (0.30, 0.10)]
    nproc = min((os.cpu_count() or 2) - 1, 12)

    cells = [(0.3, ec, cc, s) for (ec, cc) in cost_sweep for s in seeds]
    with mp.Pool(nproc) as pool:
        res = pool.map(_cell, cells)
    by = {}
    for r in res:
        by.setdefault((r["args"][1], r["args"][2]), []).append(r)

    L = []
    L.append("=" * 110)
    L.append("Exp 267 (R3b) — do predator/prey ROLES emerge once OFFENSE (aggr) and DEFENSE (escape) are DECOUPLED? RAW.")
    L.append(f"single pop (n_pred0=0), cannibalism + cannibal_defense_by_escape, aggr+escape evolve; seeds {seeds}; {nproc} procs.")
    L.append("ROLES iff aggr BIMODAL (predator clade aggr>0.5 + prey clade aggr<0.2 coexist) AND aggr-escape anti-correlated (prey defend).")
    L.append("=" * 110)
    L.append("")
    L.append(f"{'esc_cost':>8} {'cann_cost':>9} | {'aggr p10/p50/p90':>18} {'esc_mean':>8} | {'%pred':>6} {'%prey':>6} | {'esc[pred]':>9} {'esc[prey]':>9} | {'corr(aggr,esc)':>14} | extinct")
    L.append("-" * 110)
    for (ec, cc) in cost_sweep:
        rows = by[(ec, cc)]
        live = [r for r in rows if not r["extinct"] and not r.get("exploded")]
        ext = sum(r["extinct"] for r in rows)
        if live:
            pp = f"{np.mean([r['aggr_p10'] for r in live]):.2f}/{np.mean([r['aggr_p50'] for r in live]):.2f}/{np.mean([r['aggr_p90'] for r in live]):.2f}"
            em = float(np.mean([r["esc_mean"] for r in live]))
            fp = float(np.mean([r["frac_pred"] for r in live])) * 100
            fy = float(np.mean([r["frac_prey"] for r in live])) * 100
            ep = float(np.nanmean([r["esc_of_pred"] for r in live]))
            ey = float(np.nanmean([r["esc_of_prey"] for r in live]))
            cr = float(np.nanmean([r["corr"] for r in live]))
            L.append(f"{ec:>8.2f} {cc:>9.2f} | {pp:>18} {em:>8.2f} | {fp:>5.1f}% {fy:>5.1f}% | {ep:>9.2f} {ey:>9.2f} | {cr:>14.3f} | {ext}/{len(rows)}")
        else:
            L.append(f"{ec:>8.2f} {cc:>9.2f} | (no live runs) extinct={ext}/{len(rows)}")
    L.append("")
    L.append("  per-seed (esc_cost=0.15): aggr_p50 | %pred | %prey | esc[prey] | corr")
    for r in sorted(by[(0.15, 0.10)], key=lambda r: r["args"][3]):
        if r["extinct"]:
            L.append(f"    seed {r['args'][3]}: EXTINCT")
        else:
            L.append(f"    seed {r['args'][3]}: aggr_p50={r['aggr_p50']:.2f} %pred={r['frac_pred']*100:.0f} %prey={r['frac_prey']*100:.0f} esc[prey]={r['esc_of_prey']:.2f} corr={r['corr']:.2f} (n={r['n']})")
    L.append("")
    L.append("PREDECLARED READOUT (controller adjudicates):")
    L.append("  ROLES_EMERGE iff a PREDATOR clade (aggr>0.5) AND a PREY clade (aggr<0.2 with high escape) robustly")
    L.append("  COEXIST (both non-trivial %), with aggr-escape anti-correlated; CONVERGES iff one clade fixes;")
    L.append("  decoupling at least BREAKS R3's runaway iff a defended prey clade survives (vs R3's 100% max-predator).")

    out = "\n".join(L)
    print(out)
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "outputs")
    os.makedirs(outdir, exist_ok=True)
    fname = "exp267_smoke.txt" if smoke else "exp267.txt"
    with open(os.path.join(outdir, fname), "w") as f:
        f.write(out + "\n")
    print(f"\n[written to experiments/outputs/{fname}]")


if __name__ == "__main__":
    main()
