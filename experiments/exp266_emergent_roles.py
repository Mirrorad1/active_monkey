"""experiments/exp266_emergent_roles.py — Exp 266 (R3 of emergent-intraspecific-competition):
do PREDATOR/PREY ROLES emerge from a single population via cannibalism?

R1/R2 showed the contest trait CONVERGES (everyone ~one aggression level). R3 asks the harder
question — "what IS a predator/prey" — by starting a SINGLE population (no separate predator species)
with a heritable aggression/predatoriness `aggr` and a gated CANNIBALISM mechanic: a predatory prey
(high aggr) kills + consumes a conspecific (low aggr) for a reproduction boost, while being predatory
trades off against foraging. Because predators eat NON-predators (kill_prob = aggr_i*(1-aggr_j)), the
selection is FREQUENCY-DEPENDENT — a stable predator/prey POLYMORPHISM (distinct roles) CAN emerge.

HYPOTHESIS / PREDICTION: the aggr distribution DIVERGES into a BIMODAL split — a high-aggr "predator"
clade + a low-aggr "prey/forager" clade that coexist (emergent roles) — rather than converging to a
single level. PREDECLARED FALSIFIER: if aggr stays UNIMODAL (Sarle bimodality coefficient < 0.555, one
clade fixes or it is a single converged level), roles do NOT differentiate — one strategy wins or it
drifts; report it. (Also NO_VERDICT if the population collapses / explodes.)

Single population: n_pred0=0 (no separate predator species), driven by step() with prey-only extinction
(run() would halt on zero-predators). enable_cannibalism, aggr free-evolving from aggr0. Measure the
END aggr DISTRIBUTION (Sarle bimodality coefficient + predator-clade/prey-clade fractions) PER SEED
(mean-of-opposites-guard: a bimodal population is exactly what "roles" means — so report the shape, not
just the mean). RAW — controller adjudicates. FRESH seeds 740-749. multiprocessing.
"""
import sys
import os
import math
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim

SEEDS = list(range(740, 750))   # 10 seeds
HORIZON = 3000
POP_CAP = 100_000


def cfg(aggr0, cannibal_cost, cannibal_gain, horizon=HORIZON):
    return PatchMosaicConfig(
        n_patches=8, topology="ring", K_prey_local=300.0,
        migration_rate_prey=0.05, migration_rate_pred=0.0,
        async_mode="rotating", async_amplitude=0.4, async_period=50.0,
        refuge_mode="none",
        enable_trait_evolution=True, enable_contest=False, enable_cannibalism=True,
        mutation_rate=0.15, mutation_sd=0.06, aggr_mutation_sd=0.05,
        escape_cost=0.0, escape_baseline=1.0, prey_escape=1.0, pred_attack=1.0, aggr0=aggr0,
        cannibal_cost=cannibal_cost, cannibal_gain=cannibal_gain,
        freeze_prey_trait=False, freeze_predator_trait=True,
        trait_min=0.0, trait_max=4.0, track_lineages=True,
        horizon=horizon, n_prey0_per_patch=40, n_pred0_per_patch=0)   # SINGLE population


def bimod_coeff(x):
    x = np.asarray([v for v in x if not math.isnan(v)], dtype=float)
    n = len(x)
    if n < 4 or x.std() == 0:
        return float("nan")
    m, s = x.mean(), x.std()
    sk = (((x - m) ** 3).mean()) / s ** 3
    ku = (((x - m) ** 4).mean()) / s ** 4
    return (sk ** 2 + 1) / (ku + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def _cell(args):
    aggr0, cannibal_cost, cannibal_gain, seed = args
    c = cfg(aggr0, cannibal_cost, cannibal_gain)
    sim = PatchMosaicSim(c, seed)
    exploded = False
    for _ in range(HORIZON):
        if sum(len(p.prey) for p in sim.patches) == 0:
            break
        sim.step()
        if sum(len(p.prey) for p in sim.patches) > POP_CAP:
            exploded = True
            break
    aggrs = [cr.aggr for p in sim.patches for cr in p.prey]
    n = len(aggrs)
    if n == 0:
        return dict(args=args, n=0, extinct=True, exploded=exploded)
    arr = np.asarray(aggrs)
    return dict(args=args, n=n, extinct=False, exploded=exploded,
                mean=float(arr.mean()), std=float(arr.std()), bimod=bimod_coeff(aggrs),
                frac_pred=float((arr > 0.5).mean()), frac_prey=float((arr < 0.2).mean()),
                frac_mid=float(((arr >= 0.2) & (arr <= 0.5)).mean()),
                p10=float(np.percentile(arr, 10)), p50=float(np.percentile(arr, 50)), p90=float(np.percentile(arr, 90)))


def main():
    smoke = "--smoke" in sys.argv
    seeds = SEEDS[:2] if smoke else SEEDS
    cost_sweep = [0.10] if smoke else [0.05, 0.10, 0.20]
    nproc = min((os.cpu_count() or 2) - 1, 12)

    cells = [(0.3, cc, 0.5, s) for cc in cost_sweep for s in seeds]
    with mp.Pool(nproc) as pool:
        res = pool.map(_cell, cells)

    by = {}
    for r in res:
        by.setdefault(r["args"][1], []).append(r)   # group by cannibal_cost

    L = []
    L.append("=" * 104)
    L.append("Exp 266 (R3) — do PREDATOR/PREY ROLES emerge from a single population via cannibalism? RAW.")
    L.append(f"single pop (n_pred0=0), enable_cannibalism, aggr0=0.3, aggr free-evolving; seeds {seeds}; {nproc} procs.")
    L.append("ROLES_EMERGE iff aggr DISTRIBUTION is BIMODAL (predator clade aggr>0.5 + prey clade aggr<0.2 coexist).")
    L.append("=" * 104)
    L.append("")
    L.append(f"{'cannibal_cost':>13} | {'mean aggr':>9} {'std':>6} {'bimod':>6} | {'%pred(>0.5)':>11} {'%prey(<0.2)':>11} {'%mid':>6} | {'p10/p50/p90':>16} | extinct")
    L.append("-" * 104)
    for cc in cost_sweep:
        rows = by[cc]
        live = [r for r in rows if not r["extinct"] and not r.get("exploded")]
        ext = sum(r["extinct"] for r in rows)
        if live:
            mean = float(np.mean([r["mean"] for r in live]))
            std = float(np.mean([r["std"] for r in live]))
            bm = float(np.nanmean([r["bimod"] for r in live]))
            fp = float(np.mean([r["frac_pred"] for r in live]))
            fy = float(np.mean([r["frac_prey"] for r in live]))
            fm = float(np.mean([r["frac_mid"] for r in live]))
            pp = f"{np.mean([r['p10'] for r in live]):.2f}/{np.mean([r['p50'] for r in live]):.2f}/{np.mean([r['p90'] for r in live]):.2f}"
            L.append(f"{cc:>13.2f} | {mean:>9.3f} {std:>6.3f} {bm:>6.3f} | {fp*100:>10.1f}% {fy*100:>10.1f}% {fm*100:>5.1f}% | {pp:>16} | {ext}/{len(rows)}")
        else:
            L.append(f"{cc:>13.2f} | (no live runs) extinct={ext}/{len(rows)}")
    L.append("")
    L.append("  per-seed (cannibal_cost=0.10): bimod | mean | %pred | %prey")
    for r in sorted(by[0.10], key=lambda r: r["args"][3]):
        if r["extinct"]:
            L.append(f"    seed {r['args'][3]}: EXTINCT")
        else:
            L.append(f"    seed {r['args'][3]}: bimod={r['bimod']:.3f} mean={r['mean']:.3f} %pred={r['frac_pred']*100:.0f} %prey={r['frac_prey']*100:.0f} (n={r['n']})")
    L.append("")
    L.append("PREDECLARED READOUT (controller adjudicates):")
    L.append("  ROLES_EMERGE iff the aggr distribution is robustly BIMODAL (bimod>0.555) with BOTH a")
    L.append("  predator clade (aggr>0.5) and a prey clade (aggr<0.2) coexisting; CONVERGES iff unimodal")
    L.append("  (one clade / single level); NO_VERDICT iff collapse/explosion.")

    out = "\n".join(L)
    print(out)
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "outputs")
    os.makedirs(outdir, exist_ok=True)
    fname = "exp266_smoke.txt" if smoke else "exp266.txt"
    with open(os.path.join(outdir, fname), "w") as f:
        f.write(out + "\n")
    print(f"\n[written to experiments/outputs/{fname}]")


if __name__ == "__main__":
    main()
