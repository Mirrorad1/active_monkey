"""experiments/exp265_refuge_free_coupling.py — Exp 265 (R2 follow-up): do REFUGES decouple
intraspecific aggression from predation pressure?

R2 (Exp 264) found that, on the REFUGE-bearing BOTH regime, intraspecific aggression robustly
co-emerged (aggr@end~0.86) even under a co-evolving predator that suppressed GLOBAL prey density to
N/K~0.05 — the explanation being that refuge patches crowd LOCALLY even when the global predator thins
the population. This experiment tests that explanation directly: REMOVE the refuge (keep migration +
asynchrony for coexistence, the Exp-258 MIG_ONLY_ASYNC rescue) and ask whether aggression's emergence
now COUPLES to predation pressure.

HYPOTHESIS / PREDICTION: refuge-free, predation directly controls prey crowding everywhere, so aggr@end
should DECLINE as predation (attack_a) rises (washed out once a strong predator suppresses prey below K
— no crowding -> no contest payoff). WITH refuge, aggr@end should stay HIGH across predation (the
refuges supply local crowding regardless). So the refuge×predation grid should show an INTERACTION:
refuge decouples aggression from predation; refuge-free couples it.

PREDECLARED FALSIFIER: if refuge-free aggr@end stays high across predation (no decline) just like the
refuge case, then the refuges are NOT what decoupled aggression from predation in R2 — the R2
orthogonality has another cause. (Conversely, the clean result is the interaction above.)

3-trait co-evolution (escape+aggr+attack), contest ON, single varied axes = refuge_mode x attack_a.
Per-seed dispersion reported (mean-of-opposites-guard). RAW — controller adjudicates. FRESH seeds 730-739.
trait_max=4.0. multiprocessing.
"""
import sys
import os
import math
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim

SEEDS = list(range(730, 740))      # 10 seeds
ATTACK_SWEEP = [0.01, 0.03, 0.05, 0.07]
HORIZON = 3000
K_PREY = 300.0                     # default K_prey_local (N/K uses this)


def cfg(refuge_mode, attack_a, horizon=HORIZON):
    ref = (dict(refuge_mode="per_patch", refuge_predator_access=0.30, refuge_fraction=0.25)
           if refuge_mode == "per_patch" else dict(refuge_mode="none"))
    return PatchMosaicConfig(
        n_patches=8, topology="ring", attack_a=attack_a, K_pred_local=40.0, pred_self_limit_hmax=0.05,
        migration_rate_prey=0.05, migration_rate_pred=0.05,
        async_mode="rotating", async_amplitude=0.4, async_period=50.0,
        enable_trait_evolution=True, enable_contest=True,
        mutation_rate=0.15, mutation_sd=0.06, aggr_mutation_sd=0.05,
        escape_cost=0.15, escape_baseline=1.0, contest_cost=0.10, contest_seize=0.50, contest_dissipation=0.0,
        prey_escape=1.0, pred_attack=1.0, aggr0=0.0,
        freeze_prey_trait=False, freeze_predator_trait=False,
        trait_min=0.0, trait_max=4.0, track_lineages=True,
        horizon=horizon, n_prey0_per_patch=40, n_pred0_per_patch=8, **ref)


def _cell(args):
    refuge_mode, attack_a, seed = args
    sim = PatchMosaicSim(cfg(refuge_mode, attack_a), seed)
    while (any(p.prey for p in sim.patches) and any(p.predators for p in sim.patches)) and sim.t < HORIZON:
        sim.step()
    prey_esc = [c.trait for p in sim.patches for c in p.prey]
    prey_agg = [c.aggr for p in sim.patches for c in p.prey]
    pred_atk = [c.trait for p in sim.patches for c in p.predators]
    n_prey, n_pred = len(prey_esc), len(pred_atk)
    extinct = not (prey_esc and pred_atk)
    return dict(refuge_mode=refuge_mode, attack_a=attack_a, seed=seed,
                aggr=float(np.mean(prey_agg)) if prey_agg else float("nan"),
                esc=float(np.mean(prey_esc)) if prey_esc else float("nan"),
                atk=float(np.mean(pred_atk)) if pred_atk else float("nan"),
                nk=n_prey / (8 * K_PREY), n_prey=n_prey, n_pred=n_pred, extinct=extinct)


def main():
    smoke = "--smoke" in sys.argv
    seeds = SEEDS[:2] if smoke else SEEDS
    sweep = [0.01, 0.05] if smoke else ATTACK_SWEEP
    nproc = min((os.cpu_count() or 2) - 1, 12)

    cells = [(rm, aa, s) for rm in ("per_patch", "none") for aa in sweep for s in seeds]
    with mp.Pool(nproc) as pool:
        res = pool.map(_cell, cells)

    grid = {}
    for r in res:
        grid.setdefault((r["refuge_mode"], r["attack_a"]), []).append(r)

    L = []
    L.append("=" * 100)
    L.append("Exp 265 (R2 follow-up) — do REFUGES decouple intraspecific aggression from predation? RAW.")
    L.append(f"3-trait co-evolution, contest ON; refuge x attack_a grid; seeds {seeds}; {nproc} procs; trait_max=4.0.")
    L.append("=" * 100)
    L.append("")
    L.append(f"{'refuge':>10} {'attack_a':>9} | {'aggr@end':>9} {'aggr_sd':>8} | {'mean N/K':>8} | {'esc@end':>8} {'atk@end':>8} | extinct | per-seed aggr")
    L.append("-" * 116)
    summary = {}
    for rm in ("per_patch", "none"):
        for aa in sweep:
            rows = grid[(rm, aa)]
            live = [r for r in rows if not r["extinct"]]
            use = live if live else rows
            aggr = [r["aggr"] for r in use if not math.isnan(r["aggr"])]
            am = float(np.mean(aggr)) if aggr else float("nan")
            asd = float(np.std(aggr)) if aggr else float("nan")
            nk = float(np.mean([r["nk"] for r in use]))
            em = float(np.mean([r["esc"] for r in use if not math.isnan(r["esc"])])) if use else float("nan")
            km = float(np.mean([r["atk"] for r in use if not math.isnan(r["atk"])])) if use else float("nan")
            ext = sum(r["extinct"] for r in rows)
            summary[(rm, aa)] = dict(aggr=am, nk=nk, ext=ext, n=len(rows))
            ps = " ".join(f"{r['aggr']:.2f}" if not math.isnan(r['aggr']) else "--" for r in sorted(rows, key=lambda r: r["seed"]))
            L.append(f"{rm:>10} {aa:>9.2f} | {am:>9.3f} {asd:>8.3f} | {nk:>8.3f} | {em:>8.3f} {km:>8.3f} | {ext:>2}/{len(rows)} | {ps}")
        L.append("")

    L.append("PREDECLARED READOUT (controller adjudicates the interaction):")
    for rm in ("per_patch", "none"):
        seq = ", ".join(f"a{aa:g}: aggr={summary[(rm,aa)]['aggr']:.3f} (N/K {summary[(rm,aa)]['nk']:.2f}, ext {summary[(rm,aa)]['ext']}/{summary[(rm,aa)]['n']})" for aa in sweep)
        L.append(f"  {rm:>10}: {seq}")
    L.append("  COUPLING (hypothesis) iff refuge-free aggr@end DECLINES with attack_a while refuge stays HIGH;")
    L.append("  refuges decouple intraspecific crowding from global predation iff that interaction holds.")

    out = "\n".join(L)
    print(out)
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "outputs")
    os.makedirs(outdir, exist_ok=True)
    fname = "exp265_smoke.txt" if smoke else "exp265.txt"
    with open(os.path.join(outdir, fname), "w") as f:
        f.write(out + "\n")
    print(f"\n[written to experiments/outputs/{fname}]")


if __name__ == "__main__":
    main()
