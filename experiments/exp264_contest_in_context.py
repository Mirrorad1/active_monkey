"""experiments/exp264_contest_in_context.py — Exp 264 (R2 of emergent-intraspecific-competition):
the R1 emerged contest aggression, IN CONTEXT. Two parts:

PART A — SCARCITY-DEPENDENCE BASIN MAP. R1 showed aggression invades 9/10 scarce vs 0/10 abundant.
Here we MAP P(aggression invades from rarity) vs predation pressure (the scarcity knob attack_a;
predator FROZEN), to trace the basin transition. Uses invasion-from-rarity (rare aggr=0.5 into aggr=0
resident) — NOT the resident-0 50/50 gradient (which has the always-win artifact). multiprocessing.

PART B — PREDATOR INTEGRATION (the novel question). Turn the predator's attack trait back ON and let
prey escape AND prey aggression co-evolve WITH it (3-trait co-evolution) on the Exp-259 collapse-prone
BOTH regime that produced predator-dominance (prey escape collapsed to 0.68). SINGLE CAUSAL BIT =
enable_contest (OFF = Exp-259-like escape+attack co-evolution, aggr inert; ON = prey can also contest
each other). Does intraspecific contest change the predator-prey outcome?

HYPOTHESIS / PREDICTION (Part B): a STRONG co-evolving predator suppresses prey BELOW K (reproduction
headroom = abundance), and R1 showed contest does NOT pay under abundance — so the predator may WASH
OUT intraspecific aggression (aggr stays ~0), leaving the Exp-259 predator-dominance unchanged. The
alternative: in the refuge/patches prey crowd locally -> contest pays there -> aggr rises and buffers
or alters the prey-escape collapse. PREDECLARED FALSIFIER for "contest matters in context": if the ON
and OFF arms are indistinguishable (esc@3k, atk@3k, pops, extinction all within seed noise) AND aggr@3k
stays ~0 under the co-evolving predator, then the emerged aggression is WASHED OUT once the predator
co-evolves (it only emerges when predation is weak enough to let prey crowd).

Reads PER-SEED dispersion (mean-of-opposites-guard). RAW — controller adjudicates. trait_max=4.0
(Exp-259's value). FRESH seeds: Part A 620-629, Part B 720-729.
"""
import sys
import os
import math
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim
from experiments.exp263_intraspecific_contest_emergence import invasion  # reuse R1 invasion-from-rarity

SEEDS_A = list(range(620, 630))   # Part A basin map: 10 seeds per attack_a
SEEDS_B = list(range(720, 730))   # Part B co-evolution: 10 seeds per arm
ATTACK_SWEEP = [0.01, 0.02, 0.04, 0.06, 0.08]   # weak (scarce) -> strong (abundant) predation
HORIZON_B = 3000


# ---------------- Part A: invasion across predation pressure (multiprocessing) ----------------
def _invasion_cell(args):
    attack_a, seed = args
    r = invasion(attack_a, seed)        # rare aggr=0.5 into aggr=0 resident; predator frozen at attack_a
    return (attack_a, seed, r)


# ---------------- Part B: 3-trait co-evolution, contest ON vs OFF ----------------
def cfg_B(enable_contest, horizon=HORIZON_B):
    return PatchMosaicConfig(
        n_patches=8, topology="ring", attack_a=0.05, K_pred_local=40.0, pred_self_limit_hmax=0.05,
        migration_rate_prey=0.05, migration_rate_pred=0.05,
        async_mode="rotating", async_amplitude=0.4, async_period=50.0,
        refuge_mode="per_patch", refuge_predator_access=0.30, refuge_fraction=0.25,
        enable_trait_evolution=True, enable_contest=enable_contest,
        mutation_rate=0.15, mutation_sd=0.06, aggr_mutation_sd=0.05,
        escape_cost=0.15, escape_baseline=1.0, contest_cost=0.10, contest_seize=0.50, contest_dissipation=0.0,
        prey_escape=1.0, pred_attack=1.0, aggr0=0.0,
        freeze_prey_trait=False, freeze_predator_trait=False,   # escape + aggr + attack all co-evolve
        trait_min=0.0, trait_max=4.0, track_lineages=True,
        horizon=horizon, n_prey0_per_patch=40, n_pred0_per_patch=8)


def _coevo_cell(args):
    enable_contest, seed = args
    sim = PatchMosaicSim(cfg_B(enable_contest), seed)
    while (any(p.prey for p in sim.patches) and any(p.predators for p in sim.patches)) and sim.t < HORIZON_B:
        sim.step()
    prey_esc = [c.trait for p in sim.patches for c in p.prey]
    prey_agg = [c.aggr for p in sim.patches for c in p.prey]
    pred_atk = [c.trait for p in sim.patches for c in p.predators]
    n_prey, n_pred = len(prey_esc), len(pred_atk)
    extinct = not (prey_esc and pred_atk)
    return dict(enable_contest=enable_contest, seed=seed,
                esc=float(np.mean(prey_esc)) if prey_esc else float("nan"),
                aggr=float(np.mean(prey_agg)) if prey_agg else float("nan"),
                atk=float(np.mean(pred_atk)) if pred_atk else float("nan"),
                n_prey=n_prey, n_pred=n_pred, extinct=extinct)


def _summ(rows, key):
    vals = [r[key] for r in rows if not math.isnan(r[key])]
    return (float(np.mean(vals)) if vals else float("nan"),
            float(np.std(vals)) if vals else float("nan"))


def main():
    smoke = "--smoke" in sys.argv
    sa = SEEDS_A[:2] if smoke else SEEDS_A
    sb = SEEDS_B[:2] if smoke else SEEDS_B
    sweep = [0.01, 0.08] if smoke else ATTACK_SWEEP
    nproc = min((os.cpu_count() or 2) - 1, 12)

    L = []
    L.append("=" * 100)
    L.append("Exp 264 (R2) — emerged contest aggression IN CONTEXT. RAW — controller adjudicates.")
    L.append(f"Part A basin map seeds {sa}; Part B co-evolution seeds {sb}; {nproc} procs; trait_max=4.0.")
    L.append("=" * 100)
    L.append("")

    # PART A: basin map
    L.append("(A) SCARCITY BASIN MAP: P(aggression invades from rarity) vs predation pressure (predator FROZEN):")
    L.append(f"{'attack_a':>9} | {'invaded':>9} | {'mean f_final':>12} | regime")
    L.append("-" * 60)
    cellsA = [(aa, s) for aa in sweep for s in sa]
    with mp.Pool(nproc) as pool:
        resA = pool.map(_invasion_cell, cellsA)
    byA = {aa: [] for aa in sweep}
    for aa, s, r in resA:
        if r is not None:
            byA[aa].append(r)
    for aa in sweep:
        rs = byA[aa]
        inv = sum(r["invaded"] for r in rs)
        mf = float(np.mean([r["f_final"] for r in rs])) if rs else float("nan")
        regime = "SCARCE (weak pred)" if aa <= 0.02 else ("transition" if aa <= 0.04 else "ABUNDANT (strong pred)")
        L.append(f"{aa:>9.2f} | {inv:>4}/{len(rs):<3} | {mf:>12.3f} | {regime}")
    L.append("")

    # PART B: co-evolution, contest ON vs OFF
    L.append("(B) PREDATOR INTEGRATION: 3-trait co-evolution (escape+aggr+attack), single bit = enable_contest, horizon=" + str(HORIZON_B))
    L.append(f"{'arm':>14} | {'esc@end':>8} {'aggr@end':>9} {'atk@end':>8} | {'esc_sd':>6} | {'n_prey':>7} {'n_pred':>7} | extinct")
    L.append("-" * 100)
    cellsB = [(ec, s) for ec in (False, True) for s in sb]
    with mp.Pool(nproc) as pool:
        resB = pool.map(_coevo_cell, cellsB)
    byB = {False: [r for r in resB if r["enable_contest"] is False],
           True: [r for r in resB if r["enable_contest"] is True]}
    summB = {}
    for ec in (False, True):
        rows = byB[ec]
        e_m, e_s = _summ(rows, "esc"); a_m, _ = _summ(rows, "aggr"); k_m, _ = _summ(rows, "atk")
        np_m, _ = _summ(rows, "n_prey"); npd_m, _ = _summ(rows, "n_pred")
        ext = sum(r["extinct"] for r in rows)
        summB[ec] = dict(esc=e_m, esc_sd=e_s, aggr=a_m, atk=k_m, n_prey=np_m, n_pred=npd_m, ext=ext, n=len(rows))
        arm = "contest ON" if ec else "contest OFF"
        L.append(f"{arm:>14} | {e_m:>8.3f} {a_m:>9.3f} {k_m:>8.3f} | {e_s:>6.3f} | {np_m:>7.1f} {npd_m:>7.1f} | {ext}/{len(rows)}")
    L.append("")
    # per-seed esc for the mean-of-opposites-guard
    for ec in (False, True):
        arm = "ON " if ec else "OFF"
        escs = " ".join(f"{r['esc']:.2f}" for r in sorted(byB[ec], key=lambda r: r["seed"]))
        aggs = " ".join(f"{r['aggr']:.2f}" for r in sorted(byB[ec], key=lambda r: r["seed"]))
        L.append(f"  per-seed esc@end [{arm}]: {escs}")
        L.append(f"  per-seed aggr@end[{arm}]: {aggs}")
    L.append("")

    L.append("PREDECLARED READOUT (controller adjudicates):")
    pa = [(aa, sum(r['invaded'] for r in byA[aa]) / max(1, len(byA[aa]))) for aa in sweep]
    L.append(f"  (A) P(invade) vs attack_a: " + ", ".join(f"a{aa:g}={p:.2f}" for aa, p in pa))
    off, on = summB[False], summB[True]
    L.append(f"  (B) OFF: esc@end={off['esc']:.3f} atk@end={off['atk']:.3f} aggr={off['aggr']:.3f} (aggr inert when contest OFF) ext={off['ext']}/{off['n']}")
    L.append(f"  (B) ON : esc@end={on['esc']:.3f} atk@end={on['atk']:.3f} aggr@end={on['aggr']:.3f} ext={on['ext']}/{on['n']}")
    L.append(f"  contest WASHED OUT iff aggr@end~0 under the co-evolving predator AND ON~OFF on esc/atk/pops;")
    L.append(f"  contest MATTERS iff aggr@end rises and ON differs from OFF (buffers/shifts the prey-escape outcome).")

    out = "\n".join(L)
    print(out)
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "outputs")
    os.makedirs(outdir, exist_ok=True)
    fname = "exp264_smoke.txt" if smoke else "exp264.txt"
    with open(os.path.join(outdir, fname), "w") as f:
        f.write(out + "\n")
    print(f"\n[written to experiments/outputs/{fname}]")


if __name__ == "__main__":
    main()
