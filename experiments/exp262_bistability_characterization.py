"""experiments/exp262_bistability_characterization.py — Exp 262: characterize the Exp-261
BISTABILITY. Exp-261 found the costed-vs-costed Red Queen is bistable at intermediate attack_cost
(near-zero MEAN D = mean-of-opposites; per-seed split prey-dom-at-ceiling vs prey-collapsed), but
two confounds were unresolved: (a) prey in the prey-dom clade were RAILED at trait_max=4.0, and
(b) at horizon=3000 traits were still moving. So: is the bistability REAL, or a ceiling/horizon
artifact that resolves to one attractor (or a true interior equilibrium) once the ceiling is raised
and the run extended?

HYPOTHESIS / PREDICTION: the bistability is REAL — with trait_max raised (4->8) and horizon extended
(3000->5000), the intermediate attack_cost still show a BIMODAL per-seed regime split (large sd(D),
~prey-dom vs ~pred-dom), and attack_cost smoothly tunes the BASIN PROBABILITY P(prey-dom) from ~0 at
ac=0 to ~1 at ac=0.40. PREDECLARED FALSIFIER: if at the higher ceiling + longer horizon the
intermediate costs RESOLVE to a single regime (unimodal, small sd(D)) OR settle to a genuine interior
plateau with BOTH traits co-equal per-seed (|D| small per replicate, not a mean-of-opposites), then
the Exp-261 bistability was a ceiling/horizon artifact, not a real two-attractor structure.

METHOD: same robust BOTH regime + gated traits as Exp-259/260/261 (escape_cost=0.15, attack_cost
swept, both traits mutable, mutation_rate=0.15). trait_max=8.0 (raised), horizon=5000 (extended).
attack_cost sweep {0.0,0.05,0.10,0.15,0.25,0.40}; seeds 700-714 (15, superset of Exp-261's 10 for a
finer basin estimate). Per (ac,seed) is an INDEPENDENT deterministic run -> multiprocessing.Pool.
Per the mean-of-opposites-guard: report per-seed DISPERSION sd(D), the per-seed REGIME SPLIT
(prey-dom D<=-0.4 / pred-dom D>=+0.4 / balanced |D|<=0.3), bimodality, and ceiling-rail fraction at
the NEW ceiling — NOT just the mean. RAW NUMBERS — the controller adjudicates.
"""
import sys
import os
import math
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim

SWEEP = [0.0, 0.05, 0.10, 0.15, 0.25, 0.40]
SEEDS = list(range(700, 715))   # 15 seeds
HORIZON = 5000
TRAIT_MAX = 8.0
CKPTS = (1000, 2500, 5000)


def cfg(attack_cost, trait_max, horizon):
    return PatchMosaicConfig(
        n_patches=8, topology="ring", attack_a=0.05, K_pred_local=40.0, pred_self_limit_hmax=0.05,
        migration_rate_prey=0.05, migration_rate_pred=0.05,
        async_mode="rotating", async_amplitude=0.4, async_period=50.0,
        refuge_mode="per_patch", refuge_predator_access=0.30, refuge_fraction=0.25,
        enable_trait_evolution=True, mutation_rate=0.15, mutation_sd=0.06,
        escape_cost=0.15, escape_baseline=1.0, attack_cost=attack_cost, attack_baseline=1.0,
        prey_escape=1.0, pred_attack=1.0, freeze_prey_trait=False, freeze_predator_trait=False,
        trait_min=0.0, trait_max=trait_max, horizon=horizon, n_prey0_per_patch=40, n_pred0_per_patch=8)


def run_cell(args):
    """One independent (ac, seed) arms-race run. Returns end-state + trajectory checkpoints."""
    ac, seed, trait_max, horizon = args
    sim = PatchMosaicSim(cfg(ac, trait_max, horizon), seed)
    traj = {}
    while (any(p.prey for p in sim.patches) and any(p.predators for p in sim.patches)) and sim.t < horizon:
        sim.step()
        if sim.t in CKPTS:
            prey = [c.trait for p in sim.patches for c in p.prey]
            pred = [c.trait for p in sim.patches for c in p.predators]
            traj[sim.t] = (float(np.mean(prey)) if prey else None, float(np.mean(pred)) if pred else None)
    prey = [c.trait for p in sim.patches for c in p.prey]
    pred = [c.trait for p in sim.patches for c in p.predators]
    esc = float(np.mean(prey)) if prey else float("nan")
    atk = float(np.mean(pred)) if pred else float("nan")
    max_esc = float(np.max(prey)) if prey else float("nan")
    extinct = not (prey and pred)
    return dict(ac=ac, seed=seed, esc=esc, atk=atk, D=atk - esc, max_esc=max_esc,
                extinct=extinct, n_prey=len(prey), n_pred=len(pred), traj=traj)


def bimod_coeff(x):
    x = np.asarray([v for v in x if not math.isnan(v)], dtype=float)
    n = len(x)
    if n < 4 or x.std() == 0:
        return float("nan")
    m, s = x.mean(), x.std()
    sk = (((x - m) ** 3).mean()) / s ** 3
    ku = (((x - m) ** 4).mean()) / s ** 4
    return (sk ** 2 + 1) / (ku + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def summarize(rows):
    D = [r["D"] for r in rows if not math.isnan(r["D"])]
    esc = [r["esc"] for r in rows if not math.isnan(r["esc"])]
    if not D:
        return None
    preydom = sum(1 for d in D if d <= -0.40)
    preddom = sum(1 for d in D if d >= 0.40)
    bal = sum(1 for d in D if abs(d) <= 0.30)
    railed = sum(1 for r in rows if not math.isnan(r["max_esc"]) and r["max_esc"] > 0.95 * TRAIT_MAX)
    past4 = sum(1 for r in rows if not math.isnan(r["esc"]) and r["esc"] > 4.0)  # climbed past the OLD ceiling
    return dict(n=len(rows), meanD=float(np.mean(D)), sdD=float(np.std(D)),
                preydom=preydom, preddom=preddom, bal=bal,
                p_preydom=preydom / len(D), bimod=bimod_coeff(D),
                railed_new_ceiling=railed, esc_past_old_ceiling=past4,
                mean_esc=float(np.mean(esc)))


def main():
    smoke = "--smoke" in sys.argv
    sweep = [0.0, 0.15] if smoke else SWEEP
    seeds = SEEDS[:3] if smoke else SEEDS
    horizon = 1500 if smoke else HORIZON
    nproc = min((os.cpu_count() or 2) - 1, 12)

    cells = [(ac, s, TRAIT_MAX, horizon) for ac in sweep for s in seeds]
    # ceiling/horizon control: ac=0.15 at the OLD trait_max=4.0 + OLD horizon=3000 to compare regimes
    if not smoke:
        cells += [(0.15, s, 4.0, 3000) for s in seeds]
        cells += [(0.10, s, 4.0, 3000) for s in seeds]

    with mp.Pool(nproc) as pool:
        results = pool.map(run_cell, cells)

    # group main run (trait_max=8, h=5000) vs control (4,3000) by the cell's config — recover via cells order
    main_rows = {ac: [] for ac in sweep}
    ctrl_rows = {0.15: [], 0.10: []}
    for (ac, s, tmax, hz), r in zip(cells, results):
        if tmax == TRAIT_MAX:
            main_rows[ac].append(r)
        else:
            ctrl_rows[ac].append(r)

    L = []
    L.append("=" * 104)
    L.append(f"Exp 262 — characterize Exp-261 BISTABILITY: trait_max={TRAIT_MAX} (raised), horizon={horizon} (extended). RAW.")
    L.append(f"seeds={seeds}; basin map + per-seed regime split + bimodality (mean-of-opposites-guard). {nproc} procs.")
    L.append("=" * 104)
    L.append(f"{'attack_cost':>11} | {'meanD':>7} {'sd(D)':>6} | {'P(prey-dom)':>11} | split[prey/pred/bal] | {'bimod':>6} | {'esc>4.0':>7} {'railed@8':>8} | {'mean_esc':>8}")
    L.append("-" * 104)
    for ac in sweep:
        s = summarize(main_rows[ac])
        if s:
            L.append(f"{ac:>11.2f} | {s['meanD']:>+7.3f} {s['sdD']:>6.3f} | {s['p_preydom']:>11.2f} | {s['preydom']:>2}/{s['preddom']:>2}/{s['bal']:>2}{'':>11} | {s['bimod']:>6.3f} | {s['esc_past_old_ceiling']:>5}/{s['n']} {s['railed_new_ceiling']:>6}/{s['n']} | {s['mean_esc']:>8.3f}")
    L.append("")
    L.append("CEILING/HORIZON CONTROL (ac=0.10 & 0.15 at OLD trait_max=4.0, horizon=3000 = Exp-261 setup):")
    for ac in (0.10, 0.15):
        s = summarize(ctrl_rows.get(ac, [])); m = summarize(main_rows.get(ac, []))
        if s and m:
            L.append(f"  ac={ac}: OLD(4.0/3000) sd(D)={s['sdD']:.3f} P(prey-dom)={s['p_preydom']:.2f} split {s['preydom']}/{s['preddom']}/{s['bal']}  ->  NEW(8.0/{horizon}) sd(D)={m['sdD']:.3f} P(prey-dom)={m['p_preydom']:.2f} split {m['preydom']}/{m['preddom']}/{m['bal']} (esc>4.0: {m['esc_past_old_ceiling']}/{m['n']})")
    L.append("")
    L.append("PREDECLARED READOUT (controller adjudicates):")
    L.append(f"  basin curve P(prey-dom) vs attack_cost: " + ", ".join(f"ac{ac:g}={summarize(main_rows[ac])['p_preydom']:.2f}" for ac in sweep if summarize(main_rows[ac])))
    mono = True
    ps = [summarize(main_rows[ac])['p_preydom'] for ac in sweep if summarize(main_rows[ac])]
    mono = all(ps[i + 1] >= ps[i] - 0.15 for i in range(len(ps) - 1))
    L.append(f"    -> P(prey-dom) rises with attack_cost (tol 0.15): {mono}")
    inter = [ac for ac in sweep if 0 < ac < 0.40 and summarize(main_rows[ac]) and summarize(main_rows[ac])['sdD'] > 0.45]
    L.append(f"  intermediate costs still BIMODAL/high-dispersion (sd(D)>0.45) at the raised ceiling+horizon: {inter if inter else 'NONE -> resolves to monostable (falsifier)'}")
    L.append(f"  bistability REAL iff intermediate costs keep large sd(D) + opposite-regime split AND esc climbs past the old 4.0 ceiling (ceiling was limiting, regime unchanged).")

    out = "\n".join(L)
    print(out)
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "outputs")
    os.makedirs(outdir, exist_ok=True)
    fname = "exp262_bistability_smoke.txt" if smoke else "exp262_bistability_characterization.txt"
    with open(os.path.join(outdir, fname), "w") as f:
        f.write(out + "\n")
    print(f"\n[written to experiments/outputs/{fname}]")


if __name__ == "__main__":
    main()
