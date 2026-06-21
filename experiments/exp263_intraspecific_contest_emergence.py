"""experiments/exp263_intraspecific_contest_emergence.py — Exp 263 (R1 of the
emergent-intraspecific-competition direction): does costed prey-vs-prey CONTEST aggression
EMERGE under selection — purely from reproduction-opportunity-under-scarcity — when we add the
gated `enable_contest` affordance, with NO reward for fighting? Or does it hit the local-gradient WALL?

HYPOTHESIS / PREDICTION: a heritable, costed aggression propensity `aggr` (the OPTION to seize a
share of a competitor's reproduction opportunity, costed by contest_cost*aggr) is SELECTED — has a
positive LOCAL selection gradient at the resident AND invades from rarity — under SCARCITY (prey
saturated near K, so the crowding prize crowd_prize=contest_seize*min(1,N/K) is large) but NOT under
ABUNDANCE (a strong frozen predator holds prey well below K -> crowd_prize~0 -> only the cost bites).

SCARCITY KNOB = frozen predation pressure (attack_a). At logistic equilibrium N->K regardless of K,
so K alone is NOT a scarcity knob; predation sets how far below K prey sit. SCARCE = weak predator
(prey saturate, N/K~1); ABUNDANT = strong predator (prey suppressed, N/K small, reproduction headroom).

PREDECLARED FALSIFIER (claim WALL / no emergence if ANY): the prey-aggr local gradient under SCARCITY
is NOT positive (wins < 7/8); OR the DRIFT-NULL (contest made causally inert: contest_cost=0 AND
contest_seize=0 -> aggr changes nothing) is NON-neutral (wins>=7/8 — a measurement artifact -> NO_VERDICT);
OR the ABUNDANCE arm IS selected (aggr positive under abundance too -> not scarcity-driven -> confound).

VERDICT: AGGRESSION_EMERGES (scarce POS + invades, abundance neutral/NEG, drift-null neutral, cost-null
POS) / WALL (scarce gradient sub-threshold) / NO_VERDICT (drift-null fires / collapse / un-gated bimodality).
Reads PER-SEED dispersion (mean-of-opposites-guard), not just the win count. RAW — controller adjudicates.
Reuses ecology/evolvability/metrics.py. FRESH seeds: gradient 600-607, invasion 610-619.
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim, Critter
from ecology.evolvability.metrics import selection_coefficient_freq, count_wins, default_thresholds

SEEDS_G = list(range(600, 608))    # gradient preflight: 8 seeds
SEEDS_I = list(range(610, 620))    # invasion-from-rarity: 10 seeds
EPS = 0.1
WINDOW = 400
MIN_FOCAL = 40
ATTACK_SCARCE = 0.01               # weak predator -> prey saturate near K (scarce reproduction)
ATTACK_ABUNDANT = 0.08             # strong predator -> prey held below K (abundant reproduction)
K_PREY = 120.0


def cfg_R1(resident_aggr, attack_a, contest_cost, contest_seize, horizon,
           freeze_prey=True, mutation_rate=0.0):
    """R1 regime: ring-8, prey-aggr focal, predator FROZEN (strength = scarcity knob via attack_a).
    No refuge (prey-prey contest needs none). enable_contest ON; escape frozen at baseline."""
    return PatchMosaicConfig(
        n_patches=8, topology="ring", attack_a=attack_a, handling_h=0.02,
        K_prey_local=K_PREY, K_pred_local=40.0, pred_self_limit_hmax=0.15,
        migration_rate_prey=0.05, migration_rate_pred=0.05,
        async_mode="rotating", async_amplitude=0.4, async_period=50.0,
        refuge_mode="none",
        enable_trait_evolution=True, enable_contest=True,
        mutation_rate=mutation_rate, aggr_mutation_sd=0.05,
        escape_cost=0.15, escape_baseline=1.0, prey_escape=1.0, pred_attack=1.0,
        contest_cost=contest_cost, contest_seize=contest_seize, contest_dissipation=0.0,
        aggr0=resident_aggr, freeze_prey_trait=freeze_prey, freeze_predator_trait=True,
        trait_min=0.0, trait_max=4.0, track_lineages=True,
        horizon=horizon, n_prey0_per_patch=40, n_pred0_per_patch=8)


def _prey_aggrs(sim):
    return [c.aggr for p in sim.patches for c in p.prey]


def gradient_probe(resident_aggr, attack_a, contest_cost, contest_seize, seed, eps=EPS, window=WINDOW):
    """Breed-true 50/50 common garden on aggr: resident(aggr=a) vs mutant(aggr=a+eps).
    Returns selection coefficient s (mutant favoured iff s>0), plus mean N/K over the window."""
    cfg = cfg_R1(resident_aggr, attack_a, contest_cost, contest_seize, horizon=window, freeze_prey=True)
    sim = PatchMosaicSim(cfg, seed)
    thr = resident_aggr + eps / 2.0
    for patch in sim.patches:
        for k, c in enumerate(patch.prey):
            c.aggr = (resident_aggr + eps) if (k % 2 == 1) else resident_aggr

    def mut_frac():
        a = _prey_aggrs(sim)
        return (sum(x > thr for x in a) / len(a)) if a else None

    f0 = mut_frac()
    nk = []
    exploded = False
    for _ in range(window):
        n_prey = sum(len(p.prey) for p in sim.patches)
        if n_prey == 0 or sum(len(p.predators) for p in sim.patches) == 0:
            break
        nk.append(n_prey / (cfg.n_patches * cfg.K_prey_local))
        sim.step()
        if sum(len(p.prey) + len(p.predators) for p in sim.patches) > cfg.pop_cap:
            exploded = True
            break
    n_focal = len(_prey_aggrs(sim))
    f1 = mut_frac()
    valid = (n_focal >= MIN_FOCAL) and (not exploded)
    s = selection_coefficient_freq(f0, f1, window) if (f0 is not None and f1 is not None and valid) else float("nan")
    return s, (float(np.mean(nk)) if nk else float("nan"))


def cell(resident_aggr, attack_a, contest_cost, contest_seize, seeds=SEEDS_G):
    rows = [gradient_probe(resident_aggr, attack_a, contest_cost, contest_seize, s) for s in seeds]
    svals = [r[0] for r in rows]
    nk = float(np.nanmean([r[1] for r in rows]))
    wins, n_valid = count_wins([(s, 0.0) for s in svals], eps=0.0)
    valid_s = [s for s in svals if not math.isnan(s)]
    win_t, lose_t = default_thresholds(n_valid) if n_valid > 0 else (0, 0)
    verdict = "NO_DATA" if n_valid == 0 else ("POS" if wins >= win_t else ("NEG" if wins <= lose_t else "AMBIG"))
    return dict(wins=wins, n_valid=n_valid, mean_s=(float(np.mean(valid_s)) if valid_s else float("nan")),
                sd_s=(float(np.std(valid_s)) if valid_s else float("nan")), nk=nk, verdict=verdict, svals=svals)


def invasion(attack_a, seed, burn=300, post=600, n_mut=8, mut_aggr=0.5):
    """Rare breed-true aggr=0.5 mutant injected into an aggr=0 resident at equilibrium; does it rise?"""
    cfg = cfg_R1(0.0, attack_a, 0.10, 0.50, horizon=burn + post + 5, freeze_prey=True)
    sim = PatchMosaicSim(cfg, seed)
    for _ in range(burn):
        if sum(len(p.prey) for p in sim.patches) == 0:
            return None
        sim.step()
    for k in range(n_mut):
        patch = sim.patches[k % len(sim.patches)]
        if patch.prey:
            patch.prey.append(Critter("prey", 1.0, sim._next_cid, aggr=mut_aggr, lineage=999)); sim._next_cid += 1

    def mut_frac():
        a = _prey_aggrs(sim)
        return (sum(x > 0.25 for x in a) / len(a)) if a else 0.0

    f0 = mut_frac(); fracs = []
    for _ in range(post):
        if sum(len(p.prey) for p in sim.patches) == 0:
            break
        sim.step(); fracs.append(mut_frac())
    f_final = fracs[-1] if fracs else 0.0
    return dict(f0=f0, f_final=f_final, invaded=f_final > max(f0 * 1.5, 0.05))


def main():
    smoke = "--smoke" in sys.argv
    sg = SEEDS_G[:2] if smoke else SEEDS_G
    si = SEEDS_I[:2] if smoke else SEEDS_I
    L = []
    L.append("=" * 100)
    L.append("Exp 263 — R1: does costed prey-vs-prey CONTEST aggression EMERGE? RAW — controller adjudicates.")
    L.append(f"breed-true 50/50 aggr common garden; eps={EPS}; window={WINDOW}; gradient seeds {sg}; 7/8-strict.")
    L.append(f"scarcity knob = frozen attack_a (SCARCE={ATTACK_SCARCE} weak pred -> prey near K; ABUNDANT={ATTACK_ABUNDANT} strong pred).")
    L.append("=" * 100)
    L.append("")
    L.append("(A) PREY-AGGR local gradient at resident aggr=0.0 (contest_cost=0.10, contest_seize=0.50):")
    L.append(f"{'regime':>16} | {'wins':>7} {'verdict':>7} | {'mean s':>9} {'sd s':>7} | {'mean N/K':>8} | per-seed s")
    L.append("-" * 100)
    cells = {}
    cells["SCARCE"] = cell(0.0, ATTACK_SCARCE, 0.10, 0.50, sg)
    cells["ABUNDANT"] = cell(0.0, ATTACK_ABUNDANT, 0.10, 0.50, sg)
    for name in ("SCARCE", "ABUNDANT"):
        c = cells[name]
        ss = " ".join(f"{s:+.3f}" if not math.isnan(s) else "nan" for s in c["svals"])
        L.append(f"{name:>16} | {c['wins']:>4}/{c['n_valid']:<2} {c['verdict']:>7} | {c['mean_s']:>+9.4f} {c['sd_s']:>7.4f} | {c['nk']:>8.3f} | {ss}")
    L.append("")
    L.append("(B) CONTROLS at resident aggr=0.0, SCARCE regime:")
    cells["DRIFT_NULL"] = cell(0.0, ATTACK_SCARCE, 0.0, 0.0, sg)     # aggr causally inert -> must be NEUTRAL
    cells["COST_NULL"] = cell(0.0, ATTACK_SCARCE, 0.0, 0.50, sg)     # benefit only -> should be POS
    for name, note in (("DRIFT_NULL", "cost=0,seize=0 inert -> must NOT be POS"),
                       ("COST_NULL", "cost=0,seize=0.50 benefit-only -> should be POS")):
        c = cells[name]
        L.append(f"{name:>16} | {c['wins']:>4}/{c['n_valid']:<2} {c['verdict']:>7} | {c['mean_s']:>+9.4f} {c['sd_s']:>7.4f} | {c['nk']:>8.3f} | {note}")
    L.append("")
    L.append("(C) INVASION-FROM-RARITY (rare aggr=0.5 breed-true into aggr=0 resident):")
    for name, aa in (("SCARCE", ATTACK_SCARCE), ("ABUNDANT", ATTACK_ABUNDANT)):
        rs = [invasion(aa, s) for s in si]; rs = [r for r in rs if r]
        inv = sum(r["invaded"] for r in rs)
        mf = float(np.mean([r["f_final"] for r in rs])) if rs else float("nan")
        L.append(f"{name:>16} | invaded {inv}/{len(rs)} | mean f_final={mf:.3f}")
    L.append("")
    L.append("PREDECLARED READOUT (controller adjudicates vs the raw rows + mean-of-opposites-guard):")
    sc, ab, dn, cn = cells["SCARCE"], cells["ABUNDANT"], cells["DRIFT_NULL"], cells["COST_NULL"]
    L.append(f"  scarce gradient: {sc['verdict']} (wins {sc['wins']}/{sc['n_valid']}, s{sc['mean_s']:+.4f}, sd {sc['sd_s']:.4f}, N/K {sc['nk']:.2f})")
    L.append(f"  abundance gradient: {ab['verdict']} (s{ab['mean_s']:+.4f}, N/K {ab['nk']:.2f}) — must NOT be POS for scarcity-driven emergence")
    L.append(f"  drift-null: {dn['verdict']} (wins {dn['wins']}/{dn['n_valid']}) — POS here => NO_VERDICT (artifact)")
    L.append(f"  cost-null: {cn['verdict']} (s{cn['mean_s']:+.4f}) — benefit alone should be POS")
    L.append(f"  => AGGRESSION_EMERGES iff scarce POS + drift-null neutral + abundance not-POS; else WALL/NO_VERDICT.")

    out = "\n".join(L)
    print(out)
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "outputs")
    os.makedirs(outdir, exist_ok=True)
    fname = "exp263_smoke.txt" if smoke else "exp263.txt"
    with open(os.path.join(outdir, fname), "w") as f:
        f.write(out + "\n")
    print(f"\n[written to experiments/outputs/{fname}]")


if __name__ == "__main__":
    main()
