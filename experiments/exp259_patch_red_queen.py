"""experiments/exp259_patch_red_queen.py — Exp 259: the discrete Red Queen, finally posable, on
the patch-mosaic substrate (Exp 257 posed coexistence; Exp 258 showed it robust). Rung 4.

Does prey escape-speed invade from rarity, and does CO-EVOLUTION of the predator change the
outcome vs a STATIC predator — now on a DISCRETE SPATIAL (metapopulation) substrate, not well-mixed?
Single causal bit = `freeze_predator_trait` (True=static, False=co-evolving). Run on the robust
coexisting BOTH regime (refuge + migration + asynchrony) where coexistence is posed (Exp 257/258).
Predation already uses INDIVIDUAL traits (per prey-predator pair, individual capture credit), so
escape/attack are genuinely under individual selection (no Exp-256-style pooled-credit confound).

HYPOTHESIS / PREDICTION: (A) a rare faster-prey mutant invades from rarity (trait selectable on the
patch substrate). (B) under a CO-EVOLVING predator, predator attack ESCALATES to track prey escape
(arms race in the antagonist), absent under a static predator. PREDECLARED FALSIFIER: if the
co-evolving and static arms are indistinguishable (predator attack does not adapt; prey outcome
unchanged), co-evolution does not engage on the patch substrate.

Regime: n_patches=8 ring, BOTH (refuge access=0.30 frac=0.25, migration=0.05, async rotating amp=0.4),
collapse-prone within-patch (attack_a=0.05, K_pred=40, hmax=0.05), enable_trait_evolution.
FRESH seeds 400-409 (sanity used 300-302). RAW NUMBERS — controller judges.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim, Critter

SEEDS = list(range(400, 410))
HORIZON_B = 3000


def base_cfg(coevolve, prey_mutable, horizon, mutation_rate=0.15):
    return PatchMosaicConfig(
        n_patches=8, topology="ring", attack_a=0.05, K_pred_local=40.0, pred_self_limit_hmax=0.05,
        migration_rate_prey=0.05, migration_rate_pred=0.05,
        async_mode="rotating", async_amplitude=0.4, async_period=50.0,
        refuge_mode="per_patch", refuge_predator_access=0.30, refuge_fraction=0.25,
        enable_trait_evolution=True, mutation_rate=mutation_rate, mutation_sd=0.06,
        escape_cost=0.15, escape_baseline=1.0, prey_escape=1.0, pred_attack=1.0,
        freeze_prey_trait=(not prey_mutable), freeze_predator_trait=(not coevolve),
        horizon=horizon, n_prey0_per_patch=40, n_pred0_per_patch=8)


def _alive(sim):
    return any(p.prey for p in sim.patches) and any(p.predators for p in sim.patches)


def part_A(coevolve, seed, burn=400, post=900, n_mut=8, mut_escape=1.4, thresh=1.2):
    # prey breed TRUE (freeze_prey_trait); predator static or co-evolving.
    cfg = base_cfg(coevolve=coevolve, prey_mutable=False, horizon=burn + post + 5)
    sim = PatchMosaicSim(cfg, seed)
    for _ in range(burn):
        if not _alive(sim):
            return None
        sim.step()
    if not _alive(sim):
        return None
    # inject rare breed-true mutants (escape=1.4), spread across patches
    for k in range(n_mut):
        patch = sim.patches[k % len(sim.patches)]
        patch.prey.append(Critter("prey", mut_escape, sim._next_cid)); sim._next_cid += 1
    def mut_frac():
        prey = [c.trait for p in sim.patches for c in p.prey]
        return (sum(t > thresh for t in prey) / len(prey)) if prey else 0.0
    f0 = mut_frac(); fracs = []
    for _ in range(post):
        if not _alive(sim):
            break
        sim.step(); fracs.append(mut_frac())
    f_final = fracs[-1] if fracs else 0.0
    return dict(f0=f0, f_final=f_final, f_max=(max(fracs) if fracs else 0.0),
                invaded=f_final > f0 * 1.5, fixed=f_final > 0.9)


def part_B(coevolve, seed, horizon=HORIZON_B, checks=(500, 1000, 2000, 3000)):
    cfg = base_cfg(coevolve=coevolve, prey_mutable=True, horizon=horizon)
    sim = PatchMosaicSim(cfg, seed)
    esc_at, atk_at = {}, {}
    while _alive(sim) and sim.t < horizon:
        sim.step()
        if sim.t in checks:
            prey = [c.trait for p in sim.patches for c in p.prey]
            pred = [c.trait for p in sim.patches for c in p.predators]
            esc_at[sim.t] = float(np.mean(prey)) if prey else None
            atk_at[sim.t] = float(np.mean(pred)) if pred else None
    return dict(t_end=sim.t, esc_at=esc_at, atk_at=atk_at, extinct=not _alive(sim))


def main():
    L = []
    L.append("=" * 96)
    L.append("Exp 259 — DISCRETE RED QUEEN on the patch-mosaic substrate (BOTH regime). RAW — controller judges.")
    L.append(f"single causal bit = freeze_predator_trait (static vs co-evolving); FRESH seeds {SEEDS}")
    L.append("=" * 96)

    L.append("(A) INVASION-FROM-RARITY: rare breed-true mutant escape=1.4 into resident escape=1.0 (at equilibrium).")
    L.append(f"{'arm':>10} {'invaded':>8} {'fixed':>6} {'mean f0':>8} {'mean f_final':>12} {'n':>3}")
    L.append("-" * 52)
    for coevolve in (False, True):
        arm = "co-evolve" if coevolve else "static"
        rs = [part_A(coevolve, s) for s in SEEDS]; rs = [r for r in rs if r]
        inv = sum(r["invaded"] for r in rs); fix = sum(r["fixed"] for r in rs)
        L.append(f"{arm:>10} {inv:>5}/{len(rs)} {fix:>4}/{len(rs)} {np.mean([r['f0'] for r in rs]):>8.3f} {np.mean([r['f_final'] for r in rs]):>12.3f} {len(rs):>3}")
    L.append("")

    L.append("(B) OPEN-ENDED ARMS RACE: prey escape mutable; predator STATIC vs CO-EVOLVING. Global trait means.")
    L.append(f"{'arm':>10} {'esc@500':>8} {'esc@1k':>7} {'esc@2k':>7} {'esc@3k':>7} {'atk@3k':>7} {'extinct@end':>11}")
    L.append("-" * 70)
    summ = {}
    for coevolve in (False, True):
        arm = "co-evolve" if coevolve else "static"
        rs = [part_B(coevolve, s) for s in SEEDS]
        def mean_at(rs, d, t):
            vals = [r[d].get(t) for r in rs if r[d].get(t) is not None]
            return np.mean(vals) if vals else float("nan")
        e5, e1, e2, e3 = (mean_at(rs, "esc_at", t) for t in (500, 1000, 2000, 3000))
        a3 = mean_at(rs, "atk_at", 3000)
        ext = sum(r["extinct"] for r in rs)
        summ[arm] = (e3, a3)
        L.append(f"{arm:>10} {e5:>8.2f} {e1:>7.2f} {e2:>7.2f} {e3:>7.2f} {a3:>7.2f} {ext:>9}/{len(rs)}")
    L.append("")
    L.append("RED QUEEN SIGNATURE (single causal bit = predator co-evolution):")
    L.append(f"  static:    esc@3k={summ['static'][0]:.2f} atk@3k={summ['static'][1]:.2f} (attack FROZEN)")
    L.append(f"  co-evolve: esc@3k={summ['co-evolve'][0]:.2f} atk@3k={summ['co-evolve'][1]:.2f}")
    L.append("  Red Queen ENGAGES iff co-evolve predator attack RISES (atk@3k >> static's frozen ~1.0) = arms race in the antagonist.")
    out = "\n".join(L)
    print(out)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
              "experiments", "outputs", "exp259_patch_red_queen.txt"), "w") as f:
        f.write(out + "\n")
    print("\n[written to experiments/outputs/exp259_patch_red_queen.txt]")


if __name__ == "__main__":
    main()
