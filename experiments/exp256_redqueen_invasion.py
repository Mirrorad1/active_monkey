"""experiments/exp256_redqueen_invasion.py — Exp 256: the Red Queen invasion test, finally
posable, on the well-mixed Bazykin substrate (Exp 255) where predator-prey COEXIST stably.

This is the original pre-registered experiment (spec 1a2c634): does prey escape-speed invade from
rarity, and does CO-EVOLUTION of the predator change the evolutionary outcome vs a STATIC predator?
The single causal bit is `freeze_predator_trait` (True = static predator; False = co-evolving).
Escape-speed is expressed by construction (faster prey -> lower capture), so the confound that
voided the spatial Rung-1 is gone.

HYPOTHESIS / PREDICTION (Red Queen): under a STATIC predator, prey escape-speed invades from rarity
and rises to a finite ESS (cost balances the fixed escape benefit), then PLATEAUS. Under a
CO-EVOLVING predator, the predator's attack trait TRACKS rising prey escape (arms race), so
selection on escape does not saturate -> SUSTAINED directional escalation of BOTH traits (the Red
Queen). PREDECLARED FALSIFIER: if the co-evolving and static arms are indistinguishable (no
sustained escalation under co-evolution; predator attack does not track prey escape), co-evolution
does not change the outcome on this substrate and the Red Queen escape is NOT demonstrated.

RAW NUMBERS — controller judges.
"""
import sys
import os

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ecology.wellmixed import WellMixedSim, WellMixedConfig, Critter

SEEDS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Part A — invasion-from-rarity: a rare breed-true faster mutant into a resident.
# ---------------------------------------------------------------------------
def part_A(coevolve, seed, burn=400, post=800, n_mut=6, mut_escape=1.4, thresh=1.2):
    # prey breed TRUE (clean two-class invasion); predator static or co-evolving.
    cfg = WellMixedConfig(
        freeze_prey_trait=True,
        freeze_predator_trait=(not coevolve),
        mutation_rate=(0.1 if coevolve else 0.0),
        mutation_sd=0.05, horizon=10 ** 9,
        prey_escape0=1.0, pred_attack0=1.0,
    )
    sim = WellMixedSim(cfg, seed)
    for _ in range(burn):
        if not sim.prey or not sim.predators:
            return None
        sim.step()
    if not sim.prey or not sim.predators:
        return None
    for _ in range(n_mut):
        sim.prey.append(Critter("prey", mut_escape, sim._next_cid))
        sim._next_cid += 1
    f0 = sum(1 for p in sim.prey if p.trait > thresh) / len(sim.prey)
    fracs = []
    for _ in range(post):
        if not sim.prey or not sim.predators:
            break
        sim.step()
        if sim.prey:
            fracs.append(sum(1 for p in sim.prey if p.trait > thresh) / len(sim.prey))
    f_final = fracs[-1] if fracs else 0.0
    return {"f0": f0, "f_final": f_final, "f_max": (max(fracs) if fracs else 0.0),
            "invaded": f_final > f0 * 1.5, "fixed": f_final > 0.9}


# ---------------------------------------------------------------------------
# Part B — open-ended arms race: prey escape mutable; predator static vs co-evolving.
# ---------------------------------------------------------------------------
def part_B(coevolve, seed, horizon=3000, checks=(250, 500, 1000, 2000, 3000)):
    cfg = WellMixedConfig(
        freeze_prey_trait=False, mutation_rate=0.15, mutation_sd=0.06,
        freeze_predator_trait=(not coevolve),
        horizon=horizon, prey_escape0=1.0, pred_attack0=1.0,
    )
    sim = WellMixedSim(cfg, seed)
    esc_at, atk_at = {}, {}
    esc_series = []
    for _ in range(horizon):
        if not sim.prey or not sim.predators:
            break
        sim.step()
        if sim.prey:
            esc_series.append(float(np.mean([p.trait for p in sim.prey])))
        if sim.t in checks:
            esc_at[sim.t] = float(np.mean([p.trait for p in sim.prey])) if sim.prey else None
            atk_at[sim.t] = float(np.mean([q.trait for q in sim.predators])) if sim.predators else None
    # late-window slope of prey escape (per 1000 steps) — sustained escalation if > ~0
    late = esc_series[-1000:] if len(esc_series) > 1200 else esc_series
    slope = (late[-1] - late[0]) / max(1, len(late)) * 1000 if len(late) > 10 else float("nan")
    return {"t_end": sim.t, "esc_at": esc_at, "atk_at": atk_at,
            "esc_final": (esc_series[-1] if esc_series else None),
            "late_slope_per_1k": slope, "extinct": (not sim.prey or not sim.predators)}


def main():
    L = []
    L.append("=" * 100)
    L.append("Exp 256 — RED QUEEN invasion test on the well-mixed Bazykin substrate. RAW — controller judges.")
    L.append(f"single causal bit = freeze_predator_trait (static) vs co-evolving; seeds={SEEDS}")
    L.append("=" * 100)

    # Part A
    L.append("(A) INVASION-FROM-RARITY: rare breed-true mutant escape=1.4 into resident escape=1.0 at equilibrium.")
    L.append(f"{'arm':>10} {'seed':>4} {'f0':>7} {'f_final':>8} {'f_max':>7} {'invaded':>8} {'fixed':>6}")
    L.append("-" * 60)
    for coevolve in (False, True):
        arm = "co-evolve" if coevolve else "static"
        inv = fix = n = 0
        for s in SEEDS:
            r = part_A(coevolve, s)
            if r is None:
                L.append(f"{arm:>10} {s:>4}   (resident failed to establish)")
                continue
            n += 1; inv += int(r["invaded"]); fix += int(r["fixed"])
            L.append(f"{arm:>10} {s:>4} {r['f0']:>7.3f} {r['f_final']:>8.3f} {r['f_max']:>7.3f} "
                     f"{str(r['invaded']):>8} {str(r['fixed']):>6}")
        L.append(f"  -> {arm}: invaded {inv}/{n}, fixed {fix}/{n}")
        L.append("")

    # Part B
    L.append("(B) OPEN-ENDED ARMS RACE: prey escape mutable; predator STATIC vs CO-EVOLVING. Trait means at checkpoints.")
    L.append(f"{'arm':>10} {'seed':>4} {'esc@250':>8} {'esc@1k':>7} {'esc@2k':>7} {'esc@3k':>7} "
             f"{'atk@3k':>7} {'slope/1k':>9} {'extinct':>7}")
    L.append("-" * 80)
    summ = {}
    for coevolve in (False, True):
        arm = "co-evolve" if coevolve else "static"
        escs, atks, slopes = [], [], []
        for s in SEEDS:
            r = part_B(coevolve, s)
            e = r["esc_at"]; a = r["atk_at"]
            L.append(f"{arm:>10} {s:>4} {str(round(e.get(250) or 0,2)):>8} {str(round(e.get(1000) or 0,2)):>7} "
                     f"{str(round(e.get(2000) or 0,2)):>7} {str(round(e.get(3000) or 0,2)):>7} "
                     f"{str(round(a.get(3000) or 0,2)):>7} {r['late_slope_per_1k']:>9.3f} {str(r['extinct']):>7}")
            if e.get(3000) is not None:
                escs.append(e[3000])
            if a.get(3000) is not None:
                atks.append(a[3000])
            if r["late_slope_per_1k"] == r["late_slope_per_1k"]:
                slopes.append(r["late_slope_per_1k"])
        summ[arm] = (np.mean(escs) if escs else float('nan'),
                     np.mean(atks) if atks else float('nan'),
                     np.mean(slopes) if slopes else float('nan'))
        L.append(f"  -> {arm}: mean esc@3k={summ[arm][0]:.2f}, mean atk@3k={summ[arm][1]:.2f}, mean late-slope/1k={summ[arm][2]:.3f}")
        L.append("")

    L.append("RED QUEEN SIGNATURE (single causal bit = predator co-evolution):")
    L.append(f"  static:    esc@3k={summ['static'][0]:.2f} atk@3k={summ['static'][1]:.2f} late-slope={summ['static'][2]:.3f}")
    L.append(f"  co-evolve: esc@3k={summ['co-evolve'][0]:.2f} atk@3k={summ['co-evolve'][1]:.2f} late-slope={summ['co-evolve'][2]:.3f}")
    L.append("  Red Queen CONFIRMED iff co-evolve shows predator attack TRACKING prey escape (atk@3k >> static)")
    L.append("  AND sustained escalation (co-evolve late-slope > static late-slope, static ~plateaued).")
    out = "\n".join(L)
    print(out)
    with open(os.path.join(_repo_root, "experiments", "outputs", "exp256_redqueen_invasion.txt"), "w") as f:
        f.write(out + "\n")
    print("\n[written to experiments/outputs/exp256_redqueen_invasion.txt]")


if __name__ == "__main__":
    main()
