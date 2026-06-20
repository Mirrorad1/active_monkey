"""experiments/exp256b_cost_calibration.py — Exp 256b: nail the Red Queen signature clean by
raising the escape COST so prey escape settles at a finite ESS (below the trait cap) under a
STATIC predator, and confirm a CO-EVOLVING predator still drives the arms race (attack tracks
escape; prey escape pushed higher).

Exp 256 (fixed model) showed the Red Queen arms race (predator attack 1.0->3.05 under co-evolution
vs frozen 1.0 static), but the weak default escape_cost=0.15 let prey escape run toward the trait
cap (4.0) in BOTH arms, compressing the prey-escape contrast. Here we sweep escape_cost so the
static arm PLATEAUS at a finite ESS well below the cap; the clean prediction:
  - STATIC: prey escape -> finite ESS (below cap), predator attack flat.
  - CO-EVOLVE: predator attack TRACKS UP (arms race), prey escape pushed HIGHER than the static ESS.

PREDECLARED FALSIFIER: if at a cost where the static arm plateaus below the cap, the co-evolving
arm does NOT push prey escape higher AND does NOT raise predator attack, then the arms race was a
cap artifact and the Red Queen is not robustly demonstrated.

RAW NUMBERS — controller judges.
"""
import sys
import os

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ecology.wellmixed import WellMixedSim, WellMixedConfig

SEEDS = [0, 1, 2, 3, 4]
HORIZON = 3000


def run_B(coevolve, seed, escape_cost):
    cfg = WellMixedConfig(
        freeze_prey_trait=False, mutation_rate=0.15, mutation_sd=0.06,
        freeze_predator_trait=(not coevolve),
        escape_cost=escape_cost,
        horizon=HORIZON, prey_escape0=1.0, pred_attack0=1.0,
    )
    sim = WellMixedSim(cfg, seed)
    esc_series = []
    for _ in range(HORIZON):
        if not sim.prey or not sim.predators:
            break
        sim.step()
        if sim.prey:
            esc_series.append(float(np.mean([p.trait for p in sim.prey])))
    esc_final = float(np.mean(esc_series[-200:])) if len(esc_series) > 200 else (esc_series[-1] if esc_series else None)
    atk_final = float(np.mean([q.trait for q in sim.predators])) if sim.predators else None
    extinct = (not sim.prey or not sim.predators)
    return esc_final, atk_final, extinct


def main():
    L = []
    L.append("=" * 96)
    L.append("Exp 256b — Red Queen cost calibration: does the arms race survive a finite-ESS escape cost?")
    L.append(f"sweep escape_cost; STATIC vs CO-EVOLVE predator; trait cap = 4.0; horizon={HORIZON} seeds={SEEDS}")
    L.append("=" * 96)
    L.append(f"{'esc_cost':>8} {'arm':>10} {'esc_final':>10} {'atk_final':>10} {'extinct':>8}")
    L.append("-" * 60)
    summary = {}
    for cost in (0.15, 0.30, 0.50):
        for coevolve in (False, True):
            arm = "co-evolve" if coevolve else "static"
            escs, atks, ext = [], [], 0
            for s in SEEDS:
                e, a, x = run_B(coevolve, s, cost)
                if e is not None:
                    escs.append(e)
                if a is not None:
                    atks.append(a)
                ext += int(x)
            me = np.mean(escs) if escs else float("nan")
            ma = np.mean(atks) if atks else float("nan")
            summary[(cost, arm)] = (me, ma, ext)
            L.append(f"{cost:>8.2f} {arm:>10} {me:>10.2f} {ma:>10.2f} {ext:>8}")
        L.append("")

    L.append("RED QUEEN READ (per escape_cost): static esc(ESS) / atk vs co-evolve esc / atk")
    for cost in (0.15, 0.30, 0.50):
        s_e, s_a, _ = summary[(cost, "static")]
        c_e, c_a, _ = summary[(cost, "co-evolve")]
        below_cap = "static<cap" if s_e < 3.5 else "static~cap"
        race = "ATTACK TRACKS (co>static)" if c_a > s_a + 0.5 else "no attack tracking"
        push = "prey pushed higher" if c_e > s_e + 0.2 else "no extra prey push"
        L.append(f"  cost={cost}: static esc={s_e:.2f} atk={s_a:.2f} [{below_cap}] | "
                 f"co-evolve esc={c_e:.2f} atk={c_a:.2f} -> {race}; {push}")
    out = "\n".join(L)
    print(out)
    with open(os.path.join(_repo_root, "experiments", "outputs", "exp256b_cost_calibration.txt"), "w") as f:
        f.write(out + "\n")
    print("\n[written to experiments/outputs/exp256b_cost_calibration.txt]")


if __name__ == "__main__":
    main()
