"""Exp 237 — food-gradient PERCEPTION substrate + the Gate C/D frequency-dependence catch.

Exp 235/236 showed climb_ability inert: the forager never reaches the sealed plateau (local
greedy), and memory-navigation doesn't rescue it (reach/cross/survive trilemma). The human's pick:
give the creature a resource-gradient SENSE (a "scent": distance-decayed sum of the live resource
field) so it perceives + navigates toward the rich plateau. enable_food_sense, byte-identical OFF.

WHAT THIS DIAGNOSTIC SHOWS (all gen-0, mutation off, via the Exp 235 manip-check instrument):
  (1) Perception solves REACH: gate-open plateau intake 1.0% (OFF) -> ~62% (ON).
  (2) ANTI-GAMING (L40): on a FLAT-resource world (plateau NOT richer), plateau share collapses to
      ~33% = the uniform-occupancy null (plateau is 48/144 cells). A high-index artifact would push
      toward 100%. So the lift is genuinely food-driven, not an index/position artifact.
  (3) The gen-0 MONOMORPHIC benefit curve is FLAT across climb_ability (~64% at every value) — there
      is NO per-capita benefit to climbing once perception gets everyone to the plateau (saturation).

THE GATE C/D CATCH (the headline, from the locomotion preflight at this regime — see
results/preflight/locomotion_local_gradient/, run via:
  uv run --python .venv python -m ecology.evolvability --config experiments/configs/preflight/locomotion_local_gradient.yaml):
  - local_pairwise_gradient (Gate C, 50/50 head-to-head): mutant h=0.10 "wins" 7/8 (mean_s +0.008).
  - invasion_from_rarity (Gate D, rare ~5% mutant): DOES_NOT_INVADE (1/8).
  A trait that wins at 50/50 but cannot invade from rarity, with a FLAT monomorphic benefit curve, is
  POSITIVE FREQUENCY-DEPENDENCE / a priority effect (a faster climber edges a slower one in direct
  competition for the shared depletable plateau, but gains nothing in a monomorphic population and
  cannot bootstrap from rarity), NOT directional selection. Invasion-from-rarity is the adaptive-
  dynamics gold standard => climb_ability does NOT cleanly evolve; the local-gradient wall effectively
  holds. The Preflight aggregate verdict PASS_LOCAL_GRADIENT is MISLEADING here (it weights the
  pairwise gate and does not flag the Gate C/D disagreement) — see L41.

Predeclared falsifier / hypothesis: the prediction was that perception makes climb_ability expressible
AND (if the wall extends) the gradient is flat/negative. FALSIFIER of a clean escape: a PASS that does
NOT survive the invasion-from-rarity gate + a flat monomorphic benefit curve is NOT an escape — it is
frequency-dependence. (Confirmed: that is exactly what happened.)

Run:  PYTHONPATH=. uv run --python .venv python experiments/exp237_perception.py
Honest research only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp235_manip_check as mc  # noqa: E402

GATE_C_REGIME = {
    "enable_food_sense": True, "terrain_gates_movement": True,
    "regen_rate": 0.6, "initial_resource": 0.8,
    "climb_cost_floor": 0.0, "climb_cost_slope": 0.10,
    "terrain_ridge_height": 0.15, "terrain_gate_softness": 0.08,
    "terrain_food_concentration": 2.0, "horizon": 400,
}


def main() -> None:
    print("=" * 72)
    print("Exp 237 — food-gradient perception + the Gate C/D frequency-dependence catch")
    print("=" * 72)

    print("\n[1] REACH — gate-open plateau intake, food_sense OFF vs ON (canonical config):")
    off = mc.manipulation_check({"enable_food_sense": False, "terrain_gates_movement": False})
    on = mc.manipulation_check({"enable_food_sense": True, "terrain_gates_movement": False})
    print(f"    food_sense OFF: {off['resident_share']*100:5.1f}%  (Exp 235/236 local-greedy floor)")
    print(f"    food_sense ON : {on['resident_share']*100:5.1f}%  -> perception solves REACH")

    print("\n[2] ANTI-GAMING (L40) — FLAT-resource world (plateau NOT richer): plateau share must")
    print("    collapse to the ~33% uniform null (48/144 cells), NOT rise toward 100%:")
    flat = mc.manipulation_check({"enable_food_sense": True, "terrain_gates_movement": False,
                                  "terrain_food_concentration": 0.0})
    print(f"    flat-world plateau share: {flat['resident_share']*100:5.1f}%  "
          f"(food-driven, not an index artifact)")

    print("\n[3] gen-0 MONOMORPHIC benefit curve at the Gate C regime — FLAT => benefit SATURATES:")
    r = mc.manipulation_check(GATE_C_REGIME)
    for c, s in sorted(r["shares_grid"].items()):
        print(f"    climb {c:<5} -> {s*100:5.1f}% plateau intake")
    print(f"    resident(0.05)={r['resident_share']*100:.1f}%  mutant(0.10)={r['mutant_share']*100:.1f}%"
          f"  marginal={r['marginal']*100:+.2f}pp  (no per-capita benefit to climbing)")

    print("\n[4] THE CATCH (locomotion preflight at this regime, n_valid=8/8):")
    print("    Gate C local_pairwise_gradient (50/50): POSITIVE_LOCAL_GRADIENT, mutant wins 7/8, mean_s +0.008")
    print("    Gate D invasion_from_rarity (rare ~5%): DOES_NOT_INVADE, increase 1/8")
    print("    => FLAT monomorphic benefit + pairwise-win + no-invasion-from-rarity = POSITIVE")
    print("       FREQUENCY-DEPENDENCE (priority/interference on the shared plateau), NOT directional")
    print("       selection. climb_ability does NOT cleanly evolve; the local-gradient wall holds.")
    print("    Invasion-from-rarity is the binding adaptive-dynamics criterion (L41). The Preflight")
    print("    aggregate PASS_LOCAL_GRADIENT is misleading when Gate C and Gate D disagree.")


if __name__ == "__main__":
    main()
