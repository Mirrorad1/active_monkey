"""Exp 236 — navigation-policy diagnostic (Loop B research, environmental-complexity Rung 1b).

Exp 235 found climb_ability behaviorally INERT: the local greedy forager never navigates to the
sealed plateau. The human's pick was to add a navigation-capable forage policy (enable_navigation,
byte-identical OFF). This diagnostic measures, with the HONEST (neutral lowest-index target
tie-break) navigation, whether navigation makes climb_ability expressible-and-evolvable-testable.

THE GAMING CATCH (reproducible via git, the methodology headline of this iteration):
the first navigation build (commit ce78046) reported 63.7% gate-open plateau intake — but that was
an ARTIFACT of a HIGHER-INDEX target tie-break: plateau cells have high indices, so creatures went
there by spatial fiat, NOT by seeking food. The de-gamed build (commit 19a586e, neutral lowest-index
tie-break, the codebase convention) reproduces the HONEST numbers below. The contrast 63.7% -> ~0.5%
across those two commits IS the catch; the validator caught it by re-running with the neutral tie-break.

Predeclared falsifier / hypothesis: the prediction is that navigation makes climb_ability expressible
(a large 0.05->0.10 plateau-access marginal with the gate closed). FALSIFIER: if, even with honest
navigation, the gate-closed marginal is ~0 (navigator retreats on a failed crossing roll; persistence
starves the population), then navigation does NOT rescue the locomotion test on this substrate
(NULL/INVALID — reach/cross/survive cannot be jointly satisfied).

Run:  PYTHONPATH=. uv run --python .venv python experiments/exp236_nav_diagnostic.py
Honest research only; no gradient batch is run (the substrate is shown unable to pose the question).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp235_manip_check as mc  # noqa: E402


def _share(nav, gates, **over):
    r = mc.manipulation_check({"enable_navigation": nav, "terrain_gates_movement": gates, **over})
    return r


def main() -> None:
    print("=" * 72)
    print("Exp 236 — navigation-policy diagnostic (HONEST neutral tie-break, commit 19a586e)")
    print("=" * 72)

    print("\n[1] COMFORTABLE config, GATE OPEN (crossing free) — does honest navigation chase the")
    print("    distant plateau, or stay in the sufficient basin? (the gaming artifact was 63.7% here)")
    off = _share(False, False)
    on = _share(True, False)
    print(f"    nav OFF (local greedy): {off['resident_share']*100:5.1f}%   (Exp 235 ceiling ~1%)")
    print(f"    nav ON  (food-driven):  {on['resident_share']*100:5.1f}%   "
          f"<- honest; NOT the 63.7% index artifact")

    print("\n[2] SCARCITY (regen 0.05, lean basin conc 2.5), GATE OPEN — can forced ranging make")
    print("    navigation DISCOVER the rich plateau? (food-driven expressibility)")
    for horiz in (200, 800):
        r = _share(True, False, regen_rate=0.05, terrain_food_concentration=2.5, horizon=horiz)
        print(f"    nav ON scarce, horizon {horiz}: {r['resident_share']*100:5.1f}% plateau intake")

    print("\n[3] SCARCITY, GATE CLOSED — is climb_ability a real marginal (0.05 blocked, 0.10 more)?")
    print("    THE DECISIVE TEST: if the marginal is ~0, navigation does NOT rescue the experiment.")
    for soft in (0.05, 0.10):
        r = _share(True, True, regen_rate=0.05, terrain_food_concentration=2.5,
                   terrain_gate_softness=soft, terrain_ridge_height=0.15, horizon=400)
        g = r["shares_grid"]
        print(f"    soft={soft}: resid(0.05)={r['resident_share']*100:4.1f}% "
              f"mut(0.10)={r['mutant_share']*100:4.1f}% marginal={r['marginal']*100:+5.1f}pp | "
              f"curve {' '.join(f'{c}:{s*100:.0f}' for c, s in sorted(g.items()))}")

    print("\n[4] VERDICT: navigation reaches the plateau ONLY under scarcity + gate-open; with the")
    print("    closed stochastic gate the climb marginal is ~0 (retreat on failed roll; persistence")
    print("    starves the population). NULL/INVALID — reach/cross/survive cannot be jointly")
    print("    satisfied on this gridworld. The expressibility prerequisite needs PERCEPTION of")
    print("    distant food (Exp 237), not just path-planning over a memory map.")


if __name__ == "__main__":
    main()
