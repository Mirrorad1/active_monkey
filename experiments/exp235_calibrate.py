"""Exp 235 Rung 1 — geometry calibration + bottleneck diagnostic (Loop B research helper).

The first-guess geometry FAILED the manipulation check's marginal criterion: the
gen-0 plateau_intake_share curve was NOISE in climb_ability, not monotone.  Hypothesis:
the comfort-gated foraging policy is purely LOCAL/greedy (moves only to the best adjacent
cell, never path-plans), so reaching the rim is a slow random-walk bottleneck and
climb_ability — which only gates *crossing* an already-adjacent upslope edge — is swamped
by diffusion-to-rim.

KEY DIAGNOSTIC (one of the predeclared deflation controls, doubling as the diagnosis):
run with terrain_gates_movement=False (GATE OPEN — everyone crosses freely).
  - If plateau access is STILL low with the gate open  => the bottleneck is the POLICY
    (creatures don't climb even when they can); NO gate calibration helps => substrate
    redesign needed (a NULL/INVALID per the card, not a wall result).
  - If gate-open access is HIGH but resident (gate-closed) access is LOW => the gate IS
    the bottleneck and a (ridge_height, softness, leanness) cell can pass the manip check.

Then a sweep over ridge_height x softness x food_concentration looking for a cell with
resident_share <= 0.25 (sealed) AND marginal >= 0.10 (large 0.05->0.10 unlock).

Run:  uv run --python .venv python experiments/exp235_calibrate.py
Honest research only; this PRE-validates the substrate before any gradient batch.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import sibling helper

import exp235_manip_check as mc  # noqa: E402


def _fmt(x: float) -> str:
    return "nan" if x != x else f"{x*100:5.1f}%"


def gate_open_diagnostic() -> dict:
    """Plateau access with the gate OPEN (climb irrelevant) — the policy's ceiling."""
    res = mc.manipulation_check({"terrain_gates_movement": False})
    return res


def main() -> None:
    print("=" * 72)
    print("Exp 235 calibration — bottleneck diagnostic + geometry sweep")
    print(f"thresholds: resident_share <= {mc.RESIDENT_SHARE_THRESHOLD:.2f} (sealed), "
          f"marginal >= {mc.MARGINAL_THRESHOLD:.2f} (large unlock)")
    print(f"gen-0: {mc.GEN0_N_SEEDS} seeds, horizon {mc.GEN0_HORIZON}, "
          f"resident climb {mc.RESIDENT_CLIMB}, mutant {mc.MUTANT_CLIMB}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. GATE-OPEN DIAGNOSTIC — the policy's plateau-access ceiling.
    # ------------------------------------------------------------------
    print("\n[1] GATE-OPEN diagnostic (terrain_gates_movement=False, crossing free):")
    go = gate_open_diagnostic()
    print(f"    plateau_intake_share with free crossing (resident climb): "
          f"{_fmt(go['resident_share'])}")
    print(f"    full climb grid (should be ~flat — climb is irrelevant when gate open):")
    for c, s in sorted(go["shares_grid"].items()):
        print(f"      climb {c:<5} -> {_fmt(s)}")
    ceiling = go["resident_share"]
    if ceiling != ceiling or ceiling < 0.30:
        print(f"    >> POLICY-BOTTLENECK SIGNAL: even with free crossing the population "
              f"reaches only {_fmt(ceiling)} of its food on the plateau.")
        print(f"    >> The local greedy policy does not exploit the sealed plateau; "
              f"climb_ability cannot be expressed. Gate calibration will NOT help.")
    else:
        print(f"    >> Gate-open access is {_fmt(ceiling)} (>=30%): the policy CAN exploit "
              f"the plateau; the gate is a candidate bottleneck. Proceeding to sweep.")

    # ------------------------------------------------------------------
    # 2. GEOMETRY SWEEP (gate closed).
    # ------------------------------------------------------------------
    print("\n[2] GEOMETRY SWEEP (gate closed):")
    ridge_heights = [0.05, 0.10, 0.15, 0.25]
    softnesses = [0.02, 0.05, 0.10]
    concentrations = [2.0, 2.5]   # 2.5 => leaner basin (out_factor lower), still conserved
    header = f"{'ridge':>6} {'soft':>5} {'conc':>5} | {'resid':>6} {'mutant':>6} {'marg':>6} | seal? marg? BOTH?"
    print("    " + header)
    print("    " + "-" * len(header))
    passing = []
    for conc in concentrations:
        for rh in ridge_heights:
            for sf in softnesses:
                try:
                    r = mc.manipulation_check({
                        "terrain_ridge_height": rh,
                        "terrain_gate_softness": sf,
                        "terrain_food_concentration": conc,
                        "terrain_gates_movement": True,
                    })
                except ValueError as e:
                    print(f"    {rh:>6} {sf:>5} {conc:>5} | ValueError: {e}")
                    continue
                seal = "Y" if r["withheld_pass"] else "."
                marg = "Y" if r["marginal_pass"] else "."
                both = "** " if r["both_pass"] else "   "
                print(f"    {rh:>6} {sf:>5} {conc:>5} | "
                      f"{_fmt(r['resident_share'])} {_fmt(r['mutant_share'])} {_fmt(r['marginal']):>6} | "
                      f"  {seal}    {marg}   {both}")
                if r["both_pass"]:
                    passing.append((rh, sf, conc, r))

    print("\n[3] SUMMARY:")
    if passing:
        print(f"    {len(passing)} geometry cell(s) PASS both criteria:")
        for rh, sf, conc, r in passing:
            print(f"      ridge={rh} soft={sf} conc={conc} -> "
                  f"resident {_fmt(r['resident_share'])}, marginal {_fmt(r['marginal'])}")
    else:
        print("    NO geometry cell passes both criteria with the current local greedy policy.")
        print("    => If the gate-open diagnostic also showed low access, the honest verdict is")
        print("       a POLICY/SUBSTRATE wall: climb_ability is behaviorally inert because the")
        print("       local forager never paths to the sealed plateau. This is a NULL/INVALID")
        print("       (substrate redesign), not a local-gradient wall result — log it as such.")


if __name__ == "__main__":
    main()
