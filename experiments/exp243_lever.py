"""Exp 243 — final lever search: can a lower regen_rate make the field genuinely
deplete (availability in the L40 band (0.05, 0.85)) while the population stays viable
(persist >= 30) AND damps under Mechanism A (hmax=0.20)?

Context: the A-strength search found a "damped band" at regen=0.5, but the FULL
certify_run audit rejected it — availability ~0.98-0.99 >> 0.85 (degenerate_depletion):
the field is saturated/non-depleting, so "stability" is just A culling on a full field
(the Exp-242 invalid regime). The viability<->depletion tension: viable populations
barely deplete the field. This sweep tests the one untested lever (regen DOWN) seeking a
NON-DEGENERATE damped equilibrium. If availability only drops below 0.85 once the pop is
starving/extinct (no overlap), the tension is binding -> clean NO-GO confirming Exp-242.

Applies the FULL certify_run gate (incl. the L40 non-degeneracy audit) to every cell.

HYPOTHESIS: adding Mechanism A (crowding mortality) + Mechanism B (floored regen) yields a
damped, non-degenerate equilibrium (a stable population that genuinely depletes the resource
field). PREDECLARED FALSIFIER: if no cell passes certify_run's non-degeneracy audit (0/N
cells, availability outside the L40 band, or population non-viable) across the full
regen/A-strength sweep, the viability-vs-depletion tension is BINDING and the substrate
cannot host an evolvability test (NO-GO).
"""
from __future__ import annotations
import sys
from ecology.evolvability.cert_run import run_cert
from ecology.evolvability import stability as S

HMAX = 0.20
KC = 60.0
THETA = 1.0
SLOPE = 0.05
HORIZON = 2500
BURN_IN = 0.6
PARAMS = dict(hmax=HMAX, Kc=KC, theta=THETA)

REGENS = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15]   # 0.5 = the degenerate reference; sweep DOWN
SPEEDS = [1.5, 2.0]
SEEDS = [1, 2]

lines = []
def emit(s):
    print(s, flush=True)
    lines.append(s)

emit("Exp 243 lever search: regen DOWN vs availability/viability/damping")
emit(f"fixed: hmax={HMAX} Kc={KC} theta={THETA} rate_scale=0 slope={SLOPE} cap=2.0 layout=bump h={HORIZON}")
emit(f"L40 non-degenerate band: availability in (0.05, 0.85); persistence floor min_N>=30")
emit("")
emit(f'{"regen":>6} {"speed":>5} {"seed":>4} {"n_eq":>6} {"avail":>6} {"flux":>6} {"explod":>6} {"osc":>11} {"PASS":>5}  failed_gates / nd_reasons')
emit("-"*108)

winners = []
for regen in REGENS:
    for speed in SPEEDS:
        for seed in SEEDS:
            r = run_cert(speed=speed, hmax=HMAX, Kc=KC, theta=THETA, regen_rate=regen,
                         rate_scale=0.0, layout="bump", seed=seed, horizon=HORIZON,
                         burn_in=BURN_IN, speed_cost_slope=SLOPE)
            cr = S.certify_run(r, PARAMS)
            osc = S.oscillation_verdict(r["N"])["classification"]
            failed = [k for k, v in cr["checks"].items() if not v]
            avail = r["availability_mean"]
            in_band = 0.05 < avail < 0.85
            emit(f'{regen:>6} {speed:>5} {seed:>4} {r["n_eq"]:>6.1f} {avail:>6.3f} '
                 f'{r["interbump_flux"]:>6.3f} {str(r["exploded"]):>6} {osc:>11} '
                 f'{str(cr["passes"]):>5}  failed={failed} nd={cr["nd_reasons"]}')
            if cr["passes"]:
                winners.append((regen, speed, seed, r["n_eq"], avail))
            # the key diagnostic: does availability enter the band while still viable+damped?
            elif in_band and r["n_eq"] >= 30 and osc == "DAMPED":
                winners.append(("near", regen, speed, seed, r["n_eq"], avail))

emit("")
emit("="*108)
if winners:
    emit(f"NON-DEGENERATE candidate cells found ({len(winners)}): {winners}")
    emit("=> the viability<->depletion tension is NOT binding; a genuine equilibrium may exist. Expand to a band.")
else:
    emit("NO cell is non-degenerate (availability in-band AND persist>=30 AND DAMPED AND passes certify_run).")
    emit("=> If availability only drops <0.85 where the population starves, the viability<->depletion")
    emit("   tension is BINDING -> clean NO-GO confirming Exp-242 CAN'T-POSE with a mechanism.")

out = "/Users/mirro/Projects/active-loop/experiments/outputs/exp243_lever.txt"
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
emit("")
emit(f"wrote {out}")
