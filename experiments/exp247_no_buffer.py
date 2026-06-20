"""Exp 247 — the no-energy-buffer escape (the one lever the Exp-246 trade-off named).

HYPOTHESIS: the stability-vs-strong-competition trade-off (Exp 246) comes from the consumer's
ENERGY BUFFER. Creatures ride out a depleted field on stored energy -> a lagged demographic
overshoot -> oscillation -> which is why Mechanism A (mortality) is needed to stabilize ->
A caps N BELOW the resource-limited density -> competition stays weak. SHRINKING the buffer
(lower energy_capacity toward the reproduction threshold 4.2, so creatures live hand-to-mouth,
immediately intake-dependent) should make the population RESOURCE-LIMITED, reaching a STABLE
resource equilibrium WITHOUT A -> strong resource competition AND demographic stability
simultaneously -> the trade-off breaks -> the substrate becomes posable.

PREDECLARED FALSIFIER: if across the energy_capacity sweep NO low-buffer regime gives BOTH
(A-OFF stable: oscillation_verdict DAMPED AND level_cv low) AND strong competition
(%intake-drop >> the ~13% ceiling seen across Exp 243-246), while remaining viable, then the
energy buffer is NOT the lever and the stability-vs-competition trade-off is fundamental to
this consumer-resource architecture (continuous-locomotion chapter stays closed).

Config-only probe (no new mechanism): build the Exp-242 viable regime, then dataclasses.replace
the founder's energy_capacity. A-OFF = hmax=0 (crowding hazard zero). Controller-run raw numbers.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from ecology.evolvability.cert_run import _build_config
from ecology.evolvability import stability as S
from ecology.engine import Ecology

HORIZON = 2000
BURN_DROP = 50
SPEED = 2.0

def trajectory(cap, hmax, seed):
    cfg = _build_config(SPEED, hmax=hmax, Kc=60.0, theta=1.0, regen_rate=0.5, rate_scale=0.0,
                        layout="bump", horizon=HORIZON, speed_cost_slope=0.05)
    cfg = replace(cfg, founder=replace(cfg.founder, energy_capacity=cap))
    eco = Ecology(cfg, seed=seed)
    prev = {id(c): c.phenotype.resource_eaten for c in eco._alive_list}
    rows = []
    for t in range(HORIZON):
        eco.step()
        alive = eco._alive_list
        N = len(alive)
        if N == 0:
            break
        intakes = []
        cur = {}
        for c in alive:
            cur[id(c)] = c.phenotype.resource_eaten
            if id(c) in prev:
                intakes.append(c.phenotype.resource_eaten - prev[id(c)])
        prev = cur
        if t >= BURN_DROP:
            rows.append((N, float(np.mean(intakes)) if intakes else np.nan))
    return rows

def summarize(cap, hmax, seed):
    rows = trajectory(cap, hmax, seed)
    if len(rows) < 30:
        return dict(cap=cap, hmax=hmax, extinct=True, n_eq=0.0, minN=0.0,
                    cv=float("nan"), osc="-", corr=float("nan"), drop=float("nan"))
    a = np.array(rows, float)
    Ns, pc = a[:, 0], a[:, 1]
    m = ~np.isnan(pc)
    corr = float(np.corrcoef(Ns[m], pc[m])[0, 1]) if m.sum() > 2 and np.std(pc[m]) > 0 else float("nan")
    qlo, qhi = np.quantile(Ns, 0.25), np.quantile(Ns, 0.75)
    lo, hi = np.nanmean(pc[Ns <= qlo]), np.nanmean(pc[Ns >= qhi])
    drop = (lo - hi) / lo if lo else float("nan")
    return dict(cap=cap, hmax=hmax, extinct=False, n_eq=float(np.median(Ns)), minN=float(Ns.min()),
                cv=S.level_cv(Ns), osc=S.oscillation_verdict(Ns)["classification"], corr=corr, drop=drop)

if __name__ == "__main__":
    print("Exp 247 no-energy-buffer escape @ regime (regen=0.5, slope=0.05, speed=2.0; threshold=4.2 fixed)")
    print("Shrinking energy_capacity (the buffer above threshold 4.2). A-OFF=hmax 0; A-ON=hmax 0.20.")
    print(f'{"cap":>5} {"A":>4} {"extinct":>7} {"n_eq":>6} {"minN":>5} {"CV":>6} {"osc":>11} {"corr":>7} {"%drop":>7}')
    print("-" * 70)
    for cap in (10.0, 7.0, 5.5, 4.8):
        for hmax, lbl in ((0.0, "OFF"), (0.20, "ON")):
            r = summarize(cap, hmax, 1)
            d = f'{r["drop"]*100:5.1f}%' if not np.isnan(r["drop"]) else '  nan'
            print(f'{cap:>5} {lbl:>4} {str(r["extinct"]):>7} {r["n_eq"]:>6.1f} {r["minN"]:>5.0f} '
                  f'{r["cv"]:>6.3f} {r["osc"]:>11} {r["corr"]:>7.3f} {d:>7}')
    print()
    print("ESCAPE WORKS iff some LOW cap gives A-OFF: osc=DAMPED AND CV<=0.15 (stable without A)")
    print("AND strong competition %drop >> 13% (the Exp-243..246 ceiling), still viable (n_eq>=30).")
    print("If A-OFF stays OSCILLATORY / high-CV at every viable cap, the buffer is NOT the lever -> closed.")
