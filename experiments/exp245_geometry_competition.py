"""Exp 245 — geometry redesign for STRONGER competition: does concentrating the food
(sharper bumps) steepen the density-dependence of per-capita intake, while staying viable
and navigable?

Exp 244 showed the substrate has REAL but WEAK competition (per-capita intake falls only
~7-9% with N) because the 5 bumps (sigma=1.5) are so broad they nearly tile the arena —
food isn't concentrated, so creatures don't crowd the same depletable cells. This probe
SHARPENS the bumps (lower sigma) while holding TOTAL FOOD roughly constant (amplitude scaled
~ 1/sigma^2, since a Gaussian's integral ~ sigma^2) so the comparison isolates CONCENTRATION
from "less food". Measures, per sigma, via the Exp-244 method (step the GO regime tick-by-tick,
regress per-capita intake vs N over the natural density trajectory):
  - competition strength: corr(N, per-capita intake) and the %-intake-drop low-N -> high-N
  - viability: n_eq (median N over window), min N (persistence), extinct?
A sharper sigma with a MUCH steeper intake-vs-N drop AND still viable => stronger competition
that could bound the speed benefit (toward an ESS). Extinction at sharp sigma => navigation
breaks (can't sense distant concentrated patches) / too little local food — a real limit.

HYPOTHESIS: concentrating food into sharper bumps (lower sigma, same total food) steepens
density-dependent competition (larger %intake-drop with N) because creatures crowd fewer,
faster-depleted cells. PREDECLARED FALSIFIER: if the %intake-drop stays weak (< ~20%)
across the full sigma sweep at every seed, bump geometry is NOT the lever strengthening
competition — the refill-rate-vs-revisit-rate tension is geometry-independent.
"""
from __future__ import annotations
import numpy as np
from ecology.evolvability.cert_run import _build_config
from ecology.engine import Ecology
from ecology.continuous_world import _GRID_CELLS

HORIZON = 1500
BURN_DROP = 50
SPEED = 2.0
REF_SIGMA = 1.5
REF_AMP = 1.0

def trajectory(sigma, amp, seed):
    cfg = _build_config(SPEED, hmax=0.20, Kc=60.0, theta=1.0, regen_rate=0.5, rate_scale=0.0,
                        layout="bump", horizon=HORIZON, speed_cost_slope=0.05,
                        bump_sigma=sigma, bump_amplitude=amp)
    eco = Ecology(cfg, seed=seed)
    cw = eco.cont_world
    cap = cw.capacity
    ncell = _GRID_CELLS * _GRID_CELLS
    prev = {id(c): c.phenotype.resource_eaten for c in eco._alive_list}
    rows = []
    for t in range(HORIZON):
        eco.step()
        alive = eco._alive_list
        N = len(alive)
        if N == 0:
            break
        whole = sum(float(cw._resource[ri][ci]) for ri in range(_GRID_CELLS)
                    for ci in range(_GRID_CELLS)) / (ncell * cap)
        intakes = []
        cur = {}
        for c in alive:
            cur[id(c)] = c.phenotype.resource_eaten
            if id(c) in prev:
                intakes.append(c.phenotype.resource_eaten - prev[id(c)])
        prev = cur
        if t >= BURN_DROP:
            rows.append((N, whole, float(np.mean(intakes)) if intakes else np.nan))
    return rows

def summarize(sigma, seed):
    amp = REF_AMP * (REF_SIGMA / sigma) ** 2   # hold total food ~constant
    rows = trajectory(sigma, amp, seed)
    if len(rows) < 30:
        return dict(sigma=sigma, amp=amp, seed=seed, extinct=True, steps=len(rows),
                    n_eq=0.0, minN=0.0, corr=float("nan"), drop=float("nan"), whole=float("nan"))
    arr = np.array(rows, float)
    Ns, whole, pc = arr[:, 0], arr[:, 1], arr[:, 2]
    m = ~np.isnan(pc)
    corr = float(np.corrcoef(Ns[m], pc[m])[0, 1]) if m.sum() > 2 and np.std(pc[m]) > 0 else float("nan")
    qlo, qhi = np.quantile(Ns, 0.25), np.quantile(Ns, 0.75)
    lo_int = np.nanmean(pc[Ns <= qlo]); hi_int = np.nanmean(pc[Ns >= qhi])
    drop = (lo_int - hi_int) / lo_int if lo_int else float("nan")  # fraction intake falls low->high N
    return dict(sigma=sigma, amp=amp, seed=seed, extinct=False, steps=len(rows),
                n_eq=float(np.median(Ns)), minN=float(Ns.min()), corr=corr, drop=drop,
                whole=float(np.mean(whole)))

if __name__ == "__main__":
    print("Exp 245 geometry competition @ GO regime (hmax=0.20,Kc=60,regen=0.5,slope=0.05,speed=2.0)")
    print("amplitude scaled ~1/sigma^2 to hold TOTAL FOOD ~constant; isolating CONCENTRATION.")
    print(f'{"sigma":>5} {"amp":>6} {"seed":>4} {"extinct":>7} {"n_eq":>6} {"minN":>5} {"corr(N,intake)":>14} {"%intake_drop":>12} {"whole_avail":>11}')
    print("-"*92)
    for sigma in (1.5, 1.0, 0.75, 0.6, 0.5):
        for seed in (1, 2):
            r = summarize(sigma, seed)
            dropp = f'{r["drop"]*100:6.1f}%' if not np.isnan(r["drop"]) else '   nan'
            print(f'{r["sigma"]:>5} {r["amp"]:>6.2f} {r["seed"]:>4} {str(r["extinct"]):>7} '
                  f'{r["n_eq"]:>6.1f} {r["minN"]:>5.0f} {r["corr"]:>14.3f} {dropp:>12} {r["whole"]:>11.3f}')
    print()
    print("READ: bigger %intake_drop (and more-negative corr) at sharper sigma = STRONGER competition.")
    print("extinct=True at sharp sigma = navigation/local-food failure (concentration too aggressive).")
