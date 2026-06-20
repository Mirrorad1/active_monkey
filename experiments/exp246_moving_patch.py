"""Exp 246 — the moving-patch lever: does forcing the whole population to crowd ONE small
drifting food patch create STRONG resource competition (steep density-dependence of intake)?

The 4 prior levers (founder, A-strength, regen, bump geometry) all left competition WEAK
(~5-9% intake drop) because a grazed cell refills before a sparse population revisits it.
A single small DRIFTING patch forces all creatures into one tiny region (high local density
-> the same few cells revisited faster than refill -> standing-crop depletion) while still
requiring locomotion to TRACK the drift. This probe sweeps patch sigma x drift period x
amplitude at the stabilizer regime and measures, via the Exp-244 method:
  - competition strength: corr(N, per-capita intake) and %-intake-drop low-N -> high-N
  - viability: n_eq, min N, extinct?
  - patch depletion: mean availability (resource/capacity) in the patch region (within 2*sigma
    of the moving center) -- the RIGHT local-depletion measure for a moving patch.
STRONG competition (big %drop, low patch-availability, still viable) => the substrate can be
posed with genuine competition. Extinct => the small patch can't sustain a crowding population
(scarcity/tracking failure). Weak again => even forced crowding can't beat the refill rate.

HYPOTHESIS: forcing the population to crowd one drifting patch creates strong resource
competition (high local density -> revisit rate exceeds refill rate -> standing-crop
depletion -> steep per-capita intake decline with N). PREDECLARED FALSIFIER: if %intake-drop
stays weak (< ~20%) and patch availability stays high (> ~0.70) across the patch-size /
amplitude / drift-period sweep, the moving patch does NOT strengthen competition — the
refill-rate-vs-revisit-rate wall is general and the substrate cannot host a clean ESS test.
"""
from __future__ import annotations
import math
import numpy as np
from ecology.evolvability.cert_run import _build_config
from ecology.engine import Ecology
from ecology.continuous_world import _GRID_CELLS, _CELL_SIZE, ARENA_W, ARENA_H

HORIZON = 1200
BURN_DROP = 100   # patch needs time to gather the population
SPEED = 2.0

def patch_center(t, R, period):
    ang = 2.0 * math.pi * t / period
    return (ARENA_W / 2.0 + R * math.cos(ang), ARENA_H / 2.0 + R * math.sin(ang))

def run(sigma, amp, period, seed, R=3.0):
    cfg = _build_config(SPEED, hmax=0.20, Kc=60.0, theta=1.0, regen_rate=0.5, rate_scale=0.0,
                        layout="bump", horizon=HORIZON, speed_cost_slope=0.05,
                        moving_patch=True, patch_sigma=sigma,
                        patch_amplitude=amp, patch_orbit_radius=R,
                        patch_period=period)
    eco = Ecology(cfg, seed=seed)
    cw = eco.cont_world
    cap = cw.capacity
    prev = {id(c): c.phenotype.resource_eaten for c in eco._alive_list}
    rows = []
    r2 = (2.0 * sigma) ** 2
    for t in range(HORIZON):
        eco.step()
        alive = eco._alive_list
        N = len(alive)
        if N == 0:
            break
        # patch center the creatures just grazed (rho used _patch_t before step_regen advanced it)
        cx, cy = patch_center(max(0, cw._patch_t - 1), R, period)
        pvals = []
        for ri in range(_GRID_CELLS):
            cyc = (ri + 0.5) * _CELL_SIZE
            for ci in range(_GRID_CELLS):
                cxc = (ci + 0.5) * _CELL_SIZE
                if (cxc - cx) ** 2 + (cyc - cy) ** 2 <= r2:
                    pvals.append(float(cw._resource[ri][ci]) / cap)
        intakes = []
        cur = {}
        for c in alive:
            cur[id(c)] = c.phenotype.resource_eaten
            if id(c) in prev:
                intakes.append(c.phenotype.resource_eaten - prev[id(c)])
        prev = cur
        if t >= BURN_DROP:
            rows.append((N, float(np.mean(pvals)) if pvals else np.nan,
                         float(np.mean(intakes)) if intakes else np.nan))
    return rows

def summarize(sigma, amp, period, seed):
    rows = run(sigma, amp, period, seed)
    if len(rows) < 30:
        return dict(sigma=sigma, amp=amp, period=period, seed=seed, extinct=True,
                    steps=len(rows), n_eq=0, minN=0, corr=float("nan"), drop=float("nan"),
                    patch_avail=float("nan"))
    a = np.array(rows, float)
    Ns, pav, pc = a[:, 0], a[:, 1], a[:, 2]
    m = ~np.isnan(pc)
    corr = float(np.corrcoef(Ns[m], pc[m])[0, 1]) if m.sum() > 2 and np.std(pc[m]) > 0 else float("nan")
    qlo, qhi = np.quantile(Ns, 0.25), np.quantile(Ns, 0.75)
    lo, hi = np.nanmean(pc[Ns <= qlo]), np.nanmean(pc[Ns >= qhi])
    drop = (lo - hi) / lo if lo else float("nan")
    return dict(sigma=sigma, amp=amp, period=period, seed=seed, extinct=False, steps=len(rows),
                n_eq=float(np.median(Ns)), minN=float(Ns.min()), corr=corr, drop=drop,
                patch_avail=float(np.nanmean(pav)))

if __name__ == "__main__":
    print("Exp 246 moving-patch competition @ stabilizer regime (hmax=0.20,Kc=60,regen=0.5,slope=0.05,speed=2.0)")
    print("patch orbit R=3.0; sweep sigma x amplitude x drift-period (period=steps/orbit; bigger=slower).")
    print(f'{"sigma":>5} {"amp":>5} {"period":>6} {"seed":>4} {"extinct":>7} {"n_eq":>6} {"minN":>5} '
          f'{"corr(N,intk)":>12} {"%drop":>7} {"patch_avail":>11}')
    print("-" * 96)
    grid = [(0.8, 4.0, 300), (0.8, 8.0, 300), (0.8, 8.0, 600), (0.8, 8.0, 150),
            (1.2, 6.0, 300), (1.2, 12.0, 300), (0.6, 12.0, 300)]
    for (sigma, amp, period) in grid:
        for seed in (1,):
            r = summarize(sigma, amp, period, seed)
            d = f'{r["drop"]*100:6.1f}%' if not np.isnan(r["drop"]) else '   nan'
            pa = f'{r["patch_avail"]:.3f}' if not np.isnan(r["patch_avail"]) else 'nan'
            print(f'{r["sigma"]:>5} {r["amp"]:>5} {r["period"]:>6} {r["seed"]:>4} {str(r["extinct"]):>7} '
                  f'{r["n_eq"]:>6.1f} {r["minN"]:>5.0f} {r["corr"]:>12.3f} {d:>7} {pa:>11}')
    print()
    print("READ: STRONG competition = big %drop AND low patch_avail (<0.85) AND still viable (n_eq>=30,")
    print("not extinct). Extinct = small patch can't sustain a crowding population. Weak %drop + high")
    print("patch_avail = even forced crowding can't beat the refill rate (the wall is fully general).")
