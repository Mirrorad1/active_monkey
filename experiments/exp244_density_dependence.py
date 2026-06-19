"""Exp 244 — the CORRECT non-degeneracy arbiter: is per-capita intake density-dependent?

Exp 243's NO-GO rested on a whole-field availability proxy (mean over all 576 sub-cells),
which is diluted by ~500 unoccupied full cells and cannot tell genuine population-level
resource competition from local self-depletion. The decisive question for whether the
substrate can host an invasion test: as the population density N rises, does per-capita
intake (and the availability creatures experience) FALL? If yes -> genuine resource
competition -> non-degenerate -> POSABLE. If per-capita intake is ~flat in N, the
population is regulated by Mechanism A (imposed mortality), NOT by resource competition,
so the field is effectively non-limiting -> DEGENERATE (the Exp-242 invalid regime),
confirming the NO-GO with the right arbiter.

Method (no engine change; events_hash untouched): step the GO regime tick-by-tick over the
natural density trajectory (founders -> equilibrium, incl. oscillation excursions), and per
step record N, whole-field availability, occupied-cell availability, and per-capita intake
(delta resource_eaten for creatures alive across both steps). Bin by N; report the trend.
"""
from __future__ import annotations
import numpy as np
from ecology.evolvability.cert_run import _build_config
from ecology.engine import Ecology
from ecology.continuous_world import _GRID_CELLS, _CELL_SIZE

HORIZON = 2000
BURN_DROP = 50   # drop the first 50 steps (founder transient) from the analysis

def trajectory(speed, seed, regen=0.5, hmax=0.20):
    cfg = _build_config(speed, hmax=hmax, Kc=60.0, theta=1.0, regen_rate=regen,
                        rate_scale=0.0, layout="bump", horizon=HORIZON, speed_cost_slope=0.05)
    eco = Ecology(cfg, seed=seed)
    cw = eco.cont_world
    cap = cw.capacity
    ncell = _GRID_CELLS * _GRID_CELLS
    prev_eaten = {id(c): c.phenotype.resource_eaten for c in eco._alive_list}
    rows = []  # (N, whole_avail, occ_avail, percap_intake)
    for t in range(HORIZON):
        eco.step()
        alive = eco._alive_list
        N = len(alive)
        if N == 0:
            break
        whole = sum(float(cw._resource[ri][ci]) for ri in range(_GRID_CELLS)
                    for ci in range(_GRID_CELLS)) / (ncell * cap)
        occ = []
        intakes = []
        cur_eaten = {}
        for c in alive:
            ph = c.phenotype
            cur_eaten[id(c)] = ph.resource_eaten
            pc = getattr(ph, "pos_cont", None)
            if pc is not None:
                ri, ci = cw._cell_idx(pc[0], pc[1])
                occ.append(float(cw._resource[ri][ci]) / cap)
            if id(c) in prev_eaten:
                intakes.append(ph.resource_eaten - prev_eaten[id(c)])
        prev_eaten = cur_eaten
        occ_avail = float(np.mean(occ)) if occ else float("nan")
        percap = float(np.mean(intakes)) if intakes else float("nan")
        if t >= BURN_DROP:
            rows.append((N, whole, occ_avail, percap))
    return rows

def report(speed, seed):
    rows = trajectory(speed, seed)
    if len(rows) < 20:
        print(f"speed={speed} seed={seed}: too few steps ({len(rows)}) — likely extinct.")
        return
    arr = np.array(rows, dtype=float)
    Ns, whole, occ, pc = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    nlo, nhi = Ns.min(), Ns.max()
    print(f"\nspeed={speed} seed={seed}: N range [{nlo:.0f},{nhi:.0f}] over {len(rows)} steps")
    # correlations of intake / availability with N (negative => density-dependent)
    def corr(x):
        m = ~np.isnan(x)
        return float(np.corrcoef(Ns[m], x[m])[0, 1]) if m.sum() > 2 and np.std(x[m]) > 0 else float("nan")
    print(f"  corr(N, percap_intake)   = {corr(pc):+.3f}  (negative => intake falls with density)")
    print(f"  corr(N, whole_avail)     = {corr(whole):+.3f}")
    print(f"  corr(N, occupied_avail)  = {corr(occ):+.3f}")
    # quartile bins of N
    qs = np.quantile(Ns, [0, 0.25, 0.5, 0.75, 1.0])
    print(f"  {'N-bin':>14} {'meanN':>6} {'percap_intake':>13} {'whole_avail':>11} {'occ_avail':>9}")
    for i in range(4):
        lo, hi = qs[i], qs[i + 1]
        m = (Ns >= lo) & (Ns <= hi if i == 3 else Ns < hi)
        if m.sum() == 0:
            continue
        print(f"  [{lo:5.0f},{hi:5.0f}] {Ns[m].mean():6.1f} {np.nanmean(pc[m]):13.4f} "
              f"{np.nanmean(whole[m]):11.3f} {np.nanmean(occ[m]):9.3f}")

if __name__ == "__main__":
    print("Exp 244 density-dependence test @ GO regime (hmax=0.20, Kc=60, regen=0.5, slope=0.05, bump)")
    print("VERDICT KEY: per-capita intake clearly FALLING with N (and experienced availability")
    print("falling) => genuine resource competition => POSABLE. Flat intake => A-regulated, not")
    print("resource-regulated => DEGENERATE (Exp-243 NO-GO confirmed with the right arbiter).")
    for sp in (2.0, 1.5):
        for sd in (1, 2):
            report(sp, sd)
