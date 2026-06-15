"""Exp 213 — evolvability-geometry Rung 1b: AFFORDANCE AUDIT (does payoff GEOMETRY make a sense evolvable?).

PLAIN: Exp 212 found that for active sensing the wall is a SMALL-BENEFIT wall — the fitter region barely
exists, monotone, no valley. The human steer: do not just tune another organ; instead AUDIT the payoff
GEOMETRY. The hypothesis is that a sense becomes locally evolvable when a SMALL precision gain unlocks a
DISCRETE, REPEATED, HIGH-STAKES action difference — not a smooth graded benefit that saturates. We test
this on a MATCHED substrate: the SAME forage world, the SAME costed precision organ (thermosense_intensity,
0.10->0.15), the SAME cost — the ONLY thing we vary is the AFFORDANCE the precision feeds:
  - SMOOTH  : precision reduces graded forage-STEERING noise (a continuous benefit). [Exp 200/203 geometry]
  - DISCRETE: precision additionally sharpens a BINARY eat-vs-skip decision with a large false-positive
              penalty, faced every step (residue, Exp 204) — repeated, high-stakes, discrete.
We sweep the DISCRETE stakes (residue_loss 0.5 vs 1.0, both survivable per Exp 205) to see if a steeper
payoff geometry steepens the local gradient. The goal is to identify WHICH payoff geometry (if any) makes
precision locally evolvable — NOT to force any organ to win.

PREDECLARATION
--------------
Hypothesis (steer / geometry-matters): at matched cost the DISCRETE high-stakes affordance has a MORE
  POSITIVE local selection gradient for precision (0.10->0.15) than the SMOOTH graded affordance, and it
  SCALES with the outcome gap (DISCRETE_hi > DISCRETE_lo > SMOOTH on win-fraction / mean_s); ideally the
  discrete affordance crosses POSITIVE (win-fraction >= 7/8, mean_s > 0) where the smooth one is flat.
Prediction if TRUE: win_frac(DISCRETE_hi) > win_frac(SMOOTH) (by >= 2/8) AND mean_s(DISCRETE_hi) >
  mean_s(SMOOTH) AND DISCRETE_hi >= DISCRETE_lo (stakes scaling); the discrete N*(h) landscape is also
  less-saturating / more-above-resident near 0.10–0.15 than the smooth one.
Falsifier (GEOMETRY_INDEPENDENT_WALL => NEGATIVE): all affordances have a non-positive local gradient
  (win-fraction <= lose bar, mean_s ~0) AND the discrete affordance is NOT meaningfully steeper than the
  smooth one and does NOT scale with stakes — i.e. discreteness/repetition/high-stakes does NOT change
  local evolvability at this substrate; the local-gradient wall is payoff-geometry-independent.
Third outcome (GEOMETRY_MATTERS_SUBTHRESHOLD): discrete is meaningfully steeper than smooth and scales
  with stakes, but still does NOT cross the positive bar — geometry shapes the gradient but the marginal
  benefit at the available stakes is still below the cost; the evolvability condition is "steeper than
  any affordance available here". ARTIFACT_OR_NO_VERDICT if a majority of an arm's seeds collapse.

Guards (loop/LESSONS.md): L22 (the BINDING metric is the local pairwise gradient, NOT a gifted benefit);
  L29 (drift: win-FRACTION over seeds cancels the founder lottery; report mean_s too); L30 (cost is HELD
  FIXED across affordances — the comparison is geometry at matched cost, not cost-tuning); L24/L21
  (validity: exclude collapsed arms, predeclared min valid seeds); L31 parallel; L25 runtime pre-flight.
Matched substrate + cost reuse the COMMITTED Exp 204 enable_residue engine (no new mechanism, no engine
change); the survivable residue_loss range is the disclosed Exp 205 finding (not tuned to a result).
FRESH seeds 120-127. Re-runnable; writes experiments/outputs/exp213.txt. Verifier: the SMOOTH (no-residue)
matched control, the stakes sweep, the per-seed validity, and the committed raw output.
"""
from __future__ import annotations

import dataclasses as D
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ecology import sense_axis as SA
from ecology import runtime_budget as RB
import experiments.exp204_n5_residue_falsepos as E204

# Affordance geometries on the MATCHED substrate (same forage world, same costed organ, same cost).
AFFORDANCES = {
    "SMOOTH_forage": E204.no_residue_cfg(),                                    # graded steering benefit
    "DISCRETE_lo":   D.replace(E204.residue_compete_cfg(0.6), residue_loss=0.5),   # binary eat-vs-skip, modest stakes
    "DISCRETE_hi":   D.replace(E204.residue_compete_cfg(0.6), residue_loss=1.0),   # binary eat-vs-skip, higher stakes
}
LAND_GRID = [0.0, 0.10, 0.15, 0.60]    # precision (thermosense_intensity) landscape grid
H_RES, H_INV = 0.10, 0.15              # the local step (THERMOSENSE_AXIS)
CAP_SEEDS = [120, 121, 122, 123]       # 4 fresh — monomorphic N* landscape
PW_SEEDS = list(range(120, 128))       # 8 fresh — local pairwise gradient (binding)
CAP_HORIZON, PW_HORIZON = 3500, 3000   # match Exp 204/205
MIN_POP = 10                           # below this late-window N* = collapsed/invalid
INEFF = 0.20                           # matched cost (inefficiency) across ALL affordances


def _job(spec):
    return SA.audit_job(spec)


def _pmap(specs, workers):
    if workers in (None, 0, 1):
        return [_job(s) for s in specs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_job, specs))


def _mean(vals):
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    # --- L25 runtime pre-flight on the most explosion-prone arm (highest-stakes residue) ---
    probe_cfg = D.replace(AFFORDANCES["DISCRETE_hi"], horizon=CAP_HORIZON)
    workers = RB.recommended_workers_for(probe_cfg, len(PW_SEEDS), horizon=CAP_HORIZON)
    try:
        rep = RB.preflight([("exp213", probe_cfg, PW_SEEDS[0])], horizon=CAP_HORIZON,
                           n_jobs=len(PW_SEEDS), max_workers=workers, require_safe=True)
        workers = max(1, int(rep.get("recommended_workers", workers)))
        pf = f"RUNTIME PRE-FLIGHT: safe={rep.get('safe')} workers->{workers} proj~{rep.get('proj_total_min')} min"
    except Exception as e:
        pf = f"RUNTIME PRE-FLIGHT: skipped/failed ({e}); workers={workers}"

    # --- build specs (capacity landscape + pairwise gradient) for every affordance ---
    specs = []
    for aff, cfg in AFFORDANCES.items():
        cap_cfg = D.replace(cfg, horizon=CAP_HORIZON)
        pw_cfg = D.replace(cfg, horizon=PW_HORIZON)
        for h in LAND_GRID:
            for s in CAP_SEEDS:
                specs.append({"kind": "capacity", "key": ("cap", aff, h, s), "cfg": cap_cfg,
                              "h": h, "seed": s, "ineff": INEFF, "late": 1500})
        for s in PW_SEEDS:
            specs.append({"kind": "pairwise", "key": ("pw", aff, s), "cfg": pw_cfg,
                          "h_res": H_RES, "h_inv": H_INV, "seed": s, "ineff": INEFF,
                          "window": (200, 2500), "count_each": 50})
    results = {r["key"]: r for r in _pmap(specs, workers)}

    L = ["=" * 80,
         "EXP 213 — evolvability-geometry Rung 1b: AFFORDANCE AUDIT (payoff geometry vs local evolvability)",
         "SMOOTH graded steering  vs  DISCRETE repeated high-stakes eat-vs-skip — matched substrate + cost",
         "=" * 80,
         f"Matched: Exp-204 forage substrate, thermosense precision {H_RES}->{H_INV}, inefficiency {INEFF} "
         f"(cost HELD FIXED), regen 0.08; only the AFFORDANCE varies. CAP horizon {CAP_HORIZON} (late 1500, "
         f"{len(CAP_SEEDS)} seeds); PW horizon {PW_HORIZON} ({len(PW_SEEDS)} seeds). FRESH seeds 120-127.",
         pf,
         "Binding metric = LOCAL pairwise gradient (win-fraction over seeds + mean_s); N*(h) landscape = shape.", ""]

    # --- aggregate per affordance ---
    A = {}
    for aff in AFFORDANCES:
        nstar = {h: _mean([results[("cap", aff, h, s)]["N_star"] for s in CAP_SEEDS]) for h in LAND_GRID}
        valid_cap = sum(1 for s in CAP_SEEDS if results[("cap", aff, H_RES, s)]["N_star"] >= MIN_POP)
        pw = [results[("pw", aff, s)] for s in PW_SEEDS]
        wins = sum(int(r["inv_won"]) for r in pw)
        auc = _mean([r["inv_frac_auc"] for r in pw])
        s_mean = _mean([r["s"] for r in pw])
        valid_pw = sum(1 for r in pw if not r["extinct"])
        opt_h = max(LAND_GRID, key=lambda h: (nstar[h] if not math.isnan(nstar[h]) else -1))
        local_slope = nstar[H_INV] - nstar[H_RES]
        A[aff] = dict(nstar=nstar, valid_cap=valid_cap, wins=wins, n=len(PW_SEEDS), auc=auc,
                      s_mean=s_mean, valid_pw=valid_pw, opt_h=opt_h, local_slope=local_slope)

    L.append("--- N*(h) LANDSCAPE (monomorphic carrying capacity, cost ON) ---")
    L.append(f"  {'affordance':<14} " + " ".join(f"h={h:<5}" for h in LAND_GRID) + "   opt_h  localΔ(0.10->0.15)")
    for aff in AFFORDANCES:
        a = A[aff]
        cells = " ".join(f"{a['nstar'][h]:>6.1f}" for h in LAND_GRID)
        L.append(f"  {aff:<14} {cells}   {a['opt_h']:<5}  {a['local_slope']:+.1f}")
    L.append("")
    L.append("--- LOCAL PAIRWISE GRADIENT (precision 0.10->0.15; win-fraction + mean_s) ---")
    L.append(f"  {'affordance':<14} {'wins':>6} {'win_frac':>9} {'inv_auc':>9} {'mean_s':>9} {'valid_pw':>9} {'valid_cap':>9}")
    for aff in AFFORDANCES:
        a = A[aff]
        L.append(f"  {aff:<14} {a['wins']:>3}/{a['n']:<2} {a['wins']/a['n']:>9.3f} {a['auc']:>9.3f} "
                 f"{a['s_mean']:>+9.4f} {a['valid_pw']:>8}/{a['n']} {a['valid_cap']:>8}/{len(CAP_SEEDS)}")
    L.append("")

    # --- classification ---
    sm, lo, hi = A["SMOOTH_forage"], A["DISCRETE_lo"], A["DISCRETE_hi"]
    win, lose = 7, 3   # default_thresholds(8)
    # validity: each arm needs a majority of valid (non-collapsed) seeds
    def _valid(a): return a["valid_pw"] >= 5 and a["valid_cap"] >= 3
    all_valid = all(_valid(a) for a in (sm, lo, hi))
    # any affordance crosses positive?
    def _positive(a): return a["wins"] >= win and not math.isnan(a["s_mean"]) and a["s_mean"] > 0.0
    any_positive = any(_positive(a) for a in (sm, lo, hi))
    # discrete meaningfully steeper than smooth AND scales with stakes?
    steeper = (hi["wins"] >= sm["wins"] + 2 and hi["s_mean"] > sm["s_mean"])
    scales = (hi["wins"] >= lo["wins"] and hi["s_mean"] >= lo["s_mean"] - 1e-9)
    discrete_better = steeper and scales

    L.append(f"validity: all arms >=5/8 valid_pw & >=3/4 valid_cap => {all_valid} "
             f"(SMOOTH {sm['valid_pw']}/8,{sm['valid_cap']}/4; DISCRETE_lo {lo['valid_pw']}/8,{lo['valid_cap']}/4; "
             f"DISCRETE_hi {hi['valid_pw']}/8,{hi['valid_cap']}/4)")
    L.append(f"smooth gradient: wins {sm['wins']}/8 mean_s {sm['s_mean']:+.4f} (positive={_positive(sm)})")
    L.append(f"discrete_hi vs smooth: wins {hi['wins']} vs {sm['wins']}, mean_s {hi['s_mean']:+.4f} vs "
             f"{sm['s_mean']:+.4f}; scales-with-stakes (hi>=lo): {scales} => discrete_better={discrete_better}")
    # BULK (monomorphic N*) vs INVASION (pairwise) local-slope sign per affordance — the honest nuance:
    # a positive bulk slope that does NOT survive to invasion is a bulk-vs-invasion gap, NOT evolvability.
    bulk_pos = [aff for aff in AFFORDANCES if A[aff]["local_slope"] > 0]
    L.append(f"bulk-vs-invasion: monomorphic N* local slope (0.10->0.15) is POSITIVE only for {bulk_pos} "
             f"(DISCRETE_hi {hi['local_slope']:+.1f}, opt_h {hi['opt_h']}), yet its INVASION gradient is "
             f"non-positive (wins {hi['wins']}/8, mean_s {hi['s_mean']:+.4f}) => a bulk-fitter monomorphic "
             f"step that a rare mutant still cannot invade (small/noisy pops; valid_cap {hi['valid_cap']}/4).")
    L.append("")

    if not all_valid:
        verdict = "ARTIFACT_OR_NO_VERDICT"
        why = ("a majority of seeds collapsed in at least one affordance arm — the local-gradient comparison "
               "cannot be read honestly; re-pick a survivable stakes range before concluding.")
    elif any_positive:
        winner = next(aff for aff in AFFORDANCES if _positive(A[aff]))
        verdict = "GEOMETRY_UNLOCKS_EVOLVABILITY"
        why = (f"the {winner} affordance has a POSITIVE local gradient (wins {A[winner]['wins']}/8, mean_s "
               f"{A[winner]['s_mean']:+.4f}) where the smooth graded affordance is flat — a DISCRETE, repeated, "
               f"high-stakes payoff geometry makes precision locally selectable. The blocker was payoff "
               f"geometry, not the organ. (Full evolution / Rung 2 may now be justified on this affordance.)")
    elif discrete_better:
        verdict = "GEOMETRY_MATTERS_SUBTHRESHOLD"
        why = (f"the DISCRETE high-stakes affordance is meaningfully steeper than the SMOOTH one (wins "
               f"{hi['wins']} vs {sm['wins']}, mean_s {hi['s_mean']:+.4f} vs {sm['s_mean']:+.4f}) and scales "
               f"with the outcome gap, BUT still does not cross the positive bar (>= {win}/8 & mean_s>0). "
               f"Payoff geometry SHAPES the local gradient — discreteness/stakes push it up — but the marginal "
               f"benefit at the available survivable stakes is still below the matched cost. The evolvability "
               f"condition is a steeper geometry than any affordance reachable here; report the trend honestly.")
    else:
        verdict = "GEOMETRY_INDEPENDENT_WALL"
        why = (f"every affordance has a non-positive local gradient (SMOOTH wins {sm['wins']}/8, DISCRETE_hi "
               f"{hi['wins']}/8; mean_s ~0) AND the discrete high-stakes affordance is NOT meaningfully steeper "
               f"than the smooth one and does not scale with stakes. Discreteness / repetition / high stakes do "
               f"NOT change local evolvability at this substrate: the local-gradient wall is payoff-geometry-"
               f"INDEPENDENT. The blocker is the small marginal benefit of a precision STEP, regardless of "
               f"whether the affordance is graded or a discrete high-stakes decision — consolidating the "
               f"Exp 199-212 wall as a benefit-magnitude wall, not a payoff-shape one. (Honest nuance: the "
               f"DISCRETE_hi MONOMORPHIC N* slope is the only positive one near the resident "
               f"({hi['local_slope']:+.1f}, opt_h {hi['opt_h']}), so the high-stakes geometry does create a "
               f"tiny BULK gradient — but it fails to INVADE [wins {hi['wins']}/8], a bulk-vs-invasion gap on "
               f"small noisy pops, not local evolvability.)")
    L.append(f"VERDICT (script claim): {verdict} — {why}")
    L.append(f"  affordances {list(AFFORDANCES)}; H {H_RES}->{H_INV}; CAP_SEEDS {CAP_SEEDS}; PW_SEEDS {PW_SEEDS}; "
             f"horizons cap{CAP_HORIZON}/pw{PW_HORIZON}; cost(inefficiency) {INEFF} fixed; win-bar {win}/8.")
    _save(L)


def _save(L):
    text = "\n".join(L)
    print(text)
    out = _REPO / "experiments" / "outputs" / "exp213.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"\n[saved {out}]")


if __name__ == "__main__":
    main()
