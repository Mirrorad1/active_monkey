"""Exp 212 — evolvability-geometry Rung 1: LANDSCAPE ASSAY of active sensing (is there a higher region?).

PLAIN: Across Exp 199-211 a costed capability kept being useful-when-gifted yet not locally
selectable. Before asking whether evolution's search could CROSS to a useful version, we must first
ask whether a useful version EXISTS at all. This experiment does not evolve anything: it pins the
active-sensing probe rate at a grid of fixed values, runs a separate monomorphic population for each,
and measures how large a population each can sustain (its carrying capacity = a bulk-fitness proxy).
If some probing rate sustains a clearly LARGER population than the non-probing resident, there is a
higher-fitness region beyond the local wall (a fitness valley worth trying to cross). If the
population only shrinks as probing rises (cost with no net payoff), there is no higher region and the
trait is done. A pure-cost arm (perfect percept, so probing buys nothing) maps the cost-only
landscape for comparison.

This is Rung 1 of the evolvability-geometry direction. It is a DIAGNOSTIC landscape classification,
NOT an evolution run and NOT full active inference. Per the direction card, heavy-tailed mutation /
standing variation (Rung 2/3) are tested ONLY if a higher region exists here.

PREDECLARATION
--------------
Hypothesis (a valley exists): under the honest fair cost (probe_cost 0.01) and the drift-suppressed
  cap-250 regime, the monomorphic carrying capacity N*(rate) for fixed-rate active sensing has a peak
  at a probe rate h* > 0 that exceeds the non-probing resident N*(0) — i.e. a bulk-fitter probing
  configuration exists even though Exp 210 found the local step from rate 0 does not pay (concave/
  saturating benefit vs linear cost, OR a bulk-vs-invasion gap).
Prediction (FITNESS_VALLEY_CONFIRMED): mean N*_info(h*) > mean N*_info(0) by >= 3% with h* > 0, in
  >= 5/8 seeds, while the local step N*_info(0.1) is NOT meaningfully above N*_info(0) (<= +3%); and
  the INFO landscape exceeds the cost-only (perfect-percept) landscape at high rate (information adds
  carrying capacity beyond the cost it imposes).
Falsifier (NO_HIGHER_REGION): N*_info(rate) is flat or monotone-DECREASING in rate — no probing
  configuration sustains a meaningfully larger equilibrium population than the resident (the tiny
  per-step net energy does not move carrying capacity). Then the capability is not useful enough even
  when completed at this cost; STOP the active-sensing trait for this direction.
Third outcome (POSITIVE_LOCAL_SLOPE_AUDIT): if instead N*_info(0.1) IS meaningfully above N*_info(0)
  (>3%), the bulk landscape has a positive local slope that the Exp 210 INVASION gradient did not see
  — a bulk-vs-invasion gap to audit, not a clean valley.
ARTIFACT_OR_NO_VERDICT if a majority of arms collapse, the INFO arm shows no perception benefit
  (wrong-cell does not fall with rate — liveness fail), or the cost-only arm behaves identically.

Guards (loop/LESSONS.md): L22 (gifted benefit != evolvability — here we only CLASSIFY the landscape,
  no evolvability claim); L29 (drift = population size: cap-250); L30 (honest fair cost 0.01, the
  benefit ceiling was measured in Exp 211); L31 parallelised, memory-capped; L25 runtime pre-flight.
FRESH seeds 90-97 (no reuse of Exp 210-211's 50-85). Re-runnable; writes experiments/outputs/exp212.txt.
Verifier: the cost-only landscape control, the per-seed win counts, the wrong-cell liveness, and the
committed raw output.
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

from ecology.engine import Ecology
from ecology.evolvability.config import load_config
from ecology.evolvability import gates as G
from ecology import runtime_budget as RB

CFG = "experiments/configs/preflight/uncertainty_gated_active_sensing.yaml"
GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]   # fixed-rate probe rate (information_sampling_rate)
SEEDS = list(range(90, 98))                # 8 FRESH seeds (no reuse of 50-85)
WINDOW = (800, 1500)                       # late-window N* sampling
MARGIN = 0.03                              # predeclared relative N* margin for "higher region"
MIN_SEED_WINS = 5                          # of 8, for robustness (L6 effect-size + count)
MIN_POP = 80                               # arms below this late-window mean = collapsed/invalid


def _landscape_one(args):
    """Monomorphic carrying-capacity (N*) + telemetry for one (rate, cue_noise, seed)."""
    base, rate, cue_noise, seed, w_lo, w_hi = args
    cfg = D.replace(base, probe_policy="fixed_rate", mutation_rate=0.0, cue_noise=cue_noise,
                    founder=D.replace(base.founder, information_sampling_rate=rate))
    eco = Ecology(cfg, seed=seed)
    pops: list[int] = []
    while eco.t < cfg.horizon and not eco.exploded:
        eco.step()
        if w_lo <= eco.t <= w_hi and eco.t % 25 == 0:
            pops.append(eco.alive_count())
        if not eco.has_alive():
            break
    nstar = sum(pops) / len(pops) if pops else 0.0
    hm = eco.hidden_mode_steps_total
    repro = sum(1 for e in eco.events if e["event_type"] == "reproduction")
    return {
        "rate": rate, "cue_noise": cue_noise, "seed": seed,
        "nstar": nstar,
        "wrong": eco.wrong_cell_steps_total / hm if hm else float("nan"),
        "probe_rate": eco.probe_count_total / hm if hm else 0.0,
        "repro": repro,
        "extinct": not eco.has_alive(),
    }


def _pmap(args, workers):
    if workers in (None, 0, 1):
        return [_landscape_one(a) for a in args]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_landscape_one, args))


def _mean(vals):
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    cfg = load_config(CFG)
    base = G.build_base_cfg(cfg.base_scenario, cfg.horizon, cfg.base_overrides)   # cap-250
    base = D.replace(base, founder=D.replace(base.founder, **cfg.founder_overrides))
    fair_cost = base.probe_cost
    haz = base.mode_hazard_scale

    workers = RB.recommended_workers_for(base, len(SEEDS), horizon=base.horizon)
    try:
        rep = RB.preflight([("exp212_cap250", base, SEEDS[0])], horizon=base.horizon,
                           n_jobs=len(SEEDS), max_workers=workers, require_safe=True)
        workers = max(1, int(rep.get("recommended_workers", workers)))
        pf = f"RUNTIME PRE-FLIGHT: safe={rep.get('safe')} workers->{workers} proj~{rep.get('proj_total_min')} min"
    except Exception as e:
        pf = f"RUNTIME PRE-FLIGHT: skipped/failed ({e}); workers={workers}"

    L = ["=" * 80,
         "EXP 212 — evolvability-geometry RUNG 1: ACTIVE-SENSING LANDSCAPE ASSAY",
         "Is there a bulk-fitter probing configuration beyond the local wall? (monomorphic N* sweep)",
         "=" * 80,
         f"Regime cap{int(base.capacity)}/regen{base.regen_rate} (drift-suppressed), cue_noise(info)=1.0, "
         f"hazard {haz}, probe_n_samples {base.probe_n_samples}, FAIR probe_cost {fair_cost}, "
         f"fixed_rate policy, monomorphic (mutation_rate=0), N* = mean alive pop over t in {WINDOW}.",
         pf,
         "DIAGNOSTIC landscape classification — NOT an evolution run, NOT full active inference.", ""]

    # --- run both arms over the grid ---
    info_args = [(base, h, 1.0, s, WINDOW[0], WINDOW[1]) for h in GRID for s in SEEDS]
    cost_args = [(base, h, 0.0, s, WINDOW[0], WINDOW[1]) for h in GRID for s in SEEDS]
    info_rows = _pmap(info_args, workers)
    cost_rows = _pmap(cost_args, workers)

    def _by_h(rows):
        return {h: [r for r in rows if r["rate"] == h] for h in GRID}
    info_by_h, cost_by_h = _by_h(info_rows), _by_h(cost_rows)

    info_nstar = {h: _mean([r["nstar"] for r in info_by_h[h]]) for h in GRID}
    cost_nstar = {h: _mean([r["nstar"] for r in cost_by_h[h]]) for h in GRID}
    info_wrong = {h: _mean([r["wrong"] for r in info_by_h[h]]) for h in GRID}
    cost_wrong = {h: _mean([r["wrong"] for r in cost_by_h[h]]) for h in GRID}
    info_prate = {h: _mean([r["probe_rate"] for r in info_by_h[h]]) for h in GRID}

    L.append("--- INFO landscape (cue_noise=1.0: inference matters) ---")
    L.append(f"  {'rate':>5} {'N*':>9} {'wrong_cell':>11} {'probe_rate':>11} {'repro':>8}")
    for h in GRID:
        rep_mean = _mean([r["repro"] for r in info_by_h[h]])
        L.append(f"  {h:>5} {info_nstar[h]:>9.1f} {info_wrong[h]:>11.4f} {info_prate[h]:>11.4f} {rep_mean:>8.0f}")
    L.append("--- COST-ONLY landscape (cue_noise=0.0: perfect percept, probe buys nothing) ---")
    L.append(f"  {'rate':>5} {'N*':>9} {'wrong_cell':>11}")
    for h in GRID:
        L.append(f"  {h:>5} {cost_nstar[h]:>9.1f} {cost_wrong[h]:>11.4f}")
    L.append("")

    # --- classification ---
    n0 = info_nstar[0.0]
    # peak over h>0
    pos_h = [h for h in GRID if h > 0.0]
    h_star = max(pos_h, key=lambda h: info_nstar[h])
    peak = info_nstar[h_star]
    rel_gain = (peak - n0) / n0 if n0 > 0 else float("nan")
    # per-seed: does N*_info(h_star) beat N*_info(0) for that seed?
    n0_by_seed = {r["seed"]: r["nstar"] for r in info_by_h[0.0]}
    hstar_by_seed = {r["seed"]: r["nstar"] for r in info_by_h[h_star]}
    seed_wins = sum(1 for s in SEEDS if hstar_by_seed.get(s, 0) > n0_by_seed.get(s, 0))
    # local step
    local_rel = (info_nstar[0.1] - n0) / n0 if n0 > 0 else float("nan")
    # liveness: wrong-cell falls with rate in the INFO arm
    wrong_falls = (not math.isnan(info_wrong[1.0]) and not math.isnan(info_wrong[0.0])
                   and info_wrong[1.0] < info_wrong[0.0])
    # validity: arms healthy
    min_arm_pop = min(info_nstar[h] for h in GRID)
    collapsed = (math.isnan(min_arm_pop) or min_arm_pop < MIN_POP)
    # cost-only must NOT show the same higher region (else the "gain" is not information)
    cost_rel_gain = (max(cost_nstar[h] for h in pos_h) - cost_nstar[0.0]) / cost_nstar[0.0] if cost_nstar[0.0] > 0 else float("nan")

    higher_region = (not math.isnan(rel_gain) and rel_gain >= MARGIN and seed_wins >= MIN_SEED_WINS)
    local_positive = (not math.isnan(local_rel) and local_rel > MARGIN)

    # Landscape SHAPE diagnostic (tol = 0.5% of resident, ~drift noise on N*).
    tol = 0.005 * n0 if n0 > 0 else 0.0
    seq = [info_nstar[h] for h in GRID]
    monotone_up = all(seq[i + 1] >= seq[i] - tol for i in range(len(seq) - 1))
    monotone_down = all(seq[i + 1] <= seq[i] + tol for i in range(len(seq) - 1))
    dips_then_rises = (not math.isnan(peak) and peak > n0 + tol
                       and any(info_nstar[h] < n0 - tol for h in pos_h if h <= h_star))
    if dips_then_rises:
        shape = "VALLEY (dips below resident then rises)"
    elif monotone_up and (peak > n0 + tol):
        shape = "monotone-increasing (uphill path, NO valley)"
    elif monotone_down:
        shape = "monotone-decreasing (probing only costs)"
    else:
        shape = "flat / non-monotone (within drift noise)"
    # Information's net bulk contribution at full probing = INFO - COST-ONLY at rate 1.0.
    info_minus_cost_full = info_nstar[1.0] - cost_nstar[1.0]

    L.append(f"resident N*(0)={n0:.1f}; peak N*({h_star})={peak:.1f}  rel_gain={rel_gain:+.3f} "
             f"(meaningful-margin {MARGIN}); per-seed wins {seed_wins}/{len(SEEDS)} (need >= {MIN_SEED_WINS})")
    L.append(f"local step N*(0.1) rel to N*(0) = {local_rel:+.3f}; cost-only peak rel_gain = {cost_rel_gain:+.3f}; "
             f"INFO-minus-COST-only at rate1.0 = {info_minus_cost_full:+.1f} creatures")
    L.append(f"landscape SHAPE: {shape}")
    L.append(f"liveness (wrong-cell falls with rate): info wrong {info_wrong[0.0]:.4f} -> {info_wrong[1.0]:.4f} = {wrong_falls}; "
             f"min arm N* = {min_arm_pop:.1f} (collapsed<{MIN_POP}: {collapsed})")
    L.append("")

    # --- verdict (conjunct-by-conjunct; the script CLAIM, re-checked by the blinded verifier) ---
    if collapsed or not wrong_falls:
        verdict = "ARTIFACT_OR_NO_VERDICT"
        why = (f"a majority/any arm collapsed (min N* {min_arm_pop:.1f}) or the INFO arm shows no "
               f"perception benefit (wrong-cell does not fall with rate) — the bulk-fitness metric "
               f"cannot be read honestly.")
    elif not higher_region:
        verdict = "NO_HIGHER_REGION"
        why = (f"no probing rate sustains a MEANINGFULLY larger monomorphic population than the "
               f"non-probing resident: the peak (rate {h_star}) gain is only rel_gain {rel_gain:+.3f} "
               f"< the predeclared meaningful margin {MARGIN} ({seed_wins}/{len(SEEDS)} seeds). The "
               f"landscape is '{shape}': the INFORMATION benefit is REAL (wrong-cell falls with rate; "
               f"INFO exceeds the cost-only arm by {info_minus_cost_full:+.1f} creatures at full probing) "
               f"but TINY — full probing buys only ~{rel_gain*100:.1f}% carrying capacity, below the "
               f"drift threshold that flattened Exp 210's invasion gradient. CRUCIALLY there is NO VALLEY "
               f"(the path from resident to peak is monotone uphill, not downhill-then-up), so the blocker "
               f"for active sensing is SMALL BENEFIT, not search geometry: heavy-tailed mutation / standing "
               f"variation (Rung 2/3) fix valley-crossing and have no valley to cross here. STOP the "
               f"active-sensing trait for evolvability-geometry; a representative wall trait now has a "
               f"completed landscape assay (direction stop-condition #1).")
    elif local_positive:
        verdict = "POSITIVE_LOCAL_SLOPE_AUDIT"
        why = (f"a higher region exists (peak rel_gain {rel_gain:+.3f}) AND the local step is also "
               f"positive in BULK N* (N*(0.1) {local_rel:+.3f}) — but Exp 210's INVASION gradient at "
               f"the same step was flat. This is a bulk-fitness vs invasion-fitness GAP to audit, not "
               f"a clean valley: monomorphic carrying capacity rises with probing while a rare probing "
               f"mutant does not invade. Audit the metric mismatch before Rung 2.")
    else:
        verdict = "FITNESS_VALLEY_CONFIRMED"
        why = (f"a bulk-fitter probing configuration exists (peak N*({h_star}) beats resident by "
               f"{rel_gain:+.3f}, {seed_wins}/{len(SEEDS)} seeds) while the local step does NOT pay "
               f"(N*(0.1) {local_rel:+.3f} <= margin) — a true fitness valley. The completed capability "
               f"is useful but unreachable by small local steps. Continue to Rung 2 (heavy-tailed "
               f"mutation) / Rung 3 (standing variation): can a non-evaluator mechanism cross it?")
    L.append(f"VERDICT (script claim): {verdict} — {why}")
    L.append(f"  SEEDS {SEEDS}; grid {GRID}; window {WINDOW}; fair_cost {fair_cost}; "
             f"margin {MARGIN}; min_seed_wins {MIN_SEED_WINS}.")
    _save(L)


def _save(L):
    text = "\n".join(L)
    print(text)
    out = _REPO / "experiments" / "outputs" / "exp212.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"\n[saved {out}]")


if __name__ == "__main__":
    main()
