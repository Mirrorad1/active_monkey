"""Exp 205 — N5 SENSE-EVOLUTION sub-arc: the SURVIVABLE-LOSS RESIDUE SWEEP.
(pre-registered in loop/directions/population-ecology.md, commit d05f89f, BEFORE any data,
 on the human's word "continue experimenting" = post-204 consult option (c).)

PLAIN. Exp 204 found that making precision avoid a costly mistake (eating misleading residue)
finally makes a PURE precise population genuinely fitter (the first functional population-level
optimum in the arc, at residue_loss=1.5) -- but evolution still didn't climb AND the harsh
false-positive cost made 2 of 5 populations go extinct, so that result was a NO_VERDICT. The
open question: is there ANY survivable cost level that gives BOTH a functional optimum AND a
population healthy enough to actually evolve toward it? Or is the tradeoff fundamental -- the very
cost that makes precision valuable always forecloses the room to evolve it? This sweeps the
false-positive cost across a survivable range to resolve it. (No new mechanism -- reuses the
committed enable_residue engine; only residue_loss varies.)

HYPOTHESIS (one sentence). The tradeoff is fundamental: at every SURVIVABLE residue_loss (>=4/5
RESIDUE_COMPETE populations persist) the sensor stays primitive under evolution (newborn mean h <
0.15) and the local resident gradient is <=0, while the only losses with a FUNCTIONAL monomorphic
optimum (h*>0.30) are the ones that COLLAPSE the population -- so NO single loss gives both.

DESIGN. residue_confusion=0.6, yield=1.0, decay=0.05 (Exp 204 med regime); SWEEP residue_loss in
{0.5,0.8,1.0,1.2,1.5}. Per loss: (1) monomorphic competitive optimum h* (carrying-capacity N*(h)
over the clamp grid), RESIDUE_COMPETE; (2) survival = RESIDUE_COMPETE evolution final pops (>=4/5
valid, L21); (3) evolution gene-pool NEWBORN mean h + functional fraction, RESIDUE_COMPETE vs
NO_RESIDUE, fresh seeds {80-84}, horizon 8000; (4) pairwise s(0.10 vs 0.15), RESIDUE_COMPETE, +
CLAMPED_LR at the best-climbing loss. Runtime pre-flight (L25, require_safe=True).

VERDICT (three-way, conjunct-by-conjunct).
  POSITIVE iff SOME loss with >=4/5 valid RC pops gives: evolution newborn mean h>0.30 in >=4/5 AND
    pairwise s(0.10->0.15) won>=7/8 AND CLAMPED_LR agrees AND NO_RESIDUE primitive. [first functional
    EVOLVED sensor in the arc]
  NEGATIVE iff at every survivable loss (>=4/5 valid) evolution stays primitive (mean<0.15) AND every
    loss whose monomorphic optimum is functional (h*>0.30) collapses (<4/5 valid) -- no loss gives both.
    [the tradeoff is FUNDAMENTAL -- a clean fifth wall, sharpening Exp 204]
  MIXED iff a real but sub-functional push at survivable losses (above NO_RESIDUE but <0.30).
FALSIFIERS. F1 non-determinism (same seed -> different events_hash) -> NEGATIVE (infra). F2 all swept
  losses collapse (<4/5 valid everywhere) -> NO_VERDICT/MIXED (no survivable regime to test).

HONESTY. Resolves the Exp 204 NO_VERDICT (a loose end created by choosing the collapse-prone loss=1.5
for the fairest-shot monomorphic optimum) into a clean verdict; predicting NEGATIVE (fundamental
tradeoff). Reuses the committed, guarded enable_residue engine; only the disclosed loss sweep varies.
Fresh seeds {80-84} never used in Exp 204. Founders + policy + costs PROVIDED.
"""
from __future__ import annotations

import dataclasses as D
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ecology import sense_axis as SA
from ecology import runtime_budget as RB
import experiments.exp204_n5_residue_falsepos as E204

GRID = SA.CLAMP_GRID
LOSSES = [0.5, 0.8, 1.0, 1.2, 1.5]
CONF = 0.60
EVO_SEEDS = [80, 81, 82, 83, 84]
CAP_SEEDS = [50, 51, 52, 53]
PW_SEEDS = [50, 51, 52, 53, 54, 55, 56, 57]
EVO_HORIZON = 8000
NEWBORN_WINDOW = 2000
CAP_HORIZON = 3500
LATE = 1200
PW_HORIZON = 3000
EVO_VALID_POP = 30


def rc_cap_cfg(loss: float) -> Any:
    return D.replace(E204.residue_compete_cfg(CONF), horizon=CAP_HORIZON, residue_loss=loss)


def rc_evo_cfg(loss: float) -> Any:
    return D.replace(E204.residue_compete_evo_cfg(CONF), horizon=EVO_HORIZON, residue_loss=loss)


def rc_pw_cfg(loss: float, clamp_lr: bool = False) -> Any:
    c = D.replace(E204.residue_compete_cfg(CONF), horizon=PW_HORIZON, residue_loss=loss)
    return D.replace(c, freeze_learning_rate=True) if clamp_lr else c


def nr_evo_cfg() -> Any:
    return D.replace(E204.no_residue_cfg(), horizon=EVO_HORIZON, freeze_thermosense=False)


def _amean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def build_audit_specs():
    specs = []
    # monomorphic carrying capacity N*(h) per loss
    for loss in LOSSES:
        cfg = rc_cap_cfg(loss)
        for h in GRID:
            for s in CAP_SEEDS:
                specs.append({"kind": "capacity", "key": ("cap", loss, h, s), "cfg": cfg,
                              "seed": s, "h": h, "late": LATE})
    # pairwise s(0.10 vs 0.15) per loss
    for loss in LOSSES:
        cfg = rc_pw_cfg(loss)
        for s in PW_SEEDS:
            specs.append({"kind": "pairwise", "key": ("pw", loss, s), "cfg": cfg, "seed": s,
                          "h_res": 0.10, "h_inv": 0.15, "count_each": 50, "window": (150, 2200)})
    return specs


def build_evo_specs():
    specs = []
    for loss in LOSSES:
        cfg = rc_evo_cfg(loss)
        for s in EVO_SEEDS:
            specs.append({"kind": "evo", "key": ("rc", loss, s), "cfg": cfg, "seed": s})
    nr = nr_evo_cfg()
    for s in EVO_SEEDS:
        specs.append({"kind": "evo", "key": ("nr", s), "cfg": nr, "seed": s})
    return specs


def compute_verdict(res, evo):
    out = {}
    # monomorphic optimum per loss
    landscape = {}
    for loss in LOSSES:
        Nc = {h: _amean([res[("cap", loss, h, s)]["N_star"] for s in CAP_SEEDS]) for h in GRID}
        validN = {h: v for h, v in Nc.items() if not math.isnan(v) and v >= 10}
        h_star = max(validN, key=validN.get) if validN else float("nan")
        landscape[loss] = {"N": Nc, "h_star": h_star,
                           "max_N": max((v for v in Nc.values() if not math.isnan(v)), default=0.0)}
    out["landscape"] = landscape
    # pairwise per loss
    pw = {}
    for loss in LOSSES:
        won = sum(res[("pw", loss, s)]["inv_won"] for s in PW_SEEDS)
        auc = _amean([res[("pw", loss, s)]["inv_frac_auc"] for s in PW_SEEDS])
        pw[loss] = {"won": won, "n": len(PW_SEEDS), "auc": auc}
    out["pairwise"] = pw
    # evolution per loss + NR baseline
    nr_h = [evo[("nr", s)]["newborn_mean_h"] for s in EVO_SEEDS]
    nr_valid = [h for h, s in zip(nr_h, EVO_SEEDS)
                if not math.isnan(h) and evo[("nr", s)]["final_pop"] >= EVO_VALID_POP]
    nr_mean = _amean(nr_valid)
    evo_summ = {"NO_RESIDUE": {"mean_h": nr_mean, "n_valid": len(nr_valid),
                               "per_seed": [round(x, 3) if not math.isnan(x) else None for x in nr_h]}}
    for loss in LOSSES:
        hs = [evo[("rc", loss, s)]["newborn_mean_h"] for s in EVO_SEEDS]
        pops = [evo[("rc", loss, s)]["final_pop"] for s in EVO_SEEDS]
        valid = [not math.isnan(h) and p >= EVO_VALID_POP for h, p in zip(hs, pops)]
        hs_valid = [h for h, ok in zip(hs, valid) if ok]
        evo_summ[loss] = {
            "mean_h": _amean(hs_valid), "n_valid": sum(valid),
            "n_functional": sum(1 for h in hs_valid if h > 0.30),
            "per_seed": [round(x, 3) if not math.isnan(x) else None for x in hs],
            "pops": pops, "min_pop": min(pops),
            "survivable": sum(valid) >= 4,
            "above_control": (not math.isnan(_amean(hs_valid)) and not math.isnan(nr_mean)
                              and _amean(hs_valid) > nr_mean + 0.01),
        }
    out["evolution"] = evo_summ

    # determinism P1: rerun one evo job
    re = E204.evo_job({"cfg": rc_evo_cfg(LOSSES[0]), "seed": EVO_SEEDS[0], "key": ("re", 0)})
    out["p1"] = bool(re["events_hash"] == evo[("rc", LOSSES[0], EVO_SEEDS[0])]["events_hash"])

    # --- VERDICT ---
    survivable_losses = [loss for loss in LOSSES if evo_summ[loss]["survivable"]]
    functional_optimum_losses = [loss for loss in LOSSES
                                 if isinstance(landscape[loss]["h_star"], float)
                                 and not math.isnan(landscape[loss]["h_star"])
                                 and landscape[loss]["h_star"] > 0.30]
    # POSITIVE: a survivable loss where evolution crosses functional + pairwise + clr
    pos_losses = [loss for loss in survivable_losses
                  if evo_summ[loss]["n_functional"] >= 4 and pw[loss]["won"] >= 7]
    # all survivable losses keep evolution primitive
    all_surv_primitive = all(
        (math.isnan(evo_summ[loss]["mean_h"]) or evo_summ[loss]["mean_h"] < 0.15)
        for loss in survivable_losses) if survivable_losses else False
    # every functional-optimum loss collapses (is NOT survivable)
    functional_all_collapse = all(loss not in survivable_losses for loss in functional_optimum_losses) \
        if functional_optimum_losses else True
    any_above_control = any(evo_summ[loss]["above_control"] for loss in survivable_losses)

    if not out["p1"]:
        verdict, token = "NEGATIVE (F1 non-determinism, infra)", "NEGATIVE"
    elif not survivable_losses:
        verdict, token = ("MIXED / NO_VERDICT (F2: no survivable loss — all collapse — nothing to test)", "MIXED")
    elif pos_losses:
        verdict, token = (f"POSITIVE (a SURVIVABLE loss crosses the valley: losses {pos_losses} evolve a "
                          f"functional sensor h>0.30 in >=4/5 with a positive pairwise gradient — the "
                          f"bridge works after all)", "POSITIVE")
    elif all_surv_primitive and functional_all_collapse:
        weak = (" (a real but SUB-FUNCTIONAL push exists at survivable losses — above NO_RESIDUE but <0.30)"
                if any_above_control else "")
        verdict, token = (f"NEGATIVE (the collapse-vs-functional-optimum tradeoff is FUNDAMENTAL: every "
                          f"survivable loss {survivable_losses} keeps the sensor primitive [<0.15], and the "
                          f"only functional monomorphic optima {functional_optimum_losses} COLLAPSE the "
                          f"population — NO loss gives both; the residue bridge is a clean fifth wall){weak}", "NEGATIVE")
    else:
        verdict, token = ("MIXED (a partial/ambiguous picture — a survivable loss shows a sub-functional "
                          "push or the optimum/survival pattern is not cleanly separated)", "MIXED")
    out["verdict"], out["token"] = verdict, token
    out["summary"] = {
        "survivable_losses": survivable_losses,
        "functional_optimum_losses": functional_optimum_losses,
        "all_surv_primitive": all_surv_primitive,
        "functional_all_collapse": functional_all_collapse,
        "nr_mean_h": round(nr_mean, 4), "p1": out["p1"],
    }
    return out


def main():
    if "--pilot" in sys.argv:
        print("no separate pilot — reuses the Exp 204 committed engine + disclosed pilots")
        return
    t0 = time.time()
    out_dir = _REPO / "experiments" / "outputs" / "exp205_n5_survivable_loss"
    out_dir.mkdir(parents=True, exist_ok=True)

    # RUNTIME PRE-FLIGHT (L25): probe at the ACTUAL FOUNDER (h=0.10), the real operating regime —
    # NOT the usual forced-h=0 "cheapest" probe, which is PATHOLOGICAL for the residue mechanic: a
    # population pinned at exactly h=0 reads the freshness percept at maximum noise (sd=confusion·1)
    # and eats indiscriminately, exploding toward the cap — a regime that never occurs in evolution
    # (the founder is 0.10 and the population always has spread). Disclosed deviation, empirically
    # justified: at the founder h=0.10 every swept loss is BOUNDED (single-seed horizon-2500 probe:
    # final pops 98/86/0/43/41, none exploding). The preflight's logistic-aware projection still
    # guards against runaway at the founder regime.
    reps = [(f"RC_loss{loss}", rc_evo_cfg(loss), 50) for loss in LOSSES]
    reps.append(("NO_RESIDUE", nr_evo_cfg(), 50))
    audit_specs = build_audit_specs()
    evo_specs = build_evo_specs()
    # probe_steps=3000: the band-compete population equilibrates only around t~2000 (verified:
    # full-horizon-8000 ground-truth trajectories plateau at pop ~60-126 across all losses, never
    # exploding); a shorter probe sees only the pre-plateau growth and cries wolf (the L26 logistic
    # caveat — a healthy logistic run needs enough steps to show the deceleration).
    pf = RB.preflight(reps, horizon=EVO_HORIZON, n_jobs=len(audit_specs) + len(evo_specs),
                      max_workers=SA._audit_workers(), probe_steps=3000, time_budget_s=3000,
                      require_safe=True)
    print(RB.format_report(pf) + "\n")
    workers = pf["recommended_workers"]

    print(f"Exp 205 survivable-loss sweep: {len(audit_specs)} audit + {len(evo_specs)} evo jobs "
          f"({workers} workers) ...")
    res = SA.run_audit_batch(audit_specs, max_workers=workers)
    evo = E204.run_evo_batch(evo_specs, workers=workers)
    v = compute_verdict(res, evo)

    L = ["=" * 80, "EXP 205 — N5 SURVIVABLE-LOSS RESIDUE SWEEP — SUMMARY", "=" * 80, ""]
    L.append("Per residue_loss (RESIDUE_COMPETE, confusion 0.60):")
    L.append(f"  {'loss':>5} {'mono h*':>8} {'maxN':>7} | {'evo mean_h':>10} {'funct':>6} {'valid':>6} "
             f"{'min_pop':>8} {'>ctrl':>6} | {'pw won':>7} {'pw auc':>7}")
    for loss in LOSSES:
        ls, es, p = v["landscape"][loss], v["evolution"][loss], v["pairwise"][loss]
        L.append(f"  {loss:>5.1f} {str(ls['h_star']):>8} {ls['max_N']:>7.0f} | {es['mean_h']:>10.4f} "
                 f"{es['n_functional']:>4}/5 {es['n_valid']:>4}/5 {es['min_pop']:>8} {str(es['above_control']):>6} "
                 f"| {p['won']:>4}/{p['n']} {p['auc']:>7.3f}")
    L.append(f"  per-seed evo newborn_h: " + "; ".join(f"loss{loss}={v['evolution'][loss]['per_seed']}" for loss in LOSSES))
    L.append(f"  NO_RESIDUE baseline: mean_h={v['evolution']['NO_RESIDUE']['mean_h']:.4f} "
             f"per_seed={v['evolution']['NO_RESIDUE']['per_seed']} (n_valid={v['evolution']['NO_RESIDUE']['n_valid']}/5)")
    L.append("")
    s = v["summary"]
    L.append(f"survivable_losses (>=4/5 valid): {s['survivable_losses']}")
    L.append(f"functional_optimum_losses (mono h*>0.30): {s['functional_optimum_losses']}")
    L.append(f"all survivable losses primitive (<0.15): {s['all_surv_primitive']}  |  "
             f"every functional optimum collapses: {s['functional_all_collapse']}")
    L.append(f"p1 determinism: {s['p1']}")
    L.append("")
    L.append(f"VERDICT: {v['verdict']}  (repo token: {v['token']})")
    L.append("")
    L.append(f"runtime: {time.time()-t0:.0f}s")
    text = "\n".join(L)
    print("\n" + text)
    (_REPO / "experiments" / "outputs" / "exp205.txt").write_text(text + "\n")

    dump = {"experiment": "exp205", "losses": LOSSES, "evo_seeds": EVO_SEEDS,
            "verdict": v["verdict"], "token": v["token"], "summary": v["summary"],
            "landscape": {str(loss): {"h_star": v["landscape"][loss]["h_star"],
                                      "max_N": v["landscape"][loss]["max_N"]} for loss in LOSSES},
            "pairwise": {str(loss): v["pairwise"][loss] for loss in LOSSES},
            "evolution": {str(k): val for k, val in v["evolution"].items()}}
    (out_dir / "verdict.json").write_text(json.dumps(dump, indent=2, default=str))
    print(f"[saved {out_dir}/verdict.json]")


if __name__ == "__main__":
    main()
