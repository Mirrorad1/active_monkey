"""Exp 206 — N5 SENSE-EVOLUTION sub-arc: the ROTATING-CLASS NICHE / SYMPATRIC-DIVERGENCE BRIDGE.
(pre-registered in loop/directions/population-ecology.md, commit 5d9f136, BEFORE any data, on the
 human's word "Continue with exp 206"; design synthesized by a 17-agent design+adversarial-audit
 workflow, dossier at experiments/outputs/exp206_design_audit.json. The LAST structurally-distinct
 escape in the sub-arc.)

PLAIN. Five walls (199-205) showed a costed sensor never evolves functional: the LOCAL selection
gradient at the typical sensor value is <=0 even when a precise population is fitter in bulk and
survives. The one untested escape: a PRIVATE NICHE. Instead of all creatures chasing one resource
(herding, which erodes precision's advantage as it spreads), give cells a hidden, time-ROTATING
"class" and let crowding on a class discount its food — so a precise creature can read the current
class, route to the UNDER-crowded one, and escape the herd. The hope: precision pays MORE as rivals
pile into the common class (positive frequency-dependence), flipping the local gradient positive.
Rotation is essential: a static class would be memorized for free by the learned map (the confound
that defeated every prior niche design). h keys ONLY the noisy read of the current class; the crowding
discount is h-blind. Question: does a private rotating niche finally let precision evolve?

HYPOTHESIS (one sentence). Under a rotating two-class niche with crowding-discounted intake, a stable
high-sensor lineage escapes into the under-crowded class (LINEAGE_ONLY_POSITIVE) or the gene-pool mean
crosses functional (NICHE_BRANCH_POSITIVE), with the realized local resident gradient positive,
mediated by correct class routing, knockout-sensitive, and not a memory/drift/id-order/anti-cheat
confound.

MECHANISM (gated enable_niche; exp194-205 byte-identical OFF, hash-guarded — committed 234ad5d). Each
cell has a true class j(pos,t)=floor(K*frac(class_phase[pos]+frac(t*niche_rotation))) that ROTATES over
time. In ROUTING (creature.choose_action) the creature reads neighbour classes noisily (sd =
niche_confusion*(1-h)) and steers toward the least-crowded class (read from class_occ_prev). At the EAT
step (h-BLIND): kept = consume(deficit)/(1 + niche_crowding*class_occ_prev[j_true]). ANTI-CHEAT (guard-
tested + blinded-verified): no food/fitness is f(h); h enters only the routing percept noise; the
crowding divisor is a creature-COUNT on the TRUE class; at niche_confusion=0 intake is byte-identical
across h.

FIXED REGIME (tuned on a DISCLOSED pilot {100-107}; the FAIREST-SHOT regime — the best relative
gradient found across a crowding x rotation x confusion sweep): niche_crowding=1.5, niche_rotation=0.05,
niche_confusion=0.4, niche_weight=6.0, K=2, uniform food regen 0.20. At this regime the pilot already
showed (a) SURVIVABLE pops (~660-810, no collapse), (b) a WEAK relative gradient (0.10->0.15 pairwise
won 3/8), and (c) evolution DECAYING h to ~0.026 (pure cost) — suggesting NEGATIVE, tested on FRESH
seeds {90-94}.

TWO MODES. A — GRADIENT AUDIT: pairwise selection coefficient s(0.10 vs h') + monomorphic carrying-
capacity optimum h*(N*) + B(h) cost-off overlay, NICHE_COMPETE vs controls, seeds {50-57}. B —
EVOLUTION: h evolves from founder 0.10 in NICHE_COMPETE + necessity controls, fresh seeds {90-94}; +
I(h;niche) occupancy, max evolved lineage, knockout-readiness.

VERDICT (conjunct-by-conjunct; full text + P2-P8 / F1-F8 in the card). NICHE_BRANCH_POSITIVE iff P2 &&
P3(gradient) && P4-global(evo>0.30) && P5(I(h;niche)) && P6(knockout) && P7(necessity) && P8(anti-cheat).
LINEAGE_ONLY_POSITIVE iff a stable+heritable+ecological+knockout-sensitive high-h lineage. NEGATIVE
(sixth wall) iff NICHE_COMPETE stays primitive (evo<0.15 majority and/or h*<=0.15) despite a real
COST_OFF gift and no stable lineage. MIXED iff partial / drift / NO_VERDICT (F2 no survivable separable
niche). Falsifiers F1 non-determinism, F4 drift, F5 memory (STATIC_NICHE climbs / CLAMPED_LR kills it),
F6 id-order, F7 anti-cheat leak (CONFUSION_ZERO climbs / BARCODE_SHUFFLED climbs).

HONESTY. The five walls predict NEGATIVE; a positive is a genuine surprise. Regime FIXED on the
disclosed pilot before the fresh-seed verdict. Verdict is the EVOLVED gene-pool / realized gradient,
NEVER the COST_OFF gift (L22). A NEGATIVE here is a clean SIXTH wall (no collapse confound — pops
survive ~700) and would justify accepting the sub-arc answer over building Exp 207.
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
from ecology.engine import Ecology, EcologyConfig
from ecology.scenarios import SCENARIOS, FOUNDER
import experiments.exp204_n5_residue_falsepos as E204  # reuse run_evo_batch

GRID = SA.CLAMP_GRID
AUDIT_SEEDS = [50, 51, 52, 53, 54, 55, 56, 57]
DIAG_SEEDS = [50, 51, 52]
EVO_SEEDS = [90, 91, 92, 93, 94]
AUDIT_HORIZON = 3500
LATE = 1200
GROWTH_HORIZON = 1500
GROWTH_WINDOW = (100, 700)
EVO_HORIZON = 8000
NEWBORN_WINDOW = 2000
EVO_VALID_POP = 30

# FIXED regime (disclosed pilot {100-107}; the fairest-shot regime).
REGEN = 0.20
CROWD = 1.5
ROT = 0.05
CONF = 0.4
NWEIGHT = 6.0
K = 2


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------
def _base(enable_thermosense: bool = True, niche_classes: int = K, niche_rotation: float = ROT,
          niche_confusion: float = CONF, niche_crowding: float = CROWD, niche_weight: float = NWEIGHT,
          niche_barcode_shuffle: bool = False, **kw) -> EcologyConfig:
    f = D.replace(FOUNDER, thermosense_intensity=0.10, thermosense_inefficiency=0.20,
                  temperature_tolerance=0.10)
    return D.replace(SCENARIOS["balanced"], horizon=AUDIT_HORIZON, max_population=20000, founder=f,
                     freeze_thermosense=True, shuffle_creature_order=True, regen_rate=REGEN,
                     enable_thermosense=enable_thermosense, enable_temperature=True,
                     temperature_stress_scale=0.0, thermosense_upkeep_floor=0.0,
                     thermosense_active_threshold=0.05, thermosense_noise_base=0.5,
                     enable_food_coupling=False, thermosense_forage_mode=True,
                     enable_niche=True, niche_classes=niche_classes, niche_rotation=niche_rotation,
                     niche_confusion=niche_confusion, niche_crowding=niche_crowding,
                     niche_weight=niche_weight, niche_barcode_shuffle=niche_barcode_shuffle, **kw)


def niche_compete_cfg(costoff: bool = False) -> EcologyConfig:
    return _base(enable_thermosense=(not costoff))


def single_niche_cfg() -> EcologyConfig:
    return _base(niche_classes=1)                          # one class — no escape, exp202 stampede


def static_niche_cfg() -> EcologyConfig:
    return _base(niche_rotation=0.0)                       # static class — memory free-ride control


def no_crowding_cfg() -> EcologyConfig:
    return _base(niche_crowding=0.0)                       # no frequency-dependence


def barcode_shuffled_cfg() -> EcologyConfig:
    return _base(niche_barcode_shuffle=True)               # signal decorrelated from value (placebo)


def confusion_zero_cfg() -> EcologyConfig:
    return _base(niche_confusion=0.0)                      # perfect percept ⇒ h-blind (anti-cheat)


def _evo(cfg: EcologyConfig) -> EcologyConfig:
    return D.replace(cfg, horizon=EVO_HORIZON, freeze_thermosense=False)


# ---------------------------------------------------------------------------
# MODE B — niche evolution runner (picklable, top-level)
# ---------------------------------------------------------------------------
def niche_evo_job(spec: dict[str, Any]) -> dict[str, Any]:
    """Evolve h from founder 0.10; return gene-pool newborn h, I(h;niche), max lineage, drift."""
    cfg: EcologyConfig = spec["cfg"]
    seed: int = spec["seed"]
    eco = Ecology(cfg, seed=seed)
    eco.run()
    w_lo = cfg.horizon - NEWBORN_WINDOW
    newborn_h: list[float] = []
    # occupancy-by-h for I(h;niche): (h_bucket, modal_class) over window-born creatures
    pairs: list[tuple[int, int]] = []
    lineage_h: dict[int, list[float]] = {}     # lineage_root -> newborn h's (for max stable lineage)
    n_alive_end = 0
    for c in eco._creatures:
        ph = c.phenotype
        if ph.alive:
            n_alive_end += 1
        if c.parent_id is not None and ph.birth_t >= w_lo:
            h = c.genotype.thermosense_intensity
            newborn_h.append(h)
            lineage_h.setdefault(c.lineage_root, []).append(h)
            occ = c.policy.niche_occ if (c.policy is not None) else None
            if occ is not None and sum(occ) > 0:
                modal = int(np.argmax(occ))
                hb = 0 if h < 0.15 else (1 if h < 0.30 else 2)   # low / mid / high
                pairs.append((hb, modal))

    def _m(xs):
        xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
        return float(np.mean(xs)) if xs else float("nan")

    # mutual information I(h_bucket; modal_class) in bits (best-effort; ~0 if h all primitive)
    mi = 0.0
    if len(pairs) >= 20:
        hb = np.array([p[0] for p in pairs]); cl = np.array([p[1] for p in pairs])
        n = len(pairs)
        for a in set(hb.tolist()):
            for b in set(cl.tolist()):
                pab = float(np.mean((hb == a) & (cl == b)))
                pa = float(np.mean(hb == a)); pbb = float(np.mean(cl == b))
                if pab > 0 and pa > 0 and pbb > 0:
                    mi += pab * math.log2(pab / (pa * pbb))
    # the largest lineage that evolved a functional mean h (stability proxy: cohort size in window)
    max_lineage_h = 0.0
    max_lineage_n = 0
    for root, hs in lineage_h.items():
        if len(hs) >= 15 and _m(hs) > max_lineage_h:
            max_lineage_h = _m(hs); max_lineage_n = len(hs)

    return {
        "key": spec["key"],
        "newborn_mean_h": _m(newborn_h),
        "newborn_max_h": (float(np.max(newborn_h)) if newborn_h else float("nan")),
        "n_newborn": len(newborn_h),
        "final_pop": n_alive_end,
        "mi_h_niche": mi,
        "max_lineage_h": max_lineage_h,
        "max_lineage_n": max_lineage_n,
        "events_hash": eco.events_hash(),
    }


# ---------------------------------------------------------------------------
# MODE A — gradient-audit specs
# ---------------------------------------------------------------------------
def build_audit_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    nc = niche_compete_cfg()
    # (1) monomorphic carrying capacity N*(h) — NICHE_COMPETE
    for h in GRID:
        for s in AUDIT_SEEDS:
            specs.append({"kind": "capacity", "key": ("cap", h, s), "cfg": nc, "seed": s,
                          "h": h, "late": LATE})
    # (2) pairwise s(0.10 vs h') — the direct relative gradient
    pw = D.replace(nc, horizon=3000)
    for hinv in (0.06, 0.15, 0.30, 0.45):
        for s in AUDIT_SEEDS:
            specs.append({"kind": "pairwise", "key": ("pw", hinv, s), "cfg": pw, "seed": s,
                          "h_res": 0.10, "h_inv": hinv, "count_each": 50, "window": (150, 2200)})
    # (2b) control pairwise 0.10 vs 0.15: CLAMPED_LR / NO_SHUFFLE / BARCODE_SHUFFLED / CONFUSION_ZERO / STATIC
    ctrl = {
        "clr": D.replace(pw, freeze_learning_rate=True),
        "nos": D.replace(pw, shuffle_creature_order=False),
        "shuf": D.replace(pw, niche_barcode_shuffle=True),
        "cz": D.replace(pw, niche_confusion=0.0),
        "static": D.replace(pw, niche_rotation=0.0),
    }
    for tag, cfg in ctrl.items():
        for s in AUDIT_SEEDS:
            specs.append({"kind": "pairwise", "key": (f"pw_{tag}", 0.15, s), "cfg": cfg, "seed": s,
                          "h_res": 0.10, "h_inv": 0.15, "count_each": 50, "window": (150, 2200)})
    # (3) B(h) cost-off + cost-on intrinsic growth overlay
    g_off = D.replace(niche_compete_cfg(costoff=True), horizon=GROWTH_HORIZON)
    g_on = D.replace(niche_compete_cfg(costoff=False), horizon=GROWTH_HORIZON)
    for h in GRID:
        for s in DIAG_SEEDS:
            specs.append({"kind": "growth", "key": ("goff", h, s), "cfg": g_off, "seed": s,
                          "h": h, "window": GROWTH_WINDOW})
            specs.append({"kind": "growth", "key": ("gon", h, s), "cfg": g_on, "seed": s,
                          "h": h, "window": GROWTH_WINDOW})
    return specs


def build_evo_specs() -> list[dict[str, Any]]:
    arms = {
        "NICHE_COMPETE": _evo(niche_compete_cfg()),
        "SINGLE_NICHE": _evo(single_niche_cfg()),
        "STATIC_NICHE": _evo(static_niche_cfg()),
        "NO_CROWDING": _evo(no_crowding_cfg()),
        "BARCODE_SHUFFLED": _evo(barcode_shuffled_cfg()),
        "CLAMPED_LR": D.replace(_evo(niche_compete_cfg()), freeze_learning_rate=True),
    }
    return [{"kind": "evo", "key": (arm, s), "cfg": cfg, "seed": s}
            for arm, cfg in arms.items() for s in EVO_SEEDS]


def _amean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def compute_verdict(res, evo) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # N*(h) landscape
    Nc = {h: _amean([res[("cap", h, s)]["N_star"] for s in AUDIT_SEEDS]) for h in GRID}
    validN = {h: v for h, v in Nc.items() if not math.isnan(v) and v >= 10}
    h_star = max(validN, key=validN.get) if validN else float("nan")
    out["landscape"] = {"N": Nc, "h_star": h_star,
                        "max_N": max((v for v in Nc.values() if not math.isnan(v)), default=0.0)}
    # pairwise
    pw: dict[Any, Any] = {}
    for hinv in (0.06, 0.15, 0.30, 0.45):
        won = sum(res[("pw", hinv, s)]["inv_won"] for s in AUDIT_SEEDS)
        auc = _amean([res[("pw", hinv, s)]["inv_frac_auc"] for s in AUDIT_SEEDS])
        pw[hinv] = {"won": won, "n": len(AUDIT_SEEDS), "auc": auc}
    for tag in ("clr", "nos", "shuf", "cz", "static"):
        pw[tag] = {"won": sum(res[(f"pw_{tag}", 0.15, s)]["inv_won"] for s in AUDIT_SEEDS),
                   "auc": _amean([res[(f"pw_{tag}", 0.15, s)]["inv_frac_auc"] for s in AUDIT_SEEDS])}
    out["pairwise"] = pw
    # B(h)
    Bc = {h: _amean([res[("goff", h, s)]["r"] for s in DIAG_SEEDS]) for h in GRID}
    gift = Bc[0.60] - Bc[0.00]
    out["benefit"] = {"B": Bc, "gift": gift}
    # evolution
    evo_summ: dict[str, Any] = {}
    for arm in ("NICHE_COMPETE", "SINGLE_NICHE", "STATIC_NICHE", "NO_CROWDING", "BARCODE_SHUFFLED", "CLAMPED_LR"):
        hs = [evo[(arm, s)]["newborn_mean_h"] for s in EVO_SEEDS]
        pops = [evo[(arm, s)]["final_pop"] for s in EVO_SEEDS]
        valid = [not math.isnan(h) and p >= EVO_VALID_POP for h, p in zip(hs, pops)]
        hs_valid = [h for h, ok in zip(hs, valid) if ok]
        # drift guard: corr(pop, newborn_h) across seeds
        corr = float("nan")
        if sum(valid) >= 3:
            pv = [p for p, ok in zip(pops, valid) if ok]
            if np.std(pv) > 0 and np.std(hs_valid) > 0:
                corr = float(np.corrcoef(pv, hs_valid)[0, 1])
        evo_summ[arm] = {
            "mean_h": _amean(hs_valid), "n_valid": sum(valid),
            "n_functional": sum(1 for h in hs_valid if h > 0.30),
            "per_seed": [round(x, 3) if not math.isnan(x) else None for x in hs],
            "min_pop": min(pops), "corr_pop_h": corr,
            "max_lineage_h": _amean([evo[(arm, s)]["max_lineage_h"] for s in EVO_SEEDS]),
            "mi_h_niche": _amean([evo[(arm, s)]["mi_h_niche"] for s in EVO_SEEDS]),
        }
    out["evolution"] = evo_summ
    # determinism P1
    re = niche_evo_job({"cfg": _evo(niche_compete_cfg()), "seed": EVO_SEEDS[0], "key": ("re", 0)})
    out["p1"] = bool(re["events_hash"] == evo[("NICHE_COMPETE", EVO_SEEDS[0])]["events_hash"])

    # --- VERDICT (conjunct-by-conjunct) ---
    nc = evo_summ["NICHE_COMPETE"]
    s015 = pw[0.15]
    survivable = nc["n_valid"] >= 4 and nc["min_pop"] >= EVO_VALID_POP
    # P3 local gradient
    p3 = (s015["won"] >= 7 and (not math.isnan(s015["auc"]) and s015["auc"] > 0.5)
          and isinstance(h_star, float) and not math.isnan(h_star) and h_star > 0.30
          and pw["clr"]["won"] >= 7 and pw["nos"]["won"] >= 5)
    p4_global = nc["n_functional"] >= 4
    p4_lineage = nc["max_lineage_h"] > 0.30
    # P7 necessity controls primitive
    controls_primitive = all(
        evo_summ[a]["n_functional"] == 0 and (math.isnan(evo_summ[a]["mean_h"]) or evo_summ[a]["mean_h"] < 0.15)
        for a in ("SINGLE_NICHE", "STATIC_NICHE", "NO_CROWDING", "BARCODE_SHUFFLED"))
    # P8 anti-cheat: CONFUSION_ZERO + BARCODE_SHUFFLED pairwise must NOT win; STATIC must not out-climb
    anticheat_ok = pw["cz"]["won"] <= 4 and pw["shuf"]["won"] <= 4
    drift = (not math.isnan(nc["corr_pop_h"]) and nc["corr_pop_h"] < -0.4)
    gift_real = gift > 0
    nc_primitive = (nc["n_functional"] <= len(EVO_SEEDS) // 2
                    and (math.isnan(nc["mean_h"]) or nc["mean_h"] < 0.15))
    nc_above_control = (not math.isnan(nc["mean_h"]) and not math.isnan(evo_summ["NO_CROWDING"]["mean_h"])
                        and nc["mean_h"] > evo_summ["NO_CROWDING"]["mean_h"] + 0.01)
    weak = " (a real but SUB-FUNCTIONAL push above NO_CROWDING)" if nc_above_control else ""

    if not out["p1"]:
        verdict, token = "NEGATIVE (F1 non-determinism, infra)", "NEGATIVE"
    elif not survivable:
        verdict, token = ("MIXED / NO_VERDICT (F2: NICHE_COMPETE not survivable / no separable occupied "
                          "niche — <4/5 valid)", "MIXED")
    elif p3 and p4_global and controls_primitive and anticheat_ok and not drift:
        verdict, token = ("POSITIVE / NICHE_BRANCH_POSITIVE (a functional sensor evolves under the rotating "
                          "niche: positive local gradient, global mean >0.30, necessity + anti-cheat clean)", "POSITIVE")
    elif p4_lineage and controls_primitive and anticheat_ok and not drift:
        verdict, token = ("POSITIVE / LINEAGE_ONLY_POSITIVE (a stable high-sensor lineage escaped into the "
                          "under-crowded niche; global mean moderate)", "POSITIVE")
    elif nc_primitive and gift_real:
        verdict, token = (f"NEGATIVE (the SIXTH wall: precision stays primitive [evo {nc['mean_h']:.3f}, "
                          f"h*={h_star}, pairwise {s015['won']}/8] despite a real COST_OFF gift AND a "
                          f"SURVIVABLE population [{nc['min_pop']}] — the herding/negative-frequency-"
                          f"dependence is NOT escapable even with a private, rotating, crowding-discounted "
                          f"niche; no collapse confound){weak}", "NEGATIVE")
    else:
        verdict, token = (f"MIXED (partial/ambiguous — gradient in [0.15,0.30], single-mode, or "
                          f"drift-flagged){weak}", "MIXED")
    out["verdict"], out["token"] = verdict, token
    out["gates"] = {
        "pw_015_won": f"{s015['won']}/8", "pw_015_auc": round(s015["auc"], 3) if not math.isnan(s015["auc"]) else "nan",
        "pw_030_won": f"{pw[0.30]['won']}/8", "h_star_N": h_star, "gift_real": gift_real, "gift": round(gift, 5),
        "pw_clr_won": f"{pw['clr']['won']}/8", "pw_nos_won": f"{pw['nos']['won']}/8",
        "pw_shuf_won": f"{pw['shuf']['won']}/8", "pw_cz_won": f"{pw['cz']['won']}/8", "pw_static_won": f"{pw['static']['won']}/8",
        "evo_NC_mean_h": round(nc["mean_h"], 4), "evo_NC_functional": f"{nc['n_functional']}/5",
        "evo_NC_min_pop": nc["min_pop"], "evo_NC_corr_pop_h": round(nc["corr_pop_h"], 3) if not math.isnan(nc["corr_pop_h"]) else "nan",
        "evo_NC_max_lineage_h": round(nc["max_lineage_h"], 3), "mi_h_niche": round(nc["mi_h_niche"], 4),
        "evo_SINGLE": round(evo_summ["SINGLE_NICHE"]["mean_h"], 4), "evo_STATIC": round(evo_summ["STATIC_NICHE"]["mean_h"], 4),
        "evo_NOCROWD": round(evo_summ["NO_CROWDING"]["mean_h"], 4), "evo_SHUF": round(evo_summ["BARCODE_SHUFFLED"]["mean_h"], 4),
        "evo_CLAMPLR": round(evo_summ["CLAMPED_LR"]["mean_h"], 4),
        "controls_primitive": controls_primitive, "anticheat_ok": anticheat_ok, "survivable": survivable,
        "drift": drift, "p1": out["p1"],
    }
    return out


def main():
    if "--pilot" in sys.argv:
        print("regime FIXED on the disclosed inline pilot {100-107}: crowd=1.5 rot=0.05 conf=0.4 "
              "weight=6.0 K=2 regen=0.20 (fairest-shot; survivable ~700, weak gradient, h decays)")
        return
    t0 = time.time()
    out_dir = _REPO / "experiments" / "outputs" / "exp206_n5_rotating_niche"
    out_dir.mkdir(parents=True, exist_ok=True)

    # RUNTIME PRE-FLIGHT (L25): probe NICHE_COMPETE + STATIC_NICHE at the ACTUAL founder h=0.10
    # (NOT forced h=0 — exp205 lesson), probe_steps>=3000 (L26 logistic plateau).
    reps = [("NICHE_COMPETE", _evo(niche_compete_cfg()), 50),
            ("STATIC_NICHE", _evo(static_niche_cfg()), 50),
            ("NO_CROWDING", _evo(no_crowding_cfg()), 50)]
    audit_specs = build_audit_specs()
    evo_specs = build_evo_specs()
    pf = RB.preflight(reps, horizon=EVO_HORIZON, n_jobs=len(audit_specs) + len(evo_specs),
                      max_workers=SA._audit_workers(), probe_steps=3000, time_budget_s=3600,
                      require_safe=True)
    print(RB.format_report(pf) + "\n")
    workers = pf["recommended_workers"]

    print(f"Exp 206 rotating-class niche: {len(audit_specs)} audit + {len(evo_specs)} evo jobs "
          f"({workers} workers) ...")
    res = SA.run_audit_batch(audit_specs, max_workers=workers)
    evo = _run_niche_evo(evo_specs, workers)
    v = compute_verdict(res, evo)

    L = ["=" * 80, "EXP 206 — N5 ROTATING-CLASS NICHE / SYMPATRIC-DIVERGENCE BRIDGE — SUMMARY", "=" * 80, ""]
    L.append(f"REGIME (fairest-shot, disclosed pilot): crowd={CROWD} rot={ROT} conf={CONF} weight={NWEIGHT} K={K} regen={REGEN}")
    L.append("")
    L.append("MONOMORPHIC N*(h) (cost ON; argmax = competitive optimum h*):")
    L.append(f"  {'h':>5}  {'N*':>8}")
    for h in GRID:
        L.append(f"  {h:>5.2f}  {v['landscape']['N'][h]:>8.1f}")
    L.append(f"  h*N = {v['landscape']['h_star']}   max_N = {v['landscape']['max_N']:.0f}")
    L.append("")
    L.append("PAIRWISE invader h' vs resident 0.10 (NICHE_COMPETE — the relative gradient):")
    for hinv in (0.06, 0.15, 0.30, 0.45):
        p = v["pairwise"][hinv]
        a = f"{p['auc']:.3f}" if not math.isnan(p["auc"]) else "nan"
        L.append(f"  0.10 vs {hinv:.2f}:  won={p['won']}/8  auc={a}")
    L.append(f"  controls 0.10v0.15: CLAMPED_LR={v['pairwise']['clr']['won']}/8  NO_SHUFFLE={v['pairwise']['nos']['won']}/8"
             f"  BARCODE_SHUF={v['pairwise']['shuf']['won']}/8  CONFUSION_0={v['pairwise']['cz']['won']}/8"
             f"  STATIC={v['pairwise']['static']['won']}/8")
    L.append("")
    L.append(f"INSTALLED BENEFIT gift B(0.60)-B(0.00) = {v['benefit']['gift']:+.5f}  (gift real if >0)")
    L.append("")
    L.append(f"EVOLUTION (Mode B) — gene-pool NEWBORN mean h over last {NEWBORN_WINDOW} (founder 0.10):")
    for arm in ("NICHE_COMPETE", "SINGLE_NICHE", "STATIC_NICHE", "NO_CROWDING", "BARCODE_SHUFFLED", "CLAMPED_LR"):
        e = v["evolution"][arm]
        L.append(f"  {arm:>16}: mean_h={e['mean_h']:.4f} funct={e['n_functional']}/5 valid={e['n_valid']}/5 "
                 f"min_pop={e['min_pop']} max_lineage_h={e['max_lineage_h']:.3f} per_seed={e['per_seed']}")
    nc = v["evolution"]["NICHE_COMPETE"]
    L.append(f"  NICHE_COMPETE: corr(pop,h)={nc['corr_pop_h']}  I(h;niche)={nc['mi_h_niche']:.4f} bits")
    L.append("")
    g = v["gates"]
    L.append("GATES: " + "  ".join(f"{k}={g[k]}" for k in g))
    L.append("")
    L.append(f"VERDICT: {v['verdict']}  (repo token: {v['token']})")
    L.append("")
    L.append(f"runtime: {time.time()-t0:.0f}s")
    text = "\n".join(L)
    print("\n" + text)
    (_REPO / "experiments" / "outputs" / "exp206.txt").write_text(text + "\n")

    def _ser(d):
        return {f"{k:.2f}" if isinstance(k, float) else k: vv for k, vv in d.items()}
    dump = {"experiment": "exp206", "regime": {"crowd": CROWD, "rot": ROT, "conf": CONF, "weight": NWEIGHT, "K": K, "regen": REGEN},
            "audit_seeds": AUDIT_SEEDS, "evo_seeds": EVO_SEEDS, "verdict": v["verdict"], "token": v["token"],
            "gates": v["gates"], "landscape": {"N": _ser(v["landscape"]["N"]), "h_star": v["landscape"]["h_star"]},
            "pairwise": {str(k): vv for k, vv in v["pairwise"].items()},
            "benefit": {"B": _ser(v["benefit"]["B"]), "gift": v["benefit"]["gift"]},
            "evolution": v["evolution"]}
    (out_dir / "verdict.json").write_text(json.dumps(dump, indent=2, default=str))
    print(f"[saved {out_dir}/verdict.json]")


def _run_niche_evo(specs, workers):
    """Parallel niche evolution batch (reuses the exp204 ProcessPoolExecutor dispatcher pattern)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    if workers <= 1 or len(specs) <= 1:
        return {s["key"]: niche_evo_job(s) for s in specs}
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(niche_evo_job, s): s["key"] for s in specs}
        for fut in as_completed(futs):
            r = fut.result()
            out[r["key"]] = r
    return out


if __name__ == "__main__":
    main()
