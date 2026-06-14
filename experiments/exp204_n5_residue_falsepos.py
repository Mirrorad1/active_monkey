"""Exp 204 — N5 SENSE-EVOLUTION sub-arc: the FRESHNESS/RESIDUE FALSE-POSITIVE BRIDGE.
(pre-registered in loop/directions/population-ecology.md, commit 1aa80bf, BEFORE any data,
 on the human's explicit word "a continue" = post-203 consult option (a).)

PLAIN. The four walls (Exp 199-202) and the Exp 203 audit showed a costed thermosense organ
never becomes a functional sensor: precision only ever bought marginally MORE food (a benefit
that saturates, so a crude sensor grabs the easy part and precision never pays its cost). This
experiment changes WHAT precision is for: eaten food leaves a misleading RESIDUE that low-
precision sensors confuse with fresh food, so precision now AVOIDS A COSTLY MISTAKE (eating
residue → an energy loss). Avoiding a loss is a steeper payoff than grabbing a little more, and
competition raises residue (rivals deplete → traces accumulate), so the value of precision should
rise with crowding. Question: does this finally make a functional sensor evolve?

HYPOTHESIS (one sentence). When precision reduces costly false positives (eating residue) and
competition raises residue density, the local selection gradient at the resident becomes positive
AND a functional sensor (h>0.30) evolves — unlike the saturating/concave four walls.

MECHANISM (gated enable_residue; exp194-203 byte-identical OFF, hash-verified). At the eat step the
creature reads a noisy freshness percept f_hat = f + N(0, residue_confusion*(1-h)) of the true fresh
fraction f = R/(R+residue); it eats iff f_hat >= eat_threshold; eating an actually-residue-dominated
cell (f < fp_threshold) costs residue_loss (a false positive). ANTI-CHEAT: intake is the UNCHANGED
consume(); h keys ONLY the percept noise; residue_loss is an action cost, identical regardless of h —
never a reward on h (guard test confusion=0 ⇒ h irrelevant).

TWO MODES.
  A — GRADIENT AUDIT (the Exp 203 instrument): clamp-grid pairwise selection coefficient s(0.10 vs h'),
      competitive optimum h* (carrying-capacity N*/R*), B(h) overlay (cost-off), in RESIDUE_COMPETE vs
      the NO_RESIDUE control, across a residue_confusion DIFFICULTY sweep.
  B — EVOLUTION (the verdict confirmation): h evolves freely from the primitive founder (0.10); does
      the gene-pool NEWBORN mean h climb >0.30, MEDIATED by a falling false-positive rate?

ARMS (cost ON, shuffle ON unless noted; depleting band as Exp 202: regen 0.08, conc 14, band 0.08):
  RESIDUE_COMPETE (primary) · NO_RESIDUE (control, residue OFF) · RESIDUE_NO_COMPETE (residue ON,
  abundant regen 0.8 → little residue) · NO_SHUFFLE (id-order) · CLAMPED_LR (freeze learning_rate).

VERDICT (three-way, conjunct-by-conjunct; full text in the card):
  POSITIVE iff RESIDUE_COMPETE: P1 determinism; P2 healthy pop; P3 pairwise s(0.10→0.15)>0 & won≥7/8 &
    competitive optimum h*>0.30; P4 evolution newborn mean h>0.30 in ≥4/5; P5 FP-rate falls with h &
    the repro advantage is FP-mediated; P6 CLAMPED_LR agrees & NO_SHUFFLE doesn't flip & no h-reward;
    P7 NO_RESIDUE & RESIDUE_NO_COMPETE stay primitive.
  NEGATIVE iff RESIDUE_COMPETE h stays primitive (evolution <0.15 majority and/or h*≤0.15) DESPITE a
    real installed FP-avoidance benefit — a FIFTH, most-general wall.
  MIXED iff partial (above primitive but <0.30, or only one mode, or weakly mediated).
FALSIFIERS. F1 non-determinism→NEGATIVE(infra). F2 collapse→NO_VERDICT. F3 NO_RESIDUE also climbs→
  not residue-mediated, DISCARD. F4 positive only at collapsed pop→DRIFT (L24). F5 vanishes under
  CLAMPED_LR→memory (L19). F6 flips under NO_SHUFFLE→id-order. F7 h rises but FP-rate doesn't fall→
  not the predeclared mechanism, reinterpret.

HONESTY. The four walls + Exp 203 predict NEGATIVE/MIXED; a POSITIVE is a genuine surprise (a new
payoff structure, not a forced result). Residue params tuned on disclosed pilots, FIXED before the
verdict run. Founders+policy+costs PROVIDED. Cost-off used ONLY for the B(h) overlay.
"""
from __future__ import annotations

import dataclasses as D
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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

GRID = SA.CLAMP_GRID
AUDIT_SEEDS = [50, 51, 52, 53, 54, 55, 56, 57]   # 8, for the >=7/8 slope bar
DIAG_SEEDS = [50, 51, 52]
EVO_SEEDS = [70, 71, 72, 73, 74]                  # 5 fresh, for the evolution verdict
AUDIT_HORIZON = 3500
LATE = 1200
GROWTH_HORIZON = 1500
GROWTH_WINDOW = (100, 700)
EVO_HORIZON = 8000
NEWBORN_WINDOW = 2000                             # newborns in [EVO_HORIZON-NEWBORN_WINDOW, EVO_HORIZON]

# Residue regime — FIXED here, tuned on DISCLOSED pilots (seeds {100,101}; deleted, not
# committed) to give the bridge its FAIREST shot before any verdict (L7/L20):
#   - residue_loss=0.5 (the card default) was too MILD: the monomorphic competitive optimum
#     stayed at the resident h*=0.10 (precision didn't even pay when gifted) — an unfair test.
#   - residue_loss=3.0 was too HARSH: it killed primitive populations outright (N*(0.10)=0),
#     so the founder could never survive long enough to climb — also unfair.
#   - residue_loss=1.5 is the SWEET SPOT: a PURE high-h population is genuinely fitter
#     (monomorphic N*(h) rises 24→52→76 across h=0.10→0.30→0.60, competitive optimum h*=0.60
#     FUNCTIONAL — the FIRST functional optimum in the whole arc) AND primitive populations
#     still survive (N*(0.10)=24), so the founder CAN in principle climb. This is the deck
#     deliberately STACKED FOR the organ (the exp199 philosophy): a NEGATIVE here is the
#     STRONG conclusion (even a genuinely-functional monomorphic optimum is un-earnable).
# The evolution pilot at this regime (loss=1.5) already SUGGESTED the bridge fails the
# REALIZED test (newborn h DECAYED 0.10→~0.05 and one of two seeds went extinct — the L22
# forced-vs-evolvable gap again); the verdict is run on FRESH seeds {70-74}.
RES_YIELD = 1.0
RES_DECAY = 0.05
RES_LOSS = 1.5
EAT_THR = 0.5
FP_THR = 0.5
# Difficulty sweep over the signature gap (percept noise sd = confusion*(1-h)).
CONF_EASY, CONF_MED, CONF_HARD = 0.30, 0.60, 1.00


# ---------------------------------------------------------------------------
# Config builders.  RESIDUE_COMPETE = the Exp 202 depleting-band compete regime
# (forage toward the band, depleting + shuffle) with the residue mechanic layered
# ON.  NO_RESIDUE is the SAME regime with enable_residue=False — the matched control
# that isolates exactly the residue/false-positive contribution (forage-navigation
# precision is present in BOTH arms; only residue discrimination differs).
# ---------------------------------------------------------------------------
def _base(enable_thermosense: bool = True, enable_food_coupling: bool = True,
          **kw) -> EcologyConfig:
    f = D.replace(FOUNDER, thermosense_intensity=0.10, thermosense_inefficiency=0.20,
                  temperature_tolerance=0.10)
    return D.replace(SCENARIOS["balanced"], horizon=AUDIT_HORIZON, max_population=20000, founder=f,
                     freeze_thermosense=True, shuffle_creature_order=True,
                     enable_thermosense=enable_thermosense, enable_temperature=True,
                     temperature_stress_scale=0.0,
                     thermosense_upkeep_floor=0.0, thermosense_active_threshold=0.05,
                     thermosense_noise_base=0.5, thermal_avoidance_weight=4.0,
                     food_optimal_base=0.5, food_optimal_amplitude=0.3, food_optimal_period=1500.0,
                     food_concentration=14.0, food_band_width=0.08,
                     enable_food_coupling=enable_food_coupling,
                     thermosense_forage_mode=True, **kw)


def residue_compete_cfg(confusion: float = CONF_MED, costoff: bool = False) -> EcologyConfig:
    return _base(regen_rate=0.08, enable_thermosense=(not costoff),
                 enable_residue=True, residue_confusion=confusion, residue_yield=RES_YIELD,
                 residue_decay=RES_DECAY, residue_loss=RES_LOSS,
                 residue_eat_threshold=EAT_THR, residue_fp_threshold=FP_THR)


def no_residue_cfg(costoff: bool = False) -> EcologyConfig:
    # the matched control: identical compete regime, residue mechanic OFF.
    return _base(regen_rate=0.08, enable_thermosense=(not costoff), enable_residue=False)


def residue_nocompete_cfg(confusion: float = CONF_MED) -> EcologyConfig:
    # residue ON but NO band-concentration competition: UNIFORM food (enable_food_coupling=False)
    # at the bounded balanced regen 0.20 (exp194 balanced ~150 pop — NOT abundant, which the
    # runtime pre-flight correctly flagged as an explosion, the Exp 202 ABUNDANT lesson/L25).
    # Uniform food ⇒ eating is spread across cells, no crowding hotspot ⇒ residue stays diffuse
    # and low-density ⇒ few residue-dominated cells ⇒ few false-positive opportunities ⇒ the
    # gradient should be FLAT.  Tests whether COMPETITION (band crowding) is what makes precision
    # pay — the necessity contrast on the competition axis.
    return _base(regen_rate=0.20, enable_food_coupling=False, enable_residue=True,
                 residue_confusion=confusion, residue_yield=RES_YIELD, residue_decay=RES_DECAY,
                 residue_loss=RES_LOSS, residue_eat_threshold=EAT_THR, residue_fp_threshold=FP_THR)


# ---------------------------------------------------------------------------
# MODE B — evolution runner (picklable, top-level for ProcessPoolExecutor).
# ---------------------------------------------------------------------------
def evo_job(spec: dict[str, Any]) -> dict[str, Any]:
    """Run ONE free-evolution ecology and return the gene-pool newborn h + FP-rate mediation.

    h evolves (freeze_thermosense=False, mutate_thermosense via enable_thermosense=True) from the
    primitive founder 0.10.  NEWBORN mean intensity over the last NEWBORN_WINDOW is the gene-pool
    selection readout (the heritable signal, robust to survivor bias — exp196-202 convention).
    Mediation: per newborn we also bucket by intensity and read the parent-life FP-rate so a climb
    can be shown to track a FALLING false-positive rate (P5), not a different channel.
    """
    cfg: EcologyConfig = spec["cfg"]
    seed: int = spec["seed"]
    eco = Ecology(cfg, seed=seed)
    eco.run()
    w_lo = cfg.horizon - NEWBORN_WINDOW
    newborn_h: list[float] = []
    # per-creature (over those born in window) intensity + realized FP-rate (mediation)
    h_lo_fp: list[float] = []     # FP-rate of low-intensity (<0.15) window-born creatures
    h_hi_fp: list[float] = []     # FP-rate of high-intensity (>=0.30) window-born creatures
    h_lo_off: list[int] = []      # offspring of low-intensity window-born
    h_hi_off: list[int] = []      # offspring of high-intensity window-born
    n_alive_end = 0
    pops: list[int] = []
    for c in eco._creatures:
        ph = c.phenotype
        if ph.alive:
            n_alive_end += 1
        if c.parent_id is not None and ph.birth_t >= w_lo:
            h = c.genotype.thermosense_intensity
            newborn_h.append(h)
            decided = ph.tp_count + ph.fp_count
            fp_rate = (ph.fp_count / decided) if decided > 0 else float("nan")
            if h < 0.15:
                if not math.isnan(fp_rate):
                    h_lo_fp.append(fp_rate)
                h_lo_off.append(ph.offspring_count)
            elif h >= 0.30:
                if not math.isnan(fp_rate):
                    h_hi_fp.append(fp_rate)
                h_hi_off.append(ph.offspring_count)

    def _m(xs):
        xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "key": spec["key"],
        "newborn_mean_h": _m(newborn_h),
        "newborn_max_h": (float(np.max(newborn_h)) if newborn_h else float("nan")),
        "n_newborn": len(newborn_h),
        "final_pop": n_alive_end,
        "extinct": n_alive_end == 0,
        "exploded": eco.exploded,
        "fp_rate_lo_h": _m(h_lo_fp),     # P5 mediation: should be HIGHER than hi
        "fp_rate_hi_h": _m(h_hi_fp),
        "offspring_lo_h": _m(h_lo_off),
        "offspring_hi_h": _m(h_hi_off),  # P5 mediation: should be >= lo if hi-h pays
        "events_hash": eco.events_hash(),
    }


def run_evo_batch(specs: list[dict[str, Any]], workers: int) -> dict[Any, dict[str, Any]]:
    if workers <= 1 or len(specs) <= 1:
        return {s["key"]: evo_job(s) for s in specs}
    out: dict[Any, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(evo_job, s): s["key"] for s in specs}
        for fut in as_completed(futs):
            r = fut.result()
            out[r["key"]] = r
    return out


# ---------------------------------------------------------------------------
# MODE A — gradient-audit specs (reuse the sense_axis dispatcher).
# ---------------------------------------------------------------------------
AUDIT_ECO = {"RESIDUE_COMPETE": residue_compete_cfg(CONF_MED), "NO_RESIDUE": no_residue_cfg()}


def build_audit_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    # (1) capacity grid: RESIDUE_COMPETE + NO_RESIDUE
    for eco, cfg in AUDIT_ECO.items():
        for h in GRID:
            for s in AUDIT_SEEDS:
                specs.append({"kind": "capacity", "key": ("cap", eco, h, s), "cfg": cfg,
                              "seed": s, "h": h, "late": LATE})
    # (2) pairwise s(0.10 vs h') in RESIDUE_COMPETE (the direct gradient sign), MED difficulty
    pw_cfg = D.replace(residue_compete_cfg(CONF_MED), horizon=3000)
    for hinv in (0.06, 0.15, 0.30, 0.45):
        for s in AUDIT_SEEDS:
            specs.append({"kind": "pairwise", "key": ("pw", hinv, s), "cfg": pw_cfg, "seed": s,
                          "h_res": 0.10, "h_inv": hinv, "count_each": 50, "window": (150, 2200)})
    # (2b) pairwise 0.10 vs 0.15 in NO_RESIDUE — the necessity contrast (should be ~neutral as 202/203)
    pw_nr = D.replace(no_residue_cfg(), horizon=3000)
    for s in AUDIT_SEEDS:
        specs.append({"kind": "pairwise", "key": ("pw_nr", 0.15, s), "cfg": pw_nr, "seed": s,
                      "h_res": 0.10, "h_inv": 0.15, "count_each": 50, "window": (150, 2200)})
    # (2c) pairwise 0.10 vs 0.15 CLAMPED_LR (memory control) + NO_SHUFFLE (id-order control)
    pw_clr = D.replace(pw_cfg, freeze_learning_rate=True)
    pw_nos = D.replace(pw_cfg, shuffle_creature_order=False)
    for s in AUDIT_SEEDS:
        specs.append({"kind": "pairwise", "key": ("pw_clr", 0.15, s), "cfg": pw_clr, "seed": s,
                      "h_res": 0.10, "h_inv": 0.15, "count_each": 50, "window": (150, 2200)})
        specs.append({"kind": "pairwise", "key": ("pw_nos", 0.15, s), "cfg": pw_nos, "seed": s,
                      "h_res": 0.10, "h_inv": 0.15, "count_each": 50, "window": (150, 2200)})
    # (2d) DIFFICULTY SWEEP on the key 0.10-vs-0.15 comparison: easy / hard (med is (2) above)
    for tag, conf in (("easy", CONF_EASY), ("hard", CONF_HARD)):
        pwc = D.replace(residue_compete_cfg(conf), horizon=3000)
        for s in AUDIT_SEEDS:
            specs.append({"kind": "pairwise", "key": ("pw_diff", tag, s), "cfg": pwc, "seed": s,
                          "h_res": 0.10, "h_inv": 0.15, "count_each": 50, "window": (150, 2200)})
    # (3) B(h) overlay: density-independent intrinsic growth, cost-OFF (gift) + cost-ON, RESIDUE_COMPETE
    g_off = D.replace(residue_compete_cfg(CONF_MED, costoff=True), horizon=GROWTH_HORIZON)
    g_on = D.replace(residue_compete_cfg(CONF_MED, costoff=False), horizon=GROWTH_HORIZON)
    for h in GRID:
        for s in DIAG_SEEDS:
            specs.append({"kind": "growth", "key": ("goff", h, s), "cfg": g_off, "seed": s,
                          "h": h, "window": GROWTH_WINDOW})
            specs.append({"kind": "growth", "key": ("gon", h, s), "cfg": g_on, "seed": s,
                          "h": h, "window": GROWTH_WINDOW})
    return specs


def _amean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def compute_verdict(res: dict[Any, dict[str, Any]], evo: dict[Any, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # --- competitive fitness landscape (RESIDUE_COMPETE + NO_RESIDUE) ---
    landscape: dict[str, dict[str, Any]] = {}
    for eco in AUDIT_ECO:
        Ncurve, Rcurve = {}, {}
        for h in GRID:
            Ncurve[h] = _amean([res[("cap", eco, h, s)]["N_star"] for s in AUDIT_SEEDS])
            Rcurve[h] = _amean([res[("cap", eco, h, s)]["R_star"] for s in AUDIT_SEEDS])
        validN = {h: v for h, v in Ncurve.items() if not math.isnan(v) and v >= 10}
        h_star_N = max(validN, key=validN.get) if validN else float("nan")
        landscape[eco] = {"N": Ncurve, "R": Rcurve, "h_star_N": h_star_N,
                          "max_N": max((v for v in Ncurve.values() if not math.isnan(v)), default=0.0)}
    out["landscape"] = landscape

    # --- pairwise selection signal (RESIDUE_COMPETE) ---
    pw: dict[Any, dict[str, Any]] = {}
    for hinv in (0.06, 0.15, 0.30, 0.45):
        won = [res[("pw", hinv, s)]["inv_won"] for s in AUDIT_SEEDS]
        auc = [res[("pw", hinv, s)]["inv_frac_auc"] for s in AUDIT_SEEDS]
        pw[hinv] = {"won_frac": sum(won), "n": len(AUDIT_SEEDS), "auc_mean": _amean(auc),
                    "s_mean": _amean([res[("pw", hinv, s)]["s"] for s in AUDIT_SEEDS])}
    for tag, key in (("nr", "pw_nr"), ("clr", "pw_clr"), ("nos", "pw_nos")):
        pw[f"{tag}_015_won"] = sum(res[(key, 0.15, s)]["inv_won"] for s in AUDIT_SEEDS)
        pw[f"{tag}_015_auc"] = _amean([res[(key, 0.15, s)]["inv_frac_auc"] for s in AUDIT_SEEDS])
    for tag in ("easy", "hard"):
        pw[f"diff_{tag}_won"] = sum(res[("pw_diff", tag, s)]["inv_won"] for s in AUDIT_SEEDS)
        pw[f"diff_{tag}_auc"] = _amean([res[("pw_diff", tag, s)]["inv_frac_auc"] for s in AUDIT_SEEDS])
    out["pairwise"] = pw

    # --- B(h) overlay ---
    Bcurve = {h: _amean([res[("goff", h, s)]["r"] for s in DIAG_SEEDS]) for h in GRID}
    Roncurve = {h: _amean([res[("gon", h, s)]["r"] for s in DIAG_SEEDS]) for h in GRID}
    gift = Bcurve[0.60] - Bcurve[0.00]
    out["benefit"] = {"B": Bcurve, "r_on": Roncurve, "gift_B060_minus_B000": gift}

    # --- evolution (Mode B) ---
    evo_summ: dict[str, dict[str, Any]] = {}
    EVO_VALID_POP = 30          # L21: a seed with final_pop below this is INVALID for the metric
    for arm in ("RESIDUE_COMPETE", "NO_RESIDUE", "RESIDUE_NO_COMPETE"):
        hs = [evo[(arm, s)]["newborn_mean_h"] for s in EVO_SEEDS]
        pops = [evo[(arm, s)]["final_pop"] for s in EVO_SEEDS]
        # L21: a seed is VALID only if it has a measurable newborn cohort at a non-collapsed pop.
        valid = [not math.isnan(h) and p >= EVO_VALID_POP for h, p in zip(hs, pops)]
        hs_valid = [h for h, ok in zip(hs, valid) if ok]
        funct = sum(1 for h in hs_valid if h > 0.30)
        evo_summ[arm] = {
            "newborn_h": hs, "mean_newborn_h": _amean(hs_valid),
            "n_functional": funct, "n_valid": sum(valid), "n_seeds": len(EVO_SEEDS),
            "final_pops": pops, "min_pop": min(pops),
            "fp_rate_lo_h": _amean([evo[(arm, s)]["fp_rate_lo_h"] for s in EVO_SEEDS]),
            "fp_rate_hi_h": _amean([evo[(arm, s)]["fp_rate_hi_h"] for s in EVO_SEEDS]),
            "offspring_lo_h": _amean([evo[(arm, s)]["offspring_lo_h"] for s in EVO_SEEDS]),
            "offspring_hi_h": _amean([evo[(arm, s)]["offspring_hi_h"] for s in EVO_SEEDS]),
        }
    out["evolution"] = evo_summ

    # --- determinism P1: rerun one evo job, compare events_hash ---
    re = evo_job({"cfg": residue_compete_evo_cfg(), "seed": EVO_SEEDS[0], "key": ("re", 0)})
    out["p1_determinism"] = bool(re["events_hash"] == evo[("RESIDUE_COMPETE", EVO_SEEDS[0])]["events_hash"])

    # --- VERDICT (conjunct-by-conjunct) ---
    rc = evo_summ["RESIDUE_COMPETE"]
    nr = evo_summ["NO_RESIDUE"]
    rnc = evo_summ["RESIDUE_NO_COMPETE"]
    forage = landscape["RESIDUE_COMPETE"]
    n = len(AUDIT_SEEDS)
    s015 = pw[0.15]
    # P3 gradient (Mode A)
    pos_resident = s015["won_frac"] >= 7 and s015["auc_mean"] > 0.5
    h_star_functional = isinstance(forage["h_star_N"], float) and not math.isnan(forage["h_star_N"]) \
        and forage["h_star_N"] > 0.30
    # P4 evolution (Mode B)
    evo_functional = rc["n_functional"] >= 4
    # P5 mediation: high-h fewer FPs AND high-h reproduces >= low-h
    mediated = (not math.isnan(rc["fp_rate_hi_h"]) and not math.isnan(rc["fp_rate_lo_h"])
                and rc["fp_rate_hi_h"] < rc["fp_rate_lo_h"]
                and rc["offspring_hi_h"] >= rc["offspring_lo_h"])
    # P6 confound-clean
    clr_agrees = pw["clr_015_won"] >= 7
    noshuffle_ok = pw["nos_015_won"] >= 5            # not an id-order artifact (still wins majority)
    # P7 necessity contrast
    no_residue_primitive = (nr["n_functional"] == 0 and (math.isnan(nr["mean_newborn_h"])
                            or nr["mean_newborn_h"] < 0.15))
    nocompete_primitive = (rnc["n_functional"] == 0 and (math.isnan(rnc["mean_newborn_h"])
                           or rnc["mean_newborn_h"] < 0.15))
    # validity / drift guards
    rc_invalid = rc["n_valid"] < 4                          # L21: need >=4/5 measurable seeds
    gift_real = gift > 0                                    # installed FP-avoidance benefit real
    h_star_functional_RC = h_star_functional               # monomorphic optimum functional (loss-tuned)
    rc_above_control = (not math.isnan(rc["mean_newborn_h"]) and not math.isnan(nr["mean_newborn_h"])
                        and rc["mean_newborn_h"] > nr["mean_newborn_h"] + 0.01)  # weak-but-real residue push
    rc_primitive = (rc["n_functional"] <= len(EVO_SEEDS) // 2
                    and (math.isnan(rc["mean_newborn_h"]) or rc["mean_newborn_h"] < 0.15))
    weak = " (a real but SUB-FUNCTIONAL residue push: RESIDUE_COMPETE newborn h exceeds NO_RESIDUE " \
           "but stays primitive — the exp203-style weak gradient)" if rc_above_control else ""

    if not out["p1_determinism"]:
        verdict, token = "NEGATIVE (F1 non-determinism, infra)", "NEGATIVE"
    elif rc_invalid:
        verdict, token = ("MIXED / NO_VERDICT (RESIDUE_COMPETE under-represented: <4/5 valid seeds — "
                          "L21; metric not licensed)", "MIXED")
    elif (pos_resident and h_star_functional_RC and evo_functional and mediated and clr_agrees
          and noshuffle_ok and no_residue_primitive and nocompete_primitive):
        verdict, token = ("POSITIVE (the residue/false-positive bridge crosses the valley: a functional "
                          "sensor evolves [h>0.30], the resident gradient is positive, it is mediated by "
                          "fewer false positives, and it is residue- AND competition-necessary)", "POSITIVE")
    elif rc_primitive and gift_real:
        verdict, token = (f"NEGATIVE (FIFTH WALL: even a GENUINELY-FUNCTIONAL monomorphic optimum "
                          f"[N*(h) rises to h*={forage['h_star_N']}] is un-earnable — the organ stays "
                          f"primitive under evolution because the LOCAL resident gradient is <=0 and the "
                          f"false-positive cost craters the population; the ceiling holds even for "
                          f"loss-avoidance){weak}", "NEGATIVE")
    else:
        verdict, token = (f"MIXED (a partial residue gradient — above primitive but not functional, or "
                          f"only one mode, or weakly mediated){weak}", "MIXED")
    out["verdict"], out["token"] = verdict, token
    out["gates"] = {
        "pw_015_won": f"{s015['won_frac']}/{n}", "pw_015_auc": round(s015["auc_mean"], 3),
        "pw_nr_won": f"{pw['nr_015_won']}/{n}", "pw_clr_won": f"{pw['clr_015_won']}/{n}",
        "pw_nos_won": f"{pw['nos_015_won']}/{n}",
        "diff_easy_won": f"{pw['diff_easy_won']}/{n}", "diff_hard_won": f"{pw['diff_hard_won']}/{n}",
        "h_star_N_RC": forage["h_star_N"], "gift_real": gift_real, "gift_B": round(gift, 4),
        "evo_RC_mean_h": round(rc["mean_newborn_h"], 4), "evo_RC_functional": f"{rc['n_functional']}/5",
        "evo_NR_mean_h": round(nr["mean_newborn_h"], 4), "evo_RNC_mean_h": round(rnc["mean_newborn_h"], 4),
        "fp_lo_h": round(rc["fp_rate_lo_h"], 3), "fp_hi_h": round(rc["fp_rate_hi_h"], 3),
        "off_lo_h": round(rc["offspring_lo_h"], 3), "off_hi_h": round(rc["offspring_hi_h"], 3),
        "mediated": mediated, "rc_n_valid": f"{rc['n_valid']}/{len(EVO_SEEDS)}",
        "rc_above_control": rc_above_control, "rc_min_pop": rc["min_pop"], "p1": out["p1_determinism"],
    }
    return out


def residue_compete_evo_cfg(confusion: float = CONF_MED) -> EcologyConfig:
    """RESIDUE_COMPETE for the EVOLUTION mode: h evolves (freeze_thermosense=False)."""
    return D.replace(residue_compete_cfg(confusion), horizon=EVO_HORIZON, freeze_thermosense=False)


def build_evo_specs() -> list[dict[str, Any]]:
    arms = {
        "RESIDUE_COMPETE": residue_compete_evo_cfg(),
        "NO_RESIDUE": D.replace(no_residue_cfg(), horizon=EVO_HORIZON, freeze_thermosense=False),
        "RESIDUE_NO_COMPETE": D.replace(residue_nocompete_cfg(), horizon=EVO_HORIZON,
                                        freeze_thermosense=False),
    }
    return [{"kind": "evo", "key": (arm, s), "cfg": cfg, "seed": s}
            for arm, cfg in arms.items() for s in EVO_SEEDS]


# ---------------------------------------------------------------------------
# Pilot (disclosed per L7): cheap regime check + param disclosure, NOT the verdict.
# ---------------------------------------------------------------------------
def pilot() -> None:
    print("PILOT (disclosed, not committed as verdict): residue regime sanity on seeds {100,101}")
    for conf in (CONF_EASY, CONF_MED, CONF_HARD):
        for s in (100, 101):
            cfg = residue_compete_evo_cfg(conf)
            eco = Ecology(cfg, seed=s)
            eco.run()
            w_lo = cfg.horizon - NEWBORN_WINDOW
            nh = [c.genotype.thermosense_intensity for c in eco._creatures
                  if c.parent_id is not None and c.phenotype.birth_t >= w_lo]
            fp = sum(c.phenotype.fp_count for c in eco._creatures)
            tp = sum(c.phenotype.tp_count for c in eco._creatures)
            maxres = float(np.max(eco.world.residue)) if eco.world.residue is not None else 0.0
            print(f"  conf={conf:.2f} seed={s}: pop={eco.alive_count()} newborn_h="
                  f"{(np.mean(nh) if nh else float('nan')):.4f} fp_rate="
                  f"{fp/max(1,fp+tp):.3f} max_residue={maxres:.2f} n_newborn={len(nh)}")


def main() -> None:
    if "--pilot" in sys.argv:
        pilot()
        return
    t0 = time.time()
    out_dir = _REPO / "experiments" / "outputs" / "exp204_n5_residue_falsepos"
    out_dir.mkdir(parents=True, exist_ok=True)

    # RUNTIME PRE-FLIGHT (binding, PROTOCOL step 3 / L25): probe the most explosion-prone arms
    # (lowest-cost h=0) and confirm population + wall-clock bounded BEFORE launching the batch.
    reps = [(name, D.replace(cfg, founder=D.replace(cfg.founder, thermosense_intensity=0.0)), 50)
            for name, cfg in {"RESIDUE_COMPETE": residue_compete_evo_cfg(),
                              "NO_RESIDUE": D.replace(no_residue_cfg(), horizon=EVO_HORIZON),
                              "RESIDUE_NO_COMPETE": D.replace(residue_nocompete_cfg(), horizon=EVO_HORIZON)}.items()]
    pf = RB.preflight(reps, horizon=EVO_HORIZON, n_jobs=len(build_evo_specs()) + len(build_audit_specs()),
                      max_workers=SA._audit_workers(), probe_steps=1200, time_budget_s=3000,
                      require_safe=True)
    print(RB.format_report(pf) + "\n")
    workers = pf["recommended_workers"]

    audit_specs = build_audit_specs()
    evo_specs = build_evo_specs()
    print(f"Exp 204 residue/false-positive bridge: {len(audit_specs)} audit + {len(evo_specs)} evo "
          f"jobs ({workers} workers) ...")
    res = SA.run_audit_batch(audit_specs, max_workers=workers)
    evo = run_evo_batch(evo_specs, workers=workers)
    v = compute_verdict(res, evo)

    L = ["=" * 80, "EXP 204 — N5 RESIDUE / FALSE-POSITIVE BRIDGE — SUMMARY", "=" * 80, ""]
    L.append("COMPETITIVE FITNESS N*(h) (cost ON; argmax N* = competitive optimum h*):")
    L.append(f"  {'h':>5}" + "".join(f"  {e:>15}" for e in AUDIT_ECO))
    for h in GRID:
        L.append(f"  {h:>5.2f}" + "".join(f"  {v['landscape'][e]['N'][h]:>15.1f}" for e in AUDIT_ECO))
    L.append(f"  h*N " + "".join(f"  {str(v['landscape'][e]['h_star_N']):>15}" for e in AUDIT_ECO))
    L.append("")
    L.append("PAIRWISE competition, invader h' vs resident 0.10 (RESIDUE_COMPETE, MED difficulty):")
    for hinv in (0.06, 0.15, 0.30, 0.45):
        p = v["pairwise"][hinv]
        L.append(f"  0.10 vs {hinv:.2f}:  invader_won={p['won_frac']}/{p['n']}  auc={p['auc_mean']:.3f}"
                 f"  s_mean={p['s_mean']:+.5f}")
    pwd = v["pairwise"]
    L.append(f"  0.10 vs 0.15 controls: NO_RESIDUE won={pwd['nr_015_won']}/{len(AUDIT_SEEDS)} (auc {pwd['nr_015_auc']:.3f})"
             f"  CLAMPED_LR won={pwd['clr_015_won']}/{len(AUDIT_SEEDS)}"
             f"  NO_SHUFFLE won={pwd['nos_015_won']}/{len(AUDIT_SEEDS)}")
    L.append(f"  difficulty sweep (0.10 vs 0.15): easy won={pwd['diff_easy_won']}/{len(AUDIT_SEEDS)} "
             f"(auc {pwd['diff_easy_auc']:.3f})  med won={v['pairwise'][0.15]['won_frac']}/{len(AUDIT_SEEDS)}"
             f"  hard won={pwd['diff_hard_won']}/{len(AUDIT_SEEDS)} (auc {pwd['diff_hard_auc']:.3f})")
    L.append("")
    b = v["benefit"]
    L.append("INSTALLED BENEFIT B(h) (cost-OFF intrinsic growth) + realized r_on:")
    L.append(f"  {'h':>5}  {'B=r_off':>9}  {'r_on':>9}")
    for h in GRID:
        L.append(f"  {h:>5.2f}  {b['B'][h]:>9.5f}  {b['r_on'][h]:>9.5f}")
    L.append(f"  gift B(0.60)-B(0.00) = {b['gift_B060_minus_B000']:+.5f}  (gift real if >0)")
    L.append("")
    L.append("EVOLUTION (Mode B) — gene-pool NEWBORN mean intensity over last "
             f"{NEWBORN_WINDOW} steps (founder 0.10):")
    for arm in ("RESIDUE_COMPETE", "NO_RESIDUE", "RESIDUE_NO_COMPETE"):
        e = v["evolution"][arm]
        L.append(f"  {arm:>18}: mean_h={e['mean_newborn_h']:.4f}  functional(>0.30)={e['n_functional']}/{e['n_seeds']}"
                 f"  per-seed={[round(x,3) for x in e['newborn_h']]}  min_pop={e['min_pop']}")
    rc = v["evolution"]["RESIDUE_COMPETE"]
    L.append(f"  MEDIATION (RESIDUE_COMPETE): FP-rate lo-h={rc['fp_rate_lo_h']:.3f} hi-h={rc['fp_rate_hi_h']:.3f}"
             f"  | offspring lo-h={rc['offspring_lo_h']:.3f} hi-h={rc['offspring_hi_h']:.3f}")
    L.append("")
    g = v["gates"]
    L.append("GATES: " + "  ".join(f"{k}={g[k]}" for k in g))
    L.append("")
    L.append(f"VERDICT: {v['verdict']}  (repo token: {v['token']})")
    L.append("")
    L.append(f"runtime: {time.time()-t0:.0f}s")
    text = "\n".join(L)
    print("\n" + text)
    (_REPO / "experiments" / "outputs" / "exp204.txt").write_text(text + "\n")

    def _ser(d):
        return {f"{k:.2f}" if isinstance(k, float) else k: vv for k, vv in d.items()}
    dump = {"experiment": "exp204", "audit_seeds": AUDIT_SEEDS, "evo_seeds": EVO_SEEDS,
            "verdict": v["verdict"], "token": v["token"], "gates": v["gates"],
            "landscape": {e: {"N": _ser(v["landscape"][e]["N"]), "h_star_N": v["landscape"][e]["h_star_N"]}
                          for e in AUDIT_ECO},
            "pairwise": {str(k): vv for k, vv in v["pairwise"].items()},
            "benefit": {"B": _ser(v["benefit"]["B"]), "gift": v["benefit"]["gift_B060_minus_B000"]},
            "evolution": v["evolution"]}
    (out_dir / "verdict.json").write_text(json.dumps(dump, indent=2))
    print(f"[saved {out_dir}/verdict.json]")


if __name__ == "__main__":
    main()
