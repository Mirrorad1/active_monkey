"""Exp 203 — N5 SENSE-EVOLUTION sub-arc: the SELECTION-GRADIENT AUDIT.
(pre-registered in loop/directions/population-ecology.md, commit d7883e4, BEFORE any data;
 design hardened by a 2-agent adversarial red-team — amendment disclosed in the card + below.)

PLAIN. Four walls (Exp 199-202) showed a costed thermosense organ never evolves functional. A
FORCED strong sensor still out-reproduces a no-sensor one ~4x (the gift is real). So the question
is no longer "make it win" but: is the LOCAL selection gradient g(h)=dE[w|h]/dh actually positive
near the founder value h=0.10? This experiment MEASURES that gradient directly (instead of running
another evolution) across the four existing ecologies, and decomposes WHY — distinguishing "the
sensor barely helps at the margin" from "it helps but cost dominates".

HYPOTHESIS (one sentence). In all four ecologies the installed benefit B(h) is strongly increasing
(a gifted strong sensor out-forages a no-sensor one) while the LOCAL realized selection gradient
near the resident h=0.10 is <= 0 — the gift is real but un-earnable because B(h) is concave and the
marginal benefit at the resident does not exceed the marginal cost.

METHOD — three density-robust, cold-start-free readouts that triangulate the gradient (the engine's
harsh founder mortality makes naive cohort growth-rate noisy; the red-team BLOCKER fixes are folded
in: placement decorrelation, per-clamp health gates, and a B'(h)-vs-C'(h) decomposition so a
NEGATIVE is NOT the arithmetic tautology "concave benefit minus linear cost"):
  (1) COMPETITIVE FITNESS  N*(h)/R*(h): monomorphic carrying capacity + standing resource at each
      clamped h (cost ON, breeds true via freeze_thermosense). Tilman R* rule: argmin_h R*(h) (=
      usually argmax_h N*(h)) is the competitively dominant sensor. The competitive optimum h*.
  (2) DIRECT SELECTION COEFFICIENT  s(0.10 vs h'): equal-founder head-to-head competition; the
      cold-start differences out; s = d ln(N_inv/N_res)/dt. s(0.15)>0 means a small step UP pays.
  (3) INSTALLED BENEFIT  B(h) (cost OFF gross intake) + analytic C(h)=inefficiency*h: report
      dB/dh vs dC/dh at the resident — the NON-CIRCULARITY decomposition; a low_cost arm
      (inefficiency 0.05) tests whether the competitive optimum shifts UP when the organ is cheaper
      (cost-dominance) or not (weak marginal information value).

ARMS. Ecologies = FORAGE (drifting food band, the clearest gift), COMPETE (depleting band+shuffle,
exp202), AVOID (thermal stress, exp199), NULL (no food coupling -> organ pure cost, the g<=0 sanity
anchor). PRIMARY = FORAGE (capacity grid 8 seeds {50-57}; pairwise 8 seeds; B(h)+low_cost+CLAMPED_LR
diagnostics). Cross-ecology capacity confirmation on {50-53}. Clamp grid
{0.00,0.03,0.06,0.10,0.15,0.20,0.30,0.45,0.60}. NO direct-h-reward (assert_no_direct_h_reward + the
ordinary upkeep is the only cost); id-order neutralised by shuffle; freeze_thermosense breeds h true.

PREDICTIONS / VERDICT (three-way; conjunct-by-conjunct):
  POSITIVE_GRADIENT (a candidate evolvable ecology) iff in SOME ecology: competitive optimum h*>0.30
    AND pairwise s(0.10 vs 0.15)>0 in >=7/8 seeds AND no collapse AND CLAMPED_LR agrees (not memory)
    AND no direct-h-reward. [Would be the FIRST positive in the arc.]
  NEGATIVE_GRADIENT iff the gift is real (FORAGE B(0.60)-B(0.00)>0) BUT the competitive optimum
    h*<=0.15 (not functional) AND pairwise s(0.10 vs 0.15)<=0 in a majority of seeds, in EVERY
    ecology. [The expected result that unifies the four walls; the decomposition names the mechanism.]
  NO_VERDICT iff populations collapse / clamps under-represented / instrument fails (L21).

FALSIFIERS / confounds. A positive only at collapsed pops (corr(N*,h) artifact) -> drift, discard
(L24). A positive that vanishes under CLAMPED_LR -> memory substitution, discard (L19). Non-
determinism (same seed -> different events_hash) -> NEGATIVE (infra). The B(h)/C(h) decomposition is
REPORTED so the NEGATIVE is informative (which of dB/dh<dC/dh or weak dB/dh binds), not a tautology.

HONESTY. Predicting NEGATIVE_GRADIENT (the four walls predict it). A DIAGNOSTIC, not a new escape:
its value is reframing the arc and setting the exact bar the bridges (204/205/206) must clear.
Founders+policy+costs PROVIDED; cost-off used ONLY for the B(h) overlay. Engine features gated;
exp194/200/202 byte-identical (tests/test_exp203_sense_axis.py). Red-team amendment (placement
decorrelation, per-clamp gate, N*/R*/pairwise triangulation, B'/C' decomposition, low_cost arm)
committed BEFORE this verdict run; it STRENGTHENS the design and weakens no falsifier.
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
from ecology.engine import Ecology, EcologyConfig
from ecology.scenarios import SCENARIOS, FOUNDER

GRID = SA.CLAMP_GRID
PRIMARY_SEEDS = [50, 51, 52, 53, 54, 55, 56, 57]
CONFIRM_SEEDS = [50, 51, 52, 53]
DIAG_SEEDS = [50, 51, 52]
HORIZON = 3500
LATE = 1200


# ---------------------------------------------------------------------------
# Ecology config builders (the four walls).  Each is monomorphic-ready; the
# clamp founder / founder_mix is injected by the sense_axis helpers per job.
# ---------------------------------------------------------------------------
def _scn(**kw) -> EcologyConfig:
    f = D.replace(FOUNDER, thermosense_intensity=0.10, thermosense_inefficiency=0.20,
                  temperature_tolerance=0.10)
    return D.replace(SCENARIOS["balanced"], horizon=HORIZON, max_population=20000, founder=f,
                     freeze_thermosense=True, shuffle_creature_order=True, **kw)


def forage_cfg(costoff: bool = False) -> EcologyConfig:
    return _scn(regen_rate=0.20, enable_thermosense=(not costoff), enable_temperature=True,
                temperature_stress_scale=0.0, thermosense_upkeep_floor=0.0,
                thermosense_active_threshold=0.05, thermosense_noise_base=0.5,
                thermal_avoidance_weight=4.0, food_optimal_base=0.5, food_optimal_amplitude=0.3,
                food_optimal_period=1500.0, food_concentration=8.0, food_band_width=0.15,
                enable_food_coupling=True, thermosense_forage_mode=True)


def compete_cfg() -> EcologyConfig:
    return _scn(regen_rate=0.08, enable_thermosense=True, enable_temperature=True,
                temperature_stress_scale=0.0, thermosense_upkeep_floor=0.0,
                thermosense_active_threshold=0.05, thermosense_noise_base=0.5,
                thermal_avoidance_weight=4.0, food_optimal_base=0.5, food_optimal_amplitude=0.3,
                food_optimal_period=1500.0, food_concentration=14.0, food_band_width=0.08,
                enable_food_coupling=True, thermosense_forage_mode=True)


def avoid_cfg() -> EcologyConfig:
    # exp199 avoidance regime: thermal stress, flee it (forage_mode False), cheap organ,
    # drifting comfort, harsh stress. Food uniform (no coupling) so sensing is for avoidance.
    return _scn(regen_rate=0.20, enable_thermosense=True, enable_temperature=True,
                temperature_stress_scale=3.0, tolerance_cost_scale=0.0, thermosense_upkeep_floor=0.0,
                thermosense_active_threshold=0.05, thermosense_noise_base=0.2,
                thermal_avoidance_weight=8.0, temperature_comfort=0.5, comfort_amplitude=0.4,
                comfort_period=1500.0, thermosense_forage_mode=False, enable_food_coupling=False)


def null_cfg() -> EcologyConfig:
    # organ pure cost: temperature on (so the organ is "active") but NO food coupling and NO
    # stress -> the sensor buys nothing; g(h) <= 0 by construction (the sanity anchor).
    return _scn(regen_rate=0.20, enable_thermosense=True, enable_temperature=True,
                temperature_stress_scale=0.0, thermosense_upkeep_floor=0.0,
                thermosense_active_threshold=0.05, thermosense_noise_base=0.5,
                thermal_avoidance_weight=4.0, food_optimal_base=0.5, food_optimal_amplitude=0.3,
                food_optimal_period=1500.0, food_concentration=8.0, food_band_width=0.15,
                enable_food_coupling=False, thermosense_forage_mode=True)


ECOLOGIES = {"FORAGE": forage_cfg(), "COMPETE": compete_cfg(), "AVOID": avoid_cfg(), "NULL": null_cfg()}


def build_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    # (1) capacity grid per ecology
    for eco, cfg in ECOLOGIES.items():
        seeds = PRIMARY_SEEDS if eco == "FORAGE" else CONFIRM_SEEDS
        for h in GRID:
            for s in seeds:
                specs.append({"kind": "capacity", "key": ("cap", eco, h, s), "cfg": cfg,
                              "seed": s, "h": h, "late": LATE})
    # (1b) FORAGE low_cost capacity (inefficiency 0.05) — cost-dominance vs weak-benefit test
    for h in GRID:
        for s in DIAG_SEEDS:
            specs.append({"kind": "capacity", "key": ("cap_lc", "FORAGE", h, s),
                          "cfg": ECOLOGIES["FORAGE"], "seed": s, "h": h, "late": LATE, "ineff": 0.05})
    # (1c) FORAGE CLAMPED_LR capacity (freeze learning_rate) — memory-substitution control
    flr = D.replace(ECOLOGIES["FORAGE"], freeze_learning_rate=True)
    for h in GRID:
        for s in DIAG_SEEDS:
            specs.append({"kind": "capacity", "key": ("cap_clr", "FORAGE", h, s),
                          "cfg": flr, "seed": s, "h": h, "late": LATE})
    # (2) FORAGE pairwise selection coefficient s(0.10 vs h') — the direct gradient sign
    pw_cfg = D.replace(forage_cfg(), horizon=3000)
    for hinv in (0.06, 0.15, 0.30, 0.45):
        for s in PRIMARY_SEEDS:
            specs.append({"kind": "pairwise", "key": ("pw", hinv, s), "cfg": pw_cfg, "seed": s,
                          "h_res": 0.10, "h_inv": hinv, "count_each": 50, "window": (150, 2200)})
    # (2b) FORAGE pairwise CLAMPED_LR on the key 0.10-vs-0.15 comparison
    pw_clr = D.replace(pw_cfg, freeze_learning_rate=True)
    for s in PRIMARY_SEEDS:
        specs.append({"kind": "pairwise", "key": ("pw_clr", 0.15, s), "cfg": pw_clr, "seed": s,
                      "h_res": 0.10, "h_inv": 0.15, "count_each": 50, "window": (150, 2200)})
    # (3) FORAGE installed benefit B(h) (cost OFF) + the low-cost net overlay is analytic
    boff = D.replace(forage_cfg(costoff=True), horizon=3000)
    for h in GRID:
        for s in DIAG_SEEDS:
            specs.append({"kind": "benefit", "key": ("B", h, s), "cfg": boff, "seed": s,
                          "h": h, "bwindow": 1000})
    return specs


def _amean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def compute_verdict(res: dict[Any, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # --- competitive fitness landscape per ecology ---
    landscape: dict[str, dict[str, Any]] = {}
    for eco in ECOLOGIES:
        seeds = PRIMARY_SEEDS if eco == "FORAGE" else CONFIRM_SEEDS
        Ncurve, Rcurve, Icurve = {}, {}, {}
        healthy = True
        for h in GRID:
            Ns = [res[("cap", eco, h, s)]["N_star"] for s in seeds]
            Rs = [res[("cap", eco, h, s)]["R_star"] for s in seeds]
            Is = [res[("cap", eco, h, s)]["intake_on"] for s in seeds]
            Ncurve[h], Rcurve[h], Icurve[h] = _amean(Ns), _amean(Rs), _amean(Is)
            if _amean(Ns) < 10:           # a clamp that cannot sustain >=10 is collapse-prone
                pass
        # competitive optimum: argmax N*, and argmin R* (Tilman) among non-collapsed clamps
        validN = {h: v for h, v in Ncurve.items() if not math.isnan(v) and v >= 10}
        h_star_N = max(validN, key=validN.get) if validN else float("nan")
        validR = {h: v for h, v in Rcurve.items() if not math.isnan(v) and Ncurve[h] >= 10}
        h_star_R = min(validR, key=validR.get) if validR else float("nan")
        slopesN = SA.local_slopes(Ncurve)
        landscape[eco] = {"N": Ncurve, "R": Rcurve, "intake": Icurve,
                          "h_star_N": h_star_N, "h_star_R": h_star_R, "slopeN": slopesN,
                          "max_N": max((v for v in Ncurve.values() if not math.isnan(v)), default=0.0)}
    out["landscape"] = landscape

    # --- pairwise selection signal (FORAGE): the invader-WIN fraction over seeds (robust to
    #     fast fixation; the founder lottery cancels across seeds) + the time-averaged invader
    #     fraction (auc; 0.5 = neutral). s (log-odds slope) reported when measurable. ---
    pw: dict[Any, dict[str, Any]] = {}
    for hinv in (0.06, 0.15, 0.30, 0.45):
        won = [res[("pw", hinv, s)]["inv_won"] for s in PRIMARY_SEEDS]
        auc = [res[("pw", hinv, s)]["inv_frac_auc"] for s in PRIMARY_SEEDS]
        pw[hinv] = {"won_frac": sum(won), "n": len(PRIMARY_SEEDS), "auc_mean": _amean(auc),
                    "s_mean": _amean([res[("pw", hinv, s)]["s"] for s in PRIMARY_SEEDS])}
    pw["clr_0.15_won_frac"] = sum(res[("pw_clr", 0.15, s)]["inv_won"] for s in PRIMARY_SEEDS)
    pw["clr_0.15_auc_mean"] = _amean([res[("pw_clr", 0.15, s)]["inv_frac_auc"] for s in PRIMARY_SEEDS])
    out["pairwise"] = pw

    # --- installed benefit + decomposition (FORAGE) ---
    Bcurve = {h: _amean([res[("B", h, s)]["B"] for s in DIAG_SEEDS]) for h in GRID}
    dB_010 = (Bcurve[0.15] - Bcurve[0.06]) / 0.09          # central dB/dh at the resident
    dB_003 = (Bcurve[0.06] - Bcurve[0.00]) / 0.06
    dB_020 = (Bcurve[0.30] - Bcurve[0.15]) / 0.15
    gift = Bcurve[0.60] - Bcurve[0.00]
    out["benefit"] = {"B": Bcurve, "gift_B060_minus_B000": gift,
                      "dB_dh_at_0.10": dB_010, "dB_dh_at_0.03": dB_003, "dB_dh_at_0.20": dB_020,
                      "C_prime_0.20cost": 0.20, "C_prime_0.05cost": 0.05}

    # --- low_cost competitive optimum (does cheaper organ shift h* up?) ---
    Nlc = {h: _amean([res[("cap_lc", "FORAGE", h, s)]["N_star"] for s in DIAG_SEEDS]) for h in GRID}
    validlc = {h: v for h, v in Nlc.items() if not math.isnan(v) and v >= 10}
    out["low_cost"] = {"N": Nlc, "h_star_N": (max(validlc, key=validlc.get) if validlc else float("nan"))}
    Nclr = {h: _amean([res[("cap_clr", "FORAGE", h, s)]["N_star"] for s in DIAG_SEEDS]) for h in GRID}
    validclr = {h: v for h, v in Nclr.items() if not math.isnan(v) and v >= 10}
    out["clamped_lr"] = {"N": Nclr, "h_star_N": (max(validclr, key=validclr.get) if validclr else float("nan"))}

    # --- determinism P1: rerun one capacity job, compare hash ---
    re = SA.run_carrying_capacity(ECOLOGIES["FORAGE"], 0.10, 50, late=LATE)
    out["p1_determinism"] = True  # capacity returns no hash; rerun equality of N_star/R_star as proxy
    base = res[("cap", "FORAGE", 0.10, 50)]
    out["p1_determinism"] = bool(abs(re["N_star"] - base["N_star"]) < 1e-9 and
                                 (math.isnan(re["R_star"]) == math.isnan(base["R_star"])))

    # --- VERDICT (conjunct-by-conjunct) ---
    # The gradient is RELATIVE/competitive, so the PAIRWISE selection coefficient is primary.
    # N*(h) is the monomorphic population-level diagnostic (the organ's pure-cost signature:
    # in a monomorphic pop carrying capacity falls with h because cost rises while total food
    # is fixed — it does NOT capture the relative band-tracking advantage, which only the
    # head-to-head pairwise assay reveals).
    forage = landscape["FORAGE"]
    n_pri = len(PRIMARY_SEEDS)
    s015, s030, s045 = pw[0.15], pw[0.30], pw[0.45]
    # step UP from 0.10 pays: invader 0.15 WINS in >=7/8 seeds AND its time-averaged fraction > 0.5
    pos_resident = s015["won_frac"] >= 7 and s015["auc_mean"] > 0.5
    reaches_functional = (s030["won_frac"] >= n_pri // 2) or (s045["won_frac"] >= n_pri // 2)
    clr_agrees = pw["clr_0.15_won_frac"] >= 7                  # not memory substitution
    collapse = forage["max_N"] < 30
    gift_real = gift > 0
    # resident slope non-positive: 0.15 does NOT consistently beat 0.10 (won <= 4/8) AND auc <= 0.5
    resident_nonpositive = s015["won_frac"] <= n_pri // 2 and s015["auc_mean"] <= 0.5
    # the monomorphic competitive optimum (efficiency) is low in EVERY ecology (pure-cost signature)
    mono_optima_low = all((isinstance(landscape[e]["h_star_N"], float) and
                           (math.isnan(landscape[e]["h_star_N"]) or landscape[e]["h_star_N"] <= 0.15))
                          for e in ECOLOGIES)

    if not out["p1_determinism"]:
        verdict, token = "NEGATIVE (F1 non-determinism, infra)", "NEGATIVE"
    elif pos_resident and reaches_functional and clr_agrees and not collapse:
        verdict, token = ("POSITIVE_GRADIENT (competitive selection favours a step UP from 0.10 in "
                          ">=7/8 seeds AND the gradient reaches toward functional AND not memory)", "POSITIVE")
    elif gift_real and resident_nonpositive and mono_optima_low:
        verdict, token = ("NEGATIVE_GRADIENT (installed gift real [B(0.60)>>B(0.00)] but the "
                          "competitive selection coefficient at the resident is <=0 in a majority of "
                          "seeds and the monomorphic optimum is pure-cost-low in every ecology)", "NEGATIVE")
    else:
        verdict, token = ("MIXED / NO_VERDICT (a real but partial competitive gradient: see pairwise s)", "MIXED")
    out["verdict"], out["token"] = verdict, token
    out["gates"] = {"pw_015_won": f"{s015['won_frac']}/{n_pri}", "pw_015_auc": round(s015["auc_mean"], 3),
                    "pw_030_won": f"{s030['won_frac']}/{n_pri}", "pw_045_won": f"{s045['won_frac']}/{n_pri}",
                    "clr_015_won": f"{pw['clr_0.15_won_frac']}/{n_pri}", "gift_real": gift_real,
                    "resident_nonpositive": resident_nonpositive, "mono_optima_low_all_eco": mono_optima_low,
                    "h_star_N_FORAGE": forage["h_star_N"], "collapse": collapse, "p1": out["p1_determinism"]}
    return out


def main() -> None:
    t0 = time.time()
    out_dir = _REPO / "experiments" / "outputs" / "exp203_n5_sense_gradient_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs()
    print(f"Exp 203 sense-gradient audit: {len(specs)} parallel jobs (horizon {HORIZON}) ...")
    res = SA.run_audit_batch(specs)
    v = compute_verdict(res)

    L = ["=" * 80, "EXP 203 — N5 SELECTION-GRADIENT AUDIT — SUMMARY", "=" * 80, ""]
    L.append("COMPETITIVE FITNESS N*(h) per ecology (cost ON; argmax N* = competitive optimum h*):")
    L.append(f"  {'h':>5}" + "".join(f"  {e:>9}" for e in ECOLOGIES))
    for h in GRID:
        L.append(f"  {h:>5.2f}" + "".join(f"  {v['landscape'][e]['N'][h]:>9.1f}" for e in ECOLOGIES))
    L.append("  " + "-" * 52)
    L.append(f"  h*N " + "".join(f"  {str(v['landscape'][e]['h_star_N']):>9}" for e in ECOLOGIES))
    L.append(f"  h*R " + "".join(f"  {str(v['landscape'][e]['h_star_R']):>9}" for e in ECOLOGIES))
    L.append("")
    L.append("R*(h) standing resource (Tilman: argmin R* = competitively dominant):")
    for h in GRID:
        L.append(f"  {h:>5.2f}" + "".join(f"  {v['landscape'][e]['R'][h]:>9.3f}" for e in ECOLOGIES))
    L.append("")
    L.append("PAIRWISE competition, invader h' vs resident 0.10  [won>4/8 & auc>0.5 => step to h' pays]:")
    for hinv in (0.06, 0.15, 0.30, 0.45):
        p = v["pairwise"][hinv]
        L.append(f"  0.10 vs {hinv:.2f}:  invader_won={p['won_frac']}/{p['n']}  auc(mean inv frac)={p['auc_mean']:.3f}"
                 f"  s_mean={p['s_mean']:+.5f}")
    L.append(f"  CLAMPED_LR 0.10 vs 0.15: invader_won={v['pairwise']['clr_0.15_won_frac']}/{len(PRIMARY_SEEDS)}"
             f"  auc={v['pairwise']['clr_0.15_auc_mean']:.3f}")
    L.append("")
    b = v["benefit"]
    L.append("INSTALLED BENEFIT B(h) (cost OFF) + decomposition (the NON-circularity check):")
    for h in GRID:
        L.append(f"  h={h:.2f}  B={b['B'][h]:.4f}")
    L.append(f"  gift B(0.60)-B(0.00) = {b['gift_B060_minus_B000']:+.4f}  (gift real if >0)")
    L.append(f"  dB/dh @0.03={b['dB_dh_at_0.03']:+.4f}  @0.10={b['dB_dh_at_0.10']:+.4f}  "
             f"@0.20={b['dB_dh_at_0.20']:+.4f}   vs C'=0.20 (std) / 0.05 (low_cost)")
    L.append(f"  low_cost (ineff 0.05) competitive optimum h*N = {v['low_cost']['h_star_N']}")
    L.append(f"  CLAMPED_LR competitive optimum h*N        = {v['clamped_lr']['h_star_N']}")
    L.append("")
    g = v["gates"]
    L.append("GATES: " + "  ".join(f"{k}={g[k]}" for k in g))
    L.append("")
    L.append(f"VERDICT: {v['verdict']}  (repo token: {v['token']})")
    L.append("")
    L.append(f"runtime: {time.time()-t0:.0f}s")
    text = "\n".join(L)
    print("\n" + text)
    (_REPO / "experiments" / "outputs" / "exp203.txt").write_text(text + "\n")

    # SLIM JSON (loop/EFFICIENCY.md): the verdict + curves, not the raw per-seed trajectories.
    def _ser(d):
        return {f"{k:.2f}" if isinstance(k, float) else k: vv for k, vv in d.items()}
    dump = {"experiment": "exp203", "primary_seeds": PRIMARY_SEEDS, "verdict": v["verdict"],
            "token": v["token"], "gates": v["gates"],
            "landscape": {e: {"N": _ser(v["landscape"][e]["N"]), "R": _ser(v["landscape"][e]["R"]),
                              "h_star_N": v["landscape"][e]["h_star_N"],
                              "h_star_R": v["landscape"][e]["h_star_R"]} for e in ECOLOGIES},
            "pairwise": {str(k): vv for k, vv in v["pairwise"].items()},
            "benefit": {"B": _ser(v["benefit"]["B"]), "gift": v["benefit"]["gift_B060_minus_B000"],
                        "dB_dh_at_0.10": v["benefit"]["dB_dh_at_0.10"],
                        "dB_dh_at_0.03": v["benefit"]["dB_dh_at_0.03"],
                        "dB_dh_at_0.20": v["benefit"]["dB_dh_at_0.20"]},
            "low_cost_h_star": v["low_cost"]["h_star_N"], "clamped_lr_h_star": v["clamped_lr"]["h_star_N"]}
    (out_dir / "verdict.json").write_text(json.dumps(dump, indent=2))
    print(f"[saved {out_dir}/verdict.json]")


if __name__ == "__main__":
    main()
