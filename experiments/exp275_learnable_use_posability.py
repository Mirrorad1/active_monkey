"""Exp 275 — learning-guided-evolution RUNG 1: is a same-axis theta-Baldwin test POSABLE?

DIRECTION: loop/directions/learning-guided-evolution.md (rung 1, reframed per L28/L45 as a
CHEAP posability pre-flight BEFORE building any within-life `theta` learner).

PLAIN. The program keeps hitting a wall: a costed sense is useful when gifted but never
evolves. The classical escape is the Baldwin effect — if a creature LEARNS to use the sense
within life, selection can assimilate that use into the genome. But this substrate already
has within-life learning (the EMA resource map `m`) and the arc found it SUBSTITUTES for the
sensor (Exp 201/202), not guides it. Before building a learnable "ability to use the sensor"
(theta, the reserved sense_axis controller trait), we check the PRECONDITION: (1) is the
LIVE-SENSOR channel separably load-bearing — does a sensor forager track the drifting food
band better than a sensorless (wander/map) one, and can map-memory NOT substitute; (2) does
"HOW WELL you use the sensor" (theta) have an expressible bad->good range for a learner to
climb and selection to assimilate?

INSTRUMENT (the honest metric — see the calibration record in the entry). Intake / carrying-
capacity metrics are CONFOUNDED here: the forage regime is demographically unstable
(overshoot->crash, extinction), so per-capita intake conflates foraging SKILL with survival
trajectory. The demographic-confound-free measure of the sensor's CORE function is IN-BAND
OCCUPANCY: the fraction of alive creatures sitting on the drifting food band (kin Exp 210
wrong-cell occupancy), read over an early window from a fixed cohort in a mild regime.

theta = band_responsiveness (the Exp 201 tracker's EMA rate). The GOOD sensor forager uses
the band-staleness tracker branch: its band estimate updates at alpha = intensity *
band_responsiveness, so band_responsiveness in [0,1] IS "how well it uses the sensor to track
the moving band" — a FAITHFUL, non-scale-invariant theta proxy (unlike thermal_avoidance_weight,
which is argmax-scale-invariant and gives an artifactual flat sweep — the first draft's trap).

DESIGN (no engine surgery — existing knobs only). Mild thermosense-FORAGE regime, COST-OFF
(enable_thermosense=False -> the organ is free; isolates the INFORMATION channel), enable_
band_staleness=True, regen 0.14 / food_concentration 6.0 (mild enough that the BLIND arm mostly
survives -> valid occupancy). Fixed cohort (initial_population 80), early window (100,500),
per step measure fraction of alive creatures with |temperature[pos] - current_food_optimal|
<= band_width. Factorial + theta sweep:
  SENSOR:  GOOD  (h=0.60, theta=band_responsiveness=0.50 default -> tracks via branch 4)
           BLIND (h=0.00 -> use_thermo FALSE -> wander/map only, branch 2)
  MAP:     ON (learning_rate=0.30) / OFF (learning_rate=0.00), freeze_learning_rate=True
  DRIFT:   FAST (period 150) / SLOW (period 600), amplitude 0.35, band_width 0.06
  THETA sweep (GOOD/OFF/FAST): band_responsiveness in {0,0.02,0.05,0.1,0.25,0.5,1.0}.
Seeds 101-108 (fresh, 8).

HYPOTHESIS. The live-sensor channel is separably load-bearing for band-tracking (sensor > blind,
most under fast drift; map can't substitute), AND theta (band_responsiveness) has a real bad->good
expressible range that gates tracking/fitness — so a learnable theta has room to climb and
something for selection to assimilate.

PREDICTION (POSABLE) — property-level, per-seed paired where applicable, 8 seeds (occ in [0,1]):
  S1 (sensor separably tracks, FAST, map OFF): occ[GOOD,OFF,FAST] >= occ[BLIND,OFF,FAST] + 0.10
     (abs) in >=6/8 valid seeds AND median ratio >= 1.25.
  S2 (map cannot substitute under FAST drift), both in >=6/8 valid seeds:
     (a) occ[BLIND,ON,FAST] <= occ[BLIND,OFF,FAST] + 0.05;  (b) occ[GOOD,OFF,FAST] >=
     occ[GOOD,ON,FAST] - 0.05.
  S3 (theta has an expressible bad->good range): over the band_responsiveness sweep on
     GOOD/OFF/FAST, max-min mean-occ >= 0.08 (with the fitness link reported: bad theta ->
     higher extinction, good theta -> survival).
  VERDICT POSABLE iff S1 AND S2 AND S3 -> the sensor channel is separably load-bearing and
  theta has a use-range that gates fitness -> Exp 276 builds the learnable+heritable theta.

FALSIFIER (CAN'T-POSE) — any one of: ~S1 sensor tracks no better than sensorless wandering;
~S2 map-memory tracks as well as the sensor; ~S3 tracking is flat across the FAITHFUL theta
proxy (no bad->good range). Any of these -> a same-axis theta-Baldwin test is not posable here.

BINDING CAVEATS (must ride the verdict). (1) Occupancy is BEHAVIORAL; the fitness link is shown
via the theta->extinction gradient, but the regime sits at the EDGE of demographic stability
(extreme theta values cause extinction) — so Exp 276 must keep the theta gradient inside the
stable band (the Exp 242-247 stability-vs-selection tension is present but navigable at
theta~0.25-0.5). (2) Cost is OFF here (info-channel isolation); it re-enters in Exp 276.
(3) band_responsiveness is ONE parameterization of theta (a tracking-rate); Exp 276's learnable
theta should be this or a close analog.

NOTE (L1/L2): the PRINTED verdict is this script's CLAIM. The EXPERIMENTS.md verdict is applied
by the controller + the blinded verifier to the committed raw numbers, conjunct by conjunct.
"""
from __future__ import annotations

import dataclasses as D
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ecology.engine import Ecology
from ecology.scenarios import SCENARIOS, FOUNDER
from ecology.sense_axis import clamp_founder

HORIZON = 500
WIN = (100, 500)
SEEDS = tuple(range(101, 109))          # 8 fresh seeds
INIT_POP = 80
GOOD_H, BLIND_H = 0.60, 0.00
LR_ON, LR_OFF = 0.30, 0.00
BAND_WIDTH = 0.06
THETA_GOOD = 0.50                       # default good theta (interior optimum) for the factorial
THETA_SWEEP = (0.0, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0)
REGEN = 0.14
DRIFTS = {"FAST": 150.0, "SLOW": 600.0}
SENSORS = {"GOOD": GOOD_H, "BLIND": BLIND_H}
MAPS = {"ON": LR_ON, "OFF": LR_OFF}


def _cfg(sensor_h: float, lr: float, period: float, resp: float):
    base = SCENARIOS["balanced"]
    cfg = D.replace(
        base,
        horizon=HORIZON, initial_population=INIT_POP, max_population=4000,
        enable_thermosense=False,            # COST OFF (organ free; h breeds true)
        enable_temperature=True, temperature_stress_scale=0.0,
        thermosense_active_threshold=0.05, thermosense_noise_base=0.5, thermal_avoidance_weight=4.0,
        enable_food_coupling=True, thermosense_forage_mode=True,
        enable_band_staleness=True, band_responsiveness=float(resp),   # theta = tracking-use quality
        food_optimal_base=0.5, food_optimal_amplitude=0.35, food_optimal_period=float(period),
        food_band_width=BAND_WIDTH, food_concentration=6.0,
        regen_rate=REGEN, freeze_learning_rate=True,
    )
    f = D.replace(FOUNDER, learning_rate=float(lr))
    return D.replace(cfg, founder=clamp_founder(f, sensor_h, 0.20), founder_mix=None)


def run_occupancy(job: dict) -> dict:
    cfg = _cfg(job["sensor_h"], job["lr"], job["period"], job["resp"])
    eco = Ecology(cfg, seed=job["seed"])
    w = eco.world
    fr: list[float] = []
    while eco.t < HORIZON and not eco.exploded:
        eco.step()
        if WIN[0] <= eco.t <= WIN[1]:
            al = eco.alive_snapshot()
            if al:
                c0 = w.current_food_optimal
                on = sum(1 for c in al if abs(float(w.temperature[c.phenotype.pos]) - c0) <= BAND_WIDTH)
                fr.append(on / len(al))
        if not eco.has_alive():
            break
    return {"key": job["key"], "occ": (float(np.mean(fr)) if fr else float("nan")),
            "extinct": eco.alive_count() == 0}


def _batch(jobs: list[dict]) -> dict:
    out: dict = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(run_occupancy, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            out[r["key"]] = r
    return out


def _paired(occ, cell_a, cell_b, delta_abs, cmp):
    ok = valid = 0
    ratios = []
    per = {}
    for s in SEEDS:
        a, b = occ.get((*cell_a, s)), occ.get((*cell_b, s))
        if a is None or b is None or a != a or b != b:
            per[s] = None
            continue
        valid += 1
        if b > 0:
            ratios.append(a / b)
        per[s] = round(a - b, 4)
        ok += int((a >= b + delta_abs) if cmp == ">=" else (a <= b + delta_abs))
    return {"n_ok": ok, "n_valid": valid, "per_seed_diff": per,
            "median_ratio": (float(np.median(ratios)) if ratios else float("nan"))}


def main() -> None:
    jobs = []
    for (sn, h), (mn, lr), (dn, per) in itertools.product(SENSORS.items(), MAPS.items(), DRIFTS.items()):
        for s in SEEDS:
            jobs.append({"sensor_h": h, "lr": lr, "period": per, "resp": THETA_GOOD,
                         "seed": s, "key": (sn, mn, dn, s)})
    for rp in THETA_SWEEP:
        for s in SEEDS:
            jobs.append({"sensor_h": GOOD_H, "lr": LR_OFF, "period": DRIFTS["FAST"], "resp": rp,
                         "seed": s, "key": ("THETA", f"r{rp:g}", "FAST", s)})

    res = _batch(jobs)
    occ = {k: v["occ"] for k, v in res.items()}

    print("=== CELL MEANS in-band occupancy (mean, sd, n_valid, n_extinct) ===")
    for sn in SENSORS:
        for mn in MAPS:
            for dn in DRIFTS:
                vals = [occ.get((sn, mn, dn, s)) for s in SEEDS]
                ext = sum(1 for s in SEEDS if res[(sn, mn, dn, s)]["extinct"])
                vv = [v for v in vals if v is not None and v == v]
                mu = float(np.mean(vv)) if vv else float("nan")
                sd = float(np.std(vv)) if vv else float("nan")
                print(f"  {sn:5s} {mn:3s} {dn:4s}: occ={mu:.4f} sd={sd:.4f} n={len(vv)} extinct={ext}")

    print("\n=== THETA sweep (GOOD/OFF/FAST): band_responsiveness -> occupancy + extinction ===")
    theta_means = {}
    for rp in THETA_SWEEP:
        vv = [occ.get(("THETA", f"r{rp:g}", "FAST", s)) for s in SEEDS]
        ext = sum(1 for s in SEEDS if res[("THETA", f"r{rp:g}", "FAST", s)]["extinct"])
        vv = [v for v in vv if v is not None and v == v]
        theta_means[rp] = float(np.mean(vv)) if vv else float("nan")
        print(f"  resp={rp:5g}: occ={theta_means[rp]:.4f}  extinct={ext}/8")
    # L21: exclude NaN (fully-extinct) theta cells from the range; the extinction gradient
    # (bad theta -> total extinction, printed above) is the FITNESS link, reported separately.
    valid_theta = [v for v in theta_means.values() if v == v]
    theta_spread = (max(valid_theta) - min(valid_theta)) if len(valid_theta) >= 2 else float("nan")

    S1 = _paired(occ, ("GOOD", "OFF", "FAST"), ("BLIND", "OFF", "FAST"), 0.10, ">=")
    S2a = _paired(occ, ("BLIND", "ON", "FAST"), ("BLIND", "OFF", "FAST"), 0.05, "<=")
    S2b = _paired(occ, ("GOOD", "OFF", "FAST"), ("GOOD", "ON", "FAST"), -0.05, ">=")
    s1_ok = S1["n_ok"] >= 6 and S1["median_ratio"] >= 1.25
    s2_ok = S2a["n_ok"] >= 6 and S2b["n_ok"] >= 6
    s3_ok = theta_spread >= 0.08
    posable = s1_ok and s2_ok and s3_ok

    print("\n=== PREDECLARED SIGNATURES (script CLAIM — verifier recomputes) ===")
    print(f"  S1 sensor-tracks    GOOD/OFF/FAST >= BLIND/OFF/FAST +0.10 & >=1.25x : {S1} -> {s1_ok}")
    print(f"  S2a map-no-subst    BLIND/ON/FAST <= BLIND/OFF/FAST +0.05           : {S2a} -> {S2a['n_ok']>=6}")
    print(f"  S2b sensor-not-lean GOOD/OFF/FAST >= GOOD/ON/FAST -0.05             : {S2b} -> {S2b['n_ok']>=6}")
    print(f"  S3 theta-has-range  band_responsiveness occ spread >= 0.08          : spread={theta_spread:.4f} -> {s3_ok}")
    verdict = "POSABLE" if posable else "CANT_POSE"
    print(f"\n[CLAIM] VERDICT = {verdict}  (S1={s1_ok} S2={s2_ok} S3={s3_ok})")
    print("[CAVEAT] behavioral+survival posability; cost-off; edge-of-stability; one theta parameterization.")

    print("\n=== JSON ===")
    print(json.dumps({
        "seeds": list(SEEDS),
        "cell_occ": {f"{sn}/{mn}/{dn}": float(np.nanmean([occ.get((sn, mn, dn, s), np.nan) for s in SEEDS]))
                     for sn in SENSORS for mn in MAPS for dn in DRIFTS},
        "theta_means": {f"r{k:g}": v for k, v in theta_means.items()},
        "theta_spread": theta_spread,
        "S1": S1, "S2a": S2a, "S2b": S2b, "claim_verdict": verdict,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
