"""Exp 202 — N5: the INTERFERENCE-COMPETITION escape — does REAL Red-Queen competition cross the valley?
(pre-registered in loop/directions/population-ecology.md BEFORE any verdict-seed data).

Hypothesis (the LAST untested sensing escape, the human's pick): a regime with REAL positive frequency
dependence — precision pays MORE as the trait spreads — via INTERFERENCE COMPETITION at a GENUINELY
DEPLETING food band, with the ascending-creature_id "eat-first" confound NEUTRALISED by a gated SHUFFLED
processing order (shuffle_creature_order; consume() UNCHANGED). If real competition makes precision pay,
a FUNCTIONAL thermosense organ evolves de-novo (gene-pool newborn mean intensity > 0.30 in COMPETE).

Mechanism (PROVIDED, gated engine; exp194-201 byte-identical, hash-verified): shuffle_creature_order
randomises the per-step processing order (rng.shuffle, ON-branch only) so a contested cell is won by
navigation skill, not birth order — the ONE id-order fix that survives the no-hidden-evaluator invariant
(it never touches _step_one_creature; consume() is unchanged). track_band_strip is a gated validity
auditor (NO rng, NOT in events_hash) confirming the band is GENUINELY depleted within a step (exp201
failed here, strip~0). Designed via an independent Codex diagnosis + a Claude design workflow that BOTH
converged on shuffle and BOTH rejected the alternatives (precision-weighted split = cross-creature
evaluator; precision-ordering = forbidden ranking; mixed founders = re-correlates id mid-run).

Arms (temp ON, stress 0, cheap efficient organ inefficiency 0.20/floor 0, founder intensity 0.10,
forage mode, conc 14, band 0.08, period 1500, amp 0.3; horizon 12000; fresh seeds {38,39,40,41,42}):
  COMPETE    : depleting band (regen 0.08) + shuffle ON + strip auditor — the primary de-novo CLIMB test.
  NO_SHUFFLE : = COMPETE but shuffle OFF (id-order ON) — isolates whether the fair queue matters at all.
  CLAMPED_LR : = COMPETE but learning_rate FROZEN — kills the resource-memory substitution confound.
  USELESS    : enable_food_coupling=False (organ pure cost) — the primitive anchor (~0.05).
  (DISCLOSED pre-verdict amendment: an ABUNDANT regen-0.8 no-competition arm was DROPPED — a no-scarcity
   regime has nothing to bound its population, so it grew toward the runaway cap (~6000): the runtime
   bottleneck AND an explosion-invalidity risk. Its no-selection-without-competition role is covered by
   USELESS. Amendment committed BEFORE the verdict run; no verdict data existed at amendment time.)
Metric = NEWBORN (gene-pool) mean thermosense_intensity, end = born in [10000,12000]; L21 validity.

Disclosed pilots (per L7): seeds {130,131,140-147}. The STRIP-GATE PASSES (strip_frac 100%, ~170
contested in-band occupants — genuine depletion, UNLIKE exp201's strip~0), so the competition substrate
is REAL. BUT the de-novo climb DECAYS to ~0.03 (below the 0.10 founder) across regimes; a lone v_narrow
seed hit 0.736 but at a COLLAPSED pop (101), and an 8-seed characterisation gave 0/8 functional with
corr(pop, intensity) = -0.92 (high intensity ONLY when the population collapses = DRIFT, not selection).
Pilots SUGGEST NEGATIVE. Regime FIXED before the verdict run.

Predictions (5 fresh seeds): P1 determinism; P2 validity (COMPETE valid >=4/5 AND strip_frac>0.5 =
genuine competition); P3 (CORE, => POSITIVE): COMPETE end mean > 0.30 in >=4/5 valid seeds AT HEALTHY
populations (a seed with final_pop < 300 is DRIFT-FLAGGED, not a clean selection datum) AND CLAMPED_LR
also > 0.30 in >=4/5.

Falsifiers: PRIMARY -> NEGATIVE (the MAXIMALLY general wall): COMPETE end mean < 0.15 in a majority of
valid seeds -> the primitive ceiling survives even REAL interference competition with id-order neutralised.
MIXED: COMPETE in [0.15, 0.30]. DRIFT FALSIFIER (=> a POSITIVE is DISCARDED as drift): functional COMPETE
seeds coincide with collapsed populations (corr(pop, intensity) strongly negative; functional only at
pop < 300). VALIDITY: if COMPETE strip_frac <= 0.5 the band did not deplete -> the escape was never tested
-> report INERT (pause-and-consult), not NEGATIVE. F1 non-determinism -> NEGATIVE (infra). Regression:
shuffle_creature_order=False must reproduce committed exp194 + exp200 hashes else NEGATIVE (infra).

Verdict rule: NEGATIVE if not P1; INERT if COMPETE strip_frac<=0.5; else POSITIVE iff P3 and not drift;
else NEGATIVE iff COMPETE majority-primitive (<0.15); else MIXED. Repo token POSITIVE/NEGATIVE/MIXED.
Diagnostics (NON-gating): corr(pop, intensity) per arm (drift indicator); per-arm mean newborn
learning_rate; max individual intensity; the frequency-dependence probe (exp202_freqdep_probe.py:
advantage(p) vs resident r, run + committed BEFORE the verdict). Predicting NEGATIVE (pilots); the human's
interference-competition hypothesis tested on FRESH seeds. Founders + costs + scheduler PROVIDED.
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

from ecology.batch import RunSpec, run_batch
from ecology.engine import EcologyConfig
from ecology.scenarios import SCENARIOS, FOUNDER

# FIXED constants (pre-registered) -------------------------------------------
SEEDS = [38, 39, 40, 41, 42]
HORIZON = 12000
WINDOW_START = 10000
CHECKPOINT_STRIDE = 1000
MIN_VALID_POP = 10        # L21 measurable cohort
HEALTHY_POP = 300         # below this, a functional reading is DRIFT-flagged
CONC, BAND, PERIOD, AMP = 14.0, 0.08, 1500.0, 0.3
DEPLETE_REGEN, ABUNDANT_REGEN = 0.08, 0.8

BASE = dict(
    enable_thermosense=True, enable_temperature=True, temperature_stress_scale=0.0,
    tolerance_cost_scale=0.0, comfort_amplitude=0.0, thermosense_upkeep_floor=0.0,
    thermosense_active_threshold=0.05, thermosense_noise_base=0.5, thermal_avoidance_weight=4.0,
    food_optimal_base=0.5, food_optimal_amplitude=AMP, food_optimal_period=PERIOD,
    food_concentration=CONC, food_band_width=BAND,
)


def founder() -> Any:
    return D.replace(FOUNDER, thermosense_intensity=0.10, thermosense_inefficiency=0.2,
                     temperature_tolerance=0.10)


def _cfg(*, regen, shuffle, coupling=True, freeze_lr=False, track=False) -> EcologyConfig:
    return D.replace(
        SCENARIOS["balanced"], horizon=HORIZON, max_population=6000, founder=founder(),
        regen_rate=regen, enable_food_coupling=coupling, thermosense_forage_mode=coupling,
        shuffle_creature_order=shuffle, track_band_strip=track, freeze_learning_rate=freeze_lr, **BASE,
    )


# ABUNDANT (regen 0.8 no-competition baseline) was DROPPED in a disclosed pre-verdict
# amendment: a no-scarcity regime has nothing to bound its population, so it grew toward
# the runaway cap (pop ~6000) — both the runtime bottleneck and an explosion-invalidity
# risk. Its "no-selection without competition" role is covered by USELESS (organ pure cost).
_ARMS: dict[str, EcologyConfig] = {
    "COMPETE":    _cfg(regen=DEPLETE_REGEN,  shuffle=True,  track=True),
    "NO_SHUFFLE": _cfg(regen=DEPLETE_REGEN,  shuffle=False, track=True),
    "CLAMPED_LR": _cfg(regen=DEPLETE_REGEN,  shuffle=True,  track=True, freeze_lr=True),
    "USELESS":    _cfg(regen=DEPLETE_REGEN,  shuffle=True,  coupling=False),
}
ARMS = list(_ARMS)


def make_arm_cfg(arm: str) -> EcologyConfig:
    return _ARMS[arm]


def _specs():
    return [RunSpec(key=(a, s), cfg=_ARMS[a], seed=s, window_start=WINDOW_START,
                    checkpoint_stride=CHECKPOINT_STRIDE,
                    trait_means=("thermosense_intensity", "learning_rate"))
            for a in ARMS for s in SEEDS]


def _arm_mean(vals):
    v = [x for x in vals if x is not None and not math.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def compute_verdict(res) -> dict[str, Any]:
    def end(a, s):
        r = res[(a, s)]
        return r["end_means"]["thermosense_intensity"] if r["valid"] else None

    nb = {a: {s: end(a, s) for s in SEEDS} for a in ARMS}
    pops = {a: {s: res[(a, s)]["final_pop"] for s in SEEDS} for a in ARMS}
    means = {a: _arm_mean(list(nb[a].values())) for a in ARMS}

    # P1 determinism: rerun COMPETE+USELESS seed 38, compare hashes
    rer = run_batch([RunSpec(key=(a, 38), cfg=_ARMS[a], seed=38, window_start=WINDOW_START,
                             checkpoint_stride=CHECKPOINT_STRIDE,
                             trait_means=("thermosense_intensity",)) for a in ("COMPETE", "USELESS")],
                    sequential=True)
    p1 = all(rer[(a, 38)]["events_hash"] == res[(a, 38)]["events_hash"] for a in ("COMPETE", "USELESS"))

    n_valid = sum(nb["COMPETE"][s] is not None for s in SEEDS)
    strip = _arm_mean([res[("COMPETE", s)].get("strip_late_frac_pos") for s in SEEDS])
    P2 = n_valid >= 4 and strip > 0.5

    # drift correlation pop vs intensity in COMPETE (valid seeds)
    xs = [(pops["COMPETE"][s], nb["COMPETE"][s]) for s in SEEDS if nb["COMPETE"][s] is not None]
    drift_corr = float(np.corrcoef([p for p, _ in xs], [n for _, n in xs])[0, 1]) if len(xs) >= 3 else float("nan")

    def func_healthy(a):  # functional AND at a healthy (non-collapsed) population
        return sum(nb[a][s] is not None and nb[a][s] > 0.30 and pops[a][s] >= HEALTHY_POP for s in SEEDS)

    P3 = func_healthy("COMPETE") >= 4 and func_healthy("CLAMPED_LR") >= 4
    drift = (func_healthy("COMPETE") < sum(nb["COMPETE"][s] is not None and nb["COMPETE"][s] > 0.30 for s in SEEDS))
    primitive = sum(nb["COMPETE"][s] is not None and nb["COMPETE"][s] < 0.15 for s in SEEDS) >= 3

    if not p1:
        verdict, token = "NEGATIVE (F1 non-determinism, infra)", "NEGATIVE"
    elif strip <= 0.5:
        verdict, token = "INERT (band did not deplete; escape never tested -> pause-and-consult)", "MIXED"
    elif P3 and not drift:
        verdict, token = "POSITIVE (interference competition crosses the valley; functional organ)", "POSITIVE"
    elif primitive:
        verdict, token = "NEGATIVE (ceiling general even under real interference competition)", "NEGATIVE"
    else:
        verdict, token = "MIXED (climbed above primitive, not functional)", "MIXED"

    return {"nb": nb, "pops": pops, "means": means, "P1": p1, "P2": P2, "P3": P3,
            "strip_frac": strip, "drift_corr": drift_corr, "drift": drift, "primitive": primitive,
            "n_valid": n_valid, "verdict": verdict, "token": token}


def main() -> None:
    t0 = time.time()
    out_dir = _REPO / "experiments" / "outputs" / "exp202_n5_interference_competition"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exp 202: {len(ARMS)} arms x {len(SEEDS)} seeds (horizon {HORIZON}, parallel) ...")
    res = run_batch(_specs())
    v = compute_verdict(res)

    L = ["=" * 76, "EXP 202 — N5: INTERFERENCE-COMPETITION ESCAPE (shuffled fair queue) — SUMMARY", "=" * 76, ""]
    L.append(f"{'Arm':<11}" + "".join(f"  seed{s}" for s in SEEDS) + f"  {'mean':>8}  {'strip%':>6}  {'maxpop':>6}")
    L.append("-" * 76)
    for a in ARMS:
        per = "".join(f"  {'EXT':>7}" if v["nb"][a][s] is None else f"  {v['nb'][a][s]:>7.4f}" for s in SEEDS)
        strip_a = _arm_mean([res[(a, s)].get("strip_late_frac_pos") for s in SEEDS])
        maxpop = max(v["pops"][a][s] for s in SEEDS)
        L.append(f"{a:<11}{per}  {v['means'][a]:>8.4f}  {strip_a*100:>5.0f}%  {maxpop:>6}")
    L += ["", "FINAL POPULATIONS:", f"{'Arm':<11}" + "".join(f"  seed{s}" for s in SEEDS)]
    for a in ARMS:
        L.append(f"{a:<11}" + "".join(f"  {v['pops'][a][s]:>6}" for s in SEEDS))
    L += ["", "EVALUATIONS:",
          f"  P1 determinism: {'PASS' if v['P1'] else 'FAIL'}",
          f"  P2 validity (>=4/5 + strip>0.5): {'PASS' if v['P2'] else 'FAIL'}  (COMPETE strip_frac={v['strip_frac']:.2f})",
          f"  COMPETE mean={v['means']['COMPETE']:.4f}  drift corr(pop,intensity)={v['drift_corr']:+.2f}",
          f"  P3 functional-healthy (>0.30 at pop>=300 in >=4/5): {'YES' if v['P3'] else 'NO'}",
          f"  drift-flagged: {'YES' if v['drift'] else 'NO'}", "",
          f"VERDICT: {v['verdict']} (repo token: {v['token']})", "", f"runtime: {time.time()-t0:.0f}s"]
    text = "\n".join(L)
    print("\n" + text)
    (_REPO / "experiments" / "outputs" / "exp202.txt").write_text(text + "\n")

    # SLIM outputs (loop/EFFICIENCY.md): verdict.json + ONE trajectories.json + .txt + script.
    per_arm_seed = {a: {str(s): {k: res[(a, s)][k] for k in
                   ("valid", "final_pop", "extinct", "max_ever", "steps_run", "events_hash")}
                   | {"end_newborn_intensity": res[(a, s)]["end_means"]["thermosense_intensity"],
                      "end_newborn_learning_rate": res[(a, s)]["end_means"]["learning_rate"],
                      "strip_late_frac_pos": res[(a, s)].get("strip_late_frac_pos"),
                      "occ_late_mean": res[(a, s)].get("occ_late_mean")} for s in SEEDS} for a in ARMS}
    (out_dir / "verdict.json").write_text(json.dumps({
        "experiment": "exp202", "seeds": SEEDS, "per_arm_seed": per_arm_seed,
        "means": v["means"], "strip_frac": v["strip_frac"], "drift_corr": v["drift_corr"],
        "verdict": v["verdict"], "token": v["token"],
        "predictions": {k: v[k] for k in ("P1", "P2", "P3", "drift", "primitive")},
    }, indent=2))
    (out_dir / "trajectories.json").write_text(json.dumps(
        {f"{a}_seed{s}": res[(a, s)]["trajectory"] for a in ARMS for s in SEEDS}, indent=2))
    print(f"[saved {out_dir}/verdict.json + trajectories.json]")


if __name__ == "__main__":
    main()
