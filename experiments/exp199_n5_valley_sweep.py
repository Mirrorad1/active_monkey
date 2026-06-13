"""Exp 199 — N5: is the primitive-sensor ceiling FUNDAMENTAL? the fitness-valley sweep
(pre-registered in loop/directions/population-ecology.md, commit 879f60e, BEFORE any data).

Hypothesis: the primitive-sensor ceiling (Exp 197/198: thermosense pinned at ~0.045) is FUNDAMENTAL:
even under aggressively shallow-valley regimes (low sensing noise, cheap efficient organ, strong
steering, harsh stress), thermosense does NOT evolve into a functional organ - the gene-pool
equilibrium rises only weakly as the valley shallows but stays primitive (<0.15), AND a seeded-
functional organ (0.50) is NOT retained (it decays toward primitive or drives extinction) - because
the benefit SATURATES (a little sensing suffices to reach safety) while the cost keeps rising with
intensity, so the functional state is selected against from both directions.

Arms (all temp ON, founder tolerance 0.10, CHEAP efficient organ inefficiency 0.20, upkeep_floor 0.0,
thermal_avoidance_weight 8.0, temperature_stress_scale 3.0, comfort amp 0.4/period 1500,
active_threshold 0.05; horizon 12000; fresh seeds {18,19,20,21,22}):
  NOISE SWEEP (founder 0.10, can a primitive organ CLIMB?): V1 noise 0.20, V2 0.10, V3 0.05, V4 0.02.
  RETENTION (founder 0.50, is a functional organ STABLE?): F noise 0.02 (best case).
Metric = NEWBORN (gene-pool) mean thermosense intensity, end = mean over creatures born in [10000,12000].

Pilots (disclosed, per L7): pilots {100,101} (deleted) - noise sweep moved the mean only 0.05->0.067
(never functional; max individual ~0.31); a seeded 0.50 organ DECAYED (0.41->0.10) or went EXTINCT.
Params/thresholds FIXED; final on FRESH seeds {18-22}.

Predictions if TRUE (5 fresh seeds; L21 validity = reached horizon with pop>=10 and a newborn cohort
in the window; extinct/collapsed arms INVALID for the metric, reported as extinctions):
  P1 determinism: same seed -> identical event hash (V4 + F on seed 18).
  P2 validity: noise-sweep arms V1-V4 (founder 0.10) persist measurably in >=4/5 seeds.
  P3 (CORE) no climbing: across V1-V4, every measurable end gene-pool mean intensity < 0.15, each V arm
     measurable in >=4/5 seeds.
  P4 (CORE) no retention: arm F end intensity < 0.20 in its measurable runs OR F extinct in >=3/5 seeds.
  P5 (supporting) monotone-but-capped: V4 mean >= V1 mean but all V < 0.15; F extinction > V extinction.

Falsifiers (-> POSITIVE, valley crossable): FP1 any V arm end mean intensity > 0.30 in >=4/5 seeds.
FP2 arm F retains end intensity > 0.30 in >=4/5 measurable seeds. F1 non-determinism -> NEGATIVE.

Verdict rule: NEGATIVE (= the finding: ceiling fundamental) iff P3 AND P4 and neither FP1 nor FP2;
POSITIVE iff FP1 or FP2; MIXED if ambiguous. Repo token POSITIVE/NEGATIVE/MIXED. NEGATIVE here is the
scientifically interesting outcome (a structural ceiling), not a failure. Base regime deliberately
FAVORABLE to a functional organ, so NEGATIVE is the STRONG conclusion. Founders + costs PROVIDED.
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

# Ensure repo root is on sys.path when script is run directly.
_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ecology.engine import Ecology, EcologyConfig
from ecology.scenarios import SCENARIOS, FOUNDER

# ---------------------------------------------------------------------------
# FIXED experiment constants (pre-registered; do NOT change)
# ---------------------------------------------------------------------------
SEEDS = [18, 19, 20, 21, 22]
HORIZON = 12000
MAX_POP = 5000
CHECKPOINT_STRIDE = 1000
NEWBORN_WINDOW_START = 10000
NEWBORN_WINDOW_END = HORIZON
ACTIVE_THRESHOLD = 0.05
MIN_VALID_POP = 10   # L21: arm/seed is VALID iff final_pop >= MIN_VALID_POP

BASE = dict(
    enable_thermosense=True,
    enable_temperature=True,
    tolerance_cost_scale=1.5,
    comfort_amplitude=0.4,
    comfort_period=1500.0,
    temperature_comfort=0.5,
    thermosense_upkeep_floor=0.0,
    thermosense_active_threshold=0.05,
    thermal_avoidance_weight=8.0,
    temperature_stress_scale=3.0,
)

# ARM definitions (noise, founder_intensity)
ARMS: dict[str, tuple[float, float]] = {
    "V1": (0.20, 0.10),
    "V2": (0.10, 0.10),
    "V3": (0.05, 0.10),
    "V4": (0.02, 0.10),
    "F":  (0.02, 0.50),
}


# ---------------------------------------------------------------------------
# Founder + config factories (importable for unit tests)
# ---------------------------------------------------------------------------

def make_founder(intensity: float) -> Any:
    """Return a founder with given thermosense_intensity; inefficiency=0.20, tolerance=0.10."""
    return D.replace(
        FOUNDER,
        thermosense_intensity=intensity,
        thermosense_inefficiency=0.2,
        temperature_tolerance=0.10,
    )


def make_cfg(noise: float, intensity: float) -> EcologyConfig:
    """Return EcologyConfig for arm with given noise and founder intensity."""
    return D.replace(
        SCENARIOS["balanced"],
        horizon=HORIZON,
        max_population=MAX_POP,
        founder=make_founder(intensity),
        thermosense_noise_base=noise,
        **BASE,
    )


def make_arm_cfg(arm: str) -> EcologyConfig:
    """Return EcologyConfig for the named arm (V1/V2/V3/V4/F)."""
    noise, intensity = ARMS[arm]
    return make_cfg(noise, intensity)


# ---------------------------------------------------------------------------
# Newborn window metric (importable for unit tests)
# ---------------------------------------------------------------------------

def newborn_mean_intensity_in_window(
    creatures: list,
    window_start: int,
    window_end: int,
) -> float:
    """Return mean thermosense_intensity for non-founder creatures with
    birth_t in [window_start, window_end].

    Returns NaN if no newborns in the window.
    This is DISTINCT from a living snapshot — it reads the birth cohort
    from the gene pool regardless of whether creatures are still alive.
    Founders (parent_id is None) are excluded.
    """
    newborns = [
        c for c in creatures
        if c.parent_id is not None
        and window_start <= c.phenotype.birth_t <= window_end
    ]
    if not newborns:
        return float("nan")
    intensities = [c.genotype.thermosense_intensity for c in newborns]
    return float(np.mean(intensities))


# ---------------------------------------------------------------------------
# L21 validity checker
# ---------------------------------------------------------------------------

def is_valid_run(eco: Ecology, end_nb_intensity: float) -> bool:
    """L21-compliant validity: reached t=12000 AND final pop >= 10 AND >= 1 newborn in window.

    Returns True iff the arm/seed is measurable (produces a valid end metric).
    """
    alive = eco._alive()
    reached_horizon = eco.t >= HORIZON and not eco.exploded
    pop_ok = len(alive) >= MIN_VALID_POP
    newborn_ok = not math.isnan(end_nb_intensity)
    return reached_horizon and pop_ok and newborn_ok


# ---------------------------------------------------------------------------
# Core runner: one arm x one seed, with checkpoint sampling
# ---------------------------------------------------------------------------

def run_arm(cfg: EcologyConfig, seed: int) -> dict[str, Any]:
    """Run one arm/seed; manual stepping; return trajectory + summary.

    Per the spec: step manually (while eco.t<12000 and not eco.exploded: eco.step()),
    sampling every CHECKPOINT_STRIDE steps.
    Do NOT write events.jsonl.
    """
    eco = Ecology(cfg, seed=seed)

    trajectory: list[dict[str, Any]] = []
    checkpoints = list(range(CHECKPOINT_STRIDE, HORIZON + 1, CHECKPOINT_STRIDE))

    while eco.t < HORIZON and not eco.exploded:
        eco.step()

        if eco.t in checkpoints:
            alive = eco._alive()
            # Sample over the most recent CHECKPOINT_STRIDE window
            window_start = eco.t - CHECKPOINT_STRIDE
            window_end = eco.t
            nb_intensity = newborn_mean_intensity_in_window(
                eco._creatures, window_start, window_end
            )
            # Max individual intensity among living creatures
            max_intensity = max(
                (c.genotype.thermosense_intensity for c in alive),
                default=float("nan"),
            )
            trajectory.append({
                "t": eco.t,
                "pop": len(alive),
                "newborn_mean_intensity": float("nan") if math.isnan(nb_intensity) else nb_intensity,
                "max_intensity": float("nan") if math.isnan(max_intensity) else max_intensity,
            })

        if not eco._alive():
            break

    alive = eco._alive()

    # End metric: newborn mean intensity for creatures born in [NEWBORN_WINDOW_START, HORIZON]
    end_nb_intensity = newborn_mean_intensity_in_window(
        eco._creatures, NEWBORN_WINDOW_START, NEWBORN_WINDOW_END
    )

    # Max individual intensity ever seen (across all ever-born creatures)
    all_intensities = [c.genotype.thermosense_intensity for c in eco._creatures]
    max_ever_intensity = max(all_intensities) if all_intensities else float("nan")

    valid = is_valid_run(eco, end_nb_intensity)

    # Extinction step: step at which population first dropped to 0 (None if not extinct)
    extinction_step: int | None = None
    if len(alive) == 0:
        # Approximate: record eco.t as the step where it ended
        extinction_step = eco.t

    summary = {
        "seed": seed,
        "steps_run": eco.t,
        "reached_horizon": eco.t >= HORIZON and not eco.exploded,
        "extinct": len(alive) == 0,
        "exploded": eco.exploded,
        "final_pop": len(alive),
        "end_newborn_intensity": end_nb_intensity,
        "max_ever_intensity": max_ever_intensity,
        "valid": valid,
        "extinction_step": extinction_step,
        "events_hash": eco.events_hash(),
    }
    return {"trajectory": trajectory, "summary": summary}


# ---------------------------------------------------------------------------
# Determinism check (P1): rerun seed 18 for arms V4 and F
# ---------------------------------------------------------------------------

def check_p1_determinism(
    hash_V4_seed18: str,
    hash_F_seed18: str,
) -> bool:
    """Rerun seed 18 for arms V4 and F; confirm event hashes match."""
    r2_V4 = run_arm(make_arm_cfg("V4"), seed=18)
    r2_F = run_arm(make_arm_cfg("F"), seed=18)
    match_V4 = r2_V4["summary"]["events_hash"] == hash_V4_seed18
    match_F = r2_F["summary"]["events_hash"] == hash_F_seed18
    return match_V4 and match_F


# ---------------------------------------------------------------------------
# Verdict logic (implement EXACTLY as specified)
# ---------------------------------------------------------------------------

def compute_verdict(end_nb: dict[str, dict[int, float | None]], p1: bool) -> dict[str, Any]:
    """Compute all prediction booleans and verdict label/token.

    end_nb[arm][seed] = mean newborn intensity if measurable, else None (extinct/collapsed).
    """
    V = ["V1", "V2", "V3", "V4"]

    def vmeas(arm: str) -> list[int]:
        return [s for s in SEEDS if end_nb[arm][s] is not None]

    P2 = all(len(vmeas(a)) >= 4 for a in V)

    P3 = (
        all(len(vmeas(a)) >= 4 for a in V)
        and all(end_nb[a][s] < 0.15 for a in V for s in vmeas(a))
    )

    FP1 = any(
        sum(
            (end_nb[a][s] is not None and end_nb[a][s] > 0.30)
            for s in SEEDS
        ) >= 4
        for a in V
    )

    f_meas = vmeas("F")
    f_extinct = sum(end_nb["F"][s] is None for s in SEEDS)

    P4 = (
        (all(end_nb["F"][s] < 0.20 for s in f_meas) and len(f_meas) > 0)
        or (f_extinct >= 3)
    )

    FP2 = (len(f_meas) >= 4) and (sum(end_nb["F"][s] > 0.30 for s in f_meas) >= 4)

    # P5: supporting — V4 mean >= V1 mean but all V < 0.15; F extinction > V extinction
    v1_vals = [end_nb["V1"][s] for s in vmeas("V1")]
    v4_vals = [end_nb["V4"][s] for s in vmeas("V4")]
    v1_mean = float(np.mean(v1_vals)) if v1_vals else float("nan")
    v4_mean = float(np.mean(v4_vals)) if v4_vals else float("nan")
    v_extinction_counts = [sum(end_nb[a][s] is None for s in SEEDS) for a in V]
    f_extinction_count = f_extinct
    P5 = (
        (not math.isnan(v4_mean) and not math.isnan(v1_mean) and v4_mean >= v1_mean)
        and P3  # all V < 0.15 (already checked in P3)
        and (f_extinction_count > max(v_extinction_counts, default=0))
    )

    # Verdict
    if not p1:
        verdict = "NEGATIVE"
        token = "NEGATIVE"
    elif FP1 or FP2:
        verdict = "POSITIVE (valley crossable)"
        token = "POSITIVE"
    elif P3 and P4:
        verdict = "NEGATIVE (ceiling FUNDAMENTAL)"
        token = "NEGATIVE"
    else:
        verdict = "MIXED"
        token = "MIXED"

    return {
        "P1": p1,
        "P2": P2,
        "P3": P3,
        "P4": P4,
        "P5": P5,
        "FP1": FP1,
        "FP2": FP2,
        "f_meas_count": len(f_meas),
        "f_extinct_count": f_extinct,
        "v1_mean": v1_mean,
        "v4_mean": v4_mean,
        "verdict": verdict,
        "token": token,
    }


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()

    out_dir = _REPO / "experiments" / "outputs" / "exp199_n5_valley_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_labels = ["V1", "V2", "V3", "V4", "F"]

    # results[arm][seed] = run result dict
    results: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in arm_labels}

    print("Exp 199: running 5 arms x 5 seeds (horizon=12000, max_pop=5000) ...")
    for arm in arm_labels:
        cfg = make_arm_cfg(arm)
        for seed in SEEDS:
            print(f"  arm {arm} seed {seed} ...", end=" ", flush=True)
            t0 = time.time()
            result = run_arm(cfg, seed=seed)
            dt = time.time() - t0
            s = result["summary"]
            tag = "EXTINCT" if s["extinct"] else f"pop={s['final_pop']}"
            nb_str = "EXTINCT" if not s["valid"] else f"{s['end_newborn_intensity']:.4f}"
            print(f"done in {dt:.1f}s  {tag}  end_nb={nb_str}  valid={s['valid']}")
            results[arm][seed] = result

            # Persist per-arm/seed trajectory + summary
            traj_path = out_dir / f"traj_arm{arm}_seed{seed}.json"
            with open(traj_path, "w") as f:
                json.dump({
                    "arm": arm,
                    "seed": seed,
                    "trajectory": result["trajectory"],
                    "summary": result["summary"],
                }, f, indent=2)

    # ---------------------------------------------------------------------------
    # P1: determinism check (seed 18, arms V4 and F)
    # ---------------------------------------------------------------------------
    print("\nP1 determinism check (seed 18, arms V4 and F) ...", end=" ", flush=True)
    hash_V4_18 = results["V4"][18]["summary"]["events_hash"]
    hash_F_18 = results["F"][18]["summary"]["events_hash"]
    p1 = check_p1_determinism(hash_V4_18, hash_F_18)
    print(f"{'PASS' if p1 else 'FAIL'}")

    # ---------------------------------------------------------------------------
    # Build end_nb dict (None = extinct/collapsed/invalid)
    # ---------------------------------------------------------------------------
    end_nb: dict[str, dict[int, float | None]] = {}
    for arm in arm_labels:
        end_nb[arm] = {}
        for seed in SEEDS:
            s = results[arm][seed]["summary"]
            if s["valid"]:
                end_nb[arm][seed] = s["end_newborn_intensity"]
            else:
                end_nb[arm][seed] = None

    # ---------------------------------------------------------------------------
    # Compute verdict
    # ---------------------------------------------------------------------------
    vdict = compute_verdict(end_nb, p1)

    # ---------------------------------------------------------------------------
    # Build summary text
    # ---------------------------------------------------------------------------
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("EXP 199 — N5: FITNESS-VALLEY SWEEP — SUMMARY")
    lines.append("=" * 72)
    lines.append("")

    # Per-arm tables
    V_ARMS = ["V1", "V2", "V3", "V4"]
    lines.append("NOISE SWEEP (founder 0.10): can the primitive organ CLIMB?")
    lines.append(f"{'Arm':<6}{'noise':<8}" + "".join(f"  seed{s}" for s in SEEDS) + f"  {'mean':>8}  {'max_indiv':>9}  {'n_ext':>5}")
    lines.append("-" * 72)
    for arm in V_ARMS:
        noise, _ = ARMS[arm]
        meas_vals = [end_nb[arm][s] for s in SEEDS if end_nb[arm][s] is not None]
        arm_mean = float(np.mean(meas_vals)) if meas_vals else float("nan")
        max_indiv = max(
            results[arm][s]["summary"]["max_ever_intensity"] for s in SEEDS
        )
        n_ext = sum(end_nb[arm][s] is None for s in SEEDS)
        per_seed = "".join(
            f"  {'EXTINCT':>7}" if end_nb[arm][s] is None else f"  {end_nb[arm][s]:>7.4f}"
            for s in SEEDS
        )
        mean_str = f"nan" if math.isnan(arm_mean) else f"{arm_mean:.4f}"
        lines.append(f"{arm:<6}{noise:<8.2f}{per_seed}  {mean_str:>8}  {max_indiv:>9.4f}  {n_ext:>5}")
    lines.append("")

    lines.append("RETENTION (founder 0.50): is a functional organ STABLE?")
    lines.append(f"{'Arm':<6}{'noise':<8}" + "".join(f"  seed{s}" for s in SEEDS) + f"  {'mean':>8}  {'max_indiv':>9}  {'n_ext':>5}")
    lines.append("-" * 72)
    arm = "F"
    noise, _ = ARMS[arm]
    meas_vals_F = [end_nb[arm][s] for s in SEEDS if end_nb[arm][s] is not None]
    f_mean = float(np.mean(meas_vals_F)) if meas_vals_F else float("nan")
    max_indiv_F = max(
        results[arm][s]["summary"]["max_ever_intensity"] for s in SEEDS
    )
    n_ext_F = sum(end_nb[arm][s] is None for s in SEEDS)
    per_seed_F = "".join(
        f"  {'EXTINCT':>7}" if end_nb[arm][s] is None else f"  {end_nb[arm][s]:>7.4f}"
        for s in SEEDS
    )
    mean_str_F = "nan" if math.isnan(f_mean) else f"{f_mean:.4f}"
    lines.append(f"{arm:<6}{noise:<8.2f}{per_seed_F}  {mean_str_F:>8}  {max_indiv_F:>9.4f}  {n_ext_F:>5}")
    lines.append("")

    # Final populations
    lines.append("FINAL POPULATIONS per arm/seed:")
    lines.append(f"{'Arm':<6}" + "".join(f"  seed{s}" for s in SEEDS))
    lines.append("-" * 50)
    for arm in arm_labels:
        pops = "".join(f"  {results[arm][s]['summary']['final_pop']:>6}" for s in SEEDS)
        lines.append(f"{arm:<6}{pops}")
    lines.append("")

    # Prediction evaluations
    lines.append("PREDICTION EVALUATIONS:")
    lines.append(f"  P1  determinism (V4+F seed 18 rerun):      {'PASS' if vdict['P1'] else 'FAIL'}")
    lines.append(f"  P2  V1-V4 validity (>=4/5 each):           {'PASS' if vdict['P2'] else 'FAIL'}")
    lines.append(f"  P3  no climbing (all V end < 0.15):         {'PASS' if vdict['P3'] else 'FAIL'}")
    lines.append(f"  P4  no retention (F<0.20 or F ext>=3):      {'PASS' if vdict['P4'] else 'FAIL'}")
    lines.append(f"  P5  monotone-but-capped (V4>=V1, F ext>V):  {'PASS' if vdict['P5'] else 'FAIL'}")
    lines.append(f"  FP1 V arm end > 0.30 in >=4/5 seeds:       {'YES (FALSIFIED)' if vdict['FP1'] else 'NO'}")
    lines.append(f"  FP2 F retains >0.30 in >=4/5 meas seeds:   {'YES (FALSIFIED)' if vdict['FP2'] else 'NO'}")
    lines.append(f"  F arm measurable seeds: {vdict['f_meas_count']}/5;  extinct: {vdict['f_extinct_count']}/5")
    lines.append(f"  V1 mean = {vdict['v1_mean']:.4f};  V4 mean = {vdict['v4_mean']:.4f}")
    lines.append("")

    lines.append(f"VERDICT: {vdict['verdict']} (repo token: {vdict['token']})")
    lines.append("")

    # Runtime
    elapsed = time.time() - t_start
    lines.append(f"runtime: {elapsed:.0f}s")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    # Write exp199.txt
    out_txt = _REPO / "experiments" / "outputs" / "exp199.txt"
    with open(out_txt, "w") as f:
        f.write(summary_text + "\n")
    print(f"\n[saved {out_txt}]")

    # ---------------------------------------------------------------------------
    # Write verdict.json
    # ---------------------------------------------------------------------------
    per_arm_seed: dict[str, Any] = {}
    for arm in arm_labels:
        per_arm_seed[arm] = {}
        for seed in SEEDS:
            s = results[arm][seed]["summary"]
            per_arm_seed[arm][str(seed)] = {
                "valid": s["valid"],
                "end_newborn_intensity": s["end_newborn_intensity"],
                "final_pop": s["final_pop"],
                "extinct": s["extinct"],
                "max_ever_intensity": s["max_ever_intensity"],
                "steps_run": s["steps_run"],
                "events_hash": s["events_hash"],
            }

    verdict_path = out_dir / "verdict.json"
    with open(verdict_path, "w") as f:
        json.dump({
            "experiment": "exp199",
            "arms": ARMS,
            "seeds": SEEDS,
            "per_arm_seed": per_arm_seed,
            "end_nb": {
                arm: {str(s): end_nb[arm][s] for s in SEEDS}
                for arm in arm_labels
            },
            "predictions": {
                "P1": vdict["P1"],
                "P2": vdict["P2"],
                "P3": vdict["P3"],
                "P4": vdict["P4"],
                "P5": vdict["P5"],
                "FP1": vdict["FP1"],
                "FP2": vdict["FP2"],
            },
            "verdict": vdict["verdict"],
            "token": vdict["token"],
            "f_meas_count": vdict["f_meas_count"],
            "f_extinct_count": vdict["f_extinct_count"],
            "v1_mean": vdict["v1_mean"],
            "v4_mean": vdict["v4_mean"],
        }, f, indent=2)
    print(f"[saved {verdict_path}]")


if __name__ == "__main__":
    main()
