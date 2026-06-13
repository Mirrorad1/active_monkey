"""Exp 200 — N5: the FORAGING escape — can a non-saturating benefit cross the fitness valley?
(pre-registered in loop/directions/population-ecology.md, commit 2b14459, BEFORE any data).

Hypothesis (the escape, from Exp 199's mechanism): a NON-saturating foraging benefit (food
concentrated in a DRIFTING thermal band; thermosense used to FORAGE - you always need food) lets a
FUNCTIONAL thermosense organ EVOLVE (gene-pool mean intensity climbs > 0.30), crossing the Exp 199
fitness valley (where AVOIDANCE sensing self-limited to primitive because a little sensing reaches
safety).

Mechanism (PROVIDED, gated engine; exp194-199 byte-identical): enable_food_coupling concentrates
resource regen in a drifting thermal band; thermosense_forage_mode steers the policy toward the
food-optimal temperature instead of away from stress; temperature_stress_scale 0 (the ONLY benefit is
foraging). The engine behavioral test confirms a strong forager (intensity 0.8) reproduces ~4x more
than a no-organ creature.

Arms (temp ON, stress 0, cheap efficient organ inefficiency 0.20/floor 0, founder intensity 0.10,
enable_thermosense, forage mode, food_concentration 8, forage weight 4, noise 0.5; horizon 12000;
fresh seeds {23,24,25,26,27}):
  WIDE   forage: food_band_width 0.15, amplitude 0.3, period 1500.
  NARROW forage: food_band_width 0.05, amplitude 0.4, period 600 (precision-demanding).
  CONTROL: enable_food_coupling=False (uniform food -> thermosense useless) - primitive baseline.
Metric = NEWBORN (gene-pool) mean thermosense intensity, end = mean over creatures born in [10000,12000].

Pilots (disclosed, per L7): pilots {100,101} (deleted) - wide + narrow + precision foraging regimes;
gene-pool mean stayed 0.04-0.09 (never functional), max individual 0.3-0.6; narrower bands gave LOWER
mean. Pilots SUGGEST the escape fails; the experiment tests on FRESH seeds {23-27}.

Predictions if TRUE (5 fresh seeds; L21 validity = pop>=10 + a newborn cohort):
  P1 determinism: same seed -> identical event hash (WIDE + CONTROL on seed 23).
  P2 validity: arms persist measurably >=4/5 seeds.
  P3 (CORE escape test): SOME foraging arm (WIDE or NARROW) end gene-pool mean intensity > 0.30 in
     >=4/5 seeds -> a functional organ evolved under foraging -> the escape WORKS.

Falsifier -> NEGATIVE (escape fails, ceiling general): both foraging arms stay primitive (end mean <
0.15) in >=3/5 seeds -> the foraging escape FAILS; the primitive ceiling holds across avoidance AND
foraging. F1 non-determinism -> NEGATIVE.

Verdict rule: POSITIVE iff P3; NEGATIVE iff both foraging arms majority-primitive; MIXED if partial.
Repo token POSITIVE/NEGATIVE/MIXED. Diagnostics (not gating): max individual intensity per arm (a
high-intensity individual exists but does not dominate -> the marginal gradient is weak: crude sensing
captures the easy benefit, a narrow band is too stochastic to select on). Predicting POSITIVE (the
human's escape hypothesis); pilots suggest NEGATIVE - an honest test. Founders + costs PROVIDED.
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
SEEDS = [23, 24, 25, 26, 27]
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
    temperature_stress_scale=0.0,
    tolerance_cost_scale=0.0,
    comfort_amplitude=0.0,
    thermosense_upkeep_floor=0.0,
    thermosense_active_threshold=0.05,
    thermosense_noise_base=0.5,
    thermal_avoidance_weight=4.0,
    food_optimal_base=0.5,
    food_concentration=8.0,
)


# ---------------------------------------------------------------------------
# Founder + config factories (importable for unit tests)
# ---------------------------------------------------------------------------

def founder() -> Any:
    """Primitive foothold 0.10, cheap efficient organ (inefficiency=0.2), tolerance=0.10."""
    return D.replace(
        FOUNDER,
        thermosense_intensity=0.10,
        thermosense_inefficiency=0.2,
        temperature_tolerance=0.10,
    )


def cfg_forage(band: float, amp: float, per: float) -> EcologyConfig:
    """Return EcologyConfig for a foraging arm with given band width, amplitude, period."""
    return D.replace(
        SCENARIOS["balanced"],
        horizon=HORIZON,
        max_population=MAX_POP,
        founder=founder(),
        enable_food_coupling=True,
        thermosense_forage_mode=True,
        food_band_width=band,
        food_optimal_amplitude=amp,
        food_optimal_period=per,
        **BASE,
    )


def cfg_control() -> EcologyConfig:
    """No food coupling -> thermosense useless -> primitive baseline."""
    return D.replace(
        SCENARIOS["balanced"],
        horizon=HORIZON,
        max_population=MAX_POP,
        founder=founder(),
        enable_food_coupling=False,
        thermosense_forage_mode=False,
        food_band_width=0.15,
        food_optimal_amplitude=0.0,
        food_optimal_period=1500.0,
        **BASE,
    )


# Fixed arm configs (pre-registered)
_ARM_CFGS: dict[str, EcologyConfig] = {
    "WIDE":    cfg_forage(0.15, 0.3, 1500.0),
    "NARROW":  cfg_forage(0.05, 0.4, 600.0),
    "CONTROL": cfg_control(),
}


def make_arm_cfg(arm: str) -> EcologyConfig:
    """Return the EcologyConfig for the named arm (WIDE/NARROW/CONTROL)."""
    return _ARM_CFGS[arm]


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
    """L21-compliant validity: reached t=HORIZON AND final pop >= MIN_VALID_POP
    AND >= 1 newborn in window.

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

    Per the spec: step manually (while eco.t<cfg.horizon and not eco.exploded: eco.step()),
    sampling every CHECKPOINT_STRIDE steps.
    Do NOT write events.jsonl.

    Uses cfg.horizon as the stopping condition (respects overrides in unit tests).
    The newborn end window is always [NEWBORN_WINDOW_START, cfg.horizon].
    """
    eco = Ecology(cfg, seed=seed)
    horizon = cfg.horizon

    trajectory: list[dict[str, Any]] = []
    checkpoints = list(range(CHECKPOINT_STRIDE, horizon + 1, CHECKPOINT_STRIDE))

    while eco.t < horizon and not eco.exploded:
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

    # End metric: newborn mean intensity for creatures born in [NEWBORN_WINDOW_START, horizon]
    end_nb_intensity = newborn_mean_intensity_in_window(
        eco._creatures, NEWBORN_WINDOW_START, horizon
    )

    # Max individual intensity ever seen (across all ever-born creatures)
    all_intensities = [c.genotype.thermosense_intensity for c in eco._creatures]
    max_ever_intensity = max(all_intensities) if all_intensities else float("nan")

    valid = is_valid_run(eco, end_nb_intensity)

    # Extinction step: step at which population first dropped to 0 (None if not extinct)
    extinction_step: int | None = None
    if len(alive) == 0:
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
# Determinism check (P1): rerun seed 23 for arms WIDE and CONTROL
# ---------------------------------------------------------------------------

def check_p1_determinism(
    hash_wide_seed23: str,
    hash_control_seed23: str,
) -> bool:
    """Rerun seed 23 for arms WIDE and CONTROL; confirm event hashes match."""
    r2_wide = run_arm(make_arm_cfg("WIDE"), seed=23)
    r2_ctrl = run_arm(make_arm_cfg("CONTROL"), seed=23)
    match_wide = r2_wide["summary"]["events_hash"] == hash_wide_seed23
    match_ctrl = r2_ctrl["summary"]["events_hash"] == hash_control_seed23
    return match_wide and match_ctrl


# ---------------------------------------------------------------------------
# Verdict logic (implement EXACTLY as specified)
# ---------------------------------------------------------------------------

def compute_verdict(
    end_nb: dict[str, dict[int, float | None]],
    p1: bool,
) -> dict[str, Any]:
    """Compute all prediction booleans and verdict label/token.

    end_nb[arm][seed] = mean newborn intensity if measurable, else None.
    """
    forage = ["WIDE", "NARROW"]

    def meas(a: str) -> list[int]:
        return [s for s in SEEDS if end_nb[a][s] is not None]

    P2 = all(len(meas(a)) >= 4 for a in ["WIDE", "NARROW", "CONTROL"])

    P3 = any(
        sum((end_nb[a][s] is not None and end_nb[a][s] > 0.30) for s in SEEDS) >= 4
        for a in forage
    )

    neg = all(
        sum((end_nb[a][s] is not None and end_nb[a][s] < 0.15) for s in SEEDS) >= 3
        for a in forage
    )

    if not p1:
        verdict = "NEGATIVE"
        token = "NEGATIVE"
    elif P3:
        verdict = "POSITIVE (foraging escape WORKS)"
        token = "POSITIVE"
    elif neg:
        verdict = "NEGATIVE (escape FAILS; ceiling general)"
        token = "NEGATIVE"
    else:
        verdict = "MIXED"
        token = "MIXED"

    return {
        "P1": p1,
        "P2": P2,
        "P3": P3,
        "neg": neg,
        "verdict": verdict,
        "token": token,
    }


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()

    out_dir = _REPO / "experiments" / "outputs" / "exp200_n5_foraging_escape"
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_labels = ["WIDE", "NARROW", "CONTROL"]

    # results[arm][seed] = run result dict
    results: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in arm_labels}

    print("Exp 200: running 3 arms x 5 seeds (horizon=12000, max_pop=5000) ...")
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
    # P1: determinism check (seed 23, arms WIDE and CONTROL)
    # ---------------------------------------------------------------------------
    print("\nP1 determinism check (seed 23, arms WIDE and CONTROL) ...", end=" ", flush=True)
    hash_wide_23 = results["WIDE"][23]["summary"]["events_hash"]
    hash_ctrl_23 = results["CONTROL"][23]["summary"]["events_hash"]
    p1 = check_p1_determinism(hash_wide_23, hash_ctrl_23)
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
    lines.append("EXP 200 — N5: FORAGING ESCAPE — SUMMARY")
    lines.append("=" * 72)
    lines.append("")

    # Per-arm table header
    lines.append(f"{'Arm':<8}" + "".join(f"  seed{s}" for s in SEEDS) + f"  {'mean':>8}  {'max_indiv':>9}  {'n_ext':>5}")
    lines.append("-" * 72)

    for arm in arm_labels:
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
        mean_str = "nan" if math.isnan(arm_mean) else f"{arm_mean:.4f}"
        lines.append(f"{arm:<8}{per_seed}  {mean_str:>8}  {max_indiv:>9.4f}  {n_ext:>5}")
    lines.append("")

    # Final populations
    lines.append("FINAL POPULATIONS per arm/seed:")
    lines.append(f"{'Arm':<8}" + "".join(f"  seed{s}" for s in SEEDS))
    lines.append("-" * 50)
    for arm in arm_labels:
        pops = "".join(f"  {results[arm][s]['summary']['final_pop']:>6}" for s in SEEDS)
        lines.append(f"{arm:<8}{pops}")
    lines.append("")

    # CONTROL equilibrium note
    ctrl_vals = [end_nb["CONTROL"][s] for s in SEEDS if end_nb["CONTROL"][s] is not None]
    ctrl_mean = float(np.mean(ctrl_vals)) if ctrl_vals else float("nan")
    ctrl_note = f"{ctrl_mean:.4f}" if not math.isnan(ctrl_mean) else "nan"
    lines.append(f"CONTROL equilibrium (thermosense useless -> should be low): mean={ctrl_note}")
    lines.append("")

    # Prediction evaluations
    lines.append("PREDICTION EVALUATIONS:")
    lines.append(f"  P1  determinism (WIDE+CONTROL seed 23 rerun):    {'PASS' if vdict['P1'] else 'FAIL'}")
    lines.append(f"  P2  validity (>=4/5 each arm):                   {'PASS' if vdict['P2'] else 'FAIL'}")
    lines.append(f"  P3  CORE escape (some forage arm >0.30 in 4/5):  {'PASS' if vdict['P3'] else 'FAIL'}")
    lines.append(f"  neg falsifier (both forage arms <0.15 in 3/5):   {'YES' if vdict['neg'] else 'NO'}")
    lines.append("")

    lines.append(f"VERDICT: {vdict['verdict']} (repo token: {vdict['token']})")
    lines.append("")

    # Runtime (placeholder — filled below)
    elapsed = time.time() - t_start
    lines.append(f"runtime: {elapsed:.0f}s")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    # Write exp200.txt
    out_txt = _REPO / "experiments" / "outputs" / "exp200.txt"
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
            "experiment": "exp200",
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
                "neg": vdict["neg"],
            },
            "verdict": vdict["verdict"],
            "token": vdict["token"],
        }, f, indent=2)
    print(f"[saved {verdict_path}]")


if __name__ == "__main__":
    main()
