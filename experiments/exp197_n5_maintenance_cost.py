"""Exp 197 — N5: complexity = EXPRESSED-MACHINERY cost — a costed, evolvable THERMOSENSE organ
under temperature (pre-registered REVISED in loop/directions/population-ecology.md, commit da1749f,
BEFORE the fresh-seed run; supersedes the scalar complexity_cost_scale version).

Hypothesis: a costed, expressed, evolvable thermosense organ (noisy temperature-gradient info the
policy uses to flee thermal stress, at upkeep = floor + inefficiency*intensity) is RETAINED under
temperature — where its info value offsets its metabolic burden — and selected AWAY without
temperature (pure cost); and because thermal tolerance is ALSO costed and the safe zone DRIFTS (no
free escapes), thermosense's fate tests whether EFFICIENT capability-bearing complexity is selected
FOR when it pays its way. NEWBORN vs standing thermosense separates heritable selection from
survivor bias.

Mechanism (PROVIDED): genotype thermosense_intensity (0=absent), thermosense_inefficiency (efficiency
multiplier, floor 0.2). intensity>0.05 => ACTIVE: noisy thermal reading -> policy steers to low-stress
cells; upkeep = 0.02 + inefficiency*intensity. Temperature = static left->right gradient [0,1] whose
COMFORT CENTER drifts (sin, amp 0.4, period 1500); outside temperature_tolerance band -> stress energy
(scale 1.0) -> 'starvation'. Thermal tolerance COSTED (1.5*tolerance/tick). Control = temperature OFF;
treatment = ON; both enable_thermosense. Founder SEEDS a primitive organ (intensity 0.20, tolerance
0.10): tests RETENTION/loss, NOT de-novo emergence (disclosed). Regime FIXED on pilots {100,101};
fresh seeds {8,9,10,11,12}.

Predictions if TRUE (5 fresh seeds; end=t5000; newborn=born in [4000,5000]):
  P1 determinism: same seed -> identical event hash, both arms.
  P2 validity: both arms persist to t=5000 (no extinction/explosion), >=4/5 seeds.
  P3 standing retention (L): treatment living thermosense active-fraction - control >= 0.15, >=4/5.
  P4 heritable retention (H): treatment newborn mean intensity - control >= 0.008 (trt>ctrl), >=4/5.
  P5 stability: treatment>control in 5/5 for BOTH living active-fraction and newborn intensity.
  Supporting: under temperature mean temperature_tolerance stays LOW (treatment < control).

Falsifiers / verdict taxonomy: SB := (living active-frac gap >=0.30 in >=3/5) AND (newborn intensity
gap <0.02 in >=3/5).
  F1 non-determinism -> NEGATIVE. F2 arms fail to persist (majority) -> INCONCLUSIVE.
  NEGATIVE if NOT P3. INCONCLUSIVE if NOT P5.
  POSITIVE-HERITABLE if P3 and P4 and NOT SB. POSITIVE-MIXED if P3 and P4 and SB.
  POSITIVE-SURVIVOR-BIAS if P3 and NOT P4.
Repo token: POSITIVE (any POSITIVE-*), NEGATIVE, or MIXED (INCONCLUSIVE), plus the fine-grained label.
Interpretation: baseline claim = thermosense RETAINED among the living under temperature where info
value offsets cost; upgrade to 'heritable selection FOR efficient capability' ONLY if NEWBORN
thermosense shifts (P4). Death energy-mediated (no thermosense_death). Params fixed pre-run; founder
seeds a primitive organ (disclosed); tolerance costed + comfort drifts (removed cheap escapes).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
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
# Experiment constants (pre-registered / fixed from pilots {100,101})
# ---------------------------------------------------------------------------
SEEDS = [8, 9, 10, 11, 12]
HORIZON = 5000
MAX_POP = 5000
CHECKPOINT_STRIDE = 250
CHECKPOINTS = list(range(CHECKPOINT_STRIDE, HORIZON + 1, CHECKPOINT_STRIDE))
NEWBORN_WINDOW_START = 4000
NEWBORN_WINDOW_END = HORIZON
ACTIVE_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Config factory (FIXED regime from pilots {100,101})
# ---------------------------------------------------------------------------
_founder = dataclasses.replace(
    FOUNDER,
    thermosense_intensity=0.20,
    temperature_tolerance=0.10,
)

REGIME = dict(
    enable_thermosense=True,
    temperature_stress_scale=1.0,
    tolerance_cost_scale=1.5,
    comfort_amplitude=0.4,
    comfort_period=1500.0,
    thermosense_upkeep_floor=0.02,
    thermosense_noise_base=0.2,
    thermosense_active_threshold=0.05,
    temperature_comfort=0.5,
    thermal_avoidance_weight=2.5,
)


def make_cfg(temp_on: bool) -> EcologyConfig:
    """Return EcologyConfig for the balanced scenario with Exp 197 thermosense parameters."""
    return dataclasses.replace(
        SCENARIOS["balanced"],
        horizon=HORIZON,
        max_population=MAX_POP,
        founder=_founder,
        enable_temperature=temp_on,
        **REGIME,
    )


# ---------------------------------------------------------------------------
# Instrumentation helpers (importable for unit tests)
# ---------------------------------------------------------------------------

def compute_living_thermosense(alive: list) -> dict[str, float]:
    """Compute thermosense metrics over the LIVING population snapshot.

    Returns:
      active_fraction  — fraction with intensity > ACTIVE_THRESHOLD
      mean_intensity   — mean intensity over all alive creatures
      mean_inefficiency — mean inefficiency over ACTIVE creatures only
      mean_tolerance   — mean temperature_tolerance
      pop              — count of alive creatures
    """
    if not alive:
        return {
            "active_fraction": float("nan"),
            "mean_intensity": float("nan"),
            "mean_inefficiency": float("nan"),
            "mean_tolerance": float("nan"),
            "pop": 0,
        }
    intensities = np.array([c.genotype.thermosense_intensity for c in alive], dtype=float)
    inefficiencies = np.array([c.genotype.thermosense_inefficiency for c in alive], dtype=float)
    tolerances = np.array([c.genotype.temperature_tolerance for c in alive], dtype=float)
    active_mask = intensities > ACTIVE_THRESHOLD
    active_fraction = float(np.mean(active_mask))
    mean_intensity = float(np.mean(intensities))
    active_ineff = inefficiencies[active_mask]
    mean_inefficiency = float(np.mean(active_ineff)) if active_ineff.size > 0 else float("nan")
    mean_tolerance = float(np.mean(tolerances))
    return {
        "active_fraction": active_fraction,
        "mean_intensity": mean_intensity,
        "mean_inefficiency": mean_inefficiency,
        "mean_tolerance": mean_tolerance,
        "pop": len(alive),
    }


def compute_newborn_thermosense(creatures: list, prev_t: int, curr_t: int) -> dict[str, float]:
    """Compute thermosense metrics for NEWBORNS: creatures with birth_t in (prev_t, curr_t].

    Only non-founders (parent_id is not None) are counted.
    Returns:
      active_fraction — fraction with intensity > ACTIVE_THRESHOLD
      mean_intensity  — mean intensity over newborns
      count           — number of newborns in window
    """
    newborns = [
        c for c in creatures
        if c.phenotype.birth_t > prev_t
        and c.phenotype.birth_t <= curr_t
        and c.parent_id is not None
    ]
    if not newborns:
        return {"active_fraction": float("nan"), "mean_intensity": float("nan"), "count": 0}
    intensities = np.array([c.genotype.thermosense_intensity for c in newborns], dtype=float)
    active_mask = intensities > ACTIVE_THRESHOLD
    return {
        "active_fraction": float(np.mean(active_mask)),
        "mean_intensity": float(np.mean(intensities)),
        "count": len(newborns),
    }


def compute_age_stratified_thermosense(alive: list) -> dict[str, float]:
    """Split alive creatures into young/mid/old AGE TERCILES; return mean intensity per bin.

    Returns dict with keys 'young', 'mid', 'old'. Each value is the mean
    thermosense_intensity of creatures in that age tercile.
    Returns NaN-filled dict on empty input or single-element input.
    """
    if not alive:
        return {"young": float("nan"), "mid": float("nan"), "old": float("nan")}
    ages = np.array([c.phenotype.age for c in alive], dtype=float)
    intensities = np.array([c.genotype.thermosense_intensity for c in alive], dtype=float)
    n = len(alive)
    t1 = n // 3
    t2 = 2 * n // 3
    order = np.argsort(ages)
    young_idx = order[:t1]
    mid_idx = order[t1:t2]
    old_idx = order[t2:]
    return {
        "young": float(np.mean(intensities[young_idx])) if young_idx.size > 0 else float("nan"),
        "mid": float(np.mean(intensities[mid_idx])) if mid_idx.size > 0 else float("nan"),
        "old": float(np.mean(intensities[old_idx])) if old_idx.size > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Core runner: one arm x one seed, with checkpoint sampling
# ---------------------------------------------------------------------------

def run_arm(temp_on: bool, seed: int) -> dict[str, Any]:
    """Run one arm (temp_on controls temperature) with one seed; return trajectory + summary."""
    cfg = make_cfg(temp_on)
    eco = Ecology(cfg, seed=seed)

    trajectory = []
    prev_checkpoint = 0
    # Track the highest creature_id that existed at the previous checkpoint for newborn detection.
    # All creatures are appended in id order; any id > prev_max_id was born after prev_checkpoint.
    prev_max_id = eco.next_id - 1  # last founder id at t=0

    while eco.t < HORIZON and not eco.exploded:
        eco.step()
        if eco.t in CHECKPOINTS:
            alive = eco._alive()
            curr_t = eco.t
            living_snap = compute_living_thermosense(alive)
            newborn_snap = compute_newborn_thermosense(eco._creatures, prev_checkpoint, curr_t)
            age_strat = compute_age_stratified_thermosense(alive)
            food = float(eco.world.resource.mean())
            snap = {
                "t": curr_t,
                "living": living_snap,
                "newborn": newborn_snap,
                "age_stratified": age_strat,
                "food": food,
            }
            trajectory.append(snap)
            prev_checkpoint = curr_t
            prev_max_id = eco.next_id - 1

        if not eco._alive():
            break

    # End-of-run metrics
    alive = eco._alive()
    final_living = compute_living_thermosense(alive)
    # Newborn end: creatures born in (NEWBORN_WINDOW_START-1, NEWBORN_WINDOW_END]
    final_newborn = compute_newborn_thermosense(
        eco._creatures, NEWBORN_WINDOW_START - 1, NEWBORN_WINDOW_END
    )
    age_strat_end = compute_age_stratified_thermosense(alive)
    final_food = float(eco.world.resource.mean())

    reached_horizon = eco.t >= HORIZON and not eco.exploded and len(alive) > 0
    extinct = len(alive) == 0
    exploded = eco.exploded

    summary = {
        "seed": seed,
        "temp_on": temp_on,
        "steps_run": eco.t,
        "reached_horizon": reached_horizon,
        "extinct": extinct,
        "exploded": exploded,
        "final_pop": len(alive),
        "final_food": final_food,
        "living_end": final_living,
        "newborn_end": final_newborn,
        "age_strat_end": age_strat_end,
        "events_hash": eco.events_hash(),
    }
    return {"trajectory": trajectory, "summary": summary}


# ---------------------------------------------------------------------------
# Determinism check (P1): rerun seed 8 for both arms and compare hashes
# ---------------------------------------------------------------------------

def check_p1_determinism(ctrl_hash_seed8: str, trt_hash_seed8: str) -> bool:
    """Rerun seed 8 for both arms; confirm events_hash matches first run."""
    ctrl_r2 = run_arm(False, seed=8)
    trt_r2 = run_arm(True, seed=8)
    ctrl_match = ctrl_r2["summary"]["events_hash"] == ctrl_hash_seed8
    trt_match = trt_r2["summary"]["events_hash"] == trt_hash_seed8
    return ctrl_match and trt_match


# ---------------------------------------------------------------------------
# Verdict logic (pre-registered; implement EXACTLY per conductor specification)
# ---------------------------------------------------------------------------

def compute_verdict(
    valid_seeds: list[int],
    liv_gap: dict[int, float],
    new_gap: dict[int, float],
    p1: bool,
) -> tuple[str, str, dict[str, Any]]:
    """Return (verdict_label, repo_token, predicates_dict).

    Thresholds and logic are pre-registered and must not be changed after the run.
    """
    P2 = len(valid_seeds) >= 4
    P3 = sum(liv_gap.get(s, 0.0) >= 0.15 for s in valid_seeds) >= 4
    P4 = sum(new_gap.get(s, 0.0) >= 0.008 for s in valid_seeds) >= 4
    P5 = all(liv_gap.get(s, -999.0) > 0 for s in valid_seeds) and all(
        new_gap.get(s, -999.0) > 0 for s in valid_seeds
    )
    SB = (
        sum(liv_gap.get(s, 0.0) >= 0.30 for s in valid_seeds) >= 3
        and sum(new_gap.get(s, 0.0) < 0.02 for s in valid_seeds) >= 3
    )

    if not p1:
        verdict, token = "NEGATIVE", "NEGATIVE"
    elif not P2:
        verdict, token = "INCONCLUSIVE", "MIXED"
    elif not P3:
        verdict, token = "NEGATIVE", "NEGATIVE"
    elif not P5:
        verdict, token = "INCONCLUSIVE", "MIXED"
    elif P3 and P4 and SB:
        verdict, token = "POSITIVE-MIXED", "POSITIVE"
    elif P3 and P4 and not SB:
        verdict, token = "POSITIVE-HERITABLE", "POSITIVE"
    elif P3 and not P4:
        verdict, token = "POSITIVE-SURVIVOR-BIAS", "POSITIVE"
    else:
        verdict, token = "INCONCLUSIVE", "MIXED"

    predicates: dict[str, Any] = {
        "P1": p1, "P2": P2, "P3": P3, "P4": P4, "P5": P5, "SB": SB,
    }
    return verdict, token, predicates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    out_dir = Path("experiments/outputs/exp197_n5_maintenance_cost")
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("Exp 197 — N5: costed evolvable THERMOSENSE organ under temperature")
    emit(f"Seeds: {SEEDS}  |  ctrl=temp_OFF  trt=temp_ON")
    emit(f"Horizon: {HORIZON}  |  max_pop: {MAX_POP}  |  checkpoint stride: {CHECKPOINT_STRIDE}")
    emit(f"Founder: thermosense_intensity=0.20, temperature_tolerance=0.10")
    emit("=" * 72)

    # ---- Run all arms x seeds ----
    ctrl_results: dict[int, dict] = {}
    trt_results: dict[int, dict] = {}

    for seed in SEEDS:
        emit(f"\nRunning seed {seed}...")
        ctrl_r = run_arm(False, seed)
        trt_r = run_arm(True, seed)
        ctrl_results[seed] = ctrl_r
        trt_results[seed] = trt_r
        cs = ctrl_r["summary"]
        ts = trt_r["summary"]
        emit(
            f"  ctrl (temp_OFF): pop={cs['final_pop']}  reached={cs['reached_horizon']}"
            f"  active_frac={cs['living_end']['active_fraction']:.4f}"
            f"  nb_intensity={cs['newborn_end']['mean_intensity']:.4f}"
        )
        emit(
            f"  trt  (temp_ON):  pop={ts['final_pop']}  reached={ts['reached_horizon']}"
            f"  active_frac={ts['living_end']['active_fraction']:.4f}"
            f"  nb_intensity={ts['newborn_end']['mean_intensity']:.4f}"
        )

        # Save per-arm trajectory + summary JSONs
        for arm_name, res in [("ctrl", ctrl_r), ("trt", trt_r)]:
            arm_dir = out_dir / f"seed{seed}_{arm_name}"
            arm_dir.mkdir(exist_ok=True)
            with open(arm_dir / "trajectory.json", "w") as f:
                json.dump(res["trajectory"], f, indent=2)
            with open(arm_dir / "summary.json", "w") as f:
                json.dump(res["summary"], f, indent=2)

    # ---- P1 determinism check (rerun seed 8, both arms) ----
    emit("\n--- P1 Determinism check (rerunning seed 8 for both arms) ---")
    ctrl_hash_8 = ctrl_results[8]["summary"]["events_hash"]
    trt_hash_8 = trt_results[8]["summary"]["events_hash"]

    ctrl_r2 = run_arm(False, seed=8)
    trt_r2 = run_arm(True, seed=8)
    ctrl_match = ctrl_r2["summary"]["events_hash"] == ctrl_hash_8
    trt_match = trt_r2["summary"]["events_hash"] == trt_hash_8
    p1_pass = ctrl_match and trt_match

    emit(f"  ctrl seed8 hash match: {ctrl_match}  ({ctrl_hash_8[:16]}...)")
    emit(f"  trt  seed8 hash match: {trt_match}  ({trt_hash_8[:16]}...)")
    emit(f"  P1 PASS: {p1_pass}")

    # ---- Validity (P2) ----
    valid_seeds = [
        s for s in SEEDS
        if ctrl_results[s]["summary"]["reached_horizon"]
        and trt_results[s]["summary"]["reached_horizon"]
    ]
    emit(f"\nValid seeds (both arms reach t={HORIZON}): {valid_seeds}")

    # ---- Per-seed gap metrics ----
    liv_gap: dict[int, float] = {}
    new_gap: dict[int, float] = {}

    emit("\n--- Per-seed metrics (t=5000) ---")
    emit(
        f"{'Seed':>5}  {'ctrl_livAF':>10}  {'trt_livAF':>9}  {'ctrl_livMI':>10}  {'trt_livMI':>9}"
        f"  {'ctrl_nbI':>8}  {'trt_nbI':>8}  {'liv_gap':>8}  {'new_gap':>8}"
        f"  {'ctrl_tol':>8}  {'trt_tol':>8}  {'valid':>5}"
    )
    for seed in SEEDS:
        cs = ctrl_results[seed]["summary"]
        ts = trt_results[seed]["summary"]
        ctrl_liv_af = cs["living_end"]["active_fraction"]
        trt_liv_af = ts["living_end"]["active_fraction"]
        ctrl_liv_mi = cs["living_end"]["mean_intensity"]
        trt_liv_mi = ts["living_end"]["mean_intensity"]
        ctrl_nb_i = cs["newborn_end"]["mean_intensity"]
        trt_nb_i = ts["newborn_end"]["mean_intensity"]
        ctrl_tol = cs["living_end"]["mean_tolerance"]
        trt_tol = ts["living_end"]["mean_tolerance"]
        l_gap = (trt_liv_af - ctrl_liv_af) if not (
            np.isnan(trt_liv_af) or np.isnan(ctrl_liv_af)
        ) else float("nan")
        n_gap = (trt_nb_i - ctrl_nb_i) if not (
            np.isnan(trt_nb_i) or np.isnan(ctrl_nb_i)
        ) else float("nan")
        valid = seed in valid_seeds
        liv_gap[seed] = l_gap if not np.isnan(l_gap) else -999.0
        new_gap[seed] = n_gap if not np.isnan(n_gap) else -999.0
        emit(
            f"{seed:>5}  {ctrl_liv_af:>10.4f}  {trt_liv_af:>9.4f}"
            f"  {ctrl_liv_mi:>10.4f}  {trt_liv_mi:>9.4f}"
            f"  {ctrl_nb_i:>8.4f}  {trt_nb_i:>8.4f}"
            f"  {l_gap:>8.4f}  {n_gap:>8.4f}"
            f"  {ctrl_tol:>8.4f}  {trt_tol:>8.4f}"
            f"  {str(valid):>5}"
        )

    # ---- Tolerance + population + food diagnostic ----
    emit("\n--- Tolerance + population + food diagnostic (ctrl vs trt, final) ---")
    emit(
        f"{'Seed':>5}  {'ctrl_pop':>9}  {'trt_pop':>8}"
        f"  {'ctrl_food':>10}  {'trt_food':>9}"
        f"  {'ctrl_tol':>8}  {'trt_tol':>8}"
    )
    for seed in SEEDS:
        cs = ctrl_results[seed]["summary"]
        ts = trt_results[seed]["summary"]
        emit(
            f"{seed:>5}  {cs['final_pop']:>9}  {ts['final_pop']:>8}"
            f"  {cs['final_food']:>10.5f}  {ts['final_food']:>9.5f}"
            f"  {cs['living_end']['mean_tolerance']:>8.4f}"
            f"  {ts['living_end']['mean_tolerance']:>8.4f}"
        )

    # ---- Age-stratified thermosense for representative seed ----
    rep_seed = 8
    emit(f"\n--- Age-stratified thermosense table (seed {rep_seed}, t=5000) ---")
    for arm_name, res in [("ctrl", ctrl_results[rep_seed]), ("trt", trt_results[rep_seed])]:
        age_e = res["summary"]["age_strat_end"]
        emit(
            f"  {arm_name}: young={age_e['young']:.4f}"
            f"  mid={age_e['mid']:.4f}"
            f"  old={age_e['old']:.4f}"
        )

    # ---- Verdict ----
    verdict, token, predicates = compute_verdict(valid_seeds, liv_gap, new_gap, p1_pass)

    emit("\n--- Predictions ---")
    emit(f"  P1 (determinism):                         {predicates['P1']}")
    emit(f"  P2 (validity >=4/5):                      {predicates['P2']}  [{len(valid_seeds)}/5 valid]")
    p3_count = sum(liv_gap.get(s, -999.0) >= 0.15 for s in valid_seeds)
    emit(f"  P3 (liv_gap>=0.15, >=4/5):                {predicates['P3']}  [{p3_count}/{len(valid_seeds)} seeds pass]")
    p4_count = sum(new_gap.get(s, -999.0) >= 0.008 for s in valid_seeds)
    emit(f"  P4 (new_gap>=0.008, >=4/5):               {predicates['P4']}  [{p4_count}/{len(valid_seeds)} seeds pass]")
    p5_liv = sum(liv_gap.get(s, -999.0) > 0 for s in valid_seeds)
    p5_nb = sum(new_gap.get(s, -999.0) > 0 for s in valid_seeds)
    emit(f"  P5 (both gaps>0, all valid):              {predicates['P5']}  [liv>0: {p5_liv}/{len(valid_seeds)}, nb>0: {p5_nb}/{len(valid_seeds)}]")
    sb_liv = sum(liv_gap.get(s, 0.0) >= 0.30 for s in valid_seeds)
    sb_nb = sum(new_gap.get(s, 0.0) < 0.02 for s in valid_seeds)
    emit(f"  SB (liv_gap>=0.30 >=3/5 AND new_gap<0.02 >=3/5): {predicates['SB']}  [liv>=0.30: {sb_liv}/{len(valid_seeds)}, nb<0.02: {sb_nb}/{len(valid_seeds)}]")

    emit(f"\nVERDICT: {verdict} (repo token: {token})")

    # ---- Runtime ----
    elapsed = time.time() - t_start
    emit(f"\nruntime: {elapsed:.1f}s")

    # ---- Write exp197.txt ----
    txt_path = Path("experiments/outputs/exp197.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[Written: {txt_path}]")

    # ---- Write verdict.json ----
    verdict_data: dict[str, Any] = {
        "experiment": "exp197",
        "verdict": verdict,
        "token": token,
        "predicates": predicates,
        "valid_seeds": valid_seeds,
        "per_seed": {},
    }
    for seed in SEEDS:
        cs = ctrl_results[seed]["summary"]
        ts = trt_results[seed]["summary"]
        verdict_data["per_seed"][str(seed)] = {
            "valid": seed in valid_seeds,
            "ctrl_living_active_fraction": cs["living_end"]["active_fraction"],
            "trt_living_active_fraction": ts["living_end"]["active_fraction"],
            "ctrl_living_mean_intensity": cs["living_end"]["mean_intensity"],
            "trt_living_mean_intensity": ts["living_end"]["mean_intensity"],
            "ctrl_newborn_mean_intensity": cs["newborn_end"]["mean_intensity"],
            "trt_newborn_mean_intensity": ts["newborn_end"]["mean_intensity"],
            "ctrl_mean_tolerance": cs["living_end"]["mean_tolerance"],
            "trt_mean_tolerance": ts["living_end"]["mean_tolerance"],
            "liv_gap": liv_gap.get(seed, float("nan")),
            "new_gap": new_gap.get(seed, float("nan")),
            "ctrl_final_pop": cs["final_pop"],
            "trt_final_pop": ts["final_pop"],
            "ctrl_final_food": cs["final_food"],
            "trt_final_food": ts["final_food"],
            "ctrl_events_hash": cs["events_hash"],
            "trt_events_hash": ts["events_hash"],
        }

    verdict_path = out_dir / "verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict_data, f, indent=2)
    print(f"[Written: {verdict_path}]")


if __name__ == "__main__":
    main()
