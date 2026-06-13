"""Exp 197 — N5: complexity as MAINTENANCE COST + heritable-selection controls
(pre-registered in loop/directions/population-ecology.md, commit e6c259e, BEFORE any data).

Hypothesis: modeling complexity as a MAINTENANCE COST — more-complex creatures burn more
baseline energy per tick (upkeep = base + cost_scale*complexity), with death emerging ONLY
through energy exhaustion ('starvation'), never a direct complexity-death rule — produces a
resource-mediated pressure that depresses the STANDING population's complexity over a long
horizon; AND if the NEWBORN (genotype/gene-pool) complexity also shifts down vs a matched
control, that pressure is HERITABLE selection, not merely survivor bias.

Mechanism (PROVIDED): per tick a creature pays cost_scale*complexity(genotype) extra energy;
energy<=0 -> 'starvation'. Senescence OFF in both arms (this isolates the maintenance channel).
Matched control = identical world/food/seeds/cap/horizon with cost_scale=0.

Pilot (disclosed, per L7): a 2-seed pilot {100,101} (deleted, not committed) fixed cost_scale
at 0.5 (both arms persist; living ~0.48->~0.25, newborn ~0.48->~0.28 in treatment). The final
verdict runs on FRESH seeds {8,9,10,11,12} never used in the pilot. Thresholds below were set
from the pilot and are NOT tuned after the final run.

Setup: balanced regime, max_population 5000 (resource carrying capacity, not the safety guard),
horizon 5000, cost_scale 0.5 (treatment) vs 0 (control), fresh seeds {8,9,10,11,12}. Per seed:
  living_gap   = control_living_end - treatment_living_end
  newborn_gap  = control_newborn_end - treatment_newborn_end
  survivor_component = living_gap - newborn_gap
  age_strat    = treatment(young_tercile - old_tercile) living complexity
('end' = value at t=5000 for living/age_strat; newborn_end = mean complexity of creatures born
in [4000,5000].)

Predictions if TRUE (5 fresh seeds, report ALL):
  P1 determinism: same seed -> identical event hash, both arms.
  P2 validity: both arms reach t=5000 without explosion/extinction, >=4/5 seeds.
  P3 standing effect (L): living_gap >= 0.05 in >=4/5 seeds.
  P4 heritable signal (H): newborn_gap >= 0.05 in >=4/5 seeds (gene pool shifts down).
  P5 stability: newborn_gap > 0 (control>treatment) for ALL valid seeds (no sign flips).

Falsifiers / verdict taxonomy (SB := survivor_component>=0.03 in >=3/5 AND age_strat>=0.02 in
>=3/5 treatment seeds):
  F1 non-determinism -> NEGATIVE. F2 arms fail to persist in a majority -> INCONCLUSIVE.
  NEGATIVE if NOT P3 and NOT P4. INCONCLUSIVE if NOT P5 (unstable).
  POSITIVE-HERITABLE if P4 and NOT SB. POSITIVE-MIXED if P4 and SB.
  POSITIVE-SURVIVOR-BIAS if P3 and NOT P4.
Repo token on the verdict line: POSITIVE (any POSITIVE-*), NEGATIVE, or MIXED (INCONCLUSIVE),
plus the fine-grained label. Interpretation discipline: baseline claim is a resource-mediated
depression of STANDING complexity; upgrade to 'heritable selection' ONLY if NEWBORN complexity
shifts down (P4). Death is energy-mediated (no complexity_death); cost_scale fixed pre-run.
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

from ecology.engine import Ecology
from ecology.scenarios import SCENARIOS
from ecology.genotype import complexity

# ---------------------------------------------------------------------------
# Experiment constants (pre-registered)
# ---------------------------------------------------------------------------
SEEDS = [8, 9, 10, 11, 12]
COST_SCALE_CTRL = 0.0
COST_SCALE_TRTM = 0.5
HORIZON = 5000
MAX_POP = 5000
# Sample every 500 steps (documented: every 250 would double runtime; we use 500)
CHECKPOINT_STRIDE = 500
CHECKPOINTS = list(range(CHECKPOINT_STRIDE, HORIZON + 1, CHECKPOINT_STRIDE))  # [500,1000,...,5000]
NEWBORN_WINDOW_START = 4000  # creatures born in [4000, 5000] -> newborn_end
NEWBORN_WINDOW_END = HORIZON


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------
def make_cfg(cost_scale: float) -> Any:
    """Return EcologyConfig for the balanced scenario with Exp 197 parameters."""
    base = SCENARIOS["balanced"]
    return dataclasses.replace(
        base,
        horizon=HORIZON,
        max_population=MAX_POP,
        complexity_cost_scale=cost_scale,
        # Senescence stays OFF (default) in both arms
        enable_senescence=False,
    )


# ---------------------------------------------------------------------------
# Instrumentation helpers (importable for unit tests)
# ---------------------------------------------------------------------------
def compute_age_stratified(alive: list) -> dict[str, float]:
    """Split alive creatures into young/mid/old AGE TERCILES; return mean complexity per bin.

    Returns a dict with keys 'young', 'mid', 'old'. Each value is the mean
    complexity of creatures in that tercile (by age). Returns NaN-filled dict
    on empty input.
    """
    if not alive:
        return {"young": float("nan"), "mid": float("nan"), "old": float("nan")}
    ages = np.array([c.phenotype.age for c in alive], dtype=float)
    complexities = np.array([complexity(c.genotype) for c in alive], dtype=float)
    n = len(alive)
    t1 = n // 3
    t2 = 2 * n // 3
    order = np.argsort(ages)
    young_idx = order[:t1] if t1 > 0 else np.array([], dtype=int)
    mid_idx = order[t1:t2] if t2 > t1 else np.array([], dtype=int)
    old_idx = order[t2:] if t2 < n else np.array([], dtype=int)
    return {
        "young": float(np.mean(complexities[young_idx])) if len(young_idx) > 0 else float("nan"),
        "mid": float(np.mean(complexities[mid_idx])) if len(mid_idx) > 0 else float("nan"),
        "old": float(np.mean(complexities[old_idx])) if len(old_idx) > 0 else float("nan"),
    }


def compute_energy_by_complexity_bucket(alive: list) -> dict[str, float]:
    """Split alive creatures into low/mid/high COMPLEXITY TERCILES; return mean energy per bucket.

    Returns dict with keys 'low', 'mid', 'high'. Every creature is in exactly one bucket.
    """
    if not alive:
        return {"low": float("nan"), "mid": float("nan"), "high": float("nan")}
    complexities = np.array([complexity(c.genotype) for c in alive], dtype=float)
    energies = np.array([c.phenotype.energy for c in alive], dtype=float)
    n = len(alive)
    t1 = n // 3
    t2 = 2 * n // 3
    order = np.argsort(complexities)
    low_idx = order[:t1] if t1 > 0 else np.array([], dtype=int)
    mid_idx = order[t1:t2] if t2 > t1 else np.array([], dtype=int)
    high_idx = order[t2:] if t2 < n else np.array([], dtype=int)
    return {
        "low": float(np.mean(energies[low_idx])) if len(low_idx) > 0 else float("nan"),
        "mid": float(np.mean(energies[mid_idx])) if len(mid_idx) > 0 else float("nan"),
        "high": float(np.mean(energies[high_idx])) if len(high_idx) > 0 else float("nan"),
    }


def compute_newborn_complexity(creatures: list, prev_t: int, curr_t: int) -> tuple[float, list]:
    """Return (mean_complexity, list_of_newborns) for creatures with birth_t in (prev_t, curr_t].

    Newborns are creatures born AFTER prev_t and AT OR BEFORE curr_t. Non-founders
    (parent_id is not None) born in this window count; we look at ALL creatures
    (alive and dead) so we capture the full birth cohort.
    """
    newborns = [
        c for c in creatures
        if c.phenotype.birth_t > prev_t and c.phenotype.birth_t <= curr_t
        and c.parent_id is not None  # exclude founders born at t=0
    ]
    if not newborns:
        return float("nan"), []
    mean_c = float(np.mean([complexity(c.genotype) for c in newborns]))
    return mean_c, newborns


def compute_parent_complexity_at_repro(newborns: list, creatures: list) -> float:
    """For newborns, look up their parent in creatures and average parent complexity."""
    if not newborns:
        return float("nan")
    id_to_creature = {c.creature_id: c for c in creatures}
    parent_complexities = []
    for nb in newborns:
        if nb.parent_id is not None and nb.parent_id in id_to_creature:
            parent_complexities.append(complexity(id_to_creature[nb.parent_id].genotype))
    if not parent_complexities:
        return float("nan")
    return float(np.mean(parent_complexities))


def compute_deaths_by_cause_cbucket(dead: list) -> dict[str, dict[str, int]]:
    """Group dead creatures by (cause_of_death x complexity_tercile).

    Returns nested dict: {cause: {bucket: count}}.
    Complexity terciles are computed over ALL dead creatures.
    """
    if not dead:
        return {}
    complexities = np.array([complexity(c.genotype) for c in dead], dtype=float)
    n = len(dead)
    t1 = n // 3
    t2 = 2 * n // 3
    order = np.argsort(complexities)
    bucket_labels = ["?"] * n
    for rank, idx in enumerate(order):
        if rank < t1:
            bucket_labels[idx] = "low"
        elif rank < t2:
            bucket_labels[idx] = "mid"
        else:
            bucket_labels[idx] = "high"
    result: dict[str, dict[str, int]] = {}
    for c, bucket in zip(dead, bucket_labels):
        cause = c.phenotype.cause_of_death or "unknown"
        if cause not in result:
            result[cause] = {"low": 0, "mid": 0, "high": 0}
        result[cause][bucket] = result[cause].get(bucket, 0) + 1
    return result


def compute_lifespan_by_cbucket(dead: list) -> dict[str, float]:
    """Mean lifespan per complexity tercile for dead creatures."""
    if not dead:
        return {"low": float("nan"), "mid": float("nan"), "high": float("nan")}
    complexities = np.array([complexity(c.genotype) for c in dead], dtype=float)
    lifespans = np.array([c.phenotype.age for c in dead], dtype=float)
    n = len(dead)
    t1 = n // 3
    t2 = 2 * n // 3
    order = np.argsort(complexities)
    low_idx = order[:t1] if t1 > 0 else np.array([], dtype=int)
    mid_idx = order[t1:t2] if t2 > t1 else np.array([], dtype=int)
    high_idx = order[t2:] if t2 < n else np.array([], dtype=int)
    return {
        "low": float(np.mean(lifespans[low_idx])) if len(low_idx) > 0 else float("nan"),
        "mid": float(np.mean(lifespans[mid_idx])) if len(mid_idx) > 0 else float("nan"),
        "high": float(np.mean(lifespans[high_idx])) if len(high_idx) > 0 else float("nan"),
    }


def compute_reproduction_by_cbucket(dead: list, alive: list) -> dict[str, dict[str, float]]:
    """Mean and total offspring_count per complexity tercile (all ever creatures)."""
    all_c = dead + alive
    if not all_c:
        return {}
    complexities = np.array([complexity(c.genotype) for c in all_c], dtype=float)
    offspring = np.array([c.phenotype.offspring_count for c in all_c], dtype=float)
    n = len(all_c)
    t1 = n // 3
    t2 = 2 * n // 3
    order = np.argsort(complexities)
    low_idx = order[:t1] if t1 > 0 else np.array([], dtype=int)
    mid_idx = order[t1:t2] if t2 > t1 else np.array([], dtype=int)
    high_idx = order[t2:] if t2 < n else np.array([], dtype=int)

    def _bucket_stats(idx: np.ndarray) -> dict[str, float]:
        if len(idx) == 0:
            return {"mean": float("nan"), "total": 0.0}
        vals = offspring[idx]
        return {"mean": float(np.mean(vals)), "total": float(np.sum(vals))}

    return {
        "low": _bucket_stats(low_idx),
        "mid": _bucket_stats(mid_idx),
        "high": _bucket_stats(high_idx),
    }


# ---------------------------------------------------------------------------
# Core runner: one arm x one seed, with checkpoint sampling
# ---------------------------------------------------------------------------
def run_arm(cost_scale: float, seed: int) -> dict[str, Any]:
    """Run one arm (cost_scale) with one seed; return trajectory + summary."""
    cfg = make_cfg(cost_scale)
    eco = Ecology(cfg, seed=seed)

    trajectory = []
    prev_checkpoint = 0

    while eco.t < HORIZON and not eco.exploded:
        eco.step()
        if eco.t in CHECKPOINTS:
            alive = eco._alive()
            curr_t = eco.t
            newborn_c, newborns = compute_newborn_complexity(eco._creatures, prev_checkpoint, curr_t)
            parent_c = compute_parent_complexity_at_repro(newborns, eco._creatures)
            age_strat = compute_age_stratified(alive)
            energy_cbucket = compute_energy_by_complexity_bucket(alive)
            snap = {
                "t": curr_t,
                "pop": len(alive),
                "living_complexity": float(np.mean([complexity(c.genotype) for c in alive])) if alive else float("nan"),
                "age_stratified": age_strat,
                "energy_by_cbucket": energy_cbucket,
                "newborn_complexity": newborn_c,
                "parent_complexity_at_repro": parent_c,
                "food": float(eco.world.resource.mean()),
            }
            trajectory.append(snap)
            prev_checkpoint = curr_t

        # Check for extinction
        if not eco._alive():
            break

    # End-of-run derived metrics
    alive = eco._alive()
    dead = [c for c in eco._creatures if not c.is_alive()]

    deaths_by_cause_cbucket = compute_deaths_by_cause_cbucket(dead)
    lifespan_by_cbucket = compute_lifespan_by_cbucket(dead)
    repro_by_cbucket = compute_reproduction_by_cbucket(dead, alive)

    # Newborn_end: creatures born in [4000, 5000]
    newborn_end_c, _ = compute_newborn_complexity(
        eco._creatures, NEWBORN_WINDOW_START - 1, NEWBORN_WINDOW_END
    )

    # living_end
    living_end_c = float(np.mean([complexity(c.genotype) for c in alive])) if alive else float("nan")

    # age_strat at end
    age_strat_end = compute_age_stratified(alive)

    # final pop + food
    final_pop = len(alive)
    final_food = float(eco.world.resource.mean())

    reached_horizon = eco.t >= HORIZON and not eco.exploded and len(alive) > 0
    extinct = len(alive) == 0
    exploded = eco.exploded

    summary = {
        "seed": seed,
        "cost_scale": cost_scale,
        "steps_run": eco.t,
        "reached_horizon": reached_horizon,
        "extinct": extinct,
        "exploded": exploded,
        "final_pop": final_pop,
        "final_food": final_food,
        "living_end": living_end_c,
        "newborn_end": newborn_end_c,
        "age_strat_end": age_strat_end,
        "deaths_by_cause_cbucket": deaths_by_cause_cbucket,
        "lifespan_by_cbucket": lifespan_by_cbucket,
        "reproduction_by_cbucket": repro_by_cbucket,
        "events_hash": eco.events_hash(),
    }
    return {"trajectory": trajectory, "summary": summary}


# ---------------------------------------------------------------------------
# Determinism check (P1): rerun seed 8 for both arms and compare hashes
# ---------------------------------------------------------------------------
def check_p1_determinism() -> bool:
    """Rerun seed 8 for both arms; confirm events_hash matches first run."""
    # We'll capture hashes from the main run and compare in main()
    # This function runs a secondary pair of runs and returns hash pairs
    results = {}
    for cost_scale, arm in [(COST_SCALE_CTRL, "ctrl"), (COST_SCALE_TRTM, "trtm")]:
        r = run_arm(cost_scale, seed=8)
        results[arm] = r["summary"]["events_hash"]
    return results


# ---------------------------------------------------------------------------
# Verdict logic (pre-registered, authoritative thresholds)
# ---------------------------------------------------------------------------
def compute_verdict(
    valid_seeds: list[int],
    living_gap: dict[int, float],
    newborn_gap: dict[int, float],
    survivor_component: dict[int, float],
    age_strat_gap: dict[int, float],
    p1: bool,
) -> tuple[str, str, dict[str, bool]]:
    """Return (verdict_label, repo_token, predicates_dict).

    Valid seeds = seeds where both arms reached t=5000.
    Thresholds are pre-registered and must not be changed after the run.
    """
    P2 = len(valid_seeds) >= 4
    P3 = sum(living_gap.get(s, 0.0) >= 0.05 for s in valid_seeds) >= 4
    P4 = sum(newborn_gap.get(s, 0.0) >= 0.05 for s in valid_seeds) >= 4
    P5 = all(newborn_gap.get(s, -999.0) > 0 for s in valid_seeds)
    SB = (
        sum(survivor_component.get(s, 0.0) >= 0.03 for s in valid_seeds) >= 3
        and sum(age_strat_gap.get(s, 0.0) >= 0.02 for s in valid_seeds) >= 3
    )

    if not p1:
        verdict, token = "NEGATIVE", "NEGATIVE"
    elif not P2:
        verdict, token = "INCONCLUSIVE", "MIXED"
    elif (not P3) and (not P4):
        verdict, token = "NEGATIVE", "NEGATIVE"
    elif not P5:
        verdict, token = "INCONCLUSIVE", "MIXED"
    elif P4 and SB:
        verdict, token = "POSITIVE-MIXED", "POSITIVE"
    elif P4 and not SB:
        verdict, token = "POSITIVE-HERITABLE", "POSITIVE"
    elif P3 and not P4:
        verdict, token = "POSITIVE-SURVIVOR-BIAS", "POSITIVE"
    else:
        verdict, token = "INCONCLUSIVE", "MIXED"

    predicates = {"P1": p1, "P2": P2, "P3": P3, "P4": P4, "P5": P5, "SB": SB}
    return verdict, token, predicates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    out_dir = Path("experiments/outputs/exp197_n5_maintenance_cost")
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []  # Accumulate output lines for exp197.txt

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("Exp 197 — N5: complexity as MAINTENANCE COST")
    emit(f"Seeds: {SEEDS}  |  cost_scale: ctrl={COST_SCALE_CTRL}  trt={COST_SCALE_TRTM}")
    emit(f"Horizon: {HORIZON}  |  max_pop: {MAX_POP}  |  checkpoint stride: {CHECKPOINT_STRIDE}")
    emit("=" * 72)

    # ---- Run all arms x seeds ----
    ctrl_results: dict[int, dict] = {}
    trtm_results: dict[int, dict] = {}

    for seed in SEEDS:
        emit(f"\nRunning seed {seed}...")
        ctrl_r = run_arm(COST_SCALE_CTRL, seed)
        trtm_r = run_arm(COST_SCALE_TRTM, seed)
        ctrl_results[seed] = ctrl_r
        trtm_results[seed] = trtm_r
        emit(f"  ctrl: pop={ctrl_r['summary']['final_pop']}  reached={ctrl_r['summary']['reached_horizon']}"
             f"  living_end={ctrl_r['summary']['living_end']:.4f}  newborn_end={ctrl_r['summary']['newborn_end']:.4f}")
        emit(f"  trt:  pop={trtm_r['summary']['final_pop']}  reached={trtm_r['summary']['reached_horizon']}"
             f"  living_end={trtm_r['summary']['living_end']:.4f}  newborn_end={trtm_r['summary']['newborn_end']:.4f}")

        # Save per-arm trajectory + summary JSONs
        for arm, results in [("ctrl", ctrl_r), ("trtm", trtm_r)]:
            arm_dir = out_dir / f"seed{seed}_{arm}"
            arm_dir.mkdir(exist_ok=True)
            with open(arm_dir / "trajectory.json", "w") as f:
                json.dump(results["trajectory"], f, indent=2)
            with open(arm_dir / "summary.json", "w") as f:
                json.dump(results["summary"], f, indent=2)

    # ---- P1 determinism check ----
    emit("\n--- P1 Determinism check (rerunning seed 8 for both arms) ---")
    p1_hashes = check_p1_determinism()
    p1_pass = True
    for arm in ["ctrl", "trtm"]:
        first_seed_results = ctrl_results[8] if arm == "ctrl" else trtm_results[8]
        first_hash = first_seed_results["summary"]["events_hash"]
        second_hash = p1_hashes[arm]
        match = first_hash == second_hash
        p1_pass = p1_pass and match
        emit(f"  {arm} seed8 hash match: {match}  ({first_hash[:16]}...)")

    # ---- Validity (P2) ----
    valid_seeds = [
        s for s in SEEDS
        if ctrl_results[s]["summary"]["reached_horizon"]
        and trtm_results[s]["summary"]["reached_horizon"]
    ]
    emit(f"\nValid seeds (both arms reach t={HORIZON}): {valid_seeds}")

    # ---- Per-seed metrics ----
    living_gap: dict[int, float] = {}
    newborn_gap: dict[int, float] = {}
    survivor_component: dict[int, float] = {}
    age_strat_gap: dict[int, float] = {}

    emit("\n--- Per-seed metrics ---")
    emit(f"{'Seed':>5}  {'ctrl_liv':>9}  {'trt_liv':>8}  {'ctrl_nb':>8}  {'trt_nb':>8}"
         f"  {'liv_gap':>8}  {'nb_gap':>8}  {'surv_comp':>9}  {'age_strat':>9}  {'valid':>5}")
    for seed in SEEDS:
        cs = ctrl_results[seed]["summary"]
        ts = trtm_results[seed]["summary"]
        ctrl_liv = cs["living_end"]
        trt_liv = ts["living_end"]
        ctrl_nb = cs["newborn_end"]
        trt_nb = ts["newborn_end"]
        l_gap = ctrl_liv - trt_liv
        n_gap = ctrl_nb - trt_nb
        s_comp = l_gap - n_gap
        # age_strat: treatment young - old
        trt_age = ts["age_strat_end"]
        a_strat = (trt_age["young"] - trt_age["old"]) if (
            not np.isnan(trt_age["young"]) and not np.isnan(trt_age["old"])
        ) else float("nan")
        valid = seed in valid_seeds
        living_gap[seed] = l_gap
        newborn_gap[seed] = n_gap
        survivor_component[seed] = s_comp
        age_strat_gap[seed] = a_strat if not np.isnan(a_strat) else 0.0
        emit(f"{seed:>5}  {ctrl_liv:>9.4f}  {trt_liv:>8.4f}  {ctrl_nb:>8.4f}  {trt_nb:>8.4f}"
             f"  {l_gap:>8.4f}  {n_gap:>8.4f}  {s_comp:>9.4f}  {a_strat:>9.4f}  {str(valid):>5}")

    # ---- Density / food diagnostic ----
    emit("\n--- Density/food diagnostic (ctrl vs trt, final) ---")
    emit(f"{'Seed':>5}  {'ctrl_pop':>9}  {'trt_pop':>8}  {'ctrl_food':>10}  {'trt_food':>9}")
    for seed in SEEDS:
        cs = ctrl_results[seed]["summary"]
        ts = trtm_results[seed]["summary"]
        emit(f"{seed:>5}  {cs['final_pop']:>9}  {ts['final_pop']:>8}"
             f"  {cs['final_food']:>10.5f}  {ts['final_food']:>9.5f}")

    # ---- Age-stratified + by-bucket tables for seed 8 (representative) ----
    rep_seed = 8
    emit(f"\n--- Age-stratified complexity table (seed {rep_seed}) ---")
    for arm, res in [("ctrl", ctrl_results[rep_seed]), ("trtm", trtm_results[rep_seed])]:
        age_e = res["summary"]["age_strat_end"]
        emit(f"  {arm}: young={age_e['young']:.4f}  mid={age_e['mid']:.4f}  old={age_e['old']:.4f}")

    emit(f"\n--- Energy by complexity bucket at end (seed {rep_seed}) ---")
    # Pull last checkpoint from trajectory
    for arm, res in [("ctrl", ctrl_results[rep_seed]), ("trtm", trtm_results[rep_seed])]:
        traj = res["trajectory"]
        if traj:
            last = traj[-1]
            eb = last["energy_by_cbucket"]
            emit(f"  {arm} t={last['t']}: low={eb['low']:.3f}  mid={eb['mid']:.3f}  high={eb['high']:.3f}")

    emit(f"\n--- Deaths by cause x complexity bucket (seed {rep_seed}, trt arm) ---")
    dbcb = trtm_results[rep_seed]["summary"]["deaths_by_cause_cbucket"]
    for cause, buckets in dbcb.items():
        emit(f"  {cause}: {buckets}")

    emit(f"\n--- Lifespan by complexity bucket (seed {rep_seed}, trt arm) ---")
    lbcb = trtm_results[rep_seed]["summary"]["lifespan_by_cbucket"]
    emit(f"  low={lbcb['low']:.2f}  mid={lbcb['mid']:.2f}  high={lbcb['high']:.2f}")

    # ---- Verdict ----
    verdict, token, predicates = compute_verdict(
        valid_seeds, living_gap, newborn_gap, survivor_component, age_strat_gap, p1_pass
    )

    emit("\n--- Predictions ---")
    emit(f"  P1 (determinism):         {predicates['P1']}")
    emit(f"  P2 (validity >=4/5):      {predicates['P2']}  [{len(valid_seeds)}/5 valid]")
    emit(f"  P3 (living_gap>=0.05, >=4/5):  {predicates['P3']}"
         f"  [{sum(living_gap.get(s,0)>=0.05 for s in valid_seeds)}/{len(valid_seeds)} seeds pass]")
    emit(f"  P4 (newborn_gap>=0.05, >=4/5): {predicates['P4']}"
         f"  [{sum(newborn_gap.get(s,0)>=0.05 for s in valid_seeds)}/{len(valid_seeds)} seeds pass]")
    emit(f"  P5 (newborn_gap>0, all valid): {predicates['P5']}"
         f"  [{sum(newborn_gap.get(s,-999)>0 for s in valid_seeds)}/{len(valid_seeds)} seeds pass]")
    emit(f"  SB (surv_comp>=0.03 >=3/5 AND age_strat>=0.02 >=3/5): {predicates['SB']}"
         f"  [surv_comp: {sum(survivor_component.get(s,0)>=0.03 for s in valid_seeds)}/{len(valid_seeds)},"
         f" age_strat: {sum(age_strat_gap.get(s,0)>=0.02 for s in valid_seeds)}/{len(valid_seeds)}]")

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
        ts = trtm_results[seed]["summary"]
        trt_age = ts["age_strat_end"]
        a_strat_val = (trt_age["young"] - trt_age["old"]) if (
            not np.isnan(trt_age["young"]) and not np.isnan(trt_age["old"])
        ) else float("nan")
        verdict_data["per_seed"][str(seed)] = {
            "valid": seed in valid_seeds,
            "ctrl_living_end": cs["living_end"],
            "trt_living_end": ts["living_end"],
            "ctrl_newborn_end": cs["newborn_end"],
            "trt_newborn_end": ts["newborn_end"],
            "living_gap": living_gap.get(seed, float("nan")),
            "newborn_gap": newborn_gap.get(seed, float("nan")),
            "survivor_component": survivor_component.get(seed, float("nan")),
            "age_strat": age_strat_gap.get(seed, float("nan")),
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
