"""Exp283 — boundary transmission accounting (predeclared before runs).

Question: Can a nonnegative heritable capability increase from zero even while
its realized selection component is negative, and can a lineage-resolved
accounting expose that distinction without perturbing the simulation?

Hypothesis: A nonnegative heritable capability can increase from zero while its
realized selection component is negative; lineage accounting can distinguish
selection from transmission without perturbing the simulation.

Predeclaration, frozen before runs: Three arms derived from Exp198's zero-founder
config, all at 3000 steps and max_population 5000: NEUTRAL (temperature OFF,
activation threshold 1.0, so this trait has neither cost nor sensing); COST_ONLY
(temperature OFF, original .05 threshold and upkeep); THERMAL (temperature ON,
original .05 threshold and upkeep). All other inherited traits still mutate.
Fresh seeds 9500..9515; excluded runtime/instrument pilot 9499. Fixed sigma=.05.
No scientific threshold changes after pilots. No steady-state claim.

Prediction: Cost-only: at least 12/16 seeds complete, finish with living trait mean >=.02,
and cumulative realized selection <=-.02, jointly. Neutral mean final >=.02 in
at least12/16 is the second conjunct. Thermal is descriptive, not a preregistered
directional comparison.

Falsifier: Any primary scientific conjunct failing yields NEGATIVE;
malformed lineage or accounting mismatch >1e-10 is INVALID. Extinction/explosion
is reported and fails the scientific completion conjunct, not excluded.

The observer supplies a standard Price decomposition; selection is realized
association, not a causal trait effect. Expected clipped drift is conditioned
on the actual newborn parents and weighted by newborns / after-population.
Observed transmission minus that benchmark is diagnostic, never a verdict gate.
Survivors count as self-descendants. No simulation or mutation rule is changed.

Run from the repository root with uv run --python .venv python <this file>
--smoke [--smoke-horizon 600] or --full. Full runs require a safe runtime
preflight; only its recommended worker count adapts. Smoke uses excluded seed
9499, and cannot earn a scientific verdict. --output-prefix preserves pilots.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from math import fsum
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ecology.engine import Ecology, EcologyConfig
from ecology.runtime_budget import preflight
from ecology.transmission import clipped_gaussian_drift, price_step
from experiments.exp198_n5_thermosense_attractor import make_cfg

ARMS = ("NEUTRAL", "COST_ONLY", "THERMAL")
SEEDS = tuple(range(9500, 9516))
PILOT_SEED = 9499
HORIZON = 3000
SIGMA = .05
ACCOUNTING_TOLERANCE = 1e-10
TIME_BUDGET_S = 1800.
OBSERVER_MARGIN = 2.
SOURCE_PATHS = (
    "ecology/transmission.py", "experiments/exp283_boundary_transmission.py",
    "tests/test_transmission.py", "ecology/engine.py", "ecology/genotype.py",
    "ecology/scenarios.py", "ecology/creature.py", "ecology/world.py",
    "ecology/runtime_budget.py", "experiments/exp198_n5_thermosense_attractor.py",
    "docs/superpowers/plans/2026-10-27-boundary-transmission-paper.md",
)


def make_config(arm: str, horizon: int = HORIZON) -> EcologyConfig:
    """Exp198's zero-founder arms, with only the predeclared changes."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    config = replace(make_cfg(arm == "THERMAL", 0.), horizon=horizon, mutation_rate=SIGMA)
    if arm == "NEUTRAL":
        config = replace(config, thermosense_active_threshold=1.)
    return config


def run_one(arm: str, seed: int, horizon: int = HORIZON, observe: bool = True,
            *, deadline: float | None = None) -> dict[str, Any]:
    """Run a single fixed arm/seed; observations use no RNG or engine mutations.

    ``deadline`` is an optional monotonic wall-clock limit shared by full-batch
    processes. Its expiry is retained as failed completion, never an exclusion.
    Birth time follows the engine: step [t,t+1) creates children with birth_t=t.
    The final newborn window is [horizon-500,horizon), including dead newborns.
    """
    start = time.monotonic()
    eco = Ecology(make_config(arm, horizon), seed=seed)
    initial = eco.alive_snapshot()
    initial_mean = fsum(c.genotype.thermosense_intensity for c in initial) / len(initial) if initial else None
    trajectory: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    newborn_window: list[float] = []
    budget_exceeded = False
    while eco.t < horizon and eco.has_alive() and not eco.exploded:
        if deadline is not None and time.monotonic() >= deadline:
            budget_exceeded = True
            break
        if observe:
            # Capture parents BEFORE the step: a reproducing parent can then die.
            before = {c.creature_id: c.genotype.thermosense_intensity for c in eco.alive_snapshot()}
        eco.step()
        if not observe:
            continue
        current = eco.alive_snapshot()
        after = {c.creature_id: c.genotype.thermosense_intensity for c in current}
        newborns = [c for c in current if c.creature_id not in before]
        parents = {c.creature_id: c.parent_id for c in newborns}
        try:
            if len(before) + len(newborns) < len(current) or len(after) != len(current):
                raise ValueError("duplicate descendant identity")
            row = price_step(before, after, parents)
            if any(after[cid] != before[cid] for cid in before.keys() & after.keys()):
                raise ValueError("survivor genotype changed during read-only observation")
            expected = (
                float(np.sum(clipped_gaussian_drift([before[c.parent_id] for c in newborns], SIGMA))) / len(after)
                if after else None
            )
            row.update(t=eco.t, expected_transmission=expected,
                       transmission_minus_expected=(row["transmission"] - expected) if expected is not None else None)
            trajectory.append(row)
            newborn_window.extend(c.genotype.thermosense_intensity for c in newborns
                                  if horizon - 500 <= c.phenotype.birth_t < horizon)
            if row["residual"] is not None and abs(row["residual"]) > ACCOUNTING_TOLERANCE:
                raise ValueError(f"Price residual exceeds tolerance: {row['residual']}")
        except (ValueError, KeyError) as exc:
            errors.append({"t": eco.t, "error": str(exc), "before": before, "after": after, "newborn_parents": parents})
            break
    final = eco.alive_snapshot()
    final_mean = fsum(c.genotype.thermosense_intensity for c in final) / len(final) if final else None
    defined_rows = [row for row in trajectory if row["residual"] is not None]
    sums = {name: fsum(row[name] for row in defined_rows) if observe else None
            for name in ("selection", "transmission", "expected_transmission", "residual")}
    # If extinction makes the last transition undefined, retain the last defined
    # population's accounting rather than treating the final empty mean as zero.
    accounted_mean = defined_rows[-1]["mean_after"] if defined_rows else initial_mean
    telescoping_residual = (
        accounted_mean - initial_mean - sums["selection"] - sums["transmission"]
        if observe and accounted_mean is not None and initial_mean is not None else None
    )
    if telescoping_residual is not None and abs(telescoping_residual) > ACCOUNTING_TOLERANCE:
        errors.append({"t": eco.t, "error": f"cumulative identity residual exceeds tolerance: {telescoping_residual}"})
    summary = {
        "arm": arm, "seed": seed, "horizon": horizon, "observed": observe,
        "steps_run": eco.t, "reached_horizon": eco.t == horizon and bool(final) and not eco.exploded and not errors,
        "extinct": not final, "exploded": eco.exploded, "budget_exceeded": budget_exceeded,
        "final_population": len(final), "initial_living_mean": initial_mean,
        "final_living_mean": final_mean,
        "final_newborn_mean_last500": fsum(newborn_window) / len(newborn_window) if newborn_window else None,
        "final_newborn_count_last500": len(newborn_window) if observe else None,
        "newborn_window": [max(0, horizon - 500), horizon], "newborn_window_right_open": True,
        "births": sum(row["births"] for row in trajectory) if observe else None,
        "deaths": sum(row["deaths"] for row in trajectory) if observe else None,
        "accounted_steps": len(defined_rows) if observe else None,
        "accounted_final_mean": accounted_mean if observe else None,
        "cumulative_selection": sums["selection"], "cumulative_transmission": sums["transmission"],
        "cumulative_expected_transmission": sums["expected_transmission"],
        "cumulative_transmission_minus_expected": sums["transmission"] - sums["expected_transmission"] if observe else None,
        "cumulative_identity_residual": sums["residual"], "telescoping_residual": telescoping_residual,
        "max_abs_identity_residual": max((abs(row["residual"]) for row in defined_rows), default=0.) if observe else None,
        "accounting_valid": not errors if observe else None,
        "events_hash": eco.events_hash(), "runtime_s": time.monotonic() - start,
    }
    return {"summary": summary, "trajectory": trajectory, "errors": errors}


def evaluate_results(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen conjunctions to all sixteen seeds, retaining failures."""
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    found = [(s["arm"], s["seed"]) for s in summaries]
    complete_batch = len(found) == len(expected) and set(found) == expected
    cost = [s for s in summaries if s["arm"] == "COST_ONLY"]
    neutral = [s for s in summaries if s["arm"] == "NEUTRAL"]
    cost_joint = [s["seed"] for s in cost if s.get("reached_horizon")
                  and s.get("final_living_mean") is not None and s["final_living_mean"] >= .02
                  and s.get("cumulative_selection") is not None and s["cumulative_selection"] <= -.02]
    neutral_joint = [s["seed"] for s in neutral if s.get("reached_horizon")
                     and s.get("final_living_mean") is not None and s["final_living_mean"] >= .02]
    valid = complete_batch and all(s.get("accounting_valid") for s in summaries)
    return {
        "verdict": "INVALID" if not valid else ("POSITIVE" if len(cost_joint) >= 12 and len(neutral_joint) >= 12 else "NEGATIVE"),
        "complete_batch": complete_batch, "accounting_valid": valid,
        "cost_only_joint_count": len(cost_joint), "cost_only_joint_seeds": cost_joint,
        "neutral_joint_count": len(neutral_joint), "neutral_joint_seeds": neutral_joint,
        "required_count": 12, "denominator": 16, "thermal_directional_gate": None,
    }


def _save_result(path: Path, result: dict, source_hashes: dict[str, str]) -> None:
    """Lossless columnar trajectories keep the full cohort evidence compact."""
    rows = result["trajectory"]
    columns = list(rows[0]) if rows else []
    payload = {"summary": result["summary"], "errors": result["errors"],
               "source_hashes": source_hashes, "columns": columns,
               "trajectory": [[row[column] for column in columns] for row in rows]}
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-horizon", type=int, default=600)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args(argv)
    horizon = HORIZON if args.full else args.smoke_horizon
    if horizon <= 0:
        parser.error("--smoke-horizon must be positive")
    prefix = args.output_prefix or _REPO / "experiments" / "outputs" / ("exp283" if args.full else "exp283_smoke")
    if not prefix.is_absolute():
        prefix = _REPO / prefix
    trajectory_dir = prefix.parent / (prefix.name + "_trajectories")
    for artifact in (prefix.with_suffix(".txt"), prefix.with_suffix(".json"), trajectory_dir):
        if artifact.exists() or artifact.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {artifact}; choose a fresh --output-prefix")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: hashlib.sha256((_REPO / name).read_bytes()).hexdigest() for name in SOURCE_PATHS}
    metadata: dict[str, Any] = {
        "experiment": 283, "mode": "full" if args.full else "smoke", "predeclaration": __doc__,
        "horizon": horizon, "seeds": list(SEEDS) if args.full else [PILOT_SEED],
        "excluded_pilot_seed": PILOT_SEED, "source_hashes": source_hashes,
        "configs": {arm: asdict(make_config(arm, horizon)) for arm in ARMS},
        "pilot": {"seed": PILOT_SEED, "excluded_from_scientific_cohort": True,
                  "runtime_probe_steps_per_config": 600 if args.full else None},
        "summaries": [], "trajectory_files": [],
    }
    start = time.monotonic()
    with prefix.with_suffix(".txt").open("w", encoding="utf-8", buffering=1) as output:
        def emit(label: str, value: Any) -> None:
            line = label + " " + json.dumps(value, sort_keys=True, allow_nan=False)
            print(line, flush=True)
            output.write(line + "\n")

        def checkpoint() -> None:
            prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

        def retain(result: dict) -> None:
            summary = result["summary"]
            path = trajectory_dir / f"{summary['arm'].lower()}_seed{summary['seed']}.json.gz"
            _save_result(path, result, source_hashes)
            metadata["summaries"].append(summary)
            metadata["trajectory_files"].append(str(path.relative_to(prefix.parent)))
            emit("RESULT", summary)
            checkpoint()

        emit("PREDECLARATION", __doc__)
        emit("SOURCE_HASHES", source_hashes)
        checkpoint()
        try:
            if args.full:
                report = preflight([(arm, make_config(arm), PILOT_SEED) for arm in ARMS],
                                   horizon=HORIZON, n_jobs=len(ARMS) * len(SEEDS), max_workers=4,
                                   probe_steps=600, time_budget_s=TIME_BUDGET_S, require_safe=True)
                # Per-job projections are rounded to tenths by preflight. Add .05
                # before multiplying so the observer margin never rounds downward.
                projected_s = ((max(r["proj_per_job_s"] for r in report["configs"]) + .05)
                               * report["n_jobs"] / report["recommended_workers"] * OBSERVER_MARGIN)
                metadata["preflight"] = report
                metadata["observer_runtime_projection"] = {
                    "factor": OBSERVER_MARGIN, "projected_batch_s": projected_s, "budget_s": TIME_BUDGET_S,
                }
                emit("PREFLIGHT", report)
                emit("OBSERVER_RUNTIME_PROJECTION", metadata["observer_runtime_projection"])
                checkpoint()
                if projected_s > TIME_BUDGET_S:
                    raise RuntimeError(f"observer-adjusted projection {projected_s:.1f}s exceeds {TIME_BUDGET_S:.0f}s budget")
                equivalence = []
                for arm in ARMS:
                    watched = run_one(arm, PILOT_SEED, horizon=120)
                    bare = run_one(arm, PILOT_SEED, horizon=120, observe=False)
                    item = {"arm": arm, "seed": PILOT_SEED, "horizon": 120,
                            "observed_events_hash": watched["summary"]["events_hash"],
                            "unobserved_events_hash": bare["summary"]["events_hash"],
                            "match": watched["summary"]["events_hash"] == bare["summary"]["events_hash"],
                            "accounting_valid": watched["summary"]["accounting_valid"]}
                    equivalence.append(item)
                metadata["pilot"]["observer_equivalence"] = equivalence
                emit("OBSERVER_EQUIVALENCE", equivalence)
                checkpoint()
                if not all(item["match"] and item["accounting_valid"] for item in equivalence):
                    raise RuntimeError("excluded-pilot observer equivalence/instrument check failed")
                deadline = time.monotonic() + TIME_BUDGET_S
                with ProcessPoolExecutor(max_workers=report["recommended_workers"]) as pool:
                    jobs = {pool.submit(run_one, arm, seed, HORIZON, deadline=deadline): (arm, seed)
                            for arm in ARMS for seed in SEEDS}
                    for future in as_completed(jobs):
                        arm, seed = jobs[future]
                        try:
                            retain(future.result())
                        except Exception as exc:
                            failure = {"arm": arm, "seed": seed, "reached_horizon": False,
                                       "accounting_valid": False, "error": f"{type(exc).__name__}: {exc}"}
                            retain({"summary": failure, "trajectory": [], "errors": [failure]})
                metadata["criteria"] = evaluate_results(metadata["summaries"])
            else:
                for arm in ARMS:
                    retain(run_one(arm, PILOT_SEED, horizon=horizon))
                metadata["criteria"] = {"verdict": "SMOKE_ONLY", "excluded_from_scientific_cohort": True}
            metadata["runtime_s"] = time.monotonic() - start
            emit("CRITERIA", metadata["criteria"])
            emit("RUNTIME_S", metadata["runtime_s"])
            checkpoint()
        except Exception as exc:
            metadata["aborted"] = f"{type(exc).__name__}: {exc}"
            metadata["runtime_s"] = time.monotonic() - start
            emit("ABORTED", metadata["aborted"])
            checkpoint()
            raise


if __name__ == "__main__":
    main()
