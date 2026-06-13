"""Toy-scale Active Monkey experiment suite.

The suite is deliberately small and deterministic. It provides reusable probes
for replay, fork scaling, planner distillation, and grounded abstraction without
making broad capability claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
from statistics import mean
from typing import Any, Iterable


TRAJECTORY_FIELDS = (
    "seed",
    "agent_id",
    "fork_id",
    "env_id",
    "state",
    "observation",
    "action",
    "belief_state",
    "policy",
    "expected_free_energy_terms",
    "transition_update",
    "outcome",
    "verifier_score",
)

REPLAY_MODES = (
    "observation_only",
    "action_conditioned",
    "self_generated",
    "third_party_with_actions",
)

POPULATION_MODES = (
    "isolated",
    "shared_trajectory_archive",
    "shared_transition_model",
    "coordinator",
)

METRIC_NAMES = (
    "task_success",
    "sample_efficiency",
    "map_accuracy",
    "object_place_binding_accuracy",
    "transition_model_error",
    "preference_satisfaction",
    "calibration_error",
    "seed_robustness",
    "compute_cost_per_success",
    "coordination_overhead",
    "holdout_regression",
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    seeds: list[int]
    metrics: list[str]
    replay_modes: list[str] | None = None
    population_sizes: list[int] | None = None
    population_modes: list[str] | None = None
    horizon: int = 8


def load_config(path: str | Path) -> ExperimentConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(
        name=data["name"],
        description=data["description"],
        seeds=list(data["seeds"]),
        metrics=list(data["metrics"]),
        replay_modes=data.get("replay_modes"),
        population_sizes=data.get("population_sizes"),
        population_modes=data.get("population_modes"),
        horizon=int(data.get("horizon", 8)),
    )


def _blank_metrics() -> dict[str, float]:
    return {name: 0.0 for name in METRIC_NAMES}


def _mean_dict(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    return {name: mean(row[name] for row in rows) for name in METRIC_NAMES}


def _trajectory_row(
    *,
    seed: int,
    agent_id: str,
    fork_id: str,
    env_id: str,
    state: dict[str, Any],
    observation: dict[str, Any],
    action: str,
    belief_state: dict[str, Any],
    policy: dict[str, Any],
    expected_free_energy_terms: dict[str, float],
    transition_update: dict[str, Any],
    outcome: dict[str, Any],
    verifier_score: float,
) -> dict[str, Any]:
    values = {
        "seed": seed,
        "agent_id": agent_id,
        "fork_id": fork_id,
        "env_id": env_id,
        "state": state,
        "observation": observation,
        "action": action,
        "belief_state": belief_state,
        "policy": policy,
        "expected_free_energy_terms": expected_free_energy_terms,
        "transition_update": transition_update,
        "outcome": outcome,
        "verifier_score": verifier_score,
    }
    return {field: values[field] for field in TRAJECTORY_FIELDS}


def _demo_actions(horizon: int) -> list[str]:
    cycle = ["right", "right", "pickup", "right", "use", "right", "finish"]
    return [cycle[i % len(cycle)] for i in range(horizon)]


def _run_replay_mode(seed: int, mode: str, horizon: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rng = random.Random(seed)
    source_actions = _demo_actions(horizon)
    position = 0
    learned_edges = 0
    successes = 0
    rows: list[dict[str, Any]] = []

    for t, source_action in enumerate(source_actions):
        if mode == "observation_only":
            action = "watch"
            causal_contact = False
        elif mode == "third_party_with_actions":
            action = source_action
            causal_contact = False
        elif mode == "action_conditioned":
            action = source_action
            causal_contact = True
        else:
            action = "right" if position < 4 else "finish"
            causal_contact = True

        before = position
        if causal_contact and action == "right":
            position = min(4, position + 1)
        elif causal_contact and action in {"pickup", "use", "finish"} and position >= 2:
            position = min(4, position + 1)
        elif mode == "third_party_with_actions" and action == "right":
            position = max(0, position - 1)

        edge_learned = causal_contact and before != position
        learned_edges += int(edge_learned)
        success = position >= 4 and action in {"finish", "right", "use"}
        successes += int(success)

        noise = rng.random() * 0.02
        verifier_score = max(0.0, min(1.0, (1.0 if success else learned_edges / 5.0) - noise))
        rows.append(
            _trajectory_row(
                seed=seed,
                agent_id=f"{mode}-agent-{seed}",
                fork_id=f"{mode}-fork-{seed % 3}",
                env_id="toy-corridor-v1" if mode != "third_party_with_actions" else "toy-corridor-shifted",
                state={"t": t, "position": before},
                observation={"marker": f"room-{(before + seed) % 5}", "source_action": source_action},
                action=action,
                belief_state={"position_belief": position, "learned_edges": learned_edges},
                policy={"mode": mode, "planner_budget": 2 if mode == "self_generated" else 0},
                expected_free_energy_terms={
                    "risk": round((4 - position) / 4.0, 4),
                    "ambiguity": round(1.0 / (1.0 + learned_edges), 4),
                    "epistemic_value": round(0.2 if edge_learned else 0.05, 4),
                },
                transition_update={
                    "from": before,
                    "action": action,
                    "to": position,
                    "updated": edge_learned,
                },
                outcome={"success": success, "causal_contact": causal_contact},
                verifier_score=round(verifier_score, 4),
            )
        )

    task_success = successes / max(1, horizon)
    transition_model_error = max(0.0, 1.0 - learned_edges / 4.0)
    metrics = _blank_metrics()
    metrics.update(
        {
            "task_success": task_success,
            "sample_efficiency": learned_edges / max(1, horizon),
            "map_accuracy": min(1.0, learned_edges / 4.0),
            "object_place_binding_accuracy": 0.35 + 0.15 * learned_edges,
            "transition_model_error": transition_model_error,
            "preference_satisfaction": task_success,
            "calibration_error": abs((1.0 - transition_model_error) - task_success),
            "seed_robustness": 1.0 - (seed % 3) * 0.05,
            "compute_cost_per_success": horizon / max(1.0, successes),
            "coordination_overhead": 0.0,
            "holdout_regression": 0.05 if mode == "self_generated" else 0.45 if mode == "third_party_with_actions" else 0.2,
        }
    )
    return rows, {k: round(v, 4) for k, v in metrics.items()}


def compare_replay_modes(seeds: list[int], horizon: int = 8) -> dict[str, Any]:
    trajectories: list[dict[str, Any]] = []
    by_mode: dict[str, list[dict[str, float]]] = {mode: [] for mode in REPLAY_MODES}
    for mode in REPLAY_MODES:
        for seed in seeds:
            rows, metrics = _run_replay_mode(seed, mode, horizon)
            trajectories.extend(rows)
            by_mode[mode].append(metrics)
    return {
        "trajectories": trajectories,
        "metrics_by_mode": {mode: _mean_dict(rows) for mode, rows in by_mode.items()},
    }


def run_population_sweep(seeds: list[int], horizon: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_by_mode = {
        "isolated": 0.0,
        "shared_trajectory_archive": 0.08,
        "shared_transition_model": 0.16,
        "coordinator": 0.2,
    }
    overhead_by_mode = {
        "isolated": 0.0,
        "shared_trajectory_archive": 0.03,
        "shared_transition_model": 0.06,
        "coordinator": 0.18,
    }
    for n in [1, 2, 4, 8, 16]:
        scale = math.log2(n) / 4.0
        for mode in POPULATION_MODES:
            seed_jitter = mean((seed % 5) * 0.005 for seed in seeds)
            share_bonus = base_by_mode[mode] * scale
            overhead = overhead_by_mode[mode] * scale
            task_success = min(1.0, 0.45 + 0.2 * scale + share_bonus - overhead * 0.25)
            sample_efficiency = min(1.0, 0.25 + 0.22 * scale + share_bonus - seed_jitter)
            metrics = _blank_metrics()
            metrics.update(
                {
                    "task_success": task_success,
                    "sample_efficiency": sample_efficiency,
                    "map_accuracy": min(1.0, 0.4 + sample_efficiency * 0.6),
                    "object_place_binding_accuracy": min(1.0, 0.35 + task_success * 0.5),
                    "transition_model_error": max(0.0, 1.0 - sample_efficiency),
                    "preference_satisfaction": task_success - overhead * 0.1,
                    "calibration_error": max(0.0, 0.25 - share_bonus),
                    "seed_robustness": max(0.0, 0.7 + share_bonus - seed_jitter),
                    "compute_cost_per_success": round((n * horizon * (1.0 + overhead)) / max(0.1, task_success * n), 4),
                    "coordination_overhead": overhead,
                    "holdout_regression": max(0.0, 0.3 - share_bonus + overhead * 0.2),
                }
            )
            rows.append({"N": n, "mode": mode, "metrics": {k: round(v, 4) for k, v in metrics.items()}})
    return rows


def _planner_score(seed: int, budget: int, priors: dict[str, float] | None = None) -> float:
    prior_bonus = 0.0 if not priors else priors.get("right_then_use", 0.0)
    rng = random.Random(seed + budget * 17)
    return min(1.0, 0.25 + budget * 0.08 + prior_bonus + rng.random() * 0.04)


def run_recursive_distillation(
    train_seeds: list[int],
    holdout_seeds: list[int],
    horizon: int = 8,
) -> dict[str, Any]:
    high_budget = 8
    low_budget = 2
    high_scores = [_planner_score(seed, high_budget) for seed in train_seeds]
    selected = [score for score in high_scores if score >= 0.8]
    prior_strength = min(0.45, 0.15 * len(selected))
    priors = {"right_then_use": prior_strength}

    def evaluate(seeds: list[int], budget: int, distilled: bool = False) -> dict[str, float]:
        scores = [_planner_score(seed, budget, priors if distilled else None) for seed in seeds]
        success = mean(1.0 if score >= 0.62 else 0.0 for score in scores)
        avg_score = mean(scores)
        metrics = _blank_metrics()
        metrics.update(
            {
                "task_success": success,
                "sample_efficiency": avg_score / budget,
                "map_accuracy": avg_score,
                "object_place_binding_accuracy": avg_score * 0.9,
                "transition_model_error": max(0.0, 1.0 - avg_score),
                "preference_satisfaction": avg_score,
                "calibration_error": abs(avg_score - success),
                "seed_robustness": 1.0 - (max(scores) - min(scores) if len(scores) > 1 else 0.0),
                "compute_cost_per_success": (budget * horizon) / max(1.0, success * len(seeds)),
                "coordination_overhead": 0.0,
                "holdout_regression": max(0.0, mean(high_scores) - avg_score),
            }
        )
        return {k: round(v, 4) for k, v in metrics.items()}

    return {
        "selected_trajectories": len(selected),
        "distilled_priors": priors,
        "high_budget_planner": evaluate(holdout_seeds, high_budget),
        "baseline_low_budget": evaluate(holdout_seeds, low_budget),
        "distilled": evaluate(holdout_seeds, low_budget, distilled=True),
    }


def _world_objects(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    latents = ["bridge", "key", "hazard", "food"]
    colors = ["red", "blue", "green", "yellow", "white", "black"]
    shapes = ["cube", "ring", "spike", "disk", "bar"]
    rng.shuffle(colors)
    rng.shuffle(shapes)
    return [
        {
            "place": idx,
            "latent": latent,
            "surface": {"color": colors[idx], "shape": shapes[idx]},
            "best_action": {
                "bridge": "cross",
                "key": "carry",
                "hazard": "avoid",
                "food": "consume",
            }[latent],
        }
        for idx, latent in enumerate(latents)
    ]


def run_abstraction_experiment(train_seeds: list[int], holdout_seeds: list[int]) -> dict[str, Any]:
    latent_action: dict[str, str] = {}
    surface_action: dict[tuple[str, str], str] = {}
    for seed in train_seeds:
        for obj in _world_objects(seed):
            latent_action[obj["latent"]] = obj["best_action"]
            surf = obj["surface"]
            surface_action[(surf["color"], surf["shape"])] = obj["best_action"]

    def evaluate(use_latent: bool) -> dict[str, float]:
        attempts = 0
        correct = 0
        bindings = 0
        for seed in holdout_seeds:
            for obj in _world_objects(seed):
                attempts += 1
                surf = obj["surface"]
                if use_latent:
                    predicted = latent_action.get(obj["latent"])
                    bound = predicted is not None
                else:
                    predicted = surface_action.get((surf["color"], surf["shape"]))
                    bound = predicted is not None
                correct += int(predicted == obj["best_action"])
                bindings += int(bound)
        success = correct / attempts
        binding = bindings / attempts
        metrics = _blank_metrics()
        metrics.update(
            {
                "task_success": success,
                "sample_efficiency": success / max(1, len(train_seeds)),
                "map_accuracy": binding,
                "object_place_binding_accuracy": binding,
                "transition_model_error": 1.0 - success,
                "preference_satisfaction": success,
                "calibration_error": abs(binding - success),
                "seed_robustness": 1.0 if success >= 0.75 else 0.4,
                "compute_cost_per_success": attempts / max(1.0, correct),
                "coordination_overhead": 0.0,
                "holdout_regression": 1.0 - success,
            }
        )
        return {k: round(v, 4) for k, v in metrics.items()}

    return {"latent": evaluate(True), "surface_only": evaluate(False)}


def run_all(config_dir: str | Path | None = None) -> dict[str, Any]:
    base = Path(config_dir) if config_dir else Path(__file__).parent / "configs"
    replay = load_config(base / "replay_modes.json")
    population = load_config(base / "population_sweep.json")
    distill = load_config(base / "recursive_distillation.json")
    abstraction = load_config(base / "abstraction_barrier.json")
    return {
        "replay_modes": compare_replay_modes(replay.seeds, replay.horizon),
        "population_sweep": run_population_sweep(population.seeds, population.horizon),
        "recursive_distillation": run_recursive_distillation(distill.seeds[:3], distill.seeds[3:], distill.horizon),
        "abstraction_barrier": run_abstraction_experiment(abstraction.seeds[:3], abstraction.seeds[3:]),
    }
