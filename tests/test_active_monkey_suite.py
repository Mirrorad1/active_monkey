"""Tests for the Active Monkey toy experiment suite."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.active_monkey_suite import (
    METRIC_NAMES,
    POPULATION_MODES,
    REPLAY_MODES,
    TRAJECTORY_FIELDS,
    ExperimentConfig,
    compare_replay_modes,
    load_config,
    run_abstraction_experiment,
    run_population_sweep,
    run_recursive_distillation,
)


def test_config_files_load_and_preserve_required_modes():
    config_dir = Path("experiments/active_monkey_suite/configs")
    config_paths = sorted(config_dir.glob("*.json"))

    assert {p.name for p in config_paths} == {
        "abstraction_barrier.json",
        "population_sweep.json",
        "recursive_distillation.json",
        "replay_modes.json",
    }

    for path in config_paths:
        config = load_config(path)
        assert isinstance(config, ExperimentConfig)
        assert config.seeds
        assert set(config.metrics).issubset(METRIC_NAMES)

    replay = load_config(config_dir / "replay_modes.json")
    assert replay.replay_modes == list(REPLAY_MODES)

    population = load_config(config_dir / "population_sweep.json")
    assert population.population_sizes == [1, 2, 4, 8, 16]
    assert population.population_modes == list(POPULATION_MODES)


def test_trajectory_schema_contains_all_required_fields():
    assert TRAJECTORY_FIELDS == (
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

    rows = compare_replay_modes(seeds=[0], horizon=6)["trajectories"]
    assert rows
    for row in rows:
        assert tuple(row.keys()) == TRAJECTORY_FIELDS
        json.dumps(row)


def test_replay_modes_separate_observation_from_intervention():
    result = compare_replay_modes(seeds=[0, 1, 2], horizon=7)
    by_mode = result["metrics_by_mode"]

    assert set(by_mode) == set(REPLAY_MODES)
    assert by_mode["self_generated"]["task_success"] > by_mode["observation_only"]["task_success"]
    assert (
        by_mode["action_conditioned"]["transition_model_error"]
        < by_mode["observation_only"]["transition_model_error"]
    )
    assert (
        by_mode["third_party_with_actions"]["holdout_regression"]
        > by_mode["self_generated"]["holdout_regression"]
    )


def test_population_sweep_covers_sizes_modes_and_coordination_tradeoff():
    rows = run_population_sweep(seeds=[0, 1], horizon=10)

    assert {(row["N"], row["mode"]) for row in rows} == {
        (n, mode) for n in [1, 2, 4, 8, 16] for mode in POPULATION_MODES
    }
    assert all(set(METRIC_NAMES).issubset(row["metrics"]) for row in rows)

    shared16 = next(
        row for row in rows if row["N"] == 16 and row["mode"] == "shared_transition_model"
    )
    isolated16 = next(row for row in rows if row["N"] == 16 and row["mode"] == "isolated")
    coordinator16 = next(row for row in rows if row["N"] == 16 and row["mode"] == "coordinator")

    assert shared16["metrics"]["sample_efficiency"] > isolated16["metrics"]["sample_efficiency"]
    assert coordinator16["metrics"]["coordination_overhead"] > shared16["metrics"]["coordination_overhead"]


def test_recursive_distillation_improves_low_budget_holdout():
    result = run_recursive_distillation(train_seeds=[0, 1, 2], holdout_seeds=[20, 21], horizon=8)

    assert result["selected_trajectories"] > 0
    assert result["distilled"]["task_success"] >= result["baseline_low_budget"]["task_success"]
    assert (
        result["distilled"]["compute_cost_per_success"]
        <= result["high_budget_planner"]["compute_cost_per_success"]
    )
    assert result["distilled"]["holdout_regression"] <= 0.25


def test_abstraction_barrier_requires_transferable_latent_categories():
    result = run_abstraction_experiment(train_seeds=[0, 1, 2], holdout_seeds=[30, 31])

    assert result["latent"]["task_success"] > result["surface_only"]["task_success"]
    assert (
        result["latent"]["object_place_binding_accuracy"]
        > result["surface_only"]["object_place_binding_accuracy"]
    )
    assert result["latent"]["holdout_regression"] < result["surface_only"]["holdout_regression"]
