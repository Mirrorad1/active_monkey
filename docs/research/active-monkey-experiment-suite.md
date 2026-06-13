# Active Monkey Toy Experiment Suite

This suite adds small, reproducible mechanism tests around replay, fork scaling,
distillation, and grounded abstraction. The framing is conservative: the tests are
toy-scale probes of causal structure in this repo's active-inference-inspired setting.
They are not presented as evidence for capacities outside these toy mechanisms.

## Common Trajectory Schema

Every replay row uses the same JSON-serializable fields:

```text
seed, agent_id, fork_id, env_id, state, observation, action, belief_state,
policy, expected_free_energy_terms, transition_update, outcome, verifier_score
```

This makes first-person rollouts, copied traces, and third-party traces comparable
without treating them as the same causal evidence.

## Replay And Intervention

`compare_replay_modes` runs four modes:

- `observation_only`: sees observations but does not condition transition updates on action.
- `action_conditioned`: replays actions with state/action/next-state updates.
- `self_generated`: acts in the world with its own low-budget policy.
- `third_party_with_actions`: sees another run's observations and actions in a shifted world.

The intended toy question is whether copied experience alone supplies the causal evidence
that intervention supplies. The expected conservative reading is that third-person traces
can help with bookkeeping but do not replace contact with the transition surface.

## Population/Fork Scaling

`run_population_sweep` covers `N = 1, 2, 4, 8, 16` across:

- `isolated`
- `shared_trajectory_archive`
- `shared_transition_model`
- `coordinator`

The metrics report success, sample efficiency, model error, seed robustness, compute cost,
and coordination overhead. This is a scaling smoke test, not a claim that larger groups
necessarily improve capability.

## Recursive Distillation

`run_recursive_distillation` runs a high-budget planner on training seeds, selects verified
successful trajectories, distills them into a tiny prior, and evaluates a low-budget planner
on held-out seeds/worlds. The output compares high-budget, low-budget baseline, and distilled
low-budget metrics, including holdout regression and compute cost per success.

## Abstraction Barrier

`run_abstraction_experiment` uses toy worlds where superficial raw features vary by seed,
while hidden affordance classes determine useful actions. A surface-only learner memorizes
feature/action pairs; the latent learner binds interaction outcomes to transferable classes.
The check is whether grounded latent categories transfer better on held-out worlds.

## Metrics

All suite arms expose the same metric names:

```text
task_success, sample_efficiency, map_accuracy, object_place_binding_accuracy,
transition_model_error, preference_satisfaction, calibration_error, seed_robustness,
compute_cost_per_success, coordination_overhead, holdout_regression
```

These names are shared across configs so later Loop B experiments can choose a subset
without changing downstream parsers.
