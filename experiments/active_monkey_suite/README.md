# Active Monkey Toy Experiment Suite

This directory contains small deterministic probes for four mechanisms:

- digital replication and replay;
- the causal limits of third-person traces compared with first-person intervention;
- population and fork scaling under several sharing regimes;
- recursive distillation from expensive planning into cheaper priors;
- grounded abstraction discovery when raw features do not transfer.

Run from the repo root:

```bash
uv run python -m experiments.active_monkey_suite --output experiments/outputs/active_monkey_suite.json
```

The suite is intentionally toy-scale. It produces reproducible metrics and trajectory
records for mechanism checks, not broad claims beyond the toy mechanisms under test.
