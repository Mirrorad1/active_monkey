# ecology — Population Ecology Simulation Substrate

Deterministic, seed-controlled gridworld ecology for active-loop Exp 194+.

## Modules

- **genotype.py** — `Genotype` dataclass, `mutate`, `is_valid`, `clamp_traits`, `founder`
- **world.py** — `GridWorld` with regenerating resource cells and local sensing
- **creature.py** — `Phenotype`, `Policy` protocol, `HomeostaticPolicy`, `Creature`
- **engine.py** — `EcologyConfig`, `Ecology` (step/run loop); NO-HIDDEN-EVALUATOR invariant
- **scenarios.py** — `SCENARIOS` dict with balanced / scarce / overabundant configs
- **recording.py** — Structured output writers (JSONL events, JSON summary/lineage/verdict, CSV traits)
- **run.py** — `run_scenario`, `determinism_check` helpers

## Key design invariants

1. **No global fitness ranking**: survival and reproduction decisions read ONLY a
   single creature's own genotype/phenotype and its local cell. The engine's
   population-level operations are: append (new creature), mark-dead, iterate.

2. **Determinism**: all RNG via `numpy.random.default_rng(seed)` with deterministically
   derived sub-seeds. Creatures processed in ascending `creature_id` order each step.
   No wall-clock or unordered set iteration enters the event stream.

3. **Policy pluggability**: `HomeostaticPolicy` implements the `Policy` protocol.
   A future pymdp value-iteration navigator can replace it with no engine changes.

## Usage

```python
from ecology.engine import Ecology
from ecology.scenarios import SCENARIOS

eco = Ecology(SCENARIOS["balanced"], seed=42)
summary = eco.run()
```

## Running Exp 194

```bash
uv run --python .venv python experiments/exp194_n5_homeostatic_population.py
```

Results written to `experiments/outputs/exp194_n5_homeostatic_population/` and
`experiments/outputs/exp194.txt`.
