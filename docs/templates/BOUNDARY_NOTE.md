# Template — BoundaryNote

A `BoundaryNote` preserves a negative result as a reusable *constraint*. It tells the next
experiment which region not to re-enter, and why. The best existing prose example is
`docs/research/local-gradient-wall.md`. Canonical form is JSON (`boundary_notes/<id>.json`).

```python
from active_loop.coalescence.schema import BoundaryNote, write_json
write_json(BoundaryNote(
    boundary_id="<kebab-id>-v0",
    source_experiments=[<int>, ...],
    failed_mechanism="<what was attempted>",
    observed_failure="<what actually happened, with numbers>",
    tested_conditions=[...],
    excluded_confounds=[...],
    implication="<what this constrains for the future>",
    next_safe_region_to_test="<the pointed escape, or 're-opens only on a human word'>",
).to_dict(), "boundary_notes/<id>.json")
```

## Field reference

| Field | Required | Meaning |
|---|---|---|
| `boundary_id` | yes | stable kebab id, versioned |
| `source_experiments` | yes | the experiments that established the wall |
| `failed_mechanism` | yes | the mechanism that did not work |
| `observed_failure` | yes | the honest failure, with the measured evidence |
| `tested_conditions` | no | the regimes/levers actually tried |
| `excluded_confounds` | no | what was ruled out (so the wall is trustworthy) |
| `implication` | no | the reusable constraint this imposes |
| `next_safe_region_to_test` | no | the one remaining escape, or "re-opens only on a word" |

**Honesty:** a negative result is a result. Do not soften `observed_failure`; do not claim a
mechanism is dead beyond the regimes actually tested. "Useful-when-gifted != locally
evolvable" is a constraint, not a failure to report.
