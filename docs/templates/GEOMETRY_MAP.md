# Template — GeometryMap

A `GeometryMap` records where in parameter space a mechanism works or fails. Many active_monkey
findings are "worked/failed only after trying" — so the reusable artifact is the *boundary*,
not a single result. Canonical form is JSON (`geometry_maps/<id>.json`).

```python
from active_loop.coalescence.schema import GeometryMap, write_json
write_json(GeometryMap(
    geometry_id="<kebab-id>-v0",
    source_experiments=[<int>, ...],
    mechanism_id="<the mechanism this maps, optional>",
    swept_parameters={"<param>": "<range/values>", ...},
    fixed_parameters={"<param>": "<value>", ...},
    metrics=[...],
    outcome_regions=[{"region": "<which corner>", "outcome": "<what happened>"}, ...],
    thresholds={"<name>": "<value/meaning>"},
    positive_regions=[...], negative_regions=[...],
    confounds_excluded=[...],
    next_experiment_suggestions=[...],
).to_dict(), "geometry_maps/<id>.json")
```

## Field reference

| Field | Required | Meaning |
|---|---|---|
| `geometry_id` | yes | stable kebab id, versioned |
| `source_experiments` | yes | the experiments that traced the surface |
| `mechanism_id` | no | the mechanism this geometry belongs to |
| `swept_parameters` / `fixed_parameters` | no | the axes vs the held constants |
| `metrics` | no | what was measured per region |
| `outcome_regions` | no | list of `{region, outcome}` — the heart of the map |
| `thresholds` | no | the zero line / gates (e.g. a constant-response ceiling) |
| `positive_regions` / `negative_regions` | no | where the mechanism pays vs doesn't |
| `confounds_excluded` | no | which artifacts were ruled out (drift, cost, granularity) |
| `next_experiment_suggestions` | no | the pointed escape, if any |

**Honesty:** an empty `positive_regions` with a populated `negative_regions` is a legitimate,
valuable artifact (a wall). Record the confounds you excluded so the boundary is trustworthy.
