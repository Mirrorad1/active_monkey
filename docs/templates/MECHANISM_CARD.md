# Template — MechanismCard

A `MechanismCard` distills a reusable scientific mechanism from one or more experiments.
Canonical form is JSON (`mechanisms/<id>/mechanism_card.json`); YAML is an optional alternate
if `pyyaml` is installed. Build it from the dataclass so it is valid by construction:

```python
from active_loop.coalescence.schema import MechanismCard, write_json
write_json(MechanismCard(
    mechanism_id="<kebab-id>-v0",
    mechanism_type="<one of MECHANISM_TYPES>",     # functional-valence-learning, hidden-state-belief,
                                                   # costed-sensing, uncertainty-gated-probing,
                                                   # costed-signaling, selection-stabilized-trait,
                                                   # transfer-invariant-abstraction, identity-self-modeling
    status="<validated|falsified|constrained|speculative|scaffold>",
    source_experiments=[<int>, ...],
    claim="<one honest sentence; conservative language only>",
    works_when=[...], fails_when=[...], required_conditions=[...],
    reusable_interface="<how the next experiment imports it>",
    inputs=[...], outputs=[...], state_requirements=[...], costs=[...],
    metrics=[...], falsifiers=[...], known_confounds=[...], next_compositions=[...],
).to_dict(), "mechanisms/<id>/mechanism_card.json")
```

## Field reference

| Field | Required | Meaning |
|---|---|---|
| `mechanism_id` | yes | stable kebab id, versioned (`...-v0`) |
| `mechanism_type` | yes | one of `MECHANISM_TYPES` |
| `claim` | yes | the reusable principle, one honest sentence |
| `status` | yes | `validated` / `falsified` / `constrained` / `speculative` / `scaffold` |
| `source_experiments` | yes | list of Exp numbers (may be empty for a pure scaffold) |
| `works_when` / `fails_when` | no | the regime boundary |
| `required_conditions` | no | what must hold (incl. the frozen scorer + controls) |
| `reusable_interface` | no | the import surface (class, checkpoint path) |
| `inputs` / `outputs` / `state_requirements` / `costs` | no | the composition contract |
| `metrics` / `falsifiers` / `known_confounds` | no | how it was judged + what could break it |
| `next_compositions` | no | which adapters/experiments come next |

**Honesty:** `status` must match the evidence. A mechanism with no run is `scaffold` or
`speculative`, never `validated`. Name the provided-vs-learned split in `required_conditions`.
Conservative terms only — see `docs/ARTIFACT_SPEC.md`.
