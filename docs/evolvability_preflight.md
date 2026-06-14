# Evolvability Preflight — Developer Guide

## What it is and WHY it exists

The Evolvability Preflight is a battery of simulation experiments that answers
**one binding question before committing to a full multi-thousand-step evolution run**:

> Does the trait have a positive LOCAL selection gradient — i.e. can a single-step
> mutant invade the resident population by natural selection?

This question was formalised in **Exp 203** (sense-gradient sweep) and the
**L22/L28 lessons** that fell out of it: a gifted/cost-off benefit (B_high > B_low)
and a monomorphic/global optimum above the resident are *necessary but not
sufficient* for evolvability.  What determines whether a trait can evolve by
cumulative small steps is the **local pairwise invasion gradient** — whether a
mutant a small distance above the resident can invade.

**Exp 207** (corner-grid cross-partial) extended the framework to two-trait
landscapes (sensor h + controller theta), identifying the pattern where neither
trait pays alone but the combination is synergistic (joint-valley plausible).

The Preflight generalises both experiments into a reusable config-driven harness.

---

## Architecture

```
ecology/evolvability/
  config.py      — PreflightConfig, ControllerAxis, load_config / from_yaml / from_json
  trait_axis.py  — TraitAxis, THERMOSENSE_AXIS, make_axis
  metrics.py     — pure numeric helpers (corner_effects, count_wins, default_thresholds, …)
  verdicts.py    — verdict enums + decision functions (gradient, benefit, invasion, aggregate)
  gates.py       — engine-coupled gate runners (GateOutcome, build_base_cfg, run_*)
  runner.py      — run_preflight() → PreflightResult (orchestration)
  report.py      — build_report(result, cfg) → Markdown string
  io.py          — filesystem helpers (new_run_dir, write_json, write_jsonl, …)
  __init__.py    — public API re-exports
  __main__.py    — CLI entry point (python -m ecology.evolvability)
```

Dependency order (strict; no cycles):
```
trait_axis → metrics → verdicts → config → io → gates → runner → report
```
`metrics`, `verdicts`, `trait_axis`, `io` are pure (no engine imports).
`gates` and `runner` import the engine.

---

## Gates

### Gate A — Gifted Benefit (`gifted_benefit`)

**Question**: Does the trait provide an installed benefit when the cost is waived?

**Measurement**: Run B(h_low) and B(h_high) with `enable_thermosense=False` (cost off).
A positive mean_delta = mean(B_high - B_low) returns a `BENEFIT` verdict.

**What it proves**: The trait CAN be useful when free.

**What it does NOT prove**: That the trait is evolvable.  A cost-free benefit
says nothing about whether a self-paying mutant can invade in competition.

Backend: thermosense only (cost-off percept path is engine-specific).

---

### Gate B — Monomorphic Sweep (`monomorphic_sweep`)

**Question**: Where does the monomorphic N*(h) landscape peak relative to the resident?

**Measurement**: For each h in a grid, run a monomorphic population and measure
mean carrying capacity N*(h).  Report the global optimum h* = argmax N*(h) and
whether it is above the resident value.

**What it proves**: The shape of the monomorphic fitness landscape.

**What it does NOT prove**: LOCAL evolvability.  A global optimum above the
resident may be separated from the resident by a fitness valley that mutants
cannot cross.  Local invasion assays (Gates C/D) are required.

Backend: thermosense only.

---

### Gate C — Local Pairwise Gradient (`local_pairwise_gradient`) **[BINDING]**

**Question**: Does the mutant (h_mut) invade the resident (h_res) in head-to-head
competition?

**Measurement**: Run h_res vs h_mut in equal-count mixed populations for each
seed.  Record the invader fraction at end of window.  A win = invader_fraction > 0.5.

**Verdict**: `GradientVerdict` — POSITIVE_LOCAL_GRADIENT, NEGATIVE_LOCAL_GRADIENT,
FLAT_OR_NOISY, or NO_VERDICT.

**What it proves**: The direction of the local selection gradient between resident
and mutant.  This is **the binding criterion** for local evolvability.

**What it does NOT prove**: Stable long-run organ evolution (many-step dynamics,
frequency dependence, drift, epistasis, etc.).

Backend: **generic** (Phase 2.5).  thermosense delegates to `sense_axis.run_pairwise_competition`
(the proven Exp 203 instrument); any other backend uses `_run_pairwise_generic`, an
equal-count common garden built on `axis.clamp_founder` / `axis.get` / the freeze
mechanism — so the one gate that actually decides evolvability works for future traits
(memory_horizon, belief_persistence, …) without `NotImplementedError`.

---

### Gate D — Invasion from Rarity (`invasion_from_rarity`)

**Question**: Can a RARE mutant (~5%) increase in frequency when starting from rarity?

**Measurement**: Seed ~5% mutants into a resident background, measure frequency
increase over the window.

**Verdict**: `InvasionVerdict` — INVADES, DOES_NOT_INVADE, FLAT_OR_NOISY, NO_VERDICT.

**What it proves**: A more stringent test than equal-count pairwise — the mutant
must increase when genuinely rare, as required by adaptive dynamics.

Backend: generic (uses axis.clamp_founder + freeze_flag).

---

### Gate E — Density-Independent Growth (`density_independent_growth`)

**Question**: Does the trait improve the intrinsic growth rate at low density?

**Measurement**: measure r(h_low) and r(h_high) from small monomorphic seeds at
low density.

**Verdict**: `BenefitVerdict` — BENEFIT, NO_BENEFIT, AMBIGUOUS.

**What it proves**: Distinguishes frequency-dependent from frequency-independent
selection.  delta_r > 0 means the trait helps intrinsically.

Backend: thermosense only.

---

### Gate F — Cost Sensitivity (`cost_sensitivity`)

**Question**: How does the pairwise gradient change across different cost (inefficiency)
levels?

**Measurement**: Sweep over cost_values, re-running the pairwise gradient for
each cost.  Report sign_change_cost and unaffordable_cost.

This is a **diagnostic gate** — no pass/fail verdict.

Backend: thermosense only.

---

### Gate G — Null Guards (`null_guards`) [ANTI-CHEAT]

**Question**: Do all null/anti-cheat guards pass?

Seven guards, each returning PASS/FAIL/NOT_IMPLEMENTED/NA:

1. **cost_off_disconnected_byte_identical** — when the trait is fully disconnected
   (enable_flag + all disconnect_overrides), the event hash must be byte-identical
   for h_low vs h_high.  If not, h is feeding into the simulation through an
   undeclared path.

2. **no_direct_h_reward** — asserts that the engine does not write fitness or
   food as f(h) directly (thermosense + freeze_flag only).

3. **trait_disabled_null** — per-capita intake must be identical when the trait
   is disconnected.

4. **population_validity** — pairwise extinct_fraction < 0.5 (requires local
   gradient gate to have run first).

5. **perfect_percept_null** — NOT_IMPLEMENTED unless enable_niche=True is active
   (niche_confusion=0 knob).

6. **frozen_memory_map** — NOT_IMPLEMENTED (documented; requires freeze_learning_rate).

7. **shuffle_order** — PASS iff shuffle_creature_order=True (neutralises id-order
   eat-first confound).

**aggregate["all_pass"]** counts only PASS/FAIL — NOT_IMPLEMENTED/NA do not fail.

---

### Gate H — Controller Cross-Partial (`controller_cross_partial`)

**Question**: Do trait h and controller theta interact synergistically?

**Measurement**: 2×2 corner grid of births/step at (h_lo, theta_lo),
(h_hi, theta_lo), (h_lo, theta_hi), (h_hi, theta_hi).  Compute corner_effects.

**Verdict**: `CrossPartialVerdict` — JOINT_VALLEY_PLAUSIBLE, CONTROLLER_PAYS_ALONE,
TRAIT_PAYS_ALONE, NO_INTERACTION, ANTAGONISTIC, NO_VERDICT.

Requires `cfg.controller` to be set (a `ControllerAxis`).  If controller is None
and this gate is listed, it is skipped with a note.

ANTAGONISTIC means a **destructive interaction** (`cross_partial < 0`: h becomes
*more* harmful as θ rises), NOT merely "h is a cost at high θ" — h being a uniform
cost while θ carries the benefit (cross_partial ≈ 0) is `CONTROLLER_PAYS_ALONE`.

When a config runs Gate H **without** the binding `local_pairwise_gradient` gate,
the aggregate cannot PASS/FAIL local evolvability; instead it surfaces the
controller verdict directly (`gradient=None` path), so a `CONTROLLER_PAYS_ALONE`
finding is reported rather than masked as `NO_VERDICT`.

Reference config: `experiments/configs/preflight/thermosense_controller_crosspartial.yaml`
(the Exp 206/207 niche regime). It reproduces Exp 207's finding —
`CONTROLLER_PAYS_ALONE`: θ (niche_weight) carries the benefit (+0.147), h is a
uniform cost at both θ, cross_partial ≈ +0.005 — so a full sensor–controller
co-adaptation batch is **not** warranted. Note the trait's `disconnect_overrides`
for the niche regime is `{enable_thermosense: false, niche_confusion: 0.0}` (a
*different* recipe than the forage regime — the channels a trait feeds are
regime-specific).

Backend: generic.

---

## Verdict Taxonomy

```
AggregateVerdict
  PASS_LOCAL_GRADIENT     — local gradient is positive AND all guards pass
  FAIL_LOCAL_GRADIENT     — gradient is negative/flat; no compensating signal
  GLOBAL_BENEFIT_ONLY     — benefit or mono optimum above resident, but no local gradient
  CONTROLLER_PAYS_ALONE   — controller drives the benefit; trait h is not the causal agent
  NO_EFFECT               — no benefit and no optimum above resident
  NO_VERDICT              — population invalid / insufficient data / null guard failure
```

Precedence (in `verdicts.aggregate_verdict`):
1. NO_VERDICT if gradient gate returns NO_VERDICT
2. PASS_LOCAL_GRADIENT if gradient=POSITIVE and guards_all_pass
3. NO_VERDICT if gradient=POSITIVE but guards_all_pass=False (suspected artifact)
4. CONTROLLER_PAYS_ALONE if crosspartial=CONTROLLER_PAYS_ALONE
5. GLOBAL_BENEFIT_ONLY if mono above-resident + survivable, or gifted=BENEFIT
6. NO_EFFECT if gifted=NO_BENEFIT and mono not above resident
7. FAIL_LOCAL_GRADIENT (residual)

---

## Output Layout

```
results/preflight/<slug>/<run_id>/
  config.json          — full PreflightConfig serialised
  config_hash.txt      — SHA-256 of config for reproducibility
  git_commit.txt       — HEAD commit at run time
  summary.json         — PreflightResult serialised (all gates + aggregate)
  summary.csv          — one row per gate, flat key numbers
  report.md            — human-readable Markdown report
  raw/
    <gate_name>.jsonl  — one row per observation (seed/h/corner), augmented
                          with slug / run_id / config_hash
```

The `raw/` sub-directory is kept separate from derived summaries.
The NO OVERWRITE policy (`io.new_run_dir`) raises `FileExistsError` rather
than silently clobbering an existing run.

---

## How to Run

### YAML config (canonical — the real run)

```sh
uv run --python .venv python experiments/run_preflight.py \
    --config experiments/configs/preflight/thermosense_local_gradient.yaml \
    [--run-id my-run-1] \
    [--output-dir results/preflight]
```

This runs the 8-seed batch (the strict 7/8 convention) and takes several minutes
(the monomorphic sweep dominates). It reproduces the closed Exp 203–207 finding:
the thermosense local gradient is **not** positive → verdict `NO_EFFECT`
(5/8 wins < 7/8). A committed example run is in
`experiments/outputs/preflight_thermosense_8seed/`.

### JSON config (CLI plumbing / CI only — NOT a scientific run)

```sh
uv run --python .venv python experiments/run_preflight.py \
    --config experiments/configs/preflight/thermosense_smoke.json \
    --output-dir /tmp/smoke \
    --run-id test1
```

⚠️ **The smoke config uses only 2 seeds**, so its win threshold is 2 — two
chance-wins spuriously read `PASS_LOCAL_GRADIENT`. It exists to exercise the CLI
and artifact plumbing, **not** to produce a meaningful verdict. With fewer than 8
seeds a `PASS` is noise; always use the 8-seed batch for a real verdict.

### Controller cross-partial (Gate H)

```sh
uv run --python .venv python experiments/run_preflight.py \
    --config experiments/configs/preflight/thermosense_controller_crosspartial.yaml
```

Runs the Exp 206/207 niche regime (5 seeds, horizon 3500) → `CONTROLLER_PAYS_ALONE`
(θ pays alone, h pure cost, cross_partial ≈ 0). A committed example run is in
`experiments/outputs/preflight_thermosense_gateH/`.

### Module form

```sh
uv run --python .venv python -m ecology.evolvability \
    --config experiments/configs/preflight/thermosense_smoke.json \
    --output-dir /tmp/smoke \
    --run-id test1
```

### Programmatic

```python
from ecology.evolvability import PreflightConfig, THERMOSENSE_AXIS, run_preflight

cfg = PreflightConfig(
    slug="my_run",
    base_scenario="balanced",
    base_overrides={...},
    founder_overrides={"temperature_tolerance": 0.10},
    trait=THERMOSENSE_AXIS,
    seeds=(38, 39),
    horizon=200,
    gates=("local_pairwise_gradient", "null_guards"),
    output_dir="/tmp/preflight",
)
result = run_preflight(cfg, run_id="r1")
print(result.aggregate_verdict)
```

---

## HOW TO ADD A NEW TRAIT AXIS

1. **Create a TraitAxis** — all fields must be set explicitly:
   ```python
   from ecology.evolvability.trait_axis import TraitAxis

   MY_AXIS = TraitAxis(
       name="my_trait",
       resident_value=0.10,
       mutant_value=0.15,
       low_value=0.0,
       high_value=0.60,
       h_trait="my_trait_intensity",       # EcologyConfig field for the trait value
       enable_flag="enable_my_trait",      # flag that turns the trait on/off
       freeze_flag="freeze_my_trait",      # flag that freezes mutation (or None)
       inefficiency_trait="my_trait_ineff",  # cost field (or None)
       inefficiency_value=0.20,
       cost_floor=0.0,
       cost_inefficiency=0.20,
       active_threshold=0.05,
       backend="my_trait",                 # controls which gates are available
       # CRUCIAL: list every config field that makes the trait causally inert
       # so the byte-identity anti-cheat guard (Guard 1) can disconnect it fully.
       disconnect_overrides={
           "enable_my_trait": False,
           "enable_my_trait_coupling": False,
           "my_trait_weight": 0.0,
           # ... any other fields that route the trait signal
       },
   )
   ```

   The `disconnect_overrides` dict is critical: it must list **every** config
   field that makes the trait causally inert.  If the trait can still affect
   the simulation through an undeclared path when these are applied, Guard 1
   (byte-identity check) will FAIL, which is the correct outcome — it means
   the disconnect recipe is incomplete.

2. **Register the axis** — add it to `BUILTIN_AXES` in `trait_axis.py` so
   `make_axis("my_trait")` works:
   ```python
   BUILTIN_AXES["my_trait"] = MY_AXIS
   ```

3. **Engine support** — the trait needs:
   - A freeze mechanism: either a `freeze_flag` in EcologyConfig that makes the
     trait breed true, OR set `mutation_rate=0` if no freeze flag exists.  Gates
     that require frozen evolution use `_freeze_kwargs(axis)` which handles both.
   - A generic cost-off percept path for Gates A, B, E, F (gifted/monomorphic/
     density-independent/cost-sensitivity).  Today these gates are thermosense-only
     because they delegate to `sense_axis.py` functions
     (`run_installed_benefit`, `run_carrying_capacity`, `run_intrinsic_growth`,
     `run_pairwise_competition`).  For a new trait, you would need to implement
     equivalent functions that measure the trait's benefit with the cost switched
     off, or add a generic pathway in `gates.py`.

4. **Gates available for any backend** — Gate C (local_pairwise_gradient, **the
   binding gate**), Gate D (invasion_from_rarity), and Gate H (controller_cross_partial)
   use `axis.clamp_founder` / `axis.get` and the freeze mechanism and work for any
   backend.  Gate G (null_guards) is generic except for Guard 2 (`no_direct_h_reward`)
   which requires thermosense + freeze_flag.  Gates A/B/E/F (gifted/monomorphic/
   density/cost) still delegate to `sense_axis.py` (thermosense only) — a new trait
   needs a generic cost-off percept path for those, but **the binding gate C does not
   block a new trait**.

5. **Write a config file** — see
   `experiments/configs/preflight/thermosense_local_gradient.yaml` as a template.
   Set `trait: my_trait` and adjust `base_overrides` to enable the trait.

6. **Add tests** — at minimum: a tiny smoke run (horizon<=200, 2 seeds) that asserts
   the artefact layout is correct and that the aggregate_verdict is a valid
   AggregateVerdict value.
