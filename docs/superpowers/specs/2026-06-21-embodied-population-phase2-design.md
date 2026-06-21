# Embodied Population Loop — Design Spec (Phase 2)

- **Date:** 2026-06-21
- **Status:** Approved for implementation planning (brainstorming complete)
- **Direction:** `embodied-physics` — Phase 2 (continues the Phase-1 substrate)
- **Depends on:** Phase 1 (`embodied/` substrate: env, trained policy checkpoint, rollout, render) on
  branch `embodied-physics-substrate`.
- **Scope of THIS spec:** Phase 2 only — a **population loop on embodied bodies, NO evolution.**
  Phase 3 (heritable trait + invasion-from-rarity) and GPU/MJX scaling are sketched, not built.

---

## 1. Motivation

Phase 1 proved a single MuJoCo quadruped can *learn to walk to food* (Brax/MJX PPO) and be watched.
The research goal is downstream: **does locomotion escape the benefit-saturation wall when the body
is real?** (the wall held for scalar movement traits across the prior arc). That verdict (Phase 3)
needs a substrate that can host the binding instrument — the evolvability Preflight's
**invasion-from-rarity** test — which in turn requires a **stable population at a carrying capacity
with genuine density-dependence**. Phase 2 builds and certifies exactly that population substrate, on
embodied bodies, *before* adding any heritable trait.

**Key enabling insight** (from mapping `ecology/`): the existing engine already runs a *variable*
population (births/deaths) with energy/reproduce/die on *continuous* positions (the Exp 238+
ContinuousWorld path) — "the continuous substrate does not change the population mechanics, only the
movement and intake sub-seams." So Phase 2 is **the ContinuousWorld population model with the swept
movement path produced by a MuJoCo physics rollout** instead of `pos += heading·speed·dt`. The
genotype, the life mechanics, and the Preflight are all reusable.

---

## 2. Goals / Non-goals

### Phase 2 goals
1. A numpy population loop in the `embodied/` sibling package: N embodied bodies in a shared arena,
   each living/eating/reproducing/dying, all driven by the ONE fixed Phase-1 policy.
2. Competition mediated by a **shared depletable food field** (ContinuousWorld-style).
3. A clean `advance(body_state, policy) -> (new_body_state, swept_path)` seam — the only place
   physics enters; the place Phase 3 later swaps for MJX batching.
4. Reuse `ecology.genotype` (Genotype/founder) and `ecology.evolvability` (the Preflight) as
   instruments; **never import/modify `ecology.engine`**.
5. **Verdict:** the embodied population **certifies stable under the FROZEN Preflight stability gate**
   AND shows genuine density-dependence (per-capita intake falls as N rises). An honest NEGATIVE
   (can't be both stable and competitive) is a real result that connects to the prior arc's wall.
6. Determinism carried over (event hash); runs on the **Mac CPU** (no GPU).

### Phase 2 non-goals (deferred)
- **No mutation / evolution / heritable trait** (Phase 3) — reproduction copies the genotype verbatim.
- **No body-body physical collision** — competition is via the shared food field only (collision
  needs a shared MJX scene; deferred to a scaling phase).
- **No GPU / MJX batching** — per-creature MuJoCo native rollouts on CPU.
- **No numbered experiment / EXPERIMENTS.md entry** — substrate work; the first numbered exp is the
  Phase-3 evolvability verdict.

---

## 3. Architecture

A new set of files in the existing `embodied/` package (sibling to the frozen numpy `ecology/`
engine). **Allowed imports:** `ecology.genotype` (pure-data leaf: `Genotype`, `founder`,
`TRAIT_BOUNDS`, `mutate`) and `ecology.evolvability` (the Preflight: `TraitAxis`, the stability
`cert_run`/`certify_run`). **Forbidden imports:** `ecology.engine` (the golden-hash-pinned step).
This is the "reuse the instruments, not the frozen engine" line established in Phase 1.

```
embodied/
  world.py          # EmbodiedWorld: MuJoCo model + shared depletable food FIELD;
                    #   advance(body_state, policy) -> (new_body_state, swept_path)   <-- THE SEAM
  population.py     # the numpy life loop: founders -> step (advance+intake+metabolism+
                    #   reproduce+die) -> variable-N python list; event log + hash
  creature.py       # EmbodiedCreature: Genotype + energy + age + MuJoCo body_state (qpos/qvel)
  policy_runner.py  # wraps the Phase-1 checkpoint as a fixed forager: obs(body, food_field) ->
                    #   action; used inside advance()
embodied/run_population.py  # entrypoint: run N founders T steps, write events + stability telemetry
tests/test_embodied_population.py
```

---

## 4. Components & data flow

### 4.1 `EmbodiedWorld` (`world.py`)
- Holds the MuJoCo model (the Phase-1 arena/body) **and a shared depletable food field** — a
  sub-cell resource grid over the arena (mirror `ecology/continuous_world.py`: capacity + regen,
  `resource_at`, `consume(swept_path, deficit)`, `step_regen()`).
- **THE SEAM:** `advance(body_state, policy) -> (new_body_state, swept_path)` resets a `MjData` to
  `body_state`, runs the fixed policy for **K control steps** (each control step advances the env's
  `n_frames` physics substeps — so a bout is K·n_frames raw substeps), and returns the new body state
  + the (x,y) swept path. One bout = one "decision step" of the population loop, the embodied analog
  of the continuous engine's one-step move. Deterministic given `body_state` (+ deterministic policy).

### 4.2 `EmbodiedCreature` (`creature.py`)
- `genotype: ecology.genotype.Genotype` (founder defaults; **copied verbatim on birth** — no mutation
  in Phase 2), `energy: float`, `age: int`, `body_state` (qpos/qvel), `alive: bool`, `id`.

### 4.3 Population loop (`population.py`) — one step = one decision bout per alive creature
For each alive creature (in a deterministic/shuffle-safe order, mirroring the ecology engine):
1. `new_body_state, swept_path = world.advance(creature.body_state, policy_runner)`
2. `intake = world.consume(swept_path, deficit)` — line-integral of the food field along the path,
   **depleting the shared field** (this is where competition bites).
3. Charge metabolism from the genotype: `baseline_metabolic_cost + movement_cost + aging_cost·age`
   (reuse the ecology cost model + founder parameters).
4. `energy += intake − costs`; cap at `energy_capacity`.
5. **Reproduce** if `energy ≥ reproduction_energy_threshold` and `age ≥ maturity_age`: spawn a child
   near the parent (a fresh body at a nearby spawn pose), transfer
   `energy·reproduction_energy_transfer_fraction`, pay the overhead, **copy the genotype**; append to
   the population list.
6. **Die** if `energy ≤ 0` (starvation). (Senescence/crowding optional, OFF by default for Phase 2 —
   density-dependence should arise from food competition, not an imposed mortality.)
Then `world.step_regen()`; append birth/death/intake events to an event stream; hash it.

### 4.4 `policy_runner.py`
- Wraps the committed Phase-1 checkpoint as a fixed forager. Builds the obs the policy expects but
  with **`food_xy` = the nearest food source in the field** (so the relative-navigation policy chains
  across the depleting field). Deterministic mean-action.

### 4.5 Data flow
```
founders (N) --> [per step: advance(K substeps) -> swept path -> intake from shared field ->
                  metabolism -> reproduce/die] --> variable-N population list
              --> event stream (births/deaths/intake) + hash
              --> stability telemetry {N(t), births, per-capita intake vs N}
              --> Preflight certify_run gate  => PASS (certified-stable + density-dependent) / NEGATIVE
```

---

## 5. Competition model (shared food, no collision)
Bodies do **not** physically collide — that requires all bodies in one MJX scene, which is
incompatible with the chosen per-creature-rollout architecture and is deferred. Each creature rolls
out in its own physics; they compete **only** through the shared depletable food field: when one
depletes food along its path, less remains for others. This is exactly the ContinuousWorld
competition mechanism, and it is sufficient for density-dependence (per-capita intake ↓ as N ↑) —
the property Phase 3 needs.

## 6. Policy reuse + fallback
Reuse the Phase-1 checkpoint (it learned *relative* navigation: "walk toward the food vector in the
obs"). With obs `food_xy = nearest food in the field`, it should chain-forage across the field.
**Risk + fallback:** if it fails to generalize to a distributed/depleting field (creatures don't
forage effectively → population starves regardless of food), **retrain the policy on a food FIELD**
(a contained task using the existing `runpod/` recipe — change the env's food to a field, retrain,
re-commit the checkpoint). This is the one real technical risk; it is bounded and has a known fix.

## 7. Determinism & honesty
The per-creature MuJoCo rollout is deterministic given the body state + the deterministic policy.
Use a seeded numpy RNG for stochastic life events (birth placement, ordering). Carry the ecology
**event-hash** discipline: same seed → byte-identical event stream → identical hash. Predeclare the
verdict gate (Section 8) before running. VALIDATION.md is binding.

## 8. Verdict / success criterion (predeclared)
Phase 2 **PASS** requires BOTH:
1. **Certified stable** under the FROZEN evolvability Preflight stability gate — reuse
   `ecology.evolvability` `cert_run`/`certify_run` thresholds: persist floor **≥30**, level CV
   **≤0.15**, cross-seed CV **≤0.25**, drift **≤0.10**, no oscillation verdict.
2. **Genuine density-dependence**: per-capita intake falls measurably as N rises (the precondition
   for invasion-from-rarity).
Anything else is an honest **NEGATIVE / NEW INSIGHT** — in particular, if the embodied substrate (like
the continuous one in Exp 243–247) can be stable OR competitive but not both, that is a real result
about the substrate, logged as such, not papered over.

---

## 9. Testing & verification
`tests/test_embodied_population.py` (fast unless marked slow):
1. **Founders live & forage** — a tiny run (few creatures, short horizon): population is non-empty,
   intake > 0, energy changes, no crash.
2. **Birth & death happen** — over a run, at least one birth and one death event are logged; the
   population list grows and shrinks; `alive_count` tracks correctly.
3. **Shared-field competition** — with 2× the creatures, per-capita intake is lower (depletion bites).
4. **Determinism** — same seed → identical event hash across two runs.
5. **(slow) Stability smoke** — a longer run produces the telemetry dict the Preflight `certify_run`
   consumes (shape/keys correct), even if it doesn't yet certify.
Verification before claiming done (verification-before-completion): actually run `run_population.py`,
report N(t), the per-capita-intake-vs-N curve, and the certify_run verdict on real numbers.

## 10. Dependencies
None new — reuses Phase-1's `mujoco` (CPU native rollout) + `ecology.genotype` + `ecology.evolvability`
(already in the repo). No GPU, no MJX, no brax needed for Phase 2 (the fixed policy is loaded as
params and run; if loading it needs brax's network rebuild, that import is already available).

## 11. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Phase-1 policy doesn't generalize to a food field | Retrain on a field (bounded; `runpod/` recipe). Test #1 surfaces it early. |
| Per-creature MuJoCo rollouts too slow for the stability horizon | Keep N small + K modest for Phase 2; the `advance` seam swaps to MJX-batched in Phase 3 if needed. |
| Stability vs competition mutually exclusive (the prior wall reappears) | That is a valid NEGATIVE verdict, not a bug — log it honestly per Section 8. |
| Accidentally importing `ecology.engine` | A test asserts `embodied/` does not import `ecology.engine`. |

## 12. Roadmap beyond Phase 2
- **Phase 3 — the science:** make a heritable trait (gait/morphology) via `ecology.evolvability`
  `TraitAxis` (with `disconnect_overrides` for the null arm); run the **monomorphic benefit curve +
  invasion-from-rarity** on the certified-stable population. The verdict: *does locomotion escape the
  benefit-saturation wall with a real body?* First numbered experiment (exp269+), first EXPERIMENTS.md
  entry.
- **Scaling (when Phase 3 needs many creatures × generations × replicates):** swap the `advance()`
  seam for an **MJX-batched** advance (vmap over bodies) — life mechanics unchanged.

## 13. Open questions / assumptions to confirm in planning
1. **K (substeps per decision bout)** — start with the env's `n_frames`-scale stride; tune so a
   creature meaningfully forages per step without the loop being too slow.
2. **Food field layout** — reuse a ContinuousWorld layout (sum-of-Gaussians) + regen rate; calibrate
   founder viability against it (the founder-calibration discipline from the prior arc).
3. **Whether the Phase-1 policy generalizes** — resolved empirically by test #1; retrain if not.
