# MJX-Batched Embodied Substrate (epoch-hybrid) — Design Spec

- **Date:** 2026-06-21
- **Status:** Approved for implementation planning (brainstorming complete)
- **Direction:** `embodied-physics` — Phase 3a (scaling); unblocks the Phase-2.5 calibration sweep
  and Phase 3 (locomotion-evolvability).
- **Depends on:** Phase 1 (env + trained checkpoint) and Phase 2 (food field, life mechanics, certify,
  the `advance` seam) on branch `embodied-physics-substrate`.
- **Scope of THIS spec:** the batched substrate + its validation. The calibration sweep that USES it
  is the first follow-on run, not part of this spec.

---

## 1. Motivation

Phase 2's population loop runs each creature's physics bout **sequentially** on CPU. The Phase-2.5
preflight showed that at a *rich* food field the population thrives (~150 bodies, sustains far above
the FROZEN persist-floor of 30) — so a stable embodied population **is** reachable — but one cell, one
seed, horizon 120 took ~**635 s**, and a real calibration sweep (intermediate richness × seeds ×
horizon 300, pops 30–150) is **many hours** and memory-risky on CPU. The per-creature CPU rollout does
not scale to the population sizes the stability floor requires.

The Phase-2 `advance(state) -> (new_state, path)` seam was **designed to be swapped** for an
MJX-batched advance. This spec builds that swap: `vmap` the bout across a fixed-capacity body buffer so
hundreds of bodies advance in one GPU call. This unblocks (a) the Phase-2.5 question — is there an
**intermediate** calibration that is **stable *and* competitive** (the prior arc's
"stability-vs-strong-competition" tension, now bracketed between collapse and runaway), and (b) Phase 3
evolvability, which needs a stable population at scale.

---

## 2. Goals / Non-goals

### Goals
1. A **fixed-capacity** batched body buffer (`MAX_POP` slots: a batched brax pipeline_state `[MAX_POP,…]`
   + an `alive` bool mask `[MAX_POP]`), with births filling free slots and deaths freeing slots.
2. `advance_batch(batched_state, targets, alive) -> (new_batched_state, swept_paths)` that **`vmap`s**
   the existing per-body bout (policy obs → action → `pipeline_step`, K control steps) across all
   slots in one jit-compiled call.
3. A batched population loop that keeps the **food field and life-loop in numpy** (reusing Phase-2's
   `FoodField` and the `life_step` energy/reproduce/die arithmetic), processing the batched results +
   doing slot management between bouts.
4. **The correctness gate:** a non-`vmap` **sequential reference** (same per-bout-target model) that the
   batched `vmap` version must match **byte-identically** (same seed/config, small `MAX_POP`).
5. Runs on **Mac CPU** for dev/test (small `MAX_POP`) and **GPU** for the real sweep (`MAX_POP` 256–512)
   — same code.
6. Determinism (event-hash) carried over.

### Non-goals (deferred)
- **No moving the life-loop or food field into JAX** (that was the rejected "full-JAX" option). They
  stay numpy.
- **No per-control-step food targets** (would force the field onto the GPU). Targets are **per-bout-fixed**.
- **No evolution** (still Phase 2's "no mutation" — genotype copied verbatim; evolution is the later
  Phase 3 science).
- **The calibration sweep + the Phase-2.5 stability verdict are NOT in this spec** — they are the first
  run on the finished substrate.
- No replacement of Phase-2's `population.py`/`world.py` loop — the batched modules live **alongside**.
  The ONE permitted touch to Phase-2 code is a behavior-preserving DRY extraction of the
  energy/reproduce/die arithmetic into a shared `embodied/life_mechanics.py` that BOTH `population.py`
  and `batched_population.py` import — allowed only if Phase-2's existing determinism `events_hash`
  test stays **byte-identical** (regression-guarded). If that extraction risks changing Phase-2's
  hash, instead leave `population.py` untouched and have the helper used only by the batched loop.

---

## 3. Architecture

New modules in `embodied/` (siblings; reuse Phase-2's `foodfield`, the `life_step` arithmetic, the
`PolicyRunner`, and `ecology.evolvability` certify; never import `ecology.engine`).

```
embodied/
  batched_world.py        # BatchedEmbodiedWorld: the MAX_POP body buffer + advance_batch (vmap),
                          #   spawn_into_slot(idx, xy), and a SEQUENTIAL reference advance for the test
  batched_population.py    # the slot-based population loop: targets -> advance_batch -> numpy life-loop
                          #   + slot fill/free; reuses FoodField + life arithmetic; events_hash
  run_batched.py           # CLI: run a batched population (+ certify), CPU or GPU
```

### 3.1 `BatchedEmbodiedWorld` (`batched_world.py`)
- Holds one `EmbodiedForageEnv` (Phase-1 physics) and `MAX_POP`.
- `init_buffer(n_founders, seed) -> (batched_state, alive)`: build a `[MAX_POP]` batched pipeline_state
  (founders spread across the arena in the first `n_founders` slots, the rest dead/placeholder) + an
  `alive` mask. (Use `jax.vmap(env.reset)` over `MAX_POP` keys, then set each slot's root xy via
  `state.q.at[slot, 0:2].set(xy)`.)
- **`advance_batch(batched_state, targets, alive) -> (new_batched_state, paths)`** — the GPU seam:
  `jax.vmap` over slots of a single-body bout that, for K control steps, builds the obs
  `concat(q[2:], qd, target - q[0:2])`, runs the deterministic policy, and `pipeline_step`s; records
  per-step torso xy. `targets` is `[MAX_POP,2]` (fixed for the bout). Dead slots are advanced too
  (cheap on GPU) and ignored by the numpy step; `alive` is passed so a future optimization can mask
  them, but correctness does not depend on masking dead slots (they're never read). jit the whole thing.
- **`spawn_into_slot(batched_state, idx, xy, seed)`** — reset one slot's body and set its root xy
  (`state.q.at[idx,0:2].set(xy)`), returning the updated batched_state (`.at[idx].set(...)` on the
  pytree).
- **`advance_sequential(batched_state, targets, alive)`** — a plain python-loop reference (NO `vmap`)
  that advances each slot's body one at a time with the SAME per-bout-target logic. Exists ONLY for the
  equivalence test (sec 6); not used in production.

### 3.2 `batched_population.py`
- Slot-aligned numpy arrays of length `MAX_POP`: `genotype[]`, `energy[]`, `age[]`, `id[]`, plus the
  `alive` mask and the batched body state. (Genotypes reuse `ecology.genotype`.)
- Per step:
  1. (numpy) `targets[i] = food.nearest_food_xy(x_i, y_i)` for alive slots; dead slots get any
     placeholder (their advance is ignored).
  2. (GPU) `new_state, paths = world.advance_batch(state, targets, alive)`.
  3. (numpy, **sorted by id over the shared field**) for each alive slot: `intake =
     food.consume(paths[i], deficit_i)`; metabolism; energy update; mark reproduce/die — reusing the
     Phase-2 `life_step` arithmetic (extracted into a shared helper so the formulas are identical).
  4. (numpy) **slot management:** deaths set `alive[i]=False`; births take a free slot index `j`
     (`alive[j]==False`), `state = world.spawn_into_slot(state, j, child_xy, child_seed)`, set
     `genotype[j]=parent_genotype` (verbatim copy), `energy[j]=transfer`, `age[j]=0`, `alive[j]=True`.
     If no free slot (population would exceed `MAX_POP`), the birth is dropped and a `capped` counter
     is incremented (logged loudly — `MAX_POP` must be set above the expected carrying capacity).
  5. (numpy) `food.step_regen()`; append `id:event` per alive slot (sorted) to the running sha256;
     record N (alive count), per-capita raw intake.
- Returns the same `PopResult` shape Phase 2 used (so `certify()` from `run_population.py` works
  unchanged).

---

## 4. Data flow
```
init_buffer(n_founders) -> (batched_state[MAX_POP], alive[MAX_POP])
 per step:
   numpy: alive slots -> nearest_food targets[MAX_POP,2]
   GPU:   advance_batch(state, targets, alive) -> (state', paths[MAX_POP, bout, 2])
   numpy (sorted-id, shared field): consume->intake, metabolism, energy, reproduce/die
   numpy: slot fill (births) / free (deaths); food.step_regen(); hash; record N, per-capita intake
 -> PopResult -> certify() (FROZEN stability gate, unchanged from Phase 2)
```

## 5. The per-bout-fixed-target variant (explicit)
Because the food field stays numpy, the nearest-food **target is computed once per bout** and held
fixed across the bout's K control steps (Phase 2 recomputed it per control step). This makes the
batched substrate a **faithful variant**, NOT byte-identical to Phase 2. `bout_steps` is the
target-staleness ↔ GPU-efficiency knob (longer bout = more GPU efficiency, staler target). The science
(stability, competition) is insensitive to this minor approximation over a short bout; it is documented
so no one expects batched ≡ Phase-2 numbers.

## 6. Determinism & the correctness gate (load-bearing)
- **Determinism:** `vmap`'d deterministic (mean-action) policy + fixed per-slot reset keys + jit +
  seeded numpy life-loop + sorted-id eating order ⇒ same seed → byte-identical `events_hash`.
- **Equivalence gate (the proof the `vmap` is correct):** `advance_sequential` (plain loop, same
  per-bout-target model) and `advance_batch` (`vmap`) must produce **byte-identical** batched states +
  paths for the same inputs (small `MAX_POP`, rounded to ~5 dp to guard last-bit jitter). And the FULL
  batched population run must byte-match a sequential-reference population run (same seed/config). This
  proves no cross-slot contamination, correct obs/target indexing, and correct slot fill/free. **This
  is the single most important test** — a `vmap` bug that mixes bodies would otherwise be invisible.

## 7. Where it runs
- **Mac CPU (dev/test):** small `MAX_POP` (e.g. 16–64), short horizons — `jax.vmap` runs on CPU. The
  equivalence + determinism + slot tests all run here, fast.
- **GPU (RunPod, the sweep):** `MAX_POP` 256–512, full horizons — the `runpod/` recipe extends to run
  `run_batched.py`. JAX on GPU `vmap`s the bodies; the numpy life-loop runs on the box's CPU between
  bouts. (Carry-forward from Phase 1: single-GPU; the device_put_replicated shim is single-device.)

## 8. Testing
`tests/test_batched_substrate.py`:
1. **Buffer init** — `init_buffer` returns a `[MAX_POP]` batched state + alive mask; founders in the
   first `n_founders` slots at their xy.
2. **advance_batch == advance_sequential** (byte-identical, small `MAX_POP`) — the equivalence gate.
3. **Determinism** — same seed → identical population `events_hash` across two batched runs.
4. **Slot reuse** — after a death frees slot j and a later birth, the birth lands in a free slot;
   `alive` count stays correct; no slot is double-occupied.
5. **MAX_POP cap is loud** — if births would exceed `MAX_POP`, the drop is counted + surfaced (no
   silent loss).
6. **Qualitative bracket (slow, small scale)** — a poor-field run trends to collapse and a rich-field
   run trends to growth (the Phase-2.5 bracket), confirming the batched dynamics match the substrate.
Verification before "done" (verification-before-completion): run a small batched population on CPU,
show the equivalence test passing + a non-degenerate N(t).

## 9. Risks & mitigations
| Risk | Mitigation |
|---|---|
| `vmap` mixes bodies / wrong indexing | The byte-identical equivalence gate (sec 6) catches it. |
| Slot fill/free bug (double-occupancy, leak) | Dedicated slot-reuse test (sec 8.4) + `alive`-count invariants. |
| `MAX_POP` too small → silent birth loss | Loud `capped` counter + test (sec 8.5); set `MAX_POP` above the preflight's observed carrying capacity (~150 → use 256–512). |
| Batched state pytree edits (`.at[idx].set`) differ on the pinned brax | Probe the batched pipeline_state structure once (as in Phase 1/2); report the accessor used. |
| GPU determinism differs from CPU | Hashes are per-device (documented in Phase 2); compare within one machine only. |

## 10. Roadmap (after this substrate)
- **First use — the Phase-2.5 calibration sweep on GPU:** intermediate field richness × seeds × horizon
  → does an **intermediate** cell certify **stable** (FROZEN gate) **and** competitive
  (density-dependent)? Resolves the bracketed stability-vs-competition question. (Its own run/entry.)
- **Phase 3 — evolvability:** make a heritable gait/morphology trait via `ecology.evolvability`
  `TraitAxis` and run invasion-from-rarity on a certified-stable batched population — the
  locomotion-evolvability verdict. First numbered experiment.

## 11. Open questions / assumptions to confirm in planning
1. **`MAX_POP` default** — start 256 (above the preflight's ~150); make it config + the loud cap.
2. **Batched pipeline_state pytree editing** — confirm `jax.vmap(env.reset)` stacks cleanly and
   `.at[slot].set(...)` works on the stacked state (probe; same discipline as Phase 1/2).
3. **Shared `life_step` arithmetic** — extract Phase-2's energy/reproduce/die formulas into a shared
   helper used by BOTH `population.py` (unchanged behavior) and `batched_population.py`, so the two
   substrates compute identical per-creature economics. (Keep Phase 2 byte-identical — add the helper
   without changing its results; a regression test guards Phase-2's existing hash.)
