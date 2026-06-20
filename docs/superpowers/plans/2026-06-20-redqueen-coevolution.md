# Red Queen Co-evolution (predator-prey) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a gated spatial predator-prey mechanism in `ecology/` and test whether prey escape-`locomotor_speed` invades from rarity under CO-EVOLVING vs STATIC predators (co-evolution = the single causal bit), gated by a six-prong Red-Queen-vs-drift discriminator.

**Architecture:** All predation behind `enable_predation` (byte-identical OFF). When ON, the per-creature loop becomes three explicit phases (A move-all → B capture-resolve on post-move positions → C eat/reproduce/die), positions read from a once-per-step frozen snapshot for order-independence. A new per-role `mutate` branch makes `mutate_predator_speed` the single differing bit between arms. A new two-role invasion runner (role-filtered, cycle-phase-resolved) measures the prey-trait verdict; the existing single-population gate is left untouched.

**Tech Stack:** Python 3, numpy, pytest. Run with `uv run --python .venv … ` from `/Users/mirro/Projects/active-loop`.

**Spec:** `docs/superpowers/specs/2026-06-20-redqueen-coevolution-design.md` (binding predeclared-falsifier contract).

## Global Constraints

- Run via `uv run --python .venv …` from repo root. FROZEN paths never edited.
- Every new behavior behind `enable_predation: bool=False`; OFF makes ZERO new rng draws / no role read / no snapshot / no phase split / no new state → byte-identical `events_hash`. Re-pin ALL golden hashes AFTER the engine surgery lands.
- `events_hash` = SHA-256 of canonical JSON of `self.events` (`engine.py:1487-1492`); `_event()` genotype dict (`engine.py:825-847`) hardcodes a 9-field subset that **omits** `locomotor_speed` — so `Genotype.role` added as the LAST field, defaulted, and omitted from `_event()` is byte-identical OFF.
- Determinism: every predation scan/capture uses a once-per-step **frozen pre-move snapshot**, ascending-creature-id order, strict-`<` with 1e-9 tie-epsilon (lowest id wins), fixed float parenthesization, NO rng in scans. ON-path determinism pinned by `_PREDATION_ON_GOLDEN_HASH` only (the ON config legitimately adds rng via predator founders + the co-evolving-arm predator mutate draw).
- Anti-cheat seam (binding): `locomotor_speed` keys ONLY swept distance `d=speed·dt` — never the capture_radius (FIXED 0.6), never capture energy (constant-rule), never heading direction, never intake magnitude.
- After each experiment LOG: run the **FULL fast suite** (`uv run --python .venv pytest -q`), not just the site/guard subset (the verifier-floor + RESUME + status-line checks only surface in the full run). Each logged experiment needs `experiments/expNN_*.py` (docstring with 'falsifier'+a hypothesis word) + `experiments/outputs/expNN.txt` + an EXPERIMENTS.md entry with a `- Verifier:` line + `experiments-data.js` (via Python) + RESUME mention within 2 + a valid `closed-*`/`active` state token + STATUS<800.
- Commit after each task (small, single-purpose); end messages with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Branch off `main` (a new branch, e.g. `exp248-coevolution`). Never stage `active_loop/cli/converse_demo.py`. `git status` before committing (live cron shares the tree).

## File Structure

- **Modify** `ecology/genotype.py` — add `role` (LAST field) + its `mutate()` skip-guard + a `mutate_predator_speed` param.
- **Modify** `ecology/creature.py` — `Phenotype.move_hx/move_hy/move_d` (+ reuse existing `Creature.lineage_root`).
- **Modify** `ecology/engine.py` — `EcologyConfig` predation flags; the gated 3-phase loop + frozen snapshot + predator pursuit + prey flee + capture/energy/`predation` death + bucket index; the eat-heading reuse; the per-role mutate branch (1116-1124); predator founder start-energy.
- **Create** `ecology/evolvability/predation_invasion.py` — the two-role invasion runner + role-aware counter + cycle-phase-resolved estimator + the predator-prey `TraitAxis`.
- **Create** `experiments/exp248_geometry_probe.py`, `exp248_preflight.py`, `exp248_expressibility.py`, `exp248_static_invasion.py`, `exp248_coevolve_invasion.py`, `exp248_discriminator.py`, `exp248_robustness.py` (+ outputs).
- **Create** `tests/test_predation.py` — OFF byte-identity, ON golden, ON-differs, shuffle-invariance, role-counter exclusion, anti-cheat seam, determinism.

---

### Task 1 (Rung 0a): `Genotype.role` + `enable_predation` gate + per-role mutate branch

**Files:**
- Modify: `ecology/genotype.py` (Genotype dataclass after `locomotor_speed` ~L122; `mutate()` signature ~L137-148; skip-guard region ~L232-234)
- Modify: `ecology/engine.py` (`EcologyConfig` flags; mutate call `1116-1124`)
- Test: `tests/test_predation.py` (new), `tests/test_ecology_continuous.py` (golden unchanged)

**Interfaces:**
- Produces: `Genotype.role: str = "prey"` (LAST field, never in TRAIT_BOUNDS/_event, never mutated). `EcologyConfig.enable_predation: bool=False`, `mutate_predator_speed: bool=False`, `freeze_prey_speed: bool=False`. `mutate(..., mutate_predator_speed: bool=False)`.

- [ ] **Step 1: Failing test** — add to `tests/test_predation.py`:
```python
import dataclasses as D
from ecology.genotype import Genotype, mutate, founder
import numpy as np

def test_role_defaults_prey_and_is_last_field():
    g = founder()
    assert g.role == "prey"
    assert list(D.fields(Genotype))[-1].name == "role"

def test_role_never_mutates():
    rng = np.random.default_rng(0)
    g = D.replace(founder(), role="predator")
    child = mutate(g, rng, 0.2, mutate_continuous_locomotion=True)
    assert child.role == "predator"  # copied verbatim, never perturbed
```

- [ ] **Step 2: Run, expect fail** — `uv run --python .venv pytest tests/test_predation.py -v` → FAIL (`role` undefined).

- [ ] **Step 3: Add the field** — in `ecology/genotype.py` after `locomotor_speed: float = 1.0` (L122):
```python
    # Exp 248: predator-prey role — LAST field, default 'prey', NEVER mutated (copied verbatim),
    # NOT in TRAIT_BOUNDS, NOT emitted in _event() → byte-identical OFF.
    role: str = "prey"
```
In `mutate()` add `mutate_predator_speed: bool = False` to the signature (after `mutate_continuous_locomotion`). The field-order loop iterates `asdict(g)`; `role` is a str (not in TRAIT_BOUNDS) — add an explicit skip BEFORE the numeric-perturbation logic so it's copied verbatim with no rng draw:
```python
        if k == "role":
            new_d[k] = v
            continue
```
(`role` is LAST, so this never shifts any upstream trait's rng draw.)

- [ ] **Step 4: Add config flags + per-role mutate branch** — in `EcologyConfig` add (near the other continuous flags):
```python
    enable_predation: bool = False          # Exp 248 master gate; OFF byte-identical
    mutate_predator_speed: bool = False     # the SINGLE causal bit (co-evolving arm = True)
    freeze_prey_speed: bool = False          # prey breed-true in the focal test (both arms)
```
Replace the `mutate_continuous_locomotion=` argument at `engine.py:1116-1124` with a per-role branch GATED on `enable_predation`:
```python
        if cfg.enable_predation:
            _mut_speed = ((c.genotype.role == "predator" and cfg.mutate_predator_speed)
                          or (c.genotype.role == "prey" and not cfg.freeze_prey_speed))
        else:
            _mut_speed = (cfg.enable_continuous_locomotion and not cfg.freeze_continuous_locomotion)
        child_geno = mutate(c.genotype, self.rng, cfg.mutation_rate,
                            mutate_thermosense=cfg.enable_thermosense,
                            freeze_learning_rate=cfg.freeze_learning_rate,
                            freeze_thermosense=cfg.freeze_thermosense,
                            mutate_memory=cfg.enable_hidden_mode,
                            mutate_active_sensing=cfg.enable_active_sensing,
                            mutate_locomotion=cfg.enable_terrain,
                            mutate_continuous_locomotion=_mut_speed)
```
(OFF path takes the `else` → byte-identical pre-existing expression.)

- [ ] **Step 5: Tests pass + golden unchanged** — `uv run --python .venv pytest tests/test_predation.py tests/test_ecology_continuous.py -v`. The `_CONTINUOUS_ON_GOLDEN_HASH` test MUST pass unchanged. Add a test asserting OFF events_hash == a config without the predation fields (self-consistency) AND that adding `role="predator"`-defaulted founders is not present on the OFF path.

- [ ] **Step 6: Commit** — `git status`; `git add ecology/genotype.py ecology/engine.py tests/test_predation.py`; commit `Exp 248 Rung 0a: Genotype.role + enable_predation gate + per-role mutate branch (byte-identical OFF)`.

---

### Task 2 (Rung 0b-i): Phenotype move-heading fields + eat-heading reuse (gated)

**Files:**
- Modify: `ecology/creature.py` (`Phenotype` ~L52-82)
- Modify: `ecology/engine.py` (move block `895-919`, eat back-projection `1014-1026`)
- Test: `tests/test_predation.py`

**Interfaces:**
- Consumes: Task 1 flags.
- Produces: `Phenotype.move_hx: float=0.0, move_hy: float=0.0, move_d: float=0.0` (never in events_hash). When `enable_predation`, the eat back-projection reuses the stored realized move heading instead of recomputing `best_heading`.

- [ ] **Step 1: Failing test** — assert that with `enable_predation=True` and NO predator present, a short run's events_hash equals the same config's run (determinism) AND that a Rung-0 invariant holds: the stored move-heading equals `best_heading(_x1,_y1)` when no predator is in range (add a debug assertion or a telemetry check in a tiny harness). Concretely: run a prey-only `enable_predation=True` config and assert it does NOT raise and reproduces across two runs.

- [ ] **Step 2: Run, expect fail** (move_hx etc. undefined / branch absent).

- [ ] **Step 3: Add Phenotype fields** — in `ecology/creature.py` after `pos_cont` (L82):
```python
    move_hx: float = 0.0   # Exp 248: realized unit move heading (predation eat-reuse); never in events_hash
    move_hy: float = 0.0
    move_d: float = 0.0
```

- [ ] **Step 4: Store realized heading in move; reuse in eat** — in the move block (`engine.py:905-910`), after computing `_hdx,_hdy,_d`, when `cfg.enable_predation` store them: `ph.move_hx, ph.move_hy, ph.move_d = _hdx, _hdy, _d` (Task 3 will replace `_hdx,_hdy` with the role-aware heading; this task just plumbs the store+reuse). In the eat block (`engine.py:1014-1024`), gate the back-projection:
```python
        if cfg.enable_predation:
            _hdx, _hdy, _d = ph.move_hx, ph.move_hy, ph.move_d   # reuse realized move heading
        else:
            _hdx, _hdy = self.cont_world.best_heading(_x1, _y1)  # existing reconstruction (byte-identical)
            _d = g.locomotor_speed * cfg.continuous_dt
        _x0e = max(0.0, min(ARENA_W, _x1 - _hdx * _d))
        _y0e = max(0.0, min(ARENA_H, _y1 - _hdy * _d))
        eaten = self.cont_world.consume(_x0e, _y0e, _x1, _y1, deficit)
```
(OFF path unchanged → byte-identical. When ON but no predator in range, Task 3's heading == `best_heading`, so eat reduces to the same segment — assert this in Rung 0b-ii.)

- [ ] **Step 5: Tests pass + goldens unchanged.**

- [ ] **Step 6: Commit** — `Exp 248 Rung 0b-i: Phenotype move-heading fields + gated eat-heading reuse (byte-identical OFF)`.

---

### Task 3 (Rung 0b-ii): 3-phase loop + frozen snapshot + pursuit/flee + capture + bucket index + ON golden

**Files:**
- Modify: `ecology/engine.py` (`step()` `1246-1327`; `_step_one_creature` `880-1241`; `EcologyConfig` predation params)
- Test: `tests/test_predation.py`

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: gated predation dynamics; `_PREDATION_ON_GOLDEN_HASH`; predator/prey headings from a frozen snapshot; capture with `cause="predation"`.

- [ ] **Step 1: Add predation params to EcologyConfig** (read only inside `if cfg.enable_predation`): `capture_radius: float=0.6`, `sensing_radius: float=3.0`, `assimilation_efficiency: float=0.6`, `strike_cost: float=0.0`, `w_food: float=1.0`, `w_flee: float=1.5`, `max_captures_per_step: int=1`, `pred_start_energy_frac: float=0.75`.

- [ ] **Step 2: Failing tests** — in `tests/test_predation.py`, add `_pred_cfg()` (a two-role `founder_mix` config, `enable_predation=True`, depletion-intake ON, prey escape=1.0 + predators pursuit=1.4, fixed seed, horizon 300) and:
```python
def test_predation_on_differs_from_off():
    on  = Ecology(_pred_cfg(enable_predation=True),  seed=5).run()["events_hash"]
    off = Ecology(_pred_cfg(enable_predation=False), seed=5).run()["events_hash"]
    assert on != off  # predation is not a silent no-op

def test_predation_on_golden_hash():
    h = Ecology(_pred_cfg(enable_predation=True), seed=5).run()["events_hash"]
    assert h == _PREDATION_ON_GOLDEN_HASH

def test_predation_on_invariant_under_shuffle():
    a = Ecology(_pred_cfg(enable_predation=True, shuffle_creature_order=False), seed=5).run()["events_hash"]
    b = Ecology(_pred_cfg(enable_predation=True, shuffle_creature_order=True),  seed=5).run()["events_hash"]
    assert a == b  # frozen pre-move snapshot ⇒ headings order-independent
```
(Pin `_PREDATION_ON_GOLDEN_HASH` from the first green run; document it as "do not repin without cause".)

- [ ] **Step 3: Implement the gated 3-phase loop.** In `step()`, when `cfg.enable_predation`: (a) build the **frozen snapshot** at loop top — two lists `(creature_id, x, y)` for alive prey and predators, each sorted ascending id; (b) **Phase A**: a move-all pass that for each alive creature computes its role-aware heading from the snapshot, advances `pos_cont`, charges movement+speed cost, and stores `move_hx/move_hy/move_d` (reuse `_step_one_creature`'s move logic factored into a `_move_phase(c)`); (c) **Phase B**: `_resolve_captures()` — predators in ascending id order each claim the nearest not-yet-captured prey within `capture_radius` (strict `<`, 1e-9 tie, lowest prey-id), `max_captures_per_step=1`; on capture set prey `alive=False, cause_of_death="predation"`, credit predator `min(deficit, prey.energy*assimilation_efficiency)` (cap), emit a `"predation"` death event (mirror the starvation death-event shape) in predator-id then prey-id order, charge `strike_cost` to pursuing predators; (d) **Phase C**: for each surviving creature run the existing eat/reproduce/death tail (skip eat for `role=="predator"`). Headings:
  - prey: `h = normalize(w_food*best_heading_food(x,y) − w_flee*Σ_{pred in snapshot within sensing_radius} unit(pred→prey)/dist)`; flee term exactly 0 when none in range (→ `h == best_heading`).
  - predator: nearest prey in snapshot within `sensing_radius` → steer toward it (PURSUIT); else `best_heading`.
  Use a deterministic **sub-cell bucket index** (24×24) for the scans (same+adjacent buckets covering `sensing_radius`; buckets in fixed order; within-bucket ascending id). When `cfg.enable_predation=False`, `step()` takes the EXACT existing single-pass path (no snapshot, no phases).

- [ ] **Step 4: Tests pass** — `uv run --python .venv pytest tests/test_predation.py tests/test_ecology_continuous.py tests/test_density_mortality.py -v`. ON-differs, ON-golden, shuffle-invariance pass; all prior goldens unchanged.

- [ ] **Step 5: Anti-cheat seam test** — add a probe asserting capture rate is invariant to `locomotor_speed` when positions are held fixed (capture_radius never f(speed)); and a no-predator-in-range run reproduces the foraging segment of an `enable_predation=False` run (the eat-heading-reuse equals `best_heading` when flee term is zero). Commit.

---

### Task 4 (Rung 1): Static-geometry expressibility probe (the cheap CAN'T-POSE gate)

**Files:** Create `experiments/exp248_geometry_probe.py`

**Interfaces:** Consumes Tasks 1-3 (`_pred_cfg`-style config). Produces a go/abort on whether escape-speed is behaviorally expressed.

- [ ] **Step 1: Write the probe** — frozen monomorphic prey at speed ∈ {1.0, 1.1} vs FROZEN predators at each static bracket {1.2, 1.4, 1.6}, no co-evolution, no invasion. Step the sim; measure (a) mean post-move predator-prey distance and (b) per-capita capture-survival (fraction of prey-steps not ending in capture) per prey speed; report the **marginal 1.0-vs-1.1 capture-survival delta** against each bracket. Controller-run (raw numbers).
- [ ] **Step 2: Run** — `uv run --python .venv python experiments/exp248_geometry_probe.py`. Write `experiments/outputs/exp248_geometry.txt`.
- [ ] **Step 3: ABORT gate** — if capture-survival is flat in speed (faster prey caught at the same rate; geometry/distance-arithmetic dominated) → **STOP, log CAN'T-POSE**, do NOT build the invasion stack. Else proceed.
- [ ] **Step 4: Commit** the probe + output.

---

### Task 5: Two-role invasion runner + role-aware counter + cycle-phase estimator + predator-prey axis

**Files:** Create `ecology/evolvability/predation_invasion.py`; Test `tests/test_predation.py`

**Interfaces:** Produces `run_invasion_from_rarity_two_role(base_cfg, prey_axis, seeds, *, static_pursuit, coevolve, ...)` returning a per-checkpoint prey-only `f_mut` series + a cycle-phase-resolved verdict; a `PREY_SPEED_AXIS` (mirrors `LOCOMOTION_CONTINUOUS_AXIS`, `h_trait="locomotor_speed"`).

- [ ] **Step 1: Failing test (role exclusion)** — seed residents (prey 1.0), mutants (prey 1.1), and a PREDATOR at speed 1.1; assert the prey-mutant tally counts only the prey-1.1, excludes the predator-1.1:
```python
def test_role_aware_counter_excludes_predator_at_mutant_value():
    from ecology.evolvability.predation_invasion import count_prey_by_value
    alive = [mk(role="prey", speed=1.0), mk(role="prey", speed=1.1), mk(role="predator", speed=1.1)]
    n_res, n_mut = count_prey_by_value(alive, h_res=1.0, h_mut=1.1)
    assert (n_res, n_mut) == (1, 1)  # predator-1.1 excluded
```
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** — `count_prey_by_value` filters `c.genotype.role=="prey"` FIRST then bins by `abs(axis.get(c.genotype)-h)<1e-6` (mirrors `gates.py:554-561` but role-filtered). `run_invasion_from_rarity_two_role`: builds `founder_mix=((prey_res,95),(prey_mut,5),(predator,n_pred))` with `freeze_prey_speed=True`, `mutate_predator_speed=coevolve`, predators clamped to `static_pursuit` (static arm) or seeded for co-evolution; steps to horizon 6000; accumulates `f_mut` only where `n_prey_total>=prey_denominator_floor=40`; invalidates a seed if prey trough `<prey_floor=20` or `exploded`. Uses `stability.py`'s cross-correlation cycle detector to find prey-peak phase points; fits a Theil-Sen slope on phase-matched peaks + compares late-vs-early phase-matched means. INVADES iff both > 0. Add a `PREY_SPEED_AXIS` TraitAxis (`h_trait="locomotor_speed"`, freeze via `freeze_prey_speed`, `disconnect_overrides={"enable_predation":False,"speed_cost_floor":0.0,"speed_cost_slope":0.0}`).
- [ ] **Step 4: Tests pass** (role-exclusion + a determinism test of the runner on a fixed seed). Commit.

---

### Task 6 (Rung 2): Two-trophic viability preflight + runtime preflight (hard gate)

**Files:** Create `experiments/exp248_preflight.py`

- [ ] **Step 1: Write** — monomorphic 120 prey (1.0) + 20 predators (1.4), horizon 1500, 5 seeds, mutation OFF. Assert ALL: both roles >0 at horizon in ≥4/5; prey trough ≥20 & predator trough ≥3; ≥1 LV cycle (predator lags prey, via `stability.py` cross-correlation); captures>0; resource not stripped; predators survive the pre-first-capture window in ≥4/5 (track per-predator energy first ~50 steps; if not, raise `pred_start_energy_frac` 0.75→0.9 and/or `n_predator_founders`). Also invoke `compute-batch-runtime-preflight` discipline: time one 6000-step run with the bucket index, project the full Rung 1-7 batch, confirm bounded oscillation (growth deceleration across cycles, not runaway).
- [ ] **Step 2: Run.** If no parameterization in a 6-cell tuning mini-grid (`assimilation_efficiency{0.5,0.6,0.7}×capture_radius{0.5,0.7}`) passes → **STOP, log CAN'T-POSE**.
- [ ] **Step 3: Commit** the preflight + output + the calibrated params. **PAUSE for human go before the heavy Rung 3-7 batch.**

---

### Task 7 (Rung 3): Full L39 expressibility decomposition

**Files:** Create `experiments/exp248_expressibility.py`

- [ ] Gen-0 monomorphic benefit curves vs FROZEN predators: ESCAPE channel (capture-survival) ALONE non-flat & monotone-then-cost-turning over escape-speed {0.5,0.75,1.0,1.25,1.5,2.0,3.0} (~5 seeds); subtract a **predation-OFF** foraging-vs-speed baseline; predator captures rise with pursuit-speed; the marginal 1.0-vs-1.1 capture delta vs each static bracket materially non-zero AND saturating by ~1.1-1.5. ABORT CAN'T-POSE if the escape curve alone is flat or escape vanishes after subtracting the foraging baseline. Run, write output, commit.

---

### Task 8 (Rung 4): STATIC-arm invasion + marginal non-degeneracy (control FIRST)

**Files:** Create `experiments/exp248_static_invasion.py`

- [ ] Invasion-from-rarity of prey 1.0→1.1 with FROZEN predators at each bracket {1.2,1.4,1.6}, ≥16 seeds, the cycle-resolved estimator. Predeclared DOES_NOT_INVADE at all brackets. Certify the static predator is a GENUINE pressure: marginal per-capita 1.0-vs-1.1 capture-rate difference above a predeclared floor AND resident predation ≥ a predeclared share of resident deaths. FAIL-degenerate/INVALID if ~0 marginal mortality (cost wall, not saturating predation); FAIL-hypothesis (NEGATIVE, stop) if escape-speed DOES invade vs static. Run, write output, commit, blind-verify.

---

### Task 9 (Rung 5): CO-EVOLVING-arm invasion (focal)

**Files:** Create `experiments/exp248_coevolve_invasion.py`

- [ ] Same invasion, `mutate_predator_speed=True`, ≥16 PAIRED seeds (shared with Rung 4), neutral 1.0/1.0 null co-run (estimator FPR <5%), the non-genotypic `lineage_root` drift null. Predeclared DOES_INVADE. **Primary verdict = the paired cross-arm contrast** (static DOES_NOT_INVADE ∧ co-evolve INVADES). Also assert the Rung-0.5 single-bit isolation (arms byte-identical to first predator reproduction). NEGATIVE if no invasion / indistinguishable from drift; VOID if exploded/extinct/denominator unmet. Run, write output, commit, blind-verify.

---

### Task 10 (Rung 6): The six-prong discriminator (decisive)

**Files:** Create `experiments/exp248_discriminator.py`

- [ ] Only if Rung 5 invades: (1) prey cross-cycle Theil-Sen slope > within-cycle amplitude AND higher in co-evolve than static; (2) faster(1.1) invades AND slower(0.9) does NOT; (3) beats the `lineage_root` drift null (<5% FPR); (4) BOTH prey & predator means escalate cross-cycle; (5) the **flee-on/capture-off** arm (capture_radius≈0) does NOT invade; (6) one-way reciprocal invasibility (rare resident 1.0 does NOT invade a common 1.1 world). DOWNGRADE-to-NEGATIVE if any prong fails. Run, write output, commit, blind-verify.

---

### Task 11 (Rung 7 + logging/coalescence): robustness + journal

**Files:** Create `experiments/exp248_robustness.py`; Modify `EXPERIMENTS.md`, RESUME.md, `loop/directions/coevolution-red-queen.md` (new card), `site/data/experiments-data.js`, the coalescence library.

- [ ] **Step 1 (separately gated):** only if Rung 6 passes — replicate static-fails/co-evolve-invades + the discriminator across ≥3 regimes (capture_radius{0.4,0.6,0.8}, assimilation{0.5,0.6,0.7}, strike_cost{0,0.05,0.1}, ratio) on fresh seeds. DOWNGRADE to ONE-REGIME CANDIDATE if knife-edge.
- [ ] **Step 2:** Log each rung's verdict to EXPERIMENTS.md (append-only, with `- Verifier:` lines) + `experiments/outputs/expNN.txt` + `experiments-data.js` (via Python) + RESUME mention + the new direction card (valid state token, STATUS<800). **Run the full fast suite after logging.**
- [ ] **Step 3:** Distil to a coalescence artifact (MechanismCard if POSITIVE = first wall-escape; else extend the local-gradient-wall BoundaryNote to co-evolution). `active-monkey coalesce validate --all`. Commit.

---

## Self-Review

**Spec coverage:** §3 mechanism → Tasks 1-3. §4 single-causal-bit → Task 1 (branch) + Task 9 (Rung-0.5 assertion). §5 viability → Task 6. §6 measurement → Task 5. §7 six-prong discriminator → Task 10. §8 rung ladder → Tasks 4,6,7,8,9,10,11. §9 falsifiers → evaluated in Tasks 4-10. §10 controls → Tasks 4,8,9,10. §11 gating/determinism → Tasks 1,2,3 (golden re-pin). §12 runtime → Task 6 preflight. §13 surface → all tasks. §14 logging/coalescence → Task 11. All sections mapped.

**Placeholder scan:** Tasks 4,6-11 describe orchestration scripts by exact inputs/outputs/pass-conditions rather than full code (the load-bearing engine surgery + the invasion runner are complete code in Tasks 1-3,5; the rung scripts are thin glue over those tested primitives + the spec's predeclared metrics). Intentional decomposition, not placeholders.

**Type consistency:** `enable_predation`/`mutate_predator_speed`/`freeze_prey_speed` (Task 1) ↔ used in Task 3 loop + Task 5 runner. `Genotype.role` (Task 1) ↔ `count_prey_by_value` filter (Task 5). `Phenotype.move_hx/move_hy/move_d` (Task 2) ↔ eat-reuse + Phase A store (Tasks 2,3). `capture_radius`/`sensing_radius`/`assimilation_efficiency` (Task 3) ↔ the probes (Tasks 4,6-10). `run_invasion_from_rarity_two_role` + `PREY_SPEED_AXIS` (Task 5) ↔ invasion rungs (Tasks 8,9). Consistent.
