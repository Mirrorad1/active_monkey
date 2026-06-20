# Spec — Red Queen co-evolution: does prey escape-speed evolve under co-evolving predators? (Exp 248+)

**Date:** 2026-06-20 · **Direction:** coevolution-red-queen (NEW) · **Status:** design approved, pre-registered

> Both a brainstorming design doc and the experiment's **predeclared-falsifier contract** (`loop/VALIDATION.md`
> binding; falsifiers below frozen before any focal run). Hardened by a 9-agent design + adversarial-audit
> workflow (3 designers → synthesize → 4 lenses [predation/determinism/viability, control-validity + Red-Queen-
> vs-drift, invasion-gate validity, build-cost] → harden); all lenses returned "needs-fix", fixes folded in.

## 1. Premise & hypothesis

The evolvability arc (Exp 199–247) established a robust **local-gradient wall**: a costed capability does not
locally evolve on these toy substrates — across senses, memory, active sensing, and locomotion — because the
benefit **saturates** against a *static/drifting* environment (the gradient at the resident is ≤0). The
continuous-locomotion sub-arc (238–247) sharpened this to a **stability-vs-strong-competition trade-off**. Every
experiment pitted a costed trait against a *static* selective pressure — exactly why benefits saturate.

The wall arc named one escape it never satisfied: a benefit that is **non-saturating** (large-per-small-step,
continuously regenerated). The one structurally-distinct way to manufacture that, never tried, is to make the
selective pressure **co-evolve**. **Hypothesis:** a costed prey escape-speed does NOT invade against a *static*
predator (benefit saturates → replicates the wall) but DOES invade against a *co-evolving* predator (the
predator's improvements continuously regenerate the gradient — the Red Queen). Honest caveat: not guaranteed —
if the *instantaneous* local gradient is still ≤0 it could wall even here; either outcome is a strong result.

## 2. Goal & scope

Build a **spatial predator-prey** mechanism in `ecology/` (gated, byte-identical OFF) and run a **single-side**
test: invasion-from-rarity of **prey escape `locomotor_speed`** (1.0→1.1) under **CO-EVOLVING vs STATIC**
predators. **Co-evolution is the single causal variable; the static arm is the binding anti-gaming control.**
Primary verdict = the cross-arm contrast (static DOES_NOT_INVADE ∧ co-evolve INVADES), gated by a six-prong
Red-Queen-vs-drift discriminator. Out of scope: a mutual two-side arms race (Rung-7 follow-up only).

## 3. The predation mechanism (gated `enable_predation`, byte-identical OFF)

- **Role field.** `Genotype.role: Literal['prey','predator'] = 'prey'`, the **LAST** field (after `locomotor_speed`),
  default `prey`. NOT in `TRAIT_BOUNDS`, **never** perturbed by `mutate()` (copied verbatim, zero rng draw,
  mirroring the `locomotor_speed` last-field skip-guard), NOT emitted in `_event()`'s genotype subset (which
  already omits `locomotor_speed`) → every existing construction / `asdict` order / `is_valid` loop / events_hash
  is byte-identical. `founder_mix` seeds both roles `((prey_geno,N_prey),(predator_geno,N_predator))`. Pursuit and
  escape both ride the **same** `locomotor_speed`; only the heading *source* differs by role.
- **Three-phase step loop (ON only).** When `enable_predation=True` the per-creature loop splits into **A** move-all
  (advance `pos_cont`, charge speed/metabolic cost) → **B** capture-resolve (post-move positions) → **C**
  eat+reproduce+die. OFF keeps the EXACT single-pass `_step_one_creature` structure (no split) → byte-identical.
  Phase ordering enforces predation/starvation mutual exclusivity (a captured prey is removed in B before its C).
- **Capture (B).** `dist(predator,prey) ≤ capture_radius=0.6` (FIXED, never f(speed)). Speed enters **only** via the
  post-move geometry it produced (the binding anti-cheat seam; a speed-coupled radius was rejected). Resolution:
  predators in ascending creature-id order, each claims the single nearest not-yet-captured prey within radius
  (strict `<`, 1e-9 tie-epsilon, lowest prey-id wins); `max_captures_per_step=1`. No rng.
- **Energy.** Prey eat the resource field as today (depletion-intake ON). Predators do NOT eat resource; on capture
  the predator gains `min(deficit, prey.energy × assimilation_efficiency=0.6)` (capped; remainder lost — a
  constant-rule transfer, never f(speed)). Captured prey dies, `cause='predation'` (new, mutually exclusive with
  starvation/senescence), event mirrors the starvation death shape, emitted predator-id then prey-id.
- **Sensing/fleeing (frozen snapshot → determinism).** At step top, copy all alive creatures' `pos_cont`/role/id into
  a read-only **pre-move snapshot** (prey/predator sub-lists, ascending id). **Predators** scan the prey-snapshot for
  the nearest within `sensing_radius=3.0` → steer toward it (PURSUIT); else `best_heading` wander. **Prey** steer a
  blended heading `normalize(w_food·best_heading_food − w_flee·Σ_{pred in range} unit(pred→prey)/dist)`,
  `w_food=1.0, w_flee=1.5`. When no predator is in range the flee term is **exactly zero** → heading reduces verbatim
  to `best_heading(food)`. `locomotor_speed` keys **only** swept distance `d=speed·dt`, never heading/intake/radius.
- **Eat-heading consistency (audit-critical fix).** Today the eat back-projection recomputes `best_heading(food)`;
  with a flee-blended move heading that would integrate intake over a segment never traversed. FIX: store the
  **realized** move heading+displacement on the phenotype in phase A; phase C's eat reuses it. Gated so OFF /
  no-predator-in-range reproduces the existing reconstruction byte-for-byte (a Rung-0 assertion checks
  stored-heading == `best_heading` when the flee term is zero).
- **Spatial cost / bucket index.** All-pairs scans are O(n_pred·n_prey); at the resource-limited oscillation peak
  n_prey can be hundreds, so **default** to a deterministic sub-cell **bucket index** (24×24 grid, scan same+adjacent
  buckets covering `sensing_radius`, buckets in fixed order, creatures within a bucket by ascending id) → O(N).

## 4. The single causal bit + arms

The engine has ONE global `mutate()` call (`engine.py:1116`); it cannot freeze prey while mutating predators on the
same field. **New per-role mutate branch** (gated on `enable_predation`):
`mutate_continuous_locomotion = (role=='predator' and mutate_predator_speed) or (role=='prey' and not freeze_prey_speed)`.
In the focal test `freeze_prey_speed=True` in **both** arms (prey 1.0/1.1 breed true); the arms differ in **exactly**
the predator-only bool `mutate_predator_speed`: **STATIC**=False (predators inherit pursuit verbatim, no draw) vs
**CO-EVOLVING**=True (pursuit mutates). The branch mirrors the last-field skip-guard so toggling the bit skips ONLY
the predator `locomotor_speed` draw — every other rng draw byte-identical between arms up to the first predator
reproduction. **Rung-0.5 assertion:** the two arms are byte-identical engine state up to that first divergence (proves
co-evolution is isolated, not an rng artifact). OFF → the call falls back to the pre-existing global expression.
STATIC arm runs at a **bracket** `static_pursuit_speed ∈ {1.2,1.4,1.6}` (not one knife-edge value); predeclared
DOES_NOT_INVADE at all brackets.

## 5. Two-trophic viability (hard gate; expect a Lotka-Volterra cycle)

Predator-prey is an oscillator — we **expect and accept a bounded limit cycle**, not extinction, as the stable state.
Prey reuse the Exp-240/247 depletion-intake physics (density-dependent carrying capacity); predators couple to prey
only via capture energy; the single-capture cap + small radius + prey fleeing form a refuge so predators can't drive
prey to zero. **Balancing knobs:** `assimilation_efficiency`, `capture_radius`, `sensing_radius`, ~1:6
predator:prey founder ratio (`n_prey_founders=120`, `n_predator_founders=20`), `pred_start_energy_frac=0.75`,
predator energy_capacity/repro_threshold. **Validity floors:** `prey_floor=20` (hard seed-invalidation trough),
`prey_denominator_floor=40` (min n_prey to accumulate a frequency checkpoint), `pred_floor=3`.

## 6. Measurement (new two-role invasion runner)

Do NOT reuse `run_invasion_from_rarity`'s verdict path (single-checkpoint `f_final>f_initial` is a cycle-phase
artifact in an oscillator; leave it untouched/golden). **New** `run_invasion_from_rarity_two_role`: a **role-aware**
counter filters `role=='prey'` FIRST, then bins by trait value (a regression test seeds a predator at speed 1.1 and
proves it's excluded — the co-evolved-predator-drifts-into-prey-value contamination). `founder_mix` = prey-resident
(1.0, ~95) + prey-mutant (1.1, ~5) + predators (~20). `f_mut = n_prey_mut/(n_prey_mut+n_prey_res)`, accumulated only
where `n_prey_total ≥ 40`; seed INVALID if prey trough < 20. **Cycle-phase-resolved verdict:** detect LV cycle
boundaries (reuse `stability.py` cross-correlation), sample `f_mut` at the same phase (prey-peaks) across cycles, fit a
robust **Theil-Sen** slope on ≥3 phase-matched peaks AND compare late-vs-early phase-matched means. **≥16 paired
seeds**; primary verdict = **cross-arm paired** contrast (shared seeds). A **neutral 1.0/1.0 null** through the
identical estimator must give <5% false-positive rate before any positive is trusted. **Expressibility (L39)** is
decomposed: ESCAPE channel = capture-survival; FORAGING channel = resource/offspring; the escape curve ALONE must be
non-flat AND survival must rise *over and above* a predation-OFF foraging baseline (subtracted).

## 7. The six-prong Red-Queen-vs-drift discriminator (the crux)

A POSITIVE wall-escape requires **all six**, in the co-evolving arm, with the static arm failing:
1. **Directionality vs cycling** — prey cross-cycle Theil-Sen slope exceeds within-cycle amplitude AND is **higher in
   co-evolve than static** (the binding Red-Queen signature; not an intra-arm correlation).
2. **Demographic-instability trap (Exp-241)** — a slower mutant (0.9) does **NOT** also invade; faster(1.1)-invades
   XOR slower(0.9)-invades (asymmetry).
3. **Drift null** — a non-genotypic `lineage_root` tag (Phenotype-tracked, no new Genotype field, no new rng) on a
   95/5 split of identical speed-1.0 prey must NOT invade through the same estimator (<5% FPR).
4. **Mechanism attribution** — **both** prey and predator mean speed escalate cross-cycle (a bare lag-correlation is
   demoted: a shared cycle produces it without reciprocal selection).
5. **Foraging-confound control (decisive)** — a **flee-on/capture-off** arm (predators present, flee-blend on,
   `capture_radius≈0` → no mortality): the faster mutant must **NOT** invade. If it does, the rise is a speed-linked
   foraging/steering artifact (the Exp-237/238 trap re-entering via steering) → VOID.
6. **Reciprocal invasibility** — rare resident (1.0) does NOT invade a common-mutant (1.1) world → one-way
   (directional). Mutual invasion = protected polymorphism (report as such, not a wall-escape); common-stays-common =
   priority effect.

## 8. Rung ladder (each names its FAIL)

- **0a** — `Genotype.role` + `enable_predation` gate + the gated per-role mutate branch (no behavior). FAIL: any
  existing golden hash changes OFF, or any new rng draw on the OFF path.
- **0b** — capture/sensing/flee machinery + the 3-phase split + frozen snapshot + eat-heading reuse + bucket index +
  `_PREDATION_ON_GOLDEN_HASH`. FAIL: ON hash == OFF (silent no-op), ON hash changes under shuffle (order-dependent), or
  the no-predator eat back-projection diverges from the existing reconstruction.
- **1 (promoted, cheap, decisive)** — static-geometry expressibility probe: a faster prey ends measurably farther from
  a pursuer AND suffers fewer captures; the marginal 1.0-vs-1.1 capture delta vs the static bracket is non-zero. **ABORT
  (CAN'T-POSE) before building the invasion stack** if capture-survival is flat in speed (geometry/distance-arithmetic
  trap).
- **2** — two-trophic viability preflight (5 seeds × 1500): both roles persist (≥4/5, troughs above floors), ≥1 LV
  cycle, captures>0, predators survive pre-first-capture, resource not stripped, not exploded. + `compute-batch-runtime-
  preflight`. FAIL: CAN'T-POSE (no viable parameterization in the tuning budget).
- **3** — full L39 expressibility decomposition (escape channel non-flat after subtracting the foraging baseline;
  predator captures rise with pursuit-speed; marginal 1.0-vs-1.1 delta vs each static bracket non-zero AND saturating).
  FAIL: CAN'T-POSE.
- **4 (control first)** — STATIC-arm invasion at each bracket, ≥16 seeds. Predeclared DOES_NOT_INVADE. + the **marginal
  non-degeneracy audit** (the static predator must impose a materially non-zero 1.0-vs-1.1 capture-rate difference AND
  resident predation ≥ a predeclared share of resident deaths). FAIL-degenerate (INVALID) if ~0 marginal mortality
  (cost wall, not saturating-predation wall); FAIL-hypothesis (NEGATIVE, stop) if escape-speed DOES invade vs static.
- **5 (focal)** — CO-EVOLVING-arm invasion (`mutate_predator_speed=True`), ≥16 seeds, neutral-null co-run. Predeclared
  DOES_INVADE. Primary verdict = paired cross-arm contrast. NEGATIVE (loggable) if it doesn't invade / indistinguishable
  from drift; VOID if exploded/extinct/denominator-floor unmet.
- **6 (discriminator)** — only if Rung-5 invades: all six prongs. DOWNGRADE-to-NEGATIVE if any prong fails.
- **7 (robustness; separately gated)** — replicate the contrast + discriminator across ≥3 regimes (capture_radius,
  assimilation, strike_cost, ratio) on fresh seeds. DOWNGRADE to ONE-REGIME CANDIDATE if knife-edge.

## 9. Predeclared falsifiers (binding, frozen)

- **STATIC INVADES** → no saturating wall to escape; POSITIVE impossible; NEGATIVE, stop.
- **Degenerate static control (L40, Exp-238)** → ~0 marginal 1.0-vs-1.1 mortality (even if total captures>0); the
  contrast is a plain cost wall, vacuous; INVALID.
- **CO-EVOLVE does NOT invade** → co-evolution didn't regenerate a gradient; clean NEGATIVE.
- **Symmetric invasion (Exp-241)** → slower(0.9) also invades / both decline; demographic instability, not selection;
  VOID.
- **Foraging-confound artifact** → faster mutant invades the flee-on/capture-off arm; steering benefit, not escape;
  VOID.
- **Mutual reciprocal invasion** → protected polymorphism / NFD, not directional escape; report as such.
- **Zero cross-cycle slope / no excess over static** → Red-Queen cycling, not escape; DOWNGRADE/NEGATIVE.
- **Indistinguishable from drift / estimator FPR ≥5%** → neutral drift / underpowered; NEGATIVE.
- **Predators didn't escalate** → gradient not regenerated by the predator response; mechanism unsupported.
- **Single bit not isolated** → arms not byte-identical to the first predator reproduction; confounded; INVALID.
- **Inexpressible trait (CAN'T-POSE, L39)** → escape-channel capture-survival flat / Rung-1 flat; STOP.
- **Non-viable two-trophic regime** → no parameterization persists; vacuous (Exp-242/243); CAN'T-POSE.
- **Explosion / unbounded-N / denominator collapse** → invasion read off a runaway or a bottleneck; INVALID (Exp-240).
- **Byte-identical ON / order-dependent** → predation wired to nothing or non-deterministic headings; build INVALID.

## 10. Anti-gaming controls

Static-predator bracket (primary); marginal non-degeneracy audit (L40); single-bit rng-isolation assertion;
flee-on/capture-off foraging control; slower-mutant asymmetry; reciprocal invasibility; non-genotypic drift null;
predation-OFF null (reduces to the closed Exp-238–247 substrate + the foraging baseline); cost-OFF/`disconnect_overrides`
null; anti-cheat seam audit (speed keys only `d`); cross-arm directionality; both-means-escalate gate; no-explosion +
denominator-floor validity gate.

## 11. Gating / determinism / byte-identity

New flags (all read only inside `if cfg.enable_predation`): `enable_predation:bool=False`, `Genotype.role`,
`mutate_predator_speed:bool=False`, `freeze_prey_speed:bool`, `capture_radius/sensing_radius/assimilation_efficiency/
strike_cost/w_food/w_flee/max_captures_per_step/pred_start_energy_frac/prey_floor/prey_denominator_floor/pred_floor`.
Reuse: `enable_continuous_locomotion`, `enable_continuous_depletion_intake=True`, `speed_cost_*`, `founder_mix`,
`continuous_dt`, `max_population`, `stability.py` cycle detector. **OFF byte-identical** across the full golden cross
(`_CONTINUOUS_ON_GOLDEN_HASH`, discrete, terrain). **ON** pinned by a new `_PREDATION_ON_GOLDEN_HASH` + an
ON-differs-from-OFF assertion + an ON-hash-invariant-under-`shuffle_creature_order` assertion. Determinism: frozen
pre-move snapshot, ascending-id scans, strict-`<` tie-breaks, capture/death events at a fixed interleaving point,
fixed float parenthesization, no rng in scans. **Re-pin ALL golden hashes AFTER the engine surgery lands.**

## 12. Runtime

Focal horizon **6000 steps** (multi-cycle phase-matching). Bucket index → O(N). Hundreds of two-population 6000-step
runs across the ladder = **hours**; a `compute-batch-runtime-preflight` (logistic-aware: distinguish exponential-then-
cycle from runaway via growth deceleration) is **mandatory** before the full batch; Rung-7 is a separately-gated batch
contingent on Rung-6. Size workers by RSS to avoid thread oversubscription.

## 13. Implementation surface

`ecology/genotype.py` (role field + last-field skip-guard); `ecology/engine.py` (`EcologyConfig` flags; the 3-phase
loop + frozen snapshot + capture/energy/`predation` death; the per-role mutate branch at ~1116; eat-heading reuse;
bucket index); `ecology/creature.py` (`Phenotype.move_hx/move_hy/move_d`, `lineage_root`; predator/prey heading);
`ecology/continuous_world.py` (bucket helper if hosted there); `ecology/evolvability/` (the new two-role invasion
runner + role-aware counter + cycle-phase-resolved estimator + the discriminator prongs); `experiments/exp248_*.py`+
(per-rung scripts + outputs/expNN.txt + the runtime preflight); `tests/test_predation.py` (OFF byte-identity, ON
golden, ON-differs, shuffle-invariance, role-counter exclusion, anti-cheat seam, determinism).

## 14. After this build

Per-rung honest logging (EXPERIMENTS.md + site + RESUME + a `coevolution-red-queen` direction card + `- Verifier:`
lines; **run the full fast suite after each log**). The headline (Rung-4-saturate ∧ Rung-5-invade ∧ Rung-6-all-prongs ∧
Rung-7-robust) would be the **first escape of the local-gradient wall** in the project — stated narrowly, with the
named caveats. A NEGATIVE/CAN'T-POSE extends the wall to co-evolution (a much stronger statement). Distil to a
coalescence MechanismCard or extend the local-gradient-wall BoundaryNote.
