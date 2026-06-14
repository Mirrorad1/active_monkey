# direction: hidden-state-memory (Phase 3)

**STATUS:** pre-registration / design (committed BEFORE any engine code or data, per
PROTOCOL step 2 + VALIDATION). Depends on Phase 2.5 (generic Gate C, PR #49) for the
binding gate to run on a non-thermosense trait.

**Question.** When the world has a slowly-switching HIDDEN state that a single noisy
observation cannot resolve, does a small heritable increase in *memory / inference
capacity* (integrating more past observations into a belief) win more reproductive
success **locally** — i.e. is information-processing capacity evolvable where a costed
scalar *sense* was not?

**Why it matters.** The sense-evolution arc (Exp 199–207, closed-negative) established
that a costed scalar sensor never becomes a functional organ at this substrate: the
LOCAL selection gradient at the resident is ≤0 across six escapes, because every sense
benefit *saturated* (a crude sensor grabbed the easy part; precision's marginal payoff
was too small to out-breed its cost). The untested, structurally-different regime is
**partial observability of a hidden state**: there, the value of an extra remembered
observation is *non-saturating up to the mixing timescale* — averaging k noisy cues cuts
estimation variance ~1/k while the belief is still wrong. This is the clean bridge from
ecology to **world models** (an internal estimate of hidden state used to choose actions
before the world reveals the answer). The correct research sequence is
**hidden state → memory/inference → active sensing → signaling**; communication is
downstream (honest signaling needs an agent with useful private beliefs first), so it is
explicitly NOT next.

**Naming honesty (binding).** "Memory" here is a FUNCTIONAL belief-integration horizon,
not a claim of episodic memory or sentience. The hidden mode, cues, payoff coupling, and
the belief estimator are PROVIDED substrate; what is TESTED is whether the *heritable*
memory_horizon has a positive local gradient. The trait is distinct from the existing
`memory_length` (a complexity-blend trait) and the learned resource map (an EMA over
cells) — those are reported as the relevant confounds to neutralise.

---

## Mechanism (PROVIDED; gated `enable_hidden_mode`, default OFF ⇒ byte-identical to
Exp 194–207, hash-guarded — the L16/L25 regression discipline)

- **Hidden mode.** A global binary mode `m(t) ∈ {0,1}` that switches with small per-step
  probability `mode_switch_prob` (slow; expected dwell ≈ 1/p ≫ 1), drawn from the WORLD
  rng (deterministic given seed). The mode is TRULY HIDDEN: creatures cannot read `m` or
  `t` directly (direct-read guard). Stochastic switching (not a fixed schedule) so no
  learnable clock exists.
- **Mode-gated food.** Each cell has a fixed `cell_type ∈ {0,1}` (a fixed spatial
  pattern). Food regenerates ONLY in cells whose `cell_type == m(t)` — so the "good" half
  of the world flips when the mode switches. A creature must be on a mode-matching cell to
  eat. (Anti-cheat: intake is the UNCHANGED `consume()`; nothing is written as
  f(memory_horizon).)
- **Noisy cue.** At its cell each step a creature observes `cue = m(t) + N(0, cue_noise)`
  (ONE rng draw, made ONLY in the gated ON branch ⇒ OFF path rng-identical). One cue is
  unreliable when `cue_noise` is non-trivial.
- **Memory / inference.** Each creature keeps a ring buffer of its last `memory_horizon`
  cue observations; belief estimate `m_hat = round(mean(buffer))`. `memory_horizon = 0`
  ⇒ react to the single current cue; `k` ⇒ average the last k (variance ~1/k). The
  ON-branch policy steers toward the nearest cell of type `m_hat` (the inferred good half).
- **Memory cost (floored, never free).** `upkeep = memory_upkeep_floor +
  memory_cost_slope · memory_horizon`, charged each tick (gated; like `thermosense_upkeep`).
- **New state:** a `memory_horizon` Genotype trait (INT, bounds e.g. [0, 12], default 0
  ⇒ regression-safe, rng-skipped when `enable_hidden_mode` off, mirroring the thermosense
  trait guard) + a per-creature cue ring buffer.

**Anti-cheat (binding; extends no-direct-h-reward + the byte-identity guard):**
- No food/fitness is written as f(memory_horizon); memory keys ONLY (a) how many cues are
  averaged into `m_hat` and (b) the upkeep cost.
- **Perfect-percept null:** at `cue_noise = 0`, `m_hat == m` for EVERY memory_horizon ⇒
  runs byte-identical across memory_horizon (proves memory pays ONLY by denoising). This
  is the disconnect recipe for the Gate-G byte-identity guard:
  `disconnect_overrides = {enable_hidden_mode: false}` (full) and a perfect-percept guard
  at `cue_noise: 0.0`.
- **Frozen-map / learning-rate guard:** freeze `learning_rate` (the EMA map) so the
  learned resource map cannot substitute for cue inference (the Exp 201 confound).

---

## Pre-registration — Phase 3 rung 1 (the LOCAL-GRADIENT PREFLIGHT; the binding test)

Run the Evolvability Preflight (NOT a full evolution batch) on the memory axis FIRST.

- **TraitAxis.** `name="memory_horizon", h_trait="memory_horizon", resident_value=1,
  mutant_value=2 (one heritable step), low_value=0, high_value=8, enable_flag="enable_hidden_mode",
  freeze_flag=None (freeze via mutation_rate=0; no engine freeze hook for this trait),
  backend="memory", disconnect_overrides={enable_hidden_mode: false}`.
- **Gates.** `local_pairwise_gradient` (BINDING; generic Gate C from Phase 2.5),
  `invasion_from_rarity`, `null_guards`. (Gates A/B/E/F stay thermosense-only — out of
  scope; the binding question is the local gradient.)
- **Regime (FIXED on a disclosed pilot before the fresh-seed verdict).** `cue_noise`,
  `mode_switch_prob`, `memory_cost_slope`, world size, horizon — tuned to the regime where
  a single cue is unreliable (cue_noise high enough), the dwell is long enough that
  averaging helps, and the memory cost is affordable. Seeds: a small smoke list +
  documented 8-seed batch.

- **Hypothesis (one sentence).** Under hidden-mode partial observability, memory_horizon
  has a POSITIVE local gradient at the resident: the single-step mutant (1→2) invades the
  resident in a fair common garden in ≥7/8 seeds, because the inference benefit is
  non-saturating near the resident (variance reduction still large) and exceeds the small
  memory upkeep — the FIRST positive local gradient in the program's trait-evolution arc.

- **Predictions if TRUE** (property-level, ≥8 seeds, report ALL):
  - **P1 determinism.** Same seed → identical event hash (gated mechanism).
  - **P2 validity.** Common-garden populations valid (not collapsed/exploded) in ≥6/8 seeds.
  - **P3 (CORE, binding).** local_pairwise_gradient verdict = POSITIVE_LOCAL_GRADIENT
    (mutant 2 beats resident 1, ≥7/8 wins, mean effect > 0).
  - **P4 (gifted/denoising sanity).** A high-memory creature reaches the mode-matching
    cells more often than a memory-0 creature when GIFTED (mechanism liveness — reported,
    not the verdict; this is the analog of the thermosense gifted check).
  - **P5 invasion.** invasion_from_rarity = INVADES (rare mutant increases).

- **Falsifiers (each ⇒ the named outcome; a NEGATIVE here is a first-class result —
  it would extend the arc's wall from scalar senses to information-processing capacity):**
  - **F1.** Non-determinism ⇒ NEGATIVE (infra).
  - **F2 (CORE).** local gradient is NEGATIVE or FLAT (mutant fails to invade) ⇒ NEGATIVE:
    even non-saturating inference does not pay locally at this substrate.
  - **F3 (anti-cheat).** byte-identity guard fails (memory leaks through an undeclared
    path), OR perfect-percept null (cue_noise=0) is NOT byte-identical across memory ⇒
    NO_VERDICT (artifact) — fix the mechanism before any verdict.
  - **F4 validity.** populations collapse in a majority ⇒ NO_VERDICT (retune the regime,
    disclosed).

- **Verdict rule.** POSITIVE (the breakthrough: capacity is locally evolvable here) iff
  P1 ∧ P2 ∧ P3 ∧ P5 and no anti-cheat guard fails. NEGATIVE iff F2 (the wall generalises).
  NO_VERDICT iff F3/F4. Only on a POSITIVE preflight do we run rung 2 (a full evolution
  batch: does memory_horizon climb de novo to the inferential optimum?).

- **Honesty stakes (written before data).** The regime is tuned (disclosed pilot) to be
  FAVOURABLE to memory (unreliable single cue, affordable cost) — so a NEGATIVE is the
  STRONG conclusion. Predicting POSITIVE and getting NEGATIVE is an honest negative, logged
  as such. The interior optimum is expected (past ~dwell, stale pre-switch cues mislead, so
  the gradient should turn negative beyond an optimum) — but the BINDING question is the
  sign at the resident (1→2), the only thing that decides evolvability. The belief
  estimator, mode dynamics, cues, and payoff coupling are PROVIDED; the heritable quantity
  is memory_horizon. Bridge claim bounded to this toy substrate.

---

## Experiment ladder
- **Rung 1 — the local-gradient preflight** (above): the binding test; cheap; gates a batch.
- **Rung 2 — full evolution** (ONLY if rung 1 is POSITIVE): does memory_horizon evolve de
  novo from 0/1 toward the inferential optimum, and stop there (interior optimum)?
- **Rung 3 — active sensing** (later): let the creature pay to SAMPLE an extra cue
  (information-seeking), bridging to active inference. Communication stays downstream.

**Stop condition.** Closed when rung 1 returns NEGATIVE (wall generalises to capacity) or
the full ladder reaches a documented optimum/wall. A NEGATIVE rung-1 is a clean, valuable
result, not a failure.

---

## RUNG-1 PILOT RESULT (2026-06-14, disclosed) — NO_VERDICT (demographic collapse / F4)

The disclosed regime pilot (pilot seeds {100,101,102}, NOT the verdict seeds) found that
the as-built **mode-gated food mechanism collapses the population** below a drift-safe size,
so the local-gradient test is not yet valid. Regimes tried (all disclosed; none adopted):
- R1 fast/cue0.6/cost0.01 (12×12, cap10): FLAT, pops 19–40.
- R2 slow/cue0.8/cost0.005 + R3 mid/cue0.7/cost0.005: read POSITIVE (3/3, mutant fixates)
  **but at pops 15–42** — drift-dominated.
- Raised capacity (cap30, regen1.0) and larger grid (20×20, cap20, slower switch): pops
  collapsed FURTHER (5–34, several **extinctions** → 0), not up.

**Drift control (decisive):** at `cue_noise=0` (perfect cue ⇒ memory has NO denoising
benefit, only extra lag+cost) the mutant **still fixated just as often** as under noise.
A real memory signal must vanish here; it did not ⇒ the apparent "wins" are small-population
**drift**, not selection (the Exp 202 trap; L21/L24). I did NOT adopt R2/R3's POSITIVE
(that would be tuning-to-a-positive on a drift artifact).

**Honest verdict: NO_VERDICT (F4 validity).** Mode-gating (only the matching half
regenerates; the other half starves) halves carrying capacity AND each switch starves the
previously-good half — too harsh to sustain a common garden.

**Proposed mechanism revision (next iteration, re-piloted):** make the hidden mode a
MILDER payoff differential — BOTH halves regenerate, the mode-matching half regenerates
MORE (or the wrong half yields reduced, not zero, food) — so correct inference still PAYS
but the population survives (the Exp 204→205 survivable-loss lesson). Then re-pilot for a
drift-safe pop (hundreds), re-confirm the perfect-percept null fixates NEITHER way, and run
the verdict on fresh seeds. Until then rung-1 has no admissible verdict.

---

## Build order (each a scoped PR; the engine change is gated + byte-identical OFF)
1. Engine mechanism (`enable_hidden_mode`: mode dynamics + cell_type + mode-gated food +
   noisy cue + per-creature cue buffer + belief steer + memory upkeep) + `memory_horizon`
   Genotype trait (regression-safe) + tests: **byte-identical OFF (hash guard)**,
   **perfect-percept null (cue_noise=0 ⇒ byte-identical across memory)**, and a
   **liveness check** (gifted high-memory out-forages memory-0). HALT if liveness fails
   (mechanism wrong).
2. `memory` TraitAxis + a preflight config (`experiments/configs/preflight/memory_horizon_local_gradient.yaml`).
3. Run the rung-1 preflight; log the verdict honestly (POSITIVE/NEGATIVE/NO_VERDICT).
4. Full evolution batch only if rung 1 is POSITIVE.
