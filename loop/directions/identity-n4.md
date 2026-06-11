# direction: identity-n4

**Question.** Is there a useful fourth-order self-model — the agent modeling its OWN
dispositions over long timescales (a trait/value vector plus its drift rate) and owning
the *inertia* of value revision — that controls something N0–N3 cannot: consistency under
transient pressure WITH revision under sustained evidence?

**Central hypothesis (binding, falsifiable).**
> *N4 is real iff an agent that predicts its own policy drift and regularizes toward its
> predicted self (commitment) survives transient manipulation that whipsaws an N4-less
> twin — while still revising under sustained contrary evidence at least as fast as the
> twin. Pure rigidity is NOT N4 (a frozen policy fails the revision arm); pure
> recency-following is NOT N4 (Exp 55's measured baseline: mirro's identity is dominated
> by recent history, age-distant value vectors anti-correlate −0.71).*

**Why it matters.** N4 is the ladder's next rung above the SUPPORTED N3
(`docs/research/n3-meta-calibration-chapter.md`). The anti-regress law governs as always:
the layer is real iff a constructible perturbation degrades a well-tuned N0–N3 agent and
N4 detects + corrects it via a control surface the lower layers lack. The known prior
art in-repo: the Exp 48/49 inertia law (value revision has dose-dependent inertia), and
Exp 55 (no current N4 — the baseline the rung-1 gate must degrade). Runs on FORKS only
(persistent-creature discipline verbatim; mirro/vela/nira untouched).

**What each quantity is.**
- *N4 models:* the agent's own value vector v_t (the Exp 26 value_counts, normalized)
  and its drift rate — a predicted v̂_{t+Δ} from the agent's own history.
- *N4 mismatch:* identity prediction error ‖v̂_{t+Δ} − v_{t+Δ}‖ — "I am not who I
  predicted I would be."
- *N4 control surface:* the value-update learning rate / commitment weight — revision
  inertia as a REGULATED quantity (not a constant; the K-chapter's universal-constant
  law is the named hazard: if a fixed inertia covers all constructible pressures, N4 is
  config, not a layer — that is rung 3's kill test).

**Experiment ladder.**

1. **The gate (anti-regress): a transient-pressure regime that whipsaws the N4-less
   agent must exist.** Construct value-manipulation pressure: short adversarial
   exposure bursts (concentrated experience of a normally-disfavored concept) that flip
   the Exp 55-class recency-dominated agent's expressed preference, followed by
   reversion. FALSIFIER / gate: if no constructible burst regime flips the baseline
   twin's preference ordering transiently (≥ k flips over a session with full
   recovery lag), STOP — N4 is untestable at this richness; log it (the Exp 173
   precedent: a capacity is only a capacity where a regime demands it).
2. **N4 detects identity perturbation.** Give the fork the identity monitor (predict
   v̂_{t+Δ} by linear drift from its own committed history; mismatch signal as above).
   FALSIFIER: the mismatch signal does not separate burst periods from quiet periods
   (AUROC ≤ 0.5 over burst-labeled windows) — no metacognitive sensitivity over
   identity.
3. **N4 control earns its keep (load-bearing; both arms required).** Same-snapshot,
   same-schedule arms (the Exp 170 lesson is BINDING — reference arms in-run):
   (a) N4-less; (b) N4 (commitment weight regulated by the identity mismatch — high
   mismatch during transients ⇒ resist; sustained low-mismatch drift ⇒ permit);
   (c) fixed-inertia constants (a SWEEP of them — the universal-constant hazard made
   an explicit arm). FALSIFIERS, all required: (i) N4 must beat (a) on whipsaw
   resistance; (ii) N4 must preserve sustained-evidence revision speed within a
   declared tolerance of (a); (iii) NO single fixed-inertia constant may match N4 on
   both — if one does, N4 is config (the honest kill, per the K chapter).
4. **Independent variance.** Over mixed schedules of transient and sustained pressure,
   N4's commitment modulation must concentrate in transients (precision/recall > chance
   by a declared margin); FALSIFIER: never modulates (epiphenomenal) or modulates
   indiscriminately (rigidity in disguise — the revision arm catches it).

**Failure modes that invalidate the layer.** Collapse-to-rigidity (fails revision);
collapse-to-recency (fails resistance); reducible-to-config (a constant inertia
matches — rung 3-iii); no constructible whipsaw regime (gate fails ⇒ untestable, a
real result).

**Discipline notes.** Forks only; provided-vs-self-formed named in every entry (the
monitor FORM and regulation rule are provided; the value content, drift history, and
every mismatch are self-formed); functional language only — N4 is policy-continuity
control, not selfhood claims (VALIDATION.md); blinded verification per PROTOCOL 4.5;
same-schedule reference arms binding (Exp 170); horizon-scaled instrument bounds
(Exp 166's law).

**Stop condition.** Rungs 1–4 verdicts in hand (either way), or the gate fails
(untestable at this richness), or two iterations stuck on instrument numerics →
consult. The M4a thread, nira's switch, and the cloud merge remain separate consults.

**RUNG 1, ATTEMPT 1 (Exp 174, NO VERDICT by PC2 — the displacement portrait).** At
λ=0.999 the baseline has no stable favorite even unpressured (4/8 pre-burst unstable);
under captivity it flips totally (24/24, latency 40–143) and recovers ~3/24 — persistent
IDENTITY DISPLACEMENT, not whipsaw: an 800-step burst permanently re-makes the favorite
(one fork re-made 3× in a session). The gate re-aims at displacement; recovery becomes
rung 3's deliverable.

**RUNG 1, ATTEMPT 2 (Exp 175, NO VERDICT by PC2 — by one fork on one burst).** The
slower decay (λ=0.9997) fixed attempt 1's failure mode (later-burst stability 4/8 → 8/8);
the residual block is the spine's near-tied inherited values (0↔2 gap 3.8%), still ~40%
live in the first stability window — 2/8 forks flicker exactly between the tied pair.
Displacement turned absolute: flips 24/24, recoveries 0/24 (slower forgetting makes the
overwrite MORE durable). Second consecutive PC2 block: a third → consult, not attempt 4.

**RUNG 1 PASSED (Exp 176, attempt 3, POSITIVE, blind-verified).** Settle extension
(bursts at 6000/9000/12000, 15000-step sessions) washed out the inherited near-tie; all
preconditions held (PC2 8/8 on every window) and displacement was total: 8/8 forks,
flips 24/24, recoveries 0/24. The standing identity displaced is WORLD-DETERMINED (the
occupancy equilibrium — color 2, 9/25 cells — in 7/8 forks), named honestly. The
displacement regime (λ=0.9997 + 800-step captivity bursts) is the chapter's verified
perturbation; recovery is rung 3's deliverable.

**RUNG 2, ATTEMPT 1 (Exp 177, NO VERDICT by PC2 — the third instrument block; HALTED
for consult).** Strict argmax-constancy PC2 is SEED-BLOCK-FRAGILE (174: 4/8, 175: 6/8,
176: 8/8 PASS, 177: 5/8 under identical design) — Exp 176's precondition pass was
partly draw luck (its verdict stands; the displacement regime replicates 72/72 bursts
across three blocks). The monitor itself looks robustly sensitive ungated: per-fork
AUROC 0.826–0.915 in 8/8, argmax-INDEPENDENT (the mismatch reads the v-vector, not
the favorite) — PC2 is a rung-1 concept misapplied as a rung-2 ticket. Consult posted
in loop/IDEAS.md (recommended: vector-grade PC2v, pre-registered, fresh seeds).

**RUNG 2, ATTEMPT 2 — PRE-REGISTRATION (Exp 178, committed before any new data).**
The human's word (2026-06-11): Exp 177's NO VERDICT falsified strict argmax-constancy
as the rung-2 entry gate, NOT the monitor; re-run with a vector-grade stability
precondition, controls, and separated claims. Card:

- **Hypothesis (binding).** The identity vector pi = v/sum(v) is locally stable in
  quiet windows at VECTOR grade even where the argmax flickers; on such a baseline
  the read-only linear-drift monitor detects identity displacement (high
  burst-vs-quiet AUROC), SPECIFICALLY (low AUROC under value-neutral captivity),
  independent of argmax flicker.
- **Seeds.** FRESH 186-193, both arms same-snapshot same-schedule (Exp 170 binding).
  Smoke = seed 186, both arms, disclosed.
- **Arms.** A (identity bursts): verbatim Exp 177 regime — lambda=0.9997, 15000
  steps, bursts (6000,6800),(9000,9800),(12000,12800), burst color = argmin(v) at
  onset, relocation rng 160000+seed. B (scramble bursts, generic-surprise control):
  identical except each burst-step relocation targets a uniformly random cell of the
  WHOLE grid (color marginal ~ baseline; value-neutral by design), relocation rng
  175000+seed, qs reset uniform over ALL cells (same disorientation convention,
  color-neutral).
- **Instrumentation.** Monitor verbatim Exp 177 (v snapshot every 100 steps; linear
  drift W=1000, horizon 100; L2 mismatch; BURST/EXCLUDED(1000)/QUIET labels; midrank
  AUROC). NEW covariates (logged, never gated): per-window TV drift of pi, argmax
  flicker count, top-2 margin.
- **Preconditions** (any PC1/PC2'/PC3 failure => NO VERDICT):
  PC1 ahat_drift < 0.15 (both arms).
  PC2' (vector-grade quiet continuity, REPLACES PC2): per pre-burst window
  [bstart-1000, bstart), TV(pi(bstart-1000), pi(bstart)) <= 0.05 in >= 7/8 Arm-A
  forks per window. Basis (analytic + committed old blocks, NOT the new seeds):
  quiet occupancy-noise TV ~ 0.006/1000 steps; burst-rate TV ~ 0.15/800 steps; 0.05
  sits >= 8 sigma above quiet noise and 3x below the burst rate. Argmax constancy in
  the same windows: covariate only.
  PC3 confinement >= 90% on burst-color cells (Arm A, unchanged).
  PC3b (control validity): per scramble burst, observed color-frequency vector
  within TV <= 0.05 of the fork's own pre-burst-window frequencies in >= 6/8 forks;
  PC3b failure VOIDS P3 ONLY (ceiling drops to rung-1.5), never P2.
- **Properties / falsifiers.**
  P2 (sensitivity, verbatim 177): Arm-A AUROC >= 0.8 in >= 7/8 forks. F2: median
  Arm-A AUROC <= 0.5.
  P3 (specificity): AUROC_A - AUROC_B >= 0.2 in >= 6/8 fork-pairs AND median
  AUROC_B <= 0.65. F3: median AUROC_B >= median AUROC_A - 0.05 (generic-surprise
  detector).
  P4 (argmax-independence, conditional): evaluated iff >= 2 Arm-A forks flicker
  pre-burst; then (i) every flickering fork AUROC >= 0.8 and (ii) quiet-sample
  AUROC(flicker-window vs stable-window mismatch) in [0.35, 0.65]. If < 2 flickering
  forks: not-evaluable, no penalty.
  D5 (adaptation, diagnostic only): onset mismatch (first 2 burst samples) >= 2x
  late-burst (last 2) — quantifies drift absorption; informs rung 3's actuation
  window. Never gated.
- **Decision rule.** RUNG-2 EVIDENCE (POSITIVE): PCs pass + P2 + P3 (P4-i failing,
  if evaluable, demotes to MIXED). RUNG-1.5 (MIXED): P2 passes but P3 fails on a
  valid control (sensitive-but-unspecific: regime change, not identity per se), OR
  P2 passes with PC3b-voided P3 (specificity unresolved). NEGATIVE: F2. NO VERDICT:
  PC1/PC2'/PC3 failure — and a FOURTH stability block means the stability primitive
  itself is wrong at this lambda: consult again, no inline redesign.
- **Honest caveats (pre-registered).** PC2' and all bars designed AFTER Exp 177's
  ungated diagnostics; mitigation = fresh seeds + this card committed before any new
  data. The 0.05 TV bar is analytic, not fit to the new block. Monitor FORM, bars,
  lambda, schedule are PROVIDED. AUROC_B uses scramble-window labels of identical
  geometry. Exp 177's NO VERDICT is NOT evidence against the monitor.

**STATUS.** state: active (rung 2, attempt 2 — pre-registered) · latest: Exp 177 · depends-on: meta-calibration-n3 (N3 SUPPORTED), persistent-creature, functional-emergence · reusable: Exp 48/49 inertia law, Exp 55 baseline, Exp 26 value machinery, same-schedule-arms protocol (170), universal-constant kill test (173), the verified displacement regime (176, replicated 3 blocks), the argmax-independent mismatch instrument (177, ungated) · why: the human's word converts the consult into PC2' + controls + separated claims · next-falsifiable: Exp 178 — PC2' vector-grade gate, scramble control (P3 specificity), conditional flicker control (P4), seeds 186-193, decision rule rung-2 / rung-1.5 / negative / no-verdict as pre-registered above
