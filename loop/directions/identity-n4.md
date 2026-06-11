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

**STATUS.** state: active (rung 1, attempt 2) · latest: Exp 174 · depends-on: meta-calibration-n3 (N3 SUPPORTED), persistent-creature, functional-emergence · reusable: Exp 48/49 inertia law, Exp 55 baseline, Exp 26 value machinery, same-schedule-arms protocol (170), universal-constant kill test (173), the displacement portrait (174) · why: the ladder's next rung; the gate's instrument needs a stable baseline identity first · next-falsifiable: Exp 175 — λ=0.9997 (declared, window-theorem band), PC2 retained, P1′ = flip ≥7/8 AND baseline recovery-failure on ≥2/3 bursts (the regime degrades the layerless agent persistently)
