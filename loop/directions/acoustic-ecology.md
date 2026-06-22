# direction: acoustic-ecology

**Question.** Under what predator-prey conditions does **acoustic information** become *locally
evolvable* — i.e. a physically constrained sound field gives prey (or predators) a drift-robust,
small-mutation fitness advantage — and when, if ever, does *voluntary* sound emission become
**functional communication** rather than mere noise or stigmergy?

**Why it matters.** The closed sense/memory/locomotion arc (Exp 199–247) established a robust
**local-gradient wall**: useful information is not enough — a trait only evolves if a *crude*
version pays a non-saturating, locally-climbable benefit. Sound is a NEW hidden-state channel that
the prior senses never had: a predator's *actions* (moving, attacking, killing) physically radiate
energy that propagates, attenuates, and arrives late — so a prey may **eavesdrop** on a predator
before contact. This direction asks whether that physical channel carries enough usable structure
to (a) measurably help when hearing is *gifted*, and only then (b) be locally evolvable, and only
much later (c) support *voluntary* signalling. It is built on the patch-mosaic predator-prey
substrate (Exp 257–262), the one place in this project where discrete spatial predator-prey
coexistence is robustly posed — the prerequisite for any acoustic eavesdropping to matter.

**Core rule (binding).** Do **not** implement communication. Implement **sound** as a physical
environmental substrate. Sound events are emitted by ordinary actions (movement, attack, feeding,
reproduction, death); they have a position, an emission time, an amplitude, a coarse frequency
spectrum (low/mid/high), and a duration. They propagate with inverse-square-like attenuation,
frequency-dependent absorption (high decays fastest), propagation delay (distance / sound_speed),
and time decay, against per-band ambient noise with an SNR detection threshold. Agents receive
ONLY imperfect local per-band intensity (optionally a directional gradient) — **never** source
position, source identity, event type, a predator/prey label, or any semantic tag. "Communication"
may be *claimed* only later, if voluntary emission becomes fitness-relevant and survives the
deflationary controls below.

**The provided-vs-earned line (binding, the crux).** Spectra (movement = low/mid quiet; attack =
broadband loud; death = broadband loud; feeding = mid; reproduction = low/mid) are **physical
signatures, not semantic labels** — the agent never learns "this band = predator." "Gifted hearing"
(a monomorphic perfect-ish listener) is trivially informative if the channel leaks; the honest tests
are (1) **shuffled** sound (decorrelate emission location from true source location ⇒ benefit must
vanish), (2) **silence / noise-only** (zero emissions ⇒ benefit must vanish), (3) **leakage audit**
(hearing must not work via exact position / identity / event-type), (4) **range calibration**
(detection must fall with distance — not be perfect at long range), and (5) **scalar-vs-banded**
(if a single scalar intensity works as well as the frequency bands, frequency is not yet
load-bearing). The binding evolvability metric (Rung 2+) is **invasion-from-rarity** (L41), never a
gifted demo or a 50/50 pairwise win.

**Experiment ladder.** (each one PROTOCOL iteration; each names its FAILURE)

1. **Acoustic Eavesdropping Expressibility Probe (Exp 268).** Build the gated acoustic substrate
   (`enable_acoustic_field`, byte-identical OFF, golden-hash guarded). On the Exp 257–259 coexisting
   patch regime (refuge + migration + asynchrony, nontrivial predation, escape possible, persistent),
   compare: (1) no field, (2) field present but no hearing (measure MI only), (3) gifted prey hearing,
   (4) gifted predator hearing, (5) gifted prey hearing + shuffled sound, (6) gifted prey hearing +
   silence/noise-only. Metrics: prey capture hazard, prey survival, predator capture success,
   population persistence, false-positive/false-negative detection, **acoustic mutual information with
   hidden predator proximity** (analysis-only — agents never receive it), cost-adjusted fitness.
   **ABORT/FAIL (do NOT run evolution):** gifted hearing has no measurable effect; shuffled/silenced
   matches real sound (channel carries no usable structure); hearing works only via leaked
   position/identity/event-type; detection perfect at long range (model too generous); scalar works as
   well as banded (frequency not load-bearing — mark, don't fail the whole probe).
2. **Local-gradient evolution of hearing (ONLY after Rung 1 passes expressibility).** A rare small
   hearing mutant (heritable `hearing_sensitivity` from near-zero) must **invade from rarity** in a
   resident deaf background (Evolvability Preflight / `trait_axis`). Required controls: zero-cost vs
   calibrated-cost hearing; shuffled sound; noise-only sound; scalar vs frequency-banded.
   **Verdict language:** PASS = hearing invades from rarity AND beats every control; MIXED = gifted
   hearing helps but small mutants do not invade; NEGATIVE = sound is useful only when gifted, free,
   or artifact-leaky.
3. **Predator counter-adaptation / acoustic Red Queen (only if Rung 2 PASSES).** Does a co-evolving
   predator's movement quiet down (stealth) or its attack stay loud, and does prey hearing track it?
   **FAIL** = no reciprocal trait change ⇒ eavesdropping is a one-sided static advantage, not a Red
   Queen.
4. **Voluntary sound emission (LATER RUNG — do not implement unless 1–3 pass).** Add a costed
   voluntary call action with evolvable frequency/amplitude/duration (kept DISABLED until earned). Do
   NOT call it an alarm. A signal becomes **"functional signalling"** only if it (a) correlates with
   hidden state, (b) changes receiver behavior, (c) improves receiver fitness, and (d) survives
   shuffled/random-playback controls. Anything less is noise or stigmergy.

**Discipline notes.** New substrate behind `enable_acoustic_field` (default False), byte-identical
OFF, golden-hash guarded (the `_RING_GOLDEN_HASH` patch-mosaic guard + a paired
acoustic-changes-behavior-ON test). The acoustic FIELD is purely observational; behavior changes
ONLY under `acoustic_response` + a `gifted_hearing` role, so "field present, no hearing" is itself
byte-identical to OFF (separating "does sound carry info" from "does acting on it help"). Spectra,
propagation constants, hearing traits, ambient noise, and SNR threshold are ALL configurable. Sound
is a physical signature, never a semantic label; agents receive only imperfect per-band intensity.
Conservative language only — functional, costed, heritable, eavesdropping, local gradient — no
claims of communication, intent, or agency beyond measured behavior. Honest prior: the local-gradient
wall has held across every prior channel, so the EXPECTED Rung-2 outcome is NEGATIVE/MIXED; Rung 1 is
the honest gate that decides whether evolution is even worth running.

**Stop condition.** Exhausted when Rung 1 returns a clear expressibility verdict (PASS ⇒ run Rung 2;
ABORT/FAIL ⇒ log the boundary and stop — sound carries no locally-usable structure on this
substrate), and — if Rung 1 passes — Rung 2 has an invasion-from-rarity verdict with its controls.
Then write the verdict + control evidence to EXPERIMENTS.md and distil to a coalescence
MechanismCard (acoustic-eavesdropping) or a BoundaryNote (sound-not-locally-evolvable).

**STATUS.** state: active · latest: Exp 268 (pending) · depends-on: patch-mosaic-red-queen (posed
coexistence), population-ecology (local-gradient wall) · reusable: ecology/acoustic.py +
enable_acoustic_field on ecology/patchmosaic.py (byte-identical OFF) · why: sound is a new physical
hidden-state channel (eavesdrop predator actions before contact) the prior senses never had ·
next-falsifiable: does gifted hearing measurably help AND beat shuffled/silent controls without
leaking position/identity (Exp 268)?
