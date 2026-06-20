# direction: patch-mosaic-red-queen

**Question.** The homogeneous spatial-agent predator-prey chapter (Exp 248–254c) closed **CAN'T-POSE**:
across six gated natural mechanisms (logistic prey growth, decoupled resource-independent births, Type III
functional response, predator interference, predator self-limiting mortality, and combinations) on a single
shared spatial arena, the substrate could not host robust predator-prey coexistence — only predator
starvation, oscillatory collapse, or boom-bust. The diagnosis (Exp 250 capstone) was **substrate-level
destabilization**: the Bazykin-style ecology is stable as an ODE, but the discrete spatial-agent substrate
destabilizes it through discreteness, energy-buffer demographic lag, and spatial-stochastic encounters
(Exp 255 confirmed it: the SAME ecology coexists effortlessly when well-mixed). **Does the failure mean
discrete spatial predator-prey coexistence is impossible — or only that a single homogeneous arena is too
globally synchronized? Can a discrete-agent PATCH MOSAIC (semi-isolated local patches + local migration +
per-patch asynchrony + refugia) achieve GLOBAL bounded coexistence even when LOCAL patches individually
boom-bust, collapse, or rotate through phases — via metapopulation rescue/recolonization?**

**Why it matters.** It separates two very different conclusions. If a homogeneous discrete arena cannot
coexist but a patch mosaic CAN, the wall was **spatial synchronization**, not discreteness per se —
and discrete spatial predator-prey (the prerequisite for a discrete spatial Red Queen) becomes posable
through ecologically honest structure (asynchrony + refugia + limited migration), not another stabilizer
knob. If even a patch mosaic synchronizes and collapses, the discrete-substrate obstruction is deeper.
Either way it is a **substrate-posing** result, NOT a Red Queen / trait-evolution claim: co-evolution can
only be tested AFTER robust global coexistence is posed. This is explicitly pre-Red-Queen.

**Not this (binding framing).** NOT a homogeneous-grid stabilizer knob (do not reopen the Exp-248 arena).
NOT a well-mixed / population-sum ODE (a well-mixed Bazykin may appear ONLY as a clearly-labelled
REFERENCE_ONLY calibration oracle, never as the result). NOT a Red Queen evolution claim. Agents stay
DISCRETE individuals; survival/reproduction/migration are individual events; NO external evaluator selects,
protects, ranks, or teleports agents by global fitness. All randomness deterministic under fixed seed; any
shared engine path stays `enable_*`-gated, byte-identical OFF, golden-hash guarded.

**Experiment ladder.** (each one PROTOCOL iteration; each names its FAILURE)

1. **Exp 257 — patch-mosaic posing preflight (the binding test).** A minimal discrete patch mosaic
   (8–16 patches on a ring): each patch holds discrete prey + predators with LOCAL individual-based
   predator-prey dynamics, per-patch capacity, a per-patch refuge (predator-access) parameter, a per-patch
   phase offset (asynchrony), and local neighbor migration (separate prey/predator rates). Sweep migration
   (0 / low / med / high), asynchrony (synchronized / random / rotating / noisy), and refuge (none / weak /
   med / strong), with a homogeneous baseline that must reproduce the prior collapse class. Verdict =
   CAN_POSE_GLOBAL_COHABITATION iff global prey AND predator persist bounded over the horizon across the
   predeclared seed threshold, WITH local extinctions/recolonizations occurring (proving it is not a hidden
   well-mixed fixed point) AND cross-patch synchrony lower than the synchronized control AND intermediate
   migration beating both zero and high. **FAIL** = SYNCHRONIZED_COLLAPSE / PREDATOR_STARVATION /
   PREY_COLLAPSE / BOOM_BUST_UNBOUNDED across the sweep ⇒ patch structure does not rescue coexistence; the
   discrete obstruction is deeper than synchronization — log it.
2. **Load-bearing controls (gate before any claim).** Show medium-migration + asynchrony beats:
   high-migration (well-mixed/synchronized collapse), synchronized-phases (asynchrony is load-bearing),
   no-refuge and too-strong-refuge (refuge band is load-bearing, and does not trivially starve predators).
   **FAIL** = the passing arm is indistinguishable from a degenerate control ⇒ NO_VERDICT / confound.
3. **Topology / scale robustness (only if 1–2 pose coexistence).** Ring → 2D patch grid / small-world;
   8 → 16 → 32 patches; confirm the rescue regime is not knife-edge. **FAIL** = coexistence only at one
   hand-tuned point ⇒ fragile, not a robust substrate.
4. **Red Queen trait-invasion (ONLY after 1–3 pose robust coexistence).** Introduce a costed heritable
   prey-escape / predator-attack trait and test invasion-from-rarity under co-evolving vs static antagonist
   on the patch substrate. **FAIL** = trait inert / not expressed, or no rarity invasion. (Blocked until
   coexistence is posed.)

**Discipline notes.** Reuse the Exp 248–256 predator-prey toolkit (predation, logistic prey growth, Type III
response, predator interference, predator self-limit, the individual-based `ecology/wellmixed.py` as the
within-patch dynamics) — these live on branch `exp235-terrain-locomotion`, NOT yet on main; the patch-mosaic
direction depends on them. One new mechanism per iteration. Success may include LOCAL extinctions as long as
the GLOBAL mosaic persists (metapopulation persistence is a valid pass). Conservative language only.

**Stop condition.** Exhausted when Exp 257 + its controls (1–2) return a clear verdict: either
CAN_POSE_GLOBAL_COHABITATION (+ load-bearing controls) ⇒ proceed to topology robustness then the Red Queen
trait test; or a robust FAIL class across the sweep ⇒ the discrete obstruction is not mere synchronization,
log a BoundaryNote and close. Either way write the verdict + the load-bearing-control evidence to EXPERIMENTS.md.

**STATUS.** state: active · latest: Exp 260 (one-sided Red Queen local-gradient preflight — MIXED: predator-attack PASSES as a costless conditional ratchet (8/8 vs fast prey), prey-escape FAILS the local-gradient floor (sub-threshold + interior ESS ~2.5-3); both drift nulls neutral; the asymmetry behind Exp-259 predator-dominance) · ladder: 257 CAN_POSE · 258 robust topology×scale (POSITIVE-PARTIAL) · 259 Red Queen engages→predator-dominance (POSITIVE-SINGLE) · 260 refines: prey climb is weak/ESS-bounded, not a sweep (MIXED) · depends-on: Exp 248–256 toolkit + ecology/wellmixed.py + ecology/evolvability/metrics.py · next: does a COSTED predator attack symmetrize the arms race, or is the asymmetry terminal → close the direction?
