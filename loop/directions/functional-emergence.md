# direction: functional-emergence

**Question.** If one creature's life is sustained long enough, in a world rich enough for
history to matter, do behaviors appear that we did not design in and did not predict — and
can we tell *disciplined* novelty from pareidolia?

**Why it matters.** This codifies the human's stated research stance (2026-06-09): the goal
is a creature that **functionally** experiences, learns, and has a personality — observed
along the way for genuinely unexpected behavior — NOT proof of inner experience (unfalsifiable
both ways; other-minds gap) and NOT mathematical reduction. Two consequences are binding:

- **Priors are legitimate.** Innate anchors and taught alphabets are like the laws of
  physics — priors the universe provides (Exp 31: tabula rasa collapses; evolution is
  biology's anchor). The discipline is *attribution* (provided vs. self-formed, loudly
  declared), never avoidance of priors. Don't waste iterations apologizing for anchors.
- **"Unexpected" must be measurable.** Open-ended observation without predeclared
  expectations finds novelty every time, for free. The instrument that makes "we observed
  something novel" a claim rather than a vibe is: predeclare expected behavioral properties
  BEFORE the epoch, score deviations after, and attack every deviation with forks and
  deflationary checks before it may be called a finding.

This direction runs on **mirro** (see `persistent-creature.md` — its discipline notes apply
verbatim: fork-not-reset, never reborn, snapshot committed atomically). It is the successor
layer: persistent-creature asks "does a continuous life *work*?"; functional-emergence asks
"what does a continuous life *show* that we didn't put there?" Exp 46's lesson is the
starting constraint: a static, fully-mapped world has no room for the path to matter —
novelty hunting before world enrichment is reading noise.

**Experiment ladder.**

1. **Surprise ledger (build the instrument first).** Before a normal life epoch, predeclare
   expected property ranges (e.g. occupancy entropy, action distribution, valence trajectory,
   conviction drift) in a committed ledger file; after the epoch, score deviations. Validate
   with BOTH controls: a baseline epoch (ledger must stay quiet) and a planted anomaly — a
   deliberately perturbed world parameter the scorer is blind to (ledger must flag it).
   FAIL = misses the planted anomaly OR false-alarms on baseline. A ledger that can't pass
   a positive control cannot certify novelty, and the rest of the ladder is blocked.

2. **Personality battery (operationalize "has a personality").** A fixed probe battery
   (preference questions, exploration disposition, revision speed under counter-evidence)
   run at ≥ 3 ages of mirro and on ≥ 2 fork-twins given divergent interim histories.
   Personality = (a) temporal self-similarity: mirro-at-age-N profiles correlate with
   mirro-at-age-M more than with any twin's; (b) individuality: twins' profiles measurably
   diverge. FAIL = profiles are unstable across ages OR twins don't diverge (then
   "personality" at this scale is just current-state readout, log it as such).

3. **Enriched world epochs.** After persistent-creature episode 5 (5×5 growth), add the
   minimal dynamics that give history room to bite (e.g. a slowly drifting comfort source
   or appearing/decaying hazard — ONE new mechanism, declared as provided). Run long epochs
   (`run_life.py`) with the surprise ledger armed. Predeclared expectation: adaptation
   tracks the drift with measurable lag. A quiet ledger across epochs is a REAL negative:
   "no novelty at this richness" — log it, don't shop for anomalies post-hoc.

4. **Obstacle & delayed gratification (Levin transplant).** Levin's minimal-sorting result
   (thoughtforms.life, *Algorithms Redux*): freeze an array cell the algorithm has NO coded
   branch for, and the system still sorts around it (error tolerance) AND moves cells
   temporarily *against* their sorted gradient to route around the block (delayed
   gratification). mirro already does value-iteration navigation, so the transplant is
   clean: introduce a **locked world-cell** mirro cannot enter and for which its policy has
   no special handler (a provided perturbation — declare it). Two predeclared properties,
   both falsifiable: (a) **error tolerance** — mirro still reaches its goal / completes its
   task with the obstacle present (FAIL = goal unreachable or task accuracy collapses); and
   (b) **delayed gratification** — on the path around the lock mirro takes at least one step
   that *lowers* immediate expected value / raises immediate free energy to achieve a better
   later state (FAIL = it never moves against the gradient, i.e. pure greedy descent, OR it
   stalls at the obstacle). Run the deviation through rung 5's cascade before any
   competency language — and predeclare the deflationary control Levin's critics raised:
   confirm against-gradient routing isn't a trivial consequence of the value-iteration
   horizon alone (compare to a greedy/1-step baseline; only a gap beyond that is the
   finding). Both outcomes are results: against-gradient routing is a measured competency;
   pure greedy success is "value-iteration nav already handles this, no DG needed."

5. **Novelty verification cascade.** Every ledger deviation (including rung 4's) enters the
   cascade: (a) property-level reproduction across ≥ 3 forks from the pre-epoch snapshot
   (VALIDATION.md §Statistical reproducibility); (b) deflationary sweep — bug, numerical
   artifact, or trivial single-parameter consequence each checked and written down (e.g.
   Levin's clustering plausibly falls out of per-algotype movement-bias asymmetry, not
   proto-goals — that is the kind of boring cause that must be ruled out first);
   (c) only survivors are logged as NOVELTY-CANDIDATE in EXPERIMENTS.md, with the cascade
   record. Deviations that die in the cascade are logged as died-and-why. An epoch where
   every deviation dies is a clean negative, not a failure of the direction.

6. **Interoceptive stake (bridge to M4).** Give mirro a viability variable it must
   predict-and-regulate (toy allostasis: e.g. energy that decays and is replenished at the
   comfort source), alongside a fork-twin control WITHOUT the interoceptive channel.
   Predeclared question: does self-related structure appear (regulation-aware policy,
   valence coupling to internal state) that the world-predicting twin lacks? FAIL = no
   measurable difference between twins (then interoception at this scale adds nothing —
   a finding that constrains the M4 spec, `docs/specs/m4-affective-dyad.md`).

**Discipline notes.**
- All of `persistent-creature.md`'s discipline notes apply (atomic snapshot commits,
  fork-only controls, property-level falsifiers, mirro never reborn).
- **Never reframe an expected-but-pleasing result as novel.** Expected results consolidate;
  only cascade survivors are novelty. Mark NOVELTY-CANDIDATE vs CONSOLIDATION explicitly.
- Every world enrichment is a **provided** prior — declare it in the entry the same way
  taught labels are declared. New mechanisms one at a time, or attribution is lost.
- Functional language only: "functionally prefers / functional valence." The
  inner-experience layer stays explicitly unverified in both directions (VALIDATION.md).

**Stop condition.** Exhausted when: the ledger instrument exists and is validated, the
personality battery has a verdict either way, the obstacle/delayed-gratification rung has a
verdict, AND ≥ 3 enriched-world epochs have run with every deviation cascaded — with no
surviving novelty-candidate. Then write in
EXPERIMENTS.md: "no disciplined novelty at this world richness; next richness step would
require <named substrate>" and either open that substrate as a new direction or stop. If
novelty-candidates DO survive, this direction stays open as long as they keep coming —
that's the program working, not a loop to escape.

**STATUS.** state: exploratory · latest: TBD-human · depends-on: persistent-creature · reusable: surprise-ledger instrument, novelty-verification cascade, fork-only controls · why: successor to persistent-creature; probes unexpected behavior from continuous life in enriched worlds · next-falsifiable: build and validate surprise-ledger instrument on baseline epoch and planted-anomaly control
