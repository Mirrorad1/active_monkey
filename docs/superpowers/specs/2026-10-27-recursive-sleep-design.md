# Sleep cycles for the research lab and its creatures

Date: 2026-10-27. Status: proposed design; no implementation or new experimental result.

## Intent

Apply one learning lifecycle at two levels of active-loop:

1. The **research lab** learns from conducting experiments: retain evidence, consolidate
   findings and boundaries, rehearse their application, and improve future experimental design.
2. A **creature inside an experiment** learns from experience: retain encounters, consolidate
   a predictive model, rehearse decisions, and act using the resulting knowledge. Later,
   ecological selection can test whether paying for that machinery is advantageous.

The human requested both levels, including learning, knowledge, and selection. Short-session
offline replay is the first candidate creature test, rather than the entire scope. The
particular staging and budgets below are design choices by Codex, not prior human decisions.

The central question is whether accumulated experience changes subsequent performance, beyond
what equally resourced replay or ordinary archival retrieval already accomplishes.

## Inspiration and limits of the analogy

Google's [sleep overview](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/)
separates wake interaction from offline consolidation and synthetic practice. The
[underlying paper, sections 3.2–3.4](https://arxiv.org/html/2606.03979v2), implements consolidation
with new low-rank experts in slower memory blocks, knowledge seeding from a pre-update teacher,
and a separate dreaming objective. Those are concrete trainable-model mechanisms.

Here, the first lab implementation would change persistent artifacts and the material supplied
to a research session. It would **not update the hosted assistant's weights**. The creature
implementation would update its own numerical learner. Neither implementation would initially
reproduce the paper's MoE architecture or establish its benchmark claims. Use “sleep-inspired”
for this adaptation. Literal knowledge seeding into expanded capacity is a later, separately
controlled experiment, not a name for saving a Markdown summary.

## One lifecycle, two adapters

```text
WAKE: encounter evidence and act
  -> RECORD: retain the episode and its provenance
  -> CONSOLIDATE: build a bounded, durable representation
  -> REHEARSE: use that representation on candidate situations
  -> TEST / SELECT: measure consequences under declared rules
  -> RETAIN: carry warranted changes into the next wake period
```

Each stage consumes an explicit state and returns a new state plus a receipt. Sharing this
contract does not require identical learning algorithms or a recursive agent swarm.

| Stage | Research-lab adapter | Creature adapter |
| --- | --- | --- |
| Wake | Conduct one preregistered experiment | Observe, act, receive sensory consequences |
| Record | Preserve script, raw output, checkpoints, verdict, caveats | Bounded private log of observed transitions |
| Consolidate | Evidence-linked claims, failure boundaries, reusable methods | Fit or refine a learned predictive model from recorded experience |
| Rehearse | Draft counterexamples, controls, predictions, and candidate studies | Replay observations; later simulate with its learned model |
| Test/select | Independent review and fresh tests of candidate methods | Behavior tests; later survival and reproduction in the world |
| Retain | Versioned active brief and supported procedural advice | Persistent learned state; inherited capacity is tracked separately |

Three different uses of “selection” must remain identifiable: choosing memories for a bounded
buffer, testing candidate procedures/models, and ecological differential reproduction. Success
in the first two does not demonstrate the third.

## Shared records and invariants

A sleep receipt records scope (`lab` or `creature`), before/after state references and hashes,
source episode IDs, algorithm/configuration version, compute/memory budget, generated items,
rejected items and reasons, evaluation partition, metrics, and retention decision. References
resolve to existing artifacts; missing evidence stays explicitly missing.

Keep three kinds of state separate:

- **Experience:** immutable records of what actually happened, including failures.
- **Knowledge:** revisable estimates or claims, each with supporting evidence and scope.
- **Procedure:** how memories are sampled, models updated, or studies designed.

A synthetic item carries `origin=generated`, its generator state, and source IDs. It cannot
become an observed fact merely by being rehearsed. Retiring an item from the bounded active
memory never deletes historical evidence. Corrections supersede earlier claims through new
records; they do not rewrite an old experiment.

The lab's knowledge never becomes privileged input to a creature. World hidden state and
correct-answer mappings remain evaluator-only. Shared machinery means shared lifecycle and
provenance conventions, not shared access to answers.

## Adapter A: research practice

### Existing pieces to reuse

- `EXPERIMENTS.md` and `experiment_bundles/`: experience and reproducibility records.
- `mechanisms/`, `boundary_notes/`, `geometry_maps/`: semantic knowledge and limits, following
  `docs/ARTIFACT_SPEC.md` and the current coalescence validator.
- `loop/LESSONS.md`: existing procedural memory, with its stable rule numbers.
- `meta_monkey/`: passive process episodes and advisory preflight.
- `loop/compose.py`: eventual entry point for supplying a bounded brief to the conductor.

The missing component is a versioned sleep receipt plus a small, source-linked **active brief**:
what the selected evidence establishes, what it rules out, where the next question remains
open, and which controls are load-bearing. Summaries are read aids; linked evidence is the
authority. Use existing artifact types where they fit rather than duplicating the library.

### A lab sleep cycle

1. At the end of one completed iteration, index its committed bundle and passive process
   record. A default review window is three newly completed iterations, with a hard ceiling
   of eight retrieved bundles. Unfinished runs are marked unfinished and cannot support a
   scientific result.
2. Retrieve relevant earlier mechanisms and boundaries. Resolve conflicting statuses by
   preserving both evidence trails and identifying the later correction or narrower scope.
   A raw numerical claim requires an output reference, not only a prior summary.
3. Propose a compact active brief (initial cap: 1,500 tokens), with separate supported
   findings, negative constraints, and untested hypotheses. New procedural advice remains a
   candidate unless already mandated by the protocol or supported by a demonstrated guard.
4. Rehearse: apply the candidate advice to a counterexample and draft up to three falsifiable
   next studies, including the simplest competing explanation. These are study proposals,
   not simulated scientific results.
5. Independently review source fidelity, omitted negative evidence, and whether a proposed
   study could fail. Retain a source-faithful brief as an advisory input. Promoting a new
   process rule requires the usual reviewed change and a durable guard when applicable.
6. Supply the brief at the next wake, alongside the binding protocol and source pointers.
   Record which sources were actually consulted and whether the candidate advice was used.

This adds structured consolidation and rehearsal around the existing workflow. It does not
give `meta_monkey.preflight` authority to select experiments, grade science, or change its
current advisory-only contract. The conductor still follows `loop/ROUTING.md`; independent
verification still follows `loop/PROTOCOL.md` §4.5.

### How to measure lab improvement

First perform a source-fidelity/integration audit on a fixed, disclosed archival set. This
can establish that the brief is accurate and usable; the present assistant has read the
history, so this is **not a blinded test of learning or foresight**.

Then preregister a prospective comparison on fresh design tasks, with a fixed model/config,
source cutoff, token/tool-call budget, generation budget, and review rubric. Compare (a)
ordinary access to the archive plus current lessons against (b) the same access plus the
sleep brief and rehearsal. Charge summary construction, retrieval, and rehearsal to the
treatment budget; give the baseline equal total compute and source access. A separate reviewer
scores proposals without condition labels before outcomes are exposed.

Primary outcomes: missed required controls, unfalsifiable predictions, evidence-attribution
errors, and later invalidated designs per task. Also report total time/cost and useful studies
completed. A higher proportion of positive experiments is **not** a success metric. Exact task
count, allocation, effect threshold, and stopping rule must be fixed before running this
prospective test. Until then, describe the lab component as an instrument, not an improvement.

## Adapter B: creatures and ecology

### State and integration

The creature needs: a bounded episodic buffer, a persistent learned model, a replay budget,
separate random streams for learning/rehearsal and the environment, and checkpointable sleep
state. Prefer an adapter to `AgentState` / `AgentCheckpoint` rather than a replacement format.
Ecology's `snapshot` / `restore` / `fork` provide counterfactual histories.

`ecology.runtime.replay` re-executes a simulation; it is **not memory rehearsal**.
`ecology.runtime.distill` picks a founder genotype from successful lineages; it is **not
within-creature knowledge consolidation**. Keep these functions' meanings intact.

Record only information the animal actually receives: observation, chosen action, observed
outcome, lived timestamp, and its prior internal state when needed for faithful inference.
Reconstructing a trace must preserve action/outcome timing. No access to unchosen rewards,
the true resource field, hidden labels, future outcomes, or another creature's private log.

### Replay is computation, not additional evidence

Start with observed replay before adding synthetic dreams. Every real encounter has a unique
ID and one evidence weight. Repeated optimization passes may refine latent assignments or a
compressed model, but may not repeatedly add the same encounter as independent evidence.

This is especially important for `DirectHeadAgent`: `observe_feedback` both decays Dirichlet
counts and incorporates observations. Calling it repeatedly in sleep would change effective
forgetting and confidence, confounding replay with a different evidence-weighting rule.

A candidate inference algorithm should instead refit from an immutable pre-window parameter
checkpoint plus unique logged observations, rebuilding the window's sufficient statistics on
each pass rather than cumulatively appending them. Decay is indexed by **lived time**, not
replay-pass count. Priors and already consolidated evidence are not re-added on every pass.
The exact refit objective must be specified and tested before implementing the first run.

Preserve the live creature's belief continuity and age; a replay workspace is a counterfactual
copy, not a reset of the lived individual. Archive its before/after state. If a method must
recompute current beliefs, document that inference from retained history explicitly.

Later dreams use only the learned world model. Generated outcomes are predictions, with no
new evidential weight. Evaluate dreamed policies through subsequent real interaction or an
independent held-out instrument. If replay already fails, do not hide that failure by adding
dreaming, capacity expansion, and a new policy simultaneously.

### Costs and inheritance

At the individual-learning stage, impose sleep bouts and measure their effect; do not call
this evolved sleep. In a later ecology rung, sleep consumes simulation time, energy, bounded
memory and foregone foraging. The world and other creatures continue during an individual's
sleep. Sleeping does not confer invulnerability. First test imposed sleep's net benefit at
fair cost before adding heritable scheduling/capacity parameters.

Inherited state includes learning/sleep **capacity and priors**. Acquired episodic buffers,
learned maps, and realized parameters start empty or from the declared innate prior in a new
offspring. Copying them across reproduction would be a separately named cultural/Lamarckian
treatment. Preserve the genotype/phenotype split from Exp 275–277.

Ecological selection is births and deaths under the existing engine, without a global judge
protecting promising sleepers. Sleep helping a gifted individual does not establish that its
capacity invades from rarity, evolves, or escapes the local-gradient wall. Use the existing
evolvability preflight, including neutral/drift and invasion controls, before those claims.

## Staged experiments and failure conditions

Experiment numbers are reserved only at launch after checking the live log. These are
candidate rungs, not predeclared results or amendments to earlier verdicts.

| Rung | Question and comparison | What would fail |
| --- | --- | --- |
| L0 | Can a lab sleep cycle produce an accurate, bounded brief from committed sources? | Unsupported claims, missing negatives, broken provenance, or unbounded context |
| C0 | Can fixed short-session records support better inference after offline processing? | No meaningful improvement over matched rehearsal, or gains depend on duplicate evidence |
| C1 | Does retained knowledge survive interference and help subsequent wake behavior? | Only replay-set gains, forgetting old tasks, or inability to revise after real change |
| C2 | Does imposed sleep benefit an ecological creature after its costs? | No realized net benefit, or an invalid/unstable selection substrate |
| C3 | Can sleep capacity be selected from rarity, with knowledge kept distinct from genes? | Neutral drift, failure to invade, or improvements caused by copying learned state into offspring |
| L1 | Does the lab adapter improve fresh experimental design at equal budget? | No preregistered design-quality benefit, or improvement explained by extra compute/source access |

L0 can be built without changing the animal. C0 is the first small-scale application of the
same lifecycle; C2/C3 reconnect it to ecology once there is a useful learner. L1 is a separate
test, so a positive creature result cannot validate the research procedure that found it.

### Recommended first creature instrument (C0)

Use the existing short-session affective dyad, whose learning-side limitation was isolated
in Exp 219–221, as a cheap mechanism test before an ecology port. This is a staging choice,
not a replacement of the broader population question.

Fork identical end-of-wake checkpoints after 100 lived interactions. The replay arms receive
exactly the same logged history and no further environment queries. Compare:

- Sleep with no learning (state-preservation control).
- Ordinary replay/refitting, with the same evidence semantics, budget, and capacity.
- The proposed consolidation/rehearsal rule, changing one specified factor.
- A longer-wake reference with 300 real encounters, disclosed as extra experience rather
  than an equal-data control.

Retain the constant-response and broken-association controls. A shuffled replay trace is
synthetic corrupted data and must remain a diagnostic, never real evidence. Candidate
parameters, stored-trace IDs, prior-evidence weights, replay steps and elapsed physical
time must be logged. Allocate enough fresh seeds for effect-size-plus-count criteria under
`loop/VALIDATION.md`; do not choose seed IDs or thresholds from observed improvements.

Use read-only `correct_select` for discrimination and realized behavior on a separately
reserved evaluation stream. Fix evaluation precision to separate learning from exploitation.
Measure retention on a previously acquired task separately from adaptation after a true world
change. Distinguish held-out episodes of a familiar mapping from transfer to unseen mappings;
the former is not compositional generalization.

Before this can run, write an exact inference objective, intervention, compute accounting,
fresh seed set, effect-size threshold and falsifier into the runner docstring. A replay gain
over idle alone establishes usefulness of extra computation; a gain over matched ordinary
replay is required to support the additional sleep mechanism. If exact once-weighted inference
is already optimal for this learner, an offline null is a result, not a reason to count records
twice. An expressiveness failure may motivate a separately preregistered neural/model-based
learner later.

## Implementation boundary and delivery order

After design approval, write the implementation plan, then build one bounded slice at a time:

1. Lab sleep receipts + source-linked active brief, using existing artifacts and advisory
   preflight. Verify source resolution, negative retention, deterministic serialization,
   budget enforcement, and inability of generated proposals to overwrite evidence.
2. A standalone C0 runner/helper and focused tests for once-weighted evidence, correct temporal
   binding, immutable input checkpoints, and sleep-OFF equivalence. Smoke-time before batching.
3. Run and independently verify **one** preregistered experiment. Commit script, raw output,
   append-only log, site entry and required snapshots together under the existing protocol.
4. Only after the corresponding gates pass, add cross-session retention, an ecology adapter,
   heritable sleep capacity, and the prospective lab comparison.

Do not edit FROZEN paths, silently activate a new `/loop`, change routing/validation authority,
merge unrelated work, or reset a persistent creature as part of these slices. No new dependencies
are needed for the design. Any later dependency is an explicit implementation-plan decision.

The useful end state is a lab that carries justified knowledge forward while studying
creatures that do the same. Whether either improves, and whether selection favors the
machinery, remain three independently testable questions.

## Implementation status (2026-10-27)

The approved first slice L0/C0 is implemented; see the execution plan completion evidence,
`meta/sleep/README.md`, Exp279 and its bundle. C0 is POSITIVE-SINGLE at the stated toy scope;
L0 demonstrates reviewed evidence fidelity/integration, not improved lab research quality.
C1 retention/revision, C2/C3 costed ecology/inheritance, and L1 prospective quality are unrun.

## C1 continuation outcome (2026-10-27)

Exp280 is NEGATIVE / NEW INSIGHT under its independently verified all-conjunct rule.
Unchanged-code retention passes; acquisition, revision and both baseline comparisons fail.
Direction halts at C1; no automatic ecology escalation. Reusable stronger incremental
refitting and a source-linked negative brief are retained; online-life, ecology and lab-quality
benefits remain unestablished. See Exp280 raw/audit/verifier/bundle and continuation plan.
