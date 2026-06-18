# direction: social-emergence

**Question.** If mirro is treated as a **common ancestor** — forked into peer lines that
speciate by living divergent histories, then reunited through a communication channel — do
**interactions between them** produce behavior (coordination, social transmission of value,
dialects) that we did not design into either one?

**Why it matters.** It is the honest next rung of the moonshot after the solo arc
(perceive → want → act → form values → answer in words). Every result so far is about ONE
mind in isolation. PREMISE.md's recipe (embodiment + grounding + continuous registered
experience + one anchor + taught labels) has only ever been exercised solipsistically. This
direction asks whether the SAME recipe, run on two clade-mates who can sense each other,
yields *social* structure — the first place "emergent behavior we didn't put there" could
appear between agents rather than inside one. It also gives the persistence substrate its
natural shape: not a single sacred spine but a **family tree** rooted at mirro, where a
branch lived long enough in a different world is, functionally, a different species.

This direction runs on the clade. All of `persistent-creature.md` and
`functional-emergence.md` discipline applies verbatim (fork-not-reset of the trunk,
atomic snapshot commits, predeclared property-level falsifiers, novelty cascade). Two
substrate facts are the starting constraints:
- **Creatures are currently solipsistic.** A `Creature` lives alone in a grid and has *no
  representation of another agent*. Interaction requires NEW provided substrate — a shared
  world and/or a channel — and every piece of it is declared as a prior, like taught labels.
- **The common ancestor is a frozen git commit.** Forking stamps `lineage:
  ["mirro@AGE#HASH"]`; record that ancestor commit in every clade experiment so any
  divergence is causally attributable to post-fork history (the Exp 26/47 logic, on a tree).

**Experiment ladder.**

1. **Promote a branch to a committed peer spine (clade infrastructure — consolidation).**
   Fork mirro at a committed checkpoint into a second *committed* line
   (`creature/state/<name>/`), raise it in a divergent world, and verify: lineage records
   the shared-ancestor hash; both lines are independently resumable; mirro's trunk is
   byte-identical afterward (untouched). FAIL = lineage omits the ancestor hash, the lines
   are not independently loadable, or promoting the branch advances/corrupts the trunk. No
   emergence claim — this is the family-tree plumbing.

2. **Shared-world co-presence (minimal multi-agent substrate).** Place two clade-mates in
   one grid where each one's cell is observable to the other via a NEW `other-agent-here`
   sensory modality (declared prior). No communication yet. Predeclared: each still
   self-localizes and maps at its solo baseline — co-presence alone must not break solo
   competence. FAIL = adding the other-agent modality drops map accuracy or localization
   below the solo baseline (then the substrate is mis-specified; fix before climbing).

3. **Social transmission of value (Exp 14 grounding, creature→creature).** One clade-mate
   emits a cue/utterance that enters the other's sensory stream as the M4 *extrinsic*
   channel (`docs/specs/m4-affective-dyad.md` §3). Test whether creature B's values shift
   toward what A signals, against a fork-twin of B with the channel SEVERED. Predeclared
   property + threshold on value-share divergence. FAIL = channel-on and channel-severed
   twins are indistinguishable (then social transmission adds nothing at this scale — a real
   negative, logged as such).

4. **Coordination over a shared comfort source.** Two clade-mates, one comfort source both
   value. Do their policies develop measurable coordination or avoidance versus a baseline
   of two creatures who cannot sense each other? Predeclared coordination metric. FAIL =
   behavior is statistically indistinguishable from two solipsists who happen to share a
   world. Any deviation enters the `functional-emergence.md` rung-5 cascade (≥3 fork
   reproductions + deflationary sweep) before the word "coordination" is used.

5. **Dialect divergence and (mis)communication (frontier-adjacent — declare the ceiling).**
   Two clade-mates taught words SEPARATELY develop measurably different word↔concept maps
   (the toy claim). When the M4 channel couples them, do the maps converge, hold as stable
   dialects, or break down? Predeclared convergence/divergence metric. **CEILING (binding):**
   genuine emergent *compositional grammar* between them is the documented open problem
   (`open_problem.html`) — do NOT claim it. The tractable, honest claim is convergence vs.
   divergence of TAUGHT-label maps under coupling, never grammar from scratch.

**Discipline notes.**
- The trunk (mirro) and every promoted peer spine obey the continuity guard
  (`tests/test_creature_continuity.py`): never silently reset; an intentional restart must
  be an explicit logged `rebirth` event. Recovery from a bad epoch = `git checkout` the
  prior committed snapshot (each checkpoint is a restore point) — not an in-place wipe.
- A "different history" is a **branch**, not a trunk reset; a branch run long enough is a
  peer species. A from-scratch, no-inheritance **newborn** is a *separate root* (not a mirro
  descendant) — legitimate only when an experiment needs a zero-history baseline (cf. Exp 57).
- Every shared-world / channel mechanism is a PROVIDED prior — declare it in the entry the
  same way taught labels are declared. One new mechanism per iteration or attribution is lost.
- Functional language only ("functionally coordinates / functional social valence"). The
  inner-experience layer stays explicitly unverified both ways (VALIDATION.md).

**Stop condition.** Exhausted when: peer-spine infrastructure and the co-presence substrate
exist and are validated, the social-transmission and coordination rungs have verdicts either
way with every deviation cascaded, and the dialect rung reaches its convergence/divergence
verdict (or hits the declared grammar ceiling). Then write in EXPERIMENTS.md either the
surviving social novelty-candidate(s) with their cascade record, or "no disciplined social
novelty at this substrate; the next step would require <named substrate>" — and open that
substrate as a new direction or stop. Surviving candidates keep this direction open; that is
the program working, not a loop to escape.

**STATUS.** state: CLOSED-POSITIVE (ladder Exp 63–74; synthesis Exp 74) · latest: Exp 74 · depends-on: persistent-creature, functional-emergence · reusable: clade infrastructure (vela = first committed peer spine, Exp 63), shared-world/multi-agent substrates, social-transmission metrics, coordination probes · why: first test of RECIPE on multi-agent emergent behavior; one inertia law explained every social effect (value transmission Exp 66 BREAKTHROUGH young-receivers-only, dialect convergence Exp 73), and NONE of it was undesigned — coordination deflated to stigmergic unilateral lock-in (Exp 70–72) · CLOSURE: no undesigned social novelty at this substrate; named next substrates = distal other-agent sensing, graded-uncertainty (opened, Exp 85–91), **self-other modeling** (now opened as `loop/directions/self-other-modeling.md`) · next-falsifiable: NONE here — the ladder is closed; the descendant direction `self-other-modeling` carries the open question (model the OTHER's hidden state, the solipsism gap this closure named). Do NOT re-run Rung 1: vela already exists as the committed peer spine.
