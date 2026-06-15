# Glossary

Short definitions for project-specific terms. These are operational definitions for this
repo, not claims about biology or subjective experience.

**Ecology pressure:** Resource limits, depletion, competition, reproduction, mutation, and
death that make some traits costly or useful.

**Costly trait:** A capability with upkeep or opportunity cost. It must improve outcomes
enough to pay for itself.

**Sensor:** A trait that makes some hidden world variable observable, usually with noise.

**Organ-like persistence:** A costly capability that remains in the population because it
keeps improving action-relevant outcomes under pressure.

**Local evolutionary gradient:** Whether a small heritable step near the current trait value
increases realized fitness after costs.

**Gifted benefit:** The benefit seen when a strong trait is installed by hand. It can be real
without being evolvable.

**Global optimum:** A high-performing endpoint when tested directly or as a monomorphic
population.

**Local path:** The sequence of small heritable steps evolution can actually take from the
resident value.

**Live probation:** A candidate structure or trait change is installed provisionally and
kept only if live sequential experience improves under the gate.

**Prequential evaluation:** Scoring a model by its next-step predictive performance over a
stream, rather than by a separate static test set.

**Belief-like state:** An internal estimate used for prediction or action in the toy model.
It does not imply consciousness.

**Active sensing:** The agent takes an action to improve its observations, not just to get
food directly. Programmer translation: instead of only reacting to logs, the service pays
to add better observability before deciding what to do.

**First-person state:** Agent-side variables available to or carried by the creature, such
as local observations, energy, memory, uncertainty, prediction error, and action history.

**Third-person state:** Observer-side ground truth, such as absolute position, world layout,
resource state, births, deaths, and lineage.

**World model:** The creature's internal model of how observations and transitions work.

**Active inference:** Action and perception are coupled. The agent acts partly to reduce
uncertainty about hidden state, not only to maximize immediate reward. Programmer
translation: like a runtime system choosing diagnostics that reduce ambiguity before
taking a risky production action. This repo has not achieved full active inference in the
ecology; active sensing is the current bridge test.

**Falsifier:** A predeclared observation that would weaken or kill a hypothesis.

**Honest wall:** A negative result that survives checks and becomes useful because it marks
a real boundary in this setup.

**Evolvability Preflight:** A cheap diagnostic layer that measures local gradients or
load-bearing premises before running expensive evolution batches.
