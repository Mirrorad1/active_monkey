# active_monkey

A tiny artificial-life lab for testing when costly traits become worth maintaining
under ecological pressure.

- **Live tracker:** https://mirrorad1.github.io/active_monkey/
- **Claims ledger:** [docs/CLAIMS.md](docs/CLAIMS.md)
- **Experiment map:** [docs/EXPERIMENT_INDEX.md](docs/EXPERIMENT_INDEX.md)
- **Lab notebook:** [EXPERIMENTS.md](EXPERIMENTS.md)

## What This Is

active_monkey is a one-person, LLM-assisted research lab. It uses small artificial
ecologies, fixed seeds, committed scripts, raw outputs, and falsifier-first notes to
study when a capability is locally adaptive rather than merely present.

The preferred public framing is narrow:

> When do costly capabilities - sensing, active information gathering,
> memory/belief-like internal state, and eventually communication - become worth
> maintaining because they improve action-relevant outcomes under pressure?

The project is deliberately toy-scale. Its value is auditability: a skeptical reader
should be able to trace a claim to the experiment entry, script, raw output, caveat,
and falsifier.

## The Core Question

The current research program asks when information becomes useful to the system itself.
In this repo, "useful" means it changes action, survival, reproduction, prediction,
or persistence in the toy world. A variable does not count as meaningful just because
it exists in the code or is visible to an outside observer.

Current spine:

```text
ecological pressure
-> costly sensing
-> local-gradient tests
-> passive sensor/memory walls
-> active sensing
-> communication and world-model substrate
```

## Why Ecology Instead Of A Single Agent

A single agent can look capable because the evaluator rewards a behavior directly or
because a shortcut is available. Ecology adds pressure: finite resources, depletion,
mutation, reproduction, competition, lineage, and death.

That pressure makes a stricter test possible. A costly sensor should not count as
meaningful because it is gifted to a creature. It should persist only when the local
evolutionary path pays for its upkeep in this setup.

## Current Research Spine

The early active-inference toys explored language, memory, valence, embodiment, and
queryable toy opinions. The current public spine is the ecology chapter: can costly
traits, including information-gathering actions, become locally adaptive under
pressure?

For the current state of the evidence, start with:

- [docs/CLAIMS.md](docs/CLAIMS.md) - claim-by-claim ledger with caveats and falsifiers.
- [docs/EXPERIMENT_INDEX.md](docs/EXPERIMENT_INDEX.md) - chapter map for the 209-entry notebook.
- [docs/research/sense-axis-organ-evolution.md](docs/research/sense-axis-organ-evolution.md) - current thermosense / sense-evolution synthesis.
- [docs/research/local-gradient-wall.md](docs/research/local-gradient-wall.md) - synthesis connecting costly senses to Phase 3 hidden-state memory.
- [sense-evolution.html](sense-evolution.html) - public write-up of the closed-negative sense-evolution chapter.

## Current Strongest Results

Current evidence suggests:

- Costly traits should not count as meaningful just because they exist; in this lab
  they must improve action-relevant outcomes under pressure.
- Disembodied symbol streams produced useful early lessons, but fully unsupervised
  latent structure collapsed in this setup; embodiment, grounding, registered
  continuous experience, and at least one structural anchor were needed for the
  stronger toy chain.
- Live probation was a better acceptance standard than replay-only scoring for
  structure growth in the continuous-creature chapter; under normalized-density
  evaluation, the growth wall fell in Exp 154.
- Several thermosense experiments found that gifted or global usefulness is not
  enough. A functional sensor can still be un-evolvable when the local selection
  gradient at the resident is non-positive.
- Phase 3 hidden-state memory extended that wall: passive cue integration via
  `memory_horizon` and continuous `belief_persistence` did not produce a robust
  local adaptive gradient, even though larger gifted jumps can be useful.
- Recent work motivates an Evolvability Preflight layer: measure local trait
  gradients and load-bearing premises before spending compute on full evolution
  batches.
- The next structural test is active sensing: whether a costly probe or sampling
  action can improve future action selection enough to beat the resident in a fair
  common garden.

These are toy-scale claims about this codebase and its current substrates, not claims
about biology, consciousness, AGI, or inevitable emergence.

## What This Is Not

- Not a consciousness claim.
- Not AGI.
- Not a claim that emergence is inevitable.
- Not a biological realism claim.
- Not a benchmark suite for general AI systems.
- Not a hidden LLM inside the agents.

LLMs are part of the research workflow: they help propose experiments, write code,
run checks, and document results. The studied system is the committed toy ecology
and its agents.

## Evidence Standard

Serious results should preserve:

- a predeclared question, pass/fail bar, or falsifier where practical;
- fixed seeds or clearly stated stochastic scope;
- committed experiment scripts and raw outputs;
- null results and instrument failures in the append-only log;
- caveats that say what the result does not show;
- no direct reward for the representation or trait being claimed.

The lab notebook is not polished away. Negative results and honest walls remain part
of the evidence chain.

## How To Reproduce

The default public commands are:

```bash
uv sync
uv run --python .venv pytest -q
uv run --python .venv active-monkey-converse-demo
uv run --python .venv python experiments/exp154_growth_confirmation.py
uv run --python .venv python tools/check_docs.py
```

Some experiment batches are slow or stochastic. See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for representative runs, raw-output
locations, seed notes, and limitations.

## Reader Map

- New reader: [docs/CLAIMS.md](docs/CLAIMS.md), then [docs/EXPERIMENT_INDEX.md](docs/EXPERIMENT_INDEX.md).
- Engineer: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), then [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
- ML / active-inference reader: [docs/RELATED_WORK.md](docs/RELATED_WORK.md), then [docs/research/problem2-phase-picture.md](docs/research/problem2-phase-picture.md).
- Artificial-life reader: [docs/research/sense-axis-organ-evolution.md](docs/research/sense-axis-organ-evolution.md), then [ecology/README.md](ecology/README.md).
- Nontechnical reader: [sense-evolution.html](sense-evolution.html), then [journey.html](journey.html).
- Term lookup: [docs/GLOSSARY.md](docs/GLOSSARY.md).
- Future direction: [docs/ROADMAP.md](docs/ROADMAP.md).
