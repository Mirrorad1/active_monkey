# active_monkey

Teaching a machine to understand the world from surprise alone.

**Live tracker:** https://mirrorad1.github.io/active_monkey/
**The lab notebook:** [EXPERIMENTS.md](EXPERIMENTS.md)
**Related work and prior art:** [docs/RELATED_WORK.md](docs/RELATED_WORK.md)

## What this is

A toy active-inference lab: persistent agents at deliberately small scale
(4×4 gridworlds, ≤16 observation classes, lives of a few thousand steps) that learn,
act, form preferences, forget, and revise their world models through lived
experience. Built on the active-inference framework (Friston et al.) and pymdp
patterns — the framework is not ours; the testbed, benchmark discipline, and
negative-results catalog are.

## What this is not

- Not a consciousness claim. "Wants," "feels," "opinion" are functional labels for
  measured quantities (valence = −free energy), documented as such.
- Not AGI or a scaling claim. Nothing here is claimed to transfer beyond toy scale.
- Not "model expansion works." As of Exp 153, five growth designs failed honestly;
  the current best hypothesis (the emission-convention finding) is one predeclared
  re-test from confirmation. See EXPERIMENTS.md Exp 143–153.
- Not self-graded only: every experiment verdict since Exp 152 is checked by a
  blinded verifier (loop/PROTOCOL.md step 4.5), and earlier chapters were
  retro-audited (Exp 140, Exp 102).

## wait, is this just an LLM?

no. there are two layers here, and the LLM is not the part that's learning.

- **the scaffolding (an LLM).** an autonomous agent does the science. it poses a hypothesis, writes the experiment, runs it, reads the result, and logs what happened. then it goes again, on its own, without stopping. it is the researcher, not the mind being studied.
- **the subject (a mind built from math).** what it studies is a tiny active inference agent. there is no language model inside it, and nothing pretrained. the embodied creature gets no dataset — the early language experiments streamed a small text corpus as sensory input, but the embodied creature that followed does not. just a small generative model dropped into a world, updating itself to be less surprised by what it senses. everything it comes to know, it built from its own experience.

so when you watch it learn, that isn't the language model talking. it's a little mind made of math, figuring out its world from nothing. the LLM just makes it possible to keep running these experiments without me babysitting every one.

## how it works

each experiment, start to finish:

> a hypothesis → build the agent → drop it in a tiny world → measure its surprise → keep what lowers it → log it honestly → again

every result lands in [EXPERIMENTS.md](EXPERIMENTS.md), wins and dead-ends both. the site auto-deploys as new experiments land, so the tracker stays live.

so far it's climbed from "can it even learn letters" to a creature that knows where it is in a tiny world, learns facts about that world, wants things, and forms preferences of its own. not conscious, but maybe the first rungs of what an inside is built from.

## honest framing

this is a toy, and i'm honest about that. small worlds, small models, no claim that any of it is conscious. what it does show, in miniature, is structure emerging from experience with no labels and nothing pretrained. it's my braindump while i learn the math behind the theory instead of just the ideas. corrections welcome.

built on [pymdp](https://github.com/infer-actively/pymdp) for the active inference machinery.

## Reproduce

All results were produced inside the committed lockfile environment.
```bash
uv sync                                  # creates .venv from uv.lock
uv run --python .venv pytest -q          # fast suite (~2s)
uv run --python .venv python converse_demo.py        # capstone demo
uv run --python .venv python experiments/exp145_m3c_live_probation.py  # any experiment re-runs from its script
```
Every experiment script is committed together with its raw outputs under
`experiments/outputs/`; seeds are fixed in-script; headline numbers can be
recomputed from the committed rows (see `tools/`).
