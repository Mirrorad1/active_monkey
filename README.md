# active_monkey

A documented self-lab for studying how tiny artificial systems become more creature-like under pressure.

**Live tracker:** https://mirrorad1.github.io/active_monkey/  
**Lab notebook:** [EXPERIMENTS.md](EXPERIMENTS.md)  
**Related work and prior art:** [docs/RELATED_WORK.md](docs/RELATED_WORK.md)

## What this is

active_monkey is a small artificial-life research project. It uses tiny ecologies, fixed seeds, raw outputs, and public experiment notes to ask a narrow question:

> When does information become meaningful to the system itself?

The current research spine is:

```text
ecology pressure
→ sensing
→ organ-like persistence
→ memory
→ belief
→ communication
→ world models
```

The point is not to make the smartest agent first. The point is to build the conditions where sensing, memory, and belief have a reason to exist.

## Why an ecology?

A single agent can look interesting for the wrong reason. An ecology gives the experiment pressure: resource limits, depletion, competition, lineage, mutation, and failure.

Food is not just a feature. Food matters because no food means the creature stops continuing. Temperature is not just a column in a dataset. It matters only if it changes action, survival, reproduction, or persistence.

That is why the project keeps asking whether a capability is actually useful, not just present.

## Current framing

The near-term question is not simply "can a sensor appear?"

The cleaner question is:

> Under what conditions does a costly sensor become worth maintaining?

A one-off sensor is not enough. The stronger target is a sensor-to-organ bridge: a capability that persists because it improves action-relevant outcomes under pressure, without direct reward for the capability itself.

From there the next track is first-person state:

```text
third-person state:
  position, energy, resources, births, deaths, lineage

first-person state:
  local observations, internal energy, memory, beliefs, uncertainty, prediction error, action
```

The project does not claim to measure experience. It tries to make belief-like internal state measurable in a controlled world where the outside observer still knows the ground truth.

## What this is not

- Not a consciousness claim.
- Not AGI.
- Not a claim that emergence is inevitable.
- Not a benchmark project.
- Not a hidden LLM inside the creature.

The LLM is part of the research workflow. It helps propose experiments, write code, run checks, and document results. The subject being studied is the small ecology and the agents inside it.

## Standard of evidence

A result is interesting only if the obvious shortcuts are made hard to confuse with the claim.

The project tries to preserve:

- fixed seeds and reproducible scripts
- raw outputs under `experiments/outputs/`
- predeclared pass/fail bars when a claim is serious
- null results and honest walls
- explicit notes about what would kill a hypothesis
- no direct reward for the thing we are claiming emerged

The lab notebook is part of the artifact. Failed explanations stay visible because they narrow the map.

## Where deep active inference fits

Deep active inference is not the starting point. It becomes useful later if explicit belief variables stop scaling.

For now, the priority is to keep the substrate legible: ecology first, then sensing, then persistent capabilities, then explicit belief state. Learned latent models can be added once there is enough structure to know what problem they are solving.

## Reproduce

All results were produced inside the committed lockfile environment.

```bash
uv sync
uv run --python .venv pytest -q
uv run --python .venv python converse_demo.py
uv run --python .venv python experiments/exp145_m3c_live_probation.py
```

Every experiment script is committed together with its raw outputs under `experiments/outputs/`; seeds are fixed in-script; headline numbers can be recomputed from the committed rows where tooling exists.
