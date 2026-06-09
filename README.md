# active_monkey

Teaching a machine to understand the world from surprise alone.

**Live tracker:** https://mirrorad1.github.io/active_monkey/
**The lab notebook:** [EXPERIMENTS.md](EXPERIMENTS.md)

## what this is

an agent that learns from scratch using one rule from neuroscience. minimize surprise. nothing pretrained. the embodied creature gets no dataset — word-labels are taught, but every value and concept self-forms from experience. it's an idea called active inference: a mind is just something that tries to predict its world and be less surprised by it. i'm building it from the bottom up to poke at one question. not what can think, but what it's like to be a thing that does.

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
