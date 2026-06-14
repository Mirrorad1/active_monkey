# Roadmap

This roadmap separates current evidence from long-term motivation. The ambition is to
study sensing, memory, belief-like internal state, and communication under pressure. The
current evidence is narrower and toy-scale.

## Now: Public Legibility And Evolvability Preflight

Current work should make the lab easier to audit without changing the science record.
That means clearer front-door docs, a claim ledger, an experiment index, reproducibility
notes, and consistency checks.

The current research method to stabilize is Evolvability Preflight: before spending compute
on full evolution batches, measure whether the local trait gradient or design premise is
positive enough to justify the batch.

## Next: General Trait Preflight For Costly Senses

The thermosense arc closed negative in this substrate. The next reusable infrastructure
step is a general trait preflight layer that can apply to other sense-like capabilities:
sight-like spatial sensing, hearing-like localization, communication reliability, or
memory machinery.

The target is not "make every trait evolve." The target is a falsifiable diagnostic:
which environments create a locally positive path for a costly trait, and which only make
the trait useful when gifted?

## Later: Memory, Belief-Like State, And Internal Model Persistence

Once trait preflight is general, the project can return to memory and belief-like state
inside ecology. The important distinction is first-person versus third-person state: the
creature's internal estimates should be measured separately from the observer's ground
truth.

Near-term infrastructure likely includes a unified AgentState surface that can fork,
replay, restore, and compare creature/ecology histories without losing provenance.

## Future: Communication And World-Model Substrate

Communication remains motivation, not current evidence. A communication result would need
to show that signals change another creature's beliefs or actions under pressure, without
directly rewarding the signal token itself.

World models are also future motivation. Learned latent models should be introduced only
when explicit belief variables stop scaling and there is a concrete, audited problem for
the learned model to solve.

## Standing Guardrails

- Keep long-term ambition separate from current evidence.
- Preserve null results and honest walls.
- Do not rename packages or imports casually.
- Do not weaken traceability to improve presentation.
- Any new public claim needs a falsifier, script, output, caveat, and reader path.
