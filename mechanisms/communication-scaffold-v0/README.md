# communication-scaffold-v0

**Status: scaffold — NOT validated.** Communication has not been demonstrated in
active_monkey. This card exists to hold the place honestly and to specify what a real test
would require.

## What exists
- `active_loop/benchmarks/comm_v0.py` — a v0 sender/receiver signaling **existence test**:
  costed signaling that beats shuffled/muted controls (~1.9 bits mutual information). This is
  toy machinery, not a selection-pressure or emergence result.

## What does NOT exist
- No ecology/selection experiment in which proto-communication emerges. `source_experiments`
  is honestly empty.
- Creatures are currently **solipsistic** — no representation of another agent. Interaction
  requires NEW provided substrate (a shared world and/or a channel), each piece declared as a
  prior.

## What a real test requires
A sender/receiver split, an explicit message cost, a receiver belief-like update conditioned
on the message, and **shuffled-message and muted-message controls** (if shuffling/muting the
message does not destroy the receiver's advantage, there is no information transfer).

Emergent compositional grammar (language from scratch) is the documented **open problem**
(`open_problem.html`); the tractable honest claim is convergence vs divergence of
**taught-label** maps under coupling — not language emergence. Do not represent this mechanism
as validated.
