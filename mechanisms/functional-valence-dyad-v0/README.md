# functional-valence-dyad-v0

**Status: validated (toy scale).** A symbolic dyadic agent that learns which response earns
approval for each inferred intent-like state, from functional-valence feedback over a long
session, judged by a frozen constant-unfakeable scorer.

This is **functional valence only** — a scalar feedback signal and a posterior over a hidden
intent-like factor. It is **not** sentience, consciousness, subjective feeling, or
natural-language understanding. "Talking to it" means feeding it integer codes and `+/-`
feedback.

## Files
- `mechanism_card.json` — the `MechanismCard` (claim, works_when/fails_when, falsifiers).
- `adapters.json` — an `AdapterCard`: the dyad's belief-like state → an active-sensing probe.
  This is a composition **hypothesis**; Exp 211 showed margin-gated probing did not beat a
  fixed-rate control, so it is not a validated bridge.
- `scorer_refs.json` — a `ScorerCard` pinning `eval/affect_score.py` by sha256.
- runnable checkpoint: `artifacts/active-monkey-affect-dyad-v0` (see `docs/ARTIFACTS.md`).

## The honest core
- **Learned:** which response earns `+` per inferred intent (Dirichlet counts on the valence
  head). **Provided:** the generative-model structure, the scripted partner, the optimistic
  POS prior, the precision schedule, the 300-turn session.
- **Load-bearing:** the long (~300-turn) session — short sessions block learning itself
  (Exp 221). The precision schedule (annealed γ 1→8) fixes exploitation, not learning.
- **Falsifier:** a constant-reply control must FAIL (it scores `genuine_fraction = 0`,
  verdict False — verified Exp 222). The metric beats the 1/3 constant-response ceiling, not
  merely the 1/5 uniform floor.

Source experiments: 215–222, 225. Full chapter: `docs/research/m4a-affective-dyad-chapter.md`.
