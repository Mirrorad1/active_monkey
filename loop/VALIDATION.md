# VALIDATION — brutal-honesty rules (binding)

The product of this project is TRUE statements about what works and what doesn't.
A flattering false log is worthless. These rules bind every experiment entry.

## Before interpreting any result
- [ ] Compare the raw output against the PREDECLARED prediction and falsifier — not
      against what would make a good story.
- [ ] If the result matches the falsifier, the verdict is NEGATIVE. Write it that way.
      A negative result is a finding, often the most valuable kind.
- [ ] If you reran with different seeds/hyperparameters until it worked, the entry must
      say so and report the failures. Seed-shopping presented as a clean win is fraud.
- [ ] Check the trivial explanation first: would a baseline (random policy, frozen
      beliefs, shuffled labels) score the same? If you didn't run the obvious baseline,
      say "no baseline" in the entry.

## Required honesty distinctions (this project's known traps)
- **Provided vs. self-formed.** If any component was given to the creature (an anchor,
  an extraction algorithm, a template, a map), the entry must name it. "The creature
  did X" when the harness did X is the cardinal sin here.
- **Consolidation vs. new insight.** If the result was predictable from Exp 1–40, tag
  it CONSOLIDATION. Don't inflate confirmations into discoveries.
- **Ceiling honesty.** Results near the documented walls (emergent grammar, tabula-rasa
  structure) must not be framed as cracking them. Partial ≠ solved.
- **Toy honesty.** State the scale. Don't extrapolate toy success to claims about
  scaled behavior without running the scale test.
- **BREAKTHROUGH is a claim that must survive the hostile-reviewer test:** what could
  the system demonstrably do for the first time here? If the answer is "the same as
  before, but bigger / cleaner / again", it is a POSITIVE-SINGLE. Apply this test
  before writing BREAKTHROUGH in any entry or curated record.

## Validity gates (added after Exp 69's self-invalidation)
- A validity gate must test the instrument's INPUT (did the stimulus/condition occur:
  events happened, the channel fired, scarcity bound), never the mechanism's OUTPUT or any
  agent's response to it. An output-conditioned gate smuggles a particular result into
  "validity" and can invalidate exactly the runs where the interesting regime appears
  (Exp 69: requiring the monopolist's own gate to trip). Corollary: avoid raw
  threshold-state checks for engagement (Exp 69's 0.0009 near-miss) — count events.
- Verdict tags are mandatory: POSITIVE / NEGATIVE / MIXED + CONSOLIDATION / NEW INSIGHT.
- Ban unfalsifiable glow-words ("remarkably", "impressively", "emergent" without a
  measured criterion). Numbers over adjectives.
- The "Honest caveat" line is mandatory, even for clean positives. If you truly can't
  find a caveat, write what you tried.
- Never edit a past entry to make it age better. Corrections are NEW entries that cite
  the old one.

## Reproducibility
- An experiment whose script and raw output are not committed in the repo fails the
  self-audit by definition. A log-only claim is an unverified claim.
- The script must live at `experiments/expNN_<slug>.py` and the raw output at
  `experiments/outputs/expNN.txt`; both must be committed in the same commit as the
  EXPERIMENTS.md entry. Anything written to `/tmp` and not moved into the repo before
  the session ends is gone.

## Statistical reproducibility & the persistent creature (binding)

- **Predeclared predictions/falsifiers for stochastic experiments must be PROPERTY-LEVEL
  with explicit thresholds** — e.g. "map accuracy >= 8/9", "favorite stays correct in
  >= 2 of 3 seeds". Post-hoc exact numbers are not admissible as the predeclared standard.
  Exact-number match remains the standard ONLY for deterministic seeded scripts.

- **When stochasticity matters, run >= 3 seeds and report ALL** (fractions passing, not
  just the successful count). A single-seed result must say "single seed" explicitly in
  the EXPERIMENTS.md entry; it cannot be promoted to a general finding without the
  multi-seed sweep.

- **Persistent-creature experiments:** commit the creature's state snapshot BEFORE and
  AFTER the episode. The before-hash is the resumable starting point — anyone can re-run
  the experiment from the committed state. Record `name`, `age_steps`, and `state_hash`
  (before/after) in the EXPERIMENTS.md entry. Exact re-run does not apply to a lived
  creature; **resume-from-snapshot** replaces it as the reproducibility unit.

- **Lifelong-individual claims** ("its history made it X") require a counterfactual
  control: a `fork()` twin run on the alternative history, or a cohort. An uncontrolled
  biography anecdote is an anecdote — label it explicitly as such. Do not present it as
  a finding without the control.

- **The biography (BIOGRAPHY.jsonl) is append-only**, same rule as EXPERIMENTS.md.
  Never edit past biography entries; corrections are new entries that cite the old ones.

## Human consults & escalation (ratified 2026-06-10)

When a thread reaches a genuine decision point, the loop may post a **CONSULT** to
`loop/IDEAS.md` stating: the evidence, the options, a clearly-recommended option, and the
predeclared falsifier that still binds. The human has **ratified silence-as-consent** for
these, under strict guardrails:

- The loop may proceed on its **stated recommended option** if the human resumes the loop
  without redirection. Silence counts as approval ONLY for that recommended option — never
  for a different branch, and never for anything not framed as a bounded CONSULT.
- The verdict must stay **falsifier-bound**: if the predeclared test then fails, the thread
  **HALTS for explicit human input** — it may not reinterpret the failure and continue.
- The disposition is recorded under the CONSULT bullet in `IDEAS.md` ("→ taken up as Exp NN
  (option X): treated as standing authorization …"). Never silently.
- Precedent: Exp 58 consult → Exp 60 (option a), which then killed the candidate as lawful
  on its predeclared falsifier. That is the sanctioned pattern.

## Self-audit (when asked, or every ~10 experiments)
Reread the last N entries as a hostile reviewer: would each claim survive a replication
by someone who only has the entry text? Log discrepancies as a correction entry.
