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

## Language rules for EXPERIMENTS.md entries
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

## Self-audit (when asked, or every ~10 experiments)
Reread the last N entries as a hostile reviewer: would each claim survive a replication
by someone who only has the entry text? Log discrepancies as a correction entry.
