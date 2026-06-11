# direction: red-team

**Question.** Which of our own positive findings (Exp 1–40) fails to survive a hostile
replication or a stronger baseline?

**Why it matters.** The log's value is its honesty. A finding that only holds for one
seed, one scale, or against no baseline is not a finding. Killing one of our own claims
is NEW INSIGHT by definition.

**Experiment ladder.**
1. Baseline audit: pick the 3 strongest claims (e.g. Exp 20/26/34) and run the trivial
   baseline each lacked (random policy / shuffled history / shuffled labels). FAIL (for
   us) = baseline matches the claimed result — log a correction entry.
2. Seed sweep: rerun a headline experiment over 10 seeds; report the full distribution,
   not the best. FAIL = headline result is a tail outcome.
3. Ablation honesty: remove ONE recipe ingredient at a time from a working experiment
   and confirm each is actually necessary (Exp 31 did this for the anchor; do it for
   continuous-registration and grounding). FAIL = an ingredient was decorative.
4. Claim-vs-text audit: reread the last 10 EXPERIMENTS.md entries as a referee; any
   claim not supported by the recorded numbers gets a correction entry.

**Stop condition.** All four passes done. Every survived claim gets noted as
"red-teamed, survived"; every kill gets a correction entry citing the original.

**STATUS.** state: active · latest: TBD-human · depends-on: TBD-human · reusable: hostile-replication patterns, baseline audits, seed-sweep protocols · why: standing directive; killing our own claims is new insight; log's credibility depends on honesty · next-falsifiable: pick 3 strongest claims (Exp 20/26/34 candidates) and run the trivial baseline each lacked (random policy, shuffled history, shuffled labels)
