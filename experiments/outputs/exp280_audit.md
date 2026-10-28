# Exp280 controller artifact audit

Raw JSON SHA256: 9ab1d33809052d9aee6afb9e3689738483463add6bdc9a5bddbb73758dc458f4.
All six source hashes match the source-frozen pilot and current executed files. No rerun,
source patch, seed exclusion or threshold change followed pilot/confirmatory outcomes.

Loaded every one of176 saved AgentStates and matched raw checkpoint_refs content hashes.
Independently recomputed count mass from saved anchor/fit p0/p1: maximum absolute error
5.684341886080801e-13 (<1e-6), all64 fitted states (S0 plus three replay arms,16 seeds).
Minimum joint regularized objective increment .0008627407476069493 across S0/final traces.
Seeded cohort: initial lived_turns0; acquired100; source_before/after200. No mirro/nira
spine changed. All donor before/after hashes agree; exact per-seed hashes in raw JSON.

Full runtime 525.8846783330664 seconds; peak RSS 9389.09696MB;
physical RAM 38.654705664GB. One excluded pilot9399 projected conservatively
746.9048713445663 seconds. Actual within1800sec ceiling.

Each full-history replay arm:4000 fitting +4200 monitoring =8200 posterior evaluations.
Shared S0 overhead:2000 fitting +2100 monitoring =4100. This is matched record-posterior
work, not identical FLOPs/time. All event streams/raw policy distributions retained.

Scope: artifact checks, not independent simulation. Independent blinded grading is stored
separately in exp280_verification.md. Pre-run design/instrument reviews and prior lab brief
use are recorded in exp280_review.md. No causal L1 research-quality evaluation was run.
