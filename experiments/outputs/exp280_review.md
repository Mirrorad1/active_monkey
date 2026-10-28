# Exp280 pre-run review and lab knowledge use

Independent design reviewer /root/retention_design_check found the plan buildable with no
material blocker. Clarifications implemented before freeze: stage1 gamma(1,8,100), alpha1;
stage2 explicitly disables the schedule and sets gamma4; separate phase2 code stream
seed+400000; incremental inference uses current full model before count replacement.

Fresh instrument reviewer /root/retention_instrument_review inspected the plan, runner,
incremental helper and tests without Exp280 outputs: no material blockers. Focused review
checks52 passed. The strengthened installation test uses changed A/pA, preserving live
belief/RNG/age and B/C/D/pB. Minor limitation: stage2 gamma was code-inspected rather than
verified by a direct behavioral-specific regression. This is not scientific verification.

Previous reviewed lab brief used: meta/sleep/exp279-replay-v1, receipt hash
9257338c3bbc976cafd8227649836324afe9763aca4b516c42f371791609bf83. Its generated rehearsal
prompted separate retention/revision scores and the stronger sequential refresh baseline.
This usage record is not a causal demonstration of better research quality (L1 unrun).

One source-matched excluded pilot9399; all six instruments passed. Runtime forecast is
based only on pilot resource measurements, never on favorable behavioral outcomes. No
seed, learning-rate, horizon or threshold changes followed pilot outcomes. Source hashes
are retained in exp280_smoke.json and the confirmatory JSON.
