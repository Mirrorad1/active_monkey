# Exp280 independent blinded verification

Reviewer: /root/retention_blind_verifier. Allowed inputs: predeclared runner docstring and
raw exp280.txt/exp280.json only. Printed verdict/result claims ignored; no independent rerun.

Independent verdict: **NEGATIVE** (retention passes; all other named tests fail).
Reconstructed every partition from six-code policy argmax: old [0,1,2,3,0,1], current
[0,1,2,0,1,2]. All exported scores/chosen responses match. All16 seeds9400..9415 retained.

| Conjunct | Recomputed | Required | Outcome |
|---|---:|---:|---|
| S0 old-all acquisition mean | .4375 | >=.5 | FAIL |
| unchanged joint-S0 mean | 0 | >=-.05 | PASS |
| unchanged no-loss count | 13/16 | >=12/16 | PASS |
| changed-NEW joint-S0 mean | 1/12 | >=1/3 | FAIL |
| changed-NEW positive count | 5/16 | >=12/16 | FAIL |
| current joint-idle mean | 1/48 | >=1/6 | FAIL |
| current joint-idle positive count | 4/16 | >=12/16 | FAIL |
| current joint-incremental mean | 1/48 | >=1/6 | FAIL |
| current joint-incremental positive count | 3/16 | >=12/16 | FAIL |

Arm means (old-all / unchanged / changed-NEW / current-all):
S0 .437500/.479167/.250000/.364583;
idle .354167/.458333/.312500/.385417;
joint .375000/.479167/.333333/.406250;
incremental .343750/.416667/.354167/.385417;
chronology .125000/.145833/.145833/.145833;
constant .333333/.333333/.333333/.333333.

Available raw checks: all200 unique ordered seed/time events in each row; both phases
visit all codes16 or17 times. Feedback matches the declared mapping (POS2/NEU1/NEG0).
Receipt IDs match the shared history;8200 posterior evaluations per final refit and4100
for common S0. Evidence arithmetic matches within5.69e-13. All S0/joint21-value objective
traces increase (minimum increments .00164646 and .000862741 respectively). Final refits
share the same start hash per seed; source before/after hashes match16/16. All policy and
action distributions are finite, nonnegative and normalized within1e-6; independently
reconstructed expected POS matches. All exported instrument/evaluation assertions true.

Audit limits: JSON supplies checkpoint references/hashes rather than contents. This blind
review cannot independently certify checkpoint roundtrips, complete donor belief/RNG/age,
tensor-level once-weighting, receipt authenticity, common heldout uniforms or implementation
fidelity. No contradicting evidence found. Controller separately loaded all176 AgentStates
and recomputed64 tensor masses (exp280_audit.md), complementing this raw-policy audit.

Scope: negative for this familiar-code screen with weak initial acquisition and one
provided learner/history/schedule. No later online-life benefit, compositional transfer,
ecological fitness, evolved sleep or improved lab research established.
