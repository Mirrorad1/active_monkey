# Exp 279 independent blinded verification

Reviewer: /root/sleep_blind_verifier. Inputs: runner predeclared docstring and raw
exp279.txt/exp279.json, ignoring printed verdict claims. No independent scientific rerun.

Independent verdict: **POSITIVE**.

| Conjunct | Raw reconstruction | Required | Outcome |
|---|---:|---:|---|
| joint minus idle mean csel | 1/3 | >=1/6 | PASS |
| joint > idle seeds | 14/16 | >=12/16 | PASS |
| joint minus chronology mean csel | 5/12 | >=1/6 | PASS |
| joint > chronology seeds | 15/16 | >=12/16 | PASS |

Reconstructed all csel scores from raw policy argmax; exact agreement. Unique ordered
seeds 9300..9315, confirmatory flag, 100 unique IDs/times per seed; observed outcomes
match the harness mapping. Independently reconstructed evidence mass 380.83141154516375;
48 reported replay receipts agree within 1.14e-13 (<1e-6). All 320 joint objective
increments positive, minimum .0018845472357043. All 48 replay receipts count 2000 fitting
+2100 monitoring =4100 posterior evaluations. Chronology/joint event IDs match; chronology
objectives repeat after the first pass in 16/16 seeds. Live before/after hashes and
read-only evaluation hashes match throughout. Constant response independently gives 2/6.

Verifier scope: artifact-level gates, not direct inspection of fitted count tensors or
unit-test implementation. Controller separately loaded all 160 saved AgentStates and
checked raw checkpoint hashes; all matched. Current source hashes matched at execution.

Limits: warm-start global refit versus redundant restarted sequential control is an
algorithm-package comparison, not isolated replay order or strongest possible baseline.
Posterior evaluations do not equal FLOPs. Broken outcomes retain wake knowledge.
Held-out codes use the same known mapping, not transfer. Only 9/16 joint seeds reach
half-correct; no reliable rescue, retention, ecological or evolved-sleep conclusion.
