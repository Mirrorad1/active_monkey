# Exp 278 blinded verification

Independent verifier received only the predeclared module docstring and the raw
JSON in `experiments/outputs/exp278.txt`. It ignored emitted summaries, utilities,
pass flags, and verdicts; recomputed utilities from 5,760 raw post-update reward
points; and reconstructed the moment and input checks. Verification preceded
the atomic commit of script, output, and log. This is a raw-data audit, not an
independent rerun of the experiment.

**Verdict: NEGATIVE. Instruments pass.** Every comparison requires mean advantage
at least 0.01 and at least 6/8 positive paired seed differences.

| Regime | Baseline | repair_mv minus baseline | Positive seeds | Both gates |
|---|---|---:|---:|---|
| Sparse | keep | +0.011687262184 | 8/8 | PASS |
| Sparse | reset_all | -0.008141321058 | 0/8 | FAIL |
| Sparse | reset_m | +0.012363512713 | 8/8 | PASS |
| Sparse | reset_moments_keep_clock | -0.026395663373 | 0/8 | FAIL |
| Sparse | reset_clock | +0.020334003537 | 8/8 | PASS |
| Sparse | refresh_current | -0.000105173756 | 0/8 | FAIL |
| Dense | keep | +0.024045357793 | 8/8 | PASS |
| Dense | reset_all | -0.008567444838 | 0/8 | FAIL |
| Dense | reset_m | +0.014804789726 | 8/8 | PASS |
| Dense | reset_moments_keep_clock | -0.029115252773 | 0/8 | FAIL |
| Dense | reset_clock | +0.032742325263 | 8/8 | PASS |
| Dense | refresh_current | -0.000255396267 | 1/8 | FAIL |

The no-change repair_mv and keep trajectories agree exactly in all eight seeds.
The maximum replacement-versus-forward-replay discrepancy is
3.469446951953614e-17 for m and 3.2526065174565133e-19 for v. Intervention weights
are identical across arms. Inspected second moments are nonnegative, with minimum
0.0. Target changes match the prescribed indices and modulo-three replacement.
Changed sampled-reward counts are 577–619 for sparse, 1,619–1,772 for dense, and
zero for no-change. Raw rewards, advantages, and fixed-path gradients reproduce
exactly. Reset conventions match their declarations. Final-state expected rewards
match final trajectory scores.

The result improves over keeping stale state but fails against full resets and
current-weight refresh. Scope: eight seeds, one toy bandit, 30 recovery updates;
no counterfactual clean-training reconstruction or LLM-performance claim.

Final repository validation required the explicit labels `Hypothesis:` and `Falsifier:` in the module
docstring; the hypothesis and falsifying rule were already present. After that documentation-only
change, the controller reran the excluded smoke and main run without altering any
configuration. All per-seed scientific records were byte-identical after canonical
JSON serialization excluding per-job runtime. Main result SHA-256:
`96c37421861412421fe2b83db1a9b65eeae167ea4d91fc97c81899f0acaa3623`;
smoke: `a3e8f14faa269283b0986411091a2fe3bf57362eb3c12c6798aac146d9d208f5`.
The final raw files carry the corrected script hash. Main simulation time was
2.982 seconds on the first run, 0.520 seconds on an intermediate documentation-only rerun, and 3.010 seconds on the final rerun; neither
includes JSON serialization or process startup. The blinded verdict is unchanged.
