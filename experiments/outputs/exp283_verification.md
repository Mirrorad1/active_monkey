# Exp283 detached numerical verification

## Restricted inputs and scope

This verification used only `experiments/outputs/exp283_verifier_packet.json`, its frozen predeclaration, and the 48 compressed JSON ledgers listed in that packet. No source code, manuscript, experiment log, RESUME, analysis, or parent interpretation was read. Printed summary, verdict, and accounting claims were not used as evidence. Calculations were recomputed from the per-step trajectory rows; arm and seed identifiers were obtained from the listed ledger filenames.

There are 48 distinct ledger paths and 144,000 per-step rows. Each of the NEUTRAL, COST_ONLY, and THERMAL arms contains exactly the 16 seeds 9500–9515. No seeds were excluded or duplicated.

## Frozen decision and conjunct mapping

**Verdict: NEGATIVE under the frozen numerical decision rule.** The cost-only joint criterion passes for **1/16 seeds**, below the required **12/16**. Neutral passes **16/16**. The frozen rule says that failure of any primary scientific conjunct yields NEGATIVE; the descriptive thermal results do not change that decision.

| Frozen conjunct | Recomputed evidence | Result |
|---|---|---|
| Cost-only: at least 12/16 seeds jointly complete 3000 steps, finish with living trait mean at least 0.02, and have cumulative realized selection at most −0.02 | Completion 16/16; selection threshold 16/16; final-mean threshold 1/16; joint 1/16 | FAIL |
| Neutral: final living trait mean at least 0.02 in at least 12/16 seeds | 16/16; final means range from 0.03746801869831033 to 0.17242695212085246 | PASS |
| Accounting mismatch does not exceed 1e−10 | Largest checked absolute discrepancy is 1.1102230246251565e−16 | PASS |
| Malformed lineage invalidates a run | Aggregate rows do not independently expose pedigrees; see limitations below | Not independently certifiable from these inputs |
| Thermal directional comparison | Descriptive only | No verdict gate |

All 48 trajectories have exactly 3000 rows with consecutive times 1–3000, a zero initial living trait mean, positive populations throughout, and no recorded population above 5000. There are no recorded extinctions or explosions. The smallest recorded population is 2; the largest is 599.

| Arm | Completed | Final mean ≥0.02 | Cumulative selection ≤−0.02 | Joint count | Final population range |
|---|---:|---:|---:|---:|---:|
| NEUTRAL | 16/16 | 16/16 | 14/16 | 14/16 | 484–567 |
| COST_ONLY | 16/16 | 1/16 | 16/16 | 1/16 | 309–514 |
| THERMAL | 16/16 | 16/16 | 16/16 | 16/16 | 74–89 |

Only the cost-only joint count and neutral endpoint count are scientific decision gates. Neutral selection and joint counts, and all thermal comparisons, are descriptive.

## Exact cost-only endpoint and selection evidence

Cumulative selection was recomputed as the sum of `mean_ancestor − mean_before` over all 3000 rows. Every cost-only seed completed; only seed 9512 reached the frozen endpoint threshold.

| Seed | Final living mean | Cumulative selection | Joint pass |
|---|---:|---:|:---:|
| 9500 | 0.014690690714177442 | −0.4375179137291388 | No |
| 9501 | 0.016678620872691112 | −0.29006312235725656 | No |
| 9502 | 0.017421828000923335 | −0.3243992202272177 | No |
| 9503 | 0.013026695038626548 | −0.26736720401059977 | No |
| 9504 | 0.013241377854893356 | −0.33322406341944355 | No |
| 9505 | 0.01802046834928398 | −0.31135204957843365 | No |
| 9506 | 0.016837919428045387 | −0.4033447298787771 | No |
| 9507 | 0.014604438181733449 | −0.32638718018309626 | No |
| 9508 | 0.012525790738868307 | −0.3764609433435423 | No |
| 9509 | 0.015373469050638057 | −0.31928890851429154 | No |
| 9510 | 0.014242389554026366 | −0.4183906359370215 | No |
| 9511 | 0.019037105658418492 | −0.26975789603616124 | No |
| 9512 | 0.02189454178854684 | −0.2504511547380311 | Yes |
| 9513 | 0.015930432346953835 | −0.30606229382857586 | No |
| 9514 | 0.01689075135227521 | −0.3002544676897447 | No |
| 9515 | 0.014137578111869246 | −0.37459836793343215 | No |

The cost-only mean final living trait across the 16 seeds is 0.015909631065123185. This average is descriptive; the decision uses the frozen per-seed joint count.

## Ledger arithmetic and continuity checks

For each row, the following quantities were independently recomputed from the three recorded means:

- Selection S = `mean_ancestor − mean_before`.
- Transmission T = `mean_after − mean_ancestor`.
- Delta = `mean_after − mean_before`.
- Identity residual = delta − S − T.

Cumulative quantities were recomputed with `math.fsum`; endpoint comparisons used the last `mean_after` minus the first `mean_before`. All row values were finite. Population, birth, and death counts were nonnegative integers. No checked discrepancy exceeded 1e−10.

| Check | Maximum absolute discrepancy |
|---|---:|
| Reported selection minus recomputed S | 0.0 |
| Reported transmission minus recomputed T | 5.0306980803327406e−17 |
| Reported delta minus recomputed delta | 0.0 |
| Recomputed per-step identity residual | 1.734723475976807e−18 |
| Reported residual minus recomputed residual | 5.0306980803327406e−17 |
| Population balance: `n_after − n_before − births + deaths` | 0.0 |
| Population continuity between adjacent rows | 0.0 |
| Mean continuity between adjacent rows | 0.0 |
| Cumulative population balance | 0.0 |
| Endpoint change minus cumulative delta | 1.734723475976807e−18 |
| Endpoint change minus cumulative S plus cumulative T | 5.551115123125783e−17 |
| Cumulative delta minus cumulative S minus cumulative T | 1.1102230246251565e−16 |
| Sum of recomputed per-step identity residuals | 1.734723475976807e−18 |
| Reported transmission-minus-expected diagnostic minus recomputed T minus recorded expected transmission | 5.0306980803327406e−17 |
| Sum of per-step T-minus-expected diagnostics minus difference of cumulative T and cumulative recorded expected transmission | 7.28583859910259e−17 |

The transmission-minus-expected diagnostic was checked only for arithmetic consistency with the recorded expected-transmission values. It was not a verdict gate, and the expected-transmission benchmark itself was not independently regenerated from parent traits.

## Descriptive thermal result

All 16 thermal seeds complete and have final living means from 0.036802791130175115 to 0.04878699004602491, with cumulative selection from −0.5041979315402864 to −0.1882139366059834. These observations are descriptive under the frozen predeclaration and do not rescue the failed cost-only conjunct.

## Explicit limits

These aggregate ledgers do not independently reconstruct pedigrees or simulate worlds. Their internal numerical consistency cannot certify actual parent–descendant assignments, lineage correctness, engine correctness, arm configuration, mutation parameters, or observer nonperturbation. They also cannot independently reconstruct the expected clipped drift conditioned on actual newborn parents. Those claims require evidence outside this detached numerical input scope.

No numerical invalidity was detected in the available ledger checks. The numerical scientific verdict is NEGATIVE because the frozen cost-only endpoint threshold fails for 15 of the 16 seeds; this is separate from claims that the aggregate ledgers cannot independently certify.
