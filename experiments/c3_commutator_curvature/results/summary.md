# C3 experiment results

- tokenizer: `regex-word-fallback`  |  test instances: 200  |  seeds (stochastic): [1, 2, 3, 4, 5]
- tau = 0.01, residue_threshold = 0.5

## Primary criterion: acc(C3) - acc(solo_delta_greedy) >= 0.10 on B/C/D
- **BCD@0.35**: C3=100.0%  solo=55.4%  delta=+44.6pts  -> PASS
- **BCD@0.25**: C3=91.9%  solo=37.8%  delta=+54.1pts  -> PASS

**Primary: PASS**

## Secondary: C3 must not lose >0.03 vs solo on Family A
- **A@0.25**: C3=76.9%  solo=76.9%  delta=+0.0pts -> PASS

## Accuracy by method x budget (all families)
| method | r=0.75 | r=0.5 | r=0.35 | r=0.25 | r=0.15 |
|---|---|---|---|---|---|
| full_prompt | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| random_delete | 49.6% | 17.5% | 7.4% | 3.3% | 0.6% |
| length_greedy | 17.5% | 5.5% | 0.5% | 0.0% | 0.0% |
| solo_delta_greedy | 84.0% | 80.0% | 66.5% | 48.0% | 17.0% |
| c3_residue_guarded | 100.0% | 100.0% | 99.5% | 88.0% | 52.5% |

## Compression ratio (avg retained / full tokens)
| method | r=0.75 | r=0.5 | r=0.35 | r=0.25 | r=0.15 |
|---|---|---|---|---|---|
| full_prompt | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| random_delete | 0.730 | 0.482 | 0.332 | 0.232 | 0.132 |
| length_greedy | 0.732 | 0.481 | 0.331 | 0.230 | 0.131 |
| solo_delta_greedy | 0.730 | 0.482 | 0.332 | 0.233 | 0.133 |
| c3_residue_guarded | 0.730 | 0.482 | 0.333 | 0.231 | 0.133 |

## Dangerous-pair violation rate
| method | r=0.75 | r=0.5 | r=0.35 | r=0.25 | r=0.15 |
|---|---|---|---|---|---|
| full_prompt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| random_delete | 0.173 | 0.466 | 0.621 | 0.695 | 0.731 |
| length_greedy | 0.515 | 0.675 | 0.730 | 0.740 | 0.740 |
| solo_delta_greedy | 0.160 | 0.200 | 0.330 | 0.460 | 0.655 |
| c3_residue_guarded | 0.000 | 0.000 | 0.000 | 0.000 | 0.090 |

## Fragile-group erasure rate
| method | r=0.75 | r=0.5 | r=0.35 | r=0.25 | r=0.15 |
|---|---|---|---|---|---|
| full_prompt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| random_delete | 0.071 | 0.259 | 0.417 | 0.566 | 0.730 |
| length_greedy | 0.453 | 0.641 | 0.749 | 0.812 | 0.900 |
| solo_delta_greedy | 0.061 | 0.083 | 0.146 | 0.228 | 0.489 |
| c3_residue_guarded | 0.000 | 0.000 | 0.000 | 0.000 | 0.035 |

## Ablations -- accuracy by budget (all families)
| method | r=0.75 | r=0.5 | r=0.35 | r=0.25 | r=0.15 |
|---|---|---|---|---|---|
| c3_pairs_100 | 100.0% | 100.0% | 99.5% | 88.0% | 52.5% |
| c3_pairs_25 | 88.4% | 85.5% | 74.7% | 57.6% | 26.9% |
| c3_pairs_10 | 85.5% | 81.8% | 69.2% | 51.7% | 20.9% |
| c3_no_danger | 84.0% | 80.0% | 66.5% | 48.0% | 17.0% |
| c3_random_edges | 84.4% | 79.4% | 63.5% | 39.6% | 13.4% |

## Ablations -- avg pair tests used
| method | r=0.75 | r=0.5 | r=0.35 | r=0.25 | r=0.15 |
|---|---|---|---|---|---|
| c3_pairs_100 | 260.560 | 260.560 | 260.560 | 260.560 | 260.560 |
| c3_pairs_25 | 65.140 | 65.140 | 65.140 | 65.140 | 65.140 |
| c3_pairs_10 | 26.055 | 26.055 | 26.055 | 26.055 | 26.055 |
| c3_no_danger | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| c3_random_edges | 260.560 | 260.560 | 260.560 | 260.560 | 260.560 |

## Per-family accuracy at r=0.25 (main methods)
| method | A | B | C | D |
|---|---|---|---|---|
| full_prompt | 100.0% | 100.0% | 100.0% | 100.0% |
| random_delete | 1.2% | 0.7% | 11.3% | 0.9% |
| length_greedy | 0.0% | 0.0% | 0.0% | 0.0% |
| solo_delta_greedy | 76.9% | 25.5% | 39.1% | 51.1% |
| c3_residue_guarded | 76.9% | 94.5% | 100.0% | 80.9% |

## Failure cases: 10 C3-wins, 10 C3-losses written to results/failure_cases.jsonl
