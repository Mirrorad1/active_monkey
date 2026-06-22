# C3 extensions — targeted pairs (#1) & higher-order residue (#2)

## Study 1 — targeted pair proposal (k=2 data, families B/C/D)

Can a cheap lexical proposer match full-enumeration accuracy at far fewer pair tests, and beat uniform sampling?

| method | budget | accuracy | avg pair tests | true-danger recall | fragile erasure |
|---|---|---|---|---|---|
| solo | 0.35 | 55.4% | 0.0 | 0.0% | 0.146 |
| c3_enumerate | 0.35 | 100.0% | 308.3 | 100.0% | 0.000 |
| c3_lexical | 0.35 | 85.8% | 25.8 | 68.1% | 0.042 |
| c3_uniform_25 | 0.35 | 66.5% | 77.1 | 27.2% | 0.107 |
| c3_uniform_10 | 0.35 | 59.1% | 30.8 | 10.6% | 0.132 |
| solo | 0.25 | 37.8% | 0.0 | 0.0% | 0.228 |
| c3_enumerate | 0.25 | 91.9% | 308.3 | 100.0% | 0.000 |
| c3_lexical | 0.25 | 75.7% | 25.8 | 68.1% | 0.079 |
| c3_uniform_25 | 0.25 | 50.8% | 77.1 | 27.2% | 0.174 |
| c3_uniform_10 | 0.25 | 42.8% | 30.8 | 10.6% | 0.206 |

## Study 2 — higher-order (k>=3) residue (B/D families, kcover=3)

Pairwise sigma is blind to k>=3 covers; an order-3 detector should recover them.

| method | budget | accuracy | avg group tests | fragile erasure |
|---|---|---|---|---|
| solo | 0.35 | 41.5% | 0.0 | 0.171 |
| c3_pairwise_enum | 0.35 | 41.5% | 287.9 | 0.171 |
| c3_order3_lexical | 0.35 | 61.7% | 32.3 | 0.112 |
| c3_order3_enum | 0.35 | 98.9% | 2560.7 | 0.000 |
| solo | 0.25 | 24.5% | 0.0 | 0.268 |
| c3_pairwise_enum | 0.25 | 24.5% | 287.9 | 0.268 |
| c3_order3_lexical | 0.25 | 51.1% | 32.3 | 0.162 |
| c3_order3_enum | 0.25 | 91.5% | 2560.7 | 0.000 |
