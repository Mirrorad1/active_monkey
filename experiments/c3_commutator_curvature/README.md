# C3 — Commutator-Curvature Compression (first falsifiable test)

A standalone, LLM-free experiment testing one narrow, falsifiable hypothesis
about prompt compression. **No broad novelty claims.**

## 1. Hypothesis

> A compression policy that detects **second-order deletion residue** preserves
> task accuracy under tight token budgets better than a policy that scores edits
> **independently**.

Operationally: `c3_residue_guarded` should beat `solo_delta_greedy` (the strong
independent-scoring baseline) on prompts that contain *jointly* fragile spans.

## 2. Exact math

Let `loss(x)` be the task loss of a (possibly compressed) prompt `x`. Here it is
a **deterministic, binary** oracle (§3): `0.0` if the retained spans still let the
evaluator derive the gold answer, else `1.0`.

**First-order (solo) sensitivity** of deleting span *i*:

```
delta_i = loss(x without span i) - loss(x)
```

**Second-order residue** of the *pair* (i, j):

```
sigma_ij = loss(x without spans i and j) - loss(x without i) - loss(x without j) + loss(x)
         = loss(x \ {i,j}) - delta_i - delta_j        (since loss(x) = 0)
```

A **dangerous pair** is one where each deletion looks safe alone but is jointly
harmful:

```
delta_i ≈ 0,   delta_j ≈ 0,   sigma_ij ≥ residue_threshold
```

C3 builds a **danger graph** (edge *i—j* per dangerous pair) over the
candidate-safe spans (`delta ≤ tau`) and then deletes greedily by lowest delta /
highest token savings, **never deleting both endpoints of any danger edge**. If
the budget cannot be met otherwise, it deletes the lowest-sigma protected span as
a last resort and logs `forced_violation`.

## 3. Why pure deletion does *not* test literal non-commutativity

For pure **deletion** edits on **disjoint spans**, edit order commutes:
`delete(i) ∘ delete(j) = delete(j) ∘ delete(i)`. So there is no literal
commutator `[A,B] = AB - BA` to measure. The real object here is the **mixed
second partial difference** `sigma_ij` (a discrete cross-curvature / Hessian
off-diagonal term), not a commutator. The word "commutator" is reserved for
**optional non-deletion edits** (paraphrase, reorder, substitution), which are
out of scope for this first test. This experiment tests *residue*, and the name
"C3" is kept only as the project label.

## 4. Dataset families

Synthetic prompts of 12–40 numbered spans mixing relevant facts, distractors,
redundant fragile commitments, format constraints, negations, numbers/units and
cross-span dependencies. A required commitment is "satisfied" iff ≥1 of its
provider spans is retained; the gold answer needs **all** commitments satisfied.

| family | structure | what it probes |
|---|---|---|
| **A** Simple critical facts | each commitment has **one** provider | C3 must **not invent fake curvature**: deleting a critical span fails for everyone; there are no safe pairs to flag |
| **B** Redundant fragile facts | commitments stated in **two** weak places (2-cover) | the canonical residue case — either alone is safe, both lose the commitment (e.g. "report in kg" + "pound results rejected") |
| **C** Scattered relation facts | chained 2-covers **sharing a bridge span** (a node in two danger edges) | C3's graph reasoning (export-of-K inference: rule ∧ K-blue ∧ K-not-archived) |
| **D** Low-salience format | format/"do not" rules repeated weakly in **short** spans | independent salience deletes them; C3 must keep ≥1 per group |

Hidden ground truth (`hidden_required_commitments`, `hidden_fragile_groups`,
`hidden_dangerous_pairs`) is used **only** for final diagnostics — never by any
selector.

## 5. How to run

```bash
# from repo root
python experiments/c3_commutator_curvature/generate_dataset.py --n 1000 --seed 7
python experiments/c3_commutator_curvature/run_experiment.py \
    --dataset experiments/c3_commutator_curvature/data/synthetic_c3.jsonl --seed 7
# (this repo: prefix with `uv run --python .venv` )
```

Outputs land in `results/`: `summary.json`, `summary.md`, `failure_cases.jsonl`.

## 6. Results (n=1000, 200-instance test split, tau=0.01, residue_threshold=0.5)

Accuracy by method × budget ratio (all families):

| method | 0.75 | 0.50 | 0.35 | 0.25 | 0.15 |
|---|---|---|---|---|---|
| full_prompt | 100% | 100% | 100% | 100% | 100% |
| random_delete | 49.6% | 17.5% | 7.4% | 3.3% | 0.6% |
| length_greedy | 17.5% | 5.5% | 0.5% | 0.0% | 0.0% |
| solo_delta_greedy | 84.0% | 80.0% | 66.5% | 48.0% | 17.0% |
| **c3_residue_guarded** | **100%** | **100%** | **99.5%** | **88.0%** | **52.5%** |

Per-family accuracy at the tight 0.25 budget:

| method | A | B | C | D |
|---|---|---|---|---|
| solo_delta_greedy | 76.9% | 25.5% | 39.1% | 51.1% |
| c3_residue_guarded | 76.9% | 94.5% | 100% | 80.9% |

Success criteria:

- **Primary** — `acc(C3) − acc(solo)` on B/C/D: **+44.6 pts @0.35**, **+54.1 pts
  @0.25** (≥ +10 required) → **PASS**.
- **Secondary** — Family A: C3 = solo = 76.9% (Δ 0.0, ≤ 0.03 loss allowed) → **PASS**.

Dangerous-pair violation rate at 0.25: solo 0.46 → C3 **0.00**. C3 reaches the
budget (compression ratio ≈ target, `forced_violation_rate` = 0 down to 0.25;
0.09 only at the extreme 0.15) — it wins by *better selection*, not by refusing
to compress.

Ablations (accuracy, all families):

| variant | 0.35 | 0.25 | avg pair tests | note |
|---|---|---|---|---|
| C3 all pairs | 99.5% | 88.0% | 261 | full residue |
| C3 25% pairs | 74.7% | 57.6% | 65 | captures *part* of the win |
| C3 10% pairs | 69.2% | 51.7% | 26 | barely above solo |
| C3 danger disabled | 66.5% | 48.0% | 0 | **collapses exactly onto solo** ✓ |
| C3 random edges (same count) | 63.5% | 39.6% | 261 | **worse than solo** ✗ |

## 7. Interpretation

- The win is **real and mechanism-attributable**: disabling danger edges
  reproduces `solo_delta_greedy` *to the digit*, and the danger edges C3
  discovers match the hidden dangerous pairs exactly. Independent scoring
  provably cannot see these pairs (both endpoints have `delta = 0`).
- **Real residue matters, not just "a graph":** giving C3 the *same number* of
  **random** protective edges makes it **worse than solo** (it wastes budget
  protecting non-fragile spans, forcing real providers out). This is the key
  control against "any constraint graph would help."
- **Cost wall:** the win needs near-full pair enumeration. Uniform 25%/10%
  sampling recovers only a fraction, because true danger pairs are a *sparse*
  subset of all safe pairs. Practical C3 on a real LLM (where each `loss` call is
  a forward pass) would need **targeted** pair proposal, not uniform sampling.
- **Pairwise blind spot:** `sigma_ij` is second-order only. A commitment covered
  by **k ≥ 3** spans has `sigma_ij = 0` for every pair (deleting any two leaves a
  cover), so pairwise C3 cannot protect it — it needs k-th-order residue. The
  datasets here use 2-covers by design; this is an honest, known limitation.

## 8. Does this support, weakly support, or falsify the idea?

**Weakly supports.** The hypothesis is *not* falsified — second-order residue
detection clears the primary bar by a wide margin and does no harm on Family A.
But the support is **weak in scope**: the result is a clean *mechanism
proof-of-concept on synthetic data whose fragility is, by construction, exactly
2-cover redundancy*. The deterministic oracle reveals that structure noiselessly;
C3 is, candidly, exploiting synthetic redundancy. What the experiment genuinely
establishes is **necessity and sufficiency of the mechanism in principle** plus
two practical caveats (sparse-pair cost; random-graph harm). It does **not** yet
show the residue signal survives a noisy LLM loss or that uniform sampling is
viable.

## 9. Next experiment recommendation

1. **Real LLM-probe follow-up** (warranted): replace the binary oracle with a
   real `loss` (NLL of the gold continuation, or judged correctness) on a small
   model. Test whether `sigma_ij` is still detectable above noise — calibrate
   `tau`/`residue_threshold` on the dev split only.
2. **Targeted pair proposal** to beat the enumeration wall: propose candidate
   pairs by surface cues (same entity/unit/format keyword, co-reference) and
   measure recall of true danger pairs vs. uniform sampling.
3. **Higher-order residue** for k≥3 redundancy: greedy k-cover detection or a
   submodular-coverage guard, since pairwise sigma is structurally blind to it.

## 10. Follow-up results (#1 targeted pairs, #2 higher-order, #3 LLM-probe)

Run: `python generate_dataset.py --n 1000 --seed 7 --kcover 3 --out data/synthetic_c3_k3.jsonl`
then `python run_extensions.py`. Outputs `results/extensions_summary.{md,json}`.

**#1 — Targeted pair proposal (beats the enumeration wall).** A cheap lexical
proposer (spans sharing a rare stemmed content token) recovers most of C3's win
at ~12× fewer pair tests, and dominates uniform sampling (B/C/D test split):

| method | acc @0.35 | acc @0.25 | avg pair tests | true-danger recall |
|---|---|---|---|---|
| solo_delta_greedy | 55.4% | 37.8% | 0 | — |
| C3 full enumerate | 100% | 91.9% | 308 | 100% |
| **C3 lexical** | **85.8%** | **75.7%** | **26** | 68% |
| C3 uniform 25% | 66.5% | 50.8% | 77 | 27% |
| C3 uniform 10% | 59.1% | 42.8% | 31 | 11% |

Lexical at ~26 tests beats uniform at 77 tests — surface co-reference is a far
better pair prior than uniform sampling. Recall 68% (imperfect: some 2-cover
phrasings share no stemmed token), so it trails full enumeration but recovers
most of the gap cheaply.

**#2 — Higher-order (k≥3) residue (closes the pairwise blind spot).** On a
`--kcover 3` dataset (B/D), pairwise C3 is provably blind:

| method | acc @0.25 | group tests | note |
|---|---|---|---|
| solo_delta_greedy | 24.5% | 0 | |
| C3 pairwise enumerate | 24.5% | 288 | **identical to solo** — σ_ij=0 for every pair of a 3-cover |
| C3 order-3 lexical | 51.1% | 32 | recovers part, cheaply |
| C3 order-3 enumerate | 91.5% | 2561 | recovers fully, expensive |

Order-3 minimal-failing-group detection genuinely closes the blind spot
(`order-3 enumerate` ≈ ceiling); the lexical proposer makes it affordable
(32 vs 2561 tests) at the cost of recall — the cheap↔complete tradeoff, explicit.

**#3 — Real LLM-probe (RunPod, pull-and-run).** `llm_probe/` is a self-contained
torch/HF probe that replaces the binary oracle with the **NLL of the gold answer**
under a small causal LM and measures whether `sigma_ij` for the true dangerous
pair separates from random safe pairs (ROC-AUC) + whether NLL-thresholded C3
beats solo on greedy accuracy. Not run here (local venv is CPU/torch-less); see
`llm_probe/README_RUNPOD.md` for the clone→run→retrieve flow. AUC≥0.75 would mean
the residue signal survives a real model; AUC≈0.5 would falsify C3's premise on
that model.

## Files

- `common.py` — tokenizer, span↔prompt, the deterministic evaluator / loss oracle.
- `generate_dataset.py` — the four-family synthetic generator.
- `baselines.py` — full, random, length-greedy, solo_delta_greedy + `compute_deltas`.
- `c3_selector.py` — residue-guarded selector (+ sampling / no-danger / random-edge ablation knobs).
- `run_experiment.py` — runs all methods × budgets, metrics, criteria, failure cases.
- `extensions.py` — lexical pair/group proposer (#1) + higher-order minimal-failing-group selector (#2).
- `run_extensions.py` — runs the two follow-up studies → `results/extensions_summary.{md,json}`.
- `llm_probe/` — self-contained RunPod LLM-probe (#3): `llm_probe.py`, `setup_runpod.sh`, `README_RUNPOD.md`.
- `results/` — `summary.json`, `summary.md`, `failure_cases.jsonl`, `extensions_summary.{json,md}`.
- `data/synthetic_c3.jsonl` (k=2), `data/synthetic_c3_k3.jsonl` (k≥3 for study #2).
