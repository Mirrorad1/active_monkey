# sparse-llm — selection as a measurable instrument

A research line on **sparse attention**, run with the program's discipline (cheapest posable
test first, predeclared falsifiers, non-degenerate controls, scale only on proven insufficiency).

## The thesis
Sparse attention's hard part isn't the sparsity — it's the **selection**: *which* keys to attend
to, when computing the full q·k scores is the n² cost you're avoiding (you need attention to
decide where to attend). Most work reports averaged perplexity, which **launders** the silent loss
of a specific long-range dependency. So we make **selection a first-class, swappable, recall-scored
object** and measure it directly.

A `Selector` (see `selection.py`) is just `(q, K, budget, qpos, rng) -> indices`:
- `exact_topk` — the **ceiling** (true top-budget by q·k); every sparse selector is graded against it.
- `window` — content-blind positional baseline (the "fixed gait"); the **non-degenerate control**.
- `random` — floor.
- `block_topk` — the **coarse-to-fine meta-selector** ("an abstraction over the selection"): pool keys
  into blocks, score blocks by `q·mean(block)`, keep the top blocks, then exact-topk within them.

## What "improve a model" means here (be precise)
Changing selection on a **frozen** pretrained model changes the *compute/context tradeoff, not its
knowledge*. "Improve" = (1) cheaper inference at iso-quality, (2) longer usable context than the
base, or (3) better **recall-per-FLOP** than other sparse methods — **never** "smarter." If you want
smarter, selection is the wrong lever (that's data/training).

## The rung ladder (scale only when a rung is proven insufficient)
1. **Synthetic recall gate** — `synthetic_recall.py`. No model, no data, seconds on CPU. Does a cheap
   selector recover the keys exact attention needs, across relevance geometries? **[DONE — see below]**
2. **Training-free transplant** — monkeypatch the selector into a frozen open model (Llama/Qwen/Mistral),
   reuse Q/K/V/O as-is, measure on real long-context tasks (needle-in-haystack, RULER, LongBench) vs
   dense and vs H2O/StreamingLLM. *[next]*
3. **Train only the selector** (LoRA-for-selection) or LoRA the model to the pattern — only if rung 2
   degrades past tolerance. *[gated by rung 2]*

## Rung-1 finding (recall@32 / 512, d=64, r=3 planted, 300 trials)
- **Bench VALID** (non-degenerate): `exact_topk` ≈ 1.0 on every geometry; `window` ≈ 0.0 on `far`
  (it *can* detect failure) and high on `local`.
- **Coarse-to-fine has a real, SNR-dependent cost.** On scattered (`multi`) relevance, as signal-to-noise
  drops, exact selection still recovers the keys (1.000) but `block_topk` collapses:

  | alpha (SNR) | exact_topk | block_topk | gap |
  |---|---|---|---|
  | 3.0 | 1.000 | 1.000 | +0.000 |
  | 1.5 | 1.000 | 0.912 | +0.088 |
  | 1.0 | 1.000 | 0.704 | +0.296 |
  | 0.7 | 1.000 | 0.544 | +0.456 |

  **Mechanism:** the relevant key's signal is *diluted by its block's distractor neighbours in the mean
  summary*, so the block is pruned even though the key itself is trivially findable. The improvement
  handle is the **block summary** (max-pool / smaller blocks / query-aware or learned pooling) — this
  harness measures any such change directly.

## Rung-1b finding — can a cheaper-than-dense summary close the gap? (`block_summary_fix.py`)
Searched the fix (summary mean vs max, block size, pool_factor) against an honest **selection-cost**
column (q·vector dot products; dense = m = 512). At the hard SNR (alpha=1.0, where mean fell to 0.70):

| variant | cost | recall@1.0 |
|---|---|---|
| exact (ceiling) | 512 (dense) | 1.000 |
| mean b16 pf2 | 96 (19%) | 0.703 |
| **max** b16 pf2 | 96 (19%) | **0.647** (max *backfires* — grabs the most-positive distractor per dim regardless of q's sign) |
| **mean b8** pf2 | 128 (25%) | **0.938** (granularity is the real lever; +6% cost, still misses 0.95) |
| mean b16 pf4 | 160 (31%) | 0.868 |

**No sub-dense static summary clears the predeclared bar (recall ≥ exact−0.05 at sub-dense cost).**
Dilution is intrinsic to block prefiltering at this budget. This *proves the cheap rung insufficient*
(the scale-on-proven-insufficiency trigger): the irreducible residue is a **learned, query-aware block
summary** — that's the novelty lever, now justified rather than assumed (and it needs training → rung 2/3).
Norm-based or other plant-exploiting "fixes" are refused as degeneracies.

## Predeclared falsifiers (binding)
- Bench validity: if `exact_topk` is not ≈1.0 on all geometries OR `window` is not ≈0 on `far`, the
  instrument can't detect failure → results are void (the script aborts).
- Research: if `block_topk` tracks `exact_topk` within 0.05 everywhere exact succeeds → coarse-to-fine
  is free; if it drops below while exact still recovers → block pooling has a real cost (the gap is the
  phenomenon). *[Outcome so far: real cost at low SNR.]*

## Run
```
PYTHONPATH=sparse-llm uv run --python .venv python sparse-llm/synthetic_recall.py
```

## Honesty note
The selector primitives here (top-k = MIPS, block = coarse-to-fine, window = banded, LSH = ANN) are
known objects — the geometries are *not* novel. The contribution being pursued is the **measurement
framing** (selection as a directly recall-scored, swappable instrument) and any selector whose
recall-per-FLOP beats the known ones. Reduce every new idea to {ANN, low-rank, banded, learned router}
first; the irreducible residue is the only novelty claim.
