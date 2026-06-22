"""sparse-llm — rung 1b: can a cheaper-than-dense block summary close the low-SNR gap?

Rung-1 found the coarse-to-fine meta-selector's failure: block-MEAN pooling dilutes a scattered
relevant key's signal, so its block is pruned (recall 1.000 -> 0.544 as SNR drops) even though
exact selection still finds it. Here we search the fix and MEASURE it honestly against cost —
because a "fix" that just keeps every block is dense selection wearing a hat.

Levers (each keeps the per-block summary O(d), i.e. genuinely sub-dense):
  - summary: mean vs elementwise-max (max should be less diluted by distractor neighbours)
  - block size: smaller blocks = fewer distractors per summary (but more blocks to score)
  - pool_factor: keep more candidate blocks (but a larger fine-rank pool)

Cost = q·vector dot products to do the SELECTION (selection.block_select_cost):
exact = m (= dense, no saving); window/random = 0. The win is recall -> exact at cost << m.

PREDECLARED: a variant "fixes" the gap iff at the hard SNR (alpha=1.0, where mean fell to ~0.70)
it reaches recall >= exact-0.05 AND costs < m (still sub-dense). Report the recall/cost frontier;
do not call a variant a fix if it only matches exact by paying ~dense cost.

Run: PYTHONPATH=sparse-llm uv run --python .venv python sparse-llm/block_summary_fix.py
"""
import numpy as np

from selection import exact_topk, make_block_selector, block_select_cost
from synthetic_recall import make_task, recall

N, D, R, BUDGET, TRIALS = 512, 64, 3, 32, 300
M = N                      # eligible count at qpos=N-1 is N
TOL = 0.05

# (label, selector, cost) — cost in q·vector dot products for the SELECTION step.
VARIANTS = [
    ("exact",        exact_topk,                                M),
    ("mean b16 pf2", make_block_selector(16, 2, "mean"), block_select_cost(M, BUDGET, 16, 2)),
    ("max  b16 pf2", make_block_selector(16, 2, "max"),  block_select_cost(M, BUDGET, 16, 2)),
    ("mean b8  pf2", make_block_selector(8, 2, "mean"),  block_select_cost(M, BUDGET, 8, 2)),
    ("max  b8  pf2", make_block_selector(8, 2, "max"),   block_select_cost(M, BUDGET, 8, 2)),
    ("mean b16 pf4", make_block_selector(16, 4, "mean"), block_select_cost(M, BUDGET, 16, 4)),
]
ALPHAS = [3.0, 2.0, 1.5, 1.0, 0.7]


def sweep():
    rng = np.random.default_rng(0)
    # recall[label][alpha]
    rec = {lab: {} for lab, _, _ in VARIANTS}
    for alpha in ALPHAS:
        for _ in range(TRIALS):
            q, K, qpos, Rset = make_task("multi", N, D, R, alpha, rng)
            for lab, fn, _ in VARIANTS:
                rec[lab].setdefault(alpha, []).append(recall(fn(q, K, BUDGET, qpos, rng=rng), Rset))
    return {lab: {a: float(np.mean(v)) for a, v in d.items()} for lab, d in rec.items()}


def main():
    rec = sweep()
    cost = {lab: c for lab, _, c in VARIANTS}

    print(f"sparse-llm rung-1b — block-summary fix search (multi geometry, recall@{BUDGET}/{N}, "
          f"r={R}, trials={TRIALS})")
    print(f"selection cost (q·vec dot products; dense=m={M}):")
    for lab, _, c in VARIANTS:
        print(f"    {lab:<12} {c:>4}   ({'dense' if c == M else f'{c/M:.0%} of dense'})")

    print(f"\n{'variant':<12} " + " ".join(f"a={a:<4}" for a in ALPHAS))
    for lab, _, _ in VARIANTS:
        print(f"{lab:<12} " + " ".join(f"{rec[lab][a]:>5.3f}" for a in ALPHAS))

    # predeclared fix test at the hard SNR
    hard = 1.0
    exact_hard = rec["exact"][hard]
    print(f"\n[fix test @ alpha={hard}]  exact={exact_hard:.3f}, target recall >= {exact_hard - TOL:.3f}, "
          f"cost < {M} (sub-dense):")
    winners = []
    for lab, _, c in VARIANTS:
        if lab == "exact":
            continue
        r = rec[lab][hard]
        ok = (r >= exact_hard - TOL) and (c < M)
        if ok:
            winners.append((lab, r, c))
        print(f"    {lab:<12} recall={r:.3f}  cost={c:<4} ({c/M:.0%})  -> {'FIX' if ok else '-'}")

    print("\nVERDICT:", (
        "no sub-dense block-summary variant recovers the gap at the hard SNR — the dilution is "
        "intrinsic to block prefiltering at this budget; the residue (a learned/query-aware summary, "
        "or abandoning blocks for direct cheap scoring) is the open lever."
        if not winners else
        "best sub-dense fix at the hard SNR: " +
        min(winners, key=lambda w: w[2]).__repr__() +
        " (lowest-cost variant reaching exact-0.05) — block pooling's cost is closable by a cheaper "
        "summary/granularity, and this is the recall-per-cost handle to push."))


if __name__ == "__main__":
    main()
