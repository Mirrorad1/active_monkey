"""sparse-llm — selection mechanisms as a swappable, recall-scored instrument.

A Selector answers the only question sparse attention really asks: given a query and a key
cache, WHICH keys do we attend to, at a fixed budget? Making selection a clean, swappable
interface lets us measure RECALL (did we keep the keys exact attention would have used?)
directly — instead of trusting an averaged perplexity that launders capability loss.

Every selector has the signature (q, K, budget, qpos, rng) -> int array of selected indices
(<= budget). Causal: only keys at positions <= qpos are eligible.

  exact_topk : the CEILING for content selection (true top-budget by q·k). Everything is graded
               against this — any sparse selector's job is to recover what it would pick.
  window     : content-BLIND positional baseline (the most-recent budget keys). The "fixed gait":
               high recall when relevance is local, ~0 when it is far -> the non-degenerate control.
  random     : floor baseline.
  block_topk : the coarse-to-fine META-selector (the "abstraction over selection") — pool keys
               into blocks, score blocks by q·mean(block), keep the top blocks as a candidate pool,
               then take exact top-budget within it. Tests whether a cheap block prefilter KEEPS the
               relevant keys, or dilutes scattered relevance away in the block mean.
"""
import numpy as np


def _eligible(qpos, n):
    """Causal mask: positions 0..qpos are attendable from query position qpos."""
    return np.arange(qpos + 1)


def exact_topk(q, K, budget, qpos, rng=None):
    elig = _eligible(qpos, K.shape[0])
    s = K[elig] @ q
    return elig[np.argsort(-s)[:budget]]


def window(q, K, budget, qpos, rng=None):
    n = K.shape[0]
    lo = max(0, qpos - budget + 1)
    return np.arange(lo, min(qpos + 1, n))


def random_sel(q, K, budget, qpos, rng):
    elig = _eligible(qpos, K.shape[0])
    if elig.size <= budget:
        return elig
    return rng.choice(elig, size=budget, replace=False)


def block_topk(q, K, budget, qpos, rng=None, block=16, pool_factor=2):
    """Coarse prefilter (block level) -> fine select (token level), budget-matched.

    Keep top blocks until the candidate pool >= pool_factor*budget members, then take exact
    top-budget within the pool. A relevant key is lost iff its BLOCK is pruned in the coarse
    step (its boosted q·k diluted by the block's distractor neighbours in the mean summary).
    """
    elig = _eligible(qpos, K.shape[0])
    m = elig.size
    if m <= budget:
        return elig
    nblocks = int(np.ceil(m / block))
    members = [elig[b * block:(b + 1) * block] for b in range(nblocks)]
    summaries = np.array([K[mem].mean(axis=0) for mem in members])
    order = np.argsort(-(summaries @ q))
    target = min(m, pool_factor * budget)
    cand = []
    for bi in order:
        cand.extend(members[bi].tolist())
        if len(cand) >= target:
            break
    cand = np.array(cand)
    s = K[cand] @ q
    return cand[np.argsort(-s)[:budget]]


def _summary(Kblock, kind):
    """Cheap O(d) per-block representative used to score the block against q."""
    if kind == "mean":
        return Kblock.mean(axis=0)
    if kind == "max":
        return Kblock.max(axis=0)          # elementwise max — less dilution by distractors
    raise ValueError(kind)


def make_block_selector(block=16, pool_factor=2, summary="mean"):
    """Parametrised coarse-to-fine selector. The rung-1b fix-search varies block size, how many
    candidate blocks to keep (pool_factor), and the block summary (mean vs max)."""
    def sel(q, K, budget, qpos, rng=None):
        elig = _eligible(qpos, K.shape[0])
        m = elig.size
        if m <= budget:
            return elig
        nblocks = int(np.ceil(m / block))
        members = [elig[b * block:(b + 1) * block] for b in range(nblocks)]
        S = np.array([_summary(K[mem], summary) for mem in members])
        order = np.argsort(-(S @ q))
        target = min(m, pool_factor * budget)
        cand = []
        for bi in order:
            cand.extend(members[bi].tolist())
            if len(cand) >= target:
                break
        cand = np.array(cand)
        return cand[np.argsort(-(K[cand] @ q))[:budget]]
    return sel


def block_select_cost(m, budget, block=16, pool_factor=2):
    """Selection overhead in q·vector dot products: score every block + fine-rank the pool.
    Compare to exact_topk's m (= dense cost, no saving) and window/random's 0."""
    if m <= budget:
        return 0
    return int(np.ceil(m / block)) + min(m, pool_factor * budget)


SELECTORS = {
    "exact_topk": exact_topk,
    "block_topk": block_topk,
    "window": window,
    "random": random_sel,
}
