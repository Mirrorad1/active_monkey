"""Compression baselines for the C3 experiment.

Every method takes (instance, budget_tokens, ...) and returns a *retained* set
of span ids whose rendered prompt fits the token budget. None of them read the
hidden_* ground-truth fields; solo_delta_greedy only queries the scalar loss
oracle.
"""

from __future__ import annotations

import random
from typing import Callable

from common import prompt_token_count, span_tokens, make_loss_oracle


def _greedy_delete(spans, order, budget_tokens, forbid=None):
    """Delete spans in `order` until the rendered prompt fits budget_tokens.

    `forbid(span_id, deleted_set)` -> True means "skipping this deletion".
    Returns (retained_ids:set, deleted:list, forced:bool). `forced` is unused
    here (C3 sets it); kept for a uniform return shape.
    """
    retained = {s["id"] for s in spans}
    deleted = []
    if prompt_token_count(spans, retained) <= budget_tokens:
        return retained, deleted, False
    for sid in order:
        if sid not in retained:
            continue
        if forbid is not None and forbid(sid, set(deleted)):
            continue
        retained.discard(sid)
        deleted.append(sid)
        if prompt_token_count(spans, retained) <= budget_tokens:
            break
    return retained, deleted, False


def compute_deltas(instance, loss: Callable) -> dict[int, float]:
    """delta_i = loss(x without span i) - loss(x).  Base loss is 0 by design."""
    spans = instance["spans"]
    all_ids = [s["id"] for s in spans]
    base = loss(all_ids)
    deltas = {}
    for sid in all_ids:
        deltas[sid] = loss([i for i in all_ids if i != sid]) - base
    return deltas


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def full_prompt(instance, budget_tokens, **_):
    return {s["id"] for s in instance["spans"]}, {"deleted": [], "pair_tests": 0}


def random_delete(instance, budget_tokens, seed=0, **_):
    spans = instance["spans"]
    rng = random.Random(seed * 100003 + hash(instance["id"]) % 100003)
    order = [s["id"] for s in spans]
    rng.shuffle(order)
    retained, deleted, _ = _greedy_delete(spans, order, budget_tokens)
    return retained, {"deleted": deleted, "pair_tests": 0}


def length_greedy(instance, budget_tokens, **_):
    """Independent-salience baseline: delete shortest (lowest-density) spans first."""
    spans = instance["spans"]
    order = [s["id"] for s in sorted(spans, key=lambda s: (span_tokens(s), s["id"]))]
    retained, deleted, _ = _greedy_delete(spans, order, budget_tokens)
    return retained, {"deleted": deleted, "pair_tests": 0}


def solo_delta_greedy(instance, budget_tokens, deltas=None, **_):
    """Main baseline C3 must beat: delete spans with lowest individual loss first."""
    spans = instance["spans"]
    loss = make_loss_oracle(instance)
    if deltas is None:
        deltas = compute_deltas(instance, loss)
    # lowest delta first; ties broken by biggest token savings (fastest to budget)
    order = [s["id"] for s in sorted(
        spans, key=lambda s: (deltas[s["id"]], -span_tokens(s), s["id"]))]
    retained, deleted, _ = _greedy_delete(spans, order, budget_tokens)
    return retained, {"deleted": deleted, "pair_tests": 0, "deltas": deltas}
