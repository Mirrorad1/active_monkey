"""C3 selector: residue-guarded compression.

Core idea (see README for math): a pair of spans can each be individually safe
to delete (delta_i ~= 0, delta_j ~= 0) yet jointly break the task
(sigma_ij = loss(without i,j) - delta_i - delta_j >= threshold). Such a pair is
a *danger edge*. C3 deletes low-delta spans greedily but never deletes BOTH
endpoints of a danger edge.

The selector only ever sees:
  - the span list (text + ids, for token accounting)
  - a scalar loss(retained_ids) oracle
It never reads hidden_required_commitments / fragile_groups / dangerous_pairs.
"""

from __future__ import annotations

import itertools
import random
from typing import Callable

from common import prompt_token_count, span_tokens, make_loss_oracle
from baselines import compute_deltas


def _sampled_pairs(safe, fraction, seed):
    all_pairs = list(itertools.combinations(safe, 2))
    if fraction >= 1.0:
        return all_pairs, len(all_pairs)
    k = max(0, int(round(len(all_pairs) * fraction)))
    rng = random.Random(seed * 7919 + 17)
    rng.shuffle(all_pairs)
    return all_pairs[:k], len(all_pairs)


def c3_select(
    instance,
    budget_tokens,
    *,
    tau: float = 0.01,
    residue_threshold: float = 0.5,
    pair_fraction: float = 1.0,
    danger_edges_enabled: bool = True,
    random_edges: bool = False,
    deltas: dict | None = None,
    seed: int = 0,
):
    spans = instance["spans"]
    all_ids = [s["id"] for s in spans]
    loss = make_loss_oracle(instance)
    if deltas is None:
        deltas = compute_deltas(instance, loss)

    # ---- candidate-safe spans: individually harmless to delete -------------
    safe = [i for i in all_ids if deltas[i] <= tau]

    danger_adj: dict[int, set] = {i: set() for i in all_ids}
    danger_sigma: dict[tuple, float] = {}
    pair_tests = 0
    n_all_pairs = 0

    if danger_edges_enabled:
        tested, n_all_pairs = _sampled_pairs(safe, pair_fraction, seed)
        if random_edges:
            # Ablation: keep the SAME number of edges as the real residue would
            # produce, but place them on random safe pairs. Tests whether real
            # second-order residue (not just "some constraint graph") matters.
            real_edges = []
            for (i, j) in itertools.combinations(safe, 2):
                pair_tests += 1
                sigma = loss([k for k in all_ids if k not in (i, j)]) - deltas[i] - deltas[j]
                if sigma >= residue_threshold:
                    real_edges.append((i, j, sigma))
            n_edges = len(real_edges)
            rng = random.Random(seed * 104729 + 3)
            pool = list(itertools.combinations(safe, 2))
            rng.shuffle(pool)
            for (i, j) in pool[:n_edges]:
                danger_adj[i].add(j); danger_adj[j].add(i)
                danger_sigma[(min(i, j), max(i, j))] = 1.0  # placeholder weight
        else:
            for (i, j) in tested:
                pair_tests += 1
                sigma = loss([k for k in all_ids if k not in (i, j)]) - deltas[i] - deltas[j]
                if sigma >= residue_threshold:
                    danger_adj[i].add(j); danger_adj[j].add(i)
                    danger_sigma[(min(i, j), max(i, j))] = sigma

    # ---- guarded greedy deletion -----------------------------------------
    retained = set(all_ids)
    deleted: list[int] = []
    forced_violation = False
    forced_pairs: list = []

    def fits():
        return prompt_token_count(spans, retained) <= budget_tokens

    def forbidden(sid):
        # deleting sid would complete a danger edge whose other endpoint is gone
        return any(n not in retained for n in danger_adj[sid])

    tok = {s["id"]: span_tokens(s) for s in spans}
    order = sorted(all_ids, key=lambda i: (deltas[i], -tok[i], i))

    if not fits():
        for sid in order:
            if sid not in retained:
                continue
            if danger_edges_enabled and forbidden(sid):
                continue
            retained.discard(sid)
            deleted.append(sid)
            if fits():
                break

    # ---- forced relaxation: only if budget still unmet and all remaining
    #      deletable spans are danger-protected ----------------------------
    if not fits():
        while not fits():
            # candidates: retained spans whose deletion would violate an edge
            cands = []
            for sid in retained:
                viol_sigmas = [danger_sigma.get((min(sid, n), max(sid, n)), 0.0)
                               for n in danger_adj[sid] if n not in retained]
                if viol_sigmas:
                    cands.append((min(viol_sigmas), deltas[sid], -tok[sid], sid))
            if not cands:
                # nothing left to delete at all
                break
            cands.sort()
            _, _, _, sid = cands[0]
            for n in danger_adj[sid]:
                if n not in retained:
                    forced_pairs.append(sorted((sid, n)))
            retained.discard(sid)
            deleted.append(sid)
            forced_violation = True

    return retained, {
        "deleted": deleted,
        "deltas": deltas,
        "pair_tests": pair_tests,
        "n_all_pairs": n_all_pairs,
        "n_danger_edges": len(danger_sigma),
        "danger_edges": [list(k) for k in danger_sigma.keys()],
        "danger_sigma": {f"{a}-{b}": v for (a, b), v in danger_sigma.items()},
        "forced_violation": forced_violation,
        "forced_pairs": forced_pairs,
    }
