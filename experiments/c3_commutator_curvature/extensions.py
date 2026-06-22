"""C3 extensions (follow-ups #1 and #2).

#1 TARGETED PAIR PROPOSAL -- beat the C(n,2) enumeration wall.
   True danger pairs are the providers of a shared commitment, so they tend to
   share topical vocabulary ("kilograms"/"pounds", "JSON", "blue"/"archived").
   A cheap lexical proposer (spans sharing a rare stemmed content token) yields
   a small candidate set; we test sigma only on those. Measured against uniform
   sampling at matched test budget, plus recall of the true danger pairs.

#2 HIGHER-ORDER (k>=3) RESIDUE -- close the documented pairwise blind spot.
   A commitment covered by k>=3 spans has sigma_ij = 0 for EVERY pair (deleting
   any two leaves a cover); only the full k-group is a minimal failing set.
   We generalize danger edges to danger HYPEREDGES: a group G is a "minimal
   failing group" if loss(x without G) >= threshold but loss(x without any
   proper subset of G) < threshold. The guard forbids deleting ALL members of
   any minimal failing group (keep >=1). Candidate groups come from the same
   lexical proposer, so higher-order detection stays cheap.

Neither feature reads hidden_* fields: the proposer sees only span TEXT, and
detection uses only the scalar loss oracle.
"""

from __future__ import annotations

import itertools
import re
from typing import Callable

from common import prompt_token_count, span_tokens, make_loss_oracle
from baselines import compute_deltas

_STOP = set("the a an of to in on for and or not but with without is are was were be "
            "this that these those it its as at by from into per must always all any "
            "no nor only than then so if when each every here above given".split())


def _stem(tok: str) -> str:
    tok = tok.lower()
    tok = re.sub(r"[^a-z0-9]", "", tok)
    for suf in ("based", "ing", "ions", "ion", "ers", "er", "ies", "es", "ed", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def content_tokens(text: str) -> set[str]:
    toks = set()
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", text):
        s = _stem(w)
        if len(s) >= 3 and s not in _STOP and not s.isdigit():
            toks.add(s)
    return toks


def lexical_groups(spans, candidate_ids, max_order=2):
    """Propose candidate groups (size 2..max_order) of spans that share a
    content token. Returns a de-duplicated list of sorted id-tuples.

    Buckets candidate spans by stemmed token, then emits in-bucket combinations.
    This is O(sum_t C(|bucket_t|, order)) -- far below C(n, order) when shared
    vocabulary is sparse (which it is: only co-committed spans share keywords).
    """
    cset = set(candidate_ids)
    tok_index: dict[str, list[int]] = {}
    by_id = {s["id"]: s for s in spans}
    for sid in candidate_ids:
        for t in content_tokens(by_id[sid]["text"]):
            tok_index.setdefault(t, []).append(sid)
    groups = set()
    for order in range(2, max_order + 1):
        for ids in tok_index.values():
            ids = [i for i in ids if i in cset]
            if len(ids) >= order:
                for combo in itertools.combinations(sorted(set(ids)), order):
                    groups.add(combo)
    return [list(g) for g in groups]


def _is_minimal_failing(all_ids, group, loss, deltas, threshold):
    """True iff deleting the whole group fails but every proper subset is safe.

    Uses delta cache for size-1 subsets; queries loss for the others.
    """
    rest = [k for k in all_ids if k not in set(group)]
    if loss(rest) < threshold:
        return False
    g = list(group)
    for r in range(1, len(g)):
        for sub in itertools.combinations(g, r):
            keep = [k for k in all_ids if k not in set(sub)]
            if r == 1:
                if deltas[sub[0]] >= threshold:
                    return False
            elif loss(keep) >= threshold:
                return False
    return True


def c3_select_ext(
    instance,
    budget_tokens,
    *,
    tau: float = 0.01,
    residue_threshold: float = 0.5,
    max_order: int = 2,
    proposer: str = "enumerate",   # "enumerate" | "lexical"
    deltas: dict | None = None,
):
    """C3 with a pluggable group proposer and arbitrary max_order.

    proposer="enumerate": all safe combinations up to max_order (the exhaustive
                          ceiling; expensive).
    proposer="lexical":   only lexically-linked candidate groups (cheap).
    """
    spans = instance["spans"]
    all_ids = [s["id"] for s in spans]
    loss = make_loss_oracle(instance)
    if deltas is None:
        deltas = compute_deltas(instance, loss)
    safe = [i for i in all_ids if deltas[i] <= tau]

    if proposer == "lexical":
        candidate_groups = lexical_groups(spans, safe, max_order=max_order)
    else:
        candidate_groups = []
        for order in range(2, max_order + 1):
            candidate_groups.extend(list(c) for c in itertools.combinations(safe, order))

    group_tests = 0
    hyperedges: list[list[int]] = []
    for g in candidate_groups:
        group_tests += 1
        if _is_minimal_failing(all_ids, g, loss, deltas, residue_threshold):
            hyperedges.append(sorted(g))

    # membership: id -> list of hyperedges containing it
    member_of: dict[int, list[set]] = {i: [] for i in all_ids}
    edge_sets = [set(h) for h in hyperedges]
    for es in edge_sets:
        for i in es:
            member_of[i].append(es)

    retained = set(all_ids)
    deleted: list[int] = []
    forced_violation = False
    tok = {s["id"]: span_tokens(s) for s in spans}

    def fits():
        return prompt_token_count(spans, retained) <= budget_tokens

    def forbidden(sid):
        # deleting sid completes a hyperedge iff every OTHER member already gone
        for es in member_of[sid]:
            if all((m == sid) or (m not in retained) for m in es):
                return True
        return False

    order = sorted(all_ids, key=lambda i: (deltas[i], -tok[i], i))
    if not fits():
        for sid in order:
            if sid not in retained or forbidden(sid):
                continue
            retained.discard(sid); deleted.append(sid)
            if fits():
                break
    if not fits():
        # forced: sacrifice the protection that loses the smallest group first
        while not fits():
            cands = [(len(es := next((e for e in member_of[sid]
                                      if all((m == sid) or (m not in retained) for m in e)), set())),
                      deltas[sid], -tok[sid], sid)
                     for sid in retained if forbidden(sid)]
            if not cands:
                break
            cands.sort()
            sid = cands[0][-1]
            retained.discard(sid); deleted.append(sid)
            forced_violation = True

    return retained, {
        "deleted": deleted,
        "group_tests": group_tests,
        "n_candidate_groups": len(candidate_groups),
        "hyperedges": hyperedges,
        "n_hyperedges": len(hyperedges),
        "forced_violation": forced_violation,
        "max_order": max_order,
        "proposer": proposer,
    }


def true_danger_pair_recall(instance, found_edges):
    """Recall of the hidden 2nd-order dangerous pairs among found pairwise edges."""
    truth = {tuple(sorted(p)) for p in instance["hidden_dangerous_pairs"]}
    if not truth:
        return None
    found = {tuple(sorted(e)) for e in found_edges if len(e) == 2}
    return len(truth & found) / len(truth)
