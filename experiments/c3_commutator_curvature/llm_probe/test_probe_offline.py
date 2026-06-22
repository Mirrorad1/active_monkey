"""Offline (no-torch) guards for llm_probe coverage + bootstrap logic.

The LM class imports torch lazily, so everything here runs on a CPU venv.
Run: python test_probe_offline.py
"""

import random

import llm_probe as L


def _approx(a, b, eps=1e-9):
    return abs(a - b) <= eps


def test_collect_split_coverage():
    # A: true pair safe + tested ; B: gated out ; C: safe but dropped by cap
    cache = {
        "A": (0.0, {0: 0.0, 1: 0.0, 2: 0.0}, {(0, 1): 2.0, (0, 2): 0.0}),
        "B": (0.0, {0: 0.5, 1: 0.0, 2: 0.0}, {(1, 2): 0.0}),
        "C": (0.0, {0: 0.0, 1: 0.0, 2: 0.0}, {(0, 2): 0.1}),
    }
    split = [{"id": x, "hidden_dangerous_pair": [0, 1]} for x in ("A", "B", "C")]
    recs, cov = L.collect_split(cache, split, tau=0.05)
    assert cov["n"] == 3
    assert cov["safe_gated"] == 2, cov          # A, C
    assert cov["tested"] == 1, cov              # A only
    assert cov["missing_gate"] == 1, cov        # B (delta[0]=0.5 > tau)
    assert cov["missing_cap"] == 1, cov         # C (safe but not in sigma)
    # records: A has true=2.0, B/C have true=None
    assert recs[0]["true"] == 2.0 and recs[1]["true"] is None and recs[2]["true"] is None
    print("ok test_collect_split_coverage")


def test_pooled_and_bootstrap_auc():
    cache = {
        "A": (0.0, {0: 0.0, 1: 0.0, 2: 0.0}, {(0, 1): 2.0, (0, 2): 0.0}),
        "C": (0.0, {0: 0.0, 1: 0.0, 2: 0.0}, {(0, 1): 3.0, (0, 2): 0.5}),
    }
    split = [{"id": x, "hidden_dangerous_pair": [0, 1]} for x in ("A", "C")]
    recs, _ = L.collect_split(cache, split, tau=0.05)
    assert _approx(L.pooled_auc(recs), 1.0)     # true sigmas (2,3) > rand (0,0.5)
    ci1 = L.bootstrap_auc(recs, 500, random.Random(1))
    ci2 = L.bootstrap_auc(recs, 500, random.Random(1))
    assert ci1 == ci2, "bootstrap must be deterministic for a fixed rng"
    assert ci1 is not None and 0.0 <= ci1[0] <= ci1[1] <= 1.0
    # empty positive class -> None (not a spurious AUC)
    empty = [{"true": None, "rand": [0.1, 0.2]}]
    assert L.pooled_auc(empty) is None
    print("ok test_pooled_and_bootstrap_auc")


def test_bootstrap_mean_ci():
    vals = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    ci = L.bootstrap_mean_ci(vals, 1000, random.Random(7))
    assert ci is not None and ci[0] <= sum(vals) / len(vals) <= ci[1]
    assert L.bootstrap_mean_ci([], 100, random.Random(0)) is None
    print("ok test_bootstrap_mean_ci")


def test_dataset_and_selectors():
    rng = random.Random(7)
    inst = L.make_instance(rng, 0)
    dp = inst["hidden_dangerous_pair"]
    ids = [s["id"] for s in inst["spans"]]
    delta = {i: 0.0 for i in ids}
    q = inst["question"]
    bt = L.token_count(L.render(inst["spans"], ids, q)) // 3
    # C3 with a real danger edge on the true pair must keep >=1 endpoint
    r_c3 = L.select_c3(inst["spans"], delta, {tuple(dp): 2.0}, 0.5, bt, q)
    assert (dp[0] in r_c3) or (dp[1] in r_c3)
    print("ok test_dataset_and_selectors")


if __name__ == "__main__":
    test_collect_split_coverage()
    test_pooled_and_bootstrap_auc()
    test_bootstrap_mean_ci()
    test_dataset_and_selectors()
    print("ALL OFFLINE TESTS PASSED")
