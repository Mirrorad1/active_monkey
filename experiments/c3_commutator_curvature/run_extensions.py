"""Run the two C3 follow-up studies and write results/extensions_summary.{md,json}.

Study 1 (targeted pair proposal): on the k=2 dataset, does a cheap lexical
  proposer recover C3's accuracy at far fewer pair tests than full enumeration,
  and beat uniform sampling? Reports accuracy, pair tests, true-danger recall.

Study 2 (higher-order residue): on a k>=3 dataset, pairwise C3 should fail the
  k-covers (collapse toward solo); an order-3 detector should recover them.

Usage:
    python experiments/c3_commutator_curvature/generate_dataset.py --n 1000 --seed 7
    python experiments/c3_commutator_curvature/generate_dataset.py --n 1000 --seed 7 \
        --kcover 3 --out experiments/c3_commutator_curvature/data/synthetic_c3_k3.jsonl
    python experiments/c3_commutator_curvature/run_extensions.py
"""

from __future__ import annotations

import json
import math
import os
import time

from common import (load_dataset, split_dataset, is_correct, make_loss_oracle,
                    prompt_token_count)
from baselines import compute_deltas, solo_delta_greedy
from c3_selector import c3_select
from extensions import c3_select_ext, true_danger_pair_recall

BUDGETS = (0.35, 0.25)
TAU = 0.01
RT = 0.5
HERE = os.path.dirname(os.path.abspath(__file__))


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _erased(instance, retained):
    deleted = {s["id"] for s in instance["spans"]} - set(retained)
    g = instance["hidden_fragile_groups"]
    return (sum(1 for grp in g if set(grp) <= deleted), len(g))


def precompute(insts):
    out = []
    for i in insts:
        loss = make_loss_oracle(i)
        out.append({"full": prompt_token_count(i["spans"]),
                    "deltas": compute_deltas(i, loss)})
    return out


def study1(test, pres):
    """k=2 dataset: lexical proposer vs enumerate vs uniform sampling."""
    families = {"B", "C", "D"}
    rows = {}
    methods = ["solo", "c3_enumerate", "c3_lexical", "c3_uniform_25", "c3_uniform_10"]
    for b in BUDGETS:
        for m in methods:
            acc, tests, recalls, er_n, er_d = [], [], [], 0, 0
            t0 = time.time()
            for inst, pre in zip(test, pres):
                if inst["family"] not in families:
                    continue
                bt = math.ceil(b * pre["full"])
                seeds = [1, 2, 3, 4, 5] if "uniform" in m else [0]
                for sd in seeds:
                    if m == "solo":
                        r, info = solo_delta_greedy(inst, bt, deltas=pre["deltas"])
                        edges = []
                    elif m == "c3_enumerate":
                        r, info = c3_select_ext(inst, bt, deltas=pre["deltas"],
                                                proposer="enumerate", max_order=2,
                                                tau=TAU, residue_threshold=RT)
                        edges = info["hyperedges"]; tests.append(info["group_tests"])
                    elif m == "c3_lexical":
                        r, info = c3_select_ext(inst, bt, deltas=pre["deltas"],
                                                proposer="lexical", max_order=2,
                                                tau=TAU, residue_threshold=RT)
                        edges = info["hyperedges"]; tests.append(info["group_tests"])
                    else:
                        frac = 0.25 if "25" in m else 0.10
                        r, info = c3_select(inst, bt, pair_fraction=frac, deltas=pre["deltas"],
                                            seed=sd, tau=TAU, residue_threshold=RT)
                        edges = info["danger_edges"]; tests.append(info["pair_tests"])
                    acc.append(1.0 if is_correct(inst, r) else 0.0)
                    rec = true_danger_pair_recall(inst, edges)
                    if rec is not None:
                        recalls.append(rec)
                    n, d = _erased(inst, r); er_n += n; er_d += d
            rows[(m, b)] = {"accuracy": _mean(acc), "avg_pair_tests": _mean(tests),
                            "danger_recall": _mean(recalls),
                            "fragile_erasure": (er_n / er_d) if er_d else 0.0,
                            "runtime_s": time.time() - t0}
    return methods, rows


def study2(test, pres):
    """k>=3 dataset: pairwise vs order-3 detection on B/D families."""
    families = {"B", "D"}
    rows = {}
    methods = ["solo", "c3_pairwise_enum", "c3_order3_lexical", "c3_order3_enum"]
    for b in BUDGETS:
        for m in methods:
            acc, tests, er_n, er_d = [], [], 0, 0
            t0 = time.time()
            for inst, pre in zip(test, pres):
                if inst["family"] not in families:
                    continue
                bt = math.ceil(b * pre["full"])
                if m == "solo":
                    r, info = solo_delta_greedy(inst, bt, deltas=pre["deltas"])
                else:
                    order = 2 if "pairwise" in m else 3
                    prop = "lexical" if "lexical" in m else "enumerate"
                    r, info = c3_select_ext(inst, bt, deltas=pre["deltas"], proposer=prop,
                                            max_order=order, tau=TAU, residue_threshold=RT)
                    tests.append(info["group_tests"])
                acc.append(1.0 if is_correct(inst, r) else 0.0)
                n, d = _erased(inst, r); er_n += n; er_d += d
            rows[(m, b)] = {"accuracy": _mean(acc), "avg_group_tests": _mean(tests),
                            "fragile_erasure": (er_n / er_d) if er_d else 0.0,
                            "runtime_s": time.time() - t0}
    return methods, rows


def main():
    k2 = load_dataset(os.path.join(HERE, "data", "synthetic_c3.jsonl"))
    _, _, test2 = split_dataset(k2)
    pres2 = precompute(test2)
    print(f"study1: {len(test2)} test instances (k=2)")
    m1, r1 = study1(test2, pres2)

    k3_path = os.path.join(HERE, "data", "synthetic_c3_k3.jsonl")
    m2 = r2 = test3 = None
    if os.path.exists(k3_path):
        k3 = load_dataset(k3_path)
        _, _, test3 = split_dataset(k3)
        pres3 = precompute(test3)
        print(f"study2: {len(test3)} test instances (k>=3)")
        m2, r2 = study2(test3, pres3)
    else:
        print(f"!! {k3_path} missing -- generate it with --kcover 3 to run study2")

    out = {"study1_targeted_pairs": {f"{m}@{b}": r1[(m, b)] for m in m1 for b in BUDGETS}}
    if r2:
        out["study2_higher_order"] = {f"{m}@{b}": r2[(m, b)] for m in m2 for b in BUDGETS}
    with open(os.path.join(HERE, "results", "extensions_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    md = ["# C3 extensions — targeted pairs (#1) & higher-order residue (#2)\n"]
    md.append("## Study 1 — targeted pair proposal (k=2 data, families B/C/D)\n")
    md.append("Can a cheap lexical proposer match full-enumeration accuracy at far fewer "
              "pair tests, and beat uniform sampling?\n")
    md.append("| method | budget | accuracy | avg pair tests | true-danger recall | fragile erasure |")
    md.append("|---|---|---|---|---|---|")
    for b in BUDGETS:
        for m in m1:
            d = r1[(m, b)]
            md.append(f"| {m} | {b} | {d['accuracy']*100:.1f}% | {d['avg_pair_tests']:.1f} | "
                      f"{d['danger_recall']*100:.1f}% | {d['fragile_erasure']:.3f} |")
    if r2:
        md.append("\n## Study 2 — higher-order (k>=3) residue (B/D families, kcover=3)\n")
        md.append("Pairwise sigma is blind to k>=3 covers; an order-3 detector should recover them.\n")
        md.append("| method | budget | accuracy | avg group tests | fragile erasure |")
        md.append("|---|---|---|---|---|")
        for b in BUDGETS:
            for m in m2:
                d = r2[(m, b)]
                md.append(f"| {m} | {b} | {d['accuracy']*100:.1f}% | "
                          f"{d['avg_group_tests']:.1f} | {d['fragile_erasure']:.3f} |")
    with open(os.path.join(HERE, "results", "extensions_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print("\n=== STUDY 1 (targeted pairs) ===")
    for b in BUDGETS:
        print(f" budget {b}:")
        for m in m1:
            d = r1[(m, b)]
            print(f"   {m:16s} acc={d['accuracy']:.3f} pairtests={d['avg_pair_tests']:6.1f} "
                  f"recall={d['danger_recall']:.3f} erasure={d['fragile_erasure']:.3f}")
    if r2:
        print("\n=== STUDY 2 (higher-order k>=3) ===")
        for b in BUDGETS:
            print(f" budget {b}:")
            for m in m2:
                d = r2[(m, b)]
                print(f"   {m:18s} acc={d['accuracy']:.3f} grouptests={d['avg_group_tests']:7.1f} "
                      f"erasure={d['fragile_erasure']:.3f}")
    print("\nwrote results/extensions_summary.{json,md}")


if __name__ == "__main__":
    main()
