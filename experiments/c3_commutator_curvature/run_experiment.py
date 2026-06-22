"""C3 experiment runner.

Loads the synthetic dataset, computes per-instance second-order residue ground
truth (delta_i, sigma over hidden dangerous pairs), runs every compression
method at every budget ratio, and writes results + failure analysis.

Usage:
    python experiments/c3_commutator_curvature/run_experiment.py \
        --dataset experiments/c3_commutator_curvature/data/synthetic_c3.jsonl --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict

from common import (load_dataset, split_dataset, prompt_token_count,
                    is_correct, derive_answer, make_loss_oracle, TOKENIZER_NAME)
from baselines import (compute_deltas, full_prompt, random_delete,
                       length_greedy, solo_delta_greedy)
from c3_selector import c3_select

BUDGETS = [0.75, 0.50, 0.35, 0.25, 0.15]
SEEDS = [1, 2, 3, 4, 5]
TAU = 0.01
RESIDUE_THRESHOLD = 0.5


def precompute(instance):
    """Per-instance intrinsics: full token count, deltas, hidden-pair sigmas."""
    loss = make_loss_oracle(instance)
    deltas = compute_deltas(instance, loss)
    all_ids = [s["id"] for s in instance["spans"]]
    pair_sigma = {}
    for (i, j) in instance["hidden_dangerous_pairs"]:
        s = loss([k for k in all_ids if k not in (i, j)]) - deltas[i] - deltas[j]
        pair_sigma[(i, j)] = s
    return {
        "full_tokens": prompt_token_count(instance["spans"]),
        "deltas": deltas,
        "pair_sigma": pair_sigma,
    }


def eval_retained(instance, retained, pre):
    spans_ids = {s["id"] for s in instance["spans"]}
    deleted = spans_ids - set(retained)
    deltas = pre["deltas"]
    # dangerous pair violations / sigma
    viol_sigmas = []
    for (i, j), sig in pre["pair_sigma"].items():
        if i in deleted and j in deleted:
            viol_sigmas.append(sig)
    # fragile group erasure
    groups = instance["hidden_fragile_groups"]
    erased = sum(1 for g in groups if set(g) <= deleted)
    return {
        "correct": is_correct(instance, retained),
        "tokens": prompt_token_count(instance["spans"], retained),
        "deleted_deltas": [deltas[i] for i in deleted],
        "n_pair_violations": len(viol_sigmas),
        "viol_sigmas": viol_sigmas,
        "n_groups": len(groups),
        "n_groups_erased": erased,
    }


def run_method(name, instance, budget_tokens, pre, seed):
    deltas = pre["deltas"]
    if name == "full_prompt":
        r, info = full_prompt(instance, budget_tokens)
    elif name == "random_delete":
        r, info = random_delete(instance, budget_tokens, seed=seed)
    elif name == "length_greedy":
        r, info = length_greedy(instance, budget_tokens)
    elif name == "solo_delta_greedy":
        r, info = solo_delta_greedy(instance, budget_tokens, deltas=deltas)
    elif name == "c3_residue_guarded":
        r, info = c3_select(instance, budget_tokens, tau=TAU,
                            residue_threshold=RESIDUE_THRESHOLD, deltas=deltas, seed=seed)
    # ablations
    elif name == "c3_pairs_100":
        r, info = c3_select(instance, budget_tokens, pair_fraction=1.0,
                            deltas=deltas, seed=seed, tau=TAU, residue_threshold=RESIDUE_THRESHOLD)
    elif name == "c3_pairs_25":
        r, info = c3_select(instance, budget_tokens, pair_fraction=0.25,
                            deltas=deltas, seed=seed, tau=TAU, residue_threshold=RESIDUE_THRESHOLD)
    elif name == "c3_pairs_10":
        r, info = c3_select(instance, budget_tokens, pair_fraction=0.10,
                            deltas=deltas, seed=seed, tau=TAU, residue_threshold=RESIDUE_THRESHOLD)
    elif name == "c3_no_danger":
        r, info = c3_select(instance, budget_tokens, danger_edges_enabled=False,
                            deltas=deltas, seed=seed, tau=TAU, residue_threshold=RESIDUE_THRESHOLD)
    elif name == "c3_random_edges":
        r, info = c3_select(instance, budget_tokens, random_edges=True,
                            deltas=deltas, seed=seed, tau=TAU, residue_threshold=RESIDUE_THRESHOLD)
    else:
        raise ValueError(name)
    return r, info


STOCHASTIC = {"random_delete", "c3_pairs_25", "c3_pairs_10", "c3_random_edges"}


def aggregate(name, instances, pres, budget_ratio):
    """Return aggregated metrics for one method at one budget over `instances`."""
    seeds = SEEDS if name in STOCHASTIC else [0]
    acc = defaultdict(list)            # family -> [correct...]
    tokens, ratios = [], []
    pair_viol_flags, group_erase_num, group_total = [], 0, 0
    deleted_deltas, viol_sigmas, pair_tests, forced = [], [], [], []
    t0 = time.time()
    for inst, pre in zip(instances, pres):
        budget_tokens = math.ceil(budget_ratio * pre["full_tokens"])
        for seed in seeds:
            retained, info = run_method(name, inst, budget_tokens, pre, seed)
            m = eval_retained(inst, retained, pre)
            acc[inst["family"]].append(1.0 if m["correct"] else 0.0)
            acc["ALL"].append(1.0 if m["correct"] else 0.0)
            tokens.append(m["tokens"])
            ratios.append(m["tokens"] / pre["full_tokens"] if pre["full_tokens"] else 0)
            pair_viol_flags.append(1.0 if m["n_pair_violations"] > 0 else 0.0)
            group_erase_num += m["n_groups_erased"]
            group_total += m["n_groups"]
            deleted_deltas.extend(m["deleted_deltas"])
            viol_sigmas.extend(m["viol_sigmas"])
            pair_tests.append(info.get("pair_tests", 0))
            forced.append(1.0 if info.get("forced_violation") else 0.0)
    runtime = time.time() - t0
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    return {
        "accuracy": mean(acc["ALL"]),
        "accuracy_by_family": {f: mean(acc[f]) for f in ("A", "B", "C", "D") if acc[f]},
        "avg_tokens": mean(tokens),
        "compression_ratio": mean(ratios),
        "dangerous_pair_violation_rate": mean(pair_viol_flags),
        "fragile_group_erasure_rate": (group_erase_num / group_total) if group_total else 0.0,
        "avg_delta_deleted": mean(deleted_deltas),
        "avg_sigma_violated": mean(viol_sigmas),
        "avg_pair_tests": mean(pair_tests),
        "forced_violation_rate": mean(forced),
        "runtime_s": runtime,
        "n_seeds": len(seeds),
    }


def family_acc(name, instances, pres, budget_ratio, families):
    """Accuracy over a subset of families (for the success criteria)."""
    seeds = SEEDS if name in STOCHASTIC else [0]
    vals = []
    for inst, pre in zip(instances, pres):
        if inst["family"] not in families:
            continue
        budget_tokens = math.ceil(budget_ratio * pre["full_tokens"])
        for seed in seeds:
            retained, _ = run_method(name, inst, budget_tokens, pre, seed)
            vals.append(1.0 if is_correct(inst, retained) else 0.0)
    return sum(vals) / len(vals) if vals else 0.0


def collect_failure_cases(instances, pres, budget_ratio=0.25):
    """10 cases where solo failed & C3 succeeded; 10 where C3 failed/both failed."""
    c3_wins, c3_loses = [], []
    for inst, pre in zip(instances, pres):
        budget_tokens = math.ceil(budget_ratio * pre["full_tokens"])
        r_solo, _ = solo_delta_greedy(inst, budget_tokens, deltas=pre["deltas"])
        r_c3, info_c3 = c3_select(inst, budget_tokens, tau=TAU,
                                  residue_threshold=RESIDUE_THRESHOLD, deltas=pre["deltas"])
        solo_ok = is_correct(inst, r_solo)
        c3_ok = is_correct(inst, r_c3)
        del_solo = sorted({s["id"] for s in inst["spans"]} - set(r_solo))
        del_c3 = sorted({s["id"] for s in inst["spans"]} - set(r_c3))
        rec = {
            "id": inst["id"], "family": inst["family"], "budget_ratio": budget_ratio,
            "question": inst["question"],
            "spans": [{"id": s["id"], "text": s["text"], "role": s["role"]} for s in inst["spans"]],
            "solo_deleted": del_solo, "c3_deleted": del_c3,
            "c3_danger_edges": info_c3["danger_edges"],
            "c3_danger_sigma": info_c3["danger_sigma"],
            "hidden_dangerous_pairs": inst["hidden_dangerous_pairs"],
            "solo_answer": derive_answer(inst, r_solo),
            "c3_answer": derive_answer(inst, r_c3),
            "gold_answer": inst["gold_answer"],
            "solo_correct": solo_ok, "c3_correct": c3_ok,
            "c3_compressed_prompt": "\n".join(
                f"[{s['id']}] {s['text']}" for s in inst["spans"] if s["id"] in r_c3),
        }
        if solo_ok is False and c3_ok is True and len(c3_wins) < 10:
            broken = [p for p in inst["hidden_dangerous_pairs"]
                      if p[0] in del_solo and p[1] in del_solo]
            rec["explanation"] = (
                f"solo deleted both endpoints of dangerous pair(s) {broken}; each had "
                f"delta~=0 so solo saw them as free, but jointly they erase a required "
                f"commitment. C3 flagged the danger edge and kept one endpoint.")
            c3_wins.append(rec)
        elif (c3_ok is False) and len(c3_loses) < 10:
            rec["explanation"] = (
                "C3 failed: either a required commitment needed deletion to meet budget "
                "(forced violation) or the failing set is higher-order (>2 spans) which "
                "pairwise sigma cannot detect." if not solo_ok
                else "C3 failed while solo succeeded (rare): budget forced a danger-edge "
                "violation that solo happened to avoid via a different deletion order.")
            c3_loses.append(rec)
    return c3_wins, c3_loses


def fmt_table(rows, methods, budgets, metric, pct=False):
    header = "| method | " + " | ".join(f"r={b}" for b in budgets) + " |"
    sep = "|" + "---|" * (len(budgets) + 1)
    lines = [header, sep]
    for m in methods:
        cells = []
        for b in budgets:
            v = rows[(m, b)][metric]
            cells.append(f"{v*100:.1f}%" if pct else f"{v:.3f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--dataset", default=os.path.join(here, "data", "synthetic_c3.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    data = load_dataset(args.dataset)
    train, dev, test = split_dataset(data)
    print(f"loaded {len(data)} (train={len(train)} dev={len(dev)} test={len(test)})  "
          f"tokenizer={TOKENIZER_NAME}")

    pres_test = [precompute(i) for i in test]
    pres_dev = [precompute(i) for i in dev]

    main_methods = ["full_prompt", "random_delete", "length_greedy",
                    "solo_delta_greedy", "c3_residue_guarded"]
    ablations = ["c3_pairs_100", "c3_pairs_25", "c3_pairs_10",
                 "c3_no_danger", "c3_random_edges"]

    rows = {}
    for m in main_methods + ablations:
        for b in BUDGETS:
            rows[(m, b)] = aggregate(m, test, pres_test, b)
        print(f"  done {m}")

    # success criteria
    bcd = {"B", "C", "D"}
    crit = {}
    for b in (0.35, 0.25):
        c3 = family_acc("c3_residue_guarded", test, pres_test, b, bcd)
        solo = family_acc("solo_delta_greedy", test, pres_test, b, bcd)
        crit[f"BCD@{b}"] = {"c3": c3, "solo": solo, "delta": c3 - solo, "pass": (c3 - solo) >= 0.10}
    # secondary: family A, c3 should not lose >0.03 vs solo (use tightest 0.25)
    a_c3 = family_acc("c3_residue_guarded", test, pres_test, 0.25, {"A"})
    a_solo = family_acc("solo_delta_greedy", test, pres_test, 0.25, {"A"})
    crit["A@0.25"] = {"c3": a_c3, "solo": a_solo, "delta": a_c3 - a_solo,
                      "pass": (a_solo - a_c3) <= 0.03}
    # dev sanity (held-out, NOT used to tune anything)
    dev_crit = {}
    for b in (0.35, 0.25):
        dev_crit[f"BCD@{b}"] = {
            "c3": family_acc("c3_residue_guarded", dev, pres_dev, b, bcd),
            "solo": family_acc("solo_delta_greedy", dev, pres_dev, b, bcd)}

    primary_pass = all(crit[k]["pass"] for k in ("BCD@0.35", "BCD@0.25"))
    secondary_pass = crit["A@0.25"]["pass"]

    c3_wins, c3_loses = collect_failure_cases(test, pres_test, 0.25)

    summary = {
        "config": {"tokenizer": TOKENIZER_NAME, "budgets": BUDGETS, "seeds": SEEDS,
                   "tau": TAU, "residue_threshold": RESIDUE_THRESHOLD,
                   "n_total": len(data), "n_test": len(test), "n_dev": len(dev)},
        "criteria": {"primary_pass": primary_pass, "secondary_pass": secondary_pass,
                     "detail": crit, "dev_sanity": dev_crit},
        "results": {f"{m}@{b}": rows[(m, b)] for m in main_methods + ablations for b in BUDGETS},
        "failure_case_counts": {"c3_wins": len(c3_wins), "c3_loses": len(c3_loses)},
    }
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    with open(os.path.join(here, "results", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(here, "results", "failure_cases.jsonl"), "w") as f:
        for rec in c3_wins + c3_loses:
            f.write(json.dumps(rec) + "\n")

    # markdown
    md = []
    md.append("# C3 experiment results\n")
    md.append(f"- tokenizer: `{TOKENIZER_NAME}`  |  test instances: {len(test)}  |  "
              f"seeds (stochastic): {SEEDS}")
    md.append(f"- tau = {TAU}, residue_threshold = {RESIDUE_THRESHOLD}\n")
    md.append("## Primary criterion: acc(C3) - acc(solo_delta_greedy) >= 0.10 on B/C/D")
    for k in ("BCD@0.35", "BCD@0.25"):
        c = crit[k]
        md.append(f"- **{k}**: C3={c['c3']*100:.1f}%  solo={c['solo']*100:.1f}%  "
                  f"delta={c['delta']*100:+.1f}pts  -> {'PASS' if c['pass'] else 'FAIL'}")
    md.append(f"\n**Primary: {'PASS' if primary_pass else 'FAIL'}**\n")
    md.append("## Secondary: C3 must not lose >0.03 vs solo on Family A")
    c = crit["A@0.25"]
    md.append(f"- **A@0.25**: C3={c['c3']*100:.1f}%  solo={c['solo']*100:.1f}%  "
              f"delta={c['delta']*100:+.1f}pts -> {'PASS' if c['pass'] else 'FAIL'}\n")
    md.append("## Accuracy by method x budget (all families)")
    md.append(fmt_table(rows, main_methods, BUDGETS, "accuracy", pct=True))
    md.append("\n## Compression ratio (avg retained / full tokens)")
    md.append(fmt_table(rows, main_methods, BUDGETS, "compression_ratio"))
    md.append("\n## Dangerous-pair violation rate")
    md.append(fmt_table(rows, main_methods, BUDGETS, "dangerous_pair_violation_rate"))
    md.append("\n## Fragile-group erasure rate")
    md.append(fmt_table(rows, main_methods, BUDGETS, "fragile_group_erasure_rate"))
    md.append("\n## Ablations -- accuracy by budget (all families)")
    md.append(fmt_table(rows, ablations, BUDGETS, "accuracy", pct=True))
    md.append("\n## Ablations -- avg pair tests used")
    md.append(fmt_table(rows, ablations, BUDGETS, "avg_pair_tests"))
    md.append("\n## Per-family accuracy at r=0.25 (main methods)")
    fam_header = "| method | A | B | C | D |\n|---|---|---|---|---|"
    md.append(fam_header)
    for m in main_methods:
        fa = rows[(m, 0.25)]["accuracy_by_family"]
        md.append(f"| {m} | " + " | ".join(f"{fa.get(x,0)*100:.1f}%" for x in "ABCD") + " |")
    md.append(f"\n## Failure cases: {len(c3_wins)} C3-wins, {len(c3_loses)} C3-losses "
              f"written to results/failure_cases.jsonl")
    with open(os.path.join(here, "results", "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print("\n=== SUMMARY ===")
    print(f"PRIMARY (BCD@0.35 & @0.25, delta>=0.10): {'PASS' if primary_pass else 'FAIL'}")
    for k in ("BCD@0.35", "BCD@0.25"):
        c = crit[k]
        print(f"  {k}: C3={c['c3']:.3f} solo={c['solo']:.3f} delta={c['delta']:+.3f}")
    print(f"SECONDARY (A, c3 not >0.03 below solo): {'PASS' if secondary_pass else 'FAIL'} "
          f"(C3={a_c3:.3f} solo={a_solo:.3f})")
    print(f"failure cases: {len(c3_wins)} wins, {len(c3_loses)} losses")
    print("wrote results/summary.json, summary.md, failure_cases.jsonl")


if __name__ == "__main__":
    main()
