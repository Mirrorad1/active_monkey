"""C3 real-LLM probe (follow-up #3) -- RunPod single-GPU, self-contained.

The deterministic experiment showed second-order residue *detection* works when
the loss oracle is noiseless. THIS asks the real question: does the residue
signal survive a real, noisy LLM loss?

We replace the binary oracle with the **NLL of the gold answer** under a small
causal LM (default Qwen2.5-0.5B-Instruct):

    loss(retained) = mean negative log-likelihood of the gold answer tokens
                     given the retained prompt (teacher-forced)

    delta_i  = loss(x without span i) - loss(x)
    sigma_ij = loss(x without i,j) - loss(x without i) - loss(x without j) + loss(x)

Each instance has a single 2-cover commitment: the answer is stated in TWO
places (the hidden dangerous pair) plus distractors that name competing values.
Removing one mention should barely move the NLL (delta ~ 0); removing BOTH
should spike it (sigma > 0) -- IF the model actually uses cross-span redundancy.

Three things, reported honestly:
  Q1a (signal quality): is sigma for the TRUE dangerous pair separated from sigma
     for random safe pairs? (medians + ROC-AUC, test-only AND all-split, with a
     bootstrap CI). If the real model has no measurable residue, C3 cannot work.
  Q1b (signal coverage): does the true pair even REACH the AUC? It only does if
     both endpoints pass the safe gate (delta<=tau) AND survive the pair cap.
     AUC is conditional on tested pairs, so a high AUC over few-but-survivable
     pairs OVERSTATES viability. We report the tested rate + why pairs go missing.
  Q2 (utility): does NLL-thresholded C3 beat solo_delta_greedy on greedy-decode
     answer accuracy under tight budgets? (threshold calibrated on a dev split;
     per-budget delta with a bootstrap CI.)

The verdict requires BOTH quality (test AUC>=0.75) AND coverage (tested>=0.8).

Self-contained: no imports from the parent experiment package, so it runs from
a fresh clone. Standard HF transformers only.

Run:
    python llm_probe.py --n 60 --model Qwen/Qwen2.5-0.5B-Instruct --device cuda
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time

# ---------------------------------------------------------------------------
# Self-contained natural-language dataset (single 2-cover commitment / instance)
# ---------------------------------------------------------------------------

TOPICS = [
    dict(dim="unit", gold="kilograms",
         mentions=["The final answer must be reported in kilograms.",
                   "Use kilograms for the result; pounds are not accepted."],
         distractors=["An old worksheet used pounds.", "Some bins were weighed in grams.",
                      "Freight is sometimes quoted in tonnes."]),
    dict(dim="currency", gold="euros",
         mentions=["All monetary totals must be given in euros.",
                   "Report values in euros; do not use dollars."],
         distractors=["The first quote came in US dollars.", "A vendor listed prices in yen.",
                      "Historical books were in francs."]),
    dict(dim="timezone", gold="UTC",
         mentions=["Every timestamp must be expressed in UTC.",
                   "Use UTC for all times; local time is rejected."],
         distractors=["Some logs were in PST.", "The office runs on CET.",
                      "An app defaulted to IST."]),
    dict(dim="format", gold="JSON",
         mentions=["The output format must be JSON.",
                   "Return the answer as JSON, not plain text."],
         distractors=["A draft was written in YAML.", "Logs are stored as CSV.",
                      "The wiki uses Markdown."]),
    dict(dim="color", gold="blue",
         mentions=["Only blue containers may be exported.",
                   "Export is restricted to blue containers."],
         distractors=["Red containers stay onshore.", "Green bins are for recycling.",
                      "Yellow crates hold tools."]),
]

FILLER = [
    "Shipment {a} left dock {b} on day {c}.",
    "Operator {a} logged {b} minutes on line {c}.",
    "Truck {a} carried {b} crates to bay {c}.",
    "Sensor {a} pinged {b} times near gate {c}.",
    "Team {a} closed {b} tickets in week {c}.",
]


def make_instance(rng, idx):
    topic = rng.choice(TOPICS)
    spans = []
    # two redundant mentions of the commitment (the hidden dangerous pair)
    m1, m2 = topic["mentions"]
    spans.append({"text": m1, "role": "mention"})
    spans.append({"text": m2, "role": "mention"})
    # competing-value distractors + neutral filler
    for d in topic["distractors"]:
        spans.append({"text": d, "role": "distractor"})
    n_filler = rng.randint(4, 8)
    for _ in range(n_filler):
        spans.append({"text": FILLER[rng.randrange(len(FILLER))].format(
            a=rng.randint(1, 99), b=rng.randint(1, 99), c=rng.randint(1, 31)), "role": "filler"})
    rng.shuffle(spans)
    for i, s in enumerate(spans):
        s["id"] = i
    mention_ids = sorted(s["id"] for s in spans if s["role"] == "mention")
    question = (f"In what {topic['dim']} must the final answer be given? "
                f"Reply with a single word.")
    return {
        "id": f"llm_{idx:04d}", "dim": topic["dim"], "gold_answer": topic["gold"],
        "spans": spans, "question": question,
        "hidden_dangerous_pair": mention_ids,  # diagnostics only
    }


def render(spans, retained, question):
    keep = set(retained)
    lines = [f"[{s['id']}] {s['text']}" for s in spans if s["id"] in keep]
    return ("You are given numbered facts. Some constrain the answer.\n"
            + "\n".join(lines) + f"\n\n{question}")


# ---------------------------------------------------------------------------
# LLM loss / generation
# ---------------------------------------------------------------------------

class LM:
    def __init__(self, model_name, device, dtype):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        td = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=td).to(device).eval()
        self.device = device

    def _prefix(self, user_text):
        msgs = [{"role": "user", "content": user_text}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def gold_nll(self, user_text, answer):
        torch = self.torch
        prefix = self._prefix(user_text)
        pids = self.tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids
        full = self.tok(prefix + answer, return_tensors="pt", add_special_tokens=False).input_ids
        full = full.to(self.device)
        labels = full.clone()
        labels[:, : pids.shape[1]] = -100
        with torch.no_grad():
            return float(self.model(full, labels=labels).loss.item())

    def greedy(self, user_text, max_new=6):
        torch = self.torch
        prefix = self._prefix(user_text)
        ids = self.tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().lower()


# ---------------------------------------------------------------------------
# Residue computation + selectors
# ---------------------------------------------------------------------------

def token_count(text):
    return len(text.split())


def residue(lm, inst, tau, max_safe_pairs, rng):
    spans = inst["spans"]; q = inst["question"]; gold = inst["gold_answer"]
    ids = [s["id"] for s in spans]
    base = lm.gold_nll(render(spans, ids, q), gold)
    delta = {}
    for i in ids:
        delta[i] = lm.gold_nll(render(spans, [k for k in ids if k != i], q), gold) - base
    safe = [i for i in ids if delta[i] <= tau]
    pairs = list(itertools.combinations(safe, 2))
    if len(pairs) > max_safe_pairs:
        rng.shuffle(pairs); pairs = pairs[:max_safe_pairs]
    sigma = {}
    for (i, j) in pairs:
        nij = lm.gold_nll(render(spans, [k for k in ids if k not in (i, j)], q), gold)
        sigma[(i, j)] = nij - (delta[i] + base) - (delta[j] + base) + base
    return base, delta, sigma


def select_solo(spans, delta, budget_tokens, q):
    ids = [s["id"] for s in spans]
    full = token_count(render(spans, ids, q))
    order = sorted(ids, key=lambda i: (delta[i], -token_count(spans[i]["text"]), i))
    retained = set(ids)
    for sid in order:
        if token_count(render(spans, retained, q)) <= budget_tokens:
            break
        retained.discard(sid)
    return retained


def select_c3(spans, delta, sigma, threshold, budget_tokens, q):
    ids = [s["id"] for s in spans]
    adj = {i: set() for i in ids}
    for (i, j), s in sigma.items():
        if s >= threshold:
            adj[i].add(j); adj[j].add(i)
    order = sorted(ids, key=lambda i: (delta[i], -token_count(spans[i]["text"]), i))
    retained = set(ids); deleted = set()
    for sid in order:
        if token_count(render(spans, retained, q)) <= budget_tokens:
            break
        if any(n in deleted for n in adj[sid]):
            continue
        retained.discard(sid); deleted.add(sid)
    # forced relax if still over budget
    if token_count(render(spans, retained, q)) > budget_tokens:
        for sid in order:
            if sid in retained:
                retained.discard(sid)
                if token_count(render(spans, retained, q)) <= budget_tokens:
                    break
    return retained


def auc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def collect_split(cache, split, tau):
    """Per-instance sigma records + true-pair COVERAGE accounting for one split.

    A true dangerous pair only reaches the AUC if BOTH its endpoints pass the
    safe gate (delta <= tau) AND it survives the pair-cap subsample. If it does
    not, it is silently absent from the positive class -- so AUC alone overstates
    viability. We track exactly why each true pair is/ isn't tested.

    Returns (recs, cov) where recs[i] = {"true": float|None, "rand": [floats]}
    and cov has counts: n, safe_gated, tested, missing_gate, missing_cap.
    """
    recs = []
    cov = {"n": 0, "safe_gated": 0, "tested": 0, "missing_gate": 0, "missing_cap": 0}
    for inst in split:
        _, delta, sigma = cache[inst["id"]]
        tp = tuple(inst["hidden_dangerous_pair"])  # sorted [a,b]
        both_safe = (delta.get(tp[0], 1e9) <= tau) and (delta.get(tp[1], 1e9) <= tau)
        tested = tp in sigma
        cov["n"] += 1
        cov["safe_gated"] += int(both_safe)
        cov["tested"] += int(tested)
        cov["missing_gate"] += int(not both_safe)
        cov["missing_cap"] += int(both_safe and not tested)
        recs.append({"true": sigma.get(tp), "rand": [v for k, v in sigma.items() if k != tp]})
    return recs, cov


def pooled_auc(recs):
    pos = [r["true"] for r in recs if r["true"] is not None]
    neg = [v for r in recs for v in r["rand"]]
    return auc(pos, neg)


def bootstrap_auc(recs, B, rng):
    """Resample INSTANCES (the correct unit) with replacement; 95% CI of AUC."""
    if not recs:
        return None
    n, ests = len(recs), []
    for _ in range(B):
        sample = [recs[rng.randrange(n)] for _ in range(n)]
        a = pooled_auc(sample)
        if a is not None:
            ests.append(a)
    if not ests:
        return None
    ests.sort()
    return [ests[int(0.025 * len(ests))], ests[int(0.975 * len(ests))]]


def bootstrap_mean_ci(vals, B, rng):
    """95% CI for the mean of a list (e.g. per-instance c3-minus-solo deltas)."""
    if not vals:
        return None
    n, ests = len(vals), []
    for _ in range(B):
        ests.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    ests.sort()
    return [ests[int(0.025 * B)], ests[int(0.975 * B)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tau", type=float, default=0.05, help="NLL delta threshold for 'safe'")
    ap.add_argument("--budgets", default="0.5,0.35,0.25")
    ap.add_argument("--max_safe_pairs", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    budgets = [float(x) for x in args.budgets.split(",")]

    data = [make_instance(rng, i) for i in range(args.n)]
    n_dev = max(8, args.n // 4)
    dev, test = data[:n_dev], data[n_dev:]

    print(f"loading {args.model} on {args.device} ({args.dtype}) ...")
    lm = LM(args.model, args.device, args.dtype)
    t0 = time.time()

    # ---- compute residue for all instances ----
    cache = {}  # id -> (base, delta, sigma)
    for k, inst in enumerate(data):
        base, delta, sigma = residue(lm, inst, args.tau, args.max_safe_pairs, rng)
        cache[inst["id"]] = (base, delta, sigma)
        if (k + 1) % 10 == 0:
            print(f"  residue {k+1}/{len(data)}  ({time.time()-t0:.0f}s)")

    # ---- per-split sigma records + true-pair COVERAGE (honest reachability) ----
    all_recs, cov = collect_split(cache, data, args.tau)
    test_recs, _ = collect_split(cache, test, args.tau)
    dev_recs, _ = collect_split(cache, dev, args.tau)

    all_pos = [r["true"] for r in all_recs if r["true"] is not None]
    all_neg = [v for r in all_recs for v in r["rand"]]
    med = lambda xs: sorted(xs)[len(xs)//2] if xs else 0.0

    # threshold calibrated on DEV ONLY (midpoint between dev medians)
    dev_true = [r["true"] for r in dev_recs if r["true"] is not None]
    dev_rand = [v for r in dev_recs for v in r["rand"]]
    threshold = (med(dev_true) + med(dev_rand)) / 2 if dev_true else 0.5

    all_split_auc = pooled_auc(all_recs)
    test_only_auc = pooled_auc(test_recs)
    boot_rng = random.Random(args.seed + 1)
    test_auc_ci = bootstrap_auc(test_recs, 2000, boot_rng)

    tested_rate = cov["tested"] / cov["n"] if cov["n"] else 0.0
    gate_rate = cov["safe_gated"] / cov["n"] if cov["n"] else 0.0

    # ---- Q2: utility (C3 vs solo on test, greedy accuracy) ----
    def correct(pred, gold):
        return gold.lower() in pred.lower()

    util = {}
    for b in budgets:
        c3_acc, solo_acc, deltas_pi = [], [], []
        for inst in test:
            spans, q, gold = inst["spans"], inst["question"], inst["gold_answer"]
            base, delta, sigma = cache[inst["id"]]
            full_tok = token_count(render(spans, [s["id"] for s in spans], q))
            bt = max(1, int(round(b * full_tok)))
            r_solo = select_solo(spans, delta, bt, q)
            r_c3 = select_c3(spans, delta, sigma, threshold, bt, q)
            sc = 1.0 if correct(lm.greedy(render(spans, r_solo, q)), gold) else 0.0
            cc = 1.0 if correct(lm.greedy(render(spans, r_c3, q)), gold) else 0.0
            solo_acc.append(sc); c3_acc.append(cc); deltas_pi.append(cc - sc)
        util[b] = {"c3": sum(c3_acc)/len(c3_acc), "solo": sum(solo_acc)/len(solo_acc),
                   "delta": sum(deltas_pi)/len(deltas_pi),
                   "delta_ci95": bootstrap_mean_ci(deltas_pi, 2000, random.Random(args.seed + 2))}

    # ---- verdict: require BOTH signal quality AND signal coverage ----
    auc_ok = (test_only_auc or 0.0) >= 0.75
    cov_ok = tested_rate >= 0.8
    if auc_ok and cov_ok:
        verdict = ("residue DETECTABLE and reliably reachable in real model "
                   "(test AUC >= 0.75 AND true-pair tested rate >= 0.8)")
    else:
        why = []
        if not auc_ok:
            why.append(f"signal WEAK (test AUC {test_only_auc} < 0.75)")
        if not cov_ok:
            why.append(f"signal NOT RELIABLY REACHABLE (true-pair tested rate "
                       f"{tested_rate:.2f} < 0.8; {cov['missing_gate']} gated out, "
                       f"{cov['missing_cap']} dropped by pair cap)")
        verdict = "C3 premise NOT supported here -- " + "; ".join(why)

    summary = {
        "model": args.model, "n": args.n, "n_test": len(test), "n_dev": len(dev),
        "tau": args.tau, "max_safe_pairs": args.max_safe_pairs,
        "runtime_s": time.time() - t0,
        "detectability": {
            "true_pair_sigma_median": med(all_pos),
            "random_pair_sigma_median": med(all_neg),
            "true_pair_sigma_mean": sum(all_pos)/len(all_pos) if all_pos else None,
            "random_pair_sigma_mean": sum(all_neg)/len(all_neg) if all_neg else None,
            # AUC: report BOTH splits explicitly; never AUC alone
            "all_split_roc_auc_true_vs_random": all_split_auc,
            "test_only_roc_auc_true_vs_random": test_only_auc,
            "test_only_roc_auc_ci95": test_auc_ci,
            # coverage: how reliably the true pair even reaches the AUC
            "true_pair_safe_gate_rate": gate_rate,
            "true_pair_tested_rate": tested_rate,
            "true_pair_missing_due_to_gate": cov["missing_gate"],
            "true_pair_missing_due_to_pair_cap": cov["missing_cap"],
            "n_instances_for_coverage": cov["n"],
            "n_true_tested": len(all_pos), "n_random": len(all_neg),
            "calibrated_threshold": threshold,
        },
        "utility_c3_vs_solo": util,
        "verdict_hint": verdict,
    }
    with open(os.path.join(args.out, "llm_probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== C3 LLM PROBE ===")
    print(f"model={args.model}  n_test={len(test)}  runtime={summary['runtime_s']:.0f}s")
    d = summary["detectability"]
    print(f"Q1a detectability: sigma median true={d['true_pair_sigma_median']:.3f} "
          f"vs random={d['random_pair_sigma_median']:.3f}")
    print(f"     AUC test-only={test_only_auc} CI95={test_auc_ci}  all-split={all_split_auc}")
    print(f"Q1b coverage: true-pair tested rate={tested_rate:.2f} "
          f"(safe-gated={gate_rate:.2f}; missing: gate={cov['missing_gate']}, "
          f"cap={cov['missing_cap']} of {cov['n']})")
    print(f"     calibrated threshold (dev)={threshold:.3f}")
    print("Q2 utility (greedy accuracy):")
    for b in budgets:
        u = util[b]
        print(f"   budget {b}: C3={u['c3']:.3f}  solo={u['solo']:.3f}  "
              f"delta={u['delta']:+.3f} CI95={u['delta_ci95']}")
    print(f"VERDICT HINT: {verdict}")
    print("wrote results/llm_probe_summary.json")


if __name__ == "__main__":
    main()
