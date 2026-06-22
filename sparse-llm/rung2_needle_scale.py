"""sparse-llm — rung 2 STRESS: does "~6% density preserves retrieval" hold as context GROWS?

The 1k-ctx needle result is only suggestive. The real claim is about LONG context, so hold density
constant (budget = density * ctx) and sweep context length. If block's needle accuracy stays ~dense as
ctx grows, the cheap training-free selector genuinely scales; if it decays with length at fixed density,
the claim was a short-context artifact. Dense is printed per length — if the 0.5B model stops retrieving
even dense, the test is inconclusive above that length (need a stronger model / GPU).

Run: HF_HOME=.../hf-cache PYTHONPATH=sparse-llm <.venv>/bin/python sparse-llm/rung2_needle_scale.py
"""
import torch
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

from rung2_transplant import sparse_attention_forward, _CFG
from rung2_needle import build, score, MODEL

DEPTHS = [0.1, 0.5, 0.9]
LENGTHS = [1024, 2048, 4096]
DENSITY = 0.06
MODES = ["exact", "block", "window"]


def acc(model, tok, ctx):
    vals = []
    for dp in DEPTHS:
        ids, ans_start, answer = build(tok, ctx, dp)
        vals.append(score(model, ids, ans_start, answer)[1])
    return sum(vals) / len(vals)


def main():
    AttentionInterface.register("sparse_sel", sparse_attention_forward)
    tok = AutoTokenizer.from_pretrained(MODEL)
    eager = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager",
                                                 dtype=torch.float32).eval()
    sparse = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="sparse_sel",
                                                  dtype=torch.float32).eval()
    print(f"rung-2 STRESS: needle accuracy vs context, density fixed at {DENSITY:.0%} "
          f"(depths {DEPTHS}, {MODEL})", flush=True)
    for ctx in LENGTHS:
        _CFG["budget"] = None
        dense = acc(eager, tok, ctx)
        bud = max(1, round(DENSITY * ctx))
        line = f"ctx={ctx:>5}  DENSE acc={dense:.2f}  | density={DENSITY:.0%} (budget={bud}): "
        cells = []
        for m in MODES:
            _CFG["budget"], _CFG["mode"] = bud, m
            cells.append(f"{m}={acc(sparse, tok, ctx):.2f}")
        flag = "" if dense >= 0.5 else "  <-- dense already fails; inconclusive here"
        print(line + "  ".join(cells) + flag, flush=True)

    print("\nReading: if block tracks DENSE across all lengths -> the ~6% training-free selector scales; "
          "if block decays as ctx grows while dense holds -> the claim was short-context only. window is "
          "the content-blind control (should fail as soon as the needle falls outside the recent budget).",
          flush=True)


if __name__ == "__main__":
    main()
