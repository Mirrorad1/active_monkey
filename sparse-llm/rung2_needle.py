"""sparse-llm — rung 2 (real retrieval): needle-in-a-haystack on a frozen long-context model.

Does training-free key selection preserve actual LONG-RANGE RETRIEVAL (not just KL on short text)?
Plant a verbatim fact (the "needle") at a controlled depth in a long filler context, then teacher-force
the answer and measure, per selector/budget: the model's mean log-prob of the correct answer tokens and
greedy token accuracy. Dense is the reference (must retrieve, else there's nothing to preserve).

The honest test: at the answer position the query must attend BACK to the needle's keys. A content-blind
window drops an early needle (retrieval fails); exact/block should keep it. Reuses the rung-2 transplant
(GQA-safe) + its correctness gate.

Run: HF_HOME=.../hf-cache <sparse-llm/.venv>/bin/python sparse-llm/rung2_needle.py
"""
import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

from rung2_transplant import sparse_attention_forward, _CFG

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
NEEDLE = "The special access code for the vault is 73912."
QUESTION = "What is the special access code for the vault?"
ANSWER = " 73912"
FILLER = (
    "The history of science is the study of the development of science. Science is a body of empirical, "
    "theoretical, and practical knowledge about the natural world. Researchers form hypotheses, design "
    "experiments to test them, gather and analyze data, and draw conclusions that refine theories over "
    "many iterations. Modern science is divided into the natural sciences, the social sciences, and the "
    "formal sciences. The methods underlying scientific inquiry have changed dramatically over time. "
) * 40


def build(tok, ctx_tokens, depth):
    fill = tok(FILLER, return_tensors="pt").input_ids[0][:ctx_tokens]
    needle = tok(NEEDLE, return_tensors="pt").input_ids[0]
    suffix = tok(f"\n\nQuestion: {QUESTION}\nAnswer:", return_tensors="pt").input_ids[0]
    answer = tok(ANSWER, return_tensors="pt").input_ids[0]
    cut = int(depth * len(fill))
    ids = torch.cat([fill[:cut], needle, fill[cut:], suffix, answer]).unsqueeze(0)
    return ids, ids.shape[1] - len(answer), answer


@torch.no_grad()
def score(model, ids, ans_start, answer):
    logits = model(ids).logits[0]
    lp, correct = [], []
    for i, tid in enumerate(answer):
        pos = ans_start - 1 + i
        lp.append(F.log_softmax(logits[pos], -1)[tid].item())
        correct.append(int(logits[pos].argmax().item() == tid.item()))
    return sum(lp) / len(lp), sum(correct) / len(correct)


def main():
    AttentionInterface.register("sparse_sel", sparse_attention_forward)
    tok = AutoTokenizer.from_pretrained(MODEL)
    eager = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager",
                                                 dtype=torch.float32).eval()
    sparse = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="sparse_sel",
                                                  dtype=torch.float32).eval()
    ids, ans_start, answer = build(tok, 1024, 0.5)
    print(f"{MODEL}  seq={ids.shape[1]}  answer_tokens={len(answer)}")

    # --- correctness gate ---
    _CFG["budget"] = None
    d_full = (eager(ids).logits - sparse(ids).logits).abs().max().item()
    _CFG["budget"], _CFG["mode"] = 8, "exact"
    d_tight = (eager(ids).logits - sparse(ids).logits).abs().max().item()
    gate = (d_full < 1e-3) and (d_tight > 1e-2)
    print(f"[gate] None vs eager max|Δ|={d_full:.2e} (<1e-3); budget=8 vs eager={d_tight:.2e} (>1e-2) "
          f"=> {'PASS' if gate else 'FAIL'}")
    if not gate:
        print("ABORT: transplant not validated on this model.")
        return

    depths = [0.1, 0.5, 0.9]
    budgets = [256, 128, 64, 32]
    modes = ["exact", "block", "window", "random"]

    # dense reference (averaged over depths)
    _CFG["budget"] = None
    dn = [score(eager, *build(tok, 1024, dp)) for dp in depths]
    dlp = sum(x[0] for x in dn) / len(dn)
    dac = sum(x[1] for x in dn) / len(dn)
    print(f"\nneedle retrieval (ctx~1024, depths {depths}); DENSE: answer_logprob={dlp:.3f} "
          f"greedy_acc={dac:.2f}")
    print(f"{'budget':>7} " + " ".join(f"{m:>20}" for m in modes))
    for bud in budgets:
        cells = []
        for m in modes:
            _CFG["budget"], _CFG["mode"] = bud, m
            res = [score(sparse, *build(tok, 1024, dp)) for dp in depths]
            lp = sum(x[0] for x in res) / len(res)
            ac = sum(x[1] for x in res) / len(res)
            cells.append(f"lp={lp:>6.2f} acc={ac:.2f}")
        print(f"{bud:>7} " + " ".join(f"{c:>20}" for c in cells))

    print("\nReading: dense should retrieve (acc~1). A selector preserves retrieval if its acc stays ~1 "
          "and answer_logprob ~ dense. window should fail at low budget when the needle is early (depth "
          "0.1) — the content-blind control. The exact/block vs window gap is the long-range payoff.")


if __name__ == "__main__":
    main()
