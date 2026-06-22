"""sparse-llm — the SELECTION vs ABSTRACTION boundary (run existing methods on the hard case).

Compaction is broader than selection: you can KEEP a subset of tokens (selection/eviction) OR
RECOMBINE them into new shorter tokens (abstraction/summary). The geometry-ceiling showed selection is
structurally capped when relevant tokens are fragmented. The honest question (per "try existing methods,
don't argue from literature"): on the cases where selection is capped, does an EXISTING abstraction
method (model summary) recover what selection can't — and where does it LOSE?

Query-AGNOSTIC compaction (compress before knowing the question), then retrieval. Two regimes:
  density : many scattered facts, budget too small to keep them raw -> abstraction should win (recombines).
  verbatim: one exact string -> selection should win (summary corrupts exact tokens) = non-degenerate control.

Arms at matched budget k: dense (ref) / selection (keep top-k tokens by received attention, H2O-style,
query-agnostic) / abstraction (model summarizes context into <=k tokens). Metric: retrieval accuracy.

Run: HF_HOME=.../hf-cache PYTHONPATH=sparse-llm <.venv>/bin/python sparse-llm/boundary_test.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
K = 48                         # compaction budget (tokens kept)
FILLER = ("Routine maintenance logs were filed in the archive without incident. "
          "Staff rotated shifts and recorded nothing of note that day. ") * 6


@torch.no_grad()
def gen(model, tok, prompt, max_new=48):
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def ask(model, tok, context, question):
    p = (f"<|im_start|>system\nAnswer using only the context.<|im_end|>\n"
         f"<|im_start|>user\nContext:\n{context}\n\n{question}<|im_end|>\n<|im_start|>assistant\n")
    return gen(model, tok, p, max_new=24)


def select_topk(model, tok, context, k):
    """Query-agnostic selection: keep the k context tokens that RECEIVE the most attention."""
    ids = tok(context, return_tensors="pt").input_ids
    with torch.no_grad():
        att = model(ids, output_attentions=True).attentions   # tuple[L][1,H,T,T]
    recv = torch.stack([a[0].sum(0).sum(0) for a in att]).sum(0)  # [T] attention received
    T = ids.shape[1]
    keep = torch.sort(torch.topk(recv, min(k, T)).indices).values
    return tok.decode(ids[0, keep], skip_special_tokens=True)


def summarize(model, tok, context, k):
    p = (f"<|im_start|>system\nYou extract facts. Output EVERY access code and the master key you find, "
         f"one per line as 'vault N: CODE' or 'master key: CODE', verbatim. Output nothing else.<|im_end|>\n"
         f"<|im_start|>user\n{context}<|im_end|>\n<|im_start|>assistant\n")
    return gen(model, tok, p, max_new=k)


def density_task(n=6):
    codes = {f"{i}": f"{(i*13717 % 90000)+10000}" for i in range(1, n+1)}
    parts = [FILLER]
    for v, c in codes.items():
        parts.append(f" The access code for vault {v} is {c}. ")
        parts.append(FILLER)
    return "".join(parts), codes


def verbatim_task():
    code = "8F3K-99X2-QP7M-2210"
    ctx = FILLER + f" The master key is {code}. " + FILLER
    return ctx, code


def acc_density(model, tok, context, codes):
    hits = 0
    for v, c in codes.items():
        a = ask(model, tok, context, f"What is the access code for vault {v}? Reply with the number only.")
        hits += int(c in a)
    return hits / len(codes)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager",
                                                 dtype=torch.float32).eval()
    print(f"boundary test: {MODEL}, budget k={K}", flush=True)

    # ---------- DENSITY regime ----------
    ctx, codes = density_task(6)
    full_tok = tok(ctx, return_tensors="pt").input_ids.shape[1]
    sel = select_topk(model, tok, ctx, K)
    summ = summarize(model, tok, ctx, K)
    print(f"\n[density] {len(codes)} scattered facts, context={full_tok} tok -> budget {K}", flush=True)
    print(f"  dense       acc={acc_density(model, tok, ctx, codes):.2f}", flush=True)
    print(f"  selection   acc={acc_density(model, tok, sel, codes):.2f}  (kept {len(tok(sel).input_ids)} tok)", flush=True)
    print(f"  abstraction acc={acc_density(model, tok, summ, codes):.2f}  (summary {len(tok(summ).input_ids)} tok)", flush=True)
    print(f"  summary text: {summ[:160]!r}", flush=True)

    # ---------- VERBATIM regime (control: selection should win) ----------
    ctx2, code2 = verbatim_task()
    sel2 = select_topk(model, tok, ctx2, K)
    summ2 = summarize(model, tok, ctx2, K)
    q = "What is the master key? Reply with the key only."
    print(f"\n[verbatim] one exact string {code2}", flush=True)
    print(f"  dense       hit={int(code2 in ask(model, tok, ctx2, q))}", flush=True)
    print(f"  selection   hit={int(code2 in ask(model, tok, sel2, q))}", flush=True)
    print(f"  abstraction hit={int(code2 in ask(model, tok, summ2, q))}  (summary {summ2[:80]!r})", flush=True)

    print("\nReading: density -> abstraction > selection means recombination recovers what subset-keeping "
          "can't at this budget. verbatim -> selection > abstraction confirms abstraction loses exact "
          "tokens (not universally better). Where BOTH lose = the measured residual (the real opening).", flush=True)


if __name__ == "__main__":
    main()
