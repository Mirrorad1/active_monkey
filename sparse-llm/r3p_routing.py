"""sparse-llm — R3': per-context paradigm-feasibility routing (extract vs abstract).

Claim: a context's oracle-relevance GEOMETRY predicts which compaction paradigm is feasible.
4-corner grid with the non-degenerate controls and the residual cell baked in:

                 templated/compressible        verbatim/incompressible
  clustered      both easy                      EXTRACT should win (control)
  scattered      ABSTRACT should win (boundary) BOTH should struggle (the residual = real opening)

Query-AGNOSTIC compaction to a TIGHT budget (density*ctx), then retrieval. The LOCAL run was void
(short ctx + generous budget -> control didn't fire); this needs LONG context + TIGHT budget + a fair
extractive baseline -> GPU. Arms:
  extractive  : keep top SENTENCES by received attention (coherent, fair)
  abstractive : model summary (<= budget tokens)

Env: SPARSE_MODEL (Qwen/Qwen2.5-7B-Instruct), SPARSE_LENGTHS ("2048 4096"), SPARSE_DENSITY (0.04),
     SPARSE_NFACTS (10). NB output_attentions memory ~ L*ctx^2 -> keep ctx<=4096 for a 7B on one 80GB GPU.
Run: see runpod/sparse_llm_r3p.sh
"""
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("SPARSE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LENGTHS = [int(x) for x in os.environ.get("SPARSE_LENGTHS", "2048 4096").split()]
DENSITY = float(os.environ.get("SPARSE_DENSITY", "0.04"))
N = int(os.environ.get("SPARSE_NFACTS", "10"))
FILLER = ("Routine maintenance logs were filed in the archive without incident. "
          "Staff rotated shifts and recorded nothing of note that day. ")


def device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def make_facts(verbatim):
    import hashlib
    facts = []
    for i in range(1, N + 1):
        if verbatim:
            h = hashlib.sha1(f"vault{i}".encode()).hexdigest().upper()
            v = f"{h[:4]}-{h[4:8]}-{h[8:12]}"
        else:
            v = f"{(i * 13717 % 90000) + 10000}"
        facts.append((i, v))
    return facts, ("key" if verbatim else "code")


def build(tok, verbatim, scattered, target):
    facts, kind = make_facts(verbatim)
    sents = [f"The access {kind} for vault {v} is {c}." for v, c in facts]
    fids = tok(FILLER * 400, add_special_tokens=False).input_ids
    if scattered:
        per = max(8, target // (N + 1))
        parts = []
        for i in range(N + 1):
            parts.append(tok.decode(fids[:per]))
            if i < N:
                parts.append(" " + sents[i] + " ")
        ctx = "".join(parts)
    else:
        half = target // 2
        ctx = tok.decode(fids[:half]) + " " + " ".join(sents) + " " + tok.decode(fids[:half])
    return ctx, facts, kind


@torch.no_grad()
def gen(model, tok, prompt, device, max_new=32):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def ask(model, tok, context, kind, v, device):
    p = (f"<|im_start|>system\nAnswer using only the context, value only.<|im_end|>\n<|im_start|>user\n"
         f"Context:\n{context}\n\nWhat is the access {kind} for vault {v}?<|im_end|>\n<|im_start|>assistant\n")
    return gen(model, tok, p, device, 24)


def acc(model, tok, context, facts, kind, device):
    return sum(int(str(c) in ask(model, tok, context, kind, v, device)) for v, c in facts) / len(facts)


def extract_sentences(model, tok, context, k, device):
    sents = [s for s in re.split(r'(?<=[.!?])\s+', context) if s.strip()]
    ids = tok(context, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        att = model(ids, output_attentions=True).attentions
    recv = torch.stack([a[0].float().sum(0).sum(0) for a in att]).sum(0).cpu()
    del att
    sent_tok = [tok(s, add_special_tokens=False).input_ids for s in sents]
    scores, idx = [], 0
    for st in sent_tok:
        L = len(st)
        scores.append(float(recv[idx:idx + L].sum()) if idx + L <= len(recv) else 0.0)
        idx += L
    order = sorted(range(len(sents)), key=lambda i: -scores[i])
    kept, used = set(), 0
    for i in order:
        used += len(sent_tok[i]); kept.add(i)
        if used >= k:
            break
    return " ".join(sents[i] for i in sorted(kept))


def summarize(model, tok, context, k, kind, device):
    p = (f"<|im_start|>system\nOutput EVERY '{kind}' you find, one per line as 'vault N: VALUE', verbatim. "
         f"Nothing else.<|im_end|>\n<|im_start|>user\n{context}<|im_end|>\n<|im_start|>assistant\n")
    return gen(model, tok, p, device, k)


def main():
    device, dtype = device_dtype()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager",
                                                 dtype=dtype).to(device).eval()
    print(f"R3' routing: {MODEL} on {device}/{dtype}, density={DENSITY:.0%}, {N} facts, lengths={LENGTHS}",
          flush=True)
    for ctx_len in LENGTHS:
        k = max(8, round(DENSITY * ctx_len))
        print(f"\n=== ctx~{ctx_len} budget k={k} ({DENSITY:.0%}) ===", flush=True)
        print(f"{'corner':<20} {'dense':>6} {'extract':>8} {'abstract':>9}  pred / emp", flush=True)
        hits = 0
        for verbatim in (False, True):
            for scattered in (False, True):
                ctx, facts, kind = build(tok, verbatim, scattered, ctx_len)
                nt = tok(ctx, return_tensors="pt").input_ids.shape[1]
                d = acc(model, tok, ctx, facts, kind, device)
                ex = acc(model, tok, extract_sentences(model, tok, ctx, k, device), facts, kind, device)
                ab = acc(model, tok, summarize(model, tok, ctx, k, kind, device), facts, kind, device)
                pred = "abstract" if (scattered and not verbatim) else "extract"
                emp = "extract" if ex > ab + 1e-9 else "abstract" if ab > ex + 1e-9 else "tie"
                hits += int(pred == emp)
                name = f"{'scat' if scattered else 'clust'}+{'verb' if verbatim else 'templ'}"
                print(f"{name:<20} {d:>6.2f} {ex:>8.2f} {ab:>9.2f}  {pred:<8}/ {emp}  (ctx {nt})", flush=True)
        print(f"  routing predicted==empirical: {hits}/4 corners", flush=True)
    print("\nValidity: the control (clust+verb / verbatim -> extract) MUST fire, else void. Residual = "
          "scat+verb both-low. Routing holds if pred==emp across corners at TIGHT budget.", flush=True)


if __name__ == "__main__":
    main()
