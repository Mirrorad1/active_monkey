"""sparse-llm — rung 2: training-free selector transplant into a FROZEN real model (GPT-2).

We register a custom attention that mirrors transformers' eager path EXACTLY, then injects a
per-(head,query) key SELECTION before softmax. Frozen weights, reused Q/K/V/O — only *which keys
each query attends to* changes. Cheapest real model first (GPT-2 small, CPU, seconds); scale to a
long-context model only if this rung proves insufficient.

CORRECTNESS GATE (two-sided, runs before any result):
  (1) budget=None (no selection) must equal the stock eager model to float precision
      -> our attention replication is exact.
  (2) a tight budget must DIFFER from eager -> the selection hook is actually live.
If either fails, every sparse number below is meaningless and we abort.

PROBE: mean KL( dense || sparse ) of the next-token distribution over a fixed text, per selector
per budget. This measures how much a selection rule perturbs the model's own computation without
needing GPT-2 to be good at a task. Expectation: exact (keep the model's own top-k) << window
(recent-k, content-blind) < random; and KL -> 0 as budget -> full.

Run: /Users/mirro/Projects/active-loop/sparse-llm/.venv/bin/python sparse-llm/rung2_transplant.py
"""
import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

# selection config read by the custom attention at every layer/forward
_CFG = {"budget": None, "mode": "exact", "block": 16, "pool_factor": 2}

TEXT = (
    "The history of science is the study of the development of science, including both the "
    "natural and social sciences. Science is a body of empirical, theoretical, and practical "
    "knowledge about the natural world, produced by researchers making use of observation and "
    "experiment. Modern science is typically divided into three major branches: the natural "
    "sciences, which study nature in the broadest sense; the social sciences, which study "
    "individuals and societies; and the formal sciences, which study logic and mathematics. "
    "Throughout history, the methods and philosophy underlying scientific inquiry have changed "
    "dramatically, from the natural philosophy of the ancient world to the experimental method "
    "that emerged during the seventeenth century scientific revolution. Researchers form "
    "hypotheses, design experiments to test them, gather and analyze data, and draw conclusions "
    "that either support or contradict the original idea, refining theories over many iterations."
) * 3  # ~ a few hundred tokens, enough to make budget bite


def _apply_selection(attn, query, key, budget, mode, block, pool_factor):
    """Set non-selected key logits to -inf per (batch, head, query). attn already includes the
    additive causal mask (disallowed positions are very negative)."""
    b, h, q, k = attn.shape
    if k <= budget:
        return attn
    neg = torch.finfo(attn.dtype).min
    if mode == "exact":
        kth = torch.topk(attn, budget, dim=-1).values[..., -1:]      # keep model's own top-budget
        return attn.masked_fill(attn < kth, neg)
    if mode == "window":
        qp = torch.arange(q, device=attn.device).view(1, 1, q, 1)
        kp = torch.arange(k, device=attn.device).view(1, 1, 1, k)
        keep = (kp <= qp) & (kp > qp - budget)                       # most-recent budget keys
        return attn.masked_fill(~keep, neg)
    if mode == "random":
        # per (b,h,q) keep `budget` of the allowed (kp<=qp) keys at random (seeded by position)
        qp = torch.arange(q, device=attn.device).view(1, 1, q, 1)
        kp = torch.arange(k, device=attn.device).view(1, 1, 1, k)
        allowed = (kp <= qp)
        g = torch.Generator(device=attn.device).manual_seed(0)
        score = torch.rand(attn.shape, generator=g, device=attn.device)
        score = score.masked_fill(~allowed, -1.0)
        kth = torch.topk(score, budget, dim=-1).values[..., -1:]
        return attn.masked_fill(score < kth, neg)
    if mode == "block":
        # coarse-to-fine meta-selector on the REAL model: pool keys into blocks, pick top blocks
        # by q·mean(block) as a candidate pool, then exact top-budget within it.
        B = block
        nb = (k + B - 1) // B
        Kp = F.pad(key, (0, 0, 0, nb * B - k))                      # [b,h,nb*B,d]
        summ = Kp.view(b, h, nb, B, -1).mean(3)                     # [b,h,nb,d] block means
        bscore = torch.matmul(query, summ.transpose(-1, -2))       # [b,h,q,nb]
        qp = torch.arange(q, device=attn.device).view(1, 1, q, 1)
        bfirst = (torch.arange(nb, device=attn.device) * B).view(1, 1, 1, nb)
        bscore = bscore.masked_fill(bfirst > qp, neg)              # drop fully-future blocks
        m = min(nb, max(1, (pool_factor * budget + B - 1) // B))
        sel = torch.topk(bscore, m, dim=-1).indices               # top candidate blocks
        block_sel = torch.zeros(b, h, q, nb, dtype=torch.bool, device=attn.device)
        block_sel.scatter_(-1, sel, True)
        cand = block_sel.repeat_interleave(B, dim=-1)[..., :k]     # candidate keys
        masked = attn.masked_fill(~cand, neg)
        kth = torch.topk(masked, min(budget, k), dim=-1).values[..., -1:]
        return masked.masked_fill(masked < kth, neg)               # fine top-budget within pool
    raise ValueError(mode)


def sparse_attention_forward(module, query, key, value, attention_mask, scaling=None,
                             dropout=0.0, **kwargs):
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    q_len, k_len = attn_weights.shape[-2], attn_weights.shape[-1]
    neg = torch.finfo(attn_weights.dtype).min
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[..., :k_len]
    else:
        # transformers 5.x hands a custom attn impl no framework mask -> enforce causality here.
        causal = torch.triu(torch.full((q_len, k_len), neg, dtype=attn_weights.dtype,
                                       device=attn_weights.device), diagonal=1 + (k_len - q_len))
        attn_weights = attn_weights + causal
    if _CFG["budget"] is not None:
        attn_weights = _apply_selection(attn_weights, query, key, _CFG["budget"], _CFG["mode"],
                                        _CFG["block"], _CFG["pool_factor"])
    attn_weights = F.softmax(attn_weights, dim=-1).type(value.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value).transpose(1, 2)
    return attn_output, attn_weights


def load(tok_name="gpt2"):
    AttentionInterface.register("sparse_sel", sparse_attention_forward)
    tok = AutoTokenizer.from_pretrained(tok_name)
    eager = AutoModelForCausalLM.from_pretrained(tok_name, attn_implementation="eager",
                                                 dtype=torch.float32).eval()
    sparse = AutoModelForCausalLM.from_pretrained(tok_name, attn_implementation="sparse_sel",
                                                  dtype=torch.float32).eval()
    return tok, eager, sparse


@torch.no_grad()
def logits_for(model, ids):
    return model(ids).logits


def main():
    tok, eager, sparse = load()
    ids = tok(TEXT, return_tensors="pt").input_ids[:, :512]
    n = ids.shape[1]
    print(f"gpt2 transplant — seq len {n}")

    dense = logits_for(eager, ids)

    # --- correctness gate ---
    _CFG["budget"] = None
    repl = logits_for(sparse, ids)
    d_full = (dense - repl).abs().max().item()
    _CFG["budget"], _CFG["mode"] = 8, "exact"
    tight = logits_for(sparse, ids)
    d_tight = (dense - tight).abs().max().item()
    gate = (d_full < 1e-4) and (d_tight > 1e-2)
    print(f"[gate] budget=None vs eager max|Δ|={d_full:.2e} (want <1e-4); "
          f"budget=8 vs eager max|Δ|={d_tight:.2e} (want >1e-2) => "
          f"{'PASS' if gate else 'FAIL'}")
    if not gate:
        print("ABORT: transplant not validated.")
        return

    # --- probe: mean KL(dense || sparse) of next-token dist, per selector per budget ---
    logp_dense = F.log_softmax(dense, dim=-1)
    p_dense = logp_dense.exp()

    def mean_kl(sp_logits):
        logp = F.log_softmax(sp_logits, dim=-1)
        return (p_dense * (logp_dense - logp)).sum(-1).mean().item()  # KL(dense||sparse), nats

    budgets = [256, 128, 64, 32]
    modes = ["exact", "block", "window", "random"]
    print(f"\nmean KL(dense || sparse) over {n} positions, lower=better preserved:")
    print(f"{'budget':>7} " + " ".join(f"{m:>9}" for m in modes))
    for bud in budgets:
        row = []
        for m in modes:
            _CFG["budget"], _CFG["mode"] = bud, m
            row.append(mean_kl(logits_for(sparse, ids)))
        print(f"{bud:>7} " + " ".join(f"{v:>9.4f}" for v in row))

    print("\nReading: exact = keep the model's OWN top-k (best any selector can do at budget). If "
          "exact KL stays ~0 while window/random blow up, CONTENT selection is what matters and a "
          "cheap content-selector is worth chasing; if even exact KL is large at a budget, that "
          "budget is too small for the model regardless of selector.")


if __name__ == "__main__":
    main()
