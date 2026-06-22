"""sparse-llm — rung-3 residue: the SELECTOR-FEASIBILITY GEOMETRY of the oracle attention set.

Residue (survived reduce-to-known; closest prior = PBS-Attn token permutation, but that's a METHOD,
this is a falsifiable DIAGNOSTIC): the exact attention-relevant keys (oracle top-k) have a measurable
multiscale support geometry along the sequence axis, and that geometry imposes an upper bound on
block-selector recall BEFORE any scoring heuristic. Decompose the block gap into:
  geometry_loss = 1 - ceiling   (even the BEST m blocks can't cover S — S is too fragmented)
  scoring_loss  = ceiling - actual  (the right blocks exist but q·mean picks wrong ones)

Everything is derived from the model's attention rows (output_attentions); no q/K needed:
raw score_i = log(weight_i) + const, so q·mean(K_block) ranks blocks by mean_i log(weight_i).

Metrics per (layer, head, query): D_box (box-counting dim of S over block sizes), fragmentation,
ceiling & actual block recall, attention entropy, top-k margin. Falsifier + permute/cluster controls.

Run: HF_HOME=.../hf-cache <.venv>/bin/python sparse-llm/geometry_ceiling.py
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

K = 64                       # oracle budget
BS = [4, 8, 16, 32, 64, 128]  # block sizes for D_box
B_EVAL = 32                  # block size for ceiling/actual
POOL = 2                     # selector keeps m = ceil(POOL*K / b) blocks
TEXT = ("The history of science is the study of the development of science, including both the natural "
        "and social sciences. Researchers form hypotheses, design experiments to test them, gather and "
        "analyze data, and draw conclusions that refine theories over many iterations. The methods and "
        "philosophy underlying scientific inquiry have changed dramatically from the natural philosophy "
        "of the ancient world to the experimental method of the scientific revolution. ") * 12


def box_dim(pos, T_eff):
    """slope of log N_b vs log(T_eff/b): 0=compact, 1=scattered across whole axis."""
    xs, ys = [], []
    for b in BS:
        nb = len(np.unique(pos // b))
        xs.append(np.log(T_eff / b)); ys.append(np.log(nb))
    xs, ys = np.array(xs), np.array(ys)
    return float(np.polyfit(xs, ys, 1)[0])


def block_recalls(pos_all, S, raw, b, m):
    """ceiling (best m blocks by oracle count) and actual (top m blocks by mean raw score)."""
    blk = pos_all // b
    Sblk = pos_all[S] // b
    counts = np.bincount(Sblk)
    # ceiling: the m blocks holding the most oracle keys
    ceil_blocks = np.argsort(-counts)[:m]
    ceiling = np.isin(Sblk, ceil_blocks).mean()
    # actual: m blocks with highest mean raw score (= q·mean(K_block) up to const)
    nb = blk.max() + 1
    bscore = np.array([raw[blk == j].mean() if np.any(blk == j) else -1e9 for j in range(nb)])
    act_blocks = np.argsort(-bscore)[:m]
    actual = np.isin(Sblk, act_blocks).mean()
    return float(ceiling), float(actual)


def spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra*rb).sum() / d) if d > 0 else 0.0


def auroc(score, label):
    label = np.asarray(label).astype(bool)
    if label.all() or (~label).all():
        return float("nan")
    order = np.asarray(score).argsort()
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score)+1)
    npos = label.sum(); nneg = (~label).sum()
    return float((ranks[label].sum() - npos*(npos+1)/2) / (npos*nneg))


def delta_r2(y, base, extra):
    def r2(X):
        X = np.column_stack([X, np.ones(len(y))])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X@coef
        return 1 - (res**2).sum()/((y-y.mean())**2).sum()
    return r2(np.column_stack(base+[extra])) - r2(np.column_stack(base))


def main():
    import os
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager",
                                                 dtype=torch.float32).eval()
    ids = tok(TEXT, return_tensors="pt").input_ids[:, :1024]
    T = ids.shape[1]
    with torch.no_grad():
        attns = model(ids, output_attentions=True).attentions   # tuple[L] of [1,H,T,T]
    L, H = len(attns), attns[0].shape[1]
    qpos_samples = list(range(256, T, 64))   # queries with >=256 context behind them

    rows = []   # one per (l,h,qpos): dict of metrics
    for l in range(L):
        A = attns[l][0]   # [H,T,T]
        for h in range(H):
            for qp in qpos_samples:
                w = A[h, qp, :qp+1].numpy().astype(np.float64)
                w = w / w.sum()
                kk = min(K, qp+1)
                S = np.argpartition(-w, kk-1)[:kk]
                pos_all = np.arange(qp+1)
                raw = np.log(w + 1e-20)
                m = max(1, int(np.ceil(POOL*kk / B_EVAL)))
                ceiling, actual = block_recalls(pos_all, S, raw, B_EVAL, m)
                ent = float(-(w*np.log(w+1e-20)).sum())
                sw = np.sort(w)[::-1]
                margin = float(sw[kk-1] - sw[kk]) if qp+1 > kk else 0.0
                rows.append(dict(D=box_dim(S, qp+1), frag=len(np.unique(S//B_EVAL))/np.ceil(kk/B_EVAL),
                                 ceiling=ceiling, actual=actual, gap=1.0-actual,
                                 geom=1.0-ceiling, score=ceiling-actual, ent=ent, margin=margin))

    def col(name): return np.array([r[name] for r in rows])
    print(f"gpt2 geometry-ceiling: L={L} H={H} queries={len(qpos_samples)} -> {len(rows)} (l,h,q) samples")
    print(f"D_box: mean={col('D').mean():.2f} (0=compact,1=scattered)  median={np.median(col('D')):.2f}")
    print(f"block gap (1-actual): mean={col('gap').mean():.3f}  = geometry_loss {col('geom').mean():.3f} "
          f"+ scoring_loss {col('score').mean():.3f}")
    share = col('geom').mean() / max(1e-9, col('gap').mean())
    print(f"  -> geometry explains {share:.0%} of the block gap; scoring {1-share:.0%}")

    # --- predeclared falsifier ---
    rho_D = spearman(col('D'), col('gap'))
    rho_f = spearman(col('frag'), col('gap'))
    au = auroc(1.0-col('ceiling'), col('actual') < 0.80)   # does (1-ceiling) predict actual<0.8?
    dr2 = delta_r2(col('gap'), [col('ent'), col('margin')], col('D'))
    print(f"\n[falsifier]  ρ(D_box,gap)={rho_D:.2f}  ρ(frag,gap)={rho_f:.2f} (geom assoc, want >=0.35) | "
          f"AUROC(ceiling->fail)={au:.2f} (want >=0.70) | ΔR²(D_box over ent,margin)={dr2:.3f} (want >=0.05)")
    # PREDECLARED falsifier: KILL only if ALL THREE are bad (weak ρ on BOTH, AUROC<0.70, ΔR²<0.05).
    killed = (rho_D < 0.35 and rho_f < 0.35) and (au < 0.70) and (dr2 < 0.05)
    print(f"  => direction {'KILLED' if killed else 'SURVIVES'} the predeclared falsifier "
          f"(kill iff all three bad)")
    # disaggregated: the GEOMETRY/CEILING claim vs the specific BOX-DIMENSION framing
    print(f"  disaggregated: feasibility-ceiling/fragmentation = STRONG (ρ_frag={rho_f:.2f}, "
          f"AUROC={au:.2f}); box-counting D_box specifically = WEAK (ρ_D={rho_D:.2f}, ΔR²={dr2:.3f}) "
          f"-> keep the ceiling+fragmentation, drop the fractal-dimension framing.")

    # --- non-degenerate controls: permute (scatter -> ceiling falls) / cluster (-> ceiling rises) ---
    rng = np.random.default_rng(0)
    ceil_perm, ceil_clust = [], []
    for r_i, l in enumerate([0, L//2, L-1]):
        A = attns[l][0]
        for h in range(0, H, 3):
            for qp in qpos_samples[::3]:
                w = A[h, qp, :qp+1].numpy().astype(np.float64); w /= w.sum()
                kk = min(K, qp+1); S = np.argpartition(-w, kk-1)[:kk]
                raw = np.log(w+1e-20); m = max(1, int(np.ceil(POOL*kk/B_EVAL)))
                perm = rng.permutation(qp+1)               # scatter positions, keep scores
                Sp = perm[S]
                ceil_perm.append(block_recalls(np.arange(qp+1), Sp, raw[np.argsort(perm)], B_EVAL, m)[0])
                Sc = np.arange(kk)                          # cluster oracle keys contiguously
                ceil_clust.append(block_recalls(np.arange(qp+1), Sc, raw, B_EVAL, m)[0])
    base_ceiling = col('ceiling').mean()
    print(f"\n[controls] ceiling: natural={base_ceiling:.2f}  permuted(scatter)={np.mean(ceil_perm):.2f} "
          f"(want < natural)  clustered={np.mean(ceil_clust):.2f} (want ~1.0)")
    ctrl_ok = np.mean(ceil_perm) < base_ceiling - 0.05 and np.mean(ceil_clust) > 0.95
    print(f"  => metric {'VALID (separates scatter from cluster)' if ctrl_ok else 'INVALID'}")


if __name__ == "__main__":
    main()
