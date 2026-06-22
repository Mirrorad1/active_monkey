"""sparse-llm — rung 1: the cheap, posable RECALL GATE (seconds on CPU, no model, no data).

QUESTION: does a cheap selector recover the keys EXACT attention would attend to, across
different relevance GEOMETRIES (local / far / scattered)? In particular — your idea — does the
coarse-to-fine block META-selector preserve scattered/far relevance, or does block pooling
dilute it away?

METHOD: plant a known relevant set R (the keys a task REQUIRES) at controlled distances from
the query, and build key vectors so the planted keys have the highest q·k (so exact_topk
recovers R ~perfectly = a VALID ceiling). Then measure recall@budget = |selected ∩ R| / |R|
per selector per geometry. SNR (how far the planted q·k sticks above the distractor noise) is
the knob: at high SNR everything is easy; the science is where selectors start to FAIL.

PREDECLARED bench-validity control (NON-DEGENERATE — the instrument must be able to detect
failure, else the comparison is void): exact_topk recall ≈ 1.0 on every geometry AND window
recall ≈ 0 on 'far'. If that fails -> ABORT, do not report selector numbers.

PREDECLARED research falsifier: sweep SNR on scattered ('multi') relevance. If block_topk
recall tracks exact_topk within TOL across the whole sweep, coarse-to-fine is ~free (the
abstraction costs nothing here). If block_topk drops BELOW exact_topk while exact still
recovers R, block pooling has a real, measurable cost on scattered relevance -- and the SNR at
which the gap opens is the phenomenon.

Run: PYTHONPATH=sparse-llm uv run --python .venv python sparse-llm/synthetic_recall.py
"""
import numpy as np

from selection import SELECTORS

TOL = 0.05


def make_task(geometry, n, d, r, alpha, rng):
    """Return (q, K, qpos, R). Planted keys R get +alpha*q so their q·k stands above noise."""
    q = rng.standard_normal(d)
    q /= np.linalg.norm(q)
    K = rng.standard_normal((n, d)) / np.sqrt(d)
    qpos = n - 1
    if geometry == "local":
        pool = np.arange(qpos - 32, qpos)        # relevance is recent
    elif geometry == "far":
        pool = np.arange(0, n // 4)              # relevance is far back
    elif geometry == "multi":
        pool = np.arange(0, qpos)                # relevance scattered anywhere before q
    else:
        raise ValueError(geometry)
    R = rng.choice(pool, size=r, replace=False)
    K[R] += alpha * q
    return q, K, qpos, np.sort(R)


def recall(sel_idx, R):
    return len(np.intersect1d(sel_idx, R)) / len(R)


def mean_recall(geometry, alpha, *, n, d, r, budget, trials, rng):
    acc = {s: [] for s in SELECTORS}
    for _ in range(trials):
        q, K, qpos, R = make_task(geometry, n, d, r, alpha, rng)
        for sname, fn in SELECTORS.items():
            acc[sname].append(recall(fn(q, K, budget, qpos, rng=rng), R))
    return {s: float(np.mean(v)) for s, v in acc.items()}


def main():
    n, d, r, budget = 512, 64, 3, 32
    trials = 300
    rng = np.random.default_rng(0)
    cfg = dict(n=n, d=d, r=r, budget=budget, trials=trials)

    # --- 1) geometry table at high SNR ---
    print(f"sparse-llm rung-1 recall gate  (recall@{budget}/{n}, d={d}, r={r} planted, "
          f"trials={trials}, alpha=6.0)")
    geoms = ["local", "far", "multi"]
    table = {g: mean_recall(g, 6.0, rng=rng, **cfg) for g in geoms}
    print(f"{'geometry':<8} " + " ".join(f"{s:>11}" for s in SELECTORS))
    for g in geoms:
        print(f"{g:<8} " + " ".join(f"{table[g][s]:>11.3f}" for s in SELECTORS))

    # --- 2) predeclared bench-validity control (non-degenerate) ---
    ceiling_ok = all(table[g]["exact_topk"] > 0.98 for g in geoms)
    window_fails_far = table["far"]["window"] < 0.02
    valid = ceiling_ok and window_fails_far
    print(f"\n[control] exact_topk ~1.0 all geoms: {ceiling_ok} | window ~0 on far: "
          f"{window_fails_far}  => bench {'VALID' if valid else 'INVALID'}")
    if not valid:
        print("ABORT: instrument cannot detect failure; selector comparison is void.")
        return

    # --- 3) predeclared research falsifier: SNR sweep on scattered relevance ---
    print("\n[falsifier] coarse-to-fine vs exact ceiling as SNR drops (geometry='multi'):")
    print(f"{'alpha':>6} {'exact_topk':>11} {'block_topk':>11} {'window':>9} "
          f"{'gap(ex-bl)':>11}")
    gaps = {}
    for alpha in [6.0, 3.0, 2.0, 1.5, 1.0, 0.7]:
        m = mean_recall("multi", alpha, rng=rng, **cfg)
        gap = m["exact_topk"] - m["block_topk"]
        gaps[alpha] = (m["exact_topk"], gap)
        print(f"{alpha:>6.1f} {m['exact_topk']:>11.3f} {m['block_topk']:>11.3f} "
              f"{m['window']:>9.3f} {gap:>+11.3f}")

    # The phenomenon: an SNR band where EXACT still recovers R (>0.9) but block pooling loses it.
    cost_band = [a for a, (ex, gap) in gaps.items() if ex > 0.9 and gap > TOL]
    print("\nVERDICT:", (
        f"coarse-to-fine has a REAL cost — at alpha in {sorted(cost_band)} exact selection still "
        f"recovers scattered relevance (>0.9) but the block prefilter dilutes it (gap > {TOL}). "
        f"The meta-selector's block pooling is the failure mode; that gap is the phenomenon to study."
        if cost_band else
        f"coarse-to-fine tracks exact selection within {TOL} wherever exact succeeds — block "
        f"pooling is ~free at this budget/blocksize. Push budget/blocksize or r to find its wall."))


if __name__ == "__main__":
    main()
