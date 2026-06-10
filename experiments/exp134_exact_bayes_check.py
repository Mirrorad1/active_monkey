"""
Exp 134 exact-Bayes check — is the C3 mechanism anomaly an implementation artifact?

The Exp 134 fresh-seed rerun confirmed the headline sign reversal (C1, C2) but failed
the strict mechanism conjuncts (C3): 7/46 runs had the tabular argmax NOT at the
majority-count corner. The static-state posterior q_c propto prod_t A[o_t, c] is
order-independent, and for the A-vs-D pair its log-odds are exactly
(n0 - n3) * log(A[0,A]/A[0,D]), so exact Bayes MUST put the argmax (among A, D) at
the majority corner whenever n0 != n3. Suspect: the experiment's multiply-then-
renormalize filter lets individual q entries underflow to exact float 0, after which
they can never recover — a ratchet that makes the implementation order-dependent.

Prediction: recomputing the same 46 fresh-seed runs (ratio >= 5.71 cells, seeds 8..15)
in exact log-space gives argmax == majority corner in 46/46, and the exact-Bayes
held-out NLL remains the same catastrophic magnitude as the floored filter's (the
headline reversal is NOT an artifact). Falsifier: any exact log-space mismatch
(the mechanism story is wrong, not the arithmetic).
"""
import numpy as np
from scipy.special import logsumexp


def run_check(L, sigma, seed, cell_index):
    corners = np.array([[0, 0], [L, 0], [0, L], [L, L]], float)
    Lam = np.linalg.inv(sigma**2 * np.eye(2))
    logw = np.zeros((4, 4))
    for k in range(4):
        for c in range(4):
            d = corners[c] - corners[k]
            logw[k, c] = -0.5 * d @ Lam @ d - np.log(2 * np.pi * sigma**2)
    logA = logw - logsumexp(logw, axis=0, keepdims=True)
    rng = np.random.default_rng(seed * 1000 + cell_index)
    train = rng.choice([0, 3], size=200)
    holdout = rng.choice([0, 3], size=200)
    n0 = int((train == 0).sum())
    n3 = 200 - n0

    # exact log-space posterior (order-independent by construction)
    logq = np.zeros(4)
    for w in train:
        logq = logq + logA[w, :]
    logq -= logsumexp(logq)
    exact_argmax = int(np.argmax(logq))

    # the experiment's multiply-then-renormalize filter (underflow ratchet)
    q = np.ones(4) / 4
    A = np.exp(logA)
    for w in train:
        q = q * A[w, :]
        s = q.sum()
        q = np.ones(4) / 4 if s < 1e-300 else q / s
    impl_argmax = int(np.argmax(q))

    majority = 0 if n0 > n3 else (3 if n3 > n0 else -1)
    qe = np.exp(logq)
    nll_exact = float(np.mean([-np.log(A[w, :] @ qe + 1e-300) for w in holdout]))
    nll_impl = float(np.mean([-np.log(A[w, :] @ q + 1e-300) for w in holdout]))
    return n0, n3, majority, exact_argmax, impl_argmax, nll_exact, nll_impl


def main():
    cells = [(L, s) for L in (0.5, 1.0, 2.0, 4.0) for s in (0.175, 0.35, 0.7)]
    print(f"{'L':>4} {'sig':>5} {'seed':>4} {'n0-n3':>5} {'maj':>3} "
          f"{'exact':>5} {'impl':>4} {'nll_ex':>9} {'nll_im':>9}")
    mismatch_exact = mismatch_impl = nties = ntotal = 0
    for ci, (L, s) in enumerate(cells):
        if L / s < 5.7:
            continue
        for seed in range(8, 16):
            n0, n3, maj, ea, ia, ne, ni = run_check(L, s, seed, ci)
            if maj == -1:
                nties += 1
                continue
            ntotal += 1
            if ea != maj:
                mismatch_exact += 1
            if ia != maj:
                mismatch_impl += 1
            if ea != maj or ia != maj:
                tag = "  <-- EXACT MISMATCH" if ea != maj else "  <-- impl artifact"
                print(f"{L:>4} {s:>5} {seed:>4} {n0 - n3:>+5} {maj:>3} "
                      f"{ea:>5} {ia:>4} {ne:>9.2f} {ni:>9.2f}{tag}")
    print(f"\nties skipped (n0 == n3): {nties}")
    print(f"exact log-space argmax != majority: {mismatch_exact}/{ntotal}")
    print(f"floored-filter argmax != majority:  {mismatch_impl}/{ntotal}")
    verdict = "CONFIRMED" if mismatch_exact == 0 else "NOT-CONFIRMED"
    print(f"VERDICT: artifact hypothesis {verdict} "
          f"(exact Bayes deterministic at the majority corner; "
          f"all anomalies belong to the floored filter)")


if __name__ == "__main__":
    main()
