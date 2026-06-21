"""experiments/exp261_perseed.py — Exp 261 per-seed decomposition (controller-owned re-derivation
of the verdict-critical bistability evidence). Re-runs the co-evolving arms race per seed and reports
the D=atk@3k-esc@3k distribution + regime clades, to test whether the near-zero MEAN D at the balanced
costs is a stable mutual plateau or a bimodal mean-of-opposites. PREDICTION/FALSIFIER: if sd(D) across
seeds is small and most seeds are individually |D|<=0.30 with both traits co-escalated, the symmetry is
real; if sd(D) dwarfs |mean D| and seeds split into opposite prey-dom/pred-dom clades, it is bistable
(NOT symmetrized) — the pre-registered most-dangerous confound."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from experiments.exp261_costed_predator_attack import armsrace, SEEDS_B

L = ["Exp 261 PER-SEED decomposition (D = atk@3k - esc@3k). RAW — controller owns this measurement.", ""]
for ac in (0.0, 0.10, 0.15, 0.40):
    rs = [armsrace(ac, s) for s in SEEDS_B]
    D = [r['atk_at'][3000] - r['esc_at'][3000] for r in rs]
    esc = [r['esc_at'][3000] for r in rs]; atk = [r['atk_at'][3000] for r in rs]
    preydom = sum(1 for d in D if d <= -0.40); preddom = sum(1 for d in D if d >= 0.40)
    bal = sum(1 for d in D if abs(d) <= 0.30)
    both_hi = sum(1 for e, a in zip(esc, atk) if e > 1.5 and a > 1.5)
    prey_collapsed = sum(1 for e in esc if e < 1.0)
    # Sarle bimodality coefficient of D: (skew^2+1)/kurt
    Dn = np.array(D); n = len(Dn); m = Dn.mean(); s = Dn.std()
    sk = (((Dn-m)**3).mean())/s**3 if s>0 else 0; ku = (((Dn-m)**4).mean())/s**4 if s>0 else 0
    bc = (sk**2 + 1) / (ku + 3*(n-1)**2/((n-2)*(n-3))) if n > 3 and ku!=0 else float('nan')
    L.append(f"ac={ac:.2f}: meanD={m:+.3f} sd(D)={s:.3f}  clades[prey-dom={preydom} pred-dom={preddom} bal|D|<=0.3={bal}]  both>1.5={both_hi} prey_collapsed<1.0={prey_collapsed}  bimodCoeff={bc:.3f}")
    L.append(f"        per-seed D: " + " ".join(f"{d:+.2f}" for d in D))
    L.append(f"        per-seed (esc,atk): " + " ".join(f"({e:.1f},{a:.1f})" for e, a in zip(esc, atk)))
out = "\n".join(L); print(out)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "exp261_perseed.txt"), "w").write(out + "\n")
