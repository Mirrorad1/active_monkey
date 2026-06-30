"""
Exp 274 — identity-ecological RUNG 1c-MF: can MODEL-FREE learnability of the environmental identity
surface be REPAIRED so it escapes the trap-geometry trap WITHOUT a map?

PROVENANCE (branch identity-ecological; origin reaches Exp 271, this line adds 272/273 -> Exp 274).
Rung 1c (Exp 272): a MODEL-BASED learner (map by free exploration -> plan) matches the optimal escape;
a naive MODEL-FREE Q-learner gets TRAPPED on the trap color (G_segreg color 0): frozen greedy-Q HOLDs
on an interior attack cell -> closure ~ -0.2 (worse than passive). Root cause (Exp 272 debug): (i)
lr=lr0/(1+visits) FREEZES the high-traffic attack-region Q before the escape value propagates; (ii)
DEEPER -- learning UNDER the pull biases experience away from the refuge (a coverage/experience block),
the model-free analog of MB-free's map gap (L46/L47). The lr-fix alone (slower decay) recovered colors
1/2 but NOT color 0 (necessary-not-sufficient).

REPAIR HYPOTHESIS (H_mf_repairable). EXPLORING STARTS fix it: during Q-learning UNDER THE PULL,
periodically RESTART the learner at a uniformly-random cell so it EXPERIENCES all states incl. the
refuge/escape routes -> the escape-Q propagates -> frozen greedy escapes. FAIRNESS/KERNEL (binding):
MF learns Q for the PULL kernel (the dynamics it acts in); exploring starts keep the pull and only fix
STATE COVERAGE -- they are observation-only (the restart is an exploration device; the agent still
learns Q from {state,action,observed reward}; it is NOT handed the map/policy and does NOT learn under
the wrong no-pull kernel). The frozen-greedy eval starts at a FIXED pos=0 (worst case) for ALL arms, so
escape cannot be a random-refuge-start artifact.

PREDICTION / PREDECLARATION. Three MF variants (frozen-greedy closure under the pull, eval pos=0):
  MF-baseline  (lr_tau=1 == lr0/(1+visits), eps0=0.2, no restart) -> the trapped baseline.
  MF-lrfix     (lr_tau=2000, eps0=0.5, no restart)                -> isolates the lr effect.
  MF-restart   (lr_tau=2000, eps0=0.5, exploring starts every 100)-> the repair.
Anchors: passive (f-anchor), optimal (omniscient upper bound), MB-free (model-based learnable ref).
PREDICT (per color, >=7/8 seeds, 8 seeds): MF-lrfix recovers colors 1/2 (closure>=0.80) but NOT the
trap color 0; MF-restart recovers ALL colors INCLUDING color 0 (closure>=0.80) with learning-coverage
~1.0; the ISOLATION MF-restart - MF-lrfix on color 0 is large and positive (exploring-starts/coverage
is the binding cause, not lr).
FALSIFIER (H_mf_unrepairable): MF-restart STILL caps on color 0 (closure < 0.80 in >=2/8 seeds) even
with full exploring-starts coverage -> a genuine model-free wall beyond coverage.
EQUIVALENCE GATE (faithful reuse, asserted): MF-baseline on G_segreg color 0 reproduces the committed
trap -> closure < 0.0 (it HOLDs on attack). Anchors confirm the target is reachable (optimal escapes;
closure denominator non-degenerate). hypothesis/prediction/falsifier are predeclared above.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
from active_loop.creature import World
import experiments.exp272_learnable_actuator as L

ALPHA = L.A_PRIMARY          # 0.5
COLORS = [0, 1, 2]
N_SEEDS = 8
SEED_BAR = 7
CLOSURE_BAR = 0.80
BUDGET = 50000
RESTART_K = 100
LRFIX_TAU = 2000.0
LRFIX_EPS0 = 0.5
BASE_EPS0 = 0.2


def learn_mf_variant(w, color, alpha, learn_seed, budget, lr_tau, eps0, restart_period,
                     eps1=0.02, gamma=L.GAMMA):
    """Parameterized tabular Q-learning UNDER THE PULL (does NOT mutate the imported learn_mf).
    lr_tau: lr = MF_LR0/(1+visits/lr_tau) (tau=1 == the baseline lr0/(1+visits)). restart_period:
    None = no exploring starts; K = teleport to a uniform-random cell every K steps (exploring start,
    keeping the PULL kernel). Returns (Q, learning_coverage). Q-update sees only {pos,a,nxt,reward}."""
    n = w.n_cells
    cost = (np.asarray(w.cmap) == color).astype(float)
    pull_sets, pull_lens = L._pull_env(w, color)
    act_next = L._move_table(w)
    ss = np.random.SeedSequence(learn_seed).spawn(5)
    r_eps = np.random.default_rng(ss[0]).random(budget)
    r_act = np.random.default_rng(ss[1]).integers(0, L.N_ACTIONS, budget)
    r_pull = np.random.default_rng(ss[2]).random(budget)
    r_pidx = np.random.default_rng(ss[3]).random(budget)
    r_restart = np.random.default_rng(ss[4]).integers(0, n, budget)
    Q = np.zeros((n, L.N_ACTIONS))
    visits = np.zeros((n, L.N_ACTIONS))
    visited = set()
    pos = 0
    for step in range(budget):
        if restart_period and (step % restart_period == 0):
            pos = int(r_restart[step])                 # EXPLORING START (uniform cell; pull kernel kept)
        eps = eps0 - (eps0 - eps1) * step / budget
        a = int(r_act[step]) if r_eps[step] < eps else L._greedy_action(Q[pos], act_next[pos])
        if r_pull[step] < alpha:
            cs = pull_sets[pos]; nxt = int(cs[int(r_pidx[step] * pull_lens[pos])])
        else:
            nxt = int(act_next[pos, a])
        visits[pos, a] += 1.0
        lr = L.MF_LR0 / (1.0 + visits[pos, a] / lr_tau)
        Q[pos, a] += lr * (-cost[nxt] + gamma * Q[nxt].max() - Q[pos, a])
        pos = nxt
        visited.add(pos)
    return Q, len(visited) / n


def free_explore_fixed(w, seed, budget):
    rng = np.random.default_rng(seed)
    counts = rng.random((w.n_colors, w.n_cells)) * 1e-6     # unvisited -> random label (bug-fixed)
    pos = 0
    for _ in range(budget):
        pos = int(w.move(pos, int(rng.integers(0, 4))))
        counts[int(w.cmap[pos]), pos] += 1.0
    return counts.argmax(axis=0)


def main():
    out = []
    def p(s=""):
        print(s); out.append(s)

    manifest = json.load(open("creature/state/mirro/manifest.json"))
    G_mirro = World.from_dict(manifest["world"])
    G_segreg = World(rows=5, cols=5, n_colors=3, cmap=list(L.SEG_CMAP))

    p("=" * 82)
    p("Exp 274 — RUNG 1c-MF: can model-free learnability be REPAIRED (exploring starts vs lr-fix)?")
    p(f"  alpha={ALPHA} seeds={N_SEEDS} budget={BUDGET} restart_K={RESTART_K} bars: closure>={CLOSURE_BAR} in >={SEED_BAR}/{N_SEEDS}")
    p("=" * 82)

    def cl(fp, fo, f):
        return L.closure(fp, fo, f)

    def mf_arm(w, color, lr_tau, eps0, restart):
        cls, covs = [], []
        for s in range(N_SEEDS):
            Q, cov = learn_mf_variant(w, color, ALPHA, L.LEARN_SEED_BASE + s, BUDGET, lr_tau, eps0, restart)
            f = L.freeze_eval_mf(w, color, ALPHA, Q, s + L.EVAL_SEED_OFFSET)["f"]
            cls.append(f); covs.append(cov)
        return np.array(cls), float(np.mean(covs))

    def mb_free_arm(w, color):
        fs = []
        for s in range(N_SEEDS):
            ch = free_explore_fixed(w, L.LEARN_SEED_BASE + s, BUDGET)
            pol = L.optimal_on_estimated(w, np.asarray(ch), color, ALPHA)
            fs.append(L.eval_mb_policy(w, color, ALPHA, pol, s + L.EVAL_SEED_OFFSET)["f"])
        return np.array(fs)

    # ---- L25 RUNTIME PREFLIGHT (worst variant-cell = MF-restart on G_mirro color 0) ----
    t0 = time.time()
    _ = mf_arm(G_mirro, 0, LRFIX_TAU, LRFIX_EPS0, RESTART_K)
    cell_s = (time.time() - t0) / N_SEEDS
    proj = cell_s * N_SEEDS * (2 * 3 * 4)   # 2 geoms x 3 colors x ~4 MF arms (upper bound)
    p(f"L25 PREFLIGHT: per-seed MF-restart(N=25) {cell_s:.3f}s -> projected ~{proj:.0f}s (cap 300s)")
    assert proj < 300, f"L25 ABORT: projected {proj:.0f}s exceeds cap"
    p("  L25 PREFLIGHT: PASS")
    p("")

    # ---- EQUIVALENCE GATE: MF-baseline reproduces the trap on G_segreg color 0 (closure < 0) ----
    fp0, fo0 = (float(np.mean([L.walk(G_segreg, 0, ALPHA, k, s)["f"] for s in range(N_SEEDS)]))
                for k in ("passive", "avoid"))
    base_cl0, _ = mf_arm(G_segreg, 0, 1.0, BASE_EPS0, None)
    eq = float(np.mean(cl(fp0, fo0, base_cl0)))
    p(f"EQUIVALENCE GATE: MF-baseline G_segreg color 0 closure = {eq:.4f} (want < 0.0 = reproduces trap)")
    assert eq < 0.0, f"EQUIV FAIL: MF-baseline closure {eq} not < 0 (trap not reproduced)"
    p("  EQUIVALENCE GATE: PASS (faithful reuse; MF-baseline huddles on attack)")
    p("")

    # ---- main: per geometry x color, all arms ----
    isolation = {}
    verdict_rows = []
    for gname, w in [("G_segreg", G_segreg), ("G_mirro", G_mirro)]:
        p("-" * 82)
        p(f"GEOMETRY: {gname}")
        p(f"  {'col':>3} {'fp':>5} {'fo':>5} | {'MFbase':>6} | {'MFlrfix':>7} {'covL':>5} | {'MFrestart':>9} {'covR':>5} | {'MBfree':>6} | repaired?")
        for c in COLORS:
            fp = float(np.mean([L.walk(w, c, ALPHA, "passive", s)["f"] for s in range(N_SEEDS)]))
            fo = float(np.mean([L.walk(w, c, ALPHA, "avoid", s)["f"] for s in range(N_SEEDS)]))
            base, _ = mf_arm(w, c, 1.0, BASE_EPS0, None)
            lrf, covL = mf_arm(w, c, LRFIX_TAU, LRFIX_EPS0, None)
            res, covR = mf_arm(w, c, LRFIX_TAU, LRFIX_EPS0, RESTART_K)
            mbf = mb_free_arm(w, c)
            cb = cl(fp, fo, base); clf = cl(fp, fo, lrf); cr = cl(fp, fo, res); cmb = cl(fp, fo, mbf)
            res_ok = int(np.sum(cr >= CLOSURE_BAR))
            repaired = res_ok >= SEED_BAR
            isolation[(gname, c)] = (float(np.mean(clf)), float(np.mean(cr)))
            verdict_rows.append((gname, c, float(np.mean(cb)), float(np.mean(clf)), float(np.mean(cr)),
                                 res_ok, repaired, covR))
            p(f"  {c:>3} {fp:>5.2f} {fo:>5.2f} | {np.mean(cb):>6.2f} | {np.mean(clf):>7.3f} {covL:>5.2f} | "
              f"{np.mean(cr):>9.3f} {covR:>5.2f} | {np.mean(cmb):>6.3f} | {'REPAIRED' if repaired else 'no'} ({res_ok}/8)")
        p("")

    # ---- VERDICT ----
    p("=" * 82)
    all_repaired = all(r[6] for r in verdict_rows)
    # trap color (segreg c0 + mirro c0): lr-fix-alone insufficient, restart fixes -> coverage is the cause
    trap = [(g, c) for (g, c) in isolation if c == 0]
    lrfix_fails_c0 = any(isolation[k][0] < CLOSURE_BAR for k in trap)
    restart_fixes_c0 = all(isolation[k][1] >= CLOSURE_BAR for k in trap)
    p("ISOLATION (trap color 0): MFlrfix closure vs MFrestart closure:")
    for k in trap:
        p(f"  {k[0]} c0: lrfix={isolation[k][0]:.3f}  restart={isolation[k][1]:.3f}  (exploring-starts effect={isolation[k][1]-isolation[k][0]:+.3f})")
    p(f"  lr-fix-alone fails the trap color somewhere: {lrfix_fails_c0}  | restart fixes the trap color: {restart_fixes_c0}")
    if all_repaired and restart_fixes_c0 and lrfix_fails_c0:
        verdict = "REPAIRED (model-free learnable via exploring starts; coverage was the binding cause, not lr)"
    elif all_repaired:
        verdict = "REPAIRED (model-free learnable; lr-fix may already suffice -- check isolation)"
    elif not restart_fixes_c0:
        verdict = "UNREPAIRABLE-ON-TRAP (exploring starts do not fix color 0 -> a deeper model-free wall)"
    else:
        verdict = "PARTIAL"
    p(f"PRINTED_VERDICT: {verdict}")
    with open("experiments/outputs/exp274.txt", "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
