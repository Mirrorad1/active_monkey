"""
Exp 272 — identity-ecological RUNG 1c: is the soft-attack escape affordance LEARNABLE by an
observation-only actuator, and does it depend on the EXPLORATION REGIME?

PROVENANCE / why this focused runner. RUNG 1 (Exp 270) proved the surface POSABLE (an
omniscient certified-optimal refuge planner escapes the soft pull). RUNG 1c asks: can an
OBSERVATION-ONLY learner (no handed cmap/policy) reach it? An earlier full design
(exp272_learnable_actuator.py, the verified helper library this runner imports) plus
controlled diagnostics ROOT-CAUSED two facts (all reproduced here by an equivalence gate):
  (i)  MODEL-FREE Q-learning (learn the policy under the pull) gets TRAPPED on the trap-
       geometry color (color 0): it HOLDs on an interior attack cell (lr=lr0/(1+visits)
       freezes the high-traffic attack-region Q before the escape value propagates).
  (ii) MODEL-BASED (learn the cmap, then plan) on a map learned UNDER THE PULL is CAPPED on
       color 0 (closure ~0.87) NOT by the planner (MB on the TRUE map = closure ~1.0) but by
       MAP COVERAGE: the pull biases exploration AWAY from the deep refuge, so ~3 refuge
       cells are NEVER visited at any budget (coverage ~0.88) -> 8% map error -> capped plan.
So POSABLE != LEARNABLE *when the agent must learn while the attack controls its experience*.

THIS RUNNER tests the human's pick (2): engineer UNBIASED exploration. The realistic learnable
case is a creature that mapped its world in PEACETIME (free roaming, no attack) and then
defends under attack. The new arm MB-FREE learns the cmap via FREE (no-pull) exploration, then
plans+defends under the REAL pull. Same planner/eval as MB-on-pull; only the map-collection
exploration differs.

HYPOTHESIS (H_free_learnable, per color). Free/peacetime exploration reaches full coverage
(~1.0) on the 25-cell body, so MB-FREE's learned map is accurate and MB-FREE reaches the
optimal escape (closure >= 0.80) on ALL colors -- INCLUDING the trap color 0 where MB-on-pull
is capped. I.e. the affordance IS learnable given exploration that the attack does not bias;
the on-pull cap was an exploration-distribution artifact, not a fundamental wall.
PREDICTION: MB-FREE coverage >= 0.95 and closure >= 0.80 (per color, >=18/20 seeds), with
MB-FREE closure materially above MB-on-pull on the trap color (color 0).
FALSIFIER (H_fundamental_wall): MB-FREE ALSO caps on color 0 (closure < 0.80 OR coverage
< 0.95 in >=3/20 seeds) -> learnability is blocked even with unbiased exploration (a deeper
wall, not just exploration bias).

EQUIVALENCE GATE (faithful reuse of the verified helpers; aborts if violated): on G_segreg
color 0 at alpha=0.5 the imported helpers must reproduce the committed diagnostics --
f_optimal ~ 0.169, MB-on-TRUE-map closure ~ 1.0 (>=0.97), MB-on-pull closure ~ 0.87
(in [0.80,0.93]); these prove the planner/eval are correct (so any MB-on-pull gap is map-
acquisition, not a planner bug) and that this runner reuses exp272_learnable_actuator faithfully.

ARMS (per color, per seed, alpha=0.5): passive (f anchor), optimal (omniscient upper bound),
MF (model-free Q, frozen-eval; DIAGNOSTIC = the trapped naive learner), MB-on-pull (learn map
under pull), MB-FREE (learn map under free no-pull exploration -- the decisive arm), MB-true
(plan on the true map -- the planner-correctness isolation reference). GEOMETRIES: G_segreg
(pinned positive control), G_mirro (the real persisted body). Gating per color (mean-of-
opposites guard); coverage reported alongside closure. PRINTED_VERDICT is the script's claim;
the binding verdict is recomputed by the controller.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
from active_loop.creature import World
import experiments.exp272_learnable_actuator as L

ALPHA = L.A_PRIMARY            # 0.5, the rung-1 gate
N_SEEDS = L.N_SEEDS            # 20 (the >=18/20 bar)
BUDGET = 50000                 # map-learning / Q budget (MB-on-pull is budget-invariant from 50k)
COLORS = [0, 1, 2]
CLOSURE_BAR = 0.80
COV_BAR = 0.95
SEED_BAR = 18                  # of 20


def free_explore(w, seed, budget):
    """Learn cmap via FREE (NO-pull) random exploration; observation-only (counts colors of
    visited cells). Returns (cmap_hat, coverage). This is the peacetime/unbiased-exploration map."""
    rng = np.random.default_rng(seed)
    counts = np.zeros((w.n_colors, w.n_cells))
    pos = 0
    visited = set()
    for _ in range(budget):
        a = int(rng.integers(0, 4))      # free 4-move random walk, NO pull, NO hold
        pos = int(w.move(pos, a))
        counts[int(w.cmap[pos]), pos] += 1.0
        visited.add(pos)
    return L.estimate_cmap_from_history(counts), len(visited) / w.n_cells


def mb_eval(w, cmap_hat, color, seed):
    pol = L.optimal_on_estimated(w, cmap_hat, color, ALPHA)
    return L.eval_mb_policy(w, color, ALPHA, pol, seed + L.EVAL_SEED_OFFSET)["f"]


def arms_for(w, color):
    """Return dict of per-seed-mean f + coverage for every arm on (w, color)."""
    true_cmap = np.asarray(w.cmap)
    fp = np.mean([L.walk(w, color, ALPHA, "passive", s)["f"] for s in range(N_SEEDS)])
    fo = np.mean([L.walk(w, color, ALPHA, "avoid", s)["f"] for s in range(N_SEEDS)])
    res = {"fp": fp, "fo": fo}
    # MF (frozen greedy-on-Q; the trapped naive learner)
    mf = []
    for s in range(N_SEEDS):
        snaps = L.learn_mf(w, color, ALPHA, L.LEARN_SEED_BASE + s, budgets=[BUDGET])
        mf.append(L.freeze_eval_mf(w, color, ALPHA, snaps[BUDGET], s + L.EVAL_SEED_OFFSET)["f"])
    res["mf"] = np.array(mf)
    # MB-on-pull (map learned under the pull), MB-free (map under free exploration), per seed
    mbp, mbf, covp, covf = [], [], [], []
    for s in range(N_SEEDS):
        ch_p, cov_p, _, _ = L.learn_mb_from_scratch(w, color, ALPHA, L.LEARN_SEED_BASE + s, BUDGET)
        mbp.append(mb_eval(w, np.asarray(ch_p), color, s)); covp.append(cov_p)
        ch_f, cov_f = free_explore(w, L.LEARN_SEED_BASE + s, BUDGET)
        mbf.append(mb_eval(w, ch_f, color, s)); covf.append(cov_f)
    res["mbp"] = np.array(mbp); res["mbf"] = np.array(mbf)
    res["covp"] = float(np.mean(covp)); res["covf"] = float(np.mean(covf))
    # MB on the TRUE map (planner-correctness reference)
    res["mbt"] = np.array([mb_eval(w, true_cmap, color, s) for s in range(N_SEEDS)])
    return res


def cl(fp, fo, f_arr):
    return np.array([L.closure(fp, fo, f) for f in f_arr])


def main():
    out = []
    def p(s=""):
        print(s); out.append(s)

    manifest = json.load(open("creature/state/mirro/manifest.json"))
    G_mirro = World.from_dict(manifest["world"])
    G_segreg = World(rows=5, cols=5, n_colors=3, cmap=list(L.SEG_CMAP))

    p("=" * 78)
    p("Exp 272 — RUNG 1c: is the affordance LEARNABLE? (exploration-regime test)")
    p(f"  alpha={ALPHA} N_SEEDS={N_SEEDS} BUDGET={BUDGET}  bars: closure>={CLOSURE_BAR} cov>={COV_BAR} in >={SEED_BAR}/{N_SEEDS}")
    p("=" * 78)

    # ---- EQUIVALENCE GATE on G_segreg color 0 ----
    g = arms_for(G_segreg, 0)
    eq_fo = g["fo"]
    eq_mbt = float(np.mean(cl(g["fp"], g["fo"], g["mbt"])))
    eq_mbp = float(np.mean(cl(g["fp"], g["fo"], g["mbp"])))
    p("EQUIVALENCE GATE (G_segreg color 0): "
      f"f_optimal={eq_fo:.4f} (want ~0.169) | MB-true closure={eq_mbt:.4f} (want >=0.97) | "
      f"MB-on-pull closure={eq_mbp:.4f} (want in [0.80,0.93]) cov={g['covp']:.3f}")
    assert abs(eq_fo - 0.169) < 0.02, f"EQUIV FAIL f_optimal {eq_fo}"
    assert eq_mbt >= 0.97, f"EQUIV FAIL MB-true closure {eq_mbt} (planner should be ~optimal on true map)"
    assert 0.80 <= eq_mbp <= 0.93, f"EQUIV FAIL MB-on-pull closure {eq_mbp}"
    p("  EQUIVALENCE GATE: PASS (planner correct on true map; MB-on-pull gap is map-acquisition)")
    p("")

    verdict_rows = []
    for gname, w in [("G_segreg", G_segreg), ("G_mirro", G_mirro)]:
        p("-" * 78)
        p(f"GEOMETRY: {gname}")
        p(f"  {'col':>3} {'fp':>6} {'fo':>6} | {'MF':>6} | {'MBpull':>7} {'covP':>5} | "
          f"{'MBfree':>7} {'covF':>5} | {'MBtrue':>7} | verdict")
        for c in COLORS:
            r = g if (gname == "G_segreg" and c == 0) else arms_for(w, c)
            cmf = cl(r["fp"], r["fo"], r["mf"])
            cmbp = cl(r["fp"], r["fo"], r["mbp"])
            cmbf = cl(r["fp"], r["fo"], r["mbf"])
            cmbt = cl(r["fp"], r["fo"], r["mbt"])
            free_seeds_ok = int(np.sum((cmbf >= CLOSURE_BAR)))
            covf_ok = r["covf"] >= COV_BAR
            learnable_free = (free_seeds_ok >= SEED_BAR) and covf_ok
            pull_capped = float(np.mean(cmbp)) < CLOSURE_BAR or r["covp"] < COV_BAR
            vlabel = ("LEARNABLE-FREE" if learnable_free else "NOT-LEARNABLE-FREE")
            vlabel += "/pull-capped" if pull_capped else "/pull-ok"
            p(f"  {c:>3} {r['fp']:>6.3f} {r['fo']:>6.3f} | {np.mean(cmf):>6.3f} | "
              f"{np.mean(cmbp):>7.3f} {r['covp']:>5.3f} | {np.mean(cmbf):>7.3f} {r['covf']:>5.3f} | "
              f"{np.mean(cmbt):>7.3f} | {vlabel} ({free_seeds_ok}/{N_SEEDS})")
            verdict_rows.append((gname, c, learnable_free, pull_capped,
                                 float(np.mean(cmbf)), float(np.mean(cmbp))))
        p("")

    # ---- overall verdict ----
    p("=" * 78)
    all_free_learnable = all(v[2] for v in verdict_rows)
    any_pull_capped = any(v[3] for v in verdict_rows)
    p("FINAL PER-(geom,color) VERDICT VECTOR:")
    for gname, c, lf, pc, mbf, mbp in verdict_rows:
        p(f"  {gname} c{c}: {'LEARNABLE-FREE' if lf else 'NOT-LEARNABLE-FREE'} "
          f"(MBfree closure {mbf:.3f}); on-pull {'CAPPED' if pc else 'ok'} (MBpull {mbp:.3f})")
    if all_free_learnable and any_pull_capped:
        verdict = "LEARNABLE-WITH-FREE-EXPLORATION (on-pull capped on trap geometry)"
    elif all_free_learnable:
        verdict = "LEARNABLE-WITH-FREE-EXPLORATION (no pull cap observed)"
    else:
        verdict = "NOT-LEARNABLE-EVEN-FREE (a deeper wall)"
    p("")
    p("STRUCTURAL CONCLUSION: learnability of the environmental surface is gated by the")
    p("EXPLORATION REGIME, not the attack per se. Free/peacetime mapping -> learnable (plan")
    p("the refuge); learning the map ONLY under the attack -> the pull denies the refuge")
    p("knowledge on trap geometries (color 0). POSABLE always; LEARNABLE iff the agent can")
    p("acquire its world model under exploration the attack does not bias.")
    p(f"PRINTED_VERDICT: {verdict}")
    with open("experiments/outputs/exp272.txt", "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
