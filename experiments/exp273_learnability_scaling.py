"""
Exp 273 — identity-ecological RUNG 1c-SCALE: does observation-only MB-FREE learnability SURVIVE
scaling to larger worlds where free-exploration coverage is NOT trivial? (focused, compute-bounded)

PROVENANCE (branch identity-ecological; origin reaches Exp 271, this line adds Exp 272 -> this is
Exp 273). Rung 1c (Exp 272, LEARNABLE) showed a MODEL-BASED learner that maps its world by FREE
(no-pull) peacetime exploration then plans the refuge MATCHES the omniscient optimal (closure ~1.0,
coverage 1.0) on mirro's 5x5 body -- but the 25-cell world is mapped by a free random walk in
~O(200) steps << the 50k budget, so coverage 1.000 was TRIVIAL. This experiment tests whether the
LEARNABLE finding survives when free-exploration coverage is forced sub-1.0.

DESIGN (a prior workflow attempt was UNDER-RANGED -- its artifact gate could not fire because the top
size N=400 still had fixed-budget coverage 0.955; and a latent free_explore bug labeled unvisited
cells color 0. Both fixed here): the worlds are TILED-mirro (the real 5x5 aliased 3-color pattern
tiled reps x reps -> (5*reps)^2 cells), preserving the identity-ecological framing at scale.

HYPOTHESIS (H_scale_survives). MB-FREE learnability is BUDGET-PROPORTIONAL, not size-limited: with a
free-exploration budget SCALED to the world (>= ~32 x N_cells, comfortably above the empirical
cover-time knee ~16 x N_cells), MB-FREE recovers full coverage (>= 0.95) and reaches the optimal
escape (per-color closure >= 0.80) at EVERY size. The decisive scientific claim is the COUPLING:
closure TRACKS coverage (falls when coverage falls under a budget-down sweep, recovers when coverage
recovers), proving "the map is the gap" persists at scale rather than collapsing into a new wall.

PREDICTION / PREDECLARATION (three readouts):
  (A) SURVIVAL (the SCALES test; uses policy iteration, sizes reps={1,2,3,4} -> N={25,100,225,400}):
      free budget = 32 x N_cells. PREDICT MB-FREE global coverage >= 0.95 AND per-color closure
      >= 0.80 in >= 7/8 seeds at EVERY size, every color; MB-on-TRUE-map closure >= 0.95 at every
      size (planner scales -> the planner is NOT the gap).
  (B) COVERAGE-COST LAW / COUPLING (sizes reps=3 -> N=225): free budget sweep {1,2,4,8,16,32} x
      N_cells. PREDICT a monotone-rising coverage curve and an MB-FREE closure curve that RISES WITH
      coverage (low closure at small budget, >= 0.80 by the ~16x knee).
  (C) ARTIFACT CONFIRMATION (coverage-only, cheap -- NO policy iteration; sizes reps={1,2,4,6} ->
      N={25,100,400,900}): at a FIXED budget (5000) coverage MUST DECLINE with N, dropping BELOW 0.90
      at N=900 (the regime the 5x5 never enters). PREDICT cov(N=900) < 0.90. If cov(N=900) >= 0.90 the
      size range is still too small -> NO-VERDICT-still-trivial (not SCALES).

FALSIFIER (H_scale_wall). MB-FREE caps even at the SCALED (32x) budget: per-color closure < 0.80 OR
global coverage < 0.95 in >= 2/8 seeds at >= 1 size -> learnability does NOT survive scaling (a deeper
map-acquisition wall, not under-budgeting). Both verdicts reachable: the budget-down sweep's smallest
budget (1x N_cells) is a predeclared NEGATIVE witness (degraded map -> low closure); the 32x knee is
the POSITIVE witness.

NO-OMNISCIENCE / ISOLATION (per size): MB-on-TRUE-map closure ~1.0 confirms the planner is correct at
scale, so any MB-FREE gap is map ACQUISITION. free_explore is observation-only (color-blind uniform
4-move walk; unvisited cells get a seeded-random label via tiny prior noise -- the FIX for the
zeros-argmax bug, so unvisited cells are NOT systematically the attack color). Per size, per color,
per seed (mean-of-opposites guard). L25 RUNTIME PREFLIGHT smoke-times the worst cell and aborts if the
projection exceeds a cap. PRINTED_VERDICT is the script's claim; the binding verdict is recomputed by
the controller.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from active_loop.creature import World
import experiments.exp272_learnable_actuator as L

ALPHA = L.A_PRIMARY          # 0.5
COLORS = [0, 1, 2]
N_SEEDS = 8
SEED_BAR = 7                 # of 8
CLOSURE_BAR = 0.80
COV_BAR = 0.95
KNEE_MULT = 32               # scaled free-budget = 32 x N_cells (>> cover-time knee ~16x)
FIXED_BUDGET = 5000          # artifact-confirmation fixed budget
MIRRO5 = None                # set in main from manifest


def tiled_world(reps):
    """Tile the real 5x5 mirro aliased pattern reps x reps -> a (5*reps)^2 World (same 3 colors)."""
    base = np.array(MIRRO5).reshape(5, 5)
    big = np.tile(base, (reps, reps))
    R = 5 * reps
    return World(rows=R, cols=R, n_colors=3, cmap=[int(x) for x in big.reshape(-1)])


def free_explore_fixed(w, seed, budget):
    """Observation-only FREE (no-pull) random walk -> cmap_hat + coverage. FIX vs the buggy version:
    counts init = tiny seeded per-(color,cell) noise so UNVISITED cells argmax to a RANDOM color, not
    systematically color 0 (removes the phantom-attack/refuge bias in the low-coverage regime)."""
    rng = np.random.default_rng(seed)
    n = w.n_cells
    counts = rng.random((w.n_colors, n)) * 1e-6      # unvisited -> random tiny -> random argmax
    pos = 0
    visited = set()
    for _ in range(budget):
        pos = int(w.move(pos, int(rng.integers(0, 4))))
        counts[int(w.cmap[pos]), pos] += 1.0
        visited.add(pos)
    return counts.argmax(axis=0), len(visited) / n


def mb_eval(w, cmap_hat, color, seed):
    pol = L.optimal_on_estimated(w, np.asarray(cmap_hat), color, ALPHA)
    return L.eval_mb_policy(w, color, ALPHA, pol, seed + L.EVAL_SEED_OFFSET)["f"]


def anchors(w, color):
    fp = float(np.mean([L.walk(w, color, ALPHA, "passive", s)["f"] for s in range(N_SEEDS)]))
    fo = float(np.mean([L.walk(w, color, ALPHA, "avoid", s)["f"] for s in range(N_SEEDS)]))
    return fp, fo


def main():
    global MIRRO5
    import json
    MIRRO5 = list(json.load(open("creature/state/mirro/manifest.json"))["world"]["cmap"])
    out = []
    def p(s=""):
        print(s); out.append(s)

    p("=" * 80)
    p("Exp 273 — RUNG 1c-SCALE: does MB-free learnability survive scaling? (tiled-mirro worlds)")
    p(f"  alpha={ALPHA} seeds={N_SEEDS} knee={KNEE_MULT}xN_cells bars: closure>={CLOSURE_BAR} cov>={COV_BAR} in >={SEED_BAR}/{N_SEEDS}")
    p("=" * 80)

    def cl(fp, fo, f):
        return L.closure(fp, fo, f)

    # ---- L25 RUNTIME PREFLIGHT: smoke-time the worst survival cell (N=400) ----
    w400 = tiled_world(4)
    t0 = time.time()
    fp, fo = anchors(w400, 0)
    ch, cov = free_explore_fixed(w400, 0, KNEE_MULT * w400.n_cells)
    _ = mb_eval(w400, ch, 0, 0)
    _ = mb_eval(w400, np.asarray(w400.cmap), 0, 0)   # MB-true (planner ref)
    cell_s = time.time() - t0
    proj = cell_s * (4 * 3 * N_SEEDS) + cell_s * (6 * N_SEEDS) + 0.05 * (4 * N_SEEDS)
    p(f"L25 PREFLIGHT: worst-cell(N=400) {cell_s:.3f}s -> projected upper bound ~{proj:.0f}s (cap 240s)")
    assert proj < 240, f"L25 ABORT: projected {proj:.0f}s exceeds cap"
    p("  L25 PREFLIGHT: PASS")
    p("")

    # ---- (C) ARTIFACT CONFIRMATION: coverage declines with N at fixed budget (no PI) ----
    p("(C) ARTIFACT CONFIRMATION — free-walk coverage at FIXED budget=5000 vs size (need cov(N=900)<0.90):")
    art = {}
    for reps in [1, 2, 4, 6]:
        w = tiled_world(reps); N = w.n_cells
        covs = [free_explore_fixed(w, s, FIXED_BUDGET)[1] for s in range(N_SEEDS)]
        art[N] = float(np.mean(covs))
        p(f"   reps={reps} N={N:4d}: mean coverage @5000 = {art[N]:.3f}")
    artifact_ok = art[900] < 0.90
    p(f"   ARTIFACT GATE: cov(N=900)={art[900]:.3f} < 0.90 ? {artifact_ok}")
    p("")

    # ---- (A) SURVIVAL: MB-free at scaled knee budget vs size, per color ----
    p(f"(A) SURVIVAL — MB-free at budget={KNEE_MULT}xN_cells, per size x color (8 seeds):")
    p(f"   {'reps':>4} {'N':>4} {'col':>3} {'fp':>5} {'fo':>5} | {'MBfree':>7} {'gcov':>5} {'ok/8':>4} | {'MBtrue':>7} | scale?")
    survival = []
    for reps in [1, 2, 3, 4]:
        w = tiled_world(reps); N = w.n_cells; B = KNEE_MULT * N
        for c in COLORS:
            fp, fo = anchors(w, c)
            mbf, covf, mbt = [], [], []
            for s in range(N_SEEDS):
                ch, cov = free_explore_fixed(w, L.LEARN_SEED_BASE + s, B)
                mbf.append(cl(fp, fo, mb_eval(w, ch, c, s))); covf.append(cov)
                mbt.append(cl(fp, fo, mb_eval(w, np.asarray(w.cmap), c, s)))
            mbf = np.array(mbf); covf = float(np.mean(covf)); mbt = np.array(mbt)
            ok = int(np.sum(mbf >= CLOSURE_BAR))
            scales = (ok >= SEED_BAR) and (covf >= COV_BAR)
            survival.append((N, c, float(np.mean(mbf)), covf, ok, float(np.mean(mbt)), scales))
            p(f"   {reps:>4} {N:>4} {c:>3} {fp:>5.2f} {fo:>5.2f} | {np.mean(mbf):>7.3f} {covf:>5.3f} {ok:>2}/8 | "
              f"{np.mean(mbt):>7.3f} | {'SCALES' if scales else 'WALL'}")
    p("")

    # ---- (B) COVERAGE-COST LAW / COUPLING at N=225 ----
    p("(B) COVERAGE-COST LAW — N=225 (reps=3) color 0, budget sweep {1,2,4,8,16,32}xN_cells:")
    p(f"   {'mult':>4} {'budget':>7} {'gcov':>5} {'MBfree_cl':>9}")
    w = tiled_world(3); N = w.n_cells; fp, fo = anchors(w, 0)
    coupling = []
    for mult in [1, 2, 4, 8, 16, 32]:
        B = mult * N
        cls, covs = [], []
        for s in range(N_SEEDS):
            ch, cov = free_explore_fixed(w, L.LEARN_SEED_BASE + s, B)
            cls.append(cl(fp, fo, mb_eval(w, ch, 0, s))); covs.append(cov)
        coupling.append((mult, float(np.mean(covs)), float(np.mean(cls))))
        p(f"   {mult:>4} {B:>7} {np.mean(covs):>5.3f} {np.mean(cls):>9.3f}")
    p("")

    # ---- VERDICT ----
    p("=" * 80)
    all_scale = all(s[6] for s in survival)
    planner_scales = all(s[5] >= 0.95 for s in survival)
    cov_rises = coupling[-1][1] - coupling[0][1] > 0.3
    cl_rises = (coupling[-1][2] - coupling[0][2] > 0.3) and (coupling[-1][2] >= CLOSURE_BAR)
    p("FINAL VERDICT INPUTS:")
    p(f"  artifact gate (cov(900)<0.90): {artifact_ok}  [{art[900]:.3f}]")
    p(f"  survival all-sizes-colors SCALES at 32x knee: {all_scale}")
    p(f"  planner scales (MB-true>=0.95 all sizes): {planner_scales}")
    p(f"  coupling (coverage & closure rise together, closure>=bar at knee): cov_rise={cov_rises} cl_rise={cl_rises}")
    if artifact_ok and all_scale and planner_scales and cl_rises:
        verdict = "SCALES (learnability is budget-proportional; the map is the gap, at scale)"
    elif not artifact_ok:
        verdict = "NO-VERDICT-still-trivial (artifact gate failed; size range too small)"
    elif not all_scale:
        verdict = "SCALE-WALL (MB-free caps at the scaled knee on >=1 size -> a deeper acquisition wall)"
    else:
        verdict = "MIXED"
    p(f"PRINTED_VERDICT: {verdict}")
    with open("experiments/outputs/exp273.txt", "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
