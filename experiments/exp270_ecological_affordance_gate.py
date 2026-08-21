"""
Exp 270 — identity-ecological RUNG 1: the affordance gate (soft-attack escapability probe).

PROVENANCE. RUNG 1 / ATTEMPT 2 of the identity-ecological direction, on the clean main-based
line: origin/main ended at Exp 269, so this is Exp 270. (The prior sparse-llm research arc
that briefly occupied numbers 270-276 -- including ATTEMPT 1, a no-attack free-roam probe that
was NO-VERDICT/INVALID-by-degeneracy: gifted oracles huddled on a refuge cell and hit trivial
bounds f=1.000/0.000 IDENTICALLY on the aliased body AND a segregated control, geometry-blind
-- was removed from history per the human's decision, so 270 is the next free number here.)
The attempt-1 degeneracy is exactly why this attempt adds the resistible soft attack: huddling
must no longer be free.

QUESTION (necessary condition for the whole direction). On mirro's real persisted 5x5,
3-color, ALIASED world body, can the STRONGEST FAIR movement policy keep its observation
stream (cell-color "diet") off a target "attack" color under a SOFT spatial attack that
captures a passive walker — and is that escapability governed by the world's REFUGE
GEOMETRY (the largest connected non-attack region and its depth) rather than being either
universally-afforded or universally-blocked?

SOFT ATTACK (the resistible pull; NO teleport, NO qs/belief reset — see SCOPE). At each
step, with probability alpha the agent is pulled ONE cell along the BFS distance-gradient
toward the nearest attack-color cell (uniform random among neighbors that strictly reduce
distance-to-nearest-attack; if already ON an attack cell it stays — absorbing under the
pull); with probability (1-alpha) the agent's own policy chooses the move. Exactly ONE cell
transition per step. obs(t) is read from the post-transition cell of step t-1, matching the
real step phase in experiments/exp174_n4_gate.py L244 (obs taken at the start of a step from
where the previous step left the body).

SCOPE (predeclared, deliberate). The soft pull isolates the SPATIAL-EXPOSURE channel only.
It deliberately OMITS the qs/belief-reset half of full N4 captivity (exp174 L287-291 sets qs
uniform over attack cells). The question at rung 1 is strictly whether the OBSERVATION
STREAM can be steered off the attack color; the belief-disorientation channel is out of
scope here. The gated quantity is the observation-stream diet fraction (the Exp 176/192
stream-composition quantity), NOT the LAMBDA-decayed argmax-v identity itself; the v-store
identity displacement is computed and reported as an ungated faithfulness diagnostic
(v[attack]-share and whether argmax(v) flips) to confirm diet is a load-bearing proxy on
this body. RUNG 2+ adds the belief channel and the v-store gate; this rung gates on stream.

POLICIES (both move every TURN; the no-op is allowed only as a HOLD that the pull still
overrides, so standing still cannot win — the pull keeps re-asserting the attack diet unless
the agent actively relocates to a pull-resistant refuge). passive = uniform-random among the
four moves with the wall-clamp no-op FORBIDDEN (a genuine cell change every turn — the
"captured walker" baseline; its no-op-forbidden degree-weighted stationary diet, NOT raw
abundance, is the reference). avoid = the OPTIMAL refuge-seeking planner: value iteration on
the exact 25-state alpha-mixed transition kernel (known World.move graph + known cmap +
known pull) minimizing long-run discounted attack-color occupancy, hold ALLOWED. This is the
honestly-strongest fair avoider (an exact optimum, not a myopic greedy heuristic — greedy
BFS-away UNDER-performs and would manufacture a false NEGATIVE on scattered colors), so a
NEGATIVE result is geometry-bound, not heuristic-bound.

GEOMETRIES. G_mirro = the real persisted mirro body (World.from_dict of the committed
manifest; cmap = [0,1,0,1,0,1,0,1,0,1,2,2,2,1,0,2,2,2,0,1,2,2,2,1,0], counts 0:8,1:8,2:9;
color 2 is a contiguous 3x3 block, colors 0/1 are scattered checkerboard). G_segreg =
POSITIVE-CONTROL with a PINNED, predeclared layout
cmap = [0,0,0,0,1, 0,0,0,1,1, 0,2,2,1,1, 2,2,2,2,1, 2,2,2,1,1] (same 8/8/9 counts), chosen
by the fixed rule "each attack color's non-attack complement is a SINGLE connected component"
(verified: refuge sizes 17/17/16, max-depths 5/4/4 for colors 0/1/2) — so a deep connected
elsewhere provably exists for ALL THREE colors. The layout is fixed before any run and is NOT
tuned for instrument performance.

HYPOTHESIS (H_afford, per color). With the OPTIMAL avoider, escapability is a function of
refuge depth vs pull strength: where a deep-enough connected non-attack region is reachable,
the optimal avoider drives attack-color diet well below the captured passive walker's; as
alpha rises the 1-cell/step pull eventually overwhelms any 25-cell refuge and escape fails.

NULL / FALSIFIER (H_no_elsewhere, per color). No connected refuge of sufficient depth exists
(or alpha is high enough) that even the optimal avoider cannot keep its diet off the attack
color — movement barely helps; the avoider's diet stays near the passive walker's.

PREDICTION / PREDECLARATION (verified in-silico on the real cmap + pinned G_segreg before
locking; see build spec for the verification command). Gating is PER COLOR x PER GEOMETRY x
PER ALPHA — never on a mean over the structurally-heterogeneous colors (mean-of-opposites
guard; color 2 block vs colors 0/1 scatter would cancel). Reachability of BOTH verdicts is
predeclared and verified: on G_mirro the optimal avoider ESCAPES all three colors at
alpha<=0.6 (POSITIVE reachable) and FAILS at alpha>=0.8 (gap collapses to ~0.07; NEGATIVE-
gate reachable); the per-color-at-alpha=0.5 split seen with a greedy avoider is a planning-
horizon ARTIFACT, not a hard wall, which is itself the reportable finding.

VERDICT MAP. NO-VERDICT unless both preconditions pass (PC-deficit AND PC-instrument). Given
preconditions, the verdict is a PER-COLOR vector at each gated alpha, plus the across-alpha
escapability curve and the refuge-depth covariate; the direction-level conclusion is the
structural map "escape is afforded iff a connected non-attack refuge of sufficient depth is
reachable under the pull, measured by the optimal avoider", with POSITIVE / NEGATIVE-gate /
MIXED assigned per (color, geometry, alpha) cell by the predeclared bars below.
"""
from __future__ import annotations

import json
import math
from collections import deque

import numpy as np

from active_loop.creature import World

# ---------------------------------------------------------------------------
# Predeclared parameters
# ---------------------------------------------------------------------------
BURN = 1000
MEAS = 8000
ALPHAS = [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
A_LOW = 0.5   # POSITIVE-side gate
A_HIGH = 0.8  # NEGATIVE-side gate
N_SEEDS = 20
GAMMA = 0.999
LAMBDA = 0.999  # v-store decay (exp174 L73)
COLORS = [0, 1, 2]

SEG_CMAP = [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1]


# ---------------------------------------------------------------------------
# Core API (pure, deterministic; all take a World `w` and use w.move / w.cmap)
# ---------------------------------------------------------------------------
def neighbors(w: World, cell: int):
    """The four move results (up,down,left,right); wall-clamp yields self."""
    return [w.move(cell, a) for a in range(4)]


def dist_to_attack(w: World, color: int) -> np.ndarray:
    """BFS distance to the nearest attack-color cell over the move graph."""
    n = w.n_cells
    d = np.full(n, -1, dtype=int)
    q = deque()
    for s in range(n):
        if w.cmap[s] == color:
            d[s] = 0
            q.append(s)
    while q:
        x = q.popleft()
        for y in set(neighbors(w, x)):
            if y != x and d[y] == -1:
                d[y] = d[x] + 1
                q.append(y)
    return d


def pull_cands(w: World, color: int, pos: int, d: np.ndarray):
    """Cells the pull may move to from `pos`: neighbors strictly reducing d.

    If on an attack cell (d==0) -> absorbing -> [pos]; if no strict-reducer -> [pos].
    """
    if d[pos] == 0:
        return [pos]
    cands = sorted({nb for nb in neighbors(w, pos) if nb != pos and d[nb] < d[pos]})
    return cands if cands else [pos]


def refuge_components(w: World, color: int):
    """Sorted-desc connected-non-attack component sizes over the move graph."""
    n = w.n_cells
    seen = set()
    comps = []
    for s in range(n):
        if w.cmap[s] == color or s in seen:
            continue
        q = deque([s])
        seen.add(s)
        sz = 0
        while q:
            x = q.popleft()
            sz += 1
            for y in set(neighbors(w, x)):
                if y != x and w.cmap[y] != color and y not in seen:
                    seen.add(y)
                    q.append(y)
        comps.append(sz)
    return sorted(comps, reverse=True)


def _tie_argmin(vals, cells, tie_tol: float = 1e-7) -> int:
    """Deterministic argmin: lowest cell index among near-optimal (within tie_tol).

    Without a tie tolerance, two equally-optimal next-cells would flip on FP noise
    and the greedy policy never stabilizes; the tolerance + lowest-index rule makes
    the optimal policy unique and reproducible byte-for-byte.
    """
    mn = min(vals)
    return min(c for v, c in zip(vals, cells) if v <= mn + tie_tol)


_POLICY_CACHE: dict = {}


def optimal_avoid_policy(w: World, color: int, alpha: float,
                         gamma: float = GAMMA, tol: float = 1e-9,
                         max_iter: int = 5000):
    """EXACT optimal refuge-seeking policy via POLICY ITERATION (certified).

    Minimizes long-run gamma-discounted attack-color occupancy. Transition from s:
    with prob alpha -> uniform over pull_sets[s] (the soft pull, absorbing on attack
    cells); with prob (1-alpha) -> the avoider's chosen next cell (hold ALLOWED). The
    cost (1 on an attack cell) is charged on the LANDING cell, matching the
    simulation's obs phase.

    Policy iteration is EXACT for this finite MDP: policy evaluation solves the linear
    system V = r_pi + gamma P_pi V (r_pi = P_pi @ cost); policy improvement picks the
    move minimizing q_next[m] = cost[m] + gamma*V[m] (the alpha pull term is action-
    independent), using the SAME deterministic tie-tolerant _tie_argmin on the SAME
    V[0]-anchored q as the prior relative-VI implementation (so an already-optimal cell
    yields the identical tie-break -> byte-identical simulation). We then CERTIFY
    optimality by the Bellman residual of the returned policy's value (asserted
    < 1e-7). This REPLACES the earlier relative-VI policy-stability proxy, which could
    halt before the values propagated and return a sub-optimal policy on slow-mixing
    cells (it did, on 2 positive-control cells). Cached per (cmap, color, alpha).
    """
    key = (tuple(w.cmap), int(color), round(float(alpha), 6))
    cached = _POLICY_CACHE.get(key)
    if cached is not None:
        return cached
    n = w.n_cells
    d = dist_to_attack(w, color)
    cost = np.array([1.0 if w.cmap[s] == color else 0.0 for s in range(n)])
    pull_sets = [pull_cands(w, color, s, d) for s in range(n)]
    # hold ALLOWED -> include `s` itself among the reachable next-cells.
    move_sets = [sorted(set(neighbors(w, s)) | {s}) for s in range(n)]
    # init policy: myopic max-distance move (any valid init; PI finds the exact optimum)
    policy = {s: move_sets[s][int(np.argmax([d[m] for m in move_sets[s]]))]
              for s in range(n)}
    V = np.zeros(n)
    converged = False
    for _ in range(max_iter):
        # policy evaluation: exact linear solve of V = P_pi@cost + gamma P_pi V
        P = np.zeros((n, n))
        for s in range(n):
            ps = pull_sets[s]
            wp = alpha / len(ps)
            for c in ps:
                P[s, c] += wp
            P[s, policy[s]] += (1.0 - alpha)
        V = np.linalg.solve(np.eye(n) - gamma * P, P @ cost)
        # policy improvement (alpha pull term is action-independent); V[0]-anchored q
        # reproduces the prior implementation's tie-break scale exactly.
        q_anch = cost + gamma * (V - V[0])
        new_policy = {s: _tie_argmin([q_anch[m] for m in move_sets[s]], move_sets[s])
                      for s in range(n)}
        if new_policy == policy:
            converged = True
            break
        policy = new_policy
    if not converged:
        raise RuntimeError(
            f"policy iteration did not converge: color={color} alpha={alpha}")
    # optimality certificate: Bellman optimality residual of the returned policy's V
    q_next = cost + gamma * V
    pull_v = np.array([q_next[pull_sets[s]].mean() for s in range(n)])
    pol_v = np.array([min(q_next[m] for m in move_sets[s]) for s in range(n)])
    cert_resid = float(np.max(np.abs(alpha * pull_v + (1.0 - alpha) * pol_v - V)))
    assert cert_resid < 1e-7, (
        f"optimality certificate FAILED color={color} alpha={alpha} "
        f"resid={cert_resid:.2e}")
    _POLICY_CACHE[key] = policy
    return policy


def walk(w: World, color: int, alpha: float, kind: str, seed: int,
         burn: int = BURN, meas: int = MEAS) -> dict:
    """Simulate one walk; return diet fraction + home-range + v-store diagnostics.

    RNG order per step (single stream): (1) alpha-Bernoulli, (2) pull tie-break,
    (3) policy/random move tie-break. obs read from the post-transition cell.
    """
    rng = np.random.default_rng(seed)
    n = w.n_cells
    d = dist_to_attack(w, color)
    pull_sets = [pull_cands(w, color, s, d) for s in range(n)]
    # passive: changing neighbors (no-op FORBIDDEN)
    change_sets = [sorted({nb for nb in neighbors(w, s) if nb != s}) for s in range(n)]
    if kind == "avoid":
        policy = optimal_avoid_policy(w, color, alpha)
    elif kind != "passive":
        raise ValueError(f"unknown kind {kind!r}")

    pos = 0  # deterministic start; burn-in washes it out
    visited = {}
    attack_hits = 0
    # LAMBDA-EWMA v-store (faithfulness diagnostic; one-hot belief -> weight=1.0)
    v = np.zeros(w.n_colors)
    total_steps = burn + meas
    for t in range(total_steps):
        pulled = rng.random() < alpha
        if pulled:
            cands = pull_sets[pos]
            nxt = int(cands[rng.integers(0, len(cands))])
        else:
            if kind == "passive":
                cands = change_sets[pos]
                nxt = int(cands[rng.integers(0, len(cands))])
            else:
                nxt = int(policy[pos])
        pos = nxt
        obs = int(w.cmap[pos])
        if t >= burn:
            visited[pos] = visited.get(pos, 0) + 1
            if obs == color:
                attack_hits += 1
            v *= LAMBDA
            v[obs] += 1.0  # predictability_weight = 1.0 (one-hot belief)
    f = attack_hits / meas
    homerange = len(visited)
    most_visited_frac = max(visited.values()) / meas if visited else 0.0
    vsum = v.sum()
    v_attack_share = float(v[color] / vsum) if vsum > 0 else 0.0
    argmax_v_is_attack = bool(int(np.argmax(v)) == color)
    return {
        "f": f,
        "homerange": homerange,
        "most_visited_frac": most_visited_frac,
        "v_attack_share": v_attack_share,
        "argmax_v_is_attack": argmax_v_is_attack,
    }


def greedy_avoid_policy(w: World, color: int):
    """Myopic BFS-away avoider (DIAGNOSTIC ONLY): move to max-distance neighbor."""
    n = w.n_cells
    d = dist_to_attack(w, color)
    policy = {}
    for s in range(n):
        ms = sorted(set(neighbors(w, s)) | {s})
        vals = [d[m] for m in ms]
        policy[s] = ms[int(np.argmax(vals))]
    return policy


def walk_greedy(w: World, color: int, alpha: float, seed: int,
                burn: int = BURN, meas: int = MEAS) -> float:
    """Greedy-avoider diet fraction (DIAGNOSTIC; optimal-vs-greedy contrast)."""
    rng = np.random.default_rng(seed)
    n = w.n_cells
    d = dist_to_attack(w, color)
    pull_sets = [pull_cands(w, color, s, d) for s in range(n)]
    policy = greedy_avoid_policy(w, color)
    pos = 0
    attack_hits = 0
    for t in range(burn + meas):
        pulled = rng.random() < alpha
        if pulled:
            cands = pull_sets[pos]
            pos = int(cands[rng.integers(0, len(cands))])
        else:
            pos = int(policy[pos])
        if t >= burn and w.cmap[pos] == color:
            attack_hits += 1
    return attack_hits / meas


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def run_seeds(w, color, alpha, kind):
    res = [walk(w, color, alpha, kind, seed) for seed in range(N_SEEDS)]
    return res


def stats(vals):
    a = np.asarray(vals, dtype=float)
    mean = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0
    return mean, se


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def preflight(G_mirro, G_segreg, manifest_cmap, lines):
    def p(s):
        print(s)
        lines.append(s)

    p("=" * 78)
    p("PREFLIGHT ASSERTIONS")
    p("=" * 78)
    # (1) counts
    import collections
    cm_m = dict(sorted(collections.Counter(G_mirro.cmap).items()))
    cm_s = dict(sorted(collections.Counter(G_segreg.cmap).items()))
    assert sorted(cm_m.values()) == [8, 8, 9], f"mirro counts {cm_m}"
    assert sorted(cm_s.values()) == [8, 8, 9], f"segreg counts {cm_s}"
    p(f"(1) color counts OK  mirro={cm_m}  segreg={cm_s}")
    # (2) mirro cmap byte-for-byte
    assert list(G_mirro.cmap) == list(manifest_cmap), "mirro cmap != manifest"
    p(f"(2) G_mirro.cmap == committed manifest cmap  ({list(G_mirro.cmap)})")
    # (3) seg single connected refuge per color
    for col in COLORS:
        comps = refuge_components(G_segreg, col)
        assert len(comps) == 1, f"seg color {col} refuge comps {comps}"
        p(f"(3) G_segreg color {col}: single refuge, size {comps[0]}, "
          f"maxdepth {int(dist_to_attack(G_segreg, col).max())}")
    # (4) policy converges everywhere
    for (gname, w) in [("G_mirro", G_mirro), ("G_segreg", G_segreg)]:
        for col in COLORS:
            for a in ALPHAS:
                optimal_avoid_policy(w, col, a)  # raises if not converged
    p("(4) optimal_avoid_policy is CERTIFIED Bellman-optimal (residual < 1e-7) "
      "for every (geom,color,alpha) [policy iteration + optimality-certificate assert]")
    # (5) BOTH-VERDICTS-REACHABLE pre-registration on G_mirro (in-silico, mean over seeds)
    pos_witness = None
    neg_witness = None
    for col in COLORS:
        fp_low, _ = stats([r["f"] for r in run_seeds(G_mirro, col, A_LOW, "passive")])
        fa_low, _ = stats([r["f"] for r in run_seeds(G_mirro, col, A_LOW, "avoid")])
        if (fp_low - fa_low) >= 0.20 and pos_witness is None:
            pos_witness = (col, fp_low, fa_low, fp_low - fa_low)
        fp_hi, _ = stats([r["f"] for r in run_seeds(G_mirro, col, A_HIGH, "passive")])
        fa_hi, _ = stats([r["f"] for r in run_seeds(G_mirro, col, A_HIGH, "avoid")])
        if (fp_hi - fa_hi) <= 0.10 and neg_witness is None:
            neg_witness = (col, fp_hi, fa_hi, fp_hi - fa_hi)
    assert pos_witness is not None, "no P-afford-reachable witness at alpha=0.5"
    assert neg_witness is not None, "no F-afford-reachable witness at alpha>=0.8"
    p(f"(5) BOTH-VERDICTS-REACHABLE: POSITIVE witness color={pos_witness[0]} "
      f"gap={pos_witness[3]:.3f} (a=0.5); NEGATIVE-gate witness color={neg_witness[0]} "
      f"gap={neg_witness[3]:.3f} (a=0.8)")
    p("")
    return {"pos_witness": pos_witness, "neg_witness": neg_witness}


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    manifest = json.load(open("creature/state/mirro/manifest.json"))
    G_mirro = World.from_dict(manifest["world"])
    G_segreg = World(rows=5, cols=5, n_colors=3, cmap=list(SEG_CMAP))
    geoms = [("G_mirro", G_mirro), ("G_segreg", G_segreg)]

    p("#" * 78)
    p("# Exp 270 — identity-ecological RUNG 1: the affordance gate")
    p("#   (soft-attack escapability probe on mirro's real aliased body)")
    p("#" * 78)
    p("KEY BARS:")
    p("  PC-deficit (G_mirro, per color): f_passive(0.5)-f_passive(0.0) >= 0.10 AND")
    p("    > 3*SE, in >=18/20 seeds. Else color is NO-VERDICT.")
    p("  PC-instrument (G_segreg, per color): f_passive_seg(0.5)-f_avoid_seg(0.5) >= 0.30")
    p("    AND f_passive_seg(0.5) >= 0.40, in >=18/20 seeds. Else G_mirro NO-VERDICT.")
    p("  P-afford (POSITIVE): gap = f_passive-f_avoid >= 0.20 AND > 3*SE, >=18/20 seeds,")
    p("    AND avoider homerange >= 6 AND most_visited_frac <= 0.50,")
    p("    AND ESCAPE(alpha) = gap(alpha)-gap(0) > 0 (else downgraded to MIXED).")
    p("  F-afford (NEGATIVE-gate): mean gap <= 0.10 AND gap <= 0.10 in >=18/20 seeds.")
    p("  MIXED: gap in (0.10, 0.20) band, or P-afford fails a side-condition.")
    p(f"  N_SEEDS={N_SEEDS}  BURN={BURN}  MEAS={MEAS}  GAMMA={GAMMA}  LAMBDA={LAMBDA}")
    p(f"  ALPHAS={ALPHAS}  gates: a_low={A_LOW} a_high={A_HIGH}")
    p("")

    pre = preflight(G_mirro, G_segreg, manifest["world"]["cmap"], lines)

    # ----- collect raw per-seed results: results[(gname,color,alpha,kind)] = list[dict]
    raw = {}
    for gname, w in geoms:
        for col in COLORS:
            for a in ALPHAS:
                for kind in ("passive", "avoid"):
                    raw[(gname, col, a, kind)] = run_seeds(w, col, a, kind)

    def fmeans(gname, col, a, kind):
        return [r["f"] for r in raw[(gname, col, a, kind)]]

    # ----- PC-deficit (G_mirro, per color)
    p("=" * 78)
    p("PRECONDITION: PC-deficit  (G_mirro per color; soft attack captures passive walker)")
    p("=" * 78)
    pc_deficit = {}
    for col in COLORS:
        fp0 = np.array(fmeans("G_mirro", col, 0.0, "passive"))
        fp5 = np.array(fmeans("G_mirro", col, A_LOW, "passive"))
        diff = fp5 - fp0
        mean = float(diff.mean())
        se = float(diff.std(ddof=1) / math.sqrt(N_SEEDS))
        n_pass = int(np.sum(diff >= 0.10))
        ok = (mean >= 0.10) and (mean > 3 * se) and (n_pass >= 18)
        pc_deficit[col] = ok
        p(f"  color {col}: deficit mean={mean:.4f} SE={se:.4f} 3SE={3*se:.4f} "
          f"seeds>=0.10: {n_pass}/20  -> {'PASS' if ok else 'FAIL'}")
    p("")

    # ----- PC-instrument (G_segreg, per color, RELATIVE)
    p("=" * 78)
    p("PRECONDITION: PC-instrument  (G_segreg per color; optimal avoider escapes control)")
    p("=" * 78)
    pc_instr = {}
    for col in COLORS:
        fps = np.array(fmeans("G_segreg", col, A_LOW, "passive"))
        fas = np.array(fmeans("G_segreg", col, A_LOW, "avoid"))
        gap = fps - fas
        gmean = float(gap.mean())
        gse = float(gap.std(ddof=1) / math.sqrt(N_SEEDS))
        fp_mean = float(fps.mean())
        n_pass = int(np.sum((gap >= 0.30)))
        ok = (gmean >= 0.30) and (fp_mean >= 0.40) and (n_pass >= 18)
        pc_instr[col] = ok
        p(f"  color {col}: seg gap mean={gmean:.4f} SE={gse:.4f}  "
          f"f_passive_seg(0.5)={fp_mean:.4f}  seeds gap>=0.30: {n_pass}/20 "
          f"-> {'PASS' if ok else 'FAIL'}")
    p("")

    # ----- Per-geometry per-color per-alpha tables + verdicts
    json_table = []
    verdict_vector = {}  # (gname,col) -> {alpha: verdict}
    p("=" * 78)
    p("MAIN SWEEP TABLES")
    p("=" * 78)
    for gname, w in geoms:
        for col in COLORS:
            cell_verdicts = {}
            usable = pc_deficit.get(col, False) and pc_instr.get(col, False)
            # gap at alpha=0 for ESCAPE baseline subtraction
            fp0 = np.array(fmeans(gname, col, 0.0, "passive"))
            fa0 = np.array(fmeans(gname, col, 0.0, "avoid"))
            gap0 = float((fp0 - fa0).mean())
            p(f"--- {gname}  color {col}  "
              f"(PC-deficit={'PASS' if pc_deficit.get(col) else 'FAIL'}, "
              f"PC-instrument={'PASS' if pc_instr.get(col) else 'FAIL'}, "
              f"usable={usable}) ---")
            hdr = (f"  {'alpha':>5} {'f_pass':>7} {'SE_p':>6} {'f_avoid':>7} "
                   f"{'SE_a':>6} {'gap':>7} {'SE_gap':>6} {'ESCAPE':>7} "
                   f"{'hrng':>5} {'mvfrac':>6} {'pass#':>5} {'verdict':>10}")
            p(hdr)
            for a in ALPHAS:
                pres = raw[(gname, col, a, "passive")]
                ares = raw[(gname, col, a, "avoid")]
                fp = np.array([r["f"] for r in pres])
                fa = np.array([r["f"] for r in ares])
                gap = fp - fa
                fp_m, fp_se = stats(fp)
                fa_m, fa_se = stats(fa)
                gap_m = float(gap.mean())
                gap_se = float(gap.std(ddof=1) / math.sqrt(N_SEEDS))
                escape = gap_m - gap0
                hr_m = float(np.mean([r["homerange"] for r in ares]))
                mv_m = float(np.mean([r["most_visited_frac"] for r in ares]))
                gap_sd = float(gap.std(ddof=1))
                # verdict (only meaningful for the two gated alphas + curve; assign all)
                n_pos = int(np.sum(gap >= 0.20))
                n_neg = int(np.sum(gap <= 0.10))
                verdict = "—"
                if a == 1.0:
                    verdict = "analytic1.0"
                elif a == 0.0:
                    verdict = "no-attack"
                else:
                    p_ok = (gap_m >= 0.20 and gap_m > 3 * gap_se and n_pos >= 18
                            and hr_m >= 6 and mv_m <= 0.50)
                    f_ok = (gap_m <= 0.10 and n_neg >= 18)
                    if p_ok and escape > 0:
                        verdict = "P-afford"
                    elif p_ok and escape <= 0:
                        verdict = "MIXED(esc)"
                    elif f_ok:
                        verdict = "F-afford"
                    else:
                        verdict = "MIXED"
                    # flag seed-level dispersion (mean-of-opposites at seed level)
                    if abs(gap_m) > 1e-6 and gap_sd > abs(gap_m):
                        verdict += "*disp"
                cell_verdicts[a] = verdict
                pass_n = n_pos if (a not in (0.0, 1.0)) else 0
                p(f"  {a:>5.2f} {fp_m:>7.4f} {fp_se:>6.4f} {fa_m:>7.4f} {fa_se:>6.4f} "
                  f"{gap_m:>7.4f} {gap_se:>6.4f} {escape:>7.4f} {hr_m:>5.1f} "
                  f"{mv_m:>6.3f} {pass_n:>5d} {verdict:>10}")
                json_table.append({
                    "geom": gname, "color": col, "alpha": a,
                    "f_passive": fp_m, "SE_passive": fp_se,
                    "f_avoid": fa_m, "SE_avoid": fa_se,
                    "gap": gap_m, "SE_gap": gap_se, "escape": escape,
                    "homerange": hr_m, "most_visited_frac": mv_m,
                    "gap_sd": gap_sd, "n_pos": n_pos, "n_neg": n_neg,
                    "verdict": verdict,
                })
            verdict_vector[(gname, col)] = cell_verdicts
            p("")

    # ----- Refuge-depth covariate table
    p("=" * 78)
    p("REFUGE-DEPTH COVARIATE  (max BFS dist + largest connected refuge size)")
    p("=" * 78)
    p(f"  {'geom':>9} {'color':>5} {'maxdepth':>8} {'largest_refuge':>14} "
      f"{'all_comps':>20}")
    refuge_json = []
    for gname, w in geoms:
        for col in COLORS:
            d = dist_to_attack(w, col)
            comps = refuge_components(w, col)
            p(f"  {gname:>9} {col:>5} {int(d.max()):>8} {comps[0]:>14} "
              f"{str(comps):>20}")
            refuge_json.append({"geom": gname, "color": col,
                                "maxdepth": int(d.max()),
                                "largest_refuge": comps[0],
                                "components": comps})
    p("")

    # ----- Optimal-vs-greedy contrast (G_mirro, alpha=0.5; headline methodological point)
    p("=" * 78)
    p("OPTIMAL-vs-GREEDY CONTRAST  (G_mirro, alpha=0.5; greedy fails scattered colors)")
    p("=" * 78)
    p(f"  {'color':>5} {'f_passive':>9} {'f_opt':>7} {'f_greedy':>8} "
      f"{'opt_gap':>7} {'greedy_gap':>10}")
    greedy_json = []
    for col in COLORS:
        fp_m, _ = stats(fmeans("G_mirro", col, A_LOW, "passive"))
        fo_m, _ = stats(fmeans("G_mirro", col, A_LOW, "avoid"))
        fg = [walk_greedy(G_mirro, col, A_LOW, seed) for seed in range(N_SEEDS)]
        fg_m, _ = stats(fg)
        p(f"  {col:>5} {fp_m:>9.4f} {fo_m:>7.4f} {fg_m:>8.4f} "
          f"{fp_m-fo_m:>7.4f} {fp_m-fg_m:>10.4f}")
        greedy_json.append({"color": col, "f_passive": fp_m, "f_optimal": fo_m,
                            "f_greedy": fg_m, "optimal_gap": fp_m - fo_m,
                            "greedy_gap": fp_m - fg_m})
    p("")

    # ----- v-share monotone-equivalence table (faithfulness diagnostic)
    p("=" * 78)
    p("LAMBDA-EWMA v-STORE FAITHFULNESS  (does v_attack_share track f? argmax flip?)")
    p("=" * 78)
    p(f"  {'geom':>9} {'color':>5} {'alpha':>5} {'kind':>8} {'f':>7} "
      f"{'v_share':>8} {'argmaxV=atk':>11}")
    vshare_json = []
    for gname, w in geoms:
        for col in COLORS:
            for a in (A_LOW, A_HIGH):
                for kind in ("passive", "avoid"):
                    res = raw[(gname, col, a, kind)]
                    f_m, _ = stats([r["f"] for r in res])
                    vs_m, _ = stats([r["v_attack_share"] for r in res])
                    amaxfrac = float(np.mean([1.0 if r["argmax_v_is_attack"] else 0.0
                                              for r in res]))
                    p(f"  {gname:>9} {col:>5} {a:>5.2f} {kind:>8} {f_m:>7.4f} "
                      f"{vs_m:>8.4f} {amaxfrac:>11.2f}")
                    vshare_json.append({"geom": gname, "color": col, "alpha": a,
                                        "kind": kind, "f": f_m,
                                        "v_attack_share": vs_m,
                                        "argmax_v_is_attack_frac": amaxfrac})
    p("")

    # ----- Mixing robustness (one representative cell; not a gate)
    p("=" * 78)
    p("MIXING ROBUSTNESS  (G_mirro, color 0, alpha=0.6; not a gate)")
    p("=" * 78)
    rep_w, rep_col, rep_a = G_mirro, 0, 0.6
    f_full = []
    f_half = []
    for seed in range(N_SEEDS):
        rfull = walk(rep_w, rep_col, rep_a, "passive", seed, burn=BURN, meas=MEAS)
        rhalf = walk(rep_w, rep_col, rep_a, "passive", seed, burn=BURN, meas=MEAS // 2)
        f_full.append(rfull["f"])
        f_half.append(rhalf["f"])
    f_full_m, f_full_se = stats(f_full)
    f_half_m, _ = stats(f_half)
    meas_ok = abs(f_full_m - f_half_m) <= 3 * f_full_se
    burn_fs = []
    for b in (500, 1000, 2000):
        fb = [walk(rep_w, rep_col, rep_a, "passive", seed, burn=b, meas=MEAS)["f"]
              for seed in range(N_SEEDS)]
        burn_fs.append(float(np.mean(fb)))
    burn_ok = (max(burn_fs) - min(burn_fs)) <= 3 * f_full_se
    mix_ok = meas_ok and burn_ok
    p(f"  f(MEAS)={f_full_m:.4f}+-{f_full_se:.4f}  f(MEAS/2)={f_half_m:.4f}  "
      f"|diff|<=3SE: {meas_ok}")
    p(f"  f over BURN in {{500,1000,2000}}: {[round(x,4) for x in burn_fs]}  "
      f"spread<=3SE: {burn_ok}")
    p(f"  MIXING CHECK: {'PASS' if mix_ok else 'FAIL'}")
    p("")

    # ----- alpha endpoints notes
    p("NOTE: alpha=0.0 is the no-attack baseline (no pull; pure policy/random walk).")
    p("NOTE: alpha=1.0 stated ANALYTICALLY: pull every step -> absorbing into the")
    p("      attack region, f=1.0 with zero policy DOF (not a free measurement).")
    p("")

    # ----- Final per-color VERDICT VECTOR
    p("=" * 78)
    p("FINAL PER-COLOR VERDICT VECTOR")
    p("=" * 78)
    final_verdicts = {}
    for gname, _ in geoms:
        for col in COLORS:
            usable = pc_deficit.get(col, False) and pc_instr.get(col, False)
            cv = verdict_vector[(gname, col)]
            # the direction gates G_mirro on both preconditions; G_segreg is the control
            if gname == "G_mirro" and not usable:
                tag = "NO-VERDICT (precondition failed)"
            else:
                tag = (f"a_low({A_LOW})={cv[A_LOW]}  a_high({A_HIGH})={cv[A_HIGH]}")
            p(f"  {gname} color {col}: {tag}")
            final_verdicts[f"{gname}_color{col}"] = {
                "usable": usable,
                "verdicts_by_alpha": {str(k): v for k, v in cv.items()},
            }
    p("")
    p("DIRECTION-LEVEL STRUCTURAL CONCLUSION:")
    p("  Escape is afforded iff a connected non-attack refuge of sufficient depth is")
    p("  reachable under the pull, measured by the OPTIMAL avoider. Per (color,geom,")
    p("  alpha) the bars above assign P-afford / F-afford / MIXED; gating is per-color")
    p("  (never on a structurally-heterogeneous mean). G_segreg is the positive control")
    p("  proving the instrument can detect escape when geometry permits it.")
    p("")
    printed = ("PRINTED_VERDICT: per-(geom,color,alpha) vector above; this is the SCRIPT'S "
               "claim (not the adjudicated result). See FINAL PER-COLOR VERDICT VECTOR.")
    p(printed)

    # ----- write JSON (deterministic, sorted keys)
    out = {
        "experiment": "Exp 270 — identity-ecological RUNG 1 affordance gate",
        "params": {"BURN": BURN, "MEAS": MEAS, "N_SEEDS": N_SEEDS,
                   "GAMMA": GAMMA, "LAMBDA": LAMBDA, "ALPHAS": ALPHAS,
                   "a_low": A_LOW, "a_high": A_HIGH},
        "preflight": {
            "pos_witness": list(pre["pos_witness"]),
            "neg_witness": list(pre["neg_witness"]),
        },
        "pc_deficit": {str(c): bool(pc_deficit[c]) for c in COLORS},
        "pc_instrument": {str(c): bool(pc_instr[c]) for c in COLORS},
        "table": json_table,
        "refuge_covariate": refuge_json,
        "optimal_vs_greedy": greedy_json,
        "vstore_faithfulness": vshare_json,
        "mixing_check": {"meas_ok": bool(meas_ok), "burn_ok": bool(burn_ok),
                         "pass": bool(mix_ok),
                         "f_full": f_full_m, "f_half": f_half_m,
                         "burn_fs": burn_fs},
        "final_verdicts": final_verdicts,
    }
    with open("experiments/outputs/exp270_results.json", "w") as fh:
        json.dump(out, fh, sort_keys=True, indent=2)
    p("WROTE experiments/outputs/exp270_results.json")


if __name__ == "__main__":
    main()
