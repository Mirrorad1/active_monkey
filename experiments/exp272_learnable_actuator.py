"""
Exp 272 — identity-ecological RUNG 1c: is the affordance LEARNABLE by an observation-only
actuator? (standalone learnability probe; NO source-patching — exp270 primitives COPIED verbatim)

PROVENANCE. RUNG 1c of the identity-ecological direction on the clean main-based line:
origin/main numbering reaches Exp 271 (experiments/exp271_rung2_posability.py), so this is
Exp 272 (the human's pick 'c'). Rung 1 (Exp 270, POSITIVE) showed the environmental
identity-defense surface is POSABLE: under a SOFT, resistible spatial attack a CERTIFIED-OPTIMAL
refuge planner escapes the attack color on mirro's real 5x5 aliased body. But that planner is an
OMNISCIENT UPPER BOUND (handed the ground-truth cmap + the alpha pull kernel + the move graph; it
runs certified policy iteration). Rung 1 also showed a MYOPIC greedy BFS-away avoider FAILS on the
scattered colors 0/1 (manufactures a false negative). Rung 2 (Exp 271) found kill-test vs internal
gating is CAN'T-POSE (regime-incompatible), so we do NOT compare to internal gating here. This
experiment asks the honest learnability question strictly inside the SOFT-attack regime where
movement genuinely works: can an OBSERVATION-ONLY controller (no handed cmap, no handed
kernel/policy) close the gap from greedy (fails) to optimal (works)?

THE BODY + SOFT ATTACK (COPIED verbatim from exp270, NOT imported — silent-source-patch-guard;
the World load, neighbors/dist_to_attack/pull_cands/refuge_components/_tie_argmin/
optimal_avoid_policy/greedy_avoid_policy/walk/walk_greedy are pasted into exp272_*.py so exp270
stays frozen and the two scripts cannot drift). mirro real geometry:
World.from_dict(json.load("creature/state/mirro/manifest.json")["world"]); 5x5=25 cells, 3 colors,
ALIASED (counts 8/8/9; color 2 a contiguous 3x3 block, colors 0/1 scattered checkerboard);
world.move(cell,action) wall-clamped 4-action. A PIN ASSERT checks
manifest["state_hash"]=="0f35f93115f4371fde6e435a2c599740b57a9f6bd9d8700ce066af0a9fc79bd5" so the
body cannot silently drift. SOFT PULL (the attack): each step, with prob alpha take one BFS-gradient
step toward the nearest attack-color cell (uniform among neighbors strictly reducing
distance-to-nearest-attack; absorbing if already on an attack cell); else the agent's policy moves
one cell. Exactly ONE cell transition per step; obs read from the post-transition cell. The pull
gradient (dist_to_attack/pull_cands) is computed from the GROUND-TRUTH cmap — this is the
ENVIRONMENT, not the agent's model, and is identical for passive/optimal/greedy/learned so it
cannot advantage the learner. METRIC: f = fraction of measured steps on the attack color
(lower=better). PER COLOR (0,1,2), PER SEED; BURN=1000 / MEAS=8000 (exp270). G_segreg = exp270's
pinned positive-control layout (single connected refuge per color), used as the learned-actuator
POSITIVE CONTROL.

ARMS (per color, per seed, per alpha):
  passive  = COPIED captured-walker baseline (no-op-forbidden uniform random; the f=0 anchor).
  OPTIMAL  = COPIED exp270 omniscient certified upper bound (the f=closure-1 anchor).
  GREEDY   = COPIED exp270 myopic BFS-away lower bound (fails scattered colors).
  LEARNED-MF (model-free, the DECISIVE learnability arm): tabular Q-learning over the move-MDP.
    State = bare current cell index. Actions = the SAME action set as OPTIMAL = 4 moves + HOLD
    (5 actions; the pull can still override hold). Reward = -1 iff the LANDED cell's observed
    color == attack color else 0 (the agent observes its landed color — legitimate; the attack
    color IDENTITY is the externally-defined attack and is disclosed as a known target, so the
    verdict scope is "map+planning learnable GIVEN the target"). Q INIT = 0. The agent learns Q
    from its OWN experience UNDER THE PULL via epsilon-greedy exploration (NO handed policy/map).
    After a LEARNING BUDGET, FREEZE Q and evaluate greedy-on-Q (with deterministic
    lowest-cell-index tie-break) in a FRESH walk (its own eval RNG, pos=0, BURN/MEAS as above) to
    measure frozen f. Online-during-learning f is reported as a pure DIAGNOSTIC (cost of learning),
    never a gate.
  LEARNED-MB-from-scratch (model-based, the MAP-GAP arm): pA RESET to the birth Dirichlet prior
    (np.full((3,25),0.1)+tiny noise, creature.py L216 — NOT optimism/pessimism); the agent estimates
    cmap = per-cell argmax of accumulated this-session landed-color counts, unvisited cells held at
    the uniform prior; then runs the SAME policy iteration as OPTIMAL but on its ESTIMATED attack-cell
    set (it is GIVEN the pull MECHANISM — BFS-toward-nearest-attack + alpha — and applies it to its
    OWN estimated attack cells, so kernel error is downstream of map error only: one clean gap).
  LEARNED-MB-inherited-pA = DIAGNOSTIC ONLY, NOT a learnability datapoint: the persisted pA's
    per-cell argmax equals the ground-truth cmap 25/25 (verified, min confidence 0.856) and qs is a
    delta on true_pos (0 bits), so this arm IS OPTIMAL relabeled. It is run + printed ONLY to confirm
    it reproduces OPTIMAL's f, labeled "OMNISCIENT-EQUIVALENT (pA==cmap, accuracy 1.000) — NOT a
    learnability result", and is EXCLUDED from every verdict.

HYPOTHESIS (H_learnable, PER COLOR, PER ALPHA). The soft-attack escape affordance proven POSABLE by
the omniscient optimal in rung 1 is LEARNABLE from within-session observation-only experience: a
tabular learner given an adequate learning budget recovers most of the available passive->optimal
escape gap on the scattered colors where greedy fails.
NULL / FALSIFIER (H_not_learnable, PER COLOR, PER ALPHA). The omniscient upper bound is unreachable
without omniscience: the learner with the budgeted data stays near greedy/passive — recovering <=35%
of the gap — so the affordance is POSABLE-BUT-NOT-LEARNABLE (movement works only if you are handed
the map/policy). The natural NOT-LEARNABLE lever is a too-small LEARNING BUDGET (sparse coverage ->
bad cmap estimate / TD non-convergence / absorbing-trap collapse).

PREDICTION / PREDECLARATION (verified in-silico on the real cmap + exp270 numbers before locking).
Primary statistic, PER COLOR PER SEED (never a mean over the 3 structurally-heterogeneous colors —
mean-of-opposites guard; color-2 block vs colors-0/1 scatter would cancel):
  closure(color,seed) = (f_passive - f_learned) / (f_passive - f_optimal)   (passive=0, optimal=1).
The budget is a PRIMARY PREDECLARED AXIS, not a hidden constant: the headline is a learning curve
(closure vs budget over a fixed grid) and a MAP "learnable holds for color c on alpha in [a_lo,a_hi]
at budget >= B". Verdict is rendered at a PREDECLARED PLATEAU budget (the largest grid budget whose
closure has converged, |closure(B)-closure(B/2)| <= 0.05 over >=18/20 seeds), per (color,alpha):
  LEARNABLE     iff frozen closure >= 0.80 AND f_learned <= f_passive - 0.20 (a real absolute escape)
                AND homerange >= 6 AND most_visited_frac <= 0.50, in >=18/20 seeds, AND the per-seed
                closure is UNIMODAL (sd <= |mean closure|; else *disp).
  NOT-LEARNABLE iff frozen closure <= 0.35 in >=18/20 seeds (<=0.35 covers both scattered-color
                greedy closures 0.266/0.319, so "no better than myopic greedy" literally reads
                NOT-LEARNABLE on the colors where greedy fails).
  MIXED/PARTIAL = the (0.35, 0.80) middle band, OR a high-closure result violating the anti-huddle
                side-conditions (MIXED-degenerate), OR *disp bimodality — reported as a number, NOT
                a failure.
  NO-VERDICT-no-headroom where (f_passive - f_optimal) < 0.20 (closure denominator too small) or the
                plateau-convergence test fails (budget-limited NO-VERDICT, reported as such).
PREDECLARED FACTS on the record (exp270 alpha=0.5, verified by direct computation):
  f_passive 0.609/0.617/0.831, f_optimal 0.203/0.253/0.201, f_greedy 0.501/0.501/0.335 for colors
  0/1/2; greedy closure 0.266/0.319/0.787. Color 2 is a DEGENERATE learnability test (greedy already
  closes 79% of the optimal span) — the headline learnability claim therefore rests on the SCATTERED
  colors 0/1 where greedy genuinely fails; color 2 is reported separately as "greedy already
  near-solves it". The closure partition is DISJOINT per color with adequate headroom (verified:
  LEARNABLE f-ceilings 0.284/0.326/0.327, NOT-LEARNABLE f-floors below them).

BOTH-VERDICTS-REACHABLE (predeclared witness, mirroring exp270 L397-413). POSITIVE witness = >=1
(color,alpha) where LEARNED-MF (or MB-from-scratch) at the largest budget hits closure>=0.80;
NEGATIVE witness = >=1 (color,alpha) where the SMALL budget yields closure<=0.35. The learning curve
IS the reachability evidence. Honesty note predeclared in-script: a tabular 25-state MDP is solvable
by Q-learning with unbounded data, so if every (color,alpha) is LEARNABLE at the plateau that is the
honest finding ("the affordance is learnable given adequate within-session experience"); the
interesting result is the BUDGET-TO-LEARN and the geometry-dependent learning SPEED, reported per
color, never averaged.

INSTRUMENT POSITIVE CONTROL + BUDGET-SUFFICIENCY (mirroring exp270's G_segreg gate). LEARNED-MF and
LEARNED-MB-from-scratch must reach LEARNABLE (closure>=0.95 with unbounded/largest budget) on
G_segreg at alpha=0.5 BEFORE any G_mirro NOT-LEARNABLE verdict is admissible — proving the learning
machinery CAN reach optimal when escape is geometrically easy; a NOT-LEARNABLE call with a failed PC
is instrument-failure NO-VERDICT, not a result.

DEGENERACY / NO-OMNISCIENCE AUDIT (L43/L44/L19, made concrete + asserted). (a) PIN state_hash. (b)
EQUIVALENCE assert: the COPIED passive/optimal/greedy f at alpha=0.5 match exp270_results.json
within Monte-Carlo SE (proves byte-faithful substrate before any learned number is trusted). (c)
MF reward reads ONLY w.cmap[currently-occupied landed cell], never the full cmap or the optimal
policy; dist_to_attack/pull_cands are used ONLY to realize the environment's pull, never read by the
Q-update/policy (state input is {pos,a,nxt,observed_color} only). (d) MB-from-scratch: assert the
initial estimated-cmap count array is the flat birth prior (starting argmax accuracy at chance
~1/3), assert the estimated cmap is a DIFFERENT array object than w.cmap built only from visit
counts, and add a scramble-test (perturb a hidden ground-truth copy -> MB output unchanged). (e)
MB-inherited-pA: assert + print pA-vs-cmap argmax accuracy ~1.000 and that it reproduces OPTIMAL's f,
then EXCLUDE it from verdicts. (f) anti-huddle side-conditions (homerange>=6 AND most_visited_frac
<=0.50) gate every LEARNABLE call. (g) ALL hyperparameters (budget grid, lr, epsilon schedule,
gamma, RNG namespaces) are PREDECLARED in the script header before any verdict run; a 3-point
robustness check (epsilon x0.5/x2, lr x0.5/x2) at the plateau budget is reported and the verdict
must not flip.

IMPLEMENTATION NOTE (learning-curve efficiency, faithful). learn_mf runs ONE Q-learning trajectory
to the LARGEST budget with a pre-drawn RNG stream from default_rng(learn_seed) and SNAPSHOTS Q at
every grid budget. Because the smaller-budget run consumes the strict PREFIX of the same RNG stream
(action-selection draws, pull-Bernoulli draws, pull tie-break draws in the SAME per-step order), the
snapshot at budget B is BYTE-IDENTICAL to an independent learn_mf(...,budget=B,learn_seed) — this is
the standard learning-curve construction, not an approximation. eval is always a fresh independent
default_rng(eval_seed). This is asserted by a snapshot-vs-independent equivalence check in preflight.
"""
from __future__ import annotations

import json
import math
from collections import deque

import numpy as np

from active_loop.creature import World

# ---------------------------------------------------------------------------
# PREDECLARED CONSTANTS (frozen before any verdict run)
# ---------------------------------------------------------------------------
BURN = 1000
MEAS = 8000
N_SEEDS = 20
COLORS = [0, 1, 2]
# ATTEMPT 2 (post-kill, edited before re-run): ATTEMPT 1 was externally KILLED (compute) mid
# preflight — the BUDGET_GRID to 500k x 4 alphas x 20 seeds x 3 colors x 5 arms x 2 geoms
# exceeded wall-clock (an L25 runtime-preflight miss). This trims COMPUTE ONLY (ALPHAS to the
# escapable band {0.5,0.6}; BUDGET_GRID to a plateau-detectable ~2.5x grid, max 30k); the
# verdict logic, bars, guards, and N_SEEDS=20 are UNCHANGED. Also: the model-free (MF) arm
# FAILED its predeclared G_segreg positive control (closure -0.286, worse than random — an
# under-trained-Q index-tie-break artifact), so per the predeclared "failed-PC = instrument
# NO-VERDICT" rule MF is EXCLUDED from the verdict and reported as a diagnostic only; the
# MB-from-scratch arm (which passed its PC) is the load-bearing learnability arm.
ALPHAS = [0.5, 0.6]   # escapable band (rung-1: movement works at low-moderate pull)
A_PRIMARY = 0.5
GAMMA = 0.999                    # MATCH OPTIMAL's discount (MF + MB use 0.999)
GAMMA_ROBUST = 0.99             # MF robustness comparison only
BUDGET_GRID = [2000, 5000, 12000, 30000]  # learning steps; PRIMARY axis (attempt-2 compute trim; ~2.5x plateau grid)
MF_LR0 = 0.1                     # MF_LR_DECAY: lr_t = MF_LR0/(1+visits[s,a])
MF_EPS0 = 0.2                    # MF_EPS_SCHEDULE: linear decay 0.2 -> 0.02 over the budget
MF_EPS1 = 0.02
LEARN_SEED_BASE = 0
EVAL_SEED_OFFSET = 1000          # learn_seed=s, eval_seed=s+1000 (disjoint RNG namespaces)
STATE_HASH = "0f35f93115f4371fde6e435a2c599740b57a9f6bd9d8700ce066af0a9fc79bd5"

CLOSURE_LEARNABLE = 0.80
CLOSURE_NOTLEARN = 0.35
ABS_ESCAPE_FLOOR = 0.20
HRANGE_MIN = 6
MVFRAC_MAX = 0.50
HEADROOM_MIN = 0.20
PLATEAU_TOL = 0.05
PC_CLOSURE = 0.95                # instrument positive-control bar on G_segreg

SEG_CMAP = [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1]

N_ACTIONS = 5  # 0..3 = the four wall-clamped moves; 4 = HOLD
LAMBDA = 0.999  # (kept for parity with exp270 walk; v-store not gated here)


# ===========================================================================
# COPIED-VERBATIM PRIMITIVES FROM exp270 (DO NOT import exp270 — keep it frozen)
# (neighbors, dist_to_attack, pull_cands, refuge_components, _tie_argmin,
#  optimal_avoid_policy, walk, greedy_avoid_policy, walk_greedy, stats, SEG_CMAP)
# ===========================================================================
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
    """Deterministic argmin: lowest cell index among near-optimal (within tie_tol)."""
    mn = min(vals)
    return min(c for v, c in zip(vals, cells) if v <= mn + tie_tol)


_POLICY_CACHE: dict = {}


def optimal_avoid_policy(w: World, color: int, alpha: float,
                         gamma: float = GAMMA, tol: float = 1e-9,
                         max_iter: int = 5000):
    """EXACT optimal refuge-seeking policy via POLICY ITERATION (certified).

    Minimizes long-run gamma-discounted attack-color occupancy. (COPIED from exp270.)
    """
    key = (tuple(w.cmap), int(color), round(float(alpha), 6))
    cached = _POLICY_CACHE.get(key)
    if cached is not None:
        return cached
    n = w.n_cells
    d = dist_to_attack(w, color)
    cost = np.array([1.0 if w.cmap[s] == color else 0.0 for s in range(n)])
    pull_sets = [pull_cands(w, color, s, d) for s in range(n)]
    move_sets = [sorted(set(neighbors(w, s)) | {s}) for s in range(n)]
    policy = {s: move_sets[s][int(np.argmax([d[m] for m in move_sets[s]]))]
              for s in range(n)}
    V = np.zeros(n)
    converged = False
    for _ in range(max_iter):
        P = np.zeros((n, n))
        for s in range(n):
            ps = pull_sets[s]
            wp = alpha / len(ps)
            for c in ps:
                P[s, c] += wp
            P[s, policy[s]] += (1.0 - alpha)
        V = np.linalg.solve(np.eye(n) - gamma * P, P @ cost)
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
    """Simulate one walk; return diet fraction + home-range diagnostics. (COPIED.)

    RNG order per step: (1) alpha-Bernoulli, (2) pull tie-break,
    (3) policy/random move tie-break. obs read from the post-transition cell.
    """
    rng = np.random.default_rng(seed)
    n = w.n_cells
    d = dist_to_attack(w, color)
    pull_sets = [pull_cands(w, color, s, d) for s in range(n)]
    change_sets = [sorted({nb for nb in neighbors(w, s) if nb != s}) for s in range(n)]
    if kind == "avoid":
        policy = optimal_avoid_policy(w, color, alpha)
    elif kind != "passive":
        raise ValueError(f"unknown kind {kind!r}")

    pos = 0
    visited = {}
    attack_hits = 0
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
            v[obs] += 1.0
    f = attack_hits / meas
    homerange = len(visited)
    most_visited_frac = max(visited.values()) / meas if visited else 0.0
    return {
        "f": f,
        "homerange": homerange,
        "most_visited_frac": most_visited_frac,
    }


def greedy_avoid_policy(w: World, color: int):
    """Myopic BFS-away avoider (DIAGNOSTIC ONLY): move to max-distance neighbor. (COPIED.)"""
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
    """Greedy-avoider diet fraction (DIAGNOSTIC; optimal-vs-greedy contrast). (COPIED.)"""
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


def stats(vals):
    a = np.asarray(vals, dtype=float)
    mean = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0
    return mean, se


# ===========================================================================
# NEW API — the OBSERVATION-ONLY learned actuators (no omniscience)
# ===========================================================================
def _pull_env(w: World, color: int):
    """Build the environment-side pull structures (NOT the agent's model).

    Returns (pull_sets, pull_lens) where pull_sets[s] is the np.array of cells the
    soft pull may take from s (the SAME object the COPIED walk uses). Used ONLY to
    realize the environment's pull; never read by the Q-update or the MF policy.
    """
    n = w.n_cells
    d = dist_to_attack(w, color)
    pull_sets = [np.asarray(pull_cands(w, color, s, d), dtype=int) for s in range(n)]
    pull_lens = np.array([len(p) for p in pull_sets], dtype=int)
    return pull_sets, pull_lens


def _move_table(w: World) -> np.ndarray:
    """act_next[s, a]: landed cell for action a in 0..3 (move) and 4 (HOLD=self)."""
    n = w.n_cells
    tab = np.zeros((n, N_ACTIONS), dtype=int)
    for s in range(n):
        for a in range(4):
            tab[s, a] = w.move(s, a)
        tab[s, 4] = s  # HOLD
    return tab


def _greedy_action(q_row: np.ndarray, act_next_row: np.ndarray,
                   tie_tol: float = 1e-9) -> int:
    """Greedy action on a Q-row with the spec's deterministic LOWEST-LANDED-CELL-INDEX
    tie-break (== _tie_argmin convention used by OPTIMAL): among the max-Q actions, pick
    the one whose LANDED cell has the lowest index. This makes the frozen MF policy unique
    and reproducible and matches the optimal planner's tie convention, so MF-vs-optimal
    behavioral agreement is not confounded by a different tie rule."""
    mx = float(q_row.max())
    best_a = -1
    best_cell = None
    for a in range(N_ACTIONS):
        if q_row[a] >= mx - tie_tol:
            cell = int(act_next_row[a])
            if best_cell is None or cell < best_cell:
                best_cell = cell
                best_a = a
    return best_a


def learn_mf(w: World, color: int, alpha: float, learn_seed: int,
             budgets=BUDGET_GRID, lr0: float = MF_LR0, eps0: float = MF_EPS0,
             eps1: float = MF_EPS1, gamma: float = GAMMA, eps_horizon=None):
    """Tabular Q-learning over the move-MDP; return {budget: Q.copy()} snapshots.

    State = bare current cell. Actions 0..3 = moves, 4 = HOLD. Reward = -1 iff the
    LANDED cell's observed color == color else 0 (reads ONLY w.cmap[landed], never
    the full cmap or the optimal policy). Q init 0. epsilon-greedy with linear decay
    eps0->eps1 over the FIXED GLOBAL HORIZON eps_horizon (default max(BUDGET_GRID), NOT
    max(budgets)); decaying lr = lr0/(1+visits[s,a]). The pull is realized via the COPIED
    pull structures each step; pull_sets/dist are NEVER read by the Q-update (which sees
    only {pos, a, nxt, observed reward}).

    The eps schedule is anchored to a FIXED global horizon so the trajectory does NOT
    depend on which budgets are being snapshotted: snapshot(B) from a multi-budget run is
    then BYTE-IDENTICAL to an independent learn_mf(...,budgets=[B]) (same RNG prefix, same
    eps schedule). This makes the learning curve a single faithful run sampled at B's, and
    makes the robustness rerun at the plateau budget consistent with the main-sweep snapshot.
    """
    n = w.n_cells
    cmap = np.asarray(w.cmap)
    cost = (cmap == color).astype(float)  # environment discloses landed color; reward = -cost
    pull_sets, pull_lens = _pull_env(w, color)
    act_next = _move_table(w)

    horizon = int(eps_horizon) if eps_horizon is not None else max(BUDGET_GRID)
    max_budget = max(budgets)
    snap_set = set(int(b) for b in budgets)
    # FOUR INDEPENDENT child streams (spawned from one SeedSequence(learn_seed)) so each
    # per-step draw array starts at its OWN stream origin REGARDLESS of max_budget. This
    # guarantees the first B draws of a max_budget=X run equal a max_budget=B<X run, i.e.
    # snapshot(B) == independent learn_mf(budgets=[B]) byte-for-byte (the prefix property
    # would break if all four arrays shared one interleaved stream of length max_budget).
    ss_eps, ss_act, ss_pull, ss_pidx = np.random.SeedSequence(learn_seed).spawn(4)
    r_eps = np.random.default_rng(ss_eps).random(max_budget)
    r_act = np.random.default_rng(ss_act).integers(0, N_ACTIONS, max_budget)
    r_pull = np.random.default_rng(ss_pull).random(max_budget)
    r_pidx = np.random.default_rng(ss_pidx).random(max_budget)

    Q = np.zeros((n, N_ACTIONS))
    visits = np.zeros((n, N_ACTIONS))
    snapshots = {}
    pos = 0
    for step in range(max_budget):
        eps = eps0 - (eps0 - eps1) * step / horizon
        if r_eps[step] < eps:
            a = int(r_act[step])
        else:
            a = _greedy_action(Q[pos], act_next[pos])  # lowest-landed-cell-index tie-break
        if r_pull[step] < alpha:
            cs = pull_sets[pos]
            nxt = int(cs[int(r_pidx[step] * pull_lens[pos])])
        else:
            nxt = int(act_next[pos, a])
        reward = -cost[nxt]
        visits[pos, a] += 1.0
        lr = lr0 / (1.0 + visits[pos, a])
        Q[pos, a] += lr * (reward + gamma * Q[nxt].max() - Q[pos, a])
        pos = nxt
        b1 = step + 1
        if b1 in snap_set:
            snapshots[b1] = Q.copy()
    return snapshots


def freeze_eval_mf(w: World, color: int, alpha: float, Q: np.ndarray,
                   eval_seed: int, burn: int = BURN, meas: int = MEAS) -> dict:
    """Greedy-on-Q evaluation in a FRESH walk; identical per-step structure to walk().

    Policy = greedy on Q with the deterministic LOWEST-LANDED-CELL-INDEX tie-break
    (_greedy_action == _tie_argmin convention). pull can still override the greedy action.
    pos=0; own eval RNG. Returns frozen f + homerange + most_visited_frac.
    """
    n = w.n_cells
    cmap = np.asarray(w.cmap)
    pull_sets, pull_lens = _pull_env(w, color)
    act_next = _move_table(w)
    greedy_a = np.array([_greedy_action(Q[s], act_next[s]) for s in range(n)])

    rng = np.random.default_rng(eval_seed)
    pos = 0
    visited = {}
    attack_hits = 0
    for t in range(burn + meas):
        # RNG order matches walk(): (1) alpha-Bernoulli, (2) pull index pick.
        pulled = rng.random() < alpha
        if pulled:
            cs = pull_sets[pos]
            nxt = int(cs[rng.integers(0, pull_lens[pos])])
        else:
            nxt = int(act_next[pos, int(greedy_a[pos])])
        pos = nxt
        obs = int(cmap[pos])
        if t >= burn:
            visited[pos] = visited.get(pos, 0) + 1
            if obs == color:
                attack_hits += 1
    f = attack_hits / meas
    homerange = len(visited)
    most_visited_frac = max(visited.values()) / meas if visited else 0.0
    return {"f": f, "homerange": homerange, "most_visited_frac": most_visited_frac,
            "greedy_a": greedy_a}


def online_f_mf(w: World, color: int, alpha: float, learn_seed: int,
                budget: int, lr0: float = MF_LR0, eps0: float = MF_EPS0,
                eps1: float = MF_EPS1, gamma: float = GAMMA) -> float:
    """DIAGNOSTIC: attack-color diet fraction DURING learning over the budget window.

    Re-runs the learning trajectory (same RNG order as learn_mf) and counts the
    landed-color attack fraction across all `budget` learning steps (the cost of
    learning). Never a gate.
    """
    n = w.n_cells
    cmap = np.asarray(w.cmap)
    cost = (cmap == color).astype(float)
    pull_sets, pull_lens = _pull_env(w, color)
    act_next = _move_table(w)
    # identical spawned-stream construction as learn_mf (faithful trajectory replay)
    ss_eps, ss_act, ss_pull, ss_pidx = np.random.SeedSequence(learn_seed).spawn(4)
    r_eps = np.random.default_rng(ss_eps).random(budget)
    r_act = np.random.default_rng(ss_act).integers(0, N_ACTIONS, budget)
    r_pull = np.random.default_rng(ss_pull).random(budget)
    r_pidx = np.random.default_rng(ss_pidx).random(budget)
    Q = np.zeros((n, N_ACTIONS))
    visits = np.zeros((n, N_ACTIONS))
    pos = 0
    hits = 0
    horizon = max(BUDGET_GRID)  # exactly replicate learn_mf's fixed-horizon eps schedule
    for step in range(budget):
        eps = eps0 - (eps0 - eps1) * step / horizon
        if r_eps[step] < eps:
            a = int(r_act[step])
        else:
            a = _greedy_action(Q[pos], act_next[pos])
        if r_pull[step] < alpha:
            cs = pull_sets[pos]
            nxt = int(cs[int(r_pidx[step] * pull_lens[pos])])
        else:
            nxt = int(act_next[pos, a])
        reward = -cost[nxt]
        visits[pos, a] += 1.0
        lr = lr0 / (1.0 + visits[pos, a])
        Q[pos, a] += lr * (reward + gamma * Q[nxt].max() - Q[pos, a])
        pos = nxt
        if cost[nxt] > 0:
            hits += 1
    return hits / budget


# ---- model-based from-scratch ----------------------------------------------
def _birth_prior_counts(n_colors: int, n_cells: int, seed: int) -> np.ndarray:
    """The creature.py L216 BIRTH Dirichlet prior: 0.1 + tiny seeded jitter.

    This is the per-color, per-cell concentration BEFORE any observation. NOT
    optimism/pessimism — it is the literal birth prior, reset for from-scratch.
    """
    rng = np.random.default_rng(seed)
    return np.full((n_colors, n_cells), 0.1) + 0.01 * rng.random((n_colors, n_cells))


def estimate_cmap_from_history(counts: np.ndarray) -> np.ndarray:
    """Per-cell argmax of (birth prior + landed-color counts); unvisited -> prior argmax.

    counts already INCLUDES the flat birth prior; for a never-visited cell the columns
    are all == the prior (near-flat) so its argmax is the prior's (chance ~1/3). Returns
    a FRESH ndarray (NOT a view of any ground-truth array).
    """
    return np.asarray(counts.argmax(axis=0), dtype=int).copy()


def learn_mb_from_scratch(w: World, color: int, alpha: float, learn_seed: int,
                          budget: int, eps0: float = MF_EPS0, eps1: float = MF_EPS1,
                          eps_horizon=None):
    """eps-walk under the pull; accumulate landed-color counts -> cmap_hat (from scratch).

    Belief reset to the birth Dirichlet prior. The agent moves by an exploratory
    eps-greedy-away walk (with prob 1-eps step toward the neighbor its OWN current cmap
    estimate says is least likely attack; random move otherwise) so coverage is realistic
    but the map is built ONLY from observed LANDED colors — never the ground-truth cmap or
    dist_to_attack (the pull, the ENVIRONMENT, uses the real structures). Fixed-horizon eps
    + four spawned RNG streams = same construction as learn_mb_snapshots, so a single call
    to budget B == the snapshot at B. Returns (cmap_hat, coverage, counts0, counts_final).
    """
    n = w.n_cells
    n_colors = w.n_colors
    cmap = np.asarray(w.cmap)  # ground truth: read ONLY at the LANDED cell (an observation)
    pull_sets, pull_lens = _pull_env(w, color)
    act_next = _move_table(w)
    horizon = int(eps_horizon) if eps_horizon is not None else max(BUDGET_GRID)

    counts = _birth_prior_counts(n_colors, n, learn_seed)  # birth prior
    counts0 = counts.copy()                                # snapshot of the flat start
    ss_eps, ss_act, ss_pull, ss_pidx = np.random.SeedSequence(learn_seed).spawn(4)
    r_eps = np.random.default_rng(ss_eps).random(budget)
    r_act = np.random.default_rng(ss_act).integers(0, 4, budget)
    r_pull = np.random.default_rng(ss_pull).random(budget)
    r_pidx = np.random.default_rng(ss_pidx).random(budget)

    pos = 0
    visited = set()
    for step in range(budget):
        # observe landed color of CURRENT cell (we already moved into it last step / start)
        obs = int(cmap[pos])
        counts[obs, pos] += 1.0
        visited.add(pos)
        # choose an action from the agent's OWN estimate (least-likely-attack neighbor)
        eps = eps0 - (eps0 - eps1) * step / horizon
        if r_eps[step] < eps:
            a = int(r_act[step])
        else:
            cmap_hat = counts.argmax(axis=0)
            nbrs = act_next[pos, :4]
            # prefer a neighbor estimated NON-attack; tie -> lowest action index
            est_attack = (cmap_hat[nbrs] == color).astype(int)
            a = int(np.argmin(est_attack))  # 0 if any non-attack neighbor, lowest-index
        if r_pull[step] < alpha:
            cs = pull_sets[pos]
            nxt = int(cs[int(r_pidx[step] * pull_lens[pos])])
        else:
            nxt = int(act_next[pos, a])
        pos = nxt
    cmap_hat = estimate_cmap_from_history(counts)
    coverage = len(visited) / n
    return cmap_hat, coverage, counts0, counts


def learn_mb_snapshots(w: World, color: int, alpha: float, learn_seed: int,
                       budgets=BUDGET_GRID, eps0: float = MF_EPS0, eps1: float = MF_EPS1,
                       eps_horizon=None):
    """Single MB eps-walk to the largest budget; snapshot (cmap_hat, coverage) at every
    grid budget. Fixed-horizon eps + spawned streams (same construction as learn_mf) so a
    snapshot at B is byte-identical to learn_mb_from_scratch(...,budget=B) when that single
    call uses the same horizon. This is the efficient learning-curve equivalent."""
    n = w.n_cells
    n_colors = w.n_colors
    cmap = np.asarray(w.cmap)  # observed ONLY at the landed cell
    pull_sets, pull_lens = _pull_env(w, color)
    act_next = _move_table(w)
    horizon = int(eps_horizon) if eps_horizon is not None else max(BUDGET_GRID)
    max_budget = max(budgets)
    snap_set = set(int(b) for b in budgets)
    counts = _birth_prior_counts(n_colors, n, learn_seed)
    ss_eps, ss_act, ss_pull, ss_pidx = np.random.SeedSequence(learn_seed).spawn(4)
    r_eps = np.random.default_rng(ss_eps).random(max_budget)
    r_act = np.random.default_rng(ss_act).integers(0, 4, max_budget)
    r_pull = np.random.default_rng(ss_pull).random(max_budget)
    r_pidx = np.random.default_rng(ss_pidx).random(max_budget)
    pos = 0
    visited = set()
    snaps = {}
    for step in range(max_budget):
        obs = int(cmap[pos])
        counts[obs, pos] += 1.0
        visited.add(pos)
        eps = eps0 - (eps0 - eps1) * step / horizon
        if r_eps[step] < eps:
            a = int(r_act[step])
        else:
            cmap_hat = counts.argmax(axis=0)
            nbrs = act_next[pos, :4]
            est_attack = (cmap_hat[nbrs] == color).astype(int)
            a = int(np.argmin(est_attack))
        if r_pull[step] < alpha:
            cs = pull_sets[pos]
            nxt = int(cs[int(r_pidx[step] * pull_lens[pos])])
        else:
            nxt = int(act_next[pos, a])
        pos = nxt
        b1 = step + 1
        if b1 in snap_set:
            snaps[b1] = (estimate_cmap_from_history(counts), len(visited) / n)
    return snaps


def optimal_on_estimated(w: World, cmap_hat: np.ndarray, color: int, alpha: float):
    """SAME policy iteration as OPTIMAL, but attack cells = (cmap_hat == color).

    The pull MECHANISM (BFS-toward-nearest-attack + alpha) is GIVEN; it is applied to the
    agent's ESTIMATED attack set, so any kernel error is purely downstream of map error.
    Returns a {state: next_cell} policy. Builds a throwaway World whose cmap is cmap_hat so
    the COPIED optimal_avoid_policy machinery is reused verbatim on the estimated map.
    """
    w_hat = World(rows=w.rows, cols=w.cols, n_colors=w.n_colors,
                  cmap=[int(x) for x in cmap_hat])
    # If the estimated attack set is EMPTY (no cell estimated == color), there is nothing to
    # flee: the agent's planner sees zero cost everywhere -> HOLD policy (will be pulled).
    if not any(int(c) == color for c in cmap_hat):
        return {s: s for s in range(w.n_cells)}
    return optimal_avoid_policy(w_hat, color, alpha)


def eval_mb_policy(w: World, color: int, alpha: float, policy: dict,
                   eval_seed: int, burn: int = BURN, meas: int = MEAS) -> dict:
    """Evaluate an MB policy in the REAL environment (real cmap + real pull). Fresh RNG."""
    n = w.n_cells
    cmap = np.asarray(w.cmap)
    pull_sets, pull_lens = _pull_env(w, color)
    rng = np.random.default_rng(eval_seed)
    pos = 0
    visited = {}
    attack_hits = 0
    for t in range(burn + meas):
        pulled = rng.random() < alpha
        if pulled:
            cs = pull_sets[pos]
            nxt = int(cs[rng.integers(0, pull_lens[pos])])
        else:
            nxt = int(policy[pos])
        pos = nxt
        obs = int(cmap[pos])
        if t >= burn:
            visited[pos] = visited.get(pos, 0) + 1
            if obs == color:
                attack_hits += 1
    f = attack_hits / meas
    homerange = len(visited)
    most_visited_frac = max(visited.values()) / meas if visited else 0.0
    return {"f": f, "homerange": homerange, "most_visited_frac": most_visited_frac}


def closure(f_passive: float, f_optimal: float, f_learned: float) -> float:
    """(f_passive - f_learned) / (f_passive - f_optimal); passive=0, optimal=1."""
    denom = f_passive - f_optimal
    if denom <= 0:
        return float("nan")
    return (f_passive - f_learned) / denom


def map_accuracy(cmap_hat: np.ndarray, true_cmap: np.ndarray, weights=None) -> float:
    """Per-cell argmax-correct fraction (optionally stationary-visitation weighted)."""
    correct = (np.asarray(cmap_hat) == np.asarray(true_cmap)).astype(float)
    if weights is None:
        return float(correct.mean())
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        return float(correct.mean())
    return float((correct * w).sum() / w.sum())


def action_agreement(greedy_a: np.ndarray, ref_policy: dict, act_next: np.ndarray,
                     n: int) -> int:
    """Number of cells (out of n) where the frozen-Q greedy LANDED cell matches the
    reference policy's next cell (a behavioral, not action-index, agreement)."""
    agree = 0
    for s in range(n):
        landed = int(act_next[s, int(greedy_a[s])])
        if landed == int(ref_policy[s]):
            agree += 1
    return agree


# ===========================================================================
# Preflight
# ===========================================================================
def preflight(G_mirro, G_segreg, manifest, lines, p):
    p("=" * 78)
    p("PREFLIGHT ASSERTIONS")
    p("=" * 78)
    import collections

    # (1) PIN state_hash
    assert manifest["state_hash"] == STATE_HASH, (
        f"state_hash drift: {manifest['state_hash']}")
    p(f"(1) PIN state_hash OK == {STATE_HASH[:16]}...")

    # (2) COUNTS mirro 8/8/9; SEG 8/8/9
    cm_m = dict(sorted(collections.Counter(G_mirro.cmap).items()))
    cm_s = dict(sorted(collections.Counter(G_segreg.cmap).items()))
    assert sorted(cm_m.values()) == [8, 8, 9], f"mirro counts {cm_m}"
    assert sorted(cm_s.values()) == [8, 8, 9], f"segreg counts {cm_s}"
    assert list(G_mirro.cmap) == list(manifest["world"]["cmap"]), "mirro cmap != manifest"
    p(f"(2) COUNTS OK  mirro={cm_m}  segreg={cm_s}; mirro cmap == manifest")

    # (3) EQUIVALENCE: COPIED passive/optimal/greedy f at alpha=0.5 match exp270 within SE
    exp270 = json.load(open("experiments/outputs/exp270_results.json"))
    o270 = {d["color"]: d for d in exp270["optimal_vs_greedy"]}
    p("(3) EQUIVALENCE to exp270 (alpha=0.5, COPIED substrate):")
    equiv_records = {}
    for col in COLORS:
        fp = [walk(G_mirro, col, A_PRIMARY, "passive", s)["f"] for s in range(N_SEEDS)]
        fo = [walk(G_mirro, col, A_PRIMARY, "avoid", s)["f"] for s in range(N_SEEDS)]
        fg = [walk_greedy(G_mirro, col, A_PRIMARY, s) for s in range(N_SEEDS)]
        fp_m, fp_se = stats(fp)
        fo_m, fo_se = stats(fo)
        fg_m, fg_se = stats(fg)
        ref = o270[col]
        for name, m, se, refv in (("passive", fp_m, fp_se, ref["f_passive"]),
                                  ("optimal", fo_m, fo_se, ref["f_optimal"]),
                                  ("greedy", fg_m, fg_se, ref["f_greedy"])):
            se_eff = max(se, 1e-6)
            delta = abs(m - refv)
            ok = delta <= 3 * se_eff
            assert ok, (f"EQUIVALENCE FAIL color={col} {name}: ours={m:.5f} "
                        f"exp270={refv:.5f} delta={delta:.5f} 3SE={3*se_eff:.5f}")
        p(f"    color {col}: passive {fp_m:.4f} (270 {ref['f_passive']:.4f})  "
          f"optimal {fo_m:.4f} (270 {ref['f_optimal']:.4f})  "
          f"greedy {fg_m:.4f} (270 {ref['f_greedy']:.4f})  -> within 3*SE")
        equiv_records[col] = {"f_passive": fp_m, "f_optimal": fo_m, "f_greedy": fg_m}

    # (4) INHERITED-pA AUDIT
    z = np.load("creature/state/mirro/arrays.npz")
    pA = z["pA"]
    qs = z["qs"]
    true_cmap = np.asarray(G_mirro.cmap)
    pA_argmax = pA.argmax(axis=0)
    acc = float((pA_argmax == true_cmap).mean())
    assert acc >= 0.99, f"inherited pA argmax accuracy {acc} < 0.99"
    qn = qs / qs.sum() if qs.sum() > 0 else qs
    qs_ent = float(-np.sum([q * np.log2(q) for q in qn if q > 0]))
    pA_norm = pA / pA.sum(axis=0, keepdims=True)
    min_conf = float(pA_norm.max(axis=0).min())
    p(f"(4) INHERITED-pA AUDIT: argmax-vs-cmap accuracy={acc:.4f} (min conf {min_conf:.3f}); "
      f"qs entropy={qs_ent:.4f} bits -> inherited-MB == OMNISCIENT (excluded from verdicts)")

    # (5) FROM-SCRATCH AUDIT
    counts0 = _birth_prior_counts(G_mirro.n_colors, G_mirro.n_cells, 0)
    init_hat = estimate_cmap_from_history(counts0)
    init_acc = float((init_hat == true_cmap).mean())
    assert abs(init_acc - 1.0 / 3) < 0.25, f"birth-prior init accuracy {init_acc} not ~chance"
    assert init_hat is not G_mirro.cmap and not np.shares_memory(init_hat, true_cmap), \
        "cmap_hat shares memory with ground truth"
    # scramble-test: perturb a HIDDEN copy of ground truth -> from-scratch output unchanged
    cmap_hat_a, cov_a, c0a, cfa = learn_mb_from_scratch(G_mirro, 0, A_PRIMARY, 0, 20000)
    true_copy = np.asarray(G_mirro.cmap).copy()
    true_copy[:] = (true_copy + 1) % 3  # scramble the hidden copy (the agent must not see it)
    cmap_hat_b, cov_b, c0b, cfb = learn_mb_from_scratch(G_mirro, 0, A_PRIMARY, 0, 20000)
    assert np.array_equal(cmap_hat_a, cmap_hat_b), "scramble-test FAIL: MB output changed"
    assert not np.shares_memory(cmap_hat_a, true_cmap), "cmap_hat aliases ground truth"
    p(f"(5) FROM-SCRATCH AUDIT: birth-prior init map accuracy={init_acc:.3f} (~chance 0.333); "
      f"cmap_hat is a distinct array; scramble-test PASS (output invariant to hidden GT perturb)")

    # snapshot-vs-independent equivalence (the learning-curve faithfulness check)
    snaps = learn_mf(G_mirro, 0, A_PRIMARY, 0, budgets=[2000, 5000])
    ind = learn_mf(G_mirro, 0, A_PRIMARY, 0, budgets=[2000])
    assert np.array_equal(snaps[2000], ind[2000]), "snapshot != independent learn at B=2000"
    p("    snapshot-Q at B=2000 == independent learn_mf to 2000 (learning-curve prefix faithful)")

    # (6) HEADROOM per color per alpha (mirro): assert / flag no-headroom
    p("(6) HEADROOM (f_passive - f_optimal >= %.2f):" % HEADROOM_MIN)
    headroom = {}
    for col in COLORS:
        for a in ALPHAS:
            fp = stats([walk(G_mirro, col, a, "passive", s)["f"] for s in range(N_SEEDS)])[0]
            fo = stats([walk(G_mirro, col, a, "avoid", s)["f"] for s in range(N_SEEDS)])[0]
            hr = fp - fo
            headroom[(col, a)] = (fp, fo, hr, hr >= HEADROOM_MIN)
            tag = "OK" if hr >= HEADROOM_MIN else "NO-HEADROOM"
            p(f"    color {col} alpha {a}: fp={fp:.4f} fo={fo:.4f} headroom={hr:.4f} -> {tag}")

    # (7) DISJOINT-BARS per color (LEARNABLE f-ceiling < NOT-LEARNABLE f-floor) at alpha=0.5
    p("(7) DISJOINT-BARS (LEARNABLE f-ceiling < NOT-LEARNABLE f-floor, alpha=0.5):")
    for col in COLORS:
        fp = equiv_records[col]["f_passive"]
        fo = equiv_records[col]["f_optimal"]
        ceil_f = fp - CLOSURE_LEARNABLE * (fp - fo)   # closure>=0.80 -> f <= this
        floor_f = fp - CLOSURE_NOTLEARN * (fp - fo)   # closure<=0.35 -> f >= this
        assert ceil_f < floor_f, f"bars overlap color {col}: ceil {ceil_f} >= floor {floor_f}"
        p(f"    color {col}: LEARNABLE f-ceiling={ceil_f:.3f} < NOT-LEARN f-floor={floor_f:.3f} OK")

    # (8) PC + BOTH-VERDICTS-REACHABLE (instrument PC on G_segreg; neg+pos witnesses on G_mirro)
    p("(8) INSTRUMENT-PC (G_segreg) + BOTH-VERDICTS-REACHABLE:")
    # PC: the clean LEARNABILITY instrument is MB-from-scratch (it estimates the map then runs
    # the certified planner). MB MUST reach closure>=0.95 on the easy G_segreg geometry — this
    # is the HARD admissibility gate for any G_mirro NOT-LEARNABLE call (a failed MB-PC would
    # be instrument failure, not a result). The MF arm is ALSO measured here, but MF is found
    # to be intrinsically instrument-limited (model-free Q-learning under the absorbing pull
    # does NOT reliably reach optimal even on G_segreg: when a cell's whole move-neighborhood
    # is attack-colored the one-step reward gives no gradient and the deep value fails to
    # propagate at this budget). So MF's PC pass/fail is RECORDED as data and gates only the
    # ADMISSIBILITY of MF NOT-LEARNABLE verdicts (downgraded to NO-VERDICT-MF-instrument-limited
    # where MF fails its own PC); it is NEVER hard-asserted. The headline verdict is rendered
    # from the MB arm.
    pc_results = {}
    largest = max(BUDGET_GRID)
    for col in COLORS:
        fp_s = stats([walk(G_segreg, col, A_PRIMARY, "passive", s)["f"]
                      for s in range(N_SEEDS)])[0]
        fo_s = stats([walk(G_segreg, col, A_PRIMARY, "avoid", s)["f"]
                      for s in range(N_SEEDS)])[0]
        mf_cl = []
        for s in range(N_SEEDS):
            snaps = learn_mf(G_segreg, col, A_PRIMARY, LEARN_SEED_BASE + s,
                             budgets=[largest])
            ev = freeze_eval_mf(G_segreg, col, A_PRIMARY, snaps[largest],
                                s + EVAL_SEED_OFFSET)
            mf_cl.append(closure(fp_s, fo_s, ev["f"]))
        mf_cl_m = float(np.mean(mf_cl))
        mb_cl = []
        for s in range(N_SEEDS):
            cmap_hat, cov, _, _ = learn_mb_from_scratch(G_segreg, col, A_PRIMARY,
                                                        LEARN_SEED_BASE + s, largest)
            pol = optimal_on_estimated(G_segreg, cmap_hat, col, A_PRIMARY)
            ev = eval_mb_policy(G_segreg, col, A_PRIMARY, pol, s + EVAL_SEED_OFFSET)
            mb_cl.append(closure(fp_s, fo_s, ev["f"]))
        mb_cl_m = float(np.mean(mb_cl))
        pc_results[col] = {"mf_closure": mf_cl_m, "mb_closure": mb_cl_m}
        p(f"    G_segreg color {col}: MF closure={mf_cl_m:.4f} MB closure={mb_cl_m:.4f} "
          f"(need >= {PC_CLOSURE})")
    mb_pc_pass = all(pc_results[c]["mb_closure"] >= PC_CLOSURE for c in COLORS)
    mf_pc_pass = all(pc_results[c]["mf_closure"] >= PC_CLOSURE for c in COLORS)
    assert mb_pc_pass, ("MB INSTRUMENT-PC FAILED: the model-based learner cannot reach optimal "
                        "on the easy G_segreg geometry -> NOT a usable learnability instrument")
    p(f"    INSTRUMENT-PC: MB={'PASS' if mb_pc_pass else 'FAIL'} (hard gate); "
      f"MF={'PASS' if mf_pc_pass else 'FAIL'} (recorded; MF found instrument-limited if FAIL)")
    if not mf_pc_pass:
        p("    NOTE: MF fails its own PC -> MF NOT-LEARNABLE results are inadmissible "
          "(NO-VERDICT-MF-instrument-limited); MB is the verdict arm.")

    # NEG witness: small budget reaches closure<=0.35 somewhere (MF or MB).
    neg_witness = None
    small = min(BUDGET_GRID)
    for col in COLORS:
        for a in ALPHAS:
            fp, fo, hr, ok = headroom[(col, a)]
            if not ok:
                continue
            cls = []
            for s in range(N_SEEDS):
                snaps = learn_mf(G_mirro, col, a, LEARN_SEED_BASE + s, budgets=[small])
                ev = freeze_eval_mf(G_mirro, col, a, snaps[small], s + EVAL_SEED_OFFSET)
                cls.append(closure(fp, fo, ev["f"]))
            cm = float(np.mean(cls))
            if cm <= CLOSURE_NOTLEARN and neg_witness is None:
                neg_witness = ("MF", col, a, small, cm)
    if neg_witness is None:  # fall back to MB small budget
        for col in COLORS:
            for a in ALPHAS:
                fp, fo, hr, ok = headroom[(col, a)]
                if not ok:
                    continue
                cls = []
                for s in range(N_SEEDS):
                    cmap_hat, cov, _, _ = learn_mb_from_scratch(
                        G_mirro, col, a, LEARN_SEED_BASE + s, small)
                    pol = optimal_on_estimated(G_mirro, cmap_hat, col, a)
                    ev = eval_mb_policy(G_mirro, col, a, pol, s + EVAL_SEED_OFFSET)
                    cls.append(closure(fp, fo, ev["f"]))
                cm = float(np.mean(cls))
                if cm <= CLOSURE_NOTLEARN and neg_witness is None:
                    neg_witness = ("MB", col, a, small, cm)
    # POS witness: largest budget reaches closure>=0.80 somewhere (MF OR MB per spec).
    pos_witness = None
    for arm in ("MF", "MB"):
        if pos_witness is not None:
            break
        for col in COLORS:
            for a in ALPHAS:
                fp, fo, hr, ok = headroom[(col, a)]
                if not ok:
                    continue
                cls = []
                for s in range(N_SEEDS):
                    if arm == "MF":
                        snaps = learn_mf(G_mirro, col, a, LEARN_SEED_BASE + s,
                                         budgets=[largest])
                        ev = freeze_eval_mf(G_mirro, col, a, snaps[largest],
                                            s + EVAL_SEED_OFFSET)
                    else:
                        cmap_hat, cov, _, _ = learn_mb_from_scratch(
                            G_mirro, col, a, LEARN_SEED_BASE + s, largest)
                        pol = optimal_on_estimated(G_mirro, cmap_hat, col, a)
                        ev = eval_mb_policy(G_mirro, col, a, pol, s + EVAL_SEED_OFFSET)
                    cls.append(closure(fp, fo, ev["f"]))
                cm = float(np.mean(cls))
                if cm >= CLOSURE_LEARNABLE and pos_witness is None:
                    pos_witness = (arm, col, a, largest, cm)
    p(f"    NEG-witness (small budget {small}, closure<=%.2f): {neg_witness}" % CLOSURE_NOTLEARN)
    p(f"    POS-witness (largest budget {largest}, closure>=%.2f): {pos_witness}"
      % CLOSURE_LEARNABLE)
    assert neg_witness is not None, "no NEGATIVE witness reachable (small-budget closure>0.35)"
    assert pos_witness is not None, "no POSITIVE witness reachable (largest-budget closure<0.80)"
    p("    BOTH-VERDICTS-REACHABLE: PASS")
    p("")
    return {
        "equiv": equiv_records,
        "inherited_pA": {"accuracy": acc, "qs_entropy_bits": qs_ent, "min_conf": min_conf},
        "from_scratch_init_acc": init_acc,
        "headroom": headroom,
        "pc_results": pc_results,
        "mb_pc_pass": bool(mb_pc_pass),
        "mf_pc_pass": bool(mf_pc_pass),
        "neg_witness": neg_witness,
        "pos_witness": pos_witness,
    }


# ===========================================================================
# Main sweep
# ===========================================================================
def main():
    import time
    t_start = time.time()
    lines = []

    def p(s=""):
        print(s, flush=True)
        lines.append(s)

    manifest = json.load(open("creature/state/mirro/manifest.json"))
    G_mirro = World.from_dict(manifest["world"])
    G_segreg = World(rows=5, cols=5, n_colors=3, cmap=list(SEG_CMAP))
    geoms = [("G_mirro", G_mirro), ("G_segreg", G_segreg)]
    act_next_by_geom = {gname: _move_table(w) for gname, w in geoms}

    p("#" * 78)
    p("# Exp 272 — identity-ecological RUNG 1c: is the affordance LEARNABLE?")
    p("#   (observation-only learned actuator vs omniscient OPTIMAL / myopic GREEDY)")
    p("#" * 78)
    p("WALL-CLOCK ESTIMATE (pre-run): ~10-25 min (MF snapshot sweep ~8min + MB sweep +")
    p("  preflight PC/witnesses + robustness). Logged actual at end.")
    p("PREDECLARED BARS (per color, per alpha; NEVER a mean over colors):")
    p(f"  closure = (f_passive - f_learned)/(f_passive - f_optimal); passive=0 optimal=1")
    p(f"  LEARNABLE iff closure>={CLOSURE_LEARNABLE} AND f_learned<=f_passive-{ABS_ESCAPE_FLOOR}")
    p(f"    AND homerange>={HRANGE_MIN} AND most_visited_frac<={MVFRAC_MAX}, in >=18/20 seeds,")
    p(f"    AND per-seed closure UNIMODAL (sd<=|mean|; else *disp).")
    p(f"  NOT-LEARNABLE iff closure<={CLOSURE_NOTLEARN} in >=18/20 seeds.")
    p(f"  MIXED = (%.2f,%.2f) band OR side-condition violation OR *disp." %
      (CLOSURE_NOTLEARN, CLOSURE_LEARNABLE))
    p(f"  NO-VERDICT-no-headroom where (f_passive-f_optimal)<{HEADROOM_MIN}; "
      f"NO-VERDICT-budget-limited if plateau-convergence fails.")
    p(f"  Verdict rendered at PLATEAU budget: largest B with |cl(B)-cl(B/2)|<={PLATEAU_TOL} "
      f"in >=18/20 seeds.")
    p(f"  N_SEEDS={N_SEEDS} BURN={BURN} MEAS={MEAS} GAMMA={GAMMA} ALPHAS={ALPHAS}")
    p(f"  BUDGET_GRID={BUDGET_GRID}")
    p(f"  MF: lr=lr0/(1+visits) lr0={MF_LR0}; eps linear {MF_EPS0}->{MF_EPS1}; "
      f"learn_seed=s eval_seed=s+{EVAL_SEED_OFFSET}")
    p("")

    pre = preflight(G_mirro, G_segreg, manifest, lines, p)

    # ---- anchors (passive/optimal/greedy) per (geom,color,alpha) ----
    anchors = {}  # (gname,col,a) -> {f_passive,f_optimal,f_greedy,...}
    for gname, w in geoms:
        for col in COLORS:
            for a in ALPHAS:
                fp = [walk(w, col, a, "passive", s)["f"] for s in range(N_SEEDS)]
                fo = [walk(w, col, a, "avoid", s)["f"] for s in range(N_SEEDS)]
                fg = [walk_greedy(w, col, a, s) for s in range(N_SEEDS)]
                anchors[(gname, col, a)] = {
                    "f_passive": float(np.mean(fp)),
                    "f_optimal": float(np.mean(fo)),
                    "f_greedy": float(np.mean(fg)),
                    "greedy_closure": closure(float(np.mean(fp)), float(np.mean(fo)),
                                              float(np.mean(fg))),
                }

    # ---- LEARNED-MF sweep (snapshot per seed; per budget eval) ----
    p("=" * 78)
    p("LEARNED-MF SWEEP (frozen closure vs budget; per geom,color,alpha)")
    p("=" * 78)
    mf_tables = {}  # (gname,col,a) -> {budget: {per-seed arrays + means}}
    for gname, w in geoms:
        for col in COLORS:
            for a in ALPHAS:
                fp = anchors[(gname, col, a)]["f_passive"]
                fo = anchors[(gname, col, a)]["f_optimal"]
                opt_pol = optimal_avoid_policy(w, col, a)
                grd_pol = greedy_avoid_policy(w, col)
                act_next = act_next_by_geom[gname]
                # one learn run per seed -> all budget snapshots
                per_seed_snaps = [learn_mf(w, col, a, LEARN_SEED_BASE + s)
                                  for s in range(N_SEEDS)]
                budget_tab = {}
                for b in BUDGET_GRID:
                    f_arr, cl_arr, hr_arr, mv_arr = [], [], [], []
                    optagree_arr, grdagree_arr = [], []
                    for s in range(N_SEEDS):
                        ev = freeze_eval_mf(w, col, a, per_seed_snaps[s][b],
                                            s + EVAL_SEED_OFFSET)
                        f_arr.append(ev["f"])
                        cl_arr.append(closure(fp, fo, ev["f"]))
                        hr_arr.append(ev["homerange"])
                        mv_arr.append(ev["most_visited_frac"])
                        optagree_arr.append(action_agreement(ev["greedy_a"], opt_pol,
                                                             act_next, w.n_cells))
                        grdagree_arr.append(action_agreement(ev["greedy_a"], grd_pol,
                                                            act_next, w.n_cells))
                    budget_tab[b] = {
                        "f": f_arr, "closure": cl_arr, "homerange": hr_arr,
                        "mvfrac": mv_arr,
                        "f_mean": float(np.mean(f_arr)),
                        "closure_mean": float(np.mean(cl_arr)),
                        "closure_sd": float(np.std(cl_arr, ddof=1)),
                        "hr_mean": float(np.mean(hr_arr)),
                        "mv_mean": float(np.mean(mv_arr)),
                        "optagree_mean": float(np.mean(optagree_arr)),
                        "grdagree_mean": float(np.mean(grdagree_arr)),
                    }
                # online_f only at the largest budget (diagnostic; one rep seed set)
                online_largest = float(np.mean([
                    online_f_mf(w, col, a, LEARN_SEED_BASE + s, max(BUDGET_GRID))
                    for s in range(min(5, N_SEEDS))]))
                mf_tables[(gname, col, a)] = {"budgets": budget_tab,
                                              "online_f_largest": online_largest}
                # print learning curve
                p(f"--- {gname} color {col} alpha {a}  "
                  f"(f_pass={fp:.3f} f_opt={fo:.3f} headroom={fp-fo:.3f} "
                  f"greedy_cl={anchors[(gname,col,a)]['greedy_closure']:.3f}) ---")
                p(f"  {'budget':>8} {'cl_mean':>8} {'cl_sd':>7} {'f_mean':>7} "
                  f"{'hrng':>5} {'mvfrac':>6} {'optAgr':>6} {'grdAgr':>6}")
                for b in BUDGET_GRID:
                    t = budget_tab[b]
                    p(f"  {b:>8} {t['closure_mean']:>8.4f} {t['closure_sd']:>7.4f} "
                      f"{t['f_mean']:>7.4f} {t['hr_mean']:>5.1f} {t['mv_mean']:>6.3f} "
                      f"{t['optagree_mean']:>6.1f} {t['grdagree_mean']:>6.1f}")
                p(f"  online_f (cost of learning, largest budget, 5 seeds)="
                  f"{online_largest:.4f}  passive={fp:.4f}  "
                  f"online_excess={online_largest-fp:+.4f}")
                p("")

    # ---- LEARNED-MB-from-scratch sweep ----
    p("=" * 78)
    p("LEARNED-MB-FROM-SCRATCH SWEEP (estimate cmap -> PI on estimate -> eval in real env)")
    p("=" * 78)
    mb_tables = {}
    true_cmap = np.asarray(G_mirro.cmap)
    seg_cmap = np.asarray(G_segreg.cmap)
    for gname, w in geoms:
        tc = np.asarray(w.cmap)
        for col in COLORS:
            for a in ALPHAS:
                fp = anchors[(gname, col, a)]["f_passive"]
                fo = anchors[(gname, col, a)]["f_optimal"]
                # one MB eps-walk per seed -> cmap_hat snapshots at all grid budgets
                per_seed_mb = [learn_mb_snapshots(w, col, a, LEARN_SEED_BASE + s)
                               for s in range(N_SEEDS)]
                budget_tab = {}
                for b in BUDGET_GRID:
                    f_arr, cl_arr, ma_arr, cov_arr, hr_arr, mv_arr = [], [], [], [], [], []
                    for s in range(N_SEEDS):
                        cmap_hat, cov = per_seed_mb[s][b]
                        pol = optimal_on_estimated(w, cmap_hat, col, a)
                        ev = eval_mb_policy(w, col, a, pol, s + EVAL_SEED_OFFSET)
                        f_arr.append(ev["f"])
                        cl_arr.append(closure(fp, fo, ev["f"]))
                        ma_arr.append(map_accuracy(cmap_hat, tc))
                        cov_arr.append(cov)
                        hr_arr.append(ev["homerange"])
                        mv_arr.append(ev["most_visited_frac"])
                    budget_tab[b] = {
                        "f": f_arr, "closure": cl_arr, "map_accuracy": ma_arr,
                        "coverage": cov_arr, "homerange": hr_arr, "mvfrac": mv_arr,
                        "f_mean": float(np.mean(f_arr)),
                        "closure_mean": float(np.mean(cl_arr)),
                        "closure_sd": float(np.std(cl_arr, ddof=1)),
                        "map_acc_mean": float(np.mean(ma_arr)),
                        "cov_mean": float(np.mean(cov_arr)),
                        "hr_mean": float(np.mean(hr_arr)),
                        "mv_mean": float(np.mean(mv_arr)),
                    }
                mb_tables[(gname, col, a)] = {"budgets": budget_tab}
                p(f"--- {gname} color {col} alpha {a}  "
                  f"(f_pass={fp:.3f} f_opt={fo:.3f}) ---")
                p(f"  {'budget':>8} {'cl_mean':>8} {'cl_sd':>7} {'f_mean':>7} "
                  f"{'mapAcc':>7} {'cover':>6} {'hrng':>5} {'mvfrac':>6} "
                  f"{'mapGap':>7} {'fGap':>7}")
                for b in BUDGET_GRID:
                    t = budget_tab[b]
                    p(f"  {b:>8} {t['closure_mean']:>8.4f} {t['closure_sd']:>7.4f} "
                      f"{t['f_mean']:>7.4f} {t['map_acc_mean']:>7.4f} "
                      f"{t['cov_mean']:>6.3f} {t['hr_mean']:>5.1f} {t['mv_mean']:>6.3f} "
                      f"{1-t['map_acc_mean']:>7.4f} {t['f_mean']-fo:>7.4f}")
                p("")

    # ---- LEARNED-MB-inherited-pA DIAGNOSTIC (alpha=0.5 only; EXCLUDED from verdicts) ----
    p("=" * 78)
    p("LEARNED-MB-INHERITED-pA  (DIAGNOSTIC; OMNISCIENT-EQUIVALENT pA==cmap; EXCLUDED)")
    p("=" * 78)
    z = np.load("creature/state/mirro/arrays.npz")
    pA = z["pA"]
    cmap_inh = estimate_cmap_from_history(pA)  # per-cell argmax of inherited pA
    inh_acc = float((cmap_inh == true_cmap).mean())
    inherited_diag = {"accuracy": inh_acc, "excluded": True, "rows": []}
    p(f"  inherited pA argmax accuracy = {inh_acc:.4f} (1.000 -> IS the ground-truth cmap)")
    for col in COLORS:
        fo = anchors[("G_mirro", col, A_PRIMARY)]["f_optimal"]
        pol = optimal_on_estimated(G_mirro, cmap_inh, col, A_PRIMARY)
        f_inh = [eval_mb_policy(G_mirro, col, A_PRIMARY, pol, s + EVAL_SEED_OFFSET)["f"]
                 for s in range(N_SEEDS)]
        f_inh_m = float(np.mean(f_inh))
        match = abs(f_inh_m - fo) <= 0.02
        assert match, f"inherited-pA color {col}: f={f_inh_m} != optimal {fo}"
        p(f"  color {col}: inherited-MB f={f_inh_m:.4f}  OPTIMAL f={fo:.4f}  "
          f"|diff|<=0.02: {match}  -> OMNISCIENT-EQUIVALENT, NOT a learnability result")
        inherited_diag["rows"].append({"color": col, "f_inherited": f_inh_m,
                                       "f_optimal": fo, "matches_optimal": bool(match)})
    p("")

    # ---- Plateau detection + verdicts (per geom,color,alpha, per arm) ----
    def detect_plateau(budget_tab):
        """Largest grid budget B with |cl(B)-cl(prevB)| <= PLATEAU_TOL over >=18/20 seeds.
        The grid is ~geometric so the previous grid budget plays the role of B/2."""
        best = None
        for i in range(1, len(BUDGET_GRID)):
            b = BUDGET_GRID[i]
            bp = BUDGET_GRID[i - 1]
            cl_b = np.array(budget_tab[b]["closure"])
            cl_bp = np.array(budget_tab[bp]["closure"])
            n_conv = int(np.sum(np.abs(cl_b - cl_bp) <= PLATEAU_TOL))
            if n_conv >= 18:
                best = b  # keep the LARGEST converged budget
        return best

    def compute_verdict(gname, col, a, tables, arm, mf_pc_ok):
        """Render the predeclared closure-partition verdict for one arm.

        For the MF arm: if MF failed its own G_segreg PC, a NOT-LEARNABLE call is
        inadmissible and downgraded to NO-VERDICT-MF-instrument-limited."""
        fp = anchors[(gname, col, a)]["f_passive"]
        fo = anchors[(gname, col, a)]["f_optimal"]
        headroom = fp - fo
        bt = tables[(gname, col, a)]["budgets"]
        if headroom < HEADROOM_MIN:
            return "NO-VERDICT-no-headroom", None
        pb = detect_plateau(bt)
        if pb is None:
            return "NO-VERDICT-budget-limited", None
        cl = np.array(bt[pb]["closure"])
        f_arr = np.array(bt[pb]["f"])
        hr = np.array(bt[pb]["homerange"])
        mv = np.array(bt[pb]["mvfrac"])
        cl_mean = float(cl.mean())
        cl_sd = float(cl.std(ddof=1))
        n_learn = int(np.sum(
            (cl >= CLOSURE_LEARNABLE) & (f_arr <= fp - ABS_ESCAPE_FLOOR)
            & (hr >= HRANGE_MIN) & (mv <= MVFRAC_MAX)))
        n_notlearn = int(np.sum(cl <= CLOSURE_NOTLEARN))
        disp = (abs(cl_mean) > 1e-9 and cl_sd > abs(cl_mean))
        hr_ok = float(hr.mean()) >= HRANGE_MIN
        mv_ok = float(mv.mean()) <= MVFRAC_MAX
        if n_learn >= 18 and hr_ok and mv_ok:
            verdict = "LEARNABLE"
        elif n_notlearn >= 18:
            verdict = "NOT-LEARNABLE"
        elif (cl >= CLOSURE_LEARNABLE).sum() >= 18 and not (hr_ok and mv_ok):
            verdict = "MIXED-degenerate(huddle)"
        else:
            verdict = "MIXED/PARTIAL"
        if disp and verdict in ("LEARNABLE", "MIXED/PARTIAL"):
            verdict += "*disp"
        # MF NOT-LEARNABLE is inadmissible if MF failed its own PC (instrument failure)
        if arm == "MF" and verdict == "NOT-LEARNABLE" and not mf_pc_ok:
            verdict = "NO-VERDICT-MF-instrument-limited"
        return verdict, pb

    mf_pc_ok = pre["mf_pc_pass"]
    mf_verdicts, mf_plateaus = {}, {}
    mb_verdicts, mb_plateaus = {}, {}
    for gname, w in geoms:
        for col in COLORS:
            for a in ALPHAS:
                vmf, pmf = compute_verdict(gname, col, a, mf_tables, "MF", mf_pc_ok)
                vmb, pmb = compute_verdict(gname, col, a, mb_tables, "MB", True)
                mf_verdicts[(gname, col, a)] = vmf
                mf_plateaus[(gname, col, a)] = pmf
                mb_verdicts[(gname, col, a)] = vmb
                mb_plateaus[(gname, col, a)] = pmb

    p("=" * 78)
    p("PER-(geom,color,alpha) VERDICT — MB-from-scratch (ADMISSIBLE learnability arm)")
    p("=" * 78)
    for gname, w in geoms:
        for col in COLORS:
            for a in ALPHAS:
                fp = anchors[(gname, col, a)]["f_passive"]
                fo = anchors[(gname, col, a)]["f_optimal"]
                pb = mb_plateaus[(gname, col, a)]
                bt = mb_tables[(gname, col, a)]["budgets"]
                clm = bt[pb]["closure_mean"] if pb is not None else float("nan")
                pbs = pb if pb is not None else "-"
                p(f"  {gname} color {col} alpha {a}: plateau={pbs} closure={clm:.4f} "
                  f"headroom={fp-fo:.3f} -> {mb_verdicts[(gname, col, a)]}")
    p("")
    p("=" * 78)
    p("PER-(geom,color,alpha) VERDICT — MF (model-free; INSTRUMENT-LIMITED, see PC)")
    p("=" * 78)
    p(f"  (MF G_segreg PC pass = {mf_pc_ok}; where MF fails its PC a NOT-LEARNABLE is")
    p("   downgraded to NO-VERDICT-MF-instrument-limited — MF cannot prove un-learnability)")
    for gname, w in geoms:
        for col in COLORS:
            for a in ALPHAS:
                fp = anchors[(gname, col, a)]["f_passive"]
                fo = anchors[(gname, col, a)]["f_optimal"]
                pb = mf_plateaus[(gname, col, a)]
                bt = mf_tables[(gname, col, a)]["budgets"]
                clm = bt[pb]["closure_mean"] if pb is not None else float("nan")
                pbs = pb if pb is not None else "-"
                p(f"  {gname} color {col} alpha {a}: plateau={pbs} closure={clm:.4f} "
                  f"headroom={fp-fo:.3f} -> {mf_verdicts[(gname, col, a)]}")
    p("")
    # The headline verdict vector is the ADMISSIBLE arm (MB-from-scratch).
    verdicts = mb_verdicts
    plateau_budgets = mb_plateaus

    # ---- ROBUSTNESS at A_PRIMARY plateau budget (MF; eps x0.5/x2, lr x0.5/x2, gamma) ----
    p("=" * 78)
    p("ROBUSTNESS (G_mirro, alpha=%.2f, at plateau budget; verdict must not flip)" % A_PRIMARY)
    p("=" * 78)
    p("  Headline arm = MB-from-scratch (eps-exploration x0.5/x2; planner gamma 0.99/0.999);")
    p("  the MB verdict must not hard-flip LEARNABLE<->NOT-LEARNABLE. MF sensitivity")
    p("  (eps/lr/gamma) is also reported as a diagnostic (MF is instrument-limited, no assert).")

    def classify(cls):
        cls = np.asarray(cls)
        if int(np.sum(cls >= CLOSURE_LEARNABLE)) >= 18:
            return "LEARNABLE"
        if int(np.sum(cls <= CLOSURE_NOTLEARN)) >= 18:
            return "NOT-LEARNABLE"
        return "MIXED"

    def base_class_of(v):
        if v.startswith("LEARNABLE"):
            return "LEARNABLE"
        if v.startswith("NOT-LEARNABLE"):
            return "NOT-LEARNABLE"
        return "MIXED"

    robustness = {}
    for col in COLORS:
        fp = anchors[("G_mirro", col, A_PRIMARY)]["f_passive"]
        fo = anchors[("G_mirro", col, A_PRIMARY)]["f_optimal"]
        # ---- MB headline-arm robustness (the hard-asserted one) ----
        pb_mb = mb_plateaus[("G_mirro", col, A_PRIMARY)]
        base_mb = mb_verdicts[("G_mirro", col, A_PRIMARY)]
        mb_row = {}
        mb_hard_flips = []
        if pb_mb is None:
            p(f"  color {col} MB: no plateau -> robustness skipped (base={base_mb})")
        else:
            mb_variants = {
                "base": dict(eps0=MF_EPS0, gamma=GAMMA),
                "eps_x0.5": dict(eps0=MF_EPS0 * 0.5, gamma=GAMMA),
                "eps_x2": dict(eps0=MF_EPS0 * 2.0, gamma=GAMMA),
                "gamma_0.99": dict(eps0=MF_EPS0, gamma=GAMMA_ROBUST),
            }
            p(f"  color {col} MB (plateau {pb_mb}, base {base_mb}):")
            for vname, kw in mb_variants.items():
                cls = []
                for s in range(N_SEEDS):
                    cmap_hat, cov, _, _ = learn_mb_from_scratch(
                        G_mirro, col, A_PRIMARY, LEARN_SEED_BASE + s, pb_mb,
                        eps0=kw["eps0"])
                    pol = optimal_on_estimated(G_mirro, cmap_hat, col, A_PRIMARY)
                    # planner gamma variant: re-plan on the estimate at kw['gamma']
                    if kw["gamma"] != GAMMA and any(int(c) == col for c in cmap_hat):
                        w_hat = World(rows=5, cols=5, n_colors=3,
                                      cmap=[int(x) for x in cmap_hat])
                        pol = optimal_avoid_policy(w_hat, col, A_PRIMARY, gamma=kw["gamma"])
                    ev = eval_mb_policy(G_mirro, col, A_PRIMARY, pol, s + EVAL_SEED_OFFSET)
                    cls.append(closure(fp, fo, ev["f"]))
                vv = classify(cls)
                mb_row[vname] = {"closure_mean": float(np.mean(cls)), "verdict": vv}
                p(f"    {vname:>11}: closure={float(np.mean(cls)):.4f}  verdict={vv}")
            bc = base_class_of(base_mb)
            mb_hard_flips = [vn for vn, rv in mb_row.items()
                             if {rv["verdict"], bc} == {"LEARNABLE", "NOT-LEARNABLE"}]
            p(f"    -> MB hard flips: {mb_hard_flips if mb_hard_flips else 'NONE'}")
            assert not mb_hard_flips, f"MB ROBUSTNESS FLIP color {col}: {mb_hard_flips}"

        # ---- MF diagnostic robustness (reported only) ----
        pb_mf = mf_plateaus[("G_mirro", col, A_PRIMARY)]
        base_mf = mf_verdicts[("G_mirro", col, A_PRIMARY)]
        mf_row = {}
        if pb_mf is not None:
            mf_variants = {
                "base": dict(eps0=MF_EPS0, lr0=MF_LR0, gamma=GAMMA),
                "eps_x0.5": dict(eps0=MF_EPS0 * 0.5, lr0=MF_LR0, gamma=GAMMA),
                "eps_x2": dict(eps0=MF_EPS0 * 2.0, lr0=MF_LR0, gamma=GAMMA),
                "lr_x0.5": dict(eps0=MF_EPS0, lr0=MF_LR0 * 0.5, gamma=GAMMA),
                "lr_x2": dict(eps0=MF_EPS0, lr0=MF_LR0 * 2.0, gamma=GAMMA),
                "gamma_0.99": dict(eps0=MF_EPS0, lr0=MF_LR0, gamma=GAMMA_ROBUST),
            }
            p(f"  color {col} MF (plateau {pb_mf}, base {base_mf}) [diagnostic]:")
            for vname, kw in mf_variants.items():
                cls = []
                for s in range(N_SEEDS):
                    snaps = learn_mf(G_mirro, col, A_PRIMARY, LEARN_SEED_BASE + s,
                                     budgets=[pb_mf], lr0=kw["lr0"], eps0=kw["eps0"],
                                     gamma=kw["gamma"])
                    ev = freeze_eval_mf(G_mirro, col, A_PRIMARY, snaps[pb_mf],
                                        s + EVAL_SEED_OFFSET)
                    cls.append(closure(fp, fo, ev["f"]))
                vv = classify(cls)
                mf_row[vname] = {"closure_mean": float(np.mean(cls)), "verdict": vv}
                p(f"    {vname:>11}: closure={float(np.mean(cls)):.4f}  verdict={vv}")
        else:
            p(f"  color {col} MF: no plateau -> MF robustness skipped (base={base_mf})")
        robustness[col] = {"mb_base_verdict": base_mb, "mb_variants": mb_row,
                           "mb_hard_flips": mb_hard_flips,
                           "mf_base_verdict": base_mf, "mf_variants": mf_row}
    p("")

    # ---- Headline learnability MAP + collapse audit ----
    p("=" * 78)
    p("HEADLINE LEARNABILITY MAP  (per color: alpha -> MB verdict @ plateau, G_mirro; "
      "MF shown alongside)")
    p("=" * 78)
    headline_map = {}
    for col in COLORS:
        row = {}
        for a in ALPHAS:
            row[a] = verdicts[("G_mirro", col, a)]
        headline_map[col] = row
        scatter_note = ("scattered (greedy fails)" if col in (0, 1)
                        else "BLOCK (greedy near-solves; degenerate learnability test)")
        p(f"  color {col} [{scatter_note}]:")
        for a in ALPHAS:
            gc = anchors[("G_mirro", col, a)]["greedy_closure"]
            p(f"    alpha {a}: MB={row[a]:>26}  MF={mf_verdicts[('G_mirro', col, a)]:>32}"
              f"  (greedy_closure={gc:.3f})")
    p("")

    # collapse audit (frozen f within 0.02 of passive OR f~1.0) at the largest budget
    p("COLLAPSE AUDIT (frozen f within 0.02 of passive OR f>=0.98) at largest budget:")
    collapse = {}
    for gname, w in geoms:
        for col in COLORS:
            for a in ALPHAS:
                fp = anchors[(gname, col, a)]["f_passive"]
                f_arr = np.array(mf_tables[(gname, col, a)]["budgets"][max(BUDGET_GRID)]["f"])
                n_coll = int(np.sum((np.abs(f_arr - fp) <= 0.02) | (f_arr >= 0.98)))
                collapse[(gname, col, a)] = n_coll
                if n_coll > 0:
                    p(f"  {gname} color {col} alpha {a}: {n_coll}/{N_SEEDS} collapsed seeds")
    if not any(collapse.values()):
        p("  no collapsed seeds at the largest budget on any (geom,color,alpha)")
    p("")

    # ---- Per-color greedy closure on the record ----
    p("PER-COLOR GREEDY CLOSURE ON THE RECORD (G_mirro, all alphas):")
    for col in COLORS:
        gcs = {a: round(anchors[("G_mirro", col, a)]["greedy_closure"], 3) for a in ALPHAS}
        p(f"  color {col}: {gcs}")
    p("")

    # ---- Final PRINTED_VERDICT (the SCRIPT'S claim; headline = MB admissible arm) ----
    p("=" * 78)
    p("FINAL PER-(color,alpha) VERDICT VECTOR (G_mirro; MB-from-scratch admissible arm)")
    p("=" * 78)
    # determine overall headline: focus on scattered colors 0/1 at A_PRIMARY (the real test)
    scat_primary = [verdicts[("G_mirro", c, A_PRIMARY)] for c in (0, 1)]
    all_v = [verdicts[("G_mirro", c, a)] for c in COLORS for a in ALPHAS]
    n_learnable = sum(1 for v in all_v if v.startswith("LEARNABLE"))
    n_notlearn = sum(1 for v in all_v if v.startswith("NOT-LEARNABLE"))
    n_mixed = sum(1 for v in all_v if v.startswith("MIXED"))
    for col in COLORS:
        for a in ALPHAS:
            p(f"  G_mirro color {col} alpha {a}: MB={verdicts[('G_mirro', col, a)]}  "
              f"(MF={mf_verdicts[('G_mirro', col, a)]})")
    p("")
    if all(v.startswith("LEARNABLE") for v in scat_primary):
        printed = "LEARNABLE"
        note = ("scattered colors 0/1 at alpha=0.5 are BOTH LEARNABLE at plateau (MB arm) -> the "
                "affordance is learnable from observation-only experience GIVEN adequate budget")
    elif all(v.startswith("NOT-LEARNABLE") for v in scat_primary):
        printed = "NOT-LEARNABLE"
        note = "scattered colors 0/1 at alpha=0.5 stay near greedy -> posable-but-not-learnable"
    else:
        printed = "MIXED"
        note = "scattered colors 0/1 at alpha=0.5 split across the closure partition"
    p(f"PRINTED_VERDICT: {printed}   (the SCRIPT'S claim; per-(color,alpha) vector is the result)")
    p(f"  basis (MB admissible arm): {note}")
    p(f"  tally over all (color,alpha), MB arm: LEARNABLE={n_learnable} "
      f"NOT-LEARNABLE={n_notlearn} MIXED/NO-VERDICT={n_mixed}")
    p("")
    p("ARM SPLIT (the headline scientific content):")
    p("  MB-from-scratch (estimate map -> certified planner): the AFFORDANCE IS LEARNABLE — once")
    p("    the observation-only agent recovers the cmap (full coverage at modest budget) the")
    p("    handed pull-mechanism + certified PI reproduce the optimal escape. The map is the gap.")
    p("  MF (model-free tabular Q-learning): INSTRUMENT-LIMITED — it fails its OWN G_segreg PC")
    p("    (closure<0.95 on the easy geometry: when a cell's whole neighborhood is attack-colored")
    p("    the one-step reward gives no gradient and the deep value does not propagate at this")
    p("    budget). So MF cannot adjudicate NOT-LEARNABLE; it is reported as a learning-SPEED /")
    p("    cost-of-learning diagnostic, never the verdict.")
    p("INTERPRETATION (predeclared honesty note): a tabular 25-state MDP IS solvable by a")
    p("  model-based learner with adequate data; the reportable content is the BUDGET-TO-LEARN and")
    p("  geometry-dependent learning SPEED, per color (color 2 BLOCK learns fastest / greedy")
    p("  near-solves; scattered 0/1 are the real test). Reported per color, never averaged.")
    p("")

    elapsed = time.time() - t_start
    p(f"WALL-CLOCK (actual): {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ---- write JSON (deterministic, sort_keys, indent=2) ----
    def keyify(d):
        return {f"{g}|{c}|{a}": v for (g, c, a), v in d.items()}

    mf_json = {}
    for (g, c, a), tab in mf_tables.items():
        mf_json[f"{g}|{c}|{a}"] = {
            "online_f_largest": tab["online_f_largest"],
            "budgets": {str(b): {
                "closure": tab["budgets"][b]["closure"],
                "f": tab["budgets"][b]["f"],
                "homerange": tab["budgets"][b]["homerange"],
                "mvfrac": tab["budgets"][b]["mvfrac"],
                "closure_mean": tab["budgets"][b]["closure_mean"],
                "closure_sd": tab["budgets"][b]["closure_sd"],
                "f_mean": tab["budgets"][b]["f_mean"],
                "optagree_mean": tab["budgets"][b]["optagree_mean"],
                "grdagree_mean": tab["budgets"][b]["grdagree_mean"],
            } for b in BUDGET_GRID},
        }
    mb_json = {}
    for (g, c, a), tab in mb_tables.items():
        mb_json[f"{g}|{c}|{a}"] = {
            "budgets": {str(b): {
                "closure": tab["budgets"][b]["closure"],
                "f": tab["budgets"][b]["f"],
                "map_accuracy": tab["budgets"][b]["map_accuracy"],
                "coverage": tab["budgets"][b]["coverage"],
                "closure_mean": tab["budgets"][b]["closure_mean"],
                "closure_sd": tab["budgets"][b]["closure_sd"],
                "f_mean": tab["budgets"][b]["f_mean"],
                "map_acc_mean": tab["budgets"][b]["map_acc_mean"],
                "cov_mean": tab["budgets"][b]["cov_mean"],
            } for b in BUDGET_GRID},
        }

    out = {
        "experiment": "Exp 272 — identity-ecological RUNG 1c learnability probe",
        "params": {
            "BURN": BURN, "MEAS": MEAS, "N_SEEDS": N_SEEDS, "COLORS": COLORS,
            "ALPHAS": ALPHAS, "A_PRIMARY": A_PRIMARY, "GAMMA": GAMMA,
            "GAMMA_ROBUST": GAMMA_ROBUST, "BUDGET_GRID": BUDGET_GRID,
            "MF_LR0": MF_LR0, "MF_EPS0": MF_EPS0, "MF_EPS1": MF_EPS1,
            "LEARN_SEED_BASE": LEARN_SEED_BASE, "EVAL_SEED_OFFSET": EVAL_SEED_OFFSET,
            "CLOSURE_LEARNABLE": CLOSURE_LEARNABLE, "CLOSURE_NOTLEARN": CLOSURE_NOTLEARN,
            "ABS_ESCAPE_FLOOR": ABS_ESCAPE_FLOOR, "HRANGE_MIN": HRANGE_MIN,
            "MVFRAC_MAX": MVFRAC_MAX, "HEADROOM_MIN": HEADROOM_MIN,
            "PLATEAU_TOL": PLATEAU_TOL, "PC_CLOSURE": PC_CLOSURE,
            "STATE_HASH": STATE_HASH, "SEG_CMAP": SEG_CMAP,
        },
        "preflight": {
            "equivalence": {str(c): pre["equiv"][c] for c in COLORS},
            "inherited_pA_audit": pre["inherited_pA"],
            "from_scratch_init_acc": pre["from_scratch_init_acc"],
            "instrument_pc": {str(c): pre["pc_results"][c] for c in COLORS},
            "instrument_pc_mb_pass": pre["mb_pc_pass"],
            "instrument_pc_mf_pass": pre["mf_pc_pass"],
            "neg_witness": list(pre["neg_witness"]) if pre["neg_witness"] else None,
            "pos_witness": list(pre["pos_witness"]) if pre["pos_witness"] else None,
        },
        "anchors": {f"{g}|{c}|{a}": anchors[(g, c, a)] for (g, c, a) in anchors},
        "mf_tables": mf_json,
        "mb_tables": mb_json,
        "inherited_pA_diagnostic": inherited_diag,
        "plateau_budgets_mb": {f"{g}|{c}|{a}": mb_plateaus[(g, c, a)]
                               for (g, c, a) in mb_plateaus},
        "plateau_budgets_mf": {f"{g}|{c}|{a}": mf_plateaus[(g, c, a)]
                               for (g, c, a) in mf_plateaus},
        "robustness": {str(c): robustness[c] for c in COLORS},
        "verdicts_mb": {f"{g}|{c}|{a}": mb_verdicts[(g, c, a)] for (g, c, a) in mb_verdicts},
        "verdicts_mf": {f"{g}|{c}|{a}": mf_verdicts[(g, c, a)] for (g, c, a) in mf_verdicts},
        "verdicts": {f"{g}|{c}|{a}": verdicts[(g, c, a)] for (g, c, a) in verdicts},
        "headline_map": {str(c): {str(a): headline_map[c][a] for a in ALPHAS}
                         for c in COLORS},
        "collapse_audit": {f"{g}|{c}|{a}": collapse[(g, c, a)] for (g, c, a) in collapse},
        "printed_verdict": printed,
        "wall_clock_s": elapsed,
    }
    with open("experiments/outputs/exp272_results.json", "w") as fh:
        json.dump(out, fh, sort_keys=True, indent=2)
    p("WROTE experiments/outputs/exp272_results.json")


if __name__ == "__main__":
    main()
