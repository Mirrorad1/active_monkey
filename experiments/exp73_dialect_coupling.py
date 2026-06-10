"""Exp 73 — rung 5: dialects under coupling — convergence is mass-gated, or it is not.

Social-emergence direction, rung 5 (loop/directions/social-emergence.md): two clade-mates
taught word<->color maps SEPARATELY (a real dialect difference, taught not self-formed —
declared) are coupled through a grounded word channel. Do the maps converge, hold as
stable dialects, or break down? And does the value-mass inertia law (Exp 65-67) extend to
VOCAB mass — light vocabularies converge where heavy ones stand firm?

CEILING (binding, declared up front): this tests TAUGHT-LABEL map dynamics under coupling.
No grammar, no compositionality, no language-from-scratch claim — that ceiling stands
(open_problem.html).

Channel (provided, the one new mechanism): at every SAME-CELL event (exact shared
referent: same cell = same observed color c), each creature SPEAKS its current best word
for c (_word_for_color logic on its own evolving vocab) and the OTHER, observing the same
c, Dirichlet-learns: vocab[heard_word][c] += its own exp(-H) predictability weight (the
M4 grounding gate, as Exp 65/66). Speakers do not self-update; the exchange is symmetric
(both speak, both listen, every event). Listening can ADD new words to a vocab (a heard
word not yet known is initialized at the teach_word prior 0.1 before the increment).

Severed control (analytic, declared): nothing in live() touches vocab, so an uncoupled
twin's vocab is FROZEN — the severed endpoint equals the phase-1 taught state exactly; no
simulation needed.

Dialects (phase 1, provided): P taught the identity map {w0->color0, w1->color1,
w2->color2}; Q taught the shifted map {w0->color1, w1->color2, w2->color0}. Same three
word tokens, disjoint referents: argmax dialect distance starts at 1.0 by construction.
Arms: LIGHT (teach n=8 per word) and HEAVY (teach n=40 per word) — the mass manipulation.
Dose arithmetic stated before running: ~80 same-cell events spread over 3 colors with gate
~0.9 gives roughly +24 counts per (word, color) pairing from the partner's usage —
above LIGHT's taught 8, below HEAVY's taught 40 — so LIGHT should flip argmaxes and HEAVY
should not. That is the at-stake prediction, not a certainty.

Metrics: per shared word w, cosine similarity between P's and Q's normalized vocab[w]
vectors, averaged over the union of words (absent word = uniform prior vector, declared);
argmax dialect distance = fraction of shared words whose argmax colors differ.
Gates: G1 >= 40 same-cell events per run (Exp 64 saw ~80; below 40 the dose story is
unpowered -> run INVALID).
Predeclared predictions:
  P1 (channel transmits labels): final mean cosine(P,Q) > the frozen severed value in
     >= 4/5 seeds, in EACH arm.
  P2 (mass law extends to vocab): LIGHT final argmax dialect distance < 1.0 in >= 4/5
     seeds AND HEAVY final argmax dialect distance == 1.0 in >= 4/5 seeds.
Falsifiers:
  F1 = P1 fails in either arm -> the channel does not transmit labels at this dose;
     rung 5 NEGATIVE at this substrate.
  F2 = HEAVY converges like LIGHT (HEAVY distance < 1.0 in >= 4/5) -> the inertia law
     does NOT extend to vocab mass — logged as a real negative for the law's generality.
  (P2's LIGHT half failing alone with P1 passing = dose insufficient for argmax flips:
     MIXED, dose iteration named.)
Provided priors declared: both dialect maps and all teaching; the channel rule and its
gate; the world (mirro's grid); random-walk policy; starts (cells 0 and 24 — the
kidnapped-start transient is ~20-30 steps per Exp 72 P1, negligible here, declared).
Self-formed: only the usage statistics (who says what when) and the gate weights.
Spines never live, never saved; mirro forked once ("exp73-twin-template"); vela untouched.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
from numpy.random import default_rng

from active_loop.creature import Creature

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEPS = 2000
SEEDS = [0, 1, 2, 3, 4]
WORDS = ["w0", "w1", "w2"]
LIGHT_N = 8
HEAVY_N = 40
START_P = 0
START_Q = 24
MIRRO_DIR = Path("creature/state/mirro")
RNG_BASE = 950000

# ---------------------------------------------------------------------------
# Load mirro — read-only committed spine
# ---------------------------------------------------------------------------

mirro = Creature.load(MIRRO_DIR)
print(f"mirro: name={mirro.name!r} age={mirro.age_steps} "
      f"hash={mirro._state_hash()[:12]} true_pos={mirro.true_pos} "
      f"world={mirro.world.rows}x{mirro.world.cols} n_colors={mirro.world.n_colors}")

# Fork once: template shares mirro's history but is unbound
template = mirro.fork("exp73-twin-template")
print(f"template vocab (should be empty): {template.vocab}")
assert len(template.vocab) == 0, f"Expected empty vocab, got {len(template.vocab)} words"

world = mirro.world
n_colors = world.n_colors
B = world.transition_matrix()  # precomputed once for all runs

print()

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def word_vec(c: Creature, w: str) -> np.ndarray:
    """Normalised vocab[w] vector; absent word -> uniform prior 0.1 vector."""
    if w in c.vocab:
        v = c.vocab[w].copy()
    else:
        v = np.ones(n_colors) * 0.1
    total = v.sum()
    if total > 0:
        return v / total
    return np.ones(n_colors) / n_colors


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mean_cosine(P: Creature, Q: Creature) -> float:
    """Mean cosine similarity over the UNION of vocab keys."""
    union_words = set(P.vocab.keys()) | set(Q.vocab.keys())
    if not union_words:
        return 1.0  # both empty — trivially identical
    sims = [cosine(word_vec(P, w), word_vec(Q, w)) for w in union_words]
    return float(np.mean(sims))


def argmax_distance(P: Creature, Q: Creature) -> float:
    """Fraction of INTERSECTION words whose argmax colors differ."""
    intersection = set(P.vocab.keys()) & set(Q.vocab.keys())
    if not intersection:
        return 0.0
    diffs = [
        int(np.argmax(P.vocab[w])) != int(np.argmax(Q.vocab[w]))
        for w in intersection
    ]
    return float(np.mean(diffs))


def best_word_for_color(c: Creature, color: int):
    """Inline replication of _word_for_color on the CURRENT evolving vocab."""
    if not c.vocab:
        return None
    best_word, best_score = None, -1.0
    for word, counts in c.vocab.items():
        total = counts.sum()
        if total > 0:
            score = float(counts[color] / total)
            if score > best_score:
                best_score, best_word = score, word
    return best_word


# ---------------------------------------------------------------------------
# Severed (frozen) reference — analytic, no simulation needed
# ---------------------------------------------------------------------------
# Build one LIGHT and one HEAVY pair to measure the frozen (severed) cosine
# before phase 2.  These are the same pre-phase-2 vocabs we'll compute
# cosine_before from inside the per-run loop.

# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

# Storage: results[arm][seed] = dict
ARMS = ["LIGHT", "HEAVY"]
ARM_N = {"LIGHT": LIGHT_N, "HEAVY": HEAVY_N}

# For P1 checks: cosine_before (severed reference) vs cosine_after (coupled)
# For P2 checks: dist_after
results_store = {arm: {} for arm in ARMS}

print("arm    seed  events  cos_before  cos_after  dist_before  dist_after")
print("-" * 70)

for arm in ARMS:
    n_teach = ARM_N[arm]

    for seed in SEEDS:
        # --- Phase 1: teach dialects ---
        P = copy.deepcopy(template)
        Q = copy.deepcopy(template)
        P.true_pos = START_P
        Q.true_pos = START_Q

        # P: identity map {w0->0, w1->1, w2->2}
        for wi, color in enumerate(range(n_colors)):
            P.teach_word(WORDS[wi], color, n=n_teach)

        # Q: shifted map {w0->1, w1->2, w2->0}
        for wi, color in enumerate([(1, 2, 0)[wi] for wi in range(n_colors)]):
            Q.teach_word(WORDS[wi], color, n=n_teach)

        # Severed / frozen reference metrics (phase-1 state)
        cosine_before = mean_cosine(P, Q)
        dist_before = argmax_distance(P, Q)

        # --- Phase 2: coupled random walk ---
        rng_P = default_rng(RNG_BASE + 1000 * seed + 0)
        rng_Q = default_rng(RNG_BASE + 1000 * seed + 1)

        events = 0
        spoken_pairs = {}  # diagnostic: (word_P, word_Q) -> count

        for _step in range(STEPS):
            # Read positions simultaneously at start of step
            pos_P = P.true_pos
            pos_Q = Q.true_pos

            # --- live()-math block for P ---
            A_hat_P = P._A_hat()
            obs_P = int(world.cmap[pos_P])
            qs_upd_P = A_hat_P[obs_P, :] * P.qs
            denom = qs_upd_P.sum()
            if denom > 0:
                qs_upd_P = qs_upd_P / denom
            else:
                qs_upd_P = np.ones(world.n_cells) / world.n_cells
            P.pA[obs_P, :] += qs_upd_P
            map_cell_P = int(np.argmax(qs_upd_P))
            pred_P = A_hat_P[:, map_cell_P]
            h_P = -np.sum(pred_P * np.log(pred_P + 1e-12))
            w_P = float(np.exp(-h_P))
            P.value_counts[obs_P] += w_P
            action_P = int(rng_P.integers(0, 4))
            new_pos_P = world.move(pos_P, action_P)
            new_qs_P = B[:, :, action_P] @ qs_upd_P

            # --- live()-math block for Q ---
            A_hat_Q = Q._A_hat()
            obs_Q = int(world.cmap[pos_Q])
            qs_upd_Q = A_hat_Q[obs_Q, :] * Q.qs
            denom = qs_upd_Q.sum()
            if denom > 0:
                qs_upd_Q = qs_upd_Q / denom
            else:
                qs_upd_Q = np.ones(world.n_cells) / world.n_cells
            Q.pA[obs_Q, :] += qs_upd_Q
            map_cell_Q = int(np.argmax(qs_upd_Q))
            pred_Q = A_hat_Q[:, map_cell_Q]
            h_Q = -np.sum(pred_Q * np.log(pred_Q + 1e-12))
            w_Q = float(np.exp(-h_Q))
            Q.value_counts[obs_Q] += w_Q
            action_Q = int(rng_Q.integers(0, 4))
            new_pos_Q = world.move(pos_Q, action_Q)
            new_qs_Q = B[:, :, action_Q] @ qs_upd_Q

            # Advance qs (after both updates, before channel)
            P.qs = new_qs_P
            Q.qs = new_qs_Q

            # Apply moves simultaneously
            P.true_pos = new_pos_P
            Q.true_pos = new_pos_Q

            # --- Channel: same-cell event uses START-OF-STEP positions ---
            if pos_P == pos_Q:
                c_color = int(world.cmap[pos_P])  # shared referent: same cell, same color

                # Compute spoken words FIRST from pre-update vocabs
                # (vocabs ARE updated by listening below, but words computed from
                #  each creature's own vocab BEFORE this step's listen — using
                #  the vocab as it stands now, which reflects all prior exchanges)
                word_spoken_P = best_word_for_color(P, c_color)
                word_spoken_Q = best_word_for_color(Q, c_color)

                # Both updates THEN happen symmetrically
                if word_spoken_P is not None:
                    # Q listens to P
                    if word_spoken_P not in Q.vocab:
                        Q.vocab[word_spoken_P] = np.ones(n_colors) * 0.1
                    Q.vocab[word_spoken_P][c_color] += w_Q

                if word_spoken_Q is not None:
                    # P listens to Q
                    if word_spoken_Q not in P.vocab:
                        P.vocab[word_spoken_Q] = np.ones(n_colors) * 0.1
                    P.vocab[word_spoken_Q][c_color] += w_P

                events += 1

                # Diagnostic: track spoken pairs
                pair = (word_spoken_P, word_spoken_Q)
                spoken_pairs[pair] = spoken_pairs.get(pair, 0) + 1

        # --- Post-run metrics ---
        cosine_after = mean_cosine(P, Q)
        dist_after = argmax_distance(P, Q)

        results_store[arm][seed] = {
            "events": events,
            "cosine_before": cosine_before,
            "cosine_after": cosine_after,
            "dist_before": dist_before,
            "dist_after": dist_after,
            "P": P,
            "Q": Q,
            "spoken_pairs": spoken_pairs,
        }

        print(f"{arm:<6} {seed}     {events:4d}    {cosine_before:.4f}    {cosine_after:.4f}"
              f"       {dist_before:.4f}       {dist_after:.4f}")

        # G1 gate: per-run validity
        if events < 40:
            print(f"\nG1 FAIL — RUN INVALID (arm={arm} seed={seed} events={events} < 40)")
            sys.exit(1)

print()

# ---------------------------------------------------------------------------
# Diagnostic: per-word argmax table for seed 0 of each arm
# ---------------------------------------------------------------------------

print("--- DIAGNOSTIC: per-word argmax (seed 0) ---")
for arm in ARMS:
    r = results_store[arm][0]
    P = r["P"]
    Q = r["Q"]
    print(f"\n  {arm} seed=0:")
    print(f"  {'word':<6}  {'P argmax':>8}  {'Q argmax':>8}  {'match':>6}")
    for w in WORDS:
        pa = int(np.argmax(P.vocab[w])) if w in P.vocab else "absent"
        qa = int(np.argmax(Q.vocab[w])) if w in Q.vocab else "absent"
        match = "YES" if pa == qa else "no"
        print(f"  {w:<6}  {str(pa):>8}  {str(qa):>8}  {match:>6}")
    print(f"  spoken pairs: {r['spoken_pairs']}")

print()

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------

checks: list[tuple[str, bool, str]] = []


def check(name: str, predicate_fn):
    """Run predicate_fn(); record (name, pass, detail). Exceptions count as FAIL."""
    try:
        passed, detail = predicate_fn()
        checks.append((name, passed, detail))
    except Exception as exc:
        checks.append((name, False, f"exception: {exc}"))


def _p1_arm(arm: str):
    """P1: cosine_after > cosine_before (severed) in >=4/5 seeds."""
    pass_seeds = []
    seed_vals = []
    for seed in SEEDS:
        r = results_store[arm][seed]
        ok = r["cosine_after"] > r["cosine_before"]
        pass_seeds.append(ok)
        seed_vals.append(
            f"s{seed}:before={r['cosine_before']:.4f},after={r['cosine_after']:.4f},"
            f"{'ok' if ok else 'FAIL'}"
        )
    n_pass = sum(pass_seeds)
    return n_pass >= 4, f"passes={n_pass}/5  [{'; '.join(seed_vals)}]"


def _p2_light():
    """P2-LIGHT: dist_after < 1.0 in >=4/5 seeds."""
    pass_seeds = []
    seed_vals = []
    for seed in SEEDS:
        r = results_store["LIGHT"][seed]
        ok = r["dist_after"] < 1.0
        pass_seeds.append(ok)
        seed_vals.append(f"s{seed}:dist_after={r['dist_after']:.4f},{'ok' if ok else 'FAIL'}")
    n_pass = sum(pass_seeds)
    return n_pass >= 4, f"passes={n_pass}/5  [{'; '.join(seed_vals)}]"


def _p2_heavy():
    """P2-HEAVY: dist_after == 1.0 in >=4/5 seeds."""
    pass_seeds = []
    seed_vals = []
    for seed in SEEDS:
        r = results_store["HEAVY"][seed]
        ok = r["dist_after"] == 1.0
        pass_seeds.append(ok)
        seed_vals.append(f"s{seed}:dist_after={r['dist_after']:.4f},{'ok' if ok else 'FAIL'}")
    n_pass = sum(pass_seeds)
    return n_pass >= 4, f"passes={n_pass}/5  [{'; '.join(seed_vals)}]"


check("P1-LIGHT", lambda: _p1_arm("LIGHT"))
check("P1-HEAVY", lambda: _p1_arm("HEAVY"))
check("P2-LIGHT", _p2_light)
check("P2-HEAVY", _p2_heavy)

print("--- PROPERTY CHECKS ---")
check_map = {}
for name, passed, detail in checks:
    verdict = "PASS" if passed else "FAIL"
    print(f"{verdict}  {name}: {detail}")
    check_map[name] = passed

print()

# ---------------------------------------------------------------------------
# Falsifier map and final verdict
# ---------------------------------------------------------------------------

p1_light = check_map["P1-LIGHT"]
p1_heavy = check_map["P1-HEAVY"]
p2_light = check_map["P2-LIGHT"]
p2_heavy = check_map["P2-HEAVY"]

# F2: HEAVY converges like LIGHT — inertia law does NOT extend to vocab mass
# (HEAVY distance < 1.0 in >=4/5)
heavy_converged = sum(
    1 for seed in SEEDS if results_store["HEAVY"][seed]["dist_after"] < 1.0
) >= 4

f1_fires = (not p1_light) or (not p1_heavy)
f2_fires = heavy_converged  # P2-HEAVY fails because HEAVY itself converged

if f2_fires:
    print("F2: HEAVY converges like LIGHT -> inertia law does NOT extend to vocab mass")
    print("EXP73: F2 — inertia law does not extend to vocab")
elif f1_fires:
    failing = []
    if not p1_light:
        failing.append("P1-LIGHT")
    if not p1_heavy:
        failing.append("P1-HEAVY")
    print(f"F1: channel does not transmit labels at this dose — rung 5 NEGATIVE")
    print(f"EXP73: FAIL {failing}")
elif p1_light and p1_heavy and not p2_light:
    # P1 passes but P2-LIGHT fails — dose insufficient for argmax flips
    print("MIXED: channel transmits labels (P1 passes) but dose insufficient for argmax flips (P2-LIGHT fails)")
    print("EXP73: MIXED [P2-LIGHT]")
elif p1_light and p1_heavy and p2_light and p2_heavy:
    print("RUNG 5: CONVERGENCE IS MASS-GATED — light dialects merge, heavy dialects stand "
          "(stable dialects under coupling at heavy mass)")
    print("EXP73: PASS")
else:
    # Partial failures not covered above
    failing = [n for n, p in check_map.items() if not p]
    print(f"EXP73: FAIL {failing}")
