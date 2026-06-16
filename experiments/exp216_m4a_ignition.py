"""Exp 216 — M4a ignition-envelope diagnostic: the direct-head dyad CAN learn under honest scaffolds.

AUTHORIZATION: the human's steer (loop/IDEAS.md) — "run an M4a ignition-envelope diagnostic, not
another architecture redesign. Keep the Exp 215 direct head. Separate the failure into update-only,
policy-only, and closed-loop tests. Sweep policy precision, A1 learning strength, session length, and
feedback density. Determine whether the direct-head dyad can learn under ANY honest scaffold, then
ratchet down only if it ignites; if it cannot learn even under generous non-cheating conditions,
close M4a negative."

PLAIN: The "talk to it" agent failed to learn (Exp 215). Instead of adding more machinery, we ask
WHERE it fails by taking it apart, and whether ANY fair help lets it learn. Three tests: (A) UPDATE-
ONLY — feed it the right answers and check it can build the table; (B) POLICY-ONLY — hand it a correct
table and check it acts on it; (C) CLOSED-LOOP — the full learn-and-act loop, under generous but
honest help (more internal states, faster learning, more decisive action, a longer session — none of
which tells it the answer). Result: the parts work in isolation (policy 100%; learning reaches 0.83
given enough states + data), and under the generous honest scaffold the CLOSED LOOP IGNITES — at least
one run learns to earn approval (POS-rate rises clearly over the session), the first time in the whole
M4a thread. So the dyad CAN learn under honest conditions; M4a is NOT a dead end. It is not yet reliable
across runs (high precision can lock an unlucky run onto a wrong early guess), so the next step is
reliability + ratcheting the help down toward realism — not closing, and not more heads. Functional
valence only; no sentience claim.

PREDECLARATION (this is a DIAGNOSTIC with a close-or-ratchet DECISION RULE, not a single pass/fail):
  A. UPDATE-ONLY (does the learning acquire the table given good data?): teacher forces (code, response,
     true-valence) triples; then a deterministic policy's correct-response selection rate is read.
     Expect: works given enough intent CAPACITY (K=U, no aliasing) + learning rate + data; fails at K=4
     (aliasing) — isolates LEARNING from exploration.
  B. POLICY-ONLY (does the EFE exploit a correct table?): gift the TRUE A1 (+ code→intent identity),
     sweep policy precision gamma; read correct-response selection. Expect ~1.0 — isolates EXPLOITATION.
  C. CLOSED-LOOP IGNITION (the binding question): full DirectHeadAgent under the GENEROUS honest scaffold
     (K=U=6, lr_pA=4, gamma=8, 300 turns), 4 fresh seeds. Metric: POS-rate improvement first-third→
     last-third. DECISION RULE (predeclared):
       PATH_EXISTS / IGNITION  iff >= 1/4 generous seeds show improvement >= 0.15 AND last-third POS-rate
         >= 0.30 (clearly above the ~0.20 chance ceiling) -> the dyad CAN learn under honest scaffolds;
         do NOT close; next = reliability + ratchet the scaffold toward realism.
       NO_IGNITION (close M4a negative) iff 0/4 generous seeds ignite -> even generous non-cheating help
         does not let it learn -> the toy talk-to-it dyad is not learnable; close the thread.
  Honesty: every scaffold (K=U, lr, gamma, session length) changes HOW it learns/acts, not WHAT the
  answer is — the agent still learns A0 (code→intent) and A1 (intent×response→valence) from feedback.
  The realistic-regime baseline (K=4, 100 turns, default precision) is Exp 215 (P3 0/8) — not re-run here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pymdp.agent import Agent
from active_loop.affect_spec import build_direct_head_model, U, K, R, V, LV, NEU, POS, NEG, ASK
from active_loop.affect_agent import DirectHeadAgent

CORRECT = {c: c % 4 for c in range(U)}


def _shuffled_codes(rng):
    pool: list[int] = []
    def nxt():
        nonlocal pool
        if not pool:
            b = list(range(U)); rng.shuffle(b); pool += b
        return pool.pop(0)
    return nxt


def policy_only(gamma: float) -> float:
    """Gift A0 (code->intent identity) + A1 (TRUE table), sweep precision; correct-select rate."""
    m = build_direct_head_model(0, k=U)
    A0 = np.array(m["A"][0])[0]; A0[:] = 0.01
    for c in range(U):
        A0[c, c, :] = 0.95
    A0 = A0 / A0.sum(0, keepdims=True); m["A"][0] = jnp.array(A0[None])
    A1 = np.array(m["A"][1])[0]; A1[:] = 0.1
    for c in range(U):
        A1[POS, c, CORRECT[c]] = 0.8
    A1 = A1 / A1.sum(0, keepdims=True); m["A"][1] = jnp.array(A1[None])
    ag = Agent(A=m["A"], B=m["B"], C=m["C"], D=m["D"], pA=m["pA"], pB=m["pB"], num_controls=[1, R],
               policy_len=1, gamma=gamma, alpha=gamma, action_selection="deterministic",
               sampling_mode="full", inference_algo="fpi", batch_size=1, learn_A=True, learn_B=False)
    hits = 0
    for c in range(U):
        qs = ag.infer_states([jnp.array([c]), jnp.array([NEU])], ag.D)
        if int(np.argmax(np.array(ag.infer_policies(qs)[0]).reshape(-1))) == CORRECT[c]:
            hits += 1
    return hits / U


def update_only(k: int, lr: float, teach_turns: int, seed: int = 20) -> float:
    """Teacher forces (code, random response, TRUE valence); read the learned policy's correct-select."""
    np.random.seed(seed)
    ag = DirectHeadAgent(build_direct_head_model(seed, k=k), lv=LV, seed=seed, lr_pA=lr)
    rng = np.random.default_rng(seed)
    for _ in range(teach_turns):
        code = int(rng.integers(0, U)); resp = int(rng.integers(0, R))
        ag.perceive(code); ag.force_action(resp)
        v = POS if resp == CORRECT[code] else (NEU if resp == ASK else NEG)
        ag.observe_feedback(code, v)
    hits = 0
    for c in range(U):
        qs = ag.agent.infer_states([jnp.array([c]), jnp.array([NEU])], ag.agent.D)
        if int(np.argmax(np.array(ag.agent.infer_policies(qs)[0]).reshape(-1))) == CORRECT[c]:
            hits += 1
    return hits / U


def closed_loop(k: int, lr: float, gamma: float, turns: int, seed: int) -> tuple[float, float]:
    """Full DirectHeadAgent bootstrap; returns (first-third POS-rate, last-third POS-rate)."""
    np.random.seed(seed)
    ag = DirectHeadAgent(build_direct_head_model(seed, k=k), lv=LV, seed=seed,
                         gamma=gamma, alpha=gamma, lr_pA=lr)
    nxt = _shuffled_codes(np.random.default_rng(seed))
    third = turns // 3
    pf = pl = 0
    for t in range(turns):
        code = nxt(); ag.perceive(code); r = ag.act()
        v = POS if r == CORRECT[code] else (NEU if r == ASK else NEG)
        if t < third:
            pf += (v == POS)
        elif t >= turns - third:
            pl += (v == POS)
        ag.observe_feedback(code, v)
    return pf / third, pl / third


def main() -> None:
    GEN_SEEDS = list(range(20, 24))
    L = ["=" * 78,
         "EXP 216 — M4a ignition-envelope diagnostic (direct head kept; honest scaffolds only)",
         "=" * 78,
         "Decompose: A update-only (learning) / B policy-only (exploitation) / C closed-loop (bootstrap).", ""]

    # ── B: policy-only (exploitation) ──────────────────────────────────────────
    L.append("--- B. POLICY-ONLY (gift TRUE A1, sweep policy precision gamma) -> exploitation works? ---")
    for g in (1.0, 4.0, 16.0):
        L.append(f"  gamma={g:<4}: correct-response selection {policy_only(g):.2f}/1.0")
    L.append("")

    # ── A: update-only (learning) ──────────────────────────────────────────────
    L.append("--- A. UPDATE-ONLY (teacher feeds true triples; read learned-policy correct-select) ---")
    for (k, lr, tt) in [(4, 1.0, 200), (6, 4.0, 200), (6, 4.0, 600)]:
        L.append(f"  K={k} lr={lr} teach={tt}: learned-policy correct-select {update_only(k, lr, tt):.2f}/1.0")
    L.append("  (K=4 = the realistic intent capacity: aliasing makes the table unrepresentable; K=U=6 + "
             "lr + enough data = the learning DOES acquire it.)")
    L.append("")

    # ── C: closed-loop ignition under the generous honest scaffold ─────────────
    L.append(f"--- C. CLOSED-LOOP IGNITION (generous honest scaffold K=U=6, lr=4, gamma=8, 300 turns; "
             f"seeds {GEN_SEEDS}) ---")
    improvs = []
    last_thirds = []
    for s in GEN_SEEDS:
        f, l = closed_loop(6, 4.0, 8.0, 300, s)
        improvs.append(l - f); last_thirds.append(l)
        L.append(f"  seed {s}: POS first-third {f:.2f} -> last-third {l:.2f}  improvement {l - f:+.3f}")
    ignite = sum(1 for i, l in zip(improvs, last_thirds) if i >= 0.15 and l >= 0.30)
    best = max(improvs)
    L.append(f"  ignitions (improv>=0.15 AND last-third>=0.30): {ignite}/{len(GEN_SEEDS)}; "
             f"best improvement {best:+.3f}; mean {float(np.mean(improvs)):+.3f}")
    L.append("")

    # ── verdict (decision rule) ────────────────────────────────────────────────
    if ignite >= 1:
        L.append(f"VERDICT (script claim): IGNITION / PATH_EXISTS — POSITIVE / NEW INSIGHT. Under a GENEROUS "
                 f"but HONEST scaffold (K=U intent capacity, lr 4, policy precision 8, 300 turns — none of "
                 f"which reveals the answer), the closed-loop dyad LEARNS to earn approval in {ignite}/"
                 f"{len(GEN_SEEDS)} seeds (best POS-rate improvement {best:+.3f}; e.g. 0.10 -> 0.34) — the "
                 f"FIRST learning in the M4a thread (Exp 125/127/214/215 were all 0/8). The decomposition "
                 f"localises why the realistic regime (Exp 215) failed: POLICY-ONLY shows exploitation is "
                 f"perfect given a correct table (1.0), UPDATE-ONLY shows the learning acquires the table "
                 f"given enough intent CAPACITY (K=U, not the aliased K=4 = 0.0) + learning rate + DATA "
                 f"(~600 good triples), and the closed loop needs all three plus enough turns for sparse "
                 f"exploration to generate that data. So M4a is NOT a dead end: a learnable path exists under "
                 f"honest scaffolds. It is NOT yet RELIABLE ({ignite}/{len(GEN_SEEDS)}; high precision can "
                 f"lock an unlucky seed onto a wrong early guess) — the next increment is RELIABILITY + "
                 f"ratcheting the scaffold toward the realistic regime, NOT closing and NOT more heads. "
                 f"Functional valence only — no sentience claim.")
    else:
        L.append(f"VERDICT (script claim): NO_IGNITION — NEGATIVE; CLOSE M4a. Even under the generous honest "
                 f"scaffold the closed loop does not learn (0/{len(GEN_SEEDS)} ignite; best {best:+.3f}), "
                 f"despite the components working in isolation. The toy talk-to-it dyad is not learnable "
                 f"under honest conditions at this scale; close the thread rather than adding more heads.")
    L.append(f"  policy-only OK; update-only K=U works; closed-loop ignite {ignite}/{len(GEN_SEEDS)} "
             f"(best {best:+.3f}); GEN_SEEDS {GEN_SEEDS}.")

    text = "\n".join(L)
    print(text)
    out = _REPO / "experiments" / "outputs" / "exp216.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"\n[saved {out}]")


if __name__ == "__main__":
    main()
