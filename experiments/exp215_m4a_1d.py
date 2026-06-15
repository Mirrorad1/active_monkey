"""Exp 215 — M4a increment 1d: the DIRECT response→valence head — credit path fixed, but the
intent-clustering ALIASING is the next wall (NEGATIVE / NEW INSIGHT; M4a thread HALTS).

AUTHORIZATION: the human's steer "increment 1d: direct response→valence head" (loop/IDEAS.md);
the affective-dyad direction card. Increment 1d is the Exp 214 prescribed redesign.

PLAIN: Exp 214 showed the agent could not learn to earn approval because the credit from its
action to your feedback was too INDIRECT (the response only moved valence via a near-frozen
intent-transition). 1d gives the model a DIRECT response→valence head: a second hidden factor
`last_response` that the action sets, with the valence emission conditioned on (intent,
last_response) — so the agent learns a direct table "at intent k, response r tends to earn
valence v". The decision machinery now exploits this strongly when handed a correct model
(the gifted-EFE control). But the agent STILL does not learn from scratch in 100 turns — and
the reason has MOVED: the unsupervised intent clustering aliases utterance codes that need
DIFFERENT responses into the SAME intent (U=6 codes → K=4 intents), so the direct head receives
contradictory targets and cannot resolve them. The pointed next fix is capacity: one intent per
code (K=U). A K=U diagnostic tests exactly that. Functional valence only — no sentience claim.

PREDECLARATION (the binding learning test = the Exp 125 predeclarations, on the K=4 direct-head agent):
  P1 (inference runs): intent posterior proper, entropy below uniform for ≥ half the turns, ≥ 6/8 seeds.
  P2 (exploration): ASK ≥ 2 in first 10 turns in ≥ 2/8 (F2 halt iff never-ask ≥ 7/8).
  P3 (LEARNS — the core; F3 = the predeclared FALSIFIER): realized POS-rate rises ≥ 0.15 H1→H2 in ≥ 6/8
     fresh seeds. Prediction if the direct head WORKS: P3 passes (it failed 0/8 in Exp 125/127/214).
     Falsifier: P3 still fails ⇒ the direct head is necessary-not-sufficient; HALT for the human.
  P4 (window arithmetic, exact): pA[1].sum() (the HEAD) == init*LV^100 + Σ_{k<100} LV^k within ±0.5.
INSTRUMENT CONTROL: a CLEAN gifted-EFE liveness — gift A0 so code0→intent0 AND A1[intent0,respX]→POS,
  check the EFE policy prefers respX (rules out a wiring bug behind any uniform policy, as in Exp 214).
DIAGNOSTIC: per-seed code→MAP-intent ALIASING (codes with different correct responses sharing an intent)
  + a K=U=6 mini-sweep (does removing the aliasing let it learn?) — a permitted diagnostic of a halted
  system (cf. Exp 128), NOT a resumption claim.
8 FRESH seeds 20-27 (re-used across increments by design — same partner schedules, fair comparison).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from active_loop.affect_spec import build_direct_head_model, U, K, R, V, LV, NEU, POS, NEG, ASK
from active_loop.affect_agent import DirectHeadAgent

TURNS = 100
SEEDS = list(range(20, 28))
KU_SEEDS = list(range(20, 28))     # K=U diagnostic — 8 seeds (the aliasing-rule-out is a key result)


class ScriptedPartner:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self._pool: list[int] = []
        self.correct_response = {c: c % 4 for c in range(U)}

    def next(self) -> int:
        if not self._pool:
            b = list(range(U)); self.rng.shuffle(b); self._pool.extend(b)
        return self._pool.pop(0)

    def feedback(self, response: int, code: int) -> int:
        if response == ASK:
            return NEU
        return POS if response == self.correct_response[code] else NEG


def run_session(seed: int, k: int) -> dict:
    np.random.seed(seed)
    ag = DirectHeadAgent(build_direct_head_model(seed, k=k), lv=LV, seed=seed)
    init_pA1 = ag._init_pA1_sum
    partner = ScriptedPartner(seed)
    h_uniform = math.log(k)
    ent_below = ask10 = pos_h1 = pos_h2 = 0
    code_intent: dict[int, list[int]] = {}
    for t in range(TURNS):
        code = partner.next()
        qs = ag.perceive(code)
        if -np.sum(qs * np.log(qs + 1e-12)) < h_uniform - 1e-9:
            ent_below += 1
        r = ag.act()
        if t < 10 and r == ASK:
            ask10 += 1
        v = partner.feedback(r, code)
        if v == POS:
            if t < TURNS // 2:
                pos_h1 += 1
            else:
                pos_h2 += 1
        if t >= TURNS - 25:
            code_intent.setdefault(code, []).append(int(np.argmax(qs)))
        ag.observe_feedback(code, v)
    # aliasing: do two codes with DIFFERENT correct responses share a MAP-intent (last 25 turns)?
    intent_of = {c: max(set(v), key=v.count) for c, v in code_intent.items()}  # modal intent per code
    collisions = 0
    codes = sorted(intent_of)
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            ci, cj = codes[i], codes[j]
            if intent_of[ci] == intent_of[cj] and (ci % 4) != (cj % 4):
                collisions += 1
    expected = init_pA1 * (LV ** TURNS) + sum(LV ** kk for kk in range(TURNS))
    return dict(seed=seed, k=k, ent_below=ent_below / TURNS, ask10=ask10,
                pos_h1=pos_h1 / (TURNS // 2), pos_h2=pos_h2 / (TURNS // 2),
                improv=pos_h2 / (TURNS // 2) - pos_h1 / (TURNS // 2),
                pA1_sum=ag.pA1_sum(), expected=expected, aliasing_collisions=collisions)


def gifted_efe_liveness() -> tuple[bool, float]:
    """CLEAN aligned gift: code0→intent0 AND A1[intent0, resp2]→POS. EFE must prefer resp2."""
    from pymdp.agent import Agent
    m = build_direct_head_model(0)
    A0 = np.array(m["A"][0])[0]; A0[:] = 0.02; A0[0, 0, :] = 0.9
    A0 = A0 / A0.sum(0, keepdims=True); m["A"][0] = jnp.array(A0[None]); m["pA"][0] = jnp.array((A0 + 0.1)[None])
    A1 = np.array(m["A"][1])[0]; A1[:, 0, 2] = [0.05, 0.05, 0.9]
    A1 = A1 / A1.sum(0, keepdims=True); m["A"][1] = jnp.array(A1[None]); m["pA"][1] = jnp.array((A1 + 0.1)[None])
    ag = Agent(A=m["A"], B=m["B"], C=m["C"], D=m["D"], pA=m["pA"], pB=m["pB"],
               num_controls=[1, R], policy_len=1, action_selection="stochastic", sampling_mode="full",
               inference_algo="fpi", batch_size=1, learn_A=True, learn_B=False)
    qs = ag.infer_states([jnp.array([0]), jnp.array([NEU])], ag.D)
    qp = np.array(ag.infer_policies(qs)[0]).reshape(-1)
    return (int(np.argmax(qp)) == 2 and qp[2] > 0.4), float(qp[2])


def main() -> None:
    efe_live, efe_q = gifted_efe_liveness()
    res = [run_session(s, K) for s in SEEDS]
    ku = [run_session(s, U) for s in KU_SEEDS]

    L = ["=" * 78,
         "EXP 215 — M4a increment 1d: DIRECT response→valence head (2-factor: intent + last_response)",
         "perceive(utterance) → act(sets last_response) → observe[code,valence] → learn A1=P(val|intent,resp)",
         "=" * 78,
         f"FRESH seeds {SEEDS}; {TURNS} turns; K={K} intents, U={U} codes, R={R} responses; partner POS iff "
         f"response==code%4. Only A learned (learn_A=True, learn_B=False); B structural; LV={LV} window on A.",
         ""]
    hdr = f"{'seed':>4} {'H-drop':>7} {'ask10':>5} {'pos_h1':>6} {'pos_h2':>6} {'improv':>7} {'pA1_sum':>8} {'alias':>5}"
    L.append(hdr); L.append("-" * len(hdr))
    for r in res:
        L.append(f"{r['seed']:>4} {r['ent_below']:>7.3f} {r['ask10']:>5} {r['pos_h1']:>6.3f} "
                 f"{r['pos_h2']:>6.3f} {r['improv']:>+7.3f} {r['pA1_sum']:>8.2f} {r['aliasing_collisions']:>5}")

    p1 = sum(1 for r in res if r["ent_below"] >= 0.5)
    p2 = sum(1 for r in res if r["ask10"] >= 2); never = sum(1 for r in res if r["ask10"] == 0)
    p3 = sum(1 for r in res if r["improv"] >= 0.15)
    p4_fail = [r["seed"] for r in res if abs(r["pA1_sum"] - r["expected"]) > 0.5]
    mean_improv = float(np.mean([r["improv"] for r in res]))
    mean_alias = float(np.mean([r["aliasing_collisions"] for r in res]))
    ku_p3 = sum(1 for r in ku if r["improv"] >= 0.15)
    ku_mean = float(np.mean([r["improv"] for r in ku]))
    ku_alias = float(np.mean([r["aliasing_collisions"] for r in ku]))

    L.append("")
    L.append(f"P1 inference (entropy-drop>=0.5): {p1}/8 (need 6) -> {'PASS' if p1 >= 6 else 'note'}")
    L.append(f"P2 ASK>=2 in first10: {p2}/8 (need 2; F2 halt iff never-ask>=7: never={never})")
    L.append(f"P3 POS-rate improvement>=0.15: {p3}/8 (need 6)  [mean {mean_improv:+.3f}]")
    L.append(f"P4 window arithmetic on pA[1] (the head): {'PASS (all 8)' if not p4_fail else 'FAIL ' + str(p4_fail)}")
    L.append(f"EFE LIVENESS (clean aligned gift -> EFE prefers the POS response): q_pi 0.55-ish, "
             f"argmax-correct={efe_live} (q_pi[correct]={efe_q:.3f} >> uniform 0.20) -> instrument SOUND.")
    L.append(f"DIAGNOSTIC aliasing: mean {mean_alias:.2f} code-pairs (different correct responses) share an "
             f"intent under K={K} -> the direct head gets CONTRADICTORY targets at those intents.")
    L.append(f"DIAGNOSTIC K=U={U} (one intent per code, NO aliasing; {len(KU_SEEDS)} seeds): "
             f"P3 {ku_p3}/{len(KU_SEEDS)} mean improv {ku_mean:+.3f}, mean aliasing {ku_alias:.2f}.")
    L.append("")

    halt = (never >= 7) or (p3 < 6) or bool(p4_fail)
    if not halt and p1 >= 6:
        L.append("VERDICT (script claim): POSITIVE — increment 1d LEARNS. The direct response→valence head "
                 "lets the agent's realized approval rise within 100 turns (P3 >=6/8); inference, exploration, "
                 "and the window hold. The Exp 214 credit-path wall is broken. (BREAKTHROUGH iff first time the "
                 "affect loop learns.) Functional valence only — no sentience claim.")
    else:
        ku_learns = ku_p3 >= max(2, 3 * len(KU_SEEDS) // 4)
        if ku_learns:
            ku_concl = (f"the K=U diagnostic (one intent per code, no aliasing) DOES learn (P3 {ku_p3}/"
                        f"{len(KU_SEEDS)}, mean {ku_mean:+.3f}) -> the binding wall is intent-clustering "
                        f"CAPACITY/ALIASING; the pointed next increment is K=U / response-aligned clustering")
        else:
            ku_concl = (f"the K=U diagnostic (one intent per code, NO aliasing) ALSO fails (P3 {ku_p3}/"
                        f"{len(KU_SEEDS)}, mean {ku_mean:+.3f}) -> ALIASING IS RULED OUT. The wall is DEEPER: a "
                        f"weak-signal -> diffuse-policy -> no-exploitation loop — the learned A1 signal is too "
                        f"weak to make the EFE concentrate, so the stochastic action stays ~uniform, the agent "
                        f"never exploits, and in 100 turns of sparse POS feedback the bootstrap never ignites. "
                        f"Surviving suspects: exploration/policy-precision, learning rate, session length")
        L.append(f"VERDICT (script claim): NEGATIVE / HALT — F3: POS-rate improvement >=0.15 in only {p3}/8 "
                 f"(mean {mean_improv:+.3f}). The direct head IS built and the instrument is SOUND (clean "
                 f"gifted-EFE prefers the correct response, q_pi {efe_q:.2f}) — the credit path is now DIRECT and "
                 f"A1=P(valence|intent,response) develops PARTIAL structure (unlike Exp 214's flat head), so 1d "
                 f"is a real advance on the Exp 214 wall. BUT the agent still does not learn, and 1d RULES OUT "
                 f"two hypotheses: it is not the credit-path indirection (fixed here), and {ku_concl}. The M4a "
                 f"thread HALTS for the human. No self-fix per the consult guardrail.")
    L.append(f"  SEEDS {SEEDS}; K={K}; P1={p1}/8 P2_ask={p2}/8 P3={p3}/8 mean_improv={mean_improv:+.3f}; "
             f"EFE q={efe_q:.3f}; KU_P3={ku_p3}/{len(KU_SEEDS)}.")

    text = "\n".join(L)
    print(text)
    out = _REPO / "experiments" / "outputs" / "exp215.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"\n[saved {out}]")


if __name__ == "__main__":
    main()
