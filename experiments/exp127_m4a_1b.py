"""Exp 127 — M4a increment 1b: B-learning enabled — the halted thread resumes on the
human's word.

RESUMPTION: the human, present and notified of the Exp 125 halt (whose one pending
question was the recommended resumption), said "continue" — recorded in loop/IDEAS.md as
the word on the RECOMMENDED OPTION ONLY: enable the spec's own learn_B (the severed
response->valence pathway), identical predeclarations, FRESH seeds 8-15. The same
guardrail binds: any failed predeclared property HALTS the thread again.

Increment 1b = Exp 125's exact validation with pB learning enabled in observe_feedback.
Predeclared (identical to Exp 125, fresh seeds):
  P1 (inference runs): proper posterior; entropy below uniform for >= half of turns, in
     >= 6/8 seeds.
  P2 (exploration reflex): ASK >= 2 in first 10 turns in >= 2/8 seeds; F2 = ASK never in
     first 10 in >= 7/8 -> HALT.
  P3 (learns to feel positive): POS rate improvement H1->H2 >= 0.15 in >= 6/8 seeds.
     F3 = fails -> HALT (deeper diagnosis for the human: turns budget / learning rates /
     model capacity).
  P4 (window wiring): pA[0].sum() after 100 turns == init*LV^100 + Sigma LV^k within
     +-0.5; F4 -> HALT. (pB now also decays; its sum reported as diagnostic, not banded
     — its per-turn added mass under the JAX update is not unit-normalized a priori.)
All falsifiers HALT. Seeds 8-15. No creature state is touched.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from active_loop.affect_spec import build_dyad_model, U, R, LV
from active_loop.affect_agent import AffectAgent

# ── Constants ─────────────────────────────────────────────────────────────────
TURNS = 100
SEEDS = range(8, 16)
NEG, NEU, POS = 0, 1, 2
ASK_IDX = 4  # R-1; must match affect_spec.ASK


# ── Scripted Partner ──────────────────────────────────────────────────────────

class ScriptedPartner:
    """Cycles utterance codes on a seeded shuffled schedule; gives deterministic feedback."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self._pool: list[int] = []
        # teaching map: correct response for each utterance code (never ASK)
        self.correct_response = {c: c % 4 for c in range(U)}

    def _refill(self) -> None:
        batch = list(range(U))
        self.rng.shuffle(batch)
        self._pool.extend(batch)

    def next(self) -> int:
        if not self._pool:
            self._refill()
        return self._pool.pop(0)

    def feedback(self, response: int, code: int) -> int:
        """Return POS if response == correct, NEU if response == ASK, NEG otherwise."""
        if response == ASK_IDX:
            return NEU
        if response == self.correct_response[code]:
            return POS
        return NEG


# ── Session runner ─────────────────────────────────────────────────────────────

def run_session(seed: int) -> dict:
    np.random.seed(seed)

    model = build_dyad_model(seed)
    ag = AffectAgent(model, lv=LV, seed=seed)
    init_pA0_sum = ag._init_pA0_sum

    partner = ScriptedPartner(seed)

    h_uniform = math.log(max(1, 4))  # ln(K)

    entropy_below_uniform = 0
    ask_first10 = 0
    pos_h1 = 0
    pos_h2 = 0
    valence_idx = NEU  # first turn: neutral prior

    for t in range(TURNS):
        code = partner.next()
        qs = ag.perceive(code, valence_idx)

        # P1 entropy check
        h = -np.sum(qs * np.log(qs + 1e-12))
        if h < h_uniform - 1e-9:
            entropy_below_uniform += 1

        response = ag.act()

        # P2 ASK in first 10
        if t < 10 and response == ASK_IDX:
            ask_first10 += 1

        valence_idx = partner.feedback(response, code)

        # P3 POS rate by half
        if valence_idx == POS:
            if t < TURNS // 2:
                pos_h1 += 1
            else:
                pos_h2 += 1

        ag.observe_feedback(code, valence_idx)

    final_pA0_sum = ag.pA0_sum()
    final_pB0_sum = ag.pB0_sum()

    # P4: expected sum after TURNS turns with per-turn added mass ~1.0
    # expected = init_sum * LV^n + sum_{k=0..n-1} LV^k
    expected_sum = init_pA0_sum * (LV ** TURNS) + sum(LV ** k for k in range(TURNS))

    return dict(
        seed=seed,
        entropy_drop_frac=entropy_below_uniform / TURNS,
        ask_first10=ask_first10,
        pos_rate_h1=pos_h1 / (TURNS // 2),
        pos_rate_h2=pos_h2 / (TURNS // 2),
        improvement=pos_h2 / (TURNS // 2) - pos_h1 / (TURNS // 2),
        pA0_sum=final_pA0_sum,
        expected_sum=expected_sum,
        init_pA0_sum=init_pA0_sum,
        pB0_sum=final_pB0_sum,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    results = []
    for seed in SEEDS:
        print(f"  running seed {seed} ...", flush=True)
        results.append(run_session(seed))

    # ── Print table ──────────────────────────────────────────────────────────
    header = (
        f"{'seed':>4}  {'H-drop%':>7}  {'ask10':>5}  "
        f"{'pos_h1':>6}  {'pos_h2':>6}  {'improv':>6}  "
        f"{'pA0_sum':>9}  {'exp_sum':>9}  {'pB0_sum':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['seed']:>4}  {r['entropy_drop_frac']:>7.3f}  {r['ask_first10']:>5}  "
            f"{r['pos_rate_h1']:>6.3f}  {r['pos_rate_h2']:>6.3f}  {r['improvement']:>6.3f}  "
            f"{r['pA0_sum']:>9.3f}  {r['expected_sum']:>9.3f}  {r['pB0_sum']:>9.3f}"
        )

    # ── P1: inference runs ────────────────────────────────────────────────────
    p1_pass = sum(1 for r in results if r["entropy_drop_frac"] >= 0.5)
    if p1_pass < 6:
        print(f"\nP1 FAIL: only {p1_pass}/8 seeds had entropy-drop fraction >= 0.5")
        print("(informational; not a HALT falsifier)")
    else:
        print(f"\nP1 PASS: {p1_pass}/8 seeds entropy-drop fraction >= 0.5")

    # ── P2: exploration reflex ────────────────────────────────────────────────
    p2_ask_seeds = sum(1 for r in results if r["ask_first10"] >= 2)
    p2_never_ask = sum(1 for r in results if r["ask_first10"] == 0)
    if p2_never_ask >= 7:
        print(f"M4a THREAD HALTED — F2: ASK never chosen in first 10 turns in "
              f"{p2_never_ask}/8 seeds — epistemic drive dead.")
        sys.exit(1)
    if p2_ask_seeds >= 2:
        print(f"P2 PASS: ASK >= 2 times in first 10 turns in {p2_ask_seeds}/8 seeds")
    else:
        print(f"P2 note: ASK >= 2 times in first 10 turns in only {p2_ask_seeds}/8 seeds "
              f"(threshold 2/8 — not a HALT)")

    # ── P3: learns to feel positive ───────────────────────────────────────────
    p3_pass = sum(1 for r in results if r["improvement"] >= 0.15)
    if p3_pass < 6:
        print(f"M4a THREAD HALTED — F3: POS-rate improvement >= 0.15 in only "
              f"{p3_pass}/8 seeds (need 6). Core loop does not learn.")
        sys.exit(1)
    print(f"P3 PASS: POS-rate improvement >= 0.15 in {p3_pass}/8 seeds")

    # ── P4: window wiring exact arithmetic ───────────────────────────────────
    p4_fail_seeds = []
    for r in results:
        if abs(r["pA0_sum"] - r["expected_sum"]) > 0.5:
            p4_fail_seeds.append(
                f"seed {r['seed']}: got {r['pA0_sum']:.4f}, expected {r['expected_sum']:.4f}"
            )
    if p4_fail_seeds:
        print(f"M4a THREAD HALTED — F4: window wiring mis-wired:")
        for s in p4_fail_seeds:
            print(f"  {s}")
        sys.exit(1)
    print("P4 PASS: pA[0].sum() matches LV-decay arithmetic within +-0.5 in all seeds")

    print(
        "\nEXP127: M4A CORE LEARNS — increment 2 (converse REPL + frozen scorer) unlocked"
    )


if __name__ == "__main__":
    main()
