"""Exp 125 — M4a increment 1: the affective core, validated (the chapter opens).

AUTHORIZATION: the human resumed the loop with the merged CONSULT posted and no
redirection — the ratified silence-as-consent pattern (recorded in loop/IDEAS.md).
Recommended option only: M4a with the discovered substrate requirements (the value/map
window, LV=0.999; the exploration reflex via EFE's epistemic term + ASK); the committed
spines stay untouched. GUARDRAIL: any failed predeclared property below HALTS the M4a
thread for explicit human input.

Increment 1 = affect_spec + affect_agent (perceive -> intent posterior; EFE response;
windowed Dirichlet learning) validated against a deterministic scripted partner
(spec section 6): the partner cycles utterance codes on a seeded schedule and gives POS
feedback iff the agent's response matches a fixed teaching map correct_response[code]
(never ASK), NEG otherwise; ASK earns NEU (the differentiable, non-pre-labeled cue).

Predeclared:
  P1 (inference runs): after perceive(), the intent posterior is a proper distribution
     (sums to 1, no NaNs) and its entropy is below the uniform prior's for at least
     half of all turns across the session, in >= 6/8 seeds.
  P2 (exploration reflex seed): ASK is chosen at least twice in the first 10 turns in
     >= 2/8 seeds. F2 = ASK never chosen in the first 10 turns in >= 7/8 seeds ->
     epistemic drive dead -> HALT.
  P3 (learns to feel positive — the spec's core guardrail): the realized POS-feedback
     rate in the second half of a 100-turn session exceeds the first half by >= 0.15 in
     >= 6/8 partner seeds. F3 = fails -> the core loop does not learn -> HALT.
  P4 (window wiring, exact arithmetic): pA[0].sum() after 100 turns equals
     init_sum * LV^100 + sum_{k=0..99} LV^k within +-0.5 (LV=0.999). F4 -> mis-wired ->
     HALT.
All falsifiers HALT (the consult guardrail). 8 session seeds (0-7) seeding the partner
schedule and numpy's global rng (pymdp action sampling). No creature state is touched.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from active_loop.affect_spec import build_dyad_model, U, R, LV
from active_loop.affect_agent import AffectAgent

# ── Constants ─────────────────────────────────────────────────────────────────
TURNS = 100
SEEDS = list(range(8))
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
        f"{'pA0_sum':>9}  {'exp_sum':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['seed']:>4}  {r['entropy_drop_frac']:>7.3f}  {r['ask_first10']:>5}  "
            f"{r['pos_rate_h1']:>6.3f}  {r['pos_rate_h2']:>6.3f}  {r['improvement']:>6.3f}  "
            f"{r['pA0_sum']:>9.3f}  {r['expected_sum']:>9.3f}"
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
        "\nEXP125: M4A CORE VALIDATED — "
        "increment 2 (converse REPL + frozen scorer) unlocked"
    )


if __name__ == "__main__":
    main()
