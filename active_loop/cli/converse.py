"""M4a AffectAgent conversational REPL (increment 1e).

Two modes:
  interactive (default): read utterance codes from stdin; print intent belief, response, and
    valence readout; read feedback (+/-/empty); learn.
  --demo: non-interactive scripted-partner session; prints each turn and a summary line.

HONEST BANNER (printed at startup):
  Functional valence only (= -free-energy + grounded approval), NOT subjective feeling.
  The agent clusters your utterance CODES into latent intents and learns (Dirichlet) which
  response earns your +. Intent labels are coarse toy clusters, not language understanding
  (the documented ceiling).
"""
from __future__ import annotations

import argparse

import numpy as np

from active_loop.affect_spec import (
    build_direct_head_model,
    constant_response_ceiling,
    U, R, LV, POS, NEU, ASK,
)
from active_loop.affect_agent import DirectHeadAgent

# ── Frozen winning config (Exp 220 sched_full) ──────────────────────────────
K = 4
OPTIMISM = 2.0
LR = 4.0
TURNS_DEFAULT = 300

CORRECT = {c: c % 4 for c in range(U)}
CEIL = constant_response_ceiling(CORRECT, R)

RESPONSE_NAMES = ["GREET", "MIRROR", "SOOTHE", "PLAY", "ASK"]

BANNER = """\
[M4a AffectAgent]
  Functional valence only (= -free-energy + grounded approval), NOT subjective feeling.
  The agent clusters your utterance CODES into latent intents and learns (Dirichlet) which
  response earns your +. Intent labels are coarse toy clusters, not language understanding
  (the documented ceiling).
"""

FEEDBACK_MAP = {"+": POS, "-": 0, "": NEU}  # 0 == NEG


def _make_agent(seed: int = 0, turns: int = TURNS_DEFAULT) -> DirectHeadAgent:
    return DirectHeadAgent(
        build_direct_head_model(seed, k=K),
        seed=seed,
        gamma=1.0,
        alpha=1.0,
        lr_pA=LR,
        lv=LV,
        optimism=OPTIMISM,
        gamma_schedule=(1.0, 8.0, turns),
    )


def _shuffled_codes(rng):
    """Infinite iterator of codes from exhaustive shuffled blocks (same as Exp 220)."""
    pool: list[int] = []

    def nxt():
        nonlocal pool
        if not pool:
            b = list(range(U))
            rng.shuffle(b)
            pool += b
        return pool.pop(0)

    return nxt


def run_interactive(seed: int = 0, turns: int = TURNS_DEFAULT, agent_factory=None) -> None:
    """Interactive REPL: user types code then feedback, agent learns.

    agent_factory(seed, turns) -> agent (defaults to the frozen winning config).  An
    artifact-backed agent can be supplied without changing the REPL behavior.
    """
    print(BANNER)
    print(f"Utterance codes: 0..{U - 1}.  Feedback: +  -  (empty=neutral).  Type 'q' to quit.")
    print()

    ag = (agent_factory or _make_agent)(seed, turns)
    pos_count = 0
    total = 0

    while True:
        try:
            line = input(f"[turn {total + 1}] utterance code (0-{U - 1}) or q> ").strip()
        except EOFError:
            break
        if line.lower() == "q":
            break
        try:
            code = int(line)
        except ValueError:
            print(f"  Enter an integer 0..{U - 1} or 'q'.")
            continue
        if not (0 <= code < U):
            print(f"  Code must be 0..{U - 1}.")
            continue

        belief = ag.perceive(code)
        r = ag.act()
        vr = ag.valence_readout()
        print(f"  intent belief: {np.round(belief, 3).tolist()}  "
              f"response: {RESPONSE_NAMES[r]}  valence: {vr:.3f}")

        try:
            fb = input("  feedback (+/-/empty=neutral)> ").strip()
        except EOFError:
            fb = ""
        valence = FEEDBACK_MAP.get(fb, NEU)
        ag.observe_feedback(code, valence)
        pos_count += (valence == POS)
        total += 1
        pos_rate = pos_count / total
        print(f"  [learned. running POS rate: {pos_rate:.3f}]")

    if total > 0:
        print(f"\n[session ended after {total} turns. final POS rate: {pos_count/total:.3f}]")


def run_demo(seed: int = 0, turns: int = 60, agent_factory=None) -> None:
    """Non-interactive scripted-partner demo session (same CORRECT mapping as the scorer).

    agent_factory(seed, turns) -> agent (defaults to the frozen winning config).
    """
    print(BANNER)
    print(f"[demo] scripted-partner session: seed={seed} turns={turns}")
    print()

    np.random.seed(seed)
    ag = (agent_factory or _make_agent)(seed, turns)
    nxt = _shuffled_codes(np.random.default_rng(seed))
    third = turns // 3
    pf = pl = 0

    for t in range(turns):
        code = nxt()
        belief = ag.perceive(code)
        r = ag.act()
        valence = POS if r == CORRECT[code] else (NEU if r == ASK else 0)  # 0==NEG
        if t < third:
            pf += (valence == POS)
        elif t >= turns - third:
            pl += (valence == POS)
        label = "POS" if valence == POS else ("NEU" if valence == NEU else "NEG")
        print(f"  t={t:3d} code={code} -> {RESPONSE_NAMES[r]:6s} [{label}]")
        ag.observe_feedback(code, valence)

    first_rate = pf / third
    last_rate = pl / third
    csel = ag.correct_select(CORRECT)
    print()
    print(f"[demo] first-third POS={first_rate:.3f}  last-third POS={last_rate:.3f}  "
          f"correct_select={csel:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M4a AffectAgent REPL (increment 1e): interactive or scripted-partner demo."
    )
    parser.add_argument("--demo", action="store_true",
                        help="Non-interactive scripted-partner demo (no stdin reads).")
    parser.add_argument("--turns", type=int, default=TURNS_DEFAULT,
                        help=f"Session length (default {TURNS_DEFAULT}; demo default 60).")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed (default 0).")
    args = parser.parse_args()

    if args.demo:
        demo_turns = args.turns if args.turns != TURNS_DEFAULT else 60
        run_demo(seed=args.seed, turns=demo_turns)
    else:
        run_interactive(seed=args.seed, turns=args.turns)


if __name__ == "__main__":
    main()
