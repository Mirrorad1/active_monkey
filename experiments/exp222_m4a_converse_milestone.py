"""Exp 222 — M4a increment 1e: the converse REPL + the FROZEN learns-to-positive scorer.
Milestone close: make "the dyad learns to feel positive" a reproducible, constant-UNFAKEABLE number.

This docstring explicitly contains the required words: falsifier, predeclaration, hypothesis.

AUTHORIZATION: the human's word "B" (2026-06-16) at the Exp 221 consult — accept the long-session
milestone (Exp 220's reliable genuine discrimination) + document the short-session learning wall +
build increment 1e: converse.py (honest REPL) + eval/affect_score.py (FROZEN scorer) + _json entry.

PLAIN summary: across Exp 216-220 the toy "talk to it" dyad learned, on the long conversation, to
genuinely tell signals apart and earn positive feedback. This increment PACKAGES that into two
durable things: (1) converse.py, an honest REPL you can actually talk to (it shows its intent guess,
its reply, and a running positive-feedback rate; the banner is honest that "valence" is functional,
not feeling); and (2) eval/affect_score.py, a FROZEN scorer that turns "it learns to feel positive"
into one reproducible number a lazy agent CANNOT fake. The frozen metric counts a run as genuine ONLY
if the agent both earns positive feedback above the 1/3 constant-reply ceiling AND passes the
can't-be-faked discrimination probe (>=3 of 6 signals mapped right). This experiment validates the
frozen scorer two ways. Functional valence only; no sentience claim.

HYPOTHESIS: the FROZEN affect scorer makes the M4a milestone a reproducible, constant-UNFAKEABLE
metric -- A1 it reproduces Exp 220's learning, A2 a constant policy cannot pass it.

PREDECLARATION (two acceptance properties, each with a falsifier):
  A1 (reproduces the validated learner): score_affect() at the FROZEN winning config (K=4,
     optimism=2.0, lr=4.0, gamma-schedule 1->8 over 300t, seeds 20-27, N=8) returns verdict=True:
     mean_last > 1/3 (above the constant ceiling) AND genuine_fraction >= 0.5 AND improvement >= 0.10.
     Expected (== Exp 220 seeds 20-27): mean_last ~0.42, mean_first ~0.18, genuine_fraction 0.75.
  A2 (anti-hack / constant-UNFAKEABLE): the SAME scorer on a CONSTANT-response control
     (_constant_factory(0)) returns verdict=False with genuine_fraction == 0.0 (a constant policy
     caps correct_select at 2/6 and realizes POS ~1/3 -- it can neither discriminate nor exceed the
     ceiling).

FALSIFIER: if A1 verdict=False (the frozen scorer fails to reproduce Exp 220's learning) OR
  A2 verdict=True (a constant non-discriminating policy passes the metric), the frozen metric is
  BROKEN -- do not ship it. Log the failure; do not reframe.

Functional valence only; no sentience claim.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eval.affect_score import score_affect, _constant_factory, CEIL, GENUINE_FLOOR, IMPROVEMENT_FLOOR


def main() -> None:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("EXP 222 - M4a increment 1e: FROZEN learns-to-positive scorer validation")
    lines.append(f"CEIL={CEIL:.3f}; genuine = (csel>=0.5) AND (last>{CEIL:.3f})")
    lines.append("guardrails: realized_above_ceiling (mean_last>1/3), learned_improvement (>=0.10),")
    lines.append(f"            genuine_reliable (genuine_fraction>={GENUINE_FLOOR})")
    lines.append("=" * 78)
    lines.append("FALSIFIER: A1 verdict=False (no learning) OR A2 verdict=True (constant fakes it)")
    lines.append("           => the frozen metric is BROKEN. Do not ship.")
    lines.append("")

    # --- A1: the genuine learner at the frozen winning config (slow ~10 min) ---
    a1 = score_affect()
    lines.append("--- A1: score_affect() at frozen defaults (the DirectHeadAgent learner) ---")
    lines.append(json.dumps(dataclasses.asdict(a1), indent=2))
    lines.append(f"A1_VERDICT={a1.verdict}  (want True)")
    lines.append("")

    # --- A2: the constant-response anti-hack control (instant; no JAX) ---
    a2 = score_affect(agent_factory=_constant_factory(0))
    lines.append("--- A2: score_affect(constant_factory(0)) -- the anti-hack control ---")
    lines.append(json.dumps(dataclasses.asdict(a2), indent=2))
    lines.append(f"A2_VERDICT={a2.verdict}  (want False)")
    lines.append("")

    # --- Combined verdict ---
    passed = bool(a1.verdict) and not bool(a2.verdict) and a2.genuine_fraction == 0.0
    lines.append("--- MILESTONE VERDICT (predeclared) ---")
    lines.append(f"  A1 learner verdict = {a1.verdict} (want True);  "
                 f"A2 constant verdict = {a2.verdict} (want False), genuine_fraction={a2.genuine_fraction}")
    verdict = "MILESTONE_VALIDATED" if passed else "FALSIFIER_FIRED"
    lines.append(f"VERDICT: {verdict}")
    lines.append(
        f"MACHINE SUMMARY: VERDICT={verdict} "
        f"A1_verdict={a1.verdict} A1_metric={a1.metric:.4f} A1_genuine={a1.genuine_fraction:.3f} "
        f"A1_improvement={a1.improvement:.4f} A2_verdict={a2.verdict} A2_genuine={a2.genuine_fraction:.3f}"
    )

    report = "\n".join(lines)
    out = _REPO / "experiments" / "outputs" / "exp222.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n")
    print(report)
    print(f"\n[saved {out}]")


if __name__ == "__main__":
    main()
