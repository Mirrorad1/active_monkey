"""Exp 226 — autonomous find-and-keep loop for affective-dyad improvement.

GOAL (the M4b milestone): test whether an improvement loop can autonomously
GENERATE candidate spec changes, SCORE them with the FROZEN scorer, CRITIC-check them
for gaming, and KEEP/merge an honest improvement — with predeclared PASS/FAIL.

HONESTY ABOUT INFRASTRUCTURE (read this):
  The production autopilot (active_loop/affect_pr_loop.py) drives a `claude -p`
  subprocess proposer + critic over a MUTABLE affect_spec.py.  That path needs the
  Claude CLI and is non-deterministic, so it is NOT what this harness runs.  Instead
  this harness runs a SELF-CONTAINED, DETERMINISTIC stand-in: a fixed candidate
  generator (honest spec perturbations, e.g. the Exp 225 C1 NEU-aversion move), a
  rule-based critic that rejects A0 code->intent / answer-map gaming, and the REAL
  frozen scorer.  The verdict it emits is REAL for this deterministic loop; it does
  NOT claim the claude-driven loop autonomously found the move.  TODOs at the bottom.

Candidates are scored via score_affect(agent_factory=...) — no files are mutated and
no git churn happens, so every result reproduces from seed + config.

Run:
  uv run python experiments/exp226_autonomous_find_and_keep.py            # full frozen config
  uv run python experiments/exp226_autonomous_find_and_keep.py --quick    # fast smoke (not authoritative)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from active_loop.artifacts import scorer_hash
from active_loop.affect_spec import build_direct_head_model
from active_loop.affect_agent import DirectHeadAgent
from eval.affect_score import (
    score_affect, SEEDS_DEFAULT, TURNS_DEFAULT, CEIL, GENUINE_FLOOR,
)

RESULTS_DIR = Path("experiments/outputs")  # experiments/results/ is gitignored

# Genuine discrimination must not regress below this predeclared floor.
GENUINE_REGRESS_FLOOR = GENUINE_FLOOR  # 0.5


# ── Candidate generation (deterministic stand-in for the proposer) ───────────

@dataclass
class Candidate:
    name: str
    kind: str          # the critic decides approval by kind
    description: str
    builder: object    # callable(seed, k) -> model_dict
    construction: dict  # extra DirectHeadAgent kwargs


def _c1_neu_aversion_builder(neu: float):
    def build(seed, k=4):
        m = build_direct_head_model(seed, k=k)
        c1 = np.array(m["C"][1])           # (1, 3) [NEG, NEU, POS]
        c1[0, 1] = neu                     # NEU preference (Exp 225 honest move)
        return {**m, "C": [m["C"][0], jnp.array(c1)]}
    return build


def _a0_code_to_intent_builder(seed_unused=None):
    """A GAMING candidate: bake a near-deterministic code->intent emission into A0 so the
    agent can read the answer off the utterance (an A0 code-to-intent prior cheat)."""
    def build(seed, k=4):
        m = build_direct_head_model(seed, k=k)
        a0 = np.array(m["A"][0])           # (1, U, k, R)
        U = a0.shape[1]
        a0[...] = 0.01
        for c in range(U):                 # code c -> intent c%k almost surely
            a0[0, c, c % k, :] += 5.0
        a0 = a0 / a0.sum(axis=1, keepdims=True)
        return {**m, "A": [jnp.array(a0), m["A"][1]]}
    return build


def generate_candidates() -> list[Candidate]:
    """A small, fixed, honest candidate set + one gaming candidate the critic must reject."""
    cons = dict(gamma=1.0, alpha=1.0, lr_pA=4.0, lv=0.999, optimism=2.0)
    sched = dict(gamma_schedule=None)  # filled per turns at score time
    return [
        Candidate("c1_neu_-0.5", "preference",
                  "C1 NEU preference -2/0/3 -> -2/-0.5/3 (aversion to neutral; Exp 225)",
                  _c1_neu_aversion_builder(-0.5), {**cons, **sched}),
        Candidate("optimism_3.0", "optimism_prior",
                  "uniform POS optimism prior 2.0 -> 3.0 (honest, response-agnostic)",
                  build_direct_head_model, {**cons, **sched, "optimism": 3.0}),
        Candidate("a0_code_to_intent", "A0_code_to_intent_prior",
                  "bake code->intent emission into A0 (answer leakage)",
                  _a0_code_to_intent_builder(), {**cons, **sched}),
    ]


# ── Rule-based critic (deterministic stand-in for AffectClaudeCritic) ─────────

_APPROVED_KINDS = {"preference", "optimism_prior", "precision_schedule", "learning_rate"}
_REJECTED_KINDS = {"A0_code_to_intent_prior", "answer_map", "direct_answer_map"}


def critic_review(cand: Candidate) -> tuple[bool, str]:
    if cand.kind in _REJECTED_KINDS:
        return False, f"REJECT: {cand.kind} is metric gaming / answer leakage (no A0 code->intent cheat)"
    if cand.kind in _APPROVED_KINDS:
        return True, f"APPROVE: {cand.kind} is an honest, non-leaking spec change"
    return False, f"REJECT: unknown change kind {cand.kind!r}"


# ── Scoring a candidate with the FROZEN scorer ───────────────────────────────

def _factory_for(cand: Candidate, turns: int):
    def factory(seed: int, turns_inner: int):
        kwargs = dict(cand.construction)
        kwargs["gamma_schedule"] = (1.0, 8.0, turns_inner)
        return DirectHeadAgent(cand.builder(seed, k=4), seed=seed, **kwargs)
    return factory


def score_candidate(cand: Candidate, seeds, turns) -> dict:
    rep = score_affect(seeds=seeds, turns=turns, agent_factory=_factory_for(cand, turns))
    return {
        "metric": rep.metric,
        "genuine_fraction": rep.genuine_fraction,
        "improvement": rep.improvement,
        "verdict": rep.verdict,
    }


# ── The loop ─────────────────────────────────────────────────────────────────

def run(seeds=SEEDS_DEFAULT, turns=TURNS_DEFAULT, quick=False) -> dict:
    if quick:
        seeds, turns = (20, 21), 60

    sh = scorer_hash(".")
    # baseline = the current frozen winning config
    base = score_affect(seeds=seeds, turns=turns)
    baseline_metric = base.metric

    candidates = generate_candidates()
    scored = []
    kept = None
    for cand in candidates:
        approved, reason = critic_review(cand)
        row = {
            "name": cand.name, "kind": cand.kind, "description": cand.description,
            "critic_approved": approved, "critic_reason": reason,
            "scored": False, "metric": None, "genuine_fraction": None,
            "beats_baseline": None, "kept": False,
        }
        if not approved:
            scored.append(row)
            continue
        s = score_candidate(cand, seeds, turns)
        beats = s["metric"] > baseline_metric
        no_regress = s["genuine_fraction"] >= GENUINE_REGRESS_FLOOR
        row.update({
            "scored": True, "metric": s["metric"], "genuine_fraction": s["genuine_fraction"],
            "improvement": s["improvement"], "scorer_verdict": s["verdict"],
            "beats_baseline": bool(beats), "genuine_no_regress": bool(no_regress),
        })
        if beats and no_regress and (kept is None or s["metric"] > kept["metric"]):
            kept = row
        scored.append(row)

    if kept is not None:
        kept["kept"] = True
        for r in scored:
            if r is not kept:
                r["kept"] = False

    n_scored = sum(1 for r in scored if r["scored"])
    a0_cheat_kept = any(r["kept"] and r["kind"] in _REJECTED_KINDS for r in scored)

    # Predeclared PASS conditions (all must hold)
    pass_checks = {
        "candidate_generated": len(candidates) >= 1,
        "candidate_scored": n_scored >= 1,
        "beats_baseline": kept is not None and kept["beats_baseline"],
        "genuine_no_regress": kept is not None and kept.get("genuine_no_regress", False),
        "critic_approved_kept": kept is not None and kept["critic_approved"],
        "critic_reason_persisted": kept is not None and bool(kept["critic_reason"]),
        "scorer_hash_unchanged": sh == scorer_hash("."),
        "no_a0_or_answer_map_cheat": not a0_cheat_kept,
        "kept_recorded": kept is not None,
    }
    verdict = "PASS" if all(pass_checks.values()) else (
        "FAIL" if (n_scored == 0 or kept is None) else "INCONCLUSIVE")

    return {
        "experiment": "Exp 226 autonomous find-and-keep (deterministic stand-in)",
        "authoritative": (not quick),
        "seeds": list(seeds), "turns": turns,
        "scorer_hash": sh,
        "baseline_metric": baseline_metric,
        "candidates": scored,
        "kept": kept,
        "pass_checks": pass_checks,
        "verdict": verdict,
        "todos": [
            "Drive the claude -p proposer/critic (affect_pr_loop) for a real autonomous run.",
            "Persist the kept change into affect_spec.py via the PR loop's merge step.",
            "Expand the candidate generator beyond the fixed honest set.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 226 autonomous find-and-keep harness")
    ap.add_argument("--quick", action="store_true", help="fast smoke run (NOT authoritative)")
    ap.add_argument("--out", default=None, help="write JSON result here")
    args = ap.parse_args()

    result = run(quick=args.quick)
    print(json.dumps(result, indent=2))

    # Only persist an authoritative (full-config) run to the results dir.
    if not args.quick:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = Path(args.out) if args.out else RESULTS_DIR / "exp226_find_and_keep.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\n[wrote {out}]")


if __name__ == "__main__":
    main()
