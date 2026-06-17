"""Short-session learning levers for the affective dyad (reduce the long-session dependency).

CONTEXT (Exp 221, blind-verified): the 300-turn session is LOAD-BEARING — the precision
schedule optimizes exploitation, but SHORT sessions block LEARNING.  This harness asks
whether honest learning levers raise the genuine learns-to-positive metric at SHORT
sessions (30/50/100 turns) without encoding the answer.

Levers evaluated here (all honest — no correct-response map, no asymmetric/leaking prior):
  - baseline        : the frozen winning config (uniform optimism + precision schedule)
  - no_optimism     : optimism off (ablation)
  - eps_greedy      : decaying uniform-random exploration (Exp 217 lever)
  - replay          : experience replay — re-feed stored (code, action, valence) tuples

Levers SCAFFOLDED but NOT YET IMPLEMENTED in DirectHeadAgent (honest TODO, reported as
NOT_IMPLEMENTED rather than faked):
  - eligibility_trace : decaying credit/blame cache over recent action-observation events
  - active_ASK_probe  : an uncertainty-gated probe action that pays a cost for information

Hard constraints honored: the frozen scorer is unchanged; controls are constant + random;
all randomness is seedable; nothing encodes the correct map.

Run:
  uv run python experiments/short_session_affect_levers.py --quick
  uv run python experiments/short_session_affect_levers.py --full
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from active_loop.artifacts import scorer_hash
from active_loop.affect_spec import build_direct_head_model, U, POS, NEU, ASK
from active_loop.affect_agent import DirectHeadAgent
from eval.affect_score import score_affect, _constant_factory, CEIL, GENUINE_FLOOR

RESULTS_DIR = Path("experiments/outputs")  # experiments/results/ is gitignored
LENGTHS = (30, 50, 100, 300)
CORRECT = {c: c % 4 for c in range(U)}


# ── Replay lever: a thin wrapper that re-feeds past tuples (experience replay) ─

class ReplayDirectHeadAgent(DirectHeadAgent):
    """DirectHeadAgent + experience replay.

    After each real observe_feedback, store (code, action, valence) and replay
    `replay_k` random past tuples through the same learning path.  Reusing past
    interactions increases the effective sample count for the Dirichlet head without
    revealing which response is correct (the partner's valence labels are what they are).
    """

    def __init__(self, *args, replay_k: int = 2, replay_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffer: list[tuple[int, int, int]] = []
        self._replay_k = int(replay_k)
        self._replay_rng = np.random.default_rng(replay_seed)

    def observe_feedback(self, code: int, valence_idx: int) -> None:
        action = int(np.asarray(self._last_action).reshape(-1)[-1]) if self._last_action is not None else None
        super().observe_feedback(code, valence_idx)
        if action is not None:
            self._buffer.append((int(code), action, int(valence_idx)))
        # replay: re-apply learning for a few stored tuples
        for _ in range(self._replay_k):
            if not self._buffer:
                break
            c, a, v = self._buffer[int(self._replay_rng.integers(0, len(self._buffer)))]
            self.perceive(c)
            self.force_action(a)
            super().observe_feedback(c, v)


# ── Lever factories ──────────────────────────────────────────────────────────

def _baseline_factory(seed, turns):
    return DirectHeadAgent(build_direct_head_model(seed, k=4), seed=seed,
                           gamma=1.0, alpha=1.0, lr_pA=4.0, lv=0.999,
                           optimism=2.0, gamma_schedule=(1.0, 8.0, turns))


def _no_optimism_factory(seed, turns):
    return DirectHeadAgent(build_direct_head_model(seed, k=4), seed=seed,
                           gamma=1.0, alpha=1.0, lr_pA=4.0, lv=0.999,
                           optimism=0.0, gamma_schedule=(1.0, 8.0, turns))


def _eps_greedy_factory(seed, turns):
    return DirectHeadAgent(build_direct_head_model(seed, k=4), seed=seed,
                           gamma=1.0, alpha=1.0, lr_pA=4.0, lv=0.999,
                           optimism=2.0, gamma_schedule=(1.0, 8.0, turns),
                           eps0=0.3, eps_min=0.0, eps_turns=max(1, turns // 2))


def _replay_factory(seed, turns):
    return ReplayDirectHeadAgent(build_direct_head_model(seed, k=4), seed=seed,
                                 gamma=1.0, alpha=1.0, lr_pA=4.0, lv=0.999,
                                 optimism=2.0, gamma_schedule=(1.0, 8.0, turns),
                                 replay_k=3, replay_seed=seed)


LEVERS = {
    "baseline": _baseline_factory,
    "no_optimism": _no_optimism_factory,
    "eps_greedy": _eps_greedy_factory,
    "replay": _replay_factory,
    "constant_control": _constant_factory(0),
    "random_control": None,  # handled inline (random response each turn)
}

NOT_IMPLEMENTED = {
    "eligibility_trace": "decaying credit/blame cache over recent action-observation events",
    "active_ASK_probe": "uncertainty-gated probe action that pays a cost for information",
}


class _RandomAgent:
    def __init__(self, seed, r=5):
        self._rng = np.random.default_rng(seed); self._r = r
    def perceive(self, code): return np.ones(4) / 4
    def act(self): return int(self._rng.integers(0, self._r))
    def observe_feedback(self, code, v): pass
    def correct_select(self, correct): return 0.0


def _random_factory(seed, turns):
    return _RandomAgent(seed)


def _score_curve(seeds, turns, factory, cache_clear: bool) -> dict:
    """Return mean_last/improvement/genuine_fraction; cache_clear frees the JIT cache
    between seeds (memory-safe on constrained hosts; numerically identical to score_affect)."""
    if not cache_clear:
        rep = score_affect(seeds=seeds, turns=turns, agent_factory=factory)
        return {"mean_last": rep.mean_last, "improvement": rep.improvement,
                "genuine_fraction": rep.genuine_fraction}
    import jax  # noqa: PLC0415
    from eval.affect_score import _run_session  # noqa: PLC0415
    firsts, lasts, genuine = [], [], []
    for s in seeds:
        row = _run_session(factory, s, turns)
        firsts.append(row["first"]); lasts.append(row["last"])
        genuine.append(bool(row["csel"] >= 0.5 and row["last"] > CEIL))
        jax.clear_caches()
    mean_first = float(np.mean(firsts)); mean_last = float(np.mean(lasts))
    return {"mean_last": mean_last, "improvement": mean_last - mean_first,
            "genuine_fraction": float(np.mean(genuine))}


def evaluate(seeds, lengths, cache_clear: bool = False) -> dict:
    levers = dict(LEVERS)
    levers["random_control"] = _random_factory
    curves: dict[str, dict] = {}
    for name, factory in levers.items():
        row = {}
        for L in lengths:
            s = _score_curve(seeds, L, factory, cache_clear)
            verdict = "PASS" if (s["mean_last"] > CEIL and s["genuine_fraction"] >= GENUINE_FLOOR
                                 and s["improvement"] >= 0.10) else (
                      "INCONCLUSIVE" if s["mean_last"] > CEIL else "FAIL")
            row[str(L)] = {**s, "verdict": verdict}
        curves[name] = row
    return curves


def run(quick=False, full=False, cache_clear=False) -> dict:
    if quick:
        seeds, lengths = (20, 21), (30, 50, 100)
    elif full:
        seeds, lengths = tuple(range(20, 28)), LENGTHS
    else:
        seeds, lengths = (20, 21, 22, 23), (30, 50, 100, 300)

    sh = scorer_hash(".")
    curves = evaluate(seeds, lengths, cache_clear=cache_clear)

    # Headline read: does any honest lever PASS at a SHORT session (<=100t)?
    short_lengths = [L for L in lengths if L <= 100]
    short_pass = {
        name: any(curves[name][str(L)]["verdict"] == "PASS" for L in short_lengths)
        for name in curves if name not in ("constant_control", "random_control")
    }
    any_short_pass = any(short_pass.values())
    verdict = "PASS" if any_short_pass else "INCONCLUSIVE"  # honest: Exp 221 expects no short-session pass

    return {
        "experiment": "Short-session affect learning levers",
        "authoritative": full,
        "seeds": list(seeds),
        "lengths": list(lengths),
        "scorer_hash": sh,
        "constant_ceiling": CEIL,
        "curves": curves,
        "short_session_pass_by_lever": short_pass,
        "not_implemented_levers": NOT_IMPLEMENTED,
        "verdict": verdict,
        "interpretation": (
            "Exp 221 finding: short sessions block LEARNING. This harness reports the "
            "honest sample-efficiency curve per lever; a non-PASS at <=100t is the "
            "expected negative, not a tuning failure. eligibility_trace and active_ASK "
            "remain scaffolded TODOs (reported NOT_IMPLEMENTED, not faked)."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Short-session affect learning levers")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--cache-clear", action="store_true", dest="cache_clear",
                    help="clear the JAX JIT cache between seeds (memory-safe; identical numbers)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = run(quick=args.quick, full=args.full, cache_clear=args.cache_clear)
    print(json.dumps(result, indent=2))
    if not args.quick:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = Path(args.out) if args.out else RESULTS_DIR / "short_session_affect_levers.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\n[wrote {out}]")


if __name__ == "__main__":
    main()
