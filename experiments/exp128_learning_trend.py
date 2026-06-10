"""Exp 128 — the halt-mandated learning-trend diagnostic: does the M4a core learn at ANY
horizon?

Exp 127's second F3 halt left two suspects: (a) scale — the joint A+B bootstrap needs
more than 100 turns; (b) timing — credit lands post-transition. This is the predeclared
PURE DIAGNOSTIC that separates them: the as-built increment-1b agent (no changes of any
kind), two 1000-turn scripted-partner sessions, POS rate per 100-turn bin. Procedural
note (recorded): pure diagnostics of a halted system are falsifier-mandated work by this
program's own precedent (Exp 82, Exp 112) and do not resume the build; any design change
or build resumption still awaits the human's explicit word.

Predeclared readout:
  D1 (scale was the issue): POS rate over turns 701-1000 exceeds turns 1-300 by >= 0.10
     in >= 1/2 sessions -> the bootstrap converges, just slowly; recommended resumption
     (human word): re-pose P3 at the measured horizon.
  D2 (scale is NOT the issue): |difference| < 0.05 in BOTH sessions -> the timing/
     architecture suspect stands; recommended next (human word): the valence-timing
     redesign.
  Between the bands: reported as inconclusive, honestly.
Also reported: ASK rate per bin (epistemic action should decay as uncertainty resolves —
its failure to decay is itself diagnostic), and the final-bin POS rate vs the 0.25
chance ceiling for one-correct-of-four non-ask responses.
Seeds 16-17. No creature state touched; the M4a thread remains HALTED regardless of
outcome.
"""
from __future__ import annotations

import math

import numpy as np

from active_loop.affect_spec import build_dyad_model, U, R, LV
from active_loop.affect_agent import AffectAgent

# ── Constants ─────────────────────────────────────────────────────────────────
TURNS = 1000
SEEDS = [16, 17]
BIN_SIZE = 100
N_BINS = TURNS // BIN_SIZE  # 10

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

    partner = ScriptedPartner(seed)

    # Per-turn flags
    pos_flags: list[int] = []
    ask_flags: list[int] = []

    valence_idx = NEU  # first turn: neutral prior

    for t in range(TURNS):
        code = partner.next()
        ag.perceive(code, valence_idx)

        response = ag.act()

        ask_flags.append(1 if response == ASK_IDX else 0)

        valence_idx = partner.feedback(response, code)

        pos_flags.append(1 if valence_idx == POS else 0)

        ag.observe_feedback(code, valence_idx)

    # Bin the per-turn flags into N_BINS bins of BIN_SIZE each
    pos_bins: list[float] = []
    ask_bins: list[float] = []
    for b in range(N_BINS):
        start = b * BIN_SIZE
        end = start + BIN_SIZE
        pos_bins.append(sum(pos_flags[start:end]) / BIN_SIZE)
        ask_bins.append(sum(ask_flags[start:end]) / BIN_SIZE)

    # first300 = bins 0,1,2 (turns 1-300); last300 = bins 7,8,9 (turns 701-1000)
    first300 = sum(pos_bins[0:3]) / 3
    last300 = sum(pos_bins[7:10]) / 3
    diff = last300 - first300

    return dict(
        seed=seed,
        pos_bins=pos_bins,
        ask_bins=ask_bins,
        first300=first300,
        last300=last300,
        diff=diff,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    session_results = []

    for seed in SEEDS:
        print(f"  running seed {seed} (1000 turns) ...", flush=True)
        r = run_session(seed)
        session_results.append(r)

        # Print bin rows for this session
        bin_labels = "  ".join(
            f"{b * BIN_SIZE + 1:>4}-{(b + 1) * BIN_SIZE:<4}" for b in range(N_BINS)
        )
        pos_row = "  ".join(f"{v:>8.3f}" for v in r["pos_bins"])
        ask_row = "  ".join(f"{v:>8.3f}" for v in r["ask_bins"])

        print(f"\n  seed {seed} — POS rate per 100-turn bin:")
        print(f"    bins : {bin_labels}")
        print(f"    POS  : {pos_row}")
        print(f"    ASK  : {ask_row}")
        print(
            f"    first300={r['first300']:.3f}  last300={r['last300']:.3f}  "
            f"diff={r['diff']:+.3f}"
        )
        final_bin_pos = r["pos_bins"][-1]
        print(
            f"    final-bin POS rate: {final_bin_pos:.3f}  "
            f"(chance ceiling for non-ask = 0.25)"
        )

    # ── Verdict ───────────────────────────────────────────────────────────────
    diffs = [r["diff"] for r in session_results]
    d1 = diffs[0]
    d2 = diffs[1]

    d1_hit = sum(1 for d in diffs if d >= 0.10)   # sessions meeting D1 criterion
    d2_hit = all(abs(d) < 0.05 for d in diffs)    # both sessions meet D2 criterion

    print()
    if d1_hit >= 1:
        print(
            f"EXP128: D1 — SCALE (trend found; diff={d1:.2f}/{d2:.2f})"
        )
    elif d2_hit:
        print(
            "EXP128: D2 — NOT SCALE (no trend; timing suspect stands)"
        )
    else:
        print(
            f"EXP128: INCONCLUSIVE (diffs {d1:+.3f}/{d2:+.3f})"
        )


if __name__ == "__main__":
    main()
