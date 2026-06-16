"""Exp 218 — M4a ratchet: minimal honest scaffold for reliable ignition under optimism aid.

AUTHORIZATION: the human's 'continue' at the Exp 217 halt -> run increment 1g
(ratchet the scaffold toward realism)

PLAIN: Exp 217 showed the optimism aid (uniform +2.0 on the POS prior) gives RELIABLE
ignition (7/8 seeds) but ONLY on the generous scaffold (K=U=6, gamma=8, 300 turns).
This experiment holds optimism ON at the Exp 217-winning value (OPTIMISM=2.0, LR=4.0)
and RATCHETS the scaffold toward the realistic regime one knob at a time.  The goal is
to find the MINIMAL honest scaffold at which reliable ignition (>=6/8 seeds) still
survives, and to report exactly where (and on which knob) it breaks.

PREDECLARATION (made before running any cell; falsifier named in advance):

  Seven cells, each varying ONE scaffold knob from the Exp 217 anchor:
    anchor    (gamma=8, K=6, 300t) — consistency control; expect ~7/8 (Exp 217 optimistic result)
    gamma4    (gamma=4, K=6, 300t) — halve policy precision
    gamma1    (gamma=1, K=6, 300t) — realistic policy precision (Exp 215/216 default)
    K5        (gamma=8, K=5, 300t) — reduce intent capacity by 1
    K4        (gamma=8, K=4, 300t) — realistic intent capacity (Exp 215/216 default)
    turns100  (gamma=8, K=6, 100t) — cut turns to 1/3
    realistic (gamma=1, K=4, 100t) — full realistic corner (Exp 215/216 failed 0/8 here)

  CELLS are run in that exact order.  Honesty: optimism is held IDENTICAL across all
  cells (uniform +2.0 on the POS prior, no leakage); the knobs (gamma, K, turns) change
  difficulty, never the answer.

  Metric per seed: ignite(seed) = (improv >= 0.15 AND last_third_POS_rate >= 0.30)
    where improv = last_third_POS_rate - first_third_POS_rate.

  HYPOTHESIS: honest optimism extends reliable ignition (>=6/8) at least one step
  beyond the anchor toward the realistic regime.

  DECISION RULE (predeclared; reliability bar = ignitions >= 6/8):
    REALISTIC_RELIABLE     iff reliable["realistic"]
      -> optimism rescues the full realistic regime; strong/breakthrough candidate.
    PARTIAL_RATCHET        iff not reliable["realistic"] but any non-anchor cell is reliable
      -> report the MINIMAL honest scaffold (most-realistic reliable non-anchor cell)
         and the blocking knob(s).
    NO_RATCHET / FALSIFIED iff NO non-anchor cell reaches >=6/8
      -> hypothesis REFUTED; reliability is fragile to ANY ratchet step; log NEGATIVE.

  FALSIFIER: if no cell beyond the anchor reaches >=6/8 ignitions, the increment-1g
  hypothesis ("honest optimism extends reliable ignition toward realism") is falsified
  (NO_RATCHET).  This is a NEGATIVE result; do not reframe.

Honesty note: optimism is a uniform +2.0 elevation of pA1[:,POS,:,:] — it is IDENTICAL
for every (intent, response) pair, so no particular response is favoured.  The scaffold
knobs (gamma, K, turns) change how difficult the problem is, not what the answer is.
Functional valence only; no sentience claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from active_loop.affect_spec import build_direct_head_model, U, LV, NEU, POS, NEG, ASK
from active_loop.affect_agent import DirectHeadAgent

# ── correct-response mapping (same as Exp 217) ──────────────────────────────
CORRECT = {c: c % 4 for c in range(U)}

# ── 8 seeds — same as Exp 217 ────────────────────────────────────────────────
SEEDS = list(range(20, 28))

# ── held constants: the Exp 217 optimism winner ──────────────────────────────
OPTIMISM = 2.0
LR = 4.0

# ── scaffold cells (ordered dict; ladder from generous -> realistic) ──────────
CELLS: dict[str, dict[str, Any]] = {
    "anchor":    dict(gamma=8.0, K=6, turns=300),   # consistency control; expect ~7/8
    "gamma4":    dict(gamma=4.0, K=6, turns=300),
    "gamma1":    dict(gamma=1.0, K=6, turns=300),
    "K5":        dict(gamma=8.0, K=5, turns=300),
    "K4":        dict(gamma=8.0, K=4, turns=300),
    "turns100":  dict(gamma=8.0, K=6, turns=100),
    "realistic": dict(gamma=1.0, K=4, turns=100),   # full realistic corner (Exp 215/216: 0/8)
}


def _shuffled_codes(rng):
    """Infinite iterator of codes from exhaustive shuffled blocks (same as Exp 217)."""
    pool: list[int] = []

    def nxt():
        nonlocal pool
        if not pool:
            b = list(range(U))
            rng.shuffle(b)
            pool += b
        return pool.pop(0)

    return nxt


def closed_loop(cell_name: str, seed: int) -> tuple[float, float]:
    """Full DirectHeadAgent bootstrap for one cell/seed.

    Returns (first_third_POS_rate, last_third_POS_rate).
    Structure IDENTICAL to exp217.closed_loop: perceive -> act -> valence ->
    observe_feedback; valence = POS iff r == CORRECT[code], NEU iff r == ASK, NEG otherwise.
    """
    np.random.seed(seed)
    cfg = CELLS[cell_name]
    ag = DirectHeadAgent(
        build_direct_head_model(seed, k=cfg["K"]),
        seed=seed,
        gamma=cfg["gamma"],
        alpha=cfg["gamma"],
        lr_pA=LR,
        lv=LV,
        optimism=OPTIMISM,
    )
    nxt = _shuffled_codes(np.random.default_rng(seed))
    turns = cfg["turns"]
    third = turns // 3
    pf = pl = 0
    for t in range(turns):
        code = nxt()
        ag.perceive(code)
        r = ag.act()
        v = POS if r == CORRECT[code] else (NEU if r == ASK else NEG)
        if t < third:
            pf += (v == POS)
        elif t >= turns - third:
            pl += (v == POS)
        ag.observe_feedback(code, v)
    return pf / third, pl / third


def run_cell(name: str, seeds: list[int] = SEEDS) -> list[dict]:
    """Run one cell over all seeds; return list of per-seed result dicts."""
    rows = []
    for seed in seeds:
        first, last = closed_loop(name, seed)
        improv = last - first
        ignite = (improv >= 0.15 and last >= 0.30)
        rows.append(dict(seed=seed, first=round(first, 4), last=round(last, 4),
                         improv=round(improv, 4), ignite=ignite))
    return rows


def aggregate(results: dict[str, list[dict]]) -> str:
    """Apply the predeclared decision rule; return the full human-readable report."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("EXP 218 — M4a ratchet: minimal honest scaffold for reliable ignition")
    lines.append("OPTIMISM=2.0 LR=4.0 held constant; scaffold knobs ratcheted toward realism")
    lines.append("=" * 78)
    lines.append("")

    cell_ignitions: dict[str, int] = {}
    cell_mean_last: dict[str, float] = {}

    for name in CELLS:
        if name not in results:
            continue
        rows = results[name]
        cfg = CELLS[name]
        lines.append(f"--- CELL: {name}  (gamma={cfg['gamma']}, K={cfg['K']}, turns={cfg['turns']}) ---")
        n_ignite = 0
        mean_last = 0.0
        for row in rows:
            flag = "  IGNITE" if row["ignite"] else ""
            lines.append(f"  seed {row['seed']}: first {row['first']:.2f} -> last {row['last']:.2f}"
                         f"  improv {row['improv']:+.3f}{flag}")
            if row["ignite"]:
                n_ignite += 1
            mean_last += row["last"]
        mean_last /= len(rows)
        cell_ignitions[name] = n_ignite
        cell_mean_last[name] = mean_last
        reliable_tag = "RELIABLE" if n_ignite >= 6 else "not"
        lines.append(f"  ignitions: {n_ignite}/{len(rows)}   mean_last: {mean_last:.3f}   {reliable_tag} (>=6/8)")
        lines.append("")

    # ── anchor consistency check ─────────────────────────────────────────────
    lines.append("--- ANCHOR CONSISTENCY (expect ~7/8, accept 6-8 as consistent with Exp 217) ---")
    anc_ig = cell_ignitions.get("anchor", 0)
    if anc_ig >= 6:
        lines.append(f"  anchor={anc_ig}/8 -> PASS (consistent with Exp 217 optimistic 7/8)")
    else:
        lines.append(f"  anchor={anc_ig}/8 -> FLAG")
        lines.append("  *** HARNESS DRIFT? anchor did not reproduce Exp 217 optimistic result ***")
        lines.append("  *** Do NOT interpret downstream cells without resolving this.         ***")
    lines.append("")

    # ── break-point analysis per knob ────────────────────────────────────────
    reliable: dict[str, bool] = {n: (cell_ignitions.get(n, 0) >= 6) for n in CELLS}

    lines.append("--- BREAK-POINT ANALYSIS per knob ---")
    # Gamma ladder: anchor (8) -> gamma4 (4) -> gamma1 (1); K=6, turns=300
    lines.append("  gamma knob (K=6, 300t): anchor(8) -> gamma4(4) -> gamma1(1)")
    for name in ["anchor", "gamma4", "gamma1"]:
        ig = cell_ignitions.get(name, None)
        tag = "RELIABLE" if reliable.get(name, False) else "breaks"
        if ig is not None:
            lines.append(f"    {name}: {ig}/8 -> {tag}")
    gamma_break = next((n for n in ["gamma4", "gamma1"] if not reliable.get(n, True)), None)
    lines.append(f"  gamma break-point: {gamma_break if gamma_break else 'holds to the end (gamma1 reliable)'}")
    lines.append("")

    # K ladder: anchor (6) -> K5 (5) -> K4 (4); gamma=8, turns=300
    lines.append("  K knob (gamma=8, 300t): anchor(6) -> K5(5) -> K4(4)")
    for name in ["anchor", "K5", "K4"]:
        ig = cell_ignitions.get(name, None)
        tag = "RELIABLE" if reliable.get(name, False) else "breaks"
        if ig is not None:
            lines.append(f"    {name}: {ig}/8 -> {tag}")
    k_break = next((n for n in ["K5", "K4"] if not reliable.get(n, True)), None)
    lines.append(f"  K break-point: {k_break if k_break else 'holds to the end (K4 reliable)'}")
    lines.append("")

    # Turns ladder: anchor (300) -> turns100 (100); gamma=8, K=6
    lines.append("  turns knob (gamma=8, K=6): anchor(300) -> turns100(100)")
    for name in ["anchor", "turns100"]:
        ig = cell_ignitions.get(name, None)
        tag = "RELIABLE" if reliable.get(name, False) else "breaks"
        if ig is not None:
            lines.append(f"    {name}: {ig}/8 -> {tag}")
    turns_break = "turns100" if not reliable.get("turns100", True) else None
    lines.append(f"  turns break-point: {turns_break if turns_break else 'holds to 100 (reliable)'}")
    lines.append("")

    # ── headline verdict ──────────────────────────────────────────────────────
    lines.append("--- VERDICT (predeclared decision rule) ---")
    lines.append("")

    non_anchor_reliable = [n for n in CELLS if n != "anchor" and reliable.get(n, False)]

    if reliable.get("realistic", False):
        verdict = "REALISTIC_RELIABLE"
        real_ig = cell_ignitions.get("realistic", 0)
        real_ml = cell_mean_last.get("realistic", 0.0)
        detail = (
            f"Optimism (OPTIMISM=2.0) rescues the FULL realistic regime (gamma=1, K=4, 100t): "
            f"{real_ig}/8 ignitions, mean_last={real_ml:.3f}.  "
            f"This is a strong/breakthrough candidate: reliable ignition now achievable "
            f"at the Exp 215/216 parameter set.  "
            f"Next: replicate / tighten analysis or move to next increment."
        )
    elif non_anchor_reliable:
        verdict = "PARTIAL_RATCHET"
        # Pick the 'most realistic' reliable non-anchor cell.
        # Realism ordering (least to most realistic among non-anchor): turns100, K5, K4, gamma4, gamma1, realistic
        realism_order = ["turns100", "K5", "K4", "gamma4", "gamma1", "realistic"]
        most_realistic = max(non_anchor_reliable, key=lambda n: realism_order.index(n))
        # Identify blocking knobs: non-anchor cells that are NOT reliable AND represent
        # the realistic values of each knob (gamma1, K4, turns100).
        blocking = []
        if not reliable.get("gamma1", False) and reliable.get("gamma4", False):
            blocking.append("gamma (breaks between 4 and 1)")
        elif not reliable.get("gamma4", False):
            blocking.append("gamma (breaks at 4 already)")
        if not reliable.get("K4", False) and reliable.get("K5", False):
            blocking.append("K (breaks between 5 and 4)")
        elif not reliable.get("K5", False):
            blocking.append("K (breaks at 5 already)")
        if not reliable.get("turns100", False):
            blocking.append("turns (breaks at 100)")
        blocking_str = "; ".join(blocking) if blocking else "none identified"
        detail = (
            f"Optimism extends reliability beyond the anchor on cells: {non_anchor_reliable}.  "
            f"Minimal honest scaffold (most realistic reliable non-anchor cell): {most_realistic} "
            f"(gamma={CELLS[most_realistic]['gamma']}, K={CELLS[most_realistic]['K']}, "
            f"turns={CELLS[most_realistic]['turns']}).  "
            f"Blocking knob(s): {blocking_str}.  "
            f"realistic cell is NOT reliable ({cell_ignitions.get('realistic', 0)}/8)."
        )
    else:
        verdict = "NO_RATCHET / FALSIFIED"
        detail = (
            f"NEGATIVE RESULT: hypothesis REFUTED.  "
            f"No non-anchor cell reaches >=6/8 ignitions.  "
            f"Reliability is fragile to ANY ratchet step away from the generous anchor scaffold.  "
            f"The optimism aid (OPTIMISM=2.0) does not generalise beyond (gamma=8, K=6, 300t).  "
            f"Do NOT reframe as partial success.  Log as NEGATIVE and reconsider direction."
        )

    lines.append(f"VERDICT: {verdict}")
    lines.append(f"  {detail}")
    lines.append("")

    # machine summary
    anc_ig  = cell_ignitions.get("anchor",   0)
    g1_ig   = cell_ignitions.get("gamma1",   0)
    k4_ig   = cell_ignitions.get("K4",       0)
    t100_ig = cell_ignitions.get("turns100", 0)
    real_ig = cell_ignitions.get("realistic",0)
    lines.append(
        f"MACHINE SUMMARY: VERDICT={verdict} "
        f"anchor={anc_ig}/8 gamma1={g1_ig}/8 K4={k4_ig}/8 "
        f"turns100={t100_ig}/8 realistic={real_ig}/8"
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 218 — M4a ratchet scaffold cells")
    parser.add_argument("--cell", metavar="NAME", help="Run one cell over SEEDS and write its JSON")
    parser.add_argument("--aggregate", action="store_true",
                        help="Read per-cell JSONs and write the full report")
    args = parser.parse_args()

    out_dir = _REPO / "experiments" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cell:
        name = args.cell
        if name not in CELLS:
            print(f"Unknown cell: {name!r}. Valid cells: {list(CELLS)}", file=sys.stderr)
            sys.exit(1)
        rows = run_cell(name, SEEDS)
        path = out_dir / f"exp218_{name}.json"
        path.write_text(json.dumps(rows, indent=2) + "\n")
        n_ig = sum(1 for r in rows if r["ignite"])
        mean_last = sum(r["last"] for r in rows) / len(rows)
        print(f"[exp218 {name}] ignitions={n_ig}/{len(rows)} mean_last={mean_last:.3f}  saved {path}")

    elif args.aggregate:
        results: dict[str, list[dict]] = {}
        for name in CELLS:
            path = out_dir / f"exp218_{name}.json"
            if not path.exists():
                print(f"Missing {path} — run --cell {name} first", file=sys.stderr)
                sys.exit(1)
            results[name] = json.loads(path.read_text())
        report = aggregate(results)
        txt_path = out_dir / "exp218.txt"
        txt_path.write_text(report + "\n")
        print(report)
        print(f"\n[saved {txt_path}]")

    else:
        # No args: run all cells serially then aggregate
        results: dict[str, list[dict]] = {}
        for name in CELLS:
            print(f"[exp218] running cell={name} ...")
            rows = run_cell(name, SEEDS)
            path = out_dir / f"exp218_{name}.json"
            path.write_text(json.dumps(rows, indent=2) + "\n")
            n_ig = sum(1 for r in rows if r["ignite"])
            mean_last = sum(r["last"] for r in rows) / len(rows)
            print(f"  {name}: ignitions={n_ig}/{len(rows)} mean_last={mean_last:.3f}  saved {path}")
            results[name] = rows
        report = aggregate(results)
        txt_path = out_dir / "exp218.txt"
        txt_path.write_text(report + "\n")
        print(report)
        print(f"\n[saved {txt_path}]")


if __name__ == "__main__":
    main()
