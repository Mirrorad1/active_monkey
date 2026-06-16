"""Exp 220 — M4a precision schedule (anneal gamma) at N=16: does explore-then-exploit fix the
learn-but-don't-exploit decoupling found in Exp 219?

This docstring explicitly contains the required words: falsifier, predeclaration, hypothesis.

AUTHORIZATION: the human's 'continue' at the Exp 219 halt -> increment 1i: a precision
schedule (anneal gamma low->high) at bigger N to fix the learn-but-don't-exploit decoupling.

PLAIN summary: Exp 219 found PARTIAL genuine discrimination at K=4 (some cells >=2/8 seeds)
but no cell reached the >=6/8 reliability bar. A plausible blocker: fixed high-gamma
(high decisiveness) early in the session may suppress exploration before the A1 table is
learned, while fixed low-gamma wastes the learned table. A precision SCHEDULE — annealing
gamma low->high over the session — lets the agent explore early (low decisiveness) then
exploit the learned A1 table (high decisiveness). This is the learn-then-exploit hypothesis.

PREDECLARATION: run four cells (N=16 seeds each, K=4, turns=300). Two fixed-gamma controls
(gamma=4 and gamma=8 from Exp 219) and two scheduled cells (gamma annealing 1->8 over first
half or full session). Optimism held constant at 2.0 (same as Exp 219). The schedule changes
DECISIVENESS over time — it does NOT reveal the answer; correct_select is constant-unfakeable.

CELLS:
  fixed_g4:   gamma fixed at 4.0 throughout (Exp 219 best fixed-K4 approximate baseline).
  fixed_g8:   gamma fixed at 8.0 throughout.
  sched_half: gamma annealed 1.0->8.0 over first 150 of 300 turns, then held at 8.0.
  sched_full: gamma annealed 1.0->8.0 across the full 300-turn session.

DECISION RULE (predeclared; reliability bar genuine >= 12/16):
  genuine(seed) = (correct_select >= 0.5) AND (last_third_POS_rate > CEIL=0.333).
  best_fixed = max(genuine over fixed_g4, fixed_g8).
  best_sched  = max(genuine over sched_half, sched_full).
  SCHEDULE_WINS iff best_sched >= 12 AND (best_sched - best_fixed) >= 3 -> the precision
    schedule delivers RELIABLE genuine discrimination at K=4 and beats fixed precision.
  SCHEDULE_HELPS iff (best_sched - best_fixed) >= 3 but best_sched < 12 -> schedule helps
    but not yet reliable.
  NO_SCHEDULE_BENEFIT / FALSIFIED iff (best_sched - best_fixed) < 3 -> the schedule does
    not beat fixed precision.

FALSIFIER: if no schedule cell beats the best fixed cell by >=3 genuine seeds out of 16,
  the "annealing gamma fixes the learn-but-don't-exploit decoupling" hypothesis is refuted.
  Log NEGATIVE, do not reframe.

Functional valence only; no sentience claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from active_loop.affect_spec import build_direct_head_model, U, R, LV, NEU, POS, NEG, ASK, constant_response_ceiling
from active_loop.affect_agent import DirectHeadAgent

CORRECT = {c: c % 4 for c in range(U)}
CEIL = constant_response_ceiling(CORRECT, R)   # == 1/3
SEEDS = list(range(20, 36))                    # N=16
OPTIMISM = 2.0
LR = 4.0
K = 4
TURNS = 300

CELLS: OrderedDict[str, dict] = OrderedDict([
    ("fixed_g4",   dict(gamma=4.0, gamma_schedule=None)),
    ("fixed_g8",   dict(gamma=8.0, gamma_schedule=None)),
    ("sched_half", dict(gamma=1.0, gamma_schedule=(1.0, 8.0, 150))),
    ("sched_full", dict(gamma=1.0, gamma_schedule=(1.0, 8.0, 300))),
])


def _shuffled_codes(rng):
    """Infinite iterator of codes from exhaustive shuffled blocks (same as Exp 218/219)."""
    pool: list[int] = []

    def nxt():
        nonlocal pool
        if not pool:
            b = list(range(U))
            rng.shuffle(b)
            pool += b
        return pool.pop(0)

    return nxt


def closed_loop(cell_name: str, seed: int) -> tuple[float, float, float]:
    """Full DirectHeadAgent bootstrap for one cell/seed.

    Returns (first_third_POS_rate, last_third_POS_rate, correct_select).
    Structure mirrors exp219.closed_loop: perceive -> act -> valence -> observe_feedback;
    valence = POS iff r == CORRECT[code], NEU iff r == ASK, NEG otherwise.
    """
    np.random.seed(seed)
    cfg = CELLS[cell_name]
    ag = DirectHeadAgent(
        build_direct_head_model(seed, k=K),
        seed=seed,
        gamma=cfg["gamma"],
        alpha=cfg["gamma"],
        lr_pA=LR,
        lv=LV,
        optimism=OPTIMISM,
        gamma_schedule=cfg["gamma_schedule"],
    )
    nxt = _shuffled_codes(np.random.default_rng(seed))
    third = TURNS // 3
    pf = pl = 0
    for t in range(TURNS):
        code = nxt()
        ag.perceive(code)
        r = ag.act()
        valence = POS if r == CORRECT[code] else (NEU if r == ASK else NEG)
        if t < third:
            pf += (valence == POS)
        elif t >= TURNS - third:
            pl += (valence == POS)
        ag.observe_feedback(code, valence)
    cs = ag.correct_select(CORRECT)
    return (round(pf / third, 4), round(pl / third, 4), round(cs, 4))


def run_cell(name: str, seeds=SEEDS) -> list[dict]:
    """Run one cell over the given seeds; return list of per-seed result dicts."""
    rows = []
    for seed in seeds:
        first, last, csel = closed_loop(name, seed)
        improv = round(last - first, 4)
        genuine = (csel >= 0.5) and (last > CEIL)
        rows.append(dict(seed=seed, first=first, last=last, csel=csel,
                         improv=improv, genuine=genuine))
    return rows


def _csel_histogram(rows: list[dict]) -> str:
    """Produce a histogram of correct_select distribution across seeds."""
    buckets = [0.0, 0.17, 0.33, 0.5, 0.67, 0.83, 1.0]
    counts = {b: 0 for b in buckets}
    for row in rows:
        cs = row["csel"]
        # assign to closest bucket
        best = min(buckets, key=lambda b: abs(b - cs))
        counts[best] += 1
    parts = []
    for b in buckets:
        parts.append(f"{b:.2f}:{counts[b]}")
    return " | ".join(parts)


def aggregate(results: dict[str, list[dict]]) -> str:
    """Apply the predeclared decision rule; return the full human-readable report."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("EXP 220 — M4a precision schedule (anneal gamma) at N=16")
    lines.append("OPTIMISM=2.0 LR=4.0 held constant; schedule changes DECISIVENESS not the answer")
    lines.append(f"CEIL={CEIL:.3f}; genuine = (csel>=0.5) AND (last>{CEIL:.3f})")
    lines.append("=" * 78)
    lines.append("FALSIFIER: if no schedule cell beats the best fixed cell by >=3 genuine seeds")
    lines.append("  (out of 16), the 'annealing gamma fixes the decoupling' hypothesis is REFUTED.")
    lines.append("")

    genuine_counts: dict[str, int] = {}
    mean_csel: dict[str, float] = {}
    mean_last: dict[str, float] = {}

    for name in CELLS:
        if name not in results:
            continue
        rows = results[name]
        cfg = CELLS[name]
        sched_str = f"schedule={cfg['gamma_schedule']}" if cfg["gamma_schedule"] else "fixed"
        lines.append(f"--- CELL: {name}  (gamma={cfg['gamma']}, {sched_str}) ---")
        n_genuine = 0
        csel_sum = 0.0
        last_sum = 0.0
        for row in rows:
            flag = "  GENUINE" if row["genuine"] else ""
            lines.append(f"  seed {row['seed']}: first {row['first']:.4f} -> last {row['last']:.4f}"
                         f"  csel {row['csel']:.4f}  genuine={row['genuine']}{flag}")
            if row["genuine"]:
                n_genuine += 1
            csel_sum += row["csel"]
            last_sum += row["last"]
        n = len(rows)
        mean_c = csel_sum / n
        mean_l = last_sum / n
        genuine_counts[name] = n_genuine
        mean_csel[name] = mean_c
        mean_last[name] = mean_l
        hist = _csel_histogram(rows)
        lines.append(f"  genuine {n_genuine}/{n}, mean_csel {mean_c:.3f}, mean_last {mean_l:.3f}")
        lines.append(f"  csel distribution: {hist}")
        lines.append("")

    # --- Headline verdict ---
    fixed_cells = ["fixed_g4", "fixed_g8"]
    sched_cells = ["sched_half", "sched_full"]

    available_fixed = [c for c in fixed_cells if c in genuine_counts]
    available_sched = [c for c in sched_cells if c in genuine_counts]

    if not available_fixed or not available_sched:
        lines.append("--- INCOMPLETE: missing cell results; run all cells before aggregating ---")
        return "\n".join(lines)

    best_fixed_cell = max(available_fixed, key=lambda c: genuine_counts[c])
    best_sched_cell = max(available_sched, key=lambda c: genuine_counts[c])
    best_fixed = genuine_counts[best_fixed_cell]
    best_sched = genuine_counts[best_sched_cell]
    any_cell_reliable = any(genuine_counts.get(c, 0) >= 12 for c in CELLS)

    lines.append("--- HEADLINE: SCHEDULE vs FIXED (predeclared decision rule) ---")
    lines.append(f"  best_fixed: {best_fixed_cell}={best_fixed}/16")
    lines.append(f"  best_sched: {best_sched_cell}={best_sched}/16")
    lines.append(f"  any cell >= 12/16 genuine: {any_cell_reliable}")
    lines.append("")

    gap = best_sched - best_fixed
    if best_sched >= 12 and gap >= 3:
        verdict = "SCHEDULE_WINS"
        detail = (
            f"The precision schedule delivers RELIABLE genuine discrimination at K=4 "
            f"(best_sched={best_sched}/16 >= 12, gap={gap} >= 3). "
            f"The learn-then-exploit annealing strategy fixes the decoupling. "
            f"Best schedule cell: {best_sched_cell}. Best fixed cell: {best_fixed_cell}={best_fixed}/16."
        )
    elif gap >= 3:
        verdict = "SCHEDULE_HELPS"
        detail = (
            f"The precision schedule improves genuine discrimination over fixed gamma "
            f"(gap={gap} >= 3) but does not yet reach the reliability bar of 12/16 "
            f"(best_sched={best_sched}/16). Schedule helps but more work needed. "
            f"Best schedule cell: {best_sched_cell}. Best fixed cell: {best_fixed_cell}={best_fixed}/16."
        )
    else:
        verdict = "NO_SCHEDULE_BENEFIT_FALSIFIED"
        detail = (
            f"FALSIFIER FIRES: the schedule does not beat fixed gamma by >=3 seeds "
            f"(best_sched={best_sched}/16, best_fixed={best_fixed}/16, gap={gap}). "
            f"The 'annealing gamma fixes the learn-but-don't-exploit decoupling' hypothesis "
            f"is REFUTED under the predeclared falsifier. Log NEGATIVE, do not reframe."
        )

    lines.append(f"VERDICT: {verdict}")
    lines.append(f"  {detail}")
    if any_cell_reliable:
        lines.append("  NOTE: At least one cell reached >=12/16 genuine — reliable genuine discrimination at K=4 IS achievable.")
    else:
        lines.append("  NOTE: No cell reached >=12/16 genuine — reliable genuine discrimination at K=4 not yet demonstrated.")

    fg4 = genuine_counts.get("fixed_g4", "?")
    fg8 = genuine_counts.get("fixed_g8", "?")
    sh  = genuine_counts.get("sched_half", "?")
    sf  = genuine_counts.get("sched_full", "?")
    lines.append(
        f"MACHINE SUMMARY: VERDICT={verdict} ceil={CEIL:.3f} "
        f"fixed_g4={fg4}/16 fixed_g8={fg8}/16 sched_half={sh}/16 sched_full={sf}/16 "
        f"best_sched={best_sched_cell}:{best_sched} best_fixed={best_fixed_cell}:{best_fixed}"
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 220 — M4a precision schedule cells")
    parser.add_argument("--cell", metavar="NAME",
                        help="Run one cell and write/update its JSON")
    parser.add_argument("--seeds", metavar="SEEDS",
                        help="Comma-separated subset of seeds (for parallel chunking; "
                             "merges into any existing JSON for that cell)")
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
        seeds = SEEDS
        if args.seeds:
            seeds = [int(s.strip()) for s in args.seeds.split(",")]

        rows = run_cell(name, seeds)

        # Merge into any existing JSON for this cell (so chunked runs accumulate)
        path = out_dir / f"exp220_{name}.json"
        if path.exists() and args.seeds:
            existing = json.loads(path.read_text())
            by_seed = {r["seed"]: r for r in existing}
            for r in rows:
                by_seed[r["seed"]] = r
            rows = sorted(by_seed.values(), key=lambda r: r["seed"])

        path.write_text(json.dumps(rows, indent=2) + "\n")
        n_genuine = sum(1 for r in rows if r["genuine"])
        mean_c = sum(r["csel"] for r in rows) / len(rows)
        print(f"[exp220 {name}] genuine={n_genuine}/{len(rows)} mean_csel={mean_c:.3f}  saved {path}")

    elif args.aggregate:
        results: dict[str, list[dict]] = {}
        for name in CELLS:
            path = out_dir / f"exp220_{name}.json"
            if not path.exists():
                print(f"Missing {path} — run --cell {name} first", file=sys.stderr)
                sys.exit(1)
            results[name] = json.loads(path.read_text())
        report = aggregate(results)
        txt_path = out_dir / "exp220.txt"
        txt_path.write_text(report + "\n")
        print(report)
        print(f"\n[saved {txt_path}]")

    else:
        # No args: run all cells (all seeds) then aggregate
        results: dict[str, list[dict]] = {}
        for name in CELLS:
            print(f"[exp220] running cell={name} ...")
            rows = run_cell(name, SEEDS)
            path = out_dir / f"exp220_{name}.json"
            path.write_text(json.dumps(rows, indent=2) + "\n")
            n_genuine = sum(1 for r in rows if r["genuine"])
            mean_c = sum(r["csel"] for r in rows) / len(rows)
            print(f"  {name}: genuine={n_genuine}/{len(rows)} mean_csel={mean_c:.3f}  saved {path}")
            results[name] = rows
        report = aggregate(results)
        txt_path = out_dir / "exp220.txt"
        txt_path.write_text(report + "\n")
        print(report)
        print(f"\n[saved {txt_path}]")


if __name__ == "__main__":
    main()
