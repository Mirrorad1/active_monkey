"""Exp 221 — M4a increment 1j: SEPARATE the precision schedule from session length. Does
annealing gamma RESCUE the short session, or is the long 300t session load-bearing?

This docstring explicitly contains the required words: falsifier, predeclaration, hypothesis.

AUTHORIZATION: the human's /lab affective-dyad + "go" -> Exp 220's predeclared Next
(increment 1j): run the precision schedule at the SHORT session at K=4, N>=16, and report
genuine at the stricter csel>=0.67 bar.

PLAIN summary: Exp 220 found that gradually annealing decisiveness (gamma 1->8) over the
conversation made the talk-to-it agent reliably tell signals apart (13/16) at the realistic
capacity K=4 -- BUT every run used the LONG 300-turn conversation. Exp 219 had shown SHORT
conversations block LEARNING. So the schedule's win was never separated from the long session.
This experiment runs the SAME annealing schedule at SHORT conversations (100/150/200 turns)
and at 300 turns, each against a fixed-decisiveness control of the same length, to ask: does
the schedule rescue the short conversation, or was the long session doing the work?

HYPOTHESIS: annealing gamma 1->8 across the session makes genuine discrimination RELIABLE at
K=4 even at the SHORT session -- i.e. the SCHEDULE, not the 300t length, does the work.

PREDECLARATION: 8 cells = 2 conditions x 4 session lengths, N=16 seeds each, K=4. Condition
'sched' anneals gamma 1->8 across the FULL session (gamma_schedule=(1.0,8.0,L)); condition
'fixed' holds gamma=4.0 (the Exp 220 best fixed control). Optimism=2.0 and lr=4.0 held constant
(same as Exp 219/220). The schedule changes DECISIVENESS over time, not the answer;
correct_select is constant-unfakeable. Seeds 20-35 (same as Exp 220), so the 300t cells are a
REGRESSION ANCHOR: sched_300 must reproduce Exp 220 sched_full (13/16) and fixed_300 must
reproduce Exp 220 fixed_g4 (7/16) byte-for-byte.

CELLS (gamma_schedule scales with the cell's own turns):
  sched_100 / sched_150 / sched_200 / sched_300: gamma=1.0, gamma_schedule=(1.0,8.0,L), turns=L
  fixed_100 / fixed_150 / fixed_200 / fixed_300: gamma=4.0, gamma_schedule=None,      turns=L

METRIC (predeclared, identical to Exp 219/220):
  genuine(seed)        = (correct_select >= 0.5)  AND (last_third_POS_rate > CEIL=0.333).
  genuine_strict(seed) = (correct_select >= 0.67) AND (last_third_POS_rate > CEIL=0.333).
  Report per cell: genuine count, genuine_strict count, mean_csel, mean_last, csel histogram.

DECISION RULE (predeclared):
  short_sched = {sched_100, sched_150, sched_200}.  reliability bar = genuine >= 12/16.
  For each length L, gap_L = genuine(sched_L) - genuine(fixed_L).
  RESCUE_SUCCESS iff max(genuine over short_sched) >= 12 AND that winning short cell beats its
    matched fixed control (same L) by >= 3  -> annealing rescues the short session; 300t NOT
    necessary.
  LENGTH_NECESSARY (FALSIFIER FIRES) iff NO short_sched cell reaches >= 12 genuine -> the long
    300t session is load-bearing; the schedule does not substitute for session length.
    Log NEGATIVE, do not reframe.
  SCHEDULE_NOT_DECISIVE iff a short_sched cell >= 12 BUT does not beat its matched fixed by >= 3
    -> reliability reached at short length but not attributable to the schedule (length/optimism
    does the work) -> MIXED.

FALSIFIER: if no short sched cell (sched_100/150/200) reaches >= 12/16 genuine, the
  "annealing gamma rescues the short session" hypothesis is REFUTED (the 300t session is
  load-bearing). Log NEGATIVE.

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

CELLS: OrderedDict[str, dict] = OrderedDict([
    ("sched_100", dict(gamma=1.0, gamma_schedule=(1.0, 8.0, 100), turns=100)),
    ("sched_150", dict(gamma=1.0, gamma_schedule=(1.0, 8.0, 150), turns=150)),
    ("sched_200", dict(gamma=1.0, gamma_schedule=(1.0, 8.0, 200), turns=200)),
    ("sched_300", dict(gamma=1.0, gamma_schedule=(1.0, 8.0, 300), turns=300)),
    ("fixed_100", dict(gamma=4.0, gamma_schedule=None, turns=100)),
    ("fixed_150", dict(gamma=4.0, gamma_schedule=None, turns=150)),
    ("fixed_200", dict(gamma=4.0, gamma_schedule=None, turns=200)),
    ("fixed_300", dict(gamma=4.0, gamma_schedule=None, turns=300)),
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
    TURNS = cfg["turns"]
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
        genuine_strict = (csel >= 0.67) and (last > CEIL)
        rows.append(dict(seed=seed, first=first, last=last, csel=csel,
                         improv=improv, genuine=genuine, genuine_strict=genuine_strict))
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
    lines.append("EXP 221 — M4a precision schedule: SEPARATE schedule from session length")
    lines.append("OPTIMISM=2.0 LR=4.0 held constant; schedule changes DECISIVENESS not the answer")
    lines.append(f"CEIL={CEIL:.3f}; genuine = (csel>=0.5) AND (last>{CEIL:.3f})")
    lines.append(f"         genuine_strict = (csel>=0.67) AND (last>{CEIL:.3f})")
    lines.append("=" * 78)
    lines.append("FALSIFIER: if no short sched cell (sched_100/150/200) reaches >= 12/16 genuine,")
    lines.append("  the 'annealing gamma rescues the short session' hypothesis is REFUTED.")
    lines.append("")

    genuine_counts: dict[str, int] = {}
    genuine_strict_counts: dict[str, int] = {}
    mean_csel: dict[str, float] = {}
    mean_last: dict[str, float] = {}

    for name in CELLS:
        if name not in results:
            continue
        rows = results[name]
        cfg = CELLS[name]
        sched_str = f"schedule={cfg['gamma_schedule']}" if cfg["gamma_schedule"] else "fixed"
        lines.append(f"--- CELL: {name}  (gamma={cfg['gamma']}, {sched_str}, turns={cfg['turns']}) ---")
        n_genuine = 0
        n_genuine_strict = 0
        csel_sum = 0.0
        last_sum = 0.0
        for row in rows:
            flag = "  GENUINE" if row["genuine"] else ""
            strict_flag = " STRICT" if row.get("genuine_strict", False) else ""
            lines.append(f"  seed {row['seed']}: first {row['first']:.4f} -> last {row['last']:.4f}"
                         f"  csel {row['csel']:.4f}  genuine={row['genuine']}{flag}{strict_flag}")
            if row["genuine"]:
                n_genuine += 1
            if row.get("genuine_strict", False):
                n_genuine_strict += 1
            csel_sum += row["csel"]
            last_sum += row["last"]
        n = len(rows)
        mean_c = csel_sum / n
        mean_l = last_sum / n
        genuine_counts[name] = n_genuine
        genuine_strict_counts[name] = n_genuine_strict
        mean_csel[name] = mean_c
        mean_last[name] = mean_l
        hist = _csel_histogram(rows)
        lines.append(f"  genuine {n_genuine}/{n}, genuine_strict {n_genuine_strict}/{n}, "
                     f"mean_csel {mean_c:.3f}, mean_last {mean_l:.3f}")
        lines.append(f"  csel distribution: {hist}")
        lines.append("")

    # --- Gap table and headline verdict ---
    lengths = [100, 150, 200, 300]
    short_sched_cells = ["sched_100", "sched_150", "sched_200"]

    # Check all required cells are present
    required = [f"sched_{L}" for L in lengths] + [f"fixed_{L}" for L in lengths]
    missing = [c for c in required if c not in genuine_counts]
    if missing:
        lines.append(f"--- INCOMPLETE: missing cell results: {missing}; run all cells before aggregating ---")
        return "\n".join(lines)

    lines.append("--- GAP TABLE: sched_L - fixed_L for each session length ---")
    for L in lengths:
        sc = f"sched_{L}"
        fc = f"fixed_{L}"
        gap = genuine_counts[sc] - genuine_counts[fc]
        lines.append(f"  L={L}: sched={genuine_counts[sc]}/16  fixed={genuine_counts[fc]}/16  gap={gap:+d}")
    lines.append("")

    lines.append("--- MEAN CSEL PER LENGTH (learning-blocker visibility) ---")
    for L in lengths:
        sc = f"sched_{L}"
        fc = f"fixed_{L}"
        lines.append(f"  L={L}: sched mean_csel={mean_csel[sc]:.3f}  fixed mean_csel={mean_csel[fc]:.3f}")
    lines.append("")

    # Regression anchor
    sched_300_g = genuine_counts.get("sched_300", "?")
    fixed_300_g = genuine_counts.get("fixed_300", "?")
    regression_match = (sched_300_g == 13 and fixed_300_g == 7)
    lines.append(f"regression: sched_300={sched_300_g}/16 (exp220 expected 13), "
                 f"fixed_300={fixed_300_g}/16 (exp220 expected 7), match={regression_match}")
    lines.append("")

    # Decision rule
    # best short sched cell by genuine count
    best_short_sched_cell = max(short_sched_cells, key=lambda c: genuine_counts[c])
    best_short_genuine = genuine_counts[best_short_sched_cell]
    best_short_L = int(best_short_sched_cell.split("_")[1])
    matched_fixed_cell = f"fixed_{best_short_L}"
    matched_fixed_genuine = genuine_counts[matched_fixed_cell]
    gap_best = best_short_genuine - matched_fixed_genuine

    any_short_sched_reliable = any(genuine_counts[c] >= 12 for c in short_sched_cells)

    if any_short_sched_reliable and gap_best >= 3:
        verdict = "RESCUE_SUCCESS"
        detail = (
            f"Annealing gamma rescues the short session: best short sched cell "
            f"{best_short_sched_cell}={best_short_genuine}/16 >= 12 AND beats matched "
            f"fixed ({matched_fixed_cell}={matched_fixed_genuine}/16) by gap={gap_best} >= 3. "
            f"The 300t long session is NOT necessary; the schedule does the work."
        )
    elif not any_short_sched_reliable:
        verdict = "LENGTH_NECESSARY_FALSIFIED"
        short_summary = ", ".join(f"{c}={genuine_counts[c]}/16" for c in short_sched_cells)
        detail = (
            f"FALSIFIER FIRES: no short sched cell reaches >= 12/16 genuine. "
            f"Short sched results: {short_summary}. "
            f"The 300t session is load-bearing; the schedule does NOT rescue the short session. "
            f"The 'annealing gamma rescues the short session' hypothesis is REFUTED. Log NEGATIVE, do not reframe."
        )
    else:
        # any_short_sched_reliable but gap < 3
        verdict = "SCHEDULE_NOT_DECISIVE"
        detail = (
            f"Short session reliability reached ({best_short_sched_cell}={best_short_genuine}/16 >= 12) "
            f"but not attributable to the schedule: gap vs matched fixed "
            f"({matched_fixed_cell}={matched_fixed_genuine}/16) is {gap_best} < 3. "
            f"Length/optimism does the work, not the annealing schedule. MIXED result."
        )

    lines.append(f"VERDICT: {verdict}")
    lines.append(f"  {detail}")
    lines.append("")

    # Machine summary
    sc100 = genuine_counts.get("sched_100", "?")
    sc150 = genuine_counts.get("sched_150", "?")
    sc200 = genuine_counts.get("sched_200", "?")
    sc300 = genuine_counts.get("sched_300", "?")
    fc100 = genuine_counts.get("fixed_100", "?")
    fc150 = genuine_counts.get("fixed_150", "?")
    fc200 = genuine_counts.get("fixed_200", "?")
    fc300 = genuine_counts.get("fixed_300", "?")
    sc100s = genuine_strict_counts.get("sched_100", "?")
    sc150s = genuine_strict_counts.get("sched_150", "?")
    sc200s = genuine_strict_counts.get("sched_200", "?")
    sc300s = genuine_strict_counts.get("sched_300", "?")
    fc100s = genuine_strict_counts.get("fixed_100", "?")
    fc150s = genuine_strict_counts.get("fixed_150", "?")
    fc200s = genuine_strict_counts.get("fixed_200", "?")
    fc300s = genuine_strict_counts.get("fixed_300", "?")
    lines.append(
        f"MACHINE SUMMARY: VERDICT={verdict} ceil={CEIL:.3f} "
        f"sched_100={sc100}/16 sched_150={sc150}/16 sched_200={sc200}/16 sched_300={sc300}/16 "
        f"fixed_100={fc100}/16 fixed_150={fc150}/16 fixed_200={fc200}/16 fixed_300={fc300}/16 "
        f"strict: sched_100={sc100s}/16 sched_150={sc150s}/16 sched_200={sc200s}/16 sched_300={sc300s}/16 "
        f"fixed_100={fc100s}/16 fixed_150={fc150s}/16 fixed_200={fc200s}/16 fixed_300={fc300s}/16"
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 221 — M4a precision schedule vs session length")
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
        path = out_dir / f"exp221_{name}.json"
        if path.exists() and args.seeds:
            existing = json.loads(path.read_text())
            by_seed = {r["seed"]: r for r in existing}
            for r in rows:
                by_seed[r["seed"]] = r
            rows = sorted(by_seed.values(), key=lambda r: r["seed"])

        path.write_text(json.dumps(rows, indent=2) + "\n")
        n_genuine = sum(1 for r in rows if r["genuine"])
        n_genuine_strict = sum(1 for r in rows if r.get("genuine_strict", False))
        mean_c = sum(r["csel"] for r in rows) / len(rows)
        print(f"[exp221 {name}] genuine={n_genuine}/{len(rows)} genuine_strict={n_genuine_strict}/{len(rows)} "
              f"mean_csel={mean_c:.3f}  saved {path}")

    elif args.aggregate:
        results: dict[str, list[dict]] = {}
        for name in CELLS:
            path = out_dir / f"exp221_{name}.json"
            if not path.exists():
                print(f"Missing {path} — run --cell {name} first", file=sys.stderr)
                sys.exit(1)
            results[name] = json.loads(path.read_text())
        report = aggregate(results)
        txt_path = out_dir / "exp221.txt"
        txt_path.write_text(report + "\n")
        print(report)
        print(f"\n[saved {txt_path}]")

    else:
        # No args: run all cells (all seeds) then aggregate
        results: dict[str, list[dict]] = {}
        for name in CELLS:
            print(f"[exp221] running cell={name} ...")
            rows = run_cell(name, SEEDS)
            path = out_dir / f"exp221_{name}.json"
            path.write_text(json.dumps(rows, indent=2) + "\n")
            n_genuine = sum(1 for r in rows if r["genuine"])
            n_genuine_strict = sum(1 for r in rows if r.get("genuine_strict", False))
            mean_c = sum(r["csel"] for r in rows) / len(rows)
            print(f"  {name}: genuine={n_genuine}/{len(rows)} genuine_strict={n_genuine_strict}/{len(rows)} "
                  f"mean_csel={mean_c:.3f}  saved {path}")
            results[name] = rows
        report = aggregate(results)
        txt_path = out_dir / "exp221.txt"
        txt_path.write_text(report + "\n")
        print(report)
        print(f"\n[saved {txt_path}]")


if __name__ == "__main__":
    main()
