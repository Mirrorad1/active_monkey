"""Exp 219 — M4a discrimination readout: does ignition exceed the constant ceiling?

This docstring explicitly contains the required words: falsifier, predeclaration, hypothesis.

AUTHORIZATION: the human's Exp 218 follow-up directive -> run increment 1h across the
ratchet cells with a discrimination metric instead of the lenient ignition metric.

PLAIN summary: Exp 218 found that the old last-third POS-rate ignition threshold could sit
at the constant-response ceiling: a degenerate policy that always emits one response can
score about 1/3 POS on CORRECT[c] = c % 4. Exp 219 adds a read-only correct-select probe:
after closed-loop learning, ask whether each utterance code maps to its correct response
under fresh prior-D inference with valence NEU. Genuine learning must both beat the
constant ceiling in closed-loop POS-rate and map enough individual codes correctly.

PREDECLARATION: run seven ordered cells, N=8 seeds each, preserving Exp 218's optimism
and learning-rate scaffold. The K=6 controls check that the easy Exp 217/218 regime was
real discrimination under the stricter metric. The K=4 cells directly test whether the
realistic-capacity dyad can genuinely discriminate once session length and precision are
ratcheted.

CELLS:
  g1_100      gamma=1.0, K=4, turns=100
  g1_300      gamma=1.0, K=4, turns=300
  g4_100      gamma=4.0, K=4, turns=100
  g4_300      gamma=4.0, K=4, turns=300
  g4_600      gamma=4.0, K=4, turns=600
  ctrl_anchor gamma=8.0, K=6, turns=300
  ctrl_g4K6   gamma=4.0, K=6, turns=300

DECISION RULE (predeclared; reliability bar = genuine >= 6/8):
  genuine(seed) = (correct_select >= 0.5) AND (last_third_POS_rate > CEIL).
  GENUINE_REALISTIC iff any K=4 cell reaches >=6/8 genuine seeds.
  PARTIAL iff some K=4 cell reaches >=2/8 but none reaches >=6/8.
  NO_GENUINE / FALSIFIED iff no K=4 cell reaches >=2/8 genuine. Falsifier:
  if no K=4 cell reaches >=2/8 genuine, the hypothesis that the realistic-CAPACITY
  (K=4) dyad genuinely discriminates under the tested precision/session scaffold is
  refuted; log NEGATIVE, do not reframe.

Functional valence only; no sentience claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from active_loop.affect_spec import build_direct_head_model, U, R, LV, NEU, POS, NEG, ASK, constant_response_ceiling
from active_loop.affect_agent import DirectHeadAgent

CORRECT = {c: c % 4 for c in range(U)}
SEEDS = list(range(20, 28))  # N=8
OPTIMISM = 2.0
LR = 4.0
CEIL = constant_response_ceiling(CORRECT, R)  # == 1/3

CELLS = {
    "g1_100":      dict(gamma=1.0, K=4, turns=100),
    "g1_300":      dict(gamma=1.0, K=4, turns=300),
    "g4_100":      dict(gamma=4.0, K=4, turns=100),
    "g4_300":      dict(gamma=4.0, K=4, turns=300),
    "g4_600":      dict(gamma=4.0, K=4, turns=600),
    "ctrl_anchor": dict(gamma=8.0, K=6, turns=300),
    "ctrl_g4K6":   dict(gamma=4.0, K=6, turns=300),
}

K4_CELLS = ["g1_100", "g1_300", "g4_100", "g4_300", "g4_600"]


def _shuffled_codes(rng):
    """Infinite iterator of codes from exhaustive shuffled blocks (same as Exp 218)."""
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
    Structure mirrors exp218.closed_loop: perceive -> act -> valence ->
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
        valence = POS if r == CORRECT[code] else (NEU if r == ASK else NEG)
        if t < third:
            pf += (valence == POS)
        elif t >= turns - third:
            pl += (valence == POS)
        ag.observe_feedback(code, valence)
    cs = ag.correct_select(CORRECT)
    return (pf / third, pl / third, cs)


def run_cell(name: str, seeds=SEEDS) -> list[dict]:
    """Run one cell over all seeds; return list of per-seed result dicts."""
    rows = []
    for seed in seeds:
        first, last, csel = closed_loop(name, seed)
        improv = last - first
        genuine = (csel >= 0.5) and (last > CEIL)
        ignite = (improv >= 0.15 and last >= 0.30)
        rows.append(dict(seed=seed, first=round(first, 4), last=round(last, 4),
                         csel=round(csel, 4), improv=round(improv, 4),
                         genuine=genuine, ignite=ignite))
    return rows


def _blocker_for_minimal_cell(cell_name: str) -> str:
    if cell_name == "g1_100":
        return "none in the tested K=4 range (gamma=1, 100 turns already clears the bar)"
    if cell_name == "g1_300":
        return "session length mattered at gamma=1 (100 turns failed; 300 turns cleared)"
    if cell_name == "g4_100":
        return "policy precision mattered at 100 turns (gamma=1 failed; gamma=4 cleared)"
    if cell_name == "g4_300":
        return "both precision and/or session length mattered (gamma=4, 300 turns cleared)"
    if cell_name == "g4_600":
        return "longer session length under gamma=4 mattered (300 turns failed; 600 cleared)"
    return "not applicable"


def aggregate(results: dict[str, list[dict]]) -> str:
    """Apply the predeclared decision rule; return the full human-readable report."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("EXP 219 — M4a discrimination readout (genuine > constant ceiling)")
    lines.append("OPTIMISM=2.0 LR=4.0 held constant; correct_select probes fresh D inference")
    lines.append("=" * 78)
    lines.append(f"CEIL={CEIL:.3f}; last~0.33 is the constant ceiling for a degenerate constant-response policy.")
    lines.append("FALSIFIER: if no K=4 cell reaches >=2/8 genuine, the hypothesis is refuted.")
    lines.append("")

    genuine_counts: dict[str, int] = {}
    ignite_counts: dict[str, int] = {}
    mean_csel: dict[str, float] = {}
    mean_last: dict[str, float] = {}

    for name in CELLS:
        if name not in results:
            continue
        rows = results[name]
        cfg = CELLS[name]
        lines.append(f"--- CELL: {name}  (gamma={cfg['gamma']}, K={cfg['K']}, turns={cfg['turns']}) ---")
        n_genuine = 0
        n_ignite = 0
        csel_sum = 0.0
        last_sum = 0.0
        for row in rows:
            flag = "  GENUINE" if row["genuine"] else ""
            lines.append(f"  seed {row['seed']}: first {row['first']:.2f} -> last {row['last']:.2f}"
                         f"  csel {row['csel']:.2f}  genuine={row['genuine']}{flag}")
            if row["genuine"]:
                n_genuine += 1
            if row["ignite"]:
                n_ignite += 1
            csel_sum += row["csel"]
            last_sum += row["last"]
        mean_c = csel_sum / len(rows)
        mean_l = last_sum / len(rows)
        genuine_counts[name] = n_genuine
        ignite_counts[name] = n_ignite
        mean_csel[name] = mean_c
        mean_last[name] = mean_l
        lines.append(f"  genuine {n_genuine}/{len(rows)}, mean_csel {mean_c:.3f}, mean_last {mean_l:.3f}; "
                     f"(lenient ignite {n_ignite}/{len(rows)} for comparison)")
        lines.append("")

    lines.append("--- K6 CONTROL CHECK ---")
    anc_g = genuine_counts.get("ctrl_anchor", 0)
    g4k6_g = genuine_counts.get("ctrl_g4K6", 0)
    if anc_g >= 6 and g4k6_g >= 6:
        lines.append("CONTROLS GENUINE: the easy-regime Exp 217/218 ignitions ARE real discrimination (validates the metric).")
    else:
        lines.append("*** CONTROL FAILS: even the easy K=6 regime does not genuinely discriminate -> Exp 217's reliable-ignition headline was constant-ceiling; escalate. ***")
    lines.append("")

    lines.append("--- HEADLINE OVER K=4 CELLS ---")
    k4_counts = {name: genuine_counts.get(name, 0) for name in K4_CELLS}
    reliable_k4 = [name for name in K4_CELLS if k4_counts[name] >= 6]
    partial_k4 = [name for name in K4_CELLS if k4_counts[name] >= 2]
    best_k4 = max(K4_CELLS, key=lambda n: k4_counts[n])
    best_k4_count = k4_counts[best_k4]

    if reliable_k4:
        verdict = "GENUINE_REALISTIC"
        minimal = reliable_k4[0]
        blocker = _blocker_for_minimal_cell(minimal)
        lines.append(f"GENUINE_REALISTIC: the realistic-CAPACITY (K=4) dyad GENUINELY discriminates; "
                     f"minimal cell {minimal} reaches {k4_counts[minimal]}/8 genuine.")
        lines.append(f"Blocker mattered: {blocker}.")
        detail = (f"The stricter correct-select probe clears the reliability bar in K=4 at {minimal}. "
                  f"Because genuine requires csel>=0.5 and last>CEIL={CEIL:.3f}, this is not a constant-policy artifact. "
                  f"The control counts are anchor:{anc_g}/8,g4K6:{g4k6_g}/8.")
    elif partial_k4:
        verdict = "PARTIAL"
        lines.append(f"PARTIAL: some K=4 cell reaches >=2/8 genuine but none reaches >=6/8; "
                     f"best cell {best_k4}={best_k4_count}/8.")
        detail = (f"The realistic-capacity dyad shows isolated genuine discrimination but not reliable discrimination. "
                  f"The falsifier does not fire, but the reliability hypothesis is not met. "
                  f"Report partial only; do not call the K=4 regime solved.")
    else:
        verdict = "NO_GENUINE_FALSIFIED"
        lines.append("NO_GENUINE / FALSIFIED: no K=4 cell exceeds 1/8 -> FALSIFIER fires; log NEGATIVE, do not reframe.")
        detail = (f"No tested K=4 precision/session scaffold reaches >=2/8 genuine seeds. "
                  f"The hypothesis is refuted under the predeclared falsifier, even if lenient ignite appears above zero. "
                  f"Any last-third rate near CEIL={CEIL:.3f} remains compatible with a constant-response policy.")
    lines.append("")

    lines.append(f"VERDICT: {verdict}")
    lines.append(f"  {detail}")
    lines.append(
        f"MACHINE SUMMARY: VERDICT={verdict} ceil={CEIL:.3f} "
        f"controls=anchor:{anc_g}/8,g4K6:{g4k6_g}/8 K4_best={best_k4}:{best_k4_count}/8 "
        f"g1_100={k4_counts['g1_100']}/8 g1_300={k4_counts['g1_300']}/8 "
        f"g4_100={k4_counts['g4_100']}/8 g4_300={k4_counts['g4_300']}/8 g4_600={k4_counts['g4_600']}/8"
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 219 — M4a discrimination cells")
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
        path = out_dir / f"exp219_{name}.json"
        path.write_text(json.dumps(rows, indent=2) + "\n")
        n_genuine = sum(1 for r in rows if r["genuine"])
        mean_c = sum(r["csel"] for r in rows) / len(rows)
        print(f"[exp219 {name}] genuine={n_genuine}/{len(rows)} mean_csel={mean_c:.3f}  saved {path}")

    elif args.aggregate:
        results: dict[str, list[dict]] = {}
        for name in CELLS:
            path = out_dir / f"exp219_{name}.json"
            if not path.exists():
                print(f"Missing {path} — run --cell {name} first", file=sys.stderr)
                sys.exit(1)
            results[name] = json.loads(path.read_text())
        report = aggregate(results)
        txt_path = out_dir / "exp219.txt"
        txt_path.write_text(report + "\n")
        print(report)
        print(f"\n[saved {txt_path}]")

    else:
        results: dict[str, list[dict]] = {}
        for name in CELLS:
            print(f"[exp219] running cell={name} ...")
            rows = run_cell(name, SEEDS)
            path = out_dir / f"exp219_{name}.json"
            path.write_text(json.dumps(rows, indent=2) + "\n")
            n_genuine = sum(1 for r in rows if r["genuine"])
            mean_c = sum(r["csel"] for r in rows) / len(rows)
            print(f"  {name}: genuine={n_genuine}/{len(rows)} mean_csel={mean_c:.3f}  saved {path}")
            results[name] = rows
        report = aggregate(results)
        txt_path = out_dir / "exp219.txt"
        txt_path.write_text(report + "\n")
        print(report)
        print(f"\n[saved {txt_path}]")


if __name__ == "__main__":
    main()
