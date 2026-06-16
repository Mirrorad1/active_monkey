"""Exp 217 — M4a cold-start break: can honest exploration push the direct-head dyad to reliable ignition?

AUTHORIZATION: the human's 'continue' at the Exp 216 halt -> run increment 1f (break the exploration cold-start)

PLAIN: Exp 216 showed the direct-head dyad CAN ignite under a generous honest scaffold, but NOT reliably
(1-2/4 seeds, depending on lucky early exploration).  The hypothesis is that the failure is an EXPLORATION
COLD-START: early uniform exploration is too sparse for the agent to find the POS signal quickly, and high
policy precision then locks it onto a wrong guess before the table is built.  If so, adding honest
exploration (eps-greedy random overrides, or an optimistic POS prior that broadens initial policy coverage)
should break the cold-start and increase ignition across seeds — WITHOUT any information about which
response is correct (HONEST = same treatment for every response column).

PREDECLARATION (predeclaration made before running any arm; falsifier named in advance):
  Three arms, all on the SAME generous scaffold (K=6, lr_pA=4, gamma=8, 300 turns, seeds 20-27 / N=8):
    baseline   — no exploration flags (reproduces Exp 216 parameter set; the regression-check arm).
    epsgreedy  — eps-greedy decaying exploration: eps0=0.5, eps_min=0.05, eps_turns=200.
    optimistic — optimistic POS prior: optimism=2.0 (uniform +2.0 to pA1[:,POS,:,:] before Agent build).

  Metric per seed: improv = last_third_POS_rate - first_third_POS_rate.
    ignite(seed) = (improv >= 0.15 AND last_third >= 0.30).

  DECISION RULE (predeclared; applies to best exploration arm = max ignitions among {epsgreedy, optimistic}):
    RELIABLE / COLD_START_BREAKABLE  iff best_ig >= 5/8 AND best_mean_last >= 0.40 AND best_ig - base_ig >= 3
      -> honest exploration breaks the cold-start; next = ratchet the scaffold toward realism (Exp 218/1g).
    PARTIAL iff (best_ig - base_ig) >= 2 but not RELIABLE
      -> exploration helps but is not sufficient alone; investigate further.
    NOT_COLD_START / FALSIFIED iff (best_ig - base_ig) <= 1
      -> the exploration-cold-start hypothesis is REFUTED; the bottleneck is elsewhere
         (learning-signal strength / structural); reconsider, do not claim reliability.

  FALSIFIER: if BOTH exploration arms fail to improve over baseline by more than 1 ignition, the
  hypothesis that "exploration cold-start is the main bottleneck" is refuted.  Log as NEGATIVE and
  do not reframe.

Honesty: every scaffold parameter changes HOW the agent explores/exploits, not WHAT the answer is.
The optimism increment and the eps-greedy overrides are identical for all (intent, response) pairs.
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

# ── correct-response mapping (same as Exp 216) ──────────────────────────────
CORRECT = {c: c % 4 for c in range(U)}

# ── 8 seeds: 20-23 overlap Exp 216 for the regression check ─────────────────
SEEDS = list(range(20, 28))

# ── arm -> extra DirectHeadAgent kwargs ─────────────────────────────────────
ARMS: dict[str, dict[str, Any]] = {
    "baseline":   {},
    "epsgreedy":  {"eps0": 0.5, "eps_min": 0.05, "eps_turns": 200},
    "optimistic": {"optimism": 2.0},
}

# ── shared generous scaffold ─────────────────────────────────────────────────
_SCAFFOLD = dict(gamma=8.0, alpha=8.0, lr_pA=4.0, lv=LV)


def _shuffled_codes(rng):
    """Infinite iterator of codes drawn from exhaustive shuffled blocks (same as Exp 216)."""
    pool: list[int] = []
    def nxt():
        nonlocal pool
        if not pool:
            b = list(range(U)); rng.shuffle(b); pool += b
        return pool.pop(0)
    return nxt


def closed_loop(arm: str, seed: int, turns: int = 300) -> tuple[float, float]:
    """Full DirectHeadAgent bootstrap for one arm/seed.

    Returns (first_third_POS_rate, last_third_POS_rate).
    Structure IDENTICAL to exp216.closed_loop: perceive -> act -> valence ->
    observe_feedback; valence = POS iff r == CORRECT[code], NEU iff r == ASK, NEG otherwise.
    """
    np.random.seed(seed)
    arm_kwargs = ARMS[arm]
    ag = DirectHeadAgent(
        build_direct_head_model(seed, k=6),
        seed=seed,
        **_SCAFFOLD,
        **arm_kwargs,
    )
    nxt = _shuffled_codes(np.random.default_rng(seed))
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


def run_arm(arm: str, seeds: list[int] = SEEDS) -> list[dict]:
    """Run one arm over all seeds; return list of per-seed result dicts."""
    rows = []
    for seed in seeds:
        first, last = closed_loop(arm, seed)
        improv = last - first
        ignite = (improv >= 0.15 and last >= 0.30)
        rows.append(dict(seed=seed, first=round(first, 4), last=round(last, 4),
                         improv=round(improv, 4), ignite=ignite))
    return rows


def aggregate(results: dict[str, list[dict]]) -> str:
    """Apply the predeclared decision rule; return the full human-readable report."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("EXP 217 — M4a cold-start break: honest exploration vs. baseline (8 seeds)")
    lines.append("=" * 78)
    lines.append("")

    arm_ignitions: dict[str, int] = {}
    arm_mean_last: dict[str, float] = {}

    for arm, rows in results.items():
        lines.append(f"--- ARM: {arm} ---")
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
        arm_ignitions[arm] = n_ignite
        arm_mean_last[arm] = mean_last
        lines.append(f"  ignitions: {n_ignite}/{len(rows)}   mean last-third POS-rate: {mean_last:.3f}")
        lines.append("")

    # ── regression check (seeds 20-23 vs Exp 216 baseline expectation) ──────
    lines.append("--- REGRESSION CHECK: baseline seeds 20-23 vs Exp 216 expectation ---")
    # Exp 216 expectation: seed20 improv ~+0.24 (ignites); seed21,22 ~0 (flat); seed23 ~ -0.02 (flat)
    exp216_expect = {20: (True, +0.24), 21: (False, 0.0), 22: (False, 0.0), 23: (False, -0.02)}
    tol = 0.03
    reg_ok = True
    baseline_rows = {r["seed"]: r for r in results.get("baseline", [])}
    for s, (exp_ignite, exp_improv) in exp216_expect.items():
        if s not in baseline_rows:
            lines.append(f"  seed {s}: MISSING from baseline results")
            reg_ok = False
            continue
        row = baseline_rows[s]
        match = (row["ignite"] == exp_ignite) and (abs(row["improv"] - exp_improv) <= tol)
        status = "OK" if match else "MISMATCH"
        if not match:
            reg_ok = False
        lines.append(f"  seed {s}: improv {row['improv']:+.3f} (expect ~{exp_improv:+.3f}, "
                     f"ignite={row['ignite']} expect={exp_ignite}) -> {status}")
    lines.append(f"  Regression: {'PASS' if reg_ok else 'FAIL (tolerance +-0.03)'}")
    lines.append("")

    # ── decision rule ────────────────────────────────────────────────────────
    base_ig = arm_ignitions.get("baseline", 0)
    exploration_arms = {a: arm_ignitions[a] for a in ["epsgreedy", "optimistic"] if a in arm_ignitions}
    best_arm = max(exploration_arms, key=lambda a: exploration_arms[a])
    best_ig = exploration_arms[best_arm]
    best_mean_last = arm_mean_last[best_arm]
    delta = best_ig - base_ig

    lines.append("--- VERDICT (predeclared decision rule) ---")
    lines.append(f"  baseline ignitions: {base_ig}/8")
    for a, ig in exploration_arms.items():
        lines.append(f"  {a} ignitions: {ig}/8   mean_last: {arm_mean_last[a]:.3f}")
    lines.append(f"  best exploration arm: {best_arm} ({best_ig}/8, mean_last={best_mean_last:.3f})")
    lines.append(f"  delta over baseline: {delta:+d}")
    lines.append("")

    if best_ig >= 5 and best_mean_last >= 0.40 and delta >= 3:
        verdict = "RELIABLE / COLD_START_BREAKABLE"
        detail = (f"Honest exploration ({best_arm}) breaks the cold-start: {best_ig}/8 ignitions "
                  f"(mean_last={best_mean_last:.3f}), delta={delta:+d} over baseline.  "
                  f"Next: ratchet the scaffold toward realism (Exp 218/1g).")
        machine = f"VERDICT=RELIABLE arm={best_arm} best_ig={best_ig}/8 base_ig={base_ig}/8 delta={delta:+d} mean_last={best_mean_last:.3f}"
    elif delta >= 2:
        verdict = "PARTIAL"
        detail = (f"Exploration helps but is not sufficient alone: best arm ({best_arm}) "
                  f"{best_ig}/8 ignitions (mean_last={best_mean_last:.3f}), delta={delta:+d} over "
                  f"baseline.  Cold-start partially broken; further investigation needed.")
        machine = f"VERDICT=PARTIAL arm={best_arm} best_ig={best_ig}/8 base_ig={base_ig}/8 delta={delta:+d} mean_last={best_mean_last:.3f}"
    else:
        verdict = "NOT_COLD_START / FALSIFIED"
        detail = (f"Exploration-cold-start hypothesis REFUTED: best exploration arm ({best_arm}) "
                  f"adds only delta={delta:+d} ignitions over baseline ({best_ig}/8 vs {base_ig}/8).  "
                  f"The bottleneck is elsewhere (learning-signal strength / structural).  "
                  f"Reconsider; do NOT claim reliability.")
        machine = f"VERDICT=FALSIFIED arm={best_arm} best_ig={best_ig}/8 base_ig={base_ig}/8 delta={delta:+d} mean_last={best_mean_last:.3f}"

    lines.append(f"VERDICT: {verdict}")
    lines.append(f"  {detail}")
    lines.append("")
    lines.append(f"MACHINE SUMMARY: {machine}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 217 — M4a cold-start exploration arms")
    parser.add_argument("--arm", metavar="NAME", help="Run one arm over SEEDS and write its JSON")
    parser.add_argument("--aggregate", action="store_true",
                        help="Read per-arm JSONs and write the full report")
    args = parser.parse_args()

    out_dir = _REPO / "experiments" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.arm:
        arm = args.arm
        if arm not in ARMS:
            print(f"Unknown arm: {arm!r}. Valid arms: {list(ARMS)}", file=sys.stderr)
            sys.exit(1)
        rows = run_arm(arm, SEEDS)
        path = out_dir / f"exp217_{arm}.json"
        path.write_text(json.dumps(rows, indent=2) + "\n")
        n_ig = sum(1 for r in rows if r["ignite"])
        mean_last = sum(r["last"] for r in rows) / len(rows)
        print(f"[exp217 {arm}] ignitions={n_ig}/{len(rows)} mean_last={mean_last:.3f}  saved {path}")

    elif args.aggregate:
        results: dict[str, list[dict]] = {}
        for arm in ARMS:
            path = out_dir / f"exp217_{arm}.json"
            if not path.exists():
                print(f"Missing {path} — run --arm {arm} first", file=sys.stderr)
                sys.exit(1)
            results[arm] = json.loads(path.read_text())
        report = aggregate(results)
        txt_path = out_dir / "exp217.txt"
        txt_path.write_text(report + "\n")
        print(report)
        print(f"\n[saved {txt_path}]")

    else:
        # No args: run all arms serially then aggregate
        results: dict[str, list[dict]] = {}
        for arm in ARMS:
            print(f"[exp217] running arm={arm} ...")
            rows = run_arm(arm, SEEDS)
            path = out_dir / f"exp217_{arm}.json"
            path.write_text(json.dumps(rows, indent=2) + "\n")
            n_ig = sum(1 for r in rows if r["ignite"])
            mean_last = sum(r["last"] for r in rows) / len(rows)
            print(f"  {arm}: ignitions={n_ig}/{len(rows)} mean_last={mean_last:.3f}  saved {path}")
            results[arm] = rows
        report = aggregate(results)
        txt_path = out_dir / "exp217.txt"
        txt_path.write_text(report + "\n")
        print(report)
        print(f"\n[saved {txt_path}]")


if __name__ == "__main__":
    main()
