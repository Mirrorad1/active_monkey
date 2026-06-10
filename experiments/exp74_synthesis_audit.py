"""Exp 74 — social-emergence synthesis: the ladder's verdicts, audited against the
committed raw record.

The direction's stop condition is met (all five rungs have verdicts; the one deviation was
cascaded). Per VALIDATION.md the ~10-experiment self-audit is due (Exp 64-73 is the
decade). This script is the audit instrument: every headline claim quoted in the Exp 74
synthesis entry must be PRESENT in the committed raw output of the experiment it cites.
A synthesis that quotes numbers absent from the record fails the audit.

Predeclared: AUDIT-P1 — every (file, required-substring) pair below is found verbatim in
the committed output file. Falsifier F-AUDIT: any missing pair -> the synthesis (or a past
entry) misquotes the record -> a correction entry is required and the synthesis may not be
logged as audited. No tolerance, no fuzzy matching.

This is a CONSOLIDATION artifact: no new run, no new claim — the value is the
cross-checked closing of the direction.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Audit table: (output file, [required substrings]) — every quoted headline
# must be present verbatim in the committed output.
# ---------------------------------------------------------------------------

AUDIT = [
    ("experiments/outputs/exp64.txt", [
        "RUNG 2: PASS",
        "seed=0: mirro_colocations=89",
        "PASS  P1-map-nondegradation-mirro: passes=5/5",
    ]),
    ("experiments/outputs/exp65.txt", [
        "RUNG 3: FAIL ['P1-magnitude-divergence>=0.02-in-4/5']",
        "s0:0.0160(FAIL)",
        "0.5079",
    ]),
    ("experiments/outputs/exp66.txt", [
        "EXP66: PASS",
        "ratio=5.25x",
        "passes=3/4 (need >=3)",
    ]),
    ("experiments/outputs/exp67.txt", [
        "EXP67: FAIL ['P1']",
        "accuracy=85.0% (17/20)",
        "6400_installs=1",
    ]),
    ("experiments/outputs/exp68.txt", [
        "EXP68: PASS",
        "P2-blind-null-R-in-[0.75,1.33]->=4/5-seeds: seeds_passing=5/5",
    ]),
    ("experiments/outputs/exp69.txt", [
        "G3 FAIL — gate never fired; RUN INVALID",
        "0.7509",
    ]),
    ("experiments/outputs/exp70.txt", [
        "EXP70: DEPARTURE (EXCLUSION, cascade queued)",
        "modal=EXCLUSION  modal_count=5",
        "A_wins=5/5=1.000",
    ]),
    ("experiments/outputs/exp71.txt", [
        "EXP71: CASCADE COMPLETE",
        "R2 deflation signature count (asym>0.5 AND winner=A): 8/8",
        "A wins: 7/7 (1.00)",
    ]),
    ("experiments/outputs/exp72.txt", [
        "total_violations=0",
        "winner==closer=3  fraction=0.5",
        "P2 (camping_frac >= 0.25 in >=4/5): seeds_passing=0/5  -> FAIL",
    ]),
    ("experiments/outputs/exp73.txt", [
        "EXP73: PASS",
        "RUNG 5: CONVERGENCE IS MASS-GATED",
    ]),
]

# Scripts that must exist on disk (one per experiment in the decade).
SCRIPTS = [
    "experiments/exp64_copresence.py",
    "experiments/exp65_value_transmission.py",
    "experiments/exp66_young_receiver.py",
    "experiments/exp67_sensitive_period.py",
    "experiments/exp68_comfort_baseline.py",
    "experiments/exp69_depletion_coupling.py",
    "experiments/exp70_regime_classification.py",
    "experiments/exp71_exclusion_cascade.py",
    "experiments/exp72_kidnapped_twin.py",
    "experiments/exp73_dialect_coupling.py",
]

# Static synthesis table — ladder verdicts (no computation; consolidation summary).
SYNTHESIS_TABLE = [
    ("rung 1", "Exp 63",       "clade plumbing",
     "POSITIVE (consolidation)"),
    ("rung 2", "Exp 64",       "co-presence safe; coupling inert at sharp beliefs",
     "POSITIVE (consolidation)"),
    ("rung 3", "Exp 65-67",    "adult NEGATIVE (mass-limited); young ADOPT (first transmission); ambivalence-gated",
     "NEG+BREAKTHROUGH+MIXED"),
    ("rung 4", "Exp 68-72",    "departure real; deflated to stigmergic unilateral-retreat lock-in; intrinsic, stochastic winner",
     "cascaded"),
    ("rung 5", "Exp 73",       "dialect convergence mass-gated; stable heavy dialects",
     "POSITIVE"),
    ("cross-law", "60,65-67,72,73",
     "influence = dose vs accumulated evidence mass (percepts, values, words)",
     "the direction's main yield"),
]


def main() -> int:
    repo = Path(__file__).resolve().parent.parent  # repo root

    missing: list[str] = []

    # ------------------------------------------------------------------
    # 1. Substring checks against committed output files
    # ------------------------------------------------------------------
    print("=== SUBSTRING AUDIT ===")
    total_subs = 0
    for rel_file, subs in AUDIT:
        fpath = repo / rel_file
        try:
            content = fpath.read_text(encoding="utf-8")
        except FileNotFoundError:
            for s in subs:
                total_subs += 1
                tag = "MISSING (file absent)"
                print(f"  {tag}: {rel_file!r} -> {s!r}")
                missing.append(f"{rel_file}: FILE ABSENT")
            continue

        for s in subs:
            total_subs += 1
            if s in content:
                print(f"  FOUND  : {rel_file!r} -> {s!r}")
            else:
                print(f"  MISSING: {rel_file!r} -> {s!r}")
                missing.append(f"{rel_file}: {s!r}")

    total_subs_found = total_subs - sum(
        1 for m in missing if not m.endswith("FILE ABSENT")
    )
    # Recount cleanly
    found_subs = 0
    for rel_file, subs in AUDIT:
        fpath = repo / rel_file
        if not fpath.exists():
            continue
        content = fpath.read_text(encoding="utf-8")
        for s in subs:
            if s in content:
                found_subs += 1

    # ------------------------------------------------------------------
    # 2. Script existence checks
    # ------------------------------------------------------------------
    print("\n=== SCRIPT EXISTENCE ===")
    missing_scripts: list[str] = []
    for rel_script in SCRIPTS:
        spath = repo / rel_script
        if spath.exists():
            print(f"  PRESENT: {rel_script}")
        else:
            print(f"  ABSENT : {rel_script}")
            missing_scripts.append(rel_script)

    # ------------------------------------------------------------------
    # 3. Synthesis table (static — no computation)
    # ------------------------------------------------------------------
    print("\n=== SYNTHESIS TABLE — SOCIAL-EMERGENCE LADDER ===")
    col_w = [10, 14, 62, 30]
    header = f"{'rung':<{col_w[0]}} {'experiment':<{col_w[1]}} {'finding':<{col_w[2]}} {'verdict'}"
    print(header)
    print("-" * (sum(col_w) + 3))
    for rung, exp, finding, verdict in SYNTHESIS_TABLE:
        print(f"{rung:<{col_w[0]}} {exp:<{col_w[1]}} {finding:<{col_w[2]}} {verdict}")

    # ------------------------------------------------------------------
    # 4. Final verdict
    # ------------------------------------------------------------------
    print()
    all_missing = [m for m in missing] + missing_scripts
    n_scripts = len(SCRIPTS)
    n_scripts_found = n_scripts - len(missing_scripts)

    if not all_missing:
        print(
            f"AUDIT: PASS ({found_subs}/{total_subs} substrings,"
            f" {n_scripts_found}/{n_scripts} scripts)"
        )
        return 0
    else:
        detail = "; ".join(all_missing)
        print(
            f"AUDIT: FAIL — [{detail}]"
            f" ({found_subs}/{total_subs} substrings,"
            f" {n_scripts_found}/{n_scripts} scripts)"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
