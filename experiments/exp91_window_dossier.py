"""Exp 91 — graded-uncertainty rung 5: the promotion dossier, audited against the record.

The ladder is complete (rungs 1-4: Exp 85-90). Per the direction card, rung 5 is the
promotion CONSULT — and per VALIDATION.md, a synthesis quoting numbers must verify them
against the committed raw outputs. Predeclared: AUDIT-P1 — every (file, substring) pair
below is found verbatim; any miss -> correction entry required, the dossier may not be
posted as audited.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Audit table: (output file, [required substrings]) — every quoted headline
# must be present verbatim in the committed output.
# ---------------------------------------------------------------------------

AUDIT = [
    ("experiments/outputs/exp85.txt", ["EXP85: NO-HARM PASS — rung 2 (scar healing) unlocked", "333.79"]),
    ("experiments/outputs/exp86.txt", ["EXP86: F2 — healing mechanism does not work at the claimed rate; halt ladder", "mean_diff=0.41623"]),
    ("experiments/outputs/exp87.txt", ["EXP87: REPLACEMENT CLOCK VALIDATED — ladder resumes (rung 3 next)"]),
    ("experiments/outputs/exp88.txt", ["EXP88: WINDOW THEOREM DEMONSTRATED", "ratio=2.727272727272727"]),
    ("experiments/outputs/exp89.txt", ["EXP89: THE WALL OPENS (and stands without the window) — adults are reachable at LV=0.999", "666.5"]),
    ("experiments/outputs/exp90.txt", ["S=1/8 <= 2 => empty window confirmed"]),
]

SCRIPTS = [
    "experiments/exp85_lambda_no_harm.py",
    "experiments/exp86_scar_healing.py",
    "experiments/exp87_replacement_clock.py",
    "experiments/exp88_window_theorem.py",
    "experiments/exp89_wall_reopened.py",
    "experiments/exp90_fridge_window.py",
]

# Static synthesis table — ladder verdicts (no computation; consolidation summary).
SYNTHESIS_TABLE = [
    ("rung 1", "Exp 85",
     "no-harm: costless in static worlds; equilibrium mass = arithmetic",
     "PASS 8/8"),
    ("rung 2", "Exp 86-87",
     "scars heal to ~zero; rate law corrected to the replacement clock, validated out-of-sample",
     "PASS (one authorial clock error, fixed)"),
    ("rung 3", "Exp 88",
     "THE WINDOW THEOREM: robustness horizon = adaptation horizon = 1/(1-LV), age-free; non-decay rigidity grows with age (2.73x)",
     "PASS all bands"),
    ("rung 4a", "Exp 89",
     "the adult-transmission wall OPENS at LV=0.999 (adoption 6/8); control replicates Exp 65",
     "PASS"),
    ("rung 4b", "Exp 90",
     "EMPTY WINDOW at the fridge: no rate beats fast viability; exploration reflex NECESSARY",
     "CONFIRMED"),
    ("rung 5", "Exp 91",
     "promotion CONSULT posted: spines stay non-decaying reference lines; window + reflex go into M4a design",
     "THIS ENTRY"),
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
    print("\n=== SYNTHESIS TABLE — GRADED-UNCERTAINTY LADDER ===")
    col_w = [10, 12, 74, 36]
    header = f"{'rung':<{col_w[0]}} {'experiment':<{col_w[1]}} {'finding':<{col_w[2]}} {'verdict'}"
    print(header)
    print("-" * (sum(col_w) + 3))
    for rung, exp, finding, verdict in SYNTHESIS_TABLE:
        print(f"{rung:<{col_w[0]}} {exp:<{col_w[1]}} {finding:<{col_w[2]}} {verdict}")

    # ------------------------------------------------------------------
    # 4. Final verdict
    # ------------------------------------------------------------------
    print()
    all_missing = list(missing) + missing_scripts
    n_scripts = len(SCRIPTS)
    n_scripts_found = n_scripts - len(missing_scripts)

    if not all_missing:
        print(
            f"DOSSIER AUDIT: PASS ({found_subs}/{total_subs} substrings,"
            f" {n_scripts_found}/{n_scripts} scripts)"
        )
        return 0
    else:
        detail = "; ".join(all_missing)
        print(
            f"DOSSIER AUDIT: FAIL — [{detail}]"
            f" ({found_subs}/{total_subs} substrings,"
            f" {n_scripts_found}/{n_scripts} scripts)"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
