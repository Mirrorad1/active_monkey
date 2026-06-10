"""Exp 92 — decade self-audit of Exp 75-84 (the spine epochs, accrual-law arc, and noise
model), per VALIDATION.md's every-~10-experiments duty.

Exp 91's audit covered the graded-uncertainty ladder (85-90); the preceding arc has not
been machine-audited. Predeclared: AUDIT-P1 — every (file, substring) pair below is found
verbatim in the committed output; any miss -> correction entry required. The hostile
re-read findings (claims that would not survive replication from entry text alone) are
logged in the EXPERIMENTS.md entry, not here.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Audit table: (output file, [required substrings]) — every quoted headline
# must be present verbatim in the committed output.
# ---------------------------------------------------------------------------

AUDIT = [
    ("experiments/outputs/exp75.txt", ["EXP75: UNSTABLE (F1 branch)", "0.160000"]),
    ("experiments/outputs/exp76.txt", ["EXP76: F2 — NATURAL FLIP at checkpoint 4", "-32.0552"]),
    ("experiments/outputs/exp77.txt", ["EXP77: H1 CONFIRMED (gate asymmetry is the accrual law)", "ratio=predicted/observed=-43.8195/-37.3051=1.1746"]),
    ("experiments/outputs/exp78.txt", ["EXP78: F1 — aliasing reading WRONG (correction required)"]),
    ("experiments/outputs/exp79.txt", ["EXP79: F1 — mechanism wrong (re-correction needed)", "P3 INVALID"]),
    ("experiments/outputs/exp80.txt", ["EXP80: LAW HOLDS FORWARD (P1+P2+P3)", "-44.8358"]),
    ("experiments/outputs/exp81.txt", ["EXP81: F1 — law fails out-of-individual, HALT", "-21.6836"]),
    ("experiments/outputs/exp82.txt", ["EXP82: DIAGNOSIS — dominant term = Delta_visit (-55.2 of -83.7); prediction wrong", "1.0000 (6000/6000 steps)"]),
    ("experiments/outputs/exp83.txt", ["std(D) [ddof=1]    = 121.0496", "branches flipped to favorite=0: 8/20"]),
    ("experiments/outputs/exp84.txt", ["EXP84: F1 — exponent(s) 0.61/0.72, not sqrt-t", "0.0472"]),
]

SCRIPTS = [f"experiments/exp{n}_{s}.py" for n, s in [
    (75, "undisturbed_epoch"),
    (76, "margin_watch"),
    (77, "accrual_diagnosis"),
    (78, "aliasing_isolation"),
    (79, "early_tax"),
    (80, "forward_test"),
    (81, "vela_forward"),
    (82, "drift_accounting"),
    (83, "law_error_bars"),
    (84, "noise_scaling"),
]]

# Static arc table — experiment verdicts (no computation; consolidation summary).
ARC_TABLE = [
    ("75", "value core stable, map self-heals, band design error",              "MIXED"),
    ("76", "the spine's first natural opinion flip",                            "MIXED-F2"),
    ("77", "accrual law confirmed (interpretation later corrected by 78-79)",   "POSITIVE+corrected"),
    ("78", "aliasing reading falsified; scar verified; (its own effect un-replicated by 79)", "MIXED+corrected"),
    ("79", "re-correction: layout effect was noise; effect-size rule added",    "NEGATIVE"),
    ("80", "law holds forward (evidence later downgraded by 83)",               "POSITIVE+downgraded"),
    ("81", "law fails out-of-individual (interpretation corrected by 82)",      "NEGATIVE+corrected"),
    ("82", "bit-exact replay: it was walk noise",                               "POSITIVE"),
    ("83", "sigma 4x mean; flips ~40%/epoch",                                   "MIXED"),
    ("84", "superdiffusive noise over saturating signal",                       "MIXED"),
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
                print(f"  MISSING (file absent): {rel_file!r} -> {s!r}")
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
    # 3. Arc table (static — no computation)
    # ------------------------------------------------------------------
    print("\n=== ARC TABLE — EXP 75-84 (SPINE EPOCHS / ACCRUAL-LAW / NOISE MODEL) ===")
    col_w = [4, 74, 22]
    header = f"{'exp':<{col_w[0]}} {'finding':<{col_w[1]}} {'verdict'}"
    print(header)
    print("-" * (sum(col_w) + 2))
    for exp, finding, verdict in ARC_TABLE:
        print(f"{exp:<{col_w[0]}} {finding:<{col_w[1]}} {verdict}")

    # ------------------------------------------------------------------
    # 4. Final verdict
    # ------------------------------------------------------------------
    print()
    all_missing = list(missing) + missing_scripts
    n_scripts = len(SCRIPTS)
    n_scripts_found = n_scripts - len(missing_scripts)

    if not all_missing:
        print(
            f"DECADE AUDIT: PASS ({found_subs}/{total_subs} substrings,"
            f" {n_scripts_found}/{n_scripts} scripts)"
        )
        return 0
    else:
        detail = "; ".join(all_missing)
        print(
            f"DECADE AUDIT: FAIL — [{detail}]"
            f" ({found_subs}/{total_subs} substrings,"
            f" {n_scripts_found}/{n_scripts} scripts)"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
