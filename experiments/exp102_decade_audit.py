"""Exp 102 — decade self-audit of Exp 92-101 (the audit entry and the idle-mode epochs),
per VALIDATION.md's every-~10-experiments duty.

Predeclared: AUDIT-P1 — every (file, substring) pair below is found verbatim in the
committed output; any miss -> correction entry required. The hostile re-read findings
are logged in the EXPERIMENTS.md entry. This audit also recomputes the decade's forecast
calibration: the nine |error|/sigma landings quoted in the idle entries.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Audit table: (output file, [required substrings])
# ---------------------------------------------------------------------------

AUDIT = [
    ("experiments/outputs/exp92.txt", ["DECADE AUDIT: PASS (19/19 substrings, 10/10 scripts)"]),
    ("experiments/outputs/exp93.txt", ["EXP93: EPOCH CLEAN (bands held)", "favorite: 0 -> 2"]),
    ("experiments/outputs/exp94.txt", ["EXP94: EPOCH CLEAN (bands held)", "favorite: 0 -> 2"]),
    ("experiments/outputs/exp95.txt", ["EXP95: EPOCH CLEAN (bands held)", "favorite: 2 -> 2"]),
    ("experiments/outputs/exp96.txt", ["EXP96: EPOCH CLEAN (bands held)", "favorite: 2 -> 0"]),
    ("experiments/outputs/exp97.txt", ["EXP97: EPOCH CLEAN (bands held)", "favorite: 2 -> 2"]),
    ("experiments/outputs/exp98.txt", ["EXP98: EPOCH CLEAN (bands held)", "favorite: 0 -> 0"]),
    ("experiments/outputs/exp99.txt", ["EXP99: EPOCH CLEAN (bands held)", "end_gap=+12.0043"]),
    ("experiments/outputs/exp100.txt", ["EXP100: EPOCH CLEAN (bands held)", "end_gap=-315.8570"]),
    ("experiments/outputs/exp101.txt", ["EXP101: EPOCH CLEAN (bands held)", "end_gap=-128.1940"]),
]

SCRIPTS = [
    "experiments/exp92_decade_audit.py",
    "experiments/exp93_mirro_epoch.py",
    "experiments/exp94_vela_epoch.py",
    "experiments/exp95_mirro_epoch2.py",
    "experiments/exp96_vela_epoch2.py",
    "experiments/exp97_mirro_epoch3.py",
    "experiments/exp98_vela_epoch3.py",
    "experiments/exp99_mirro_epoch4.py",
    "experiments/exp100_vela_epoch4.py",
    "experiments/exp101_mirro_epoch5.py",
]

# Calibration files: exp93-101 (nine idle-mode epochs)
CALIB_FILES = [
    "experiments/outputs/exp93.txt",
    "experiments/outputs/exp94.txt",
    "experiments/outputs/exp95.txt",
    "experiments/outputs/exp96.txt",
    "experiments/outputs/exp97.txt",
    "experiments/outputs/exp98.txt",
    "experiments/outputs/exp99.txt",
    "experiments/outputs/exp100.txt",
    "experiments/outputs/exp101.txt",
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
    # 3. Calibration recompute: extract |error|/sigma from exp93-101
    # ------------------------------------------------------------------
    print("\n=== CALIBRATION RECOMPUTE (|error|/sigma, exp93-101) ===")
    NEEDLE = "|error| / sigma   :"
    z_values: list[float] = []
    calib_missing: list[str] = []

    for rel_file in CALIB_FILES:
        fpath = repo / rel_file
        found_line = None
        try:
            for line in fpath.read_text(encoding="utf-8").splitlines():
                if NEEDLE in line:
                    found_line = line
                    break
        except FileNotFoundError:
            pass

        label = rel_file.split("/")[-1]
        if found_line is None:
            print(f"  MISSING  : {label}  — calibration line not found")
            calib_missing.append(label)
        else:
            # Extract the float after the colon
            raw = found_line.split(":")[-1].strip()
            try:
                z = float(raw)
                z_values.append(z)
                print(f"  {label}: |error|/sigma = {z:.3f}")
            except ValueError:
                print(f"  PARSE-ERROR: {label}  — could not parse {raw!r}")
                calib_missing.append(label)

    print()
    if z_values:
        mean_z = sum(z_values) / len(z_values)
        tail_count = sum(1 for z in z_values if z > 2.0)
        print(f"  n values parsed : {len(z_values)}/9")
        print(f"  mean |error|/sigma: {mean_z:.3f}")
        print(f"  tail > 2sigma   : {tail_count}/{len(z_values)}")
    else:
        mean_z = float("nan")
        tail_count = 0
        print("  No calibration values parsed.")

    # ------------------------------------------------------------------
    # 4. Final verdict
    # ------------------------------------------------------------------
    print()
    all_missing = list(missing) + missing_scripts
    n_scripts = len(SCRIPTS)
    n_scripts_found = n_scripts - len(missing_scripts)
    n_z = len(z_values)

    if not all_missing and not calib_missing:
        print(
            f"DECADE AUDIT 92-101: PASS ({found_subs}/{total_subs} substrings,"
            f" {n_scripts_found}/{n_scripts} scripts;"
            f" calibration mean |z| = {mean_z:.2f}, tail>2sigma = {tail_count}/9)"
        )
        return 0
    else:
        detail_parts = list(all_missing) + [f"calib MISSING: {f}" for f in calib_missing]
        detail = "; ".join(detail_parts)
        print(
            f"DECADE AUDIT 92-101: FAIL — [{detail}]"
            f" ({found_subs}/{total_subs} substrings,"
            f" {n_scripts_found}/{n_scripts} scripts;"
            f" calibration mean |z| = {mean_z:.2f}, tail>2sigma = {tail_count}/9)"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
