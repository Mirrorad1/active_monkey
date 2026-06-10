"""Exp 117 — the registered depth-law evaluation, scored as written.

The registration (Exp 107, verbatim): entering |gap| < 60 predicts flip with p >= 0.40;
entering |gap| >= 120 predicts hold with flip-p <= 0.10; 60-120 unbinned. Confirmation:
deep bin <= 1 flip in >= 5 entries AND razor bin >= 2 flips in >= 4 entries; else the
law fails or the sample is declared insufficient, honestly.

This scorer parses the committed outputs of the post-registration idle epochs (Exp
107-116) for the registered calls and outcomes, scores each bin exactly as registered,
and renders the verdict. Exp 112's ensemble measurement (deep flip rate 0.20 from one
deep state, n=20) is cited as context either way — the registration is NOT amended.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Registered-epoch records, hardcoded from the committed outputs.
# (output file, bin, outcome, verification substring)
# ---------------------------------------------------------------------------

RECORDS = [
    ("experiments/outputs/exp107.txt", "deep", "HELD",
     "P4 registered-call outcome: HELD as called"),
    ("experiments/outputs/exp108.txt", "unbinned", None,
     "unbinned middle, start gap +115.0"),
    ("experiments/outputs/exp109.txt", "deep", "HELD",
     "P4 registered-call outcome: HELD as called"),
    ("experiments/outputs/exp110.txt", "deep", "HELD",
     "P4 registered-call outcome: HELD as called"),
    ("experiments/outputs/exp111.txt", "deep", "FLIPPED",
     "P4 registered-call outcome: FLIPPED against the call"),
    ("experiments/outputs/exp113.txt", "deep", "HELD",
     "P4 registered-call outcome: HELD as called"),
    ("experiments/outputs/exp114.txt", "deep", "HELD",
     "P4 registered-call outcome: HELD as called"),
    ("experiments/outputs/exp115.txt", "deep", "HELD",
     "P4 registered-call outcome: HELD as called"),
    ("experiments/outputs/exp116.txt", "unbinned", None,
     "unbinned middle, start gap +94.6"),
]


def main() -> int:
    repo = Path(__file__).resolve().parent.parent  # repo root

    # ------------------------------------------------------------------
    # 1. Record verification: each hardcoded record must match its
    #    committed output file by substring.
    # ------------------------------------------------------------------
    print("=== RECORD VERIFICATION (committed outputs) ===")
    mismatched: list[str] = []
    for rel_file, _bin, _outcome, needle in RECORDS:
        fpath = repo / rel_file
        try:
            content = fpath.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"  MISSING (file absent): {rel_file!r} -> {needle!r}")
            mismatched.append(f"{rel_file}: FILE ABSENT")
            continue
        if needle in content:
            print(f"  FOUND  : {rel_file!r} -> {needle!r}")
        else:
            print(f"  MISSING: {rel_file!r} -> {needle!r}")
            mismatched.append(f"{rel_file}: {needle!r}")

    if mismatched:
        print()
        print(f"EVAL INVALID — record mismatch: {'; '.join(mismatched)}")
        return 1

    # ------------------------------------------------------------------
    # 2. Score the bins exactly as registered.
    # ------------------------------------------------------------------
    assert not any(b == "razor" for _f, b, _o, _n in RECORDS), (
        "no razor-bin epochs occurred in Exp 107-116"
    )

    deep_entries = sum(1 for _f, b, _o, _n in RECORDS if b == "deep")
    deep_flips = sum(
        1 for _f, b, o, _n in RECORDS if b == "deep" and o == "FLIPPED"
    )
    razor_entries = 0
    razor_flips = 0
    unbinned = sum(1 for _f, b, _o, _n in RECORDS if b == "unbinned")

    print()
    print("=== SCORE (as registered, Exp 107) ===")
    print(f"  deep bin  (|gap| >= 120, flip-p <= 0.10): "
          f"{deep_entries} entries, {deep_flips} flip(s)")
    print(f"  razor bin (|gap| <  60, flip-p >= 0.40): "
          f"{razor_entries} entries, {razor_flips} flip(s)")
    print(f"  unbinned middle (60-120, not scored)    : {unbinned} entries")

    # ------------------------------------------------------------------
    # 3. Verdict logic exactly as registered.
    # ------------------------------------------------------------------
    deep_ok = (deep_entries >= 5) and (deep_flips <= 1)
    razor_sufficient = razor_entries >= 4

    if deep_ok and razor_sufficient and razor_flips >= 2:
        verdict = "CONFIRMED (both bins)"
    elif deep_ok and not razor_sufficient:
        verdict = (
            f"PARTIAL — deep bin CONFIRMED AS REGISTERED (at its limit: "
            f"{deep_flips}/{deep_entries} vs <= 1 tolerated); razor bin "
            f"INSUFFICIENT SAMPLE ({razor_entries}/4 required) — the razor "
            f"side stays OPEN under the same registered numbers; a second "
            f"evaluation fires if/when >= 4 razor entries accumulate."
        )
    else:
        verdict = f"FAILED (deep bin: {deep_flips} flips in {deep_entries})"

    # ------------------------------------------------------------------
    # 4. Context (not part of the verdict) + final line.
    # ------------------------------------------------------------------
    print()
    print(
        "Exp 112 ensemble context: deep flip rate measured 0.20 (n=20, "
        "CI ~0.08-0.42) at one deep state — the registered 0.10 criterion "
        "passed at its limit and may be optimistic; cited, not amended."
    )
    print()
    print(f"EXP117: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
