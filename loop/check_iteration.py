"""Mechanical rubric checker for one loop-B iteration.

This module turns loop/PROTOCOL.md's entry-format rules into runnable checks.
Hard checks cause failure; soft checks emit warnings.  The principle is from
loop/META.md: prefer a MECHANICAL guard over a prose reminder — if a rule can
be stated in code, code should enforce it.

Usage:
    python loop/check_iteration.py [N] [--strict]

N defaults to the highest entry number found in EXPERIMENTS.md.
--strict promotes all warnings to hard failures.

Exit code 1 on any hard failure (or any warning under --strict), else 0.
"""
import argparse
import glob
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Entries from Exp 152 on must carry a `- Verifier:` line, per PROTOCOL
# step 4.5 added 2026-06-10.
VERIFIER_FLOOR = 152


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def find_entries(text: str) -> dict[int, str]:
    """Parse EXPERIMENTS.md text and return {n: entry_text} for every entry.

    An entry starts at a line matching ``^## Exp N`` and runs until the next
    such heading or EOF.  The heading line is included in the entry text.
    """
    pattern = re.compile(r"^## Exp (\d+)\b", re.MULTILINE)
    matches = list(pattern.finditer(text))
    entries: dict[int, str] = {}
    for i, m in enumerate(matches):
        n = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entries[n] = text[start:end]
    return entries


# ---------------------------------------------------------------------------
# Core checker
# ---------------------------------------------------------------------------

def check_entry(
    n: int,
    *,
    experiments_text: str,
    root: pathlib.Path,
) -> tuple[list[str], list[str]]:
    """Check entry N against the protocol rubric.

    Returns (hard_failures, warnings).  Pure enough to unit-test: the caller
    supplies the EXPERIMENTS.md text and the repo root (so tests can use
    tmp_path roots with synthetic layouts).
    """
    hard: list[str] = []
    warn: list[str] = []

    # ------------------------------------------------------------------
    # 1. Entry exists
    # ------------------------------------------------------------------
    entries = find_entries(experiments_text)
    if n not in entries:
        hard.append(f"no entry found for Exp {n} in EXPERIMENTS.md")
        # Cannot check anything further without the entry text.
        return hard, warn

    entry = entries[n]

    # ------------------------------------------------------------------
    # 2. Plain line
    # ------------------------------------------------------------------
    if not re.search(r"^- Plain:", entry, re.MULTILINE):
        hard.append(f"Exp {n}: missing `- Plain:` line")

    # ------------------------------------------------------------------
    # 3. Verdict line with POSITIVE / NEGATIVE / MIXED
    # ------------------------------------------------------------------
    has_verdict_line = bool(re.search(r"^- Verdict:", entry, re.MULTILINE))
    has_verdict_word = bool(re.search(r"\b(POSITIVE|NEGATIVE|MIXED)\b", entry))
    if not has_verdict_line:
        hard.append(f"Exp {n}: missing `- Verdict:` line")
    elif not has_verdict_word:
        hard.append(
            f"Exp {n}: `- Verdict:` line present but entry contains none of "
            "POSITIVE / NEGATIVE / MIXED"
        )

    # ------------------------------------------------------------------
    # 4. CONSOLIDATION or NEW INSIGHT
    # ------------------------------------------------------------------
    if "CONSOLIDATION" not in entry and "NEW INSIGHT" not in entry:
        hard.append(
            f"Exp {n}: entry contains neither CONSOLIDATION nor NEW INSIGHT"
        )

    # ------------------------------------------------------------------
    # 5. Honest caveat
    # ------------------------------------------------------------------
    if not re.search(r"^- Honest caveat", entry, re.MULTILINE):
        hard.append(f"Exp {n}: missing `- Honest caveat` line")

    # ------------------------------------------------------------------
    # 6. Self-grade required for POSITIVE verdicts
    # (only when the entry is POSITIVE and NOT solely NEGATIVE or MIXED)
    # ------------------------------------------------------------------
    # Determine if the entry is a POSITIVE verdict (and not only NEGATIVE/MIXED).
    # Strategy: check if the Verdict line contains POSITIVE; if the entry
    # says only NEGATIVE or only MIXED that is not a POSITIVE verdict.
    verdict_line_m = re.search(r"^- Verdict:.*$", entry, re.MULTILINE)
    verdict_line_text = verdict_line_m.group(0) if verdict_line_m else ""
    is_positive_verdict = bool(re.search(r"\bPOSITIVE\b", verdict_line_text))

    if is_positive_verdict:
        has_self_grade = bool(
            re.search(r"\b(BREAKTHROUGH|POSITIVE-SINGLE)\b", entry)
        )
        if not has_self_grade:
            hard.append(
                f"Exp {n}: POSITIVE verdict requires a self-grade "
                "(BREAKTHROUGH or POSITIVE-SINGLE)"
            )

    # ------------------------------------------------------------------
    # 7. Script exists
    # ------------------------------------------------------------------
    script_pattern = str(root / "experiments" / f"exp{n}_*.py")
    scripts = glob.glob(script_pattern)
    if not scripts:
        hard.append(
            f"Exp {n}: no script found matching "
            f"experiments/exp{n}_*.py"
        )

    # ------------------------------------------------------------------
    # 8. Raw output exists
    # ------------------------------------------------------------------
    output_path = root / "experiments" / "outputs" / f"exp{n}.txt"
    if not output_path.exists():
        hard.append(
            f"Exp {n}: raw output missing at "
            f"experiments/outputs/exp{n}.txt"
        )

    # ------------------------------------------------------------------
    # 9. Script docstring contains "falsifier" AND one of
    #    "hypothesis" / "prediction" / "predeclar" (case-insensitive)
    # ------------------------------------------------------------------
    if scripts:
        docstring_ok = False
        for script_path in scripts:
            try:
                src = pathlib.Path(script_path).read_text(encoding="utf-8")
            except OSError:
                continue
            # Extract the first triple-quoted string (module docstring).
            doc_m = re.search(
                r'^"""(.*?)"""',
                src,
                re.DOTALL | re.MULTILINE,
            ) or re.search(
                r"^'''(.*?)'''",
                src,
                re.DOTALL | re.MULTILINE,
            )
            if doc_m:
                doc = doc_m.group(1).lower()
                has_falsifier = "falsifier" in doc
                has_hyp = any(
                    kw in doc
                    for kw in ("hypothesis", "prediction", "predeclar")
                )
                if has_falsifier and has_hyp:
                    docstring_ok = True
                    break
        if not docstring_ok:
            hard.append(
                f"Exp {n}: script module docstring must contain 'falsifier' "
                "AND at least one of 'hypothesis'/'prediction'/'predeclar' "
                "(case-insensitive)"
            )

    # ------------------------------------------------------------------
    # 10. Verifier line for entries >= VERIFIER_FLOOR
    # ------------------------------------------------------------------
    if n >= VERIFIER_FLOOR:
        if not re.search(r"^- Verifier:", entry, re.MULTILINE):
            hard.append(
                f"Exp {n}: entries from Exp {VERIFIER_FLOOR}+ require a "
                "`- Verifier:` line (PROTOCOL step 4.5, added 2026-06-10)"
            )

    # ------------------------------------------------------------------
    # Soft: re-quote check
    # ------------------------------------------------------------------
    if not output_path.exists():
        warn.append(
            f"Exp {n}: output file missing — skipping re-quote check "
            f"(experiments/outputs/exp{n}.txt)"
        )
    else:
        try:
            output_text = output_path.read_text(encoding="utf-8")
        except OSError:
            output_text = ""

        decimal_tokens = set(re.findall(r"\d+\.\d+", entry))
        fraction_tokens = set(re.findall(r"\b\d+/\d+\b", entry))
        all_tokens = decimal_tokens | fraction_tokens

        missing_tokens = [
            tok for tok in sorted(all_tokens) if tok not in output_text
        ]

        cap = 25
        for tok in missing_tokens[:cap]:
            warn.append(
                f'number "{tok}" quoted in entry not found in '
                f"experiments/outputs/exp{n}.txt "
                "(derived numbers are fine — confirm it is derived, not stale)"
            )
        if len(missing_tokens) > cap:
            warn.append(
                f"... {len(missing_tokens) - cap} further re-quote warnings "
                "truncated (cap=25)"
            )

    return hard, warn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_experiments_text(root: pathlib.Path) -> str:
    return (root / "EXPERIMENTS.md").read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mechanical rubric checker for one loop-B iteration."
    )
    parser.add_argument(
        "n",
        nargs="?",
        type=int,
        default=None,
        help="Experiment number to check (default: highest entry found).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Promote all warnings to hard failures.",
    )
    args = parser.parse_args(argv)

    experiments_text = _read_experiments_text(ROOT)
    entries = find_entries(experiments_text)

    if not entries:
        print("FAIL: no entries found in EXPERIMENTS.md", file=sys.stderr)
        return 1

    n = args.n if args.n is not None else max(entries)

    hard, warn = check_entry(n, experiments_text=experiments_text, root=ROOT)

    if args.strict:
        hard = hard + [f"[--strict] {w}" for w in warn]
        warn = []

    for msg in hard:
        print(f"FAIL: {msg}")
    for msg in warn:
        print(f"WARN: {msg}")

    j = len(hard)
    k = len(warn)
    if j:
        print(f"exp {n}: FAIL ({j} hard, {k} warnings)")
        return 1
    else:
        print(f"exp {n}: PASS ({k} warnings)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
