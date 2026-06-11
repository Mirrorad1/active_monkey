"""
Exp 140 — self-audit of the continuous-substrate chapter (Exp 133-139), per
VALIDATION.md's hostile-reviewer clause (due every ~10 experiments; 7 logged since
the last audit, and the right gate before any substrate-migration decision).

Method: every headline number cited in the EXPERIMENTS.md entries for Exp 133-139
is re-checked mechanically against the COMMITTED raw outputs (the only admissible
source). A citation that cannot be found verbatim in its committed output is a
DISCREPANCY. Known-discrepancy expectations are themselves asserted so the audit
is reproducible.

Predeclared standard: an entry passes iff all its checked citations appear in the
committed output. Verdict: CLEAN iff 7/7 entries pass; otherwise list each
discrepancy with its severity (verdict-affecting vs citation-only).
"""
from pathlib import Path

OUT = Path(__file__).parent / "outputs"

# (entry, output file, [required substrings from the entry's cited numbers])
CHECKS = [
    ("Exp 133", "exp133.txt", [
        "P1 localization  (final err < 0.1): 8/8 seeds",
        "P2 contraction   (no violation in all steps): 8/8 seeds",
        "P2 sanity        (final tr < 1% prior tr):    8/8 seeds",
        "P3 twin race     (gap <= +0.1 nats early): 8/8 seeds",
        "VERDICT: POSITIVE",
        "0.001225",
    ]),
    ("Exp 134", "exp134.txt", [
        "Total snapped runs (96 total): 0",
        "Max |tr_blend - tr_pure| across all 96 pairs: 0.000000e+00",
        "Spearman rho(log(L/sigma), cell_mean_gap) = -0.9824",
        "-245.5168",
        "VERDICT: NEGATIVE",
    ]),
    ("Exp 134 rerun", "exp134_rerun.txt", [
        "Spearman rho(log(L/sigma), gap) = -0.9824",
    ]),
    ("Exp 134 exact check", "exp134_exact_check.txt", [
        "exact log-space argmax != majority: 0/46",
        "floored-filter argmax != majority:  7/46",
    ]),
    ("Exp 135", "exp135.txt", [
        "NLL n_half ratio (cont/tab) at kappa0=1, nu0=4: 1.5169",
        "mean_n_half=110.4",
        "mean_n_half=216.7",
        "VERDICT: POSITIVE",
    ]),
    ("Exp 136 (committed, post-patch re-run)", "exp136.txt", [
        "P3(c): observe() slope = 0.170",
        "P3(c): Sigma-solve slope = 0.578",
        "VERDICT: MIXED",
    ]),
    ("Exp 137", "exp137.txt", [
        "NIW loses to fixed-lr by 22.0% at v=0.0005",
        "NIW loses to fixed-lr by 139.6% at v=0.002",
        "NIW loses to fixed-lr by 515.6% at v=0.008",
        "VERDICT: NEGATIVE",
    ]),
    ("Exp 137 mech", "exp137_mech.txt", [
        "ratio=1.0036",
        "ratio=0.9892",
        "ratio=0.9996",
        "keep_mean argmin N*=80",
        "keep_mean argmin N*=20",
        "keep_mean argmin N*=10",
    ]),
    ("Exp 138", "exp138.txt", [
        "12,800,000 word-samples per seed",
        "Closed-form training: 0 word-samples",
    ]),
    ("Exp 139", "exp139.txt", [
        "ratio=1.3971",
        "ratio=1.0794",
        "ratio=1.0225",
        "VERDICT: MIXED",
    ]),
]

# The found discrepancy: the Exp 136 ENTRY cites the superseded pre-patch run's
# wall-clock numbers. These strings must be ABSENT from the committed output
# (they were replaced by 1.02x / 0.0082 s on the re-run) — asserting the absence
# makes the discrepancy itself reproducible.
KNOWN_DISCREPANCIES = [
    ("Exp 136 entry cites '1.03x' (committed output: 1.02x)", "exp136.txt",
     "d=8 full-run / d=2 full-run = 1.03x"),
    ("Exp 136 entry cites '8.3 ms' / 0.0083 s (committed output: 0.0082 s)",
     "exp136.txt", "0.0083 s"),
]


def main() -> None:
    failures = []
    for entry, fname, needles in CHECKS:
        text = (OUT / fname).read_text()
        missing = [n for n in needles if n not in text]
        status = "PASS" if not missing else f"DISCREPANCY: missing {missing}"
        print(f"{entry:<40} {fname:<28} {status}")
        if missing:
            failures.append((entry, missing))

    print()
    for desc, fname, stale in KNOWN_DISCREPANCIES:
        text = (OUT / fname).read_text()
        confirmed = stale not in text
        print(f"KNOWN DISCREPANCY {'CONFIRMED' if confirmed else 'NOT FOUND'}: {desc}")
        if not confirmed:
            failures.append((desc, [stale]))

    print()
    if failures:
        print(f"AUDIT VERDICT: {len(failures)} unexpected failure(s) — see above")
    else:
        print("AUDIT VERDICT: 6/7 entries fully clean; 1 known citation-only "
              "discrepancy (Exp 136 wall-clock numbers quote the superseded "
              "pre-patch run; verdict and all deterministic numbers unaffected). "
              "Correction logged in the Exp 140 entry; PROTOCOL step 5 gains the "
              "re-run re-quote rule.")


if __name__ == "__main__":
    main()
