# Adversarial review and repairs — 2026-10-27

**Assessment:** no mathematical or numerical error changes the result. The
substantial-growth prediction remains **NEGATIVE: 1/16, with 12/16 required**.
The contribution is a reproducible methods case study; neither Lean verification
nor this review establishes novelty, journal acceptance, or an adaptive mechanism.

Two fresh AI reviewers separately examined mathematics/formalization and empirical
evidence. They received the source and evidence, so this was source-informed
adversarial review, not a blinded experiment verdict. Both completed their reviews.
Their attempted repair turns then failed at a service usage limit; the primary
agent implemented and checked the repairs. No independent post-repair human review
is claimed.

## Findings and disposition

| Finding | Severity and effect | Repair |
|---|---|---|
| Newborn means were retained only as summaries; the original per-step table lacked child traits. | P2, reproducibility gap in the central endpoint sensitivity. No incorrect mean detected. | An additive same-seed reproduction retains 168,650 individual birth records, including child/parent IDs, birth time, and both focal traits. Every newborn mean is reconstructed exactly. |
| Engine event hashes omit focal thermosense traits. | P2, weak observer guard; source review found no actual perturbation. | Watched/bare excluded pilots now compare every creature's full genotype, including dead creatures, and engine RNG state as well as event hashes. A deliberate neutral-trait corruption escapes the old hash and is rejected by the new guard. |
| “48 seeds” overstated the independent seed count. | P3, wording. Sixteen seed values are reused in three conditions. | Manuscript now says 48 runs across 16 shared seed blocks and discloses later stream divergence. Descriptive contrasts are unaffected; future uncertainty calculations should retain pairing. |
| Lean audit inspected only namespace-prefixed theorem declarations. | P3, guard scope; no current admitted proof found. | Audit uses defining-module ownership and checks all safe logical declarations, including definitions and opaque proofs outside the namespace. Four deliberately invalid controls are rejected. |
| Terminal living census and pooled recent births use different time aggregation. | Interpretation limit. | Added a clearly post-hoc matched-window comparison, retaining the original verdict and avoiding a causal survivorship claim. |
| The draft omitted substantial AI involvement. | Submission transparency. | Manuscript now discloses AI question selection, code, analysis, proofs, writing, and review; human review remains pending. |

## Evidence for the repairs

`experiments/exp283_adversarial_audit.py` leaves all originally executed sources and
raw artifacts untouched. It runs the same fixed 48 conditions/seeds using the
original runner with an observation-only birth recorder. It checks all original
rows and all summary fields except runtime for exact equality. This is reproduction,
not a new seed cohort or confirmation. Runtime was 37.5312 seconds on the existing
environment; 144,000 rows and all 48 event hashes matched. All original source
hashes still match.

Raw outputs are `experiments/outputs/exp283_adversarial_audit.json`, `.txt`, and
`exp283_adversarial_audit_births/*.json.gz`. Each birth file includes its column
schema, original-trajectory hash, and reconstruction checks. The summary hashes
every birth file and records the audit source/test hashes and dependency versions.
Existing output files, directories, and dangling symlinks are rejected. A failed
run retains its partial directory and cannot overwrite evidence on retry.

The existing background auto-sync committed the completed additive script/raw files
as `0c7a3c6` before this review, tests, and log addendum were committed. The final
review commit therefore reconciles that snapshot rather than claiming atomic
script/raw/log publication. The shared-checkout auto-sync behavior is not fixed here.

Cost-only endpoint sensitivity, averaged equally over the sixteen seeds:

| Statistic | Mean | Seeds at least .02 |
|---|---:|---:|
| Registered terminal living census | .0159096311 | 1/16 |
| Living census averaged over the final 500 transitions | .0161102025 | 1/16 |
| Same-window living census weighted by birth count | .0161274885 | 1/16 |
| All newborns during the final 500 steps | .0295825351 | 16/16 |

Row `t` is the after-census for births stamped `t-1`; the window is birth times
`[2500,3000)`, hence rows `2501..3000`. Time alignment does not remove the descriptive
gap in this data. It does not isolate survival, parental selection, or a direct
trait effect.

The strengthened Lean audit passes for 60 safe declarations, including 47 theorem
declarations. Sixteen unsafe compiler runtime declarations are excluded explicitly;
they cannot enter safe Lean proofs. Negative controls cover a local axiom and
admitted definitions, opaque proofs, and theorems outside the usual namespace.
See `lean/check_audit.py` and `lean/verification.txt`.

Validation: 41 focused Python tests pass (34 original plus 7 additive guards).
The saved LaTeX source compiles through the built-in editor compiler, including a
five-page-limit assertion. The previously exported standalone PDF is intentionally
not regenerated. No FROZEN path or original scientific source was changed.

## Limits that remain

- One simulator, trait, mutation scale, and finite horizon; no new general mechanism.
- The primary prediction failed. Arbitrary declared case-study size bars are not
  universal adaptation criteria or a statistical significance test.
- Realized Price selection includes association, drift, linkage, and demographic
  effects. Thermal-minus-cost differences are descriptive accounting contrasts.
- “Cost-only” charges upkeep only above .05, not for every positive trait value.
- Birth records reconstruct newborn endpoints and parent-child increments, but
  are not a full per-step individual survival ledger. The engine is not formally
  verified; the strengthened observer comparison is a finite pilot check.
- Formal mathematics is established theory. Publication interest rests on the
  measurement application and reusable artifact. Human domain review is still needed.

## Durable prevention

Prevented by: `tests/test_exp283_adversarial_audit.py` (focal corruption, RNG advance,
window/count mapping, and evidence overwrite); `lean/check_audit.py` (four invalid
proof controls); the manuscript's page-limit assertion; and the reproduction guide's
explicit raw-statistic and observer-coverage requirements.

Claudeception evaluation found reusable, non-obvious lessons: an event hash is only
as complete as its serialized fields; aggregate rows may omit a headline statistic's
inputs; Lean declaration namespaces do not identify package ownership, and unsafe
compiler specializations differ from logical axioms. The verified guards and
project guides above preserve these lessons. A separate global skill would duplicate
them, so none was installed. External issue/PR publication is not part of this local
review; this document is the durable local issue and resolution record.
