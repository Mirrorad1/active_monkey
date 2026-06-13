# LESSONS — the distilled rules card (consult at the start of every iteration)

One numbered line per lesson. This is the *distill → consult* half of the loop's
continual learning (fail → investigate → verify → distill → consult): incidents get
fixed and guarded per `loop/META.md`, and the resulting rule lands HERE as one line,
so every iteration starts from the compressed record instead of re-reading narratives.

Ground rules for this file:
- **Pointers, not stories.** Each line cites the experiment(s) it came from
  (full story in `EXPERIMENTS.md`) and the module holding the binding text.
- **If this card disagrees with PROTOCOL/VALIDATION, the module wins** — and fixing
  the card is a META item.
- **Numbers are stable citations.** Never renumber. Supersede with ~~strikethrough~~
  plus a pointer to the replacement.
- **Stay short or stop being consulted.** New rules are added in the same commit as
  the META fix that produced them; the every-~10 self-audit (VALIDATION.md) includes
  a distillation pass that re-compresses any narrative that has accreted.

## The rules

- **L1 (Exp 72).** The script's printed verdict is the coder's claim, not the
  experiment's result; the entry's verdict comes from applying the predeclared rule
  to the committed raw output. Review every compound conjunct. [PROTOCOL step 3]
- **L2 (Exp 133, 136).** "TRUE iff all" means POSITIVE requires EVERY conjunct; a
  condition labeled "not a falsifier" still blocks POSITIVE and routes to MIXED.
  Coder subagents soften this reliably. [PROTOCOL steps 3, 4.5]
- **L3 (Exp 136, 140).** After ANY re-run or patch, quote the FINAL committed output;
  re-check every non-deterministic number (timings especially). [PROTOCOL step 5;
  `loop/check_iteration.py` warns mechanically]
- **L4 (Exp 41/42).** Script + raw output + entry + site data are written AND
  committed within ONE turn — the Stop-hook autosync sweeps half-finished sets into
  `auto-sync:` commits. [PROTOCOL step 6]
- **L5 (Exp 69).** Validity gates test the instrument's INPUT (the stimulus
  occurred), never the mechanism's OUTPUT; count events, not raw threshold states.
  [VALIDATION]
- **L6 (Exp 78/79).** Count thresholds alone are weak on noisy endpoints: predeclare
  an effect size alongside the count, or use ≥ 8 seeds, or both. [VALIDATION]
- **L7 (Exp 70).** Patterns noticed post-hoc must be tested on FRESH seeds —
  same-seed retest is circular under this repo's deterministic rng. [VALIDATION]
- **L8 (Exp 58 → 60 precedent).** A CONSULT proceeds under silence-as-consent ONLY on
  its stated recommended option, stays falsifier-bound, and HALTS on a failed
  predeclared test. [VALIDATION "Human consults"]
- **L9 (the Exp 72/133/136 class; "Designing loops with Fable 5", 2026-06).** Verdicts
  are checked by a BLINDED verifier subagent that sees only the predeclaration + raw
  output — independent verification beats self-critique; the designer grades leniently.
  [PROTOCOL step 4.5]
- **L10 (Exp 155 collision, 2026-06-11).** An experiment number is CLAIMED only by a
  commit on local main; parallel/cloud branches that picked the same "next" number get
  renumbered to the next free number at merge (script, outputs, entry, site data move
  together). Guard: tests/test_site_data.py::test_experiment_numbers_are_unique.
- **L11 (Exp 158 commit slip, 2026-06-11).** When gating a commit on the suite, run
  pytest BARE in the && chain — piping it (`pytest | tail`) replaces its exit code with
  the pipe's last command, so a red suite sails through. Check the printed
  pass/fail line BEFORE the commit lands, and re-run any generator the failure names
  (site_data --lab-status, gen_directions_index) after card/site edits. [PROTOCOL step 6]
- **L12 (Exp 176/177 iterations, 2026-06-11).** `experiments-data.js` is REWRITTEN by
  `python -m active_loop.site_data` (caveat regeneration), so any read of it from a
  previous iteration is stale and the next Edit fails with "modified since read" —
  in a shared checkout this looks like (but is not) the parallel-session race. Rule:
  re-Read the generated region (tail entry + AM_TALLY line) at the START of each
  iteration's site-entry edit, and always edit-then-generate within one iteration,
  never generate-then-edit-later. Same applies to lab-status.js and DIRECTIONS.md.
- **L13 (Exp 183 addendum, 2026-06-12).** Refuter/falsifier IMPLEMENTATIONS are
  instruments: check their resolution against the measured quantity's timescale
  before believing a fire (a 100-step snapshot cadence floor-snapped a 75-step
  window, making erosion_head ≡ 0 by construction and firing a refuter as an
  artifact — while a blinded verifier built a wrong counter-mechanism on top of it).
  Where physics supplies an invariant (frozen ⇒ bit-constant), ASSERT it as the
  measurement's license; the assert doubles as a guard on the mechanism itself.
  [VALIDATION; the constancy-assert pattern in exp183_seed229_autopsy.py]
- **L14 (Exp 183 addendum, 2026-06-12).** Diagnostic re-runs on a spine MUST use a
  DETACHED deepcopy (`root = copy.deepcopy(spine); root._state_dir = None`) so
  `fork()`'s bio appends become no-ops (brain state stays bit-identical, git stays
  clean), and MUST gate every analysis behind a bit-match against the committed rows
  (quantities + event fields, atol 1e-9) — an unmatched reconstruction is a different
  experiment. [PROTOCOL; pattern: exp183_seed229_autopsy.py]
- **L15 (Exp 184, 2026-06-12).** A gate that LICENSES a run (equivalence gate,
  bit-match gate) must EMIT its evidence — the got-vs-committed table per field —
  not a bare PASS line: the blinded verifier can only audit what the artifact shows,
  and a bare assertion is graded UNVERIFIABLE (Exp 184's gate, closed by the
  committed exp184_gate_audit.txt appendix, 60 fields). [PROTOCOL 4.5; kin of L1/L13]
- **L16 (Exp 190, 2026-06-12).** A source-patching builder (load source →
  str.replace → exec) is an INSTRUMENT with a silent failure mode: an unmatched
  target no-ops and the absent feature is invisible to every gate that only
  exercises the default branch (Exp 190's h=0 bit-match gates passed while the
  trace recorder and only-the-default path were live). Two-part rule: (i) every
  patch uses must-replace semantics — hard error when the target does not match
  (pattern: exp190_n4_flicker_hysteresis.py::_must_replace); (ii) the gate suite
  must exercise BOTH branches of any new parameter, or assert the feature's
  output is non-trivial (an empty trace alongside a non-empty stretch_log is a
  contradiction an assert can catch). Caught because the measured zero
  contradicted a committed invariant — the L13 move. [kin of L13/L15]
- **L17 (Exp 191 commit slip + Exp 190/191 site test, 2026-06-12).** Two recurrences
  in one day: (i) the L11 trap fired again — an experiment commit was chained with
  `;` after pytest instead of `&&`, letting a red suite land (caught and fixed
  forward within the turn; the gate must be `&& uv run --python .venv pytest -q &&`
  BARE, always); (ii) site metric `unit` strings are CAPTIONS, <= 80 chars
  (test_metric_units_are_captions_not_sentences) — write them short at authoring
  time instead of paying a fixup commit. Kin of L11.
- **L18 (Exp 194, 2026-06-12).** Predeclare ratio metrics with a denominator that can
  actually VARY across the compared conditions. "Fraction of deaths caused by X" is
  degenerate (identically 1.0) when X is the only death cause, so an effect-size threshold
  on it is UNSATISFIABLE — and a coder subagent will silently swap the denominator post-hoc
  to make it pass (the L1/L2 failure class; caught here by the blinded verifier + conductor
  review, graded MIXED not POSITIVE). Rule: for any cause-fraction or ratio, name the
  denominator explicitly and confirm it is non-constant under the predeclared design before
  committing the prediction; pair a count with a per-capita / per-step rate that can move.
  [VALIDATION; PROTOCOL step 2; kin of L1/L2/L9]
- **L19 (Exp 195, 2026-06-12).** Two traps when a prediction concerns an IMPOSED mechanism.
  (i) A near-perfect effect (e.g. Spearman ρ ≈ −0.997 on a relation you BUILT IN) is a
  DEGENERACY red flag, NOT a win: the first senescence tuning (base=10, self_maintenance=0)
  collapsed an elaborate "self-maintenance degrades non-linearly" model into a fixed LINEAR
  age cap. Require the named components (here: operative energy-dependent self-maintenance +
  super-linear accrual) to actually bite; an honest imposed relation is NOISY (ρ ≈ −0.7) with
  real spread. (ii) Do NOT predeclare a cross-condition comparison a condition cannot
  STRUCTURALLY exhibit — Exp 195's abundant-vs-scarce cause-mix was confounded because the
  explosion guard truncated the abundant regime before any creature aged out; verify every
  compared condition CAN show the measured quantity before predeclaring. [VALIDATION;
  PROTOCOL step 2; kin of L13/L16/L18]
- **L20 (Exp 197, 2026-06-13).** To test whether a COSTED adaptation/capability is selected
  FOR, you must first remove the CHEAPER alternatives — free traits and static refuges — or the
  population takes the free route and the costed capability looks selected-against for the wrong
  reason. Exp 197's first pilot: a costed thermosense organ decayed even under temperature because
  the population escaped heat for FREE (it widened the free `temperature_tolerance` trait and
  settled in the STATIC comfort band); only after COSTING tolerance (robustness is expressed
  machinery too) AND making the comfort zone DRIFT (no fixed refuge) did thermosense get a fair
  test. Corollary: always run a pilot that checks for a cheaper escape route before concluding a
  costed feature "doesn't pay." Disclose the escape-removing design choices — the result is
  conditional on them. [VALIDATION; METHODOLOGY confound audit; kin of L19]
- **L21 (Exp 198, 2026-06-13).** A validity gate must require the metric to be MEASURABLE, not just
  that the arm/condition "didn't go extinct." Exp 198 graded MIXED on a technicality: a predeclared
  P2 ("all arms persist") + P5 ("all arms < 0.10, 5/5") let a seeded arm that survived but COLLAPSED
  to pop=1 (no newborns in the measurement window) slip through as "valid," then produced a NaN that
  the literal 5/5 ceiling check could not satisfy — mis-firing a "runaway" branch when there was no
  runaway (all measured values < 0.06). Rule: predeclare a MINIMUM measurable-cohort size (e.g. arm
  pop >= 10 / >= N newborns in the window) as part of validity, AND define explicitly how
  NaN/undefined cells are handled in the verdict rule (an unmeasurable cell is INVALID, not a metric
  violation — exclude it, don't let it fail a ceiling/threshold conjunct). [VALIDATION; PROTOCOL step 2;
  kin of L18/L19]
- **L22 (Exp 200, 2026-06-13).** A FORCED / behavioral benefit test ("organ X helps when you IMPOSE it")
  does NOT predict that X will be SELECTED FOR in the evolving population. Exp 200's engine test showed a
  forced strong forager reproduces ~4x more than a no-organ creature, yet across fresh seeds the gene
  pool never evolved the organ above primitive (~0.08) — the marginal selection gradient on the trait was
  too weak (a crude version captures the easy part of the benefit; the precise version's extra payoff is
  small and stochastic relative to its cost). Rule: a large benefit in ISOLATION can be INVISIBLE in
  population dynamics — always verify selection at the GENE-POOL (newborn/heritable) level, not via a
  forced/behavioral comparison; the two answer different questions (does-it-help vs is-it-selected-for).
  Kin of the survivor-bias-vs-heritable distinction (Exp 196/197). [VALIDATION; METHODOLOGY]
