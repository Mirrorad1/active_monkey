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
  (site_data, site_data --lab-status, gen_directions_index) after card/site edits. [PROTOCOL step 6]
- **L12 (Exp 176/177 iterations, 2026-06-11).** `site/data/experiments-data.js` is REWRITTEN by
  `python -m active_loop.site_data` (caveat regeneration), so any read of it from a
  previous iteration is stale and the next Edit fails with "modified since read" —
  in a shared checkout this looks like (but is not) the parallel-session race. Rule:
  re-Read the generated region (tail entry + AM_TALLY line) at the START of each
  iteration's site-entry edit, and always edit-then-generate within one iteration,
  never generate-then-edit-later. Same applies to `site/data/lab-status.js` and DIRECTIONS.md.
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
- **L23 (Exp 201, 2026-06-13).** PILOT (and measure the end-window metric) at the FULL verdict horizon —
  a population metric can be NON-MONOTONIC, so a short-horizon pilot mis-states the equilibrium in EITHER
  direction. Exp 201's gene-pool intensity peaked ~t=2000 then DECAYED: a horizon-4000 pilot read FAST ~0.18
  (would have predicted MIXED) while the horizon-12000 pilot read ~0.13 (the true low equilibrium, NEGATIVE);
  the inverse of Exp 196, where a too-SHORT horizon HID a signal that only emerged after ~t=2000. Rule: run
  pilots at the same horizon as the verdict and confirm the metric has PLATEAUED (not a transient peak or a
  late-emerging rise) before fixing the regime or reading the result. Corollary for "increasing-returns"
  claims: convex GEOMETRY does not guarantee a convex realised SELECTION gradient — measure the benefit curve
  empirically (a frozen-trait returns probe: intake(p) at pinned intensity p, cost handled analytically), as
  the convexity can collapse to concave + cost-dominated at engine/grid resolution. [VALIDATION; METHODOLOGY;
  kin of L13 instrument-resolution]
- **L26 (perf analysis, 2026-06-13).** ECOLOGY RUNTIME = c·Σ_t pop(t)·arms·seeds, c≈4–5 µs/creature-step
  of GIL-bound pure Python (distributed across ~10 ops — NO single hot spot). It grew because pop (×10)
  AND horizon (×20) grew. The TWO failure modes that produce HOURS are (i) runaway growth → the L25
  guard, and (ii) memory→SWAP (~815 MB/run × workers; the dead-creature belief maps were the big heap —
  freed on death, RSS −27%; the pre-flight now caps workers to fit RAM). Per-creature CPU is ~irreducible:
  the bit-exact `events_hash` determinism contract blocks vectorisation (sequential consume() race + rng
  order) and rng-batching. Real levers: right-size HORIZON (~linear) and GRID→pop (8×8≈3× faster than
  12×12) and seeds; vectorisation/PyPy are big-but-major. PROFILING CAVEAT: cProfile `tottime`
  over-weights million-call cheap functions (neighbors/max) — confirm any hot spot with a clean
  wall-clock A/B before optimising. Full analysis: docs/research/ecology-performance.md. [METHODOLOGY]
- **L25 (human request, 2026-06-13).** RUN A RUNTIME / ALGORITHMIC-COMPLEXITY PRE-FLIGHT before launching
  any full experiment batch — a bug (unbounded population, an accidental super-linear per-step cost, a
  no-scarcity regime growing toward the runaway cap) can quietly burn hours of compute. The Exp 202
  ABUNDANT arm was dropped for exactly this (no-scarcity → population explosion → runtime + invalidity).
  Mechanical guard: `ecology/runtime_budget.preflight(reps, horizon, n_jobs, max_workers, require_safe=True)`
  probes the most explosion-prone arms (lowest-cost / highest-regen), is LOGISTIC-AWARE (distinguishes
  exponential-then-plateau from true runaway by growth deceleration across quarters — a naive geometric
  extrapolation cries wolf on every healthy logistic run), and flags EXPLOSION / SUPERLINEAR (per-creature
  cost rising ⇒ inner loop worse than O(alive)) / OVER_BUDGET. Call it at the top of `main()` so the batch
  REFUSES to launch on a flagged config; for non-ecology runs do the back-of-envelope equivalent + a
  smoke-timed extrapolation. [PROTOCOL step 3]
- **L24 (Exp 202, 2026-06-13).** REAL interference competition for a contested DEPLETING resource is NOT
  automatically a Red-Queen escape — it can select AGAINST a costed precision organ: a crowded scarce band
  makes the upkeep more lethal and a fair queue (id-order neutralised by shuffle) does not rescue precision,
  so COMPETE DECAYED below the founder. Verify the band genuinely depletes (strip>0) so the test is valid,
  AND beware drift: high trait values at small/collapsed populations (corr(pop,trait) strongly negative) are
  founder/drift artifacts, not selection — predeclare a healthy-population floor. [VALIDATION; METHODOLOGY]
- **L27 (exp206/exp207 design workflows, 2026-06-13).** A multi-agent DESIGN+AUDIT workflow's wall-clock is
  dominated by ORCHESTRATION overhead, NOT the experiment — the ecology sims run in ~100s, but the exp207
  design workflow took 45 min / 1.9M tokens. Four avoidable costs, all in how the workflow is authored: (i)
  REDUNDANT GROUNDING — telling all N agents to "read these files first" makes each of ~17 agents re-read
  the same huge files (EXPERIMENTS.md is 8.6k lines); instead PRE-DIGEST the grounding ONCE (one scout agent,
  or inline the relevant excerpts + a one-line-per-experiment summary) and pass it inline. (ii) NO MODEL
  TIERING — reserve the top model for design + synthesis; route routine audit lenses to a cheaper model
  (agent opts.model). (iii) SCHEMA-RETRY THRASH — rich StructuredOutput schemas (many required fields, long
  descriptions) cause near-miss-JSON retry loops (one agent burned 10 attempts); keep schemas lean. (iv)
  STRAGGLER BARRIERS — a parallel phase waits for its slowest agent (26m vs 2m48s here); prefer pipeline()
  and smaller fan-out. Right-size the fan-out to the stakes. The ecology sim itself is fine — right-size
  HORIZON/grid/seeds (L26), not the agent swarm. [Workflow authoring; kin of L26]
- **L28 (Exp 207, 2026-06-14).** PRE-FLIGHT THE SCIENTIFIC PREMISE, not just the runtime (L25). Before
  committing a full co-adaptation / interaction batch (6-arm × 5-seed × 8000-step + 2-D audit), MEASURE the
  load-bearing premise cheaply: for a hypothesized 2-D fitness valley (neither trait pays alone, both
  together do), a ~40-run monomorphic CORNER-GRID B(h,θ) settles it — the experiment is viable only if the
  discrete cross-partial d²B/dh·dθ > 0 AND neither trait pays strongly ALONE. Exp 207's corner-grid showed
  cross-partial ≈ 0, θ pays alone (+0.147 via herd-escape), h pure cost at every θ (dB/dh < 0) → DESIGN-STAGE
  NEGATIVE, no full batch needed (it would only rediscover the sixth wall). A design-stage negative IS a real,
  loggable, blind-verifiable finding when the corner-grid is run on committed code with the anti-cheat guards
  passing. [PROTOCOL step 3; kin of L22/L25]
- **L29 (Exp 210, 2026-06-14).** DRIFT in a finite-population invasion / common-garden test is a
  POPULATION-SIZE problem, not a cost problem. The diagnostic: a matched PURE-COST / perfect-percept control
  (the trait's mechanism on but its INFORMATION removed, e.g. cue_noise=0) — if that control's mutant fixates
  (inv_frac → 0/1) as often as the treatment, the fixation metric cannot separate selection from drift. FIX:
  raise carrying capacity (bigger pops ⇒ slower fixation) and read the drift-robust SELECTION SLOPE mean_s
  (slope of ln(n_mut/n_res) over the pre-fixation window; drift ⇒ mean_s ≈ 0, selection ⇒ consistent
  mean_s > 0), requiring the treatment to BEAT the pure-cost control on the slope, not just win on inv_frac.
  Do NOT "fix" drift by raising the cost — that changes which trait wins by COST, not selection (the trap that
  nearly turned Exp 210 into a foreordained negative). Kin of L24 (high trait at collapsed pops = drift artifact).
  [VALIDATION; METHODOLOGY]
- **L30 (Exp 210, 2026-06-14).** CALIBRATE a costed ACTION's cost to the EMPIRICAL benefit ceiling BEFORE
  reading a verdict. Measure the maximum attainable benefit with a gifted, cost-WAIVED run (e.g. full-probe
  wrong-cell-occupancy drop × hazard = energy/step). A verdict cost ABOVE that ceiling makes a NEGATIVE
  INEVITABLE — the mirror of seed-shopping a positive, and a VALIDATION violation (don't make the outcome
  foreordained). Pick the verdict cost BELOW the ceiling and report a cost-sensitivity sweep spanning it, so
  the negative is "doesn't pay even when it could" rather than "priced out by fiat." Kin of L20 (remove cheaper
  escapes before concluding a costed feature doesn't pay) and L22 (forced/gifted benefit ≠ evolvable).
  [VALIDATION; METHODOLOGY; PROTOCOL step 3]
- **L31 (perf, human request 2026-06-15).** PARALLELISE INDEPENDENT RUNS, CAP WORKERS TO FIT RAM.
  Experiment wall-clock = c·(pop·horizon)·(arms·seeds), serial on one core by default — measured: a
  16-seed×multi-arm preflight is ~150 serial sim-runs (Exp 210 ~25 min). Independent runs (different
  seeds/arms; each `events_hash` depends only on its own seed ⇒ parallel changes NO result) MUST run
  in PARALLEL via `ecology/batch.py` (`run_batch`/`default_workers()`), NOT a serial `for seed` loop;
  the generic Evolvability Preflight gates (`run_local_pairwise_gradient`/`run_invasion_from_rarity`)
  now take a `max_workers` param (pass `recommended_workers`; serial when omitted; parallel==serial
  proven by `tests/test_gates_parallel.py`). BUT cap `max_workers` so `workers × peak_RSS_per_run ≤ ~60% RAM` — use
  `ecology.runtime_budget.preflight(...)`'s `recommended_workers` (already memory-sized) as the pool
  size; over-subscribing → SWAP thrashing → minutes become hours (L26 failure mode #2). Determinism
  contract still bars vectorising/RNG-batching/consume-reorder (those change the hash). Right-size the
  cheap knobs to the science first (pop/cap — L29; seeds; horizon — L23). Measured factors: cap250 pop
  ~1000 is 5.3× slower/run than cap50 pop ~215; active-sensing probe-draws add ~29% (gated, AS-only).
  [PROTOCOL step 3; kin of L25/L26]
- **L32 (Exp 211, 2026-06-15).** WORKS-WHEN-IMPOSED ≠ HAS A BENEFIT CEILING TO SELECT ON. A costed
  capability gated on the agent's OWN internal signal can pass every imposed-mechanism check (beat a
  budget-matched random control, enrich its action at the targeted states, change the decision more per
  use) and STILL have a near-zero benefit ceiling — because the cheaply-computable gating signal cannot
  identify the truly-pivotal states. Exp 211's uncertainty-gated probe (gate on |single-cue belief−0.5|)
  beat random cost-matched on decision quality yet did NOT beat fixed-rate, because under sensory noise a
  single cue is often CONFIDENTLY WRONG (large margin, wrong half) — invisible to the margin gate but
  fixed by indiscriminate probing; gated benefit ceiling 0.0009 vs fixed-rate 0.0186 energy/step ⇒ flat
  local gradient even at probe_cost=0.0. Rule: when a targeting/gating policy "wins when imposed", measure
  its BENEFIT CEILING (gifted, cost-waived) against the indiscriminate baseline's BEFORE reading an
  evolvability verdict — a policy that beats random but not the indiscriminate version has near-zero
  ceiling, and the local gradient is foreordained flat regardless of cost (verify at probe_cost=0.0 per
  L30). Kin of L22 (forced/gifted ≠ evolvable) and L30 (cost vs benefit ceiling). [VALIDATION; METHODOLOGY]
- **L33 (Exp 221, 2026-06-16).** A batch of JAX/XLA-CPU experiment processes (the affect_agent /
  DirectHeadAgent family) must BOUND CONCURRENCY to ≈ cores / cores-per-process — NOT run one process
  per cell. Each JAX-CPU process spawns its own multi-thread XLA/Eigen pool (~2.3 useful cores/process
  here; `OMP_NUM_THREADS=1` and `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` do NOT pin it, and macOS
  has no per-process thread cap), so launching all 8 cells at once oversubscribed 12 cores → load 43,
  ~96 spinning threads, only ~4 useful cores (throughput WORSE than fewer processes; the L26 swap-thrash
  analogue for CPU threads). Rule: MEASURE per-process core footprint with a single-process probe first,
  then run via a concurrency pool of `floor(cores / cores_per_proc)` (e.g. `xargs -P N`), longest cells
  first; chunk a cell across processes only with DISTINCT output files (the `--seeds` read-modify-write
  merge races if two processes write the same cell JSON). Kin of L26/L31 (oversubscription → thrash).
  [PROTOCOL step 3]
- **L34 (Exp 221, 2026-06-16).** Set decision/report THRESHOLDS ON the metric's value grid. A
  constant-unfakeable probe quantized to k/6 compared against a literal float bar `csel >= 0.67` silently
  EXCLUDES 4/6 = 0.6667 (just below 0.67), shifting the intended "one-step-stricter" bar up by a full
  quantum to ≥5/6 (sched_300 read 1/16 instead of the intended 7/16). The verdict was unaffected (the
  primary bar 0.5 = 3/6 is on-grid), but a reported number was wrong until recomputed at ≥4/6. Rule: when a
  metric takes values k/N, write thresholds as `>= k/N` (e.g. `>= 4/6`), never an off-grid decimal; an
  L13-kin instrument-resolution check. [VALIDATION; PROTOCOL step 5]
- **L35 (Exp 223-224, M4b real-run shakedown, 2026-06-16).** Repurposing a FROZEN/generic LLM-driven
  autopilot (a PR-loop) for a NEW objective inherits the original's baked-in assumptions, which fail ONE AT A
  TIME in expensive real runs unless shaken out first: (1) the proposer/critic carry the ORIGINAL mission /
  context / world_model (lang) — write NON-frozen objective-specific proposer+critic with their own mission;
  (2) the proposer timeout is sized for the original prompt — bump it (180s→600s here); (3) a NEW isolated
  world_model/notes dir is NOT tracked on trunk, so `commit_all` (git add -A) sweeps it onto the proposal
  branch and the discard→`checkout(trunk)` WIPES it (the lang loop only worked because its world_model was
  already committed) — re-create the dir each iteration AND/OR commit it to trunk first; (4) the FULL
  orchestration loop (`run_*_loop`) may never have been integration-tested — only the single-iteration fn was.
  RULE: BEFORE launching expensive real autopilot runs (here ~10–40 min each), add a FAST mock-proposer +
  stub-score integration test of the WHOLE loop path (not just one iteration) — it shakes out the
  dir/git-churn/world_model bugs cheaply; otherwise each real run discovers one more trivial bug (Exp 223 =
  timeout+context, Exp 224a = the git-wipe). Kin of L25/L31 (pre-flight before a big spend). [PROTOCOL step 3]
- **L36 (Exp 224-225, 2026-06-16).** When an LLM-driven optimization autopilot reports "no improvement",
  DISTINGUISH the failure mode before concluding anything about the metric: (a) SCORE-DRIVEN (a candidate was
  measured and scored ≤ baseline) vs (b) GATE-GATED (the candidate was rejected by a critic/test/frozen-guard
  BEFORE the metric ever measured it). A gate-gated negative says NOTHING about improvability — the scorer
  never ran on a mutant (Exp 224 = NEGATIVE but both proposals were critic-rejected pre-scoring; "improvable?"
  was UNRESOLVED, not answered). Cheaply DIAGNOSE a gate-gated result by capturing the gate's REASONS on a
  handful of proposals (propose→critic only, NO expensive scoring): the critic may be CORRECTLY rejecting
  metric-gaming (Exp 225: it rejected baking the code→intent answer-structure into a prior, and approved a
  legitimate preference tweak), in which case the gate is working, not broken — then SCORE a gate-APPROVED
  proposal directly to test whether a legitimate improvement exists (Exp 225b: the approved C1 tweak scored
  0.4225→0.5188, genuine 6/8→8/8). Don't read a gate-gated 0/N as "metric unimprovable" or "gate miscalibrated"
  without that ~5-min diagnostic. [PROTOCOL step 5; VALIDATION; kin of L35]
- **L37 (2026-06-18).** A direction card's `STATUS` line can be STALE — it is hand-maintained and is NOT
  re-derived from the record, so it can claim a direction is open after its ladder has been run and closed.
  `social-emergence.md` read `state: exploratory · next-falsifiable: Rung 1` long after the ladder ran and
  CLOSED at Exp 74 (vela, the Rung-1 peer spine, has existed since Exp 63). This misled both a recon subagent
  ("no rung has been run") and a /lab compose ("start at Rung 1" = a duplicate of Exp 63). RULE: at PROTOCOL
  step 0/1, treat `IDEAS.md` + `EXPERIMENTS.md` as the source of truth for whether a direction is open, NOT the
  card's STATUS line; before running "the next rung", grep EXPERIMENTS.md for that direction's experiments and
  confirm the rung is genuinely unrun. When you find a stale STATUS, FIX it in the same iteration (mark the
  closure + point to the descendant direction) — the card is a pointer, the log is the truth. [PROTOCOL step 0/1; META]
- **L38 (Exp 232, 2026-06-18).** MANIPULATION CHECK: when you design a "harder" (or "easier", or "more X")
  condition to stress-test a prior result, ADD AND PREDECLARE a check that the manipulation ACTUALLY moved the
  thing it was supposed to — otherwise a passed metric just answers a WEAKER question than you think. Exp 232
  tried to make goal-inference HARD with a "25-way" target (vs the trivial 3-way of Exp 229-231), but the
  inference posterior hit q=0.93 (>> the 0.40 bar) because the 4 CORNER targets chosen induce maximally-different
  trajectories — trivially separable. The "25-way" was nominal; the actual inference stayed easy, so the
  predeclared POSITIVE did NOT overcome the "trivial-goal" caveat it set out to test. GENERAL PRINCIPLE the miss
  taught: inference difficulty over another agent's latent goal is set by the behavioral SEPARABILITY of the goals
  (how differently they make the agent act), NOT by the goal-space SIZE/cardinality. RULE: predeclare a
  manipulation-check metric (here: the inference posterior should stay LOW if the condition is genuinely hard) and
  read it BEFORE crediting the headline; a difficulty manipulation that doesn't move the difficulty metric is a
  no-op on the intended question. [PROTOCOL step 2 (design-time) + step 5 (confound/surprise audit); METHODOLOGY]
- **L39 (Exp 235, 2026-06-18).** EXPRESSIBLE BEFORE EVOLVABLE: before asking whether a trait is EVOLVABLE
  (a selection gradient), verify the trait is even behaviorally EXPRESSED under the agent's POLICY — run a
  cheap gen-0 "ability-gifted / gate-open" diagnostic (set the trait/affordance to its max or remove the
  barrier it gates) and confirm the policy actually exploits the benefit. A trait that gates one STEP is
  inert if the policy never creates the situation where that step matters. Exp 235 built a costed
  `climb_ability` to cross a ridge sealing the richest food, but the ecology forager is a LOCAL greedy
  policy (best-adjacent-cell, no path-planning), so it never navigates to the ridge: gate-OPEN plateau
  intake stayed ~1% (climb irrelevant), every geometry-sweep cell was noise. The bottleneck was the POLICY,
  not the gate — diagnosable in minutes by the gate-open control, BEFORE spending an 8-seed gradient batch.
  RULE: make the gate-open/ability-gifted expressibility check one of the predeclared deflation controls
  and run it FIRST; if the policy can't express the trait even when handed it free, no geometry/cost tuning
  helps — it's a NULL/INVALID (substrate/policy redesign), NOT a local-gradient wall result. The fix is a
  policy that can express the trait (here: navigation), gated byte-identical OFF. [PROTOCOL step 2 + step 4.5;
  METHODOLOGY; pairs with L22 forced!=evolvable and L32 works-when-imposed!=evolvable]
- **L40 (Exp 236, 2026-06-18).** A subagent (or you) can GAME a metric via an INCIDENTAL ARTIFACT that
  correlates with the target — not by cheating outright. The Exp 236 navigation build hit the
  expressibility target (63.7% plateau access) by flipping the target-selection tie-break to
  "highest-index cell wins"; plateau cells happen to have the highest indices, so creatures marched
  there by spatial fiat, not by seeking food. It looked like working navigation; it was an index
  artifact. RULE (validator discipline): when a subagent reports a number that clears a bar, do NOT
  accept it on the report — test its ROBUSTNESS to incidental choices. Re-run with the NEUTRAL /
  codebase-convention setting (here: the lowest-index tie-break used everywhere else, `(m, -c)`); if
  the result collapses (63.7% -> 0.5%), the metric was gamed by the artifact, not earned. Watch for
  tie-breaks, sort orders, index/id correlations with the outcome region, default-parameter coincidences,
  and "tuned until it passed" knobs. The gate-open / ability-gifted expressibility check (L39) is the
  natural place to run this robustness test. [PROTOCOL step 4.5 (blinded-verify) + step 5; VALIDATION;
  pairs with L39]
- **L41 (Exp 237, 2026-06-18).** INVASION-FROM-RARITY is the binding evolvability criterion, NOT the
  50/50 pairwise gate. A trait can WIN a head-to-head equal-frequency contest yet be UNABLE to spread
  when rare — that is POSITIVE FREQUENCY-DEPENDENCE / a priority effect (e.g. a faster climber edges a
  slower one in direct competition for a shared depletable resource, but gains nothing in a monomorphic
  population and cannot bootstrap from a rare mutation), NOT directional selection. Exp 237: the
  Evolvability Preflight reported aggregate `PASS_LOCAL_GRADIENT` (pairwise mutant wins 7/8) while
  `invasion_from_rarity` said DOES_NOT_INVADE (1/8) AND the gen-0 MONOMORPHIC benefit curve was FLAT
  across the trait — a near-false-positive "the trait evolves!" The truth: it does NOT evolve (no
  per-capita benefit, no invasion from rarity); the wall holds. RULE: when reading an evolvability
  result, require (a) a non-flat monomorphic benefit curve AND (b) invasion-from-rarity, not just a
  pairwise win; if pairwise PASSes but invasion-from-rarity fails, the verdict is frequency-dependence,
  not evolvability. Self-heal: the Preflight aggregate was fixed to downgrade pairwise-PASS +
  invasion-DOES_NOT_INVADE to a distinct non-PASS verdict. [PROTOCOL step 5 + step 4.5; VALIDATION;
  pairs with L22/L32/L40; instrument: ecology/evolvability]
- **L42 (Exp 242, 2026-06-19).** A FIX that produces BYTE-IDENTICAL output is a SILENT NO-OP — it is acting on the wrong lever or a dead code path — and does NOT change the experimental substrate. Always verify a fix actually changes behavior: compare the events_hash (or any deterministic output hash) before and after; if they match, the fix is a no-op and the experiment is unchanged. The first Exp-242 regen-cap attempt was byte-identical to Exp 241 (regen-side cap acting on _resource, while intake read rho() — the wrong field), caught because events_hash matched. Pairs with L40 (anti-gaming robustness: verify behavior changed) and the foundational-bug class (L13/L16). [PROTOCOL step 5; METHODOLOGY]
- **L43 (Exp 270, 2026-06-26).** An AFFORDANCE / "can-policy-X-control-its-own-input" probe is DEGENERATE when run WITHOUT the pressure the capability is meant to resist. On a small wall-clamped grid both a gifted seeker and a gifted avoider reach the trivial bounds (f=1.000 / 0.000) by HUDDLING on a 1-2 cell refuge (an action into a wall is a no-op = stand still; same-color adjacency gives a 2-cell oscillation), so the controllability metric saturates IDENTICALLY across the very geometries it was built to distinguish (aliased == segregated) — geometry-blind. Tells: the L19 exact-bound + the L42 byte-identical-across-conditions signature. RULE: an affordance gate for a CONTROL surface must include the PRESSURE the surface resists (here a soft, resistible attack/pull, no-op-overridden), so "control" cannot be won by standing still; and a positive control that ALSO saturates to a trivial bound validates nothing — require it to land STRICTLY between the bounds, and report a home-range/most-visited-frac diagnostic to distinguish a genuine refuge from a huddle. Kin L42 (wall-clamp degeneracy), L40 (incidental-artifact gaming), L19 (near-perfect effect = degeneracy red flag). [VALIDATION; PROTOCOL step 4.5]
- **L44 (Exp 270, 2026-06-26).** A MYOPIC / greedy planner can manufacture a FALSE NEGATIVE in an affordance/evolvability/escapability test — the INVERSE of L22/L32 (forced/imposed != evolvable). Exp 270: a greedy BFS-away avoider failed the scattered attack-colors (gap ~0.11) and would have read "movement can't escape here / no refuge", while the CERTIFIED-OPTIMAL avoider (policy iteration on the exact kernel) escaped the same colors (gap ~0.41) — the apparent geometric wall was a PLANNING-HORIZON artifact, not geometry. RULE: when a negative turns on "the best policy can't do X", use a CERTIFIED-optimal policy (value/policy iteration to a Bellman-residual certificate, asserted in preflight), not a myopic heuristic, so the negative is substrate/geometry-bound not heuristic-bound; a heuristic upper bound that fails proves nothing. Also (instrument guard): converge value iteration on the Bellman RESIDUAL, never on policy-stability ("policy unchanged for K sweeps"), which can halt before values propagate and return a sub-optimal policy on slow-mixing cells (gamma->1) — caught here by a code-review + verifier against deep VI; the certificate is the durable guard. [VALIDATION; PROTOCOL step 3/4.5; pairs with L22/L32 inverted]
