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
