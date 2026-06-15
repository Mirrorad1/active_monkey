# The local selection gradient is the binding barrier — from costed senses to costed cognition

**Status:** SYNTHESIS (closes the hidden-state-memory direction CLOSED-NEGATIVE, 2026-06-14, on
the human's word "C"). Ties together three pieces of work: the **Evolvability Preflight** framework
(the instrument), the **sense-evolution** sub-arc (Exp 199–207), and **Phase 3 hidden-state memory**
(memory_horizon + belief_persistence). Re-opens only on an explicit human word.

**Scope:** a research note spanning `loop/directions/population-ecology.md` (closed-negative) and
`loop/directions/hidden-state-memory.md` (closed-negative). Companion docs:
`docs/research/sense-axis-organ-evolution.md` (the sense arc, experiment-by-experiment) and
`docs/evolvability_preflight.md` (the framework's developer guide). PRs: #46 (framework), #49
(generic binding gate), #50 (Phase 3).

---

## 1. The one binding question

Across this whole line the decisive question for whether a costed, heritable capability can EVOLVE
is **not** "is it useful?" but:

> Does a small heritable step from the resident value (`h → h + ε`) win more reproductive success
> in a fair common garden — i.e. is the **local selection gradient at the resident** positive?

A capability can be genuinely useful when **gifted** (handed to an agent for free, or installed at
high intensity), can define a **monomorphic optimum** above the resident, can leave the population
**surviving**, and can have a **controller** that exploits it — and STILL not evolve, because none
of those is the local gradient. Those four "necessary but not sufficient" traps were each
demonstrated empirically (the L22/L28 lessons). Only the local gradient decides.

## 2. The instrument: Evolvability Preflight

`ecology/evolvability/` turns that question into a cheap, reusable, trait-agnostic battery that runs
**before** any expensive full-evolution batch (PRs #46/#49). Its gates separate the traps:
gifted-benefit (useful-when-free), monomorphic-sweep (global optimum), **local-pairwise-gradient
(the binding gate)**, invasion-from-rarity (the stringent adaptive-dynamics test), controller
cross-partial, and a null-guard anti-cheat battery. Verdicts are deliberately asymmetric — POSITIVE
is harder to reach than NEGATIVE (7/8 strict vs ≤3/8), a failed null-guard can only DOWNGRADE a
positive, and a win-count alone can never produce POSITIVE.

Three methodological disciplines proved essential and are reusable beyond this repo:
- **Drift control via a perfect-percept null.** Run the same comparison in a regime where the trait
  has *no* benefit (e.g. `cue_noise=0`). If the mutant still fixates there, the apparent "win" is
  small-population drift, not selection. This caught multiple false positives.
- **Demographic viability first.** A common-garden cold-start needs an abundant-food regime or it
  collapses to drift-sized populations — which is a *validity* failure (NO_VERDICT), not a result.
  Distinguish "the gate said FLAT" from "the population was too small to interpret."
- **Byte-identity gating + disclosed pilots + no tuning-to-a-positive.** Every mechanism is gated
  OFF byte-identical to all prior experiments (hash-anchored); regimes are fixed on disclosed pilot
  seeds before the fresh-seed verdict; a positive seen on a pilot must replicate on fresh seeds.

## 3. The evidence

**Senses (Exp 199–207), CLOSED-NEGATIVE.** Seven structurally-distinct levers — avoidance,
foraging, increasing-returns tracking, interference competition, residue/false-positive
discrimination, rotating private niche, and a sensor–controller corner grid — all give a local
gradient `g(h_res) ≤ 0`. A forced strong sensor reproduces ~4× more (the gift is real), a
monomorphic precise population can be bulk-fitter and survivable (Exp 204/205), and a controller
pays *alone* while the sensor stays pure cost (Exp 207) — yet the marginal step never pays. Full
account: `docs/research/sense-axis-organ-evolution.md`.

**Information-processing (Phase 3), CLOSED-NEGATIVE.** Hidden-state inference was the
structurally-different bet: under partial observability the value of an extra remembered
observation should be *non-saturating* (variance ~1/k), so memory/inference looked like the regime
where a costed capability might finally have a positive local gradient — the bridge from ecology to
world models. It does not, at this substrate:
- **rung-1, integer `memory_horizon` (1→2):** local FLAT_OR_NOISY 6/8, mean invader-fraction 0.67
  vs the perfect-percept drift control 0.51 (a real but *sub-threshold* denoising signal),
  invasion-from-rarity 0/8 ⇒ FAIL_LOCAL_GRADIENT.
- **rung-1b, continuous `belief_persistence` (ρ 0.5→0.55), to rule out an integer-granularity
  artifact:** local FLAT_OR_NOISY 6/8 (0.74) with the perfect-percept control ALSO 6/8 (0.72) ⇒ the
  residual is drift, **not** denoising. A *large* step (ρ 0→0.85) is POSITIVE 3/3 — so the mechanism
  is genuinely live; the capability pays when gifted big but has no selectable *local* gradient.
  Committed runs: `experiments/outputs/preflight_memory_rung1/`,
  `experiments/outputs/preflight_belief_persistence_rung1b/`.

A second, subtler property surfaced: temporal integration is **not a pure denoiser** — a longer
memory also LAGS a hidden-state switch (stale pre-switch evidence), a real cost that produces an
interior optimum and further flattens the near-resident gradient.

**Active information-seeking (Phase 4 / Rung 3, Exp 210), CLOSED-NEGATIVE.** The lag/staleness
property above motivated the next bet: an *action* — pay to sample extra cues WITHIN the current
step (a "probe"), which sharpens the current-mode belief with **zero** staleness bias (variance
~1/(1+n)). If staleness was what flattened memory's gradient (Theory A), staleness-free active
sensing should finally show a positive local gradient. It does not — staleness was **not** the
killer (Theory B):
- New gated mechanism (`enable_active_sensing` + heritable `information_sampling_rate` + a per-step
  probe; byte-identical OFF, golden-hash-guarded). Mechanism is **LIVE**: gifted full-rate probing
  cuts wrong-type-cell occupancy 0.4042 → 0.3466 (benefit ceiling ≈ 0.0345 energy/step).
- Two disclosed validity fixes the pilot forced (now reusable disciplines): **drift is a
  population-size problem** (at the Phase-3-parity cap-50 regime the pure-cost control fixated the
  mutant as often as the info arm — drift-dominated; raised carrying capacity to cap-250, pops ~950,
  and read the drift-robust selection **slope** mean_s); and **calibrate the cost to the empirical
  benefit ceiling** (a cost above the gifted ceiling makes a negative inevitable — fair cost 0.01).
- Verdict (FRESH seeds 50–65): local step rate 0.0→0.10 is **7/16 wins, mean_s −0.0004, inv 0.520**
  (neutral), and **loses** to the pure-cost perfect-percept control (11/16, +0.0024, 0.656);
  `invasion_from_rarity` **0/16**; a cost sweep 0.005–0.1 shows info ≈ pure-cost at every cost.
  The `memory_horizon` trait re-tested at the SAME cap-250 regime is also flat (4/8, −0.0034) ⇒ the
  Phase-3 result is **not** a drift artifact. Committed run: `experiments/outputs/exp210.txt`.

**Uncertainty-gated active sensing (Phase 4 / Rung 4, Exp 211), CLOSED-NEGATIVE — the wasted-budget
hypothesis REFUTED.** Exp 210's probe fired at a *fixed* rate, so the natural follow-up was: did it
fail only because it wasted most probes on non-pivotal states? Exp 211 adds a `probe_policy`
abstraction and an **uncertainty-gated** probe — probe with probability `gain ×
sigmoid(sensitivity×(threshold − action_margin))`, where `action_margin = |single-cue belief − 0.5|`
is a *creature-available* signal (no oracle). It is the first real act-to-reduce-your-own-uncertainty
step (a pre-active-inference bridge, **not** full active inference).
- **The gate works exactly as designed when IMPOSED:** at matched budget it beats budget-matched
  *random* probing on decision quality (wrong-cell 0.4071 vs 0.4103), concentrates probes at ambiguous
  states (margin@probe 0.099 vs 0.941), and changes the next action on 0.474 of probes vs random's
  0.285; the pure-cost / gate-shuffle / hidden-scramble controls do not improve decisions.
- **But it does NOT beat fixed-rate** (0.3970): under sensory noise a single cue is often *confidently
  wrong* (large action margin, wrong half), and the cheap uncertainty signal cannot flag those, while
  indiscriminate fixed-rate probing fixes them by probing everything. So the **gated benefit ceiling is
  ~0** (0.0009 vs fixed-rate's 0.0186 energy/step) — an order of magnitude smaller.
- With essentially nothing to select on, the binding local step (gain 0.50→0.55) is flat/drift **even at
  probe_cost = 0.0** (so cost cannot foreordain it): 5/16 wins, mean_s −0.0031, NEGATIVE; flat across a
  cost sweep; invasion skipped. Blind-verified. Committed run: `experiments/outputs/exp211.txt`.

So **wasted fixed-rate budget was not the issue** — those probes were not waste, and uncertainty-
targeting does not rescue the benefit. The new lesson: a costed capability can *work when imposed* yet
have a near-zero benefit ceiling because the agent's own honestly-computable gating signal is itself too
noisy to identify the pivotal states. Named (lower-value) residual levers: a richer/multi-cue uncertainty
estimate, a lower-cue-noise regime, or mutation-geometry/standing-variation rather than a single step.

## 4. The law

> **At this toy active-inference ecology, a costed heritable capability's marginal step does not pay
> near the resident — its benefit saturates faster than its cost falls — so the local selection
> gradient is ≤ the evolvability threshold, EVEN WHEN the capability is genuinely useful when gifted.
> This holds across sensing AND information-processing, and across discrete AND continuous trait
> resolution. Useful-when-gifted ≠ locally evolvable; the local gradient is the binding barrier.
> It holds for PASSIVE capabilities (senses, memory/inference) AND for a costed ACTIVE
> information-gathering ACTION, both fixed-rate (Exp 210) and UNCERTAINTY-GATED (Exp 211) —
> perception and action hit the same wall, and even an agent that probes only when it is unsure
> cannot find the pivotal states with its own honestly-computable uncertainty signal.**

## 5. What this does NOT prove

- **Not** "these capabilities can never evolve in any world." It is a statement about *this* toy
  substrate's payoff geometry; a non-saturating payoff or a richer world could change the sign. The
  preflight is exactly the cheap instrument to re-test that when a new mechanism is proposed.
- **Not** a claim about scaled systems — toy scale throughout.
- **Not** that memory/sensing is useless — both are demonstrably useful when gifted (the gift and
  the large-step results are positive). The negative is specifically about the *local* gradient.
- A FAIL preflight correctly says "do not spend a full evolution batch," not "impossible."

## 6. Reusable contributions
- A trait-agnostic Evolvability Preflight (`ecology/evolvability/`) with the binding gate generic
  across traits; thermosense, memory_horizon, and belief_persistence are wired instances.
- The drift-control / demographic-viability / disclosed-pilot disciplines above.
- A gated, byte-identical hidden-state-mode engine substrate (`enable_hidden_mode`) for any future
  partial-observability work.
