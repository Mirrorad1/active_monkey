# Roadmap

This roadmap separates current evidence from long-term motivation. The ambition is to
study sensing, memory, belief-like internal state, and communication under pressure. The
current evidence is narrower and toy-scale: passive cue integration, `memory_horizon`,
and continuous `belief_persistence` did not yet produce robust adaptive sense-organ or
belief-capacity emergence under fair local-gradient tests.

## Current State: Passive Memory Is Not The Next Knob

The thermosense and Phase 3 hidden-state-memory chapters now share the same binding
lesson: a capability can be useful when gifted, but still fail to be locally evolvable
when a small heritable step near the resident does not pay for itself.

That does **not** mean memory is solved, useless, or impossible in general. It means the
current ecology has not shown that simply adding passive cues, passive hidden-state
integration, or more memory knobs causes robust adaptive sense-organ or belief-like-state
growth. For this phase, deeper passive memory is a mostly closed-negative direction.

The reusable method remains Evolvability Preflight: before spending compute on full
evolution batches, measure whether the local trait gradient or design premise is positive
enough to justify the batch.

## Phase 4 — Active Sensing

**Phase 4 goal:** test whether information-gathering actions can become locally adaptive.

**Core question:** can a small increase in sampling or probing tendency beat the resident
in a fair common garden?

**Candidate mechanisms:**

- `probe_action`
- `sample_extra_cue`
- `lookahead_cue`
- `local_scan_radius`
- `information_sampling_rate`

**Acceptance bar:**

1. Probe action costs energy, time, or opportunity cost.
2. Probe is not directly rewarded.
3. Probe only improves future action selection.
4. Scrambling or disabling the hidden state removes the advantage.
5. Local pairwise gradient passes before full evolution.

Active sensing means the agent takes an action to improve its observations, not just to
get food directly. Programmer translation: instead of only reacting to logs, the service
pays to add better observability before deciding what to do.

This is a pre-active-inference bridge, not a claim that full active inference has been
achieved in the ecology. Active inference would couple action and perception more deeply:
the agent acts partly to reduce uncertainty about hidden state, not only to maximize
immediate reward. Programmer translation: like a runtime system choosing diagnostics that
reduce ambiguity before taking a risky production action.

## Why This Matters

Active sensing is a better bridge to active inference than simply adding more memory,
because the agent must spend resources to reduce uncertainty before acting. The immediate
scientific test is local adaptivity of costly probing or sampling actions: does a small,
costed tendency to gather better information win under the same fair common-garden
standard that rejected passive memory and passive sensor precision?

## What Would Falsify This Direction

The direction should be considered unsupported if probing only helps because of direct
reward leakage, hidden hardcoding, id-order artifacts, unfair seed conditions, or if the
advantage survives hidden-state scrambling. A negative local pairwise gradient is also a
first-class result: it would say this bridge does not pay locally in the current ecology.

## Future: Communication And World-Model Substrate

Communication remains motivation, not current evidence. A communication result would need
to show that signals change another creature's beliefs or actions under pressure, without
directly rewarding the signal token itself.

World models are also future motivation. Learned latent models should be introduced only
when explicit belief variables stop scaling and there is a concrete, audited problem for
the learned model to solve.

Near-term infrastructure may still include a unified AgentState surface that can fork,
replay, restore, and compare creature/ecology histories without losing provenance, but it
should support the active-sensing test rather than reopen passive memory as the main
scientific direction.

## Direction Update: a falsifier-first toy lab with copyable agent artifacts

active_monkey is becoming a **falsifier-first toy lab with copyable agent artifacts**. The
unit of work is a reproducible artifact + a frozen scorer + predeclared controls, so any
result — positive or negative — is checkable and forkable.

Status legend: **[done]** completed and verified · **[scaffold]** runnable scaffold ·
**[proposed]** next experiment · **[neg/inconcl]** negative or inconclusive.

- **[done] First artifact: `active-monkey-affect-dyad-v0`.** A substrate-independent
  `AgentState`/checkpoint layer (`active_loop/state.py`, safetensors, stable content hash,
  schema-mismatch refusal), a local artifact API (`active_loop/artifacts.py`), and a stable
  `active-monkey` CLI (`export` / `inspect` / `score` / `converse`). Provenance pins the
  frozen scorer (`eval/affect_score.py`) by sha256; scoring refuses to run on hash drift.
  See `docs/ARTIFACTS.md`.
- **[done] Autonomous find-and-keep, end-to-end (Exp 227).** The REAL
  `affect_pr_loop.one_affect_iteration` machinery, run on an isolated clone with a real
  `claude -p` critic + the real FROZEN scorer + a real git merge, REJECTS a gaming A0 bake
  and KEEPS the honest C1 NEU-aversion (baseline 0.4225 → merged 0.51875, verdict held;
  `experiments/exp227_real_autopilot_find_and_keep.py` → `.../exp227_real_autopilot.json`).
  The candidate generation is scripted (the two Exp 225 moves) so the run is reproducible;
  the keep decision is fully autonomous. A fully-autonomous *discovery* run (real
  `AffectClaudeProposer` generating candidates) remains the next step. The earlier
  deterministic stand-in (`experiments/exp226_autonomous_find_and_keep.py`) re-implemented the
  keep/reject logic; Exp 227 exercises the production code instead.
- **[done] Host-robust scorer (`active_loop/affect_score_fast.py`).** On a constrained host the
  binding constraint is XLA JIT exhaustion, not wall-time: the frozen agent recompiles a static
  gamma every turn, and the compiled dylibs accumulate until the CPU backend fails to
  materialize symbols, so a full 8-seed score crashes mid-run. `jax.clear_caches()` between
  seeds (per Exp 226) frees the executables and bounds the dylib count; a small worker pool adds
  a modest wall-clock win on top. Bit-identical to `eval/affect_score.py` (guarded by
  `tests/test_affect_score_fast.py`); full config completes ~13 min / 5.6 GB. Process-parallelism
  WITHOUT cache-clear is a trap — N cold caches multiply the memory and OOM.
- **[done] BeliefBench v0** (`active_loop/benchmarks/beliefbench.py`): hidden partner-type
  inference; PASS at v0 (belief updates, drives policy, scrambling hurts, transfers, below
  oracle). "belief" = posterior over a hidden state, not subjective belief.
- **[done] Comm v0** (`active_loop/benchmarks/comm_v0.py`): costed sender/receiver signaling
  ("proto-communication", NOT language); PASS at v0 (beats shuffled/muted, ~1.9 bits MI).
- **[neg/inconcl] Short-session learning levers**
  (`experiments/short_session_affect_levers.py`): honest sample-efficiency curves for
  optimism / exploration / experience-replay levers at 30/50/100/300 turns. Per Exp 221 the
  long session is load-bearing; a non-PASS at short sessions is the expected negative, not a
  tuning failure. Eligibility traces and an active ASK/probe action remain scaffolded TODOs
  (reported NOT_IMPLEMENTED, not faked).
- **[proposed] Eventual re-entry into ecology / selection** via the same AgentState surface
  (fork / replay / restore with provenance), once the belief and communication benchmarks
  have load-bearing levers.

This direction does not overstate results: BeliefBench/Comm are toy-scale existence tests
of machinery, the autonomous loop is a deterministic stand-in until the claude-driven loop
is run, and the short-session levers are honestly inconclusive/negative at short horizons.

## Standing Guardrails

- Keep long-term ambition separate from current evidence.
- Preserve null results and honest walls.
- Do not claim memory is solved, active inference has been achieved, or emergence is
  guaranteed.
- Do not rename packages or imports casually.
- Do not weaken traceability to improve presentation.
- Any new public claim needs a falsifier, script, output, caveat, and reader path.
