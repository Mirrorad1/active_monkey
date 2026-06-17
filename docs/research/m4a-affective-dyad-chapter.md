# The M4a affective-dyad chapter — "talk to it and watch it learn to feel positive"

**Status: closed-positive at toy scale (2026-06-16).** The most direct rung toward the moonshot
goal — a toy active-inference agent you talk to that infers your latent intent, has a functional
valence channel, and learns to earn your positive feedback — is reached **at the long session**, and
packaged into a runnable, honest, constant-unfakeable artifact. Functional valence only; no sentience
claim, ever.

Source spec: `docs/specs/m4-affective-dyad.md`. Code: `active_loop/affect_spec.py` (MUTABLE model
builder) + `active_loop/affect_agent.py` (`DirectHeadAgent`). Deliverables (increment 1e):
`active-monkey-converse` (`active_loop/cli/converse.py`, honest REPL), `eval/affect_score.py` +
`eval/affect_score_json.py` (FROZEN learns-to-positive scorer).
Guards: `tests/test_affect_agent.py`, `tests/test_affect_score.py`.

## The arc (Exp 125 → 222)

1. **The credit-path wall (Exp 125 / 127 / 214, all HALTED F3).** The spec's original design routed
   the response→valence credit *indirectly* through an intent-transition `B`. It never learned
   (P3 0/8): the within-turn intent-transition signal is ~0 until `A`/`B` are already learned — the
   AIF echo of the program's recurring **useful-when-gifted ≠ learnable**. Timing was a real flaw
   (Exp 214 fixed it) but not sufficient.
2. **The direct head (Exp 215).** Giving the generative model a **direct** `P(valence | intent,
   response)` emission (`DirectHeadAgent`) gave the learner a non-vanishing gradient from action
   straight to feedback. Aliasing (K=U) was ruled out as the wall — narrowing it to exploration.
3. **Ignition → reliability-on-easy (Exp 216 / 217).** The failure decomposed into a conjunction;
   its binding remainder was an **exploration cold-start**, broken by an honest optimistic POS prior
   (+2.0 uniform across correct *and* wrong cells — no leakage). 7/8 seeds learned on the generous
   scaffold.
4. **The metric honesty upgrade (Exp 218 / 219).** A caught flaw: a constant "always reply 0" policy
   already earns POS ~1/3 of the time, so a ~1/3 realized-POS rate proves nothing. The fix is the
   constant-**UNFAKEABLE** probe `correct_select` (a constant policy caps at 2/6). Under it, the
   earlier "reliable 7/8" was only ~4/8 *genuine*. The two blockers separated cleanly: **short
   session blocks LEARNING** (near-chance `correct_select`); **low precision blocks EXPLOITATION**
   (high `correct_select`, low realized POS).
5. **The precision schedule (Exp 220, POSITIVE).** Annealing decisiveness γ 1→8 across the session
   (explore-to-learn, then sharpen-to-exploit) reached **13/16 GENUINE** at the realistic capacity
   K=4 — the **first reliable genuine discrimination** in the thread — beating fixed γ4 (7/16) and
   crushing fixed γ8 (3/16, over-commits early).
6. **The boundary (Exp 221, NEGATIVE / NEW INSIGHT).** Separating the schedule from session length:
   the schedule does **not** rescue the short session (short sched 0/2/3 of 16). It beats fixed only
   at 300t (+6) and is slightly *worse* at short lengths (anneals too fast → over-commits before
   learning). `mean_csel` rises with length for **both** schedule and fixed ⇒ the schedule is an
   *exploitation* optimizer; **the long 300t session is load-bearing** for the LEARNING itself.
7. **The milestone packaging (Exp 222, increment 1e).** `active-monkey-converse` + the FROZEN scorer. The
   scorer makes "learns to feel positive" one reproducible number that a lazy policy **cannot fake**:
   A1 — `score_affect()` at the validated config returns `verdict=True` (mean_last 0.42 > the 1/3
   ceiling, genuine_fraction 0.75, improvement 0.24, reproducing Exp 220's learning); A2 — the SAME
   scorer on a constant-response control returns `verdict=False`, `genuine_fraction=0.0`.

## The provided-vs-learned ledger (honesty)

**Provided (designer-given structure / scaffold):** the intent/response/utterance-code repertoire
(K/R/U); the generative model structure (`A`/`B`/`C`/`D`) and the **direct** `P(valence|intent,
response)` head; the scripted partner (a fixed teaching policy `CORRECT = c mod 4` + the valence rule
POS/NEU/NEG); the honest **optimistic POS prior** (uniform, no leakage); the **precision schedule**
(γ 1→8); and the **long 300-turn session**. These are declared, not earned.

**Learned by the agent (Dirichlet on `A`, from lived turns):** *which response earns your `+` for each
inferred intent* — the content of "what it learned about you." This is what `correct_select` measures
(constant-unfakeable), and it is genuinely acquired: from a weak prior, in one short session, the dyad
maps ≥3–5 of 6 signals to their approval-earning response, and its realized positive-feedback rate
rises from ~0.18 to ~0.42 over the session.

## The honest boundary (where it stops)

- **The long session is load-bearing (Exp 221).** Reliable genuine learning needs ~300 turns; short
  realistic sessions (100–200t) block the *learning*, and no precision schedule rescues them. The
  named next lever for the short session is a **learning-side** one (lr / optimism / replay), not a
  precision one — left open.
- **Functional valence only.** "Valence" = −free energy + grounded-extrinsic approval; "feels
  positive" = the realized POS rate rising + the unfakeable discrimination. No subjective-experience
  claim.
- **Coarse intent, not language.** "Intent" is a small clustering over utterance *codes*; the REPL
  takes codes, not free text. Conversational language understanding remains the documented moonshot
  ceiling (`open_problem.html`).

## What this unlocks

The metric is now **FROZEN and unfakeable**, so **M4b** — the PR-style autopilot over the MUTABLE
`affect_spec.py` against `eval/affect_score.py` — becomes possible: the loop can try to raise the
genuine learns-to-positive number without being able to reward-hack it (a constant policy scores
`verdict=False`). That is the next rung, on a human's word.
