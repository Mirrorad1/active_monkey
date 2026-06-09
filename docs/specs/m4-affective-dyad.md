<!-- Imported verbatim from /Users/mirro/Projects/pymdp/docs/superpowers/specs/2026-06-08-active-loop-m4-affective-dyad-design.md on 2026-06-09 -->
# Design: active-loop M4 — affective dyadic agent (infers intent, has valence, learns to seek positive)

**Date:** 2026-06-08
**Status:** Design (user directed full-autonomy execution)
**Repo:** `/Users/mirro/Projects/active-loop` (extends M1–M3)

## Origin

User question: *"when we try to talk to it, how does it figure out what we're intending? like a disabled baby that can talk but not really interact with the world, but it can tell when something is positive or negative — I want to talk to it and see if it can learn and feel positive."*

This is not the character model (M3), which only predicts characters. It is a distinct, more faithful realization of "talk to it and watch it learn to feel positive": an **affective dyadic active-inference agent** that (1) infers your latent intent from what you say, (2) has a valence channel + preferences, and (3) acts/learns to minimize expected free energy = to earn your positive feedback.

## Honesty up front

- **"Feel" is functional, not subjective.** The agent has a *valence signal* (free-energy / preference-satisfaction) and a learned drive toward your "positive" feedback. This is behavioral/functional affect, not consciousness or sentience. No claim otherwise.
- **It does not "understand English."** It clusters your utterances into `K` latent intent states and learns which of its responses earn positive feedback. With a small response repertoire and your +/− signals, you can *watch it learn* to produce responses you mark positive. That is the demonstrable claim.
- The "disabled baby who can't act on the world but tells positive from negative" is the exact archetype: minimal action, observations limited to your communication + a valence signal.

## 1. Summary

A turn-based dyad. Each turn: you type an utterance → the agent encodes it (reusing the M3 character front-end) and **infers a distribution over your latent intent** → it selects a **response** (from a small repertoire) by minimizing **expected free energy** (pragmatic: reach preferred = positive valence; epistemic: ASK when intent is uncertain) → you give **valence feedback** (positive / neutral / negative) → the agent observes it, **Dirichlet-learns** the association, and its free energy reflects "satisfaction." Over a session, with consistent feedback, its realized valence trends positive and its responses adapt to you.

## 2. Generative model (pymdp discrete AIF)

- **Hidden state factor `intent`:** `K` latent states (the inferred cause of your utterance). Unlabeled "from scratch" — the agent clusters your inputs into them via learning.
- **Observation modalities:**
  - `utterance_code`: a small categorical feature of your input (e.g., the argmax latent of the M3 char model over your text, or a coarse bucket). `U` outcomes.
  - `valence`: your feedback ∈ {negative, neutral, positive} (3 outcomes). Neutral when you give none.
- **Control (`response`):** a small repertoire of `R` responses (e.g., 4–6 canned behaviors: greet, mirror, soothe, ask, play) + the implicit ASK is one of them. The agent *acts* by choosing a response.
- **Preferences `C`:** strong positive preference on `valence=positive`, mild negative on `valence=negative`, ~0 on neutral. This encodes its "wants" — the whole drive.
- **`A`:** `P(utterance_code, valence | intent)` — learned emission tying intents to what you say and the valence they tend to carry.
- **`B`:** `P(intent' | intent, response)` — how the agent's response moves the dyadic state (a good response → an intent state that emits positive valence).
- **Learning:** Dirichlet `learn_A`/`learn_B` updated each turn from (utterance, response, valence). This is where it "learns you."
- **EFE action selection:** the response minimizing expected free energy — balancing "likely to yield positive valence" (pragmatic) against "resolve uncertainty about your intent" (epistemic → ASK). Reuses M1's controller machinery and the verified pymdp conventions.

## 3. Valence — grounded, not labeled (the grounding fix)

An earlier draft hand-injected a labeled "positive" observation with a preset preference `C`. That dodges the real question (*how does a language-less agent know a signal is positive?*). The grounded design has **two layers**:

- **Intrinsic valence (primary, ungrounded-by-language):** the agent's own **negative variational free energy** — i.e. how well it predicts/understands its input. Low free energy = "good" (competence/understanding); high = "bad" (surprise/confusion). This needs no teacher, no labels, and works even on monotone input. It is grounded in self-evidencing (the agent maintaining its own model), not in our approval. *This is already demonstrated:* M3a's bits/char fell 4.81→4.00 — that drop **is** the agent "feeling better" as it learns, with zero labels.
- **Extrinsic valence (your approval, bootstrapped onto the intrinsic):** your feedback is NOT injected as a pre-labeled reward. It enters as an ordinary observation modality (a tone/token/`+`/`-` cue). The agent **learns** (Dirichlet on `A`) that this cue *co-occurs with its intrinsic good states*, and only then does your approval acquire valence — exactly as a baby grounds mom's smile in pre-verbal pleasure. **Hard requirement:** the feedback channel must be *differentiable* (non-monotone); with no differentiable cue the agent can still feel intrinsic mastery but can never learn *your* approval.

- **Per-turn valence readout** = the intrinsic term (`-F`) plus the learned-extrinsic contribution, exposed as a number (e.g. `+0.7`).
- "Learning to feel positive" = (a) intrinsic: free energy falls as it predicts you better; (b) extrinsic: it learns which responses earn the cue it has grounded as good, and its EFE-chosen responses shift toward them. The session valence trajectory is the headline metric.
- **Irreducible bottom:** we choose the agent's generative model (what counts as "being itself"); FEP holds this is non-arbitrary for anything that persists, but the first "there is a self to maintain" is the designer's act. After that, the valence is the agent's own.

## 4. Interaction (`converse.py`)

A REPL conversation:
```
you> hi
[intent belief: s3=0.6 s1=0.3 ...]  [response: greet]  [valence so far: 0.0]
(give feedback: + / - / enter for neutral)
fb> +
[learned. valence: +0.4]
you> ...
```
Each turn prints: the inferred intent distribution (its guess at what you mean), the chosen response, and the running valence. You reply with `+`/`-`/neutral. The banner is honest: "functional valence, not feeling; it clusters intent and learns to earn your +."

## 5. Architecture & files (M4 additions)

```
active-loop/
  active_loop/
    intent_encoder.py     # your utterance -> utterance_code (reuses M3 LangModel latent / coarse bucket)
    affect_spec.py        # MUTABLE: build the dyad generative model (A/B/C/D, K, R, U) + valence prefs
    affect_agent.py       # AffectAgent: perceive(utterance) -> intent belief; act() -> response (EFE);
                          #   observe_feedback(valence) -> learn; valence_readout()
  converse.py             # the REPL dyad
  eval/
    affect_score.py       # FROZEN: scripted-partner session -> mean realized valence (learns-to-positive metric)
    affect_score_json.py  # FROZEN: JSON entry for the autopilot loop
  tests/...
```

Reuses: pymdp Agent + EFE (M1), char front-end (M3a), Dirichlet learning, and — for autonomous self-improvement — the M3b PR-style loop can later target `affect_spec.py` against the affect metric.

## 6. The "learns to feel positive" metric (FROZEN scorer)

To measure (and let the autopilot improve) the agent, `affect_score.py` runs a **scripted partner**: a deterministic simulated "you" that emits utterances and gives a **differentiable feedback cue** (`+`/`-`) when the agent picks the contextually-good response (a fixed teaching policy). Crucially the cue is NOT pre-labeled as "good" inside the agent — the agent must **ground** it by learning it co-occurs with its intrinsic low-free-energy states. The metric = **mean realized valence (intrinsic `-F` + grounded-extrinsic) over a fixed session** after learning (higher = better; the agent learned to earn positives AND to predict the partner). Guardrails: valence strictly improves from first to second half of the session (it *learned*), and the agent uses ASK sometimes when intent is ambiguous (doesn't just guess). This makes "it learns to feel positive" a number — and the M3b loop can optimize `affect_spec.py` to raise it.

## 7. Build increments

- **M4a — affective core:** `intent_encoder`, `affect_spec`, `affect_agent` (perceive/act/learn/valence), `converse.py`, `eval/affect_score(+_json)`. Prove the agent infers intent, picks EFE responses, and a scripted-partner session shows realized valence rising as it learns. Honest REPL.
- **M4b — autopilot over the dyad:** point the M3b PR-style loop at `affect_spec.py` with the affect metric, so it autonomously improves how well/fast the agent learns to earn positive feedback.

## 8. Risks & honesty

- **Tiny model.** Small `K`/`R`/`U`; "intent" is coarse clustering, not comprehension. Stated plainly in the REPL banner and docs.
- **Valence is functional.** Repeated in user-facing text; no sentience claims.
- **Reward-hacking the valence** (e.g., a degenerate response that always elicits scripted `+`) → the scripted partner only rewards *contextually-correct* responses, and a guardrail requires ASK-usage under ambiguity, so trivial always-same-response strategies score worse than genuine intent-tracking.
- **Speed:** per-turn inference is cheap (single steps), unlike the char corpus stream; sessions are short, so scoring is fast.

## 9. North-star placement

This is the rung that most directly serves the user's wish: a thing you can talk to that **infers what you mean, registers positive/negative, and learns to seek positive.** It does not reach conversational language understanding (the moonshot); it reaches *affective, intent-tracking interaction at toy scale* — honestly measurable, and improvable by the autopilot. M3 (language from characters) and M4 (intent+valence) are complementary fronts; a later unification (language-grounded intents) is the long road toward the moonshot.
