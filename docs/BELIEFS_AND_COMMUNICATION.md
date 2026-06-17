# BeliefBench v0 and Comm v0

Two minimal, substrate-independent benchmarks that operationalize "belief-like state" and
"proto-communication" **without overclaiming**. Both are pure-numpy, seedable, and ship
predeclared falsifiers, controls, and an honest **PASS / FAIL / INCONCLUSIVE** verdict.

Neither benchmark involves natural language, semantics, or sentience.

---

## BeliefBench v0 — hidden partner-type inference

`active_loop/benchmarks/beliefbench.py`

A partner has a hidden type `z ∈ {friendly, neutral, adversarial, inconsistent}`. Each turn
the agent receives an ambiguous observation drawn from the partner's type-conditioned
emission, maintains a posterior `q(z)` by **exact Bayesian filtering**, and chooses an
action `∈ {respond_A, respond_B, probe, wait}`. `probe` pays a small cost for a *sharper*
observation next turn (an information-gathering action).

**"Belief" here = the posterior `q(z)`.** Nothing subjective is claimed.

Belief is counted as **load-bearing** only if all four hold:

1. **evidence update** — `q(z)` changes when observations change (`belief_update_magnitude > 0`);
2. **action relevance** — the policy changes as `q(z)` changes (`policy_belief_sensitivity > 0`);
3. **scramble control** — shuffling the posterior hurts reward (`reward_normal − reward_scrambled > δ`);
4. **held-out transfer** — the same machinery works on fresh seeds and a shifted partner mix.

The reward table `R(action, z)` is the **environment's public payoff** (like a preference
vector); the agent plans `argmax_a E_{q(z)}[R(a,z)]`. A hidden-type→action lookup is used
**only** inside the named ORACLE baseline, never as the agent. Baselines: constant action,
random action, no-belief reactive (acts on the current observation only), scrambled belief,
and oracle (ceiling).

Metrics: `belief_update_magnitude`, `policy_belief_sensitivity`, `reward_{normal,
scrambled, reactive, constant, random, oracle}`, `reward_delta_scrambled`, the held-out
transfer deltas, and `mean_logloss` (calibration of `q(z)` against the true `z`).

```bash
uv run python -m active_loop.benchmarks.beliefbench
```

**v0 result (seeds 0–6, 240 turns, noise 0.45): PASS.** The posterior moves with evidence,
drives the policy, beats the reactive/constant/random baselines, sits below the oracle
ceiling, and scrambling the belief roughly halves reward — and the same holds on a held-out
partner mix. This is a positive result at toy scale; it is *not* a claim about real beliefs.

---

## Comm v0 — costed sender/receiver signaling (proto-communication, NOT language)

`active_loop/benchmarks/comm_v0.py`

A Lewis-style referential signaling game: the world has a hidden state `z`; a **sender**
observes `z` and emits a message `m` at a per-message **cost**; a **receiver** sees only
`m` and chooses an action `a`; reward is `1` if `a` matches the correct action for `z`,
else `0`, minus cost. There is **no reward for any token identity** and **no hardcoded
English / semantic label** — sender and receiver learn a joint mapping by simple Roth-Erev
reinforcement.

We then test whether the learned signal actually carries hidden-state information and
changes the receiver, against controls: **shuffled** messages (decoupled from `z`),
**muted** (constant signal), **permuted** message IDs, **random sender**, **constant
sender**.

PASS requires (across seeds): `reward_normal` beats shuffled AND muted by a predeclared
delta, empirical **mutual information `I(m; z) > 0`**, and the receiver's action changes
with the message.

Metrics: `reward_{normal, shuffled, muted, permuted, random_sender, constant_sender}`,
`reward_delta_{shuffled, muted}`, `mutual_information_bits`, `message_entropy_bits`,
`receiver_policy_sensitivity`.

```bash
uv run python -m active_loop.benchmarks.comm_v0
```

**v0 result (seeds 0–7, train 4000 / eval 600, cost 0.05): PASS.** The trained protocol
earns ~0.89 vs ~0.20 for shuffled/muted/permuted controls, the message carries ~1.9 bits
about `z`, and the receiver maps different messages to different actions. We call this
**costed signaling / proto-communication** — explicitly **not** language.

---

## Honesty notes

- Both benchmarks return INCONCLUSIVE or FAIL honestly when their predeclared deltas are
  not met; they are not tuned-until-they-pass.
- The thresholds (`UPDATE_FLOOR`, `SCRAMBLE_DELTA`, `MI_FLOOR`, …) are frozen module
  constants; changing them is a benchmark version bump, not a silent edit.
- These are toy-scale existence tests of *machinery*, not claims about cognition or meaning.
