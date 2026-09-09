# Repairing optimizer memory after a verifier correction

**Research memo and executable experiment plan.** A narrow candidate was formalized and cheaply tested: after repairing a verifier bug, replace the contribution of affected training batches to Adam's first and second moments, retain the current model weights and optimizer clock, and continue with the corrected verifier. **The predeclared screen was NEGATIVE.** Moment repair improved early recovery over retaining stale state, but an ordinary full optimizer reset beat it in every seed in both corruption regimes. Novelty remains provisional; no novel, effective LLM RL method has been established.

This memo distinguishes three claims: an exact bookkeeping identity, an empirical optimization hypothesis, and possible transfer to LLM RL or an RSI system that revises its verifier. Only the second is the proposed research hypothesis. Neither the identity nor a small bandit result would establish a new general theory of learning or scalable self-improvement.

**The one hypothesis.** After a localized verifier correction, replacing the affected historical contributions to both Adam moments, evaluated at the actual historical parameters and on the actual sampled responses, yields higher mean true reward over the next 30 updates than retaining state, resetting state, or refreshing moments at the current parameters. All arms start from identical weights, receive the same corrected verifier, and have the same future rollout budget. The proposed explanation is that reward-invalidated momentum can be removed without discarding still-useful optimization history.

The hypothesis is deliberately stronger than “optimizer state matters.” It can fail if an ordinary reset is sufficient, if current-parameter refresh is better, if stale parameters make repaired history misleading, or if changing the moments cannot compensate for damage already stored in the weights. It also has a cost condition: any apparent advantage must eventually survive a comparison that spends the repair overhead on additional ordinary learning. The first executable screen tests utility before undertaking that more expensive comparison.

**Why the broad FEP idea remains closed.** Prediction-error rewards, adaptive advantages, self-generated feedback, judge ensembles, and search over prompts or code do not supply the novelty here. FREIA already combines free-energy-driven rewards with adaptive advantage shaping. DeepSeek-R1, SCoRe, and Self-Rewarding Language Models cover important combinations of reasoning RL, self-correction, and self-generated evaluation. Agent-as-a-Judge covers agentic feedback. STOP, optimizable agent graphs, DGM, and AlphaEvolve cover recursive improvement or search over scaffolds and code. A unified description of off-policy and on-policy post-training likewise does not make a new algorithm. The source table below records these closures explicitly.

Two tempting narrower alternatives also fail the reduction test. Let \(S_{ij}\) be the score of agent version \(i\) under evaluator version \(j\), with 0=old and 1=new. The interaction

\[
I=(S_{11}-S_{01})-(S_{10}-S_{00})
\]

is a standard crossed contrast. Subtracting it from the new-evaluator improvement gives exactly the old-evaluator improvement. That is not a novel correction. Using its magnitude to acquire audits is evaluator disagreement on a parent–child contrast, a query-by-committee variant. Moreover, RQGM already co-evolves agents and evaluators, and Double Ratchet already uses anchors, outer audits, and a crossed final-judge study. Neither “evolve the judge” nor “audit the judge” remains an adequate novelty claim. [RQGM](https://arxiv.org/abs/2606.26294), [Double Ratchet](https://arxiv.org/html/2607.12790v1), [conflict-aware feedback acquisition](https://ojs.aaai.org/index.php/AAAI/article/download/41104/45065).

**Formal intervention.** Let \(\theta_{s-1}\) be the actual parameters before update \(s\), \(D_s\) its recorded rollout group, \(r_s\) its original rewards, and \(r_s'\) those rewards after the verifier correction. Define

\[
g_s=\nabla_\theta L(\theta_{s-1};D_s,r_s),\qquad
g_s'=\nabla_\theta L(\theta_{s-1};D_s,r_s'),\qquad
\delta_s=g_s'-g_s.
\]

Everything other than the rewards is held at its recorded value. If the objective uses group normalization, all advantages in each affected group must be recomputed; relabeling only the changed response is insufficient. If it uses PPO clipping, gradient clipping, a KL term, dropout, or gradient accumulation, the recorded loss and optimizer-input gradient conventions must be reproduced. In particular, clipping branches may change when advantages change sign.

At correction time \(T\), for the retained affected updates \(\mathcal B\), apply

\[
m_T'=m_T+(1-\beta_1)\sum_{s\in\mathcal B}\beta_1^{T-s}\delta_s,
\]

\[
v_T'=v_T+(1-\beta_2)\sum_{s\in\mathcal B}\beta_2^{T-s}
\left(2g_s\odot\delta_s+\delta_s\odot\delta_s\right).
\]

Preserve \(\theta_T\), the step counter \(T\), the learning-rate schedule, and unaffected historical contributions. Adam then uses its usual bias correction and update. These formulas follow directly from Adam's exponential moving averages; the formulas are not claimed as new mathematical results. The cross term in the second equation is necessary: adding only squared reward-correction gradients does not replace the original second moment. [Kingma and Ba, Adam](https://arxiv.org/abs/1412.6980).

Unrolling the recurrence proves that the repaired moments equal those obtained by feeding the revised gradient sequence through the same EMA, starting from the same pre-buffer moments. This equality holds for the **recorded parameter path**. It does not produce the weights or gradients that would have occurred if the verifier had been correct throughout training. Call it a fixed-path state intervention, not exact unlearning or counterfactual retraining. It leaves policy-distribution shifts and weight damage untouched.

A useful sanity case is \(g_s'=-g_s\). Its second-moment correction is zero even though the first moment changes. This is why the experiment includes an m-only repair ablation. Binary reward changes in a three-action, group-normalized policy are not necessarily whole-gradient sign reversals, so that ablation still has a meaningful job.

For a buffer that omits older affected updates, exactness is restricted to the retained buffer. If omitted corrections satisfy \(\|\delta_s\|_\infty\le D\) and both gradient versions have coordinate magnitude at most \(G\), the omitted first-moment correction is bounded by \(D\beta_1^K\) after retaining the newest \(K\) updates, and the omitted second-moment correction by \(2GD\beta_2^K\). These are conservative geometric-tail bounds, not guarantees of good future reward. With \(\beta_2=0.999\), a short tape cannot repair a long history of corrupted second moments.

**The residual against the closest literature.** This is an inference from a bounded literature review, not proof that no prior publication contains the operation. Primary sources were checked through 8 September 2026; the closest optimizer and evaluator papers were read in full-text HTML where available. Negative exact-phrase search results carry little weight. The important novelty risks are substantive overlaps, listed first here.

| Prior work | What it already establishes or implements | What would have to remain distinct |
|---|---|---|
| Bengio, Pineau and Precup, [Correcting Momentum in Temporal Difference Learning](https://arxiv.org/html/2106.03955v1), 2021 | Gradients can become stale through changing parameters and TD targets; proposes a first-order momentum correction. | A discrete, externally validated verifier revision; exact replacement of both Adam EMA contributions on a frozen historical path. The broad “correct stale momentum” idea is already closed. |
| Ellis et al., [Adam on Local Time](https://arxiv.org/html/2412.17113v1), NeurIPS 2024 | Nonstationarity affects Adam; compares resetting moments and retaining moments with local clock resets. | Repair content using revised historical rewards, while retaining the global clock. Merely preserving useful optimizer history is already closed. |
| [Off-Policy Corrected Reward Modeling](https://arxiv.org/html/2507.15507v1), COLM 2025 | Updates reward models using importance weighting; resets policy optimizer and value-function state between stages. | Replace reward-invalidated history selectively, instead of treating the reward-model transition as a reset boundary. |
| Stewart, [Form and Function: Machine Unlearning as a Problem of Misaligned States](https://arxiv.org/html/2605.17590v1), 2026 preprint | Treats online L-BFGS unlearning as counterfactual state alignment; distinguishes parameter-only, memory-only, and replay interventions. | A utility-seeking state-only intervention, with no claim to reconstruct a realizable counterfactual model. This paper is a direct warning against overclaiming the reconstruction. |
| Guo, [Delayed Optimizer-State Transport Shapes Short-Horizon Training Decisions](https://arxiv.org/html/2608.24593v1), 2026 preprint | Uses full model–optimizer response over future paths to select loss schedules. Optimizer history is already an actionable training state. | Retrospectively substitute verifier-revised gradients; do not claim novelty for optimizer-aware short-horizon control. |
| RQGM / Double Ratchet / [J-Zero](https://arxiv.org/abs/2608.26582), 2026 | Co-adapt evaluators with agents, skills, or solvers. | The operation immediately after a validated evaluator revision, inside an adaptive optimizer. |

The potentially new contribution is therefore a small algorithmic specialization plus a falsifiable performance claim: **verifier-revision-specific replacement of first and second moments can outperform simple recovery strategies in policy-gradient learning.** A reviewer may reasonably regard it as incremental staleness correction. A positive result against weak baselines would not answer that objection. Discovery of a prior method performing the same historical reward substitution in adaptive optimizer state would close the algorithmic claim entirely, even if it used another name.

| Required closure | Source and reduction |
|---|---|
| FEP reward and advantage shaping | Huang et al., [FREIA](https://arxiv.org/html/2605.04065v1), 2026. Changing the reward or advantage shape is not the proposed contribution. |
| Reasoning RL | DeepSeek-AI, [DeepSeek-R1](https://arxiv.org/abs/2501.12948), 2025. The method would modify an optimizer transition within such training. |
| Self-correction | Kumar et al., [SCoRe](https://arxiv.org/abs/2409.12917), 2024. Revising an answer and revising optimizer history are different intervention targets. |
| Self-rewarding and process feedback | Yuan et al., [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020), 2024; Zhuge et al., [Agent-as-a-Judge](https://arxiv.org/abs/2410.10934), 2024. The source of a repaired reward is supplied, not innovated here; process/temporal reward variants have the same scope boundary. |
| Recursive code or scaffold search | Zelikman et al., [STOP](https://arxiv.org/abs/2310.02304), 2023; Zhuge et al., [Language Agents as Optimizable Graphs](https://arxiv.org/abs/2402.16823), 2024; Zhang et al., [DGM](https://arxiv.org/abs/2505.22954), 2025; Google DeepMind, [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/), 2025. These could discover an optimizer patch, but the present intervention is specified directly and contains no scaffold search. |
| Unified post-training | Zhao et al., [A Unified View of Off-Policy and On-Policy Learning](https://arxiv.org/abs/2604.07941), 2026. A framework for composing learning regimes does not supply the particular moment substitution. |
| Model-based/Bayesian RL, active inference, MoE/routing, uncertainty exploration, shaping, preferences, curricula, novelty/QD search, meta-learning | These are established method families, not newly named components. The selected experiment has no learned transition model, router, acquisition controller, curriculum, population search, or learned optimizer. It holds the recovery objective fixed and changes stored optimizer state. General meta-learning could represent the intervention; representability alone does not establish priority for this exact specialization. |

**Operational FEP interpretation.** Prediction-error minimization would fit a model of verifier mistakes from independently checked outcomes. An epistemic action would rerun a past response against an independent executable check to resolve whether its reward was wrong. An RSI loop can then revise its verifier and invoke the state intervention above. Those acquisition and inference operations are ordinary active learning, not novelty claims. The experiment below supplies the verifier correction at a fixed time to isolate the optimizer mechanism; it tests neither an FEP-derived controller nor autonomous discovery of verifier bugs. Dropping FEP terminology leaves the hypothesis and equations unchanged.

**Minimal executable screen.** The implementation is `experiments/exp278_verifier_moment_repair.py`. It is a NumPy contextual bandit with 24 trainable parameters, 16 contexts, three possible actions, and eight sampled actions per context. It uses group-relative normalized rewards and a shared linear softmax policy. It is actual policy-gradient learning at toy scale, not an LLM, language-generation benchmark, learned verifier, or autonomous RSI demonstration.

Each seed supplies normalized Gaussian context features and a random teacher determining the correct action. Run 60 clean updates, 20 updates under a corrupted verifier, and 30 clean recovery updates. The corruption replaces the correct action with the next action modulo three for either 4/16 contexts or 12/16 contexts. A no-change control keeps all labels correct. Adam uses learning rate 0.03, betas 0.9/0.999, and epsilon 1e-8. The primary endpoint is the mean exact success probability over the 30 post-intervention updates, excluding the initial shared checkpoint. It measures recovery on the fixed contexts, not generalization to unseen contexts.

The 20 corrupted batches store sampled actions and historical probabilities, allowing exact rescoring of the group-normalized gradient in this linear model. The repaired verifier, its correct labels, the corruption window, the features, and the repair algorithm are all provided by the harness. Every intervention starts from the same weights. Subsequent arms share uniform random variates but sample from their own policies; forcing identical actions after policies diverge would silently turn the continuation off-policy.

| Arm | State intervention at update 80 |
|---|---|
| Keep | Retain m, v, and clock. All future rewards are nevertheless corrected. |
| Reset all | Set m=v=0 and restart the optimizer clock. |
| Reset m | Set only m=0; retain v and clock. |
| Reset moments, keep clock | Set m=v=0; keep global clock, exposing bias-correction effects. |
| Reset clock | Keep moments and reset only the clock. Adam-Rel-inspired control, not a full reproduction of its benchmark protocol. |
| Repair m | Replace only historical first-moment contributions. Mechanism ablation. |
| Repair m and v | Apply both replacement equations. Proposed intervention. |
| Refresh at current weights | Rescore the same tape with gradients at current weights, with clipped importance weighting. State-only, equal tape length; a practical staleness control, not an exact PPO reproduction. |

Seeds 9100–9107 are fixed for all three regimes. Seed 9099 is a separate runtime and implementation smoke check, excluded from scientific aggregation. No seed, learning rate, horizon, corruption fraction, or threshold may be adjusted after viewing the smoke or main results.

Before interpreting performance, verify the replacement equations against forward EMA replay to tolerance 1e-12, unchanged intervention weights, nonnegative second moments, and the no-change identity. Check that the declared target labels changed and that sampled rewards were actually revised. These are input and instrument checks; a favorable gradient or recovery trajectory is not a validity condition. Preserve all seeds and all trajectories.

**Predeclared decision rule.** For *each* sparse and dense regime, repair-mv must improve mean recovery success by at least 0.01 over *each* of the six baseline arms (all arms except itself and repair-m), and show a positive paired difference in at least 6/8 seeds against each baseline. The no-change repair-mv and keep trajectories must match within 1e-12. Failure of an instrument check means INVALID. Failure of any scientific comparison means NEGATIVE for this strong screen, not partial confirmation disguised as success. The m-only ablation is descriptive. These are screening thresholds, not a claim of conventionally significant population-level effects; no p-value is inferred from 6/8 signs.

From the repository root, with its existing environment:

```sh
uv run --python .venv python experiments/exp278_verifier_moment_repair.py --self-test
uv run --python .venv python experiments/exp278_verifier_moment_repair.py --smoke --output experiments/outputs/exp278_smoke.json
uv run --python .venv python experiments/exp278_verifier_moment_repair.py --workers 2 --output experiments/outputs/exp278.txt
```

The run is CPU-only and makes no model API calls. The bounded work is 24 seed–regime jobs, each with 80 shared training updates and eight 30-update continuations. Repair additionally rescans 20 groups per applicable arm; those passes must be reported separately. Equal future rollouts do **not** mean equal total computation. Record measured runtime before treating the screen as cheap in practice. The output contains configuration and script hashes, intervention checks, per-seed trajectories, and baseline contrasts.

**Results and interpretation.** The main run used all eight fixed seeds and all three regimes. Its measured simulation wall time was 3.010 seconds, excluding JSON serialization and process startup. There were 7,680 optimizer updates, 983,040 sampled action decisions, 480 extra fixed-path gradient evaluations, and 480 extra current-weight gradient evaluations. No GPU or paid model call was used. Raw data, including the logged batches, states, exogenous uniforms, and per-context trajectories, are in `experiments/outputs/exp278.txt`; the excluded smoke is preserved separately.

| Mean true success over the 30 recovery updates | Sparse corruption | Dense corruption | No-change control |
|---|---:|---:|---:|
| Keep | 82.703% | 79.809% | 85.113% |
| Reset all | 84.685% | 83.071% | 85.936% |
| Reset m | 82.635% | 80.733% | 84.106% |
| Reset moments, keep clock | **86.511%** | **85.125%** | **87.533%** |
| Reset clock | 81.838% | 78.940% | 84.275% |
| Repair m | 83.854% | 82.167% | 85.113% |
| Repair m and v | 83.871% | 82.214% | 85.113% |
| Refresh at current weights | 83.882% | 82.239% | 85.085% |

The proposed repair beat keep by 1.169 and 2.405 percentage points, respectively, with 8/8 positive differences in both regimes. It lost to full reset by 0.814 and 0.857 points, with 0/8 positive differences in both. It also lost to reset-moments/keep-clock by 2.640 and 2.912 points, again in every seed. Relative to current-weight refresh it was lower by 0.011 and 0.026 points, with 0/8 and 1/8 positive differences. It passed only three of the six primary comparisons in each regime: keep, reset-m, and reset-clock. The conjunction therefore fails decisively.

The no-change repair and keep trajectories were exactly identical. Maximum error across the recorded formula/replay/intervention checks was 3.47e-17. Both-moment repair exceeded m-only repair by just 0.017 and 0.047 points; these descriptive differences do not establish that second-moment correction is necessary. Reset-moments/keep-clock also improves the no-change case, so its advantage cannot be attributed uniquely to removing erroneous reward history. Learning-rate and bias-correction effects remain plausible explanations; they were not separately identified. There was no hyperparameter tuning after the excluded smoke or main results. Repository validation required adding the explicit labels `Hypothesis:` and `Falsifier:` to the already-written hypothesis and falsifying rule. The subsequent documentation-only rerun preserved all scientific records exactly; the verification record includes their hashes and both measured runtimes.

There is an additional analytic limit to this toy: in a linear softmax policy, the probability term in the fixed-batch score gradient cancels because the group advantages sum to zero. Conditional on the sampled actions and advantages, the gradient is independent of the historical probabilities. Consequently the current-weight refresh differs here through its importance weights, and this screen cannot identify neural parameter-path staleness. The negative recovery comparison remains valid, but neither exact moment replay nor the refresh comparison demonstrates that historical neural checkpoints are useful. A nonlinear adapter is needed to test that part of the transfer argument.

**Verdict: NEGATIVE / NEW INSIGHT at this toy design.** A blinded verifier independently recomputed the comparisons from raw trajectories and agreed; its record is `experiments/outputs/exp278_verification.md`. The bookkeeping works, and retaining stale state is an inadequate sole baseline, but the proposed intervention does not justify its complexity over simple resets. This is a scoped optimization result, not proof that all possible reward-history repairs fail. It supplies no evidence of LLM improvement. The next LLM experiment below is a reproducible specification for a future investigation; the current result does not license launching it as a promising continuation.

**The minimum LLM bridge, conditional on the screen.** A positive mechanistic screen would license a small follow-up; it would not license training at scale. Use one fixed open-weight model of roughly 0.5–1.5B parameters with a small trainable adapter and one-update-per-group policy-gradient training. Use generated integer-arithmetic tasks with an exact external answer checker. Deliberately supply a training checker with one specified bug, such as accepting absolute-value equality, then restore exact signed equality. Keep true evaluation separate from the training checker. Use disjoint operands and templates for final evaluation and exclude both those tasks and their outputs from repair-tape construction.

Use eight independent training seeds and the same 60/20/30 schedule initially, with 8 prompts × 4 responses per update and at most 96 generated tokens. Retain the 20-update tape and the historical adapter checkpoints plus optimizer-input gradients. Recompute corrected gradients by replaying the original forward/backward computations, including RNG state, then perform the state-only intervention. Do not assume logits alone reconstruct a neural-network gradient. All compared arms must receive identical corrected labels. Include keep, full reset, clock reset, m-only repair, both-moment repair, and current-weight refresh; also compare spending repair time on ordinary corrected-verifier training or corrected-buffer replay. Report both rollout-matched and total-time-matched endpoints.

That design has an upper bound of 6.39 million generated training tokens for eight seeds and six 30-update continuations sharing each 80-update training prefix: 8 × (80 + 6 × 30) × 8 × 4 × 96. Evaluation and extra repair computation are additional. This is a workload estimate, not a runtime or price quote. Start with a single excluded smoke seed and cap a device run at 30 minutes; estimate the remaining workload from measured tokens/second and backward-pass time. The delivered executable implements the cheaper mechanistic screen; this LLM bridge is a conditional next experiment, not completed work or a ready-made LLM training harness.

Full-history neural replay can be expensive: for P trainable parameters and K retained updates, checkpoint plus gradient storage alone is approximately 8KP bytes in fp32, excluding activations, tokens, optimizer snapshots, and metadata. For K=20 and P=1 million this is about 160 MB, but cost grows linearly with adapter size and backward work. Trace retention may therefore erase the proposed method's benefit. The cache format and replay machinery are engineering requirements, not free inputs.

For a bridge claim, predeclare at least +1 percentage point mean verified recovery accuracy versus the best independently tuned baseline, a positive paired effect in at least 6/8 seeds, and no more than 1 point loss on unaffected tasks. Also require a positive effect under equal total compute. Add a long-corruption condition to expose finite-tape limits, and two independent verifier bug families before calling the result robust. If a reset, refresh, or ordinary replay matches it within that margin, abandon the proposed advantage. If m-only repair matches both-moment repair, remove the stronger claim that second-moment replacement is necessary.

**Assessment boundary.** There is a concrete, executable, narrowly differentiated hypothesis here, with substantial adjacent prior art. There is no justified claim yet of a genuinely new general RL method, improved LLM reasoning, autonomous verifier improvement, or scalable RSI. The best possible cheap result is evidence to justify a controlled LLM test. A failed cheap test is a reason to stop this version, not to rename it or quietly change its falsifier.
