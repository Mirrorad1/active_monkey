# RESEARCH — mathematical analysis + frontier-research scout for the active-inference language agent

**Date:** 2026-06-08
**Author:** research subagent (parallel to the autonomous experiment loop)
**Scope:** formalize the current `active-loop` character HMM; survey frontier active-inference work relevant to the moonshot; give prioritized, honest recommendations.

> **Honest one-line thesis.** The current system is a *first-order* discrete HMM over characters. Its empirically-found wall (EXPERIMENTS.md Exp 3–7) is **not capacity** (number of latent states `K`) but the **first-order Markov assumption**: the next-character distribution is conditioned on a single latent state that summarizes *only the present*, not the recent past. The lever — confirmed empirically and provable from conditional-entropy arguments — is **temporal/context depth**. The credible research path is *hierarchical, structure-learning, deeper-memory* active inference. The credible *ceiling* of the pure-discrete path is a context-limited character/word babbler with measurable, interpretable cognition — not a conversational English speaker.

---

## 1. The math of the current system

### 1.1 The generative model (as built)

Files: `active_loop/lang_model_spec.py`, `active_loop/lang_model.py`, `active_loop/alphabet.py`.

- **Alphabet / observations.** One observation modality: the current character, with `V = 28` outcomes (`a–z`, space, period). `o_t ∈ {1,…,V}`.
- **Hidden states.** One factor `s_t` with `K` latent "meaning" states (`K = 14` in the spec; experiments swept 12/30/60).
- **Control.** One trivial control factor with a single no-op action (`_NOOP`). pymdp's `Agent` requires a control factor; M3 has no real actions — it is *passive perceptual learning*.
- **Tensors** (batched, batch = 1):
  - Emission `A ∈ Δ^{V×K}`, column-stochastic: `A[o,s] = P(o_t = o | s_t = s)`.
  - Transition `B ∈ Δ^{K×K×1}`, column-stochastic: `B[s',s,0] = P(s_t = s' | s_{t-1} = s)`.
  - Initial `D ∈ Δ^K`: `D[s] = P(s_1 = s)` (uniform).
- **Dirichlet priors over the parameters** (this is what makes it *learn*):
  - `p(A) = ∏_s Dir(a_{·s} ; α^A_{·s})`, with `pA = A_CONC · 1` (`A_CONC = 0.1`).
  - `p(B) = ∏_s Dir(b_{·s} ; α^B_{·s})`, with `pB = B_CONC · 1`.

So the **full generative model** of an observed character stream `o_{1:T}` is

```
p(o_{1:T}, s_{1:T}, A, B) = p(A) p(B) · D[s_1] · ∏_{t=2}^{T} B[s_t, s_{t-1}] · ∏_{t=1}^{T} A[o_t, s_t].
```

This is a **Bayesian HMM**: a categorical HMM with conjugate Dirichlet priors over the emission and transition matrices. "Hidden states are meaning" (the design's framing) = the latent cause `s_t`; "A is meaning→character" = the emission. The key structural fact, returned to in §1.5: **`s_t` depends on the past only through `s_{t-1}` (a first-order Markov chain), and `o_t` depends on the past only through `s_t`.**

### 1.2 Variational free energy (the objective being minimized)

Active inference minimizes the **variational free energy (VFE)** `F[q]`, an upper bound on negative log model-evidence (surprise). For one factor with categorical posterior `q(s_t)` (single-step / filtering form, which is what `infer_states` computes), pymdp's `calc_vfe` (`pymdp/maths.py:448`, line 576) evaluates exactly:

```
F_t = −H[q(s_t)]                         (negative entropy of the posterior)
      + E_{q(s_t)}[ −log prior(s_t) ]      (cross-entropy to the empirical prior)
      − E_{q(s_t)}[ log P(o_t | s_t) ]     (negative expected log-likelihood = −accuracy)
```

i.e. in the canonical "complexity − accuracy" reading,

```
F_t  =  D_KL[ q(s_t) ‖ prior(s_t) ]  −  E_{q(s_t)}[ log A[o_t, s_t] ]
     =  complexity                    −  accuracy.
```

Minimizing `F` pulls `q(s_t)` toward the prior (cheap beliefs) **and** toward states that explain the observed character (accurate beliefs). At the optimum, `F` equals the negative log-evidence `−log P(o_t | history)` plus the posterior KL gap; for an exact single-factor update the gap is ~0 and `F_t ≈ −log P(o_t | o_{<t})` — the **predictive surprise**.

The repo's metric is exactly this surprise, computed as a robust proxy in `LangModel.mean_surprise` (`lang_model.py:33`):

```
pred = A · prior(s_t);   surprise_t = −log pred[o_t];   metric = mean_t surprise_t / ln 2   (bits/char).
```

The uniform baseline is `log₂ V = log₂ 28 ≈ 4.807` bits/char (`eval/lang_score.py`). Exp 1 measured held-out `4.81 → 4.00` bits/char: free energy genuinely falls as the model learns — "the agent feeling better as it understands," with zero labels. **This intrinsic free-energy drop is the grounded valence signal** the M4 design builds on (`-F` = competence).

### 1.3 Inference (perception): fixed-point iteration

`LangModel` uses `inference_algo="fpi"` (one-step factorized fixed-point iteration; `pymdp/inference.py`, `_run_one_step_inference`). For the single-factor, single-modality case the FPI fixed point is the exact Bayesian filter update:

```
q(s_t)  ∝  prior(s_t) · A[o_t, s_t],         prior(s_t) = Σ_{s'} B[s, s'] q(s_{t-1}).
```

The prior is advanced through the no-op transition by `update_empirical_prior` (`agent.py:825`), called as `prior ← B · q(s_{t-1})` each step (`lang_model.py:42,53`). This is precisely forward filtering of an HMM.

### 1.4 Expected free energy (action) and parameter learning

**Expected free energy (EFE).** Although M3 has no real actions, the machinery (used heavily by the M1 controller and the M4 affective dyad) scores a policy `π` by its **expected free energy** `G(π)`, decomposed into epistemic + pragmatic value. pymdp computes `−G` ("neg_efe", `pymdp/control.py:611–621`) as

```
−G(π) = Σ_t [  info_gain(π,t)          (epistemic: expected state information gain)
             + utility(π,t)            (pragmatic: expected log-preference, E_{qo}[C])
             − neg_param_info_gain ].  (parameter epistemic value: Dirichlet info gain over A,B)
```

with the **state epistemic term** (`compute_info_gain`, `control.py:388`):

```
info_gain = H[ q(o|π) ]  −  E_{q(s|π)}[ H[ A(·|s) ] ]   =  expected reduction in observation uncertainty,
```

and the **pragmatic term** `utility = Σ_m E_{q(o_m|π)}[ C_m ]` (`compute_expected_utility`, `control.py:422`). The optional **parameter information gain** (`calc_negative_pA/pB_info_gain`, `control.py:450,493`) is the *curiosity to learn the model itself* — directly relevant to a "baby chooses what to read" curriculum (§3, rec. 5). Action is then `a* = argmin_π G(π)` (here `action_selection="deterministic"`). This is the formal seat of the "ASK only when it matters" behavior (M1) and "ASK when intent is ambiguous" (M4): asking wins when belief entropy is high enough that the epistemic term dominates.

**Dirichlet parameter learning (`infer_parameters`).** This is the inner, fast learning timescale. The VFE-minimizing posterior over `A` is itself Dirichlet, with concentrations incremented by the expected sufficient statistics (`pymdp/learning.py:11–55`, `119–157`):

```
A update:   α^A*  =  α^A_0  +  κ · Σ_{t} o_t ⊗ s_t          (s_t = q(s_t),  o_t one-hot)
B update:   α^B*  =  α^B_0  +  κ · Σ_{t} s_t ⊗ s_{t-1} ⊗ a_{t-1}
```

and the point estimate used downstream is the Dirichlet expected value `A = E[a] = α/Σα` (`dirichlet_expected_value`). In words: **every time the model believes state `s` emitted character `o`, it adds a pseudo-count to `α^A[o,s]`; every believed transition `s'←s` adds a pseudo-count to `α^B[s',s]`.** This is exact conjugate Bayesian counting in the latent space — the active-inference analogue of "smoothed n-gram counting," but over *inferred* states rather than observed symbols. `learn_stream` (`lang_model.py:45`) accumulates `q(s_t)` for the whole pass, then applies one `infer_parameters` step per epoch.

### 1.5 Why a first-order HMM cannot capture order / long-range dependence (formal)

This is the crux finding (EXPERIMENTS.md Exp 3–7). Three complementary arguments:

**(a) The Markov factorization throws away history.** In the model of §1.1, the only path from the past to the next observation is `o_{<t} → s_{t-1} → s_t → o_t`. By the d-separation property of the chain, conditioning on `s_{t-1}` makes `o_t` independent of everything before it:

```
P(o_t | o_{<t})  =  Σ_{s_t} A[o_t, s_t]  Σ_{s_{t-1}} B[s_t, s_{t-1}]  P(s_{t-1} | o_{<t}).
```

The entire history is compressed into the *belief vector* `P(s_{t-1} | o_{<t}) ∈ Δ^K`. Crucially, the **predictive distribution is a fixed linear-then-marginal map of that belief** — there is no mechanism that depends on *which characters* produced the belief beyond what the `K`-simplex can encode. Two different histories that drive `s_{t-1}` to the same (or nearly the same) posterior produce the same prediction. That is exactly the failure mode Exp 6 names: "1 char of context can't distinguish 'm mid-word' from 'm starting the answer'."

**(b) The repeated-letter impossibility is exact.** Consider training on `"mirro "` (Exp 3, 6). In a learned model the posterior after emitting `r` collapses toward the cluster of states whose emission is peaked on `r`. From *that* belief the model must predict the next character — but in `"mirro"`, the symbol `r` is followed by `r` once and by `o` once. A first-order model has a **single** predictive distribution available after "having just emitted an `r`," so the best it can do is the marginal `P(next | last='r') = ½·r + ½·o`. The numerical check run for this report confirms it exactly:

```
After 'r'  (1-char context):  {r: 50, o: 50}     # H = 1.0 bit — irreducibly ambiguous
After 'ir' (2-char context):  {r: 50}            # H = 0   — resolved
After 'rr' (2-char context):  {o: 50}            # H = 0   — resolved
```

No value of `K` fixes this (Exp 4's negative result): adding states adds *resolution of the present emission*, not *memory of the previous one*. Capacity ≠ memory.

**(c) Conditional-entropy / order-of-the-source argument.** Natural language as a symbol source has an entropy *rate* `h = lim_{n→∞} H(o_t | o_{t-n…t-1})` that keeps falling as context `n` grows (empirically `h ≈ 1.1` bits/char for English; Shannon's bound). A model with memory depth `d` can at best achieve the order-`d` conditional entropy `H(o_t | o_{t-d…t-1})`, which for `d = 1` (a first-order HMM's *effective* order over observations) is strictly larger than the order-`d` value for any `d > 1` whenever the source has dependencies beyond lag 1 — which language massively does. So the **bits/char floor of a first-order character HMM is the order-1 conditional entropy of the corpus**, well above both the trigram value and the ~1.1-bit Shannon estimate. The lever to *lower the floor* is to **increase the model's effective Markov order** — i.e. give it memory.

**The threshold (Exp 5–7).** Making the state carry the recent context — `s_t = (last char)` (bigram, 1-char memory) or `s_t = (last 2 chars)` (trigram, 2-char memory) — raises the effective order. Exp 7's count-based control isolated the variable: `n=2 → "miro"` (drops an `r`, still ambiguous), `n=3 → exact "mirro"` and `"name." → "mirro"` (question evokes answer), `n=4 → no further gain at this scale`. **Two characters of context is the switch-on point** for exact word order *and* simple Q→A at this corpus scale. This is the empirical target the AIF model must hit.

> **Important nuance for the loop.** This does *not* mean "an HMM is order-1 forever." A *latent* HMM can in principle encode some history in `s_t` if `B` and the inference drive states to specialize as context-carriers. But with `A` and `B` both free and learned greedily on a tiny corpus, the model has no pressure to allocate its `K` states as a context buffer — it allocates them to emissions. The reliable, *engineered* way to get effective order `d` is to **make the state literally be the last `d` symbols** (a deterministic `A` that reads off the most-recent symbol, and a learned `B` = the `d`-gram transition). That is exactly the priority experiment in EXPERIMENTS.md's open threads, and it is correct.

---

## 2. Frontier-research scout

Honest labels: **[established]** = peer-reviewed, reproduced; **[emerging]** = recent, promising, not yet broadly validated; **[speculative]** = direction/claim, not demonstrated at this task.

### 2.1 Hierarchical & deep temporal active inference — *the most direct precedent* [established → emerging]

The single most relevant precedent is **Friston, Parr, de Vries et al., "Generative models, linguistic communication and active inference"** (Neurosci. Biobehav. Rev., 2020; PMC7758713). It builds **exactly the structure this repo needs next**: a *two-level* generative model where a **slow higher level** holds narrative/semantic state (prompt/question/answer, question-type, scene knowledge) and a **fast lower level** generates *word sequences* constrained top-down by the higher level. The key mechanism — **"deep diachronic" / separation of timescales**: "states at the higher level change slowly… [they] remain the same throughout a sequence of state transitions at the lower level." Word order and context-sensitivity come from the higher level *constraining* lower-level transitions, and meaning is **factorized** ("the form of a question and its content separately") to fight combinatorial explosion. It demonstrates question→answer behavior in a "Twenty Questions" world — i.e. the *same Q→A capability* Exp 7 reached with trigrams, but principled and compositional. **Honest limitation (their own):** tiny domains (3 question types, binary answers), and crucially *the deep model is hand-specified, not learned from scratch* — "the acquisition of language through learning deep models" is named as missing. This paper is the blueprint for §3 rec. 2 (words-above-characters) and the warning for §4.

Supporting/parallel work: **"Deep temporal models and active inference"** (Friston et al., 2018; PMC5998386) formalizes hierarchical HMMs with nested timescales; **"Generative models for sequential dynamics in active inference"** (2024; PMC11655747) surveys discrete-sequence models for cognition where "cognitive operations require processing of discrete sequences of items… sequences of words during linguistic communication."

### 2.2 Renormalizing Generative Models (RGM) / scale-free active inference [emerging]

**Friston et al., "From pixels to planning: scale-free active inference"** (Front. Network Physiology, 2024–2025; arXiv 2407.20292). RGMs are "discrete homologues of deep convolutional neural networks," built by a **renormalization-group operator** that coarse-grains a level into a slower, more abstract level "retaining the properties of interest while discarding irrelevant details," assuming **self-similar structure across scales**. They "learn compositionality over space and time, furnishing models of paths or orbits — events of increasing temporal depth and itinerancy," and generalize POMDPs to include *paths* as latent variables. This is the most ambitious current answer to "how do you get temporal depth and word-above-character compositionality in a *discrete* AIF model without exponential blow-up": **fixed-vocabulary coarse-graining + slower clocks per level.** Caveat: heavy machinery, largely demonstrated on vision/games, and the "self-similar across scales" prior is a strong assumption for language. Treat as the long-horizon north star for the hierarchical path.

### 2.3 Structure / model learning (learn the model, don't fix `K`) [established for small models]

Two mechanisms matter directly:

- **Bayesian model reduction (BMR)** (Friston, Parr, Zeidman; arXiv 1805.07092): start with an *over-expressive* model and analytically *remove redundant parameters* by comparing reduced priors — a post-hoc, evidence-based simplification ("learning in the absence of new data," linked to sleep/"aha" moments). For this repo: train with generous `K` (or a generous context order), then **prune** unused states/transitions to find the *minimal sufficient* model — turning "what `K`?" from a hyperparameter sweep into an inference.
- **Structure learning / "fast structure learning," "supervised structure learning"** (arXiv 2311.10300; PLOS One 2022, "Structure learning enhances concept formation in synthetic active inference agents"): *grow* state factors / add states when the data demand it, framed as optimizing *priors over structure*. "Expected free energy… can be extended to cover expected information gain over models" — i.e. the agent can be curious **about its own architecture**. This is the principled version of the autopilot loop's job (see §3 rec. 4).

### 2.4 Discrete state-space scaling limits — the honest wall [established]

**"Active Inference in Discrete State Spaces from First Principles"** (arXiv 2511.20321) and the robotics survey (arXiv 2112.01871) state the core scaling fact plainly: planning/inference in discrete POMDPs is **exponential in both the time horizon and the state-space size**, and "the restriction of using discrete categorical distributions… renders operating in any kind of continuous/fine-grained environment unfeasible." Higher-order memory via "make the state encode the last `d` symbols" costs `O(V^d)` states — `V^2 = 784` is fine, `V^3 = 21,952` is borderline, `V^4 ≈ 614k` is not, for full categorical tensors. **Mitigations named:** factorize the state (independent sub-factors), **sparse transition tensors**, hierarchy with slower upper clocks. Separately, work probing LLMs on discrete-state tasks (arXiv 2402.00795) found even LLaMA-70B "can only learn up to 9 discrete states," a sobering reminder that *discreteness itself* is hard at scale — for any architecture.

### 2.5 Free-energy valence, precision/attention, self-evidencing [established]

- **Self-evidencing**: self-organizing systems act "to achieve consistency between internal model and external world" (Friston, "FEP made simpler but not too simple," 2023). This is the formal license for the M4 claim that **intrinsic valence = `−F`** (competence/understanding) needs no teacher.
- **Emotional valence and the FEP** (Joffily & Coricelli, PLOS Comput. Biol. 2013): valence tracks the **rate of change of free energy** — *improving* prediction feels good, *worsening* feels bad. This is a concrete, implementable refinement of "valence = `−F`": use `−dF/dt`, which is differentiable, sign-meaningful, and exactly the trajectory Exp 1 (4.81→4.00) traces. **Recommended grounding for M4** (§3 rec. 6).
- **Precision/attention as inference** (active-inference attention models, arXiv 2505.03856; Parr & Friston): attention = optimizing the *precision* (inverse variance / confidence) of specific likelihood or prior terms. In this repo, precision-weighting is the principled control for "how much to trust the current prediction vs. seek context" — and the seat of the M1/M4 "ASK when uncertain" decision.

### 2.6 Active inference as a layer over LLMs [emerging/speculative]

**"Active Inference for Self-Organizing Multi-LLM Systems"** (arXiv 2412.10425) and **"Predictive Minds: LLMs as atypical active inference agents"** (arXiv 2311.10215): use AIF as a *cognitive control layer* that adapts prompts/strategies by EFE, restoring the "internal-model-update channel" that pure LLM agents lack. This is *not* the moonshot path (it borrows a pre-trained language model, violating "opinions never pre-trained"), but it is the honest, pragmatic bridge if conversational ability is ever needed *before* the from-scratch model reaches it (§4).

---

## 3. Concrete, prioritized recommendations for this repo

Each: **mechanism (math) · benefit · pymdp feasibility · honest risk/ceiling.** Ranked by expected value-per-effort toward the moonshot.

### Rank 1 — Build the AIF 2-char-context (trigram) model *within active inference*
*(already the #1 open thread; this report confirms it is correct and gives the exact construction)*

- **Mechanism.** Redefine the hidden factor as a **pair-state** `s_t = (c_{t-1}, c_t)`, so `K = V² = 784`. Make `A` **deterministic and frozen**: `A[o, (c_{prev}, c_cur)] = 1` iff `o = c_cur` (the state *is* the last two chars; reading the current char off it is exact). Then **learn only `B`**, the trigram transition `B[(c_cur, c_next), (c_prev, c_cur)] = P(c_next | c_prev, c_cur)` — note `B` is *structurally sparse*: from pair `(c_prev,c_cur)` only the `V` pairs `(c_cur, ·)` are reachable, so each column has ≤ `V` nonzeros. This is exactly the count-based trigram of Exp 7, *expressed as a Bayesian-Dirichlet HMM* so the same `infer_parameters` machinery does the learning.
- **Benefit.** Exp 7 proves this hits the target: exact `"mirro"` and `"name." → "mirro"`. It demonstrates **comprehension-as-prediction inside active inference** (the whole point), and keeps free energy as the only objective.
- **pymdp feasibility.** High. `K = 784` is tractable; `A` is a fixed deterministic likelihood (set `learn_A=False`); `B` learned with Dirichlet as now. The encode/decode in `alphabet.py` extends to pair-indexing trivially. The generation loop in `lang_model.py:64` already samples state→char→next-state; only the state space changes. **This is a few-hours build on the existing `LangModel`.**
- **Risk/ceiling.** `V²` is fine; `V³` (3-char memory) is `21,952` — still tractable with sparse `B`; `V⁴` is not, with dense tensors. So this approach **caps at ~3-char memory** before you must factorize or go hierarchical (rec. 2). It will *not* produce sentences — only locally-correct fragments and short memorized Q→A. That is the honest, measurable next rung, not the moonshot.

### Rank 2 — A hierarchical "words-above-characters" layer (two timescales)
*(the single highest-leverage architectural move; directly the PMC7758713 blueprint)*

- **Mechanism.** Add a **slow upper factor** `w_t` (latent "word/chunk" state, `K_w` states) whose transitions are slow, and a **fast lower factor** (the character model of rec. 1). The upper state **conditions** the lower transition: `B_lower` becomes `P(c_next | c_cur, w)` — i.e. *which* character sequence is generated depends on the current chunk. The upper level runs on a slower clock (one `w`-transition per several char-steps), giving **effective memory far beyond the character window** without `O(V^d)` blow-up. This is the "deep diachronic" separation-of-timescales mechanism (§2.1) and the discrete homologue of RGM coarse-graining (§2.2).
- **Benefit.** This is the structural step from *fragments* to *words and word-order*, and the natural place for **emergent "concepts"**: `w` states are unlabeled chunks the model discovers — the first place something like a proto-"opinion/disposition" could live (a slow state that biases what it says). It is also where M3 (language) and M4 (intent) *unify*: an upper "intent/topic" factor over a lower "characters" factor.
- **pymdp feasibility.** Medium. pymdp's JAX agent supports multi-factor models and factor-conditioned transitions (`B_dependencies`), and `mmp`/`vmp` sequence inference exists for deep temporal inference. The friction is wiring the two-timescale clocking (upper updates every `L` lower steps) and the conditioned `B` — non-trivial but supported. Start with `w` *given* (e.g. word boundaries = spaces) to validate, then learn it.
- **Risk/ceiling.** This is real research, not a config tweak; learning the upper level *from scratch* (vs hand-specifying it, which the precedent does) is the hard, partly-open part. Honest ceiling: short, structured utterances and topic-conditioned generation — a long way below conversation, but a genuine qualitative jump and the right direction.

### Rank 3 — Learn the model structure instead of fixing `K`/order (BMR + structure learning)
*(turns the autopilot's job from hyperparameter-poking into principled inference)*

- **Mechanism.** Two parts. **(a) BMR:** train an over-complete model (large `K` / order), then analytically compare reduced priors that zero-out states/transitions, keeping the reduction iff it *increases model evidence* (lowers free energy with a complexity saving) — `ΔF` is closed-form for Dirichlet/categorical models. **(b) Structure growth:** add a state/factor when expected free energy over *models* (information gain about structure) says the data demand capacity. Score with the **same** held-out bits/char the loop already uses.
- **Benefit.** Removes the brittle "what `K`?" and "what context order?" sweeps (Exp 4 wasted capacity; this finds the *minimal sufficient* memory automatically), and makes the autopilot loop a genuine **structure-learning** agent — the design's stated aspiration. Directly answers "learn the model structure rather than fixing K."
- **pymdp feasibility.** Medium. BMR for Dirichlet priors is a small standalone computation (no new pymdp internals needed — operate on `pA`/`pB` and `calc_vfe`). Structure *growth* (adding factors mid-run) is harder in a JAX/`equinox` static-shape world (re-instantiate the agent on growth, which the loop's keep/revert flow already accommodates). The MUTABLE surface (`lang_model_spec.py`) is the right place for the loop to propose these.
- **Risk/ceiling.** BMR is well-established and low-risk; full online structure-growth is emerging and finicky under JAX shape constraints. Ceiling: better-sized models and a more principled loop — a multiplier on rec. 1/2, not a moonshot in itself.

### Rank 4 — Make the autopilot loop optimize *temporal depth* as its primary lever
*(re-point the existing keep/revert loop at the variable that actually matters)*

- **Mechanism.** The loop currently mutates `K`, `A_CONC`, `B_CONC` (`lang_model_spec.py`). Per §1.5, the dominant lever is **context order `d`**, not `K`. Add `d` (and the rec.-1 pair-state construction) to the MUTABLE surface and let the loop search `d ∈ {1,2,3}` against held-out bits/char, with a complexity guardrail (reject `d` increases that don't beat baseline by a margin, to avoid `O(V^d)` bloat). Pair with rec. 3's BMR so the loop *earns* depth rather than guessing it.
- **Benefit.** Aligns the autonomous search with the proven lever (Exp 5/7) instead of the disproven one (Exp 4). Cheap, high-signal.
- **pymdp feasibility.** High — it is a metric/surface change plus rec. 1's model. The loop's branch→test→score→merge flow already exists.
- **Risk/ceiling.** Risk is metric-gaming via depth (memorizing tiny corpora); mitigate with held-out split (already present) and the critic. Ceiling = whatever rec. 1/2 ceilings are; this just gets there autonomously.

### Rank 5 — Curiosity / active reading via parameter information gain (M3.5)
- **Mechanism.** Use the **parameter epistemic value** already in pymdp (`calc_negative_pA/pB_info_gain`, `control.py:450,493`): let the agent *choose which corpus segment to read next* by maximizing expected Dirichlet information gain about `A`/`B`. "The baby chooses what to attend to," made literal — *active* inference over its own data diet.
- **Benefit.** Faster, sample-efficient learning; turns passive streaming into active inference; a real demo of epistemic drive.
- **pymdp feasibility.** Medium — the EFE term exists; needs a small action space over "next segment." A clean M3.5.
- **Risk/ceiling.** Helps efficiency, not the fundamental order ceiling. Do after rec. 1.

### Rank 6 — Ground valence in `−dF/dt`, not just `−F` (M4 refinement)
- **Mechanism.** Per Joffily & Coricelli (§2.5), set the intrinsic valence readout to the **rate of free-energy reduction** `−dF/dt` (improvement = positive) rather than the level `−F`. Then bootstrap the extrinsic `+`/`−` cue onto it via Dirichlet learning (the M4 grounding fix), as designed.
- **Benefit.** A differentiable, sign-meaningful, theory-backed valence — exactly the "learns to feel positive" trajectory, with a citation. Small change to `valence_readout`.
- **pymdp feasibility.** High — `infer_states(..., return_info=True)` already returns per-step VFE; differencing it is trivial.
- **Risk/ceiling.** Functional valence only (the design is honest about this). No sentience claim.

---

## 4. Honest assessment — how far can the discrete-AIF path go?

**What it can credibly reach.** With rec. 1–4, the system can become a **context-`d` character/word model whose every behavior — prediction, "understanding" (low `F`), short Q→A, topic-conditioned generation — is derived from free-energy minimization, with interpretable latent states and no pre-trained opinions.** That is a genuine, rare artifact: a *legible* cognitive system you can watch learn, with grounded intrinsic valence. The PMC7758713 precedent shows hierarchical discrete AIF can do question→answer and compositional word-sequence generation in small worlds. RGM (2407.20292) shows discrete hierarchies *can* in principle scale temporal depth and compositionality. So "structured, multi-level, emergent-state language behavior at small scale" is **reachable and worth building**.

**The fundamental barriers (stated plainly).**
1. **Exponential cost of depth in dense discrete models** (§2.4). Memory-by-state-enumeration is `O(V^d)`; you hit the wall by `d ≈ 4`. Escaping it *requires* factorization / sparsity / hierarchy (rec. 2/3), which is real research, not tuning.
2. **From-scratch structure learning of deep language models is essentially open.** Every strong AIF language demo (PMC7758713) *hand-specifies* the hierarchy; "acquisition of language through learning deep models" is named as missing by the authors themselves. Learning the *upper* (word/concept/topic) levels purely from interaction is the crux unsolved problem — and it is exactly what the moonshot's "opinions emerge, never pre-trained" demands.
3. **The entropy gap.** Count/HMM-class models bottom out around ~1.46 bits/char on English; neural LMs reach ~1.12 (Shannon's human-estimated floor). A pure discrete AIF model lives on the n-gram side of that gap. Conversational fluency lives on the other side. Discreteness itself is hard at scale even for huge models (LLaMA-70B ≈ 9 states; §2.4).
4. **"Opinions/thoughts" as emergent slow states is plausible but undemonstrated.** The natural home for a proto-opinion is a *slow upper latent* (rec. 2) that biases generation — but showing that something a human would call an "opinion" *emerges and stabilizes* from interaction alone is **[speculative]**, far beyond current results.

**A credible (long) roadmap.**
- **Now → weeks:** rec. 1 (AIF trigram, hits Exp-7 target inside AIF) → rec. 4 (loop optimizes depth) → rec. 6 (valence = `−dF/dt`). *Outcome:* exact short word order + Q→A + grounded valence, all from free energy. Solid, honest rungs.
- **Weeks → months:** rec. 3 (BMR/structure learning so the model sizes itself) → rec. 2 (two-timescale words-above-characters; start with given word boundaries, then learn them) → rec. 5 (active reading). *Outcome:* topic-/chunk-conditioned generation, the first emergent "concept" states, the M3⊕M4 unification (intent above characters).
- **Months → years (frontier, uncertain):** deepen the hierarchy toward RGM-style scale-free temporal compositionality; attack *learning* the deep structure from interaction (the open problem); only here does anything resembling "emergent disposition/opinion over conversation" become a *research question* rather than a build task. Likely needs mixed discrete-continuous models and sparsity to be tractable — and may ultimately argue for a *hybrid* (AIF control/valence/structure layer over a learned continuous sequence model, §2.6) rather than a purely discrete speaker.

**Bottom line.** The discrete-AIF path is the *right vehicle for the honest, legible, emergent-cognition part of the moonshot* — grounded valence, interpretable latent meaning, structure that the agent itself learns and prunes — and it can reach genuinely interesting small-scale language behavior. It is **not** the vehicle for conversational-English fluency on its own; that requires either escaping discreteness (continuous/hybrid sequence models) or borrowing a pre-trained language model as a substrate (which trades away the "never pre-trained" purity). The intellectually honest moonshot is therefore likely a *hybrid*: keep active inference as the seat of valence, curiosity, self-evidencing, and emergent slow-state "dispositions," layered over a more expressive sequence substrate — with the discrete rungs above as the proving ground that the cognition is real and emergent, not bolted on.

---

## Sources

- Friston, Parr, de Vries, et al. — *Generative models, linguistic communication and active inference* (Neurosci. Biobehav. Rev., 2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7758713/  *(hierarchical words-above-narrative; Q→A; the rec. 2 blueprint)*
- Friston et al. — *Deep temporal models and active inference* (2018). https://pmc.ncbi.nlm.nih.gov/articles/PMC5998386/
- *Generative models for sequential dynamics in active inference* (2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11655747/
- Friston et al. — *From pixels to planning: scale-free active inference* (RGM; 2024/2025). https://arxiv.org/abs/2407.20292 · https://pmc.ncbi.nlm.nih.gov/articles/PMC12217590/
- *Active Inference in Discrete State Spaces from First Principles* (2025). https://arxiv.org/html/2511.20321v1  *(discrete scaling limits)*
- *Active Inference in Robotics and Artificial Agents: Survey and Challenges* (2021). https://arxiv.org/pdf/2112.01871
- Friston, Parr, Zeidman — *Bayesian model reduction* (2018). https://arxiv.org/pdf/1805.07092
- *Supervised structure learning* (2023). https://arxiv.org/pdf/2311.10300
- *Structure learning enhances concept formation in synthetic active inference agents* (PLOS One, 2022). https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0277199
- Joffily & Coricelli — *Emotional valence and the free-energy principle* (PLOS Comput. Biol., 2013). https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003094  *(valence = −dF/dt; rec. 6)*
- Friston — *The free energy principle made simpler but not too simple* (2023). https://www.sciencedirect.com/science/article/pii/S037015732300203X  *(self-evidencing)*
- *An Active Inference Model of Covert and Overt Visual Attention* (2025). https://arxiv.org/pdf/2505.03856  *(precision/attention as inference)*
- *Active Inference for Self-Organizing Multi-LLM Systems* (2024). https://arxiv.org/abs/2412.10425  *(AIF layer over LLMs)*
- *Predictive Minds: LLMs as atypical active inference agents* (2023). https://arxiv.org/abs/2311.10215
- *LLMs learn governing principles of dynamical systems… in-context neural scaling law* (2024). https://arxiv.org/pdf/2402.00795  *(LLaMA-70B ≈ 9 discrete states)*
- *Cross Entropy of Neural Language Models at Infinity — A New Bound of the Entropy Rate* (2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7512401/  *(~1.12 bits/char)*
- Shannon entropy of English / Shannon-game estimates. http://mattmahoney.net/dc/entropy1.html

**Repo files grounding §1:** `active_loop/lang_model_spec.py`, `active_loop/lang_model.py`, `active_loop/alphabet.py`, `eval/lang_score.py`, `EXPERIMENTS.md`; pymdp internals `pymdp/maths.py` (`calc_vfe`, `compute_info_gain`), `pymdp/inference.py` (`update_posterior_states`), `pymdp/learning.py` (Dirichlet updates), `pymdp/control.py` (EFE decomposition), `pymdp/agent.py` (`infer_parameters`, `update_empirical_prior`).
