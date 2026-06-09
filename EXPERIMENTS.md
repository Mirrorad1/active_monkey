# active-loop — teaching experiments log

Autonomous research log toward the moonshot (an active-inference agent that learns
language from scratch). Each entry: a teaching experiment, what was tried, the honest
result, and what it implies. Append-only; newest at the bottom.

Honest framing: free energy = the reward (low surprise = "understanding"); hidden states
= meaning; intrinsic valence (prediction success) is primary, approval is grounded on top.

---

## Exp 1 — does the character HMM learn at all? (M3a baseline)
- Setup: K=12 first-order HMM, built-in English corpus, Dirichlet learn A/B.
- Result: held-out 4.81 → 4.00 bits/char. It learns: surprise drops, generation shifts
  from random to spacing/period/letter-cluster structure.
- Implication: the engine works — free energy falls as it learns. Intrinsic valence is real.

## Exp 2 — can it learn to choose actions that earn positive feedback?
- Setup: bandit; agent has a preference for a "positive" outcome, must learn which action yields it.
- Result: positive-feedback rate 0.90 → 1.00 over a session.
- Implication: preference + EFE + learning = "learns to seek positive" (functionally).
- Caveat: this injected a labeled "positive" — see the grounding fix (intrinsic valence first).

## Exp 3 — teach it to "say mirro" by repeated exposure
- Setup: K=12 first-order HMM, stream "mirro " repeatedly.
- Result: surprise 3.38 → 1.61; output became made of mirro's LETTERS (m,i,r,o,space) but
  jumbled order: "mo io riorrr". Got the ingredients, not the sequence.
- Implication: a first-order HMM learns the character palette but not the ORDER.

## Exp 4 — does MORE MEMORY (more states K) fix the order? (negative result)
- Setup: train "mirro " at K = 12, 30, 60.
- Result: NO improvement at any K — all jumbled ("rrrr imls", "rrrr imiv", "rrrr imiw").
  Question→answer ("what is your name." → "mirro.") also failed entirely.
- Implication (key finding): the wall is the FIRST-ORDER assumption, not capacity. More
  states don't add memory of recent context. Long-range structure (word order, Q→A) is
  unreachable for a plain HMM regardless of size.

## Exp 5 — does CONTEXT (state = last character) fix the order? (positive result)
- Setup: bigram-style model — state == current char (K=V=28), A≈identity, learn B = char→char.
- Result: taught "mirro" → "irro", "mihro", "mi" (mirro in order, minor r→r/r→o wobble);
  taught "the cat sat on the mat" → "tal.e th t cn tat.she sa" — real word-fragments:
  "the", "th", "tat", "sa", correct spacing/periods.
- Implication (the lever): MEMORY of recent context, not capacity, produces order and words.
  One char of memory (bigram) → fragments; two (trigram) would resolve "mirro" exactly.
  This is the direction toward the moonshot: grow the context window of the generative model.

## Exp 6 — bigram limits: repeated letters & word-context (sharpening result)
- Setup: bigram (state = last char), greedy (most-likely-next) decode, 12 epochs.
  (a) "mirro " repeated;  (b) Q→A "name. mirro. " then prompt "name. ".
- Result:
  (a) greedy from space → "mirr" then loops on r ("mirrrrrr"). Nails m-i-r-r, then can't
      tell if the 2nd r → r or → o (in "mirro" r is followed by both, equally). 1-char
      memory sees "just saw an r" identically in both cases.
  (b) "name. " → "me. me." — the bigram CONFLATES the two m's (na-M-e vs M-irro); 1 char of
      context can't distinguish "m mid-word" from "m starting the answer".
- Implication: 1-char memory is provably insufficient for (i) repeated letters and (ii)
  word/answer context. Minimum context depth = longest disambiguation needed. Confirms and
  quantifies Exp 5's lever: need ≥2-char context (trigram). This is THE next build.

## Exp 7 — context-depth control: how deep is enough? (decisive positive)
- Setup: count-based n-gram control (n=2,3,4) to ISOLATE the context-depth variable (not the
  AIF model). Greedy decode. (a) "mirro " ; (b) Q→A "name. mirro. " then "name. ".
- Result:
  (a) n=2 → "miro" (drops an r); n=3 (trigram, 2-char memory) → EXACT "mirro mirro";
      n=4 → no further gain.
  (b) n=2 → "me. me." (conflates the two m's); n=3 → "mirro. " — "name." correctly EVOKES
      the answer "mirro"; n=4 → no further gain.
- Implication (threshold found): TWO characters of context is the switch-on point for BOTH
  exact word order AND question→answer at this scale. Comprehension-as-prediction
  demonstrated: a question cue evokes the learned answer once memory ≥ the disambiguation span.
- Caveat: count-based control, not active inference. It SETS THE TARGET the AIF context model
  must match: a ≥2-char-context (trigram) active-inference generative model.

## Open threads for the loop to pursue
- **(priority) AIF 2-char-context model**: reproduce Exp 7 WITHIN active inference — pair-state
  (s = last 2 chars), deterministic A (pair→current char), learn B = trigram transitions.
  Tractable for tiny corpora (K=V*V=784; few chars/epoch). Goal: AIF says "mirro" exact + Q→A.
- Curriculum: teach short words first, then phrases (does staged exposure help / transfer?).
- Grounding bridge: bind an arbitrary feedback cue to the intrinsic free-energy valence.
- Capacity vs depth: confirm within AIF that DEPTH (memory), not state count, is the lever
  (Exp 4 showed states alone fail; Exp 5/7 show depth works).

## Exp 8 — AIF 2-char-context (pair-state) model: depth INSIDE active inference (positive)
- Setup: pymdp Agent with hidden state s=(prev,cur), K=V*V=784; near-deterministic emission
  A (state s emits char s%V); transition B structured (only (c,*) valid from (p,c)) then
  Dirichlet-LEARNED (learn_B). Greedy generative rollout: P(next char)=A·(B·belief), then
  condition belief on the emitted char. Reproduces Exp 7 as free-energy minimization, not counts.
- Result: taught "mirro " → cycles "...mirro mirro..." in EXACT order (only the cold-start
  1-char prime is ambiguous — "what preceded m?" is genuinely unknown). Q→A: after "name. "
  → "mirro." — the question→answer association is learned and recalled WITHIN the AIF model.
- Implication: RESEARCH rec #1 achieved. Temporal depth (2-char memory) works inside active
  inference; the emergent PAIR-STATES are literally "memory of recent context". Depth-as-lever
  confirmed in AIF (Exp 4 showed capacity K alone fails; Exp 8 shows learned context wins).
- Next: hierarchical words-above-characters (rec #2) — a slow chunk/intent factor over the fast
  char states; where word-order generalizes and emergent "concept/opinion" states could live.

## Exp 9 — long-range binding wall (honest negative; motivates hierarchy)
- Setup: flat AIF 2-char pair-state model trained on "sky is blue. grass is green. ";
  prime "sky is " vs "grass is ", greedy.
- Result: BOTH → "green" (identical). The flat model CANNOT bind subject→predicate — when
  predicting after "is ", the discriminating word (sky/grass, 5+ chars back) is outside the
  2-char window, so the two clauses blur into one.
- Implication (the real wall): any FIXED-order Markov model fails once a dependency spans more
  than its order. Long-range binding — subject→predicate, question→specific-answer,
  topic→opinion — REQUIRES a slow hierarchical "concept/topic" latent, NOT just deeper flat
  context. This is the boundary between toy n-gram behavior and conversation, and the precise
  formal motivation for rec #2 (hierarchy). "What do you think about X?" needs exactly this:
  a held topic-state conditioning the response — the seat of an emergent, non-pretrained opinion.
- Next: minimal 2-factor hierarchical model — a slow topic factor conditioning the fast char
  transitions; does it bind sky→blue, grass→green where flat context cannot?

## Exp 10 — hierarchy principle: a held topic binds long-range (positive, with caveat)
- Setup: topic-conditioned char transitions (the hierarchy idea) — a HELD "topic" latent
  conditioning the fast char model, realized here by per-topic transition models. Prime "is ".
- Result: topic=SKY → "blue."; topic=GRASS → "green." A held topic binds is→blue vs is→green
  exactly where flat fixed-order memory (Exp 9) collapsed both to "green."
- Implication: confirms hierarchy is the fix — a SLOW held latent conditioning FAST transitions
  achieves long-range binding no flat Markov order can. Direct line to the moonshot: "what do
  you think about X" = infer+hold topic X, then generate a response conditioned on it; that
  learned, self-derived conditioned response IS an emergent (non-pretrained) opinion.
- Caveat (the open frontier): topic was SUPERVISED/separated here. The hard part is making the
  topic EMERGE and be INFERRED from the subject ("sky"/"grass") unsupervised, and HELD across
  the clause — a genuine two-timescale generative model (slow factor inferred from the fast stream).
- Next: attempt EMERGENT topic — a 2-factor pymdp model whose slow factor is inferred from the
  fast char stream and persists (near-identity slow dynamics), binding without supervision.

## Exp 11 — emergent-topic scaffolding: 2-factor model constructs in pymdp (progress)
- Setup: probe whether pymdp supports the hierarchical model needed for an EMERGENT (inferred)
  topic — factor0 = topic (slow, near-identity B so it persists), factor1 = char, with
  B_dependencies=[[0],[1,0]] so the char transitions are CONDITIONED ON the topic. A depends
  only on the char factor (A_dependencies=[[1]]).
- Result: CONSTRUCTS and runs inference cleanly. Topic belief initializes [0.5, 0.5] (properly
  uncertain), char belief 28-dim. The multi-factor topic-conditioned machinery works here.
- Implication: the emergent-topic hierarchical model is BUILDABLE in pymdp (foundation laid) —
  not just a theoretical construct. This is the vehicle for "infer + hold the topic from the
  subject" that Exp 9/10 showed is required for long-range binding (and for opinions about X).
- Next (continuing across wakes — mid-task): full 2-factor TRAINING loop on mixed clauses
  ("sky is blue. grass is green."), then test whether the topic is INFERRED from the subject and
  HELD to bind is->blue vs is->green WITHOUT supervision. This unsupervised mixture-learning step
  is the genuinely hard, research-grade part; expect partial/negative results en route.

## Exp 12 — unsupervised emergent topic: FAILS to differentiate (honest negative; key insight)
- Setup: full 2-factor training (topic 2-state + char), B_dependencies=[[0],[1,0]] (char trans
  conditioned on topic), asymmetric B1 init, switchable B0. Trained UNSUPERVISED on mixed
  "sky is blue. grass is green."; tested binding via prime "sky is " vs "grass is ".
- Result: both → "s gree" (identical); topic belief stayed [0.5, 0.5] the ENTIRE time. Topic
  never differentiated → no binding (collapses to the flat-model failure of Exp 9).
- Implication (the crux of emergence): this is the mixture-learning LOCAL MINIMUM. Because the
  emission A does NOT depend on topic and the topic-conditioned transitions start symmetric,
  variational EM has no gradient to break symmetry — it settles on "ignore the topic". A latent
  concept needs SOME differentiating signal to crystallize; it will NOT bootstrap from pure
  symmetry. Directly relevant to the moonshot: "opinions/concepts that emerge unsupervised"
  cannot come from nothing — there must be a foothold (a cue, asymmetric grounding, curriculum,
  or structure-learning move) for a latent to attach to.
- Next moves to give the topic a foothold (try in coming cycles):
  (a) let emission depend on topic too (A_dependencies=[[0,1]]) so the SUBJECT chars inform topic;
  (b) reset/loosen topic at clause boundaries (period) so it re-infers per clause;
  (c) light semi-supervision to break symmetry, then test generalization;
  (d) structure learning / Bayesian model reduction (RESEARCH.md rec #3).

## Exp 13 — semi-supervised foothold still fails to yield test-time inference (negative)
- Setup: 2-factor (topic + first-order char), topic TEACHER-FORCED during training (each clause's
  topic pinned via the prior); at test, topic uniform and must be inferred from the subject.
  (Also fixed two bugs en route: script-on-/tmp lost the import path; teacher-forced prior dropped
  its batch axis — note for future: forced priors must be shape (1,T), not (T,).)
- Result: both "sky is "/"grass is " → "s is i"; inferred topic [0.5,0.5]; no binding; char output
  degraded too.
- Implication: scaffolding the topic in TRAINING does not by itself produce TEST-TIME inference.
  Root causes: (1) emission A is topic-INDEPENDENT, so the subject chars have NO direct pathway to
  move the topic belief — the only channel is transition-mismatch, which single-step FPI doesn't
  accumulate over the subject; (2) the first-order char factor is too weak a substrate. The topic
  needs a DIRECT evidential pathway (emission-level dependence) + a context-carrying char substrate.
- Meta (honest): Exps 11–13 confirm the predicted wall — emergent/unsupervised concept formation in
  this discrete framework is genuinely hard (matches RESEARCH.md's honest ceiling: from-scratch deep
  structure learning is largely open). Productive paths: (a) make the topic GROUNDED & directly
  inferable — A depends on topic so the subject word identifies it (this is grounding from
  experience, NOT pretrained opinions); (b) pair-state char substrate; (c) shift focus to fronts
  that show positive progress (valence-grounding bridge, curriculum) while keeping emergence as the
  long research arc.
- Next: Exp 14 — topic with EMISSION-level dependence (A_dependencies=[[0,1]]) over a pair-state
  char substrate, so the subject directly and durably identifies the topic; test unsupervised binding.

## Exp 14 — valence-grounding bridge: an arbitrary cue acquires valence from free energy (POSITIVE)
- Setup: (alternated to a positive front after Exp 11-13 emergent-topic wall.) Corpus where an
  arbitrary cue 'a' ALWAYS precedes a predictable run ("mmmm" = low free energy / "understood")
  and 'z' precedes VARIED letters (high free energy / "surprised"). The agent is NEVER told which
  is good. Pair-state char model; measure next-char uncertainty (bits) after each cue with a
  DETERMINED 2-char context (single-char prime was too noisy — both near the 4.81 uniform).
- Result: P(next | ..a) = 3.04 bits (confident); P(next | ..z) = 4.79 bits (uncertain); delta +1.75.
  The cue 'a' acquired POSITIVE valence (predicts a low-free-energy / understood state); 'z'
  negative — purely by co-occurring with the agent's OWN intrinsic free-energy states, never labeled.
- Implication (answers "how does it know positive without language?"): valence grounds in the
  agent's own prediction confidence (free energy); an arbitrary symbol becomes felt-good by
  associating with low-free-energy states. This is the M4 grounding bridge WORKING, and validates
  RESEARCH.md's valence = -dF/dt. No teacher, no labels, no pretraining. This is how a "+"/tone could
  eventually become meaningful to it — grounded on its intrinsic understanding-valence.
- Measurement note: pair-state models must be primed with a DETERMINED context (>=2 chars); a
  single-char prime leaves the state ambiguous and washes the signal out (3.04 vs 4.79 became
  visible only with the 2-char prime; was +0.03 with 1-char).
- Next: (a) close the affective loop — does the agent ACT to seek the 'a'-cue / avoid 'z' (EFE +
  this grounded valence)? (b) alternate back to emergent topic (emission-level) as the long arc.

## Exp 15 — closing the affective loop: the agent ACTS to seek its grounded-good state (POSITIVE)
- Setup: choice world — action0 -> a predictable/"understood" scene (peaked emission = low free
  energy), action1 -> a surprising scene (uniform emission = high free energy). Preference C is the
  SELF-EVIDENCING prior (prefer the predictable/comfort outcome the agent learned to expect — the
  principled AIF grounding of valence, NOT an external labeled reward). Compute EFE + policy posterior.
- Result: q(pi) seek-good = 0.79 vs seek-bad = 0.21; EFE good = -1.82 (lower) vs bad = -0.50.
  The agent chooses to occupy the low-free-energy / understood state.
- Implication: closes the affective loop. "Wanting to feel good" = acting to minimize expected free
  energy toward states the agent can predict. Together with Exp 14 (cue -> valence grounding) this is
  a minimal AFFECTIVE AGENT: it grounds valence in its OWN free energy AND acts to seek it — no RL
  reward, no labeled opinions, no pretraining of what is "good".
- Honest note: a preference C was specified, but as the self-evidencing prior (prefer
  predictable/characteristic states), which is how valence is grounded in active inference. This is
  the difference between "preference emerges from the agent's drive to persist" vs "reward injected".
- Connects to the moonshot: the "feel positive and seek it" you described, demonstrated — the wanting
  emerges from EFE + self-evidencing, it is not pre-trained in.
- Next: alternate back to emergent topic (long arc, the open problem in open_problem.html), or fuse
  valence with the language model (does the grounded valence shape what it attends to / generates?).

## Exp 16 — symmetry-breaking warm-start fails; reveals the MEAN-FIELD culprit (deep negative)
- Setup: hand the topics a perfect asymmetric foothold — topic0's B1 := sky char-transitions,
  topic1's B1 := grass char-transitions (fully differentiated). Emission A topic-independent;
  mean-field posterior q(z)q(s). Test binding BEFORE and AFTER 6 unsupervised epochs.
- Result: binding FAILED both before and after; topic belief stayed [0.5,0.5] EVEN WITH perfectly
  differentiated transitions; output degenerate ("s is i").
- Implication (mechanistic, deeper than "symmetry"): the topic posterior never updates even when
  the transitions clearly distinguish the topics. Cause = the MEAN-FIELD APPROXIMATION: q(z,s)≈q(z)q(s)
  assumes the topic and char factors are independent in the posterior, which SEVERS the cross-factor
  message "this char-sequence implies topic0". Evidence carried by transitions cannot reach q(z) under
  mean-field. So transition-only topic inference is blocked by the variational approximation itself,
  not just by symmetry. (Adds to open_problem.html: a 4th obstacle — the inference factorization.)
- Fix implied (and it mirrors biology / the baby framing): the topic must enter via EMISSION
  (A_dependencies=[[0,1]]) so OBSERVATIONS directly update q(z) — a concept grounded in what you
  SEE/HEAR (baby grounds "sky" by seeing blue), not in transition statistics alone. Alternatively,
  structured (non-mean-field) inference. Emission-grounding is the tractable, biologically-aligned path.
- Meta (user framing): babies DO solve unsupervised emergence, so this is an open ALGORITHMIC problem,
  not impossible. The missing ingredients are exactly what our toy lacks: multimodal/observational
  grounding (emission-level), embodiment/action (intervention), innate asymmetric priors, curriculum,
  precision/attention, and scale. Every WORKING result here came from a foothold; babies have them all.
- Next: topic-in-EMISSION model — A depends on topic so the subject directly, observationally
  identifies it; does observation-driven topic inference finally bind?

## DIRECTION PIVOT (strategic, user-driven) — bottom-up, embodied, mouse-level, run-to-see
Reframe of the whole emergent-concept thread, three moves:
1. **Lower the bar: mouse, not human.** The phenomenon (unsupervised structure from experience)
   shows up in a mouse forming PLACE CELLS — a spatial concept emerging from embodied wandering,
   no labels. Same magic, tractable scale. "Opinions are late, abstract place cells." Active
   inference already models foraging/T-maze/place-coding (pymdp has gridworld/T-maze envs).
2. **Emergence-from-aggregation, not top-down design.** We kept HAND-CARVING one concept slot
   ("be a topic, please") — no room for emergence. Brains: concepts SELF-ORGANIZE from many simple
   predictive units (Anderson, "More Is Different"). Build a substrate of many simple units + rich
   grounded input; let structure CONDENSE; don't carve the slot in advance.
3. **Computational irreducibility (halting analogy).** Emergence may be unprovable in advance — you
   can only RUN it and watch. So: (a) "mathematically open" != "won't happen"; (b) small negatives
   don't prove impossibility (maybe not enough substrate/scale/time); (c) this JUSTIFIES the
   experimental loop as the correct method — build-and-watch is what the problem demands.
Honest caveat: aggregation != guaranteed useful emergence (most aggregates = noise). Brains get GOOD
emergence from grounded embodiment + architecture + evolutionary selection; may need scale/time beyond
a laptop. Necessary direction, not automatically sufficient.
NEW PRIMARY THREAD: embodied mouse-level agent in a small grounded world (gridworld w/ sensory cues),
many simple predictive units / large state space, sensorimotor grounding (symmetry broken by the
world), then RUN and watch for emergent structure (place-cell-like location latents). If a location
concept self-organizes from embodied active inference -> the existence proof in miniature; then climb
place -> object -> relation -> (eventually) abstract dispositions/opinions.
Next experiments: (Exp 17) embodied gridworld AIF agent — does it learn a usable model / latent
structure from acting? (Exp 18+) inspect whether hidden states specialize by location (emergent
place fields). Keep affective loop (Exp14/15) as the valence substrate to fuse in later.

## Roadmap from RESEARCH.md (parallel math/frontier track — see RESEARCH.md)
The math formalizes WHY depth is the lever (first-order d-separation squeezes all history
through one belief; repeated-letter ambiguity is an exact 1-bit floor a 1-char model cannot
beat; order-1 conditional entropy floor >> trigram). Frontier directions, ranked:
1. **AIF 2-char-context (trigram) pair-state model** — reproduce Exp 7 inside active inference
   (s=(c-1,c), K=V^2=784, frozen deterministic A, learn sparse B). [= the priority thread above]
2. **Hierarchical words-above-characters** — a slow upper "chunk/intent" factor conditioning
   the fast char transitions (two timescales; Friston et al. linguistic-communication model).
   This is where word-order, Q→A, and emergent "concept/opinion" states live, and where the
   language track (M3) and affective track (M4) unify.
3. **Structure learning (Bayesian model reduction) instead of fixing K/order**, and re-point the
   autopilot loop to optimize TEMPORAL DEPTH (proven lever), not K (disproven, Exp 4).
Valence grounding (M4): research backs **valence = -dF/dt** (rate of free-energy reduction) as a
differentiable, theory-grounded intrinsic signal. Honest ceiling: discrete AIF reaches a legible,
emergent-state, grounded-valence context model with short Q→A — not conversational fluency alone;
the credible moonshot is AIF (valence/curiosity/emergent dispositions) over a richer sequence substrate.
