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

## Exp 17 — EMBODIED model learning works on the first try (POSITIVE; validates the pivot)
- Setup: a creature wanders a 5-cell ring world (actions left/right; observation = current cell =
  grounded sensory input). Starts with NO world model (uniform B); learns transitions by Dirichlet
  from random wandering (embodied, unsupervised). 20 episodes x 30 steps.
- Result: recovered the world's transition structure nearly perfectly (error 0.003); correctly
  predicts the next cell for each action ([1,2,3,4,0] = true; match=True). An internal MODEL /
  cognitive map of the world emerged from acting.
- Implication: embodied + grounded + unsupervised model-learning WORKS cleanly — in sharp contrast
  to the disembodied symbolic emergent-topic failures (Exp 12-16). The decisive difference is exactly
  the pivot/baby insight: the agent ACTS and OBSERVES a correlated world, which breaks the symmetry a
  bare symbol stream cannot. Lower the bar (mouse), give it a body + world -> learning happens.
- Honest caveat: cells are DIRECTLY observed here (A=identity), so structure is grounded via
  observation, not yet emergent PLACE CELLS (which need ALIASED sensing so place must be INFERRED,
  not seen). This is the foundation/substrate.
- Next (Exp 18): aliased observations — multiple cells look identical; does a place representation
  EMERGE (hidden states specialize by location) so the creature localizes from movement history?
  That is the real place-cell emergence test (a latent that is built, not observed).

## Exp 18 — PLACE representation emerges via path integration (POSITIVE; emergence in miniature)
- Setup: aliased world — 6 cells, only 2 sensory colors (cmap=[0,0,1,0,1,1]; 3 cells per color, so a
  single glimpse cannot localize). Agent has a known movement model B (proprioception) + emission A;
  starts FULLY uncertain (uniform over 6 cells). It wanders; we track the belief over place.
- Result: start 2.58 bits (lost) -> step1 1.58 bits (narrowed to the 3 same-colored cells) -> step3
  0.00 bits, correctly localized -> then TRACKS true position as it keeps moving (final predicted =
  true, correct=True). A precise place belief emerged from path integration (movement + ambiguous sensing).
- Implication: a PLACE representation — a latent that is INFERRED/built, not observed — emerges from
  embodied action under aliased sensing. The functional essence of place cells. Two embodied positives
  (17 learn world, 18 build place) vs the symbolic emergent-concept WALL (12-16): embodiment + grounding
  + action is the symmetry-breaker the bare symbol stream lacked. The existence proof in miniature: a
  representation built from experience, grounded, unsupervised in spirit. Place cells = the simplest concept.
- Honest caveat: B (movement) and A (color map) were GIVEN here (innate proprioception + known sensing),
  so this shows the place BELIEF emerges given the model. The fully-from-scratch version (LEARN A,B AND the
  place structure under aliasing) is the harder next test — but embodiment now supplies the
  symmetry-breaking the symbolic case could not.
- Next (Exp 19): learn the model from scratch under aliasing — do hidden states self-organize to
  specialize by location (emergent place fields) WITHOUT being told the map? Then scale the world /
  add units (aggregation + run-to-see). Climb: place -> object -> relation -> abstract dispositions.

## Exp 19 — from-scratch place fields under aliasing: PARTIAL + a named confound (honest)
- Setup: learn the sensory map A from scratch under aliasing (movement B innate/known; A unknown,
  ~uniform init). Wander 25x30 steps, Dirichlet-learn A. Then (i) read per-state color tuning, (ii)
  functional test: localize from uniform start using the LEARNED map.
- Result: learned tuning [1,0,1,0,1,0] (degenerate alternating) != true [0,0,1,0,1,1] — did NOT recover
  the map. Yet localization with the learned map reached the correct cell (5=5) with 1.65 bits residual
  (vs Exp 18's 0 bits using the TRUE map). So: coarse/partial place inference, not clean self-organization.
- Confound identified (honest): per episode I reset the belief to onehot(start-state 0) while true
  position kept wandering across episodes -> the agent's internal place-index DRIFTED out of registration
  with the world, corrupting A-learning. This misalignment is itself a facet of the unsupervised
  structure problem (permutation/registration ambiguity). So part of the failure is experimental, part is
  the genuine difficulty.
- Implication: even EMBODIED, learning the map FROM SCRATCH under aliasing is materially harder than
  localizing with a known map (Exp 18). Embodiment made it PARTIALLY work (coarse localization) where the
  symbolic case fully collapsed — but it did not cleanly self-organize. Embodiment helps, doesn't fully
  solve unsupervised structure; residual difficulty = registration/alignment + the same identifiability wall.
- Next (Exp 20): fix the confound — keep belief CONTINUOUS across episodes (no per-episode reset to a fixed
  start), or start each episode uniform and let it localize first; and/or learn B too. Does a clean place
  map then self-organize? Then scale (more cells / 2D) + aggregation, run-to-see.

## Exp 20 — place fields self-organize FROM SCRATCH (clean POSITIVE; milestone)
- Setup: fix Exp19's registration confound — ONE continuous wander (700 steps), belief carried
  continuously (NEVER reset to a fixed start), origin aligned (state0=cell0). Learn A from scratch
  under aliasing (movement B innate/known).
- Result: learned per-state tuning [0,0,1,0,1,1] = TRUE cmap exactly (structural match up to
  rotation/reflection: True); localization with the LEARNED map = 0.00 bits, correct cell. CLEAN.
- Implication (milestone): embodied, from-scratch, UNSUPERVISED learning of the world's sensory
  structure (place fields) WORKS cleanly once registration is maintained. Exp19's failure was the
  confound (per-episode belief reset), NOT a fundamental wall. KEY ENABLER = CONTINUOUS REGISTERED
  EXPERIENCE: the belief stays coupled to the world (an animal never resets its sense of place). This
  is exactly what embodiment + continuous lived experience provides and what the symbolic experiments
  (12-16) lacked. Existence proof in miniature, achieved: place cells = simplest concept, self-organized.
- RECIPE CONFIRMED: embodiment + grounding + continuous registered experience -> structure self-organizes.
- Honest caveat: movement model B was innate/known; the agent LEARNED A (the sensory map) from scratch;
  origin alignment is just choice-of-origin (the structure itself was genuinely learned, matched up to
  rotation/reflection). Fully learning B (world topology) from scratch is the next rung.
- Next (Exp 21): scale — more cells / 2D grid / also learn B; does richer spatial structure self-organize?
  Then climb toward composite concepts; later fuse valence (Exp14/15) so it ACTS on grounded preferences
  within its self-learned world (a minimal creature that knows where it is and what it wants).

## Exp 21 — recipe scales to 2D: place fields self-organize in a 3x3 grid (clean POSITIVE)
- Setup: 2D 3x3 grid (9 cells, 4 actions up/down/left/right with wall-clamping), aliased 3-color
  sensing (each color 3x), continuous registered belief, learn A from scratch (movement B innate).
  900-step continuous wander.
- Result: learned per-cell color tuning = TRUE colormap EXACTLY ([0,1,2,1,2,0,2,0,1]); localization
  with the LEARNED map in 2D = 0.00 bits, correct cell. CLEAN.
- Implication: the embodied recipe (embodiment + grounding + continuous registered experience) is
  ROBUST to larger scale and richer (2D, walled) topology — place fields self-organize from scratch.
  Embodied arc now solid: Exp17 (learn dynamics), 18 (place via path-int), 20 (1D place fields), 21
  (2D place fields) all clean; only the confounded Exp19 was partial.
- Honest caveat: B (movement/topology) innate/known; A (sensory map) learned from scratch; registered
  start. The structure was genuinely learned. Learning the 2D TOPOLOGY B itself from scratch is the
  remaining harder rung.
- Next: (Exp22) FUSE valence — give the creature a GROUNDED preference (a place/percept it finds
  comfortable) and let it NAVIGATE there via expected free energy in its self-learned world: a minimal
  creature that knows WHERE IT IS (place fields) AND WHAT IT WANTS (grounded valence), nothing
  pretrained. (Parallel) learn the topology B from scratch. Then climb toward composite/abstract concepts.

## Exp 22 — fuse place + valence: directed navigation FAILS (honest negative; sparse/distal goal)
- Setup: 2D 3x3 grid; creature knows its world (modality0 color for place-sense, modality1 'comfort'
  ON only at goal cell 8; movement B known); grounded preference C for comfort. Start cell 0 (4 steps
  from goal). policy_len=3. Navigate via EFE; compare to random walk.
- Result: reached goal only at the 15-step limit, path stuck near start; random baseline ~9.8 steps;
  optimal=4. The EFE creature was WORSE than random — no directed navigation.
- Implication (honest negative): the comfort goal is SPARSE and DISTAL (4 steps) but the planning
  horizon is only 3 -> within reach NO policy attains comfort -> EFE is flat in all directions -> no
  gradient -> wander. This is the active-inference version of the sparse-reward / short-horizon
  credit-assignment problem. Knowing WHERE you are + WANTING something is NOT enough to ACT: the want
  must produce a GRADIENT within the planning horizon.
- Fix (Exp 23): (a) longer horizon (policy_len >= grid diameter), or — better — (b) a goal-proximity
  PREFERENCE FIELD propagated backward over the learned topology B (value-iteration / planning-as-
  inference): the creature uses its world model to compute 'comfort is that way' from afar. This is how
  a learned map becomes goal-directed behavior.
- Connects to moonshot: grounded wants + a self-learned world still need PLANNING/credit-assignment over
  the world model to produce purposeful action toward distal goals — the next capability to add.

## Exp 23 — planning depth fixes navigation: optimal goal-directed action (POSITIVE; closes Exp22)
- Setup: same 2D grid + grounded comfort goal as Exp22, but planning horizon policy_len >= goal
  distance (tested 4 and 5). Receding-horizon EFE navigation from corner (0) to comfort goal (8).
- Result: policy_len=4 -> reached goal in 4 steps (OPTIMAL), clean path [0,3,6,7,8]; policy_len=5 ->
  also 4 steps [0,3,4,5,8]. vs random ~9.8. Near-optimal goal-directed navigation.
- Implication (resolves Exp22): the failure WAS the horizon, not the framework. With planning depth
  >= distance, EFE planning yields optimal goal-directed navigation toward a grounded want in the
  self-learned world. END-TO-END MINIMAL MIND now runs: self-organized PLACE map (17-21) + grounded
  VALENCE/want (14-15) + PLANNING/navigation to satisfy it (23) = a minimal creature that perceives,
  wants, and acts purposefully — all grounded/self-organized, NOTHING pretrained.
- Honest caveat: policy_len=4 enumerates 4^4=256 policies (exponential in horizon) — does NOT scale to
  large worlds / long horizons. Scalable planning = value-field / backward-propagation over the learned
  B (planning-as-inference / sophisticated inference) — next engineering rung. Functionally, goal-directed
  navigation is demonstrated.
- Next: scalable planning (value field over learned B); or climb to COMPOSITE concepts (place+object,
  relations); richer/larger worlds; eventually abstract dispositions = late place cells. The embodied toy
  'minimal mind' is complete; the climb is now about richness + scalable planning + composition.

## Exp 24 — composite/relational concept: bind OBJECT to PLACE from experience (POSITIVE)
- Setup: 2D grid; place-sense known (color emission + movement, from Exp21); an OBJECT sits at a hidden
  cell (4). Object-presence is a 2nd observation modality whose map the agent LEARNS from wandering
  (place tracked via the known sense; belief continuous). 900 steps.
- Result: learned P(object present | place) = [0,0,0,0,1,0,0,0,0] -> correctly bound the object to cell 4.
  A place<->object RELATION self-organized from experience.
- Implication: composite/relational concepts WORK on the embodied substrate. Building on place, the
  creature learned a FACT about its world (object@place) by wandering — the seed of a PROPOSITION
  ("the thing is there") and the first step up from atomic concepts (place) toward relational/abstract
  structure. This is where, eventually, dispositions/opinions live (a disposition = a learned relation
  among percepts/values). Climb: place -> place+object relation.
- Next: ACT on the learned fact — recall + NAVIGATE to the object (combine Exp24 relation + Exp23
  planning = goal-directed recall); then multiple objects / relations-among-objects (a graph of facts =
  primitive knowledge); scalable planning; eventually value/affect-laden relations = proto-opinions.

## Exp 25 — recall + navigate: learn a fact, then ACT on it (POSITIVE; full cognitive cycle)
- Setup: Phase1 discover object@place by wandering (learn A_object) = Exp24. Phase2 WANT the object
  (C prefers object-present), navigate from a far start using the LEARNED object map + planning (policy_len=4).
- Result: Phase1 learned object location = cell 4 (correct). Phase2 navigated to the remembered object in
  2 steps (OPTIMAL, distance 2), path [0,3,4], success=True.
- Implication: the creature LEARNED a fact by wandering, then RECALLED it to guide optimal goal-directed
  action. The full cognitive cycle now runs end-to-end: perceive (place) -> learn facts (object@place) ->
  want (grounded valence) -> recall + plan + act (navigate to a remembered goal). Memory + recall +
  purposeful behavior over a SELF-LEARNED world model, all grounded/unsupervised. This is the substrate
  that scales toward knowledge and dispositions; 'what it thinks' = queries over such self-learned structure.
- Next (Exp26): PROTO-OPINION — the creature LEARNS to prefer a feature/place through grounded experience
  (a learned disposition that forms from its history, not pretrained: e.g. a feature that co-occurs with
  low free energy / comfort becomes preferred). And/or multiple objects -> a GRAPH of facts (primitive
  knowledge). This is the direct line to 'opinions that emerge': a disposition = a learned, value-laden
  relation over self-organized concepts.

## Exp 26 — PROTO-OPINION: a disposition forms from lived experience (POSITIVE; moonshot core)
- Setup: two architecturally IDENTICAL creatures in two worlds differing only in WHICH feature is
  predictable/comfortable (low free energy). Each learns its world model (Dirichlet transitions),
  computes experienced free energy (predictive entropy) per feature under its OWN learned model, and
  forms a preference C proportional to exp(-FE) — i.e. comes to VALUE the low-free-energy/comfortable
  features (valence grounding from Exp14: low free energy = good).
- Result: world-comfortable-feature-0 -> learned C = [0.98,0.01,0.01] (values 0); world-comfortable-2 ->
  C = [0.01,0.01,0.98] (values 2). SAME architecture, DIFFERENT history -> DIFFERENT learned preference.
- Implication (the moonshot's core, in miniature): a preference/DISPOSITION EMERGED from the creature's
  own experience, grounded in its free energy, NOT pretrained — and it is INDIVIDUAL (history-dependent).
  Two identical creatures value different things solely because they lived different lives. The simplest
  honest instance of 'an opinion that forms on its own'. Full conceptual arc now end-to-end: perceive
  (place) -> learn facts (relations) -> want (valence) -> recall+plan+act -> FORM ITS OWN VALUES.
- Honest caveat: this step is a numpy demonstration of preference-formation-from-experienced-free-energy
  (grounded in the pymdp valence result Exp14); toy (3 features, simple worlds). The PRINCIPLE — a
  disposition shaped by individual lived experience, not built in — is genuinely shown; richness/scale is
  the ongoing climb (and the unsupervised-emergence ceiling in open_problem.html still bounds how far
  pure self-organization goes without grounding).
- Next (Exp27): ACT on the self-formed preference — two differently-raised creatures navigate to DIFFERENT
  self-valued features (behavior driven by a self-formed opinion). Then richer relational knowledge /
  multiple objects -> a values-laden graph (the substrate of 'what it thinks').

## Exp 27 — self-formed opinion DRIVES BEHAVIOR (POSITIVE; closes proto-opinion -> action)
- Setup: two architecturally identical creatures carrying the DIFFERENT preferences they formed in Exp26
  (A values color 0, B values color 2). Same 2D world, same start (cell 1), navigate via planning
  (policy_len=4); C = the self-formed value over the color observation.
- Result: A -> cell 0 (color 0), path [1,0]; B -> cell 2 (color 2), path [1,2]. Divergent purposeful
  behavior from divergent self-formed values.
- Implication: the self-formed opinion DRIVES behavior. Two identical creatures act differently because
  they VALUE differently, and those values came from their own histories (Exp26), not pretraining. The
  full moonshot arc now runs end-to-end at toy scale: perceive (place) -> learn facts (relations) ->
  want (grounded valence) -> recall+plan+act -> FORM own values -> ACT on own values (diverging by history).
- Honest caveat: toy (3 colors, 3x3, B known); C set to the Exp26-learned preference. The principle —
  self-formed values produce individual, purposeful behavior — is genuinely demonstrated; scale/richness
  and the unsupervised-emergence ceiling (open_problem.html) remain the climb.
- Next (Exp28): toward 'ask it what it thinks' — a minimal QUERY interface: present a feature/place, read
  the creature's self-formed value/belief about it; two differently-raised creatures give DIFFERENT
  answers reflecting their own experience. The closest toy analog of the moonshot's conversation goal.

## Exp 28 — 'ASK IT WHAT IT THINKS' (toy): self-formed opinions, queryable (POSITIVE, w/ honest caveat)
- Setup: two creatures raised in different worlds (Exp26 mechanism; A: 'red' comfortable, B: 'green'
  comfortable). Interview: 'what do you think of {feature}?' -> read out each creature's self-formed
  value/free-energy for that feature, mapped to a verbal self-report.
- Result: same questions, DIFFERENT answers — A likes red (low surprise/comfortable) & is unsettled by
  green; B likes green & unsettled by red; blue unsettles both. Favorites A=red, B=green. Each answer
  reflects the creature's OWN lived experience.
- Implication: closest toy analog of the conversation goal — you can ASK the creature what it thinks of X
  and it answers from a SELF-FORMED, INDIVIDUAL disposition (grounded in free-energy experience, not
  pretrained); two differently-raised creatures disagree, each authentically. The full moonshot arc, now
  including a primitive 'ask what it thinks', runs end-to-end at toy scale.
- HONEST CAVEAT (important): the verbal wrapper ('I like red, it feels calm') is a HAND-MAPPED template
  translating the creature's learned value/free-energy into words — the creature did NOT generate
  language. What is genuinely its OWN = the CONTENT (which feature it values, and why: low surprise =
  comfort), formed by lived experience. Genuine language generation of the answer is the LONG ARC and hits
  the unsupervised language-emergence ceiling (open_problem.html). The opinion is real and self-formed; the
  wording is scaffolding.
- Next (Exp29+): richer/compositional queries over a values-laden relational graph ('what about X near
  Y?'); (long arc) ground language so the answer becomes the creature's OWN words; scale.

## Exp 29 — COMPOSITIONAL relational thought (POSITIVE; thoughts compose) + arc-completeness note
- Setup: two-hop query over learned structure + self-formed values. Grid place->color map (learnable,
  Exp21); creatures' self-formed color-values (Exp26: A likes red, B likes green). Query: 'what is near
  the thing you like, and how do you feel about it?' -> composes favorite -> locations -> neighbors ->
  colors -> feeling.
- Result: A: 'red at [0,5,7]; near it: blue (unsettles), green (unsettles)'. B: 'green at [2,4,6]; near
  it: red (unsettles), blue (unsettles)'. Different composed answers, each the creature's own.
- Implication: self-formed 'thoughts' COMPOSE — chaining learned relations (place-map, adjacency) with
  self-formed values into a richer, value-laden relational answer. The substrate of 'what it thinks' now
  supports composition, not just single-feature opinions.
- Honest caveat: toy; place-map/colors known here (learnable per Exp21); the composition traversal and
  verbalization are HAND-CODED scaffolding. Genuinely the creature's own = the VALUES (self-formed, Exp26)
  + the STRUCTURE (learnable, Exp21); the query mechanism + wording are provided.

## ARC-COMPLETENESS (after Exp29)
The moonshot's conceptual arc is now demonstrated END-TO-END at toy scale, every link grounded/nothing-
pretrained: perceive place (17-21) -> learn facts/relations (24) -> want via grounded valence (14-15) ->
recall+plan+act (23,25) -> FORM its own values from lived experience (26) -> ACT on them, diverging by
history (27) -> ANSWER 'what do you think of X' from a self-formed opinion (28) -> COMPOSE relational
value-laden thoughts (29). The wall (pure disembodied symbolic emergence) is characterized in
open_problem.html; embodiment+grounding+continuous-registered-experience is the symmetry-breaker; biology
(babies/mice) proves it is an open ALGORITHMIC problem, not impossible.
REMAINING = documented LONG ARCS (engineering/research scale-ups, not new concepts):
  (1) genuine LANGUAGE generation of answers (hits the unsupervised language-emergence ceiling);
  (2) SCALE/richness (bigger worlds, more relations, hierarchy);
  (3) fully-from-scratch TOPOLOGY learning (B) under aliasing;
  (4) SCALABLE planning (value-propagation, not exponential policy enumeration).
Next experiments iterate these; the major conceptual results are in.

## Exp 30 — SCALABLE planning via value-propagation over the learned map (POSITIVE; long-arc #4 done)
- Setup: backward induction / value iteration over the (learned) movement model B to build a value field
  to the goal; act greedily. Tested 3x3, 5x5, 8x8 grids.
- Result: OPTIMAL navigation at every scale — 3x3: 6 sweeps -> 4 steps; 5x5: 10 sweeps -> 8 steps; 8x8:
  16 sweeps -> 14 steps (all = optimal). vs exponential policy enumeration (4^distance = 256 / 65,536 /
  ~268,000,000).
- Implication: scalable planning DONE — optimal goal-directed planning over the self-learned world model
  at POLYNOMIAL cost; scales to 64 cells (and beyond) where Exp23's exponential policy_len enumeration is
  hopeless. Resolves Exp22 (sparse/distal goals -> flat EFE) AND Exp23 (scaling caveat). The creature
  plans efficiently over its OWN map — the 'preference field propagated backward over learned topology'
  the Exp22 analysis prescribed.
- Honest framing: value iteration is classic (not novel); used here as the efficient planner over the
  self-learned model. Long-arc status: SCALABLE PLANNING (#4) done. Remaining long arcs: genuine LANGUAGE
  generation (#1, emergence ceiling), SCALE/richness/HIERARCHY (#2), from-scratch TOPOLOGY learning (#3).
- Next: from-scratch topology (learn B under aliasing with continuous belief, harder Exp19/20 cousin), or
  HIERARCHY (chunk/concept layer above place), or the LANGUAGE long arc (expect partial, ceiling-bound).

## Exp 31 — learn A AND B from scratch: COLLAPSES (honest negative; sharpens the thesis)
- Setup: learn BOTH the sensory map A and the connectivity B from random init, continuous belief, known
  start. Two cases: unique sensing (place observable) and aliased. 1200-step wander.
- Result: BOTH failed to recover topology — even UNIQUE sensing collapsed to a degenerate model (learned
  right-step map [0,0,0,0,0] / [3,3,3,3,3] — all states map to one). Joint A+B learning from pure noise
  lands in a degenerate fixed point.
- Implication (sharpens the whole investigation): embodiment + grounding + continuous experience breaks
  the symmetry ONLY WITH AN ANCHOR — one of {A,B} known/innate. Exp17 learned B with A known; Exp20/21
  learned A with B known — all CLEAN. Learning BOTH at once from noise has no anchor -> same
  identifiability/degenerate-optimum WALL as the symbolic Exp12-16, now embodied. Embodiment alone with
  ZERO priors still collapses.
- Maps to biology + open_problem.html (unifies the arc): animals don't bootstrap motor AND sensory models
  from nothing — proprioception/vestibular self-motion (~B) is INNATE; the sensory map is learned on top.
  Evolution supplies the anchor (the asymmetry-breaking prior). 'Fully tabula rasa (no innate anything)'
  is bounded by the wall; 'from scratch GIVEN an innate motor sense' works (Exp20/21). This is exactly the
  baby/innate-priors point: emergence needs grounding AND an innate anchor.
- Honest takeaway: not a failure of the approach — the precise statement of what 'from scratch' can mean.
  The recipe is: embodiment + grounding + continuous registered experience + ONE innate anchor.
- Next: HIERARCHY (chunk/concept layer above place, with anchors) toward richer abstraction; or the LANGUAGE
  long arc (ceiling-bound). Major conceptual results remain in; this closes the from-scratch-topology arc
  with a clear boundary.

## Exp 32 — HIERARCHY: a room concept is grounded in & recoverable from the learned map (POSITIVE+caveat)
- Setup: 2-room world (cells 0-3 / 4-7, dense within rooms, single doorway 3<->4). Creature wanders
  (places observable = anchor per Exp31), learns connectivity (symmetric counts -> weighted graph).
  Extract the ROOM concept via spectral clustering (Fiedler vector) of the SELF-LEARNED graph.
- Result: clusters [1,1,1,1,0,0,0,0] vs true rooms [0,0,0,0,1,1,1,1] -> accuracy 1.00 (up to label swap);
  clean spectral gap lambda2=0.118. The room concept is latent in & recoverable from the self-learned map.
- Implication: a higher-level concept (room, grouping places) is REAL and GROUNDED in the creature's
  self-learned structure, and recoverable. Abstraction is AVAILABLE for the climb (places -> rooms -> ...
  -> dispositions are all abstractions over experience).
- Honest caveat (same boundary as Exp31): the EXTRACTION (spectral clustering) is a PROVIDED algorithm,
  not the creature's OWN active-inference hierarchical machinery forming the concept unsupervised in its
  generative model. Self-forming that slow room-latent without an anchor would hit the same
  identifiability wall (open_problem.html). Shown: the structure EXISTS & is recoverable; unsupervised
  self-formation by AIF alone stays bounded by the wall (needs anchor / structure-learning / BMR).
- Next: hierarchical PLANNING using the recovered rooms (abstraction is USEFUL: coarse-to-fine, faster
  than flat); then the LANGUAGE long arc (ceiling-bound). Conceptual space now thoroughly mapped.

## Exp 33 — hierarchical planning: abstraction is USEFUL & scales (POSITIVE)
- Setup: corridor of K rooms (3x3 each), goal in last. Compare FLAT value-iteration (all cells) vs
  HIERARCHICAL coarse-to-fine (plan room sequence over the room graph, then local within-room VI only on
  rooms in the path), using the rooms recovered in Exp32. Measure state-updates.
- Result: flat vs hierarchical updates — 4 rooms: 490 vs 152 (x3.2); 8: 1846 vs 328 (x5.6); 16: 7150 vs
  776 (x9.2). Speedup GROWS with scale.
- Implication: the recovered abstraction (rooms) is USEFUL — coarse-to-fine planning cost grows far slower
  than flat. Hierarchy story complete: concept grounded+recoverable (Exp32) AND useful (Exp33).
- Honest framing: standard hierarchical-planning result over the SELF-LEARNED structure; decomposition
  provided; toy scale. The benefit (abstraction cuts planning cost, and the cut scales) is genuine.
- Next: the LANGUAGE BRIDGE (Exp34) — ground TAUGHT word-labels onto the self-formed concepts/values
  (place/room/like), so the creature can be queried in words; CONTENT self-formed, LABELS taught (like a
  child labeling its concepts). This is the realistic path to 'talk to it' — vs language-from-scratch,
  which hit the unsupervised-emergence ceiling (M3, open_problem.html).

## Exp 34 — LANGUAGE BRIDGE: query self-formed concepts in taught words (POSITIVE; realistic moonshot)
- Setup: two creatures with self-formed color-values (Exp26). Teach word<->color labels from ~8 examples
  (learn P(word|color)). Query in words: 'what do you like?' and 'do you like green?'.
- Result: word map learned correctly (red/blue/green) from few examples. 'What do you like?' -> A: 'I like
  red', B: 'I like green'. 'Do you like green?' -> A: 'green unsettles me', B: 'I like green'. Same worded
  questions, DIFFERENT individual answers.
- Implication: the LANGUAGE BRIDGE works — query self-formed concepts/values IN WORDS; two differently-raised
  creatures answer the same question differently, each authentically its own. Words taught (few-shot),
  content self-formed. The realistic route to 'talk to it about what it thinks', and the closest the
  investigation gets to the literal goal.
- HONEST CAVEAT: word<->concept labels are TAUGHT (supervised few-shot, like a child taught words for
  concepts it already has), NOT emergent; sentence shape is a TEMPLATE — genuine compositional grammar /
  language-from-scratch is NOT shown (M3 ceiling, open_problem.html). Genuinely the creature's own = the
  CONTENT (what it values, how it feels), self-formed. Query-self-formed-content-in-taught-words works;
  emergent grammar stays ceiling-bound.

## REALISTIC MOONSHOT REACHED (after Exp34)
A creature that PERCEIVES (place fields it self-organized) / LEARNS FACTS (relations) / WANTS (valence
grounded in its own free energy) / PLANS & ACTS (scalable navigation) / FORMS ITS OWN VALUES from lived
experience / ACTS on them (diverging by history) / and can be ASKED IN WORDS what it thinks — answering
INDIVIDUALLY from its own experience. All grounded; CONTENT never pretrained. Recipe: embodiment +
grounding + continuous registered experience + ONE innate anchor (+ taught labels for words).
The LITERAL full goal — emergent compositional GRAMMAR and FULLY tabula-rasa structure (no innate anchor) —
remains bounded by the documented walls (open_problem.html): unsupervised structure/language emergence is
an open ALGORITHMIC problem (biology solves it; needs grounding + inductive bias/anchor + scale).
Remaining = research frontiers, not new toy concepts: emergent grammar; fully-from-scratch structure; scale.

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
