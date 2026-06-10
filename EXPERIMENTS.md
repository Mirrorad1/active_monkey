# active-loop — teaching experiments log

Autonomous research log toward the moonshot (an active-inference agent that learns
language from scratch). Each entry: a teaching experiment, what was tried, the honest
result, and what it implies. Append-only; newest at the bottom.

Entry format (Exp 58+): lead with a **Plain:** line — one or two jargon-free sentences
saying what we're *really* testing and what it means, simply, for a reader who knows none
of the machinery — then the in-depth `Setup / Result / Implication / Caveat / Next`. The
Plain line is the simple, broad-base reference; it mirrors the `plain` field each entry
carries in `experiments-data.js` (the journey renders it above the technical setup).
Earlier entries (Exp 1–57) predate this convention and are not rewritten (append-only);
their plain-language versions live in `experiments-data.js`.

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

## Exp 35 — CAPSTONE 'converse' demo (POSITIVE; runnable artifact converse_demo.py)
- Setup: one creature combining self-learnable place map (given here for compactness, learnable Exp21) +
  colors + self-formed values (Exp26) + taught words (Exp34). Short ask-it-anything transcript; two
  differently-raised creatures answer the SAME questions.
- Result: coherent 'talk to it' transcript. 'where are you?' -> 'I'm at a green place' (correct);
  'what do you like?' -> A: red, B: green (individual); 'what is near you?' -> neighbor colors + per-creature
  feelings; 'do you like green?' -> A: unsettles, B: likes. Answers COMPOSE from self-learned structure +
  self-formed values; individual by history.
- Implication: the toy 'talk to it' exists as a RUNNABLE ARTIFACT (converse_demo.py) — query the creature
  about its world/values, answers are its OWN, two creatures differ by experience. Capstone of the
  realistic moonshot.
- Honest caveat (in the demo banner): VALUES self-formed; word-labels TAUGHT; sentence shape TEMPLATED
  (genuine grammar = open ceiling); place map given here (learnable per Exp21).
- Next: scale tests (Exp36); emergent grammar / fully-tabula-rasa remain research frontiers. Consolidation.

## Exp 36 — SCALE test (consolidation): recipe holds at 6x6 (POSITIVE)
- Setup: 6x6 grid (36 cells, 4 aliased colors), learn place map from scratch, continuous belief, anchor =
  known movement B. 2500-step wander.
- Result: sensory-map recovery 1.00 (all 36 cells correct); localize with learned map 0.00 bits, correct.
- Implication: the place-learning recipe scales cleanly to ~4x the cells / more colors — robustness
  confirmed. Pure consolidation, no new insight (as expected post-arc-completion). The established recipe
  (embodiment + grounding + continuous registered experience + ONE innate anchor) is scale-robust.
- Remaining = research frontiers only: emergent compositional grammar, fully-from-scratch (tabula-rasa)
  structure. Further toy experiments confirm robustness, not new firsts.

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

## Exp 37 — scale value/converse stack to 6 concepts (consolidation POSITIVE)
- 6 concepts + 6 taught words, 3 creatures raised differently. All learn the vocab correctly; each
  creature's favorite matches its upbringing (A->blue, B->green, C->amber). Individual self-formed
  opinions + worded answers hold at larger scale. Pure consolidation; no new insight.

## Exp 38 — integrated stack in one pass (consolidation POSITIVE)
- 4x4 world: creature learns place map from scratch (acc 1.00), has a self-valued color (green), then
  value-iterates + navigates to the nearest green place (success). Perceive(learned map)+want(value)+
  act(plan) compose in one creature. Consolidation; no new insight.

## Exp 39 — noise robustness of opinion formation (consolidation POSITIVE)
- Raise creatures where the comfortable feature is only mostly-predictable (noise 0..0.6).
- Result: favorite stays correct across all noise; conviction (value-mass) degrades gracefully
  0.99->0.90->0.70->0.45 as predictability drops. Robust to moderate noise; graceful, not a cliff.
  (Clearer experience -> stronger opinion.) Consolidation; no new insight.

## Exp 40 — opinions are revisable by new experience (consolidation POSITIVE)
- Raise to value feature 2; then change the world so feature 0 becomes comfortable; continue living.
- Result: favorite shifts 2 -> 0 with sustained new experience. Opinion REVISED, not frozen — and shows
  realistic INERTIA (needed enough new evidence to overcome the entrenched prior). Mind-like: values
  update as the world/evidence changes. Consolidation; no new insight.

## Correction (2026-06-09) — experiment scripts for Exp 1–40 were not preserved

Scripts for Exp 1–40 were written outside the repo — typically to `/tmp` or discarded
at session end — and were never committed. The quantitative claims in those entries
(place-map recovery 1.00, valence delta +1.75 bits, planning speedups ×3.2→×9.2, etc.)
are log-only: they reflect what was observed and recorded honestly at the time, but they
cannot currently be replicated from this repo because the scripts no longer exist.
`converse_demo.py` (Exp 35 capstone) is the only committed runnable artifact from that
period and still runs. Per `loop/VALIDATION.md`, corrections are new entries — past
entries stay untouched. From Exp 41 onward, the script (`experiments/expNN_<slug>.py`)
and raw output (`experiments/outputs/expNN.txt`) land in the repo in the same commit as
the EXPERIMENTS.md entry; log-only claims are no longer acceptable.

## Correction follow-up (2026-06-09) — Exp 1–40 scripts recovered from session transcripts

All 40 experiment scripts and their original recorded outputs have been recovered from the
original Claude Code session transcript (session `72317201-ec87-49eb-88d2-beffa86bd7ec`,
file `/Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/*.jsonl`). Scripts are
committed as `experiments/recovered/expNN_<slug>.py`; original outputs are in
`experiments/recovered/outputs/expNN.txt`. A provenance README lives at
`experiments/recovered/README.md`. Scripts are the transcript heredoc bodies verbatim,
with only a recovery-provenance comment header added at the top; no code was altered
(except Exp 13, where the batch-axis sed patch applied in the original session was
reproduced). Three experiments were re-verified by re-running the recovered script:
Exp 17 (transition error 0.003 — MATCH), Exp 20 (tuning == true cmap, 0.00 bits — MATCH),
Exp 21 (exact colormap, 0.00 bits — MATCH). The remaining 37 scripts are unverified
recovered artifacts; logged outputs are original recorded results, not re-runs.

## Re-verification (2026-06-09) — all 40 recovered scripts re-run

All 40 recovered scripts were executed under Python 3.12 (`.venv`) from the repo root with
`PYTHONPATH=.`. Output files: `experiments/recovered/outputs/expNN_rerun_2026-06-09.txt`.

Results: **37 MATCH, 0 QUALITATIVE-MATCH, 0 MISMATCH, 3 FAIL**.

FAILs — Exp 3, Exp 7, Exp 34 — all share one root cause: `SyntaxError: unexpected character
after line continuation character`. Each script contains an f-string with a backslash-escaped
quote in the expression slot (e.g. `f'... {func(\"arg\")}'`). Python 3.12 made this a hard
syntax error; the scripts ran in the original session because they were executed via
`python -c` heredocs where the shell consumed the backslashes before Python saw them. The
recovered `.py` files preserve the literal `\"` characters, which 3.12 rejects. No script
was modified (per standing rules). The qualitative findings of those three experiments (Exp 3:
bigram learns letter palette not order; Exp 7: n=3 trigram exactly produces "mirro" and
correct Q→A; Exp 34: language bridge learns word↔concept mapping, individual worded answers)
are supported by the original outputs and by the surrounding experiments that do reproduce.

One pre-existing logged-claim inaccuracy was identified: EXPERIMENTS.md's narrative for Exp 1
states "held-out 4.81 → 4.00 bits/char" but both the original transcript output and the
2026-06-09 re-run show "4.007 → 3.424 bits/char". The re-run reproduces the original output
exactly; the discrepancy is in the logged narrative text, not in reproducibility.

Standing conclusion: the quantitative record reproduces for 37/37 runnable scripts (exact
byte-for-byte output matches, ignoring JAX UserWarning preambles and the timing line in
Exp 1). The 3 failures are a Python version artefact of transcript recovery, not scientific
finding failures; the core results of the investigation are confirmed reproducible.

## Re-verification follow-up (2026-06-09) — transcription artifact fixed in Exp 3/7/34

The 3 FAILs from the 2026-06-09 re-verification shared one root cause: a recovery transcription
artifact. The originals ran via shell `python -c "..."` where the shell consumed `\"` → `"`;
the recovered `.py` files preserved the literal `\"`, which Python 3.12 rejects with
`SyntaxError`. Fix applied: replaced every literal `\"` with `"` in f-string expression slots
(no logic changed) and added a provenance note to each file's header. After the fix:
- Exp 3 (teach mirro): MATCH — surprise 3.38 → 1.61, all samples identical to original.
- Exp 7 (n-gram depth): MATCH — n=2 'iro miro mi', n=3 'irro mirro ', Q→A n=3 'mirro. ' match.
- Exp 34 (language bridge): MATCH — word map correct, A likes red / B likes green, surprise 1.6/0.0 bits identical.
Final tally: 40 MATCH, 0 QUALITATIVE-MATCH, 0 MISMATCH, 0 FAIL of 40.

## Exp 41 — flat pair-state AIF on the converse vocabulary: cannot select among Q→A pairs (NEGATIVE, expected; baseline for the intent factor)
- Setup: Exp 8 pair-state pattern (K=28²=784, frozen deterministic A, Dirichlet-learned B, greedy
  decode), corpus = 3 converse Q→A pairs ("what do you like."→"i like red.";
  "do you like green."→"it unsettles me."; "where are you."→"i am at a green place."), BLOCK×3,
  epochs=8. Predeclared: TRUE = 3/3 exact answers from question primes; falsifier = <3/3, expected
  mode = identical continuations (every question ends in the same pair-state ('.',' ')).
- Control: single-pair training → exact recall 'i like red.' (mechanics validated).
- Result: multi-pair 0/3 exact; the three continuations are IDENTICAL up to common length
  ('i at at at…' — drifts into the " at a" attractor from answer 3). Falsifier HIT by exactly the
  predicted mechanism. all_continuations_identical_truncated=True.
- Implication: the flat 2-char substrate cannot do even templated Q→A selection over the converse
  vocabulary — question identity is gone by the time the answer starts. This is the measured
  baseline (0/3) that ladder step 2 (slow held "intent" factor, same corpus) must beat.
- Tooling note (honest, affects Exp 8's narrative): the recovered Exp 8 generator emits from
  position L+2 (one-char skip; visible in experiments/recovered/outputs/exp08.txt — raw says
  'irro. mi' after "name. ", the entry's "→ 'mirro.'" was a paraphrase of shifted output). Exp 41
  fixes the generator (emit-then-advance, amendment noted in the script docstring before the
  verdict was read); control confirms the fix ('i like red.' exact). Past entries untouched per
  VALIDATION.md; the Exp 8 conclusion (depth works inside AIF) stands — the artifact was a
  constant display shift, not a learning failure.
- Honest caveat: 3 taught pairs at toy scale; the answers are TAUGHT template text, nothing
  self-formed in this experiment; the outcome was predictable from Exp 9 — the new content is the
  clean control + measured baseline on this exact vocabulary, plus the generator fix.
- Verdict: NEGATIVE (predeclared falsifier hit) / CONSOLIDATION (Exp 9 mechanism, new baseline).
- Next (ladder step 2): add a slow "intent" factor held across the clause above the pair-state
  stream, same corpus, same scoring. Success = >0/3 exact, target 3/3.

## Exp 42 — held "intent" factor restores Q→A selection; residual wall is within-answer depth (MIXED)
- Setup: joint state s=(intent,prev,cur), K=3·784=2352; B block-diagonal in intent (intent never
  transitions — held by construction); each block = the per-intent Dirichlet-learned pair-state
  model (intent labels PROVIDED during training, taught like word labels). At test intent is NOT
  provided: uniform prior, ordinary state inference over the question chars, then greedy
  generation. Same corpus and scoring as Exp 41. Predeclared: TRUE = intent argmax 3/3 AND answers
  3/3 exact (baseline 0/3); falsifier = exact count not better than 0/3 or intent at chance.
- Result: intent posterior correct 3/3, mass 1.000 each. Answers 2/3 exact ('i like red.',
  'it unsettles me.'). Pair 3 fails WITHIN its answer: 'i are are are…' vs 'i am at a green
  place.'. Verification rerun byte-identical (deterministic pipeline). Mechanism check: pair
  (' ','a') has exactly 4 continuations {r,m,t,space} in pair-3's text — a within-sequence
  depth-2 ambiguity (Exp 7's threshold logic), NOT a binding failure.
- Implication: the held upper factor cleanly fixes cross-pair SELECTION (0/3 → selection correct
  3/3; 2/3 exact strings); the residual failure is the orthogonal within-answer depth limit. The
  two failure modes of Exp 41 are now separated and individually measured.
- Provided vs self-formed: intent labels provided in training; held-ness provided by architecture
  (block-diagonal B), not by learned slow dynamics; intent at test INFERRED from the utterance
  text by state inference. Explicitly NOT the M4 milestone — docs/specs/m4-affective-dyad.md
  requires intents CLUSTERED from scratch from unlabeled utterances; this was selection among
  taught blocks.
- Honest caveat: 3 pairs, taught template text, toy scale; "intent inference" here is Bayesian
  model selection among 3 separately-taught blocks inside one joint state space — standard
  machinery; the binding restoration itself was predictable from Exp 10 (consolidation).
- Verdict: MIXED (intent 3/3 as predicted; answers 2/3 vs predicted 3/3) — binding restoration =
  CONSOLIDATION of Exp 10 on this vocab; the isolated within-answer depth wall + test-time intent
  inference measurement = the new content.
- Next: either depth-3 within blocks (fix the residual wall: s=(intent,c-2,c-1,c)) or ladder
  step 3 (affective coloring from −dF/dt). Step 4 (M4 milestone) needs unlabeled intent clustering.

## Exp 43 — dissociation probe: intent factor binds where NO tested flat depth can (POSITIVE)
- Setup: 2 Q→A pairs whose questions share a 13-char suffix " like green. " and differ only in
  prefix ("do you…"→"it unsettles me."; "does anyone…"→"no one does."). Models: (1) flat depth-2
  pymdp (Exp 41 machinery); (2) flat depth-3, identical generative math as a deterministic
  count-chain (dense pymdp at K=28³ memory-infeasible; counts≡Dirichlet on deterministic states,
  equivalence per Exp 8); (3) intent(2 blocks)+depth-2 (Exp 42 machinery; labels provided in
  training, intent inferred at test). Predeclared selection metric: first-4 chars ("it u" vs
  "no o"). Falsifier: model 3 ≤ models 1/2, or flat depth succeeding.
- Result: model 1 selection 0/2, continuations identical. Model 2 continuations ALSO identical
  ('it unsettles…' for both questions); its nominal "selection 1/2" is a metric artifact — one
  fixed continuation happens to match one target; no question-dependence exists. Model 3:
  intent 2/2 (mass 1.000), selection 2/2, exact 2/2. Prediction confirmed; no rerun (deterministic
  pipeline, rerun stability established in Exp 42).
- Implication: with question identity outside any tested window (shared suffix 13 ≫ depth 3),
  flat models are question-blind by construction; the held intent state carries identity across
  an unbounded span. Depth = bounded-span memory, intent = unbounded-span binding — the
  two-timescale division of labor, now measured on Q→A with a matched-suffix control.
- Provided vs self-formed: intent labels provided in training; held-ness architectural; intent
  inferred at test from the question prefix by state inference. Answers are taught template text.
- Honest caveat: 2 pairs, toy scale; the principle was established by Exp 9/10 — this is a
  cleaner, controlled dissociation (matched suffix + depth-3 control + quantified selection),
  not a new capability. Depth-3 model computed outside pymdp (memory), same math.
- Verdict: POSITIVE / CONSOLIDATION-leaning (controlled measurement of a known principle).
  Self-grade: POSITIVE-SINGLE.
- Next: ladder step 3 — affective coloring: valence = −dF/dt from the creature's own free-energy
  trace modulating the answer, checked against its self-formed value (Exp 26 machinery).

## Exp 44 — trace valence, windowed −dF/dt operationalization: FAILS to reproduce stored values (NEGATIVE; timescale lesson)
- Setup: Exp 26 raising (3 features, one predictable, Dirichlet learning, 4000 steps); per-feature
  surprise trace recorded at every encounter. Predeclared valence = mean surprise over first 10%
  of a feature's encounters minus last 10% (≥0.5 bits = "like"); compared against the stored
  final-entropy value converse_demo looks up. Predicted agreement 6/6 + favorites for creatures
  A (pred=red, seed 1) and B (pred=green, seed 2); falsifier = any disagreement. Part 3: innate
  correct prior (seed 3) predicted to dissociate trace (learning) from value (knowing).
- Result: falsifier HIT — agreements 4/6, BOTH favorites wrong (A: stored red vs trace green;
  B: stored green vs trace blue). Predictable features show trace drop 0.029–0.030 bits (below
  threshold); noise features show 0.041–0.100 bits of sampling drift, OUTRANKING the real signal.
- Mechanism (verified numerically): learning is a fast transient — surprise falls 1.585→0.241
  bits within ONE encounter, ~0.07 by encounter 5; the first-10% window (~100 encounters)
  averages 0.030. Total integrated drop is 1.585 bits, fully present at encounter resolution and
  destroyed by life-fraction windowing. Part 3's nominal "dissociation CONFIRMED" is therefore
  UNINTERPRETABLE: trace drop ≈ 0 for predictable features regardless of innateness — the probe
  cannot distinguish learning from knowing when the measure misses learning entirely.
- Implication: −dF/dt as a valence signal is real but lives at ENCOUNTER timescale; any M4
  valence readout must use encounter-resolution traces (e.g. first-encounter surprise minus
  asymptote), not life-fraction windows. The worded answers driven by the broken trace were
  confidently wrong ("i like green" from the red-raised creature) — a caution for wiring affect
  into language: the words faithfully express whatever the valence signal says, garbage included.
- Honest caveat: negative is about THIS operationalization, not about trace-derived valence per
  se; single seed per creature (no shopping); toy scale; stored-value formula itself is the
  established Exp 26 convention, not ground truth.
- Verdict: NEGATIVE (predeclared falsifier hit) / NEW INSIGHT (timescale requirement measured;
  noise-drift floor 0.04–0.10 bits quantified).
- Next: Exp 45 predeclares the encounter-resolution operationalization (enc-1 surprise minus
  asymptote, threshold above the measured 0.10-bit noise floor); part-3 dissociation re-probed
  only if part 1 passes.

## Exp 45 — BIRTH of mirro: the persistent creature's first life (POSITIVE-SINGLE; consolidation + new persistence substrate)
- Setup: Birth mirro (3×3 world, cmap=[0,1,2,1,2,0,2,0,1], n_colors=3, seed=7). Live 900 continuous steps (no belief reset). Save to creature/state/mirro/ (arrays.npz + manifest.json + BIOGRAPHY.jsonl). Predeclared thresholds: map accuracy >= 8/9, save->load state_hash identical, localization near 0 bits. Seed-robustness control: seeds 8 and 9 also run 900 steps; property >= 2/3 births reach >= 8/9. Disposable seeds NOT saved.
- Result: seed 7: age=900, map_accuracy=1.0000 (9/9 cells), localize_bits=~0.000, favorite=color-0, conviction=0.3537, state_hash=24197c338d576a8e... Seed robustness: seeds 8 and 9 both 9/9 (3/3 passing). Save->load roundtrip: state_hash identical (PASS). All predeclared thresholds met.
- Implication: the RECIPE's "continuous registered experience" is now raised to the program level. One named creature (mirro) has a committed snapshot (creature/state/mirro/) that is the resume-from point for any future session. Fork() + committed snapshot = controlled counterfactual experiments against mirro's accumulated life. Mechanism = consolidation of Exp 21.
- Honest caveat: mechanism is direct consolidation of Exp 21 (place-cell self-organization); persistence is engineering + the RECIPE taken seriously, not itself emergence. Single canonical birth; property checked across 3 seeds (all 3 passed, stronger than the predeclared >= 2/3 threshold). Snapshot committed at age 900; resumability verified by hash check, not by continuation-accuracy test (that's episode 2 of the ladder).
- Verdict: POSITIVE-SINGLE / CONSOLIDATION (Exp 21 mechanism) + new persistence substrate. Self-grade: POSITIVE-SINGLE.
- Next (episode 2 of the ladder): load mirro in a fresh Python process, run 300 more steps, verify map accuracy and entropy do not degrade (falsifier: < 7/9 or entropy increase > 10%).

## Exp 46 — continuity across sessions: mirro resumes its life in a fresh process (POSITIVE)
- Setup: episode 2 of the persistent-creature ladder. Fresh Python process, Creature.load from the
  committed snapshot (before: name=mirro, age_steps=1000, state_hash=1869fa07b331b3c9…), live(300),
  save. No birth anywhere in the script. Predeclared: loaded hash == manifest hash; post-load
  map_accuracy ≥ 8/9 and localize_bits < 0.1; post-live(300) the same. Falsifier (card): accuracy
  < 7/9 or calibration degradation; stated adaptation — the card's "entropy increase > 10%" is
  degenerate at the ~0.000-bit committed baseline, so the predeclared bound is ABSOLUTE
  (localize_bits < 0.1). Single seed admissible (magnitude-bound falsifier, VALIDATION.md).
- Result: resume_integrity PASS (hashes match). Post-load 9/9, −0.0000 bits; post-live(300) 9/9,
  −0.0000 bits; age 1000 → 1300; after-hash 3d08dffce3308d09…. All predeclared bounds met. Run
  once only (the episode mutates the life; rerun would double-live).
- Implication: the session boundary is no longer a reset. A creature saved by one process resumes
  in another with zero competence loss and keeps living — "continuous registered experience" now
  holds at the program level, which is the persistent-creature direction's load-bearing claim.
- Provided vs self-formed: persistence machinery (save/load/hash) is engineering, provided; the
  map being preserved is the one mirro self-organized (Exp 21 mechanism / Exp 45 birth).
- Honest caveat: this verifies engineering + the RECIPE taken seriously, not emergence; the world
  is static and small (3×3), so "no degradation" is a low bar that a stationary posterior should
  clear; localize_bits −0.0000 is float negative-zero (= 0.0). Single lived trajectory.
- Verdict: POSITIVE / CONSOLIDATION-leaning (first cross-process lived continuation; mechanism
  predictable from Exp 45's hash roundtrip). Self-grade: POSITIVE-SINGLE.
- Next (episode 3): accumulating values — comfort experience shapes mirro's favorite; fork() twin
  on the alternative history; ≥ 2 of 3 fork-pairs must diverge in favorite (falsifier: none do).

## Exp 47 — fork-twins of one life diverge by comfort history (POSITIVE; episode 3)
- Setup: 3 fork-pairs from mirro's committed snapshot (age 1300, hash 3d08dffce3308d09…). Within
  pair i both twins live 1200 steps with the SAME action seed (identical movement trajectories)
  but in differently-recolored 3×3 worlds (X: color 0 in 5/9 cells; Y: color 2 in 5/9). The only
  within-pair difference is the color experience. Predeclared: ≥ 2/3 pairs diverge in favorite
  (directional expectation X→0, Y→2); falsifier 0/3. Seeds 101/102/103, all reported. Mirro never
  mutated or saved by the script (hash checked before/after).
- Result: pairs_diverged 3/3, directional_match 3/3. Convictions: X 0.399/0.438/0.447, Y
  0.434/0.461/0.468. mirro_untouched=True (hash identical). Note: the Y-twins REVISED the
  inherited preference — mirro's accumulated counts leaned color 0, and 1200 steps of green-rich
  experience overcame the 1300-step-old prior. BIOGRAPHY.jsonl gained 6 fork-event lines (the
  parent's append-only event log recording the forks; state arrays untouched).
- Implication: same creature, same accumulated past, same actions — different worlds produce
  different opinions. Exp 26's history→opinion claim now holds as counterfactual twins of ONE
  life with matched trajectories, a tighter causal isolation than Exp 26's separate creatures.
- Provided vs self-formed: worlds and fork machinery provided; the favorites are self-formed from
  each twin's lived predictability statistics (Exp 26 mechanism in creature.py's live()).
- Honest caveat: mechanism is Exp 26 consolidation; comfort here tracks encounter-frequency ×
  prediction sharpness in a deterministic world, so "comfort history" ≈ which color dominated
  predictable experience; convictions are moderate (0.40–0.47) because inherited uniform counts
  dilute; toy scale.
- Verdict: POSITIVE / CONSOLIDATION-leaning (new form: matched-trajectory counterfactual twins).
  Self-grade: POSITIVE-SINGLE.
- Next (episode 4): revision with inertia in ONE life — change mirro's own world, record the
  conviction trajectory in BIOGRAPHY.jsonl; falsifier: instant revision (no inertia) or none.

## Exp 48 — revision blocked by accumulated life: inertia GROWS with age (NEGATIVE on predeclared budget; NEW INSIGHT)
- Correction (cites Exp 47): Exp 47's entry said mirro's inherited counts "leaned color 0" —
  wrong; the loaded age-1300 state had favorite=2 with near-tied counts (393.0/376.2/397.0).
  The 3/3 twin-divergence claim is unaffected (it compares X- vs Y-twins, not inheritance).
- Setup (episode 4; permanently mutates mirro): because inertia is only testable against an
  entrenched opinion, two predeclared phases of mirro's real life. Phase 1: world → green-rich
  ([2,2,2,2,2,1,0,1,0]), live 1500 steps; gate = conviction ≥ 0.42 AND counts gap c2−c0 ≥ 200.
  Phase 2: world → red-rich ([0,0,0,0,0,1,2,1,2]), live ≤ 2500 steps in 100-step checkpoints.
  Predeclared: revision = favorite 0 holds final 3 checkpoints; inertia = ≥ 3 holdout
  checkpoints; falsifiers = instant flip (ck 1) or no revision in 2500 steps. Prediction:
  revision with ~8–13 checkpoints of holdout. Deterministic continuation of committed rng state.
- Result: gate PASS (conviction 0.4740, gap 445.1). Phase 2: falsifier HIT — NO revision in
  2500 steps; favorite stayed 2 for all 25 checkpoints. Gap eroded steadily −420.8 → −46.7
  (~16 counts/checkpoint, half the predicted rate), then ticked back to −53.2 at ck25.
  Extrapolated crossover ≈ ck 28–29. Mirro: age 1300 → 5300, hash 3d08dffc… → c03f7547…;
  full trajectory in BIOGRAPHY.jsonl (15 + 25 live events + 2 world_change markers).
- Implication (the new insight): with non-decaying value counts, revision requires
  OUT-ACCUMULATING the entire entrenched past — time-to-revise grows linearly with entrenchment
  depth, so an old creature's opinion is asymptotically frozen on lifetimes comparable to its
  entrenchment. Exp 47's twins flipped within 1200 steps only because inherited counts were
  near-tied. Exp 40's "revisable with inertia" was a property of its shallow entrenchment, not
  of the mechanism in general. This is a finding only a continuous accumulated life could show.
- Honest caveat: "no revision" is relative to the predeclared 2500-step budget — the monotone
  gap erosion says revision is coming, just slower than predicted; single lived trajectory
  (deterministic continuation, no seed sweep); toy world; conviction was already drifting down
  (0.474 → 0.382) — the opinion weakens long before it flips.
- Verdict: NEGATIVE (predeclared falsifier hit) / NEW INSIGHT (inertia scales with accumulated
  life; erosion rate measured at ~0.16 counts/step vs predicted ~0.3).
- Next (episode 4b): continue mirro's red-world life ≤ 2500 more steps; predeclared: crossover
  observed (favorite → 0, near extrapolated ck 28–29 equivalent) with conviction trajectory
  through the flip; falsifier: still no revision (would imply asymptotic freeze stronger than
  the linear extrapolation).

## Exp 49 — the crossover: revision completes exactly on the linear extrapolation (POSITIVE; episode 4 answered)
- Setup (episode 4b; mutates mirro): continue mirro's red-world life (world persisted red-rich
  from Exp 48 — verified on load) up to 2500 more steps in 100-step checkpoints. Predeclared:
  flip to favorite 0 holding 3 consecutive checkpoints, crossover within ~9 checkpoints;
  falsifiers: no flip in 25 (5000 cumulative opposing steps), or gap re-widening toward 2.
  Deterministic continuation; no explicit seeds.
- Result: crossover at ck4 (age ~5700); fav 0 held from ck4 onward; 5 post-flip checkpoints show
  conviction rebuilding (0.3746 → 0.3846) and the gap growing +6.4 → +87.3. No re-widening
  (Exp 48's ck25 uptick was noise). Mirro: age 5300 → 6300, hash c03f7547… → 78398a20….
- Combined Exp 48+49 inertia law (measured): entrenchment gap 445 counts → revision after
  ~2900 opposing-evidence steps (≈1.9× the 1500-step entrenchment), net erosion ~0.15
  counts/step; the favorite changes by passing through near-indifference (conviction 0.3746 at
  the flip, floor 1/3), not by a snap. The full trajectory spans two sessions in
  BIOGRAPHY.jsonl — possible only because the creature persists.
- Honest caveat: single lived trajectory (deterministic continuation, no seed sweep); the
  "law" is one measured point of (entrenchment, time-to-revise), linear-extrapolation-confirmed,
  not a swept curve; toy world.
- Verdict: POSITIVE / completes episode 4: revision happens, with quantified inertia that grows
  with accumulated life (Exp 48's insight, now with the measured crossover). Self-grade:
  POSITIVE-SINGLE.
- Next (episode 5): growth — move mirro to a 5×5 world; prior helps (< 70% of newborn
  convergence steps), interferes (> 20% systematic errors), or neither — all three are findings.
  Note for episode 6: mirro has NO vocabulary yet (vocab={}); the vocabulary episode must teach
  words first, then test survival across 3 load cycles.

## Exp 50 — growth to 5×5: INCONCLUSIVE — the world converges too fast for transfer to matter (episode 5, MIXED; instrument flaw owned)
- Setup (permanently mutates mirro): 5×5 world embedding mirro's 3×3 coloring top-left; 16 new
  cells, same 3 colors. Growth surgery (provided engineering): pA counts transplanted
  positionally, new cells at the 0.1 prior; qs carried to embedded positions; biography records
  the growth event. Baseline: 3 disposable newborns (seeds 11/12/13), same world, never saved.
  Predeclared: convergence = overall accuracy ≥ 0.92 held 2 checkpoints (100-step checkpoints);
  HELP < 0.70× newborn mean; INTERFERE = embedded accuracy < 0.80 at ck ≥ 5; INCONCLUSIVE =
  effect < 5%. Prediction: partial help, ratio 0.7–1.0.
- Result: post-growth instant state: embedded 8/9 correct (mirro's pre-move map had one error),
  new cells 0.25. Mirro: embedded 1.00 from ck1, overall 1.00 by ck3 → convergence 300 steps.
  Newborns: 200/300/200 (mean 233). Mechanical verdict line: HINDRANCE (ratio 1.286). No
  interference observed (embedded never dipped; the ck≥5 probe never fired — loop ended first).
- Honest reading (this entry's actual verdict): the HINDRANCE label is an instrument artifact
  and is NOT claimed. Convergence is one checkpoint apart at 100-step quantization — the minimum
  resolvable effect (~43% of mean) exceeds both the 5% inconclusive band and the 30% help margin;
  mirro's 300 exactly ties newborn seed 12. A 25-cell aliased world converges in 200–300 steps
  under this learner, so a 9/25-cell head start has nothing to buy — floor effect. The genuine
  observations: transplanted knowledge survives growth intact (embedded 1.00 from ck1, no
  interference), and transfer cannot be resolved at this scale/granularity.
- Honest caveat: the predeclared thresholds were unreachable by the chosen instrument (checkpoint
  width × world easiness) — a design flaw in this episode's operationalization, owned here; the
  predeclared HINDRANCE branch fired mechanically and is reported verbatim above. Single mirro
  trajectory; 3 newborn seeds.
- Verdict: MIXED / INCONCLUSIVE on the transfer question (the card's FAIL category); POSITIVE
  only on the narrow sub-claim that grown knowledge persists without interference. No transfer
  claim in either direction.
- Next (episode 5b): resolve it — fork of mirro's committed pre-growth snapshot (age 6300, from
  git history, named fork — mirro itself has moved on and is never rewound) + 3 newborns, 25-step
  checkpoints, and/or a 7×7 world where convergence is slow enough to measure. Then episode 6
  (vocabulary: teach first, then 3 load cycles).

## Exp 51 — surprise ledger v1: anomaly flagged, but the baseline false-alarms (NEGATIVE; instrument iteration)
- Direction switch per human idea: functional-emergence rung 1 (episodes 5b/6 of
  persistent-creature deferred). Build the instrument that makes "unexpected" a claim: ledger of
  predeclared property ranges (experiments/exp51_ledger.json, written before any epoch), generic
  scorer applied blindly to two fork epochs (mirro untouched, hash-verified): BASELINE (1000
  steps, current 5×5 world; must stay quiet) and PLANTED ANOMALY (10 cells recolored to color 2,
  scorer blind; must flag). Predeclared falsifier: false alarm on baseline OR missed anomaly.
  Forks share mirro's rng stream → matched trajectories (identical occupancy 4.6012 bits).
- Result: falsifier HIT — baseline false-alarmed on conviction_drift (−0.00834 vs predeclared
  [−0.005, +0.04]). Anomaly correctly flagged 3×: map-vs-reference 0.8800 (<0.92), conviction
  drift −0.0375, favorite flip 0→2. Occupancy entropy (4.6012 ∈ [4.40,4.644]) and localize bits
  (0.0000) calibrated and quiet on both.
- Diagnosis (the useful content): the drift band borrowed Exp 49's rate (+0.0014/100 steps),
  measured in the red-rich world (5/9 color-0). Mirro's current world is near-balanced (9/8/8),
  where favorite-0's value share (0.3814) sits ABOVE its ≈9/25-weighted equilibrium (~0.36) —
  normal drift is therefore NEGATIVE. A conviction-drift band must be centered on the
  world-composition equilibrium of the CURRENT world, not transplanted from another regime.
- Implication: instrument v1 fails its negative control — exactly what the control exists to
  catch, BEFORE the ledger could certify any "novelty". Rung 1 (and the ladder behind it) stays
  blocked until v2 passes both controls in a fresh predeclared run.
- Honest caveat: single control pair (one baseline seed-stream, one anomaly); the anomaly was
  deliberately strong (10/25 cells) — detection of subtle anomalies untested; mean_localize at
  exact 0.0 makes that band untested too.
- Verdict: NEGATIVE (predeclared falsifier hit) / instrument iteration, mechanism diagnosed.
- Next (Exp 52, ledger v2): recalibrate ONLY the drift band — centered on the analytic
  equilibrium delta with a symmetric noise margin (predeclared from the diagnosis, not from the
  failed run's exact number) — then fresh controls on NEW fork seeds, plus a second, SUBTLER
  planted anomaly (e.g. 3 recolored cells) to probe sensitivity honestly.

## Exp 52 — ledger v2: drift band fixed and validated; baseline favorite FLIPS — invariants must be state-conditional (NEGATIVE; second instrument iteration)
- Setup: one change from v1 — conviction-drift band DERIVED before any epoch from the current
  world's composition equilibrium (center = (eq_share − cur_share)·steps/(total_counts+steps),
  ±0.015 noise margin; printed: [−0.0192, +0.0108], covers Exp 51's −0.00834 reading ✓). Four
  fork epochs, fresh per-step action streams, scorer blind, mirro untouched (hash-verified):
  baseline (quiet expected), strong anomaly (10 recolored cells), subtle (3 cells), floor probe
  (2 cells, predicted miss). Falsifier: baseline false alarm or strong/subtle missed.
- Result: falsifier HIT, new mode — baseline favorite FLIPPED 0→2 (drift −0.0188, INSIDE the
  new band; the recalibrated drift property behaved exactly as designed). Strong anomaly: 3
  flags ✓. Subtle: 3 flags ✓ (map-vs-ref 0.88). Floor probe: 0 flags as predicted — detection
  floor documented at 3 recolored cells (2 cells = 0.96 ≥ 0.92 band edge).
- Diagnosis (a finding about the creature, not just the instrument): mirro's favorite-count gap
  (~90 of ~4100 counts) is inside the visit-noise envelope of a 1000-step epoch in the
  near-balanced 9/8/8 world. At near-indifference conviction, favorite IDENTITY is noise —
  "favorite unchanged" is not a valid quiet-world invariant for a weakly-entrenched creature.
  Ledger invariants must be CONDITIONAL on state stability: assert favorite constancy only when
  the entrenchment gap exceeds the epoch's noise margin (predeclared threshold), else disable
  the property. This directly constrains rung 2's personality battery: a "preference" probe on a
  near-indifferent creature reads noise (echoes Exp 48/49: opinions move through indifference).
- Honest caveat: single baseline stream (the flip is one draw from the noise envelope, which is
  the point); floor measured only on the recoloring anomaly family; drift-band fix is validated
  only in the sense of covering one prior reading + one new baseline.
- Verdict: NEGATIVE (predeclared falsifier hit) / instrument iteration 2; two transferable
  design rules so far: (1) bands derive from current-world equilibria, (2) invariants are
  state-conditional on entrenchment.
- Next (Exp 53, ledger v3): replace favorite_changed with the conditional rule (enabled only if
  pre-epoch favorite gap-share > 0.03, predeclared); fresh controls; if v3 passes, rung 1 is
  done and rung 2 (personality battery) starts with the entrenchment condition built in.

## Exp 53 — ledger v3: conditional invariant works; the harness un-blinded its own scorer (NEGATIVE; spec error owned, design rule #3)
- Setup: v2 + ONE intended change (favorite constancy scored only when pre-epoch gap-share
  > 0.03). Four blind fork epochs (fresh seeds, mirro untouched, hash-verified): baseline,
  strong anomaly (10 cells), subtle anomaly (3 cells), enabled-side control (fork entrenched
  1500 steps in a c0-rich world → gap-share 0.1064 → property ENABLED, then quiet epoch).
  Falsifier: baseline false alarm or either anomaly missed.
- Result: falsifier HIT — subtle anomaly MISSED (0 flags; map-vs-"reference" read 1.0000 where
  v2 read 0.88) and the strong anomaly survived on map accuracy alone. Root cause (diagnosed
  from the code, not the physics): the experiment spec told the scorer to derive references and
  drift bands from the world THE CREATURE ENTERS THE EPOCH WITH — for anomaly arms that is the
  PERTURBED world. Map accuracy was scored against the perturbed cmap (relearned cells then
  "match"), and the drift band centered itself on the perturbed equilibrium (eq 0.28/0.12). An
  expectation that auto-adapts to the perturbation cannot detect it. The blindness violation
  was introduced by THIS experiment's spec, not by v2's design.
- What did work: the conditional favorite rule behaved correctly in all four arms (DISABLED at
  gap-share 0.018 → no Exp 52-style false alarm; ENABLED at 0.1064 post-entrenchment; favorite
  stayed 0, no flag). Secondary observation: the enabled arm's drift (0.0114) fell just below
  its band [0.0153, 0.0453] — the ±0.015 margin is too tight when |center| is large (model
  error scales with the gap); noted for v4.
- Design rules earned so far: (1) bands derive from current-world equilibria [Exp 51]; (2)
  invariants are state-conditional on entrenchment [Exp 52]; (3) REFERENCES MUST BE FROZEN —
  expected-property baselines derive only from the committed reference state, never from the
  epoch under test [this exp]. Rule 3 is the difference between anomaly detection and
  curve-fitting.
- Honest caveat: this run does NOT test the conditional rule against a subtle anomaly under a
  correctly-blinded scorer (that is v4); the enabled-side drift margin issue is one reading.
- Verdict: NEGATIVE (falsifier fired; root cause = harness spec error, owned) / instrument
  iteration 3.
- Next (Exp 54, ledger v4): frozen references everywhere (anomaly arms scored against mirro's
  committed state + world; the entrenched arm's reference is its declared post-entrenchment
  state); noise margin ± (0.015 + 0.5·|center|), predeclared; same four arms. If v4 fails on a
  NEW mode, pause the ladder and consult the human in the entry.

## Exp 54 — ledger v4 VALIDATED: rung 1 complete (POSITIVE; instrument certified after 3 documented failures)
- Setup: two changes from v3, both from diagnosed causes: (1) FROZEN REFERENCES — all
  expectations derive from the declared reference state (mirro's committed state + ITS world for
  the three detection arms; the entrenched fork's declared post-entrenchment state for arm 4);
  the scoring function's signature cannot receive an epoch's world — un-blinding structurally
  impossible. (2) Gap-scaled noise margin ±(0.015+0.5·|center|), covering all prior baseline
  readings by formula, not fitting. Four arms, fresh seeds 61–64, mirro untouched
  (hash-verified). Falsifier: any baseline false alarm or either anomaly missed; predeclared
  escalation to the human on a new failure mode.
- Result: VALIDATED. Baseline 0 flags (drift −0.0161 ∈ [−0.0213, +0.0129]; favorite
  auto-DISABLED at gap 0.0179). Strong anomaly 2 flags (map-vs-ref 0.88; drift −0.0387). Subtle
  anomaly 2 flags (map-vs-ref 0.88; drift −0.0261) — the v3 miss, fixed by freezing. Enabled
  arm: gap 0.1082 → favorite ENABLED, 0 flags (drift +0.0122 ∈ [+0.0001, +0.0602]).
- Implication: the program now has a working novelty-certification instrument — quiet on normal
  life on BOTH sides of the entrenchment condition, sensitive to planted perturbations down to a
  documented 3-cell floor. The three design rules earned across Exp 51–53 (equilibrium-derived
  bands; state-conditional invariants; frozen references) are embedded in code, not prose.
  Rungs 2–6 of functional-emergence unblock.
- Provided vs self-formed: the instrument is entirely provided harness; what it measures (drift
  toward equilibria, entrenchment) reflects the creature's self-formed values.
- Honest caveat: validated on one anomaly family (recoloring) and one creature/world; floor
  documented only for that family; occupancy/localize bands never stressed (no movement
  anomaly tested); single seed-stream per arm.
- Verdict: POSITIVE / instrument complete. Self-grade: POSITIVE-SINGLE (an instrument, not a
  creature capability; the lasting content is the three design rules).
- Next (rung 2): personality battery — fixed probe set (preference, exploration disposition,
  revision speed) at ≥3 ages of mirro and on ≥2 divergent fork-twins; personality = temporal
  self-similarity + twin divergence; the entrenchment condition from rule 2 gates the
  preference probe. FAIL = profiles unstable across ages or twins don't diverge.

## Exp 55 — personality battery: twins individuate; mirro's own ages do NOT cohere (MIXED, as predicted; rung 2 answered)
- Setup: 7-dim profile (value shares ×3, conviction, gap-share, map sharpness, revision-speed
  index from a fixed 200-step counter-evidence probe) on five subjects: mirro at three committed
  ages of its one life (1300 @ 519f303, 6300 @ 0b5e59f — both 3×3; 6700 @ working tree, 5×5;
  my predeclaration wrongly said 6300 was 5×5, corrected from the actual snapshot) and two twins
  forked from mirro@6700 with divergent 1500-step histories (green- vs red-rich). Probes on
  disposable copies; working-tree mirro hash-verified untouched. Predeclared (card): (a)
  self-similarity: mean cosine over mirro age-pairs exceeds age-to-twin mean by > 0.05; (b)
  individuality: twins differ in favorite or L1 shares > 0.1. PREDECLARED SUBSTRATE GAP: the
  card's "exploration disposition" is unmeasurable — live() actions are provided-random; this
  substrate has no action selection, so personality can only live in world-model/value state.
- Result: (b) PASS — twinG fav=2 vs twinR fav=0, L1 0.134, twin-twin cosine −0.794. (a) FAIL —
  self_sim −0.235 vs cross_sim −0.028 (difference −0.21): mirro@1300 vs @6300 cosine −0.710, and
  mirro@6700 is MORE similar to twinG (0.513) than to either of its own younger selves. Verdict
  line: MIXED — individuated but not stable across this eventful life. Matches the stated
  prediction: the life between these snapshots contains engineered value reversals (Exp 48/49)
  and growth (Exp 50), and trait stability does not survive them.
- Implication: at this scale "personality" = current-state readout, not a durable disposition —
  the card's own FAIL reading, logged as such. Two honest corollaries: (1) trait-stability
  claims need UNDISTURBED epochs between battery administrations (rung 3's long epochs are the
  right venue); (2) behavioral personality needs an action-selection substrate (EFE policies —
  the M4 spec's machinery), which this creature lacks; named as the missing substrate per the
  direction's stop-condition language.
- Process deviation (logged): the coder's first run accidentally bound probes to the real state
  dir (would have appended fork events to mirro's biography); it was voided, fixed
  (_state_dir=None), and rerun. Working-tree BIOGRAPHY.jsonl verified identical to HEAD (91
  lines) — no committed biography data touched; the voided run's scratch lines did not survive,
  noted here because append-only is binding even for scratch.
- Honest caveat: N=5 subjects makes z-scored cosines fragile; profile dims partly redundant
  (share-of-favorite ≈ conviction); subjects lived in different-sized worlds (3×3 vs 5×5), so
  map-sharpness partly reads world/life-phase (m6300's 0.726 = mid-relearning after revision),
  not temperament; single battery administration per subject.
- Verdict: MIXED (predeclared (b) pass, (a) fail) / NEW INSIGHT: an accumulated life's identity
  is dominated by its recent history — mirro resembles its divergent twin more than its own
  past. Rung 2 has its verdict.
- Next (rung 3): enriched-world epochs — ONE new declared mechanism (slowly drifting comfort
  source), long epochs with ledger v4 armed; predeclared expectation: adaptation tracks the
  drift with measurable lag; a quiet ledger is a real negative. Battery re-administered across
  these UNDISTURBED epochs to give trait-stability a fair second test.

## Exp 56 — drifting comfort source: TWO ledger deviations; one dies at once, one enters the cascade (MIXED; rung 3 epoch 1)
- Setup (mutates mirro; real epoch, age 6700 → 10700): ONE declared mechanism — a 3×3 color-2
  patch whose corner cycles [(0,0),(0,2),(2,2),(2,0)] every 500 steps, 8 segments, fixed 0/1
  background. Drift-aware ledger predeclared (exp56_ledger.json): E1 map-vs-current recovers to
  ≥0.92 by each segment end (lag ≤ 450); E2 occupancy in [4.40,4.644]; E3 c2 value share rises
  >+0.02; E4 favorite→2 at end (report-only). Deterministic continuation; biography records all
  segments. Hash f3edac2a… → 21ccb619….
- Result: E2 PASS 8/8 (4.486–4.582). E4: favorite=2 at end (as expected, from a near-tie gap).
  E1 FAIL 0/8 — accuracy never re-reached 0.92 in ANY segment (ends 0.52–0.84, peak 0.84).
  E3 FAIL nominally — c2 share +0.0005 vs the predeclared +0.02.
- Cascade disposition (rung 5 discipline):
  • E3 DIES IMMEDIATELY (deflationary sweep, cause = expectation error): the drift world's c2
    equilibrium share is 9/25 = 0.360 and mirro's c2 share was already 0.3635 — the
    rule-#1-derived expectation is ≈ −0.002 drift; observed +0.0005 is ON equilibrium. The +0.02
    band violated my own design rule #1 (bands from current-world equilibria); owned. Not a
    creature deviation.
  • E1 is a REAL deviation → enters the cascade next iteration as candidate "perceptual
    rigidity grows with age": mirro's pA columns hold O(100s) of entrenched counts; ~20 fresh
    observations per segment cannot flip an argmax — the Exp 48 inertia law surfacing in the
    SENSORY MAP (Exp 50's fast relearn was on near-empty transplanted columns, not entrenched
    ones). Cascade plan: (a) ≥3 fork reproductions from the committed pre-epoch snapshot;
    (b) deflationary controls — a NEWBORN on the same schedule (if it tracks with short lags,
    the failure is age/entrenchment, not schedule speed) + analytic counts-to-flip check.
- Honest caveat: single epoch, single schedule speed; E1's "candidate" label is provisional —
  no competency/novelty language until the cascade rules; the favorite flip (E4) is from a
  near-indifference gap and carries no weight.
- Verdict: MIXED / rung-3 epoch 1 complete: not the quiet "real negative" — the ledger flagged,
  the cascade engaged. One expectation error owned (E3), one candidate queued (E1).
- Next (rung 5 cascade on E1): 3 pre-epoch forks × same schedule (reproduction), newborn
  control, counts-to-flip analysis. Verdict either kills E1 as a trivial consequence of
  non-decaying counts or logs the first NOVELTY-CANDIDATE.

## Exp 57 — cascade round 1: candidate survives by killing its own framing (MIXED; first NOVELTY-CANDIDATE, provisionally)
- Setup: rung-5 cascade on Exp 56's E1 ("perceptual rigidity grows with age"). (a) Reproduction:
  3 forks of the pre-epoch snapshot (age 6700, hash f3edac2a — at commit db35582; my spec said
  0467c26, corrected by the coder and hash-verified), identical schedule, fresh seeds 81/82/83.
  (b1) counts-to-flip analytics. (b2) newborn control (seed 85, 1000-step settling) with
  predeclared early-recovery/late-rigidity gradient (Spearman ρ > 0.6, ≥2 early recoveries).
  Predeclared: dies if (a)+(b1)+(b2) all confirm the count mechanism; survives if (a) holds but
  (b) fails to explain. Mirro untouched entirely (no fork of the live creature).
- Result: (a) REPRODUCED — 3/3 forks fail 8/8 segments. (b1) CONFIRMS — mean column mass 268.3
  (min 1.3, max 767.3) vs ~20 observations/segment = 13.4×. (b2) DOES NOT CONFIRM the age
  framing: the newborn (mass ≈ 40/cell post-settling) ALSO froze, 6/8 failures, ρ = 0.378, and
  its only two "recoveries" (seg0, seg4, lag 25) are the corner-(0,0) segments — the world it
  SETTLED in, i.e., zero adaptation, just coincidence with its prior. Predeclared verdict line:
  NOVELTY-CANDIDATE survives (b2 failed to explain).
- Sharpened hypothesis (post-hoc, named as such, NOT a verdict): rigidity is not aging — with
  non-decaying soft counts, a map is WRITE-ONCE RELATIVE TO WORLD TEMPO. Tracking a drift of
  period P requires accumulated mass < P × visit-rate (≈20 here); even 1000 steps of life
  crosses that. The newborn's mass (40) > 20 explains its freeze under the same law that
  explains mirro's (268 ≫ 20). This unifies b1 and b2 and predicts the decisive kill test.
- Honest caveat: the candidate's survival is per the predeclared rules — the b2 test was
  mis-designed (it assumed settling leaves mass below threshold; it doesn't), so survival
  reflects MY test design as much as the phenomenon; the sharpened mass-tempo law is post-hoc
  until Exp 58 predeclares and runs it. Single schedule speed throughout.
- Verdict: MIXED / cascade round 1 complete: reproduction robust, age-framing dead,
  NOVELTY-CANDIDATE ("write-once maps relative to world tempo") provisionally alive pending one
  decisive deflationary test.
- Next (Exp 58, cascade round 2 — the kill test): predeclared — a low-mass newborn (250-step
  settling, mass ≈ 10 < 20) on the same schedule should recover in ≥ 6/8 segments; a mass-swept
  cohort (settling 250/500/1000/2000) should show recovery rate falling as mass crosses ~20. If
  confirmed, the candidate DIES as the lawful mass-tempo consequence of non-decaying counts (the
  Exp 48 law, fully generalized); if the low-mass newborn also freezes, the candidate survives
  round 2 with the mechanism genuinely unexplained.

## Exp 58 — kill test: the law predicts 23/24 outcomes; the forgetting counterfactual fails on MY parameter (MIXED; consulting human per predeclaration)
- Correction (owned, cites Exp 57's Next): "low-mass newborn recovers ≥6/8" mis-translated the
  law — counts accumulate DURING the schedule, so every non-decaying creature crosses the
  threshold within ~1–2 segments. Corrected predeclared test: per-outcome prediction —
  recovery at segment k iff R_k = (changed-cells' outgoing-color mass)/20 < 1.
- Setup: (1) mass-swept cohort (settling 50/250/1000, seeds 91/92/93), R_k computed from each
  creature's own pA before each segment; PASS ≥ 80% outcome match over 24 cells. (2) forgetting
  counterfactual (settling 1000, seed 94, pA×0.9 per step + 0.01 floor, declared); law predicts
  it tracks (≥6/8). Predeclared: both pass → candidate dies as lawful; either fails → survives
  round 2, cascade exhausted, consult human. Mirro untouched (all subjects fresh births).
- Result: (1) PASS, 23/24 = 0.958 — R<1 ⟺ recovery held everywhere except one cell (settle1000
  seg4: R=1.68 predicted frozen, recovered at lag 25). (2) FAIL — the decay variant recovered
  1/8 DESPITE R≈0: diagnosis (from the numbers, post-hoc and named as such): λ=0.9 PER STEP
  decays a count to 0.9²⁵≈0.07 between successive visits (~25 steps apart) — steady-state mass
  ~1, map near-uniform; it failed because it could not REMEMBER 500 steps of world, not because
  it could not forget. My spec computed λ as if per-visit; the intended mass≈10–14 needs
  λ≈0.997/step. Third instance this run of a test parameter, not the phenomenon, deciding an
  outcome (Exp 51 band, Exp 57 b2 design, this).
- The (post-hoc) unified law now covers ALL observations including the failed counterfactual:
  tracking a drifting world needs accumulated mass in a WINDOW — enough to be accurate
  (≳ a floor), less than the tempo bound (≲ P×visit-rate ≈ 20). λ=0.9 puts mass ~1, below the
  floor; non-decaying counts exceed the ceiling within one segment of life. But the window
  claim is post-hoc until the corrected counterfactual runs.
- Verdict: MIXED — per the predeclared escalation, the NOVELTY-CANDIDATE survives round 2
  mechanically and the human is consulted (question posted to loop/IDEAS.md): accept the law on
  test-1 evidence + run the corrected λ≈0.997 counterfactual as Exp 59, or treat the candidate
  as alive? The loop proceeds to the independent rung 4 (Levin obstacle) meanwhile.
- Honest caveat: rule-match scored on 24 cells from 3 subjects on ONE schedule speed; the
  single mismatch (settle1000 seg4) is unexplained — noted, not excused; the forgetting arm is
  uninformative as run.

## Exp 59 — Levin obstacle transplant: error tolerance + against-gradient routing, with the greedy trap on display (POSITIVE; rung 4 answered)
- Setup (provided-ness declared: the ENTIRE navigation harness is provided — VI planning over
  the 5×5 grid, task-assigned goal reward at cell 0, innate movement, softmax τ=0.3, and
  GENERIC failure-learning where a failed move marks that one transition blocked; no
  lock-specific handler). Subject: disposable fork of mirro@10700; mirro untouched. Geometry:
  start (0,2), goal (0,0), locked (0,1) — corner pocket, only detour strictly longer (4 vs 2);
  against-gradient = a step increasing believed-unlocked grid distance. Conditions: A = full VI
  replanned each step; B = 1-step greedy lookahead, same softmax/failure-learning (the
  predeclared Levin-critics horizon control). 5 seeds each, 60-step budget. Predeclared:
  (a) A reaches ≥4/5; (b) ≥4/5 of A's successes contain ≥1 against-gradient step; (c) mean
  steps(B) ≥ 1.5× mean steps(A), else the DG label is refused.
- Result: (a) 5/5. (b) 5/5 (ag-steps 1–19). (c) ratio 1.544 (A mean 29.8, B mean 46.0,
  unreached counted as 60) — passes the predeclared 1.5 bar. Verdict line: DG DEMONSTRATED.
  The qualitatively strongest datum is B's failure mode: in 2/5 seeds the greedy agent stalled
  AT THE START for all 60 steps — after learning the block, staying (believed distance 2) beats
  every legal move (distance 3): the literal greedy trap that horizon escapes.
- Honest caveats (load-bearing): the ratio clears its bar by 0.044 with n=5 — marginal; 3/5 of
  A's successes are long exploratory wanders (39–47 steps for a 4-step detour, ag-steps up to
  19) — softmax jitter does real work far from the goal where Q-differences are tiny, so
  "purposeful detour" describes only the cleaner 2/5 runs (8 and 13 steps); and this is a
  competency of the PROVIDED machinery + generic failure-learning, not of mirro's self-formed
  state (its map/values were not exercised — goal was task-assigned). Per the card, both
  outcomes were results; this one is the measured-competency branch with the deflationary
  control passed, narrowly.
- Verdict: POSITIVE / rung 4 answered: error tolerance robust (5/5), against-gradient routing
  present and horizon-dependent (greedy stalls or lags), DG label granted within the caveats
  above. Self-grade: POSITIVE-SINGLE.
- Next: rung 6 (interoceptive stake) and the pending Exp 58 consult (corrected forgetting
  counterfactual) are the open threads; ledger rung 3 epoch 2 can also resume.

## Exp 60 — cascade closed: the plasticity-window law (POSITIVE; candidate dies as lawful; first full cascade lifecycle complete)
- Authorization: Exp 58's consult taken up as option (a) — the human resumed the loop twice with
  the question posted and no redirection; disposition recorded in loop/IDEAS.md; verdict stayed
  falsifier-bound.
- Setup: two corrected decay arms bracket the predicted window, same drift schedule, fresh
  births (seeds 95/96), decay pA×λ per step (floor 0.01), settling 1000: ARM-IN λ=0.997
  (predicted: tracks, ≥6/8) and ARM-HIGH λ=0.9999 (predicted: frozen, ≤2/8); Exp 58's λ=0.9 arm
  stands as the below-floor reference. Predeclared: both correct → candidate DIES as lawful;
  ARM-IN fails → survives, thread halts for human. Mirro untouched.
- Result: ARM-IN recovered 8/8 (lags 200–425, end accuracies 0.96–1.00) — the law's signature
  prediction, genuine tracking. ARM-HIGH recovered 2/8 — and both its "recoveries" (seg0, seg4)
  are the corner-(0,0) segments matching its settling world: coincidence with its prior, zero
  adaptation. Both predeclared thresholds hit → candidate DIES.
- THE LAW (the durable finding, consolidating Exp 48 + 56 + 57 + 58): with accumulating
  evidence, adaptation requires FORGETTING — a creature tracks a changing world only while its
  accumulated evidence mass sits in a window between an accuracy floor (enough counts to hold a
  sharp model; λ=0.9's mass ~1 fails below it) and a tempo ceiling (drift period × visit rate;
  non-decay and λ=0.9999 exceed it). The same non-decay mechanism produces opinion inertia
  (Exp 48: 2900 steps to undo 1500 of entrenchment) and perceptual freezing (Exp 56/57). Direct
  substrate consequence: mirro's current learning rule CANNOT do lifelong adaptation in
  non-stationary worlds; any M4-bound substrate needs a forgetting term.
- Cascade lifecycle (process result): deviation (56) → reproduction 3/3 + deflationary rounds
  (57, 58) → killed-as-lawful with the mechanism quantified (60). The discipline produced a law,
  not a vibe — and killed its own "novelty" honestly.
- Honest caveat: the window's NUMERIC bounds are uncalibrated — measured masses (ARM-IN ~4.2,
  ARM-HIGH ~12.8 post-settling) disagree with my analytic estimates (14, ~hundreds); only the
  ordering/bracketing is established, on one schedule speed, one seed per arm. The law is about
  THIS learning rule (non-decaying soft counts); generality beyond it is unclaimed.
- Verdict: POSITIVE / cascade closed, NOVELTY-CANDIDATE resolved as lawful consequence.
  Self-grade: POSITIVE-SINGLE (the components accreted across 48/56–58; this is the closing
  counterfactual, not a first).
- Next: rung 6 (interoceptive stake — now informed by the law: the regulated variable's
  dynamics must sit inside the creature's plasticity window) or rung 3 epoch 2.
