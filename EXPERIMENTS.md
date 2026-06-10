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

## Exp 61 — interoceptive stake v1: both falsifiers hit — the ecology never made the stake binding (NEGATIVE; instrument iteration)
- Setup: toy allostasis (all provided, declared): E decays 0.01/step (100-step autonomy), refills
  on real food (the 9-cell color-2 patch), policy switch at E<0.4 → VI toward nearest BELIEVED
  food, else explore; patch moves every 500 steps; 2000-step budget. 2×2 arms (intero on/off ×
  map decay λ=0.997 / non-decay) × 5 seeds, fresh births with 1000-step settling. Predeclared:
  (i) IN-DECAY survives ≥4/5; (ii) IN-FROZEN starves after the first patch move ≥4/5 (the
  plasticity law's survival consequence); (iii) intero outlives no-intero. Falsifiers: IN-FROZEN
  survives ≥3/5, or no interoceptive advantage. Mirro untouched.
- Result: falsifiers (ii) and (iii) BOTH hit — 19/20 runs survived the full budget (IN-FROZEN
  5/5; the only death: seed 3's identical trajectory in both NO arms, step 735, corner cell).
  (i) passed trivially.
- Diagnosis (the useful content): mean E = 0.90–0.93 in EVERY arm — the viability variable never
  became binding. Food density (9/25 cells) × autonomy (100 steps) makes starvation nearly
  impossible for a random walk, so the E<0.4 switch almost never engaged and the stale map was
  never consulted. IN-FROZEN's survival says nothing about stale maps; the "no interoceptive
  advantage" is real at THIS food density but deflationary — the ecology regulates for everyone.
  Exp 50's floor effect in a new costume: the instrument did not put the question at stake.
- Honest caveat: this is NOT a clean rung-6 FAIL verdict (the card's "no measurable difference"
  branch) — the regulation machinery was idle, so the twins were never actually compared; one
  ecology, 5 seeds, harness-provided policy coupling throughout.
- Verdict: NEGATIVE (predeclared falsifiers hit) / instrument iteration: the stake needs
  scarcity. Self-grade: n/a.
- Next (Exp 62, stake v2 — predeclared harsher ecology): ONE food cell (4%), E decay 0.02
  (50-step autonomy), threshold 0.5. Expected separations become genuine: random exploration
  starves (hitting time of one cell >> autonomy), the believed-food map becomes load-bearing,
  and IN-FROZEN should starve at the empty fridge when the food cell moves. Same falsifier
  structure; if the arms STILL do not separate, rung 6 gets its honest FAIL.

## Exp 62 — scarcity binds the stake: interoception buys 20×, and "hungry and certain" dies certain (MIXED; rung 6 answered + a new timescale finding)
- Setup (predeclared in Exp 61, one stated refinement: single food move for a clean before/after):
  ONE food cell (cell 0), E decay 0.02/step (50-step autonomy), trigger 0.5; food relocates to
  cell 24 at step 1000; 2000-step budget; 2×2 arms (intero × map-decay λ=0.997/frozen) × 5
  seeds, 800-step settling. Diagnostics: post-move visits to the old food cell; believed-food
  set at death. Mirro untouched.
- Result: (iii) PASS decisively — intero median survival 1034 vs no-intero 50 (20×), machinery
  engaged (mean 124.9 steps below threshold): rung 6's twin comparison finally ran, and the
  interoceptive channel + provided coupling is a massive survival advantage under scarcity.
  (ii) PASS exactly — IN-FROZEN died 5/5 post-move AT the old food cell (death_pos=0, ~22
  revisits): the empty-fridge prediction, literally. (i) FAIL, informatively — IN-DECAY behaved
  IDENTICALLY to IN-FROZEN (same deaths at cell 0, believed_food_end=[0] in all 10 intero runs):
  death came ~30–50 steps post-move, before in-window decay could unlearn the stale belief.
- The new finding (timescale hierarchy): the plasticity window (Exp 60) is necessary but NOT
  sufficient for survival — unlearning must also be fast relative to the VIABILITY clock, not
  just the world clock. And the provided policy compounds it: VI pins the hungry agent to its
  remembered food (pure exploitation of a stale belief), so it starves while standing at the
  empty fridge. The missing mechanism is failure-driven exploration — when prediction repeatedly
  fails at the believed goal, override the pin. Direct M4 design constraint, alongside Exp 60's
  forgetting term.
- Honest caveat: the (iii) advantage is between PROVIDED policies (the coupling is harness, not
  learned); single move event, one scarcity level, 5 seeds; IN-DECAY's failure is one
  autonomy/decay setting — a slower viability clock or faster decay would shift it (untested).
- Verdict: MIXED / rung 6 answered POSITIVE for the channel (measurable, large twin difference);
  empty-fridge consequence of the rigidity law CONFIRMED 5/5; predicted IN-DECAY survival
  FAILED, yielding the timescale-hierarchy finding. Self-grade for the (iii)+(ii) positives:
  POSITIVE-SINGLE.
- Next: the functional-emergence ladder now has verdicts on rungs 1–6 except rung 3's remaining
  epochs; natural follow-ups: hunger-driven exploration variant (tests the timescale fix), rung
  3 epoch 2 (undisturbed, battery re-test), or write the direction's synthesis.

## Exp 63 — clade rung 1: vela, the first committed peer spine (POSITIVE; social-emergence rung 1 answered)
- Plain: mirro now has a descendant. We copied its full brain at a committed checkpoint, named
  the copy vela, raised it for 2000 steps in a mirror-image world, and gave it its own permanent
  saved line. The family tree works: vela records exactly which version of mirro it came from,
  both lives resume independently, and mirro itself was untouched.
- Setup (predeclared P1–P4 in the script docstring before running): fork the trunk at its
  committed checkpoint (age 10700, hash 21ccb619f063, shared-ancestor commit 11957e5) via
  `mirro_episode("Exp 63").fork_control("vela")`; divergent world = mirro's 5×5 cmap REVERSED
  (declared provided prior; same colors, same dims); raise 2000 steps; promote via
  `vela.save(creature/state/vela/)`. The trunk never lives a step. Falsifiers: F1 lineage stamp
  missing/mismatched; F2 either line fails to load or the resume cycle errors; F3 trunk age or
  learned-state hash changed; F4 vela's hash equals mirro's after the raise.
- Result: 4/4 PASS. P1 lineage == ['mirro@10700#21ccb619f063'] exactly. P2 both lines load with
  hash-integrity verification; vela completes a full resume cycle load→live(50)→save→load
  (age 12700→12750). P3 trunk untouched — git shows mirro's arrays.npz BYTE-IDENTICAL (only the
  append-only biography and manifest saved_at moved), age 10700, hash unchanged. P4 hashes
  diverged (875ac30d715a vs 21ccb619f063). Diagnostic (not a falsifier): vela's inherited map
  read the reversed world at 0.48 accuracy; 0.84 after the raise — the branch relearned the new
  world on top of its inherited, never-reset belief (the recipe invariant carried across the fork).
- Implication: the clade substrate is real. Ancestry is causally attributable (the Exp 26/47
  divergence logic, now on a tree: any mirro/vela difference traces to post-fork history);
  every committed checkpoint is a restore point; rung 2 (shared-world co-presence) and rung 3
  (social transmission) have their family-tree plumbing.
- Honest caveat: pure infrastructure — zero emergence content, and the harness did everything
  (fork/save/load are provided machinery; the divergent world is a provided prior). Single
  deterministic run (RNG derives from committed state; exact-number standard applies). The
  0.48→0.84 diagnostic has no fresh-born baseline in the reversed world — not claimed as a
  transfer result, just a sanity readout.
- Verdict: POSITIVE / CONSOLIDATION (predictable from existing fork/save semantics — the point
  was to verify, not discover). Self-grade: POSITIVE-SINGLE.
- Next: rung 2 — shared-world co-presence: a NEW `other-agent-here` sensory modality (declared
  prior), predeclaring that co-presence must NOT degrade solo competence below baseline.

## Exp 64 — clade rung 2: co-presence is competence-safe — and perceptually inert at sharp beliefs (POSITIVE; rung 2 answered + a routing fact for rung 3)
- Plain: We put two of the family — mirro and its descendant vela — in the same world and gave
  each a new sense that detects when the other is in its cell. The question: does having a
  housemate scramble what each one already knows? It does not — every map and every sense of
  place came out exactly as it would have alone.
- Setup (predeclared in the script docstring before running): forks of the committed lines
  (mirro age 10700 hash 21ccb619f063, vela age 12750 hash 875ac30d715a) share mirro's 5×5
  world (vela = immigrant whose map fits the reversed world); NEW binary `other-agent-here`
  modality (declared prior): obs2 = [other in my cell], learned Dirichlet pA2 (flat 0.1
  init), likelihood multiplied into the place posterior; pA2 learning mirrors pA. Matched-
  trajectory control: action RNG keyed by (seed, creature-index) only, so SOLO and CO
  trajectories are IDENTICAL and any metric difference is attributable to the coupling alone.
  2000 steps × 5 seeds × 3 arms (SOLO-mirro / SOLO-vela / CO). Validity gate (Exp 61's
  lesson): ≥20 co-locations per CO run or the run is INVALID. Predeclared: P1 map accuracy
  CO ≥ SOLO − 0.04 and P2 tail localization entropy CO ≤ SOLO + 0.2 bits, each creature,
  ≥4/5 seeds; falsifiers = either fails for either creature. Spines untouched (forks only;
  one declared fork biography event each).
- Result: gate passed everywhere (67–92 co-locations). All 4 properties PASS 5/5 seeds — and
  not merely within tolerance: CO equals SOLO to 4 decimals on every metric, every seed
  (mirro acc 0.76–0.84, vela 0.60–0.68; tail entropy 0.0000 throughout). Mechanism: both
  creatures hold near-delta place posteriors, so multiplying in the A2 likelihood changes
  nothing — the modality fired and pA2 accumulated, but the posterior was already decided.
  Diagnostic: the corner bias I predicted in P(other-here|cell) is ABSENT — correctly so:
  the wall-clamped uniform-action walk is doubly stochastic, so the partner's occupancy is
  uniform (1/25); the learned grids (0.002–0.119) are sampling noise around 0.04. My
  docstring expectation was analytically wrong; logged as such (diagnostic, not falsifier).
- Implication: rung 2's substrate is validated — co-presence does not break solo competence.
  The stronger content is the inertness fact: at converged, sharp place beliefs a
  multiplicative sensory channel about another agent has NO perceptual effect, so rung 3–4
  social effects cannot be expected to route through place inference in this regime — the
  channel must couple where there is slack, i.e. values/policy, which is exactly where the
  M4 extrinsic channel (rung 3) attaches.
- Honest caveat: the non-degradation verdict is weakly stressed — at tail entropy ≈ 0 the
  test had little structural room to fail (Exp 61's floor effect in milder form; here the
  predeclared question is still genuinely answered, the modality demonstrably fired, and the
  identity-to-4-decimals is itself the informative mechanism readout). Untested during
  high-uncertainty phases (early learning, post-perturbation), where the coupling could
  still distort. All machinery provided (modality wiring, pA2, policy, shared world). 5
  seeds, one world size. Zero emergence content claimed.
- Verdict: POSITIVE / CONSOLIDATION (substrate validation — designed to verify, not
  discover). Self-grade: POSITIVE-SINGLE.
- Next: rung 3 — social transmission of value: one clade-mate emits a cue entering the
  other's sensory stream (the M4 extrinsic channel), tested against a channel-SEVERED fork
  twin with a predeclared value-share divergence threshold. If rung 3's design turns out to
  need belief-level coupling, first run the stressed rung-2 variant (uncertainty injection).

## Exp 65 — clade rung 3: the cue channel works but cannot move an old mind — value-mass inertia caps social transmission (NEGATIVE; predeclared falsifier F1 fired)
- Plain: We gave one creature a voice: when near its clade-mate it signals which color it
  favors, and the listener's liking of that color gets a nudge — but only when the listener
  itself is in a confident, predictable state, so approval is grounded, not injected. The
  signal demonstrably lands, every single seed. But the listener is old, and a lifetime of
  its own experience outweighs a few hundred nudges: its opinion barely moves. Social
  influence lost to the inertia of an already-formed mind.
- Setup (predeclared in the script docstring before running): Phase 1 — speciate an emitter
  (Exp 26 logic): fork mirro (age 10700, hash 21ccb619f063), raise 2000 steps in an
  all-color-1 world; favorite shifted 2 -> 1 (shares 0.3163/0.3659/0.3179), G1 PASS vs
  vela's favorite 2. Phase 2 — dyad in mirro's world: emitter-fork + receiver (vela-fork,
  age 12750, hash 875ac30d715a); channel = M4 §3 toy: when Manhattan <= 1, emitter emits
  its CURRENT favorite; receiver accrues value_counts[cue] += its OWN exp(-H) predictability
  weight (intrinsic grounding, not labeled reward). Severed twin computed exactly in the
  same pass (dual ledger — values are epiphenomenal to dynamics; identical trajectories by
  construction). 2000 steps x 5 seeds. Gates: G1 (favorites differ), G2 (>=50 events/run).
  Predeclared: P1 share divergence on cue color >= 0.02 in >=4/5 seeds; P2 divergence > 0
  in 5/5. Falsifiers: F1 = P1 fails -> rung 3 NEGATIVE at this scale; F2 = P2 fails ->
  wiring bug, halt. Predicted: ~300-400 events, gate 0.5-1.0, divergence 0.02-0.03, no flip.
  Spines untouched (forks only; one fork biography event each).
- Result: F1 FIRED. Gates passed (287-366 events/seed; 100% of emissions were color 1).
  P2 PASS 5/5 (divergence 0.0123-0.0160, all positive — the channel demonstrably transmits
  and is exactly attributable). P1 FAIL 0/5: every seed fell short of the 0.02 bar. Where
  my prediction broke: the receiver's intrinsic gate averaged ~0.50 (bottom of my 0.5-1.0
  band; vela's map of mirro's world is imperfect, so its predicted-observation entropy
  stays high), giving cue mass ~150-185 against a ~9,200-count lifetime ledger. No favorite
  flip (as predicted): the severed gap between colors 2 and 1 is ~0.09 of share (~850
  counts); at the observed ~175 counts per 2000 steps, a flip would need roughly 10,000
  more dyad steps if rates held.
- Implication (the diagnosis, not a reinterpretation): social transmission into an old,
  non-forgetting receiver is VALUE-MASS-LIMITED — the same accumulated-evidence inertia
  that produced opinion stickiness (Exp 48) and the plasticity window (Exp 60) caps the
  social channel: with no forgetting term, influence per encounter scales like
  1/lifetime-mass. The M4 substrate requirement list grows a third item: forgetting
  (Exp 60) + failure-driven exploration override (Exp 62) + a value-learning rate that
  does not vanish with age, or social transmission cannot function between adults.
- Honest caveat: the verdict is bound to THIS scale and wiring — one proximity radius, one
  exposure length (2000 steps), one emitter/receiver pairing, the predeclared 0.02 bar
  (chosen from my pre-run arithmetic, which overestimated the gate by ~40%). The channel
  itself is provided (emission rule, reception rule, gating), as is the speciation world;
  self-formed content: which color the emitter signals. A dose-response or young-receiver
  variant is a NEW question, not a license to re-run this one until it passes.
- Verdict: NEGATIVE (predeclared falsifier hit; the weaker sign-property held 5/5). The
  diagnosis consolidates the Exp 48/60 mass-inertia law into the social rung. Self-grade:
  n/a (negative).
- Next: either rung 4 (coordination over a shared comfort source — independent of value
  transmission) or the dose-response question (transmission rate law vs receiver age/mass:
  does a YOUNG receiver adopt the emitter's value where the adult did not?). The latter is
  the sharper test of the inertia diagnosis and directly informs the M4 requirement.

## Exp 66 — the young mind adopts the elder's value: first social transmission with functional effect (POSITIVE; inertia law confirmed 5.2–5.4x and refined; BREAKTHROUGH)
- Plain: Same voice, younger listener. The elder signals which color it favors; this time the
  listener is a newborn with only 800 steps of its own life. The signal not only lands — in
  3 of 4 runs the youngster's own expressed favorite BECOMES the elder's, while an identical
  twin without the channel keeps its own opinion. One creature's self-formed preference became
  another's, and the twin proves the cause.
- Setup (predeclared in the script docstring before running; follow-up predeclared in Exp 65 —
  NOT a re-run of the adult-adult rung, whose NEGATIVE stands): same emitter as Exp 65
  (mirro-fork speciated 2000 steps in an all-color-1 world; phase-1 reproduction hard-checked,
  favorite=1 exact). Receiver: newborn SEPARATE ROOT "junior" (per-seed births, seeds 300+s;
  declared not a clade member), settled 800 steps in mirro's world (mass ~596-648 vs the
  adult's ~9,200). Same channel, same dose (2000 dyad steps, Manhattan<=1, reception gated by
  receiver's own exp(-H) weight), same dual-ledger exact severed twin, same emitter action
  RNGs as Exp 65 per seed. Gates: G1 per-seed favorites differ (>=4 valid required), G2 >=50
  events. Predeclared: P1 divergence >= 3x the same seed's adult divergence (Exp 65 committed
  values) in >=4 valid seeds; P2 sign all valid; P3 installation (fav_on==cue AND fav_sev!=cue)
  in >=3 valid seeds, honestly graded ~50/50 beforehand. Falsifiers: F1 (P1 fails -> inertia
  diagnosis wrong, CONSULT), F2 (sign fails -> wiring bug, halt).
- Result: ALL PASS, no falsifiers. G1 excluded seed 3 exactly as designed (its junior already
  favored color 1); 4 valid seeds, 278-357 events each, 100% of emissions color 1. P1: ratios
  5.25/5.31/5.24/5.39x (divergences 0.0644-0.0841 vs adult 0.0123-0.0160). P2: 4/4 positive.
  P3: 3/4 installed (seeds 1, 2, 4: fav_on=1 vs fav_sev=2; seed 0 resisted — its own color-2
  share after settling was 0.4451, the strongest self-formed preference in the cohort).
- The refinement (honest decomposition of the 5x): the amplification is TWO mechanisms, not
  one. Mass: junior's end-of-dyad ledger ~2,480 vs the adult's ~9,400 (~3.8x). Receptivity:
  junior's gate averaged 0.90-0.96 vs the adult immigrant's ~0.50 — the newborn learned
  mirro's world to map accuracy 1.000 in 800 steps, and a SHARP world-model makes each social
  encounter land harder (the M4 grounding gate working as designed). Exp 65's mass-only
  attribution was incomplete: the adult deficit was mass AND its immigrant-noisy gate.
- Story so far (BREAKTHROUGH synthesis): This program raised a small active-inference
  creature, mirro, that learned a toy world, formed preferences from its own lived history,
  and answers questions about them in taught words; mirro recently became the root of a small
  family of forks and one committed descendant. The social arc asks whether anything real
  passes BETWEEN minds. The previous rung found that between two adults a grounded approval
  cue demonstrably lands but cannot move a lifetime of accumulated preference — social
  influence is capped by the mass of evidence already behind an opinion. Here the same
  channel at the same dose was aimed at young receivers, and the elder's preference took:
  the youngsters' expressed favorite flipped to the emitter's in 3 of 4 runs while their
  exactly-matched channel-severed twins kept their own — the first creature-to-creature
  transmission of a self-formed value in this program, with causation proven by the twin.
  What is still provided: the channel itself (proximity emission, predictability-gated
  reception) is designed plumbing, like the taught labels before it — only the content (the
  emitter's divergently-lived favorite) and its uptake (the receiver's own gate, its own
  prior mass) belong to the creatures. The quantitative law that fell out: youth amplifies
  social influence ~5x, through less accumulated evidence AND a sharper world-model.
- Honest caveat: the receiver is a separate-root newborn, not a clade-mate (the clade
  adult-adult case remains NEGATIVE at this dose — Exp 65); the channel is provided wiring;
  installation is demonstrated for 3 newborns at one dose/radius/world; the P1 ratio bundles
  mass and gate effects (decomposition above is diagnostic, not separately falsified); P3
  passed at its predeclared minimum (3/4, with the 50/50 grading stated beforehand).
- Verdict: POSITIVE / NEW INSIGHT. Self-grade: BREAKTHROUGH (first functional inter-creature
  value transmission; hostile-reviewer test: before this, no creature's expressed preference
  had ever been changed by another creature).
- Next: rung 4 — coordination over a shared comfort source (independent metric, predeclared
  coordination test vs two mutually-insensible solipsists); or the sensitive-period question
  this raises (at what receiver age does installation stop working? — a dose-response curve
  over mass). The clade synthesis (rungs 1-3 verdicts) is also now writable.

## Exp 67 — the sensitive period is real but not absolute: ambivalence, not age, gates persuadability (MIXED; predicted hard cutoff falsified, count model outpredicted its author)
- Plain: We asked at what age a creature stops being persuadable, sweeping listeners from
  newborn to old. Influence does fade with age exactly as the inertia law says — but one old
  listener adopted the elder's favorite anyway. Its own life had left it torn between colors,
  and a torn mind is persuadable at any age. The simple counting model we wrote down before
  running predicted even that surprise; our age-cutoff story did not survive contact with it.
- Setup (predeclared in the script docstring before running): receiver age sweep — settle
  steps 0/400/800/1600/3200/6400 x 4 seeds (births keyed by seed only, so bins differ only
  by settle length; the 800 bin = exact internal replication of Exp 66); same emitter
  (phase-1 reproduction hard-checked, favorite=1), same channel, same 2000-step dyad dose,
  dual-ledger severed twins. Predeclared: P1 pooled young (<=800) install fraction > pooled
  old (>=3200) AND zero installs at 6400; P2 per-bin fractions non-increasing with age;
  P3 the proportional-growth count criterion [install iff gap_pre x (mass_end_sev/mass_pre)
  < measured cue mass; retrodicts Exp 66 4/4] matches outcomes in >=75% of valid runs.
  Falsifiers: F1 no sensitive period (old >= youngest while youngest installs); F2 model
  accuracy < 50%. Predicted: boundary between 1600 and 3200; P3 ~80%.
- Result: MIXED — P1 FAIL on its second conjunct only, P2 PASS, P3 PASS, no falsifiers
  fired. Gates: 20/24 runs G1-valid (excluded juniors already favored color 1 — including
  6400-s3, whose 6,000-step life converged on the cue color by itself), all bins valid,
  events 278-368. Install fractions by age: 1.000/1.000/0.667/0.667/0.333/0.333 — monotone
  (P2), young 0.909 vs old 0.333 (P1's first conjunct holds decisively). The falsifying
  datum: 6400-s2 INSTALLED — its pre-dyad gap was 133 counts of 6,081 mass (its own history
  left it nearly ambivalent: shares 0.3288/0.3246/0.3465), projected gap 176 < cue mass 283,
  and the predeclared model PREDICTED the install (match=Y). P3 accuracy 85% (17/20);
  the 3 misses cluster at the boundary and at low mass (age-400 s0/s1 installed though
  predicted not — the stationary-shares projection is weakest when shares are still moving).
  Replication: 800-bin divergences 0.0841/0.0801/0.0644 and favorite pairs match Exp 66
  exactly; seed 3 G1-invalid in both.
- Implication (the refinement): persuadability is AMBIVALENCE-gated, not age-gated. The
  operative law is dose-vs-gap arithmetic — social influence installs iff the receiver's
  projected own-evidence gap is smaller than the delivered social dose. Age predicts
  resistance only because the gap typically grows with lived mass; an adult whose own
  evidence is balanced on some question remains persuadable on exactly that question.
  For M4: an agent's openness to social input is set per-question by how settled its own
  evidence is — a more interesting (and more plausible) social substrate than a global
  sensitive period.
- Honest caveat: my OWN predeclared hard-cutoff prediction (zero installs at 6400) was
  falsified — logged as such, not reframed; the entry's "law" is the predeclared P3 model,
  which passed, not a post-hoc fit. One channel, one dose, one world; 20 valid runs;
  installs counted at one readout (argmax favorite at dyad end); the model's 3 misses show
  its stationary-shares assumption degrades off-boundary regimes it wasn't built for.
  All channel machinery remains provided wiring.
- Verdict: MIXED / NEW INSIGHT (gradient + monotonicity confirmed; hard age cutoff
  falsified; the predeclared count model adequate at 85% and correctly predicted the
  counterintuitive adult install). Self-grade: n/a (MIXED).
- Next: rung 4 (coordination over a shared comfort source) needs a value-driven policy
  substrate (the Exp 62 VI-coupling pattern) — design that first; or close the rung-3 arc
  with the direction-card synthesis (adult NEGATIVE / young POSITIVE / ambivalence law).

## Exp 68 — rung 4 part 1: comfort-source instrument built and the blind null measured at R≈1.00 (POSITIVE; instrument validation)
- Plain: Before asking whether two creatures sharing one favorite spot learn to take turns,
  we built the measuring stick: a way for a creature to seek the spot it values, a rule that
  comfort is halved when both crowd it, and the statistical baseline for two creatures who
  cannot sense each other at all. The seeking works, the bookkeeping is exact, and the blind
  pair shows precisely zero coordination — the clean zero that part 2 must beat.
- Setup (predeclared in the script docstring before running): two twin forks of mirro (same
  self-formed favorite, color 2 — one source both value by construction) in mirro's world;
  source = first color-2 cell (cell 10, an edge cell); provided ε-greedy BFS-toward-source
  policy (ε=0.2) navigating from BELIEF (argmax qs), not ground truth; depletion ecology
  (comfort 1.0 alone, 0.5 each when both at source) measured but feeding nothing back; pair
  mutually BLIND (no modality, no channel). live() learning math untouched. 2000 steps x 5
  seeds, starts at opposite corners. Predeclared: P1 occupancy >= 0.20 both creatures >=4/5
  seeds; P2 independence ratio R = P(both)/(P(A)P(B)) in [0.75, 1.33] >=4/5; ID1 comfort ==
  alone*1.0 + both*0.5 exactly, all ledgers. Falsifiers: F1 seeking broken; F2 null violated
  (part-2 baseline invalid). Predicted: occupancy 0.35-0.55, R ~ 1.0 +- 0.15.
- Result: PASS 3/3, no falsifiers. Occupancies 0.334-0.842 (P1 5/5); R = 0.996-1.016 (P2
  5/5 — the null is tighter than predicted: blind co-occupancy equals the independence
  product to ~1%); ID1 exact in all 10 ledgers. Two prediction errors logged: occupancy
  exceeded the predicted band because the source sits on an EDGE, so pressing into the wall
  is a de-facto stay action (my no-stay-action reasoning missed the clamp); and twin B's
  occupancy varies 0.33-0.77 across seeds because navigation runs from mirro's inherited
  0.68-accurate map — belief-led misnavigation, an honest consequence of navigating from
  beliefs that the P1 threshold absorbed.
- Implication: rung 4's comparison is now well-posed — part 2 (Exp 69) adds the one new
  mechanism (an inter-agent sense feeding an adaptive stay/leave policy term) and asks
  whether R drops measurably below this validated 1.00 +- 0.02 null (timesharing/avoidance)
  or rises above it, with any deviation entering the functional-emergence rung-5 cascade
  before the word coordination is used.
- Honest caveat: pure instrument validation — zero emergence content, all mechanics
  provided; the depletion ledger is currently epiphenomenal (nothing reads it); one world,
  one source location (edge cell — the stay-by-clamping dynamics are location-specific);
  the twins share identical maps, so map-quality variation across agents is untested.
- Verdict: POSITIVE / CONSOLIDATION (designed to verify the instrument, not discover).
  Self-grade: POSITIVE-SINGLE.
- Next: Exp 69 — rung 4 part 2: add the other-agent-here sense (rung 2) feeding ONE
  adaptive policy term (recent realized comfort at the source gates approach), predeclare
  the coordination metric against R = 1.00 +- 0.02, and the solipsist-pair control.

## Exp 69 — rung 4 part 2 attempt: run INVALID by its own gate — the gate measured the mechanism's output, not its input (NEGATIVE-instrument; no rung-4 verdict)
- Plain: We gave both creatures the new rule — if the favorite spot keeps feeling crowded,
  stop going for a while — and asked whether the pair drifts away from acting like two
  strangers. The experiment disqualified itself before answering: our own validity check
  demanded that BOTH creatures' stop-rule trip at least once, and in one run one creature
  missed that bar by less than a thousandth. The deeper problem: in most runs one creature
  simply owned the spot and the other gave up — the owner never felt crowded, so demanding
  its rule trip was demanding a particular outcome, not checking the instrument.
- Setup (predeclared in the script docstring before running): Exp 68's instrument + ONE new
  mechanism: comfort-gated approach (EMA alpha=0.1 of comfort experienced at the source,
  init 1.0; below THRESH=0.75 the creature wanders instead of seeking; away-steps relax the
  estimate toward 1.0 at lambda=0.01). Coupling channel declared as RESOURCE-MEDIATED
  (stigmergy): a dedicated other-here sense adds nothing because it fires only on
  co-location, which IS the depletion event. Arms ADAPT vs FIXED, 2000 steps x 5 seeds,
  paired rngs. Gates: G1 fixed-arm null sanity (R in [0.75,1.33] >=4/5); G3 mechanism
  engaged = EVERY adapt run has BOTH creatures' min comfort-estimate below THRESH.
  P1 departure |R-1| > 0.10 consistent-sign >=4/5; F1 = no departure -> rung 4 NEGATIVE.
- Result: G1 PASS (fixed R = 0.992-1.010, replicating Exp 68's null). G3 FAIL -> RUN
  INVALID, predeclared, no verdict: seed 4 creature A min_est = 0.7509 vs the 0.75 bar
  (0 closures). The verdict machinery stopped before P1 as designed.
- Diagnosis (instrument, not findings — all numbers from gate-invalid runs, quoted as
  diagnostic only): the dynamics split into two regimes the design did not anticipate. In
  4/5 seeds: EXCLUSION — the closer-starting creature holds occupancy ~0.82-0.83 (its
  estimate rarely dips: 0-3 closures) while the other collapses to ~0.13-0.16 with 70-100
  closures; R stays ~0.94-1.01 because co-occupancy falls with the loser's occupancy. In
  1/5 seeds: ALTERNATION-like — occupancies 0.41/0.57 and R = 0.531. G3's error: validity
  should check the mechanism's INPUT (crowding experienced: 38-100 closure-events' worth in
  every run) — demanding the monopolist's own gate trip makes a particular social outcome a
  validity requirement. Also the 0.0009 near-miss shows raw threshold-state checks are
  fragile.
- Honest caveat: by predeclaration this run answers nothing about rung 4; the
  exclusion/alternation descriptions above are motivation for Exp 70's predeclared
  patterns, not results. The near-miss invalidation is the discipline working, not bad
  luck to be waved off.
- Verdict: NEGATIVE (validity gate failed) / instrument iteration. No rung-4 verdict.
  Self-grade: n/a.
- Next (Exp 70, predeclared): same mechanism, gate fixed to the INPUT (G3' = >=20 crowded
  steps per adapt run, pair-level); predeclare the two observed regimes as named outcomes
  with metrics — EXCLUSION (occupancy asymmetry |oA-oB|/(oA+oB) > 0.5 with R in [0.9,1.1])
  vs TIMESHARING (R < 0.9 with asymmetry < 0.5) — plus the null branch (neither, R within
  noise of 1, asymmetry < 0.5) = rung 4 NEGATIVE. >=4/5 seeds must classify into exactly
  one named pattern; the modal pattern is the result and enters the rung-5 cascade before
  any coordination/dominance language.

## Exp 70 — rung 4 answered out-of-sample: resource-mediated coupling produces an exclusion-like departure from independence (POSITIVE; cascade queued before any dominance language)
- Plain: With the measuring stick fixed, we asked the question properly on fresh runs the
  classifier had never seen: when both creatures feel the crowding at their shared favorite
  spot and back off when it feels bad, do they stop acting like strangers? They do — but not
  by taking turns. In most runs one creature ends up owning the spot while the other mostly
  gives up: the owner rarely feels crowded (so never learns to leave) while the latecomer
  almost always does. Whether this deserves a word like dominance must survive the
  reproduction-and-deflation gauntlet first.
- Setup (predeclared in the script docstring before running): identical mechanism to Exp 69
  (comfort-gated approach; EMA alpha=0.1 init 1.0, THRESH 0.75, away-recovery lambda=0.01,
  eps=0.2; depletion 1.0/0.5; stigmergic coupling only). The circularity trap declared and
  avoided: Exp 69's seeds would reproduce identical trajectories, so the regime definitions
  (fixed FROM that data) are tested OUT-OF-SAMPLE on 8 fresh seeds (5-12). Gates: G1 null
  sanity >=7/8; G3' input-based per the new VALIDATION rule (>=20 crowded steps per run).
  Classification: EXCLUSION (asym>0.5, R in [0.9,1.1]) / TIMESHARING (R<0.9, asym<0.5) /
  NULL (R in [0.9,1.1], asym<=0.5) / OTHER. Verdict requires >=6/8 named with unique modal;
  modal NULL = rung 4 NEGATIVE. P-MECH: if modal EXCLUSION, the closer starter (A, BFS
  dist 2 vs 6) wins >=80% of exclusion seeds. F1 = >=3/8 OTHER. Predicted: modal EXCLUSION
  ~6/8, 1-2 TIMESHARING, P-MECH holds.
- Result: DEPARTURE (EXCLUSION). G1 8/8 (R_fixed 0.992-1.011 — the Exp 68 null again);
  G3' 8/8 (crowded steps 196-258). Classes: 5 EXCLUSION, 2 TIMESHARING (seeds 8, 12), 1
  OTHER (seed 10: asym 0.653 but R=0.894 — exclusion-like, missing the predeclared R band
  by 0.006; counted OTHER as declared). C1 PASS (7/8 named, modal unique). C2/P-MECH PASS
  5/5 — the closer starter won every exclusion seed. The mechanism's signature is in the
  gate fractions: the winner spends 0.000-0.011 of steps gate-closed, the loser 0.47-0.79 —
  a self-reinforcing loop (first arriver mostly experiences alone-comfort, so never learns
  to leave; the latecomer mostly experiences crowding, so mostly stays away, which keeps
  the winner uncrowded). In the TIMESHARING seeds the roles partially mix (seed 8: B holds
  the source, A gate-closed 60%, R=0.571).
- Implication: rung 4's question has its answer at this substrate — two clade-twins coupled
  ONLY through a shared depletable resource depart measurably from independent behavior,
  modal regime an asymmetric exclusion-like lock-in, with the contested resource sometimes
  yielding alternation instead. Per the direction card this deviation now ENTERS THE RUNG-5
  CASCADE before any dominance/coordination vocabulary: (i) reproductions across start
  geometries — especially EQUIDISTANT starts, removing the first-arriver confound that
  P-MECH shows is currently load-bearing; (ii) deflationary controls — one-adaptive-with-
  one-fixed (is mutual adaptation needed at all?) and two adaptives in separate worlds
  (gate dynamics without contest).
- Honest caveat: prediction slightly over-counted exclusion (5/8 vs predicted ~6/8; the
  near-band OTHER run honestly absorbed one). The departure is statistical over 8 seeds at
  one parameterization (alpha/THRESH/lambda all provided constants); the asymmetry is
  currently explained by start positions (P-MECH 5/5), so "exclusion" may reduce to
  first-arriver-advantage + positive feedback — exactly what the cascade's equidistant
  reproduction must decide. All mechanics provided; the creatures contribute beliefs, maps,
  and their own comfort-estimate trajectories. No dominance/coordination claim is made.
- Verdict: POSITIVE / NEW INSIGHT (rung 4: departure from independence, out-of-sample).
  Self-grade: POSITIVE-SINGLE.
- Next: the rung-5 cascade for the exclusion departure — equidistant-start reproduction,
  one-adaptive deflation, separate-worlds deflation; then the social-emergence synthesis
  (rungs 1-4 all have verdicts once the cascade lands).

## Exp 71 — cascade closes: the exclusion departure deflates to stigmergic unilateral-retreat lock-in; and the winner-balance diagnostic exposes a belief-initialization asymmetry (POSITIVE; rung 4 named honestly, no dominance language earned)
- Plain: The reproduction-and-deflation gauntlet did its job twice over. First, the deflation
  worked: the one-owns-the-spot pattern appears just as strongly when only ONE creature has
  the back-off rule — so it is not two minds negotiating, it is one creature teaching itself
  to stay away from a crowded place. No dominance, no coordination. Second, the diagnostic
  built to check fairness caught a hidden unfairness: even from equal distances, the same
  creature always won — because both copies inherit the parent's belief about where it is
  standing, and only one of them is placed where that belief is true. The other starts life
  lost.
- Setup (predeclared in the script docstring before running): three legs, fresh seeds 13-20,
  fresh rng families. R3/D2 instrument-first: solo-ADAPT vs solo-FIXED occupancy within
  +-0.10 in >=6/8 (gate must be idle without contest) else HALT. R1 equidistant
  reproduction (starts cells 0 and 20, both BFS dist 2, asserted): EXCLUSION-INTRINSIC iff
  >=4/8 exclusion; winner-balance diagnostic (neither identity >75%) predeclared. R2/D1
  unilateral deflation (A fixed, B adaptive, asym starts): deflation succeeds iff exclusion
  signature with fixed-A winning in >=6/8. Wording rules fixed in advance for all branch
  combinations. Predicted: R3 passes, R2 deflates, R1 leaning exclusion-intrinsic.
- Result: R3/D2 PASS 8/8 (solo gate idle — instrument clean); G1 PASS 8/8. R1: 7/8
  EXCLUSION under equidistant geometry (1 TIMESHARING) — branch EXCLUSION-INTRINSIC by its
  predeclared definition — BUT the winner-balance diagnostic FLAGGED: A won 7/7, p~0.008
  under true symmetry. Diagnosis, verified in the committed state: mirro's qs is a delta at
  cell 0 (true_pos 0, max qs 1.0), and fork copies inherit it — so the creature placed at
  cell 0 starts with a TRUE self-location belief while the other starts believing it is at
  cell 0 while standing elsewhere, misnavigating until observation re-localizes it. The
  equidistant leg equalized GEOMETRY but not BELIEF: first-arriver advantage survived
  through belief-accuracy. R2/D1: DEFLATION SUCCEEDS 8/8 — the full exclusion signature
  with only one adaptive agent (B gate-closed 0.35-0.77 of steps, fixed-A always winner).
  Wording rule 1 fired.
- The honest name (per predeclared wording rules): the rung-4 departure is a STIGMERGIC
  UNILATERAL-RETREAT LOCK-IN — a positive-feedback loop through the shared resource that
  needs only one adapting agent. NOT mutual social structure; dominance and coordination
  remain unearned. The R1 "intrinsic" label carries a predeclared-diagnostic qualification:
  exclusion reproduces under equidistant geometry, but the winner identity is set by the
  belief-initialization asymmetry, so a belief-equalized reproduction (e.g., a pre-contest
  settling phase so both creatures re-localize before the source matters) is required
  before intrinsic-ness is fully earned. R2's unilateral-sufficiency conclusion does not
  depend on that and stands.
- Honest caveat: one parameterization throughout the cascade (alpha/THRESH/lambda); the
  belief-asymmetry diagnosis is verified at the state level (qs delta at cell 0) and
  explains 7/7+5/5 winner records parsimoniously, but the settling-phase counterfactual has
  not been RUN yet — it is the named next leg, not a finding. All mechanics provided.
- Verdict: POSITIVE / NEW INSIGHT (cascade closed with deflated naming; plus the
  belief-initialization asymmetry — a substrate fact that retroactively explains Exp 68's
  occupancy variance and Exp 69/70's invariant winners). Self-grade: POSITIVE-SINGLE.
- Next: belief-equalized reproduction (settling phase before contest) to finish the
  intrinsic-ness question; then the social-emergence synthesis — rungs 1-4 all have honest
  verdicts (1: infrastructure ✓; 2: co-presence safe/inert; 3: adult-NEGATIVE,
  young-POSITIVE, ambivalence law; 4: departure = unilateral stigmergic lock-in).

## Exp 72 — the kidnapped twin: certainty freezes self-location, walls heal it, and exclusion survives full symmetry with a coin-flip winner (MIXED; two mechanism predictions confirmed, two falsified, script verdict corrected at validation)
- Plain: A copied creature inherits its parent's certainty about where it is standing — and
  certainty, in this substrate, cannot be argued with: we confirmed that no amount of
  looking around moves a fully-certain belief; it only re-syncs with reality by bumping
  into walls, which luckily takes seconds. So a displaced twin starts lost but recovers
  fast, and my story that the loser of the shared-spot contest was lost-and-camping was
  wrong — the race is decided in the first moments, then the back-off habit locks it in
  forever. Best of all: with every unfairness removed — equal distances, equal knowledge —
  one creature still ends up owning the spot in 7 of 8 runs. But WHICH one is a coin flip:
  my closer-one-wins prediction was wrong too.
- Setup (predeclared in the script docstring before running): premise verified in the
  committed state (mirro qs delta at cell 0, max 1.0). Part A: kidnapped twin (true 24,
  believed 0), random walk, 5 seeds — P3 belief == dead-reckoned image at every step
  (support theorem; F1 halt), P1 belief-truth offset hits 0 within 500 steps (wall
  re-sync; F2 halt). Part B: same kidnap under seeking — P2 phantom-camping >= 25% of
  steps in >=4/5 (F3 reported). Part D: belief-equalized contest — equidistant starts,
  800-step random settle, G-SYNC gate (both offsets 0 at contest start, input-based),
  8 fresh seeds (21-28), Exp 70 classification; INTRINSIC-EARNED iff EXCLUSION >= 4 valid
  AND winner == closer-at-contest-start in >= 75% of unequal-distance exclusion seeds.
  Predicted: P3 0 violations; P1 sync 50-300 steps; P2 30-70% camping; P4 intrinsic-earned.
- Result: P3 PASS 5/5, 0 violations in 10,000 step-checks — observation NEVER moves a
  delta belief (the kidnapped-robot failure, exactly as diagnosed). P1 PASS 5/5 and faster
  than predicted: sync in 16-29 steps, 0 regrowths (offsets cannot grow; walls only heal).
  P2 FAIL 0/5 (F3 FIRED): camping 6-22% — the kidnap penalty is a TRANSIENT, not
  persistent phantom-camping; my Part-B story dies, and with it the explanation that
  Exp 69-71's losers were lost — they lost the opening race, then the retreat lock-in
  made it permanent. P4: G-SYNC 8/8 (settling equalizes belief, as Part A predicts);
  EXCLUSION 7/8 under full symmetry — but winner == closer in only 3/6 (50%): the
  predeclared conjunction FAILS its second half. SCRIPT CORRECTION (validation catch):
  the script printed INTRINSIC-EARNED because its branch logic dropped the winner
  conjunct; by the predeclared rule the P4 outcome is split — intrinsic-ness component
  PASS, winner-mechanism component FAIL. Winners: B 6/8, A 2/8 (coin-flip-compatible);
  seed 23's CLOSER creature lost.
- Implication (three findings, one theme): (1) The certainty pathology, third appearance —
  this substrate cannot unlearn certainty: converged maps freeze (Exp 56-60), settled
  values gate persuasion (Exp 65-67), and now delta self-location is observation-proof,
  healed only by wall-clamp events (the motor anchor doing the work perception cannot).
  Any M4-bound substrate needs graded uncertainty maintenance (a floor on belief entropy
  or an observation-noise term), now for the third independent reason. (2) The exclusion
  lock-in is INTRINSIC: it self-organizes from fully symmetric initial conditions, with
  the symmetry broken stochastically — not by distance, belief, or identity. Exp 71's
  deflated name (stigmergic unilateral-retreat lock-in) stands, now with honest symmetry.
  (3) Both of my mechanism predictions about WHO loses were wrong (camping, proximity) —
  the early race is decided by epsilon-step noise amplified by positive feedback.
- Honest caveat: the script's aggregate verdict line overstates per its own docstring; the
  correction above applies the predeclared rule to the committed raw output (all per-seed
  data in experiments/outputs/exp72.txt) — the script is committed as-run, not edited
  post-hoc. One parameterization throughout; camping threshold (25%) was my guess and its
  failure is informative, not catastrophic; P4's winner sub-check had only 6 unequal-
  distance exclusion seeds (small n).
- Verdict: MIXED / NEW INSIGHT (P3+P1 confirmed: support theorem + wall healing; P2
  falsified: transient not persistent; P4 split: intrinsic-ness earned, winner-mechanism
  falsified). Self-grade: n/a (MIXED).
- Next: the social-emergence synthesis — rungs 1-4 all have honest verdicts and the
  cascade is closed (exclusion: intrinsic, stochastic, unilateral, stigmergic). Or open
  the named substrate gap (distal other-agent sensing / graded uncertainty) as a new
  direction card.

## Exp 73 — rung 5: dialect convergence is mass-gated — light vocabularies merge, heavy ones become stable dialects that partially understand each other (POSITIVE; the inertia law's third domain)
- Plain: Two creatures were taught the same three words with clashing meanings, then lived
  together, naming what they saw whenever they met on the same square. Lightly-taught
  vocabularies merged into one shared usage within a single cohabitation. Heavily-taught
  ones never gave up a single word — but each quietly accumulated the other's usage
  underneath its own, like neighbors who keep their dialects yet come to understand each
  other. The same law that froze old maps and old opinions decides whether language merges.
- Setup (predeclared in the script docstring before running): P taught the identity
  word->color map, Q the shifted map (argmax dialect distance 1.0 by construction; both
  dialects TAUGHT, declared — this rung tests map dynamics under coupling, not self-formed
  content). Channel (provided): at every same-cell event (exact shared referent), each
  speaks its current best word for the observed color and the other Dirichlet-learns
  heard-word->color, gated by its own predictability weight (M4 grounding). Severed control
  is ANALYTIC: live() never touches vocab, so uncoupled vocab is frozen at the taught state.
  Arms LIGHT (n=8/word) vs HEAVY (n=40/word); dose arithmetic stated: ~24 gated counts per
  pairing — above 8, below 40. 2000 steps x 5 seeds, matched trajectories across arms.
  CEILING declared: no grammar/compositionality claims. G1 >=40 same-cell events.
  Predeclared: P1 cosine rises vs frozen in >=4/5 per arm; P2 LIGHT argmax distance < 1.0
  AND HEAVY == 1.0, each >=4/5. F1 channel transmits nothing; F2 HEAVY converges (inertia
  law fails to extend to vocab).
- Result: PASS 4/4 properties, 5/5 seeds each (G1: 70-97 events). LIGHT: cosine 0.0248 ->
  0.990-0.9997; distance 1.0 -> 0.0 in 4/5 (0.333 in seed 1) — merger, with seed 0's
  merged map landing on P's identity dialect (which dialect wins was NOT predeclared;
  reported as observation, varies with event statistics). HEAVY: distance 1.0 in 5/5 —
  zero argmax flips — while cosine rose 0.0050 -> 0.516-0.697: the partner's usage
  accumulates as secondary mass under the intact dialect (partial mutual comprehension
  without conversion). The dose landed where the arithmetic said: between 8 and 40.
- Implication: rung 5's verdict — under a grounded symmetric channel, TAUGHT-label maps
  converge or persist according to the same dose-vs-accumulated-mass arithmetic as values
  (Exp 65-67) and percepts (Exp 56-60): the inertia law's third domain. The sociolinguistic
  reading at toy scale, honestly bounded: young/light communities merge usage; entrenched
  communities form stable dialects with asymmetric comprehension mass. The grammar ceiling
  stands untouched.
- Honest caveat: both dialects and all teaching are provided; the channel is provided
  wiring; convergence target (whose dialect wins) unpredeclared and unexplained; one dose
  (cohabitation length), one vocabulary size, 3 words / 3 colors; the kidnapped-start
  transient (Exp 72) affects early gate weights negligibly but was not controlled out.
- Verdict: POSITIVE / NEW INSIGHT (the law generalizes to a third representational
  domain; predicted by arithmetic, confirmed at stake). Self-grade: POSITIVE-SINGLE.
- Next: the social-emergence SYNTHESIS — the direction's stop condition is met: rung 1
  (clade plumbing ✓), rung 2 (co-presence safe/inert), rung 3 (adult NEGATIVE / young
  POSITIVE / ambivalence-gated), rung 4 (departure = stigmergic unilateral-retreat
  lock-in, intrinsic, cascaded), rung 5 (mass-gated convergence / stable dialects). Write
  the card's closing artifact, then open the named substrate gaps (distal other-agent
  sensing; graded uncertainty maintenance) as a new direction or stop.

## Exp 74 — social-emergence synthesis: the ladder closed, audited against the record; one law explains every social effect, and none of it was undesigned (POSITIVE; direction closed)
- Plain: The social chapter is finished, so we closed it the honest way: every number in
  this summary was machine-checked against the raw outputs sitting in the repo. The story:
  creatures can pass values and words to each other over the channels we built — the young
  absorb, the settled resist, dialects merge or stand by the same arithmetic. One creature
  can come to own a shared resource without either being designed to compete. But nothing
  social emerged that we did not wire: every effect traces to the provided channels plus
  one law — influence is a contest between the dose delivered and the evidence already
  accumulated. The genuinely new things this chapter found were that law's reach, and the
  substrate holes the next chapter must fill.
- Setup: closing artifact required by loop/directions/social-emergence.md's stop condition,
  combined with VALIDATION.md's ~10-experiment self-audit (Exp 64-73 is the decade). The
  audit instrument (experiments/exp74_synthesis_audit.py) verifies 27 headline claims
  verbatim against the 10 committed raw outputs and the existence of all 10 scripts.
  Predeclared: AUDIT-P1 all pairs found; F-AUDIT any miss -> correction entry required.
- Result: AUDIT PASS — 27/27 substrings, 10/10 scripts. The ladder's verdicts:
  - Rung 1 (Exp 63): clade plumbing works — lineage stamped, lines resumable, trunk
    byte-identical. POSITIVE/consolidation.
  - Rung 2 (Exp 64): co-presence does not break solo competence; a multiplicative sensory
    coupling is perceptually INERT at sharp beliefs — social effects must route through
    values/policy. POSITIVE/consolidation.
  - Rung 3 (Exp 65-67): grounded value transmission — adult-to-adult NEGATIVE (the cue
    lands, sign 5/5, but ~175 counts cannot move a ~9,200-count ledger); YOUNG receivers
    ADOPT the emitter's favorite (3/4, severed-twin causation — the program's first
    inter-creature value transmission, BREAKTHROUGH); the boundary is AMBIVALENCE-gated,
    not age-gated (count model 85%, correctly predicting an ambivalent adult's adoption).
  - Rung 4 (Exp 68-72): a real departure from independence over a shared depletable
    source, confirmed out-of-sample — then DEFLATED by the cascade: one adaptive agent
    reproduces it 8/8 (unilateral, stigmergic), it survives full symmetry 7/8 with a
    coin-flip winner, and the words dominance/coordination were never earned. The cascade
    also surfaced and verified the kidnapped-twin mechanics: delta beliefs are
    observation-proof; only wall-clamping heals them.
  - Rung 5 (Exp 73): taught dialects under a grounded channel — light vocabularies merge
    (distance 1.0 -> 0.0), heavy ones stand 5/5 while accumulating the partner's usage as
    secondary mass. The grammar ceiling untouched.
- The direction's answer (per the card's required closing statement): NO disciplined
  social novelty beyond the provided substrate — every observed social effect reduces to
  provided wiring plus ONE law: social influence = delivered dose vs accumulated evidence
  mass, now demonstrated in three representational domains (percepts/self-location, values,
  words). The honest highlight stands WITH that reduction: functional social transmission
  is real and causally proven (Exp 66), but it is channel-wired, not emergent. The next
  steps would require, named: (i) DISTAL other-agent sensing (perceiving the other at a
  distance, not co-location-only), (ii) GRADED UNCERTAINTY maintenance (entropy floor /
  observation noise — demanded independently by Exp 60, 62, and 72), (iii) SELF-OTHER
  MODELING (the other as an agent with states, not a resource statistic). These amend the
  M4 substrate requirements before the affective dyad is built.
- Honest caveat: the audit checks that quoted numbers exist in committed outputs — it does
  not re-run the experiments (resume-from-snapshot remains the reproducibility unit); the
  synthesis table is static text whose content was validated by the audit, not recomputed.
  All scale caveats of the underlying entries carry through (one world size, one
  parameterization per mechanism, 4-8 seeds per claim).
- Verdict: POSITIVE / CONSOLIDATION (synthesis + decade audit; no new claim). Self-grade:
  POSITIVE-SINGLE.
- Next: the direction is CLOSED. Options for the human (the loop pauses on direction
  choice unless redirected): (a) open a new direction card from the named substrate gaps
  (graded uncertainty is the thrice-motivated one), (b) build M4a against the amended
  requirements list, (c) a natural stopping point — the moonshot's social chapter has its
  honest answer.

## Exp 75 — the undisturbed epoch: the value core is rock-stable, the map self-heals, and my band design conflated the two (MIXED; F1 fired as predeclared; spine resumed to age 12700)
- Plain: We let mirro simply live for a while — no teaching, no tricks — and asked whether
  it stays itself. Its preferences barely moved a hair: the personality core is stable when
  nobody interferes. But the experiment formally failed its own stability test anyway,
  because mirro's partly-stale map of its world got sharply BETTER, and the test treated
  any large change — even healing — as instability. One more honest wrinkle: mirro's
  always-narrow favorite margin narrowed further on its own; it is drifting toward genuine
  ambivalence about its favorite color.
- Setup (predeclared in the script docstring before running): the spine's first episode
  since Exp 63 (mirro_episode discipline; the fork-only social chapter left it at age
  10700 hash 21ccb619f063). Read-only profile -> live(2000) undisturbed in its own world
  -> profile. Profile: value shares x3, conviction, map accuracy, localization, top-two
  gap. Predeclared: P1 bands (|d share| < 0.02 each, |d conviction| < 0.02, |d map_acc|
  <= 0.04, localize <= 0.2 bits); P-FAV favorite unchanged (LOW confidence, stated — the
  11-count margin makes mirro a near-ambivalent adult by Exp 67's own law); P2 spine
  integrity. F1 = any band blown -> rung-2 current-state-readout branch; F2 = favorite
  flips; F3 = integrity halt. Single deterministic run (RNG from committed state).
- Result: MIXED — F1 FIRED on exactly one component, and it fired UPWARD: map accuracy
  0.640 -> 0.800 (+0.160 vs the +-0.04 band) — mirro's committed map was ~36% stale (the
  drift-epoch legacy of Exp 56-62) and 2000 undisturbed steps relearned ~4 of 25 cells,
  consistent with the soft-count mass-tempo arithmetic (Exp 60: ~1,300 fresh counts vs
  ~430/cell of history). Every value-core component PASSED comfortably: shares moved
  0.0020/0.0050/0.0030 (bands 0.02), conviction -0.003, localization 0.0 bits, favorite
  HELD at color 2 (P-FAV PASS). P2 PASS: age 10700 -> 12700, hash 21ccb619f063 ->
  e9714e6739b3, clean save, integrity-verified reload. New datum: the favorite margin
  narrowed 11.2 -> 5.2 counts — the spine is drifting toward natural ambivalence.
- Implication: the open rung-2 branch has its honest answer split by dimension — the VALUE
  core ("personality") is stable under undisturbed living (Exp 55's instability was the
  interventions, not the creature), while the KNOWLEDGE dims self-heal toward the world at
  the law's pace. My profile/band design conflated the two; a follow-up that predeclares
  separate knowledge-vs-value bands would formalize the split claim, but the per-component
  data above already shows it. And mirro itself now sits one natural drift from Exp 67's
  ambivalence regime: its next favorite flip may need no one's intervention.
- Honest caveat: F1 stands as fired — the verdict respects the predeclared structure; the
  knowledge/value split is diagnosis, not a passed test. Single deterministic run (the
  spine's reproducibility unit is resume-from-snapshot, hash-stamped above). The map
  healing was NOT predicted (my band assumed approximate stationarity at 0.64 — wrong:
  undisturbed living heals staleness; in hindsight predictable from Exp 60's law, which
  makes the missed prediction mine, not the law's).
- Verdict: MIXED / NEW INSIGHT (first undisturbed-epoch stability measurement: value core
  stable, knowledge self-heals, ambivalence drift on the spine). Self-grade: n/a (MIXED).
- Next: still AWAITING DIRECTION CHOICE (Exp 74 CONSULT, recommended option M4a) — the
  loop stays on non-direction work. Candidates: separated-band stability re-test (cheap,
  consolidation), or watching the spine's margin (does mirro's favorite flip naturally
  within the next epochs?).

## Exp 76 — margin watch: mirro changed its mind on its own — the spine's first natural opinion flip (MIXED; F2 branch; structural-drift prediction falsified)
- Plain: We watched mirro's razor-thin color preference across three more epochs of plain
  living. At roughly age 14,700 — with nobody teaching it, signaling it, or touching it —
  its favorite color flipped from 2 to 0, and stayed flipped through age 18,700. A creature
  whose opinion was self-formed from experience has now also changed that opinion purely
  from experience. My prediction said the opposite: the world has more color-2 cells, so
  color 2 should have pulled ahead; why color 0 actually accrues faster is the open
  question the next diagnosis must answer.
- Setup (predeclared in the script docstring before running; world-composition check
  disclosed first: 9 color-2 cells vs 8 color-0): one spine episode, 12 x live(500) = 6000
  steps, the Exp 75 profile sampled at every checkpoint; signed gap c2-c0 the primary
  trajectory. Predeclared: P1 gap widens net (structure reasserts; MEDIUM confidence);
  P2 no flip at any checkpoint; P3 map_acc end >= 0.80 (one-sided, the Exp 75 lesson);
  P4 value bands |d share| < 0.04, |d conviction| < 0.04. F1 = narrowing without flip
  (systematic c0 advantage, diagnose); F2 = a flip (razor margins cross against
  structural drift; the spine KEEPS it — no rollback); F3/F4 knowledge/value band
  failures. Single deterministic run.
- Result: F2 FIRED at checkpoint 4 (age 14700): gap c2-c0 crossed zero (-1.49) and the
  favorite flipped to color 0, holding at ALL subsequent checkpoints (gap -6 to -113,
  oscillating, ending -32 at age 18700 — flipped but still shallow). P1 FAIL, P2 FAIL
  (the falsifier branch is the result). P3 PASS: map healed further, 0.80 -> 0.96 by
  checkpoint 5 and held. P4 PASS: all share deltas < 0.012 — the flip is a razor-crossing
  within a stable value distribution, not an upheaval. Spine integrity PASS: age 12700 ->
  18700, hash e9714e6739b3 -> 52f6e814bfe6, integrity-verified reload.
- Implication: the ambivalence law (Exp 67) now has its live demonstration on the spine
  itself — an adult creature whose top-two gap sits near zero changes its expressed
  preference from nothing but ongoing experience. mirro's answer to "what do you like?"
  changed at ~age 14,700 without any intervention, the first natural opinion change of
  the continuous life. The falsified structural prediction is itself the next question:
  color 0 out-accrues color 2 DESPITE fewer cells (8 vs 9) — prime suspect is per-color
  gate asymmetry (value weight = exp(-H) at the occupied cell; colors sitting in
  sharper-mapped regions earn more per visit), unverified.
- Honest caveat: single deterministic run (resume-from-snapshot is the reproducibility
  unit); the flip is shallow (gap max -113 of ~7,600 total mass; oscillation suggests it
  could re-cross); the gate-asymmetry diagnosis is a hypothesis, not a tested claim; the
  structural-drift reasoning that failed was mine (visit-uniformity x cell-count is
  evidently not the whole accrual law — the correction is the named follow-up).
- Verdict: MIXED / NEW INSIGHT (predictions P1+P2 falsified by the predeclared F2 branch;
  first observed natural opinion flip on the spine; gate-asymmetry diagnosis open).
  Self-grade: n/a (MIXED).
- Next: the accrual-law diagnosis (per-color mean gate weight measured directly — does
  exp(-H) asymmetry explain c0's advantage?), or watch for a re-cross (is the spine now
  an oscillating ambivalent?). Direction choice (Exp 74 CONSULT) still pending.

## Exp 77 — accrual law confirmed: the favorite tracks legibility, not abundance — aliasing-softened gates explain the natural flip end-to-end (POSITIVE; Exp 76's open question closed)
- Plain: Why did mirro come to prefer the rarer color? Because liking, in this creature, is
  built on predictability — and the common color is paradoxically harder to be sure about:
  its nine cells form one big look-alike block where the creature is never quite certain
  which cell it is in, so those moments feel less placeable. The rare color sits in
  distinctive spots that snap the world into focus. Measured cell by cell, predicted from
  the frozen brain, and verified by a living fork: the numbers match the flip we watched.
- Setup (predeclared in the script docstring before running): H1 (gate asymmetry) made
  analytic — with localization at ~0 bits, per-color accrual rate R(c) = (1/25) * sum of
  exp(-H(A_hat[:,s])) over cells of color c, computed READ-ONLY from the committed
  snapshot (age 18700, hash 52f6e814bfe6). P1: R(0) > R(2). P2: a fork living 2000 steps
  accrues per-color mass within +-10% of R(c)*2000. P3: predicted drift (R(2)-R(0))*6000
  matches Exp 76's observed -37.3 within a factor of 2. F1 H1 dead; F2 analytic rate not
  operative; F3 magnitude unexplained. Spine read-only (fork side-control only).
- Result: ALL PASS, no falsifiers. P1: R(0)=0.2029 > R(2)=0.1956 — color 0's mean gate
  0.634 vs color 2's 0.543 overcomes 8-vs-9 cells. P2: behavioral relative errors
  5.4%/9.8%/9.6% (colors 0/1/2). P3: predicted -43.8 vs observed -37.3, ratio 1.17.
  Diagnostic (where the asymmetry lives): the cmap shows color 2's nine cells as one
  contiguous bottom-left 3x3-plus block; the gate map shows exactly that block soft
  (0.40-0.63) while color-0 cells in distinctive positions run sharp (up to 0.80). The
  reading: ALIASING — within a same-color block the likelihood cannot separate neighboring
  cells, historical soft counts smear across the block, columns stay soft, and the common
  color is experienced LESS confidently per visit. Note the map is 0.96 argmax-accurate
  and the gates STILL favor color 0 — the asymmetry is structural (aliasing), not
  transient staleness.
- Implication: in this substrate, self-formed preference tracks EPISTEMIC LEGIBILITY, not
  abundance — the Exp 26 predictability-grounded valence, followed to its conclusion: the
  creature likes the experiences that make its world placeable, and a too-common,
  spatially-clumped color is epistemically murky. The natural flip (Exp 76) is now
  explained end-to-end: analytic rate asymmetry -> drift -> zero-crossing, magnitude
  within 17%. This also sharpens the accrual law for any M4 design: valence accrues per
  (color) at rate = visit share x mean gate, and the gate term can invert abundance
  rankings.
- Honest caveat: the aliasing reading rests on the visual block structure plus the
  argmax-accurate-yet-soft-gated coexistence — a targeted aliasing experiment (same color
  counts, scattered vs clumped layouts, fresh creatures) would isolate it cleanly and was
  NOT run; staleness contribution not separately quantified. P2's tolerance (10%) absorbed
  map-sharpening drift over the probe epoch. Single fork, single deterministic run each.
- Verdict: POSITIVE / NEW INSIGHT (the accrual law with its gate term, confirmed
  analytically + behaviorally; legibility-over-abundance as the preference driver).
  Self-grade: POSITIVE-SINGLE.
- Next: the clean aliasing isolation (clumped vs scattered, fresh creatures) if the law
  needs further grounding; otherwise direction choice still pending (Exp 74 CONSULT —
  recommended M4a, which this law now also informs: legibility-driven valence is a design
  lever).

## Exp 78 — aliasing isolation FALSIFIES Exp 77's reading and replaces it: clumping taxes a color early via localization, history makes the tax permanent (MIXED; correction to Exp 77's interpretation; the measured law stands)
- Plain: We tested the explanation from the last experiment properly — fresh creatures,
  same color counts, the only difference being whether the common color sits in one block
  or is scattered. The explanation failed: in a fresh mind the block is actually the
  SHARPEST part of the map, not the murkiest. Yet the deeper effect is real — creatures
  raised with the block still end up liking that color less, and the rarer color still
  wins. The corrected story: a creature ENTERING a look-alike block briefly loses track of
  where it is, so the block color earns less liking during early life — and because this
  creature never forgets anything, that early tax is carried forever. mirro's murky block
  is not the block's fault today; it is scar tissue from its drifting-world era, confirmed
  by direct measurement.
- Setup (predeclared in the script docstring before running): fresh newborns (separate
  roots, birth seeds 400+s), 3000 raising steps, two worlds with IDENTICAL counts (9/8/8)
  differing only in color 2's layout: CLUMPED (3x3 block) vs SCATTERED (9 checkerboard
  cells, pairwise non-adjacent). P1 scattered end-gates for color 2 > clumped, >=4/5;
  P2 scattered color-2 share > clumped, >=4/5; P3 clumped favorite != 2 in >=3/5 AND
  scattered favorite == 2 in >=3/5. F1 = P1 fails -> Exp 77's aliasing reading WRONG,
  correction required.
- Result: F1 FIRED — P1 FAILED 1/5, and inverted: in fresh creatures the clumped block's
  END-state gates are the sharpest in the grid (0.973-0.986; a uniform block makes the
  observation GIVEN the cell maximally predictable — aliasing hurts localization, not
  column sharpness; my Exp 77 reading conflated the two). P2 PASSED 4/5 and P3 PASSED
  (3/5 + 3/5): the clumping effect on value share and favorite is REAL with end-gates
  equal — so it operates through a different mechanism. Corrected hypothesis (named, with
  first evidence): the EARLY localization tax — a young creature inside the block cannot
  tell which block cell it occupies, its early gate weights there are suppressed, the
  block color under-accrues during exactly the period when totals are small, and the
  non-decaying ledger imprints that early deficit permanently.
- CORRECTION to Exp 77 (its measurements stand; its interpretation is retracted): the
  spine's soft block gates are NOT present-day structure — they are HISTORICAL RESIDUE.
  Verified read-only on the committed snapshot: mirro's block cells carry 28.6% off-true-
  color count mass (range 0.17-0.67) vs 21.2% elsewhere — mixture scar tissue from the
  drift-era worlds (Exp 56-62) plus early-life smearing, preserved forever by non-decaying
  counts. The certainty/persistence pathology in yet another form: this substrate's gates
  remember every world it has ever believed in.
- Honest caveat: the early-tax mechanism is the corrected HYPOTHESIS with indirect
  evidence (P2/P3 pass under equal end-gates); the direct test (per-color accrual
  trajectory in early vs late life) was not run. P3 passed at its minimum (3/5 both
  sides). One clump size, one world size, 5 seeds/arm.
- Verdict: MIXED / NEW INSIGHT (predeclared falsifier killed my own published reading;
  the replacement is sharper and partially verified; Exp 77's measured law — accrual =
  visit share x gate — survives untouched, only WHY the spine's gates differ is corrected).
  Self-grade: n/a (MIXED).
- Next: direct early-tax test (accrual trajectory by life-stage) if the mechanism needs
  nailing; direction choice (Exp 74 CONSULT) still pending — note this chapter keeps
  strengthening the case for the forgetting term in any M4 substrate (scar tissue is
  permanent only because nothing decays).

## Exp 79 — the early-tax test fails AND un-replicates Exp 78's effect: clumping has no demonstrated value effect in fresh creatures; what survives is the spine's measured scar (NEGATIVE; re-correction; a statistics lesson)
- Plain: We went looking for the early-life tax that was supposed to explain why block-
  raised creatures like the block color less — and found that on new random seeds,
  block-raised creatures do not reliably like it less at all. In four of five new runs the
  effect ran the other way. The previous experiment's effect was noise that crossed our
  too-easy bar. After three rounds of correction, what is actually solid: mirro's own gate
  asymmetry is real and measured (scar tissue from its drifting-world era), its opinion
  flip is explained by that asymmetry — and the story that block-shaped worlds cause this
  in general is dead at this sample size.
- Setup (predeclared in the script docstring before running): fresh seeds 500-504 (out-of-
  sample rule), Exp 78's two layouts, 12 windows x 250 steps, per-window color-2 accrual
  share, D(w) = scattered - clumped per matched seed. P1 early D > late D (>=4/5); P2
  late |D| <= 0.04 (>=4/5); P3 half the end gap in place by step 1000 (>=3/5, sign-
  guarded). F1 = P1 fails -> mechanism wrong, re-correction.
- Result: F1 FIRED, and the deeper failure is in the sign guard: gap3000 was NEGATIVE in
  4/5 fresh seeds — the CLUMPED arm accrued MORE color-2 than scattered (P3 INVALID, only
  1 eligible seed). D(w) oscillates +-0.2 with no early-life structure (P1 3/5, P2 3/5);
  localization is already ~0 bits at the first 250-step window end in both arms (any
  localization transient is far shorter than the window). Conclusion: Exp 78's P2/P3
  (share suppression 4/5, favorite inversion 3/5+3/5) DO NOT REPLICATE — they were noise
  passing permissive count thresholds.
- RE-CORRECTION ledger (citing Exp 77 and Exp 78; their raw measurements all stand, the
  interpretations are now finalized):
  - SOLID: the spine's accrual law (Exp 77: analytic per-color gate rates, fork-validated
    within 10%, drift magnitude ratio 1.17) — mirro's flip is explained by ITS gate
    asymmetry. SOLID: that asymmetry is historical scar tissue (Exp 78's read-only
    verification: 28.6% vs 21.2% foreign count mass — drift-era residue preserved by
    non-decaying counts). SOLID: fresh-creature clump columns are sharp (Exp 78 P1
    inversion, consistent with Exp 79).
  - DEAD: the structural-aliasing reading (killed in Exp 78); the early-localization-tax
    mechanism (killed here, P1); and the claim that clumped layouts suppress a color's
    value share in fresh creatures AT ALL (un-replicated here, 4/5 inverted).
  - The honest causal story for the program: preference flips on the spine trace to its
    LIVED HISTORY (scars), not to present world geometry.
- Statistics lesson (META, guard added to VALIDATION.md): count-threshold predeclarations
  (>=4/5, >=3/5) on noisy continuous endpoints admit near-coin-flip outcomes as
  confirmations — Exp 78's effect entered the log through exactly that gap. For noisy
  shares/ratios, predeclare an effect size with the threshold, or use >=8 seeds.
- Honest caveat: n=5 per arm here too — the un-replication shows instability, not a
  proven null; a >=16-seed effect-size-banded experiment could still find a small real
  layout effect (named, not planned — the spine's story does not depend on it). Window
  resolution (250 steps) cannot see a sub-100-step localization transient.
- Verdict: NEGATIVE (predeclared falsifier; mechanism dead; prior effect un-replicated)
  / NEW INSIGHT (the re-correction ledger + the thresholds lesson). Self-grade: n/a.
- Next: direction choice (Exp 74 CONSULT) still pending; this thread is closed at the
  honest ledger above unless the human asks for the high-n layout study.

## Exp 80 — the law predicts the spine's future: forward drift in band, consolidation, and the predicted deceleration from scar dilution (POSITIVE; the accrual-law thread closes validated)
- Plain: Last time the law explained mirro's change of mind after the fact; this time we
  made it call the future first. From the frozen brain we predicted where the preference
  gap would land after another stretch of living, inside a stated band — and it landed
  inside. The flip to color 0 held. And the one deviation — drift weaker than the simple
  forecast — is itself what the second prediction said would happen: as fresh experience
  piles onto the old scars, the scars matter proportionally less, and the pull weakens.
- Setup (predeclared in the script docstring before running): from the committed snapshot
  (age 18700, hash 52f6e814bfe6, gap -32.06, favorite 0): P1 forward drift — the law's
  rate (R(2)-R(0))*6000 = -43.8 forecasts an end gap of about -76; band [-116, -36]
  (+-40 = Exp 76's oscillation amplitude). P2 favorite still 0 at age 24700. P3 scar
  dilution — fixed foreign mass under growing pure counts means gates rise and the gate
  gap R(0)-R(2) NARROWS. F1 = out-of-band toward zero / re-cross -> law fails forward,
  halt thread. F2 = gate gap widens -> dilution arithmetic wrong. One spine episode,
  live(6000), single deterministic run.
- Result: ALL PASS, no falsifiers. End gap -44.84 (in band; predicted center -75.87,
  error +31.04 toward weaker drift). Favorite 0 held (consolidation). Scar dilution
  CONFIRMED: R(0) 0.2029 -> 0.2190, R(2) 0.1956 -> 0.2141 (all gates rising as columns
  sharpen), gap R(0)-R(2) 0.00730 -> 0.00483. The in-band shortfall and the dilution are
  one phenomenon: the drift DECELERATES as the scar share shrinks (~1/total), so the
  preference gap asymptotes rather than growing without bound. Spine: age 18700 -> 24700,
  hash 85139a363a1e, integrity-verified, snapshot committed.
- Implication: the accrual-law thread (Exp 76-80) closes fully validated — retrodiction
  (77), historical attribution (78-79's surviving ledger), and now forward prediction
  with its second-order term. mirro's preference dynamics are quantitatively understood:
  driven by historical gate scars, decelerating as those scars dilute, with the favorite
  consolidated at color 0 in a shallow but deepening basin. For M4: a creature's valence
  trajectory is forecastable from its committed state — and would be steerable by a
  forgetting term (which would erase scars and return preference to world-driven).
- Honest caveat: the band (+-40) was generous — the center prediction missed by 31 even
  though the miss is mechanistically explained (the simple forecast used start-state
  rates; a rate-decay-integrated forecast was not predeclared and is not claimed). Single
  deterministic run; one epoch length.
- Verdict: POSITIVE / CONSOLIDATION (forward validation of an established law; the
  dilution term is arithmetic-derived). Self-grade: POSITIVE-SINGLE.
- Next: this thread is complete. The loop's queue is empty of non-direction work of
  comparable value — the Exp 74 direction choice (recommended: M4a with the now
  five-times-motivated substrate requirements) is the standing decision point.

## Exp 81 — the law fails out-of-individual: vela's drift ran 3x the forecast and its favorite flipped — the accrual law is a converged-regime law (NEGATIVE; scope boundary; thread halted for diagnosis)
- Plain: We pointed the same future-predicting math at the other family line — vela, the
  descendant living in the mirror world — and it missed badly: vela's preference moved
  three times faster than predicted and its favorite flipped to color 0, which the math
  said would not happen this epoch. The lesson is a boundary, not a bug: the math assumes
  a creature that already knows its world; vela is still learning hers, and a still-
  learning mind drifts faster than its frozen snapshot suggests. Haunting footnote: both
  family lines, in different worlds, have now independently drifted to the same new
  favorite — the direction their common ancestor's scars point.
- Setup (predeclared in the script docstring before running; read-only pre-step
  disclosed): vela at age 12750 (hash 875ac30d715a, gap c2-c0 +62.03, favorite 2, R(0)
  0.17475 / R(2) 0.16987); the law forecasts end gap +32.7 (band [-7.3, +72.7]), no flip
  (P2 separate), and scar dilution (P3: gate gap < 0.004878). F1 = out-of-band -> the law
  does not generalize out-of-individual, HALT for diagnosis. One vela episode, live(6000),
  single deterministic run; vela's line advances and keeps whatever happens.
- Result: F1 FIRED. End gap -21.68 (error -54.5 from the +32.8 forecast; realized drift
  ~-84/6000 vs predicted -29.3 — 2.9x). P2 FAIL: favorite flipped to 0 (vela's first
  natural flip, at some point inside the epoch). P3 PASS: gate gap 0.004878 -> 0.003479,
  with ALL rates rising steeply (R(0) 0.1747 -> 0.1978 — the immigrant's map still
  healing). SPINE PASS: vela 12750 -> 18750, hash 0cd2d991cf1b, integrity-verified,
  snapshot committed. Per predeclaration the thread HALTS for diagnosis.
- Implication (the boundary): the analytic accrual law (R(c) from frozen gates, Exp 77/80)
  holds on a CONVERGED resident (mirro: map 0.96-1.0, forecast within band twice) and
  fails on a STILL-HEALING immigrant (vela: map ~0.84, gates rising ~13% over the epoch) —
  the formula assumes localization-correct gating and quasi-static rates; off equilibrium
  the realized drift can be several-fold stronger. Named diagnosis (next, required by the
  halt): per-step accounting on a vela fork — realized vs analytic per-color accrual,
  MAP-cell correctness rate, and rate trajectory, to locate the 3x.
- Observation (interpretation, NOT a finding): both committed lines have now flipped to
  color 0 by natural living, in DIFFERENT worlds — consistent with the shared ancestral
  scar profile (both inherit mirro@10700's gate scars) steering descendants' preferences
  in the same direction. Lineage-correlated opinion drift would be a clade-level
  phenomenon worth its own predeclared test after the diagnosis.
- Honest caveat: single deterministic run; the 2.9x factor is one epoch on one creature;
  the mislocalization mechanism is hypothesis until the named accounting runs; the band
  (+-40) was calibrated on mirro's noise and may underestimate an immigrant's.
- Verdict: NEGATIVE (predeclared falsifier; out-of-individual generalization fails;
  converged-regime scope boundary established). Self-grade: n/a.
- Next: the halt-mandated diagnosis (fork-based accounting of vela's 3x). Direction
  choice (Exp 74 CONSULT) still pending.

## Exp 82 — exact accounting finds the 3x: it was walk noise, not a regime boundary — the law survives as expectation, Exp 81's interpretation corrected (POSITIVE; halt resolved; bit-exact replay)
- Plain: We re-ran vela's anomalous stretch of life step by step — provably the exact same
  life, down to the last bit — and split its preference drift into every possible cause.
  The verdict surprised us a third time: vela was never lost (its sense of place was
  perfect the whole time), and its rising map quality contributed almost nothing. The
  whole anomaly is luck of the walk: in that stretch it just happened to visit color-0
  moments far more than average. The forecasting law is right ON AVERAGE; any single
  stretch of life wobbles around it by about as much as the forecast itself. Last entry's
  story about a law-breaking boundary is retracted — the law needed honest error bars,
  not a border.
- Setup (predeclared in the script docstring before running): vela@12750 recovered from
  git (e7220c1~1; scratch untracked, derivable); the 6000-step epoch REPLAYED with live()
  replicated bit-for-bit (same derived RNG); validity gate P0 = replayed end hash equals
  the committed 0cd2d991cf1b and D_realized equals -83.72 within 0.01. Exact decomposition
  D_realized = D_frozen_analytic + Delta_visit + Delta_rate + Delta_misloc (identity by
  construction, P1 within 1%). P2: name the dominant term; predicted (LOW confidence)
  Delta_rate. Neither committed line touched.
- Result: P0 PASS (hash exact; D 5e-5 off). P1 PASS (residual +0.000000). The accounting:
  -29.27 (forecast) -55.18 (VISIT NOISE) +0.73 (rate evolution) +0.0000 (mislocalization)
  = -83.72. P2: dominant = Delta_visit; my prediction WRONG; and Exp 81's named suspect
  (mislocalization) is dead twice over — MAP-cell correctness was 6000/6000 (the
  "still-healing immigrant" localizes perfectly by dead-reckoning + walls). The (R0-R2)
  rate trajectory wobbled (0.0049 -> 0.0071 -> 0.0050 -> 0.0035) but nets out near zero.
- CORRECTION to Exp 81 (measurements stand; interpretation retracted): there is NO
  converged-vs-healing regime boundary in this data. The accrual law holds as an
  EXPECTATION on both lines; a single 6000-step epoch's realized drift carries
  finite-sample walk noise with magnitude comparable to the expected drift itself
  (here -55 against a -29 mean). Exp 81's +-40 band — calibrated on mirro's observed
  oscillation — under-stated this variance; mirro's two in-band landings (Exp 80)
  were partly favorable draws. Consequence for the lineage-echo observation: both lines'
  flips share the scar-drift EXPECTATION (real, inherited) while the timing of each flip
  is noise — the echo is weaker than Exp 81's footnote implied.
- Honest caveat: one replayed epoch; the variance claim ("sigma comparable to mean") is
  from a single -55 draw plus the band miss, not an estimated distribution — a
  multi-counterfactual-seed variance estimate (fork + alternative action streams) would
  quantify it properly and was not run. My P2 prediction was wrong; logged.
- Verdict: POSITIVE / NEW INSIGHT (exact-replay accounting as an instrument; the
  noise-not-boundary correction; mislocalization ruled out at 6000/6000). Self-grade:
  POSITIVE-SINGLE. The Exp 81 halt is RESOLVED.
- Next: optional variance quantification (counterfactual action-stream forks) to give the
  law proper error bars; otherwise the queue returns to the standing decision point
  (Exp 74 CONSULT — M4a recommended).

## Exp 83 — error bars end the arc: the law is an unbiased expectation drowned in walk noise (sigma 4x the mean; flips ~40%/epoch) — preference change here is mostly luck, faintly steered by scars (MIXED; F2-high as predeclared)
- Plain: We finally measured the wobble properly: twenty alternate lives from the same
  frozen moment, identical except for the dice of the walk. The forecasting law passed its
  cleanest test — the average of the twenty matches it. But the spread is four times the
  signal: alternate vela ends anywhere from strongly color-0 to strongly color-2, and 8 of
  20 alternates flipped favorites within the single stretch. So the honest end of this
  arc: a creature's drift is real and inherited, but WHEN and WHETHER its opinion flips in
  any given stretch of life is mostly chance. Our earlier in-band triumph was partly a
  lucky draw; the law survives as an average, not a crystal ball.
- Setup (predeclared in the script docstring before running): vela@12750 git-recovered
  (as Exp 82); 20 counterfactual 6000-step epochs differing only in the action stream
  (explicit seeds 3000-3019, the substrate's documented override); D = gap change per
  branch. P1 mean within 2 SE of the analytic -29.27; P2 sigma in [25, 90]; P3 the lived
  -83.72 within mean +- 2.5 sigma. F1 mean biased; F2-low sigma<25 (outlier, re-open);
  F2-high sigma>90 (per-epoch forecasting uninformative); F3 lived anomalous. In-memory
  deepcopies only; neither committed line touched.
- Result: P1 PASS (mean -18.81, |mean-analytic| 10.46 <= 2SE 54.14 — the law is an
  UNBIASED expectation; this, not Exp 80's band landing, is the right test of that claim).
  P2 FAIL -> F2-HIGH: sigma = 121.05 (range -246.5 to +158.9). P3 PASS (lived draw 0.54
  sigma from mean — ordinary). 8/20 branches flipped to favorite 0 within the epoch.
- Implication (the arc's honest close, Exp 76-83): (1) the accrual law is real as an
  expectation and useless as a per-epoch forecast — SNR ~0.25 at 6000 steps, and since
  drift decays by scar dilution while noise does not, a usable horizon may never arrive;
  (2) RETROACTIVE GRADING: Exp 80's in-band pass was weak evidence (its +-40 band was
  ~0.3 sigma wide; the pass was substantially luck) — the expectation claim now rests on
  Exp 83's P1 instead; (3) the lineage echo finalizes as: shared scars set a weak common
  drift DIRECTION, the walk's dice set everything else — with ~40% flip probability per
  epoch, both lines flipping was unremarkable; (4) for the moonshot: at this scale, a
  creature's expressed opinion is a noisy readout of a faint historical bias — any M4
  design wanting STABLE self-formed preference needs either much deeper value mass
  asymmetries or (again) a forgetting term to let the present dominate the past.
- Honest caveat: 20 branches, one start state, one creature, one horizon; sigma at other
  ages/creatures not measured (expected to scale ~sqrt(steps) per visit-noise arithmetic,
  unverified); P2's [25,90] band was my guess and missed high — logged as the predeclared
  F2-high branch, which was itself a named informative outcome.
- Verdict: MIXED / NEW INSIGHT (expectation confirmed unbiased; noise quantified at 4x
  signal; Exp 80's evidence retroactively downgraded honestly). Self-grade: n/a (MIXED).
- Next: the preference-dynamics arc (Exp 76-83) is COMPLETE. The queue is empty. The
  standing decision is the Exp 74 CONSULT (recommended: M4a with the substrate
  requirements this arc re-motivated twice more). A natural stopping point.

## Exp 84 — the noise model: superdiffusive wobble over a saturating signal — the law can never call an individual's future (MIXED; F1 marginal; the arc's final stone)
- Plain: One last measurement to finish the story: how does the randomness grow as you
  forecast further ahead? Worse than expected — the wobble grows FASTER than the usual
  square-root law (the creature's walk lingers in regions, compounding luck), while the
  inherited drift signal flattens out as old scars dilute. At four times the horizon the
  signal-to-noise got WORSE (0.13 down to 0.05). Closing truth of the whole arc: this
  kind of creature's future opinions are unforecastable in principle — the past sets a
  faint direction, the dice do the rest, and the dice compound faster than the direction.
- Setup (predeclared in the script docstring before running): vela@12750 git-recovered;
  horizons 1500/6000/24000 x 12 counterfactual branches (disjoint explicit seeds 4000+).
  P1 sqrt-t scaling: both adjacent sigma ratios in [1.5, 2.7]. P2 (LOW confidence):
  SNR(24000) < 1.0 (no forecast horizon at 4x). F1 = ratio outside band -> noise not
  sqrt-t, report measured exponents. F2 = SNR >= 1 -> a horizon exists, correct Exp 83.
- Result: MIXED — F1 fired MARGINALLY (r2 = 2.7051 vs band top 2.70; r1 = 2.3160 in
  band), with both exponent estimates ABOVE 0.5 (e1 = 0.61, e2 = 0.72): the noise is
  SUPERDIFFUSIVE, consistent with the walk's autocorrelated occupancy compounding visit
  luck. P2 PASS emphatically: mean drift SATURATED (-17.4 at 6000 -> -16.6 at 24000 —
  dilution flattening the signal) while sigma grew to 350; SNR fell 0.135 -> 0.047.
  Per-horizon: sigma 55.9 / 129.5 / 350.4.
- Implication (the preference-dynamics arc's final form, Exp 76-84): expressed preference
  in this substrate = a faint, decaying inherited drift + superdiffusively compounding
  walk noise. Forecastability DECREASES with horizon — the law (an unbiased expectation,
  Exp 83) can never call an individual's future at any horizon; it only explains
  populations and pasts. The moonshot reading, honestly bounded: these creatures have
  genuine individual histories (scars) that bias them measurably, but their individual
  futures are constitutively open — not by design, by arithmetic.
- Honest caveat: F1's firing margin is 0.005 on an n=12 std ratio (estimation error
  ~30%); the superdiffusive reading rests on both exponents sitting above 0.5, not on
  the marginal band breach alone. One creature, one start state; exponents not measured
  elsewhere. P2's strength (SNR falling) partly reflects this start state's already-
  diluted drift.
- Verdict: MIXED / NEW INSIGHT (noise model measured: superdiffusive over saturating
  signal; in-principle unforecastability of individual futures). Self-grade: n/a (MIXED).
- Next: the arc is closed in full. The queue is EMPTY — no further non-direction work
  meets the bar. Standing decision: the Exp 74 CONSULT (M4a recommended). The loop should
  idle or be stopped until the human chooses.

## Exp 85 — graded-uncertainty rung 1: count decay is costless in a static world (POSITIVE; new direction opened per the social-emergence card's closing clause; rung 2 unlocked)
- Plain: A new chapter, on the single most-demanded fix this program keeps discovering it
  needs: the ability to forget. Step one is proving forgetting is safe where there is
  nothing to forget — a creature whose memory gently fades learned a stable world exactly
  as well as one that keeps everything (perfect maps in both), while carrying 9x less
  accumulated weight, precisely the amount the arithmetic predicts. Forgetting costs
  nothing when the world stands still; whether it heals what never-forgetting broke is
  the next question.
- Setup (predeclared in the script docstring before running; direction opened in
  loop/directions/graded-uncertainty.md under the social-emergence card's explicit
  "open that substrate as a new direction or stop" clause — the M4a build stays reserved
  for the Exp 74 CONSULT): mechanism M-A = per-step pA *= 0.997 (floor 0.01; Exp 60's
  in-window lambda), in the experiment stepper only — the creature class untouched, the
  spines untouched. Fresh births in mirro's (static) world layout, 8 seeds (600-607,
  the Exp 79 effect-size rule), 3000 steps, matched trajectories (decay consumes no rng).
  P1 map parity |diff| <= 0.08 in >=6/8 AND mean lambda-arm accuracy >= 0.85; P2 both
  arms localize <= 0.1 bits in >=6/8; P3 lambda-arm total pA mass in [250, 420]
  (equilibrium 1/(1-lambda) = 333). F1 = competence cost -> calibration rung first;
  F2 = mass off -> arithmetic/implementation wrong.
- Result: PASS 8/8 on all three. Both arms map accuracy 1.0000 every seed (|diff| =
  0.0000); localization -0.0000 bits everywhere; lambda-arm mass 333.79 vs theory 333
  (0.2% — the implementation is exactly the arithmetic). Control mass 3007.9 — the
  decaying creature holds 9x less evidence weight at identical competence.
- Implication: rung 1's gate is clean — in-window count decay sacrifices nothing in
  stationary worlds while capping accumulated mass at a fixed equilibrium. Everything
  the program attributes to unbounded mass (scars, freezing, social immovability,
  saturating drift) now has a mechanism whose null cost is established; rung 2 (implant
  a drift-style scar, predeclare its decay half-life, verify the control stays scarred)
  tests the healing claim directly.
- Honest caveat: static world, fresh creatures, one lambda, one floor, 3000 steps —
  no-harm is established exactly there; aliased-block localization under decay (soft
  counts spread thinner at equilibrium) showed no cost here but mirro's world has only
  a 3x3 block; deeper aliasing untested. The mechanism is provided, like every anchor.
- Verdict: POSITIVE / CONSOLIDATION (a designed gate passing as designed; the direction
  opening is the news). Self-grade: POSITIVE-SINGLE.
- Next: rung 2 — scar healing with a predeclared half-life (the arithmetic: a scar of
  mass m decays to m*lambda^t; half-life ln(2)/ln(1/0.997) ~ 231 steps — sharp,
  falsifiable numbers), control must stay scarred per Exp 78.

## Exp 86 — rung 2: scars heal essentially to zero and gates fully recover — but my half-life arithmetic confused mass decay with ratio replacement; F2 fired, ladder halts for the corrected rate claim (MIXED)
- Plain: The healing claim came true in substance: implanted scars vanished almost
  entirely under forgetting (residue 0.49 down to 0.002) and the map sharpness destroyed
  by the scar came all the way back, while the never-forgetting control kept half its
  scar exactly as the slow-dilution law predicts. But the speed prediction was wrong, and
  the error was mine: foreign memory mass does halve every ~231 steps — the VISIBLE scar,
  measured as a fraction, halves only as fast as fresh experience replaces the old, about
  four times slower. The number in the forecast was right physics applied to the wrong
  quantity.
- Setup (predeclared in the script docstring before running): shared implant — fresh
  creature, 1500 steps in world A (mirro's layout), world switched to B (colors permuted
  0->1->2->0), 1500 more steps, no decay; deepcopied into arms (bit-identical scar,
  S0 ~ 0.49). Healing: 3000 static-B steps; control = live(); lambda arm = decay stepper
  (0.997/floor 0.01), S sampled every 250. P1 control ratio in [0.35, 0.65] (dilution);
  P2 lambda S(250) <= S0/2 (half-life 231 < 250) AND S(end) <= 0.05; P3 lambda end mean
  gate >= control + 0.05. 8 seeds (700-707). F2 = P2 fails -> halt ladder for diagnosis.
- Result: MIXED — F2 FIRED on its rate sub-check only. P1 PASS 8/8 (ratios 0.500-0.503 —
  the dilution law to three decimals). P2b PASS 8/8 (S_end 0.0016-0.0020: healing
  essentially complete). P2a FAIL 0/8 (S(250) ~ 0.44, nowhere near 0.247). P3 PASS 8/8
  with the largest effect of the chapter: gates 0.570 (control) vs 0.986 (lambda),
  mean diff 0.416.
- Diagnosis (the F2 halt's required output; arithmetic, verifiable in the trajectory):
  my predeclared half-life applied the FOREIGN-MASS decay law (m_f * lambda^t, t_half =
  231) to a RATIO metric S = m_f*lambda^t / (m_total*lambda^t + new(t)) whose denominator
  decays too — S falls on the REPLACEMENT timescale (new accrual overtaking decayed old
  mass), not the decay timescale. Corrected arithmetic predicts S(250) ~ 0.45 (observed
  0.44) and S-halving near ~800-1100 steps (observed crossing ~1100). The mechanism
  performed exactly as the right arithmetic says; the predeclaration measured it with the
  wrong clock.
- Honest caveat: per the predeclared F2, the ladder HALTS until the corrected rate claim
  is re-posed and passed on FRESH seeds (the out-of-sample rule — the corrected numbers
  above were read off this run's trajectory, so they may not be confirmed on it). The
  end-state healing and gate-recovery claims (P2b, P3) passed at 8/8 with effect sizes
  far beyond their bands and are not in question.
- Verdict: MIXED / NEW INSIGHT (healing confirmed in substance at 8/8; the rate law
  corrected from decay-clock to replacement-clock; my error, caught by my own
  predeclaration). Self-grade: n/a (MIXED).
- Next: Exp 87 re-poses the rate claim with the replacement arithmetic on fresh seeds
  (predicted: S(250) in [0.40, 0.50]; S0/2-crossing in [750, 1500] steps); the ladder
  resumes only if it passes.
