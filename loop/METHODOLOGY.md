# METHODOLOGY — design-time, run-time, and evaluation heuristics for the research loop

> **Provenance (repo annotation, 2026-06-10):** this meta-methodology was supplied by the
> human as a standing reference for designing experiments, writing rung declarations, and
> evaluating results. It is recorded verbatim below; repo-added notes are confined to the
> final "Repo annotations" section and are marked as such.
>
> **Status: ADVISORY reference layer, internalize-don't-recite.** `loop/VALIDATION.md`
> remains the binding honesty contract and `loop/PROTOCOL.md` governs iteration mechanics;
> where this document and those conflict, PROTOCOL/VALIDATION win until the human amends
> them (the two known deltas are listed in the annotations). The intended use: consult
> Part 1 at PROTOCOL steps 1–2, Part 2 at steps 3–4, Part 3 at steps 4–5, Parts 4–6 when
> writing the Implication line and at step 7 (Reflect).

---

I am building a research program in active inference, starting from toy-scale experiments
in a personal repo. The immediate goal is to implement a continuous state substrate
(Problem 2 from our roadmap) and run a series of experiments comparing conjugate Bayesian
inference to both discrete tabular baselines and amortized (gradient-based) baselines.

But my actual goal is larger: I want to develop a research practice that can produce
findings that generalize beyond my repo, that I can recognize as significant when they
appear, and that could eventually contribute to live debates in the field (e.g., AXIOM
vs. Dreamer, the falsifiability of the Free Energy Principle, whether active inference
agents can scale).

I need a meta-methodology document — something I can keep open as a reference while
designing experiments, writing rung declarations, and evaluating results. It should be a
set of heuristics, questions, and diagnostic checks, not a rigid protocol. I want to
internalize these as instincts over time.

## Part 1: Design-time criteria

Before I write code for an experiment, what should I ask?

**Mechanism vs. outcome.** Does this experiment isolate *why* something happens, or just
measure *whether* it happens? An outcome experiment says "continuous converges faster than
discrete." A mechanism experiment says "continuous converges faster because precision
accumulation gives each observation independent weight, while discrete normalization
forces zero-sum competition across bins." Which am I running? If I'm only measuring
outcomes, what would I need to add to trace the mechanism?

**The null hypothesis.** What is the specific thing I expect *not* to happen? "Continuous
won't help" is vague. "The collapse index will be identical for continuous and discrete
models when prior strength is matched" is testable. What exactly would falsify my
expectation?

**The alternative explanations.** If I get the result I expect, what else could explain it
besides my hypothesis? List the confounds before running: Did I give one model a stronger
prior? More parameters? Better-tuned hyperparameters? A different observation sequence? If
I can't rule these out, the experiment is underdesigned.

**The comparison surface.** Am I comparing against the strongest version of the
alternative, or a strawman? If I'm testing conjugate vs. amortized, am I using a
well-tuned amortized baseline with appropriate capacity, or a 2-layer MLP with default
hyperparameters? If the latter, I'm not testing the claim — I'm testing whether I can tune
my thing better than a default thing.

**The degrees of freedom.** How many knobs am I turning? If I'm varying three things at
once (dimensionality, noise, and number of concepts) and get an interesting result, I
won't know which knob produced it. One experiment = one independent variable, unless I'm
explicitly mapping interactions and have the statistical power to do so. A sweep across
one variable with a shape prediction is one experiment. A grid search across three
variables hoping something pops out is not.

**The bridge question.** If this works at toy scale, what's the next-simplest domain where
I'd expect to see the same phenomenon? Can I name that domain now? If I can't imagine what
a bridge experiment would look like, the toy experiment might be too disconnected from
anything scalable.

## Part 2: Run-time discipline

While the experiment is running (or being coded), what practices keep me honest?

**Pre-declare the shape, not the point.** Don't predict "the crossover will be at d=8."
Predict "there will be a crossover where conjugate is faster below some d and amortized is
faster above it." Commit to the existence and direction of the phenomenon, not the exact
parameter value. If you want to predict a range ("the crossover will be between d=4 and
d=16"), that's a stronger prediction and more falsifiable — but only do it if you have a
reason from the math.

**Sweeps are experiments, not fishing expeditions.** A parameter sweep is legitimate if:
(1) the range is declared before seeing data, (2) the prediction is about the shape of the
curve (monotonic, U-shaped, phase transition at some threshold), and (3) the falsifier is
stated ("if the curve is flat, the hypothesis is wrong"). A sweep becomes fishing when you
run it, see three interesting bumps, and then write a story about the bump you like best.

**The two-phase rule for exploration.** It's fine to run exploratory sweeps with no
prediction — that's how you find phenomena. But you must label them exploratory, and any
finding from them is a hypothesis for the next experiment, not a conclusion from this one.
The standard pattern: Phase 1 = exploratory sweep, find something interesting. Phase 2 =
pre-declare the finding as a hypothesis, run on fresh seeds. Phase 1 is private or labeled
exploratory. Phase 2 is the actual experiment.

**Seed discipline.** If you run 20 seeds and 3 show the effect, you don't have an effect —
you have noise. Decide the number of seeds before running, not after seeing variance. For
toy experiments with low computational cost, 20-50 seeds is cheap and prevents fooling
yourself. Report all seeds, not the best ones.

**Log everything, decide what matters later.** The repo's logging infrastructure should
capture every metric you can think of, even ones you don't currently have hypotheses
about. You don't have to analyze all of them. But if something unexpected appears, you'll
have the data. The cost of logging an extra numpy array is zero. The cost of not logging
it and realizing later it was the key variable is restarting the experiment.

**Commit before running.** The rung declaration, the hypothesis, the predicted shape, the
falsifier — all of it goes into a git commit before the run starts. The timestamp is your
honesty proof. If you change your mind mid-run, that's a new rung, not an edit to the old
one.

## Part 3: Result evaluation

The experiment finished. The plots are in front of you. Now what?

**The "so what?" test.** Explain the result to an imaginary colleague who works on
something else (not active inference, not your repo). If they would say "ok, and?" you
have a plot, not a finding. If they would say "huh, that means X should be true in my
domain too," you have something general.

**The confound audit.** Go back to the alternative explanations you listed before running.
Can you now rule them out with the data you have? If not, the result is suggestive, not
established. Run the control experiment.

**The boundary condition check.** Don't just report that the effect exists. Report where
it *stops* existing. "Continuous outperforms discrete" is marketing. "Continuous
outperforms discrete when dimensionality < 12, observation noise < 0.4, and number of
concepts > 4" is science. The second statement tells people when *not* to use your method,
which is more useful than telling them to use it.

**The effect size, not just significance.** At toy scale with many seeds, tiny effects can
be "statistically significant." That doesn't make them meaningful. Ask: would this effect
matter in a real system? A 0.3% improvement in convergence speed is a p-value, not a
finding. A 3× improvement is a finding. A phase transition — where behavior qualitatively
changes at a threshold — is a stronger finding than a smooth 10% improvement, even if the
10% is more statistically robust.

**The surprise audit.** Did the result surprise you? If yes, that's interesting — but it
might also mean you made an error, or your intuition was miscalibrated. Surprising results
need *more* verification, not less. If the result confirmed your expectation exactly,
that's also worth scrutinizing — did it confirm because the phenomenon is robust, or
because your experimental design couldn't have produced any other outcome?

**The "what would have changed my mind?" check.** Before you saw the data, what result
would have made you abandon your hypothesis? If no possible result would have changed your
mind, you didn't have a hypothesis — you had a commitment. Go back and check whether your
pre-declared falsifier actually triggered. If the falsifier triggered and you're still
arguing for your hypothesis, you're doing advocacy, not science.

## Part 4: Generalizability indicators

How do I know if a toy finding will hold beyond my repo?

**Analytic grounding (strongest).** Can you derive the result — or at least its functional
form — from the equations before running? If the math says "collapse should occur when
noise variance exceeds prior variance," and the experiment confirms it, that's not a toy
finding. It's a mathematical necessity that will hold in any system described by those
equations. The more you can derive analytically, the less you're relying on your specific
implementation.

**Functional form stability (strong).** Does the same curve shape (exponential decay,
sigmoidal phase transition, U-shaped tradeoff) appear across different tasks, different
model sizes, different random seeds? If the shape is invariant even as parameters change,
you've probably found a structural property of the inference method, not an artifact of
your setup.

**Parameter-level prediction (strong).** Can you predict how the phenomenon changes when
you vary a specific parameter? "If I double the prior strength, the collapse threshold
should shift by X" — and then it does. This is the strongest form of generalization short
of scaling to new domains. It shows you understand the causal relationship, not just the
correlation.

**Failure mode transfer (moderate).** If a method fails in a specific way at toy scale
(e.g., discrete representations collapse under noise), does it fail in the same way in a
slightly more complex domain (e.g., a simple image-based task)? If the failure mode
transfers, the mechanism probably transfers too.

**Benchmark transfer (weaker but necessary).** Eventually, the finding has to show up in a
domain that the field cares about — Atari, MiniGrid, dSprites, something with pixels or
continuous control. This is the last step, not the first. But you should be able to name
the bridge experiment now, even if you're not running it yet.

**The "why wouldn't this scale?" question.** For any toy finding, ask: what would have to
be true for this to *not* hold at scale? Maybe your effect depends on the small state
space, or the discrete observations, or the lack of function approximation. List the
assumptions your finding relies on. If you can design an experiment that tests one of
those assumptions, you're building the bridge.

## Part 5: Recognizing the profound

What does a genuinely important finding look like at toy scale?

**It constrains what's possible.** A benchmark score tells you what worked. A phase
diagram tells you what *can* work — and what can't. If your experiment maps a boundary
between "regime where method A is appropriate" and "regime where method B is appropriate,"
you've done something more useful than declaring a winner. You've given practitioners a
decision criterion.

**It resolves a disagreement.** If two camps in the literature are arguing about
something, and your toy experiment isolates the disagreement to a single variable, you've
clarified the debate even if you haven't settled it. "The AXIOM vs. Dreamer debate reduces
to whether the environment's dimensionality exceeds the crossover point for conjugate
updates" is a more productive framing than "my benchmark score is higher."

**It predicts something surprising that turns out true.** The strongest possible finding:
you derive something from the math, it's counterintuitive, you pre-register it, and it
confirms. Even at toy scale, this is a genuine discovery. It means the math is telling you
something about reality that your intuition wouldn't have guessed.

**It reveals a hidden assumption.** If your experiment shows that a widely-used method
implicitly assumes something that isn't always true — e.g., "discrete representations
assume all states are equally dissimilar, which fails when states have a metric structure"
— you've found a conceptual contribution, not just an empirical one. These age well.

**It's simple enough to explain in one sentence.** "Continuous representations resist
noise-induced collapse better than discrete ones because precision-weighted updates
provide representational inertia." If you can say it in one sentence and it's both true
and non-obvious, you might have something lasting.

**The "would Friston cite this?" heuristic.** Not literally — but imagine the person whose
work you most respect in the field. Would this finding change how they think about
something? Would it make them say "I hadn't considered that" rather than "yes, that's
consistent with what we'd expect"? If it only confirms existing intuitions, it's
replication (valuable but not profound). If it shifts an intuition, it's a contribution.

## Part 6: Bridging to scale

When and how do I move from toy to something the field cares about?

**The bridge experiment is a scaling test, not a benchmark.** Don't jump from 2D Gaussian
mixtures to full Atari. The bridge is a single intermediate domain where you test whether
the *functional form* of your toy finding holds. dSprites (disentangled 2D shapes),
MiniGrid (symbolic but spatial), or a simple continuous control task (CartPole with pixel
inputs) are bridge domains. They're more complex than your toy setup but simple enough
that you can still run controlled experiments.

**Only bridge findings with analytic grounding or functional form stability.** If your toy
finding is "this specific hyperparameter setting works well," it won't bridge. If your
finding is "the collapse threshold depends on the ratio of noise variance to prior
variance," and you've confirmed this functional form across multiple toy parameter
settings, it's worth bridging. Bridge the mechanism, not the parameter value.

**A failed bridge is a finding.** If the phenomenon doesn't transfer to the bridge domain,
you've learned that your toy finding relied on an assumption that the bridge domain
violates. Identifying that assumption is a contribution. It tells the field: "this
mechanism matters, but only under these conditions." That's useful.

**The bridge doesn't need to beat anything.** The bridge experiment is a test of
generalization, not a competitive benchmark. You're asking: "does the same curve shape
appear?" not "does my method win?" If the curve shape transfers but your toy
implementation is too slow or simple to compete on score, that's fine — you're
demonstrating a principle, not shipping a system.

## Summary card: Questions to ask at each stage

**Designing:**
- Am I isolating a mechanism or measuring an outcome?
- What's the specific null hypothesis?
- What alternative explanations would survive my result?
- Am I comparing against the strongest version of the alternative?
- What's the bridge domain if this works?

**Running:**
- Did I pre-declare the shape, direction, and falsifier?
- Am I labeling exploration as exploration?
- Did I commit the prediction before the run?
- Am I logging everything, not just what I expect to matter?

**Evaluating:**
- Would someone outside my subfield care?
- Can I rule out the confounds I identified?
- Where does the effect stop working?
- Is the effect size meaningful, not just significant?
- Did the falsifier trigger, and am I responding honestly?

**Generalizing:**
- Can I derive this from the math?
- Does the same functional form hold across variations?
- Can I predict how the effect changes with parameters?
- What assumptions would have to be false for this to not scale?

**Recognizing:**
- Does this constrain what's possible, or just report what happened?
- Does it resolve or reframe a debate?
- Is it counterintuitive and confirmed?
- Does it reveal a hidden assumption?
- Can I say it in one sentence?

**Bridging:**
- Is this finding a mechanism or a parameter setting?
- What's the simplest domain where I'd expect the same functional form?
- Am I testing generalization, or chasing a benchmark score?

This is a living document. It should evolve as I run experiments and learn which
heuristics actually help and which are just ceremony. The goal is not to follow every rule
every time — it's to develop enough instinct that I don't need the rules, because honest,
generalizable, mechanism-focused science has become how I naturally think about
experiments.

---

## Repo annotations (added at capture, 2026-06-10 — NOT part of the supplied document)

1. **Two known deltas vs. existing repo law (PROTOCOL/VALIDATION govern until amended).**
   (a) *Seeds:* Part 2 recommends 20–50 seeds at toy scale; the continuous-substrate build
   constraints set a floor of ≥ 8. Read as: 8 is the floor, prefer 20–50 when wall-clock
   is cheap, and the count is always predeclared — never chosen after seeing variance.
   (b) *"Commit before running":* Part 2 wants the declaration committed BEFORE the run
   (timestamp as honesty proof); PROTOCOL step 6's atomicity norm commits script + output
   + entry in ONE turn (the autosync constraint). The current honesty proof is the
   docstring predeclaration written before execution within the committed atomic unit.
   Upgrading to a two-commit declare-then-run pattern is a PROTOCOL change only the human
   ratifies; until then the atomicity norm stands.
2. **The sweep rules here align with the ratified sweep clause** (loop/IDEAS.md,
   2026-06-10): predeclared grid, shape-level falsifier, exploration labeled and then
   registered + confirmed on fresh seeds (Exp 112/128 precedent). Part 2's two-phase rule
   is the same law in different words.
3. **Where this plugs into an iteration:** Part 1 → PROTOCOL steps 1–2 (Choose,
   Predeclare); Part 2 → steps 3–4 (Build, Run); Part 3 → steps 4–5 (Validate, Log —
   especially before self-grading); Parts 4–5 → the entry's Implication line (name the
   generalizability tier claimed: analytic / functional-form / parameter-level / none);
   Part 6 → step 7 (Reflect) and future direction cards. The "so what?" test is already
   institutionalized as the mandatory Plain line.
4. **Bridge domains are out of current scope.** Part 6 names dSprites / MiniGrid /
   pixel-CartPole; none is sanctioned yet — a bridge experiment would be a new direction
   card requiring its own human steer (and likely new dependencies beyond pure numpy).
   Naming the bridge in an Implication line costs nothing and is encouraged; running one
   is a direction-level decision.