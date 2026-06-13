# direction: active-monkey-mechanisms

**Question.** Can small toy experiments separate mechanisms that are easy to conflate:
replay versus intervention, single-agent versus fork/population scaling, expensive planning
versus distilled priors, and surface-feature learning versus grounded transferable abstraction?

**Why it matters.** The project already treats lived, registered experience as load-bearing.
This direction adds conservative harnesses for testing when copied traces, shared archives,
shared transition models, and distilled planning products help or fail. Either answer is useful:
positive results identify reusable mechanisms; negative results mark where passive replay or
surface correlations are insufficient.

**Experiment ladder.**
- **Replay ladder.** Compare `observation_only`, `action_conditioned`, `self_generated`, and
  `third_party_with_actions` on the common trajectory schema. FAILURE: observation-only or
  third-party traces match first-person intervention on transition learning without causal
  contact, which would mean the toy world is too weak to discriminate the mechanism.
- **Population sweep.** Run `N = 1, 2, 4, 8, 16` agents/forks under `isolated`,
  `shared_trajectory_archive`, `shared_transition_model`, and `coordinator`. FAILURE:
  all sharing modes are indistinguishable from isolated runs, or coordination overhead is not
  measurable in the coordinator arm.
- **Recursive distillation.** Run a high-budget planner, select verified successes, distill
  the successful traces into priors/transition estimates, and evaluate lower-budget held-out
  seeds/worlds. FAILURE: distilled low-budget runs do not improve over the low-budget baseline,
  or they regress badly on holdout worlds.
- **Abstraction barrier.** Train on raw interactions where superficial features vary across
  worlds and latent affordance classes determine transfer. FAILURE: a surface-only learner
  transfers as well as the latent learner, meaning the barrier is not strong enough.

**Stop condition.** Close this direction when the toy suite either (a) becomes a stable
regression harness for later Loop B experiments, or (b) fails to discriminate the mechanisms
above after one strengthening pass. Write the synthesis as toy-scale evidence only, naming
negative results without reframing them as wins.

**STATUS.** state: active · latest: suite scaffold · depends-on: active-monkey toy harness · reusable: yes · why: provides reusable configs/schema/metrics for replay, scaling, distillation, and abstraction probes · next-falsifiable: run suite as a numbered Loop B experiment with raw output and append-only verdict
