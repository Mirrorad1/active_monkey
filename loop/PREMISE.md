# PREMISE — the world model of this research program

GOAL: an active-inference agent you can talk to that formed its own opinions from lived
experience — never pretrained on having opinions or on what to think.

FEP-native framing (locked in):
- Free energy is the reward. Low surprise = "understanding"; the agent minimizes free energy.
- Hidden states = meaning. Valence = −free energy (a thing "feels good" when its consequences
  are predictable). Functional claims, not sentience claims.

THE ONE DURABLE FINDING (Exp 1–40): unsupervised emergence from a disembodied symbol stream
COLLAPSES (symmetric saddle / posterior collapse / non-identifiability / mean-field severs
cross-factor inference). What breaks the symmetry is the RECIPE:

> embodiment + grounding + continuous REGISTERED experience (belief never reset) +
> ONE innate anchor (sensory map A or motor model B — learning both from noise collapses,
> Exp 31) + taught labels for the few-shot word↔concept mapping.

On that recipe the full chain works: perceive (place fields self-organize) → learn facts →
want (grounded valence) → plan + act (value-iteration nav) → form own values (same
architecture + different history ⇒ different opinion, Exp 26) → act on them → answer in
words what it thinks (content self-formed, labels taught, Exp 28/34/35).

HONEST CEILINGS (documented research frontiers, NOT toy-crackable — don't keep banging):
- emergent compositional grammar / language-from-scratch
- fully tabula-rasa structure (no innate anchor)
Both written up in `open_problem.html`.

STATE: realistic moonshot reached at toy scale. Exp 36–40 were consolidation (each new
experiment confirms more than it discovers). Status of any new work must be judged against
that baseline: does it probe an EDGE of the recipe, or merely re-confirm it?

Verified pymdp patterns to reuse: Exp 21 (place learning), 26 (opinion formation),
30 (value-propagation planning), 34/35 (language bridge / converse). Capstone artifact:
`converse_demo.py`. Designed-but-unbuilt next rung: the M4 affective-dyad spec
(see RESUME.md §7).
