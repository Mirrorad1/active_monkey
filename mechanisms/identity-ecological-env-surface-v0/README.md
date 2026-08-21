# identity-ecological-env-surface-v0

**Status: constrained.** The **environmental identity-control surface** — defend your
functional identity by **acting on your own input stream** (retreat through space, away from
the cells that overwrite you) rather than by gating internal value updates. This is the
unoccupied pole opposite the closed N4 internal-gating arc (`boundary_notes/identity-n4-commitment-v0.json`).

On mirro's real aliased 5×5 body, under a **soft (resistible) attack** — a pull toward an
attack color, *not* teleport-and-clamp — a **certified-optimal refuge policy** drives the
attack color down to ~0.20 of its diet (vs ~0.61–0.83 for a captured passive walker; gaps
0.41/0.36/0.63 at pull α=0.5, all 3 colors). The surface is **learnable observation-only** by
a **model-based** learner that maps its world in *peacetime* (free, no-pull exploration) then
plans the refuge (closure ~1.0, coverage 1.0), and it **scales** to ≥400 cells with a
free-exploration budget ~ the random-walk cover time (~16–32× cells; budget-proportional).

**Where it fails (see `boundary_notes/identity-ecological-boundaries-v0.json`):**
- **Strong/inescapable attack** (α ≳ 0.8): a changed world that cannot be fled — revision is
  correctly forced (rung 1).
- **Map learnable only under attack**: the pull biases exploration away from the refuge,
  coverage-capping the plan (L46).
- **Model-free (tabular Q)**: *not* repaired by exploring starts — with full state coverage
  (covR=1.00) and 4× budget the trap color stays unlearned (closure −0.280 → −0.027), while
  the model-based learner reaches optimal (~1.0). The wall is **TD-convergence** under the
  discounted (γ=0.999) pull kernel + the absorbing HOLD-on-attack trap, **not** coverage
  (Exp 274, L48).
- **Head-to-head vs internal freeze-gating**: **ILL-POSED** — the freeze surface needs hard
  captivity, the movement surface needs a soft attack; mutually exclusive regimes (rung 2
  CAN'T-POSE, L45).

**Tiered verdict (the whole arc):** POSABLE (rung1) → CAN'T-POSE kill-test (rung2) → LEARNABLE
model-based (rung1c) → SCALES (rung1c-scale) → MODEL-FREE CONVERGENCE WALL (rung1c-MF).

**Honesty:** "identity" = **value-vector ordering on a 3-color simplex** — functional
policy-continuity, not selfhood or biography. The avoid *form*, the N4 mismatch trigger
(`mechanisms/identity-n4-monitor-v0`), and the attack target are **provided**; the value
content, every mismatch, and every move are **self-formed**. The 25-cell base substrate makes
free coverage trivial — scaling was tested to 400 cells with a *fired* artifact gate (L47).
This is the verdict at **this** body/richness, not a universal impossibility.

Source: Exp 270–274. Direction card: `loop/directions/identity-ecological.md`.
Reusable runners: `experiments/exp270_ecological_affordance_gate.py`,
`experiments/exp272_learnable_actuator.py`, `experiments/exp273_learnability_scaling.py`.
