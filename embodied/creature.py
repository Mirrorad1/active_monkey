"""embodied.creature — an embodied life with energy/reproduce/die (no evolution in Phase 2).

life_step runs one decision bout:
  1. Advance the body via physics (world.advance).
  2. Eat from the shared food field along the swept path.
  3. Pay metabolism (baseline + movement + aging).
  4. Die if energy <= 0.
  5. Reproduce if mature + sufficient energy (child gets a verbatim copy of the genotype).

Returns a 4-tuple (parent, child_or_None, event, intake) where:
  - parent  : updated EmbodiedCreature (alive flag set False on death)
  - child   : new EmbodiedCreature or None
  - event   : "live" | "reproduce" | "die"
  - intake  : raw food consumed this bout (float >= 0); clean density-dependence signal
              for Task 5/6.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ecology.genotype import Genotype


@dataclass
class EmbodiedCreature:
    id: int
    genotype: Genotype
    energy: float
    age: int
    body_state: object
    alive: bool = True


def life_step(
    c: EmbodiedCreature,
    world,
    rng,
    next_id: int,
) -> tuple[EmbodiedCreature, "EmbodiedCreature | None", str, float]:
    """Run one life-cycle bout for creature *c*.

    Parameters
    ----------
    c        : current creature state (not mutated — dataclass immutable by convention)
    world    : EmbodiedWorld (advance, food, spawn_pipeline_state)
    rng      : numpy.random.Generator  (for child placement offset)
    next_id  : int  id to assign to the child if reproduction occurs

    Returns
    -------
    (parent, child_or_None, event, intake)
      parent        updated EmbodiedCreature
      child_or_None new EmbodiedCreature or None
      event         "live" | "reproduce" | "die"
      intake        raw food consumed (float >= 0.0)
    """
    g = c.genotype

    # 1. Advance body via physics.
    new_state, path = world.advance(c.body_state)

    # 2. Eat from the food field along the swept path.
    deficit = max(0.0, g.energy_capacity - c.energy)
    intake = world.food.consume(path, deficit)

    # 3. Pay metabolism.
    cost = g.baseline_metabolic_cost + g.movement_cost + g.aging_cost * c.age
    energy = min(g.energy_capacity, c.energy + intake - cost)
    age = c.age + 1
    parent = replace(c, body_state=new_state, energy=energy, age=age)

    # 4. Die if energy exhausted.
    if energy <= 0.0:
        return replace(parent, alive=False), None, "die", float(intake)

    # 5. Reproduce if mature and energy above threshold.
    child = None
    event = "live"
    if age >= g.maturity_age and energy >= g.reproduction_energy_threshold:
        transfer = energy * g.reproduction_energy_transfer_fraction
        overhead = energy * g.reproduction_cost_fraction
        if energy - transfer - overhead > 0.0:
            # Spawn child near parent with a small random offset.
            px = float(new_state.q[0])
            py = float(new_state.q[1])
            off = rng.normal(0.0, 0.3, size=2)
            child_state = world.spawn_pipeline_state(
                (px + float(off[0]), py + float(off[1])), seed=next_id
            )
            # Genotype copied verbatim — no mutation in Phase 2.
            child = EmbodiedCreature(
                id=next_id,
                genotype=g,
                energy=float(transfer),
                age=0,
                body_state=child_state,
            )
            parent = replace(parent, energy=float(energy - transfer - overhead))
            event = "reproduce"

    return parent, child, event, float(intake)
