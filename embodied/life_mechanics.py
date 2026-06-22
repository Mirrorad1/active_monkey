"""embodied.life_mechanics — pure per-creature economics (energy/reproduce/die).

Identical formulas to embodied.creature.life_step (Phase 2), extracted so the
batched loop computes the SAME economics without depending on the full creature
stack.  Phase-2 files (creature.py, population.py) are NOT refactored to use
this module — keeping their hash byte-identical.

Formula source: embodied/creature.py, life_step(), lines 67-98:

    cost   = g.baseline_metabolic_cost + g.movement_cost + g.aging_cost * c.age
    energy = min(g.energy_capacity, c.energy + intake - cost)
    age    = c.age + 1                          # incremented AFTER energy calc

    # die
    if energy <= 0.0: ...

    # reproduce gate uses POST-increment `age`
    if age >= g.maturity_age and energy >= g.reproduction_energy_threshold:
        transfer = energy * g.reproduction_energy_transfer_fraction
        overhead = energy * g.reproduction_cost_fraction
        if energy - transfer - overhead > 0.0:
            parent.energy = energy - transfer - overhead
"""
from __future__ import annotations

from ecology.genotype import Genotype


def step_economics(g: Genotype, energy: float, age: int, intake: float) -> dict:
    """Run one bout of per-creature economics.

    Parameters
    ----------
    g       : Genotype — trait configuration.
    energy  : current energy before this step.
    age     : current age (PRE-increment; cost uses this value).
    intake  : food consumed this bout.

    Returns
    -------
    dict with keys:
      energy     : new energy (clamped to capacity; parent POST-reproduction if reproducing)
      age        : age + 1
      die        : True iff new_energy <= 0
      reproduce  : True iff reproduction occurred this step
      transfer   : energy transferred to child (0.0 if no reproduction)
      overhead   : parent overhead cost of reproduction (0.0 if no reproduction)
    """
    # Phase-2 formula (creature.py line 67): cost uses PRE-increment age.
    cost = g.baseline_metabolic_cost + g.movement_cost + g.aging_cost * age
    new_energy = min(g.energy_capacity, energy + intake - cost)

    out = {
        "energy": new_energy,
        "age": age + 1,
        "die": False,
        "reproduce": False,
        "transfer": 0.0,
        "overhead": 0.0,
    }

    # Phase-2 formula (creature.py line 73): die if exhausted.
    if new_energy <= 0.0:
        out["die"] = True
        return out

    # Phase-2 formula (creature.py line 79): reproduce gate uses POST-increment age.
    post_age = age + 1
    if post_age >= g.maturity_age and new_energy >= g.reproduction_energy_threshold:
        transfer = new_energy * g.reproduction_energy_transfer_fraction
        overhead = new_energy * g.reproduction_cost_fraction
        # Phase-2 formula (creature.py line 82): parent must stay positive.
        if new_energy - transfer - overhead > 0.0:
            out["reproduce"] = True
            out["transfer"] = transfer
            out["overhead"] = overhead
            out["energy"] = new_energy - transfer - overhead

    return out
