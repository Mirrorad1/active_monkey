"""embodied.population — the numpy life loop over embodied bodies on a shared food field.

Spawn N founders, then for each step:
  - iterate alive creatures (sorted by id for determinism)
  - run life_step on each (advance physics -> eat -> metabolism -> reproduce/die)
  - accumulate raw intake, collect births/deaths
  - regenerate the food field
  - record N(t) and per-capita intake
  - hash the event stream for determinism verification
"""
import hashlib
from dataclasses import dataclass, field

import numpy as np

from ecology.genotype import founder
from embodied.creature import EmbodiedCreature, life_step
from embodied.foodfield import FoodField, FoodFieldConfig
from embodied.policy_runner import DEFAULT_CKPT, PolicyRunner
from embodied.world import EmbodiedWorld


@dataclass
class PopConfig:
    n_founders: int = 20
    horizon: int = 200
    bout_steps: int = 8
    seed: int = 0
    field: FoodFieldConfig = field(default_factory=FoodFieldConfig)


@dataclass
class PopResult:
    n_series: list
    per_capita_intake: list
    births: int
    deaths: int
    events_hash: str
    final_alive: int


def run(cfg: PopConfig) -> PopResult:
    """Spawn cfg.n_founders bodies on a shared food field and run for cfg.horizon steps.

    Returns PopResult with population time-series, per-capita intake, birth/death counts,
    a deterministic events hash (sha256[:16]), and final alive count.
    """
    rng = np.random.default_rng(cfg.seed)
    ff = FoodField(cfg.field, seed=cfg.seed)
    world = EmbodiedWorld(ff, PolicyRunner(DEFAULT_CKPT), bout_steps=cfg.bout_steps)

    g0 = founder()

    # Spread founders on a grid across the arena extent.
    side = int(np.ceil(np.sqrt(cfg.n_founders)))
    xs = np.linspace(-cfg.field.extent * 0.6, cfg.field.extent * 0.6, side)
    alive: list[EmbodiedCreature] = []
    next_id = 0
    for k in range(cfg.n_founders):
        x = xs[k % side]
        y = xs[k // side]
        st = world.spawn_pipeline_state((float(x), float(y)), seed=next_id)
        alive.append(
            EmbodiedCreature(
                id=next_id,
                genotype=g0,
                energy=0.5 * g0.energy_capacity,
                age=0,
                body_state=st,
            )
        )
        next_id += 1

    h = hashlib.sha256()
    n_series: list[int] = []
    pci: list[float] = []
    births = 0
    deaths = 0

    for _ in range(cfg.horizon):
        # Sort by id for deterministic iteration order.
        alive.sort(key=lambda c: c.id)

        survivors: list[EmbodiedCreature] = []
        newborns: list[EmbodiedCreature] = []
        step_intake = 0.0

        for c in alive:
            upd, child, ev, intake = life_step(c, world, rng, next_id)
            # Raw intake — the clean density-dependence signal.
            step_intake += intake
            h.update(f"{c.id}:{ev};".encode())

            if ev == "die":
                deaths += 1
                continue

            survivors.append(upd)
            if child is not None:
                newborns.append(child)
                births += 1
                next_id += 1

        alive = survivors + newborns
        world.food.step_regen()

        n = len(alive)
        n_series.append(n)
        pci.append(step_intake / max(1, n))

        if n == 0:
            # Population extinct — pad to full horizon with zeros.
            remaining = cfg.horizon - len(n_series)
            n_series += [0] * remaining
            pci += [0.0] * remaining
            break

    return PopResult(
        n_series=n_series,
        per_capita_intake=pci,
        births=births,
        deaths=deaths,
        events_hash=h.hexdigest()[:16],
        final_alive=len(alive),
    )
