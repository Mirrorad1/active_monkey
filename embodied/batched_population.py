"""embodied.batched_population — slot-based population loop over a vmap'd body buffer.

Batched analog of Phase-2's embodied.population.run.  Uses BatchedEmbodiedWorld
for GPU-parallel physics (advance_batch), then a numpy life-loop consuming from
the shared FoodField in deterministic sorted-id order.

Key design points
-----------------
- **Deterministic order**: alive slots iterated sorted by creature-id; food
  consumed from the shared field in that order (first creature gets more).
- **Exact determinism**: events_hash (per-alive-slot "id:event" per step) is
  byte-identical across two runs with the same cfg.
- **Loud cap**: a birth that would exceed MAX_POP is dropped, capped += 1, and
  surfaced in BatchedPopResult.capped — never silently lost.
- **Raw intake**: per_capita_intake uses raw intake from consume(), not net energy.
"""
import hashlib
from dataclasses import dataclass, field

import numpy as np

from ecology.genotype import founder
from embodied.batched_world import BatchedEmbodiedWorld
from embodied.foodfield import FoodField, FoodFieldConfig
from embodied.life_mechanics import step_economics
from embodied.policy_runner import DEFAULT_CKPT, PolicyRunner
from embodied.population import PopResult


@dataclass
class BatchedPopResult(PopResult):
    """PopResult extended with a count of births dropped due to max_pop cap."""
    capped: int = 0


@dataclass
class BatchedPopConfig:
    n_founders: int = 30
    horizon: int = 200
    bout_steps: int = 8
    max_pop: int = 256
    seed: int = 0
    field: FoodFieldConfig = field(default_factory=FoodFieldConfig)


def run(cfg: BatchedPopConfig, world: BatchedEmbodiedWorld | None = None) -> BatchedPopResult:
    """Spawn cfg.n_founders bodies, run for cfg.horizon steps, return BatchedPopResult.

    Parameters
    ----------
    cfg : BatchedPopConfig
    world : BatchedEmbodiedWorld, optional
        A prebuilt world to REUSE across many runs (a sweep). The world holds no
        per-run state — its env, deterministic policy, constant offspring pose,
        and jit-compiled advance are shared safely — so reusing one world avoids
        recompiling `advance_batch` (and reloading the checkpoint) on every call.
        Its max_pop / bout_steps must match cfg. When None, a fresh world is built
        (recompiles once); the per-run buffer is always created fresh via
        world.init_buffer, so results are byte-identical either way.

    Returns
    -------
    BatchedPopResult with population time-series, per-capita intake, birth/death
    counts, deterministic events hash (sha256[:16]), final alive count, and
    the number of births dropped due to max_pop cap.
    """
    rng = np.random.default_rng(cfg.seed)
    ff = FoodField(cfg.field, seed=cfg.seed)
    if world is None:
        world = BatchedEmbodiedWorld(ff, PolicyRunner(DEFAULT_CKPT), cfg.max_pop, cfg.bout_steps)
    elif world.max_pop != cfg.max_pop or world.bout_steps != cfg.bout_steps:
        raise ValueError(
            f"reused world (max_pop={world.max_pop}, bout_steps={world.bout_steps}) "
            f"does not match cfg (max_pop={cfg.max_pop}, bout_steps={cfg.bout_steps})"
        )

    g0 = founder()

    # Spread founders on a grid across the arena extent (same as Phase-2).
    side = int(np.ceil(np.sqrt(cfg.n_founders)))
    xs = np.linspace(-cfg.field.extent * 0.6, cfg.field.extent * 0.6, side)
    founder_xys = [
        (float(xs[k % side]), float(xs[k // side]))
        for k in range(cfg.n_founders)
    ]

    state, alive = world.init_buffer(cfg.n_founders, cfg.seed, founder_xys)

    # Slot-aligned numpy metadata arrays (indexed [0..max_pop)).
    geno = [g0] * cfg.max_pop
    energy = np.zeros(cfg.max_pop, dtype=np.float64)
    energy[:cfg.n_founders] = 0.5 * g0.energy_capacity
    age = np.zeros(cfg.max_pop, dtype=int)
    # creature-id per slot; -1 = dead/unoccupied
    cid = np.full(cfg.max_pop, -1, dtype=int)
    cid[:cfg.n_founders] = np.arange(cfg.n_founders)
    next_id = cfg.n_founders

    h = hashlib.sha256()
    n_series: list[int] = []
    pci: list[float] = []
    births = 0
    deaths = 0
    capped = 0

    for _ in range(cfg.horizon):
        # Alive slot indices, sorted by creature-id for deterministic order.
        idx = np.where(alive)[0]
        idx = idx[np.argsort(cid[idx])]

        # Compute navigation targets from current positions (numpy, before GPU advance).
        q = np.asarray(state.q)  # [max_pop, q_dim]
        targets = np.zeros((cfg.max_pop, 2), dtype=np.float32)
        for i in idx:
            tx, ty = ff.nearest_food_xy(float(q[i, 0]), float(q[i, 1]))
            targets[i, 0] = tx
            targets[i, 1] = ty

        # GPU-batched advance: all max_pop slots advance simultaneously.
        state, paths = world.advance_batch(state, targets)
        paths = np.asarray(paths)  # [max_pop, bout_steps, 2]

        # Numpy life-loop: consume from shared field in deterministic sorted-id order.
        step_intake = 0.0
        pending_births: list[tuple[int, float]] = []

        for i in idx:
            g = geno[i]
            # Deficit = room to fill up to energy_capacity.
            deficit = max(0.0, g.energy_capacity - energy[i])
            intake = ff.consume(
                [tuple(p) for p in paths[i]],
                deficit=deficit,
            )
            step_intake += intake

            r = step_economics(g, energy[i], int(age[i]), intake)
            energy[i] = r["energy"]
            age[i] = r["age"]

            # Encode event into hash — same format as Phase-2.
            if r["die"]:
                ev = "die"
            elif r["reproduce"]:
                ev = "reproduce"
            else:
                ev = "live"
            h.update(f"{cid[i]}:{ev};".encode())

            if r["die"]:
                alive[i] = False
                deaths += 1
                continue

            if r["reproduce"]:
                pending_births.append((i, r["transfer"]))

        # Slot-fill for births (after all creatures have consumed this step).
        # Parents are alive and never a birth target (births fill free slots),
        # so their positions are stable across the fill — read state.q ONCE here
        # instead of once per birth (was a host sync per birth).
        q_now = np.asarray(state.q)
        birth_slots: list[int] = []
        birth_xys: list[tuple[float, float]] = []
        for parent_i, transfer in pending_births:
            free = np.where(~alive)[0]
            if free.size == 0:
                # No free slot — birth is dropped, loud count surfaced.
                capped += 1
                continue
            j = int(free[0])
            # Place offspring near parent with seeded jitter (same rng draw order
            # as before — one size=2 draw per birth, in pending order).
            px = float(q_now[parent_i, 0])
            py = float(q_now[parent_i, 1])
            off = rng.normal(0.0, 0.3, size=2)
            bx = float(px + off[0])
            by = float(py + off[1])
            geno[j] = geno[parent_i]
            energy[j] = transfer
            age[j] = 0
            cid[j] = next_id
            alive[j] = True
            birth_slots.append(j)
            birth_xys.append((bx, by))
            births += 1
            next_id += 1

        # One batched GPU scatter for ALL of this step's births (was one eager
        # env.reset + full-buffer copy per birth — the host-bound bottleneck).
        if birth_slots:
            state = world.spawn_into_slots(state, birth_slots, birth_xys)

        ff.step_regen()

        n = int(alive.sum())
        n_series.append(n)
        pci.append(step_intake / max(1, n))

        if n == 0:
            # Population extinct — pad to full horizon with zeros.
            remaining = cfg.horizon - len(n_series)
            n_series += [0] * remaining
            pci += [0.0] * remaining
            break

    return BatchedPopResult(
        n_series=n_series,
        per_capita_intake=pci,
        births=births,
        deaths=deaths,
        events_hash=h.hexdigest()[:16],
        final_alive=int(alive.sum()),
        capped=capped,
    )
