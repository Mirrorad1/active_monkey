"""ecology/engine.py — EcologyConfig, Ecology (step/run loop).

NO-HIDDEN-EVALUATOR INVARIANT:
  There is NO code path in this module where survival or reproduction reads any
  global ranking, sorted-fitness, top-K, or cross-creature comparison.
  Survival/reproduction decisions read ONLY the individual creature's own
  genotype/phenotype and its local cell's resource.
  Population-level operations are exactly: append (new creature), mark-dead
  (set alive=False on one creature), and iteration (ascending id order).
  This invariant is tested in test_no_global_fitness_selection.

RNG discipline:
  All randomness flows through numpy.random.default_rng with deterministically
  derived sub-seeds.  No set iteration or wall-clock enters the event stream.
  Creatures are processed in ascending creature_id order each step.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ecology.genotype import (
    Genotype, mutate, is_valid, complexity as genotype_complexity,
    thermosense_active, thermosense_upkeep,
)
from ecology.world import GridWorld
from ecology.creature import Creature, Phenotype


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class EcologyConfig:
    rows: int
    cols: int
    horizon: int
    initial_population: int
    founder: Genotype
    mutation_rate: float
    capacity: float
    regen_rate: float
    initial_resource: float          # fraction of capacity (0..1)
    max_population: int              # runaway guard threshold (safety, NOT a culler)
    min_survival_energy: float       # parent must stay above this after paying for child
    name: str
    log_moves: bool = False          # per-step move events are bulk, reproducible-from-seed
                                     # telemetry — OFF by default so committed event logs stay
                                     # small (birth/death/reproduction/resource_tick always log)

    # ------------------------------------------------------------------
    # Senescence model (Exp 195) — OFF by default; OFF is byte-identical to Exp 194
    #
    # Faithful tuning (disclosed; tuning documented in exp195 script):
    #   onset0=155, onset_frailty=0.65 -> onset range:
    #     low-c (0.26): onset~129; high-c (0.57): onset~98 (31-step spread).
    #   base=0.002 (moderate) so damage accumulates over MANY steps past onset --
    #     post-onset survival: well-fed 45-56 steps, starving 13-21 steps.
    #     Death age is onset + a multi-step, energy-modulated, super-linear integral.
    #     (The old base=10.0 made creatures die within 1-2 steps -- degenerate.)
    #   exp=1.5 (>1) is genuinely operative: shapes the accumulation curve over
    #     tens of steps rather than being bypassed by instant death.
    #   maintenance=1.0 and energy-dependent: a well-fed creature (energy~capacity)
    #     offsets early degradation and RESISTS aging; a starving creature of the
    #     same age+complexity senesces ~30-40 steps sooner.
    #   rate_f=2.0 adds a complexity multiplier on degradation.
    # ------------------------------------------------------------------
    enable_senescence: bool = False
    # Base degradation rate -- moderate so damage integrates over many steps past onset
    senescence_base: float = 0.002
    # Exponent >1 so degradation accelerates super-linearly with age past onset
    senescence_exp: float = 1.5
    # Onset age (at complexity=0); complex creatures age earlier
    senescence_onset0: float = 155.0
    # Fraction by which complexity shifts onset earlier (k')
    senescence_onset_frailty: float = 0.65
    # Rate multiplier from complexity for degradation accumulation (k)
    senescence_rate_frailty: float = 2.0
    # Well-fed creatures resist damage (energy-dependent self-maintenance)
    senescence_self_maintenance: float = 1.5
    # Damage threshold at which creature dies of senescence
    senescence_damage_death: float = 1.0

    # ------------------------------------------------------------------
    # Exp 197: complexity-linked maintenance cost — OFF by default;
    # OFF (scale=0.0) is byte-identical to Exp 194/196.
    #
    # Upkeep per tick = baseline_metabolic_cost + aging_cost*age
    #                   + complexity_cost_scale * genotype_complexity(g)
    # The last term is gated (L16 guard): when scale==0.0 the code path is
    # never reached, so no floating-point drift vs. the Exp 194 baseline.
    # Death still flows through the existing energy<=0 -> "starvation" path;
    # there is NO direct complexity-death rule.
    # ------------------------------------------------------------------
    complexity_cost_scale: float = 0.0   # 0 = off (byte-identical to Exp 194/196)

    # ------------------------------------------------------------------
    # Exp 197: temperature field + evolvable thermosense organ — OFF by default.
    # All defaults preserve Exp 194/195/196 byte-identical behaviour.
    #
    # enable_temperature: builds the static thermal gradient in GridWorld.
    # enable_thermosense: gates the policy thermal branch + mutate flag.
    # temperature_stress_scale=0.0 ⇒ even with enable_temperature the field
    #   is built but imposes 0 energy drain (safe for control arm piloting).
    # thermosense_upkeep_floor=0.0 ⇒ upkeep = inefficiency * intensity only
    #   (floor > 0 adds a fixed cost on top, enforcing the "never free" property
    #    even at minimum evolved inefficiency).
    # ------------------------------------------------------------------
    enable_temperature: bool = False
    enable_thermosense: bool = False
    temperature_comfort: float = 0.5
    temperature_stress_scale: float = 0.0
    thermosense_upkeep_floor: float = 0.0
    thermosense_active_threshold: float = 0.05
    thermosense_noise_base: float = 0.5
    thermal_avoidance_weight: float = 1.0

    # ------------------------------------------------------------------
    # Exp 197 extensions: thermal tolerance cost + dynamic comfort zone.
    # Both OFF by default (scale=0, amplitude=0) → byte-identical to the
    # static thermosense implementation above.
    #
    # tolerance_cost_scale: energy drain per tick = scale * temperature_tolerance.
    #   A wider tolerable band (more robustness) costs proportionally more.
    #   Gated on scale != 0.0 (L16 guard) so the OFF path is byte-identical.
    #
    # comfort_amplitude / comfort_period: the comfort center drifts as
    #   comfort(t) = temperature_comfort + comfort_amplitude * sin(2π t / period).
    #   amplitude=0 keeps current_comfort == temperature_comfort for all t
    #   → byte-identical to the static comfort implementation.
    # ------------------------------------------------------------------
    tolerance_cost_scale: float = 0.0    # 0 = off (byte-identical)
    comfort_amplitude: float = 0.0       # 0 = static comfort zone (byte-identical)
    comfort_period: float = 1000.0       # period of the sinusoidal drift in steps

    # ------------------------------------------------------------------
    # Exp 200: foraging-sense — food concentrated in a drifting thermal band.
    # ALL defaults preserve Exp 194–199 byte-identical behaviour.
    #
    # enable_food_coupling: when True, regen is concentrated where the cell
    #   temperature is near current_food_optimal (the drifting food band).
    #   When False, regen is EXACTLY as before (byte-identical).
    #
    # food_optimal_base / amplitude / period: the food-optimal temperature drifts
    #   food_opt(t) = food_optimal_base + food_optimal_amplitude*sin(2π t / period).
    #   amplitude=0 → constant band (food_optimal_base); byte-identical to static.
    #
    # food_band_width: half-width of the in-band region around food_opt.
    #   Cells within this distance get regen_rate * food_concentration;
    #   out-of-band cells get a conserved-regen out_factor (≥ 0).
    #
    # food_concentration: boost multiplier for in-band cells (1.0 = uniform).
    #   Higher values concentrate food more tightly in the band.
    #
    # thermosense_forage_mode: when True AND thermosense is active, the policy
    #   steers TOWARD current_food_optimal (where food is) rather than away from
    #   thermal stress.  When False, the existing avoid-mode logic runs unchanged.
    # ------------------------------------------------------------------
    enable_food_coupling: bool = False
    food_optimal_base: float = 0.5
    food_optimal_amplitude: float = 0.0
    food_optimal_period: float = 1500.0
    food_band_width: float = 0.15
    food_concentration: float = 1.0
    thermosense_forage_mode: bool = False

    # ------------------------------------------------------------------
    # Exp 201: band-staleness foraging + clamped-learning-rate confound control.
    # ALL defaults preserve Exp 194-200 byte-identical behaviour.
    #
    # enable_band_staleness: when True (and thermosense_forage_mode True), the
    #   forage policy steers toward each creature's PRIVATE EMA estimate of the
    #   drifting band center instead of reading world.current_food_optimal for
    #   free.  Precision (thermosense_intensity) keys the tracker responsiveness
    #   and reading noise — so precision finally buys tracking quality of a moving
    #   target.  Requires enable_food_coupling AND enable_temperature (asserted).
    # band_responsiveness: scales the tracker EMA alpha = clamp(intensity*resp,0,1).
    # freeze_learning_rate: confound-killer arm — pins learning_rate to the founder
    #   value so the learned resource map cannot substitute for the costed organ.
    # ------------------------------------------------------------------
    enable_band_staleness: bool = False
    band_responsiveness: float = 1.0
    freeze_learning_rate: bool = False

    # ------------------------------------------------------------------
    # Exp 202: interference-competition / frequency-dependence escape.
    # ALL defaults preserve Exp 194-201 byte-identical behaviour.
    #
    # shuffle_creature_order: when True, alive creatures are processed in a
    #   RANDOMISED order each step (rng.shuffle, drawing ONLY in this gated ON
    #   branch) instead of ascending creature_id. This neutralises the id-order
    #   "eat-first" confound (low-id creatures otherwise win every contested cell
    #   regardless of skill; new high-id mutants are structurally disadvantaged),
    #   so interference competition at a genuinely-depleting band is keyed to
    #   navigation skill, not birth order. consume() is UNCHANGED — this is the
    #   only id-order fix that survives the no-hidden-evaluator invariant cleanly.
    # track_band_strip: gated telemetry (NO rng, NOT in events_hash) recording how
    #   much of the food band is depleted WITHIN a step (before regen) — the
    #   go/no-go that the band is genuinely contested (exp201 failed here, strip~0).
    # ------------------------------------------------------------------
    shuffle_creature_order: bool = False
    track_band_strip: bool = False


# ---------------------------------------------------------------------------
# Ecology
# ---------------------------------------------------------------------------
class Ecology:
    """Single simulation instance.  Deterministic given cfg + seed.

    Usage:
      eco = Ecology(cfg, seed=42)
      summary = eco.run()
    """

    def __init__(self, cfg: EcologyConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed

        # Deterministically derive sub-seeds so world and creature RNG are
        # independent but fully reproducible.
        master_rng = np.random.default_rng(seed)
        world_seed = int(master_rng.integers(0, 2**31))
        main_seed = int(master_rng.integers(0, 2**31))

        self.rng = np.random.default_rng(main_seed)
        world_rng = np.random.default_rng(world_seed)

        self.world = GridWorld.from_config(
            rows=cfg.rows,
            cols=cfg.cols,
            capacity=cfg.capacity,
            regen_rate=cfg.regen_rate,
            initial_resource=cfg.initial_resource,
            rng=world_rng,
            enable_temperature=cfg.enable_temperature,
            temperature_comfort=cfg.temperature_comfort,
            thermosense_noise_base=cfg.thermosense_noise_base,
            thermal_avoidance_weight=cfg.thermal_avoidance_weight,
            thermosense_active_threshold=cfg.thermosense_active_threshold,
            forage_mode=cfg.thermosense_forage_mode,
            food_optimal_base=cfg.food_optimal_base,
            enable_food_coupling=cfg.enable_food_coupling,
            food_band_width=cfg.food_band_width,
            food_concentration=cfg.food_concentration,
            enable_band_staleness=cfg.enable_band_staleness,
            band_responsiveness=cfg.band_responsiveness,
        )

        # Exp 201 guard: band-staleness needs a drifting food band to track, which
        # requires both the food-coupling regen concentration and the temperature
        # field.  Fail loudly rather than silently no-op or None-deref.
        if cfg.enable_band_staleness:
            assert cfg.enable_food_coupling and cfg.enable_temperature, (
                "enable_band_staleness requires enable_food_coupling AND "
                "enable_temperature (a drifting thermal food band to track)"
            )

        self.t: int = 0
        self.next_id: int = 0
        self.events: list[dict[str, Any]] = []
        self.exploded: bool = False
        # Exp 202: band-strip telemetry (NOT in events_hash; populated only when
        # cfg.track_band_strip is True). A plain attribute, so the OFF path is
        # byte-identical to Exp 194-201.
        self.strip_log: list[dict[str, Any]] = []

        # Creatures stored in a list; we always iterate in ascending id order.
        # _creatures accumulates EVERY creature ever born (dead included — needed by the
        # final newborn/lineage summary). _alive_list holds ONLY currently-alive creatures
        # and is maintained incrementally so the per-step _alive() scan is O(alive), not
        # O(total-ever-born) (the latter dominated long high-turnover runs). Determinism-safe:
        # _alive() still returns the SAME sorted-by-id order, so events_hash is unchanged.
        self._creatures: list[Creature] = []
        self._alive_list: list[Creature] = []

        # Place initial_population founders at deterministic spread positions
        n_cells = cfg.rows * cfg.cols
        total = cfg.rows * cfg.cols
        step = max(1, total // cfg.initial_population)
        for i in range(cfg.initial_population):
            pos = (i * step) % total
            start_energy = cfg.founder.energy_capacity * 0.75
            ph = Phenotype(energy=start_energy, age=0, pos=pos)
            c = Creature(
                creature_id=self.next_id,
                parent_id=None,
                generation=0,
                lineage_root=self.next_id,
                genotype=cfg.founder,
                phenotype=ph,
                n_cells=n_cells,
            )
            self._creatures.append(c)
            self._alive_list.append(c)
            self.events.append(self._event("birth", c, details={"founder": True}))
            self.next_id += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _alive(self) -> list[Creature]:
        """Return alive creatures sorted by ascending creature_id (deterministic).

        Scans self._alive_list (only currently-alive creatures), maintained incrementally
        by step(); identical result to filtering self._creatures but O(alive) not
        O(total-ever-born). The sorted-by-id order is unchanged, so events_hash is unchanged.
        """
        return sorted(self._alive_list, key=lambda c: c.creature_id)

    def _event(self, event_type: str, c: Creature, details: dict | None = None) -> dict:
        g = c.genotype
        return {
            "t": self.t,
            "event_type": event_type,
            "creature_id": c.creature_id,
            "parent_id": c.parent_id,
            "generation": c.generation,
            "location": c.phenotype.pos,
            "energy": round(c.phenotype.energy, 6),
            "genotype": {
                "movement_cost": g.movement_cost,
                "baseline_metabolic_cost": g.baseline_metabolic_cost,
                "energy_capacity": g.energy_capacity,
                "reproduction_energy_threshold": g.reproduction_energy_threshold,
                "aging_cost": g.aging_cost,
                "sensor_precision": g.sensor_precision,
                "maturity_age": g.maturity_age,
                "exploration_bias": g.exploration_bias,
                "learning_rate": g.learning_rate,
            },
            "details": details or {},
        }

    # ------------------------------------------------------------------
    # Per-creature step logic (each reads ONLY its own state + local cell)
    # NO GLOBAL RANKING: reproduction/death eligibility are evaluated
    # per-creature in isolation, using only that creature's own
    # genotype thresholds and its local cell's resource value.
    # ------------------------------------------------------------------
    def _is_reproduction_eligible(self, c: Creature) -> bool:
        """Check reproduction eligibility using ONLY this creature's own state.
        No cross-creature comparison or global ranking involved.
        """
        ph = c.phenotype
        g = c.genotype
        if not ph.alive:
            return False
        if ph.age < g.maturity_age:
            return False
        if ph.energy < g.reproduction_energy_threshold:
            return False
        return True

    def _reproduction_cost(self, c: Creature) -> tuple[float, float]:
        """Return (transfer, overhead) for reproduction.
        Complexity penalty: higher-capacity / higher-sensor lineages pay MORE.
        Uses the single canonical genotype_complexity() helper (shared with senescence).
        """
        g = c.genotype
        ph = c.phenotype
        transfer = ph.energy * g.reproduction_energy_transfer_fraction
        overhead = ph.energy * g.reproduction_cost_fraction * (1.0 + genotype_complexity(g))
        return transfer, overhead

    def _step_one_creature(
        self, c: Creature, pending_children: list[Creature]
    ) -> None:
        """Execute one timestep for a single creature.  Reads only c + local cell.
        NO global ranking, NO cross-creature comparison.
        """
        ph = c.phenotype
        g = c.genotype
        cfg = self.cfg

        # 1. Sense + update learned map
        _observed = c.sense(self.world, self.rng)

        # 2. Choose + execute action; pay movement_cost only if cell changed
        old_pos = ph.pos
        new_pos = c.act(self.world, self.rng)
        if new_pos != old_pos:
            ph.energy -= g.movement_cost
            if self.cfg.log_moves:
                self.events.append(self._event("move", c, details={"from": old_pos, "to": new_pos}))

        # 3. Pay metabolic + aging cost
        ph.energy -= g.baseline_metabolic_cost + g.aging_cost * ph.age

        # Exp 197: complexity-linked maintenance cost. More complex creatures cost more
        # energy per tick to keep alive (upkeep = base + complexity_cost_scale * complexity).
        # Gated on scale != 0.0 so the OFF path is byte-identical to Exp 194/196 (L16 guard).
        if cfg.complexity_cost_scale != 0.0:
            ph.energy -= cfg.complexity_cost_scale * genotype_complexity(g)

        # Exp 197: temperature stress at the new cell — energy-mediated (death stays "starvation").
        # OFF when temperature is None (enable_temperature=False).
        if self.world.temperature is not None:
            ph.energy -= self.world.temperature_stress(
                new_pos, g.temperature_tolerance, cfg.temperature_stress_scale,
            )
            # Tolerance upkeep: wider tolerable band = more expressed machinery = higher cost.
            # Gated on scale != 0.0 (L16 guard) — OFF path is byte-identical.
            if cfg.tolerance_cost_scale != 0.0:
                ph.energy -= cfg.tolerance_cost_scale * g.temperature_tolerance

        # Exp 197: thermosense upkeep — cost of the EXPRESSED capability (floored, never free).
        # OFF when enable_thermosense=False (default) or organ inactive (intensity <= threshold).
        if cfg.enable_thermosense and thermosense_active(g, cfg.thermosense_active_threshold):
            ph.energy -= thermosense_upkeep(g, cfg.thermosense_upkeep_floor, cfg.thermosense_active_threshold)

        # 4. Eat resource at new cell; cap energy at energy_capacity
        deficit = g.energy_capacity - ph.energy
        if deficit > 0:
            eaten = self.world.consume(new_pos, deficit)
            ph.energy += eaten
            ph.resource_eaten += eaten
        ph.energy = min(ph.energy, g.energy_capacity)
        ph.age += 1

        # 5. Update stress as EMA of energy deficit fraction (0 = full, 1 = empty)
        deficit_frac = max(0.0, 1.0 - ph.energy / g.energy_capacity)
        ph.stress = 0.9 * ph.stress + 0.1 * deficit_frac  # EMA, bounded [0,1]

        # 6. REPRODUCTION — reads only this creature + local cell; no ranking
        if self._is_reproduction_eligible(c):
            transfer, overhead = self._reproduction_cost(c)
            # Capture energy at the reproduction decision point (post-eat, pre-payment)
            energy_at_repro = ph.energy
            # Only reproduce if parent stays above min_survival_energy
            if energy_at_repro - transfer - overhead > cfg.min_survival_energy:
                # Child placement: lowest-indexed neighbor (deterministic), or parent cell
                neighbors = self.world.neighbors(new_pos)
                child_pos = min(neighbors) if neighbors else new_pos

                child_geno = mutate(c.genotype, self.rng, cfg.mutation_rate,
                                    mutate_thermosense=cfg.enable_thermosense,
                                    freeze_learning_rate=cfg.freeze_learning_rate)
                child_ph = Phenotype(energy=transfer, age=0, pos=child_pos, birth_t=self.t)
                child = Creature(
                    creature_id=self.next_id,
                    parent_id=c.creature_id,
                    generation=c.generation + 1,
                    lineage_root=c.lineage_root,
                    genotype=child_geno,
                    phenotype=child_ph,
                    n_cells=cfg.rows * cfg.cols,
                )
                self.next_id += 1
                ph.energy -= (transfer + overhead)
                ph.offspring_count += 1
                pending_children.append(child)
                self.events.append(self._event("reproduction", c, details={
                    "transfer": round(transfer, 6),
                    "overhead": round(overhead, 6),
                    "child_id": child.creature_id,
                    # F4-verifiable fields: parent state at the decision point.
                    # energy_at_repro is the exact float (not rounded) so F4 checks
                    # can compare it accurately against reproduction_energy_threshold.
                    "parent_age_at_repro": ph.age,
                    "parent_energy_at_repro": energy_at_repro,
                    "parent_maturity_age": g.maturity_age,
                    "parent_repro_energy_threshold": g.reproduction_energy_threshold,
                }))
                self.events.append(self._event("birth", child, details={
                    "founder": False,
                    "parent_id": c.creature_id,
                }))
                assert is_valid(child_geno), "Mutation produced invalid genotype — engine bug"

        # 7a. Senescence degradation — ONLY when enable_senescence is True.
        #     Deterministic (no rng draws), so OFF path is byte-identical to Exp 194.
        #     Uses genotype_complexity() — the same helper used by reproduction overhead,
        #     so the two paths cannot diverge.
        if cfg.enable_senescence:
            _c = genotype_complexity(g)
            onset = cfg.senescence_onset0 * (1.0 - cfg.senescence_onset_frailty * _c)
            if ph.age > onset:
                deg = (
                    cfg.senescence_base
                    * (1.0 + cfg.senescence_rate_frailty * _c)
                    * (ph.age - onset) ** cfg.senescence_exp
                )
                maintenance = cfg.senescence_self_maintenance * (ph.energy / g.energy_capacity)
                ph.damage += max(0.0, deg - maintenance)

        # 7b. Death precedence: starvation first (acute), then senescence.
        #     Causes are mutually exclusive; senescence path only active when flag is True.
        #
        #     CRITICAL (L16 no-op guard): when enable_senescence is False, we emit the
        #     EXACT same event dict as Exp 194 (no extra keys) to preserve the hash.
        if ph.energy <= 0.0:
            ph.alive = False
            ph.cause_of_death = "starvation"
            if cfg.enable_senescence:
                # Treatment arm: include complexity in death event for P3/P5 verifiability
                self.events.append(self._event("death", c, details={
                    "cause": "starvation",
                    "age": ph.age,
                    "complexity": genotype_complexity(g),
                    "offspring_count": ph.offspring_count,
                }))
            else:
                # Control arm: byte-identical to Exp 194 (no complexity key)
                self.events.append(self._event("death", c, details={
                    "cause": "starvation",
                    "age": ph.age,
                    "offspring_count": ph.offspring_count,
                }))
        elif cfg.enable_senescence and ph.damage >= cfg.senescence_damage_death:
            ph.alive = False
            ph.cause_of_death = "senescence"
            self.events.append(self._event("death", c, details={
                "cause": "senescence",
                "age": ph.age,
                "complexity": genotype_complexity(g),
                "offspring_count": ph.offspring_count,
            }))

    # ------------------------------------------------------------------
    # Step / Run
    # ------------------------------------------------------------------
    def step(self) -> None:
        """Execute ONE timestep for all alive creatures, then regenerate world."""
        pending_children: list[Creature] = []

        # Dynamic comfort zone: update current_comfort BEFORE processing creatures.
        # When comfort_amplitude == 0, sin term == 0 → current_comfort == temperature_comfort
        # for all t → byte-identical to the static comfort implementation.
        if self.world.temperature is not None:
            self.world.current_comfort = (
                self.cfg.temperature_comfort
                + self.cfg.comfort_amplitude
                * math.sin(2.0 * math.pi * self.t / self.cfg.comfort_period)
            )

        # Exp 200: update food-optimal temperature BEFORE processing creatures.
        # When enable_food_coupling is False this block never runs — byte-identical.
        # When food_optimal_amplitude == 0 the result is constant (food_optimal_base).
        if self.cfg.enable_food_coupling:
            self.world.current_food_optimal = (
                self.cfg.food_optimal_base
                + self.cfg.food_optimal_amplitude
                * math.sin(2.0 * math.pi * self.t / self.cfg.food_optimal_period)
            )

        # Process alive creatures. OFF path (shuffle_creature_order=False) iterates in
        # ascending creature_id order — BYTE-IDENTICAL to Exp 194-201. Exp 202: when
        # shuffle_creature_order is True, a RANDOMISED order each step (rng.shuffle, drawing
        # ONLY in this gated ON branch) neutralises the id-order eat-first confound so that
        # interference competition at a contested cell is keyed to navigation skill, not birth
        # order. consume() is UNCHANGED.
        order = self._alive()
        if self.cfg.shuffle_creature_order:
            self.rng.shuffle(order)

        # Exp 202: band-strip validity instrumentation (gated; NO rng draw, NOT in events_hash) —
        # the go/no-go that the food band is GENUINELY depleted within a step (exp201: strip~0).
        _band_before = None
        if self.cfg.track_band_strip and self.world.temperature is not None and self.world.enable_food_coupling:
            _mask = np.abs(self.world.temperature - self.world.current_food_optimal) <= self.world.food_band_width
            _band_before = float(np.sum(self.world.resource[_mask]))

        for c in order:
            self._step_one_creature(c, pending_children)

        if _band_before is not None:
            _band_after = float(np.sum(self.world.resource[_mask]))
            _n_occ = sum(1 for c in order if bool(_mask[c.phenotype.pos]))
            self.strip_log.append({
                "t": self.t,
                "strip": max(0.0, _band_before - _band_after),
                "band_before": _band_before,
                "occupants": _n_occ,
            })

        # Add pending children (born creatures act from next step)
        self._creatures.extend(pending_children)
        # Maintain the incremental alive-list (perf; determinism-safe): drop creatures that
        # died this step, append the new children. O(alive), replacing the old per-step
        # O(total-ever-born) full scan of self._creatures.
        self._alive_list = [c for c in self._alive_list if c.phenotype.alive]
        self._alive_list.extend(pending_children)

        # World regeneration
        self.world.step_regen()

        # Log resource tick every 50 steps (reduce event volume)
        if self.t % 50 == 0:
            total_res = float(np.sum(self.world.resource))
            self.events.append({
                "t": self.t,
                "event_type": "resource_tick",
                "total_resource": round(total_res, 4),
                "creature_id": None,
                "parent_id": None,
                "generation": None,
                "location": None,
                "energy": None,
                "genotype": {},
                "details": {},
            })

        # RUNAWAY GUARD: safety assertion only, NEVER a fitness culler.
        # len(self._alive_list) — same value as the old full scan of self._creatures, O(1).
        alive_count = len(self._alive_list)
        if alive_count > self.cfg.max_population:
            self.exploded = True
            self.events.append({
                "t": self.t,
                "event_type": "explosion",
                "alive_count": alive_count,
                "max_population": self.cfg.max_population,
                "creature_id": None,
                "parent_id": None,
                "generation": None,
                "location": None,
                "energy": None,
                "genotype": {},
                "details": {"message": "Population exceeded max_population. Run stopped (safety guard, not a culler)."},
            })

        self.t += 1

    def run(self) -> dict[str, Any]:
        """Run until horizon, extinction, or explosion.  Return summary dict."""
        while True:
            if len(self._alive_list) == 0:   # extinction (O(1), was a full self._creatures scan)
                break
            if self.exploded:
                break
            if self.t >= self.cfg.horizon:
                break
            self.step()

        return self._compute_summary()

    def _compute_summary(self) -> dict[str, Any]:
        alive = [c for c in self._creatures if c.is_alive()]
        dead = [c for c in self._creatures if not c.is_alive()]
        all_c = self._creatures

        # Cause of death tally
        cod_tally: dict[str, int] = {}
        for c in dead:
            cause = c.phenotype.cause_of_death or "unknown"
            cod_tally[cause] = cod_tally.get(cause, 0) + 1

        # cohort_mortality: starvation_deaths / total_births (honest descriptive metric).
        # Measures what fraction of all creatures ever born died; in a scarce environment
        # more of the cohort dies, fewer survive.  EXPLORATORY / POST-HOC metric.
        #
        # starvation_death_fraction: starvation_deaths / total_deaths.
        # senescence_death_fraction: senescence_deaths / total_deaths (Exp 195 P5 metric).
        total_deaths = len(dead)
        starvation_deaths = cod_tally.get("starvation", 0)
        senescence_deaths = cod_tally.get("senescence", 0)
        cohort_mortality = starvation_deaths / len(all_c) if len(all_c) > 0 else 0.0
        starvation_death_fraction = starvation_deaths / max(1, total_deaths) if total_deaths > 0 else 0.0
        senescence_death_fraction = senescence_deaths / max(1, total_deaths) if total_deaths > 0 else 0.0

        # Births and max generation
        births = len(all_c)
        max_generation = max((c.generation for c in all_c), default=0)

        # Mean lifespan of dead
        lifespans = [c.phenotype.age for c in dead]
        mean_lifespan = float(np.mean(lifespans)) if lifespans else 0.0

        # Reproduction events
        repro_count = sum(1 for e in self.events if e["event_type"] == "reproduction")

        # Resource stats
        total_resource = float(np.sum(self.world.resource))
        mean_resource = float(np.mean(self.world.resource))

        # Last-generation trait means (mean over all creatures in max generation)
        max_gen_creatures = [c for c in all_c if c.generation == max_generation]
        last_gen_trait_means: dict[str, float] = {}
        if max_gen_creatures:
            from dataclasses import asdict
            trait_keys = list(asdict(max_gen_creatures[0].genotype).keys())
            for k in trait_keys:
                vals = [getattr(c.genotype, k) for c in max_gen_creatures]
                last_gen_trait_means[k] = float(np.mean(vals))

        return {
            "scenario": self.cfg.name,
            "seed": self.seed,
            "horizon": self.cfg.horizon,
            "steps_run": self.t,
            "final_pop": len(alive),
            "total_creatures": len(all_c),
            "births": births,
            "deaths": total_deaths,
            "cause_of_death_tally": cod_tally,
            "cohort_mortality": round(cohort_mortality, 6),
            "starvation_death_fraction": round(starvation_death_fraction, 6),
            "senescence_death_fraction": round(senescence_death_fraction, 6),
            "max_generation": max_generation,
            "reproduction_count": repro_count,
            "mean_lifespan": round(mean_lifespan, 4),
            "total_resource": round(total_resource, 4),
            "mean_resource": round(mean_resource, 6),
            "extinction": len(alive) == 0,
            "explosion": self.exploded,
            "last_gen_trait_means": last_gen_trait_means,
            "events_hash": self.events_hash(),
        }

    def events_hash(self) -> str:
        """SHA-256 over canonical JSON of events (no wall-clock, sorted keys).
        This is the determinism fingerprint.
        """
        canonical = json.dumps(self.events, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
