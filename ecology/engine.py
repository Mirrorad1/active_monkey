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
from ecology.world import _TERRAIN_GATE_SOFTNESS_DEFAULT, _TERRAIN_RIDGE_HEIGHT_DEFAULT
from ecology.world import GridWorld
from ecology.creature import Creature, Phenotype
from ecology.continuous_world import ContinuousWorld


# ---------------------------------------------------------------------------
# Exp 243 Mechanism A: pure hazard helper (module-level, no imports, no state)
# ---------------------------------------------------------------------------
def _density_mortality_p(N, hmax, Kc, theta, rate_scale=0.0, dN=0.0):
    """Exp 243 Mechanism A per-step crowding death probability (theta-logistic on N).

    p = hmax * clamp((N/Kc)**theta, 0, 1)  [ + rate_scale*max(0, dN/Kc) when rate_scale>0 ].
    Pure function of the GLOBAL scalar N (no genotype term) -> trait-flat by construction.
    """
    factor = (N / Kc) ** theta
    if factor < 0.0:
        factor = 0.0
    elif factor > 1.0:
        factor = 1.0
    p = hmax * factor
    if rate_scale > 0.0 and dN > 0.0:
        p += rate_scale * (dN / Kc)
    return p


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

    # ------------------------------------------------------------------
    # Exp 203: selection-gradient audit — clamp a sensor value and breed it true
    # while cost stays ON, and seed a polymorphic common-garden of clamp values.
    # ALL defaults preserve Exp 194-202 byte-identical behaviour.
    #
    # freeze_thermosense: when True, offspring inherit the PARENT's thermosense
    #   intensity + inefficiency UNMUTATED (the perturbation is drawn then
    #   discarded — rng stream unchanged), so a clamped sensor value breeds true
    #   across generations.  Upkeep is STILL charged (this is NOT cost-off): it is
    #   the unit of a realized-fitness-at-fixed-h measurement.  Mirrors
    #   freeze_learning_rate.  Default False ⇒ byte-identical.
    # founder_mix: when not None, an ((Genotype, count), ...) tuple seeding the
    #   initial population from EXPLICIT genotypes (a polymorphic common garden:
    #   resident + a grid of clamp values in ONE shared world) instead of
    #   initial_population copies of cfg.founder.  None ⇒ the existing
    #   single-founder seeding, byte-identical.  NO direct-h-reward: founder_mix
    #   only SEEDS genotypes; survival/reproduction read only each creature's own
    #   state (the no-hidden-evaluator invariant is untouched).
    # ------------------------------------------------------------------
    freeze_thermosense: bool = False
    founder_mix: tuple[tuple["Genotype", int], ...] | None = None

    # ------------------------------------------------------------------
    # Exp 204: residue / false-positive discrimination bridge.
    # ALL defaults preserve Exp 194-203 byte-identical behaviour (enable_residue
    # False ⇒ the eat step in _step_one_creature is the EXACT unconditional consume,
    # no extra rng draw, no residue field allocated).
    #
    # The mechanic (ON-branch only): eaten food leaves a misleading TRACE
    # (residue += eaten*residue_yield), decaying residue_decay/step.  At the eat
    # step the creature reads a NOISY freshness percept f_hat = f + N(0, sigma) of
    # the true fresh fraction f = R/(R+residue), with sigma = residue_confusion*(1-h)
    # (h = thermosense_intensity; residue_confusion = the signature-gap difficulty).
    # It EATS iff f_hat >= residue_eat_threshold; if it ate an actually
    # residue-dominated cell (f < residue_fp_threshold) it paid a FALSE POSITIVE
    # (energy -= residue_loss).  ANTI-CHEAT (binding): intake is the UNCHANGED
    # consume(); h keys ONLY the percept noise; residue_loss is an ACTION cost,
    # identical regardless of h — never a reward written as f(h).
    # ------------------------------------------------------------------
    enable_residue: bool = False
    residue_yield: float = 1.0           # residue produced per unit of food eaten
    residue_decay: float = 0.05          # per-step residue decay fraction
    residue_loss: float = 0.5            # energy cost of a false positive (eating residue)
    residue_eat_threshold: float = 0.5   # perceived fresh-fraction at/above which it eats
    residue_fp_threshold: float = 0.5    # true fresh-fraction below which an eat is a false positive
    residue_confusion: float = 0.6       # signature-gap difficulty; percept noise sd = confusion*(1-h)

    # ------------------------------------------------------------------
    # Exp 206: rotating-class niche / sympatric-divergence bridge.
    # ALL defaults preserve Exp 194-205 byte-identical behaviour (enable_niche False ⇒ no
    # niche state allocated, the eat step is the plain consume, choose_action niche branch
    # never runs — no rng-stream perturbation anywhere).
    #
    # Each cell has a hidden time-ROTATING class j(pos,t)=floor(niche_classes*frac(class_phase[pos]
    # + frac(t*niche_rotation))). Rotation makes the class non-memorizable by the learned map (the
    # confound-hunters' load-bearing fix). The sensor (thermosense_intensity h) keys ONLY a noisy
    # read of a cell's CURRENT class (in routing, creature.py); the EAT step is h-blind and applies
    # a frequency-dependent crowding discount kept = intake/(1 + niche_crowding*occ_prev[j_true]),
    # where occ_prev is the FROZEN previous-step count of creatures on that class (h-blind).
    # ANTI-CHEAT: no food/fitness is f(h); h enters only the routing percept noise sd =
    # niche_confusion*(1-h); the crowding divisor is a creature-COUNT on the TRUE class.
    # ------------------------------------------------------------------
    enable_niche: bool = False
    niche_classes: int = 2               # K (A=common, B=rarer)
    niche_rotation: float = 0.0          # per-step class-phase rotation; 0 ⇒ static (STATIC_NICHE control)
    niche_confusion: float = 0.6         # routing percept noise scale; sd = niche_confusion*(1-h)
    niche_crowding: float = 0.0          # crowding discount strength (0 ⇒ NO_CROWDING control)
    niche_weight: float = 4.0            # routing bias toward the read under-crowded class
    niche_barcode_shuffle: bool = False  # BARCODE_SHUFFLED placebo: decorrelate read-class from value

    # ------------------------------------------------------------------
    # Exp hidden-state-mode: binary hidden mode with noisy cue + memory integration.
    # ALL defaults preserve Exp 194-206 byte-identical behaviour (enable_hidden_mode False ⇒
    # no mode state, no rng draw in step(), no creature cue draw, no memory upkeep).
    #
    # enable_hidden_mode: gates the full mechanism (world regen, engine mode-switch, creature
    #   cue/routing, memory upkeep, memory_horizon mutation).
    # mode_switch_prob: per-step probability of flipping hidden_mode (ONE rng draw per step).
    # cue_noise: std-dev of the noisy cue each creature observes each step.
    # memory_upkeep_floor: fixed energy cost per step when memory_horizon > 0 (gated).
    # memory_cost_slope: per-unit memory-horizon cost per step (gated).
    #
    # ANTI-CHEAT: food intake is the UNCHANGED consume(); memory_horizon keys ONLY (a) how many
    # cues are averaged and (b) the upkeep cost.  Nothing is written as f(memory_horizon).
    # ------------------------------------------------------------------
    enable_hidden_mode: bool = False
    mode_switch_prob: float = 0.02
    cue_noise: float = 0.5
    mode_wrong_regen_factor: float = 0.3   # wrong-half regen factor (milder payoff; 0.0 = harsh gating)
    # Mode-dependent HAZARD (decouples the inference payoff from carrying capacity): with
    # mode_wrong_regen_factor=1.0 food is UNIFORM (pops stay healthy) and a creature in a
    # wrong-type cell (cell_type != hidden_mode) pays mode_hazard_scale energy per step, so
    # correct inference (avoid the wrong half) pays without gating total food. 0.0 = no hazard.
    mode_hazard_scale: float = 0.0
    memory_upkeep_floor: float = 0.0
    memory_cost_slope: float = 0.01

    # ------------------------------------------------------------------
    # Phase 4: active sensing — per-step paid probe for extra cues.
    # ALL defaults preserve pre-Phase-4 byte-identical behaviour.
    #
    # enable_active_sensing: gates the probe draws in creature.choose_action and the
    #   probe cost in _step_one_creature.  False (default) ⇒ no extra rng draws ⇒
    #   byte-identical to Phase-3 paths.
    # probe_cost: energy charged per step when the creature probed (information_sampling_rate
    #   >= rng draw).  0.0 default ⇒ probing is free (only non-zero when explicitly set).
    # probe_n_samples: number of extra cue samples drawn per probe event.
    # ------------------------------------------------------------------
    enable_active_sensing: bool = False   # defaults preserve byte-identical OFF behaviour
    probe_cost: float = 0.0
    probe_n_samples: int = 4

    # ------------------------------------------------------------------
    # Exp 211: probe-policy abstraction — HOW the within-step probe is triggered.
    # ALL defaults preserve Exp 210 behaviour: probe_policy="fixed_rate" + the default
    # OFF gate means an enable_active_sensing=True run that does NOT set probe_policy
    # reproduces Exp 210 byte-for-byte (the fixed_rate branch is the verbatim Exp 210
    # code path, golden-hash guarded). When enable_active_sensing is False the probe
    # block never runs ⇒ byte-identical regardless of probe_policy.
    #
    # probe_policy ∈ {
    #   "off"                 — no probe (== enable_active_sensing False; for completeness),
    #   "fixed_rate"          — Exp 210: probe iff u < information_sampling_rate,
    #   "uncertainty_gated"   — probe with prob = info_sampling_rate * sigmoid(
    #                            sensitivity*(threshold - action_margin)); action_margin =
    #                            |provisional_belief - 0.5| from the SINGLE fresh cue (the
    #                            creature's own ambiguity about which half to steer to),
    #   "random_cost_matched" — probe with fixed prob random_cost_matched_probe_rate
    #                            (budget-matched random TIMING; same info, same cost),
    #   "pure_cost"           — uncertainty-gated TRIGGER + cost, but the extra cues are
    #                            NOT integrated (pays the cost, gains no information),
    #   "gate_shuffle"        — uncertainty-gated but the gate reads a TIME-SHUFFLED margin
    #                            (same marginal probe rate, timing decorrelated from the
    #                            current step's uncertainty; info ON),
    #   "hidden_scramble"     — uncertainty-gated TRIGGER + cost, extra cues drawn from a
    #                            SCRAMBLED mode (no mutual information with the true mode).
    # }
    # The gate params below are CONFIG (shared by resident+mutant in the common garden);
    # the heritable knob is information_sampling_rate (the probe GAIN / cap under
    # uncertainty), so the local-gradient mutant probes MORE only where it is uncertain.
    probe_policy: str = "fixed_rate"
    uncertainty_gate_threshold: float = 0.15      # action-margin cutoff (probe when margin < ~this)
    uncertainty_gate_sensitivity: float = 20.0    # sigmoid steepness of the gate ramp
    random_cost_matched_probe_rate: float = 0.0   # fixed probe prob for random_cost_matched
    gate_shuffle_buffer: int = 64                 # ring-buffer length for the time-shuffle gate

    # ------------------------------------------------------------------
    # Exp 235: terrain / locomotion evolvability substrate — OFF by default.
    # ALL defaults preserve Exp 194-213 byte-identical behaviour (enable_terrain False ⇒
    # no elevation field, no crossing rolls, no climb_ability mutation, no climb cost).
    #
    # enable_terrain: gates ALL terrain mechanics (world elevation build, movement gate,
    #   plateau regen boost, climb cost, climb_ability mutation in mutate()).
    #   False (default) ⇒ zero new rng draws ⇒ byte-identical to Exp 194-213.
    #
    # terrain_food_concentration: regen boost for plateau cells (mirrors food_concentration
    #   for the thermal-band path; conserved-total).  0.0 (default) ⇒ uniform regen.
    #   Only read when enable_terrain is True.
    #
    # terrain_gate_softness: sigmoid ramp softness for upslope crossing probability.
    #   A sane positive default; only read when enable_terrain is True.
    #
    # climb_cost_floor / climb_cost_slope: monotone cost paid per tick when terrain is ON.
    #   total_cost = floor + slope * climb_ability (mirrors thermosense_upkeep cost shape).
    #   Both 0.0 (default) ⇒ OFF (no climb cost ⇒ byte-identical OFF path).
    #
    # terrain_gates_movement: when True (and enable_terrain), movement uses
    #   climbable_neighbors() instead of world.neighbors().  Default True so that
    #   the plateau is genuinely sealed; set False for the gate-open control.
    # ------------------------------------------------------------------
    enable_terrain: bool = False
    terrain_food_concentration: float = 0.0
    terrain_gate_softness: float = _TERRAIN_GATE_SOFTNESS_DEFAULT
    terrain_ridge_height: float = _TERRAIN_RIDGE_HEIGHT_DEFAULT   # plateau elevation (binary geometry)
    climb_cost_floor: float = 0.0
    climb_cost_slope: float = 0.0
    terrain_gates_movement: bool = True

    # ------------------------------------------------------------------
    # Exp 236: navigation-capable foraging policy — OFF by default.
    # ALL defaults preserve Exp 194-235 byte-identical behaviour.
    #
    # enable_navigation: when True (and use_thermo is False), the depleted-cell
    #   foraging branch uses a GLOBAL target-selection step before the existing
    #   local explore/exploit logic:
    #     - pick TARGET = argmax over all cells of
    #         score(cell) = m[cell] - nav_distance_penalty * manhattan_dist(pos, cell)
    #     - take ONE STEP toward TARGET among the candidate neighbors
    #       (climbable_neighbors when terrain_gates_movement is True, else neighbors())
    #     - tie-break: closer to TARGET first, then higher m, then lower index
    #     - if no admissible neighbor moves toward TARGET: fall back to best-m neighbor
    #   Resource plentiful => "stay and eat" (unchanged).  OFF => byte-identical.
    #
    # nav_distance_penalty: score discount per unit of manhattan distance.
    #   Small penalty (default 0.05) so creatures prefer closer high-value cells
    #   over very distant ones, but still navigate far if the food value is high.
    # ------------------------------------------------------------------
    enable_navigation: bool = False    # default False => byte-identical to Exp 194-235
    nav_distance_penalty: float = 0.001

    # ------------------------------------------------------------------
    # Exp 237: food-gradient perception — OFF by default.
    # ALL defaults preserve Exp 194-236 byte-identical behaviour.
    #
    # enable_food_sense: when True, creatures sense a distance-decayed resource
    #   "scent" and navigate toward the richest food each depleted step.
    #   The rich plateau (2x regen) raises scent even from the basin, so creatures
    #   persistently retry the terrain rim instead of retreating.
    #   False (default) ⇒ BYTE-IDENTICAL to Exp 194-236 (block never entered).
    # food_sense_decay: decay factor per unit of manhattan distance.
    #   scent(n) = sum_c resource[c] * (food_sense_decay ** manhattan_dist(n, c))
    # ------------------------------------------------------------------
    enable_food_sense: bool = False
    food_sense_decay: float = 0.5

    # ------------------------------------------------------------------
    # Exp 238: continuous locomotion — OFF by default.
    # ALL defaults preserve Exp 194-237 byte-identical behaviour.
    #
    # enable_continuous_locomotion: when True, a ContinuousWorld is used instead of
    #   GridWorld; creatures move d = locomotor_speed * dt per step; intake is the
    #   line integral of rho(x,y) along the swept segment.
    #   False (default) ⇒ BYTE-IDENTICAL to Exp 194-237 (no new paths entered).
    #
    # continuous_layout: resource field layout ("bump", "flat", "neutral").
    # continuous_dt: fixed timestep for continuous movement (distance = speed * dt).
    # speed_cost_floor: fixed upkeep cost per step (when enable_continuous_locomotion).
    # speed_cost_slope: per-unit locomotor_speed upkeep (monotone cost).
    # continuous_regen_rate: per-step regen rate for continuous world sub-cells.
    # continuous_capacity: max resource per sub-cell.
    # ------------------------------------------------------------------
    enable_continuous_locomotion: bool = False
    continuous_layout: str = "bump"
    continuous_dt: float = 1.0
    speed_cost_floor: float = 0.0
    speed_cost_slope: float = 0.0
    continuous_regen_rate: float = 0.05
    continuous_capacity: float = 2.0
    # Exp 240: logistic resource regeneration — OFF (False) by default, byte-identical to
    # Exp 238-239. When True, ContinuousWorld.step_regen() uses logistic formula
    # regen = regen_rate * v * (1 - v/cap) instead of flat regen_rate, producing a
    # genuine negative feedback that prevents the commons-tragedy runaway at Rung 2.
    continuous_logistic_regen: bool = False

    # Exp 242: depletion-aware intake — OFF by default, byte-identical to Exp 238-241.
    # The Exp 238-241 substrate has a silent regulation bug: continuous intake is the line
    # integral of the STRUCTURAL density field rho() (which never depletes), so the
    # depletable _resource grid has NO effect on intake and per-capita intake never falls
    # with density — faster movers always run away (no finite carrying capacity, Exp 241).
    # When True, ContinuousWorld.line_integral_intake multiplies rho by the local
    # availability fraction (resource_cell / capacity), closing the density-dependent
    # feedback so every speed reaches a finite carrying capacity. Requires
    # enable_continuous_locomotion=True. OFF path is byte-identical to Exp 238-241.
    enable_continuous_depletion_intake: bool = False

    # Exp 243: freeze locomotor_speed (breed TRUE) — OFF by default, byte-identical to Exp 238-242.
    # The certification/null runs need MONOMORPHIC populations, but engine.py couples
    # mutate_continuous_locomotion to enable_continuous_locomotion (every continuous run mutates
    # speed). When True (continuous ON), the per-child locomotor_speed rng draw
    # (genotype.py LOCOMOTION_CONTINUOUS_TRAITS skip-guard) is skipped so speed breeds true.
    # OFF (default) ⇒ byte-identical: the mutate flag is unchanged.
    freeze_continuous_locomotion: bool = False

    # Exp 243: Mechanism B — monotone floored regen for the continuous world. OFF byte-identical.
    # NOTE: only observable when enable_continuous_depletion_intake=True (intake ignores _resource otherwise).
    continuous_floored_regen: bool = False

    # Exp 243: Mechanism A — global density-dependent (crowding) mortality. OFF byte-identical.
    # Per-creature Bernoulli death roll keyed ONLY on the frozen total head-count N:
    #   p = hmax * clamp((N/Kc)**theta, 0, 1) [+ rate_scale*max(0,(N-N_prev)/Kc)].
    # A SUBSTRATE scalar, not a genotype trait (no new heritable trait, no mutate_* flag).
    enable_density_mortality: bool = False
    density_mortality_theta: float = 1.0      # 1.0 linear (default); 2.0 = Stage-2 probe
    density_mortality_hmax: float = 0.04      # per-step hazard ceiling
    density_mortality_Kc: float = 60.0        # density scale (birth-balance-derived)
    density_mortality_rate_scale: float = 0.0 # optional lead/derivative brake (default OFF)

    # Exp 245: configurable continuous bump geometry — OFF (defaults) byte-identical to Exp 238-244.
    # continuous_bump_sigma: Gaussian bump width (default 1.5 = _BUMP_SIGMA). Lower → sharper.
    # continuous_bump_amplitude: bump peak density (default 1.0 = _BUMP_AMPLITUDE).
    # continuous_bump_centers: explicit centers for the "bump" layout (None → legacy default).
    #   None (default) → uses _BUMP_CENTERS_BUMP in ContinuousWorld; byte-identical to Exp 238-244.
    continuous_bump_sigma: float = 1.5
    continuous_bump_amplitude: float = 1.0
    continuous_bump_centers: tuple | None = None

    # Exp 246: moving-patch resource mode — OFF (False) by default, byte-identical to Exp 238-245.
    # When True, the structural resource density is a SINGLE concentrated Gaussian whose center
    # drifts deterministically on a circular orbit; creatures crowd the patch and compete.
    # All fields default to their canonical values so the OFF path is perfectly byte-identical.
    # Requires enable_continuous_locomotion=True (cont_world must exist).
    continuous_moving_patch: bool = False
    continuous_patch_sigma: float = 0.8
    continuous_patch_amplitude: float = 3.0
    continuous_patch_orbit_radius: float = 3.0
    continuous_patch_period: int = 300


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
            enable_hidden_mode=cfg.enable_hidden_mode,
            mode_switch_prob=cfg.mode_switch_prob,
            cue_noise=cfg.cue_noise,
            mode_wrong_regen_factor=cfg.mode_wrong_regen_factor,
            enable_active_sensing=cfg.enable_active_sensing,
            probe_n_samples=cfg.probe_n_samples,
            # Exp 211 probe-policy params (defaults preserve Exp 210 fixed_rate behaviour).
            probe_policy=cfg.probe_policy,
            uncertainty_gate_threshold=cfg.uncertainty_gate_threshold,
            uncertainty_gate_sensitivity=cfg.uncertainty_gate_sensitivity,
            random_cost_matched_probe_rate=cfg.random_cost_matched_probe_rate,
            gate_shuffle_buffer=cfg.gate_shuffle_buffer,
            # Exp 235 terrain params — defaults are no-ops (enable_terrain=False).
            enable_terrain=cfg.enable_terrain,
            terrain_food_concentration=cfg.terrain_food_concentration,
            terrain_gate_softness=cfg.terrain_gate_softness,
            terrain_ridge_height=cfg.terrain_ridge_height,
            terrain_gates_movement=cfg.terrain_gates_movement,
            # Exp 236 navigation params — defaults are no-ops (enable_navigation=False).
            enable_navigation=cfg.enable_navigation,
            nav_distance_penalty=cfg.nav_distance_penalty,
            # Exp 237 food-sense params — defaults are no-ops (enable_food_sense=False).
            enable_food_sense=cfg.enable_food_sense,
            food_sense_decay=cfg.food_sense_decay,
        )

        # Exp 238: continuous locomotion — build ContinuousWorld as a SIBLING attribute.
        # The discrete GridWorld above is ALWAYS built (even when ON) to keep founder spawn,
        # world seam calls (step_regen for the discrete path), and all OFF-path code paths
        # identical. When enable_continuous_locomotion=True, the creature movement, intake,
        # and cost seams use self.cont_world instead of self.world; self.world is unused for
        # resource/movement (but retained for consistency with the OFF path structure).
        # OFF: self.cont_world = None; every continuous block is behind `if self.cont_world`.
        if cfg.enable_continuous_locomotion:
            self.cont_world: ContinuousWorld | None = ContinuousWorld.from_config(
                layout=cfg.continuous_layout,
                regen_rate=cfg.continuous_regen_rate,
                capacity=cfg.continuous_capacity,
                logistic_regen=cfg.continuous_logistic_regen,
                enable_depletion_intake=cfg.enable_continuous_depletion_intake,
                floored_regen=cfg.continuous_floored_regen,
                # Exp 245: configurable bump geometry (defaults byte-identical to Exp 238-244).
                bump_sigma=cfg.continuous_bump_sigma,
                bump_amplitude=cfg.continuous_bump_amplitude,
                bump_centers=cfg.continuous_bump_centers,
                # Exp 246: moving-patch resource mode (defaults byte-identical to Exp 238-245).
                moving_patch=cfg.continuous_moving_patch,
                patch_sigma=cfg.continuous_patch_sigma,
                patch_amplitude=cfg.continuous_patch_amplitude,
                patch_orbit_radius=cfg.continuous_patch_orbit_radius,
                patch_period=cfg.continuous_patch_period,
            )
        else:
            self.cont_world = None

        # Exp 201 guard: band-staleness needs a drifting food band to track, which
        # requires both the food-coupling regen concentration and the temperature
        # field.  Fail loudly rather than silently no-op or None-deref.
        if cfg.enable_band_staleness:
            assert cfg.enable_food_coupling and cfg.enable_temperature, (
                "enable_band_staleness requires enable_food_coupling AND "
                "enable_temperature (a drifting thermal food band to track)"
            )

        # Exp 204: allocate the residue trace field ONLY when the mechanic is enabled.
        # When OFF, world.residue stays None and the eat step is byte-identical (no field,
        # no extra rng draw). The field is pure state, never entering events_hash.
        if cfg.enable_residue:
            self.world.residue = np.zeros(cfg.rows * cfg.cols, dtype=np.float64)

        # Exp 206: allocate the rotating-class niche state ONLY when enabled (OFF ⇒ byte-identical).
        # class_phase is a PURE-ARITHMETIC low-discrepancy spread of cells across [0,1) (golden-ratio
        # conjugate) — NO rng, NOT temperature-derived, so the class is not a memorizable spatial
        # value. The mutual-exclusion guard (kin of the band-staleness assert) prevents the niche and
        # band-staleness routing branches from silently shadowing each other.
        if cfg.enable_niche:
            assert not cfg.enable_band_staleness, (
                "enable_niche and enable_band_staleness are mutually exclusive routing modes"
            )
            n_cells = cfg.rows * cfg.cols
            _PHI = 0.6180339887498949  # (sqrt(5)-1)/2, low-discrepancy
            _idx = np.arange(n_cells, dtype=np.float64)
            self.world.class_phase = np.mod((_idx + 0.5) * _PHI, 1.0)
            self.world.class_signal = self.world.class_phase.copy()   # t=0: omega=0
            self.world.class_occ_prev = np.zeros(cfg.niche_classes, dtype=np.int64)
            self.world.class_occ_cur = np.zeros(cfg.niche_classes, dtype=np.int64)
            self.world.enable_niche = True
            self.world.niche_classes = cfg.niche_classes
            self.world.niche_confusion = cfg.niche_confusion
            self.world.niche_weight = cfg.niche_weight
            if cfg.niche_barcode_shuffle:
                # decorrelate the routing read-class from the true crowding class: a deterministic
                # coprime-stride permutation (no rng) — reading precisely tells you nothing actionable.
                self.world.niche_read_perm = (np.arange(n_cells) * 7919) % n_cells
            else:
                self.world.niche_read_perm = None

        self.t: int = 0
        self.next_id: int = 0
        self.events: list[dict[str, Any]] = []
        self.exploded: bool = False
        # Phase 4 telemetry (NOT in events / events_hash): active-sensing probe counter
        # and hidden-mode occupancy counters for decision-quality measurement.
        self.probe_count_total: int = 0
        self.wrong_cell_steps_total: int = 0
        self.hidden_mode_steps_total: int = 0
        # Exp 211 telemetry (NOT in events / events_hash): probe pivotality + uncertainty
        # enrichment. probe_changed_action_count = probes after which m_hat (which-half
        # decision) flipped vs the no-probe (single-cue) decision. action_margin sums let
        # the experiment report mean action-margin overall vs at-probe vs without-probe
        # (the gate-enrichment test). All gated on enable_active_sensing + hidden_mode.
        self.probe_changed_action_count: int = 0
        self.action_margin_sum: float = 0.0       # sum over all hidden-mode steps
        self.action_margin_n: int = 0
        self.action_margin_at_probe_sum: float = 0.0   # sum over probe steps only
        self.action_margin_at_probe_n: int = 0
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

        # Place founders at deterministic spread positions.
        # Exp 203: founder_mix (when not None) seeds an EXPLICIT polymorphic
        # population — a common garden of clamp values in one shared world —
        # expanded from ((Genotype, count), ...).  None ⇒ initial_population
        # copies of cfg.founder, BYTE-IDENTICAL to Exp 194-202 (founders list is
        # [cfg.founder]*initial_population, so step/pos/genotype are unchanged).
        n_cells = cfg.rows * cfg.cols
        total = cfg.rows * cfg.cols
        if cfg.founder_mix is None:
            founders = [cfg.founder] * cfg.initial_population
        else:
            founders = [g for g, cnt in cfg.founder_mix for _ in range(int(cnt))]

        # Exp 235 terrain: restrict founder spawn to BASIN cells (elevation <= 0.0)
        # so founders start inside the sealed area, not on the ridge/plateau.
        # Pure arithmetic (no rng); spawn positions computed from the basin cell list.
        # When terrain is OFF, the existing placement logic is VERBATIM (byte-identical).
        if cfg.enable_terrain and self.world.elevation is not None:
            basin_cells = [c for c in range(n_cells)
                           if float(self.world.elevation[c]) <= 0.0]
            basin_step = max(1, len(basin_cells) // max(1, len(founders)))
            for i, geno in enumerate(founders):
                pos = basin_cells[(i * basin_step) % len(basin_cells)]
                start_energy = geno.energy_capacity * 0.75
                ph = Phenotype(energy=start_energy, age=0, pos=pos)
                c = Creature(
                    creature_id=self.next_id,
                    parent_id=None,
                    generation=0,
                    lineage_root=self.next_id,
                    genotype=geno,
                    phenotype=ph,
                    n_cells=n_cells,
                )
                self._creatures.append(c)
                self._alive_list.append(c)
                self.events.append(self._event("birth", c, details={"founder": True}))
                self.next_id += 1
        else:
            step = max(1, total // max(1, len(founders)))
            for i, geno in enumerate(founders):
                pos = (i * step) % total
                start_energy = geno.energy_capacity * 0.75
                ph = Phenotype(energy=start_energy, age=0, pos=pos)
                c = Creature(
                    creature_id=self.next_id,
                    parent_id=None,
                    generation=0,
                    lineage_root=self.next_id,
                    genotype=geno,
                    phenotype=ph,
                    n_cells=n_cells,
                )
                self._creatures.append(c)
                self._alive_list.append(c)
                self.events.append(self._event("birth", c, details={"founder": True}))
                self.next_id += 1

        # Exp 238: set continuous positions for founders when ON.
        # Spread founders evenly across the arena using their discrete grid pos as seed
        # (pure arithmetic, no rng; OFF path never reaches this block — byte-identical).
        if cfg.enable_continuous_locomotion and self.cont_world is not None:
            from ecology.continuous_world import ARENA_W, ARENA_H
            _rows = cfg.rows
            _cols = cfg.cols
            for c in self._creatures:
                _pos = c.phenotype.pos
                _r, _col = divmod(_pos, _cols)
                # Map discrete (row, col) to continuous arena center of each cell.
                x = ((_col + 0.5) / _cols) * ARENA_W
                y = ((_r + 0.5) / _rows) * ARENA_H
                c.phenotype.pos_cont = (x, y)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def alive_count(self) -> int:
        """Number of currently-alive creatures.  O(1); allocates nothing.

        Equivalent to ``len(self._alive())`` but WITHOUT building the throwaway copy —
        prefer this at count call sites (the old ``len(eco._alive())`` paid an O(alive)
        list copy just to read a length).
        """
        return len(self._alive_list)

    def has_alive(self) -> bool:
        """True iff at least one creature is currently alive.  O(1); allocates nothing.

        Equivalent to ``bool(self._alive())`` without the copy — prefer this at emptiness
        checks (the old ``if not eco._alive():`` copied the whole list just to test a bool).
        """
        return bool(self._alive_list)

    def alive_snapshot(self) -> list[Creature]:
        """A FRESH list copy of the alive creatures in ascending creature_id order.

        Use this when the caller iterates the living creatures or needs an independent,
        mutable list it can reorder/append to (e.g. step()'s shuffle path).  For a pure
        count or emptiness test use alive_count()/has_alive() — they do not allocate.

        Determinism / ordering invariant: self._alive_list is maintained incrementally by
        step() and is ALREADY in ascending creature_id order — founders are appended 0..N at
        init, and each step the survivors (a stable filter preserves order) are followed by
        that step's pending_children, whose ids (next_id, next_id+1, …) are strictly greater
        than every existing id.  So the list is sorted by construction and the old per-step
        ``sorted(...)`` (O(alive log alive), every step) was pure overhead.  The returned copy
        is byte-identical to that old sorted() output, so events_hash is unchanged.  Returning
        a COPY means a caller that mutates it (``rng.shuffle(order)``) cannot corrupt the
        maintained list.  Guarded by tests/test_perf_optimizations.py.
        """
        return list(self._alive_list)

    def _alive(self) -> list[Creature]:
        """Backward-compatible alias for alive_snapshot() — returns a private COPY.

        Retained so existing callers keep working unchanged; new code should call the
        accessor that matches intent — alive_count()/has_alive() for count/emptiness,
        alive_snapshot() when an independent list of the living is actually needed.  Do NOT
        expose the raw self._alive_list to external callers (mutating it corrupts engine
        bookkeeping); use alive_snapshot() for a safe, independent copy.
        """
        return self.alive_snapshot()

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
        if self.cont_world is not None:
            # Exp 238: continuous locomotion branch.
            # The SAME single rng draw path runs via c.act() for byte-identity when OFF;
            # the returned discrete cell is used as a HEADING cue (direction toward that cell)
            # then discarded — continuous physics applies instead.
            # ANTI-CHEAT: locomotor_speed keys ONLY how far we sweep (d = speed * dt).
            from ecology.continuous_world import ARENA_W, ARENA_H
            _hint_cell = c.act(self.world, self.rng)  # consumes the SAME single rng draw
            # Derive heading toward sensed-best direction from cont_world field.
            _x0, _y0 = ph.pos_cont if ph.pos_cont is not None else (0.5, 0.5)
            _hdx, _hdy = self.cont_world.best_heading(_x0, _y0)
            # Advance by d = locomotor_speed * dt (instantaneous; no momentum).
            _d = g.locomotor_speed * cfg.continuous_dt
            _x1 = max(0.0, min(ARENA_W, _x0 + _hdx * _d))
            _y1 = max(0.0, min(ARENA_H, _y0 + _hdy * _d))
            ph.pos_cont = (_x1, _y1)
            # Project to nearest discrete grid cell (byte-identical integer pos for events_hash).
            _r = int(_y1 / ARENA_H * cfg.rows)
            _col = int(_x1 / ARENA_W * cfg.cols)
            _r = max(0, min(cfg.rows - 1, _r))
            _col = max(0, min(cfg.cols - 1, _col))
            new_pos = _r * cfg.cols + _col
            ph.pos = new_pos
            # Movement cost: always charged in continuous mode (every step moves).
            ph.energy -= g.movement_cost
        else:
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

        # Exp 235: terrain climb cost — monotone in climb_ability, mirrors thermosense_upkeep
        # shape (floor + slope * ability).  OFF when enable_terrain=False (default) ⇒
        # byte-identical.  Both floor=0 and slope=0 (defaults) ⇒ zero cost even when ON.
        if cfg.enable_terrain and (cfg.climb_cost_floor != 0.0 or cfg.climb_cost_slope != 0.0):
            climb_cost = cfg.climb_cost_floor + cfg.climb_cost_slope * g.climb_ability
            ph.energy -= max(0.0, climb_cost)

        # Exp 238: continuous locomotion speed cost — monotone in locomotor_speed.
        # OFF when enable_continuous_locomotion=False (default) ⇒ byte-identical (block never
        # entered). Both floor=0 and slope=0 (defaults) ⇒ zero cost even when ON.
        # ANTI-CHEAT: cost is never a function of intake or field value.
        if cfg.enable_continuous_locomotion and (cfg.speed_cost_floor != 0.0 or cfg.speed_cost_slope != 0.0):
            speed_cost = cfg.speed_cost_floor + cfg.speed_cost_slope * g.locomotor_speed
            ph.energy -= max(0.0, speed_cost)

        # Hidden-state-mode HAZARD: a creature standing in a WRONG-type cell (cell_type !=
        # current hidden_mode) pays a small survivable energy penalty. Gated (OFF path
        # byte-identical: no draw, no read). The penalty is on CELL TYPE vs the true mode,
        # never on memory_horizon — memory only steers where the creature stands (anti-cheat).
        if (cfg.enable_hidden_mode and cfg.mode_hazard_scale != 0.0
                and self.world.cell_type is not None
                and self.world.cell_type[new_pos] != self.world.hidden_mode):
            ph.energy -= cfg.mode_hazard_scale

        # Exp hidden-state-mode: memory upkeep — cost of maintaining the belief.
        # OFF when enable_hidden_mode=False or both memory traits 0 ⇒ byte-identical.
        # memory_horizon (integer buffer, rung 1) and belief_persistence (continuous EMA,
        # rung 1b) are alternative integration mechanisms; each pays the same slope per unit.
        if cfg.enable_hidden_mode and g.memory_horizon > 0:
            ph.energy -= cfg.memory_upkeep_floor + cfg.memory_cost_slope * g.memory_horizon
        if cfg.enable_hidden_mode and g.belief_persistence > 0.0:
            ph.energy -= cfg.memory_upkeep_floor + cfg.memory_cost_slope * g.belief_persistence

        # Phase 4: active-sensing probe cost — charged when the creature probed this step.
        # OFF (enable_active_sensing=False) ⇒ probed_this_step never True ⇒ byte-identical.
        if cfg.enable_active_sensing and getattr(c.policy, "probed_this_step", False):
            ph.energy -= cfg.probe_cost
            self.probe_count_total += 1
            # Exp 211: probe pivotality (telemetry, no rng, NOT in events_hash) — the probe
            # CHANGED the next which-half decision vs the no-probe (single-cue) baseline.
            if getattr(c.policy, "last_probe_changed_action", False):
                self.probe_changed_action_count += 1
        # Phase 4 telemetry (NOT in events_hash): wrong-type-cell occupancy = decision quality.
        if cfg.enable_hidden_mode and self.world.cell_type is not None:
            self.hidden_mode_steps_total += 1
            if self.world.cell_type[new_pos] != self.world.hidden_mode:
                self.wrong_cell_steps_total += 1
            # Exp 211: action-margin enrichment telemetry (no rng, NOT in events_hash).
            # last_action_margin is set by choose_action ONLY when active sensing is on.
            if cfg.enable_active_sensing:
                _m = getattr(c.policy, "last_action_margin", None)
                if _m is not None:
                    self.action_margin_sum += _m
                    self.action_margin_n += 1
                    if getattr(c.policy, "probed_this_step", False):
                        self.action_margin_at_probe_sum += _m
                        self.action_margin_at_probe_n += 1

        # 4. Eat resource at new cell; cap energy at energy_capacity
        deficit = g.energy_capacity - ph.energy
        # Exp 238: continuous locomotion eat — line integral along the swept segment.
        # OFF (enable_continuous_locomotion=False) ⇒ byte-identical discrete path below.
        if self.cont_world is not None and ph.pos_cont is not None:
            # Exp 238: continuous eat via line integral along the swept segment.
            # ANTI-CHEAT: intake is integral of the PROVIDED rho field.
            if deficit > 0:
                _x1, _y1 = ph.pos_cont  # new (end) position after this step's move
                _hdx, _hdy = self.cont_world.best_heading(_x1, _y1)
                _d = g.locomotor_speed * cfg.continuous_dt
                from ecology.continuous_world import ARENA_W, ARENA_H
                _x0e = max(0.0, min(ARENA_W, _x1 - _hdx * _d))
                _y0e = max(0.0, min(ARENA_H, _y1 - _hdy * _d))
                eaten = self.cont_world.consume(_x0e, _y0e, _x1, _y1, deficit)
                ph.energy += eaten
                ph.resource_eaten += eaten
            # ph.energy cap, ph.age, stress: fall through to shared code below.
        elif cfg.enable_residue and self.world.residue is not None:
            # Exp 204: residue / false-positive discrimination. ONE rng draw, made ONLY
            # in this gated ON branch and ONLY when deficit > 0 (exactly when the OFF path
            # consumes) — so the OFF path (enable_residue=False) keeps the EXACT
            # unconditional-consume rng stream and is byte-identical to exp194-203.
            # ANTI-CHEAT: intake is the UNCHANGED consume(); thermosense_intensity (h) keys
            # ONLY the percept noise sigma; residue_loss is charged by the ACTION of eating
            # a residue-dominated cell, identical regardless of h — never a reward on h.
            if deficit > 0:
                avail = self.world.resource_at(new_pos)
                res_here = float(self.world.residue[new_pos])
                f = avail / (avail + res_here + 1e-9)          # true fresh fraction at the cell
                h = g.thermosense_intensity
                sigma = max(0.0, cfg.residue_confusion * (1.0 - h))
                f_hat = f + self.rng.normal(0.0, sigma)        # the noisy freshness PERCEPT
                if f_hat >= cfg.residue_eat_threshold:
                    eaten = self.world.consume(new_pos, deficit)
                    ph.energy += eaten
                    ph.resource_eaten += eaten
                    self.world.residue[new_pos] += eaten * cfg.residue_yield
                    if f < cfg.residue_fp_threshold:
                        ph.energy -= cfg.residue_loss          # false positive: ate residue
                        ph.fp_count += 1
                    else:
                        ph.tp_count += 1                       # true positive: ate fresh
                else:
                    if f >= cfg.residue_fp_threshold:
                        ph.fn_count += 1                       # false negative: skipped fresh
                    else:
                        ph.tn_count += 1                       # true negative: skipped residue
        elif cfg.enable_niche and self.world.class_phase is not None:
            # Exp 206: rotating-class niche. The eat step is fully h-BLIND (no rng here): h's
            # value is realized in ROUTING (creature.choose_action steers toward the under-crowded
            # class via an h-keyed noisy read of neighbour classes). Here we only apply the
            # frequency-dependent CROWDING discount on the cell's TRUE current class.
            # ANTI-CHEAT: intake = the unchanged consume(); the discount divisor is a creature-COUNT
            # on j_true via the FROZEN previous-step occupancy snapshot — never c_read, never h.
            K = self.world.niche_classes
            j_true = int(K * float(self.world.class_signal[new_pos]))
            if j_true >= K:                                    # guard the frac==1.0 edge
                j_true = K - 1
            self.world.class_occ_cur[j_true] += 1              # occupancy (every alive creature counts)
            if c.policy.niche_occ is None:
                c.policy.niche_occ = [0] * K
            c.policy.niche_occ[j_true] += 1                    # per-creature class telemetry (modal class)
            if deficit > 0:
                avail = self.world.resource_at(new_pos)
                want = min(deficit, avail)
                occ = float(self.world.class_occ_prev[j_true])  # frozen prev-step count of this class
                kept = want / (1.0 + cfg.niche_crowding * occ)  # h-blind crowding discount
                self.world.consume(new_pos, kept)               # deplete only the realized intake
                ph.energy += kept
                ph.resource_eaten += kept
        else:
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
                # Child placement: lowest-indexed neighbor (deterministic), or parent cell.
                # Exp 235: when terrain is ON and movement gating is ON, guard the spawn so
                # a newborn is NOT placed across an unclimbable rim (spurious extinction is
                # not a gradient result).  Use world.neighbors() directly (not
                # climbable_neighbors — spawn placement is deterministic, never probabilistic).
                if cfg.enable_terrain and cfg.terrain_gates_movement and self.world.elevation is not None:
                    all_nb = self.world.neighbors(new_pos)
                    elev_here = float(self.world.elevation[new_pos])
                    # Allow placement on basin cells (elevation <= 0.0) or same-level;
                    # never place a newborn on the ridge (1.0) or plateau (1.5) from basin.
                    safe_nb = [n for n in all_nb if float(self.world.elevation[n]) <= elev_here + 0.01]
                    neighbors = safe_nb if safe_nb else all_nb
                else:
                    neighbors = self.world.neighbors(new_pos)
                child_pos = min(neighbors) if neighbors else new_pos

                child_geno = mutate(c.genotype, self.rng, cfg.mutation_rate,
                                    mutate_thermosense=cfg.enable_thermosense,
                                    freeze_learning_rate=cfg.freeze_learning_rate,
                                    freeze_thermosense=cfg.freeze_thermosense,
                                    mutate_memory=cfg.enable_hidden_mode,
                                    mutate_active_sensing=cfg.enable_active_sensing,
                                    mutate_locomotion=cfg.enable_terrain,
                                    mutate_continuous_locomotion=(cfg.enable_continuous_locomotion
                                                                  and not cfg.freeze_continuous_locomotion))
                child_ph = Phenotype(energy=transfer, age=0, pos=child_pos, birth_t=self.t)
                # Exp 238: set continuous pos for child near parent when ON.
                if cfg.enable_continuous_locomotion and ph.pos_cont is not None:
                    from ecology.continuous_world import ARENA_W, ARENA_H
                    _pr, _pc = divmod(child_pos, cfg.cols)
                    child_ph.pos_cont = (
                        ((_pc + 0.5) / cfg.cols) * ARENA_W,
                        ((_pr + 0.5) / cfg.rows) * ARENA_H,
                    )
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

        # PERF (memory; byte-identical): a dead creature's two n_cells belief maps (m, visit_t)
        # are NEVER read again — sense()/act() run only on ALIVE creatures, and the summary +
        # newborn analysis read only genotype + phenotype. Dropping the maps on death keeps the
        # heap from growing with total-ever-born — on a long high-turnover run ~75k dead
        # creatures × 2 maps is ~175 MB of otherwise-retained numpy buffers, the dominant
        # per-run RSS and the parallel-batch swap risk. The policy object + band_estimate (a
        # float) are KEPT for post-hoc inspection; events_hash + all results are unchanged.
        if not ph.alive and c.policy is not None:
            c.policy.release_maps()

        # 7c. Exp 243 Mechanism A: global density-dependent crowding mortality.
        #     Gated; OFF path makes ZERO rng draws and emits no event. Only rolled if the
        #     creature is still alive this step (starvation/senescence take precedence).
        if cfg.enable_density_mortality and ph.alive:
            p = _density_mortality_p(
                self._dm_N_frozen, cfg.density_mortality_hmax, cfg.density_mortality_Kc,
                cfg.density_mortality_theta, cfg.density_mortality_rate_scale,
                getattr(self, "_dm_dN", 0.0))
            if p > 0.0 and self.rng.random() < p:
                ph.alive = False
                ph.cause_of_death = "crowding"
                self.events.append(self._event("death", c, details={
                    "cause": "crowding",
                    "age": ph.age,
                    "offspring_count": ph.offspring_count,
                }))
                if c.policy is not None:
                    c.policy.release_maps()

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

        # Exp 206: recompute the ROTATING class signal + reset this-step occupancy BEFORE
        # processing. Gated on enable_niche ⇒ OFF path never runs (byte-identical, no rng).
        # class_signal[pos] = frac(class_phase[pos] + omega(t)), omega = frac(t*niche_rotation):
        # a cell's true class j(pos,t)=floor(K*class_signal[pos]) ROTATES over time, so a static
        # learned map cannot memorise it (the non-memorizability fix). class_occ_prev (last step's
        # per-class occupancy) is FROZEN here and read by the discount + routing this step; the
        # eat step fills class_occ_cur, which becomes prev after the loop (deterministic,
        # order-independent ⇒ no eat-first asymmetry under shuffle). No rng; not in events_hash.
        if self.cfg.enable_niche and self.world.class_phase is not None:
            omega = (self.t * self.cfg.niche_rotation) % 1.0
            self.world.class_signal = np.mod(self.world.class_phase + omega, 1.0)
            self.world.class_occ_cur = np.zeros(self.world.niche_classes, dtype=np.int64)

        # Exp hidden-state-mode: stochastic hidden-mode flip — ONE gated rng draw per step.
        # OFF path (enable_hidden_mode=False) makes NO rng draw ⇒ byte-identical to Exp 194-206.
        if self.cfg.enable_hidden_mode:
            if self.rng.random() < self.cfg.mode_switch_prob:
                self.world.hidden_mode = 1 - self.world.hidden_mode

        # Exp 243 Mechanism A: freeze the global head-count ONCE per step, before the
        # per-creature loop and before any shuffle, so every alive creature sees the SAME N.
        if self.cfg.enable_density_mortality:
            _N_now = self.alive_count()
            if not hasattr(self, "_dm_N_prev"):
                self._dm_N_prev = _N_now          # t=0: derivative term is 0
            self._dm_dN = _N_now - self._dm_N_prev
            self._dm_N_frozen = _N_now
            self._dm_N_prev = _N_now

        # Process alive creatures. OFF path (shuffle_creature_order=False) iterates in
        # ascending creature_id order — BYTE-IDENTICAL to Exp 194-201. Exp 202: when
        # shuffle_creature_order is True, a RANDOMISED order each step (rng.shuffle, drawing
        # ONLY in this gated ON branch) neutralises the id-order eat-first confound so that
        # interference competition at a contested cell is keyed to navigation skill, not birth
        # order. consume() is UNCHANGED.
        #
        # PERF (byte-identical): only the shuffle path needs a private, mutable copy to reorder.
        # The OFF path iterates the maintained _alive_list DIRECTLY — _step_one_creature never
        # mutates _alive_list (children go to pending_children; the list is rebuilt only BELOW,
        # after this loop and the band-strip read both finish), so aliasing it here is safe and
        # saves an O(alive) list copy every step. The visited creatures and their order are
        # identical to the old `order = self._alive()` copy, so events_hash is unchanged
        # (guarded by the committed-hash regression tests + tests/test_perf_optimizations.py).
        if self.cfg.shuffle_creature_order:
            order = self.alive_snapshot()
            self.rng.shuffle(order)
        else:
            order = self._alive_list

        # Exp 202: band-strip validity instrumentation (gated; NO rng draw, NOT in events_hash) —
        # the go/no-go that the food band is GENUINELY depleted within a step (exp201: strip~0).
        _band_before = None
        if self.cfg.track_band_strip and self.world.temperature is not None and self.world.enable_food_coupling:
            _mask = np.abs(self.world.temperature - self.world.current_food_optimal) <= self.world.food_band_width
            _band_before = float(np.sum(self.world.resource[_mask]))

        for c in order:
            self._step_one_creature(c, pending_children)

        # Exp 206: freeze this step's occupancy as next step's crowding snapshot (no rng,
        # not in events_hash). OFF path never runs ⇒ byte-identical.
        if self.cfg.enable_niche and self.world.class_occ_cur is not None:
            self.world.class_occ_prev = self.world.class_occ_cur

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
        # Exp 238: continuous world regen (ON-branch only; OFF path never reaches this).
        if self.cont_world is not None:
            self.cont_world.step_regen()

        # Exp 204: residue decay (ON-branch only; no rng, not in events_hash). Placed here
        # rather than inside step_regen() so it runs regardless of which step_regen() path
        # (food-coupling vs plain) executed. OFF path never touches residue → byte-identical.
        if self.cfg.enable_residue and self.world.residue is not None:
            self.world.residue *= (1.0 - self.cfg.residue_decay)

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
        # alive_count() is O(1) — same value as the old full scan of self._creatures.
        n_alive = self.alive_count()
        if n_alive > self.cfg.max_population:
            self.exploded = True
            self.events.append({
                "t": self.t,
                "event_type": "explosion",
                "alive_count": n_alive,
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
            if not self.has_alive():         # extinction (O(1), was a full self._creatures scan)
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
