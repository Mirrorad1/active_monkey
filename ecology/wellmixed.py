"""ecology/wellmixed.py — Well-mixed (mean-field), individual-based, DETERMINISTIC
predator-prey + evolution substrate.

Exp 255 diagnosis test: does a logistic-prey + logistic-predator ecology COEXIST
stably when well-mixed (no spatial encounter stochasticity)?  The spatial-agent
substrate (Exp 238–247) collapsed; here the null hypothesis is overturned — a
well-mixed Bazykin / Type-II functional-response system coexists.

ALSO supports an invasion-from-rarity test of a prey escape-speed trait.

Design principles
-----------------
- Deterministic given (cfg, seed): single seeded numpy.random.default_rng(seed).
- Creatures processed in ascending ID order every step (no set-iteration).
- events_hash: SHA-256 of canonical JSON of per-step summary list.
- No silent failures: any explosion / early termination is flagged in the result dict.
- Self-contained: does NOT touch ecology/engine.py or any spatial-engine golden.

RNG discipline
--------------
One ``numpy.random.default_rng(seed)`` drives all decisions in strict ID-ascending
order.  Per step, draws are: (1) prey births, (2) predation kills + kill-attribution
(one extra draw per kill to attribute to a predator), (3) predator births,
(4) predator deaths.  Within each phase, creatures are processed in ascending cid order.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WellMixedConfig:
    # ---- Prey ----
    r_prey: float = 0.6          # intrinsic per-capita birth rate
    K_prey: float = 300.0        # prey carrying capacity
    escape_cost: float = 0.15    # fecundity cost per unit escape_speed above baseline
    escape_baseline: float = 1.0 # escape_speed at which no cost is paid

    # ---- Predation (Type II, escape-keyed) ----
    attack_a: float = 0.02       # baseline attack rate
    handling_h: float = 0.02     # handling time (Holling Type II)
    escape_k: float = 1.0        # how strongly escape reduces vulnerability

    # ---- Predator ----
    assimilation: float = 0.5
    pred_birth_per_capture: float = 0.35  # numerical-response: birth prob per assimilated capture
    pred_base_mortality: float = 0.05
    pred_self_limit_hmax: float = 0.15    # self-limitation at K_pred
    K_pred: float = 40.0                  # predator self-limit scale

    # ---- Evolution ----
    mutation_rate: float = 0.0
    mutation_sd: float = 0.05
    freeze_prey_trait: bool = False       # True = prey trait never changes (static arm)
    freeze_predator_trait: bool = False   # True = predator trait never changes (static arm)

    # ---- Trait bounds ----
    trait_min: float = 0.0
    trait_max: float = 4.0

    # ---- Run ----
    horizon: int = 1500
    n_prey0: int = 100
    n_pred0: int = 20
    prey_escape0: float = 1.0    # starting prey escape_speed trait
    pred_attack0: float = 1.0    # starting predator attack trait

    # ---- Safety ----
    pop_cap: int = 50_000        # explosion guard


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

@dataclass
class Critter:
    """Minimal individual.  trait = escape_speed (prey) or attack_scale (predator)."""
    role: str   # "prey" or "pred"
    trait: float
    cid: int


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class WellMixedSim:
    """Well-mixed individual-based predator-prey simulation."""

    def __init__(self, cfg: WellMixedConfig, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._next_cid = 0
        self.t = 0

        # Build initial populations
        self.prey: List[Critter] = []
        self.predators: List[Critter] = []

        for _ in range(cfg.n_prey0):
            self.prey.append(Critter("prey", cfg.prey_escape0, self._next_cid))
            self._next_cid += 1

        for _ in range(cfg.n_pred0):
            self.predators.append(Critter("pred", cfg.pred_attack0, self._next_cid))
            self._next_cid += 1

    # ------------------------------------------------------------------
    def _clamp_trait(self, v: float) -> float:
        return max(self.cfg.trait_min, min(self.cfg.trait_max, v))

    def _mutate_trait(self, parent_trait: float) -> float:
        """Apply mutation if mutation_rate triggers; clamp to bounds."""
        if self.cfg.mutation_rate > 0.0 and self.rng.random() < self.cfg.mutation_rate:
            delta = self.rng.normal(0.0, self.cfg.mutation_sd)
            return self._clamp_trait(parent_trait + delta)
        return parent_trait

    # ------------------------------------------------------------------
    def step(self) -> dict:
        """Advance one time step.  Returns per-step summary dict."""
        cfg = self.cfg
        rng = self.rng

        # (a) Snapshot population sizes
        N_prey = len(self.prey)
        N_pred = len(self.predators)

        # -- Accumulators --
        births_prey = 0
        deaths_prey_pred = 0
        births_pred = 0
        deaths_pred = 0

        # (b) Prey logistic births
        new_prey_children: List[Critter] = []
        for p in self.prey:  # already in ascending cid order (maintained by append)
            birth_p = cfg.r_prey * max(0.0, 1.0 - N_prey / cfg.K_prey)
            # fecundity cost for escape above baseline
            excess = max(0.0, p.trait - cfg.escape_baseline)
            birth_p *= max(0.0, 1.0 - cfg.escape_cost * excess)
            birth_p = max(0.0, birth_p)
            if rng.random() < birth_p:
                # Child trait: parent + mutation (unless frozen)
                if cfg.freeze_prey_trait:
                    child_trait = p.trait
                else:
                    child_trait = self._mutate_trait(p.trait)
                new_prey_children.append(Critter("prey", child_trait, self._next_cid))
                self._next_cid += 1
                births_prey += 1

        # (c) Predation — Type II, INDIVIDUAL predator attack (selectable).
        #   sat = 1 / (1 + a*h*N_prey)  (shared saturation — depends only on prey density)
        #   For each prey i, each predator j contributes hazard:
        #     v_ij = 1 / (1 + escape_k * max(0, prey_i.trait - pred_j.trait))
        #     c_ij = attack_a * sat * v_ij
        #   total hazard on prey i = sum_j(c_ij)
        #   kill_prob_i = 1 - exp(-total_haz_i)
        #   If prey i is killed, the kill is attributed to predator j with prob c_ij / total_haz.
        #   This gives higher-attack predators MORE captures -> individual selection on attack.
        sat = 1.0 / (1.0 + cfg.attack_a * cfg.handling_h * N_prey) if N_prey > 0 else 1.0

        dead_prey_mask = [False] * N_prey
        pred_captures = [0] * N_pred          # per-predator capture count THIS step
        captures_this_step = 0
        for i, p in enumerate(self.prey):
            # Compute each predator's hazard contribution on prey i
            contribs: List[float] = []
            total_haz = 0.0
            for q in self.predators:
                v_ij = 1.0 / (1.0 + cfg.escape_k * max(0.0, p.trait - q.trait))
                c = cfg.attack_a * sat * v_ij
                contribs.append(c)
                total_haz += c
            kill_prob = 1.0 - math.exp(-total_haz)
            if rng.random() < kill_prob:
                dead_prey_mask[i] = True
                deaths_prey_pred += 1
                captures_this_step += 1
                # Attribute kill to ONE predator, weighted by hazard contribution.
                # Higher-attack predators win more kills -> individual selection on attack.
                if total_haz > 0.0:
                    r = rng.random() * total_haz
                    cum = 0.0
                    for j, c in enumerate(contribs):
                        cum += c
                        if r <= cum:
                            pred_captures[j] += 1
                            break
                    else:
                        pred_captures[-1] += 1

        # (d) Predator numerical response (INDIVIDUAL captures) + self-limit
        dead_pred_mask = [False] * N_pred
        new_pred_children: List[Critter] = []
        for j, q in enumerate(self.predators):
            # Birth draw — uses THIS predator's OWN captures (individual selection)
            birth_p_pred = cfg.pred_birth_per_capture * cfg.assimilation * pred_captures[j]
            birth_p_pred = min(1.0, max(0.0, birth_p_pred))
            if rng.random() < birth_p_pred:
                if cfg.freeze_predator_trait:
                    child_trait = q.trait
                else:
                    child_trait = self._mutate_trait(q.trait)
                new_pred_children.append(Critter("pred", child_trait, self._next_cid))
                self._next_cid += 1
                births_pred += 1
            # Death draw (after birth — order: birth then death)
            death_p_pred = cfg.pred_base_mortality + cfg.pred_self_limit_hmax * (N_pred / cfg.K_pred)
            death_p_pred = min(1.0, max(0.0, death_p_pred))
            if rng.random() < death_p_pred:
                dead_pred_mask[j] = True
                deaths_pred += 1

        # (e) Apply deaths, append children
        self.prey = [p for i, p in enumerate(self.prey) if not dead_prey_mask[i]]
        self.prey.extend(new_prey_children)

        self.predators = [q for j, q in enumerate(self.predators) if not dead_pred_mask[j]]
        self.predators.extend(new_pred_children)

        self.t += 1

        return {
            "t": self.t,
            "n_prey": len(self.prey),
            "n_pred": len(self.predators),
            "births_prey": births_prey,
            "deaths_prey_pred": deaths_prey_pred,
            "births_pred": births_pred,
            "deaths_pred": deaths_pred,
        }

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Run to horizon or extinction/explosion.  Returns result dict."""
        cfg = self.cfg

        prey_series: List[int] = [len(self.prey)]
        pred_series: List[int] = [len(self.predators)]
        prey_trait_mean_series: List[float] = [
            float(np.mean([p.trait for p in self.prey])) if self.prey else 0.0
        ]
        pred_trait_mean_series: List[float] = [
            float(np.mean([q.trait for q in self.predators])) if self.predators else 0.0
        ]

        step_summaries: List[dict] = []
        exploded = False

        while self.t < cfg.horizon:
            summary = self.step()
            step_summaries.append(summary)

            n_prey = len(self.prey)
            n_pred = len(self.predators)

            prey_series.append(n_prey)
            pred_series.append(n_pred)
            prey_trait_mean_series.append(
                float(np.mean([p.trait for p in self.prey])) if self.prey else 0.0
            )
            pred_trait_mean_series.append(
                float(np.mean([q.trait for q in self.predators])) if self.predators else 0.0
            )

            # Stop conditions
            if n_prey == 0 and n_pred == 0:
                break
            if n_prey == 0 or n_pred == 0:
                # One population extinct; let the other run to horizon but
                # mark extinction so the caller can detect it.
                # Actually just break early to signal
                break
            if n_prey + n_pred > cfg.pop_cap:
                exploded = True
                break

        # Compute events_hash over the canonical step summary list
        canonical = json.dumps(step_summaries, separators=(",", ":"), sort_keys=True)
        events_hash = hashlib.sha256(canonical.encode()).hexdigest()

        extinct = (len(self.prey) == 0 or len(self.predators) == 0)

        return {
            "events_hash": events_hash,
            "prey_series": prey_series,
            "pred_series": pred_series,
            "prey_trait_mean_series": prey_trait_mean_series,
            "pred_trait_mean_series": pred_trait_mean_series,
            "t_end": self.t,
            "extinct": extinct,
            "exploded": exploded,
        }
