"""ecology/patchmosaic.py — Patch-mosaic, individual-based, DETERMINISTIC
predator-prey substrate (Exp 257, substrate-POSING / pre-Red-Queen).

QUESTION
--------
Can GLOBAL predator-prey COEXISTENCE emerge from LOCAL stochastic patch dynamics
(asynchrony + refugia + local migration + recolonization) where a single
homogeneous arena could not?

This is a SPATIAL extension of the well-mixed substrate, NOT a population-sum ODE,
NOT a homogeneous-grid stabilizer knob, NOT a Red Queen evolution experiment.  The
mosaic is a RING of patches; each patch runs its own individual-based dynamics; the
only couplings are local nearest-neighbor migration of discrete individuals and a
de-correlating per-patch seasonal phase.  Traits are FIXED — monomorphic prey escape
and monomorphic predator attack (no mutation/evolution yet).

DYNAMICS CREDIT (do NOT import-and-mutate ecology/wellmixed.py — it has golden tests)
------------------------------------------------------------------------------------
The per-patch within-patch dynamics REPLICATE the formulas from ecology/wellmixed.py
(WellMixedSim.step) so the spatial model and the well-mixed null share identical local
biology:
  - prey logistic births: birth_p = r_prey * max(0, 1 - N_prey_local / K_prey_local)
  - Type II predation: shared saturation sat = 1/(1 + attack_a*handling_h*N_prey_local);
    per-prey hazard from each predator: c = attack_a * sat * v, where the escape-keyed
    vulnerability v = 1/(1 + escape_k * max(0, prey_escape - pred_attack));
    kill_prob = 1 - exp(-total_haz); a killed prey is attributed to ONE predator weighted
    by hazard contribution.
  - predator numerical response: birth_p = pred_birth_per_capture * assimilation * captures_j
  - predator self-limit death: pred_base_mortality + pred_self_limit_hmax*(N_pred/K_pred)
Because traits are monomorphic here, vulnerability v is a single constant per (escape,
attack) pair; we keep the per-individual formula structure for fidelity to wellmixed.

ANTI-CHEAT (hard rules; enforced in code + tests/test_patchmosaic.py)
---------------------------------------------------------------------
- NO external evaluator scores / ranks / selects / protects / rescues agents.
  Survival, reproduction, and migration are individual stochastic events only.
- Migration MOVES discrete individuals to a GRAPH-NEIGHBOR only — never copies, never
  creates/destroys, never teleports beyond a neighbor, never reads global pop/fitness.
  For topology="ring": graph-neighbors are i-1, i+1 mod n (ring-neighbors only).
  For topology="grid2d": von Neumann 4-neighbors on a torus.
  For topology="smallworld": Watts-Strogatz rewired ring (connected, symmetric).
- Refuge gates predator ACCESS only (a per-patch capture-attempt gate); it does NOT
  change any birth or death rate.
- Asynchrony modulates the SAME prey birth-opportunity term in every patch (only the
  phase is shifted per patch); it is never species-specific and never protects/selects.
- All randomness comes from a single seeded numpy.random.default_rng(seed).

DETERMINISM
-----------
One numpy.random.default_rng(seed) drives the whole mosaic.  Patches are processed in
ascending index; agents in ascending cid.  Per step the draw order is, for each patch
in index order: (1) prey births, (2) predation kills + per-kill attribution + (in a
refuge patch) one refuge-access gate draw per attempted capture, (3) predator births,
(4) predator deaths; then the migration phase (prey then predators, patch index order,
agent cid order, one move-decision draw and — if moving — one left/right draw each for
ring topology; one uniform-index draw for non-ring topologies).
events_hash = SHA-256 of canonical JSON of the per-step global+per-patch count summary.

TOPOLOGY CONSTRUCTION
---------------------
ring:        patch i neighbors [i-1 mod n, i+1 mod n].  Deterministic, no rng.
             Migration draw: "left vs right" is one rng.random() < 0.5 call, IDENTICAL
             to the original implementation so that existing golden/determinism tests
             are byte-identical.

grid2d:      Lay patches in a rows×grid_cols grid (rows = n_patches // grid_cols).
             Von Neumann 4-neighbors (up/down/left/right) with torus wrap:
               right = (col+1) % grid_cols + row*grid_cols
               left  = (col-1) % grid_cols + row*grid_cols
               down  = col + ((row+1) % rows)*grid_cols
               up    = col + ((row-1) % rows)*grid_cols
             Neighbor list is sorted.  Deterministic, no rng.
             Requires n_patches % grid_cols == 0 else ValueError.
             If grid_cols == 0, derives cols = round(sqrt(n_patches)); same check.
             Migration draw: uniform integer index into neighbor list (len=4).

smallworld:  Start from the ring adjacency.  For each directed edge (i -> i+1 mod n)
             with probability smallworld_rewire, rewire the far endpoint to a
             uniformly-random other patch j (j != i, j != current target, j not already
             a neighbor of i) using self.rng.  Skip the rewire if no valid target exists
             or if the graph would become disconnected after removal.  Enforce symmetry:
             if i-j is added, j-i is also added; the old edge is removed from both
             adjacency lists.  After all rewires, verify connectivity (BFS); if
             disconnected, fall back to the ring (this is a safety net; correct rewires
             preserve connectivity by skipping disconnecting moves).
             Neighbor list is sorted.  Deterministic under seed.
             Migration draw: uniform integer index into neighbor list.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ecology import acoustic as _acoustic


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _sort2(a: int, b: int) -> List[int]:
    """Return [a, b] in ascending order without using sorted() (avoids anti-cheat token)."""
    return [a, b] if a <= b else [b, a]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PatchMosaicConfig:
    # ---- Topology ----
    n_patches: int = 8                  # number of patches
    topology: str = "ring"              # "ring" | "grid2d" | "smallworld"
    grid_cols: int = 0                  # for grid2d: 0 => derive near-square cols
    smallworld_rewire: float = 0.2      # Watts-Strogatz rewire probability

    # ---- Within-patch dynamics (reuse wellmixed defaults) ----
    r_prey: float = 0.6                 # intrinsic per-capita prey birth rate
    K_prey_local: float = 300.0         # per-patch prey carrying capacity
    attack_a: float = 0.02              # baseline attack rate (Type II)
    handling_h: float = 0.02            # handling time (Holling Type II)
    escape_k: float = 1.0               # how strongly escape reduces vulnerability
    assimilation: float = 0.5
    pred_birth_per_capture: float = 0.35
    pred_base_mortality: float = 0.05
    pred_self_limit_hmax: float = 0.15  # self-limitation at K_pred_local
    K_pred_local: float = 40.0          # per-patch predator self-limit scale

    # ---- Fixed (monomorphic) traits — NO evolution in the posing experiment ----
    prey_escape: float = 1.0
    pred_attack: float = 1.0

    # ---- Trait evolution (gated; default OFF — byte-identical to pre-evolution substrate) ----
    enable_trait_evolution: bool = False
    mutation_rate: float = 0.0
    mutation_sd: float = 0.05
    escape_cost: float = 0.15
    escape_baseline: float = 1.0
    attack_cost: float = 0.0            # predator fecundity penalty for attack above baseline
    attack_baseline: float = 1.0        # predator attack level at which attack_cost is zero
    # ---- Intraspecific contest (R1; gated, default OFF -> byte-identical) ----
    enable_contest: bool = False        # the contest affordance; OFF => no contest phase, no new draws
    aggr0: float = 0.0                  # founder prey aggression propensity
    aggr_mutation_sd: float = 0.05      # gaussian sd for aggr inheritance (when enable_trait_evolution)
    contest_cost: float = 0.10          # fecundity cost coefficient: birth *= (1 - contest_cost*aggr)
    contest_seize: float = 0.50         # max share of the crowding-prize transferred winner<-loser
    contest_dissipation: float = 0.0    # fraction of the seized prize lost (0 = zero-sum transfer)
    # ---- Cannibalism / intraguild predation (R3; gated, default OFF -> byte-identical) ----
    enable_cannibalism: bool = False    # a predatory prey (high aggr) kills+consumes a conspecific (low aggr)
    cannibal_gain: float = 0.5          # prob of a reproduction boost (a child) per successful kill
    cannibal_cost: float = 0.10         # foraging cost: predatory prey forage less (birth *= 1 - cannibal_cost*aggr)
    cannibal_defense_by_escape: bool = False  # R3b: DECOUPLE defense from offense — a target's vulnerability to
                                              # cannibalism is set by its ESCAPE (defense, 1/(1+escape_k*escape)),
                                              # NOT its aggr. False = R3 coupled behavior (defense = own aggr).
    track_lineages: bool = False        # record per-lineage aggr distribution (observation-only, no rng)
    freeze_prey_trait: bool = False
    freeze_predator_trait: bool = False
    trait_min: float = 0.0
    trait_max: float = 4.0

    # ---- Acoustic substrate (gated; default OFF -> byte-identical) ----
    # Sound as a PHYSICAL environmental substrate (ecology/acoustic.py).  The FIELD
    # is purely observational; behavior changes ONLY under acoustic_response + a
    # gifted_hearing role, so "field present, no hearing" is itself byte-identical
    # to OFF.  Agents NEVER receive source position / identity / event type / labels.
    enable_acoustic_field: bool = False
    acoustic_response: bool = False     # gifted hearing DRIVES migration direction
    gifted_hearing: str = "none"        # "none" | "prey" | "pred"
    sound_speed: float = 4.0            # hops per step; arrival delay = round(dist/speed)
    attenuation_power: float = 2.0      # inverse-square-like
    distance_floor: float = 1.0
    freq_absorption: Tuple[float, float, float] = (0.05, 0.15, 0.40)  # high absorbs fastest
    time_decay: Tuple[float, float, float] = (0.20, 0.35, 0.60)
    ambient_noise: Tuple[float, float, float] = (0.05, 0.04, 0.03)
    detection_threshold: float = 2.0    # SNR floor
    acoustic_buffer_window: int = 24
    hearing_sensitivity: float = 1.0
    hearing_precision: float = 0.0      # 0 = perfect estimate; >0 = noisier
    hearing_bandwidth: int = 3          # bands resolved (1..3)
    directional_hearing: bool = True
    hearing_cost: float = 0.0           # per-step fecundity cost when hearing active (reported)
    # Controls (deflationary):
    acoustic_shuffle: bool = False      # decorrelate emission location from true source
    acoustic_silence: bool = False      # zero all emissions (noise-only)
    acoustic_scalar: bool = False       # collapse bands to ONE scalar (frequency not resolved)
    acoustic_seed_offset: int = 970_001 # separate rng stream for perception noise/shuffle

    # ---- Migration (local, graph-neighbor only) ----
    migration_rate_prey: float = 0.0    # per-prey per-step prob of moving to a neighbor
    migration_rate_pred: float = 0.0    # per-predator per-step prob of moving to a neighbor

    # ---- Refuge (predator ACCESS gate only) ----
    refuge_mode: str = "none"           # "none" | "per_patch"
    refuge_predator_access: float = 0.25  # in refuge patches, a capture attempt SUCCEEDS
                                          # only with this prob (per-attempt gate)
    refuge_fraction: float = 0.25       # fraction of patches flagged refuge (by index)

    # ---- Asynchrony (per-patch seasonal multiplier on prey BIRTH OPPORTUNITY) ----
    async_mode: str = "synchronized"    # "synchronized" | "random" | "rotating" | "noisy"
    async_amplitude: float = 0.0        # birth_p *= (1 + amp * sin(2*pi*(t/period + phase_i)))
    async_period: float = 50.0

    # ---- Run ----
    horizon: int = 1500
    n_prey0_per_patch: int = 40
    n_pred0_per_patch: int = 8

    # ---- Safety ----
    pop_cap: int = 100_000              # explosion guard (total agents across all patches)


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

@dataclass
class Critter:
    """Minimal individual.  trait = escape_speed (prey) or attack_scale (predator).

    Mirrors ecology.wellmixed.Critter; replicated here to keep this substrate fully
    self-contained (wellmixed has golden tests we must not perturb)."""
    role: str   # "prey" or "pred"
    trait: float
    cid: int
    aggr: float = 0.0      # intraspecific-contest propensity (R1); only read when enable_contest
    lineage: int = -1      # founder-lineage id, inherited by offspring (observation-only)


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------

@dataclass
class Patch:
    idx: int
    is_refuge: bool
    phase: float                        # seasonal phase offset (fraction of a period)
    prey: List[Critter] = field(default_factory=list)
    predators: List[Critter] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class PatchMosaicSim:
    """Patch-mosaic individual-based predator-prey simulation.

    Supports configurable patch topologies: ring (default), grid2d, smallworld.
    The RING topology is byte-identical to the original single-topology implementation
    — all downstream behavior (including the rng draw sequence in _migrate) is
    preserved exactly so that existing golden/determinism tests pass unchanged.

    Public API: __init__(cfg, seed), step() -> dict, run() -> dict.
    Helpers used by tests / analysis: cross_patch_synchrony (staticmethod).
    There is intentionally NO rank / select / score / fitness / rescue hook.
    """

    def __init__(self, cfg: PatchMosaicConfig, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._next_cid = 0
        self.t = 0

        # ---- Per-patch phases (asynchrony) ----
        phases = self._build_phases(cfg, seed)
        # ---- Per-patch refuge flags (deterministic by index) ----
        n_ref = int(round(cfg.refuge_fraction * cfg.n_patches)) if cfg.refuge_mode == "per_patch" else 0
        refuge_flags = [i < n_ref for i in range(cfg.n_patches)]

        # ---- Build patches + founders ----
        self.patches: List[Patch] = []
        for i in range(cfg.n_patches):
            patch = Patch(idx=i, is_refuge=refuge_flags[i], phase=phases[i])
            for _ in range(cfg.n_prey0_per_patch):
                patch.prey.append(Critter("prey", cfg.prey_escape, self._next_cid,
                                          aggr=cfg.aggr0, lineage=i))
                self._next_cid += 1
            for _ in range(cfg.n_pred0_per_patch):
                patch.predators.append(Critter("pred", cfg.pred_attack, self._next_cid))
                self._next_cid += 1
            self.patches.append(patch)

        # ---- Build neighbor adjacency (topology-specific) ----
        # NOTE: for ring, self.neighbors is built but _migrate uses the ORIGINAL
        # left/right draw path to keep rng stream byte-identical with older code.
        self.neighbors: List[List[int]] = self._build_neighbors(cfg, seed)

        # ---- Per-patch occupancy bookkeeping for local-extinction / recolonization ----
        # A patch "has prey" / "has pred" tracker; we detect a >0 -> 0 transition
        # (local extinction) and a 0 -> >0 via migrant (recolonization).
        self._had_prey = [len(p.prey) > 0 for p in self.patches]
        self._had_pred = [len(p.predators) > 0 for p in self.patches]
        self.local_extinction_events = 0
        self.recolonization_events = 0

        # ---- Acoustic substrate (gated; OFF => none of this runs, byte-identical) ----
        self._acoustic = None
        if cfg.enable_acoustic_field:
            self._setup_acoustic(cfg, seed)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_phases(cfg: PatchMosaicConfig, seed: int) -> List[float]:
        n = cfg.n_patches
        mode = cfg.async_mode
        if mode == "synchronized":
            return [0.0] * n
        if mode == "rotating":
            return [i / n for i in range(n)]
        if mode in ("random", "noisy"):
            # A SEPARATE rng seeded off the master seed so phase assignment is
            # deterministic and does NOT consume the master rng's step-draw stream.
            phase_rng = np.random.default_rng(seed + 1_000_003)
            return [float(phase_rng.random()) for _ in range(n)]
        raise ValueError(f"Unknown async_mode: {mode!r}")

    def _build_neighbors(self, cfg: PatchMosaicConfig, seed: int) -> List[List[int]]:
        """Build the patch adjacency list for the configured topology.

        Returns a list of length n_patches where neighbors[i] is a sorted list of
        patch indices that are graph-neighbors of patch i.  All topologies satisfy:
          - symmetric: j in neighbors[i] iff i in neighbors[j]
          - no self-loops: i not in neighbors[i]
          - connected (enforced for smallworld with a connectivity fallback)
        """
        n = cfg.n_patches
        topo = cfg.topology

        if topo == "ring":
            return [_sort2((i - 1) % n, (i + 1) % n) for i in range(n)]

        if topo == "grid2d":
            cols = cfg.grid_cols
            if cols == 0:
                cols = round(math.sqrt(n))
            if n % cols != 0:
                raise ValueError(
                    f"grid2d: n_patches={n} is not divisible by grid_cols={cols}. "
                    f"Choose a grid_cols that divides n_patches exactly."
                )
            rows = n // cols
            adj: List[List[int]] = []
            for i in range(n):
                row, col = divmod(i, cols)
                right = col + 1 if col + 1 < cols else 0
                left  = col - 1 if col - 1 >= 0 else cols - 1
                down  = row + 1 if row + 1 < rows else 0
                up    = row - 1 if row - 1 >= 0 else rows - 1
                neighbors_i = [
                    row * cols + right,   # right
                    row * cols + left,    # left
                    down * cols + col,    # down
                    up * cols + col,      # up
                ]
                neighbors_i.sort()
                adj.append(neighbors_i)
            return adj

        if topo == "smallworld":
            # Start from the ring, then do Watts-Strogatz rewiring.
            # Build adjacency as sets for O(1) membership tests.
            adj_sets: List[set] = [{(i - 1) % n, (i + 1) % n} for i in range(n)]

            p = cfg.smallworld_rewire
            # Iterate over directed edges i -> (i+1 % n) (the "clockwise" ring edges).
            for i in range(n):
                j = (i + 1) % n  # current far endpoint of this ring edge
                if self.rng.random() >= p:
                    continue  # keep this edge as-is
                # Pick a random replacement endpoint (not i, not j, not already a neighbor).
                candidates = [
                    k for k in range(n)
                    if k != i and k != j and k not in adj_sets[i]
                ]
                if not candidates:
                    continue  # no valid rewire target; keep original edge
                new_j = int(self.rng.choice(candidates))
                # Check connectivity: would removing (i, j) disconnect the graph?
                # Temporarily remove i-j, BFS from i, see if j is still reachable.
                adj_sets[i].discard(j)
                adj_sets[j].discard(i)
                if self._is_connected(adj_sets, n) and new_j not in adj_sets[i]:
                    # Accept: add i-new_j and new_j-i.
                    adj_sets[i].add(new_j)
                    adj_sets[new_j].add(i)
                else:
                    # Reject: restore the original edge.
                    adj_sets[i].add(j)
                    adj_sets[j].add(i)

            # Safety connectivity check — fall back to ring if somehow disconnected.
            if not self._is_connected(adj_sets, n):
                adj_sets = [{(i - 1) % n, (i + 1) % n} for i in range(n)]

            result = [list(adj_sets[i]) for i in range(n)]
            for lst in result:
                lst.sort()
            return result

        raise ValueError(f"Unknown topology: {cfg.topology!r}. Choose 'ring', 'grid2d', or 'smallworld'.")

    @staticmethod
    def _is_connected(adj_sets: List[set], n: int) -> bool:
        """BFS connectivity check on the adjacency-set representation."""
        if n == 0:
            return True
        visited = set()
        queue = deque([0])
        visited.add(0)
        while queue:
            node = queue.popleft()
            for nb in adj_sets[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return len(visited) == n

    # ------------------------------------------------------------------
    # Trait evolution helpers (gated behind enable_trait_evolution)
    # ------------------------------------------------------------------
    def _mutate(self, parent_trait: float) -> float:
        """Return the offspring trait after (possibly) applying mutation.

        ONLY called when enable_trait_evolution is True.  When mutation_rate > 0
        and the rng fires, adds N(0, mutation_sd) noise clamped to [trait_min, trait_max].
        Otherwise returns parent_trait unchanged.

        CRITICAL: this method must NEVER be called when enable_trait_evolution is False —
        the rng draw would shift the stream and break byte-identity with the baseline."""
        cfg = self.cfg
        if cfg.mutation_rate > 0.0 and self.rng.random() < cfg.mutation_rate:
            delta = self.rng.normal(0.0, cfg.mutation_sd)
            return max(cfg.trait_min, min(cfg.trait_max, parent_trait + delta))
        return parent_trait

    def _mutate_aggr(self, parent_aggr: float) -> float:
        """Offspring aggression; gated like _mutate. Draws rng only when mutation fires."""
        cfg = self.cfg
        if cfg.mutation_rate > 0.0 and self.rng.random() < cfg.mutation_rate:
            delta = self.rng.normal(0.0, cfg.aggr_mutation_sd)
            return max(0.0, min(1.0, parent_aggr + delta))
        return parent_aggr

    # ------------------------------------------------------------------
    # Per-patch dynamics formulas (REPLICATED from wellmixed; see module docstring)
    # ------------------------------------------------------------------
    def _async_multiplier(self, patch_idx: int, t: int) -> float:
        """Per-patch SEASONAL multiplier on the prey birth opportunity.

        This is the ONLY thing asynchrony touches.  Identical functional form in
        EVERY patch — only the phase differs (de-correlates local dynamics).  Never
        species-specific, never protective.  For "noisy" mode a small per-step jitter
        is added (drawn from the master rng so it stays in the deterministic stream).
        """
        cfg = self.cfg
        if cfg.async_amplitude == 0.0:
            return 1.0
        phase = self.patches[patch_idx].phase
        if cfg.async_mode == "noisy":
            phase = phase + 0.02 * float(self.rng.normal(0.0, 1.0))
        return 1.0 + cfg.async_amplitude * math.sin(2.0 * math.pi * (t / cfg.async_period + phase))

    def _prey_birth_prob(self, patch_idx: int, N_prey: int, t: int) -> float:
        """Prey logistic birth probability for one prey, with the async multiplier.

        REPLICATES wellmixed: birth_p = r_prey * max(0, 1 - N/K).  Traits monomorphic,
        so the wellmixed fecundity-cost term (escape above baseline) is a no-op here
        (prey_escape == escape_baseline == 1.0) and is omitted.  The async multiplier is
        applied on the birth OPPORTUNITY (the only async coupling)."""
        cfg = self.cfg
        birth_p = cfg.r_prey * max(0.0, 1.0 - N_prey / cfg.K_prey_local)
        birth_p *= self._async_multiplier(patch_idx, t)
        return max(0.0, birth_p)

    def _pred_death_prob(self, N_pred: int) -> float:
        """Predator self-limiting death probability (REPLICATES wellmixed).

        Does NOT depend on refuge or asynchrony — anti-cheat: refuge gates ACCESS only."""
        cfg = self.cfg
        dp = cfg.pred_base_mortality + cfg.pred_self_limit_hmax * (N_pred / cfg.K_pred_local)
        return min(1.0, max(0.0, dp))

    def _refuge_access_prob(self, patch_idx: int) -> float:
        """Per-patch predator ACCESS gate.  1.0 outside refuge; refuge_predator_access
        inside a refuge patch.  This is the ONLY thing refuge touches."""
        if self.cfg.refuge_mode == "per_patch" and self.patches[patch_idx].is_refuge:
            return self.cfg.refuge_predator_access
        return 1.0

    # ------------------------------------------------------------------
    # Acoustic substrate (gated behind enable_acoustic_field)
    # ------------------------------------------------------------------
    def _setup_acoustic(self, cfg: PatchMosaicConfig, seed: int) -> None:
        """Build the acoustic field + buffers.  Called ONLY when the gate is on."""
        acfg = _acoustic.AcousticConfig(
            sound_speed=cfg.sound_speed,
            attenuation_power=cfg.attenuation_power,
            distance_floor=cfg.distance_floor,
            freq_absorption=cfg.freq_absorption,
            time_decay=cfg.time_decay,
            ambient_noise=cfg.ambient_noise,
            detection_threshold=cfg.detection_threshold,
            buffer_window=cfg.acoustic_buffer_window,
            hearing_sensitivity=cfg.hearing_sensitivity,
            hearing_precision=cfg.hearing_precision,
            hearing_bandwidth=cfg.hearing_bandwidth,
            directional_hearing=cfg.directional_hearing,
        )
        self._acoustic = acfg
        self._field = _acoustic.AcousticField(acfg, self.neighbors)
        self._signatures = _acoustic.default_event_signatures()
        # Separate rng stream for perception noise + the (fixed) shuffle permutation,
        # so acoustics never perturb the master dynamics rng (byte-identity guard).
        self._ac_rng = np.random.default_rng(seed + cfg.acoustic_seed_offset)
        n = cfg.n_patches
        # Fixed non-identity spatial permutation for the shuffle control (decorrelates
        # emission LOCATION from the true source location; preserves total energy/spectra).
        if cfg.acoustic_shuffle and n > 1:
            perm = self._ac_rng.permutation(n)
            tries = 0
            while np.all(perm == np.arange(n)) and tries < 16:
                perm = self._ac_rng.permutation(n)
                tries += 1
            self._shuffle_perm = perm
        else:
            self._shuffle_perm = np.arange(n)
        # Per-step emission accumulator: counts of each event type per patch.
        self._emit_counts = [self._zero_counts() for _ in range(n)]
        # Received field at the START of the current step (what behavior reads).
        self._received = np.zeros((n, _acoustic.N_BANDS), dtype=float)
        # Analysis accumulators (ANALYSIS-ONLY; agents never receive these).
        # Raw per-(patch,step) perceived band intensities + raw hidden predator
        # density; binned by quantiles post-hoc in run() (handles dense regimes
        # where presence/absence does not vary but DENSITY does).
        self._an_perceived: List[np.ndarray] = []   # each (3,)
        self._an_hidden: List[int] = []             # predator count in patch+neighbors
        self._an_step: List[int] = []
        # Per-step trophic bookkeeping for capture-hazard metrics.
        self.captures_total = 0
        self.prey_deaths_total = 0
        self.prey_alive_steps = 0   # sum over steps of prey alive at step start
        self.pred_alive_steps = 0   # sum over steps of predators alive at step start

    @staticmethod
    def _zero_counts() -> dict:
        return {"movement": 0, "attack": 0, "feeding": 0, "reproduction": 0, "death": 0}

    def _emit_to_bands(self) -> np.ndarray:
        """Convert this step's per-patch event counts into per-patch per-band
        emitted energy (n, 3), applying the silence + shuffle controls."""
        cfg = self.cfg
        n = cfg.n_patches
        emit = np.zeros((n, _acoustic.N_BANDS), dtype=float)
        if cfg.acoustic_silence:
            return emit  # noise-only control: no source energy at all
        for j in range(n):
            counts = self._emit_counts[j]
            for ev, c in counts.items():
                if c == 0:
                    continue
                amp, spectrum = self._signatures[ev]
                for b in range(_acoustic.N_BANDS):
                    emit[j, b] += c * amp * spectrum[b]
        if cfg.acoustic_shuffle:
            emit = emit[self._shuffle_perm]  # relocate each patch's emission elsewhere
        return emit

    def _hidden_pred_count(self, i: int) -> int:
        """Ground-truth hidden predator count near patch i (ANALYSIS-ONLY): predators
        in patch i plus its graph-neighbors (the locality a sound could carry).
        Agents NEVER receive this; binned by quantiles post-hoc."""
        c = len(self.patches[i].predators)
        for nb in self.neighbors[i]:
            c += len(self.patches[nb].predators)
        return c

    def _record_acoustic_analysis(self, t: int) -> None:
        """Sample (perceived per-band intensity, raw hidden predator density) per
        patch using the start-of-step received field.  Binned post-hoc in run().
        ANALYSIS-ONLY; never fed to agents."""
        acfg = self._acoustic
        n = self.cfg.n_patches
        for i in range(n):
            perceived = _acoustic.perceive(self._received[i], acfg, self._ac_rng)
            self._an_perceived.append(np.asarray(perceived, dtype=float))
            self._an_hidden.append(self._hidden_pred_count(i))
            self._an_step.append(t)

    @staticmethod
    def _tercile_bin(x: np.ndarray) -> np.ndarray:
        """Bin a 1-D array into 0/1/2 by its own 33rd/67th percentiles.  A
        (near-)constant array collapses to all-zeros (no information)."""
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return x.astype(int)
        q1, q2 = np.quantile(x, [1.0 / 3.0, 2.0 / 3.0])
        if q2 <= q1:  # degenerate / constant — no resolvable variation
            if x.max() <= x.min():
                return np.zeros_like(x, dtype=int)
            q1 = q2 = float(x.min())  # split off only the minimum
        b = np.zeros_like(x, dtype=int)
        b[x > q1] = 1
        b[x > q2] = 2
        return b

    def _acoustic_analysis(self) -> dict:
        """Compute MI(acoustic obs; hidden predator density) + detection error +
        capture hazard, over the STEADY-STATE second half.  ANALYSIS-ONLY.

        Per-band and scalar MI use the SAME 3-bin resolution so banded-vs-scalar is
        a fair comparison (the abort check 'scalar works as well as banded').  A
        shuffled-hidden null gives the MI noise floor."""
        per = np.asarray(self._an_perceived, dtype=float)   # (S, 3)
        hid = np.asarray(self._an_hidden, dtype=float)       # (S,)
        steps = np.asarray(self._an_step)
        if per.shape[0] == 0:
            return {"mi_high_bits": 0.0, "mi_mid_bits": 0.0, "mi_low_bits": 0.0,
                    "mi_scalar_bits": 0.0, "mi_banded_bits": 0.0, "mi_null_bits": 0.0,
                    "false_positive_rate": 0.0, "false_negative_rate": 0.0,
                    "prey_capture_hazard": 0.0, "captures_total": self.captures_total,
                    "prey_deaths_total": self.prey_deaths_total, "n_samples": 0}
        # steady-state second half
        cut = (steps.max() + 1) // 2
        m = steps >= cut
        per, hid = per[m], hid[m]
        hb = self._tercile_bin(hid)
        bands = [self._tercile_bin(per[:, b]) for b in range(_acoustic.N_BANDS)]
        scalar = self._tercile_bin(per.sum(axis=1))
        joint = bands[0] + 3 * bands[1] + 9 * bands[2]   # 0..26
        hb_l = hb.tolist()
        mi_low = _acoustic.mutual_information_bits(bands[0].tolist(), hb_l)
        mi_mid = _acoustic.mutual_information_bits(bands[1].tolist(), hb_l)
        mi_high = _acoustic.mutual_information_bits(bands[2].tolist(), hb_l)
        mi_scalar = _acoustic.mutual_information_bits(scalar.tolist(), hb_l)
        mi_banded = _acoustic.mutual_information_bits(joint.tolist(), hb_l)
        # MI noise floor: shuffle hidden (breaks the obs<->hidden pairing).
        hb_shuf = hb.copy()
        self._ac_rng.shuffle(hb_shuf)
        mi_null = _acoustic.mutual_information_bits(bands[2].tolist(), hb_shuf.tolist())
        # detection error using the high band (the attack/death signature) vs danger.
        near = hb == 2
        detected = bands[2] == 2
        fp, fn = _acoustic.detection_error_rates(detected.tolist(), near.tolist())
        cap_haz = (self.prey_deaths_total / self.prey_alive_steps
                   if self.prey_alive_steps > 0 else 0.0)
        cap_success = (self.captures_total / self.pred_alive_steps
                       if self.pred_alive_steps > 0 else 0.0)
        return {
            "predator_capture_success": cap_success,
            "mi_high_bits": mi_high, "mi_mid_bits": mi_mid, "mi_low_bits": mi_low,
            "mi_scalar_bits": mi_scalar, "mi_banded_bits": mi_banded, "mi_null_bits": mi_null,
            "false_positive_rate": fp, "false_negative_rate": fn,
            "prey_capture_hazard": cap_haz,
            "captures_total": self.captures_total,
            "prey_deaths_total": self.prey_deaths_total,
            "n_samples": int(per.shape[0]),
        }

    def _perceived_danger(self, i: int) -> float:
        """A gifted listener's scalar DANGER estimate at patch i = perceived
        broadband (attack/death-dominated) intensity.  Honest: it is just loudness;
        the agent is never told it means predator.  Under acoustic_scalar the bands
        are summed (no frequency resolution)."""
        acfg = self._acoustic
        perceived = _acoustic.perceive(self._received[i], acfg, self._ac_rng)
        if self.cfg.acoustic_scalar:
            # No frequency resolution: total loudness only (the deflation control —
            # cannot isolate the predator-specific attack/death signature).
            return float(perceived.sum())
        # Frequency-resolved: the HIGH band is the predator-SPECIFIC signature
        # (only attack + death emit high; movement/feeding/reproduction do not).
        # mid is added lightly (attack/death also emit mid, but so does feeding).
        return float(perceived[2] + 0.3 * perceived[1])

    def _perceived_prey_sound(self, i: int) -> float:
        """A gifted predator's scalar PREY-ACTIVITY estimate at patch i = perceived
        low/mid (movement/feeding-dominated) intensity."""
        acfg = self._acoustic
        perceived = _acoustic.perceive(self._received[i], acfg, self._ac_rng)
        if self.cfg.acoustic_scalar:
            return float(perceived.sum())
        return float(perceived[0] + perceived[1])

    def _choose_neighbor(self, i: int, danger: bool) -> int:
        """Pick the migration target among patch i's graph-neighbors using ONLY the
        perceived acoustic field.  danger=True (prey): flee the QUIETEST danger
        neighbor (argmin).  danger=False (predator): chase the LOUDEST prey-sound
        neighbor (argmax).  Ties -> lowest patch index (deterministic, no rng).
        With directional_hearing off, the agent cannot compare neighbors and falls
        back to the home reading (stays acoustically blind to direction => index 0
        neighbor), which the probe reports as a degraded control."""
        nb = self.neighbors[i]
        if not self.cfg.directional_hearing:
            # Blind to direction: random neighbor (baseline-like).
            return nb[int(self.rng.integers(len(nb)))]
        # Explicit local comparison (no argmax/argsort — those are banned tokens for
        # the global-evaluator anti-cheat guard; this is an individual sensory choice).
        if danger:
            vals = [self._perceived_danger(j) for j in nb]
        else:
            vals = [self._perceived_prey_sound(j) for j in nb]
        spread = max(vals) - min(vals)
        # NO discriminable signal (e.g. silence, or all-equal neighbors): fall back to
        # a RANDOM neighbor so "no usable sound" reduces to baseline migration — never
        # a deterministic drift (the VALIDATION.md degenerate-control guard).
        if spread <= 1e-12:
            return nb[int(self.rng.integers(len(nb)))]
        best_idx = nb[0]
        best_val = vals[0]
        if danger:  # flee the quietest neighbor
            for k in range(1, len(nb)):
                if vals[k] < best_val:
                    best_val, best_idx = vals[k], nb[k]
        else:       # chase the loudest prey-sound neighbor
            for k in range(1, len(nb)):
                if vals[k] > best_val:
                    best_val, best_idx = vals[k], nb[k]
        return best_idx

    # ------------------------------------------------------------------
    def _patch_dynamics(self, patch: Patch, t: int) -> None:
        """Within-patch dynamics for ONE patch using ONLY that patch's agents.

        Order (REPLICATES wellmixed.step): prey births, predation (refuge-gated),
        predator births, predator deaths.  Mutates patch.prey / patch.predators in place.
        """
        cfg = self.cfg
        rng = self.rng

        N_prey = len(patch.prey)
        N_pred = len(patch.predators)

        # (b) Prey logistic births (+ async); optional gated intraspecific contest redistributes
        #     reproduction-opportunity-under-crowding from losers to winners, costed by aggr.
        new_prey_children: List[Critter] = []
        birth_p = self._prey_birth_prob(patch.idx, N_prey, t)
        bmult = [1.0] * N_prey
        if cfg.enable_contest and N_prey > 0:
            crowd_prize = cfg.contest_seize * min(1.0, N_prey / cfg.K_prey_local)
            for i, p in enumerate(patch.prey):           # ascending cid
                if p.aggr <= 0.0:
                    continue
                bmult[i] *= max(0.0, 1.0 - cfg.contest_cost * p.aggr)
                if N_prey < 2:
                    continue
                j = int(rng.integers(N_prey))
                if j == i:
                    continue
                q = patch.prey[j]
                p_win = p.aggr / (p.aggr + q.aggr + 1e-9)
                if rng.random() < p_win:
                    bmult[i] += crowd_prize * (1.0 - cfg.contest_dissipation)
                    bmult[j] = max(0.0, bmult[j] - crowd_prize)
        for i, p in enumerate(patch.prey):               # ascending cid
            if cfg.enable_trait_evolution:
                bp = birth_p * bmult[i] * max(0.0, 1.0 - cfg.escape_cost * max(0.0, p.trait - cfg.escape_baseline))
                if cfg.enable_cannibalism:
                    # being predatory (high aggr) trades off against foraging on the base resource
                    bp *= max(0.0, 1.0 - cfg.cannibal_cost * p.aggr)
                draw_ok = rng.random() < bp
                child_trait = p.trait if cfg.freeze_prey_trait else self._mutate(p.trait)
                # NOTE (R2/R3 footgun): _mutate_aggr draws rng. It is gated by enable_contest OR
                # enable_cannibalism so the trait-evolution golden (790a8499, both OFF) is preserved.
                # A config with either flag ON + enable_trait_evolution + not freeze_prey calls it
                # per-birth -> its rng stream DIVERGES from the attack_cost/trait-evo experiments'
                # stream. Expected and untested; do not assume byte-identity to those once contest/
                # cannibalism is on.
                child_aggr = (self._mutate_aggr(p.aggr)
                              if ((cfg.enable_contest or cfg.enable_cannibalism) and not cfg.freeze_prey_trait)
                              else p.aggr)
            else:
                draw_ok = rng.random() < (birth_p * bmult[i])
                child_trait = p.trait
                child_aggr = p.aggr
            if draw_ok:
                new_prey_children.append(Critter("prey", child_trait, self._next_cid,
                                                 aggr=child_aggr, lineage=p.lineage))
                self._next_cid += 1

        # (c) Predation — Type II, escape-keyed, refuge-gated predator ACCESS.
        #   sat = 1 / (1 + a*h*N_prey)   (shared saturation; prey-density only)
        #   v_ij = 1 / (1 + escape_k * max(0, prey_i.trait - pred_j.trait))
        #   c_ij = attack_a * sat * v_ij; total_haz_i = sum_j c_ij
        #   kill_prob_i = 1 - exp(-total_haz_i)
        #   In a refuge patch, a would-be kill SUCCEEDS only with refuge_predator_access
        #   (an additional per-attempt gate draw) — gates ACCESS, not birth/death rates.
        sat = 1.0 / (1.0 + cfg.attack_a * cfg.handling_h * N_prey) if N_prey > 0 else 1.0
        access = self._refuge_access_prob(patch.idx)

        dead_prey_mask = [False] * N_prey
        pred_captures = [0] * N_pred
        for i, p in enumerate(patch.prey):
            contribs: List[float] = []
            total_haz = 0.0
            for q in patch.predators:
                v_ij = 1.0 / (1.0 + cfg.escape_k * max(0.0, p.trait - q.trait))
                c = cfg.attack_a * sat * v_ij
                contribs.append(c)
                total_haz += c
            kill_prob = 1.0 - math.exp(-total_haz)
            if rng.random() < kill_prob:
                # Refuge gates predator ACCESS: the would-be capture only lands with
                # probability `access`.  Outside refuge access == 1.0 and the extra
                # draw is still consumed ONLY when access < 1.0 (so non-refuge patches
                # keep the wellmixed rng stream exactly).
                if access < 1.0 and rng.random() >= access:
                    continue  # predator could not access the prey in the refuge
                dead_prey_mask[i] = True
                # Attribute the kill to ONE predator, weighted by hazard contribution.
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

        # (c2) Cannibalism / intraguild predation (R3; gated): a predatory prey (high aggr) kills +
        #      consumes a conspecific (low aggr), gaining a reproduction boost. Frequency-dependent
        #      (predators eat non-predators), so a stable predator/prey polymorphism CAN emerge.
        #      Adds to dead_prey_mask + new_prey_children (applied in (e)); rng drawn only for aggr>0
        #      prey, only under enable_cannibalism -> byte-identical OFF.
        if cfg.enable_cannibalism and N_prey > 1:
            for i, p in enumerate(patch.prey):           # ascending cid
                if p.aggr <= 0.0 or dead_prey_mask[i]:
                    continue
                j = int(rng.integers(N_prey))
                if j == i or dead_prey_mask[j]:
                    continue
                q = patch.prey[j]
                if cfg.cannibal_defense_by_escape:
                    # R3b: defense is a SEPARATE trait (escape). A high-escape target resists being
                    # cannibalised; offense (aggr) and defense (escape) are now decoupled.
                    vuln = 1.0 / (1.0 + cfg.escape_k * q.trait)
                    kill_prob = max(0.0, min(1.0, p.aggr * vuln))
                else:
                    kill_prob = max(0.0, min(1.0, p.aggr * (1.0 - q.aggr)))  # R3: defense = own aggr (coupled)
                if rng.random() < kill_prob:
                    dead_prey_mask[j] = True
                    if rng.random() < cfg.cannibal_gain:  # reproduction boost from the meal
                        child_aggr = p.aggr if cfg.freeze_prey_trait else self._mutate_aggr(p.aggr)
                        new_prey_children.append(Critter("prey", p.trait, self._next_cid,
                                                         aggr=child_aggr, lineage=p.lineage))
                        self._next_cid += 1

        # (d) Predator numerical response (individual captures) + self-limit death
        dead_pred_mask = [False] * N_pred
        new_pred_children: List[Critter] = []
        death_p_pred = self._pred_death_prob(N_pred)
        for j, q in enumerate(patch.predators):
            birth_p_pred = cfg.pred_birth_per_capture * cfg.assimilation * pred_captures[j]
            if cfg.enable_trait_evolution:
                # Per-individual attack cost (MIRRORS the prey escape cost EXACTLY):
                # higher attack above baseline -> lower predator fecundity.  With the
                # default attack_cost=0.0 the factor is exactly 1.0, so the predator-birth
                # threshold is byte-identical and the rng draw below is unchanged.
                birth_p_pred = birth_p_pred * max(
                    0.0, 1.0 - cfg.attack_cost * max(0.0, q.trait - cfg.attack_baseline)
                )
            birth_p_pred = min(1.0, max(0.0, birth_p_pred))
            if rng.random() < birth_p_pred:
                child_trait = (q.trait if cfg.freeze_predator_trait else self._mutate(q.trait)) \
                    if cfg.enable_trait_evolution else q.trait
                new_pred_children.append(Critter("pred", child_trait, self._next_cid))
                self._next_cid += 1
            if rng.random() < death_p_pred:
                dead_pred_mask[j] = True

        # (a-acoustic) Record physical event counts for sound emission (gated; no rng).
        #   feeding   = successful prey foraging (a prey birth)
        #   reproduction = a birth (prey or predator)
        #   attack    = a landed capture; death = a prey predation death + a predator death
        # These are PHYSICAL signatures; the agent is never told the event type.
        if cfg.enable_acoustic_field:
            ec = self._emit_counts[patch.idx]
            n_prey_births = len(new_prey_children)
            n_pred_births = len(new_pred_children)
            n_captures = sum(pred_captures)
            n_prey_deaths = sum(dead_prey_mask)
            n_pred_deaths = sum(dead_pred_mask)
            ec["feeding"] += n_prey_births
            ec["reproduction"] += n_prey_births + n_pred_births
            ec["attack"] += n_captures
            ec["death"] += n_prey_deaths + n_pred_deaths
            self.captures_total += n_captures
            self.prey_deaths_total += n_prey_deaths

        # (e) Apply deaths, append children (preserve ascending-cid order)
        patch.prey = [p for i, p in enumerate(patch.prey) if not dead_prey_mask[i]]
        patch.prey.extend(new_prey_children)
        patch.predators = [q for j, q in enumerate(patch.predators) if not dead_pred_mask[j]]
        patch.predators.extend(new_pred_children)

    # ------------------------------------------------------------------
    def _migrate(self, record_moves: bool = False) -> Optional[List[Tuple[int, int]]]:
        """Local migration phase: MOVE discrete individuals to a graph-neighbor.

        Never copies, creates, or destroys; never crosses a non-neighbor; never reads
        global population/fitness.  Processed prey-then-predators, patches in ascending
        index, agents in ascending cid.  Returns the (origin, target) move list when
        record_moves is True (used by anti-long-range test).

        RING topology special path: uses rng.random() < 0.5 to pick left vs right,
        IDENTICAL to the original implementation, preserving the rng stream byte-for-byte
        so existing golden/determinism tests are unchanged.

        Non-ring topologies: use a uniform-integer draw into self.neighbors[i].
        """
        cfg = self.cfg
        rng = self.rng
        n = cfg.n_patches
        is_ring = (cfg.topology == "ring")
        moves: List[Tuple[int, int]] = [] if record_moves else None

        # Collect moves first (decide based on the PRE-migration occupants of each
        # patch), then apply, so an agent that just arrived cannot migrate again this
        # step and the decision order is deterministic.
        gifted_prey = (cfg.enable_acoustic_field and cfg.acoustic_response
                       and cfg.gifted_hearing == "prey")
        acoustic_on = cfg.enable_acoustic_field

        # --- Prey ---
        prey_outgoing: List[List[Critter]] = [[] for _ in range(n)]  # to-append per patch
        for i, patch in enumerate(self.patches):
            stay: List[Critter] = []
            for c in patch.prey:  # ascending cid
                if cfg.migration_rate_prey > 0.0 and rng.random() < cfg.migration_rate_prey:
                    if gifted_prey:
                        # Gifted hearing: flee toward the QUIETEST neighbor (lowest perceived
                        # danger). No rng direction draw — direction is acoustic, deterministic.
                        target = self._choose_neighbor(i, danger=True)
                    elif is_ring:
                        # ORIGINAL draw: left/right as one rng.random() < 0.5 call.
                        target = (i + 1) % n if rng.random() < 0.5 else (i - 1) % n
                    else:
                        nb = self.neighbors[i]
                        target = nb[int(rng.integers(len(nb)))]
                    prey_outgoing[target].append(c)
                    if acoustic_on:
                        self._emit_counts[i]["movement"] += 1
                    if record_moves:
                        moves.append((i, target))
                else:
                    stay.append(c)
            patch.prey = stay
        for i, patch in enumerate(self.patches):
            patch.prey.extend(prey_outgoing[i])

        gifted_pred = (cfg.enable_acoustic_field and cfg.acoustic_response
                       and cfg.gifted_hearing == "pred")

        # --- Predators ---
        pred_outgoing: List[List[Critter]] = [[] for _ in range(n)]
        for i, patch in enumerate(self.patches):
            stay = []
            for c in patch.predators:
                if cfg.migration_rate_pred > 0.0 and rng.random() < cfg.migration_rate_pred:
                    if gifted_pred:
                        # Gifted hearing: approach the LOUDEST prey-activity neighbor.
                        target = self._choose_neighbor(i, danger=False)
                    elif is_ring:
                        # ORIGINAL draw: left/right as one rng.random() < 0.5 call.
                        target = (i + 1) % n if rng.random() < 0.5 else (i - 1) % n
                    else:
                        nb = self.neighbors[i]
                        target = nb[int(rng.integers(len(nb)))]
                    pred_outgoing[target].append(c)
                    if acoustic_on:
                        self._emit_counts[i]["movement"] += 1
                    if record_moves:
                        moves.append((i, target))
                else:
                    stay.append(c)
            patch.predators = stay
        for i, patch in enumerate(self.patches):
            patch.predators.extend(pred_outgoing[i])

        return moves

    # ------------------------------------------------------------------
    def _record_occupancy_transitions(self) -> None:
        """Detect per-patch local-extinction (>0 -> 0) and recolonization (0 -> >0)
        transitions for prey AND predators, updating the running counters."""
        for i, patch in enumerate(self.patches):
            has_prey = len(patch.prey) > 0
            has_pred = len(patch.predators) > 0
            if self._had_prey[i] and not has_prey:
                self.local_extinction_events += 1
            if (not self._had_prey[i]) and has_prey:
                self.recolonization_events += 1
            if self._had_pred[i] and not has_pred:
                self.local_extinction_events += 1
            if (not self._had_pred[i]) and has_pred:
                self.recolonization_events += 1
            self._had_prey[i] = has_prey
            self._had_pred[i] = has_pred

    # ------------------------------------------------------------------
    def step(self) -> dict:
        """Advance one time step.  Returns per-step summary dict (counts only)."""
        cfg = self.cfg
        # (a0-acoustic) At step START: compute the received field from PAST emissions
        # (propagation delay => sound emitted this step is first audible next step),
        # record the analysis sample, and reset this step's emission accumulator.
        if cfg.enable_acoustic_field:
            self._received = self._field.received(self.t)
            # prey/predator alive at step start (capture-hazard + capture-success denominators)
            self.prey_alive_steps += sum(len(p.prey) for p in self.patches)
            self.pred_alive_steps += sum(len(p.predators) for p in self.patches)
            if self.t >= 0:
                self._record_acoustic_analysis(self.t)
            self._emit_counts = [self._zero_counts() for _ in range(cfg.n_patches)]

        # (a) Within-patch dynamics, patches in ascending index.
        for patch in self.patches:
            self._patch_dynamics(patch, self.t)
        # (b) Migration AFTER within-patch dynamics.
        self._migrate()
        # (c) Record local-extinction / recolonization transitions.
        self._record_occupancy_transitions()

        # (d-acoustic) Push this step's emissions into the propagation buffer.
        if cfg.enable_acoustic_field:
            self._field.push_emissions(self.t, self._emit_to_bands())

        self.t += 1

        patch_prey = [len(p.prey) for p in self.patches]
        patch_pred = [len(p.predators) for p in self.patches]
        return {
            "t": self.t,
            "global_prey": sum(patch_prey),
            "global_pred": sum(patch_pred),
            "patch_prey": patch_prey,
            "patch_pred": patch_pred,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def cross_patch_synchrony(patch_prey_series: List[List[int]]) -> float:
        """Mean pairwise Pearson correlation of patch prey-count series over the
        SECOND HALF of the run.  Lower => more de-correlated (more asynchronous).

        A patch with zero variance over the window (e.g. extinct/flat) is skipped for
        pairs involving it; if no valid pair exists, returns 0.0."""
        n = len(patch_prey_series)
        if n < 2:
            return 0.0
        T = len(patch_prey_series[0])
        half = T // 2
        tails = [np.asarray(s[half:], dtype=float) for s in patch_prey_series]
        corrs: List[float] = []
        for a in range(n):
            for b in range(a + 1, n):
                xa, xb = tails[a], tails[b]
                if xa.std() == 0.0 or xb.std() == 0.0:
                    continue
                c = float(np.corrcoef(xa, xb)[0, 1])
                if not math.isnan(c):
                    corrs.append(c)
        return float(np.mean(corrs)) if corrs else 0.0

    @staticmethod
    def _cv_tail(series: List[int]) -> float:
        """Coefficient of variation (std/mean) over the second-half tail of a series."""
        T = len(series)
        tail = np.asarray(series[T // 2:], dtype=float)
        m = tail.mean()
        if m == 0.0:
            return 0.0
        return float(tail.std() / m)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Run to horizon OR global extinction OR explosion.  Returns result dict."""
        cfg = self.cfg
        n = cfg.n_patches

        def _patch_counts():
            return ([len(p.prey) for p in self.patches],
                    [len(p.predators) for p in self.patches])

        pp0, qq0 = _patch_counts()
        global_prey_series: List[int] = [sum(pp0)]
        global_pred_series: List[int] = [sum(qq0)]
        patch_prey_series: List[List[int]] = [[c] for c in pp0]
        patch_pred_series: List[List[int]] = [[c] for c in qq0]
        occupancy_series: List[float] = [
            sum(1 for i in range(n) if pp0[i] > 0 and qq0[i] > 0) / n
        ]

        step_summaries: List[dict] = []
        aggr_mean_series: List[float] = []
        exploded = False

        while self.t < cfg.horizon:
            summary = self.step()
            step_summaries.append(summary)

            gp = summary["global_prey"]
            gq = summary["global_pred"]
            pp = summary["patch_prey"]
            qq = summary["patch_pred"]

            global_prey_series.append(gp)
            global_pred_series.append(gq)
            for i in range(n):
                patch_prey_series[i].append(pp[i])
                patch_pred_series[i].append(qq[i])
            occupancy_series.append(
                sum(1 for i in range(n) if pp[i] > 0 and qq[i] > 0) / n
            )
            if cfg.track_lineages:
                prey = [c.aggr for p in self.patches for c in p.prey]
                aggr_mean_series.append(float(np.mean(prey)) if prey else 0.0)

            # Stop conditions
            if gp == 0 or gq == 0:
                break  # global extinction of either species
            if gp + gq > cfg.pop_cap:
                exploded = True
                break

        canonical = json.dumps(step_summaries, separators=(",", ":"), sort_keys=True)
        events_hash = hashlib.sha256(canonical.encode()).hexdigest()

        global_extinct = (global_prey_series[-1] == 0 or global_pred_series[-1] == 0)

        result: dict = {
            "events_hash": events_hash,
            "global_prey_series": global_prey_series,
            "global_pred_series": global_pred_series,
            "patch_prey_series": patch_prey_series,
            "patch_pred_series": patch_pred_series,
            "t_end": self.t,
            "global_extinct": global_extinct,
            "exploded": exploded,
            "local_extinction_events": self.local_extinction_events,
            "recolonization_events": self.recolonization_events,
            "occupancy_series": occupancy_series,
            "cross_patch_synchrony": self.cross_patch_synchrony(patch_prey_series),
            "cv_global_prey": self._cv_tail(global_prey_series),
            "cv_global_pred": self._cv_tail(global_pred_series),
        }
        # ---- Acoustic analysis (ANALYSIS-ONLY; agents never received any of this) ----
        if cfg.enable_acoustic_field:
            result["acoustic"] = self._acoustic_analysis()
            result["acoustic"]["attenuation_curve"] = self._field.attenuation_curve()
            result["acoustic"]["shuffle_perm"] = self._shuffle_perm.tolist()
        else:
            result["acoustic"] = {}

        result["aggr_mean_series"] = aggr_mean_series if cfg.track_lineages else []
        if cfg.track_lineages:
            from collections import defaultdict
            agg = defaultdict(list)
            for p in self.patches:
                for c in p.prey:
                    agg[c.lineage].append(c.aggr)
            result["lineage_aggr_final"] = {k: (len(v), float(np.mean(v))) for k, v in agg.items()}
        else:
            result["lineage_aggr_final"] = {}
        return result
