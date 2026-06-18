"""ecology/creature.py — Phenotype, Policy protocol, HomeostaticPolicy, Creature.

Policy pluggability seam:
  The Policy protocol (choose_action signature) is the intended interface for
  future navigation strategies, including a pymdp value-iteration navigator
  that would implement the same `choose_action(creature, world, rng) -> int`
  contract without any changes to the engine.

HomeostaticPolicy heuristic:
  The creature maintains a learned resource map m[cell] initialized to an
  optimistic prior.  Each observation updates m via EMA with learning_rate.
  On each step:
    - If the current cell's resource is below a depletion threshold:
        with prob exploration_bias: EXPLORE (move to least-recently-visited
        neighbor, tie-broken by lowest cell index; rng used for the coin).
        else: EXPLOIT (move to the neighbor with highest expected m[cell]).
    - Else (resource plentiful here): stay or exploit based on exploration_bias.
  This minimises expected energy-deficit: the creature moves toward cells where
  it expects high resource (exploitation) and occasionally explores to update
  beliefs about unknown cells (free-energy-reducing curiosity).  The tie-break
  on lowest cell index is fully deterministic (no set/dict ordering).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, TYPE_CHECKING

import numpy as np

from ecology.genotype import Genotype, thermosense_active

if TYPE_CHECKING:
    from ecology.world import GridWorld


_DEPLETION_THRESHOLD: float = 0.5   # resource below this triggers possible move
_OPTIMISTIC_PRIOR: float = 1.0      # initial belief about unseen cells' resource


def _sigmoid(x: float) -> float:
    """Numerically-stable logistic sigmoid, 1/(1+e^-x) ∈ (0,1).  Exp 211 gate ramp."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------------------
# Phenotype
# ---------------------------------------------------------------------------
@dataclass
class Phenotype:
    """Runtime developmental state of a creature."""
    energy: float
    age: int
    pos: int
    stress: float = 0.0
    damage: float = 0.0
    offspring_count: int = 0
    steps_moved: int = 0
    steps_explored: int = 0
    resource_eaten: float = 0.0
    alive: bool = True
    cause_of_death: str | None = None  # free string; "starvation" used here;
                                        # Exp 195 may add "temperature", "damage", etc.
    birth_t: int = 0                    # Exp 197: timestep the creature was created;
                                        # founders keep the default 0 (t=0 at __init__);
                                        # NOT emitted in any event dict / events_hash.
    # Exp 204: residue/false-positive discrimination counters — lifetime tallies of the
    # eat decision (true/false positive = eat fresh/residue; false/true negative = skip
    # fresh/residue). Pure phenotype telemetry, NEVER emitted in any event dict /
    # events_hash, so they do not affect determinism. Default 0 ⇒ untouched when the
    # residue mechanic is OFF (the counters simply never increment).
    tp_count: int = 0                   # ate an actually-fresh cell (correct eat)
    fp_count: int = 0                   # ate an actually-residue cell (costly mistake)
    fn_count: int = 0                   # skipped an actually-fresh cell (missed food)
    tn_count: int = 0                   # skipped an actually-residue cell (correct skip)


# ---------------------------------------------------------------------------
# Policy protocol
# ---------------------------------------------------------------------------
class Policy(Protocol):
    """Navigation policy interface.

    A future pymdp value-iteration navigator implements this same protocol:
      def choose_action(self, creature: Creature, world: GridWorld,
                        rng: np.random.Generator) -> int

    Returns the target cell (a valid flat position).  May return current pos
    for "stay".  Must be deterministic given rng state.
    """

    def choose_action(
        self,
        creature: "Creature",
        world: "GridWorld",
        rng: np.random.Generator,
    ) -> int: ...


# ---------------------------------------------------------------------------
# HomeostaticPolicy
# ---------------------------------------------------------------------------
class HomeostaticPolicy:
    """Simple heuristic policy maintaining a learned resource map.

    Internal state:
      m[cell]  — EMA-updated expected resource value per cell.
      visit_t[cell] — timestep of last visit (or -1 if never).

    See module docstring for the full choice logic.
    """

    def __init__(self, n_cells: int, learning_rate: float, exploration_bias: float) -> None:
        self.n_cells = n_cells
        self.learning_rate = learning_rate
        self.exploration_bias = exploration_bias
        # Optimistic prior: assume all cells have some resource
        self.m: np.ndarray = np.full(n_cells, _OPTIMISTIC_PRIOR, dtype=np.float64)
        self.visit_t: np.ndarray = np.full(n_cells, -1, dtype=np.int64)
        # Exp 201: private EMA tracker of the drifting food-band center.  None until
        # the band-staleness branch first runs (which only happens when
        # enable_band_staleness=True); a pure attribute, NEVER in events_hash, so the
        # OFF path is byte-identical to exp194-200.  Lazily initialised to
        # world.food_optimal_base (a neutral start, not the moving true value).
        self.band_estimate: float | None = None
        # Exp 206: per-creature class-occupancy tally (modal niche occupied), lazily inited to
        # [0]*K by the engine eat step.  A plain attribute, NEVER in events_hash; KEPT through
        # release_maps() (only m/visit_t are dropped) so the post-hoc I(h;niche)/knockout analysis
        # can read the dead creature's occupancy.  None when the niche mechanic is OFF.
        self.niche_occ: "list[int] | None" = None
        # Exp hidden-state-mode: rolling cue buffer for belief integration.  None until the
        # hidden-mode branch first runs (only when enable_hidden_mode=True); a plain attribute,
        # NEVER in events_hash, so the OFF path is byte-identical to exp194-206.
        self.cue_buffer: "list[float] | None" = None
        # Phase 3 rung-1b: continuous-belief EMA state (lazy; only used when
        # belief_persistence>0). Plain attribute, NEVER in events_hash.
        self.belief_ema: "float | None" = None
        # Phase 4: flag set each step by choose_action to record whether the creature
        # paid for an active-sensing probe this step.  Read by engine._step_one_creature
        # to charge probe_cost.  Plain attribute, NEVER in events_hash.
        self.probed_this_step: bool = False
        # Exp 211 (uncertainty-gated active sensing): per-step gate telemetry, all plain
        # attributes NEVER in events_hash (the new probe policies are new code paths; the
        # fixed_rate/OFF paths stay byte-identical because these are pure reads/writes).
        #   last_action_margin       = |provisional_belief - 0.5| this step (None if not set)
        #   last_probe_changed_action= did the probe flip the which-half decision this step
        #   _mhat_pre                = no-probe (single-cue) which-half decision, for the above
        #   margin_buffer            = ring buffer of recent margins (gate_shuffle time-shuffle)
        self.last_action_margin: "float | None" = None
        self.last_probe_changed_action: bool = False
        self._mhat_pre: "int | None" = None
        self.margin_buffer: "list[float] | None" = None

    def _peek_belief(self, cue: float, rho: float, k: int) -> float:
        """Provisional belief from the SINGLE (pre-probe) cue, WITHOUT mutating state.

        Mirrors the integration in choose_action exactly (EMA when rho>0, else the
        memory_horizon buffer-mean) but on the single cue and as a pure read — so the
        gate's action margin = |this - 0.5| is exactly the no-probe which-half decision
        quantity, and m_hat_pre/m_hat_post differ iff the probe changed the decision.
        No rng, no mutation ⇒ computing it leaves the fixed_rate/OFF rng stream identical."""
        if rho > 0.0:
            return cue if self.belief_ema is None else (1.0 - rho) * cue + rho * self.belief_ema
        buf = self.cue_buffer if self.cue_buffer is not None else []
        window = (buf + [cue])[-k:]
        return sum(window) / len(window)

    def update_belief(self, pos: int, observed: float, t: int) -> None:
        """Update learned map at pos with EMA; record visit time."""
        self.m[pos] += self.learning_rate * (observed - self.m[pos])
        self.visit_t[pos] = t

    def release_maps(self) -> None:
        """PERF (memory): drop the two n_cells belief maps (m, visit_t) — the dominant per-
        creature heap. Called by the engine when a creature DIES: a dead creature's maps are
        never read again (sense()/act() run only on the living; the summary uses genotype +
        phenotype; band_estimate — a single float — is kept for post-hoc inspection). Frees
        ~2.3 KB/creature, ~175 MB on a long high-turnover run, without changing any result."""
        self.m = None              # type: ignore[assignment]
        self.visit_t = None        # type: ignore[assignment]

    def choose_action(
        self,
        creature: "Creature",
        world: "GridWorld",
        rng: np.random.Generator,
    ) -> int:
        pos = creature.phenotype.pos

        # Exp 235: terrain-gated candidate neighbor set.  ONLY inside this ON branch;
        # the OFF path uses world.neighbors() VERBATIM (zero new rng draws).  The
        # crossing roll is the ONLY new rng draw, strictly inside this ON branch.
        # Gated on world.terrain_gates_movement (set True by enable_terrain in engine).
        if world.terrain_gates_movement and world.elevation is not None:
            neighbors = world.climbable_neighbors(
                pos, float(creature.genotype.climb_ability), rng
            )
            if not neighbors:
                return pos  # all edges sealed at current climb_ability — stay
        else:
            neighbors = world.neighbors(pos)

        if not neighbors:
            return pos  # trapped (shouldn't happen on 12x12 but handle it)

        # Exp hidden-state-mode: HIGHEST-PRIORITY branch — runs ONLY when the hidden mode
        # is enabled.  ONE rng draw (the noisy cue); NO further draws.  OFF path never runs
        # this block ⇒ byte-identical to Exp 194-206 (the use_thermo path below is reached
        # with IDENTICAL rng draws when enable_hidden_mode is False).
        if world.enable_hidden_mode and world.cell_type is not None:
            cue = float(world.hidden_mode) + rng.normal(0.0, world.cue_noise)
            # Phase 4 active sensing: optional probe — pay to draw extra cues this step and
            # average them into a sharper belief (variance ~1/(1+n)).  ON path only; when
            # enable_active_sensing is False NO extra rng draws happen ⇒ byte-identical to
            # the Phase-3 path.  The extra samples are ALWAYS drawn when active sensing is ON
            # (keep-the-draw / discard-the-result idiom, cf. mutate's freeze guards) so the
            # rng stream is identical across information_sampling_rate values; the trait keys
            # ONLY whether they are averaged in and whether the cost is charged.
            self.probed_this_step = False
            self.last_probe_changed_action = False
            self.last_action_margin = None
            self._mhat_pre = None
            if world.enable_active_sensing and world.probe_policy != "off":
                policy = world.probe_policy
                gain = float(creature.genotype.information_sampling_rate)
                rho_g = float(creature.genotype.belief_persistence)
                k_g = max(1, int(creature.genotype.memory_horizon))
                # Action margin = |provisional belief - 0.5|, the creature's OWN ambiguity
                # about which half to steer to, computed from the SINGLE cue (no rng, no
                # mutation).  This is the only quantity the uncertainty gate may read.
                cue0 = cue
                prov = self._peek_belief(cue0, rho_g, k_g)
                margin = abs(prov - 0.5)
                self.last_action_margin = margin
                self._mhat_pre = 1 if prov >= 0.5 else 0
                if policy == "fixed_rate":
                    # EXACT Exp 210 path — BYTE-IDENTICAL (rng draws: u, then probe_n_samples
                    # extras, in this order; everything above is pure arithmetic / reads).
                    u = rng.random()
                    extra = [float(world.hidden_mode) + rng.normal(0.0, world.cue_noise)
                             for _ in range(world.probe_n_samples)]
                    if u < gain:
                        cue = (cue + sum(extra)) / (1.0 + len(extra))
                        self.probed_this_step = True
                elif policy in ("uncertainty_gated", "pure_cost", "hidden_scramble"):
                    # Uncertainty-GATED trigger: probe prob = gain * sigmoid(sensitivity *
                    # (threshold - margin)) — high only when the which-half call is ambiguous.
                    gate_w = _sigmoid(world.uncertainty_gate_sensitivity
                                      * (world.uncertainty_gate_threshold - margin))
                    if rng.random() < gain * gate_w:
                        self.probed_this_step = True
                        if policy == "uncertainty_gated":
                            extra = [float(world.hidden_mode) + rng.normal(0.0, world.cue_noise)
                                     for _ in range(world.probe_n_samples)]
                            cue = (cue + sum(extra)) / (1.0 + len(extra))
                        elif policy == "hidden_scramble":
                            # Same trigger + cost, but the extra cues carry a mode DECORRELATED
                            # from the true hidden_mode ⇒ averaging them in gives no information.
                            scr = float(int(rng.integers(0, 2)))
                            extra = [scr + rng.normal(0.0, world.cue_noise)
                                     for _ in range(world.probe_n_samples)]
                            cue = (cue + sum(extra)) / (1.0 + len(extra))
                        # pure_cost: probed (cost charged) but cue NOT sharpened — no information.
                elif policy == "random_cost_matched":
                    # Budget-matched random TIMING: fixed prob, same info + cost as gated.
                    if rng.random() < world.random_cost_matched_probe_rate:
                        self.probed_this_step = True
                        extra = [float(world.hidden_mode) + rng.normal(0.0, world.cue_noise)
                                 for _ in range(world.probe_n_samples)]
                        cue = (cue + sum(extra)) / (1.0 + len(extra))
                elif policy == "gate_shuffle":
                    # Same gate, but read a TIME-SHUFFLED margin: same marginal probe rate,
                    # timing decorrelated from the CURRENT step's uncertainty (info ON).
                    if self.margin_buffer is None:
                        self.margin_buffer = []
                    self.margin_buffer.append(margin)
                    if len(self.margin_buffer) > world.gate_shuffle_buffer:
                        self.margin_buffer = self.margin_buffer[-world.gate_shuffle_buffer:]
                    j = int(rng.integers(0, len(self.margin_buffer)))
                    shuf = self.margin_buffer[j]
                    gate_w = _sigmoid(world.uncertainty_gate_sensitivity
                                      * (world.uncertainty_gate_threshold - shuf))
                    if rng.random() < gain * gate_w:
                        self.probed_this_step = True
                        extra = [float(world.hidden_mode) + rng.normal(0.0, world.cue_noise)
                                 for _ in range(world.probe_n_samples)]
                        cue = (cue + sum(extra)) / (1.0 + len(extra))
            rho = float(creature.genotype.belief_persistence)
            if rho > 0.0:
                # Phase 3 rung-1b: CONTINUOUS belief — an EMA with persistence rho, so a
                # genuinely small heritable step (rho -> rho+eps) can be tested. rho=0 would
                # be "react to current cue only" (== memory_horizon=0). The single cue rng
                # draw above is identical to the buffer path, so byte-identity is preserved.
                if self.belief_ema is None:
                    self.belief_ema = cue
                else:
                    self.belief_ema = (1.0 - rho) * cue + rho * self.belief_ema
                belief = self.belief_ema
            else:
                # Integer buffer-mean path (rung-1; unchanged when belief_persistence==0).
                k = max(1, int(creature.genotype.memory_horizon))
                if self.cue_buffer is None:
                    self.cue_buffer = []
                self.cue_buffer.append(cue)
                if len(self.cue_buffer) > k:
                    self.cue_buffer = self.cue_buffer[-k:]
                belief = sum(self.cue_buffer) / len(self.cue_buffer)
            m_hat = 1 if belief >= 0.5 else 0

            # Exp 211 pivotality telemetry (no rng): the probe CHANGED the which-half
            # decision iff it fired AND the post-probe m_hat differs from the no-probe
            # (single-cue) m_hat.  Read by engine._step_one_creature into
            # probe_changed_action_count.  Set only when active sensing is on.
            if self._mhat_pre is not None:
                self.last_probe_changed_action = (
                    self.probed_this_step and (m_hat != self._mhat_pre)
                )

            # Steer toward the inferred-good half (right if m_hat==1, left if m_hat==0).
            # Score lexicographically: (in-good-half, believed-resource, -index).
            # "in-good-half" = 1 if neighbor's cell_type matches m_hat, else 0.
            # This keeps belief as a meaningful secondary tie-breaker so creatures
            # don't graze only the extreme edge of the correct half.
            # No further rng draws; lowest index wins ties.
            def _score(n: int) -> tuple:
                toward = 1 if world.cell_type[n] == m_hat else 0
                belief = float(self.m[n]) if self.m is not None else 0.0
                return (toward, belief, -n)

            return max(neighbors, key=_score)

        # Exp 197: thermal-aware branch — ONLY when temperature field is present AND
        # the creature has an active thermosense organ.  When use_thermo is False,
        # the exact existing logic below runs with IDENTICAL rng draws (regression guard).
        use_thermo = (
            world.temperature is not None
            and thermosense_active(creature.genotype, world.thermosense_active_threshold)
        )

        if not use_thermo:
            # ----------------------------------------------------------------
            # Exp 237: FOOD-SENSE branch — gated on world.enable_food_sense.
            # Activates ONLY when current cell is depleted (resource <
            # _DEPLETION_THRESHOLD) and enable_food_sense is True.  OFF path
            # never runs this block ⇒ byte-identical to Exp 194-236 (the
            # enable_navigation + explore/exploit paths below are reached with
            # IDENTICAL rng draws when enable_food_sense is False).
            #
            # MECHANIC (honest, food-driven, NO index artifacts):
            #   For each admissible neighbor n (climbable_neighbors when terrain ON),
            #   compute scent(n) = sum over ALL cells c of resource[c] * decay^manhattan_dist(n, c)
            #   Move to the neighbor with highest scent; tie-break by LOWER index (NEUTRAL).
            #   The rich plateau (2x regen) raises scent even from the basin, so the
            #   creature persistently retries the rim instead of retreating downhill.
            #
            # RNG discipline: the ONLY rng draw is the terrain crossing roll already
            #   inside climbable_neighbors (computed above as `neighbors`).  No new draws.
            # ----------------------------------------------------------------
            if world.enable_food_sense:
                _fs_resource = world.resource_at(pos)
                if _fs_resource < _DEPLETION_THRESHOLD:
                    _cols = world.cols
                    _n_cells = world.rows * _cols
                    _decay = world.food_sense_decay
                    _best_scent = float("-inf")
                    _best_n = neighbors[0]
                    for _n in neighbors:
                        _nr, _nc = divmod(_n, _cols)
                        _scent = 0.0
                        for _c in range(_n_cells):
                            _cr, _cc = divmod(_c, _cols)
                            _mdist = abs(_nr - _cr) + abs(_nc - _cc)
                            _scent += float(world.resource[_c]) * (_decay ** _mdist)
                        # Neutral tie-break: higher scent wins; equal => LOWER index
                        if _scent > _best_scent or (_scent == _best_scent and _n < _best_n):
                            _best_scent = _scent
                            _best_n = _n
                    return _best_n
                # Resource plentiful here: fall through to existing logic

            # ----------------------------------------------------------------
            # Exp 236: NAVIGATION-CAPABLE branch — gated on world.enable_navigation.
            # Activates ONLY when the current cell is depleted (resource <
            # _DEPLETION_THRESHOLD) and enable_navigation is True.  When
            # enable_navigation is False, this block is never entered — the EXISTING
            # code path below runs VERBATIM with identical rng draws (OFF = byte-
            # identical to Exp 194-235; the "stay and eat" path is also unaffected).
            #
            # Mechanic (minimal, honest):
            #   TARGET = argmax over all cells of
            #     score(cell) = m[cell] - nav_distance_penalty * manhattan_dist(pos, cell)
            #   Take ONE STEP toward TARGET: among the candidate neighbors
            #   (climbable_neighbors when terrain_gates_movement is ON — so the climb
            #   gate decides whether the upslope step toward the target is available),
            #   choose the neighbor minimizing manhattan_dist(neighbor, TARGET), tie-
            #   broken by higher m then lower index.  If no neighbor moves toward TARGET
            #   (all sealed / same dist), fall back to best-m neighbor.
            #
            # RNG discipline: the ONLY new rng draw is the terrain crossing roll
            #   already INSIDE climbable_neighbors (world.terrain_gates_movement ON),
            #   which is drawn BEFORE this block via the shared `neighbors` local
            #   variable computed at the top of choose_action.  No additional rng draws
            #   are made in this branch — the `neighbors` list was already computed above.
            # ----------------------------------------------------------------
            current_resource = world.resource_at(pos)

            if world.enable_navigation and current_resource < _DEPLETION_THRESHOLD:
                # --- Navigation: head toward best-remembered distant food ---
                n_cells = world.rows * world.cols
                cols = world.cols
                penalty = world.nav_distance_penalty

                # Compute row, col of current position (fast inline divmod).
                pos_r, pos_c = divmod(pos, cols)

                # Pick TARGET = argmax score(cell) over all cells.
                # score = m[cell] - penalty * manhattan_dist(pos, cell)
                # Tie-break: LOWEST cell index wins (the codebase convention, cf. the
                # (m, -c) tie-breaks elsewhere) — a NEUTRAL spatial tie-break, so any
                # plateau-seeking is driven by learned/expected food (m), not by an
                # index artifact (plateau cells happen to have higher indices).
                # We scan all n_cells; on a 12x12 grid this is 144 iterations — fast.
                best_score = float("-inf")
                target_cell = pos  # fallback: stay (should not happen on non-empty map)
                for cell in range(n_cells):
                    cr, cc = divmod(cell, cols)
                    mdist = abs(pos_r - cr) + abs(pos_c - cc)
                    score = float(self.m[cell]) - penalty * mdist
                    if score > best_score:
                        best_score = score
                        target_cell = cell

                # Compute target row, col for distance-to-target measurement.
                tgt_r, tgt_c = divmod(target_cell, cols)
                cur_dist = abs(pos_r - tgt_r) + abs(pos_c - tgt_c)

                # Choose ONE STEP from pos toward TARGET: among admissible neighbors,
                # prefer the one minimizing manhattan_dist(neighbor, TARGET).
                # Tie-break: higher m[neighbor], then lower index.
                toward: list[int] = []
                away_or_equal: list[int] = []
                for n in neighbors:
                    nr, nc = divmod(n, cols)
                    nd = abs(nr - tgt_r) + abs(nc - tgt_c)
                    if nd < cur_dist:
                        toward.append(n)
                    else:
                        away_or_equal.append(n)

                if toward:
                    # Among toward-target neighbors, prefer smaller dist, then higher m, then lower index.
                    step = min(
                        toward,
                        key=lambda n: (
                            abs(divmod(n, cols)[0] - tgt_r) + abs(divmod(n, cols)[1] - tgt_c),
                            -float(self.m[n]),
                            n,
                        ),
                    )
                    return step
                if away_or_equal:
                    # No neighbor makes progress toward TARGET this tick — fall back to the
                    # best-m available neighbor. (Persistent stay-at-rim was tested and
                    # starves the population under scarcity; see EXPERIMENTS.md Exp 236.)
                    step = max(away_or_equal, key=lambda n: (float(self.m[n]), -n))
                else:
                    step = pos  # no admissible neighbors (extreme edge case)
                return step

            # ----------------------------------------------------------------
            # EXACT existing logic — verbatim, unchanged — rng stream identical
            # when enable_navigation is False (OFF path never touches the block
            # above, so rng draws here are byte-identical to Exp 194-235).
            # When enable_navigation is True but resource is plentiful, we also
            # reach this block (the "stay and eat" path is unchanged).
            # ----------------------------------------------------------------

            # Decide explore vs exploit
            if current_resource < _DEPLETION_THRESHOLD:
                # Resource depleted here — consider moving
                if rng.random() < self.exploration_bias:
                    # EXPLORE: go to least-recently-visited neighbor
                    # Sort by visit_t ascending (oldest / never visited first),
                    # tie-break by lowest cell index (deterministic)
                    target = min(neighbors, key=lambda c: (self.visit_t[c], c))
                    creature.phenotype.steps_explored += 1
                    return target
                else:
                    # EXPLOIT: go to neighbor with highest expected resource
                    # tie-break by lowest cell index
                    target = max(neighbors, key=lambda c: (self.m[c], -c))
                    # Negate c so larger index loses ties: max by (m, -c) means
                    # higher m wins; equal m -> lower c wins (since -c is larger)
                    return target
            else:
                # Resource plentiful here; occasionally explore, else stay
                if rng.random() < self.exploration_bias * 0.3:
                    target = min(neighbors, key=lambda c: (self.visit_t[c], c))
                    creature.phenotype.steps_explored += 1
                    return target
                return pos  # stay and eat

        elif world.enable_niche:
            # ----------------------------------------------------------------
            # Exp 206: ROTATING-CLASS NICHE routing — the ONLY site where the sensor h
            # buys anything.  The creature seeks the UNDER-CROWDED class (read from the
            # ecological crowding state class_occ_prev — like depletion, an allowed rho,
            # NOT a fitness ranking) and routes toward neighbours whose CURRENT class it
            # reads as that target.  h keys ONLY the read accuracy: sigma =
            # niche_confusion*(1-h).  A precise creature resolves the true current class
            # and lands on under-crowded cells (low crowding discount at the eat step); a
            # crude creature misreads and herds.  The class ROTATES (world.class_signal
            # recomputed each step) so a static learned map cannot substitute (the
            # non-memorizability fix).  Gated on enable_niche (mutually exclusive with
            # band-staleness, asserted in engine __init__); returns within itself drawing
            # ONLY in this ON branch, so the OFF path rng is identical.  No food/fitness is
            # f(h): h only sharpens the percept; the eat-step crowding discount is h-blind.
            # ----------------------------------------------------------------
            intensity = creature.genotype.thermosense_intensity
            sigma = max(0.0, world.niche_confusion * (1.0 - intensity))
            K = world.niche_classes
            target_class = int(np.argmin(world.class_occ_prev))   # least-crowded class to seek
            read_sig = (world.class_signal if world.niche_read_perm is None
                        else world.class_signal[world.niche_read_perm])
            fw = world.niche_weight

            def _cls_read(cell: int) -> int:
                noisy = (float(read_sig[cell]) + rng.normal(0.0, sigma)) % 1.0
                cr = int(K * noisy)
                return K - 1 if cr >= K else cr

            # one rng draw per neighbour, fixed (up/down/left/right) order ⇒ deterministic
            nb_scores = [(self.m[n] + (fw if _cls_read(n) == target_class else 0.0), -n, n)
                         for n in neighbors]
            best_score, _, best = max(nb_scores)
            current_resource = world.resource_at(pos)
            if current_resource < _DEPLETION_THRESHOLD:
                return best
            else:
                stay_score = self.m[pos] + (fw if _cls_read(pos) == target_class else 0.0)
                if best_score > stay_score:
                    return best
                return pos

        elif world.forage_mode and world.enable_band_staleness:
            # ----------------------------------------------------------------
            # Exp 201: BAND-STALENESS forage — the drifting band center is NOT
            # handed to the policy for free (that was exp200's fatal flaw).  The
            # creature must privately ESTIMATE it via an EMA tracker whose
            # responsiveness (alpha) and reading-noise are both keyed to
            # thermosense_intensity.  The static spatial gradient stays known
            # (neighbor temps read true); only the drifting CENTER is tracked, so
            # a crude tracker chronically lags a fast band into already-depleted
            # cells while a precise tracker locks on.  This sub-branch is gated on
            # enable_band_staleness and returns within itself, so the exp200 forage
            # path below is reached with IDENTICAL rng draws when the flag is OFF.
            #
            # CRITICAL L19 GUARD: food intake is NEVER written as reward=f(intensity).
            # Intensity affects ONLY the tracker's alpha and reading noise; the food a
            # creature gets falls out of where it steps and the unchanged consume()
            # depletion race.  The SLOW-band null arm is the binding falsifier: if
            # selection appears when the band barely drifts (lag harmless), the
            # relation was imposed, not earned — discard the positive.
            intensity = creature.genotype.thermosense_intensity
            forage_weight = world.thermal_avoidance_weight

            # Lazy neutral init (food_optimal_base, NOT the moving current_food_optimal).
            if self.band_estimate is None:
                self.band_estimate = world.food_optimal_base

            # ONE rng draw: a noisy observation of the drifting TRUE center, quality
            # keyed to intensity; then an intensity-keyed EMA step toward it.
            noise_sd = max(0.0, world.thermosense_noise_base * (1.0 - intensity))
            noisy_center = world.current_food_optimal + rng.normal(0.0, noise_sd)
            alpha = min(1.0, max(0.0, intensity * world.band_responsiveness))
            self.band_estimate += alpha * (noisy_center - self.band_estimate)
            est = self.band_estimate

            current_resource = world.resource_at(pos)
            if current_resource < _DEPLETION_THRESHOLD:
                target = max(
                    neighbors,
                    key=lambda n: (self.m[n] - forage_weight * abs(float(world.temperature[n]) - est), -n),
                )
                return target
            else:
                best_neighbor = max(
                    neighbors,
                    key=lambda n: (self.m[n] - forage_weight * abs(float(world.temperature[n]) - est), -n),
                )
                stay_score = self.m[pos] - forage_weight * abs(float(world.temperature[pos]) - est)
                move_score = (
                    self.m[best_neighbor]
                    - forage_weight * abs(float(world.temperature[best_neighbor]) - est)
                )
                if move_score > stay_score:
                    return best_neighbor
                return pos

        elif world.forage_mode:
            # ----------------------------------------------------------------
            # Exp 200: FORAGE mode — steer TOWARD food-optimal temperature.
            # Runs ONLY when thermosense is active AND world.forage_mode is True.
            # Avoidance mode is completely bypassed; rng draws happen here only
            # in this branch so regression conditions are unaffected.
            # ----------------------------------------------------------------
            food_opt = world.current_food_optimal
            intensity = creature.genotype.thermosense_intensity
            forage_weight = world.thermal_avoidance_weight

            # Noise decreases with higher intensity (better organ = better signal).
            noise_sd = max(0.0, world.thermosense_noise_base * (1.0 - intensity))

            # Noisy temperature reading at each neighbor; steer toward food_opt.
            # Score = learned resource estimate - forage_weight * |noisy_temp - food_opt|
            # Higher score = more food expected AND closer to the food band.
            noisy_temp: dict[int, float] = {}
            for n in neighbors:
                noisy_temp[n] = float(world.temperature[n]) + rng.normal(0.0, noise_sd)

            current_resource = world.resource_at(pos)
            if current_resource < _DEPLETION_THRESHOLD:
                # Resource depleted here — move to best neighbor for foraging.
                target = max(
                    neighbors,
                    key=lambda n: (self.m[n] - forage_weight * abs(noisy_temp[n] - food_opt), -n),
                )
                return target
            else:
                # Resource available here; check if a neighbor is significantly better
                # (close to food band AND has expected resource).
                best_neighbor = max(
                    neighbors,
                    key=lambda n: (self.m[n] - forage_weight * abs(noisy_temp[n] - food_opt), -n),
                )
                stay_score = self.m[pos] - forage_weight * abs(
                    float(world.temperature[pos]) - food_opt
                )
                move_score = (
                    self.m[best_neighbor]
                    - forage_weight * abs(noisy_temp[best_neighbor] - food_opt)
                )
                if move_score > stay_score:
                    return best_neighbor
                return pos

        else:
            # ----------------------------------------------------------------
            # Thermal-aware branch (Exp 197 avoid-mode) — runs ONLY when
            # temperature is on AND creature has active thermosense AND
            # forage_mode is False.
            # This branch may consume rng differently; that is intentional and
            # safe because it never executes in control/regression conditions.
            # ----------------------------------------------------------------
            comfort = world.current_comfort
            tolerance = creature.genotype.temperature_tolerance
            intensity = creature.genotype.thermosense_intensity
            avoidance = world.thermal_avoidance_weight

            # Noise decreases with higher intensity (better organ = better signal).
            noise_sd = max(0.0, world.thermosense_noise_base * (1.0 - intensity))

            # Estimate thermal stress at each neighbor via noisy temperature reading.
            stress_est: dict[int, float] = {}
            for n in neighbors:
                noisy = float(world.temperature[n]) + rng.normal(0.0, noise_sd)
                stress_est[n] = max(0.0, abs(noisy - comfort) - tolerance)

            # Estimate stress at current cell (no rng draw — use true value for stability).
            current_temp_stress = max(
                0.0, abs(float(world.temperature[pos]) - comfort) - tolerance
            )

            # If the current cell is highly stressed (above tolerance), prefer moving
            # to the lowest-stress neighbor regardless of resource.
            if current_temp_stress > 0.0:
                target = min(neighbors, key=lambda n: (stress_est[n], n))
                return target

            # Otherwise: balance resource and thermal stress (exploit with stress penalty).
            current_resource = world.resource_at(pos)
            if current_resource < _DEPLETION_THRESHOLD:
                # Resource depleted — move to best neighbor (high resource, low stress).
                target = max(
                    neighbors,
                    key=lambda n: (self.m[n] - avoidance * stress_est[n], -n),
                )
                return target
            else:
                # Resource plentiful here; stay unless thermally stressed (handled above).
                return pos


# ---------------------------------------------------------------------------
# Creature
# ---------------------------------------------------------------------------
class Creature:
    """A single creature with genotype, phenotype, and homeostatic policy.

    Attributes:
      creature_id   — unique integer id (ascending from 0)
      parent_id     — id of parent or None for founders
      generation    — 0 for founders, parent.generation+1 for offspring
      lineage_root  — id of the founding ancestor in this lineage
      genotype      — frozen inherited config
      phenotype     — mutable runtime state
      policy        — navigation policy (HomeostaticPolicy or any Policy)
      _t            — internal timestep counter for visit tracking
    """

    def __init__(
        self,
        creature_id: int,
        parent_id: int | None,
        generation: int,
        lineage_root: int,
        genotype: Genotype,
        phenotype: Phenotype,
        n_cells: int,
    ) -> None:
        self.creature_id = creature_id
        self.parent_id = parent_id
        self.generation = generation
        self.lineage_root = lineage_root
        self.genotype = genotype
        self.phenotype = phenotype
        self._t: int = 0
        self.policy: HomeostaticPolicy = HomeostaticPolicy(
            n_cells=n_cells,
            learning_rate=genotype.learning_rate,
            exploration_bias=genotype.exploration_bias,
        )

    # ------------------------------------------------------------------
    # Sense + act
    # ------------------------------------------------------------------
    def sense(self, world: "GridWorld", rng: np.random.Generator) -> float:
        """Get a (possibly noisy) resource reading at current cell; update belief."""
        pos = self.phenotype.pos
        reading = world.local_reading(pos, self.genotype.sensor_precision, rng)
        self.policy.update_belief(pos, reading, self._t)
        return reading

    def act(self, world: "GridWorld", rng: np.random.Generator) -> int:
        """Choose and execute a move.  Returns new position.
        Movement cost is ONLY paid if the cell changed (caller does energy deduction).
        """
        target = self.policy.choose_action(self, world, rng)
        old_pos = self.phenotype.pos
        self.phenotype.pos = target
        if target != old_pos:
            self.phenotype.steps_moved += 1
        self._t += 1
        return target

    def is_alive(self) -> bool:
        return self.phenotype.alive

    def genotype_dict(self) -> dict:
        """Return a plain dict of genotype fields for logging."""
        from dataclasses import asdict
        return asdict(self.genotype)
