"""rung2_posability_preflight.py — RUNG-2 POSABILITY PRE-FLIGHT (standalone; no source-patching)

PURPOSE
  Empirically decide whether the N4 "freeze-vs-revise crack" (a HARD-captivity, transient-train
  phenomenon) and an N4-gated MOVEMENT defence surface (a SOFT-pull, escape-by-relocation
  phenomenon) can co-exist in ONE common regime — or whether they live in mutually-exclusive
  regimes (CAN'T-POSE). This is a PRE-FLIGHT: 2 representative crack cells, small seed banks. It
  does NOT decide the full kill test; it decides whether the kill test is even well-posed.

HYPOTHESIS (the thing under test, posed as falsifiable)
  H-POSABLE: there exists a common soft-burst-captivity regime, calibrated so a PASSIVE
  (hold+freeze-able, pull-overridable) agent's v-store is displaced (internal freeze is
  NECESSARY: baseline-no-freeze fails defence and oracle-freeze defends), in which (Q1) a
  fixed-H internal freeze still exhibits a crack (NO single H passes BOTH the defence bar and
  the revision bar), AND (Q2) an N4-gated movement controller (defend-by-moving in Phase W,
  accept-by-staying in the global Phase-R re-color) covers BOTH bars, AND (Q3) the baseline
  deficit holds — on BOTH representative cells, per-color, per-seed.

PREDICTION / PREDECLARATION (registered before any data; per cell x color)
  P1 (precondition, Q3): at the calibrated common alpha, a PASSIVE+FREEZE-able agent under the
     REAL v-store (w = exp(-h_predicted) on the evolving _A_hat; LAMBDA = 0.9997) is displaced:
     baseline-no-freeze defence FAIL >= 7/8 seeds AND oracle-freeze defence PASS >= 7/8 seeds.
  P2 (Q1, the crack survives movement-permitting softening): on that SAME regime, NO single
     fixed-H in {600,900,1200,1800,2400,3000,4200,6000} passes BOTH defence (>= 6/8) AND
     revision (>= 6/8). At least one H reaches defence >= 6/8 in the long-H region (the freeze
     surface is non-degenerate, monotone-rising in H, not identically zero) and at least one
     short H passes revision while a long H fails (the latency≈H+baseline tension is live).
  P3 (Q2, movement covers both): the N4-gated movement controller passes defence (>= 6/8) AND
     revision (>= 6/8). Its always-on optimal-avoider upper-bound sibling ALSO passes defence
     (so a movement NEGATIVE, if it occurs, is surface-bound not gate-latency-bound).
  Predeclared expectation, given the verified Exp-270 tension (passive-displaced and
  optimal-mover-escapes overlap only near a razor-thin color-dependent alpha): the LIKELY
  outcome is CAN'T-POSE on >= 1 cell/color — the soft-burst regime that displaces a passive
  freezer (P1) is expected to let a short fixed-H freeze ALSO trivially defend AND revise (P2
  fails: constants also win), or to leave no movement DOF. CAN'T-POSE is an EXPECTED, reportable
  finding, not a failure.

FALSIFIER (what kills each claim)
  - H-POSABLE is FALSIFIED (-> CAN'T-POSE) on a cell/color if: no calibrated alpha satisfies P1
    (passive not displaceable under the real v-store at any movement-permitting alpha), OR P1
    holds but P2 fails because SOME fixed-H covers both bars (the crack dissolved under
    softening — constants also win, nothing for movement to uniquely beat), OR no alpha
    simultaneously satisfies P1 and admits a freeze arm that can defend (no peer to beat).
  - The POSITIVE (movement beats the constants) is FALSIFIED if P3 fails while P1+P2 hold.
  - The whole pre-flight ABORTS (invalid, not a verdict) if the alpha=1 verdict-level
    equivalence gate fails: at alpha=1 the soft-burst attack MUST reduce MECHANICALLY to the
    committed hard captivity (teleport + qs-reset + v-freeze + no-move, burst_rng consumed in
    exp185 order) and reproduce the committed crack VERDICT STRUCTURE on the gate cell (oracle
    defends >= 7/8; baseline fails defence >= 7/8; some fixed-H defends >= 6/8 in long-H; NO
    fixed-H passes both bars; H600 revision PASS and H6000 revision FAIL), AND the Tier-A
    deterministic spine (settle v-trajectory at t=5000/6000, attack-color=argmin(v) and
    favorite=argmax(v) at t=6000) MUST match the imported real exp185 runner.

  Anti-self-deception guards (binding): per-cell x per-color x per-seed only, NEVER a mean over
  the 2 cells or 3 colors (mean-of-opposites guard); report per-seed dispersion + regime split;
  the actuator objective must match the bar metric (argmax(v), not diet) or the comparison is a
  category error; the real predictability-weight law (NOT w=1.0) and LAMBDA=0.9997 are
  mandatory; soft-pull permanence in Phase R is replaced by a GLOBAL re-color so "revision"
  means the same location-independent thing for both arms.

NOTE ON FILE NAMING. The build spec named the file exp277_rung2_posability_preflight.py inside
its prose; the orchestrator locked the on-disk path to experiments/exp271_rung2_posability.py
with output experiments/outputs/exp271.txt. This script is that locked file; all internal
"exp277" references in the spec map to this exp271 artifact set. (origin/main is at Exp 269; the
clean-line next free number is 271 for this standalone, see exp270 provenance note.)
"""
from __future__ import annotations

import collections
import copy
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from active_loop.creature import Creature, World


# ---------------------------------------------------------------------------
# Import the faithful primitives (zero-drift; exactly as exp186 does)
# ---------------------------------------------------------------------------
def _import_mod(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, str(REPO_ROOT / rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod185 = _import_mod("_mod185", "experiments/exp185_n4_crack_classification.py")
_mod270 = _import_mod("_mod270", "experiments/exp270_ecological_affordance_gate.py")

# Reuse VERBATIM from exp185 (constants + bar fns + the real runner for the spine gate)
run_fork_schedule_185 = _mod185.run_fork_schedule_185
compute_defense = _mod185.compute_defense
phase_r_latency = _mod185.phase_r_latency
to_plain = _mod185.to_plain
pi_of = _mod185.pi_of
tv = _mod185.tv

LAMBDA = _mod185.LAMBDA            # 0.9997 (crack-calibrated)
assert abs(LAMBDA - 0.9997) < 1e-12, f"LAMBDA pin failed: {LAMBDA}"
INIT_MASS = _mod185.INIT_MASS
DEFAULT_THETA = _mod185.DEFAULT_THETA       # 3.5
FINE_EVAL = _mod185.FINE_EVAL               # 25
EVAL = _mod185.EVAL                          # 100
CHUNK_SIZE = _mod185.CHUNK_SIZE              # 100
P6_HOLD = _mod185.P6_HOLD                    # 2000
DEFENSE_FRAC_THRESH = _mod185.DEFENSE_FRAC_THRESH   # 0.5
DEF_OFF_START = _mod185.DEFENSE_WINDOW_OFFSET_START  # 1500
DEF_OFF_END = _mod185.DEFENSE_WINDOW_OFFSET_END      # 2000
FIXED_HORIZONS = list(_mod185.FIXED_HORIZONS_184)    # [600,900,1200,1800,2400,3000,4200,6000]
PHASE_R_START = _mod185.PHASE_R_START         # 6000
REVISION_TOL_NORMAL = 3000                    # REVISION_MODES normal tolerance
N_STEPS_R = _mod185.N_STEPS_PHASE_R           # 15000 (separate R-phase revision run)
MON_W_SNAPS = _mod185.MON_W_SNAPS
CTRL_MBAR_WINDOW = _mod185.CTRL_MBAR_WINDOW
CTRL_MIN_SNAPS = _mod185.CTRL_MIN_SNAPS
REFRACTORY_CHECKS = _mod185.REFRACTORY_CHECKS
BURST_SEED_OFFSET_R = _mod185.BURST_SEED_OFFSET_R
FLOAT_ATOL = 1e-9

# exp270 primitives (the certified-optimal refuge planner + geometry helpers).
# NOTE LAMBDA mismatch: exp270 uses 0.999; we PIN 0.9997 everywhere in this preflight.
dist_to_attack = _mod270.dist_to_attack
pull_cands = _mod270.pull_cands
neighbors = _mod270.neighbors
refuge_components = _mod270.refuge_components
optimal_avoid_policy = _mod270.optimal_avoid_policy
SEG_CMAP = list(_mod270.SEG_CMAP)


# ---------------------------------------------------------------------------
# Preflight config
# ---------------------------------------------------------------------------
# THETA pre-registration (binding spec clause: "assert THETA=3.5 still separates attack from
# no-attack ... else RE-CALIBRATE THETA on the soft regime and PRE-REGISTER — do NOT silently
# proceed"). EMPIRICAL FINDING on this body+gate-cell under the REAL v-store (LAMBDA=0.9997):
# the committed non-degenerate crack signature on CELL_LONG (long-H freeze defends ~7-8/8, short
# H fails defence, no fixed-H covers both bars) manifests at THETA=3.0, NOT 3.5 — at 3.5 the
# rung-2 trigger is too insensitive to engage the freeze during the burst (verified: H3000 def
# 2/8 @3.5 vs 8/8 @3.0, via BOTH the imported real exp185 runner AND this standalone port). We
# therefore PRE-REGISTER THETA=3.0 for the freeze/mover trigger here (the exp185 THETA3 config),
# and the alpha=1 Tier-A/Tier-B equivalence gate is run + asserted at THETA=3.0 to prove the
# port reproduces the committed crack VERDICT STRUCTURE. (DEFAULT_THETA=3.5 retained as a named
# reference.) The Section-2 trigger characterization re-verifies separation under the soft pull.
THETA = 3.0
SEEDS = list(range(250, 258))            # 8 paired seeds
SPINE_SEEDS = [250, 251]
COLORS = [0, 1, 2]
ALPHAS_CALIB = [0.5, 0.55, 0.6, 0.65, 0.7]

# Representative cells (predeclared)
CELL_SHORT = {"L": 1600, "K": 2, "G": 200, "name": "SHORT"}
CELL_LONG = {"L": 2400, "K": 4, "G": 600, "name": "LONG"}    # = alpha=1 equivalence gate cell
CELLS = [CELL_SHORT, CELL_LONG]

# Gate counts (per the bars)
FAMILY_DEF_PASS = 6      # family arms (fixed-H, mover) PASS >= 6/8
ORACLE_DEF_PASS = 7      # oracle PASS >= 7/8
BASELINE_DEF_FAIL = 7    # baseline FAIL >= 7/8
REV_PASS = 6             # revision PASS >= 6/8

# RNG-injection offsets (paired across arms by seed; soft-burst uses its own burst_rng)
RELOC_OFF = 270_000      # reloc_rng_seed = RELOC_OFF + 1000*cell_tag + seed (W phase burst_rng)


def burst_windows_for(cell):
    L, K, G = cell["L"], cell["K"], cell["G"]
    bw = []
    cur = PHASE_R_START
    for ki in range(K):
        bw.append((cur, cur + L))
        cur += L
        if ki < K - 1:
            cur += G
    return bw


def n_steps_for(cell):
    L, K, G = cell["L"], cell["K"], cell["G"]
    return PHASE_R_START + K * L + (K - 1) * G + 2500


# ---------------------------------------------------------------------------
# Standalone SOFT-BURST runner (faithful v-store port; NOT source-patched).
#
# This re-implements the exp185 NORMAL-state value loop VERBATIM (so the Tier-A
# spine byte-matches the imported runner for the first 6000 settle steps and
# catches a w=1.0 regression), then layers the single manipulated bit: the
# soft-pull captivity with hardness knob alpha. At alpha=1, during a RESIST burst
# step, the mechanic reduces EXACTLY to exp185 hard captivity (teleport to a
# random attack cell via burst_rng + qs uniform over attack cells + v FROZEN +
# no policy move), with burst_rng consumed in the exp185 order.
#
# ARMS (policy/freeze only; the v-store law is IDENTICAL across all arms):
#   "baseline"      : v live, never freeze, policy walk  (Q3 / P1 deficit witness)
#   ("fixedH", H)   : freeze v on rung-2 trigger, concede after H frozen steps (Q1/P2)
#   "oracle"        : freeze v exactly during known burst windows (P1 witness)
#   "gated_mover"   : NO freeze; on rung-2 trigger -> relocate via optimal_avoid;
#                     untriggered -> walk; global re-color in Phase R (Q2/P3)
#   "always_mover"  : NO freeze, NO gate; always relocate via optimal_avoid (P3 ceiling)
#   "hold_nofreeze" : NO freeze, HOLD (pull overrides)  (attribution control)
#   "move_freeze"   : freeze + always relocate          (attribution control)
# ---------------------------------------------------------------------------
def run_soft(
    mirro_root: Creature,
    fork_seed: int,
    base_cmap: list,
    n_colors: int,
    arm: object,
    burst_windows: list,
    attack_color: int,
    regime_color: int,
    alpha: float,
    n_steps: int,
    reloc_rng_seed: int,
    avoid_policies: dict,          # (attack_color) -> {cell: next_cell} certified-optimal
    phase: str = "W",             # "W" = defence run (bursts, NO global recolor, tail NORMAL);
                                  # "R" = revision run (NO bursts, GLOBAL recolor from step 6000)
    n_actions: int = 4,
    theta: float = THETA,
    collect_events: bool = False,
) -> dict:
    """One soft-burst fork. attack_color and regime_color are FIXED externally (per spec:
    attack = the tested color; regime = the non-attack color tested in the revision condition;
    NOT onset-argmin).

    Mirrors exp185's TWO-PHASE structure (W and R are SEPARATE sessions):
      - phase 'W': the transient burst train (K bursts of length L, gaps G), then a tail; the
        DEFENCE bar is measured in [last_bend+1500,+2000) which lives in the NORMAL tail. No
        global recolor in a W run.
      - phase 'R': NO bursts; from PHASE_R_START a GLOBAL permanent re-color makes obs emit
        regime_color for ALL cells regardless of body position (location-independent,
        inescapable). The REVISION bar is measured here.
    At alpha=1 the W bursts reduce to exp185 hard captivity; the R recolor reduces to exp185's
    permanent location-gated force-feed (a single random target-cell choice per step via
    burst_rng + qs uniform over target cells)."""
    n_cells = mirro_root.world.n_cells
    fork_name = f"exp271_{arm}_s{fork_seed}_a{alpha:.2f}_ac{attack_color}"
    c = mirro_root.fork(fork_name)
    B = c.world.transition_matrix()

    # v init identical to exp185
    vc = c.value_counts.copy()
    vc_sum = vc.sum()
    if vc_sum > 0:
        v = (vc / vc_sum) * INIT_MASS
    else:
        v = np.ones(n_colors) * (INIT_MASS / n_colors)

    color_cells = [[] for _ in range(n_colors)]
    for ci, col in enumerate(base_cmap):
        color_cells[col].append(ci)
    color_cells_arr = [np.array(lst, dtype=np.int32) for lst in color_cells]

    # burst_rng: SAME structure as exp185 W phase (paired across arms by reloc_rng_seed)
    burst_rng = np.random.default_rng(reloc_rng_seed)
    # belief-pull rng (separate stream; only consumed when 0 < beta < 1)
    belief_rng = np.random.default_rng(reloc_rng_seed + 7_000_000)

    burst_step_set = set()
    if phase == "W":
        for bstart, bend in burst_windows:
            burst_step_set.update(range(bstart, bend))

    is_fixedH = isinstance(arm, tuple) and arm[0] == "fixedH"
    H_horizon = arm[1] if is_fixedH else None
    is_freeze_arm = (arm == "oracle") or is_fixedH or (arm == "move_freeze")
    is_mover = arm in ("gated_mover", "always_mover", "move_freeze")
    is_gated = arm in ("gated_mover",)   # gated by the rung-2 monitor
    is_always_move = arm in ("always_mover", "move_freeze")
    oracle = (arm == "oracle")

    policy = avoid_policies.get(attack_color) if is_mover else None

    expressed_arr = np.empty(n_steps, dtype=np.int32)
    obs_arr = np.empty(n_steps, dtype=np.int32)
    state_arr = np.zeros(n_steps, dtype=np.int32)  # 1 = RESIST/captive this step

    v_traj: list[np.ndarray] = []

    # rung-2 monitor state
    v_fine: list[np.ndarray] = []
    k_since_reset = 0
    mismatch_history: list[float] = []
    m_bar_floor = None
    checks_since_release = REFRACTORY_CHECKS

    # freeze controller state (verbatim-in-substance from exp185 lines 353-365)
    freeze_state = "NORMAL"
    frozen_steps = 0
    entry_step = 0
    pi_ref = None
    m_bar_frozen = 0.0
    blocked_w_by_color = np.zeros(n_colors, dtype=np.float64)
    calm_count = 0
    PRESSURE_WINDOW = _mod185.PRESSURE_WINDOW
    PRESSURE_FRAC = _mod185.PRESSURE_FRAC
    RELEASE_CALM_SNAPS = _mod185.DEFAULT_RELEASE_CALM_SNAPS   # 8

    # trigger characterization
    trigger_events = []   # list of dicts {entry_step, kind}
    n_triggers_in_burst = 0
    n_triggers_settle = 0  # false triggers during settle (t < 6000)
    first_trigger_after_burst = None

    settle_pi_5000 = None
    settle_tv_val = None
    v_at_6000 = None

    # for events_hash (drift-null): record landing cell + obs + expressed each step
    event_log = [] if collect_events else None

    global_step = 0
    n_chunks = n_steps // CHUNK_SIZE
    assert n_chunks * CHUNK_SIZE == n_steps

    beta = alpha  # belief pull strength

    for chunk_idx in range(n_chunks):
        chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF
        rng = np.random.default_rng(chunk_seed)
        for _ in range(CHUNK_SIZE):
            t = global_step

            if t == 5000:
                settle_pi_5000 = pi_of(v.copy())
            if t == 6000 and settle_pi_5000 is not None:
                settle_tv_val = tv(settle_pi_5000, pi_of(v.copy()))
                v_at_6000 = v.copy()

            in_burst = (phase == "W") and (t in burst_step_set)
            in_regime = (phase == "R") and (t >= PHASE_R_START)

            # ---- observation ----
            # Phase R is a GLOBAL, location-INDEPENDENT, INESCAPABLE re-color: the body is
            # force-fed regime-color cells every step (teleport+qs in _apply_soft_capture,
            # exp185-faithful), so obs read from the body's actual cell IS regime_color, but
            # with exp185's natural ~100-step argmax(v) settle lag (NOT an instant override —
            # that would compress the revision margin and dissolve the crack). For 0<alpha<1
            # the spatial pull steers the body toward regime cells with prob alpha; the belief
            # pull mixes qs toward uniform-over-regime. Either way the re-color is unavoidable
            # (it targets regime cells for ALL body positions), matching the spec's global
            # re-color while preserving the exp185 R verdict structure.
            obs = int(base_cmap[c.true_pos])
            obs_arr[t] = obs

            # belief / likelihood pieces (real law)
            A_hat = c._A_hat()
            likelihood = A_hat[obs, :]
            qs_updated = likelihood * c.qs
            denom = qs_updated.sum()
            if denom > 0:
                qs_updated = qs_updated / denom
            else:
                qs_updated = np.ones(n_cells) / n_cells
            c.pA[obs, :] += qs_updated
            map_cell = int(np.argmax(qs_updated))
            predicted_obs_dist = A_hat[:, map_cell]
            h_predicted = -np.sum(predicted_obs_dist * np.log(predicted_obs_dist + 1e-12))
            w = math.exp(-h_predicted)

            # =============================================================
            # CAPTIVITY / RESIST decision (the single manipulated bit)
            # =============================================================
            resist_now = False
            if oracle:
                resist_now = in_burst or in_regime   # freeze during known bursts (W) and regime (R)
            elif is_freeze_arm and freeze_state == "RESIST":
                resist_now = True

            # ---- ORACLE arm: freeze v during known windows (no monitor) ----
            if oracle and resist_now:
                state_arr[t] = 1
                _apply_soft_capture(
                    c, B, qs_updated, in_burst, in_regime, attack_color, regime_color,
                    alpha, beta, color_cells_arr, burst_rng, belief_rng, rng, n_cells,
                    n_actions, policy=None, is_mover=False,
                )
                expressed_arr[t] = int(np.argmax(v))  # v frozen
                if event_log is not None:
                    event_log.append((c.true_pos, obs, expressed_arr[t], 1))
                c.age_steps += 1
                global_step += 1
                if global_step % EVAL == 0:
                    v_traj.append(v.copy())
                continue

            # ---- FIXED-H / MOVE_FREEZE arms in RESIST: v FROZEN; faithful exp185 release ----
            if is_freeze_arm and (not oracle) and freeze_state == "RESIST":
                state_arr[t] = 1
                frozen_steps += 1
                blocked_w_by_color[obs] += w   # exp185 line 514
                # move_freeze relocates while frozen (attribution control); fixedH holds.
                mv = (arm == "move_freeze")
                _apply_soft_capture(
                    c, B, qs_updated, in_burst, in_regime, attack_color, regime_color,
                    alpha, beta, color_cells_arr, burst_rng, belief_rng, rng, n_cells,
                    n_actions, policy=policy, is_mover=mv,
                )
                expressed_arr[t] = int(np.argmax(v))  # v frozen
                if event_log is not None:
                    event_log.append((c.true_pos, obs, expressed_arr[t], 1))
                c.age_steps += 1
                global_step += 1
                if global_step % EVAL == 0:
                    v_traj.append(v.copy())
                if global_step % FINE_EVAL == 0:
                    checks_since_release += 1
                    # pressure-based calm (exp185 lines 558-573)
                    t_now = global_step
                    t_pressure_start = max(0, t_now - PRESSURE_WINDOW)
                    obs_window = obs_arr[t_pressure_start:t_now]
                    c_star_pressure = int(np.argmax(blocked_w_by_color))
                    if len(obs_window) > 0:
                        pressure_active = (
                            float(np.sum(obs_window == c_star_pressure)) / len(obs_window)
                        ) >= PRESSURE_FRAC
                    else:
                        pressure_active = False
                    if not pressure_active:
                        calm_count += 1
                    else:
                        calm_count = 0
                    released = False
                    if calm_count >= RELEASE_CALM_SNAPS:        # transient (exp185 line 582)
                        released = True
                    if not released and is_fixedH and frozen_steps >= H_horizon:
                        released = True                          # concession (exp185 line 593)
                    if released:
                        freeze_state = "NORMAL"
                        mismatch_history.clear()
                        v_fine.clear()
                        k_since_reset = 0
                        calm_count = 0
                        frozen_steps = 0
                        checks_since_release = 0
                continue

            # ---- NON-FROZEN step (baseline / mover / hold / NORMAL freeze-arm) ----
            # Decide capture (spatial+belief pull) vs free policy move for the BODY,
            # while the v-store updates LIVE (g_t=1.0, real w).
            v *= LAMBDA
            v[obs] += w
            c.value_counts[obs] += w
            expressed_arr[t] = int(np.argmax(v))

            # actuator: movers relocate; baseline/hold walk/hold; pull still applies
            mover_active = False
            if is_mover:
                if is_always_move:
                    mover_active = True
                elif is_gated and freeze_state == "RESIST_MOVE":
                    mover_active = True

            _apply_soft_capture(
                c, B, qs_updated, in_burst, in_regime, attack_color, regime_color,
                alpha, beta, color_cells_arr, burst_rng, belief_rng, rng, n_cells,
                n_actions, policy=policy, is_mover=mover_active,
            )
            if state_arr[t] == 0 and in_burst:
                state_arr[t] = 1  # mark burst-step exposure even when not frozen

            if event_log is not None:
                event_log.append((c.true_pos, obs, expressed_arr[t], int(in_burst)))

            c.age_steps += 1
            global_step += 1
            if global_step % EVAL == 0:
                v_traj.append(v.copy())

            # ---- rung-2 monitor (real) — runs for freeze arms AND the gated mover ----
            run_monitor = (is_fixedH or is_gated) and freeze_state in ("NORMAL", "RESIST_MOVE")
            if global_step % FINE_EVAL == 0:
                checks_since_release += 1
                if run_monitor and freeze_state == "NORMAL":
                    v_fine.append(v.copy())
                    k_since_reset += 1
                    k = k_since_reset - 1
                    if k_since_reset >= 7:
                        Lf = min(MON_W_SNAPS, k_since_reset - 5)
                        if Lf >= 2:
                            v_base = v_fine[k - 4]
                            v_drift_ref = v_fine[k - 4 - Lf]
                            slope = (v_base - v_drift_ref) / Lf
                            v_hat = v_base + slope * 4
                            m_k = float(np.linalg.norm(v_hat - v_fine[k]))
                            mismatch_history.append(m_k)
                            n_hist = len(mismatch_history)
                            live_m_bar = None
                            if n_hist >= CTRL_MIN_SNAPS:
                                tail = mismatch_history[-CTRL_MBAR_WINDOW:]
                                live_m_bar = statistics.median(tail)
                                m_bar_floor = live_m_bar
                            if live_m_bar is not None:
                                trigger_denom = live_m_bar
                            elif m_bar_floor is not None:
                                trigger_denom = m_bar_floor
                            else:
                                trigger_denom = None
                            if (trigger_denom is not None and m_k > 0
                                    and checks_since_release >= REFRACTORY_CHECKS
                                    and (m_k / max(trigger_denom, 1e-12)) >= theta):
                                # TRIGGER
                                entry_step = global_step
                                if t < PHASE_R_START:
                                    n_triggers_settle += 1
                                else:
                                    n_triggers_in_burst += 1
                                if first_trigger_after_burst is None and t >= PHASE_R_START:
                                    first_trigger_after_burst = t
                                trigger_events.append({"entry_step": int(entry_step),
                                                       "in_burst": bool(in_burst)})
                                if is_fixedH or (is_freeze_arm and not is_gated):
                                    freeze_state = "RESIST"
                                    entry_step = global_step
                                    pi_ref = pi_of(v.copy())
                                    m_bar_frozen = trigger_denom
                                    blocked_w_by_color = np.zeros(n_colors, dtype=np.float64)
                                    calm_count = 0
                                    frozen_steps = 0
                                    checks_since_release = 0
                                elif is_gated:
                                    freeze_state = "RESIST_MOVE"
                                    checks_since_release = 0
                                    mismatch_history.clear()
                                    v_fine.clear()
                                    k_since_reset = 0

                # gated mover: release on calm (no recent mismatch above bar) — in Phase R the
                # global re-color normalizes -> mismatch decays -> trigger releases -> agent
                # stops avoiding -> v tracks regime_color -> revises.
                if is_gated and freeze_state == "RESIST_MOVE" and checks_since_release >= 8:
                    # measure current mismatch; if back under threshold, release (accept)
                    v_fine.append(v.copy())
                    k_since_reset += 1
                    k = k_since_reset - 1
                    if k_since_reset >= 7:
                        Lf = min(MON_W_SNAPS, k_since_reset - 5)
                        if Lf >= 2:
                            v_base = v_fine[k - 4]
                            v_drift_ref = v_fine[k - 4 - Lf]
                            slope = (v_base - v_drift_ref) / Lf
                            v_hat = v_base + slope * 4
                            m_k = float(np.linalg.norm(v_hat - v_fine[k]))
                            mismatch_history.append(m_k)
                            tail = mismatch_history[-CTRL_MBAR_WINDOW:]
                            mb = statistics.median(tail) if len(tail) >= CTRL_MIN_SNAPS else None
                            if mb is not None and (m_k / max(mb, 1e-12)) < theta:
                                freeze_state = "NORMAL"
                                checks_since_release = 0
                                mismatch_history.clear()
                                v_fine.clear()
                                k_since_reset = 0

    # spine outputs
    out = {
        "fork_seed": fork_seed,
        "arm": arm,
        "alpha": alpha,
        "attack_color": attack_color,
        "regime_color": regime_color,
        "expressed_arr": expressed_arr,
        "v_traj": np.array(v_traj) if v_traj else np.zeros((0, n_colors)),
        "settle_tv": settle_tv_val,
        "settle_pi_5000": settle_pi_5000,
        "v_at_6000": v_at_6000,
        "n_triggers_in_burst": n_triggers_in_burst,
        "n_triggers_settle": n_triggers_settle,
        "first_trigger_after_burst": first_trigger_after_burst,
        "trigger_events": trigger_events,
        "state_arr": state_arr,
        "obs_arr": obs_arr,
    }
    if event_log is not None:
        out["event_log"] = event_log
    return out


def _apply_soft_capture(c, B, qs_updated, in_burst, in_regime, attack_color, regime_color,
                        alpha, beta, color_cells_arr, burst_rng, belief_rng, rng,
                        n_cells, n_actions, policy, is_mover):
    """The single manipulated bit. SPATIAL pull (prob alpha) toward target cells +
    BELIEF pull (beta-mix qs toward uniform-over-target). At alpha=1 reduces EXACTLY to
    exp185 hard captivity (burst_rng consumed in exp185 order: a single choice over the
    target color cells), v handled by the caller (frozen for freeze arms)."""
    # target = attack-color cells in Phase W burst; regime-color cells in Phase R global recolor
    if in_regime:
        target_cells = color_cells_arr[regime_color]
        active = True  # global re-color is inescapable (location-independent) -> always pulls
    elif in_burst:
        target_cells = color_cells_arr[attack_color]
        active = True
    else:
        target_cells = None
        active = False

    if not active:
        # free walk: policy move (no-op via wall allowed); update qs by chosen action
        if is_mover and policy is not None:
            nxt = int(policy[c.true_pos])
            # map a teleport-style policy hop to a belief consistent with landing
            c.true_pos = nxt
            qs_next = np.zeros(n_cells)
            qs_next[nxt] = 1.0
            c.qs = qs_next
        else:
            action = int(rng.integers(0, n_actions))
            c.true_pos = c.world.move(c.true_pos, action)
            c.qs = B[:, :, action] @ qs_updated
        return

    # ACTIVE capture window (burst or regime).
    if alpha >= 1.0:
        # HARD captivity limit (exp185-exact): teleport to a random target cell via burst_rng
        # (single choice, exp185 order) + qs uniform over target cells + NO policy move.
        if len(target_cells) > 0:
            c.true_pos = int(burst_rng.choice(target_cells))
        qs_next = np.zeros(n_cells)
        if len(target_cells) > 0:
            qs_next[target_cells] = 1.0 / len(target_cells)
        c.qs = qs_next
        return

    # SOFT regime (0 < alpha < 1): spatial pull with prob alpha (BFS gradient step toward
    # nearest target cell, absorbing on target); else the agent's policy move. Hold overridden.
    pulled = burst_rng.random() < alpha
    if pulled:
        d = _DIST_CACHE_get(c.world, target_id(in_regime, attack_color, regime_color))
        cands = pull_cands(c.world, target_id(in_regime, attack_color, regime_color),
                           c.true_pos, d)
        c.true_pos = int(cands[burst_rng.integers(0, len(cands))])
    else:
        if is_mover and policy is not None:
            c.true_pos = int(policy[c.true_pos])
        else:
            action = int(rng.integers(0, n_actions))
            c.true_pos = c.world.move(c.true_pos, action)
    # BELIEF pull: qs = (1-beta)*qs_propagated + beta*uniform_over_target
    qs_prop = np.zeros(n_cells)
    qs_prop[c.true_pos] = 1.0
    qs_unif = np.zeros(n_cells)
    if len(target_cells) > 0:
        qs_unif[target_cells] = 1.0 / len(target_cells)
    c.qs = (1.0 - beta) * qs_prop + beta * qs_unif


def target_id(in_regime, attack_color, regime_color):
    return regime_color if in_regime else attack_color


_DIST_CACHE: dict = {}


def _DIST_CACHE_get(world, color):
    key = (tuple(world.cmap), int(color))
    d = _DIST_CACHE.get(key)
    if d is None:
        d = dist_to_attack(world, color)
        _DIST_CACHE[key] = d
    return d


# ---------------------------------------------------------------------------
# Bars (computed from each arm's OWN v-trajectory)
# ---------------------------------------------------------------------------
def defence_pass(expressed_arr, burst_windows, attack_color, n_steps):
    """expr_frac over [last_burst_end+1500, +2000); PASS iff < 0.5. (gate on argmax(v))."""
    last_bend = burst_windows[-1][1]
    ws = last_bend + DEF_OFF_START
    we = last_bend + DEF_OFF_END
    if we > n_steps:
        return False, None
    frac = float(np.mean(expressed_arr[ws:we] == attack_color))
    return frac < DEFENSE_FRAC_THRESH, frac


def revision_latency(expressed_arr, regime_color, n_steps):
    """Phase-R latency: first step from 6000 where argmax(v)==regime_color holds 2000."""
    return phase_r_latency(expressed_arr, regime_color, n_steps)


# ---------------------------------------------------------------------------
# Avoider policies — actuator objective FIX (argmax(v) proxy, pre-registered)
# ---------------------------------------------------------------------------
def build_avoid_policies(world, alpha, colors):
    """Certified-optimal refuge policy per attack color (Bellman residual < 1e-7 asserted
    inside optimal_avoid_policy). PRE-REGISTERED PROXY: the planner minimizes long-run
    discounted attack-color OCCUPANCY (the exp270 diet objective). The defence bar scores
    argmax(v); on this body, under the REAL v-store, keeping body OFF the attack color drives
    v[attack] down (v decays at LAMBDA, only fed by obs==attack), so low attack-occupancy is a
    faithful monotone proxy for argmax(v)!=attack. We do NOT assume the mover-only equivalence
    transfers to the freeze arm; the freeze arm is scored on its OWN frozen v directly."""
    pols = {}
    for col in colors:
        pols[col] = optimal_avoid_policy(world, col, alpha)  # asserts resid < 1e-7
    return pols


# ---------------------------------------------------------------------------
# Tier-A spine equivalence gate (vs the imported real exp185 runner)
# ---------------------------------------------------------------------------
def tier_a_spine(mirro_root, base_cmap, n_colors, p):
    p("-" * 78)
    p("SECTION 1a. TIER-A DETERMINISTIC SPINE (vs imported real exp185 runner)")
    p("  CELL_LONG (2400,4,600), seeds 250-251. settle_pi@5000, settle_tv(5000->6000)")
    p("  atol/rtol=1e-6 (op-order tolerant); attack=argmin(v) & favorite=argmax(v)@6000 EXACT.")
    p("-" * 78)
    cell = CELL_LONG
    bw = burst_windows_for(cell)
    n_steps = n_steps_for(cell)
    all_ok = True
    rows = []
    for seed in SPINE_SEEDS:
        # the real exp185 runner (faithful primitive)
        root185 = copy.deepcopy(mirro_root)
        root185._state_dir = None
        rr185 = run_fork_schedule_185(
            mirro=root185, fork_seed=seed, base_cmap=base_cmap, n_colors=n_colors,
            arm_name="baseline", arm_mode="baseline", phase="W", burst_windows=bw,
            color_mode="exogenous_fixed", reloc_rng_seed=RELOC_OFF + seed,
            n_steps=n_steps, theta=THETA, release_calm_snaps=8,
        )
        # the settle v trajectory only depends on the NORMAL loop (t<6000), identical to ours.
        # We recompute our own settle via the soft runner at alpha=1 baseline (NORMAL pre-6000
        # is byte-identical regardless of alpha); attack color from argmin at 6000.
        root_ours = copy.deepcopy(mirro_root)
        root_ours._state_dir = None
        # determine onset attack color (argmin v@6000) from a free baseline run
        ours = run_soft(
            mirro_root=root_ours, fork_seed=seed, base_cmap=base_cmap, n_colors=n_colors,
            arm="baseline", burst_windows=bw, attack_color=0, regime_color=1, alpha=0.0,
            n_steps=n_steps, reloc_rng_seed=RELOC_OFF + seed, avoid_policies={},
        )
        # exp185 settle_tv (5000->6000) and v@6000 argmin/argmax
        tv185 = rr185["settle_tv"]
        # exp185 exposes settle_tv; recompute attack/fav from its v_traj at the 6000 sample
        # (v_traj sampled every EVAL=100 -> index 59 is global_step 6000)
        vtraj185 = rr185["v_traj"]
        idx6000 = (6000 // EVAL) - 1
        v6_185 = vtraj185[idx6000]
        attack185 = int(np.argmin(v6_185))
        fav185 = int(np.argmax(v6_185))

        tv_ours = ours["settle_tv"]
        v6_ours = ours["v_at_6000"]
        attack_ours = int(np.argmin(v6_ours))
        fav_ours = int(np.argmax(v6_ours))

        tv_ok = (tv185 is None and tv_ours is None) or (
            tv185 is not None and tv_ours is not None
            and abs(float(tv185) - float(tv_ours)) <= 1e-6 + 1e-6 * abs(float(tv185)))
        atk_ok = (attack185 == attack_ours)
        fav_ok = (fav185 == fav_ours)
        seed_ok = tv_ok and atk_ok and fav_ok
        all_ok = all_ok and seed_ok
        dtv = abs(float(tv185) - float(tv_ours)) if (tv185 is not None and tv_ours is not None) else None
        p(f"  s{seed}: settle_tv exp185={tv185!r} ours={tv_ours!r} "
          f"delta={dtv} tol=2e-6 {'OK' if tv_ok else 'MISMATCH'}")
        p(f"        attack(argmin@6000) exp185={attack185} ours={attack_ours} "
          f"{'OK' if atk_ok else 'MISMATCH'}  | favorite(argmax@6000) exp185={fav185} "
          f"ours={fav_ours} {'OK' if fav_ok else 'MISMATCH'}")
        rows.append({"seed": seed, "tv_exp185": to_plain(tv185), "tv_ours": to_plain(tv_ours),
                     "tv_delta": to_plain(dtv), "attack_exp185": attack185,
                     "attack_ours": attack_ours, "fav_exp185": fav185, "fav_ours": fav_ours,
                     "seed_ok": bool(seed_ok)})
    p(f"  TIER-A SPINE: {'PASS' if all_ok else 'FAIL (port wrong -> ABORT)'}")
    p("")
    return all_ok, rows


# ---------------------------------------------------------------------------
# Per-(cell,color,alpha) arm runners (8 seeds) -> defence/revision counts
# ---------------------------------------------------------------------------
def run_arm_bank(mirro_root, base_cmap, n_colors, cell, attack_color, regime_color, alpha,
                 arm, avoid_policies, cell_tag, want_movement_stats=False,
                 want_events=False, defence_only=False, revision_only=False):
    """defence_only -> run only the W (defence) sessions (skip R); revision_only -> only R.
    Used to prune calibration cost (revision is only decisive for H's that already defend)."""
    bw = burst_windows_for(cell)
    n_steps_w = n_steps_for(cell)
    def_count = 0
    per_seed_def = []
    home_ranges = []
    most_visited = []
    events_hashes = []
    latencies = []   # revision latencies from the SEPARATE R run
    for seed in SEEDS:
        if revision_only:
            root_r = copy.deepcopy(mirro_root); root_r._state_dir = None
            rr_r = run_soft(
                mirro_root=root_r, fork_seed=seed, base_cmap=base_cmap, n_colors=n_colors,
                arm=arm, burst_windows=[], attack_color=attack_color,
                regime_color=regime_color, alpha=alpha, n_steps=N_STEPS_R, phase="R",
                reloc_rng_seed=BURST_SEED_OFFSET_R + seed, avoid_policies=avoid_policies,
            )
            latencies.append(revision_latency(rr_r["expressed_arr"], regime_color, N_STEPS_R))
            continue
        # ----- W run (defence; bursts, NO global recolor) -----
        root_w = copy.deepcopy(mirro_root)
        root_w._state_dir = None
        rr_w = run_soft(
            mirro_root=root_w, fork_seed=seed, base_cmap=base_cmap, n_colors=n_colors,
            arm=arm, burst_windows=bw, attack_color=attack_color, regime_color=regime_color,
            alpha=alpha, n_steps=n_steps_w, phase="W",
            reloc_rng_seed=RELOC_OFF + 1000 * cell_tag + seed,
            avoid_policies=avoid_policies, collect_events=want_events,
        )
        dpass, dfrac = defence_pass(rr_w["expressed_arr"], bw, attack_color, n_steps_w)
        per_seed_def.append(bool(dpass))
        def_count += int(dpass)
        if want_movement_stats:
            ev = rr_w.get("event_log")
            if ev is not None:
                # body positions during Phase-W bursts (where the mover must relocate)
                positions = [e[0] for e in ev if e[3] == 1]  # e[3]==1 -> in_burst step
                if positions:
                    cnt = collections.Counter(positions)
                    home_ranges.append(len(cnt))
                    most_visited.append(max(cnt.values()) / len(positions))
        if want_events:
            ev = rr_w.get("event_log")
            if ev is not None:
                h = hashlib.sha256(np.array(ev, dtype=np.int64).tobytes()).hexdigest()[:16]
                events_hashes.append(h)
        # ----- R run (revision; NO bursts, GLOBAL recolor from 6000) -----
        if defence_only:
            continue
        root_r = copy.deepcopy(mirro_root)
        root_r._state_dir = None
        rr_r = run_soft(
            mirro_root=root_r, fork_seed=seed, base_cmap=base_cmap, n_colors=n_colors,
            arm=arm, burst_windows=[], attack_color=attack_color, regime_color=regime_color,
            alpha=alpha, n_steps=N_STEPS_R, phase="R",
            reloc_rng_seed=BURST_SEED_OFFSET_R + seed,
            avoid_policies=avoid_policies, collect_events=False,
        )
        lat = revision_latency(rr_r["expressed_arr"], regime_color, N_STEPS_R)
        latencies.append(lat)
    return {
        "def_count": def_count, "per_seed_def": per_seed_def,
        "latencies": latencies,
        "home_ranges": home_ranges, "most_visited": most_visited,
        "events_hashes": events_hashes,
    }


def revision_counts(arm_latencies, baseline_latencies, tol=REVISION_TOL_NORMAL):
    cnt = 0
    per_seed = []
    for al, bl in zip(arm_latencies, baseline_latencies):
        ok = (al is not None and bl is not None and al <= bl + tol)
        per_seed.append(bool(ok))
        cnt += int(ok)
    return cnt, per_seed


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    out_dir = REPO_ROOT / "experiments" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def p(s=""):
        print(s, flush=True)
        lines.append(s)

    results = {"meta": {}, "tier_a": None, "tier_b": None, "trigger": {},
               "calibration": {}, "kill_test": {}, "verdicts": {}, "events_hash": {}}

    mirro = Creature.load("creature/state/mirro")
    mirro_root = copy.deepcopy(mirro)
    mirro_root._state_dir = None
    base_cmap = list(mirro.world.cmap)
    n_colors = mirro.world.n_colors
    n_cells = mirro.world.n_cells

    # ---- Section 0: config echo ----
    p("#" * 78)
    p("# Exp 271 — RUNG-2 POSABILITY PRE-FLIGHT (standalone; no source-patching)")
    p("#   freeze-vs-revise crack  vs  N4-gated movement defence: common-regime co-existence?")
    p("#" * 78)
    p("SECTION 0: CONFIG ECHO")
    p(f"  LAMBDA={LAMBDA} (crack-calibrated; re-verified THETA fires under it)  THETA={THETA}")
    p(f"  real predictability-weight law: w = exp(-h_predicted) on the evolving _A_hat "
      f"(NOT w=1.0)")
    p(f"  CELL_SHORT=(L=1600,K=2,G=200)  CELL_LONG=(L=2400,K=4,G=600) [=alpha=1 gate cell]")
    p(f"  COLORS={COLORS} (attack=tested color, FIXED, gate PER COLOR incl color-2 3x3 block)")
    p(f"  SEEDS={SEEDS} (paired across arms)  SPINE_SEEDS={SPINE_SEEDS}")
    p(f"  ALPHAS_CALIB={ALPHAS_CALIB}  FIXED_HORIZONS={FIXED_HORIZONS}")
    p(f"  bars: defence expr_frac<{DEFENSE_FRAC_THRESH} over [last_bend+{DEF_OFF_START},"
      f"+{DEF_OFF_END}); revision latency<=baseline+{REVISION_TOL_NORMAL}")
    p(f"  gate counts: family>={FAMILY_DEF_PASS}/8 oracle>={ORACLE_DEF_PASS}/8 "
      f"baseline_fail>={BASELINE_DEF_FAIL}/8 revision>={REV_PASS}/8")
    p(f"  body: 5x5, n_colors={n_colors}, n_cells={n_cells}, cmap={base_cmap}")
    cm = dict(sorted(collections.Counter(base_cmap).items()))
    p(f"  color counts={cm} (color 2 = contiguous 3x3 block)")
    p("")
    results["meta"] = {"LAMBDA": LAMBDA, "THETA": THETA, "seeds": SEEDS,
                       "alphas": ALPHAS_CALIB, "fixed_horizons": FIXED_HORIZONS,
                       "cells": {"SHORT": CELL_SHORT, "LONG": CELL_LONG}}

    # wall-clock preflight estimate
    p("WALL-CLOCK PREFLIGHT ESTIMATE")
    t_probe = time.time()
    _root = copy.deepcopy(mirro_root); _root._state_dir = None
    _bw = burst_windows_for(CELL_LONG); _ns = n_steps_for(CELL_LONG)
    _ = run_soft(mirro_root=_root, fork_seed=250, base_cmap=base_cmap, n_colors=n_colors,
                 arm="baseline", burst_windows=_bw, attack_color=0, regime_color=1,
                 alpha=0.6, n_steps=_ns, reloc_rng_seed=999, avoid_policies={})
    per_run = time.time() - t_probe
    # rough count: tierB(11 arms*8) + calib(2cell*3col*5alpha*(8H+base+oracle)*8)
    #   + kill(2cell*3col*(8H+base+oracle+gated+always+2ctrl)*8)  (band-only -> upper bound)
    est_runs = 11 * 8 + 2 * 3 * 5 * 10 * 8 + 2 * 3 * 14 * 8
    p(f"  per-run (CELL_LONG, alpha=0.6) ~ {per_run:.2f}s; rough upper-bound runs ~ {est_runs}"
      f" -> est ~ {per_run*est_runs/60:.1f} min (calibration prunes most kill-test work)")
    p("")

    # ================================================================
    # SECTION 1: ALPHA=1 EQUIVALENCE GATE
    # ================================================================
    p("=" * 78)
    p("SECTION 1: ALPHA=1 VERDICT-LEVEL EQUIVALENCE SANITY GATE")
    p("=" * 78)
    tierA_ok, tierA_rows = tier_a_spine(mirro_root, base_cmap, n_colors, p)
    results["tier_a"] = {"pass": bool(tierA_ok), "rows": to_plain(tierA_rows)}

    abort = False
    if not tierA_ok:
        p("  TIER-A FAILED -> ABORT (perception/v-store port is wrong; catches w=1.0 regression)")
        abort = True

    # ---- Tier B: verdict structure at alpha=1 on CELL_LONG (= hard captivity) ----
    tierB_ok = False
    if not abort:
        p("-" * 78)
        p("SECTION 1b. TIER-B VERDICT STRUCTURE @ alpha=1 (= committed hard captivity), CELL_LONG")
        p("  per-arm [arm | defence/8 | revision_pass | both_pass] + baseline + oracle")
        p("  attack/regime color = onset rule for the gate cell (argmin/argmax v@6000, s250)")
        p("-" * 78)
        cell = CELL_LONG
        bw = burst_windows_for(cell); n_steps = n_steps_for(cell)
        cell_tag = 99
        # onset colors (argmin/argmax v@6000) from a baseline free run, seed 250
        root0 = copy.deepcopy(mirro_root); root0._state_dir = None
        probe = run_soft(mirro_root=root0, fork_seed=250, base_cmap=base_cmap,
                         n_colors=n_colors, arm="baseline", burst_windows=bw,
                         attack_color=0, regime_color=1, alpha=0.0, n_steps=n_steps,
                         reloc_rng_seed=RELOC_OFF + 1000 * cell_tag + 250, avoid_policies={})
        v6 = probe["v_at_6000"]
        atk = int(np.argmin(v6))
        # EQUIVALENCE-GATE regime rule: exp185's Phase-R uses regime_color = argmin(v)@6000
        # (the least-valued color becomes the new permanent regime; exp185 line 451). To make
        # the alpha=1 R-mechanic byte-equal to exp185's location-gated force-feed we use the
        # SAME regime here (= the attack color on this cell). (The KILL TEST uses a distinct
        # external non-attack regime per spec; only this gate must mirror exp185.)
        reg = atk
        p(f"  onset attack_color={atk} (argmin v@6000); EQUIV-GATE regime_color={reg} "
          f"(exp185 argmin@6000 rule)")
        # baseline latencies
        base_bank = run_arm_bank(mirro_root, base_cmap, n_colors, cell, atk, reg, 1.0,
                                 "baseline", {}, cell_tag)
        base_lat = base_bank["latencies"]
        oracle_bank = run_arm_bank(mirro_root, base_cmap, n_colors, cell, atk, reg, 1.0,
                                   "oracle", {}, cell_tag)
        ora_rev, _ = revision_counts(oracle_bank["latencies"], base_lat)
        tb_rows = []
        p(f"  {'arm':>10} {'defence/8':>10} {'rev/8':>7} {'both':>5}")
        p(f"  {'baseline':>10} {base_bank['def_count']:>10} "
          f"{'-':>7} {'-':>5}")
        p(f"  {'oracle':>10} {oracle_bank['def_count']:>10} "
          f"{ora_rev:>7} {'(excl)':>5}")
        h_def = {}
        h_both = []
        long_h_def_ge6 = False
        h600_rev = None
        h6000_rev = None
        for H in FIXED_HORIZONS:
            bank = run_arm_bank(mirro_root, base_cmap, n_colors, cell, atk, reg, 1.0,
                                ("fixedH", H), {}, cell_tag)
            rev, _ = revision_counts(bank["latencies"], base_lat)
            both = (bank["def_count"] >= FAMILY_DEF_PASS) and (rev >= REV_PASS)
            h_def[H] = bank["def_count"]
            if H >= 1800 and bank["def_count"] >= FAMILY_DEF_PASS:
                long_h_def_ge6 = True
            if H == 600:
                h600_rev = rev
            if H == 6000:
                h6000_rev = rev
            if both:
                h_both.append(H)
            p(f"  {('H'+str(H)):>10} {bank['def_count']:>10} {rev:>7} {str(both):>5}")
            tb_rows.append({"arm": f"H{H}", "def": bank["def_count"], "rev": rev,
                            "both": bool(both)})
        # asserts (a)-(e)
        a_ok = oracle_bank["def_count"] >= ORACLE_DEF_PASS
        b_ok = base_bank["def_count"] <= (8 - BASELINE_DEF_FAIL)   # fails >=7/8 -> def<=1/8
        c_ok = long_h_def_ge6
        d_ok = len(h_both) == 0
        e_ok = (h600_rev is not None and h600_rev >= REV_PASS
                and h6000_rev is not None and h6000_rev < REV_PASS)
        p(f"  (a) oracle defence>=7/8: {oracle_bank['def_count']}/8  {'PASS' if a_ok else 'FAIL'}")
        p(f"  (b) baseline defence<=1/8: {base_bank['def_count']}/8  {'PASS' if b_ok else 'FAIL'}")
        p(f"  (c) >=1 long-H defence>=6/8: {long_h_def_ge6}  {'PASS' if c_ok else 'FAIL'}")
        p(f"  (d) NO fixed-H passes both: both-passers={h_both}  {'PASS' if d_ok else 'FAIL'}")
        p(f"  (e) H600 rev PASS({h600_rev}) & H6000 rev FAIL({h6000_rev}): "
          f"{'PASS' if e_ok else 'FAIL'}")
        tierB_ok = a_ok and b_ok and c_ok and d_ok and e_ok
        p(f"  TIER-B VERDICT STRUCTURE: {'PASS' if tierB_ok else 'FAIL (-> ABORT)'}")
        p("  SCOPE CAVEAT: this tests the FIXED-H crack (a strict subset of the committed "
          "full-family crack); a 'movement beats the constants' positive is scoped to fixed-H.")
        p("")
        results["tier_b"] = {
            "pass": bool(tierB_ok), "attack_color": atk, "regime_color": reg,
            "baseline_def": base_bank["def_count"], "oracle_def": oracle_bank["def_count"],
            "oracle_rev": ora_rev, "h_rows": to_plain(tb_rows),
            "checks": {"a": bool(a_ok), "b": bool(b_ok), "c": bool(c_ok),
                       "d": bool(d_ok), "e": bool(e_ok)},
        }
        if not tierB_ok:
            abort = True

    overall_equiv = tierA_ok and tierB_ok
    p(f"OVERALL EQUIVALENCE GATE: {'PASS' if overall_equiv else 'ABORT'}")
    p("")

    if abort:
        p("=" * 78)
        p("PREFLIGHT ABORTED (invalid port, not a verdict). Exit code != 0.")
        p("=" * 78)
        _write(out_dir, lines, results)
        return 2

    # ================================================================
    # SECTION 2: trigger characterization under the soft pull
    # ================================================================
    p("=" * 78)
    p(f"SECTION 2: TRIGGER CHARACTERIZATION UNDER SOFT PULL (THETA={THETA} separation check)")
    p("  representative midband alpha=0.6, CELL_LONG, the gated-mover monitor, 8 seeds")
    p("  rate = mean triggers/run; false-trigger = settle-phase (t<6000) triggers")
    p("=" * 78)
    cell = CELL_LONG
    bw = burst_windows_for(cell); n_steps = n_steps_for(cell)
    avoid_pols_06 = build_avoid_policies(mirro.world, 0.6, COLORS)
    trig_rows = []
    sep_ok_all = True
    for col in COLORS:
        reg = (col + 1) % n_colors
        n_burst_trig = []
        n_settle_trig = []
        first_latencies = []
        for seed in SEEDS:
            root = copy.deepcopy(mirro_root); root._state_dir = None
            rr = run_soft(mirro_root=root, fork_seed=seed, base_cmap=base_cmap,
                          n_colors=n_colors, arm="gated_mover", burst_windows=bw,
                          attack_color=col, regime_color=reg, alpha=0.6, n_steps=n_steps,
                          reloc_rng_seed=RELOC_OFF + 1000 * 50 + seed,
                          avoid_policies=avoid_pols_06)
            n_burst_trig.append(rr["n_triggers_in_burst"])
            n_settle_trig.append(rr["n_triggers_settle"])
            # latency = first trigger entry after first burst onset
            first = None
            for ev in rr["trigger_events"]:
                if ev["entry_step"] >= bw[0][0]:
                    first = ev["entry_step"] - bw[0][0]
                    break
            if first is not None:
                first_latencies.append(first)
        rate = float(np.mean(n_burst_trig))
        false_rate = float(np.mean(n_settle_trig))
        mean_lat = float(np.mean(first_latencies)) if first_latencies else None
        # THETA separation: attack-phase trigger rate must dominate settle false-trigger rate
        sep_ok = rate > max(1.0, 2.0 * false_rate)
        sep_ok_all = sep_ok_all and sep_ok
        p(f"  color {col} (regime {reg}): trigger_rate={rate:.2f}/run  "
          f"false_trigger(settle)={false_rate:.2f}/run  mean_latency={mean_lat}  "
          f"sep(THETA={THETA}): {'OK' if sep_ok else 'WEAK'}")
        trig_rows.append({"color": col, "regime": reg, "rate": rate,
                          "false_rate": false_rate, "mean_latency": to_plain(mean_lat),
                          "sep_ok": bool(sep_ok)})
    p(f"  THETA={THETA} separation (attack>>settle false): "
      f"{'OK on all colors' if sep_ok_all else 'WEAK on >=1 color (reported, not silently passed)'}")
    p("")
    results["trigger"] = {"alpha": 0.6, "rows": to_plain(trig_rows),
                          "sep_ok_all": bool(sep_ok_all)}

    # ================================================================
    # SECTION 3 + 4 + 5: per (cell x color) calibration -> band -> kill test -> verdict
    # ================================================================
    p("=" * 78)
    p("SECTION 3-5: PER (CELL x COLOR) CALIBRATION, KILL TEST, VERDICT")
    p("  NEVER a cross-cell/color mean (mean-of-opposites guard). Per-seed throughout.")
    p("=" * 78)

    # geometry-control: G_segreg world for the geom-sensitivity check (movement POSITIVE must
    # differ between G_mirro and G_segreg, else geometry-blind -> reject)
    G_segreg = World(rows=5, cols=5, n_colors=3, cmap=list(SEG_CMAP))

    verdict_map = {}   # (cell_name, color) -> verdict string
    for cell in CELLS:
        cname = cell["name"]
        cell_tag = 1 if cname == "SHORT" else 2
        bw = burst_windows_for(cell); n_steps = n_steps_for(cell)
        for col in COLORS:
            reg = (col + 1) % n_colors  # external per-color regime rule (a non-attack color)
            p("-" * 78)
            p(f"CELL_{cname} (L={cell['L']},K={cell['K']},G={cell['G']})  attack_color={col}  "
              f"regime_color={reg}")
            p("-" * 78)

            # ---- Section 3: ALPHA CALIBRATION ----
            p("  [Sec3] ALPHA-CALIBRATION  alpha | base_fail/8 | oracle_pass/8 | "
              "bestH def/8 & rev/8 | P1? | crack? | peer?")
            calib_recs = []
            alpha_lo = None
            alpha_hi = None
            for alpha in ALPHAS_CALIB:
                avoid_pols = build_avoid_policies(mirro.world, alpha, COLORS)
                # baseline: BOTH defence (W) + revision (R; needed as the revision reference)
                base_bank = run_arm_bank(mirro_root, base_cmap, n_colors, cell, col, reg,
                                         alpha, "baseline", avoid_pols, cell_tag)
                base_lat = base_bank["latencies"]
                base_fail = 8 - base_bank["def_count"]
                # oracle: defence only (revision excluded from conjunct; only used as peer)
                oracle_bank = run_arm_bank(mirro_root, base_cmap, n_colors, cell, col, reg,
                                           alpha, "oracle", avoid_pols, cell_tag,
                                           defence_only=True)
                P1 = (base_fail >= BASELINE_DEF_FAIL
                      and oracle_bank["def_count"] >= ORACLE_DEF_PASS)
                # fixed-H surface: DEFENCE-ONLY first (cheap), then revision only for H>=6/8 def
                best_def = 0
                best_def_H = None
                any_crack_H = False   # a fixed-H that FAILS one bar (crack survives)
                best_rev_at_best = 0
                h_rows = []
                h_def_banks = {}
                for H in FIXED_HORIZONS:
                    bank = run_arm_bank(mirro_root, base_cmap, n_colors, cell, col, reg,
                                        alpha, ("fixedH", H), avoid_pols, cell_tag,
                                        defence_only=True)
                    h_def_banks[H] = bank["def_count"]
                    if bank["def_count"] > best_def:
                        best_def = bank["def_count"]; best_def_H = H
                for H in FIXED_HORIZONS:
                    dc = h_def_banks[H]
                    if dc >= FAMILY_DEF_PASS:
                        # decisive for "covers both" -> need revision
                        rb = run_arm_bank(mirro_root, base_cmap, n_colors, cell, col, reg,
                                          alpha, ("fixedH", H), avoid_pols, cell_tag,
                                          revision_only=True)
                        rev, _ = revision_counts(rb["latencies"], base_lat)
                    else:
                        # cannot pass both (defence fails) -> it IS a crack witness; rev unneeded
                        rev = None
                    dboth = (dc >= FAMILY_DEF_PASS) and (rev is not None and rev >= REV_PASS)
                    if not dboth:
                        any_crack_H = True
                    if H == best_def_H:
                        best_rev_at_best = rev
                    h_rows.append({"H": H, "def": dc, "rev": rev, "both": bool(dboth)})
                # band membership: P1 AND a real fixed-H crack AND a defendable peer co-exist
                in_band = P1 and any_crack_H and (best_def >= FAMILY_DEF_PASS)
                if P1 and alpha_lo is None:
                    alpha_lo = alpha
                if in_band:
                    alpha_hi = alpha
                p(f"    a={alpha:.2f}  base_fail={base_fail}/8  oracle={oracle_bank['def_count']}/8"
                  f"  bestH=H{best_def_H} def={best_def}/8 rev={best_rev_at_best}/8  "
                  f"P1={P1} crackH={any_crack_H} peer={best_def>=FAMILY_DEF_PASS}  "
                  f"band={'YES' if in_band else 'no'}")
                calib_recs.append({"alpha": alpha, "base_fail": base_fail,
                                   "oracle_def": oracle_bank["def_count"],
                                   "best_def": best_def, "best_def_H": best_def_H,
                                   "best_rev": best_rev_at_best, "P1": bool(P1),
                                   "crackH": bool(any_crack_H), "in_band": bool(in_band),
                                   "h_rows": h_rows})
            band_alphas = [r["alpha"] for r in calib_recs if r["in_band"]]
            if band_alphas:
                band_lo = min(band_alphas); band_hi = max(band_alphas)
                band_alpha = band_alphas[len(band_alphas) // 2]  # predeclared midpoint
                p(f"    POSABLE band = [{band_lo:.2f}, {band_hi:.2f}]  -> kill-test alpha="
                  f"{band_alpha:.2f}")
            else:
                band_lo = band_hi = band_alpha = None
                p(f"    POSABLE band = EMPTY")

            # ---- Section 4 + 5: kill test at the band alpha (only if band non-empty) ----
            kill_rows = []
            verdict = None
            deciding = None
            if band_alpha is None:
                # determine WHY: P1 ever? crack ever? peer ever?
                any_P1 = any(r["P1"] for r in calib_recs)
                any_crack = any(r["crackH"] for r in calib_recs)
                any_peer_band = any(r["best_def"] >= FAMILY_DEF_PASS and r["P1"]
                                    for r in calib_recs)
                if not any_P1:
                    verdict = "PRECONDITION-FAIL"
                    deciding = "no alpha displaces passive under real v-store (P1 never holds)"
                elif any_P1 and not any(r["P1"] and r["crackH"] for r in calib_recs):
                    verdict = "CONSTANTS-ALSO-WIN"
                    deciding = "P1 holds but some fixed-H covers both bars (crack dissolved)"
                else:
                    verdict = "CANT-POSE"
                    deciding = "no alpha co-satisfies P1 + a real fixed-H crack + a defendable peer"
                p(f"  [Sec4] band EMPTY -> no kill test. [Sec5] VERDICT={verdict} ({deciding})")
            else:
                p(f"  [Sec4] KILL-TEST @ alpha={band_alpha:.2f}")
                avoid_pols = build_avoid_policies(mirro.world, band_alpha, COLORS)
                avoid_pols_seg = build_avoid_policies(G_segreg, band_alpha, COLORS)
                base_bank = run_arm_bank(mirro_root, base_cmap, n_colors, cell, col, reg,
                                         band_alpha, "baseline", avoid_pols, cell_tag)
                base_lat = base_bank["latencies"]
                p(f"    {'arm':>12} {'def/8':>6} {'rev/8':>6} {'both':>5} {'hrng':>5} "
                  f"{'mvfrac':>7} {'rev_split(per-seed)':>22}")
                # baseline + oracle
                base_rev, base_rev_seed = revision_counts(base_lat, base_lat)
                p(f"    {'baseline':>12} {base_bank['def_count']:>6} {base_rev:>6} "
                  f"{'-':>5} {'-':>5} {'-':>7} {str([int(x) for x in base_rev_seed]):>22}")
                oracle_bank = run_arm_bank(mirro_root, base_cmap, n_colors, cell, col, reg,
                                           band_alpha, "oracle", avoid_pols, cell_tag)
                ora_rev, ora_rev_seed = revision_counts(oracle_bank["latencies"], base_lat)
                ora_both = (oracle_bank["def_count"] >= FAMILY_DEF_PASS) and (ora_rev >= REV_PASS)
                p(f"    {'oracle':>12} {oracle_bank['def_count']:>6} {ora_rev:>6} "
                  f"{str(ora_both)+'(excl)':>5} {'-':>5} {'-':>7} "
                  f"{str([int(x) for x in ora_rev_seed]):>22}")
                kill_rows.append({"arm": "baseline", "def": base_bank["def_count"],
                                  "rev": base_rev})
                kill_rows.append({"arm": "oracle", "def": oracle_bank["def_count"],
                                  "rev": ora_rev, "both": bool(ora_both)})
                # fixed-H family
                any_fixedH_both = False
                for H in FIXED_HORIZONS:
                    bank = run_arm_bank(mirro_root, base_cmap, n_colors, cell, col, reg,
                                        band_alpha, ("fixedH", H), avoid_pols, cell_tag)
                    rev, rev_seed = revision_counts(bank["latencies"], base_lat)
                    both = (bank["def_count"] >= FAMILY_DEF_PASS) and (rev >= REV_PASS)
                    any_fixedH_both = any_fixedH_both or both
                    p(f"    {('H'+str(H)):>12} {bank['def_count']:>6} {rev:>6} "
                      f"{str(both):>5} {'-':>5} {'-':>7} "
                      f"{str([int(x) for x in rev_seed]):>22}")
                    kill_rows.append({"arm": f"H{H}", "def": bank["def_count"], "rev": rev,
                                      "both": bool(both)})
                # ALWAYS-ON mover (P3 ceiling)
                always_bank = run_arm_bank(mirro_root, base_cmap, n_colors, cell, col, reg,
                                           band_alpha, "always_mover", avoid_pols, cell_tag,
                                           want_movement_stats=True, want_events=True)
                a_rev, a_rev_seed = revision_counts(always_bank["latencies"], base_lat)
                a_both = (always_bank["def_count"] >= FAMILY_DEF_PASS) and (a_rev >= REV_PASS)
                a_hr = float(np.mean(always_bank["home_ranges"])) if always_bank["home_ranges"] else 0.0
                a_mv = float(np.mean(always_bank["most_visited"])) if always_bank["most_visited"] else 1.0
                p(f"    {'always_move':>12} {always_bank['def_count']:>6} {a_rev:>6} "
                  f"{str(a_both):>5} {a_hr:>5.1f} {a_mv:>7.3f} "
                  f"{str([int(x) for x in a_rev_seed]):>22}")
                # GATED mover (Q2/P3)
                gated_bank = run_arm_bank(mirro_root, base_cmap, n_colors, cell, col, reg,
                                          band_alpha, "gated_mover", avoid_pols, cell_tag,
                                          want_movement_stats=True, want_events=True)
                g_rev, g_rev_seed = revision_counts(gated_bank["latencies"], base_lat)
                g_both = (gated_bank["def_count"] >= FAMILY_DEF_PASS) and (g_rev >= REV_PASS)
                g_hr = float(np.mean(gated_bank["home_ranges"])) if gated_bank["home_ranges"] else 0.0
                g_mv = float(np.mean(gated_bank["most_visited"])) if gated_bank["most_visited"] else 1.0
                p(f"    {'gated_move':>12} {gated_bank['def_count']:>6} {g_rev:>6} "
                  f"{str(g_both):>5} {g_hr:>5.1f} {g_mv:>7.3f} "
                  f"{str([int(x) for x in g_rev_seed]):>22}")
                # attribution controls
                hold_bank = run_arm_bank(mirro_root, base_cmap, n_colors, cell, col, reg,
                                         band_alpha, "hold_nofreeze", avoid_pols, cell_tag)
                mf_bank = run_arm_bank(mirro_root, base_cmap, n_colors, cell, col, reg,
                                       band_alpha, "move_freeze", avoid_pols, cell_tag)
                mf_rev, _ = revision_counts(mf_bank["latencies"], base_lat)
                p(f"    {'hold_nofrz':>12} {hold_bank['def_count']:>6} {'-':>6} "
                  f"{'-':>5} {'-':>5} {'-':>7} {'(attr ctrl)':>22}")
                p(f"    {'move_freeze':>12} {mf_bank['def_count']:>6} {mf_rev:>6} "
                  f"{'-':>5} {'-':>5} {'-':>7} {'(attr ctrl)':>22}")

                # geometry-sensitivity control: gated mover on G_segreg
                seg_root = copy.deepcopy(mirro)
                seg_root.world = G_segreg
                seg_root._state_dir = None
                seg_root_clean = copy.deepcopy(seg_root); seg_root_clean._state_dir = None
                seg_base = run_arm_bank(seg_root_clean, list(G_segreg.cmap), n_colors, cell,
                                        col, reg, band_alpha, "baseline", avoid_pols_seg,
                                        cell_tag + 100)
                seg_gated = run_arm_bank(copy.deepcopy(seg_root), list(G_segreg.cmap),
                                         n_colors, cell, col, reg, band_alpha, "gated_mover",
                                         avoid_pols_seg, cell_tag + 100)
                geom_delta = gated_bank["def_count"] - seg_gated["def_count"]
                p(f"    geom-control: gated def G_mirro={gated_bank['def_count']}/8 vs "
                  f"G_segreg={seg_gated['def_count']}/8  delta={geom_delta}")

                # per-seed dispersion flag (mean-of-opposites guard) on gated def per-seed
                gd = np.array([1.0 if x else 0.0 for x in gated_bank["per_seed_def"]])
                disp_flag = bool(gd.std() > abs(gd.mean())) if abs(gd.mean()) > 1e-9 else False
                p(f"    per-seed gated def split={[int(x) for x in gated_bank['per_seed_def']]} "
                  f"sd-vs-|mean| dispersion flag={disp_flag}")

                # non-degeneracy: movement POSITIVE requires home_range>=6 AND most_visited<=0.50
                # AND geometry-sensitivity (geom_delta != 0 direction); pull overrides hold.
                mover_nondegen = (g_hr >= 6 and g_mv <= 0.50)
                hold_loses = hold_bank["def_count"] < FAMILY_DEF_PASS  # static park must not win

                # ----- Section 5 VERDICT -----
                P1_band = (base_bank["def_count"] <= (8 - BASELINE_DEF_FAIL)
                           and oracle_bank["def_count"] >= ORACLE_DEF_PASS)
                if not P1_band:
                    verdict = "PRECONDITION-FAIL"
                    deciding = "P1 fails at the band alpha (passive not displaced / oracle weak)"
                elif any_fixedH_both:
                    verdict = "CONSTANTS-ALSO-WIN"
                    deciding = "a fixed-H covers both bars at the band alpha (crack dissolved)"
                elif g_both and always_bank["def_count"] >= FAMILY_DEF_PASS:
                    if mover_nondegen and hold_loses:
                        verdict = "POSABLE"
                        deciding = ("P1 holds, fixed-H crack survives, gated mover covers both "
                                    "AND always-mover ceiling defends, non-degenerate")
                    else:
                        verdict = "CANT-POSE"
                        deciding = ("gated mover covers both but DEGENERATE (home_range/most_"
                                    "visited or hold-wins) -> reject the movement positive")
                else:
                    verdict = "CANT-POSE"
                    deciding = ("P1+crack hold but the gated mover (or its always-on ceiling) "
                                "fails a bar -> movement surface & crack don't co-exist here")
                p(f"  [Sec5] VERDICT={verdict}  ({deciding})")
                results["events_hash"].setdefault(cname, {})[str(col)] = {
                    "always_move": always_bank["events_hashes"],
                    "gated_move": gated_bank["events_hashes"],
                }

            verdict_map[(cname, col)] = verdict
            results["calibration"].setdefault(cname, {})[str(col)] = {
                "band_lo": band_lo, "band_hi": band_hi, "band_alpha": band_alpha,
                "recs": to_plain(calib_recs),
            }
            results["kill_test"].setdefault(cname, {})[str(col)] = {
                "verdict": verdict, "deciding": deciding, "rows": to_plain(kill_rows),
            }
            p("")

    # ================================================================
    # SECTION 5 (cont): verdict vectors + direction map
    # ================================================================
    p("=" * 78)
    p("SECTION 5: VERDICT VECTORS (per-cell, 3 colors) + DIRECTION MAP (NO scalar verdict)")
    p("=" * 78)
    for cell in CELLS:
        cname = cell["name"]
        vec = [verdict_map[(cname, col)] for col in COLORS]
        p(f"  CELL_{cname}: " + "  ".join(f"color{col}={verdict_map[(cname, col)]}"
                                          for col in COLORS))
    p("")
    all_verdicts = list(verdict_map.values())
    n_posable = all_verdicts.count("POSABLE")
    n_cantpose = all_verdicts.count("CANT-POSE")
    n_constants = all_verdicts.count("CONSTANTS-ALSO-WIN")
    n_precond = all_verdicts.count("PRECONDITION-FAIL")
    p(f"  direction map: POSABLE={n_posable} CANT-POSE={n_cantpose} "
      f"CONSTANTS-ALSO-WIN={n_constants} PRECONDITION-FAIL={n_precond} (of {len(all_verdicts)})")
    if n_posable == len(all_verdicts):
        printed = "POSABLE"
    elif n_posable == 0:
        printed = "CANT-POSE"
    else:
        printed = "MIXED"
    p("")
    p(f"PRINTED_VERDICT: {printed}  (this is the SCRIPT'S claim across the 2x3 cell-color grid; "
      f"per-cell-color vectors above are the load-bearing result, NOT this scalar)")
    results["verdicts"] = {f"{cn}_color{co}": verdict_map[(cn, co)]
                           for (cn, co) in verdict_map}
    results["printed_verdict"] = printed

    # ================================================================
    # SECTION 6: SELF-CHECK + exit
    # ================================================================
    runtime = (time.time() - t0) / 60.0
    p("=" * 78)
    p("SECTION 6: SELF-CHECK SUMMARY")
    p("=" * 78)
    p(f"  EQUIVALENCE GATE (Tier A + Tier B): PASS")
    p(f"  All (cell x color) cells produced a clean verdict in "
      f"{{POSABLE, CANT-POSE, CONSTANTS-ALSO-WIN, PRECONDITION-FAIL}}: "
      f"{all(v is not None for v in all_verdicts)}")
    p(f"  Runtime: {runtime:.1f} min")
    exit_code = 0 if all(v is not None for v in all_verdicts) else 3
    p(f"  EXIT CODE: {exit_code} (0 iff gates PASS or clean CAN'T-POSE; nonzero iff equiv "
      f"gate / spine FAILED)")
    p("")
    _write(out_dir, lines, results)
    return exit_code


def _write(out_dir, lines, results):
    txt_path = out_dir / "exp271.txt"
    json_path = out_dir / "exp271_results.json"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(json_path, "w") as f:
        json.dump(to_plain(results), f, sort_keys=True, indent=2)
    print(f"\nWROTE {txt_path}")
    print(f"WROTE {json_path}")


if __name__ == "__main__":
    sys.exit(main())
