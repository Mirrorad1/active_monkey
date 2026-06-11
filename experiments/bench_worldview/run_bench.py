"""Worldview benchmark runner — T15, rigor-fairness-upgrade spec.

Runs the agent simulation loop across the four benchmark world families
(A=learnable, B=noisy, C=aliased, D=nonstationary) and two mechanisms
(none, grow) using the extracted lab machinery from active_loop.

CLI
---
    python experiments/bench_worldview/run_bench.py \\
        --world {A,B,C,D} \\
        --mechanism {none,grow} \\
        --seeds N \\
        --layouts N \\
        --convention {unnormalized,normalized}

Mechanisms ``decay``, ``random_accept``, ``replay_accept``, ``bigger_fixed``,
``oracle`` are reserved for T16 and will be rejected with
"not yet implemented".

Outputs (per run)
-----------------
- JSON rows to ``experiments/bench_worldview/outputs/<tag>_rows.json``
- ``summary.md`` under ``experiments/bench_worldview/outputs/<tag>_summary.md``
- ``verdict.json`` via ``active_loop.verdict.write_verdict`` ONLY when
  ``experiments/bench_worldview/bars.json`` exists (predeclared by the research
  loop). If absent, exits after writing rows/summary with the message
  "awaiting predeclaration — bars.json not found" and the verdict file is
  OMITTED.

Simulation-loop reuse decision
-------------------------------
Lifted from ``experiments/exp154_growth_confirmation.py`` (the Exp 154 pattern
for the run_arm_b_normalized phase loop), rather than from
``active_loop/creature_continuous.py``.  Reason: ContinuousCreature is a
persisted species tied to a fixed 16-color non-aliased world with its own
file I/O, biography, and state management layers.  The benchmark needs a
lightweight, world-agnostic loop over parameterised worlds from worlds.py.
The exp154 phase loop (ContinuousPlace + NIW components + growth machinery)
is exactly the right abstraction — it is the minimal faithful reuse and avoids
dragging in the creature persistence layer.

The winning configuration for the ``grow`` mechanism (Exp 154):
    batch-jump EM + live probation + normalized convention
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Repo root on sys.path when run as a script from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from active_loop.continuous import NIW
from active_loop.creature_continuous import ContinuousPlace
from active_loop.growth import (
    ALARM_THRESH,
    BURN_IN_COV_FLOOR,
    CEILING_MEAN_THRESH,
    CEILING_SLOPE_THRESH,
    COLOR_SURPRISE_WINDOW,
    JUMP_COOLDOWN,
    KEEP_MARGIN,
    K_CANDIDATES,
    K_PENALTY,
    MIN_REPLAY_PAIRS,
    PRE_SPAWN_WINDOW,
    PROBATION_STEPS,
    SPAWN_BUDGET,
    SPAWN_INTERVAL,
    SURPRISE_WINDOW,
    EM_COV_FLOOR,
    EM_ITERS,
    ConventionT,
    alarmed_colors_with_budget,
    mixture_emission_moments,
    mixture_predictive_logprobs,
    pick_round_robin_color,
    select_best_k,
    _copy_color_components,
)
from active_loop.verdict import write_verdict
from active_loop.worlds import (
    aliased,
    analytic_floor,
    learnable,
    noisy,
    nonstationary,
    ALIASED_LAYOUT_SEEDS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NIW prior (matches exp153/exp154 exactly)
_D = 2
_KAPPA0 = 1.0
_NU0 = 4.0
_S0_SCALE = 0.35 ** 2 * (_NU0 - _D - 1)   # = 0.1225
_Q_SCALE = 0.05                             # process noise

# Phase timing (smoke run uses shorter T — see pick_T())
_T_PHASE1_FULL = 2000
_T_PHASE2_FULL = 6000
_T_SMOKE_PHASE1 = 600    # SMOKE: smallest T that lets B's noise floor show
_T_SMOKE_PHASE2 = 1400   # SMOKE: smallest T that lets C's alarm show (need ~50-step burn-in)
# Rationale documented in summary.md as SMOKE

# Eval window: last fraction of phase2
_FINAL_EVAL_FRAC = 0.25  # last 25% of phase2 used for final-surprise eval

# Mechanism whitelist
_IMPLEMENTED_MECHANISMS = {"none", "grow"}
_RESERVED_MECHANISMS = {
    "decay", "random_accept", "replay_accept", "bigger_fixed", "oracle"
}

# Output dir relative to repo root
_BENCH_DIR = Path(__file__).resolve().parent
_OUTPUTS_DIR = _BENCH_DIR / "outputs"

# ---------------------------------------------------------------------------
# Action / world helpers
# ---------------------------------------------------------------------------

def _cell_centers(rows: int, cols: int) -> np.ndarray:
    """Shape (rows*cols, 2): center of each cell as (x=col, y=row)."""
    return np.array(
        [[float(c), float(r)] for r in range(rows) for c in range(cols)],
        dtype=float,
    )


def _move_cell(cell: int, action: int, rows: int, cols: int) -> int:
    """Wall-clamped grid move (0=up, 1=down, 2=left, 3=right)."""
    r, c = divmod(cell, cols)
    if action == 0:
        r = max(0, r - 1)
    elif action == 1:
        r = min(rows - 1, r + 1)
    elif action == 2:
        c = max(0, c - 1)
    else:
        c = min(cols - 1, c + 1)
    return r * cols + c


_ACTION_DELTA = {
    0: np.array([0.0, -1.0]),
    1: np.array([0.0, +1.0]),
    2: np.array([-1.0, 0.0]),
    3: np.array([+1.0, 0.0]),
}


def _check_ceiling(surprise_buf: deque) -> bool:
    """Return True if the mean/slope conjunction fires on surprise_buf."""
    if len(surprise_buf) < SURPRISE_WINDOW:
        return False
    arr = np.array(surprise_buf)
    if float(np.mean(arr)) <= CEILING_MEAN_THRESH:
        return False
    xs = np.arange(len(arr), dtype=float)
    slope = float(np.polyfit(xs, arr, 1)[0])
    return abs(slope) < CEILING_SLOPE_THRESH


# ---------------------------------------------------------------------------
# World instantiation helpers
# ---------------------------------------------------------------------------

def _build_world_and_cmap(
    world_key: str,
    layout_seed: int,
) -> tuple[dict, np.ndarray]:
    """Return (world_dict, cmap_array) for the given world key and layout seed.

    For A (learnable) and B (noisy): layout_seed is used as the action RNG seed;
    the cmap is fixed (from worlds.learnable()).
    For C (aliased): layout_seed controls the color layout.
    For D (nonstationary): layout_seed controls the post-remap permutation seed;
    the remap fires at the midpoint of phase2.
    """
    if world_key == "A":
        w = learnable()
        cmap = np.array(w["cmap"], dtype=int)
        return w, cmap

    elif world_key == "B":
        w = noisy()
        cmap = np.array(w["cmap"], dtype=int)
        return w, cmap

    elif world_key == "C":
        w = aliased(layout_seed=layout_seed)
        cmap = np.array(w["cmap"], dtype=int)
        return w, cmap

    elif world_key == "D":
        base = learnable()
        w = nonstationary(base=base, remap_at_step=0, remap_seed=layout_seed)
        # remap_at_step is set per-run (at mid-phase2); for now store the dict
        # The actual remap_at_step is wired in run_single_seed()
        cmap = np.array(w["cmap"], dtype=int)
        return w, cmap

    else:
        raise ValueError(f"Unknown world key {world_key!r}")


def _obs_color(
    true_cell: int,
    cmap_current: np.ndarray,
    world: dict,
    rng: np.random.Generator,
) -> int:
    """Draw observation color for current cell under the world's noise model."""
    true_color = int(cmap_current[true_cell])
    if world["kind"] == "noisy":
        p_true = float(world["p_true"])
        n_colors = int(world["n_colors"])
        if rng.random() < p_true:
            return true_color
        else:
            # Uniform over the OTHER n_colors-1 colors
            others = [c for c in range(n_colors) if c != true_color]
            return int(rng.choice(others))
    return true_color


# ---------------------------------------------------------------------------
# Core simulation loop (world-agnostic, mechanism-parameterised)
# ---------------------------------------------------------------------------

def run_single_seed(
    world_key: str,
    world: dict,
    cmap_initial: np.ndarray,
    mechanism: str,
    convention: ConventionT,
    actions: np.ndarray,
    run_rng: np.random.Generator,
    t_phase1: int,
    t_phase2: int,
) -> dict:
    """Run one seed of the benchmark simulation.

    Simulation-loop pattern: lifted directly from exp154 run_arm_b_normalized.
    ContinuousPlace + NIW mixture components + optional grow mechanism.

    The place update always uses the UNNORMALIZED integral (footprint conjugate
    — unchanged from exp153/exp154). The convention argument controls only the
    PREDICTIVE evaluation (surprise, alarm, ceiling, probation scoring).

    Parameters
    ----------
    world_key    : 'A' | 'B' | 'C' | 'D'
    world        : world dict from worlds.py
    cmap_initial : int array of cell→color mapping (pre-remap for D)
    mechanism    : 'none' or 'grow'
    convention   : 'unnormalized' or 'normalized'
    actions      : pre-drawn action sequence, shape (t_phase1+t_phase2,)
    run_rng      : numpy Generator for EM / K-selection
    t_phase1     : phase-1 step count
    t_phase2     : phase-2 step count

    Returns
    -------
    dict with keys: world_key, mechanism, convention, plateau,
    final_surprise, drop, alarm_events, growth_accepted, growth_reverted,
    final_ceiling_events, phase1_ceiling_events, final_comps_per_color,
    attempt_records, loc_median_final500 (phase2), phase1_loc_median_final500.
    """
    t_total = t_phase1 + t_phase2
    rows = int(world.get("rows", 4))
    cols = int(world.get("cols", 4))
    n_cells = rows * cols
    n_colors = int(world["n_colors"])

    cell_centers = _cell_centers(rows, cols)
    arena = (0.0, float(cols - 1), 0.0, float(rows - 1))
    arena_center = np.array([(cols - 1) / 2.0, (rows - 1) / 2.0])

    S0 = _S0_SCALE * np.eye(_D)
    Q = np.diag([_Q_SCALE ** 2, _Q_SCALE ** 2])

    # Place belief
    cp = ContinuousPlace(arena_center.copy(), np.diag([4.0, 4.0]), arena)

    # NIW components — one per color, initial unimodal
    components: list[list[tuple[float, NIW]]] = []
    counts: list[list[int]] = []
    for k in range(n_colors):
        niw0 = NIW(m=arena_center.copy(), kappa=_KAPPA0, nu=_NU0, S=S0.copy())
        components.append([(1.0, niw0)])
        counts.append([1])

    # --- Mechanism: grow state ---
    spawn_budget = [SPAWN_BUDGET] * n_colors   # max extra components per color

    surprise_buf: deque = deque(maxlen=SURPRISE_WINDOW)
    replay_buf: deque = deque(maxlen=max(400, PRE_SPAWN_WINDOW))
    color_surprise_bufs: list[deque] = [
        deque(maxlen=COLOR_SURPRISE_WINDOW) for _ in range(n_colors)
    ]
    color_pre_jump_bufs: list[deque] = [
        deque(maxlen=PRE_SPAWN_WINDOW) for _ in range(n_colors)
    ]
    last_attempt_step: list[int] = [-JUMP_COOLDOWN - 1] * n_colors

    # Probation state
    probation_color: int = -1
    probation_start_phase2_t: int = -1
    probation_pre_jump_mean: float = float("inf")
    probation_color_snap: list[tuple[float, NIW]] = []
    probation_color_counts_snap: list[int] = []
    probation_observations: list[float] = []

    attempt_records: list[dict] = []
    alarm_event_steps: list[int] = []  # phase2 steps when alarm triggered an attempt

    # Diagnostics
    phase1_surprise_vals: list[float] = []
    phase2_surprise_vals: list[float] = []
    phase1_loc_errors: list[float] = []
    phase2_loc_errors: list[float] = []
    phase1_ceiling_count = 0
    phase2_final_ceiling_count = 0

    # Nonstationary world: cmap flips at midpoint of phase2
    cmap_current = cmap_initial.copy()
    cmap_after = None
    remap_at_step = None
    if world_key == "D":
        cmap_after = np.array(world.get("cmap_after", cmap_initial), dtype=int)
        remap_at_step = t_phase1 + t_phase2 // 2

    true_cell = 0
    obs_rng = np.random.default_rng(int(run_rng.integers(0, 2**32)))

    for t in range(t_total):
        # Nonstationary remap
        if world_key == "D" and remap_at_step is not None and t == remap_at_step:
            cmap_current = cmap_after

        obs_k = _obs_color(true_cell, cmap_current, world, obs_rng)

        Sigma_p_diag = np.maximum(
            np.array([cp.Sigma[0, 0], cp.Sigma[1, 1]]), 1e-9
        )
        mu_p = cp.mu

        replay_buf.append((obs_k, mu_p.copy(), Sigma_p_diag.copy()))

        # Surprise under the chosen convention
        log_probs = mixture_predictive_logprobs(
            mu_p, Sigma_p_diag, components, convention=convention
        )
        surprise_t = float(-log_probs[obs_k])
        surprise_buf.append(surprise_t)

        is_phase1 = t < t_phase1
        phase2_t = t - t_phase1

        if is_phase1:
            phase1_surprise_vals.append(surprise_t)
            phase1_loc_errors.append(
                float(np.linalg.norm(mu_p - cell_centers[true_cell]))
            )
            if _check_ceiling(surprise_buf):
                phase1_ceiling_count += 1
        else:
            phase2_surprise_vals.append(surprise_t)
            phase2_loc_errors.append(
                float(np.linalg.norm(mu_p - cell_centers[true_cell]))
            )

            color_surprise_bufs[obs_k].append(surprise_t)
            color_pre_jump_bufs[obs_k].append(surprise_t)

            if probation_color == obs_k:
                probation_observations.append(surprise_t)

            is_ceiling = _check_ceiling(surprise_buf)
            if is_ceiling and phase2_t >= t_phase2 - max(100, t_phase2 // 4):
                phase2_final_ceiling_count += 1

            # --- Probation resolution ---
            if probation_color >= 0 and mechanism == "grow":
                elapsed = phase2_t - probation_start_phase2_t
                if elapsed >= PROBATION_STEPS:
                    pc = probation_color
                    prob_mean = (
                        float(np.mean(probation_observations))
                        if probation_observations
                        else float("inf")
                    )
                    keep = prob_mean <= probation_pre_jump_mean - KEEP_MARGIN

                    if attempt_records:
                        attempt_records[-1]["probation_mean"] = prob_mean
                        attempt_records[-1]["delta"] = (
                            probation_pre_jump_mean - prob_mean
                        )
                        attempt_records[-1]["kept"] = keep

                    if keep:
                        spawn_budget[pc] -= 1
                    else:
                        components[pc] = probation_color_snap
                        counts[pc] = probation_color_counts_snap

                    probation_color = -1
                    probation_start_phase2_t = -1
                    probation_pre_jump_mean = float("inf")
                    probation_color_snap = []
                    probation_color_counts_snap = []
                    probation_observations = []

            # --- Alarm check and growth attempt ---
            if (
                mechanism == "grow"
                and probation_color < 0
                and phase2_t > 0
                and phase2_t % SPAWN_INTERVAL == 0
            ):
                eligible = alarmed_colors_with_budget(
                    color_surprise_bufs, spawn_budget
                )
                # Apply cooldown
                eligible = [
                    k for k in eligible
                    if (phase2_t - last_attempt_step[k]) >= JUMP_COOLDOWN
                ]

                if eligible:
                    jump_color = pick_round_robin_color(eligible, last_attempt_step)
                    last_attempt_step[jump_color] = phase2_t
                    alarm_event_steps.append(phase2_t)

                    color_replay_pairs = [
                        (mu_s.copy(), sig_s.copy())
                        for (obs_c, mu_s, sig_s) in replay_buf
                        if obs_c == jump_color
                    ]

                    if len(color_replay_pairs) >= MIN_REPLAY_PAIRS:
                        pre_jump_buf = list(color_pre_jump_bufs[jump_color])
                        pre_jump_mean = (
                            float(np.mean(pre_jump_buf))
                            if pre_jump_buf
                            else float("inf")
                        )

                        snap_comps = _copy_color_components(
                            components[jump_color]
                        )
                        snap_counts = list(counts[jump_color])

                        best_K, best_comps, best_n_eff, nll_by_k = select_best_k(
                            color_replay_pairs, run_rng
                        )

                        components[jump_color] = best_comps
                        n_total = len(color_replay_pairs)
                        new_counts = [
                            max(1, int(round(
                                float(best_n_eff[j]) * n_total / sum(best_n_eff)
                            )))
                            for j in range(best_K)
                        ]
                        counts[jump_color] = new_counts

                        probation_color = jump_color
                        probation_start_phase2_t = phase2_t
                        probation_pre_jump_mean = pre_jump_mean
                        probation_color_snap = snap_comps
                        probation_color_counts_snap = snap_counts
                        probation_observations = []

                        nll_serial = {
                            str(K): {"nll": nll_by_k[K][0], "score": nll_by_k[K][1]}
                            for K in K_CANDIDATES
                        }
                        attempt_records.append({
                            "color": jump_color,
                            "K_chosen": best_K,
                            "nll_by_k": nll_serial,
                            "pre_jump_mean": pre_jump_mean,
                            "probation_mean": float("nan"),
                            "delta": float("nan"),
                            "kept": None,
                        })

        # --- Place update (UNCHANGED: unnormalized conjugate) ---
        mu_mix, Sigma_mix_diag, hard_idx = mixture_emission_moments(
            mu_p, Sigma_p_diag, obs_k, components
        )
        cp.update(mu_mix, np.diag(Sigma_mix_diag))

        post_mu = cp.mu
        post_Sigma_diag = np.maximum(
            np.array([cp.Sigma[0, 0], cp.Sigma[1, 1]]), 1e-9
        )

        old_niw = components[obs_k][hard_idx][1]
        new_niw = old_niw.update_moments(post_mu, np.diag(post_Sigma_diag))
        components[obs_k][hard_idx] = (
            components[obs_k][hard_idx][0], new_niw
        )
        counts[obs_k][hard_idx] += 1
        total_k = sum(counts[obs_k])
        components[obs_k] = [
            (counts[obs_k][j] / total_k, niw)
            for j, (_, niw) in enumerate(components[obs_k])
        ]

        action = int(actions[t])
        true_cell = _move_cell(true_cell, action, rows, cols)
        cp.predict_clamped_moments(_ACTION_DELTA[action], Q)

    # --- Handle in-flight probation at end of run ---
    if probation_color >= 0 and mechanism == "grow":
        pc = probation_color
        components[pc] = probation_color_snap
        counts[pc] = probation_color_counts_snap
        if attempt_records:
            attempt_records[-1]["kept"] = False
            attempt_records[-1]["probation_mean"] = (
                float(np.mean(probation_observations))
                if probation_observations
                else float("nan")
            )

    # --- Compute results ---
    final_eval_n = max(1, int(t_phase2 * _FINAL_EVAL_FRAC))
    final_surprise = float(np.mean(phase2_surprise_vals[-final_eval_n:]))
    p1_plateau_n = min(500, max(100, t_phase1 // 4))
    plateau = float(np.mean(phase1_surprise_vals[-p1_plateau_n:]))

    decided = [r for r in attempt_records if r.get("kept") is not None]
    kept = [r for r in decided if r.get("kept") is True]
    growth_accepted = len(kept)
    growth_reverted = len([r for r in decided if not r.get("kept")])

    final_comps = [len(components[k]) for k in range(n_colors)]

    loc_arr_p2 = np.array(phase2_loc_errors) if phase2_loc_errors else np.array([float("nan")])
    loc_arr_p1 = np.array(phase1_loc_errors) if phase1_loc_errors else np.array([float("nan")])
    loc_med_p2 = float(np.median(loc_arr_p2[-min(500, len(loc_arr_p2)):]))
    loc_med_p1 = float(np.median(loc_arr_p1[-min(500, len(loc_arr_p1)):]))

    return {
        "world_key": world_key,
        "mechanism": mechanism,
        "convention": convention,
        "plateau": plateau,
        "final_surprise": final_surprise,
        "drop": plateau - final_surprise,
        "alarm_events": len(alarm_event_steps),
        "growth_accepted": growth_accepted,
        "growth_reverted": growth_reverted,
        "final_ceiling_events": phase2_final_ceiling_count,
        "phase1_ceiling_events": phase1_ceiling_count,
        "final_comps_per_color": final_comps,
        "attempt_records": attempt_records,
        "loc_median_final500_p2": loc_med_p2,
        "loc_median_final500_p1": loc_med_p1,
    }


# ---------------------------------------------------------------------------
# Multi-layout / multi-seed orchestration
# ---------------------------------------------------------------------------

def _layout_seeds_for_world(world_key: str, n_layouts: int) -> list[int]:
    """Return layout seeds for a given world key.

    For C (aliased): use the canonical ALIASED_LAYOUT_SEEDS.
    For D (nonstationary): use fixed remap seeds (permutation-diversity).
    For A, B: single layout (layout_seed unused for cmap but still varies action RNG).
    """
    if world_key == "C":
        seeds = list(ALIASED_LAYOUT_SEEDS)
        return seeds[:n_layouts]
    elif world_key == "D":
        return [17, 23, 31][:n_layouts]
    else:
        # A, B: single canonical layout
        return [42][:n_layouts]


def run_bench(
    world_key: str,
    mechanism: str,
    n_seeds: int,
    n_layouts: int,
    convention: ConventionT,
    t_phase1: int,
    t_phase2: int,
) -> list[dict]:
    """Run the full matrix and return JSON-serialisable result rows."""
    t_total = t_phase1 + t_phase2
    layout_seeds = _layout_seeds_for_world(world_key, n_layouts)
    rows: list[dict] = []

    for layout_idx, layout_seed in enumerate(layout_seeds):
        world, cmap = _build_world_and_cmap(world_key, layout_seed)

        # For world D: bake remap_at_step into a per-layout world dict
        if world_key == "D":
            remap_at = t_phase1 + t_phase2 // 2
            # Rebuild nonstationary with correct remap step
            from active_loop.worlds import learnable as _learnable, nonstationary as _nonstationary
            base = _learnable()
            world = _nonstationary(base=base, remap_at_step=remap_at, remap_seed=layout_seed)
            cmap = np.array(world["cmap"], dtype=int)

        for seed in range(n_seeds):
            action_rng = np.random.default_rng(1000 + layout_seed * 100 + seed)
            actions = action_rng.integers(0, 4, size=t_total)
            run_rng = np.random.default_rng(2000 + layout_seed * 100 + seed)

            result = run_single_seed(
                world_key=world_key,
                world=world,
                cmap_initial=cmap,
                mechanism=mechanism,
                convention=convention,
                actions=actions,
                run_rng=run_rng,
                t_phase1=t_phase1,
                t_phase2=t_phase2,
            )

            row = {
                "bench": "worldview",
                "world": world_key,
                "mechanism": mechanism,
                "convention": convention,
                "layout_seed": layout_seed,
                "layout_idx": layout_idx,
                "seed": seed,
                "t_phase1": t_phase1,
                "t_phase2": t_phase2,
                **{
                    k: v
                    for k, v in result.items()
                    if k not in ("attempt_records", "world_key", "mechanism", "convention")
                },
                "n_attempts": len(result["attempt_records"]),
                "attempt_records": result["attempt_records"],
            }
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Grading against bars.json
# ---------------------------------------------------------------------------

def _grade(rows: list[dict], bars: dict) -> dict:
    """Grade rows against predeclared bars.

    Returns a verdict-ready arms dict.
    """
    arms: dict[str, dict] = {}
    for arm_name, bar_spec in bars.items():
        subset = [
            r for r in rows
            if r.get("world") == bar_spec.get("world")
            and r.get("mechanism") == bar_spec.get("mechanism")
        ]
        if not subset:
            arms[arm_name] = {
                "pass": False,
                "reason": f"no rows matched bar spec {bar_spec}",
            }
            continue

        metric = bar_spec.get("metric", "final_surprise")
        direction = bar_spec.get("direction", "le")  # "le" or "ge"
        threshold = float(bar_spec["threshold"])
        agg = bar_spec.get("agg", "mean")

        vals = [r[metric] for r in subset if metric in r and not math.isnan(r[metric])]
        if not vals:
            arms[arm_name] = {
                "pass": False,
                "reason": f"metric {metric!r} missing or all NaN",
            }
            continue

        agg_val = float(np.mean(vals)) if agg == "mean" else float(np.median(vals))

        if direction == "le":
            passed = agg_val <= threshold
        else:
            passed = agg_val >= threshold

        arms[arm_name] = {
            "pass": passed,
            "reason": (
                f"{agg}({metric})={agg_val:.4f} "
                f"{'<=' if direction == 'le' else '>='} {threshold:.4f}: "
                f"{'PASS' if passed else 'FAIL'}"
            ),
        }

    return arms


# ---------------------------------------------------------------------------
# Summary markdown builder
# ---------------------------------------------------------------------------

def _build_summary(
    rows: list[dict],
    world_key: str,
    mechanism: str,
    convention: str,
    t_phase1: int,
    t_phase2: int,
    smoke: bool,
) -> str:
    """Build summary.md content."""
    lines = ["# Worldview Benchmark — Summary", ""]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines += [
        f"Generated: {now}",
        f"World: {world_key}  Mechanism: {mechanism}  Convention: {convention}",
        f"Phase1: {t_phase1} steps  Phase2: {t_phase2} steps",
        "",
    ]

    if smoke:
        lines += [
            "## SMOKE RUN",
            "",
            "**This is NOT a scientific run.** Step counts chosen as the smallest T that still",
            "lets the key world-specific phenomena show:",
            f"- Phase1 T={t_phase1}: enough for place-belief convergence (A/B/C) and",
            "  noise-floor stabilisation (B requires ~50 obs/color at 0.7 noise to plateau).",
            f"- Phase2 T={t_phase2}: enough for the alarm to fire in C (aliased world needs",
            f"  ~{COLOR_SURPRISE_WINDOW} per-color obs + {SPAWN_INTERVAL}-step check interval;",
            f"  {t_phase2} steps gives ~{t_phase2 // 4} per-color obs in a 4-color world).",
            "  World D (nonstationary) remap fires at mid-phase2.",
            "",
            f"Smoke T_phase1={_T_SMOKE_PHASE1}, T_phase2={_T_SMOKE_PHASE2} chosen empirically.",
            "",
        ]

    if not rows:
        lines.append("No rows generated.")
        return "\n".join(lines)

    # Group by layout_seed x seed
    from itertools import groupby

    lines.append("## Results Table")
    lines.append("")
    lines.append(
        "| world | mechanism | layout_seed | seed | plateau | final_surprise | drop | alarm_events | growth_accepted |"
    )
    lines.append(
        "|-------|-----------|-------------|------|---------|----------------|------|--------------|-----------------|"
    )

    for r in rows:
        plateau = r.get("plateau", float("nan"))
        fs = r.get("final_surprise", float("nan"))
        drop = r.get("drop", float("nan"))
        alarms = r.get("alarm_events", 0)
        accepted = r.get("growth_accepted", 0)
        lines.append(
            f"| {r['world']} | {r['mechanism']} | {r['layout_seed']} | {r['seed']} "
            f"| {plateau:.4f} | {fs:.4f} | {drop:.4f} | {alarms} | {accepted} |"
        )

    lines.append("")

    # Aggregate stats
    plateaus = [r["plateau"] for r in rows if "plateau" in r]
    finals = [r["final_surprise"] for r in rows if "final_surprise" in r]
    drops = [r["drop"] for r in rows if "drop" in r]
    alarms_total = sum(r.get("alarm_events", 0) for r in rows)
    accepted_total = sum(r.get("growth_accepted", 0) for r in rows)

    lines += [
        "## Aggregate",
        "",
        f"- Mean plateau: {np.mean(plateaus):.4f} nats" if plateaus else "",
        f"- Mean final surprise: {np.mean(finals):.4f} nats" if finals else "",
        f"- Mean drop: {np.mean(drops):.4f} nats" if drops else "",
        f"- Total alarm events: {alarms_total}",
        f"- Total growth accepted: {accepted_total}",
        "",
        "## Grading",
        "",
        "bars.json absent — awaiting predeclaration. No verdict issued.",
        "",
    ]

    return "\n".join(l for l in lines)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Worldview benchmark runner (T15, rigor-fairness-upgrade spec)",
    )
    parser.add_argument("--world", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument(
        "--mechanism",
        required=True,
        choices=list(_IMPLEMENTED_MECHANISMS) + list(_RESERVED_MECHANISMS),
    )
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--layouts", type=int, default=3)
    parser.add_argument(
        "--convention", default="normalized", choices=["unnormalized", "normalized"]
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use smoke step counts (overrides full T)",
    )
    parser.add_argument(
        "--t-phase1", type=int, default=None,
        help="Override phase1 step count (default: full or smoke)",
    )
    parser.add_argument(
        "--t-phase2", type=int, default=None,
        help="Override phase2 step count (default: full or smoke)",
    )
    args = parser.parse_args(argv)

    # Reject not-yet-implemented mechanisms
    if args.mechanism in _RESERVED_MECHANISMS:
        print(
            f"mechanism {args.mechanism!r} not yet implemented "
            f"(reserved for T16); implemented: {sorted(_IMPLEMENTED_MECHANISMS)}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Step counts
    smoke = args.smoke
    if args.t_phase1 is not None:
        t_phase1 = args.t_phase1
    else:
        t_phase1 = _T_SMOKE_PHASE1 if smoke else _T_PHASE1_FULL

    if args.t_phase2 is not None:
        t_phase2 = args.t_phase2
    else:
        t_phase2 = _T_SMOKE_PHASE2 if smoke else _T_PHASE2_FULL

    tag = (
        f"bench_{args.world}_{args.mechanism}_{args.convention}"
        f"_s{args.seeds}_l{args.layouts}"
    )

    print(
        f"[bench] world={args.world} mechanism={args.mechanism} "
        f"convention={args.convention} seeds={args.seeds} layouts={args.layouts} "
        f"t1={t_phase1} t2={t_phase2} smoke={smoke}"
    )

    rows = run_bench(
        world_key=args.world,
        mechanism=args.mechanism,
        n_seeds=args.seeds,
        n_layouts=args.layouts,
        convention=args.convention,
        t_phase1=t_phase1,
        t_phase2=t_phase2,
    )

    # Write JSON rows
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = _OUTPUTS_DIR / f"{tag}_rows.json"
    with rows_path.open("w") as fh:
        for row in rows:
            # Serialise — replace NaN with null for JSON compliance
            def _ser(v):
                if isinstance(v, float) and math.isnan(v):
                    return None
                return v
            safe_row = {k: _ser(v) if not isinstance(v, (list, dict)) else v for k, v in row.items()}
            fh.write(json.dumps(safe_row) + "\n")
    print(f"[bench] wrote {len(rows)} rows → {rows_path}")

    # Write summary
    summary_path = _OUTPUTS_DIR / f"{tag}_summary.md"
    summary = _build_summary(
        rows, args.world, args.mechanism, args.convention,
        t_phase1, t_phase2, smoke=smoke,
    )
    summary_path.write_text(summary, encoding="utf-8")
    print(f"[bench] wrote summary → {summary_path}")

    # Grade against bars.json (if it exists)
    bars_path = _BENCH_DIR / "bars.json"
    if not bars_path.exists():
        print("[bench] awaiting predeclaration — bars.json not found")
        return

    bars = json.loads(bars_path.read_text())
    arms = _grade(rows, bars)

    all_pass = all(a["pass"] for a in arms.values())
    verdict_str = "POSITIVE" if all_pass else (
        "MIXED" if any(a["pass"] for a in arms.values()) else "NEGATIVE"
    )

    verdict_path = _OUTPUTS_DIR / f"{tag}_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment=f"bench_{args.world}_{args.mechanism}",
        arms=arms,
        verdict=verdict_str,
        halted=False,
        notes=f"bars.json loaded; world={args.world} mechanism={args.mechanism}",
    )
    print(f"[bench] verdict={verdict_str} → {verdict_path}")


if __name__ == "__main__":
    main()
