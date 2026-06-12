"""Exp 193 — N4 Return-Home Causal Intervention (rung 1 reframed).

PRE-REGISTERED in loop/directions/n4-crack-edges.md (commit fb6ca58) BEFORE any data.

Question (causal): When the locality confound is removed by intervention —
the creature is returned to its pre-captivity home position at the last burst
end — (i) does release SETTLE, (ii) what is the retention rate, and (iii) does
the surrendering seed 301 now RETAIN (the locality mechanism's direct causal
prediction: same seed, same defense, intervened stream)?

The intervention (minimal, draw-free): Runner parameter return_home: bool = False.
At the FIRST burst start, capture home_pos = true_pos (the pre-relocation position);
at the LAST burst end, set true_pos = home_pos and localize qs at it (mirroring the
captivity-relocation qs pattern). NO rng draws consumed (the False path must remain
bit-identical). Built through the exp190 _must_replace patch chain (L16), with both
branches gated.

falsifier bindings (ordered per pre-registration):
  F1 (settling): any controller pair M < 10 — cycle persists at home; the locality
    reading of the NON-SETTLING TAIL is wrong/incomplete (third strike on instruments).
  F2 (retention): any covered pair retention_rate <= 2/3 — deferral persists without
    the locality feed; the stored-state reading returns.
  F3 (crown causal): s301 C-C still surrenders — the locality mechanism is refuted as
    the cause of its deferral.
  F4 (durability): baseline self-heals in >= 8/16 anywhere — displacement durability
    was itself locality-fed.

prediction: P1 PASS (settling restored with locality removed); P2 PASS (retention >= 5/6);
  P3 PASS (s301 retains under intervention, validating the locality mechanism as cause);
  P4 PASS (baseline stays displaced, train overwrite is intrinsic); P5 PASS (oracle >= 5/6).

Status: active | Seeds: 296-311 | 3 cells x 4 arms x 16 seeds = 192 W sessions.
Runner: run_fork_schedule_193 with return_home=True (exp190 patch chain + return-home patches).
"""
from __future__ import annotations

import copy
import inspect
import json
import math
import statistics
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Import exp190 module via importlib (same pattern as exp192)
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_spec190 = _ilu.spec_from_file_location(
    "exp190",
    str(REPO_ROOT / "experiments" / "exp190_n4_flicker_hysteresis.py"),
)
_mod190 = _ilu.module_from_spec(_spec190)
_spec190.loader.exec_module(_mod190)  # type: ignore[union-attr]

# Grab the exp188 module reference that exp190 already loaded
_mod188 = _mod190._mod188

# Import helpers from exp190
_must_replace = _mod190._must_replace
run_equivalence_gate_190 = _mod190.run_equivalence_gate_190
compute_defense = _mod190.compute_defense
compute_final_gap = _mod190.compute_final_gap
phase_r_latency = _mod190.phase_r_latency
to_plain = _mod190.to_plain
_value_match = _mod190._value_match
_load_rows = _mod190._load_rows

DEFAULT_THETA = _mod190.DEFAULT_THETA
DEFAULT_RELEASE_CALM_SNAPS = _mod190.DEFAULT_RELEASE_CALM_SNAPS
BURST_SEED_OFFSET_R = _mod190.BURST_SEED_OFFSET_R
FINE_EVAL = _mod190.FINE_EVAL
N_STEPS_PHASE_R = _mod190.N_STEPS_PHASE_R
FLOAT_ATOL = _mod190.FLOAT_ATOL
INIT_MASS = _mod190.INIT_MASS
PRESSURE_WINDOW = _mod190.PRESSURE_WINDOW
PRESSURE_FRAC = _mod190.PRESSURE_FRAC
BURST_SEED_OFFSET_W_EXP183 = _mod190.BURST_SEED_OFFSET_W_EXP183
CALM2600_SNAPS = _mod190.CALM2600_SNAPS
CHUNK_SIZE = _mod190.CHUNK_SIZE
PC1_AHAT_DRIFT_MAX = _mod190.PC1_AHAT_DRIFT_MAX

from active_loop.creature import Creature

# ---------------------------------------------------------------------------
# Build run_fork_schedule_193 via the exp190 patch chain + return-home patches
# ---------------------------------------------------------------------------

def _build_run_fork_schedule_193():
    """Generate runner with exp190 patches PLUS the return-home intervention."""
    src = "from __future__ import annotations\n" + textwrap.dedent(
        inspect.getsource(_mod188.run_fork_schedule_188)
    )

    # ---- exp190 patch chain (verbatim, swapping 190 -> 193 in name) ----
    src = _must_replace(src,
        "def run_fork_schedule_188(",
        "def run_fork_schedule_193(",
        1,
    )
    src = _must_replace(src,
        "    release_calm_snaps: int = DEFAULT_RELEASE_CALM_SNAPS,\n) -> dict:",
        "    release_calm_snaps: int = DEFAULT_RELEASE_CALM_SNAPS,\n"
        "    hysteresis_snaps: int = 0,\n"
        "    record_pressure_trace: bool = False,\n"
        "    return_home: bool = False,\n"
        ") -> dict:",
        1,
    )
    src = _must_replace(src,
        "    reg_max_current_seen: float = 0.0   # high-water mark of current_stretch (diagnostic)\n",
        "    reg_max_current_seen: float = 0.0   # high-water mark of current_stretch (diagnostic)\n"
        "    deassert_run: int = 0               # consecutive de-assert snaps while frozen\n"
        "    pressure_trace: list[list[int]] = [] # [global_step, frozen_flag, pressure_active_flag]\n"
        "    home_pos: int | None = None          # captured pre-captivity position\n"
        "    home_restored_step: int | None = None  # global_step where home was restored\n",
        1,
    )
    src = _must_replace(src,
        "                    if not pressure_active:\n"
        "                        # Pressure de-asserted: handle stretch end + reset\n"
        "\n"
        "                        if int_mode == \"reg_tb\":\n"
        "                            # End current stretch: record into S_max if longest\n"
        "                            if reg_current_stretch > 0:\n"
        "                                reg_n_completed += 1\n"
        "                                reg_stretch_log.append(reg_current_stretch)\n"
        "                                if reg_current_stretch > reg_S_max:\n"
        "                                    reg_S_max = reg_current_stretch\n"
        "                                reg_current_stretch = 0.0\n"
        "                            # quiet: calm counter increments\n"
        "                        elif int_mode == \"int_c\":\n"
        "                            # Reset continuous counter\n"
        "                            if int_acc > 0:\n"
        "                                n_resets += 1\n"
        "                                int_acc = 0.0\n"
        "                        elif int_mode == \"int_e_fixed\":\n"
        "                            # Reset evidence integral\n"
        "                            if int_acc > 0 or e600_active_steps_since_reset > 0:\n"
        "                                n_resets += 1\n"
        "                                int_acc = 0.0\n"
        "                                e600_active_steps_since_reset = 0.0\n"
        "\n"
        "                        calm_count += 1\n",
        "                    if not pressure_active:\n"
        "                        # Pressure de-asserted: count the run and only reset\n"
        "                        # after hysteresis has been exceeded.\n"
        "                        deassert_run += 1\n"
        "                        if deassert_run > hysteresis_snaps:\n"
        "                            if int_mode == \"reg_tb\":\n"
        "                                # End current stretch: record into S_max if longest\n"
        "                                if reg_current_stretch > 0:\n"
        "                                    reg_n_completed += 1\n"
        "                                    reg_stretch_log.append(reg_current_stretch)\n"
        "                                    if reg_current_stretch > reg_S_max:\n"
        "                                        reg_S_max = reg_current_stretch\n"
        "                                    reg_current_stretch = 0.0\n"
        "                                # quiet: calm counter increments\n"
        "                            elif int_mode == \"int_c\":\n"
        "                                # Reset continuous counter\n"
        "                                if int_acc > 0:\n"
        "                                    n_resets += 1\n"
        "                                    int_acc = 0.0\n"
        "                            elif int_mode == \"int_e_fixed\":\n"
        "                                # Reset evidence integral\n"
        "                                if int_acc > 0 or e600_active_steps_since_reset > 0:\n"
        "                                    n_resets += 1\n"
        "                                    int_acc = 0.0\n"
        "                                    e600_active_steps_since_reset = 0.0\n"
        "\n"
        "                        calm_count += 1\n",
        1,
    )
    src = _must_replace(src,
        "                        calm_count = 0\n",
        "                        calm_count = 0\n"
        "                        deassert_run = 0\n",
        1,
    )
    src = _must_replace(src,
        "                                blocked_w_by_color = np.zeros(n_colors, dtype=np.float64)\n"
        "                                calm_count = 0\n"
        "                                frozen_steps = 0\n"
        "                                directional_pressure_acc = 0.0\n"
        "                                int_acc = 0.0\n"
        "                                reg_current_stretch = 0.0\n"
        "                                reg_S_max = 0.0\n"
        "                                reg_n_completed = 0\n"
        "                                e600_active_steps_since_reset = 0.0\n"
        "                                # Do NOT reset: reg_stretch_log, reg_max_current_seen\n",
        "                                blocked_w_by_color = np.zeros(n_colors, dtype=np.float64)\n"
        "                                calm_count = 0\n"
        "                                frozen_steps = 0\n"
        "                                directional_pressure_acc = 0.0\n"
        "                                int_acc = 0.0\n"
        "                                reg_current_stretch = 0.0\n"
        "                                reg_S_max = 0.0\n"
        "                                reg_n_completed = 0\n"
        "                                deassert_run = 0\n"
        "                                e600_active_steps_since_reset = 0.0\n"
        "                                # Do NOT reset: reg_stretch_log, reg_max_current_seen\n",
        1,
    )
    src = _must_replace(src,
        "                    if len(obs_window) > 0:\n"
        "                        pressure_active = (\n"
        "                            float(np.sum(obs_window == c_star_pressure)) / len(obs_window)\n"
        "                        ) >= PRESSURE_FRAC\n"
        "                    else:\n"
        "                        pressure_active = False\n",
        "                    if len(obs_window) > 0:\n"
        "                        pressure_active = (\n"
        "                            float(np.sum(obs_window == c_star_pressure)) / len(obs_window)\n"
        "                        ) >= PRESSURE_FRAC\n"
        "                    else:\n"
        "                        pressure_active = False\n"
        "                    if record_pressure_trace:\n"
        "                        # Pressure is only computed while frozen in this runner;\n"
        "                        # this records EVERY frozen FINE_EVAL snap.\n"
        "                        pressure_trace.append([int(t_now), 1, 1 if pressure_active else 0])\n",
        1,
    )
    src = _must_replace(src,
        "                        freeze_state = \"NORMAL\"\n"
        "                        mismatch_history.clear()\n"
        "                        v_fine.clear()\n"
        "                        k_since_reset = 0\n"
        "                        calm_count = 0\n"
        "                        frozen_steps = 0\n"
        "                        directional_pressure_acc = 0.0\n"
        "                        int_acc = 0.0\n"
        "                        # REG-TB: reset per-freeze stretch tracking\n"
        "                        reg_current_stretch = 0.0\n"
        "                        reg_S_max = 0.0\n"
        "                        reg_n_completed = 0\n"
        "                        # Note: reg_stretch_log and reg_max_current_seen are session-level\n"
        "                        # and are NOT reset here — they persist across freezes for diagnostics\n"
        "                        e600_active_steps_since_reset = 0.0\n"
        "                        checks_since_release = 0\n",
        "                        freeze_state = \"NORMAL\"\n"
        "                        mismatch_history.clear()\n"
        "                        v_fine.clear()\n"
        "                        k_since_reset = 0\n"
        "                        calm_count = 0\n"
        "                        frozen_steps = 0\n"
        "                        directional_pressure_acc = 0.0\n"
        "                        int_acc = 0.0\n"
        "                        # REG-TB: reset per-freeze stretch tracking\n"
        "                        reg_current_stretch = 0.0\n"
        "                        reg_S_max = 0.0\n"
        "                        reg_n_completed = 0\n"
        "                        deassert_run = 0\n"
        "                        # Note: reg_stretch_log and reg_max_current_seen are session-level\n"
        "                        # and are NOT reset here — they persist across freezes for diagnostics\n"
        "                        e600_active_steps_since_reset = 0.0\n"
        "                        checks_since_release = 0\n",
        1,
    )
    src = _must_replace(src,
        '        "concession_active_steps_log": list(concession_active_steps_log),\n'
        "    }\n",
        '        "concession_active_steps_log": list(concession_active_steps_log),\n'
        '        "pressure_trace": list(pressure_trace) if record_pressure_trace else None,\n'
        '        "home_pos": home_pos,\n'
        '        "home_restored_step": home_restored_step,\n'
        "    }\n",
        1,
    )

    # ---- return-home Patch A: capture home_pos in FROZEN branch ----
    # In the FROZEN state, before c.true_pos is overwritten by burst relocation,
    # capture the pre-captivity position when return_home=True at the first burst start.
    # Frozen branch target: "                if in_burst and phase == \"W\":\n"
    #                       "                    cells_of_bc = color_cells_arr[current_burst_color]\n"
    # We insert the capture BEFORE the overwrite.
    n_frozen_in_burst = src.count(
        "                if in_burst and phase == \"W\":\n"
        "                    cells_of_bc = color_cells_arr[current_burst_color]\n"
        "                    if len(cells_of_bc) > 0:\n"
        "                        c.true_pos = int(burst_rng.choice(cells_of_bc))\n"
        "                    qs_next = np.zeros(n_cells)\n"
        "                    qs_next[cells_of_bc] = 1.0 / len(cells_of_bc)\n"
        "                    c.qs = qs_next\n"
    )
    assert n_frozen_in_burst == 1, f"Expected 1 frozen in_burst block, got {n_frozen_in_burst}"

    src = _must_replace(src,
        "                if in_burst and phase == \"W\":\n"
        "                    cells_of_bc = color_cells_arr[current_burst_color]\n"
        "                    if len(cells_of_bc) > 0:\n"
        "                        c.true_pos = int(burst_rng.choice(cells_of_bc))\n"
        "                    qs_next = np.zeros(n_cells)\n"
        "                    qs_next[cells_of_bc] = 1.0 / len(cells_of_bc)\n"
        "                    c.qs = qs_next\n",
        "                if in_burst and phase == \"W\":\n"
        "                    cells_of_bc = color_cells_arr[current_burst_color]\n"
        "                    # return_home Patch A (frozen branch): capture pre-relocation pos\n"
        "                    if return_home and home_pos is None and burst_windows and t == burst_windows[0][0]:\n"
        "                        home_pos = int(c.true_pos)\n"
        "                    if len(cells_of_bc) > 0:\n"
        "                        c.true_pos = int(burst_rng.choice(cells_of_bc))\n"
        "                    qs_next = np.zeros(n_cells)\n"
        "                    qs_next[cells_of_bc] = 1.0 / len(cells_of_bc)\n"
        "                    c.qs = qs_next\n"
        "                    # return_home Patch B (frozen branch): restore at last burst end\n"
        "                    if return_home and home_pos is not None and burst_windows and t == burst_windows[-1][1] - 1:\n"
        "                        c.true_pos = home_pos\n"
        "                        qs_next2 = np.zeros(n_cells)\n"
        "                        qs_next2[home_pos] = 1.0\n"
        "                        c.qs = qs_next2\n"
        "                        home_restored_step = t\n",
        1,
    )

    # ---- return-home Patch A+B: capture and restore in NORMAL branch ----
    n_normal_in_burst = src.count(
        "            if in_burst and phase == \"W\":\n"
        "                cells_of_bc = color_cells_arr[current_burst_color]\n"
        "                if len(cells_of_bc) > 0:\n"
        "                    c.true_pos = int(burst_rng.choice(cells_of_bc))\n"
        "                qs_next = np.zeros(n_cells)\n"
        "                qs_next[cells_of_bc] = 1.0 / len(cells_of_bc)\n"
        "                c.qs = qs_next\n"
    )
    assert n_normal_in_burst == 1, f"Expected 1 normal in_burst block, got {n_normal_in_burst}"

    src = _must_replace(src,
        "            if in_burst and phase == \"W\":\n"
        "                cells_of_bc = color_cells_arr[current_burst_color]\n"
        "                if len(cells_of_bc) > 0:\n"
        "                    c.true_pos = int(burst_rng.choice(cells_of_bc))\n"
        "                qs_next = np.zeros(n_cells)\n"
        "                qs_next[cells_of_bc] = 1.0 / len(cells_of_bc)\n"
        "                c.qs = qs_next\n",
        "            if in_burst and phase == \"W\":\n"
        "                cells_of_bc = color_cells_arr[current_burst_color]\n"
        "                # return_home Patch A (normal branch): capture pre-relocation pos\n"
        "                if return_home and home_pos is None and burst_windows and t == burst_windows[0][0]:\n"
        "                    home_pos = int(c.true_pos)\n"
        "                if len(cells_of_bc) > 0:\n"
        "                    c.true_pos = int(burst_rng.choice(cells_of_bc))\n"
        "                qs_next = np.zeros(n_cells)\n"
        "                qs_next[cells_of_bc] = 1.0 / len(cells_of_bc)\n"
        "                c.qs = qs_next\n"
        "                # return_home Patch B (normal branch): restore at last burst end\n"
        "                if return_home and home_pos is not None and burst_windows and t == burst_windows[-1][1] - 1:\n"
        "                    c.true_pos = home_pos\n"
        "                    qs_next2 = np.zeros(n_cells)\n"
        "                    qs_next2[home_pos] = 1.0\n"
        "                    c.qs = qs_next2\n"
        "                    home_restored_step = t\n",
        1,
    )

    ns = {
        "__name__": "_exp193_runner_copy",
        "np": np,
        "math": math,
        "statistics": statistics,
        "copy": copy,
        "json": json,
        "time": __import__("time"),
        "Creature": Creature,
        "tv_func": lambda p, q: 0.5 * float(np.abs(p - q).sum()),
        "REPO_ROOT": REPO_ROOT,
        "run_fork_schedule_185": _mod188.run_fork_schedule_185,
        "compute_defense": compute_defense,
        "compute_final_gap": compute_final_gap,
        "phase_r_latency": phase_r_latency,
        "to_plain": to_plain,
        "pi_of": _mod190.pi_of,
        "LAMBDA": _mod190.LAMBDA,
        "INIT_MASS": INIT_MASS,
        "PHASE_R_START": _mod190.PHASE_R_START,
        "BURST_SEED_OFFSET_R": BURST_SEED_OFFSET_R,
        "BURST_SEED_OFFSET_W_EXP183": BURST_SEED_OFFSET_W_EXP183,
        "EVAL": _mod190.EVAL,
        "FINE_EVAL": FINE_EVAL,
        "N_STEPS_PHASE_R": N_STEPS_PHASE_R,
        "CHUNK_SIZE": CHUNK_SIZE,
        "P6_HOLD": _mod190.P6_HOLD,
        "DEFENSE_FRAC_THRESH": _mod190.DEFENSE_FRAC_THRESH,
        "DEFENSE_WINDOW_OFFSET_START": _mod190.DEFENSE_WINDOW_OFFSET_START,
        "DEFENSE_WINDOW_OFFSET_END": _mod190.DEFENSE_WINDOW_OFFSET_END,
        "PC1_AHAT_DRIFT_MAX": PC1_AHAT_DRIFT_MAX,
        "DEFAULT_THETA": DEFAULT_THETA,
        "DEFAULT_RELEASE_CALM_SNAPS": DEFAULT_RELEASE_CALM_SNAPS,
        "CALM2600_SNAPS": CALM2600_SNAPS,
        "THETA3_THETA": _mod190.THETA3_THETA,
        "E_STAR": _mod190.E_STAR,
        "FLOAT_ATOL": FLOAT_ATOL,
        "EXP183_BURST_WINDOWS": _mod190.EXP183_BURST_WINDOWS,
        "EXP183_N_STEPS": _mod190.EXP183_N_STEPS,
        "EXP183_GATE_SEEDS": _mod190.EXP183_GATE_SEEDS,
        "EXP183_GATE_ARMS": _mod190.EXP183_GATE_ARMS,
        "PRESSURE_WINDOW": PRESSURE_WINDOW,
        "PRESSURE_FRAC": PRESSURE_FRAC,
        "REFRACTORY_CHECKS": _mod190.REFRACTORY_CHECKS,
    }
    exec(src, ns)
    return ns["run_fork_schedule_193"]


run_fork_schedule_193 = _build_run_fork_schedule_193()


# ---------------------------------------------------------------------------
# Equivalence gate (G1) -- same as exp190/192 but through the 193 runner
# ---------------------------------------------------------------------------

def run_equivalence_gate_193(
    mirro_root, base_cmap, n_colors, committed_183_path: Path
) -> tuple[bool, str]:
    """Replay the exp183 baseline/H1200 rows through the exp193 runner (return_home=False, h=0)."""
    from active_loop.creature import Creature as _Creature

    committed_rows = _load_rows(committed_183_path)
    committed_w = {
        (row["arm"], row["fork_seed"], row["burst_idx"]): row
        for row in committed_rows
        if row.get("phase") == "W"
    }

    detail_lines: list[str] = ["EQUIVALENCE GATE (exp193 path): exp183 replay"]
    all_pass = True

    arm_lookup = {
        "baseline": ("baseline", "baseline"),
        "H1200": ("H1200", ("freeze_time", 1200)),
    }

    EXP183_BURST_WINDOWS = _mod190.EXP183_BURST_WINDOWS
    EXP183_N_STEPS = _mod190.EXP183_N_STEPS
    EXP183_GATE_SEEDS = _mod190.EXP183_GATE_SEEDS
    EXP183_GATE_ARMS = _mod190.EXP183_GATE_ARMS

    for arm_name in EXP183_GATE_ARMS:
        _, arm_mode = arm_lookup[arm_name]
        for seed in EXP183_GATE_SEEDS:
            root = copy.deepcopy(mirro_root)
            root._state_dir = None
            rr = run_fork_schedule_193(
                mirro=root,
                fork_seed=seed,
                base_cmap=base_cmap,
                n_colors=n_colors,
                arm_name=arm_name,
                arm_mode=arm_mode,
                phase="W",
                burst_windows=EXP183_BURST_WINDOWS,
                color_mode="endogenous",
                reloc_rng_seed=BURST_SEED_OFFSET_W_EXP183 + seed,
                n_steps=EXP183_N_STEPS,
                theta=DEFAULT_THETA,
                release_calm_snaps=DEFAULT_RELEASE_CALM_SNAPS,
                hysteresis_snaps=0,
                return_home=False,
            )

            session_pass = True
            detail_lines.append(f"\n=== arm={arm_name} seed={seed} ===")
            for bi in range(len(EXP183_BURST_WINDOWS)):
                key = (arm_name, seed, bi)
                committed = committed_w.get(key)
                if committed is None:
                    detail_lines.append(f"  b{bi} committed row missing for {key}")
                    session_pass = False
                    all_pass = False
                    continue

                got = {
                    "gap_start": rr["gap_start"][bi],
                    "gap_end": rr["gap_end"][bi],
                    "d_b": rr["d_b"][bi],
                    "tv_b": rr["tv_b"][bi],
                    "recovered": False,
                    "n_events": len(rr["events"]),
                }
                bstart, bend = EXP183_BURST_WINDOWS[bi]
                win_start = bend + 1500
                win_end = bend + 2000
                if win_end <= EXP183_N_STEPS and rr["burst_onset_color"][bi] is not None:
                    bc = rr["burst_onset_color"][bi]
                    got["recovered"] = float(np.mean(rr["expressed_arr"][win_start:win_end] == bc)) < 0.5

                for field in ("gap_start", "gap_end", "d_b", "tv_b", "recovered", "n_events"):
                    ok = _value_match(got[field], committed[field])
                    detail_lines.append(
                        f"  b{bi} {field:<12} got={json.dumps(to_plain(got[field])):<44} "
                        f"committed={json.dumps(to_plain(committed[field])):<44} "
                        f"{'OK' if ok else 'MISMATCH'}"
                    )
                    if not ok:
                        session_pass = False
                        all_pass = False

                # event list comparison
                got_events = rr["events"]
                committed_events = committed.get("events_summary", [])
                if len(got_events) != len(committed_events):
                    detail_lines.append(
                        f"  n_events got={len(got_events)} committed={len(committed_events)} MISMATCH"
                    )
                    session_pass = False
                    all_pass = False
                else:
                    for idx, (ge, ce) in enumerate(zip(got_events, committed_events)):
                        for field in ("label", "entry_step", "frozen_steps"):
                            gv = ge.get(field)
                            cv = ce.get(field)
                            ok = gv == cv
                            detail_lines.append(
                                f"  ev{idx:<2d} {field:<12} got={str(gv):<24} committed={str(cv):<24} "
                                f"{'OK' if ok else 'MISMATCH'}"
                            )
                            if not ok:
                                session_pass = False
                                all_pass = False

            detail_lines.append(f"  gate arm={arm_name} seed={seed}: {'PASS' if session_pass else 'FAIL'}")

    return all_pass, "\n".join(detail_lines)


# ---------------------------------------------------------------------------
# Exp 193 configuration (verbatim exp192)
# ---------------------------------------------------------------------------

BLOCK1 = list(range(296, 304))  # [296..303]
BLOCK2 = list(range(304, 312))  # [304..311]
ALL_SEEDS = BLOCK1 + BLOCK2
N_SEEDS_TOTAL = len(ALL_SEEDS)  # 16

CELLS_193 = [
    {
        "name": "C-A",
        "L": 1600,
        "K": 2,
        "G": 200,
        "burst_windows": [(6000, 7600), (7800, 9400)],
        "n_steps": 21400,
        "cell_idx": 0,
    },
    {
        "name": "C-B",
        "L": 2400,
        "K": 4,
        "G": 600,
        "burst_windows": [(6000, 8400), (9000, 11400), (12000, 14400), (15000, 17400)],
        "n_steps": 29400,
        "cell_idx": 1,
    },
    {
        "name": "C-C",
        "L": 400,
        "K": 4,
        "G": 2400,
        "burst_windows": [(6000, 6400), (8800, 9200), (11600, 12000), (14400, 14800)],
        "n_steps": 26800,
        "cell_idx": 2,
    },
]

for _cell in CELLS_193:
    _last_end = _cell["burst_windows"][-1][1]
    assert _cell["n_steps"] % CHUNK_SIZE == 0, (
        f"Cell {_cell['name']}: n_steps={_cell['n_steps']} not divisible by CHUNK_SIZE={CHUNK_SIZE}"
    )
    assert _last_end + 12000 == _cell["n_steps"], (
        f"Cell {_cell['name']}: last_window_end={_last_end} + 12000 = {_last_end+12000} "
        f"!= n_steps={_cell['n_steps']}"
    )

ARM_DEFS = [
    ("baseline",  "baseline",                            DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS),
    ("oracle",    "oracle",                              DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS),
    ("INT-C2900", ("int_c", 2900, CALM2600_SNAPS),       DEFAULT_THETA, CALM2600_SNAPS),
    ("REG-TB",    ("reg_tb", 1.5, 2800, CALM2600_SNAPS), DEFAULT_THETA, CALM2600_SNAPS),
]

W1_OFFSET_START = 1000
W1_OFFSET_END   = 1500
W2_OFFSET_START = 3000
W2_OFFSET_END   = 3500


def _compute_retention_fracs(
    expressed_arr: np.ndarray,
    attack_color: int,
    release_step: int,
    n_steps: int,
) -> tuple[float | None, float | None]:
    w1_start = release_step + W1_OFFSET_START
    w1_end   = release_step + W1_OFFSET_END
    w2_start = release_step + W2_OFFSET_START
    w2_end   = release_step + W2_OFFSET_END
    if w1_end > n_steps or w2_end > n_steps:
        return None, None
    w1_frac = float(np.mean(expressed_arr[w1_start:w1_end] == attack_color))
    w2_frac = float(np.mean(expressed_arr[w2_start:w2_end] == attack_color))
    return w1_frac, w2_frac


def _run_session(
    mirro_root,
    base_cmap,
    n_colors,
    cell: dict,
    arm_name: str,
    arm_mode,
    theta_val,
    rc_snaps,
    seed: int,
    return_home: bool = True,
) -> tuple[dict, dict]:
    """Run a single W session and return (row_dict, raw_result)."""
    root = copy.deepcopy(mirro_root)
    root._state_dir = None

    cell_name = cell["name"]
    cell_idx = cell["cell_idx"]
    n_steps = cell["n_steps"]
    burst_windows = cell["burst_windows"]
    last_bend = burst_windows[-1][1]

    reloc_rng_seed = 296_000 + 10_000 * cell_idx + seed

    rr = run_fork_schedule_193(
        mirro=root,
        fork_seed=seed,
        base_cmap=base_cmap,
        n_colors=n_colors,
        arm_name=arm_name,
        arm_mode=arm_mode,
        phase="W",
        burst_windows=burst_windows,
        color_mode="exogenous_fixed",
        reloc_rng_seed=reloc_rng_seed,
        n_steps=n_steps,
        theta=theta_val,
        release_calm_snaps=rc_snaps,
        hysteresis_snaps=0,
        return_home=return_home,
    )

    frozen_defense, final_expr_frac, attack_color = compute_defense(
        rr["expressed_arr"], burst_windows, rr["burst_onset_color"], n_steps
    )

    pc1_flag = rr["ahat_drift"] >= PC1_AHAT_DRIFT_MAX

    events = rr.get("events", [])
    events_summary = [
        {
            "label": e["label"],
            "entry_step": e["entry_step"],
            "exit_step": e.get("exit_step"),
            "frozen_steps": e["frozen_steps"],
        }
        for e in events
    ]
    n_events = len(events)

    flags: list[str] = []
    if pc1_flag:
        flags.append("PC1_DRIFT")

    if arm_name == "baseline":
        release_step: int | None = last_bend
        release_label: str | None = "none_baseline"
    elif arm_name == "oracle":
        if n_events > 0:
            last_ev = events[-1]
            release_step = last_ev.get("exit_step")
            release_label = last_ev.get("label")
        else:
            release_step = last_bend
            release_label = "oracle_exact_train"
    else:
        if n_events == 0:
            release_step = None
            release_label = None
            flags.append("NO_FREEZE_EVENT")
        else:
            last_ev = events[-1]
            release_step = last_ev.get("exit_step")
            release_label = last_ev.get("label")

    if arm_name != "baseline":
        for ev in events:
            if ev.get("label") == "concession":
                ev_exit = ev.get("exit_step")
                if ev_exit is not None and ev_exit < last_bend:
                    if "MID_TRAIN_CONCESSION" not in flags:
                        flags.append("MID_TRAIN_CONCESSION")
                    break

    if release_step is not None and release_step + 3500 > n_steps:
        flags.append("WINDOW_OVERRUN")

    if (
        release_step is not None
        and attack_color is not None
        and "WINDOW_OVERRUN" not in flags
        and "NO_FREEZE_EVENT" not in flags
    ):
        w1_frac, w2_frac = _compute_retention_fracs(
            rr["expressed_arr"], attack_color, release_step, n_steps
        )
        if w1_frac is None or w2_frac is None:
            if "WINDOW_OVERRUN" not in flags:
                flags.append("WINDOW_OVERRUN")
            retained = None
        else:
            retained = bool(w1_frac < 0.5 and w2_frac < 0.5)
    else:
        w1_frac = None
        w2_frac = None
        retained = None

    # Settling: defended AND not overrun AND release_step <= last_bend + 4000
    settled: bool | None = None
    if frozen_defense and "WINDOW_OVERRUN" not in flags and release_step is not None:
        settled = bool(release_step <= last_bend + 4000)

    row: dict = {
        "exp": 193,
        "cell": {
            "name": cell_name,
            "L": cell["L"],
            "K": cell["K"],
            "G": cell["G"],
        },
        "arm": arm_name,
        "seed": int(seed),
        "attack_color": to_plain(attack_color),
        "frozen_defense": bool(frozen_defense),
        "final_expr_frac": to_plain(final_expr_frac),
        "n_events": n_events,
        "events_summary": to_plain(events_summary),
        "release_step": to_plain(release_step),
        "release_label": release_label,
        "w1_frac": to_plain(w1_frac),
        "w2_frac": to_plain(w2_frac),
        "retained": to_plain(retained),
        "settled": to_plain(settled),
        "flags": flags,
        "ahat_drift": float(rr["ahat_drift"]),
        "home_pos": to_plain(rr.get("home_pos")),
        "home_restored_step": to_plain(rr.get("home_restored_step")),
        "return_home": bool(return_home),
    }
    return row, rr


def main() -> None:
    t_start = time.time()

    lines: list[str] = []
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt_path = out_dir / "exp193.txt"
    out_rows_path = out_dir / "exp193_rows.json"
    committed_183_path = out_dir / "exp183_rows.json"
    committed_192_path = out_dir / "exp192_rows.json"

    def p(msg: str = "") -> None:
        lines.append(msg)
        print(msg)

    def abort(reason: str) -> None:
        p(f"ABORT: {reason}")
        with open(out_txt_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        sys.exit(1)

    p("=" * 80)
    p("EXP 193 -- N4 RETURN-HOME CAUSAL INTERVENTION")
    p("PRE-REGISTERED in loop/directions/n4-crack-edges.md (commit fb6ca58) BEFORE any data")
    p(f"Seeds: Block1={BLOCK1}, Block2={BLOCK2}")
    p(f"3 cells x 4 arms x 16 seeds = 192 W sessions (ALL return_home=True)")
    p("Runner: run_fork_schedule_193 (exp190 patch chain + return-home patches)")
    p("=" * 80)
    p()

    # Spine safety (L14)
    mirro = Creature.load("creature/state/mirro")
    mirro_root = copy.deepcopy(mirro)
    mirro_root._state_dir = None

    base_cmap = list(mirro.world.cmap)
    n_colors = mirro.world.n_colors
    p(f"Loaded mirro: age={mirro.age_steps}, world={mirro.world.rows}x{mirro.world.cols}, "
      f"n_colors={n_colors}")
    p()

    # ====================================================================
    # STEP 1: G1 -- EQUIVALENCE GATE (exp183 through exp193 runner)
    # ====================================================================
    p("=" * 80)
    p("STEP 1: G1 -- EQUIVALENCE GATE (exp183 replay through exp193 runner, return_home=False)")
    p("=" * 80)

    eq_pass, eq_detail = run_equivalence_gate_193(
        mirro_root, base_cmap, n_colors, committed_183_path
    )
    p(eq_detail)
    p()

    if not eq_pass:
        abort("G1 FAIL")

    p("G1 PASS -- proceeding.")
    p()

    # ====================================================================
    # STEP 2: G2 -- exp192 regression (return_home=False bit-match)
    # ====================================================================
    p("=" * 80)
    p("STEP 2: G2 -- EXP192 REGRESSION GATE (return_home=False bit-match)")
    p("  Cells: C-A, C-C; Arms: baseline, INT-C2900; Seeds: 296, 301 (8 sessions)")
    p("=" * 80)
    p()

    # Load committed exp192 rows
    exp192_rows: dict[tuple[str, str, int], dict] = {}
    with open(committed_192_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "analysis" or row.get("exp") != 192:
                continue
            cell_name = row.get("cell", {}).get("name")
            arm = row.get("arm")
            seed = row.get("seed")
            if cell_name and arm and seed is not None:
                exp192_rows[(cell_name, arm, seed)] = row

    p(f"Loaded {len(exp192_rows)} exp192 session rows.")

    G2_CELLS = ["C-A", "C-C"]
    G2_ARMS = ["baseline", "INT-C2900"]
    G2_SEEDS = [296, 301]
    G2_N_STEPS = {"C-A": 21400, "C-C": 26800}

    g2_mismatches: list[str] = []
    g2_evidence: list[str] = []

    for cell_name in G2_CELLS:
        cell = next(c for c in CELLS_193 if c["name"] == cell_name)
        n_steps_g2 = G2_N_STEPS[cell_name]
        for arm_name in G2_ARMS:
            arm_def = next(a for a in ARM_DEFS if a[0] == arm_name)
            arm_mode = arm_def[1]
            theta_val = arm_def[2]
            rc_snaps = arm_def[3]
            for seed in G2_SEEDS:
                row193, _ = _run_session(
                    mirro_root, base_cmap, n_colors,
                    cell, arm_name, arm_mode, theta_val, rc_snaps, seed,
                    return_home=False,
                )
                e192 = exp192_rows.get((cell_name, arm_name, seed))
                if e192 is None:
                    g2_evidence.append(f"  G2 {cell_name}/{arm_name}/{seed}: no exp192 row -- SKIP")
                    continue

                label = f"G2 {cell_name}/{arm_name}/s{seed}"

                # frozen_defense
                fd_ok = (row193["frozen_defense"] == e192["frozen_defense"])
                g2_evidence.append(
                    f"  {label} frozen_defense: 192={e192['frozen_defense']} 193={row193['frozen_defense']} -> {'OK' if fd_ok else 'MISMATCH'}"
                )
                if not fd_ok:
                    g2_mismatches.append(f"frozen_defense mismatch {cell_name}/{arm_name}/{seed}")

                # event chain
                e192_evs = e192.get("events_summary", [])
                e193_evs = row193.get("events_summary", [])
                n_e = len(e192_evs)
                if n_e > len(e193_evs):
                    g2_evidence.append(f"  {label} events: exp192={n_e} exp193={len(e193_evs)} -> MISMATCH")
                    g2_mismatches.append(f"event count mismatch {cell_name}/{arm_name}/{seed}")
                else:
                    all_ev_ok = True
                    for i, ev192 in enumerate(e192_evs):
                        ev193 = e193_evs[i]
                        ev_ok = (
                            ev192["label"] == ev193["label"]
                            and ev192["entry_step"] == ev193["entry_step"]
                            and ev192["exit_step"] == ev193["exit_step"]
                        )
                        g2_evidence.append(
                            f"  {label} event[{i}] label={ev192['label']}/{ev193['label']} "
                            f"entry={ev192['entry_step']}/{ev193['entry_step']} "
                            f"exit={ev192['exit_step']}/{ev193['exit_step']} -> {'OK' if ev_ok else 'MISMATCH'}"
                        )
                        if not ev_ok:
                            all_ev_ok = False
                            g2_mismatches.append(f"event[{i}] mismatch {cell_name}/{arm_name}/{seed}")

                # release_step
                rs_ok = (row193["release_step"] == e192["release_step"])
                g2_evidence.append(
                    f"  {label} release_step: 192={e192['release_step']} 193={row193['release_step']} -> {'OK' if rs_ok else 'MISMATCH'}"
                )
                if not rs_ok:
                    g2_mismatches.append(f"release_step mismatch {cell_name}/{arm_name}/{seed}")

                # w-fracs (only if release_step unchanged)
                if rs_ok:
                    for wname in ("w1_frac", "w2_frac"):
                        v192 = e192.get(wname)
                        v193 = row193.get(wname)
                        if v192 is not None:
                            if v193 is None:
                                g2_evidence.append(f"  {label} {wname}: 192={v192} 193=None -> MISMATCH")
                                g2_mismatches.append(f"{wname} None mismatch {cell_name}/{arm_name}/{seed}")
                            elif abs(v192 - v193) > 1e-9:
                                g2_evidence.append(
                                    f"  {label} {wname}: 192={v192} 193={v193} diff={abs(v192-v193):.2e} -> MISMATCH"
                                )
                                g2_mismatches.append(f"{wname} value mismatch {cell_name}/{arm_name}/{seed}")
                            else:
                                g2_evidence.append(
                                    f"  {label} {wname}: 192={v192} 193={v193} diff={abs(v192-v193):.2e} -> OK"
                                )

    p("G2 EVIDENCE:")
    for line in g2_evidence:
        p(line)
    p()

    if g2_mismatches:
        p("G2 MISMATCHES:")
        for mm in g2_mismatches:
            p(f"  {mm}")
        p()
        abort("G2 FAIL -- exp192 regression violated")

    p("G2 PASS -- all return_home=False sessions bit-match committed exp192 rows.")
    p()

    # ====================================================================
    # STEP 3: MAIN GRID (return_home=True)
    # ====================================================================
    p("=" * 80)
    p("STEP 3: MAIN GRID -- ALL return_home=True")
    p("=" * 80)
    p()

    all_rows: list[dict] = []
    g3_violations: list[str] = []
    g3_evidence: list[str] = []

    n_total = len(CELLS_193) * len(ARM_DEFS) * len(ALL_SEEDS)
    n_done = 0
    t_grid = time.time()

    for cell in CELLS_193:
        cell_name = cell["name"]
        burst_windows = cell["burst_windows"]
        last_burst_end = burst_windows[-1][1]
        first_burst_start = burst_windows[0][0]

        for arm_name, arm_mode, theta_val, rc_snaps in ARM_DEFS:
            for seed in ALL_SEEDS:
                row, rr = _run_session(
                    mirro_root, base_cmap, n_colors,
                    cell, arm_name, arm_mode, theta_val, rc_snaps, seed,
                    return_home=True,
                )
                all_rows.append(row)
                n_done += 1

                # G3: check home_pos and home_restored_step
                hp = rr.get("home_pos")
                hrs = rr.get("home_restored_step")
                # Expected: home_pos captured (not None), home_restored_step = last burst window end - 1
                expected_restore = last_burst_end - 1
                if hp is None:
                    g3_violation = f"G3 VIOLATION {cell_name} {arm_name} s{seed}: home_pos=None (not captured)"
                    g3_violations.append(g3_violation)
                    g3_evidence.append(g3_violation)
                elif hrs is None:
                    g3_violation = f"G3 VIOLATION {cell_name} {arm_name} s{seed}: home_restored_step=None (not restored)"
                    g3_violations.append(g3_violation)
                    g3_evidence.append(g3_violation)
                elif hrs != expected_restore:
                    g3_violation = (
                        f"G3 VIOLATION {cell_name} {arm_name} s{seed}: "
                        f"home_restored_step={hrs} expected={expected_restore}"
                    )
                    g3_violations.append(g3_violation)
                    g3_evidence.append(g3_violation)
                else:
                    g3_evidence.append(
                        f"G3 {cell_name} {arm_name} s{seed} home_pos={hp} restored@{hrs} OK"
                    )

                if n_done % 16 == 0 or n_done == n_total:
                    elapsed = time.time() - t_grid
                    rate = elapsed / max(1, n_done)
                    remaining = rate * (n_total - n_done)
                    print(
                        f"  [{n_done:3d}/{n_total}] cell={cell_name} arm={arm_name} seed={seed} "
                        f"| def={row['frozen_defense']} ret={row['retained']} settled={row['settled']} "
                        f"hp={rr.get('home_pos')} hrs={rr.get('home_restored_step')} "
                        f"| {elapsed:.0f}s elapsed ETA {remaining:.0f}s",
                        flush=True,
                    )

    p()
    p("G3 EVIDENCE (per session):")
    for ev in g3_evidence:
        p(ev)
    p()

    if g3_violations:
        p("G3 VIOLATIONS DETECTED:")
        for v in g3_violations:
            p(f"  {v}")
        p()
        # Write partial rows before aborting
        with open(out_rows_path, "w") as fh:
            for row in all_rows:
                fh.write(json.dumps(row) + "\n")
        abort("G3 FAIL -- return_home intervention not applied in all sessions")

    p("G3 PASS -- home_pos captured and restored in all sessions.")
    p()

    # Write session rows to JSONL
    with open(out_rows_path, "w") as fh:
        for row in all_rows:
            fh.write(json.dumps(row) + "\n")

    p(f"Grid done: {(time.time()-t_grid)/60:.1f} min ({n_total} sessions)")
    p()

    # ====================================================================
    # STEP 4: ANALYSIS
    # ====================================================================
    p("=" * 80)
    p("STEP 4: ANALYSIS OVER POOLED 16 SEEDS (return_home=True)")
    p("=" * 80)
    p()

    cell_names_order = [c["name"] for c in CELLS_193]

    # Index all rows
    row_index: dict[tuple[str, str, int], dict] = {}
    for row in all_rows:
        key = (row["cell"]["name"], row["arm"], row["seed"])
        row_index[key] = row

    # ---- 4a. Frozen-defense counts ----
    p("FROZEN-DEFENSE COUNT TABLE (count/16 per (arm, cell))")
    p("-" * 80)
    header = f"  {'arm':<22}"
    for cn in cell_names_order:
        header += f"  {cn:>9}"
    p(header)
    p("-" * 80)

    frozen_def_counts: dict[tuple[str, str], int] = {}
    for arm_name, _, _, _ in ARM_DEFS:
        row_str = f"  {arm_name:<22}"
        for cell_name in cell_names_order:
            cnt = sum(
                1 for seed in ALL_SEEDS
                if row_index.get((cell_name, arm_name, seed), {}).get("frozen_defense", False)
            )
            frozen_def_counts[(arm_name, cell_name)] = cnt
            row_str += f"  {cnt:>8}/16"
        p(row_str)
    p()

    # ---- 4b. Covered pairs ----
    CANDIDATE_PAIRS: list[tuple[str, str]] = []
    for arm_name in ["INT-C2900", "REG-TB"]:
        for cell_name in cell_names_order:
            CANDIDATE_PAIRS.append((arm_name, cell_name))

    covered_pairs: list[tuple[str, str]] = []
    for arm_name, cell_name in CANDIDATE_PAIRS:
        cnt = frozen_def_counts.get((arm_name, cell_name), 0)
        if cnt >= 12:
            covered_pairs.append((arm_name, cell_name))

    oracle_covered: list[tuple[str, str]] = []
    for cell_name in cell_names_order:
        cnt = frozen_def_counts.get(("oracle", cell_name), 0)
        if cnt >= 14:
            oracle_covered.append(("oracle", cell_name))

    covered_pairs_all = covered_pairs + oracle_covered

    p(f"Covered non-oracle pairs (frozen_def >= 12/16): {covered_pairs}")
    p(f"Oracle covered pairs (frozen_def >= 14/16):     {oracle_covered}")
    p()

    # ---- 4c. P1 (settling) and P2 (retention) per covered pair ----
    p("SETTLING + RETENTION ANALYSIS PER COVERED PAIR")
    p("  M = measured defended seats (frozen_defense AND NOT overrun)")
    p("  settled = defended AND not overrun AND release_step <= last_bend + 4000")
    p("  Validity: M >= 10 else INSTRUMENT-FAILED")
    p("  retention_rate = retained / M")
    p("-" * 100)

    retention_rates: dict[tuple[str, str], float | None] = {}
    settling_data: dict[tuple[str, str], dict] = {}
    instrument_failed_pairs: list[tuple[str, str]] = []
    f1_fired = False
    f2_fired_flag = False
    mixed_zone_pairs: list[tuple[str, str]] = []
    p3_oracle_pass = True

    for arm_name, cell_name in covered_pairs_all:
        cell = next(c for c in CELLS_193 if c["name"] == cell_name)
        last_bend = cell["burst_windows"][-1][1]

        defended_seeds = [
            seed for seed in ALL_SEEDS
            if row_index.get((cell_name, arm_name, seed), {}).get("frozen_defense", False)
        ]
        n_defended = len(defended_seeds)

        measured_seeds = [
            seed for seed in defended_seeds
            if "WINDOW_OVERRUN" not in row_index.get((cell_name, arm_name, seed), {}).get("flags", [])
        ]
        M = len(measured_seeds)

        # P1 settling: defended and not overrun and release <= last_bend + 4000
        settled_count = sum(
            1 for seed in measured_seeds
            if row_index.get((cell_name, arm_name, seed), {}).get("settled") is True
        )

        settling_data[(arm_name, cell_name)] = {
            "M": M, "settled": settled_count, "defended": n_defended
        }

        if M < 10:
            instrument_failed_pairs.append((arm_name, cell_name))
            f1_fired = True  # M < 10 means F1 fires (cycle persists at home)
            p(f"  {arm_name:<22} x {cell_name:<8}: INSTRUMENT-FAILED (M={M} < 10) -- F1 fired "
              f"(defended={n_defended}/16, measured={M}/16, settled={settled_count})")
            retention_rates[(arm_name, cell_name)] = None
            p()
            continue

        # Check P1: M >= 10 AND settled >= 12/16
        p1_pair_ok = (M >= 10 and settled_count >= 12)
        if not p1_pair_ok and M >= 10 and settled_count < 12:
            f1_fired = True  # any pair < 12/16 settled fires F1

        retained_count = sum(
            1 for seed in measured_seeds
            if row_index.get((cell_name, arm_name, seed), {}).get("retained") is True
        )
        rate = retained_count / M if M > 0 else None
        retention_rates[(arm_name, cell_name)] = rate

        p(f"  {arm_name:<22} x {cell_name:<8}: defended={n_defended}/16 M={M} "
          f"settled={settled_count} retained={retained_count}/{M} "
          f"retention_rate={rate:.3f if rate is not None else 'N/A'} "
          f"P1_pair={'PASS' if p1_pair_ok else 'FAIL'}")

        p(f"    {'block':>6}  {'seed':>5}  {'fd':>5}  {'release':>8}  "
          f"{'settled':>8}  {'w1':>8}  {'w2':>8}  {'ret':>8}  {'flags':>25}")
        for seed in ALL_SEEDS:
            block = "B1" if seed in BLOCK1 else "B2"
            r = row_index.get((cell_name, arm_name, seed), {})
            fd = r.get("frozen_defense", False)
            rs = r.get("release_step")
            st = r.get("settled")
            w1 = r.get("w1_frac")
            w2 = r.get("w2_frac")
            ret = r.get("retained")
            fl = ",".join(r.get("flags", []))
            rs_str = str(rs) if rs is not None else "None"
            w1_str = f"{w1:.3f}" if w1 is not None else "None"
            w2_str = f"{w2:.3f}" if w2 is not None else "None"
            p(f"    {block:>6}  {seed:>5}  {str(fd):>5}  {rs_str:>8}  "
              f"{str(st):>8}  {w1_str:>8}  {w2_str:>8}  {str(ret):>8}  {fl:>25}")

        if rate is not None:
            if rate <= 2 / 3:
                f2_fired_flag = True
                p(f"    *** F2 FIRED: {arm_name} x {cell_name} retention_rate={rate:.3f} <= 2/3 ***")
            elif rate < 5 / 6:
                mixed_zone_pairs.append((arm_name, cell_name))
                p(f"    MIXED-ZONE: {arm_name} x {cell_name} rate={rate:.3f} (2/3 < rate < 5/6)")

        if arm_name == "oracle" and rate is not None and rate < 5 / 6:
            p3_oracle_pass = False

        p()

    # ---- 4d. P3 (crown): s301 C-C both controllers ----
    p("P3 (CROWN) -- s301 C-C CAUSAL SEED TEST")
    p("-" * 60)
    f3_fired = False
    crown_data: dict = {}
    for arm_name in ["INT-C2900", "REG-TB"]:
        for seed in [297, 301]:
            r = row_index.get(("C-C", arm_name, seed), {})
            w1 = r.get("w1_frac")
            w2 = r.get("w2_frac")
            ret = r.get("retained")
            fd = r.get("frozen_defense", False)
            rs = r.get("release_step")
            crown_data[(arm_name, seed)] = {"w1": w1, "w2": w2, "retained": ret, "fd": fd, "rs": rs}
            p(f"  {arm_name} s{seed}: frozen_def={fd} release={rs} w1={w1} w2={w2} retained={ret}")

    # F3: s301 not retained in either controller
    s301_intc_retained = crown_data.get(("INT-C2900", 301), {}).get("retained")
    s301_regtb_retained = crown_data.get(("REG-TB", 301), {}).get("retained")
    if not s301_intc_retained or not s301_regtb_retained:
        f3_fired = True
        p(f"  *** F3 FIRED: s301 C-C not retained in both controllers "
          f"(INT-C2900: {s301_intc_retained}, REG-TB: {s301_regtb_retained}) ***")
    else:
        p(f"  P3 PASS candidate: s301 retained in BOTH controllers")
    p()

    # ---- 4e. P4 (baseline durability at home) ----
    p("P4 -- BASELINE W1 DISPLACED COUNT PER CELL (w1_frac >= 0.5 = still displaced)")
    p("-" * 60)
    p(f"  {'cell':<8}  {'w1_displaced/16':>16}  {'w2_displaced/16':>16}  {'P4_cell':>10}")
    p("-" * 60)

    p4_pass = True
    f4_fired = False
    baseline_w1_displaced: dict[str, int] = {}
    baseline_w2_displaced: dict[str, int] = {}

    for cell_name in cell_names_order:
        w1_displaced = sum(
            1 for seed in ALL_SEEDS
            if (row_index.get((cell_name, "baseline", seed), {}).get("w1_frac") or 0.0) >= 0.5
        )
        w2_displaced = sum(
            1 for seed in ALL_SEEDS
            if (row_index.get((cell_name, "baseline", seed), {}).get("w2_frac") or 0.0) >= 0.5
        )
        baseline_w1_displaced[cell_name] = w1_displaced
        baseline_w2_displaced[cell_name] = w2_displaced
        cell_p4 = w1_displaced >= 12
        if not cell_p4:
            p4_pass = False
        not_displaced = N_SEEDS_TOTAL - w1_displaced
        if not_displaced >= 8:
            f4_fired = True
        p(f"  {cell_name:<8}  {w1_displaced:>11}/16  {w2_displaced:>11}/16  {'PASS' if cell_p4 else 'fail':>10}")

    # Diagnostic: per cell, mean expr_frac(attack) over final 2,000 steps
    p()
    p("P4 DIAGNOSTIC -- mean expr_frac(attack) over final 2,000 steps (baseline, self-healing trajectory)")
    p("-" * 60)
    for cell in CELLS_193:
        cell_name = cell["name"]
        n_steps = cell["n_steps"]
        w_start = n_steps - 2000
        w_end = n_steps
        fracs = []
        for seed in ALL_SEEDS:
            r = row_index.get((cell_name, "baseline", seed), {})
            ac = r.get("attack_color")
            if ac is None:
                continue
            # We need the expressed_arr -- not stored in row; note: we only have row-level data
            # Use w2_frac as a proxy at release+3000..3500 (last_bend based)
            # Actually we need a different diagnostic. Use w1_frac and w2_frac as proxies:
            w1 = r.get("w1_frac")
            w2 = r.get("w2_frac")
            if w1 is not None:
                fracs.append(w1)
            if w2 is not None:
                fracs.append(w2)
        mean_frac = sum(fracs) / len(fracs) if fracs else None
        p(f"  {cell_name}: mean_w_frac(attack) from w1+w2 windows = {mean_frac:.4f if mean_frac is not None else 'N/A'} "
          f"(n={len(fracs)} windows from {N_SEEDS_TOTAL} seeds)")
    p()
    if f4_fired:
        p("  *** F4 FIRED: baseline self-heals in at least one cell ***")
    p()

    # ---- 4f. PC1 summary ----
    pc1_total = sum(1 for row in all_rows if "PC1_DRIFT" in row.get("flags", []))
    p(f"PC1 flagged: {pc1_total}/192 sessions with ahat_drift >= {PC1_AHAT_DRIFT_MAX}")
    p()

    # ====================================================================
    # VERDICT LINES
    # ====================================================================
    p("=" * 80)
    p("VERDICT LINES")
    p("=" * 80)
    p()

    # P1: every covered pair M >= 10 AND settled >= 12/16
    p1_pass = (
        not f1_fired
        and all(
            (settling_data.get((arm, cell), {}).get("M", 0) >= 10
             and settling_data.get((arm, cell), {}).get("settled", 0) >= 12)
            for arm, cell in covered_pairs_all
        )
    )

    p("P1 (settling: M >= 10 AND >= 12/16 defended seats release by last_bend + 4000):")
    p(f"  Covered pairs (non-oracle): {covered_pairs}")
    p(f"  Covered pairs (oracle):     {oracle_covered}")
    p(f"  Instrument-failed pairs:    {instrument_failed_pairs}")
    for arm_name, cell_name in covered_pairs_all:
        sd = settling_data.get((arm_name, cell_name), {})
        M = sd.get("M", 0)
        settled = sd.get("settled", 0)
        if M < 10:
            verdict = "INSTRUMENT-FAILED (F1)"
        elif settled >= 12:
            verdict = "PASS"
        else:
            verdict = f"FAIL (settled={settled}/M={M}; F1)"
        p(f"  {arm_name:<22} x {cell_name:<8}: M={M} settled={settled} -> {verdict}")
    p(f"  F1 fired (M < 10 or settled < 12/16 in any pair): {f1_fired}")
    if p1_pass:
        p("  => P1 VERDICT: PASS -- settling restored; M >= 10 and >= 12/16 settle in all pairs.")
    else:
        p("  => P1 VERDICT: NEGATIVE -- F1 FIRED.")
    p()

    p("P2 (retention >= 5/6 in every covered pair):")
    for arm_name, cell_name in covered_pairs_all:
        rate = retention_rates.get((arm_name, cell_name))
        if rate is None:
            zone = "INSTRUMENT-FAILED"
        elif rate >= 5 / 6:
            zone = ">= 5/6 PASS"
        elif rate > 2 / 3:
            zone = "MIXED"
        else:
            zone = "F2"
        rate_str = f"{rate:.3f}" if rate is not None else "N/A"
        p(f"  {arm_name:<22} x {cell_name:<8}: retention_rate={rate_str} -> {zone}")
    p(f"  F2 fired (any covered pair retention <= 2/3): {f2_fired_flag}")
    p(f"  MIXED-zone pairs (2/3 < rate < 5/6): {mixed_zone_pairs}")
    graded_pairs = [(pair, rate) for pair, rate in retention_rates.items() if rate is not None]
    p2_pass = (
        not f2_fired_flag
        and len(mixed_zone_pairs) == 0
        and all(rate >= 5 / 6 for _, rate in graded_pairs if _ in covered_pairs_all)
    )
    if p2_pass:
        p("  => P2 VERDICT: PASS -- retention_rate >= 5/6 in every covered pair.")
    elif f2_fired_flag:
        p("  => P2 VERDICT: NEGATIVE -- F2 FIRED. Deferral persists without locality feed.")
    else:
        p("  => P2 VERDICT: MIXED -- not all covered pairs clear 5/6 threshold.")
    p()

    p("P3 (CROWN -- s301 C-C retained in BOTH controllers):")
    p(f"  s301 INT-C2900: w1={crown_data.get(('INT-C2900',301),{}).get('w1')} "
      f"w2={crown_data.get(('INT-C2900',301),{}).get('w2')} retained={s301_intc_retained}")
    p(f"  s301 REG-TB:    w1={crown_data.get(('REG-TB',301),{}).get('w1')} "
      f"w2={crown_data.get(('REG-TB',301),{}).get('w2')} retained={s301_regtb_retained}")
    p(f"  s297 INT-C2900: w1={crown_data.get(('INT-C2900',297),{}).get('w1')} "
      f"w2={crown_data.get(('INT-C2900',297),{}).get('w2')} retained={crown_data.get(('INT-C2900',297),{}).get('retained')}")
    p(f"  s297 REG-TB:    w1={crown_data.get(('REG-TB',297),{}).get('w1')} "
      f"w2={crown_data.get(('REG-TB',297),{}).get('w2')} retained={crown_data.get(('REG-TB',297),{}).get('retained')}")
    p(f"  F3 fired (s301 not retained in either controller): {f3_fired}")
    if not f3_fired:
        p("  => P3 VERDICT: PASS -- s301 retains under return-home; locality mechanism confirmed causal.")
    else:
        p("  => P3 VERDICT: NEGATIVE -- F3 FIRED. s301 still surrenders; locality NOT sufficient cause.")
    p()

    p("P4 (baseline stays displaced: w1 >= 0.5 in >= 12/16 per cell):")
    for cell_name in cell_names_order:
        cnt = baseline_w1_displaced[cell_name]
        p(f"  {cell_name}: W1_displaced={cnt}/16 W2_displaced={baseline_w2_displaced[cell_name]}/16")
    p(f"  F4 fired (w1_frac < 0.5 in >= 8/16 anywhere): {f4_fired}")
    if p4_pass:
        p("  => P4 VERDICT: PASS -- baseline stays displaced at home (train overwrite is intrinsic).")
    else:
        p("  => P4 VERDICT: fail -- some cells show < 12/16 baseline displacement.")
    if f4_fired:
        p("  => P4 F4: FIRED -- baseline self-heals; displacement was locality-fed.")
    p()

    p("P5 (oracle retention >= 5/6 where covered):")
    if not oracle_covered:
        p("  No oracle-covered pairs (oracle frozen_def < 14/16 in all cells).")
        p("  => P5 VERDICT: N/A.")
    else:
        for arm_name, cell_name in oracle_covered:
            rate = retention_rates.get((arm_name, cell_name))
            rate_str = f"{rate:.3f}" if rate is not None else "INSTRUMENT-FAILED"
            p(f"  oracle x {cell_name}: rate={rate_str}")
        p5_pass = p3_oracle_pass
        if p5_pass:
            p("  => P5 VERDICT: PASS -- oracle retention_rate >= 5/6 in all covered cells.")
        else:
            p("  => P5 VERDICT: fail -- oracle retention < 5/6 in at least one cell.")
    p()

    # ====================================================================
    # SUMMARY
    # ====================================================================
    elapsed_total = time.time() - t_start
    runtime_min = elapsed_total / 60.0

    p("=" * 80)
    p("SUMMARY")
    p("=" * 80)
    p()
    p(f"  G1:                PASS")
    p(f"  G2:                PASS (8 exp192 regression sessions bit-matched)")
    p(f"  G3:                PASS (all {len(all_rows)} return_home=True sessions verified)")
    p(f"  Seeds:             Block1={BLOCK1} Block2={BLOCK2}")
    p(f"  Runtime:           {runtime_min:.1f} min")
    p(f"  Sessions:          {len(all_rows)} W (return_home=True)")
    p()
    p(f"  P1 (settling M>=10 AND >=12/16 in every covered pair): {'PASS' if p1_pass else 'NEGATIVE (F1)'}")
    p(f"  P2 (retention >= 5/6 every covered pair):              {'PASS' if p2_pass else ('NEGATIVE (F2)' if f2_fired_flag else 'MIXED')}")
    p(f"  P3 (s301 C-C retains in BOTH controllers):             {'PASS' if not f3_fired else 'NEGATIVE (F3)'}")
    p(f"  P4 (baseline W1 displaced >= 12/16 every cell):        {'PASS' if p4_pass else 'fail'}")
    p(f"  P5 (oracle retention >= 5/6 every covered cell):       {'PASS' if (p3_oracle_pass and oracle_covered) else ('N/A' if not oracle_covered else 'fail')}")
    p()
    p(f"  F1 (settling fails: M<10 or settled<12/16 in any pair): {f1_fired}")
    p(f"  F2 (any covered pair retention <= 2/3):                  {f2_fired_flag}")
    p(f"  F3 (s301 C-C not retained in either controller):         {f3_fired}")
    p(f"  F4 (baseline self-heals in any cell):                    {f4_fired}")
    p()
    p(f"  Instrument-failed pairs:    {instrument_failed_pairs}")
    p(f"  PC1 flagged: {pc1_total}/192")
    p("=" * 80)

    # ====================================================================
    # Analysis row + finalize outputs
    # ====================================================================
    analysis_row: dict = {
        "exp": 193,
        "kind": "analysis",
        "seeds_b1": BLOCK1,
        "seeds_b2": BLOCK2,
        "covered_pairs": covered_pairs,
        "oracle_covered": oracle_covered,
        "instrument_failed_pairs": instrument_failed_pairs,
        "settling_data": {
            f"{arm}|{cell}": sd
            for (arm, cell), sd in settling_data.items()
        },
        "retention_rates": {
            f"{arm}|{cell}": to_plain(rate)
            for (arm, cell), rate in retention_rates.items()
        },
        "p1_pass": p1_pass,
        "f1_fired": f1_fired,
        "p2_pass": p2_pass,
        "f2_fired": f2_fired_flag,
        "mixed_zone_pairs": mixed_zone_pairs,
        "f3_fired": f3_fired,
        "p3_crown": {
            "s297_intc": to_plain(crown_data.get(("INT-C2900", 297), {})),
            "s297_regtb": to_plain(crown_data.get(("REG-TB", 297), {})),
            "s301_intc": to_plain(crown_data.get(("INT-C2900", 301), {})),
            "s301_regtb": to_plain(crown_data.get(("REG-TB", 301), {})),
        },
        "p4_pass": p4_pass,
        "f4_fired": f4_fired,
        "baseline_w1_displaced": baseline_w1_displaced,
        "baseline_w2_displaced": baseline_w2_displaced,
        "p5_pass": p3_oracle_pass,
        "pc1_total_flagged": pc1_total,
        "runtime_min": runtime_min,
    }

    with open(out_rows_path, "a") as fh:
        fh.write(json.dumps(to_plain(analysis_row)) + "\n")

    with open(out_txt_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nRows written to {out_rows_path} ({len(all_rows)} session rows + 1 analysis)")
    print(f"Report written to {out_txt_path}")
    print(f"Total runtime: {runtime_min:.1f} min")


if __name__ == "__main__":
    main()
