"""Exp 190 — N4 flicker-hysteresis runner module (runner + gates ONLY).

This module generates a verbatim copy of the exp188 runner (`run_fork_schedule_190`,
built by source-patching exp188's `run_fork_schedule_188` with `_must_replace` —
every patch hard-fails if its target does not match; see L16) with two additions:

- `hysteresis_snaps` (default 0 = exp188-verbatim): a de-assert run of <= h
  consecutive FINE_EVAL snaps does NOT end/reset the continuity clock; tolerated
  de-assert snaps add nothing to any accumulator (active-only counting unchanged);
  calm/transient release untouched.
- `record_pressure_trace` (default False): per-snap [step, frozen, pressure_active]
  trace, pure observation (bit-match-gated against committed exp189 rows).

Gates: `run_equivalence_gate_190` (the standing exp183 gate through this code
path) and `run_exp189_regression_gate_190` (h=0 bit-match against committed
exp189 rows), both with emitted got-vs-committed evidence tables (L15).

THE PLANNED FRESH-SEED GRID WAS NEVER RUN — deliberately. The design-time
diagnostic (experiments/exp190_flicker_diagnostic.py, which holds the Exp 190
predeclaration: prediction, falsifier-bound admissibility + buildability rules)
found NO admissible hysteresis constant: every h that rescues revision bridges
the attack-train gaps (the timescale-overlap law). With no parameter to test,
a fresh-seed run would have been theater. See the Exp 190 entry in EXPERIMENTS.md.
"""
from __future__ import annotations

import copy
import inspect
import json
import math
import statistics
import sys
import textwrap
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Import exp188 via importlib, matching the exp189 pattern.
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_spec188 = _ilu.spec_from_file_location(
    "exp188",
    str(REPO_ROOT / "experiments" / "exp188_n4_regulated_controller.py"),
)
_mod188 = _ilu.module_from_spec(_spec188)
_spec188.loader.exec_module(_mod188)  # type: ignore[union-attr]

run_fork_schedule_188 = _mod188.run_fork_schedule_188
run_equivalence_gate_188 = _mod188.run_equivalence_gate_188
compute_defense = _mod188.compute_defense
compute_final_gap = _mod188.compute_final_gap
phase_r_latency = _mod188.phase_r_latency
to_plain = _mod188.to_plain
pi_of = _mod188.pi_of

LAMBDA = _mod188.LAMBDA
INIT_MASS = _mod188.INIT_MASS
PHASE_R_START = _mod188.PHASE_R_START
BURST_SEED_OFFSET_R = _mod188.BURST_SEED_OFFSET_R
BURST_SEED_OFFSET_W_EXP183 = _mod188.BURST_SEED_OFFSET_W_EXP183
EVAL = _mod188.EVAL
FINE_EVAL = _mod188.FINE_EVAL
N_STEPS_PHASE_R = _mod188.N_STEPS_PHASE_R
CHUNK_SIZE = _mod188.CHUNK_SIZE
P6_HOLD = _mod188.P6_HOLD
DEFENSE_FRAC_THRESH = _mod188.DEFENSE_FRAC_THRESH
DEFENSE_WINDOW_OFFSET_START = _mod188.DEFENSE_WINDOW_OFFSET_START
DEFENSE_WINDOW_OFFSET_END = _mod188.DEFENSE_WINDOW_OFFSET_END
PC1_AHAT_DRIFT_MAX = _mod188.PC1_AHAT_DRIFT_MAX

DEFAULT_THETA = _mod188.DEFAULT_THETA
DEFAULT_RELEASE_CALM_SNAPS = _mod188.DEFAULT_RELEASE_CALM_SNAPS
CALM2600_SNAPS = _mod188.CALM2600_SNAPS
THETA3_THETA = _mod188.THETA3_THETA
E_STAR = _mod188.E_STAR
FLOAT_ATOL = _mod188.FLOAT_ATOL

EXP183_BURST_WINDOWS = _mod188.EXP183_BURST_WINDOWS
EXP183_N_STEPS = _mod188.EXP183_N_STEPS
EXP183_GATE_SEEDS = _mod188.EXP183_GATE_SEEDS
EXP183_GATE_ARMS = _mod188.EXP183_GATE_ARMS

PRESSURE_WINDOW = _mod188.PRESSURE_WINDOW
PRESSURE_FRAC = _mod188.PRESSURE_FRAC
REFRACTORY_CHECKS = _mod188.__dict__.get("REFRACTORY_CHECKS", 8)

from active_loop.creature import Creature


def _must_replace(src: str, old: str, new: str, count: int = 1) -> str:
    """str.replace that REFUSES to no-op silently (the L13/L15 instrument rule:
    a source patch is an instrument; a target that does not match is a hard
    error, never a silent pass-through)."""
    n_found = src.count(old)
    if n_found < count:
        raise RuntimeError(
            f"_must_replace: target found {n_found}x, need {count}. "
            f"Target starts: {old[:120]!r}"
        )
    out = src.replace(old, new, count)
    if out == src:
        raise RuntimeError(f"_must_replace: replace was a no-op. Target: {old[:120]!r}")
    return out


def _build_run_fork_schedule_190():
    """Generate a verbatim copy of exp188's runner with the requested deltas."""
    src = "from __future__ import annotations\n" + textwrap.dedent(
        inspect.getsource(_mod188.run_fork_schedule_188)
    )

    src = _must_replace(src,
        "def run_fork_schedule_188(",
        "def run_fork_schedule_190(",
        1,
    )
    src = _must_replace(src,
        "    release_calm_snaps: int = DEFAULT_RELEASE_CALM_SNAPS,\n) -> dict:",
        "    release_calm_snaps: int = DEFAULT_RELEASE_CALM_SNAPS,\n"
        "    hysteresis_snaps: int = 0,\n"
        "    record_pressure_trace: bool = False,\n"
        ") -> dict:",
        1,
    )
    src = _must_replace(src,
        "    reg_max_current_seen: float = 0.0   # high-water mark of current_stretch (diagnostic)\n",
        "    reg_max_current_seen: float = 0.0   # high-water mark of current_stretch (diagnostic)\n"
        "    deassert_run: int = 0               # consecutive de-assert snaps while frozen\n"
        "    pressure_trace: list[list[int]] = [] # [global_step, frozen_flag, pressure_active_flag]\n",
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
        "    }\n",
        1,
    )

    ns = {
        "__name__": "_exp190_runner_copy",
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
        "pi_of": pi_of,
        "LAMBDA": LAMBDA,
        "INIT_MASS": INIT_MASS,
        "PHASE_R_START": PHASE_R_START,
        "BURST_SEED_OFFSET_R": BURST_SEED_OFFSET_R,
        "BURST_SEED_OFFSET_W_EXP183": BURST_SEED_OFFSET_W_EXP183,
        "EVAL": EVAL,
        "FINE_EVAL": FINE_EVAL,
        "N_STEPS_PHASE_R": N_STEPS_PHASE_R,
        "CHUNK_SIZE": CHUNK_SIZE,
        "P6_HOLD": P6_HOLD,
        "DEFENSE_FRAC_THRESH": DEFENSE_FRAC_THRESH,
        "DEFENSE_WINDOW_OFFSET_START": DEFENSE_WINDOW_OFFSET_START,
        "DEFENSE_WINDOW_OFFSET_END": DEFENSE_WINDOW_OFFSET_END,
        "PC1_AHAT_DRIFT_MAX": PC1_AHAT_DRIFT_MAX,
        "DEFAULT_THETA": DEFAULT_THETA,
        "DEFAULT_RELEASE_CALM_SNAPS": DEFAULT_RELEASE_CALM_SNAPS,
        "CALM2600_SNAPS": CALM2600_SNAPS,
        "THETA3_THETA": THETA3_THETA,
        "E_STAR": E_STAR,
        "FLOAT_ATOL": FLOAT_ATOL,
        "EXP183_BURST_WINDOWS": EXP183_BURST_WINDOWS,
        "EXP183_N_STEPS": EXP183_N_STEPS,
        "EXP183_GATE_SEEDS": EXP183_GATE_SEEDS,
        "EXP183_GATE_ARMS": EXP183_GATE_ARMS,
        "PRESSURE_WINDOW": PRESSURE_WINDOW,
        "PRESSURE_FRAC": PRESSURE_FRAC,
        "REFRACTORY_CHECKS": REFRACTORY_CHECKS,
    }
    exec(src, ns)
    return ns["run_fork_schedule_190"]


run_fork_schedule_190 = _build_run_fork_schedule_190()


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _value_match(got, exp) -> bool:
    if got is None or exp is None:
        return got is exp
    if isinstance(got, (int, float, np.integer, np.floating)) and isinstance(
        exp, (int, float, np.integer, np.floating)
    ):
        return abs(float(got) - float(exp)) <= FLOAT_ATOL
    if isinstance(got, (list, tuple)) and isinstance(exp, (list, tuple)):
        if len(got) != len(exp):
            return False
        return all(_value_match(g, e) for g, e in zip(got, exp))
    return got == exp


def _fmt(v) -> str:
    return json.dumps(to_plain(v), sort_keys=True)


def _compare_event_lists(got_events: list[dict], committed_events: list[dict]) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok_all = True
    if len(got_events) != len(committed_events):
        lines.append(
            f"  n_events got={len(got_events)} committed={len(committed_events)} MISMATCH"
        )
        return False, lines
    for idx, (ge, ce) in enumerate(zip(got_events, committed_events)):
        for field in ("label", "entry_step", "frozen_steps"):
            got = ge.get(field)
            exp = ce.get(field)
            ok = got == exp
            lines.append(
                f"  ev{idx:<2d} {field:<12} got={got!s:<24} committed={exp!s:<24} "
                f"{'OK' if ok else 'MISMATCH'}"
            )
            if not ok:
                ok_all = False
    return ok_all, lines


def _compare_fields(session_label: str, got_row: dict, committed_row: dict, fields: list[str]) -> tuple[bool, str]:
    lines = [f"=== {session_label} ==="]
    ok_all = True
    for field in fields:
        got = got_row.get(field)
        exp = committed_row.get(field)
        ok = _value_match(got, exp)
        lines.append(
            f"  {field:<18} got={_fmt(got):<44} committed={_fmt(exp):<44} "
            f"{'OK' if ok else 'MISMATCH'}"
        )
        if not ok:
            ok_all = False
    return ok_all, "\n".join(lines)


def run_equivalence_gate_190(
    mirro_root, base_cmap, n_colors, committed_183_path: Path
) -> tuple[bool, str]:
    """Replay the exp183 baseline/H1200 rows through the exp190 runner."""
    committed_rows = _load_rows(committed_183_path)
    committed_w = {
        (row["arm"], row["fork_seed"], row["burst_idx"]): row
        for row in committed_rows
        if row.get("phase") == "W"
    }

    detail_lines: list[str] = ["EQUIVALENCE GATE: exp183 replay"]
    all_pass = True

    arm_lookup = {
        "baseline": ("baseline", "baseline"),
        "H1200": ("H1200", ("freeze_time", 1200)),
    }

    for arm_name in EXP183_GATE_ARMS:
        _, arm_mode = arm_lookup[arm_name]
        for seed in EXP183_GATE_SEEDS:
            root = copy.deepcopy(mirro_root)
            root._state_dir = None
            rr = run_fork_schedule_190(
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
                        f"  b{bi} {field:<12} got={_fmt(got[field]):<44} committed={_fmt(committed[field]):<44} "
                        f"{'OK' if ok else 'MISMATCH'}"
                    )
                    if not ok:
                        session_pass = False
                        all_pass = False

                ok_events, ev_lines = _compare_event_lists(rr["events"], committed.get("events_summary", []))
                detail_lines.extend(ev_lines)
                if not ok_events:
                    session_pass = False
                    all_pass = False

            detail_lines.append(f"  gate arm={arm_name} seed={seed}: {'PASS' if session_pass else 'FAIL'}")

    return all_pass, "\n".join(detail_lines)


def run_exp189_regression_gate_190(
    mirro_root, base_cmap, n_colors, committed_189_rows_path: Path
) -> tuple[bool, str]:
    """Replay the committed exp189 regression sessions through hysteresis=0."""
    rows = _load_rows(committed_189_rows_path)
    w_index = {
        (row["cell"]["name"], row["arm"], row["seed"]): row
        for row in rows
        if row.get("kind") == "W"
    }
    r_index = {
        (row["arm"], row["seed"]): row
        for row in rows
        if row.get("kind") == "R"
    }

    detail_lines: list[str] = ["EXP189 REGRESSION GATE: hysteresis=0"]
    all_pass = True

    # W: E1 / REG-TB / seed 280
    root = copy.deepcopy(mirro_root)
    root._state_dir = None
    rr_w = run_fork_schedule_190(
        mirro=root,
        fork_seed=280,
        base_cmap=base_cmap,
        n_colors=n_colors,
        arm_name="REG-TB",
        arm_mode=("reg_tb", 1.5, 2800, 104),
        phase="W",
        burst_windows=[
            (6000, 7200),
            (7800, 10200),
            (10800, 14000),
        ],
        color_mode="exogenous_fixed",
        reloc_rng_seed=280_000,
        n_steps=16500,
        theta=DEFAULT_THETA,
        release_calm_snaps=104,
        hysteresis_snaps=0,
    )
    got_w_defense, got_w_frac, got_w_attack = compute_defense(
        rr_w["expressed_arr"],
        [(6000, 7200), (7800, 10200), (10800, 14000)],
        rr_w["burst_onset_color"],
        16500,
    )
    committed_w = w_index[("E1", "REG-TB", 280)]
    w_fields = [
        ("defense", got_w_defense, committed_w["defense"]),
        ("final_expr_frac", got_w_frac, committed_w["final_expr_frac"]),
        ("final_gap", compute_final_gap(rr_w["v_traj"], rr_w["expressed_arr"], 16500), committed_w["final_gap"]),
        ("stretch_log", rr_w.get("stretch_log", []), committed_w["stretch_log"]),
        ("S_max_final", rr_w.get("S_max_final", 0.0), committed_w["S_max_final"]),
        ("max_current_stretch", rr_w.get("max_current_stretch", 0.0), committed_w["max_current_stretch"]),
        ("n_events", len(rr_w["events"]), committed_w["n_events"]),
    ]
    for field, got, exp in w_fields:
        ok = _value_match(got, exp)
        detail_lines.append(
            f"W E1 REG-TB seed=280 {field:<18} got={_fmt(got):<44} committed={_fmt(exp):<44} "
            f"{'OK' if ok else 'MISMATCH'}"
        )
        if not ok:
            all_pass = False
    ok_events, ev_lines = _compare_event_lists(rr_w["events"], committed_w.get("events_summary", []))
    detail_lines.extend([f"W E1 REG-TB seed=280 {line}" for line in ev_lines])
    if not ok_events:
        all_pass = False

    # R: REG-TB / seed 281
    root = copy.deepcopy(mirro_root)
    root._state_dir = None
    rr_r1 = run_fork_schedule_190(
        mirro=root,
        fork_seed=281,
        base_cmap=base_cmap,
        n_colors=n_colors,
        arm_name="REG-TB",
        arm_mode=("reg_tb", 1.5, 2800, 104),
        phase="R",
        burst_windows=[],
        color_mode="endogenous",
        reloc_rng_seed=BURST_SEED_OFFSET_R + 281,
        n_steps=N_STEPS_PHASE_R,
        theta=DEFAULT_THETA,
        release_calm_snaps=104,
        hysteresis_snaps=0,
    )
    lat1 = phase_r_latency(rr_r1["expressed_arr"], rr_r1["regime_color"], N_STEPS_PHASE_R)
    committed_r1 = r_index[("REG-TB", 281)]
    r1_fields = [
        ("latency", lat1, committed_r1["latency"]),
        ("n_resets", rr_r1.get("n_resets", 0), committed_r1["n_resets"]),
        ("n_completed_stretches", rr_r1.get("n_completed_stretches", 0), committed_r1["n_completed_stretches"]),
        ("stretch_log", rr_r1.get("stretch_log", []), committed_r1["stretch_log"]),
        ("n_events", len(rr_r1["events"]), committed_r1["n_events"]),
    ]
    for field, got, exp in r1_fields:
        ok = _value_match(got, exp)
        detail_lines.append(
            f"R REG-TB seed=281 {field:<18} got={_fmt(got):<44} committed={_fmt(exp):<44} "
            f"{'OK' if ok else 'MISMATCH'}"
        )
        if not ok:
            all_pass = False
    detail_lines.append(
        "R REG-TB seed=281 committed row has no events_summary; event list printed from the fresh run only."
    )
    detail_lines.append(f"R REG-TB seed=281 fresh_events={_fmt(rr_r1['events'])}")

    # R: INT-C2900 / seed 286
    root = copy.deepcopy(mirro_root)
    root._state_dir = None
    rr_r2 = run_fork_schedule_190(
        mirro=root,
        fork_seed=286,
        base_cmap=base_cmap,
        n_colors=n_colors,
        arm_name="INT-C2900",
        arm_mode=("int_c", 2900, 104),
        phase="R",
        burst_windows=[],
        color_mode="endogenous",
        reloc_rng_seed=BURST_SEED_OFFSET_R + 286,
        n_steps=N_STEPS_PHASE_R,
        theta=DEFAULT_THETA,
        release_calm_snaps=104,
        hysteresis_snaps=0,
    )
    lat2 = phase_r_latency(rr_r2["expressed_arr"], rr_r2["regime_color"], N_STEPS_PHASE_R)
    committed_r2 = r_index[("INT-C2900", 286)]
    r2_fields = [
        ("latency", lat2, committed_r2["latency"]),
        ("n_resets", rr_r2.get("n_resets", 0), committed_r2["n_resets"]),
        ("n_completed_stretches", rr_r2.get("n_completed_stretches", 0), committed_r2["n_completed_stretches"]),
        ("stretch_log", rr_r2.get("stretch_log", []), committed_r2["stretch_log"]),
        ("n_events", len(rr_r2["events"]), committed_r2["n_events"]),
    ]
    for field, got, exp in r2_fields:
        ok = _value_match(got, exp)
        detail_lines.append(
            f"R INT-C2900 seed=286 {field:<18} got={_fmt(got):<44} committed={_fmt(exp):<44} "
            f"{'OK' if ok else 'MISMATCH'}"
        )
        if not ok:
            all_pass = False
    detail_lines.append(
        "R INT-C2900 seed=286 committed row has no events_summary; event list printed from the fresh run only."
    )
    detail_lines.append(f"R INT-C2900 seed=286 fresh_events={_fmt(rr_r2['events'])}")

    return all_pass, "\n".join(detail_lines)


def main() -> None:
    print("exp190 runner module - run gates via the diagnostic")


if __name__ == "__main__":
    main()
