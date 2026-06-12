"""Exp 190 — the flicker-robust concession attempt (Exp 189 consult option (a)):
the DESIGN-TIME DIAGNOSTIC that was to fix the hysteresis constant h before a
fresh-seed pre-registered run.

Authorized by the human's explicit word "(a)" on the Exp 189 consult (2026-06-12,
in-session; recorded in loop/IDEAS.md): ONE pre-registered attempt at a
flicker-robust concession form; a falsifier fire closes the question as
refuted-as-buildable.

FORM SELECTION (fixed at design time, before this instrument ran):
- total-active-time-since-freeze-entry: REFUTED BY ARITHMETIC, no run needed —
  defending E1/E3 requires the concession budget to cover the train's total
  active pressure (~7,175 steps at E1: stretches ~1325+2525+3325) while revision
  requires concession <= ~2,900 (T0-class budget inside the +3000 bar);
  the interval is empty by > 4,000 steps. Dismissed; not tested.
- DE-ASSERT HYSTERESIS (the chosen form, this instrument): a de-assert run of
  <= h consecutive FINE_EVAL snaps does not end/reset the continuity clock;
  active-only counting unchanged (tolerated de-assert snaps add NOTHING — the
  quiet-accumulation guard stays binding); calm/transient release untouched.

PREDECLARED INSTRUMENT + RULES (fixed in the build spec BEFORE any trace data):
- Re-run committed exp189 sessions (REG-TB R x seeds 280-287; W probes E1
  seeds 280/286, E3 seeds 280/286) with per-snap pressure tracing at
  hysteresis_snaps=0, gated by (i) the exp183 equivalence gate and (ii) a NEW
  exp189 regression gate, both with emitted got-vs-committed evidence (L15),
  plus a per-session bit-match of stretch_log/latency/defense against the
  committed rows (L14 spine safety: detached deepcopy). ANY mismatch aborts.
- Measure de-assert run lengths (snaps) while frozen: R-side runs = the
  changed-world onset "flickers"; W-side runs = inter-burst gap runs vs
  in-burst runs.
- h-ADMISSIBILITY RULE: h is admissible iff max R-side run <= h/2 AND min
  W-side gap run >= 2h (revision needs flickers bridged, defense needs gap
  runs NOT bridged; 2x margins both ways). h=8 evaluated first; else h in
  [2,20] searched; none admissible => "the hysteresis form is unbuildable at
  this body".
- STEP 4 DECISION RULE (the buildability bar; fixed before the sweep ran):
  the hysteresis form is BUILDABLE iff SOME h gives
    (i)  revision: latency - same-seed committed baseline <= +3000 on
         >= 6/8 committed seeds (bars verbatim Exp 186/188/189), AND
    (ii) defense intact in BOTH E1 and E3 (no mid-train concession; defense
         bool True at the probed seeds) — the Exp 189 separation bar needs
         >= 2 cells and E1/E3 are the only candidates the constants concede.
  SAME-SEED NOTE: the sweep runs on the committed seeds 280-287 — the
  controller's BEST case (h chosen with the answer key in hand). A negative
  here is a fortiori a negative on fresh seeds; a positive would have
  LICENSED (not replaced) the fresh-seed pre-registered run.

INSTRUMENT HISTORY (honest, L13 class): the first build's trace recorder
silently no-opped (source patches with wrong indentation matched nothing,
str.replace passed through), yielding empty traces and a vacuous "NO
ADMISSIBLE h". Caught in main-model review: a de-assert count of 0 contradicts
the committed stretch_log (4 completed stretches require >= 3 frozen
de-asserts). Repaired via _must_replace (a source patch that cannot no-op) in
the runner builder; every patch now hard-fails on mismatch; all gates
re-passed. This output is from the repaired instrument only.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import importlib.util as _ilu

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

_spec190 = _ilu.spec_from_file_location(
    "exp190",
    str(REPO_ROOT / "experiments" / "exp190_n4_flicker_hysteresis.py"),
)
_mod190 = _ilu.module_from_spec(_spec190)
_spec190.loader.exec_module(_mod190)  # type: ignore[union-attr]

run_equivalence_gate_190 = _mod190.run_equivalence_gate_190
run_exp189_regression_gate_190 = _mod190.run_exp189_regression_gate_190
run_fork_schedule_190 = _mod190.run_fork_schedule_190
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
INT_CALM_SNAPS = 104
N_STEPS_PHASE_R = _mod190.N_STEPS_PHASE_R
FLOAT_ATOL = _mod190.FLOAT_ATOL
INIT_MASS = _mod190.INIT_MASS
PRESSURE_WINDOW = _mod190.PRESSURE_WINDOW
PRESSURE_FRAC = _mod190.PRESSURE_FRAC
BURST_SEED_OFFSET_W_EXP183 = _mod190.BURST_SEED_OFFSET_W_EXP183

from active_loop.creature import Creature

SEEDS_189 = [280, 281, 282, 283, 284, 285, 286, 287]

EXP189_E1_WINDOWS = [(6000, 7200), (7800, 10200), (10800, 14000)]
EXP189_E3_WINDOWS = [(6000, 7200), (8400, 10800), (12000, 15200)]

TRACE_CASES_R = [seed for seed in SEEDS_189]
TRACE_CASES_W = [(280, "E1"), (286, "E1"), (280, "E3"), (286, "E3")]


def _load_committed_rows(path: Path) -> list[dict]:
    return _load_rows(path)


def _fmt(v) -> str:
    return json.dumps(to_plain(v), sort_keys=True)


def _lines_append(lines: list[str], msg: str) -> None:
    lines.append(msg)
    print(msg)


def _pressure_runs(trace: list[list[int]], burst_windows: list[tuple[int, int]] | None = None) -> list[dict]:
    """Summarize maximal de-assert runs from frozen pressure snapshots only."""
    episodes: list[tuple[list[int], list[int]]] = []
    cur_steps: list[int] = []
    cur_vals: list[int] = []
    prev_step: int | None = None

    for step, frozen, active in trace:
        if int(frozen) != 1:
            continue
        step = int(step)
        active = int(active)
        if prev_step is not None and step - prev_step != FINE_EVAL:
            if cur_steps:
                episodes.append((cur_steps, cur_vals))
            cur_steps = [step]
            cur_vals = [active]
        else:
            cur_steps.append(step)
            cur_vals.append(active)
        prev_step = step

    if cur_steps:
        episodes.append((cur_steps, cur_vals))

    gap_windows: list[tuple[int, int]] = []
    if burst_windows:
        for idx in range(len(burst_windows) - 1):
            gap_windows.append((burst_windows[idx][1], burst_windows[idx + 1][0]))

    runs: list[dict] = []
    for steps, vals in episodes:
        start_idx: int | None = None
        for idx, val in enumerate(vals):
            if val == 0 and start_idx is None:
                start_idx = idx
            elif val == 1 and start_idx is not None:
                run_steps = steps[start_idx:idx]
                runs.append(_classify_run(run_steps, gap_windows, burst_windows))
                start_idx = None
        if start_idx is not None:
            run_steps = steps[start_idx:]
            runs.append(_classify_run(run_steps, gap_windows, burst_windows))
    return runs


def _classify_run(
    run_steps: list[int],
    gap_windows: list[tuple[int, int]],
    burst_windows: list[tuple[int, int]] | None,
) -> dict:
    if not run_steps:
        return {"length": 0, "kind": "EMPTY", "steps": []}

    span_start = run_steps[0]
    span_end = run_steps[-1]
    if burst_windows is None:
        kind = "R"
    else:
        kind = "IN-BURST"
        for gstart, gend in gap_windows:
            if any(gstart <= s < gend for s in run_steps):
                kind = "GAP"
                break

    return {
        "length": len(run_steps),
        "kind": kind,
        "span": [span_start, span_end],
        "steps": run_steps,
    }


def _compare_row_fields(prefix: str, got_row: dict, committed_row: dict, fields: list[str]) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok_all = True
    for field in fields:
        got = got_row.get(field)
        exp = committed_row.get(field)
        ok = _value_match(got, exp)
        lines.append(
            f"{prefix} {field:<18} got={_fmt(got):<44} committed={_fmt(exp):<44} "
            f"{'OK' if ok else 'MISMATCH'}"
        )
        if not ok:
            ok_all = False
    return ok_all, lines


def _record_session_lines(lines: list[str], prefix: str, runs: list[dict]) -> None:
    run_desc = ", ".join(f"{r['length']}:{r['kind']}" for r in runs) if runs else "[]"
    gap_runs = [r["length"] for r in runs if r["kind"] == "GAP"]
    in_burst_runs = [r["length"] for r in runs if r["kind"] == "IN-BURST"]
    min_gap = min(gap_runs) if gap_runs else None
    _lines_append(
        lines,
        f"{prefix} runs={run_desc} | in_burst_count={len(in_burst_runs)} | gap_lengths={gap_runs} | min_gap={min_gap}",
    )


def main() -> None:
    lines: list[str] = []
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt_path = out_dir / "exp190.txt"
    committed_183_path = out_dir / "exp183_rows.json"
    committed_189_path = out_dir / "exp189_rows.json"

    def p(msg: str = "") -> None:
        _lines_append(lines, msg)

    p("=" * 80)
    p("EXP 190 — N4 FLICKER HYSTERESIS DIAGNOSTIC")
    p("=" * 80)
    p()

    mirro = Creature.load("creature/state/mirro")
    mirro_root = copy.deepcopy(mirro)
    mirro_root._state_dir = None
    base_cmap = list(mirro.world.cmap)
    n_colors = mirro.world.n_colors
    p(f"Loaded mirro: age={mirro.age_steps}, world={mirro.world.rows}x{mirro.world.cols}, n_colors={n_colors}")
    p()

    # ------------------------------------------------------------------
    # STEP 1: gates
    # ------------------------------------------------------------------
    p("=" * 80)
    p("STEP 1: GATE CHECKS")
    p("=" * 80)
    eq_pass, eq_detail = run_equivalence_gate_190(mirro_root, base_cmap, n_colors, committed_183_path)
    reg_pass, reg_detail = run_exp189_regression_gate_190(mirro_root, base_cmap, n_colors, committed_189_path)
    p(eq_detail)
    p()
    p(reg_detail)
    p()
    if not (eq_pass and reg_pass):
        p("GATE FAILURE — aborting before trace collection.")
        with open(out_txt_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        raise SystemExit(1)
    p("GATE CHECKS PASS.")
    p()

    # ------------------------------------------------------------------
    # STEP 2: pressure trace collection and bit-match
    # ------------------------------------------------------------------
    p("=" * 80)
    p("STEP 2: PRESSURE TRACE COLLECTION")
    p("=" * 80)

    committed_rows = _load_committed_rows(committed_189_path)
    committed_w = {
        (row["cell"]["name"], row["arm"], row["seed"]): row
        for row in committed_rows
        if row.get("kind") == "W"
    }
    committed_r = {
        (row["arm"], row["seed"]): row
        for row in committed_rows
        if row.get("kind") == "R"
    }

    pressure_sessions: dict[tuple, dict] = {}
    all_pass = True

    # R sessions: REG-TB across all 8 seeds.
    for seed in TRACE_CASES_R:
        root = copy.deepcopy(mirro_root)
        root._state_dir = None
        rr = run_fork_schedule_190(
            mirro=root,
            fork_seed=seed,
            base_cmap=base_cmap,
            n_colors=n_colors,
            arm_name="REG-TB",
            arm_mode=("reg_tb", 1.5, 2800, INT_CALM_SNAPS),
            phase="R",
            burst_windows=[],
            color_mode="endogenous",
            reloc_rng_seed=BURST_SEED_OFFSET_R + seed,
            n_steps=N_STEPS_PHASE_R,
            theta=DEFAULT_THETA,
            release_calm_snaps=INT_CALM_SNAPS,
            hysteresis_snaps=0,
            record_pressure_trace=True,
        )
        pressure_sessions[("R", seed)] = rr
        committed = committed_r[("REG-TB", seed)]
        lat = phase_r_latency(rr["expressed_arr"], rr["regime_color"], N_STEPS_PHASE_R)
        got = {
            "latency": lat,
            "n_resets": rr.get("n_resets", 0),
            "n_completed_stretches": rr.get("n_completed_stretches", 0),
            "stretch_log": rr.get("stretch_log", []),
        }
        exp = {
            "latency": committed["latency"],
            "n_resets": committed["n_resets"],
            "n_completed_stretches": committed["n_completed_stretches"],
            "stretch_log": committed["stretch_log"],
        }
        ok, lines_out = _compare_row_fields(
            f"R seed={seed}",
            got,
            exp,
            ["latency", "n_resets", "n_completed_stretches", "stretch_log"],
        )
        for line in lines_out:
            p(line)
        if not ok:
            all_pass = False

    # W sessions: REG-TB E1 and E3 at seeds 280 and 286.
    w_cases = [
        ("E1", 280, EXP189_E1_WINDOWS, 16500, 0),
        ("E1", 286, EXP189_E1_WINDOWS, 16500, 0),
        ("E3", 280, EXP189_E3_WINDOWS, 17700, 2),
        ("E3", 286, EXP189_E3_WINDOWS, 17700, 2),
    ]
    for cell_name, seed, burst_windows, n_steps, cell_idx in w_cases:
        root = copy.deepcopy(mirro_root)
        root._state_dir = None
        rr = run_fork_schedule_190(
            mirro=root,
            fork_seed=seed,
            base_cmap=base_cmap,
            n_colors=n_colors,
            arm_name="REG-TB",
            arm_mode=("reg_tb", 1.5, 2800, INT_CALM_SNAPS),
            phase="W",
            burst_windows=burst_windows,
            color_mode="exogenous_fixed",
            reloc_rng_seed=280_000 + 10_000 * cell_idx + seed,
            n_steps=n_steps,
            theta=DEFAULT_THETA,
            release_calm_snaps=INT_CALM_SNAPS,
            hysteresis_snaps=0,
            record_pressure_trace=True,
        )
        pressure_sessions[("W", cell_name, seed)] = rr
        committed = committed_w[(cell_name, "REG-TB", seed)]
        defense, frac, attack_color = compute_defense(rr["expressed_arr"], burst_windows, rr["burst_onset_color"], n_steps)
        got = {
            "stretch_log": rr.get("stretch_log", []),
            "S_max_final": rr.get("S_max_final", 0.0),
            "max_current_stretch": rr.get("max_current_stretch", 0.0),
            "defense": defense,
        }
        exp = {
            "stretch_log": committed["stretch_log"],
            "S_max_final": committed["S_max_final"],
            "max_current_stretch": committed["max_current_stretch"],
            "defense": committed["defense"],
        }
        ok, lines_out = _compare_row_fields(
            f"W {cell_name} seed={seed}",
            got,
            exp,
            ["stretch_log", "S_max_final", "max_current_stretch", "defense"],
        )
        for line in lines_out:
            p(line)
        if not ok:
            all_pass = False

    if not all_pass:
        p("TRACE BIT-MATCH FAILURE — aborting before analysis.")
        with open(out_txt_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        raise SystemExit(1)

    p()
    p("TRACE BIT-MATCH PASS.")
    p()

    # ------------------------------------------------------------------
    # STEP 3: analysis
    # ------------------------------------------------------------------
    p("=" * 80)
    p("STEP 3: RUN-LENGTH ANALYSIS")
    p("=" * 80)

    r_seed_to_runs: dict[int, list[dict]] = {}
    w_session_runs: dict[tuple[str, int], list[dict]] = {}

    for seed in TRACE_CASES_R:
        rr = pressure_sessions[("R", seed)]
        runs = _pressure_runs(rr.get("pressure_trace", []), None)
        r_seed_to_runs[seed] = runs
        run_lengths = [r["length"] for r in runs]
        global_max = max(run_lengths) if run_lengths else 0
        p(f"R seed={seed} run_lengths={run_lengths} global_max={global_max}")

    global_max_r = max(
        (max((r["length"] for r in runs), default=0) for runs in r_seed_to_runs.values()),
        default=0,
    )
    p(f"R global max flicker snaps = {global_max_r}")
    p()

    min_w_gap = None
    for cell_name, seed, burst_windows, _, _ in w_cases:
        rr = pressure_sessions[("W", cell_name, seed)]
        runs = _pressure_runs(rr.get("pressure_trace", []), burst_windows)
        w_session_runs[(cell_name, seed)] = runs
        gap_runs = [r["length"] for r in runs if r["kind"] == "GAP"]
        in_burst_count = sum(1 for r in runs if r["kind"] == "IN-BURST")
        if gap_runs:
            session_min_gap = min(gap_runs)
            min_w_gap = session_min_gap if min_w_gap is None else min(min_w_gap, session_min_gap)
        else:
            session_min_gap = None
        p(
            f"W {cell_name} seed={seed} in_burst_count={in_burst_count} gap_run_lengths={gap_runs} min_gap={session_min_gap}"
        )

    p()
    p(f"min W gap run snaps = {min_w_gap}")
    p()

    h = 8
    admissible_8 = (
        global_max_r <= h / 2
        and min_w_gap is not None
        and min_w_gap >= 2 * h
    )
    p("=" * 80)
    p("h-ADMISSIBILITY CHECK")
    p("=" * 80)
    p(
        f"h={h} snaps ({h * FINE_EVAL} steps): global_max_R={global_max_r} <= {h/2} and "
        f"min_W_gap={min_w_gap} >= {2*h} => {'ADMISSIBLE' if admissible_8 else 'INADMISSIBLE'}"
    )

    if not admissible_8:
        p()
        p("SEARCHING h in [2, 20]...")
        admissible_h: list[int] = []
        for candidate in range(2, 21):
            ok = (
                global_max_r <= candidate / 2
                and min_w_gap is not None
                and min_w_gap >= 2 * candidate
            )
            if ok:
                admissible_h.append(candidate)
            p(
                f"  h={candidate:2d} -> {'OK' if ok else 'NO'} "
                f"(R margin {global_max_r} <= {candidate/2}, W margin {min_w_gap} >= {2*candidate})"
            )
        if admissible_h:
            p(f"ADMISSIBLE h values: {admissible_h}")
        else:
            p("NO ADMISSIBLE h — the hysteresis form is unbuildable at this body")
    p()

    # ------------------------------------------------------------------
    # STEP 4: counterfactual h sweep THROUGH THE REAL RUNNER (committed seeds —
    # the controller's best case; decision rule predeclared in the docstring)
    # ------------------------------------------------------------------
    p("=" * 80)
    p("STEP 4: COUNTERFACTUAL h SWEEP (real runner, committed seeds 280-287)")
    p("  buildable iff SOME h: revision diff <= +3000 in >= 6/8 seeds AND")
    p("  E1 and E3 defense both intact at the probed seeds")
    p("=" * 80)
    p()

    baseline_lat = {seed: committed_r[("baseline", seed)]["latency"] for seed in SEEDS_189}
    committed_regtb_lat = {seed: committed_r[("REG-TB", seed)]["latency"] for seed in SEEDS_189}
    p(f"committed baseline latencies: {baseline_lat}")
    p(f"committed REG-TB (h=0) latencies: {committed_regtb_lat}")
    p()

    H_SWEEP_R = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 46, 52, 58, 64, 70, 76, 84, 92, 104]
    rev_results: dict[int, dict[int, int | None]] = {}
    p("--- R side (revision) ---")
    for h_val in H_SWEEP_R:
        diffs: dict[int, int | None] = {}
        n_pass = 0
        for seed in SEEDS_189:
            root = copy.deepcopy(mirro_root)
            root._state_dir = None
            rr = run_fork_schedule_190(
                mirro=root,
                fork_seed=seed,
                base_cmap=base_cmap,
                n_colors=n_colors,
                arm_name="REG-TB",
                arm_mode=("reg_tb", 1.5, 2800, INT_CALM_SNAPS),
                phase="R",
                burst_windows=[],
                color_mode="endogenous",
                reloc_rng_seed=BURST_SEED_OFFSET_R + seed,
                n_steps=N_STEPS_PHASE_R,
                theta=DEFAULT_THETA,
                release_calm_snaps=INT_CALM_SNAPS,
                hysteresis_snaps=h_val,
            )
            lat = phase_r_latency(rr["expressed_arr"], rr["regime_color"], N_STEPS_PHASE_R)
            d = (lat - baseline_lat[seed]) if lat is not None else None
            diffs[seed] = d
            if d is not None and d <= 3000:
                n_pass += 1
        rev_results[h_val] = diffs
        diff_str = " ".join(
            f"s{seed}:{diffs[seed] if diffs[seed] is not None else 'None'}" for seed in SEEDS_189
        )
        p(f"  h={h_val:3d}: rev_pass={n_pass}/8  diffs: {diff_str}")
    p()

    p("--- W side (defense) ---")
    C1_WINDOWS = [(6000, 8400), (9000, 11400), (12000, 14400)]
    W_PROBE_CASES = [
        ("E1", EXP189_E1_WINDOWS, 16500, 0),
        ("E3", EXP189_E3_WINDOWS, 17700, 2),
        ("C1", C1_WINDOWS, 16900, 5),
    ]
    H_SWEEP_W = [8, 20, 21, 24, 33, 46, 58, 76]
    w_defense: dict[tuple[int, str], list[bool]] = {}
    for h_val in H_SWEEP_W:
        row_parts = []
        for cell_name, burst_windows, n_steps, cell_idx in W_PROBE_CASES:
            oks = []
            for seed in (280, 286):
                root = copy.deepcopy(mirro_root)
                root._state_dir = None
                rr = run_fork_schedule_190(
                    mirro=root,
                    fork_seed=seed,
                    base_cmap=base_cmap,
                    n_colors=n_colors,
                    arm_name="REG-TB",
                    arm_mode=("reg_tb", 1.5, 2800, INT_CALM_SNAPS),
                    phase="W",
                    burst_windows=burst_windows,
                    color_mode="exogenous_fixed",
                    reloc_rng_seed=280_000 + 10_000 * cell_idx + seed,
                    n_steps=n_steps,
                    theta=DEFAULT_THETA,
                    release_calm_snaps=INT_CALM_SNAPS,
                    hysteresis_snaps=h_val,
                )
                defense, _, _ = compute_defense(
                    rr["expressed_arr"], burst_windows, rr["burst_onset_color"], n_steps
                )
                n_concessions = sum(
                    1 for e in rr["events"] if e.get("label") == "concession"
                )
                oks.append(bool(defense))
                row_parts.append(
                    f"{cell_name}/s{seed}:def={'T' if defense else 'F'},conc={n_concessions}"
                )
            w_defense[(h_val, cell_name)] = oks
        p(f"  h={h_val:3d}: " + "  ".join(row_parts))
    p()

    p("--- BUILDABILITY TABLE (decision rule from the docstring) ---")
    p(f"  {'h':>4}  {'rev_pass':>8}  {'E1_def':>7}  {'E3_def':>7}  {'C1_def':>7}  {'BUILDABLE':>10}")
    any_buildable = False
    for h_val in H_SWEEP_R:
        n_pass = sum(
            1 for seed in SEEDS_189
            if rev_results[h_val][seed] is not None and rev_results[h_val][seed] <= 3000
        )
        # W defense at the nearest probed h at-or-below h_val (defense is
        # monotone non-increasing in h: larger h bridges strictly more runs)
        probed_at_or_below = [hw for hw in H_SWEEP_W if hw <= h_val]
        h_w = max(probed_at_or_below) if probed_at_or_below else None
        if h_w is None:
            e1_ok = e3_ok = c1_ok = True  # h below all probes: no gap bridged
        else:
            e1_ok = all(w_defense[(h_w, "E1")])
            e3_ok = all(w_defense[(h_w, "E3")])
            c1_ok = all(w_defense[(h_w, "C1")])
        buildable = (n_pass >= 6) and e1_ok and e3_ok
        if buildable:
            any_buildable = True
        p(
            f"  {h_val:>4}  {n_pass:>7}/8  {'OK' if e1_ok else 'LOST':>7}  "
            f"{'OK' if e3_ok else 'LOST':>7}  {'OK' if c1_ok else 'LOST':>7}  "
            f"{'YES' if buildable else 'no':>10}"
        )
    p()
    if any_buildable:
        p("STEP 4 VERDICT: a buildable h EXISTS on the committed seeds — the")
        p("fresh-seed pre-registered run is LICENSED at that h.")
    else:
        p("STEP 4 VERDICT: NO h IS BUILDABLE — revision rescue requires bridging")
        p("de-assert runs at or above the W gap-run scale; the de-assert-run-length")
        p("channel cannot separate changed-world onset from attack-train gaps at")
        p("this body. THE HYSTERESIS FORM IS REFUTED-AS-BUILDABLE AT DESIGN TIME.")
    p()

    with open(out_txt_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    p(f"Output written to {out_txt_path}")


if __name__ == "__main__":
    main()
