"""Exp 192 — N4 Retention PRICED (RUNG 1 completed with correctly-sized instrument).

PRE-REGISTERED in loop/directions/n4-crack-edges.md (commit a2e1e1b) BEFORE any data.

Question: With the instrument sized to the measured release dynamics, what is the
post-release retention RATE of the spanning defenses — and what distinguishes the one
observed surrendering freeze (s301 C-C: entry pre-attack on a settle-phase false alarm)
from its retaining structural twin (s297: entry at first-burst onset)?

Instrument corrections (each from an Exp 191 measurement):
  Tails 6,600 -> 12,000 (max observed final release was bend+6,550 via the re-freeze
  cycle; windows now fit releases to bend+8,500): n_steps 21,400 / 29,400 / 26,800
  for C-A / C-B / C-C. Unmeasured seats (WINDOW_OVERRUN) are excluded from BOTH
  numerator and denominator. Validity gate per covered pair: M = measured defended
  seats >= 10 of the pooled 16, else the pair is INSTRUMENT-FAILED (logged, not graded).

Arms (4): baseline; oracle (witness); INT-C2900; REG-TB. CALM2600-H6000 dropped
(failed its rung-1 precondition everywhere).

Seeds (two blocks, pooled for the bars):
  Block 1 = 296-303 (COMPLETION block: same seeds, longer tails).
  Block 2 = FRESH 304-311 (unbiased block).
  W-only: 3 cells x 4 arms x 16 seeds = 192 sessions.

Gates:
  G1: the exp183 equivalence gate through the exp190 runner (h=0), evidence emitted.
  G2 (prefix bit-match, the instrument license): for every Block-1 session, all
  quantities that lie within the exp191 session span must reproduce the committed
  exp191 rows exactly -- frozen_defense, the event chain (labels, entry/exit steps)
  up to the old n_steps, and every w-frac that was measured in exp191 (atol 1e-9).
  Any mismatch -> abort.

Bars (predeclared; pooled 16 seeds):
  Covered pairs: {INT-C2900, REG-TB} x {C-A, C-B, C-C} with frozen-defense >= 12/16;
  oracle covered where >= 14/16.
  retention_rate = retained / M over measured defended seats.

  P1: every covered pair >= 5/6. F1 (falsifier): any covered pair <= 2/3 (deferral is
    common, not rare). Between -> MIXED with per-pair detail.

  P2: baseline displaced in W1 >= 12/16 per cell (durability at the new horizon).
    F2 (falsifier): self-healing (w1_frac < 0.5) in >= 8/16 anywhere.

  P3: oracle retention >= 5/6 where covered.

  Prior prediction: Exp 191's measured seats ran 10/11 -- P1 is predicted to PASS
  unless s301-class deferral recurs at ~10%+ rate; either outcome prices the phenomenon.

The mechanism diagnostic (Part C; diagnostic-only, no bar):
  Re-run C-C x INT-C2900 x seeds {297, 301} at the ORIGINAL exp191 n_steps (21,400),
  bit-match gated against the committed exp191 rows, with v-trajectory dumps: v at
  freeze entry, at each burst boundary, at release, at release+500/+1000; the
  favorite-vs-attack margin through those points; and the post-release observation
  composition over [release, release+1000).
  H-mech check: s301's surrender traces to its AT-ENTRY state -- the freeze entered at
  2,025, BEFORE settle consolidation, so its frozen v carries a smaller
  favorite-over-attack margin at release than s297's (entry 6,075, post-settle);
  the alternative (comparable margins, divergence from post-release dynamics alone)
  would point at absorption/locality instead.

falsifier bindings (ordered per pre-registration):
  F1: any covered pair (arm, cell) has retention_rate <= 2/3 -- deferral is common.
  F2: baseline self-heals (w1_frac < 0.5 in >= 8/16 seeds) in any cell.

prediction: P1 PASS -- mechanism predicts retention (frozen-plateau law + quiet tail);
  P2 PASS -- baseline displacement persists; P3 PASS -- oracle is the exact-train witness.

Status: active | Seeds: 296-311 (blocks 1+2) | 3 cells x 4 arms x 16 seeds = 192 W sessions.
Runner: run_fork_schedule_190 with hysteresis_snaps=0 (exp188-verbatim code path).
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Import exp190 runner and helpers via importlib (verbatim exp191 pattern)
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_spec190 = _ilu.spec_from_file_location(
    "exp190",
    str(REPO_ROOT / "experiments" / "exp190_n4_flicker_hysteresis.py"),
)
_mod190 = _ilu.module_from_spec(_spec190)
_spec190.loader.exec_module(_mod190)  # type: ignore[union-attr]

run_equivalence_gate_190 = _mod190.run_equivalence_gate_190
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
# Exp 192 configuration
# ---------------------------------------------------------------------------

BLOCK1 = list(range(296, 304))  # [296..303]
BLOCK2 = list(range(304, 312))  # [304..311]
ALL_SEEDS = BLOCK1 + BLOCK2
N_SEEDS_TOTAL = len(ALL_SEEDS)  # 16

# Cells (3): tail extended to 12,000 steps.
# n_steps = settle(6000) + train + tail(12000); last_window_end + 12000 == n_steps
CELLS_192 = [
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

# Sanity checks
for _cell in CELLS_192:
    _last_end = _cell["burst_windows"][-1][1]
    assert _cell["n_steps"] % CHUNK_SIZE == 0, (
        f"Cell {_cell['name']}: n_steps={_cell['n_steps']} not divisible by CHUNK_SIZE={CHUNK_SIZE}"
    )
    assert _last_end + 12000 == _cell["n_steps"], (
        f"Cell {_cell['name']}: last_window_end={_last_end} + 12000 = {_last_end+12000} "
        f"!= n_steps={_cell['n_steps']}"
    )

# Arms (4) -- CALM2600-H6000 dropped
ARM_DEFS = [
    ("baseline",  "baseline",                            DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS),
    ("oracle",    "oracle",                              DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS),
    ("INT-C2900", ("int_c", 2900, CALM2600_SNAPS),       DEFAULT_THETA, CALM2600_SNAPS),
    ("REG-TB",    ("reg_tb", 1.5, 2800, CALM2600_SNAPS), DEFAULT_THETA, CALM2600_SNAPS),
]

assert len(ARM_DEFS) == 4, f"Expected 4 arms, got {len(ARM_DEFS)}"

# Retention window offsets from release_step
W1_OFFSET_START = 1000   # [release+1000, release+1500)
W1_OFFSET_END   = 1500
W2_OFFSET_START = 3000   # [release+3000, release+3500)
W2_OFFSET_END   = 3500


def _compute_retention_fracs(
    expressed_arr: np.ndarray,
    attack_color: int,
    release_step: int,
    n_steps: int,
) -> tuple[float | None, float | None]:
    """Compute W1 and W2 expr_frac(attack) after release. Returns (None, None) on overrun."""
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
) -> dict:
    """Run a single W session and return its row dict."""
    root = copy.deepcopy(mirro_root)
    root._state_dir = None

    cell_name = cell["name"]
    cell_idx = cell["cell_idx"]
    n_steps = cell["n_steps"]
    burst_windows = cell["burst_windows"]
    last_bend = burst_windows[-1][1]

    reloc_rng_seed = 296_000 + 10_000 * cell_idx + seed

    rr = run_fork_schedule_190(
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

    row: dict = {
        "exp": 192,
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
        "flags": flags,
        "ahat_drift": float(rr["ahat_drift"]),
    }
    return row, rr


def main() -> None:
    t_start = time.time()

    lines: list[str] = []
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt_path = out_dir / "exp192.txt"
    out_rows_path = out_dir / "exp192_rows.json"
    committed_183_path = out_dir / "exp183_rows.json"
    committed_191_path = out_dir / "exp191_rows.json"

    def p(msg: str = "") -> None:
        lines.append(msg)
        print(msg)

    p("=" * 80)
    p("EXP 192 -- N4 RETENTION PRICED (RUNG 1 COMPLETED WITH CORRECTLY-SIZED INSTRUMENT)")
    p("PRE-REGISTERED in loop/directions/n4-crack-edges.md (commit a2e1e1b) BEFORE any data")
    p(f"Seeds: Block1={BLOCK1}, Block2={BLOCK2}")
    p(f"3 cells x 4 arms x 16 seeds = 192 W sessions")
    p("Runner: run_fork_schedule_190 with hysteresis_snaps=0 (exp188-verbatim code path)")
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
    # STEP 1: EQUIVALENCE GATE (G1)
    # ====================================================================
    p("=" * 80)
    p("STEP 1: G1 -- EQUIVALENCE GATE (exp183 replay through exp190 runner)")
    p("=" * 80)

    eq_pass, eq_detail = run_equivalence_gate_190(
        mirro_root, base_cmap, n_colors, committed_183_path
    )
    p(eq_detail)
    p()

    if not eq_pass:
        p("G1 FAIL -- aborting.")
        with open(out_txt_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        sys.exit(1)

    p("G1 PASS -- proceeding.")
    p()

    # ====================================================================
    # Load committed exp191 rows for G2
    # ====================================================================
    exp191_rows: dict[tuple[str, str, int], dict] = {}
    with open(committed_191_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "analysis":
                continue
            # Only seeds in BLOCK1 (296-303), arms in exp192 ARM_DEFS (no CALM2600-H6000)
            cell_name = row.get("cell", {}).get("name")
            arm = row.get("arm")
            seed = row.get("seed")
            if cell_name and arm and seed is not None:
                exp191_rows[(cell_name, arm, seed)] = row

    p(f"Loaded {len(exp191_rows)} exp191 session rows for G2 comparisons.")
    p()

    # ====================================================================
    # STEP 2: BLOCK-1 SESSIONS + G2 GATE
    # ====================================================================
    p("=" * 80)
    p("STEP 2: BLOCK-1 SESSIONS (seeds 296-303) + G2 PREFIX BIT-MATCH GATE")
    p("=" * 80)
    p()

    all_rows: list[dict] = []
    g2_mismatches: list[str] = 0
    g2_mismatches = []
    g2_comparisons: list[str] = []

    t_grid = time.time()
    n_b1_total = len(CELLS_192) * len(ARM_DEFS) * len(BLOCK1)
    n_done = 0

    for cell in CELLS_192:
        cell_name = cell["name"]
        for arm_name, arm_mode, theta_val, rc_snaps in ARM_DEFS:
            for seed in BLOCK1:
                row, rr = _run_session(
                    mirro_root, base_cmap, n_colors,
                    cell, arm_name, arm_mode, theta_val, rc_snaps, seed
                )
                all_rows.append(row)
                n_done += 1

                if n_done % 12 == 0 or n_done == n_b1_total:
                    elapsed = time.time() - t_grid
                    rate = elapsed / max(1, n_done)
                    remaining = rate * (n_b1_total - n_done)
                    print(
                        f"  B1 [{n_done:3d}/{n_b1_total}] cell={cell_name} arm={arm_name} seed={seed} "
                        f"| def={row['frozen_defense']} ret={row['retained']} "
                        f"| {elapsed:.0f}s elapsed, ETA {remaining:.0f}s",
                        flush=True,
                    )

                # ---- G2 comparison ----
                e191 = exp191_rows.get((cell_name, arm_name, seed))
                if e191 is None:
                    # No exp191 row for this (arm maybe was CALM2600-H6000 only, or arm not in 191)
                    cmp_note = f"  G2 {cell_name}/{arm_name}/{seed}: no exp191 row (arm not in exp191 or filtered) -- SKIP"
                    g2_comparisons.append(cmp_note)
                    continue

                # (a) frozen_defense
                mismatch_this = False
                fd_match = (row["frozen_defense"] == e191["frozen_defense"])
                cmp_note = f"  G2 {cell_name}/{arm_name}/{seed}: frozen_defense exp191={e191['frozen_defense']} exp192={row['frozen_defense']} -> {'OK' if fd_match else 'MISMATCH'}"
                g2_comparisons.append(cmp_note)
                if not fd_match:
                    g2_mismatches.append(f"MISMATCH frozen_defense {cell_name}/{arm_name}/{seed}: {e191['frozen_defense']} vs {row['frozen_defense']}")
                    mismatch_this = True

                # (b) event chain: every exp191 event must appear identically at head of exp192 list
                e191_events = e191.get("events_summary", [])
                e192_events = row.get("events_summary", [])

                # exp191 n_steps for this cell (old values)
                OLD_N_STEPS = {"C-A": 16000, "C-B": 24000, "C-C": 21400}
                old_n = OLD_N_STEPS[cell_name]

                # Events from exp191 must appear at head; exp192 may have more (tail events)
                # An exp191 event with exit_step >= old_n_steps cannot exist (all exits were within span)
                n_e191 = len(e191_events)

                if n_e191 > len(e192_events):
                    g2_mismatches.append(
                        f"MISMATCH event count {cell_name}/{arm_name}/{seed}: exp191 has {n_e191} events but exp192 has {len(e192_events)}"
                    )
                    g2_comparisons.append(
                        f"    G2 event chain: exp191={n_e191} exp192={len(e192_events)} -> MISMATCH (exp191 has more events)"
                    )
                    mismatch_this = True
                else:
                    all_ev_ok = True
                    for i, ev191 in enumerate(e191_events):
                        ev192 = e192_events[i]
                        ev_ok = (
                            ev191["label"] == ev192["label"]
                            and ev191["entry_step"] == ev192["entry_step"]
                            and ev191["exit_step"] == ev192["exit_step"]
                        )
                        g2_comparisons.append(
                            f"    G2 event[{i}] {cell_name}/{arm_name}/{seed}: "
                            f"label={ev191['label']}/{ev192['label']} "
                            f"entry={ev191['entry_step']}/{ev192['entry_step']} "
                            f"exit={ev191['exit_step']}/{ev192['exit_step']} "
                            f"-> {'OK' if ev_ok else 'MISMATCH'}"
                        )
                        if not ev_ok:
                            g2_mismatches.append(
                                f"MISMATCH event[{i}] {cell_name}/{arm_name}/{seed}: "
                                f"exp191={ev191} exp192={ev192}"
                            )
                            all_ev_ok = False
                            mismatch_this = True
                    if n_e191 < len(e192_events):
                        g2_comparisons.append(
                            f"    G2 {cell_name}/{arm_name}/{seed}: "
                            f"exp192 has {len(e192_events)-n_e191} additional tail event(s) beyond exp191 -- allowed"
                        )

                # (c) w-frac comparison: only if release_step unchanged
                release_moved = (row["release_step"] != e191["release_step"])
                if release_moved:
                    g2_comparisons.append(
                        f"    G2 w-frac {cell_name}/{arm_name}/{seed}: release moved "
                        f"exp191={e191['release_step']} -> exp192={row['release_step']} "
                        f"(new tail events) -- w-frac compare SKIPPED"
                    )
                else:
                    # Compare w-fracs that were measured in exp191 (not null)
                    for wname in ("w1_frac", "w2_frac"):
                        v191 = e191.get(wname)
                        v192 = row.get(wname)
                        if v191 is not None:
                            if v192 is None:
                                g2_comparisons.append(
                                    f"    G2 {wname} {cell_name}/{arm_name}/{seed}: "
                                    f"exp191={v191} exp192=None -> MISMATCH (was measured, now None)"
                                )
                                g2_mismatches.append(
                                    f"MISMATCH {wname} {cell_name}/{arm_name}/{seed}: exp191={v191} exp192=None"
                                )
                                mismatch_this = True
                            elif abs(v191 - v192) > 1e-9:
                                g2_comparisons.append(
                                    f"    G2 {wname} {cell_name}/{arm_name}/{seed}: "
                                    f"exp191={v191} exp192={v192} diff={abs(v191-v192):.2e} -> MISMATCH"
                                )
                                g2_mismatches.append(
                                    f"MISMATCH {wname} {cell_name}/{arm_name}/{seed}: "
                                    f"exp191={v191} exp192={v192} diff={abs(v191-v192):.2e}"
                                )
                                mismatch_this = True
                            else:
                                g2_comparisons.append(
                                    f"    G2 {wname} {cell_name}/{arm_name}/{seed}: "
                                    f"exp191={v191} exp192={v192} -> OK"
                                )
                        # If v191 is None: not measured in exp191, no comparison required

    # Print G2 evidence
    p("G2 EVIDENCE TABLE (one line per compared field):")
    for cmp in g2_comparisons:
        p(cmp)
    p()

    if g2_mismatches:
        p("G2 MISMATCHES DETECTED:")
        for mm in g2_mismatches:
            p(f"  {mm}")
        p()
        p("G2 FAIL -- prefix bit-match violated; aborting.")
        with open(out_txt_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        sys.exit(1)

    p("G2 PASS -- all Block-1 prefixes match exp191 committed rows.")
    p()

    # ====================================================================
    # STEP 3: BLOCK-2 SESSIONS (seeds 304-311)
    # ====================================================================
    p("=" * 80)
    p("STEP 3: BLOCK-2 SESSIONS (fresh seeds 304-311)")
    p("=" * 80)
    p()

    n_b2_total = len(CELLS_192) * len(ARM_DEFS) * len(BLOCK2)
    n_done2 = 0
    t_b2 = time.time()

    for cell in CELLS_192:
        cell_name = cell["name"]
        for arm_name, arm_mode, theta_val, rc_snaps in ARM_DEFS:
            for seed in BLOCK2:
                row, _ = _run_session(
                    mirro_root, base_cmap, n_colors,
                    cell, arm_name, arm_mode, theta_val, rc_snaps, seed
                )
                all_rows.append(row)
                n_done2 += 1

                if n_done2 % 12 == 0 or n_done2 == n_b2_total:
                    elapsed = time.time() - t_b2
                    rate = elapsed / max(1, n_done2)
                    remaining = rate * (n_b2_total - n_done2)
                    print(
                        f"  B2 [{n_done2:3d}/{n_b2_total}] cell={cell_name} arm={arm_name} seed={seed} "
                        f"| def={row['frozen_defense']} ret={row['retained']} "
                        f"| {elapsed:.0f}s elapsed, ETA {remaining:.0f}s",
                        flush=True,
                    )

    p()
    p(f"Block-2 done: {(time.time()-t_b2)/60:.1f} min ({n_b2_total} sessions)")
    p()

    # ====================================================================
    # Write session rows to JSONL (after blocks 1+2)
    # ====================================================================
    with open(out_rows_path, "w") as fh:
        for row in all_rows:
            fh.write(json.dumps(row) + "\n")

    # ====================================================================
    # STEP 4: PART C -- Mechanism diagnostic
    # ====================================================================
    p("=" * 80)
    p("STEP 4: PART C -- MECHANISM DIAGNOSTIC")
    p("C-C x INT-C2900 x seeds {297, 301} at ORIGINAL exp191 n_steps=21400")
    p("=" * 80)
    p()

    PART_C_CELL = next(c for c in CELLS_192 if c["name"] == "C-C")
    PART_C_N_STEPS_ORIG = 21400
    PART_C_SEEDS = [297, 301]
    PART_C_ARM = ("INT-C2900", ("int_c", 2900, CALM2600_SNAPS), DEFAULT_THETA, CALM2600_SNAPS)
    arm_name_c, arm_mode_c, theta_c, rc_snaps_c = PART_C_ARM

    # Burst windows for C-C (original span)
    burst_windows_c = PART_C_CELL["burst_windows"]
    last_bend_c = burst_windows_c[-1][1]  # 14800

    # Build a mini cell dict for the original n_steps
    cell_c_orig = {
        "name": "C-C",
        "L": PART_C_CELL["L"],
        "K": PART_C_CELL["K"],
        "G": PART_C_CELL["G"],
        "burst_windows": burst_windows_c,
        "n_steps": PART_C_N_STEPS_ORIG,
        "cell_idx": PART_C_CELL["cell_idx"],
    }

    part_c_data = {}

    for seed in PART_C_SEEDS:
        p(f"--- Part C: C-C / INT-C2900 / seed={seed} at n_steps={PART_C_N_STEPS_ORIG} ---")

        root = copy.deepcopy(mirro_root)
        root._state_dir = None

        reloc_rng_seed = 296_000 + 10_000 * PART_C_CELL["cell_idx"] + seed

        rr = run_fork_schedule_190(
            mirro=root,
            fork_seed=seed,
            base_cmap=base_cmap,
            n_colors=n_colors,
            arm_name=arm_name_c,
            arm_mode=arm_mode_c,
            phase="W",
            burst_windows=burst_windows_c,
            color_mode="exogenous_fixed",
            reloc_rng_seed=reloc_rng_seed,
            n_steps=PART_C_N_STEPS_ORIG,
            theta=theta_c,
            release_calm_snaps=rc_snaps_c,
            hysteresis_snaps=0,
        )

        frozen_defense_c, _, attack_color_c = compute_defense(
            rr["expressed_arr"], burst_windows_c, rr["burst_onset_color"], PART_C_N_STEPS_ORIG
        )
        events_c = rr.get("events", [])

        # Bit-match gate vs exp191
        e191_c = exp191_rows.get(("C-C", "INT-C2900", seed))
        if e191_c is None:
            p(f"  WARNING: no exp191 row for C-C/INT-C2900/seed={seed}")
        else:
            fd_ok = (frozen_defense_c == e191_c["frozen_defense"])
            p(f"  Bit-match frozen_defense: exp191={e191_c['frozen_defense']} exp192={frozen_defense_c} -> {'OK' if fd_ok else 'MISMATCH'}")
            # events
            e191_evs = e191_c.get("events_summary", [])
            evs_c = [
                {"label": e["label"], "entry_step": e["entry_step"], "exit_step": e.get("exit_step")}
                for e in events_c
            ]
            ev_ok = (len(evs_c) == len(e191_evs)) and all(
                evs_c[i]["label"] == e191_evs[i]["label"]
                and evs_c[i]["entry_step"] == e191_evs[i]["entry_step"]
                and evs_c[i]["exit_step"] == e191_evs[i]["exit_step"]
                for i in range(len(e191_evs))
            )
            p(f"  Bit-match events: exp191={len(e191_evs)} exp192={len(evs_c)} -> {'OK' if ev_ok else 'MISMATCH'}")
            # w-fracs
            for wname in ("w1_frac", "w2_frac"):
                v191 = e191_c.get(wname)
                v192_c = None
                if v191 is not None:
                    # compute w-frac from the run
                    rs_c = e191_c.get("release_step")
                    if rs_c is not None and attack_color_c is not None:
                        offset_s = W1_OFFSET_START if wname == "w1_frac" else W2_OFFSET_START
                        offset_e = W1_OFFSET_END if wname == "w1_frac" else W2_OFFSET_END
                        ws = rs_c + offset_s
                        we = rs_c + offset_e
                        if we <= PART_C_N_STEPS_ORIG:
                            v192_c = float(np.mean(rr["expressed_arr"][ws:we] == attack_color_c))
                    diff = abs(v191 - v192_c) if v192_c is not None else None
                    wf_ok = diff is not None and diff <= 1e-9
                    p(f"  Bit-match {wname}: exp191={v191} exp192={v192_c} diff={diff} -> {'OK' if wf_ok else 'MISMATCH'}")

        # v-trajectory dump
        # v_traj is np.array of shape (n_snaps, n_colors); snap i -> step (i+1)*EVAL
        v_traj = rr.get("v_traj", None)
        obs_arr = rr.get("obs_arr", None)

        # EVAL constant: v_traj snaps are recorded every EVAL steps
        EVAL_STEPS = 100  # from exp188: EVAL=100

        # Determine key checkpoints
        # release_step from events
        if events_c:
            release_step_c = events_c[-1].get("exit_step")
        else:
            release_step_c = last_bend_c

        freeze_entry_c = events_c[0]["entry_step"] if events_c else None

        # Key steps to dump
        key_steps = []
        if freeze_entry_c is not None:
            key_steps.append(("freeze_entry", freeze_entry_c))
        for i, (ws, we) in enumerate(burst_windows_c):
            key_steps.append((f"burst{i+1}_start", ws))
            key_steps.append((f"burst{i+1}_end", we))
        if release_step_c is not None:
            key_steps.append(("release", release_step_c))
            key_steps.append(("release+500", release_step_c + 500))
            key_steps.append(("release+1000", release_step_c + 1000))

        # Build v_traj lookup: step -> v_array
        # v_traj shape: (n_snaps, n_colors); snap index i -> global_step = (i+1)*EVAL_STEPS
        if v_traj is not None and v_traj.ndim == 2 and v_traj.shape[0] > 0:
            v_traj_dict = {(i + 1) * EVAL_STEPS: v_traj[i] for i in range(v_traj.shape[0])}
            v_steps_sorted = sorted(v_traj_dict.keys())
        else:
            v_traj_dict = {}
            v_steps_sorted = []

        attack_color_c_int = int(attack_color_c) if attack_color_c is not None else None

        p(f"  attack_color={attack_color_c_int}")
        p(f"  release_step={release_step_c}")
        p()
        p(f"  {'checkpoint':<20} {'step':>6}  {'argmax':>6}  {'margin':>8}  v_vector")
        p(f"  {'-'*70}")

        at_release_margin = None

        for (label, target_step) in key_steps:
            if not v_traj_dict:
                p(f"  {label:<20} {target_step:>6}  v_traj unavailable")
                continue
            # Find nearest snapshot
            nearest = min(v_steps_sorted, key=lambda s: abs(s - target_step))
            v = v_traj_dict[nearest]
            argmax_v = int(np.argmax(v))
            v_str = " ".join(f"{x:.4f}" for x in v)
            if attack_color_c_int is not None and argmax_v == attack_color_c_int:
                # argmax IS the attack color
                # margin = v[attack] - v[second-best]
                v_sorted_idx = np.argsort(v)[::-1]
                second_best = int(v_sorted_idx[1])
                margin = -(v[argmax_v] - v[second_best])  # negative (attack winning)
            elif attack_color_c_int is not None:
                margin = v[argmax_v] - v[attack_color_c_int]
            else:
                margin = float("nan")
            p(f"  {label:<20} {nearest:>6}  {argmax_v:>6}  {margin:>8.4f}  [{v_str}]")

            if label == "release":
                at_release_margin = float(margin)

        part_c_data[seed] = {
            "release_step": release_step_c,
            "at_release_margin": at_release_margin,
        }

        # Post-release observation composition [release, release+1000)
        p()
        if obs_arr is not None and release_step_c is not None and attack_color_c_int is not None:
            obs_start = release_step_c
            obs_end = min(release_step_c + 1000, PART_C_N_STEPS_ORIG)
            obs_window = obs_arr[obs_start:obs_end]
            if len(obs_window) > 0:
                obs_frac_attack = float(np.mean(obs_window == attack_color_c_int))
                p(f"  Post-release obs composition [release, release+1000): "
                  f"frac_attack={obs_frac_attack:.4f} ({int(np.sum(obs_window == attack_color_c_int))}/{len(obs_window)} obs == attack_color)")
            else:
                p(f"  Post-release obs composition: window empty")
        else:
            p(f"  obs composition unavailable (obs_arr not in runner result dict)")
        p()

    # H-mech check
    p("H-MECH CHECK (falsifiable expectation):")
    m297 = part_c_data.get(297, {}).get("at_release_margin")
    m301 = part_c_data.get(301, {}).get("at_release_margin")
    p(f"  s297 at-release margin (entry 6075, post-settle):    {m297}")
    p(f"  s301 at-release margin (entry 2025, pre-settle):     {m301}")
    if m297 is not None and m301 is not None:
        if m297 > m301:
            p(f"  H-mech CONFIRMED: s297 margin ({m297:.4f}) > s301 margin ({m301:.4f}) -- "
              f"settle consolidation explains the deferral gap")
        else:
            p(f"  H-mech NOT CONFIRMED: s297 margin ({m297:.4f}) <= s301 margin ({m301:.4f}) -- "
              f"divergence from post-release dynamics; absorption/locality candidate")
    p()

    # ====================================================================
    # STEP 5: ANALYSIS OVER POOLED 16 SEEDS
    # ====================================================================
    p("=" * 80)
    p("STEP 5: ANALYSIS OVER POOLED 16 SEEDS")
    p("=" * 80)
    p()

    cell_names_order = [c["name"] for c in CELLS_192]

    # Index all rows
    row_index: dict[tuple[str, str, int], dict] = {}
    for row in all_rows:
        key = (row["cell"]["name"], row["arm"], row["seed"])
        row_index[key] = row

    # ---- 5a. Frozen-defense counts per (arm, cell) ----
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

    # ---- 5b. Covered pairs ----
    # Non-oracle: {INT-C2900, REG-TB} x {C-A, C-B, C-C} covered if frozen_def >= 12/16
    CANDIDATE_PAIRS: list[tuple[str, str]] = []
    for arm_name in ["INT-C2900", "REG-TB"]:
        for cell_name in cell_names_order:
            CANDIDATE_PAIRS.append((arm_name, cell_name))

    covered_pairs: list[tuple[str, str]] = []
    for arm_name, cell_name in CANDIDATE_PAIRS:
        cnt = frozen_def_counts.get((arm_name, cell_name), 0)
        if cnt >= 12:
            covered_pairs.append((arm_name, cell_name))

    # Oracle covered: cells where oracle frozen_def >= 14/16
    oracle_covered: list[tuple[str, str]] = []
    for cell_name in cell_names_order:
        cnt = frozen_def_counts.get(("oracle", cell_name), 0)
        if cnt >= 14:
            oracle_covered.append(("oracle", cell_name))

    covered_pairs_all = covered_pairs + oracle_covered

    p(f"Covered non-oracle pairs (frozen_def >= 12/16): {covered_pairs}")
    p(f"Oracle covered pairs (frozen_def >= 14/16):     {oracle_covered}")
    p()

    # ---- 5c. Per-covered-pair retention analysis with M validity gate ----
    p("RETENTION ANALYSIS PER COVERED PAIR")
    p("  M = measured defended seats (frozen_defense AND NOT overrun)")
    p("  Validity: M >= 10 else INSTRUMENT-FAILED")
    p("  retention_rate = retained / M")
    p("-" * 90)

    retention_rates: dict[tuple[str, str], float | None] = {}
    instrument_failed_pairs: list[tuple[str, str]] = []
    f1_fired = False
    mixed_zone_pairs: list[tuple[str, str]] = []
    p3_oracle_pass = True

    for arm_name, cell_name in covered_pairs_all:
        defended_seeds = [
            seed for seed in ALL_SEEDS
            if row_index.get((cell_name, arm_name, seed), {}).get("frozen_defense", False)
        ]
        n_defended = len(defended_seeds)

        if n_defended == 0:
            retention_rates[(arm_name, cell_name)] = None
            p(f"  {arm_name:<22} x {cell_name:<8}: n_defended=0 -- SKIP")
            continue

        # M = measured (frozen_defense AND NOT WINDOW_OVERRUN)
        measured_seeds = [
            seed for seed in defended_seeds
            if "WINDOW_OVERRUN" not in row_index.get((cell_name, arm_name, seed), {}).get("flags", [])
        ]
        M = len(measured_seeds)

        if M < 10:
            instrument_failed_pairs.append((arm_name, cell_name))
            p(f"  {arm_name:<22} x {cell_name:<8}: INSTRUMENT-FAILED (M={M} < 10) "
              f"(defended={n_defended}/16, measured={M}/16)")
            retention_rates[(arm_name, cell_name)] = None
            p()
            continue

        retained_count = sum(
            1 for seed in measured_seeds
            if row_index.get((cell_name, arm_name, seed), {}).get("retained") is True
        )
        rate = retained_count / M
        retention_rates[(arm_name, cell_name)] = rate

        p(f"  {arm_name:<22} x {cell_name:<8}: defended={n_defended}/16 M={M} "
          f"retained={retained_count}/{M} retention_rate={rate:.3f}")

        # Per-seed table (block, seed, frozen_defense, release_step, w1_frac, w2_frac, retained/overrun)
        p(f"    {'block':>6}  {'seed':>5}  {'frozen_def':>10}  {'release':>8}  "
          f"{'w1_frac':>8}  {'w2_frac':>8}  {'retained':>9}  {'flags':>30}")
        for seed in ALL_SEEDS:
            block = "B1" if seed in BLOCK1 else "B2"
            row = row_index.get((cell_name, arm_name, seed), {})
            fd = row.get("frozen_defense", False)
            rs = row.get("release_step")
            w1 = row.get("w1_frac")
            w2 = row.get("w2_frac")
            ret = row.get("retained")
            fl = ",".join(row.get("flags", []))
            rs_str = str(rs) if rs is not None else "None"
            w1_str = f"{w1:.3f}" if w1 is not None else "None"
            w2_str = f"{w2:.3f}" if w2 is not None else "None"
            p(f"    {block:>6}  {seed:>5}  {str(fd):>10}  {rs_str:>8}  "
              f"{w1_str:>8}  {w2_str:>8}  {str(ret):>9}  {fl:>30}")

        # Threshold check
        if rate <= 2 / 3:
            f1_fired = True
            p(f"    *** F1 FIRED: {arm_name} x {cell_name} retention_rate={rate:.3f} <= 2/3 ***")
        elif rate < 5 / 6:
            mixed_zone_pairs.append((arm_name, cell_name))
            p(f"    MIXED-ZONE: {arm_name} x {cell_name} rate={rate:.3f} (2/3 < rate < 5/6)")
        p()

    # P1 pass: no F1, no mixed zone, all valid covered pairs >= 5/6
    graded_pairs = [(pair, rate) for pair, rate in retention_rates.items() if rate is not None]
    p1_pass = (
        not f1_fired
        and len(mixed_zone_pairs) == 0
        and all(rate >= 5 / 6 for _, rate in graded_pairs if _ in covered_pairs_all)
    )

    # P3: oracle retention_rate >= 5/6 in every covered oracle cell
    for pair in oracle_covered:
        rate = retention_rates.get(pair)
        if rate is not None and rate < 5 / 6:
            p3_oracle_pass = False

    # ---- 5d. P2: baseline W1 displaced count per cell ----
    p("P2: BASELINE W1 DISPLACED COUNT PER CELL (w1_frac >= 0.5 = still displaced)")
    p("-" * 60)
    p(f"  {'cell':<8}  {'w1_displaced/16':>16}  {'P2_cell':>10}")
    p("-" * 60)

    p2_pass = True
    f2_fired = False
    baseline_w1_displaced: dict[str, int] = {}

    for cell_name in cell_names_order:
        displaced_count = sum(
            1 for seed in ALL_SEEDS
            if (row_index.get((cell_name, "baseline", seed), {}).get("w1_frac") or 0.0) >= 0.5
        )
        baseline_w1_displaced[cell_name] = displaced_count
        cell_p2 = displaced_count >= 12
        if not cell_p2:
            p2_pass = False
        not_displaced = N_SEEDS_TOTAL - displaced_count
        if not_displaced >= 8:
            f2_fired = True
        p(f"  {cell_name:<8}  {displaced_count:>11}/16  {'PASS' if cell_p2 else 'fail':>10}")
    p()
    if f2_fired:
        p("  *** F2 FIRED: baseline self-heals in at least one cell ***")
    p()

    # ---- 5e. PC1 summary ----
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

    p("P1 (retention >= 5/6 in every covered pair):")
    p(f"  Covered pairs (non-oracle): {covered_pairs}")
    p(f"  Covered pairs (oracle):     {oracle_covered}")
    p(f"  Instrument-failed pairs:    {instrument_failed_pairs}")
    for arm_name, cell_name in covered_pairs_all:
        rate = retention_rates.get((arm_name, cell_name))
        rate_str = f"{rate:.3f}" if rate is not None else "INSTRUMENT-FAILED"
        if rate is None:
            zone = "INSTRUMENT-FAILED"
        elif rate >= 5 / 6:
            zone = ">= 5/6"
        elif rate > 2 / 3:
            zone = "MIXED"
        else:
            zone = "F1"
        p(f"  {arm_name:<22} x {cell_name:<8}: rate={rate_str} -> {zone}")
    p(f"  F1 fired (any covered pair <= 2/3): {f1_fired}")
    p(f"  MIXED-zone pairs (2/3 < rate < 5/6): {mixed_zone_pairs}")
    if p1_pass:
        p("  => P1 VERDICT: PASS -- retention_rate >= 5/6 in every covered pair.")
    elif f1_fired:
        p("  => P1 VERDICT: NEGATIVE -- F1 FIRED. Deferral is common (rate <= 2/3) in covered pair(s).")
    else:
        p("  => P1 VERDICT: MIXED -- not all covered pairs clear 5/6 threshold.")
    p()

    p("P2 (baseline W1 displaced >= 12/16 per cell):")
    for cell_name in cell_names_order:
        cnt = baseline_w1_displaced[cell_name]
        p(f"  {cell_name}: W1_displaced={cnt}/16")
    p(f"  F2 fired (w1_frac < 0.5 in >= 8/16 anywhere): {f2_fired}")
    if p2_pass:
        p("  => P2 VERDICT: PASS -- baseline stays displaced post-release in all cells.")
    else:
        p("  => P2 VERDICT: fail -- some cells show < 12/16 baseline displacement.")
    if f2_fired:
        p("  => P2 F2: FIRED -- baseline self-heals in at least one cell.")
    p()

    p("P3 (oracle retention_rate >= 5/6 in every covered oracle cell):")
    if not oracle_covered:
        p("  No oracle-covered pairs (oracle frozen_def < 14/16 in all cells).")
        p("  => P3 VERDICT: N/A.")
    else:
        for arm_name, cell_name in oracle_covered:
            rate = retention_rates.get((arm_name, cell_name))
            rate_str = f"{rate:.3f}" if rate is not None else "INSTRUMENT-FAILED"
            p(f"  oracle x {cell_name}: rate={rate_str}")
        if p3_oracle_pass:
            p("  => P3 VERDICT: PASS -- oracle retention_rate >= 5/6 in all covered cells.")
        else:
            p("  => P3 VERDICT: fail -- oracle retention < 5/6 in at least one cell.")
    p()

    # ====================================================================
    # SUMMARY BLOCK
    # ====================================================================
    elapsed_total = time.time() - t_start
    runtime_min = elapsed_total / 60.0

    p("=" * 80)
    p("SUMMARY")
    p("=" * 80)
    p()
    p(f"  G1:                PASS")
    p(f"  G2:                PASS (Block-1 prefix bit-match verified)")
    p(f"  Seeds:             Block1={BLOCK1} Block2={BLOCK2}")
    p(f"  Runtime:           {runtime_min:.1f} min")
    p(f"  Sessions:          {len(all_rows)} W")
    p()
    p(f"  P1 (retention >= 5/6 every covered pair): {'PASS' if p1_pass else ('NEGATIVE (F1)' if f1_fired else 'MIXED')}")
    p(f"  P2 (baseline W1 displaced >= 12/16 every cell): {'PASS' if p2_pass else 'fail'}")
    p(f"  P3 (oracle retention >= 5/6 every covered cell): {'PASS' if (p3_oracle_pass and oracle_covered) else ('N/A' if not oracle_covered else 'fail')}")
    p()
    p(f"  F1 (any covered pair retention <= 2/3): {f1_fired}")
    p(f"  F2 (baseline self-heals in any cell):   {f2_fired}")
    p()
    p(f"  Instrument-failed pairs: {instrument_failed_pairs}")
    p(f"  PC1 flagged: {pc1_total}/192")
    p()
    p(f"  H-mech check:")
    p(f"    s297 at-release margin: {part_c_data.get(297, {}).get('at_release_margin')}")
    p(f"    s301 at-release margin: {part_c_data.get(301, {}).get('at_release_margin')}")
    p("=" * 80)

    # ====================================================================
    # Analysis row + finalize outputs
    # ====================================================================
    analysis_row: dict = {
        "exp": 192,
        "kind": "analysis",
        "seeds_b1": BLOCK1,
        "seeds_b2": BLOCK2,
        "covered_pairs": covered_pairs,
        "oracle_covered": oracle_covered,
        "instrument_failed_pairs": instrument_failed_pairs,
        "retention_rates": {
            f"{arm}|{cell}": to_plain(rate)
            for (arm, cell), rate in retention_rates.items()
        },
        "p1_pass": p1_pass,
        "f1_fired": f1_fired,
        "mixed_zone_pairs": mixed_zone_pairs,
        "p2_pass": p2_pass,
        "f2_fired": f2_fired,
        "p3_pass": p3_oracle_pass,
        "p3_na": not oracle_covered,
        "baseline_w1_displaced": baseline_w1_displaced,
        "pc1_total_flagged": pc1_total,
        "part_c": {
            str(seed): {
                "release_step": part_c_data[seed].get("release_step"),
                "at_release_margin": part_c_data[seed].get("at_release_margin"),
            }
            for seed in PART_C_SEEDS
        },
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
