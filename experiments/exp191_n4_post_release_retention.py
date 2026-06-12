"""Exp 191 — N4 Post-Release Retention (RUNG 1).

PRE-REGISTERED in loop/directions/n4-crack-edges.md (commit 55e7705) BEFORE any data.

Question: When a gap-spanning defense (continuity clock or frozen-time constant)
releases in the quiet tail, does the defended identity PERSIST, or does the deferred
displacement land after release? Both honest outcomes matter: retention strengthens
every banked defense claim; deferral bounds them all.

Mechanism stakes (written before data): During freeze v is bit-constant (the
frozen-plateau law); spanning arms pay ONE detection-head dose (~72-97 w-weighted
units) per train; post-release the world is quiet and free roaming adds no systematic
mass to the attack color -- retention is the mechanism-predicted outcome. But the
rung-1 autopsy (Exp 183 addendum) showed post-release expression dynamics decide
recovery with NO deterministic gap threshold below ~80 units of slack -- margin-class
seeds can plausibly flip after release. F1 is live, not a strawman.

Bars (predeclared):
  Covered pairs = (arm, cell) with banked defense AND fresh-seed frozen-defense >= 6/8:
    expected {INT-C2900, REG-TB} x {C-A, C-B, C-C}, CALM2600-H6000 x C-A,
    oracle x cells where frozen-defense >= 7/8.
  retention_rate(arm, cell) = #(frozen-defense AND W1 AND W2) / #(frozen-defense).

  P1 (retention): retention_rate >= 5/6 in EVERY covered pair. F1 (falsifier):
    any covered pair <= 1/2 -- the spanning defense DEFERS rather than defends there;
    the chapter's defense claims get explicitly bounded (logged as a new finding).
    Between 1/2 and 5/6 -> MIXED, per-pair detail reported.

  P2 (durability control): baseline stays displaced post-release -- expr_frac(attack)
    >= 0.5 in W1 in >= 6/8 seeds per cell. F2 (falsifier): baseline self-heals
    (< 0.5 in W1 in >= 4/8 anywhere) -- displacement is transient at these tails and
    the chapter's stakes weaken; report with both hands.

  P3 (witness): oracle retention_rate >= 5/6 wherever it defends >= 7/8.

  Precondition check: fresh-seed frozen-defense within +-2 seeds of banked levels
    per covered pair (a larger swing is flagged, not silently absorbed).

Verdict semantics: P1+P2 -> the banked defense results are REAL defenses
  (retention), chapter strengthened; F1 -> deferral named and bounded; F2 -> the
  displacement-durability premise itself is bounded at long tails.

falsifier bindings (ordered per pre-registration):
  F1: any covered pair (arm, cell) has retention_rate <= 1/2 -- deferral, not defense.
  F2: baseline self-heals (w1_frac < 0.5 in >= 4/8 seeds) in any cell.

prediction: P1 PASS -- mechanism predicts retention (frozen-plateau law + quiet tail);
  P2 PASS -- baseline displacement persists (no driving force removed); P3 PASS --
  oracle is the exact-train witness and sets the ceiling.

Status: active | Seeds: 296-303 | 3 cells x 5 arms x 8 seeds = 120 W sessions.
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
# Import exp190 runner and helpers via importlib (verbatim exp190_flicker_diagnostic pattern)
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
# Exp 191 configuration
# ---------------------------------------------------------------------------

SEEDS = [296, 297, 298, 299, 300, 301, 302, 303]
N_SEEDS = len(SEEDS)

# Cells (3): fixed-length, fresh geometry, tail extended to 6600 steps.
# n_steps = settle(6000) + train + tail(6600); last_window_end + 6600 == n_steps
CELLS_191 = [
    {
        "name": "C-A",
        "L": 1600,
        "K": 2,
        "G": 200,
        "burst_windows": [(6000, 7600), (7800, 9400)],
        "n_steps": 16000,
        "cell_idx": 0,
    },
    {
        "name": "C-B",
        "L": 2400,
        "K": 4,
        "G": 600,
        "burst_windows": [(6000, 8400), (9000, 11400), (12000, 14400), (15000, 17400)],
        "n_steps": 24000,
        "cell_idx": 1,
    },
    {
        "name": "C-C",
        "L": 400,
        "K": 4,
        "G": 2400,
        "burst_windows": [(6000, 6400), (8800, 9200), (11600, 12000), (14400, 14800)],
        "n_steps": 21400,
        "cell_idx": 2,
    },
]

# Sanity checks
for _cell in CELLS_191:
    _last_end = _cell["burst_windows"][-1][1]
    assert _cell["n_steps"] % CHUNK_SIZE == 0, (
        f"Cell {_cell['name']}: n_steps={_cell['n_steps']} not divisible by CHUNK_SIZE={CHUNK_SIZE}"
    )
    assert _last_end + 6600 == _cell["n_steps"], (
        f"Cell {_cell['name']}: last_window_end={_last_end} + 6600 = {_last_end+6600} "
        f"!= n_steps={_cell['n_steps']}"
    )

# Arms (5) -- verbatim exp189 ARM_DEFS tuple convention:
# (name, arm_mode, theta, release_calm_snaps)
ARM_DEFS = [
    ("baseline",       "baseline",                                        DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS),
    ("oracle",         "oracle",                                          DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS),
    ("INT-C2900",      ("int_c", 2900, CALM2600_SNAPS),                   DEFAULT_THETA, CALM2600_SNAPS),
    ("REG-TB",         ("reg_tb", 1.5, 2800, CALM2600_SNAPS),             DEFAULT_THETA, CALM2600_SNAPS),
    ("CALM2600-H6000", ("freeze_time", 6000),                             DEFAULT_THETA, CALM2600_SNAPS),
]

assert len(ARM_DEFS) == 5, f"Expected 5 arms, got {len(ARM_DEFS)}"

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


def main() -> None:
    t_start = time.time()

    lines: list[str] = []
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt_path = out_dir / "exp191.txt"
    out_rows_path = out_dir / "exp191_rows.json"
    committed_183_path = out_dir / "exp183_rows.json"

    def p(msg: str = "") -> None:
        lines.append(msg)
        print(msg)

    p("=" * 80)
    p("EXP 191 -- N4 POST-RELEASE RETENTION (RUNG 1)")
    p("PRE-REGISTERED in loop/directions/n4-crack-edges.md (commit 55e7705) BEFORE any data")
    p(f"Seeds: {SEEDS} | 3 cells x 5 arms x 8 seeds = 120 W sessions")
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
    # STEP 1: EQUIVALENCE GATE
    # ====================================================================
    p("=" * 80)
    p("STEP 1: EQUIVALENCE GATE (exp183 replay through exp190 runner)")
    p("=" * 80)

    eq_pass, eq_detail = run_equivalence_gate_190(
        mirro_root, base_cmap, n_colors, committed_183_path
    )
    p(eq_detail)
    p()

    if not eq_pass:
        p("EQUIVALENCE GATE FAIL -- aborting before grid.")
        with open(out_txt_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        sys.exit(1)

    p("EQUIVALENCE GATE PASS -- proceeding to grid.")
    p()

    # ====================================================================
    # STEP 2: W SESSIONS (3 cells x 5 arms x 8 seeds = 120)
    # ====================================================================
    p("=" * 80)
    p("STEP 2: W SESSIONS (3 cells x 5 arms x 8 seeds = 120 sessions)")
    p("=" * 80)
    p()

    all_rows: list[dict] = []

    t_grid = time.time()
    n_total = len(CELLS_191) * len(ARM_DEFS) * N_SEEDS
    n_done = 0

    for cell in CELLS_191:
        cell_name = cell["name"]
        cell_L = cell["L"]
        cell_K = cell["K"]
        cell_G = cell["G"]
        cell_idx = cell["cell_idx"]
        n_steps = cell["n_steps"]
        burst_windows = cell["burst_windows"]
        last_bend = burst_windows[-1][1]

        for arm_name, arm_mode, theta_val, rc_snaps in ARM_DEFS:
            for seed in SEEDS:
                root = copy.deepcopy(mirro_root)
                root._state_dir = None

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

                # Defense (banked instrument: [last_bend+1500, last_bend+2000))
                frozen_defense, final_expr_frac, attack_color = compute_defense(
                    rr["expressed_arr"], burst_windows, rr["burst_onset_color"], n_steps
                )

                # PC1 flag
                pc1_flag = rr["ahat_drift"] >= PC1_AHAT_DRIFT_MAX

                # Events summary with exit_step
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

                # Determine release_step and release_label
                flags: list[str] = []
                if pc1_flag:
                    flags.append("PC1_DRIFT")

                if arm_name == "baseline":
                    # Baseline: no freeze; release_step = last burst end (last_bend)
                    release_step: int | None = last_bend
                    release_label: str | None = "none_baseline"
                elif arm_name == "oracle":
                    # Oracle uses the legacy exp185 runner (exact-train freeze, witness);
                    # it does not emit events via the new event system. Its freeze covers
                    # burst windows exactly and releases at last_bend. Treat as last_bend.
                    # (If n_events > 0 for some future runner variant, use last event instead.)
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

                # MID_TRAIN_CONCESSION: any concession event with exit_step < last burst end
                if arm_name != "baseline":
                    for ev in events:
                        if ev.get("label") == "concession":
                            ev_exit = ev.get("exit_step")
                            if ev_exit is not None and ev_exit < last_bend:
                                if "MID_TRAIN_CONCESSION" not in flags:
                                    flags.append("MID_TRAIN_CONCESSION")
                                break

                # WINDOW_OVERRUN: release_step + 3500 > n_steps
                if release_step is not None and release_step + 3500 > n_steps:
                    flags.append("WINDOW_OVERRUN")

                # Retention windows W1 and W2
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
                    "exp": 191,
                    "cell": {
                        "name": cell_name,
                        "L": cell_L,
                        "K": cell_K,
                        "G": cell_G,
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
                all_rows.append(row)

                n_done += 1
                if n_done % 12 == 0 or n_done == n_total:
                    elapsed = time.time() - t_grid
                    rate = elapsed / max(1, n_done)
                    remaining = rate * (n_total - n_done)
                    print(
                        f"  [{n_done:3d}/{n_total}] cell={cell_name} arm={arm_name} seed={seed} "
                        f"| def={bool(frozen_defense)} ret={retained} "
                        f"| {elapsed:.0f}s elapsed, ETA {remaining:.0f}s",
                        flush=True,
                    )

    t_grid_done = time.time()
    p()
    p(f"W grid done: {(t_grid_done - t_grid)/60:.1f} min ({n_total} sessions)")
    p()

    # ====================================================================
    # Write session rows to JSONL
    # ====================================================================
    with open(out_rows_path, "w") as fh:
        for row in all_rows:
            fh.write(json.dumps(row) + "\n")

    # ====================================================================
    # STEP 3: ANALYSIS
    # ====================================================================
    p("=" * 80)
    p("STEP 3: ANALYSIS")
    p("=" * 80)
    p()

    cell_names_order = [c["name"] for c in CELLS_191]

    # Index rows by (cell_name, arm, seed)
    row_index: dict[tuple[str, str, int], dict] = {}
    for row in all_rows:
        key = (row["cell"]["name"], row["arm"], row["seed"])
        row_index[key] = row

    # ---- 3a. Frozen-defense counts per (arm, cell) ----
    p("FROZEN-DEFENSE COUNT TABLE (count/8 per (arm, cell))")
    p("-" * 70)
    header = f"  {'arm':<22}"
    for cn in cell_names_order:
        header += f"  {cn:>8}"
    p(header)
    p("-" * 70)

    frozen_def_counts: dict[tuple[str, str], int] = {}
    for arm_name, _, _, _ in ARM_DEFS:
        row_str = f"  {arm_name:<22}"
        for cell_name in cell_names_order:
            cnt = sum(
                1 for seed in SEEDS
                if row_index.get((cell_name, arm_name, seed), {}).get("frozen_defense", False)
            )
            frozen_def_counts[(arm_name, cell_name)] = cnt
            row_str += f"  {cnt:>7}/8"
        p(row_str)
    p()

    # ---- 3b. Covered pairs ----
    # Non-oracle: {INT-C2900, REG-TB} x {C-A, C-B, C-C} + CALM2600-H6000 x C-A
    # covered if frozen_def >= 6/8
    CANDIDATE_PAIRS: list[tuple[str, str]] = []
    for arm_name in ["INT-C2900", "REG-TB"]:
        for cell_name in cell_names_order:
            CANDIDATE_PAIRS.append((arm_name, cell_name))
    CANDIDATE_PAIRS.append(("CALM2600-H6000", "C-A"))

    covered_pairs: list[tuple[str, str]] = []
    for arm_name, cell_name in CANDIDATE_PAIRS:
        cnt = frozen_def_counts.get((arm_name, cell_name), 0)
        if cnt >= 6:
            covered_pairs.append((arm_name, cell_name))

    # Oracle covered: cells where oracle frozen_def >= 7/8
    oracle_covered: list[tuple[str, str]] = []
    for cell_name in cell_names_order:
        cnt = frozen_def_counts.get(("oracle", cell_name), 0)
        if cnt >= 7:
            oracle_covered.append(("oracle", cell_name))

    covered_pairs_all = covered_pairs + oracle_covered

    p(f"Covered non-oracle pairs (frozen_def >= 6/8): {covered_pairs}")
    p(f"Oracle covered pairs (frozen_def >= 7/8):     {oracle_covered}")
    p()

    # Precondition check: fresh-seed vs banked (assumed 8/8 for main arms)
    BANKED_DEF: dict[tuple[str, str], int] = {
        ("INT-C2900",      "C-A"): 8,
        ("INT-C2900",      "C-B"): 8,
        ("INT-C2900",      "C-C"): 8,
        ("REG-TB",         "C-A"): 8,
        ("REG-TB",         "C-B"): 8,
        ("REG-TB",         "C-C"): 8,
        ("CALM2600-H6000", "C-A"): 8,
    }
    p("PRECONDITION CHECK (fresh-seed frozen-defense vs banked levels, +-2 tolerance):")
    for (arm_name, cell_name), banked in BANKED_DEF.items():
        fresh = frozen_def_counts.get((arm_name, cell_name), 0)
        swing = abs(fresh - banked)
        flag = "OK" if swing <= 2 else "FLAGGED (swing > 2)"
        p(f"  {arm_name:<22} {cell_name:<8} banked={banked}/8 fresh={fresh}/8 swing={swing} {flag}")
    p()

    # ---- 3c. Per-covered-pair retention analysis ----
    p("RETENTION ANALYSIS PER COVERED PAIR")
    p("  retention_rate = #(frozen_defense AND retained) / #(frozen_defense)")
    p("-" * 80)

    retention_rates: dict[tuple[str, str], float | None] = {}
    f1_fired = False
    mixed_zone_pairs: list[tuple[str, str]] = []
    p3_oracle_pass = True

    for arm_name, cell_name in covered_pairs_all:
        defended_seeds = [
            seed for seed in SEEDS
            if row_index.get((cell_name, arm_name, seed), {}).get("frozen_defense", False)
        ]
        n_defended = len(defended_seeds)

        if n_defended == 0:
            retention_rates[(arm_name, cell_name)] = None
            p(f"  {arm_name:<22} x {cell_name:<8}: n_defended=0 -- SKIP")
            continue

        retained_count = sum(
            1 for seed in defended_seeds
            if row_index.get((cell_name, arm_name, seed), {}).get("retained") is True
        )
        rate = retained_count / n_defended
        retention_rates[(arm_name, cell_name)] = rate

        p(f"  {arm_name:<22} x {cell_name:<8}: retained={retained_count}/{n_defended} "
          f"retention_rate={rate:.3f}")

        # Per-seed table
        p(f"    {'seed':>6}  {'frozen_def':>10}  {'w1_frac':>8}  {'w2_frac':>8}  {'retained':>9}  {'flags':>20}")
        for seed in SEEDS:
            row = row_index.get((cell_name, arm_name, seed), {})
            fd = row.get("frozen_defense", False)
            w1 = row.get("w1_frac")
            w2 = row.get("w2_frac")
            ret = row.get("retained")
            fl = ",".join(row.get("flags", []))
            w1_str = f"{w1:.3f}" if w1 is not None else "None"
            w2_str = f"{w2:.3f}" if w2 is not None else "None"
            p(f"    {seed:>6}  {str(fd):>10}  {w1_str:>8}  {w2_str:>8}  {str(ret):>9}  {fl:>20}")

        # Threshold check
        if rate <= 0.5:
            f1_fired = True
            p(f"    *** F1 FIRED: {arm_name} x {cell_name} retention_rate={rate:.3f} <= 1/2 ***")
        elif rate < 5 / 6:
            mixed_zone_pairs.append((arm_name, cell_name))
            p(f"    MIXED-ZONE: {arm_name} x {cell_name} rate={rate:.3f} (1/2 < rate < 5/6)")
        p()

    # P1 pass: no F1, no mixed zone, all valid covered pairs >= 5/6
    p1_pass = (
        not f1_fired
        and len(mixed_zone_pairs) == 0
        and all(
            (retention_rates.get(pair) or 0.0) >= 5 / 6
            for pair in covered_pairs_all
            if retention_rates.get(pair) is not None
        )
    )

    # P3: oracle retention_rate >= 5/6 in every covered oracle cell
    for arm_name, cell_name in oracle_covered:
        rate = retention_rates.get((arm_name, cell_name))
        if rate is not None and rate < 5 / 6:
            p3_oracle_pass = False

    # ---- 3d. P2: baseline W1 displaced count per cell ----
    p("P2: BASELINE W1 DISPLACED COUNT PER CELL (w1_frac >= 0.5 = still displaced)")
    p("-" * 60)
    p(f"  {'cell':<8}  {'w1_displaced/8':>15}  {'P2_cell':>10}")
    p("-" * 60)

    p2_pass = True
    f2_fired = False
    baseline_w1_displaced: dict[str, int] = {}

    for cell_name in cell_names_order:
        displaced_count = sum(
            1 for seed in SEEDS
            if (row_index.get((cell_name, "baseline", seed), {}).get("w1_frac") or 0.0) >= 0.5
        )
        baseline_w1_displaced[cell_name] = displaced_count
        cell_p2 = displaced_count >= 6
        if not cell_p2:
            p2_pass = False
        not_displaced = N_SEEDS - displaced_count
        if not_displaced >= 4:
            f2_fired = True
        p(f"  {cell_name:<8}  {displaced_count:>10}/8  {'PASS' if cell_p2 else 'fail':>10}")
    p()
    if f2_fired:
        p("  *** F2 FIRED: baseline self-heals in at least one cell ***")
    p()

    # ---- 3e. PC1 summary ----
    pc1_total = sum(1 for row in all_rows if "PC1_DRIFT" in row.get("flags", []))
    p(f"PC1 flagged: {pc1_total}/120 sessions with ahat_drift >= {PC1_AHAT_DRIFT_MAX}")
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
    for arm_name, cell_name in covered_pairs_all:
        rate = retention_rates.get((arm_name, cell_name))
        rate_str = f"{rate:.3f}" if rate is not None else "None"
        if rate is None:
            zone = "N/A"
        elif rate >= 5 / 6:
            zone = ">= 5/6"
        elif rate > 0.5:
            zone = "MIXED"
        else:
            zone = "F1"
        p(f"  {arm_name:<22} x {cell_name:<8}: rate={rate_str} -> {zone}")
    p(f"  F1 fired (any covered pair <= 1/2): {f1_fired}")
    p(f"  MIXED-zone pairs (1/2 < rate < 5/6): {mixed_zone_pairs}")
    if p1_pass:
        p("  => P1 VERDICT: PASS -- retention_rate >= 5/6 in every covered pair.")
    elif f1_fired:
        p("  => P1 VERDICT: NEGATIVE -- F1 FIRED. Deferral detected in covered pair(s).")
    else:
        p("  => P1 VERDICT: MIXED -- not all covered pairs clear 5/6 threshold.")
    p()

    p("P2 (baseline W1 displaced >= 6/8 per cell):")
    for cell_name in cell_names_order:
        cnt = baseline_w1_displaced[cell_name]
        p(f"  {cell_name}: W1_displaced={cnt}/8")
    p(f"  F2 fired (< 0.5 in >= 4/8 anywhere): {f2_fired}")
    if p2_pass:
        p("  => P2 VERDICT: PASS -- baseline stays displaced post-release in all cells.")
    else:
        p("  => P2 VERDICT: fail -- some cells show < 6/8 baseline displacement.")
    if f2_fired:
        p("  => P2 F2: FIRED -- baseline self-heals in at least one cell.")
    p()

    p("P3 (oracle retention_rate >= 5/6 in every covered oracle cell):")
    if not oracle_covered:
        p("  No oracle-covered pairs (oracle frozen_def < 7/8 in all cells).")
        p("  => P3 VERDICT: N/A.")
    else:
        for arm_name, cell_name in oracle_covered:
            rate = retention_rates.get((arm_name, cell_name))
            rate_str = f"{rate:.3f}" if rate is not None else "None"
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
    p(f"  Gate:              PASS")
    p(f"  Seeds:             {SEEDS}")
    p(f"  Runtime:           {runtime_min:.1f} min")
    p(f"  Sessions:          {n_total} W")
    p()
    p(f"  P1 (retention >= 5/6 every covered pair): {'PASS' if p1_pass else ('NEGATIVE (F1)' if f1_fired else 'MIXED')}")
    p(f"  P2 (baseline W1 displaced >= 6/8 every cell): {'PASS' if p2_pass else 'fail'}")
    p(f"  P3 (oracle retention >= 5/6 every covered cell): {'PASS' if (p3_oracle_pass and oracle_covered) else ('N/A' if not oracle_covered else 'fail')}")
    p()
    p(f"  F1 (any covered pair retention <= 1/2): {f1_fired}")
    p(f"  F2 (baseline self-heals in any cell):   {f2_fired}")
    p()
    p("=" * 80)

    # ====================================================================
    # Analysis row + finalize outputs
    # ====================================================================
    analysis_row: dict = {
        "exp": 191,
        "kind": "analysis",
        "seeds": SEEDS,
        "covered_pairs": covered_pairs,
        "oracle_covered": oracle_covered,
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
