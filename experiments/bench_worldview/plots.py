"""T17 — worldview bench plots and tables.

Standalone module: generates PNG surprise-trace plots, appends a world x
mechanism summary table to summary.md, builds a 4x4 world x response
confusion-matrix scaffold and a nats-regret-vs-oracle table, and reports
structure-economy statistics.

Usage (standalone over committed rows):
    uv run --python .venv python experiments/bench_worldview/plots.py

Or import generate_all() and call it from run_bench.py after rows are written.

Design notes:
- Reads only *committed* (or freshly written) rows under outputs/.
- Requires matplotlib (confirmed available in the venv).
- All PNGs are written next to the rows files in outputs/.
- The world x mechanism summary table is APPENDED to an aggregate
  summary.md (outputs/aggregate_summary.md) — not to per-run summaries
  (those are run_bench outputs).
- Regret vs oracle uses oracle rows as the baseline; rows without a
  matching oracle row are marked "pending".
- Spec: rigor-fairness-upgrade T17.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_BENCH_DIR = Path(__file__).resolve().parent
_OUTPUTS_DIR = _BENCH_DIR / "outputs"

# ---------------------------------------------------------------------------
# Matplotlib import (T17 spec: emit SVG or note if unavailable; it IS available)
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

# ---------------------------------------------------------------------------
# World and mechanism ordering (display order)
# ---------------------------------------------------------------------------
_WORLD_ORDER = ["A", "B", "C", "D"]
_WORLD_LABELS = {
    "A": "A (learnable)",
    "B": "B (noisy)",
    "C": "C (aliased)",
    "D": "D (nonstat.)",
}
_MECH_ORDER = [
    "none", "decay", "random_accept", "replay_accept",
    "bigger_fixed", "grow", "oracle",
]
_MECH_LABELS = {
    "none": "none",
    "decay": "decay (Exp 137)",
    "random_accept": "random_accept",
    "replay_accept": "replay_accept (Exp 144)",
    "bigger_fixed": "bigger_fixed",
    "grow": "grow (Exp 145/153/154)",
    "oracle": "oracle (upper bound)",
}

# Alarm/probation interval constants (from growth.py — duplicated to avoid
# importing the whole module just for plotting)
try:
    from active_loop.growth import PROBATION_STEPS, SPAWN_INTERVAL
except ImportError:
    PROBATION_STEPS = 400
    SPAWN_INTERVAL = 50

# ---------------------------------------------------------------------------
# Row loading
# ---------------------------------------------------------------------------

def load_all_rows() -> list[dict]:
    """Load all *_rows.json files from the outputs directory."""
    rows: list[dict] = []
    for path in sorted(_OUTPUTS_DIR.glob("*_rows.json")):
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _group_rows(rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    """Group rows by (world, mechanism, convention)."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row.get("world", "?"), row.get("mechanism", "?"),
               row.get("convention", "normalized"))
        groups[key].append(row)
    return dict(groups)


# ---------------------------------------------------------------------------
# Surprise trace plots (per run)
# ---------------------------------------------------------------------------

def _make_surprise_trace_plot(
    row: dict,
    out_path: Path,
) -> None:
    """Draw a per-run surprise trace with alarm-event and probation-window bands.

    Since the run loop does not store step-by-step surprise values (too large
    for JSON rows), we synthesize a representative trace from the summary
    statistics stored in the row: plateau → final_surprise transition, with
    alarm_events marked and probation windows shaded.

    Synthetic trace construction:
    - Phase1 (steps 0..t_phase1): flat at plateau (final phase1 estimate)
    - Phase2 (steps t_phase1..t_total): linear decay from plateau to
      final_surprise over t_phase2 steps, with noise N(0, 0.05)
    - alarm_event_steps: uniformly spaced over the phase2 period (the runner
      records count but not step indices in the row summary; we place them
      at their natural SPAWN_INTERVAL grid positions)
    - Probation band: each alarm event → shaded window of PROBATION_STEPS

    This is a structural/visual scaffold, not a replay of recorded values.
    The recorded values are the per-run JSON rows.
    """
    if not _MPL_AVAILABLE:
        return

    t_phase1 = int(row.get("t_phase1", 600))
    t_phase2 = int(row.get("t_phase2", 1400))
    t_total = t_phase1 + t_phase2

    plateau = float(row.get("plateau", 1.0))
    final_s = float(row.get("final_surprise", plateau))
    n_alarm = int(row.get("alarm_events", 0))
    mechanism = row.get("mechanism", "?")
    world = row.get("world", "?")
    seed = row.get("seed", 0)
    layout_seed = row.get("layout_seed", 0)

    # Synthetic surprise trace
    rng = np.random.default_rng(int(seed) * 100 + int(layout_seed))
    p1_noise = rng.normal(0, 0.04, t_phase1)
    p2_base = np.linspace(plateau, final_s, t_phase2)
    p2_noise = rng.normal(0, 0.04, t_phase2)
    trace = np.concatenate([
        np.full(t_phase1, plateau) + p1_noise,
        p2_base + p2_noise,
    ])

    fig, ax = plt.subplots(figsize=(10, 4))

    # Phase boundary
    ax.axvline(t_phase1, color="gray", linestyle="--", alpha=0.5, lw=0.8, label="phase2 start")

    # Probation windows (shaded)
    if n_alarm > 0:
        # Place alarms at uniform SPAWN_INTERVAL positions in phase2
        alarm_steps = [
            t_phase1 + SPAWN_INTERVAL * (i + 1)
            for i in range(n_alarm)
            if t_phase1 + SPAWN_INTERVAL * (i + 1) < t_total
        ]
        for i, alarm_t in enumerate(alarm_steps):
            prob_end = min(alarm_t + PROBATION_STEPS, t_total)
            ax.axvspan(
                alarm_t, prob_end,
                alpha=0.12, color="orange",
                label="probation window" if i == 0 else None,
            )
            ax.axvline(
                alarm_t, color="red", alpha=0.7, lw=0.8,
                label="alarm event" if i == 0 else None,
            )

    # Surprise trace
    ax.plot(np.arange(t_total), trace, lw=0.9, color="steelblue", alpha=0.85, label="surprise (nats)")

    # Reference lines
    ax.axhline(plateau, color="gray", linestyle=":", alpha=0.4, lw=0.8)
    ax.axhline(final_s, color="steelblue", linestyle=":", alpha=0.4, lw=0.8)

    ax.set_xlabel("step")
    ax.set_ylabel("surprise (nats)")
    title = (
        f"world={world}  mechanism={mechanism}  "
        f"seed={seed}  layout={layout_seed}\n"
        f"plateau={plateau:.3f}  final={final_s:.3f}  "
        f"alarms={n_alarm}  accepted={row.get('growth_accepted', 0)}"
    )
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, t_total)

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def generate_surprise_traces(rows: list[dict]) -> list[Path]:
    """Generate one PNG per row.  Returns list of written paths."""
    written: list[Path] = []
    for row in rows:
        world = row.get("world", "X")
        mech = row.get("mechanism", "none")
        conv = row.get("convention", "normalized")
        seed = row.get("seed", 0)
        layout = row.get("layout_seed", 0)
        t1 = row.get("t_phase1", 600)
        t2 = row.get("t_phase2", 1400)
        fname = (
            f"trace_{world}_{mech}_{conv}_s{seed}_l{layout}_"
            f"t{t1}x{t2}.png"
        )
        out_path = _OUTPUTS_DIR / fname
        _make_surprise_trace_plot(row, out_path)
        written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# World × mechanism summary table
# ---------------------------------------------------------------------------

def _mean_or_na(vals: list[float], fmt: str = ".3f") -> str:
    if not vals:
        return "N/A"
    v = [x for x in vals if not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return "N/A"
    return format(float(np.mean(v)), fmt)


def build_summary_table(rows: list[dict]) -> str:
    """Build a world x mechanism summary table as markdown.

    Columns: world | mechanism | n_runs | mean_plateau | mean_final_surprise
             | mean_drop | total_alarms | total_accepted | total_reverted
             | mean_comps_used | comps_needed
    """
    groups = _group_rows(rows)

    lines = [
        "## World × Mechanism Summary Table",
        "",
        "| world | mechanism | n | plateau | final_surprise | drop "
        "| alarms | accepted | reverted | comps_used | comps_needed |",
        "|-------|-----------|---|---------|----------------|------"
        "|--------|----------|----------|------------|--------------|",
    ]

    for world in _WORLD_ORDER:
        for mech in _MECH_ORDER:
            for conv in ("normalized",):
                key = (world, mech, conv)
                if key not in groups:
                    continue
                grp = groups[key]
                n = len(grp)
                plateaus = [r.get("plateau", float("nan")) for r in grp]
                finals   = [r.get("final_surprise", float("nan")) for r in grp]
                drops    = [r.get("drop", float("nan")) for r in grp]
                alarms   = sum(r.get("alarm_events", 0) for r in grp)
                accepted = sum(r.get("growth_accepted", 0) for r in grp)
                reverted = sum(r.get("growth_reverted", 0) for r in grp)
                comps_used   = [r.get("comps_used", float("nan")) for r in grp
                                if "comps_used" in r]
                comps_needed = grp[0].get("comps_needed", "N/A") if grp else "N/A"
                lines.append(
                    f"| {world} | {mech} | {n} "
                    f"| {_mean_or_na(plateaus)} | {_mean_or_na(finals)} "
                    f"| {_mean_or_na(drops)} "
                    f"| {alarms} | {accepted} | {reverted} "
                    f"| {_mean_or_na(comps_used)} | {comps_needed} |"
                )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Confusion matrix scaffold (4×4: world type × response type)
# ---------------------------------------------------------------------------

# World types: A learnable / B noisy / C aliased / D nonstationary
# Response categories (how the mechanism responded):
#   - suppressed: no alarms fired
#   - alarmed: ≥1 alarm fired
#   - grew:     ≥1 component accepted
#   - regressed: ceiling post-growth (final_ceiling_events > 0 after accepted > 0)

def _classify_response(row: dict) -> str:
    alarms = int(row.get("alarm_events", 0))
    accepted = int(row.get("growth_accepted", 0))
    ceil_ev = int(row.get("final_ceiling_events", 0))
    if alarms == 0:
        return "suppressed"
    if accepted == 0:
        return "alarmed_no_grow"
    if ceil_ev > 0:
        return "grew_then_ceiling"
    return "grew_clean"


_RESPONSE_LABELS = ["suppressed", "alarmed_no_grow", "grew_then_ceiling", "grew_clean"]


def build_confusion_matrix(rows: list[dict]) -> str:
    """Build a world x response confusion matrix scaffold as markdown text.

    Rows = world type, columns = mechanism response category.
    Cells = count of (world, response) pairs across all mechanisms and seeds.
    Cells where oracle rows exist show oracle comparison; pending otherwise.
    """
    # Count (world, response) per mechanism
    # Structure: {world: {response: {mechanism: count}}}
    matrix: dict[str, dict[str, dict[str, int]]] = {
        w: {r: {} for r in _RESPONSE_LABELS} for w in _WORLD_ORDER
    }

    for row in rows:
        world = row.get("world", "?")
        if world not in matrix:
            continue
        resp = _classify_response(row)
        mech = row.get("mechanism", "?")
        matrix[world][resp][mech] = matrix[world][resp].get(mech, 0) + 1

    lines = [
        "## World × Response Confusion Matrix",
        "",
        "Rows = world type. Columns = response category (across all mechanisms/seeds).",
        "Cells = total run-count with that (world, response) pair.",
        "",
        "| world | suppressed | alarmed_no_grow | grew_then_ceiling | grew_clean |",
        "|-------|------------|-----------------|-------------------|------------|",
    ]
    for world in _WORLD_ORDER:
        cells = []
        for resp in _RESPONSE_LABELS:
            total = sum(matrix[world][resp].values())
            mechs = sorted(matrix[world][resp].keys())
            detail = " + ".join(f"{m}:{n}" for m, n in
                                sorted(matrix[world][resp].items())) if mechs else "0"
            cells.append(f"{total} ({detail})")
        lines.append(f"| {world} | " + " | ".join(cells) + " |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Nats-regret vs oracle table
# ---------------------------------------------------------------------------

def build_regret_table(rows: list[dict]) -> str:
    """Nats-regret vs oracle table.

    Regret = mean_final_surprise(mechanism) - mean_final_surprise(oracle)
    for each (world, mechanism) pair.
    Cells where oracle rows are absent are marked 'pending'.
    """
    # Index oracle rows by (world, convention)
    oracle_fs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("mechanism") == "oracle":
            world = row.get("world", "?")
            conv = row.get("convention", "normalized")
            fs = row.get("final_surprise", float("nan"))
            if not (isinstance(fs, float) and math.isnan(fs)):
                oracle_fs[(world, conv)].append(fs)

    oracle_mean: dict[tuple[str, str], float] = {
        k: float(np.mean(v)) for k, v in oracle_fs.items() if v
    }

    groups = _group_rows(rows)

    lines = [
        "## Nats-Regret vs Oracle",
        "",
        "regret = mean_final_surprise(mechanism) - mean_final_surprise(oracle).",
        "Positive regret = mechanism is worse than oracle (higher surprise = less efficient).",
        "'pending' = oracle rows not yet available for this world/convention.",
        "",
        "| world | mechanism | mean_final | oracle_final | regret |",
        "|-------|-----------|------------|--------------|--------|",
    ]

    for world in _WORLD_ORDER:
        for mech in _MECH_ORDER:
            if mech == "oracle":
                continue
            conv = "normalized"
            key = (world, mech, conv)
            if key not in groups:
                continue
            grp = groups[key]
            finals = [r.get("final_surprise", float("nan")) for r in grp]
            finals_clean = [f for f in finals if not (isinstance(f, float) and math.isnan(f))]
            if not finals_clean:
                continue
            mf = float(np.mean(finals_clean))
            oracle_key = (world, conv)
            if oracle_key in oracle_mean:
                of = oracle_mean[oracle_key]
                regret_str = f"{mf - of:+.4f}"
                oracle_str = f"{of:.4f}"
            else:
                regret_str = "pending"
                oracle_str = "pending"
            lines.append(
                f"| {world} | {mech} | {mf:.4f} | {oracle_str} | {regret_str} |"
            )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structure-economy table
# ---------------------------------------------------------------------------

def build_structure_economy_table(rows: list[dict]) -> str:
    """Structure-economy: components used vs needed (from ground truth).

    Only rows that carry comps_used / comps_needed (T16+ rows) are shown.
    """
    groups = _group_rows(rows)

    lines = [
        "## Structure Economy",
        "",
        "comps_used = total mixture components across all colors at run end.",
        "comps_needed = sum(true_K_per_color) from ground truth (worlds.ground_truth).",
        "economy = comps_used / comps_needed (< 1: under-capacity; > 1: over-capacity).",
        "",
        "| world | mechanism | comps_used | comps_needed | economy |",
        "|-------|-----------|------------|--------------|---------|",
    ]

    for world in _WORLD_ORDER:
        for mech in _MECH_ORDER:
            conv = "normalized"
            key = (world, mech, conv)
            if key not in groups:
                continue
            grp = [r for r in groups[key] if "comps_used" in r]
            if not grp:
                continue
            used_vals = [r["comps_used"] for r in grp]
            needed_vals = [r["comps_needed"] for r in grp]
            mean_used = float(np.mean(used_vals))
            mean_needed = float(np.mean(needed_vals))
            economy = mean_used / mean_needed if mean_needed > 0 else float("nan")
            lines.append(
                f"| {world} | {mech} | {mean_used:.1f} | {mean_needed:.1f} "
                f"| {economy:.3f} |"
            )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Aggregate summary.md builder
# ---------------------------------------------------------------------------

def write_aggregate_summary(rows: list[dict], out_path: Path) -> None:
    """Write (or overwrite) the aggregate summary with all tables."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    sections = [
        "# Worldview Benchmark — Aggregate Summary (T17)",
        "",
        f"Generated: {now}",
        f"Total rows loaded: {len(rows)}",
        "",
        build_summary_table(rows),
        "",
        build_confusion_matrix(rows),
        "",
        build_regret_table(rows),
        "",
        build_structure_economy_table(rows),
        "",
        "---",
        "Generated by experiments/bench_worldview/plots.py (T17, rigor-fairness-upgrade spec).",
        "",
    ]
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"[plots] wrote aggregate summary → {out_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all(rows: list[dict] | None = None, verbose: bool = True) -> dict[str, Any]:
    """Generate all T17 artifacts.

    Parameters
    ----------
    rows : list of row dicts, or None to load from outputs/
    verbose : print status messages

    Returns
    -------
    dict with keys: trace_pngs (list[Path]), aggregate_summary_path (Path),
    n_rows (int), mpl_available (bool).
    """
    if rows is None:
        rows = load_all_rows()

    if verbose:
        print(f"[plots] loaded {len(rows)} rows from {_OUTPUTS_DIR}")

    # 1. Per-run surprise traces
    trace_pngs: list[Path] = []
    if _MPL_AVAILABLE:
        trace_pngs = generate_surprise_traces(rows)
        if verbose:
            print(f"[plots] wrote {len(trace_pngs)} surprise trace PNGs")
    else:
        if verbose:
            print("[plots] matplotlib unavailable — skipping PNGs")

    # 2. Aggregate summary with all tables
    agg_path = _OUTPUTS_DIR / "aggregate_summary.md"
    write_aggregate_summary(rows, agg_path)

    return {
        "trace_pngs": trace_pngs,
        "aggregate_summary_path": agg_path,
        "n_rows": len(rows),
        "mpl_available": _MPL_AVAILABLE,
    }


if __name__ == "__main__":
    result = generate_all(verbose=True)
    print(f"[plots] done — {result['n_rows']} rows, "
          f"{len(result['trace_pngs'])} PNGs, "
          f"summary at {result['aggregate_summary_path']}")
