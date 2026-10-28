"""Descriptive analysis of Exp283; no new predictions or replacement verdict.

Recompute the frozen scientific counts, aggregate full ledgers, retain the
post-hoc living/newborn endpoint comparison as sensitivity only, and compare
the IO-guard rerun against the archived first run. Does not simulate a world.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from math import fsum
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/outputs"
PAPER = ROOT / "docs/research/boundary-transmission"


def main():
    raw = json.loads((OUT / "exp283.json").read_text())
    first = json.loads((OUT / "exp283_initial_archive/exp283.json").read_text())
    old_summaries = {(s["arm"], s["seed"]): s for s in first["summaries"]}
    assert len(raw["summaries"]) == len(old_summaries) == 48
    records = {}
    all_rows = 0
    max_error = 0.
    for relative in raw["trajectory_files"]:
        path = OUT / relative
        with gzip.open(path, "rt") as stream:
            payload = json.load(stream)
        s = payload["summary"]
        key = s["arm"], s["seed"]
        assert key not in records
        with gzip.open(OUT / "exp283_initial_archive" / relative, "rt") as stream:
            original = json.load(stream)
        assert payload["columns"] == original["columns"]
        assert payload["trajectory"] == original["trajectory"]
        for field, value in s.items():
            if field != "runtime_s":
                assert value == old_summaries[key][field], (key, field)
        rows = [dict(zip(payload["columns"], r)) for r in payload["trajectory"]]
        assert len(rows) == 3000 and s["reached_horizon"]
        for row in rows:
            assert row["n_after"] == row["n_before"] + row["births"] - row["deaths"]
            error = abs(row["mean_after"] - row["mean_before"] - row["selection"] - row["transmission"])
            max_error = max(max_error, error)
            assert error < 1e-10
        assert abs(fsum(r["selection"] for r in rows) - s["cumulative_selection"]) < 1e-12
        assert abs(fsum(r["transmission"] for r in rows) - s["cumulative_transmission"]) < 1e-12
        assert sum(r["births"] for r in rows) == s["births"]
        all_rows += len(rows)
        records[key] = s
    for source, expected in raw["source_hashes"].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == expected, source

    metrics = ["final_living_mean", "final_newborn_mean_last500", "cumulative_selection",
               "cumulative_transmission", "cumulative_expected_transmission",
               "cumulative_transmission_minus_expected", "births", "final_population"]
    report = {"scope": "Frozen criterion recomputation plus explicitly post-hoc descriptive sensitivity; not a new confirmation",
              "rows_checked": all_rows, "max_identity_error": max_error,
              "rerun_exact_numeric_match": True, "rerun_excludes": ["runtime", "source_hashes", "gzip_header"],
              "final_run_runtime_s": raw["runtime_s"], "first_run_runtime_s": first["runtime_s"], "arms": {}}
    for arm in ["NEUTRAL", "COST_ONLY", "THERMAL"]:
        group = [records[arm, seed] for seed in range(9500, 9516)]
        report["arms"][arm] = {
            k: {"mean": float(np.mean([s[k] for s in group])),
                "min": min(s[k] for s in group), "max": max(s[k] for s in group)} for k in metrics}
        report["arms"][arm].update(
            completes=sum(s["reached_horizon"] for s in group),
            living_above_bar=sum(s["final_living_mean"] >= .02 for s in group),
            newborn_above_bar_posthoc=sum(s["final_newborn_mean_last500"] >= .02 for s in group),
            negative_differential=sum(s["cumulative_selection"] < 0 for s in group),
            differential_below_bar=sum(s["cumulative_selection"] <= -.02 for s in group))
    cost = [records["COST_ONLY", seed] for seed in range(9500, 9516)]
    cost_count = sum(s["reached_horizon"] and s["final_living_mean"] >= .02
                     and s["cumulative_selection"] <= -.02 for s in cost)
    neutral_count = report["arms"]["NEUTRAL"]["living_above_bar"]
    report["frozen_counts"] = {"cost_joint": cost_count, "neutral": neutral_count, "required": 12,
                               "verdict": "POSITIVE" if cost_count >= 12 and neutral_count >= 12 else "NEGATIVE"}
    thermal, cost_group = report["arms"]["THERMAL"], report["arms"]["COST_ONLY"]
    report["thermal_minus_cost_posthoc"] = {k: thermal[k]["mean"]-cost_group[k]["mean"]
                                            for k in ["final_living_mean", "cumulative_selection", "cumulative_transmission"]}
    (OUT / "exp283_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    packet = {"instructions": "Apply only the frozen docstring; ignore printed verdict/aggregate claims. Recompute from raw ledgers. No repository bootstrap or parent interpretation provided.",
              "predeclaration": raw["predeclaration"],
              "raw_ledger_files": [str(OUT / p) for p in raw["trajectory_files"]]}
    (OUT / "exp283_verifier_packet.json").write_text(json.dumps(packet, indent=2) + "\n")

    # Inline pgfplots data make the manuscript source self-contained.
    lines = [r"\begin{tikzpicture}", r"\begin{axis}[width=0.94\linewidth,height=5.5cm,",
             r"xmin=-.55,xmax=.06,ymin=0,ymax=.185,grid=major,grid style={gray!20},",
             r"xlabel={Cumulative realized selection differential $\sum_t S_t$},",
             r"ylabel={Final living trait mean $\bar h_H$},",
             r"scaled ticks=false,xtick={-.5,-.4,-.3,-.2,-.1,0},ytick={0,.05,.10,.15},",
             r"tick label style={font=\small,/pgf/number format/fixed,/pgf/number format/precision=2},label style={font=\small},",
             r"legend style={at={(0.02,0.98)},anchor=north west,font=\small,draw=none,fill=white},",
             r"legend cell align=left]",
             r"\addplot[gray,dashed,forget plot] coordinates {(-.55,.02) (.06,.02)};",
             r"\addplot[gray,thin,forget plot] coordinates {(0,0) (0,.185)};"]
    for arm, color, mark, label in [("NEUTRAL", "gray", "triangle*", "Neutral"),
                                    ("COST_ONLY", "red!75!black", "*", "Cost-only"),
                                    ("THERMAL", "blue!70!black", "square*", "Thermal (descriptive)")]:
        coords = " ".join(f"({records[arm,s]['cumulative_selection']:.12f},{records[arm,s]['final_living_mean']:.12f})" for s in range(9500,9516))
        lines.append(r"\addplot[only marks,mark="+mark+",mark size=1.8pt,color="+color+"] coordinates {"+coords+"};")
        lines.append(r"\addlegendentry{"+label+"}")
    lines += [r"\end{axis}", r"\end{tikzpicture}"]
    (PAPER / "scatter-data.tex").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
