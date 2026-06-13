"""ecology/run.py — run_scenario and determinism_check helpers.

run_scenario:
  Builds and runs an Ecology for the named scenario+seed, writes all five
  structured output files, returns a summary dict.

determinism_check:
  Runs the scenario twice fresh and compares events_hash.  Returns True iff
  the two hashes match.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

import numpy as np

from ecology.engine import Ecology
from ecology.scenarios import SCENARIOS, FOUNDER
from ecology.recording import (
    write_events_jsonl,
    write_population_summary,
    write_lineage_tree,
    write_trait_distribution_csv,
    write_verdict,
)


def run_scenario(name: str, seed: int, outdir: str) -> dict[str, Any]:
    """Run scenario `name` with `seed`, write structured outputs, return summary."""
    cfg = SCENARIOS[name]
    eco = Ecology(cfg, seed=seed)
    summary = eco.run()

    run_dir = os.path.join(outdir, f"{name}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    # 1. Events JSONL
    write_events_jsonl(
        os.path.join(run_dir, "events.jsonl"),
        eco.events,
    )

    # 2. Population summary
    write_population_summary(
        os.path.join(run_dir, "summary.json"),
        summary,
    )

    # 3. Lineage tree
    write_lineage_tree(
        os.path.join(run_dir, "lineage.json"),
        eco._creatures,
    )

    # 4. Trait distribution CSV (by generation)
    trait_rows = _build_trait_rows(eco._creatures)
    write_trait_distribution_csv(
        os.path.join(run_dir, "traits.csv"),
        trait_rows,
    )

    # 5. Placeholder verdict (per-run; top-level verdict written by the experiment)
    write_verdict(
        os.path.join(run_dir, "verdict.json"),
        {"scenario": name, "seed": seed, "summary": summary},
    )

    return summary


def determinism_check(name: str, seed: int) -> bool:
    """Run the scenario twice; return True iff events_hash matches."""
    cfg = SCENARIOS[name]
    eco1 = Ecology(cfg, seed=seed)
    eco1.run()
    hash1 = eco1.events_hash()

    eco2 = Ecology(cfg, seed=seed)
    eco2.run()
    hash2 = eco2.events_hash()

    return hash1 == hash2


def _build_trait_rows(creatures: list[Any]) -> list[dict[str, Any]]:
    """Build per-generation trait statistics rows for CSV."""
    from dataclasses import asdict as _asdict
    gen_map: dict[int, list[Any]] = {}
    for c in creatures:
        g = c.generation
        gen_map.setdefault(g, []).append(c)

    rows = []
    for gen in sorted(gen_map.keys()):
        cs = gen_map[gen]
        trait_d = _asdict(cs[0].genotype)
        for trait_name in trait_d:
            vals = [getattr(c.genotype, trait_name) for c in cs]
            rows.append({
                "generation": gen,
                "trait": trait_name,
                "mean": round(float(np.mean(vals)), 8),
                "std": round(float(np.std(vals)), 8),
                "count": len(vals),
            })
    return rows
