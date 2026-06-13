"""ecology/recording.py — Structured output writers for Exp 194+.

All writers produce deterministic output (no wall-clock, no set iteration).
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any


def write_events_jsonl(path: str, events: list[dict[str, Any]]) -> None:
    """Write events as newline-delimited JSON (one event per line)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, sort_keys=True, ensure_ascii=True) + "\n")


def write_population_summary(path: str, summary: dict[str, Any]) -> None:
    """Write the run summary as a single JSON object."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=True)


def write_lineage_tree(
    path: str,
    creatures: list[Any],  # list of Creature objects
) -> None:
    """Write parent → children adjacency as JSON.

    Format: {"creature_id": {"parent_id": int|null, "generation": int, "children": [...]}}
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tree: dict[str, Any] = {}
    for c in creatures:
        cid = str(c.creature_id)
        tree[cid] = {
            "parent_id": c.parent_id,
            "generation": c.generation,
            "lineage_root": c.lineage_root,
            "offspring_count": c.phenotype.offspring_count,
            "children": [],
        }
    # Populate children lists
    for c in creatures:
        if c.parent_id is not None:
            pid = str(c.parent_id)
            if pid in tree:
                tree[pid]["children"].append(c.creature_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, sort_keys=True, ensure_ascii=True)


def write_trait_distribution_csv(
    path: str,
    rows: list[dict[str, Any]],
) -> None:
    """Write trait distribution data as CSV.

    Each row should be a dict with at least: generation, trait, mean, std, count.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("generation,trait,mean,std,count\n")
        return
    fieldnames = sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_verdict(path: str, verdict_dict: dict[str, Any]) -> None:
    """Write verdict JSON — machine-readable for the verifier loop."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(verdict_dict, f, indent=2, sort_keys=True, ensure_ascii=True)
