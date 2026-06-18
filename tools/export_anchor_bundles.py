"""Export one curated milestone ExperimentBundle per coalescence card.

Run from the repo root:
    PYTHONPATH=. uv run --python .venv python tools/export_anchor_bundles.py

Each entry below is a load-bearing ANCHOR experiment for a seeded mechanism/boundary card —
materialising the experiments -> evidence -> bundles -> cards chain so every card's
source_experiments resolves to a citable bundle object. Bundles are exported at the HONESTLY
achievable level from the inventory (the exporter refuses to over-claim) and reference original
committed files in place (never copy or mutate raw data). Re-running overwrites the manifests
with a fresh created_at/repo_commit; the referenced evidence is unchanged.
"""
from __future__ import annotations

from pathlib import Path

from active_loop.coalescence.export import export_bundle
from active_loop.coalescence.inventory import build_inventory

ROOT = Path(__file__).resolve().parent.parent

# (experiment n, the card it anchors). Level is taken from the inventory (achievable max).
ANCHORS = [
    (31, "recipe-symmetry-breaking-v0 / disembodied-stream-collapse-v0 (the anchor law: A+B from noise collapses)"),
    (35, "recipe-symmetry-breaking-v0 (answers in words; content self-formed, labels taught)"),
    (220, "functional-valence-dyad-v0 (precision schedule: first reliable genuine discrimination)"),
    (154, "online-structure-growth-v0 (the growth wall fell under normalized evaluation)"),
    (168, "meta-calibration-n3-v0 (the ratchet law; N3 agency over metacognition)"),
    (188, "identity-n4-commitment-v0 (REG-TB: the N4 crack closed at fixed-L)"),
    (205, "costly-sensing-wall-v0 (the fitness valley is the sole barrier)"),
    (211, "active-sensing-benefit-wall-v0 (uncertainty-gated probing; benefit ceiling ~0)"),
    (209, "hidden-state-memory-boundary-v0 (continuous belief-persistence; wall not a granularity artifact)"),
]


def main() -> int:
    inv = {e["n"]: e for e in build_inventory(str(ROOT))["experiments"]}
    for n, anchor_for in ANCHORS:
        rec = inv[n]
        level = rec["backfill_level_possible"]
        eid = rec["experiment_id"]
        out = ROOT / "experiment_bundles" / eid
        manifest = export_bundle(eid, level, out, repo_root=str(ROOT))
        print(f"{eid:>7}  {level:<18}  anchors {anchor_for}")
        # leave a breadcrumb of which card this bundle anchors (honest provenance note)
        readme = out / "README.md"
        if readme.exists():
            text = readme.read_text(encoding="utf-8")
            note = f"\n\n## Anchors\n\nLoad-bearing evidence for: {anchor_for}\n"
            if "## Anchors" not in text:
                readme.write_text(text + note, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
