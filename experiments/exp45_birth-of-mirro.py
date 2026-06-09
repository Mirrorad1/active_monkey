"""Exp 45 — BIRTH of mirro: the persistent creature's first life.

Hypothesis: a persistent creature born once with the innate movement anchor and
living one continuous 900-step life self-organizes its sensory map, and its
committed snapshot is exactly resumable.

Prediction: map accuracy >= 8/9 cells at age 900 (Exp 21/36 predict 9/9);
save->load roundtrip state_hash identical; localization near 0 bits.

Falsifier: accuracy < 8/9 by age 1800, or roundtrip hash mismatch.

Note: mechanism = consolidation of Exp 21 (place-cell self-organization, pure
numpy, 3x3 world, aliased cmap).  The new element is the program-level persistent,
committed, resumable life + biography.  This is POSITIVE-SINGLE, not BREAKTHROUGH.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# World configuration — Exp 21's 3x3 aliased colormap
# ---------------------------------------------------------------------------
WORLD = World(rows=3, cols=3, cmap=[0, 1, 2, 1, 2, 0, 2, 0, 1], n_colors=3)
STATE_DIR = Path(__file__).resolve().parent.parent / "creature" / "state"


def run_birth(seed: int, name: str, save: bool = False) -> dict:
    """Birth and run one creature for 900 steps.  Returns diagnostics."""
    c = Creature.birth(name, WORLD, seed=seed)
    if save:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        c.save(STATE_DIR / name)  # bind state_dir so biography is written

    c.live(900)
    if save:
        c.save(STATE_DIR / name)

    learned = c.sensory_map()
    true = list(WORLD.cmap)
    correct = sum(l == t for l, t in zip(learned, true))

    return {
        "name": name,
        "seed": seed,
        "age": c.age_steps,
        "map_accuracy": c.map_accuracy(),
        "correct_cells": correct,
        "total_cells": WORLD.n_cells,
        "localize_bits": c.localize_bits(),
        "favorite": c.favorite(),
        "conviction": c.conviction(),
        "state_hash": c._state_hash(),
        "creature": c,
    }


def check_roundtrip(state_path: Path) -> tuple[str, str, bool]:
    """Save->load roundtrip: verify state_hash is identical."""
    c_loaded = Creature.load(state_path)
    loaded_hash = c_loaded._state_hash()
    manifest_hash = __import__("json").loads((state_path / "manifest.json").read_text())["state_hash"]
    match = loaded_hash == manifest_hash
    return loaded_hash, manifest_hash, match


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("=" * 60)
print("Exp 45 — BIRTH of mirro: the persistent creature's first life")
print("=" * 60)
print()
print(f"World: {WORLD.rows}x{WORLD.cols}, cmap={WORLD.cmap}, n_colors={WORLD.n_colors}")
print(f"Seed: 7  |  Steps: 900")
print()

# --- (a) Birth mirro and live 900 steps ---
print("--- Birth mirro (seed=7) and live 900 steps ---")
result = run_birth(seed=7, name="mirro", save=True)

print(f"age:           {result['age']}")
print(f"learned map:   {result['creature'].sensory_map()}")
print(f"true cmap:     {list(WORLD.cmap)}")
print(f"map_accuracy:  {result['map_accuracy']:.4f}  ({result['correct_cells']}/{result['total_cells']} cells)")
print(f"localize_bits: {result['localize_bits']:.4f} bits")
print(f"favorite:      color-{result['favorite']}  conviction={result['conviction']:.4f}")
print(f"state_hash:    {result['state_hash']}")
print()

# --- Predeclared threshold check ---
threshold_met = result['correct_cells'] >= 8
print(f"Predeclared threshold (>= 8/9): {'PASS' if threshold_met else 'FAIL'}")
print()

# --- (b) Seed-robustness control: seeds 8 and 9 ---
print("--- Seed-robustness control (seeds 8, 9) — disposable, not saved ---")
seed8 = run_birth(seed=8, name="mirro_ctrl_8", save=False)
seed9 = run_birth(seed=9, name="mirro_ctrl_9", save=False)

print(f"seed 8: accuracy={seed8['map_accuracy']:.4f} ({seed8['correct_cells']}/{seed8['total_cells']})")
print(f"seed 9: accuracy={seed9['map_accuracy']:.4f} ({seed9['correct_cells']}/{seed9['total_cells']})")

seeds_passing = sum([
    result['correct_cells'] >= 8,
    seed8['correct_cells'] >= 8,
    seed9['correct_cells'] >= 8,
])
print(f"Seeds passing >= 8/9: {seeds_passing}/3")
property_met = seeds_passing >= 2
print(f"Property (>= 2 of 3 reach 8/9): {'PASS' if property_met else 'FAIL'}")
print()

# --- (c) Save -> load roundtrip ---
print("--- Save/load roundtrip: state_hash integrity ---")
state_path = STATE_DIR / "mirro"
loaded_hash, manifest_hash, roundtrip_ok = check_roundtrip(state_path)
print(f"manifest state_hash:  {manifest_hash[:32]}...")
print(f"reloaded state_hash:  {loaded_hash[:32]}...")
print(f"roundtrip match:      {'PASS — identical' if roundtrip_ok else 'FAIL — MISMATCH'}")
print()

# --- Summary ---
print("=" * 60)
print("SUMMARY")
print("=" * 60)
verdict = "POSITIVE-SINGLE" if (threshold_met and property_met and roundtrip_ok) else "NEGATIVE"
print(f"Verdict:               {verdict}")
print(f"Map accuracy (seed 7): {result['map_accuracy']:.4f} ({result['correct_cells']}/{result['total_cells']})")
print(f"Threshold (>= 8/9):    {'PASS' if threshold_met else 'FAIL'}")
print(f"Property (>= 2/3 seeds >= 8/9): {'PASS' if property_met else 'FAIL'} ({seeds_passing}/3 passing)")
print(f"Roundtrip hash match:  {'PASS' if roundtrip_ok else 'FAIL'}")
print(f"Mechanism:             consolidation of Exp 21 (place-cell self-organization)")
print(f"New element:           program-level persistent committed resumable life + biography")
print()
if not threshold_met:
    print("FALSIFIER HIT: map accuracy < 8/9 at age 900 — stopping, not tweaking.")
    sys.exit(1)
if not property_met:
    print("FALSIFIER HIT: fewer than 2/3 seeds reach >= 8/9 — stopping, not tweaking.")
    sys.exit(1)
if not roundtrip_ok:
    print("FALSIFIER HIT: roundtrip hash mismatch — stopping, not tweaking.")
    sys.exit(1)
print("mirro is born. snapshot committed to creature/state/mirro/")
print(f"state_hash (short):    {result['state_hash'][:16]}")
