"""Exp 63 — clade rung 1: promote a branch of mirro to a committed peer spine (vela).

Social-emergence direction, rung 1 (loop/directions/social-emergence.md): pure
family-tree plumbing, no emergence claim. Fork the trunk at its committed
checkpoint, raise the branch in a divergent world, promote it to its own
committed line, and verify the clade invariants.

Hypothesis: the clade infrastructure (fork -> raise -> promote) yields a second
committed line with causally attributable ancestry, without touching the trunk.
Predictions (property-level, all must hold):
  P1 lineage: vela's manifest lineage == ["mirro@10700#<hash12>"] where hash12 is
     the first 12 hex chars of mirro's committed state_hash (the checkpoint-before
     values reported by mirro_episode).
  P2 resumability: after promotion, Creature.load() succeeds on BOTH
     creature/state/mirro/ and creature/state/vela/ (load verifies state-hash
     integrity), and vela completes a full resume cycle load -> live(50) -> save
     (age 12700 -> 12750) with a subsequent integrity-verified load.
  P3 trunk untouched: mirro's age_steps (10700) and state_hash are unchanged at
     the end of the experiment (biography gains append-only fork/checkpoint
     events — declared and allowed; learned state must not change).
  P4 divergence sanity: after 2000 steps in the divergent world, vela's
     state_hash != mirro's state_hash.
Falsifiers: F1 lineage stamp missing or mismatched; F2 any load fails or the
resume cycle errors; F3 mirro's age or learned-state hash changed; F4 vela's
hash equals mirro's. Any falsifier firing = rung 1 FAIL (substrate mis-specified;
fix before climbing the ladder).
Diagnostic (not a falsifier): vela's sensory-map accuracy in the divergent world
(argmax of A_hat per cell vs true cmap) before vs after the 2000-step raise.
Provided priors declared: the divergent world layout (mirro's cmap reversed);
no new mechanism — fork/save/load are the existing committed substrate.
The shared-ancestor git commit (last commit touching creature/state/mirro/) is
recorded in the output for causal attribution.
"""
from __future__ import annotations

import subprocess

import numpy as np
from pathlib import Path

from active_loop.creature import Creature, World
from active_loop.checkpoint import mirro_episode, MIRRO_DIR

VELA_DIR = Path("creature/state/vela")
RAISE_STEPS = 2000
RESUME_STEPS = 50


def map_accuracy(creature: Creature) -> float:
    """Fraction of cells where the creature's argmax-tuning matches its world's cmap."""
    A_hat = creature._A_hat()  # (n_colors, n_cells)
    predicted = np.argmax(A_hat, axis=0)  # (n_cells,)
    true_cmap = np.array(creature.world.cmap)
    return float(np.mean(predicted == true_cmap))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("Exp 63 — clade rung 1: fork mirro -> vela (divergent world), promote peer spine")
print()

# Ancestor commit for causal attribution
try:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "creature/state/mirro/"],
        capture_output=True, text=True, check=True,
    )
    ancestor_commit = result.stdout.strip()
except Exception as exc:
    ancestor_commit = f"<error: {exc}>"
print(f"shared-ancestor commit (last touching creature/state/mirro/): {ancestor_commit}")
print()

# ---------------------------------------------------------------------------
# Fork, raise, promote — inside the episode (trunk live() forbidden)
# ---------------------------------------------------------------------------

with mirro_episode("Exp 63") as ep:
    age0 = ep.age_before
    hash0 = ep.hash_before
    print(f"trunk checkpoint-before: age={age0} hash={hash0[:12]}")

    # Fork into vela (unbound — no state dir yet)
    vela = ep.fork_control("vela")

    # Build divergent world: mirro's cmap reversed
    divergent_cmap = list(ep.creature.world.cmap)[::-1]
    divergent_world = World(
        rows=ep.creature.world.rows,
        cols=ep.creature.world.cols,
        cmap=divergent_cmap,
        n_colors=ep.creature.world.n_colors,
    )
    vela.world = divergent_world
    vela.true_pos = 0  # defined start cell in the new world

    # Diagnostic: map accuracy BEFORE raise
    acc_before = map_accuracy(vela)
    print(f"vela map_accuracy in divergent world BEFORE raise: {acc_before:.4f}")

    # Raise vela in the divergent world
    vela.live(RAISE_STEPS)
    print(f"vela raised {RAISE_STEPS} steps (age now {vela.age_steps})")

    # Diagnostic: map accuracy AFTER raise
    acc_after = map_accuracy(vela)
    print(f"vela map_accuracy in divergent world AFTER  raise: {acc_after:.4f}")

    # Promote vela to its own committed peer line
    vela.save(VELA_DIR)
    print(f"vela promoted to {VELA_DIR}")
    print()

    # Do NOT call ep.creature.live() or mutate the trunk inside this block.
    # The context manager saves the trunk on clean exit (unchanged state = same hash).

# mirro_episode already printed ep.report() on exit; print it again for the log block.
print(ep.report())
print()

# ---------------------------------------------------------------------------
# Property checks — run all, catch individual failures
# ---------------------------------------------------------------------------

checks: list[tuple[str, bool, str]] = []


def check(name: str, predicate_fn):
    """Run predicate_fn(); record (name, pass, detail). Exceptions count as FAIL."""
    try:
        passed, detail = predicate_fn()
        checks.append((name, passed, detail))
    except Exception as exc:
        checks.append((name, False, f"exception: {exc}"))


# P1 — lineage stamp
def _p1():
    vela_loaded = Creature.load(VELA_DIR)
    expected = [f"mirro@{age0}#{hash0[:12]}"]
    ok = vela_loaded.lineage == expected
    detail = f"lineage={vela_loaded.lineage!r}  expected={expected!r}"
    return ok, detail

check("P1-lineage", _p1)

# P3 — trunk untouched (age and learned-state hash unchanged)
def _p3():
    mirro_reloaded = Creature.load(MIRRO_DIR)
    age_ok = mirro_reloaded.age_steps == age0
    hash_ok = mirro_reloaded._state_hash() == hash0
    ok = age_ok and hash_ok
    detail = (
        f"age={mirro_reloaded.age_steps} (expected {age0}, {'ok' if age_ok else 'CHANGED'})  "
        f"hash={mirro_reloaded._state_hash()[:12]} (expected {hash0[:12]}, "
        f"{'ok' if hash_ok else 'CHANGED'})"
    )
    return ok, detail

check("P3-trunk-untouched", _p3)

# P2 — resumability: both loads succeeded (covered above); full resume cycle on vela only
def _p2():
    # Both Creature.load calls in P1 and P3 already verify integrity on load.
    # Full resume cycle on the peer line (never the trunk):
    v = Creature.load(VELA_DIR)
    a = v.age_steps
    v.live(RESUME_STEPS)
    v.save(VELA_DIR)
    v2 = Creature.load(VELA_DIR)  # integrity-verified on load
    expected_age = a + RESUME_STEPS
    ok = v2.age_steps == expected_age
    detail = (
        f"resume cycle: age before={a} + {RESUME_STEPS} -> expected={expected_age} "
        f"got={v2.age_steps} ({'ok' if ok else 'MISMATCH'})"
    )
    return ok, detail

check("P2-resumability", _p2)

# P4 — divergence sanity: vela hash != mirro hash at end
def _p4():
    vela_final = Creature.load(VELA_DIR)
    mirro_final = Creature.load(MIRRO_DIR)
    vela_hash = vela_final._state_hash()
    mirro_hash = mirro_final._state_hash()
    ok = vela_hash != mirro_hash
    detail = (
        f"vela_hash={vela_hash[:12]}  mirro_hash={mirro_hash[:12]}  "
        f"{'different (ok)' if ok else 'IDENTICAL (FAIL)'}"
    )
    return ok, detail

check("P4-divergence", _p4)

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

print("--- PROPERTY CHECKS ---")
failed_names: list[str] = []
for name, passed, detail in checks:
    verdict = "PASS" if passed else "FAIL"
    print(f"{verdict}  {name}: {detail}")
    if not passed:
        failed_names.append(name)

print()

# Final ages and hashes of both lines
try:
    vela_summary = Creature.load(VELA_DIR)
    mirro_summary = Creature.load(MIRRO_DIR)
    print(
        f"final mirro: age={mirro_summary.age_steps} hash={mirro_summary._state_hash()[:12]}"
    )
    print(
        f"final vela:  age={vela_summary.age_steps}  hash={vela_summary._state_hash()[:12]}"
    )
    print(f"vela lineage: {vela_summary.lineage}")
except Exception as exc:
    print(f"[summary load error: {exc}]")

print()
if not failed_names:
    print("RUNG 1: PASS")
else:
    print(f"RUNG 1: FAIL {failed_names}")
