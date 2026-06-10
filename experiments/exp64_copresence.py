"""Exp 64 — clade rung 2: shared-world co-presence must not break solo competence.

Social-emergence direction, rung 2 (loop/directions/social-emergence.md): the minimal
multi-agent substrate. Two clade-mates (forks of the committed mirro and vela lines)
are placed in ONE grid; each senses the other via a NEW `other-agent-here` binary
modality. No communication, no value coupling — perception only.

Hypothesis: coupling a learned other-agent-here likelihood multiplicatively into
place inference does NOT degrade solo competence — neither for a converged resident
(mirro-fork in its home world) nor for a relearning immigrant (vela-fork, whose map
fits the reversed world).

Design control: per (creature, seed), the action sequence comes from the same seeded
RNG in both arms, so SOLO and CO trajectories are IDENTICAL; any metric difference is
attributable to the modality coupling alone.

Predeclared predictions (property-level, both must hold):
  P1 map non-degradation: for EACH creature, in >=4/5 seeds,
     end map_accuracy(CO) >= end map_accuracy(SOLO) - 0.04  (one cell of 25).
  P2 localization non-degradation: for EACH creature, in >=4/5 seeds,
     mean post-observation belief entropy over the final 200 steps satisfies
     loc_CO <= loc_SOLO + 0.2 bits.
Validity gate (instrument, not verdict — Exp 61's lesson): every CO run must log
  >=20 co-location events (expected ~80 on 5x5); fewer = the modality never fired,
  run INVALID, redesign before any verdict.
Falsifiers: F1 = P1 fails for either creature; F2 = P2 fails for either creature.
Either firing = rung 2 FAIL: the co-presence substrate is mis-specified; fix before
climbing the ladder.
Diagnostic (not a falsifier): the learned A2 structure — does P(other-here | cell)
pick up the other's wall-clamped-walk occupancy bias (corners over-visited)?

Provided priors declared: the shared world (mirro's grid + cmap); the other-here
modality wiring (obs2 = [other in my cell]; learned Dirichlet pA2 init flat 0.1;
likelihood coupled multiplicatively into the place posterior); the harness random-walk
policy (as in live()); start positions (each creature's committed true_pos). The
spines NEVER live in this experiment: forks only, neither committed line is saved.
Value accumulation stays exactly live()'s mechanism (A1-only predictability weight);
the new modality does not enter values in this rung — one new mechanism per iteration.
"""
from __future__ import annotations

import copy
import sys

import numpy as np
from pathlib import Path

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEPS = 2000
SEEDS = [0, 1, 2, 3, 4]
TAIL = 200  # final-window step count for localization metric
MIRRO_DIR = Path("creature/state/mirro")
VELA_DIR = Path("creature/state/vela")

# ---------------------------------------------------------------------------
# Load committed spines (read-only — NEVER call .live() or .save() on these)
# ---------------------------------------------------------------------------

print("Exp 64 — clade rung 2: shared-world co-presence")
print()

mirro = Creature.load(MIRRO_DIR)
vela = Creature.load(VELA_DIR)

print(f"mirro: name={mirro.name!r} age={mirro.age_steps} "
      f"hash={mirro._state_hash()[:12]} true_pos={mirro.true_pos}")
print(f"vela:  name={vela.name!r}  age={vela.age_steps}  "
      f"hash={vela._state_hash()[:12]} true_pos={vela.true_pos}")
print()

# Fork templates (appends one biography event per spine — append-only, allowed)
mirro_t = mirro.fork("exp64-mirro-template")
vela_t = vela.fork("exp64-vela-template")

# Shared world: mirro's grid (the immigrant vela gets the wrong map — its pA stays)
shared = mirro.world
vela_t.world = shared  # re-point to mirro's world; pA still encodes reversed cmap

# ---------------------------------------------------------------------------
# Stepper
# ---------------------------------------------------------------------------


def run_arm(creatures, steps, rngs, use_modality):
    """Run one arm (solo or co) for `steps` steps.

    Parameters
    ----------
    creatures : list of Creature copies (1 for solo, 2 for co).
        Index 0 = mirro-fork, index 1 = vela-fork.
    steps : int
    rngs : list of np.random.Generator, one per creature (indexed by position, not arm).
    use_modality : bool — if True, couple the other-agent-here modality.

    Returns
    -------
    results : list of dicts, one per creature, with keys:
        map_acc, tail_entropy_bits, colocations (only if use_modality), pA2 (only if use_modality)
    """
    world = shared  # all creatures share this
    n_cells = world.n_cells
    B = world.transition_matrix()  # (n_cells, n_cells, 4) — precomputed once

    # Per-creature modality state (NOT stored on Creature objects)
    if use_modality:
        pA2_list = [np.full((2, n_cells), 0.1) for _ in creatures]
        colocation_counts = [0] * len(creatures)
    else:
        pA2_list = [None] * len(creatures)
        colocation_counts = [0] * len(creatures)

    # Per-creature entropy history (all steps; we'll slice the tail later)
    entropy_histories = [[] for _ in creatures]

    for step in range(steps):
        # Read all positions simultaneously at the start of the step
        positions_before = [c.true_pos for c in creatures]

        new_positions = []
        qs_updated_list = []

        for i, c in enumerate(creatures):
            obs1 = int(world.cmap[positions_before[i]])
            A1_hat = c._A_hat()
            likelihood = A1_hat[obs1, :].copy()  # shape (n_cells,)

            if use_modality:
                # Determine other agent's position (simultaneous read)
                other_idx = 1 - i  # works for 2-agent case
                other_pos = positions_before[other_idx]
                obs2 = 1 if (other_pos == positions_before[i]) else 0

                # A2_hat: normalize pA2 over axis 0 so each column sums to 1
                pA2 = pA2_list[i]
                col_sums = pA2.sum(axis=0, keepdims=True)
                col_sums = np.where(col_sums == 0, 1.0, col_sums)
                A2_hat = pA2 / col_sums  # (2, n_cells)

                likelihood = likelihood * A2_hat[obs2, :]

                if obs2 == 1:
                    colocation_counts[i] += 1

            # Belief update: qs ∝ likelihood * qs_prior
            qs_prior = c.qs
            qs_upd = likelihood * qs_prior
            denom = qs_upd.sum()
            if denom > 0:
                qs_upd = qs_upd / denom
            else:
                qs_upd = np.ones(n_cells) / n_cells

            # Dirichlet count learning for A1
            c.pA[obs1, :] += qs_upd

            # Modality learning for A2
            if use_modality:
                pA2_list[i][obs2, :] += qs_upd

            # Value accumulation — exactly live()'s mechanism (A1 only, even in CO)
            map_cell = int(np.argmax(qs_upd))
            predicted_obs_dist = A1_hat[:, map_cell]  # P(obs | map_cell) — uses pre-update A1_hat
            h_predicted = -np.sum(predicted_obs_dist * np.log(predicted_obs_dist + 1e-12))
            predictability_weight = np.exp(-h_predicted)
            c.value_counts[obs1] += predictability_weight

            # Per-step metric: entropy in BITS of post-observation belief
            p = qs_upd  # already normalized
            ent_bits = float(-np.sum(p * np.log2(p + 1e-300)))
            entropy_histories[i].append(ent_bits)

            # Sample action (uses creature's dedicated rng — index-based, not arm-based)
            action = int(rngs[i].integers(0, 4))
            new_pos = world.move(positions_before[i], action)
            new_positions.append(new_pos)

            # Advance belief through B
            c.qs = B[:, :, action] @ qs_upd

            qs_updated_list.append(qs_upd)

        # Apply all moves simultaneously
        for i, c in enumerate(creatures):
            c.true_pos = new_positions[i]

    # Collect results
    results = []
    for i, c in enumerate(creatures):
        tail_entropies = entropy_histories[i][-TAIL:]
        mean_tail_entropy = float(np.mean(tail_entropies)) if tail_entropies else float("nan")
        r = {
            "map_acc": c.map_accuracy(),
            "tail_entropy_bits": mean_tail_entropy,
        }
        if use_modality:
            r["colocations"] = colocation_counts[i]
            r["pA2"] = pA2_list[i].copy()
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

# Storage: results[seed][arm_key][creature_idx]
# arm_key in ("SOLO-mirro", "SOLO-vela", "CO")
all_results = {seed: {} for seed in SEEDS}

print("seed  arm          creature   map_acc  tail_ent_bits  colocations")
print("-" * 68)

for seed in SEEDS:
    for arm in ("SOLO-mirro", "SOLO-vela", "CO"):
        # Build per-run deep copies (templates are never mutated)
        if arm == "SOLO-mirro":
            run_creatures = [copy.deepcopy(mirro_t)]
            creature_indices = [0]  # mirro is index 0
        elif arm == "SOLO-vela":
            run_creatures = [copy.deepcopy(vela_t)]
            creature_indices = [1]  # vela is index 1
        else:  # CO
            run_creatures = [copy.deepcopy(mirro_t), copy.deepcopy(vela_t)]
            creature_indices = [0, 1]

        # Build rngs: each creature's rng depends only on (seed, creature_index)
        # so trajectories are identical across arms
        rngs = [
            np.random.default_rng(100000 + 1000 * seed + i)
            for i in creature_indices
        ]

        use_mod = (arm == "CO")
        results = run_arm(run_creatures, STEPS, rngs, use_modality=use_mod)

        all_results[seed][arm] = results

        # Print one line per (seed, arm, creature)
        for k, res in enumerate(results):
            c = run_creatures[k]
            coloc_str = f"  {res['colocations']}" if use_mod else "  -"
            print(
                f"  {seed}   {arm:<12} {c.name:<14} "
                f"{res['map_acc']:.4f}   {res['tail_entropy_bits']:.4f}"
                f"{coloc_str}"
            )

print()

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------

checks: list[tuple[str, bool, str]] = []


def check(name: str, predicate_fn):
    """Run predicate_fn(); record (name, pass, detail). Exceptions count as FAIL."""
    try:
        passed, detail = predicate_fn()
        checks.append((name, passed, detail))
    except Exception as exc:
        checks.append((name, False, f"exception: {exc}"))


# ---------------------------------------------------------------------------
# Validity gate: every CO run must have >=20 co-location events
# ---------------------------------------------------------------------------

print("--- VALIDITY GATE ---")
gate_ok = True
for seed in SEEDS:
    co_res = all_results[seed]["CO"]
    mirro_coloc = co_res[0]["colocations"]
    vela_coloc = co_res[1]["colocations"]
    gate_seed = (mirro_coloc >= 20) and (vela_coloc >= 20)
    if not gate_seed:
        gate_ok = False
    status = "OK" if gate_seed else "FAIL"
    print(f"  seed={seed}: mirro_colocations={mirro_coloc}  vela_colocations={vela_coloc}  [{status}]")

if not gate_ok:
    print()
    print("GATE FAIL — RUN INVALID")
    sys.exit(1)

print("  Gate: PASS (all seeds >=20 co-locations)")
print()

# ---------------------------------------------------------------------------
# P1 — map non-degradation: CO acc >= SOLO acc - 0.04 in >=4/5 seeds, per creature
# ---------------------------------------------------------------------------

def _p1_mirro():
    pass_seeds = []
    seed_vals = []
    for seed in SEEDS:
        acc_solo = all_results[seed]["SOLO-mirro"][0]["map_acc"]
        acc_co = all_results[seed]["CO"][0]["map_acc"]
        ok = acc_co >= acc_solo - 0.04
        pass_seeds.append(ok)
        seed_vals.append(f"s{seed}:solo={acc_solo:.4f},co={acc_co:.4f},{'ok' if ok else 'FAIL'}")
    n_pass = sum(pass_seeds)
    return n_pass >= 4, f"passes={n_pass}/5  [{'; '.join(seed_vals)}]"


def _p1_vela():
    pass_seeds = []
    seed_vals = []
    for seed in SEEDS:
        acc_solo = all_results[seed]["SOLO-vela"][0]["map_acc"]
        acc_co = all_results[seed]["CO"][1]["map_acc"]
        ok = acc_co >= acc_solo - 0.04
        pass_seeds.append(ok)
        seed_vals.append(f"s{seed}:solo={acc_solo:.4f},co={acc_co:.4f},{'ok' if ok else 'FAIL'}")
    n_pass = sum(pass_seeds)
    return n_pass >= 4, f"passes={n_pass}/5  [{'; '.join(seed_vals)}]"


check("P1-map-nondegradation-mirro", _p1_mirro)
check("P1-map-nondegradation-vela", _p1_vela)

# ---------------------------------------------------------------------------
# P2 — localization non-degradation: loc_CO <= loc_SOLO + 0.2 bits in >=4/5 seeds
# ---------------------------------------------------------------------------

def _p2_mirro():
    pass_seeds = []
    seed_vals = []
    for seed in SEEDS:
        ent_solo = all_results[seed]["SOLO-mirro"][0]["tail_entropy_bits"]
        ent_co = all_results[seed]["CO"][0]["tail_entropy_bits"]
        ok = ent_co <= ent_solo + 0.2
        pass_seeds.append(ok)
        seed_vals.append(f"s{seed}:solo={ent_solo:.4f},co={ent_co:.4f},{'ok' if ok else 'FAIL'}")
    n_pass = sum(pass_seeds)
    return n_pass >= 4, f"passes={n_pass}/5  [{'; '.join(seed_vals)}]"


def _p2_vela():
    pass_seeds = []
    seed_vals = []
    for seed in SEEDS:
        ent_solo = all_results[seed]["SOLO-vela"][0]["tail_entropy_bits"]
        ent_co = all_results[seed]["CO"][1]["tail_entropy_bits"]
        ok = ent_co <= ent_solo + 0.2
        pass_seeds.append(ok)
        seed_vals.append(f"s{seed}:solo={ent_solo:.4f},co={ent_co:.4f},{'ok' if ok else 'FAIL'}")
    n_pass = sum(pass_seeds)
    return n_pass >= 4, f"passes={n_pass}/5  [{'; '.join(seed_vals)}]"


check("P2-localization-nondegradation-mirro", _p2_mirro)
check("P2-localization-nondegradation-vela", _p2_vela)

# ---------------------------------------------------------------------------
# Print property check results
# ---------------------------------------------------------------------------

print("--- PROPERTY CHECKS ---")
failed_names: list[str] = []
for name, passed, detail in checks:
    verdict = "PASS" if passed else "FAIL"
    print(f"{verdict}  {name}: {detail}")
    if not passed:
        failed_names.append(name)

print()

# ---------------------------------------------------------------------------
# Diagnostic: learned P(other-here | cell) from seed 0 CO run
# ---------------------------------------------------------------------------

print("--- DIAGNOSTIC (not a falsifier): learned P(other-here | cell), seed=0 CO ---")
co_res_s0 = all_results[0]["CO"]
cols = shared.cols
corner_cells = {0, 4, 20, 24}
center_cell = {12}

for creature_label, ci in [("mirro", 0), ("vela", 1)]:
    pA2 = co_res_s0[ci]["pA2"]  # shape (2, n_cells)
    # Normalize: P(other-here=1 | cell) = pA2[1, cell] / (pA2[0, cell] + pA2[1, cell])
    col_sums = pA2.sum(axis=0)
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    p_other_here = pA2[1, :] / col_sums  # shape (n_cells,)

    print(f"\n  {creature_label}: P(other-here | cell) as 5x5 grid:")
    for row in range(shared.rows):
        row_vals = p_other_here[row * cols: (row + 1) * cols]
        print("    " + "  ".join(f"{v:.3f}" for v in row_vals))

    corner_mean = float(np.mean([p_other_here[c] for c in corner_cells]))
    center_mean = float(np.mean([p_other_here[c] for c in center_cell]))
    print(f"  {creature_label}: mean P(other-here) — corners={corner_mean:.3f}  center={center_mean:.3f}")

print()

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

if not failed_names:
    print("RUNG 2: PASS")
else:
    print(f"RUNG 2: FAIL {failed_names}")
