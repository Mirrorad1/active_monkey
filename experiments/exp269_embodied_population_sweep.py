"""Exp 269 — Embodied population stability-vs-competition calibration sweep.

QUESTION (Phase-2.5 of the embodied-physics direction): with a REAL articulated body
and real physics (MuJoCo/MJX quadruped, fixed Phase-1 walk-to-food gait, NO evolution),
is there any food-field calibration where an embodied population is BOTH stable (certifies
under the FROZEN Preflight stability gate) AND competitive (density-dependent: per-capita
intake falls as N rises)? The prior continuous-locomotion / patch-mosaic arc found a
"stability-vs-strong-competition" wall; does a real body dissolve it, or does it recur?

PREDECLARED FALSIFIER: if NO cell in the (capacity x regen) grid is both stable and
density-dependent, the wall recurs on the embodied substrate (substrate-general).

WHERE THE WORK LIVES: the sweep engine is `embodied.sweep_phase2p5` on the MJX-batched
substrate (embodied/batched_population.py + batched_world.py), reusing the FROZEN
`ecology.evolvability.stability` gate via `embodied.run_population.certify`. The full grid
is GPU-only (it batches 256 bodies through MJX physics); it ran on a RunPod H100 80GB:

    unset LD_LIBRARY_PATH
    export JAX_PLATFORM_NAME=cuda CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.
    python -m embodied.sweep_phase2p5 --capacities 8 12 16 22 30 --regens 0.3 0.5 \
        --seeds 0 1 2 --horizon 300 --max-pop 256

Raw output: experiments/outputs/exp269.txt. Verdict: NEGATIVE / NEW INSIGHT — 0/10 cells
stable; competitive XOR stable (rich cells are density-dependent but boom-bust; poor cells
are calm only as they collapse).

THIS SCRIPT reproduces the verdict-CRITICAL, CPU-deterministic check the controller owns:
that the FROZEN stability gate is NON-DEGENERATE — it can return stable=True for a genuinely
stable input — so "stable 0/3 in every cell" is a real signal and not a stuck gate
(VALIDATION.md: a NEGATIVE from an instrument that can never say POSITIVE is worthless).
Run: uv run --python .venv python experiments/exp269_embodied_population_sweep.py
"""
import numpy as np

from embodied.population import PopResult
from embodied.run_population import certify


def _mk(N, pci):
    N = np.asarray(N, dtype=float)
    return PopResult(n_series=[int(x) for x in N], per_capita_intake=list(pci),
                     births=0, deaths=0, events_hash="x", final_alive=int(N[-1]))


def non_degeneracy_proof():
    """The gate must PASS a stable input and FAIL boom-bust / collapse inputs."""
    T = 300
    rng = np.random.default_rng(0)

    # (1) genuinely stable: tight band ~50, no drift, density-dependent intake
    N_stable = 50 + rng.normal(0, 1.5, T)
    pci_dd = 2.0 - 0.01 * N_stable + rng.normal(0, 0.02, T)
    vs = certify(_mk(N_stable, pci_dd))

    # (2) boom-bust like the rich cells (cap=30/0.5): peak 256 -> final ~125, big swings
    N_bb = np.concatenate([np.linspace(30, 256, 80),
                           256 - 120 * np.abs(np.sin(np.linspace(0, 6, 220)))])
    vb = certify(_mk(N_bb, 2.0 - 0.01 * N_bb))

    # (3) collapse like the poor cells (cap=8): crashes toward 0
    N_col = np.concatenate([np.linspace(30, 49, 40), np.linspace(49, 0, 260)])
    vc = certify(_mk(N_col, 1.0 + rng.normal(0, 0.01, T)))

    print("FROZEN-gate non-degeneracy check (re-applied to synthetic inputs):")
    for name, v in [("STABLE  ", vs), ("BOOMBUST", vb), ("COLLAPSE", vc)]:
        print(f"  {name} -> stable={v['stable']!s:5}  dd={v['density_dependent']!s:5}  "
              f"persist={v['persistence']:.0f}  cv={v['level_cv']:.3f}  "
              f"drift={v['drift']:.3f}  {v['oscillation']}")

    assert vs["stable"] is True, "gate must PASS a genuinely stable input (else degenerate)"
    assert vs["density_dependent"] is True
    assert vb["stable"] is False, "boom-bust must FAIL the gate"
    assert vc["stable"] is False, "collapse must FAIL the gate"
    print("PASS: gate is non-degenerate -> the sweep's 0/10 stable verdict is a real NEGATIVE.")


if __name__ == "__main__":
    non_degeneracy_proof()
