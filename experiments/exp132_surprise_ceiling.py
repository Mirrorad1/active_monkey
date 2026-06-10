"""Exp 132 — the surprise-ceiling hypothesis: does the fixed state space hit a measurable
ceiling in a standard life run? (The structure-learning directive's opening experiment.)

The human's hypothesis (directive, 2026-06-10): the current fixed state space hits a
measurable surprise ceiling in at least one modality during a standard life run. The
loop's predeclared counter-prediction, stated before running: it does NOT — the standard
world is deterministic and fully learnable, so surprise falls to a near-zero floor and
the model is structurally ADEQUATE here; the ceiling detector should stay quiet. Either
outcome is recorded as-is. To make the test mean something either way, a POSITIVE
CONTROL arm runs the same creature in a world with genuinely irreducible observation
noise (each visit: true color with p=0.7, else uniform among the other two; analytic
irreducible surprise = 0.7 ln(1/0.7) + 0.3 ln(1/0.15) ~ 0.82 nats > the 0.7-nat ceiling
threshold) — there the detector MUST fire, or the instrument itself is mis-calibrated.

Arms (8 fresh birth seeds 3100-3107 each, 3000 steps, start cell 12):
  ARM-STD:   mirro's committed world layout (deterministic cmap).
  ARM-NOISE: same layout wrapped in a NoisyCmap (p_true=0.7, its own seeded rng so the
             creature's action stream is untouched — trajectory determinism preserved).
Predeclared:
  P1 (instrument validation): ARM-NOISE ceiling fires (>= 1 ceiling event by step 3000)
     in >= 6/8 seeds. F1 = fails -> the detector is mis-calibrated; fix before any
     structural claim (no verdict on the hypothesis).
  P2 (the directive's hypothesis, scored as stated): the hypothesis is CONFIRMED iff
     ARM-STD fires in >= 4/8 seeds; the loop's counter-prediction is 0/8. Either way is
     the finding.
Also reported per arm: final window mean surprise, final slope, ceiling-event count,
final map accuracy. The single modality (color) and single factor (place) are this
substrate's whole observation/state structure — "at least one modality" = this one.
No creature state touched (fresh separate roots only); FROZEN untouched.
"""
import json
from pathlib import Path

import numpy as np

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SEEDS = 8
BIRTH_SEED_BASE = 3100
NOISE_SEED_BASE = 9000
TOTAL_STEPS = 3000
N_CHUNKS = 6
CHUNK = TOTAL_STEPS // N_CHUNKS  # 500
START_CELL = 12
P_TRUE = 0.7


class NoisyCmap:
    """Color map wrapper with irreducible observation noise.

    Each lookup returns the true color with probability p_true, otherwise a
    uniform draw among the other colors.  Uses its OWN seeded rng so the
    creature's action stream is untouched (trajectory determinism preserved).
    """

    def __init__(self, base_cmap, n_colors, p_true, seed):
        self.base = list(base_cmap)
        self.n_colors = int(n_colors)
        self.p_true = float(p_true)
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, s):
        true = self.base[s]  # raises IndexError past the end (sequence protocol)
        if self.rng.random() < self.p_true:
            return true
        others = [c for c in range(self.n_colors) if c != true]
        return int(self.rng.choice(others))

    def __len__(self):
        return len(self.base)


def load_base_world() -> World:
    """Read mirro's committed world layout (read-only, as exp85 does)."""
    manifest = json.loads(
        Path("creature/state/mirro/manifest.json").read_text()
    )
    return World.from_dict(manifest["world"])


def map_accuracy_vs_base(c: Creature, base_cmap) -> float:
    """Learned argmax tuning vs the BASE (noiseless) cmap, computed manually."""
    A_hat = c._A_hat()
    learned = [int(np.argmax(A_hat[:, s])) for s in range(c.world.n_cells)]
    return sum(l == t for l, t in zip(learned, base_cmap)) / len(base_cmap)


def run_seed(arm: str, idx: int, base_world: World,
             total_steps: int = TOTAL_STEPS, n_chunks: int = N_CHUNKS) -> dict:
    """Run one (arm, seed) life; return final metrics + ceiling count + map acc."""
    base_cmap = list(base_world.cmap)
    c = Creature.birth(f"exp132-{arm}-s{idx}", base_world,
                       seed=BIRTH_SEED_BASE + idx)
    if arm == "NOISE":
        # Same dims; World is a dataclass — construct directly with the noisy cmap.
        # move() and transition_matrix() never touch cmap, so the innate B is intact.
        c.world = World(
            rows=base_world.rows,
            cols=base_world.cols,
            cmap=NoisyCmap(base_cmap, base_world.n_colors, P_TRUE,
                           seed=NOISE_SEED_BASE + idx),
            n_colors=base_world.n_colors,
        )
    c.true_pos = START_CELL

    chunk = total_steps // n_chunks
    chunk_metrics = []
    for _ in range(n_chunks):
        c.live(chunk)  # end-of-live ceiling check fires once per chunk
        chunk_metrics.append(c.surprise_metrics())

    final = chunk_metrics[-1]
    if arm == "NOISE":
        map_acc = map_accuracy_vs_base(c, base_cmap)
    else:
        map_acc = c.map_accuracy()
    return dict(
        arm=arm,
        idx=idx,
        birth_seed=BIRTH_SEED_BASE + idx,
        final_mean=final["mean"],
        final_slope=final["slope"],
        n_ceiling_events=len(final["events"]),
        map_acc=map_acc,
        chunk_metrics=chunk_metrics,
    )


def main() -> None:
    base_world = load_base_world()

    results = {"STD": [], "NOISE": []}
    for arm in ("STD", "NOISE"):
        for idx in range(N_SEEDS):
            results[arm].append(run_seed(arm, idx, base_world))

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    for arm in ("STD", "NOISE"):
        print(f"\nARM-{arm}  ({TOTAL_STEPS} steps, {N_CHUNKS} chunks of {CHUNK}, "
              f"start cell {START_CELL})")
        print(f"{'seed':>6} {'final_mean':>11} {'final_slope':>12} "
              f"{'n_ceiling':>10} {'map_acc':>8}")
        for r in results[arm]:
            print(f"{r['birth_seed']:>6} {r['final_mean']:>11.4f} "
                  f"{r['final_slope']:>+12.6f} {r['n_ceiling_events']:>10d} "
                  f"{r['map_acc']:>8.3f}")

    # ------------------------------------------------------------------
    # Predeclared checks
    # ------------------------------------------------------------------
    m = sum(1 for r in results["NOISE"] if r["n_ceiling_events"] >= 1)
    k = sum(1 for r in results["STD"] if r["n_ceiling_events"] >= 1)

    print(f"\nP1 (instrument validation): ARM-NOISE fires in {m}/8 seeds "
          f"(need >= 6/8): {'PASS' if m >= 6 else 'FAIL -> F1'}")
    print(f"P2 (directive's hypothesis): ARM-STD fires in {k}/8 seeds "
          f"(CONFIRMED iff >= 4/8; loop's counter-prediction: 0/8)")

    if m < 6:
        print("\nEXP132: F1 — DETECTOR MIS-CALIBRATED; no verdict")
    elif k >= 4:
        print(f"\nEXP132: HYPOTHESIS CONFIRMED (std fires {k}/8)")
    else:
        counter = "exact" if k == 0 else "directionally right"
        print(f"\nEXP132: HYPOTHESIS NOT CONFIRMED (std fires {k}/8; "
              f"counter-prediction {counter}) — instrument validated {m}/8 on noise")


if __name__ == "__main__":
    main()
