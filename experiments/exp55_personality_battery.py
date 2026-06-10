"""Exp 55 — personality battery (functional-emergence rung 2).

Fixed probe battery on five subjects: mirro at three committed ages of its ONE life
(1300 @ git 519f303, 6300 @ git 0b5e59f, 6700 @ working tree) and two twins forked
from mirro@6700 with divergent 1500-step interim histories (twin-G: green-rich world
[cells 4,9,12,15,18,21 -> 2... see code]; twin-R: red-rich [same cells -> 0]).
Battery (run on disposable copies; subjects never mutated): P1 value-share vector (3);
P2 conviction + gap_share (2); P3 map sharpness = mean column entropy of A_hat (1);
P4 revision-speed = favorite's share drop after a fixed 200-step counter-evidence
protocol on a probe fork (world 60% recolored to the runner-up color) (1). Profile =
7-dim, z-scored across subjects before similarity.
PREDECLARED LIMITATION: the card's 'exploration disposition' is NOT measurable -- this
substrate has no action selection (live() actions are provided-random); personality can
only live in world-model/value state here. Named substrate gap.
Criteria (card): (a) temporal self-similarity -- mean cosine(sim) over mirro age-pairs
exceeds mean cosine over (mirro-age, twin) pairs by > 0.05; (b) individuality -- twins
differ in favorite OR L1 share-distance > 0.1.
FAIL (card): (a) fails (profiles unstable across ages) OR (b) fails (twins identical).
Prediction (stated): (b) passes; (a) likely FAILS because mirro's life between these
snapshots contains engineered value reversals (Exp 48/49 entrenchment+revision, Exp 50
growth) -- if so, log honestly: personality at this scale is current-state readout
under an eventful history; stability would need undisturbed epochs.
Seeds: twins' interim lives base 71/72; revision probes base 73+subject_index. All listed.
"""

import math
import subprocess
import tempfile
import hashlib
from pathlib import Path

import numpy as np

from active_loop.creature import Creature

REPO = Path(__file__).parent.parent


def _git_materialize(commit: str, tmpdir: Path) -> Creature:
    """Materialize creature/state/mirro from a git commit into tmpdir."""
    for fname in ("manifest.json", "BIOGRAPHY.jsonl"):
        data = subprocess.check_output(
            ["git", "show", f"{commit}:creature/state/mirro/{fname}"],
            cwd=REPO,
        )
        (tmpdir / fname).write_bytes(data)
    # arrays.npz is binary — must write raw bytes
    with open(tmpdir / "arrays.npz", "wb") as fh:
        subprocess.run(
            ["git", "show", f"{commit}:creature/state/mirro/arrays.npz"],
            cwd=REPO, stdout=fh, check=True,
        )
    return Creature.load(tmpdir)


def _hash_dir(path: Path) -> str:
    h = hashlib.sha256()
    for fname in sorted(["arrays.npz", "manifest.json", "BIOGRAPHY.jsonl"]):
        p = path / fname
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def battery(subject: Creature, idx: int) -> list:
    """Compute 7-dim personality profile without mutating subject."""
    s = subject.fork(f"probe_{idx}")
    total = s.value_counts.sum()
    shares = s.value_counts / (total if total > 0 else 1.0)
    fav = int(np.argmax(s.value_counts))
    sorted_vals = np.sort(s.value_counts)[::-1]
    runner = int(np.argsort(s.value_counts)[::-1][1])
    runner_share = sorted_vals[1] / (total if total > 0 else 1.0)
    conv = s.conviction()
    gap_share = float(shares[fav] - runner_share)

    # P3: map sharpness = mean column entropy of A_hat (nats)
    A_hat = s._A_hat()
    col_entropies = -np.sum(A_hat * np.log(A_hat + 1e-12), axis=0)
    sharpness = float(np.mean(col_entropies))

    # P4: revision index
    rfork = subject.fork(f"rev_{idx}")
    n_cells = rfork.world.n_cells
    recolor_n = math.ceil(0.6 * n_cells)
    cmap = list(rfork.world.cmap)
    recolored = 0
    for ci in range(n_cells):
        if recolored >= recolor_n:
            break
        if cmap[ci] != runner:
            cmap[ci] = runner
            recolored += 1
    rfork.world.cmap = cmap
    pre_fav_share = float(rfork.value_counts[fav] / (rfork.value_counts.sum() if rfork.value_counts.sum() > 0 else 1.0))
    base_seed = 73 + idx
    for i in range(200):
        rfork.live(1, seed=(base_seed * 1_000_003 + i) & 0xFFFFFFFFFFFFFFFF)
    post_total = rfork.value_counts.sum()
    post_fav_share = float(rfork.value_counts[fav] / (post_total if post_total > 0 else 1.0))
    revision_index = pre_fav_share - post_fav_share

    profile = [float(shares[0]), float(shares[1]), float(shares[2]),
               conv, gap_share, sharpness, revision_index]
    return profile


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    # --- Load subjects ---
    print("=== Loading subjects ===")
    wt_mirro_dir = REPO / "creature" / "state" / "mirro"
    hash_before = _hash_dir(wt_mirro_dir)

    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        m1300 = _git_materialize("519f303", Path(td1))
        print(f"mirro@1300: age={m1300.age_steps}, world={m1300.world.rows}x{m1300.world.cols}")
        m6300 = _git_materialize("0b5e59f", Path(td2))
        print(f"mirro@6300: age={m6300.age_steps}, world={m6300.world.rows}x{m6300.world.cols}")

        m6700 = Creature.load(wt_mirro_dir)
        m6700._state_dir = None  # detach so fork/live events don't write back to working tree
        print(f"mirro@6700: age={m6700.age_steps}, world={m6700.world.rows}x{m6700.world.cols}")

        # --- Build twins from mirro@6700 ---
        print("\n=== Building twins ===")
        tg = m6700.fork("twin_g")
        cmap_g = list(tg.world.cmap)
        for ci in [4, 9, 12, 15, 18, 21]:
            if ci < len(cmap_g):
                cmap_g[ci] = 2
        tg.world.cmap = cmap_g
        for i in range(1500):
            tg.live(1, seed=(71 * 1_000_003 + i) & 0xFFFFFFFFFFFFFFFF)
        print(f"twin_G: age={tg.age_steps}, fav={tg.favorite()}, value_counts={tg.value_counts.round(2)}")

        tr = m6700.fork("twin_r")
        cmap_r = list(tr.world.cmap)
        for ci in [4, 9, 12, 15, 18, 21]:
            if ci < len(cmap_r):
                cmap_r[ci] = 0
        tr.world.cmap = cmap_r
        for i in range(1500):
            tr.live(1, seed=(72 * 1_000_003 + i) & 0xFFFFFFFFFFFFFFFF)
        print(f"twin_R: age={tr.age_steps}, fav={tr.favorite()}, value_counts={tr.value_counts.round(2)}")

        # --- Battery ---
        print("\n=== Running battery ===")
        subjects = [m1300, m6300, m6700, tg, tr]
        labels = ["m1300", "m6300", "m6700", "twinG", "twinR"]
        profiles = []
        for i, (subj, lbl) in enumerate(zip(subjects, labels)):
            p = battery(subj, i)
            profiles.append(p)
            print(f"{lbl}: profile={[round(x, 4) for x in p]}  fav={subj.favorite()}  world={subj.world.rows}x{subj.world.cols}")

        # --- z-score ---
        P = np.array(profiles)  # (5, 7)
        mu = P.mean(axis=0)
        sigma = P.std(axis=0)
        sigma_safe = np.where(sigma == 0, 1.0, sigma)
        Pz = (P - mu) / sigma_safe

        # --- Cosine similarity matrix ---
        print("\n=== Cosine similarity matrix (z-scored profiles) ===")
        n = len(labels)
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sim[i, j] = cosine(Pz[i], Pz[j])

        header = f"{'':>7}" + "".join(f"{lbl:>8}" for lbl in labels)
        print(header)
        for i, lbl in enumerate(labels):
            row = f"{lbl:>7}" + "".join(f"{sim[i, j]:8.3f}" for j in range(n))
            print(row)

        # --- Criteria ---
        self_sim = (sim[0, 1] + sim[0, 2] + sim[1, 2]) / 3.0
        cross_pairs = [(i, j) for i in range(3) for j in [3, 4]]
        cross_sim = np.mean([sim[i, j] for i, j in cross_pairs])
        print(f"\nself_sim (mirro age-pairs): {self_sim:.4f}")
        print(f"cross_sim (mirro-ages vs twins): {cross_sim:.4f}")
        diff = self_sim - cross_sim
        print(f"difference: {diff:.4f}")
        crit_a = diff > 0.05
        print(f"Criterion (a) [self_sim - cross_sim > 0.05]: {'PASS' if crit_a else 'FAIL'}")

        shares_g = np.array(profiles[3][:3])
        shares_r = np.array(profiles[4][:3])
        fav_g = tg.favorite()
        fav_r = tr.favorite()
        l1_dist = float(np.sum(np.abs(shares_g - shares_r)))
        print(f"\ntwinG fav={fav_g}, twinR fav={fav_r}, L1(shares)={l1_dist:.4f}")
        crit_b = (fav_g != fav_r) or (l1_dist > 0.1)
        print(f"Criterion (b) [fav differs OR L1 > 0.1]: {'PASS' if crit_b else 'FAIL'}")

        if crit_a and crit_b:
            verdict = "personality CONFIRMED (stable + individuated)"
        elif crit_b and not crit_a:
            verdict = "MIXED: individuated but not stable across this eventful life"
        elif crit_a and not crit_b:
            verdict = "MIXED: stable but twins failed to individuate"
        else:
            verdict = "falsifier HIT (neither)"
        print(f"\nVERDICT: {verdict}")

        # --- mirro untouched check ---
        hash_after = _hash_dir(wt_mirro_dir)
        print(f"\nmirro_untouched: hash_before={hash_before}  hash_after={hash_after}  {'OK' if hash_before == hash_after else 'MUTATED!'}")


if __name__ == "__main__":
    main()
