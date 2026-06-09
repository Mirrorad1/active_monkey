"""Exp 50 — growth: mirro moves to a 5x5 world (episode 5; permanently mutates mirro).

Does an accumulated life transfer to a larger world? The 5x5 world embeds mirro's
current 3x3 coloring in its top-left block; 16 new cells carry a fixed balanced
pattern of the same 3 colors. Mirro GROWS: pA counts transplant positionally into the
embedded block (new cells start at the 0.1 prior); place belief carries to embedded
positions with epsilon mass on new cells (knowledge never reset; location uncertainty
honestly spread). The array surgery is harness engineering, PROVIDED. Baseline: 3
disposable newborns (seeds 11/12/13) in the same 5x5 world, never saved.
Everyone lives up to 4000 steps, checkpoints every 100.
Predeclared (direction card episode 5): convergence = first checkpoint with overall
map_accuracy >= 0.92 held 2 consecutive checkpoints. HELP = mirro converges in < 70%
of the newborn mean. INTERFERE = embedded-cell accuracy < 80% at any checkpoint >= 5.
INCONCLUSIVE (FAIL) = effect size < 5% in either direction. All three are findings.
Prediction (stated, uncertain): partial help below threshold (ratio 0.7-1.0); the 9
transplanted columns are locked-correct but confidently-wrong likelihoods on new cells
may cause mislocalization interference newborns do not have.
Deterministic continuation for mirro; newborn seeds 11/12/13 all reported.
"""
import numpy as np
from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# 1. Build 5x5 world
# ---------------------------------------------------------------------------
# Embedded block (top-left 3x3) = mirro's current cmap
embedded = [[0, 0, 0], [0, 0, 1], [2, 1, 2]]

# Remaining 16 cells (row-major non-embedded positions):
# cols 3-4 of rows 0-2  ->  6 positions
# all of rows 3-4        -> 10 positions
fill_pattern = [1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

cmap5 = []
fill_idx = 0
for r in range(5):
    for c in range(5):
        if r < 3 and c < 3:
            cmap5.append(embedded[r][c])
        else:
            cmap5.append(fill_pattern[fill_idx])
            fill_idx += 1

world5x5 = World(rows=5, cols=5, cmap=cmap5, n_colors=3)

print("5x5 cmap rows:")
for r in range(5):
    print(" ", cmap5[r*5:(r+1)*5])

embedded_indices = set(r * 5 + c for r in range(3) for c in range(3))
new_indices = set(range(25)) - embedded_indices

# ---------------------------------------------------------------------------
# 2. Load mirro, print BEFORE
# ---------------------------------------------------------------------------
STATE_DIR = "creature/state/mirro"
c = Creature.load(STATE_DIR)
vc_str = " ".join(f"c{i}={c.value_counts[i]:.1f}" for i in range(c.world.n_colors))
print(f"\nBEFORE: age={c.age_steps} map_acc={c.map_accuracy():.4f} {vc_str} hash={c._state_hash()[:16]}")

# ---------------------------------------------------------------------------
# 3. GROW mirro in place
# ---------------------------------------------------------------------------
old_pA = c.pA.copy()    # (3, 9)
old_qs = c.qs.copy()    # (9,)
old_pos = c.true_pos    # int 0-8

new_pA = np.full((3, 25), 0.1)
for r in range(3):
    for col in range(3):
        new_pA[:, r * 5 + col] = old_pA[:, r * 3 + col]

new_qs = np.full(25, 1e-6)
for r in range(3):
    for col in range(3):
        new_qs[r * 5 + col] = old_qs[r * 3 + col]
new_qs /= new_qs.sum()

old_r, old_c_pos = divmod(old_pos, 3)
new_true_pos = old_r * 5 + old_c_pos

c.world = world5x5
c.pA = new_pA
c.qs = new_qs
c.true_pos = new_true_pos

c._bio_append({
    "event": "growth",
    "age_steps": c.age_steps,
    "summary": "Exp 50: world grows 3x3 -> 5x5 (embedded top-left); pA/qs transplanted, 16 new cells at prior"
})

# Immediate post-growth accuracy
sm = c.sensory_map()
emb_acc = sum(sm[i] == cmap5[i] for i in embedded_indices) / len(embedded_indices)
new_acc = sum(sm[i] == cmap5[i] for i in new_indices) / len(new_indices)
print(f"Post-growth immediate: embedded_acc={emb_acc:.4f} new_acc={new_acc:.4f} overall={c.map_accuracy():.4f}")

# ---------------------------------------------------------------------------
# 4. Mirro checkpoint loop (up to 40 x 100 steps)
# ---------------------------------------------------------------------------
STEPS_PER_CK = 100
MAX_CK = 40
CONV_ACC = 0.92
CONV_HOLD = 2

mirro_records = []   # (ck, overall, emb_acc, new_acc)
mirro_conv_ck = None
consec = 0

print("\n--- Mirro checkpoints ---")
for ck in range(1, MAX_CK + 1):
    c.live(STEPS_PER_CK)
    sm = c.sensory_map()
    ov = c.map_accuracy()
    ea = sum(sm[i] == cmap5[i] for i in embedded_indices) / len(embedded_indices)
    na = sum(sm[i] == cmap5[i] for i in new_indices) / len(new_indices)
    mirro_records.append((ck, ov, ea, na))
    print(f"  ck{ck:02d} age={c.age_steps} overall={ov:.4f} emb={ea:.4f} new={na:.4f}")
    if ov >= CONV_ACC:
        consec += 1
        if consec >= CONV_HOLD and mirro_conv_ck is None:
            mirro_conv_ck = ck - (CONV_HOLD - 1)  # first ck that started the streak
            print(f"  ** Mirro converged at ck {mirro_conv_ck} (held {CONV_HOLD}) **")
            if ck >= mirro_conv_ck + CONV_HOLD:
                break
    else:
        consec = 0
    if mirro_conv_ck is not None and ck >= mirro_conv_ck + CONV_HOLD - 1:
        break

# ---------------------------------------------------------------------------
# 5. Newborns
# ---------------------------------------------------------------------------
newborn_conv_steps = []
print("\n--- Newborn baselines ---")
for seed in (11, 12, 13):
    nb = Creature.birth(f"baseline5x5_s{seed}", world5x5, seed=seed)
    nb_conv_ck = None
    nb_consec = 0
    print(f"  Seed {seed}:")
    for ck in range(1, MAX_CK + 1):
        nb.live(STEPS_PER_CK)
        ov = nb.map_accuracy()
        print(f"    ck{ck:02d} overall={ov:.4f}")
        if ov >= CONV_ACC:
            nb_consec += 1
            if nb_consec >= CONV_HOLD and nb_conv_ck is None:
                nb_conv_ck = ck - (CONV_HOLD - 1)
                print(f"    ** Seed {seed} converged at ck {nb_conv_ck} **")
                if ck >= nb_conv_ck + CONV_HOLD - 1:
                    break
        else:
            nb_consec = 0
    steps_val = nb_conv_ck * STEPS_PER_CK if nb_conv_ck is not None else None
    newborn_conv_steps.append(steps_val)
    print(f"  Seed {seed} convergence_steps={steps_val}")

# ---------------------------------------------------------------------------
# 6. Save mirro, print AFTER
# ---------------------------------------------------------------------------
c.save(STATE_DIR)
sm_after = c.sensory_map()
ov_after = c.map_accuracy()
ea_after = sum(sm_after[i] == cmap5[i] for i in embedded_indices) / len(embedded_indices)
na_after = sum(sm_after[i] == cmap5[i] for i in new_indices) / len(new_indices)
print(f"\nAFTER: age={c.age_steps} overall={ov_after:.4f} emb={ea_after:.4f} new={na_after:.4f} hash={c._state_hash()[:16]}")

# ---------------------------------------------------------------------------
# 7. Summary & verdict
# ---------------------------------------------------------------------------
mirro_steps = mirro_conv_ck * STEPS_PER_CK if mirro_conv_ck is not None else None
valid_nb = [s for s in newborn_conv_steps if s is not None]
nb_mean = sum(valid_nb) / len(valid_nb) if valid_nb else None

print(f"\n--- Summary ---")
print(f"mirro_convergence_steps={mirro_steps}")
print(f"newborn_convergence_steps={newborn_conv_steps} mean={nb_mean}")

if mirro_steps is not None and nb_mean is not None:
    ratio = mirro_steps / nb_mean
else:
    ratio = None
print(f"ratio={ratio:.3f}" if ratio is not None else "ratio=None")

# Embedded acc min from ck5 onward
emb_from_ck5 = [ea for (ck, ov, ea, na) in mirro_records if ck >= 5]
emb_min = min(emb_from_ck5) if emb_from_ck5 else None
print(f"embedded_acc_min_after_ck5={emb_min:.4f}" if emb_min is not None else "embedded_acc_min_after_ck5=None")

# Verdict (INTERFERE takes precedence)
interfere_triggered = emb_min is not None and emb_min < 0.80
interfere_ck = None
if interfere_triggered:
    for (ck, ov, ea, na) in mirro_records:
        if ck >= 5 and ea < 0.80:
            interfere_ck = ck
            break

if interfere_triggered:
    print(f"VERDICT: INTERFERE (embedded acc {emb_min:.4f} < 0.80 at ck {interfere_ck})")
elif ratio is None:
    print("VERDICT: INCONCLUSIVE (no convergence data)")
elif ratio < 0.70:
    print(f"VERDICT: HELP (ratio {ratio:.3f} < 0.70)")
elif ratio < 0.95:
    print(f"VERDICT: PARTIAL HELP (ratio {ratio:.3f} in [0.70, 0.95))")
elif ratio <= 1.05:
    print(f"VERDICT: NO EFFECT (ratio {ratio:.3f} in [0.95, 1.05])")
elif ratio <= 1.10:
    print(f"VERDICT: HINDRANCE (ratio {ratio:.3f} > 1.05)")
else:
    print(f"VERDICT: INCONCLUSIVE (effect < 5%)" if abs(ratio - 1.0) < 0.05 else f"VERDICT: HINDRANCE (ratio {ratio:.3f} > 1.05)")
