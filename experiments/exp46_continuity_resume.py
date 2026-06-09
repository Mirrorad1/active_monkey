"""Exp 46 — continuity across sessions: mirro resumes a saved life in a fresh process.

Hypothesis: mirro's competence survives a save->load across Python processes and
continued living -- the RECIPE's 'continuous registered experience' holds at the
session boundary, not just within one script run.
Prediction if TRUE: (a) loaded state hash == committed manifest state_hash (load
integrity); (b) post-load map_accuracy 9/9 (>= 8/9 required) and localize_bits < 0.1;
(c) after live(300) in this fresh process: map_accuracy >= 8/9 and localize_bits < 0.1.
Falsifier (direction-card thresholds): post-resume map accuracy < 7/9, or calibration
degradation. Stated adaptation: the card's 'entropy increase > 10%' is degenerate at
mirro's ~0.000-bit committed baseline, so the predeclared bound is ABSOLUTE:
localize_bits must stay < 0.1 bits. Single seed (mirro's own continued rng) is
admissible: the falsifier is a magnitude bound (VALIDATION.md, statistical section).
This script never births; the state comes only from the committed snapshot on disk.
"""
import json
from pathlib import Path

from active_loop.creature import Creature

STATE = "creature/state/mirro"

# 1. Read manifest
manifest = json.loads((Path(STATE) / "manifest.json").read_text())
manifest_age = manifest["age_steps"]
manifest_hash = manifest["state_hash"]
print(f"manifest_age={manifest_age}  manifest_hash={manifest_hash[:16]}")

# 2. Load creature
c = Creature.load(STATE)
loaded_hash = c._state_hash()
hash_match = loaded_hash == manifest_hash
print(f"loaded: name={c.name} age={c.age_steps}")
print(f"loaded_hash={loaded_hash[:16]}  hash_match={hash_match}")

# 3. BEFORE metrics
acc_before = c.map_accuracy()
bits_before = c.localize_bits()
n_cells = c.world.n_cells
acc_before_n = round(acc_before * n_cells)
print(f"BEFORE  map_accuracy={acc_before:.4f} ({acc_before_n}/{n_cells})  localize_bits={bits_before:.4f}")

# 4. Live 300 steps
c.live(300)
age_after = c.age_steps

# 5. AFTER metrics
acc_after = c.map_accuracy()
bits_after = c.localize_bits()
acc_after_n = round(acc_after * n_cells)
print(f"AFTER   map_accuracy={acc_after:.4f} ({acc_after_n}/{n_cells})  localize_bits={bits_after:.4f}  age={age_after}")

# 6. Save
c.save(STATE)
new_hash = c._state_hash()
print(f"saved: age={c.age_steps} new_hash={new_hash[:16]}")

# 7. Summary
resume_ok = "PASS" if hash_match else "FAIL"
print(f"\nresume_integrity={resume_ok}")
print(f"post_load:    acc={acc_before_n}/{n_cells} bits={bits_before:.4f}")
print(f"post_live300: acc={acc_after_n}/{n_cells} bits={bits_after:.4f}")

# Verdict logic
if (hash_match and
        acc_before_n >= 8 and bits_before < 0.1 and
        acc_after_n >= 8 and bits_after < 0.1):
    print("VERDICT: prediction CONFIRMED")
elif acc_before_n < 7 or acc_after_n < 7 or bits_before >= 0.1 or bits_after >= 0.1:
    reasons = []
    if acc_before_n < 7:
        reasons.append(f"post-load acc {acc_before_n}/9 < 7/9")
    if acc_after_n < 7:
        reasons.append(f"post-live acc {acc_after_n}/9 < 7/9")
    if bits_before >= 0.1:
        reasons.append(f"post-load bits {bits_before:.4f} >= 0.1")
    if bits_after >= 0.1:
        reasons.append(f"post-live bits {bits_after:.4f} >= 0.1")
    print(f"VERDICT: falsifier HIT ({'; '.join(reasons)})")
else:
    reasons = []
    if not hash_match:
        reasons.append("hash mismatch")
    if 7 <= acc_before_n < 8:
        reasons.append(f"post-load acc {acc_before_n}/9 in 7-8/9 zone")
    if 7 <= acc_after_n < 8:
        reasons.append(f"post-live acc {acc_after_n}/9 in 7-8/9 zone")
    print(f"VERDICT: MIXED ({'; '.join(reasons)})")
