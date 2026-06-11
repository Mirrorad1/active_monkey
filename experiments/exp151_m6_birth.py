"""
Exp 151 — continuous-creature rung M6: the birth of nira (the species line; the
migration's final rung — engineering + exactness verification, per the card).

Card loop/directions/continuous-creature.md (M6). Guardrail: any failed
predeclared property HALTS for explicit human input.

nira: the first committed continuous-substrate creature — the M1-M5 machinery
packaged as a persistent life under creature/state/nira/, clade rules (mirro
remains the tabular root ancestor; nira is a NEW species line, not a fork);
its native world is the half-noisy 16-color 4x4 world of M4/M5 (noisy left,
p=0.6). Born at seed 0. Belief never reset; biography append-only; the
continuity guard extended to its line.

Predeclarations (engineering exactness, TRUE iff all):
- P1 round-trip: save -> load -> state_hash identical; AND two fresh loads each
  living 100 further steps (same explicit seed) produce identical hashes
  (resume determinism).
- P2 guard coverage: the nira continuity test passes against the saved state;
  the full fast suite stays green (mirro/vela guards untouched).
- P3 first-life sanity (the packaged class reproduces the experiment bands):
  after live(4000): map_formed_count >= 14/16, final localization error <= 0.5,
  reliable_share > 0.55 with the favorite color in the reliable (right) half;
  after teaching 16 words (n=8): answer_what_do_you_like names the favorite's
  word.
Falsifiers (any HALTS): hash mismatch or resume nondeterminism; guard gap or
suite regression; any sanity band missed (the packaging changed behavior).
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent
STATE_DIR = ROOT / "creature" / "state" / "nira"

# Add repo root to path for PYTHONPATH-less invocation
sys.path.insert(0, str(ROOT))

from active_loop.creature_continuous import ContinuousCreature  # noqa: E402

N_COLORS = 16


def main() -> None:
    print("=" * 72)
    print("Exp 151 — continuous-creature rung M6: the birth of nira")
    print("=" * 72)
    print()

    # ------------------------------------------------------------------ #
    # Birth nira                                                           #
    # ------------------------------------------------------------------ #
    print("Birthing nira (seed=0, noisy_half='left', noise_p=0.6) ...")
    nira = ContinuousCreature.birth("nira", seed=0, noisy_half="left", noise_p=0.6)
    print(f"  Born: {nira}")
    print()

    # ------------------------------------------------------------------ #
    # Live 4000 steps (Phase A: uniform-random wander)                    #
    # ------------------------------------------------------------------ #
    print("Living 4000 steps (uniform-random wander) ...")
    nira.live(4000)
    print(f"  After live(4000): {nira}")

    map_formed = nira.map_formed_count()
    loc_err = nira._localization_error()
    rel_share = nira.reliable_share()
    fav = nira.favorite()

    # Reliable set for noisy_half='left': cols 2,3 -> colors 2,3,6,7,10,11,14,15
    reliable_colors = nira._reliable_colors()
    fav_in_reliable = fav in reliable_colors

    print(f"  map_formed_count = {map_formed}/16")
    print(f"  localization_error = {loc_err:.4f}")
    print(f"  reliable_share = {rel_share:.4f}")
    print(f"  favorite color = {fav}  (in reliable half: {fav_in_reliable})")
    print()

    # ------------------------------------------------------------------ #
    # Teach 16 words (one per color, n=8)                                 #
    # ------------------------------------------------------------------ #
    print("Teaching 16 words (w0..w15, n=8 each) ...")
    for color_idx in range(N_COLORS):
        nira.teach_word(f"w{color_idx}", color_idx, n=8)
    print(f"  Vocab size: {len(nira.vocab)} words")
    print()

    # ------------------------------------------------------------------ #
    # Ask: what do you like? + do you like <word>?                        #
    # ------------------------------------------------------------------ #
    print("=" * 72)
    print("NIRA'S FIRST WORDS")
    print("=" * 72)
    print()
    ans_like = nira.answer_what_do_you_like()
    print(f"  Q: What do you like?")
    print(f"  A: \"{ans_like}\"")
    print()

    fav_word = f"w{fav}"
    ans_own = nira.answer_do_you_like(fav_word)
    print(f"  Q: Do you like {fav_word}? (own favorite)")
    print(f"  A: \"{ans_own}\"")
    print()

    # Ask about one color from the noisy (left) half
    noisy_colors = set(range(N_COLORS)) - reliable_colors
    noisy_example = sorted(noisy_colors)[0]
    noisy_word = f"w{noisy_example}"
    ans_noisy = nira.answer_do_you_like(noisy_word)
    print(f"  Q: Do you like {noisy_word}? (noisy-half color {noisy_example})")
    print(f"  A: \"{ans_noisy}\"")
    print()

    # ------------------------------------------------------------------ #
    # Save to creature/state/nira/                                        #
    # ------------------------------------------------------------------ #
    print(f"Saving to {STATE_DIR} ...")
    nira.save(STATE_DIR)
    hash_post_save = nira.state_hash()
    print(f"  Saved. state_hash = {hash_post_save[:20]}...")
    print()

    # ------------------------------------------------------------------ #
    # P1: round-trip + resume determinism                                  #
    # (operates on COPIES — nira's committed life stays at post-teach point) #
    # ------------------------------------------------------------------ #
    print("-" * 72)
    print("P1: round-trip + resume determinism check")

    c_reload = ContinuousCreature.load(STATE_DIR)
    hash_reload = c_reload.state_hash()
    p1_roundtrip = (hash_reload == hash_post_save)
    print(f"  Reload hash match: {p1_roundtrip}  "
          f"(saved={hash_post_save[:16]}... loaded={hash_reload[:16]}...)")

    # Two fresh loads, same explicit seed, 100 steps each — NOT saved
    # Unbind state_dir so live() does NOT append to nira's biography
    c_resume_a = ContinuousCreature.load(STATE_DIR)
    c_resume_a._state_dir = None
    c_resume_b = ContinuousCreature.load(STATE_DIR)
    c_resume_b._state_dir = None
    c_resume_a.live(100, seed=42)
    c_resume_b.live(100, seed=42)
    hash_a = c_resume_a.state_hash()
    hash_b = c_resume_b.state_hash()
    p1_resume = (hash_a == hash_b)
    print(f"  Resume determinism (100 steps, seed=42): {p1_resume}  "
          f"(a={hash_a[:16]}... b={hash_b[:16]}...)")

    p1_ok = p1_roundtrip and p1_resume
    print(f"  P1 RESULT: {'PASS' if p1_ok else 'FAIL'}")
    print()

    # ------------------------------------------------------------------ #
    # P2: guard coverage — print instructions (pytest runs separately)    #
    # ------------------------------------------------------------------ #
    print("-" * 72)
    print("P2: guard coverage (nira continuity test + full suite)")
    print("  The orchestrator should run pytest after this script completes.")
    print("  Expected: nira continuity tests now RUN (not skip); full suite green.")
    print("  Command: uv run --python .venv pytest 2>&1 | tail -1")
    print("  Expected: 113 passed (108 original + 2 new unit tests + 3 nira continuity guards), 0 skipped")
    print()

    # ------------------------------------------------------------------ #
    # P3: first-life sanity                                               #
    # ------------------------------------------------------------------ #
    print("-" * 72)
    print("P3: first-life sanity bands")

    p3_map = (map_formed >= 14)
    p3_loc = (loc_err <= 0.5)
    p3_rel = (rel_share > 0.55) and fav_in_reliable

    # Check what_do_you_like names the favorite's word
    fav_word_named = fav_word  # e.g. "w10"
    p3_word = (ans_like == f"I like {fav_word_named}")

    print(f"  map_formed_count >= 14/16: {p3_map}  (got {map_formed})")
    print(f"  localization_error <= 0.5: {p3_loc}  (got {loc_err:.4f})")
    print(f"  reliable_share > 0.55 AND fav in reliable half: {p3_rel}  "
          f"(share={rel_share:.4f}, fav={fav}, in_reliable={fav_in_reliable})")
    print(f"  answer_what_do_you_like names favorite word '{fav_word_named}': {p3_word}  "
          f"(answer='{ans_like}')")

    p3_ok = p3_map and p3_loc and p3_rel and p3_word
    print(f"  P3 RESULT: {'PASS' if p3_ok else 'FAIL'}")
    print()

    # ------------------------------------------------------------------ #
    # Verdict                                                             #
    # ------------------------------------------------------------------ #
    print("=" * 72)
    print("PREDECLARED CHECK VALUES")
    print("=" * 72)
    print(f"  P1 round-trip hash match:    {p1_roundtrip}")
    print(f"  P1 resume determinism:       {p1_resume}")
    print(f"  P1 COMBINED:                 {p1_ok}")
    print()
    print(f"  P3 map_formed >= 14/16:      {p3_map}  ({map_formed}/16)")
    print(f"  P3 loc_err <= 0.5:           {p3_loc}  ({loc_err:.4f})")
    print(f"  P3 rel_share > 0.55 + fav:   {p3_rel}  (share={rel_share:.4f}, fav_in_reliable={fav_in_reliable})")
    print(f"  P3 word answer correct:      {p3_word}  ('{ans_like}')")
    print(f"  P3 COMBINED:                 {p3_ok}")
    print()
    print("  P2: see pytest output (run after this script).")
    print()

    all_ok = p1_ok and p3_ok
    if all_ok:
        verdict = "POSITIVE — all predeclared engineering properties satisfied."
    else:
        verdict = "NEGATIVE — HALT: at least one predeclared property failed:"
        if not p1_ok:
            verdict += "\n    P1 FAILED: hash mismatch or resume nondeterminism"
        if not p3_ok:
            verdict += "\n    P3 FAILED: sanity band(s) missed — packaging changed behavior"

    print("=" * 72)
    print(f"VERDICT: {verdict}")
    print("=" * 72)

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
