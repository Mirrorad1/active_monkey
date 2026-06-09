"""Exp 44 — affective coloring: valence from the creature's own free-energy trace.

Hypothesis: for a normally-raised creature (Exp 26), valence derived from its own
surprise TRACE (-dF/dt, operationalized as mean surprise over the first 10% of a
feature's encounters minus the last 10%) reproduces the like/unsettle verdicts of the
stored self-formed value (final-entropy formula) that converse_demo.py looks up.
Prediction if TRUE: for 2 creatures raised on different features, favorites agree
(argmax trace-valence == argmax stored value) and all 6 per-feature verdicts agree
(predeclared threshold: trace drop >= 0.5 bits = 'i like it', else 'it unsettles me';
stored: value > 0.5 = like).
Falsifier: any favorite or per-feature verdict disagrees.
Part 3 (designed dissociation, predeclared): a creature whose predictable feature's
correct transition row is given INNATELY (counts pre-seeded, nothing to learn there).
Prediction: trace drop ~0 bits for that feature (nothing was learned) despite stored
value high -- i.e. -dF/dt is the valence of LEARNING, -F the valence of KNOWING; they
coincide only when knowledge was acquired in-life. Either outcome logged honestly.
Seed: fixed (1 and 2 for the two creatures, 3 for part 3); single seeds, no shopping.
"""
import numpy as np, math

F = 3
WORDS = ["red", "blue", "green"]


def raise_with_trace(predictable, rng, steps=4000, innate=False):
    # world: feature f's successor deterministic iff f == predictable, else uniform
    P = {f: (np.eye(F)[(f + 1) % F] if f == predictable else np.ones(F) / F) for f in range(F)}
    counts = np.ones((F, F)) * 0.1
    if innate:  # part 3: give the correct row innately (strong pre-seeded counts)
        counts[predictable] = P[predictable] * 100.0 + 0.1
    surprises = {f: [] for f in range(F)}   # per-feature surprise trace, in encounter order
    f = rng.integers(F)
    for _ in range(steps):
        nxt = rng.choice(F, p=P[f])
        pred = counts[f] / counts[f].sum()
        surprises[f].append(-math.log2(pred[nxt] + 1e-12))
        counts[f, nxt] += 1
        f = nxt
    learned = counts / counts.sum(1, keepdims=True)
    ent = np.array([-np.sum(learned[g] * np.log(learned[g] + 1e-12)) for g in range(F)]) / math.log(2)
    value = np.exp(-3.0 * ent); value /= value.sum()
    return value, surprises


def trace_valence(surprises):
    out = np.zeros(F)
    for f in range(F):
        s = surprises[f]; k = max(1, len(s) // 10)
        out[f] = float(np.mean(s[:k]) - np.mean(s[-k:]))   # bits of surprise drop
    return out


def stored_verdict(value, f):
    return "i like it" if value[f] > 0.5 else "it unsettles me"


def trace_verdict(drop):
    return "i like it" if drop >= 0.5 else "it unsettles me"


def print_creature(label, predictable, rng, innate=False):
    value, surprises = raise_with_trace(predictable, rng, innate=innate)
    tv = trace_valence(surprises)
    print(f"\n--- {label} (predictable={WORDS[predictable]}, innate={innate}) ---")
    print(f"{'feature':<8} {'word':<6} {'stored_val':>10} {'stored_verd':<18} {'trace_drop':>10} {'trace_verd':<18} {'agree'}")
    agreements = []
    for f in range(F):
        sv = stored_verdict(value, f)
        tv_v = trace_verdict(tv[f])
        agree = (sv == tv_v)
        agreements.append(agree)
        print(f"  f={f}    {WORDS[f]:<6} {value[f]:>10.3f} {sv:<18} {tv[f]:>10.3f} {tv_v:<18} {agree}")
    fav_stored = int(np.argmax(value))
    fav_trace = int(np.argmax(tv))
    fav_agree = (fav_stored == fav_trace)
    print(f"  favorites: stored={WORDS[fav_stored]}, trace={WORDS[fav_trace]}, agree={fav_agree}")
    # Demo conversation (from trace verdicts)
    green_verd = trace_verdict(tv[2])
    fav_word = WORDS[fav_trace]
    print(f"  you> do you like green.")
    print(f"  it > green: {green_verd}.")
    print(f"  you> what do you like.")
    print(f"  it > i like {fav_word}.")
    return agreements, fav_agree, value, tv


def main():
    print("=== Exp 44: trace valence vs stored value ===")

    # Parts 1+2: creatures A and B
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    agrees_a, fav_a, val_a, tv_a = print_creature("creature-A", 0, rng_a)
    agrees_b, fav_b, val_b, tv_b = print_creature("creature-B", 2, rng_b)

    all_agrees = agrees_a + agrees_b
    n_agree = sum(all_agrees)
    print(f"\n=== Summary part 1+2 ===")
    print(f"verdict_agreements={n_agree}/6")
    print(f"favorites_agree=A:{fav_a}, B:{fav_b}")
    if n_agree == 6 and fav_a and fav_b:
        print("VERDICT part1: prediction CONFIRMED")
    else:
        details = []
        if n_agree < 6:
            bad = [f"f{i%3}({'AB'[i//3]})" for i, v in enumerate(all_agrees) if not v]
            details.append(f"verdict mismatch at {bad}")
        if not fav_a:
            details.append(f"favorite mismatch A: stored={WORDS[int(np.argmax(val_a))]}, trace={WORDS[int(np.argmax(tv_a))]}")
        if not fav_b:
            details.append(f"favorite mismatch B: stored={WORDS[int(np.argmax(val_b))]}, trace={WORDS[int(np.argmax(tv_b))]}")
        print(f"VERDICT part1: falsifier HIT ({'; '.join(details)})")

    # Part 3: innate creature C (predictable=1, seed=3)
    rng_c = np.random.default_rng(3)
    _, _, val_c, tv_c = print_creature("creature-C", 1, rng_c, innate=True)
    stored_like_f1 = (stored_verdict(val_c, 1) == "i like it")
    trace_drop_f1 = tv_c[1]
    dissociation_observed = stored_like_f1 and (trace_drop_f1 < 0.5)
    print(f"\n=== Summary part 3 ===")
    print(f"  feature 1 stored_verdict: {stored_verdict(val_c, 1)}, trace_drop: {trace_drop_f1:.3f} bits")
    print(f"  dissociation_observed={dissociation_observed}")
    if dissociation_observed:
        print("VERDICT part3: dissociation CONFIRMED (trace=learning, value=knowing)")
    else:
        print(f"VERDICT part3: no dissociation (innate prior still shows trace drop {trace_drop_f1:.3f} bits)")


if __name__ == "__main__":
    main()
