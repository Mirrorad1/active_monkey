"""Capstone 'converse' demo (Exp 35): a creature you can ask things, answering from its OWN
self-learned world + self-formed values + taught word-labels. Toy scale, honest:
 - place map / colors: self-learnable (Exp20/21)         [here: given, to keep the demo compact]
 - VALUES over colors: SELF-FORMED from experience (Exp26) [genuinely the creature's own]
 - word<->color labels: TAUGHT from a few examples (Exp34) [like a child labeling its concepts]
 - sentence shape: TEMPLATE (genuine grammar is the open ceiling, see open_problem.html)
Run: python converse_demo.py   (PYTHONPATH=. or from repo root)
"""
from __future__ import annotations
import numpy as np, math

G = 3
CMAP = np.array([0, 1, 2, 1, 2, 0, 2, 0, 1])     # color at each cell (self-learnable: Exp21)
WORDS = ["red", "blue", "green"]
F = 3


def neighbors(cell):
    r, c = divmod(cell, G); out = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < G and 0 <= nc < G: out.append(nr * G + nc)
    return out


def raise_creature(predictable, rng, steps=4000):
    """Form values from experience (Exp26): a color whose consequences are predictable feels good."""
    P = {f: (np.eye(F)[(f + 1) % F] if f == predictable else np.ones(F) / F) for f in range(F)}
    counts = np.ones((F, F)) * 0.1; f = rng.integers(F)
    for _ in range(steps):
        nxt = rng.choice(F, p=P[f]); counts[f, nxt] += 1; f = nxt
    learned = counts / counts.sum(1, keepdims=True)
    ent = np.array([-np.sum(learned[g] * np.log(learned[g] + 1e-12)) for g in range(F)]) / math.log(2)
    value = np.exp(-3.0 * ent); value /= value.sum()
    return value, ent


def teach_words(rng, n=8):
    wc = np.ones((F, F)) * 0.1
    for _ in range(n):
        c = rng.integers(F); wc[c, c] += 1
    return wc / wc.sum(0, keepdims=True)


class Creature:
    def __init__(self, predictable, seed):
        rng = np.random.default_rng(seed)
        self.value, self.ent = raise_creature(predictable, rng)
        self.WC = teach_words(rng)
        self.pos = 4  # where it currently stands

    def word_for(self, color): return WORDS[int(np.argmax(self.WC[:, color]))]
    def color_for(self, word): return int(np.argmax(self.WC[WORDS.index(word), :]))
    def feel(self, color):
        return "I like it" if self.value[color] > 0.5 else "it unsettles me"

    def ask(self, q):
        if q == "where are you?":
            return f"I'm at a {self.word_for(int(CMAP[self.pos]))} place."
        if q == "what do you like?":
            return f"I like {self.word_for(int(np.argmax(self.value)))}."
        if q == "what is near you?":
            cols = sorted({int(CMAP[n]) for n in neighbors(self.pos)})
            return "near me: " + ", ".join(f"{self.word_for(c)} ({self.feel(c)})" for c in cols)
        if q.startswith("do you like "):
            w = q[len("do you like "):].rstrip("?")
            c = self.color_for(w)
            return f"{w}: {self.feel(c)}."
        return "I don't understand that."


QUESTIONS = ["where are you?", "what do you like?", "what is near you?", "do you like green?"]


def main():
    print("=== converse with two creatures raised differently (same questions) ===")
    for label, pred, seed in [("creature-A (raised among red)", 0, 1),
                              ("creature-B (raised among green)", 2, 2)]:
        c = Creature(pred, seed)
        print(f"\n{label}:")
        for q in QUESTIONS:
            print(f"  you> {q}\n  it > {c.ask(q)}")
    print("\n[honest: VALUES self-formed from experience; word-labels taught; sentence shape templated.]")


if __name__ == "__main__":
    main()
