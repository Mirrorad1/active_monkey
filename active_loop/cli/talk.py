"""Talk to the active-inference character babbler: seed a prefix, get a continuation.

Honest banner: this is a character-level active-inference model, not a chatbot;
output quality reflects held-out free energy (bits/char).
"""
from __future__ import annotations

import argparse

from active_loop.lang_model import LangModel
from active_loop.cli._paths import repo_root
from eval.lang_score import score_language

CORPUS = repo_root() / "data" / "corpus.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--length", type=int, default=120)
    args = ap.parse_args()

    print("training the active-inference character model...")
    lm = LangModel(seed=0)
    lm.learn_stream(CORPUS.read_text(), epochs=args.epochs)
    report = score_language(epochs=args.epochs)
    print(f"[char-level active-inference babbler — {report.bits_per_char:.2f} bits/char; "
          f"baseline {report.baseline_bits:.2f}]")
    print("type a seed prefix (blank to quit):")
    while True:
        try:
            seed = input("> ")
        except EOFError:
            break
        if not seed:
            break
        print(seed + lm.generate(seed, n=args.length))


if __name__ == "__main__":
    main()
