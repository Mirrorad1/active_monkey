"""FROZEN: train the char HMM on the corpus and score held-out bits/char + guardrails."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

from active_loop.alphabet import V
from active_loop.lang_model import LangModel

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus.txt"
LN2 = math.log(2)


@dataclass(frozen=True)
class LangReport:
    bits_per_char: float
    baseline_bits: float
    guardrails: dict
    verdict: bool


def score_language(epochs: int = 6) -> LangReport:
    text = CORPUS.read_text()
    split = int(len(text) * 0.8)
    train, held = text[:split], text[split:]

    lm = LangModel(seed=0)
    lm.learn_stream(train, epochs=epochs)
    nats = lm.mean_surprise(held)
    bits = float(nats) / LN2
    baseline = math.log(V) / LN2

    finite = math.isfinite(bits)
    beats = bool(finite and bits < baseline)
    guardrails = {"finite": finite, "beats_baseline": beats}
    return LangReport(bits, baseline, guardrails, all(guardrails.values()))
