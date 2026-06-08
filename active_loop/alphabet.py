"""Fixed 28-symbol alphabet (a-z, space, period) and corpus normalization."""
from __future__ import annotations

import string

ALPHABET = string.ascii_lowercase + " ."
V = len(ALPHABET)  # 28
_INDEX = {c: i for i, c in enumerate(ALPHABET)}


def normalize(text: str) -> str:
    """Lowercase; map any symbol not in ALPHABET to a space."""
    return "".join(c if c in _INDEX else " " for c in text.lower())


def encode(text: str) -> list[int]:
    """Normalize then map each char to its index."""
    return [_INDEX[c] for c in normalize(text)]


def decode(indices: list[int]) -> str:
    return "".join(ALPHABET[i] for i in indices)
