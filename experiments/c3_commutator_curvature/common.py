"""Shared utilities for the C3 (Commutator-Curvature Compression) experiment.

This module holds the three pieces every method shares:

1. A tokenizer (tiktoken if available, else a deterministic word/punct fallback).
2. Span <-> prompt (de)serialization.
3. A deterministic, LLM-free task evaluator that defines the *loss oracle*.

The loss oracle is the heart of the experiment. In a real deployment `loss(x)`
would be an LLM forward pass on the retained prompt. Here we substitute a
deterministic evaluator so we can isolate the *selection mechanism* before
spending any LLM calls (see README, "Why a deterministic evaluator").

LOSS DEFINITION (binary):

    loss(retained) = 0.0  if the retained spans still let the evaluator derive
                          the gold answer
                   = 1.0  otherwise

A required commitment is "satisfied" iff at least one of its provider spans is
retained. The gold answer is derivable iff *every* required commitment is
satisfied. Distractor spans provide no commitment, so deleting them never
changes the loss. A commitment with two provider spans is "redundant": deleting
either provider alone is harmless (delta ~= 0), but deleting both loses the
commitment (the second-order residue sigma_ij = 1).

IMPORTANT: the selector is only ever handed a scalar `loss(retained_ids)`
closure. It never sees `hidden_required_commitments`, `hidden_fragile_groups`,
or `hidden_dangerous_pairs`. Those exist only for final diagnostics.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Iterable, Sequence

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised only when tiktoken is installed
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def token_count(text: str) -> int:
        return len(_ENC.encode(text))

    TOKENIZER_NAME = "tiktoken/cl100k_base"
except Exception:  # fallback: deterministic word + punctuation tokenizer
    _WORD_RE = re.compile(r"\w+|[^\w\s]")

    def token_count(text: str) -> int:
        return len(_WORD_RE.findall(text))

    TOKENIZER_NAME = "regex-word-fallback"


# ---------------------------------------------------------------------------
# Span <-> prompt
# ---------------------------------------------------------------------------

def render_prompt(spans: Sequence[dict], retained_ids: Iterable[int] | None = None) -> str:
    """Render numbered spans into a prompt string.

    If retained_ids is given, only those spans are rendered (in id order),
    preserving their original [n] numbering so references stay stable.
    """
    keep = None if retained_ids is None else set(retained_ids)
    lines = []
    for s in spans:
        if keep is None or s["id"] in keep:
            lines.append(f"[{s['id']}] {s['text']}")
    return "\n".join(lines)


def prompt_token_count(spans: Sequence[dict], retained_ids: Iterable[int] | None = None) -> int:
    return token_count(render_prompt(spans, retained_ids))


def span_tokens(span: dict) -> int:
    return token_count(span["text"])


# ---------------------------------------------------------------------------
# Evaluator / loss oracle
# ---------------------------------------------------------------------------

def commitments_satisfied(instance: dict, retained: set[int]) -> list[str]:
    """Return the list of required-commitment ids that are NOT satisfied."""
    missing = []
    for c in instance["hidden_required_commitments"]:
        if not (set(c["provider_spans"]) & retained):
            missing.append(c["cid"])
    return missing


def is_correct(instance: dict, retained_ids: Iterable[int]) -> bool:
    retained = set(retained_ids)
    return len(commitments_satisfied(instance, retained)) == 0


def derive_answer(instance: dict, retained_ids: Iterable[int]) -> str:
    """Deterministic answer derivation, used for failure-case explanations.

    If all commitments hold -> gold answer. Otherwise, if a retained distractor
    targets a now-missing commitment, that distractor "dominates" and yields its
    wrong value; otherwise the answer degrades to UNKNOWN.
    """
    retained = set(retained_ids)
    missing = set(commitments_satisfied(instance, retained))
    if not missing:
        return instance["gold_answer"]
    # contradiction-causing distractor domination
    for s in instance["spans"]:
        if s["id"] in retained and s.get("role") == "distractor":
            tgt = s.get("contradicts")
            if tgt in missing and "wrong_value" in s:
                return s["wrong_value"]
    return "UNKNOWN"


def make_loss_oracle(instance: dict) -> Callable[[Iterable[int]], float]:
    """Return loss(retained_ids) -> {0.0, 1.0}. The ONLY view the selector gets."""

    def loss(retained_ids: Iterable[int]) -> float:
        return 0.0 if is_correct(instance, retained_ids) else 1.0

    return loss


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_dataset(path: str, instances: Sequence[dict]) -> None:
    with open(path, "w") as f:
        for inst in instances:
            f.write(json.dumps(inst) + "\n")


def split_dataset(instances: list[dict], train=0.6, dev=0.2):
    """Deterministic contiguous split (data is already shuffled at generation)."""
    n = len(instances)
    n_train = int(n * train)
    n_dev = int(n * dev)
    return (
        instances[:n_train],
        instances[n_train:n_train + n_dev],
        instances[n_train + n_dev:],
    )
