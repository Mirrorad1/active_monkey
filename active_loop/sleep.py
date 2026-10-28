"""Content-addressed receipts shared by lab and creature sleep adapters."""
from __future__ import annotations

import hashlib
import json


def canonical_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_receipt(scope, before, after, sources, budget, metrics, decision) -> dict:
    if scope not in ('lab', 'creature'):
        raise ValueError('scope must be lab or creature')
    receipt = dict(schema_version=1, algorithm='sleep-v1', scope=scope,
                   before_hash=canonical_hash(before), after_hash=canonical_hash(after),
                   sources=sources, budget=budget, metrics=metrics, decision=decision)
    receipt['content_hash'] = canonical_hash(receipt)
    return receipt
