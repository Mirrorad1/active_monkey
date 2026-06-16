"""FROZEN: run score_affect() and print the AffectScoreReport as JSON (used by the outer loop)."""
from __future__ import annotations

import dataclasses
import json

from eval.affect_score import score_affect


def main() -> None:
    report = score_affect()
    print(json.dumps(dataclasses.asdict(report)))


if __name__ == "__main__":
    main()
