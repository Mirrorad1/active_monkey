"""FROZEN: run the suite and print the ScoreReport as JSON (used by the outer loop)."""
from __future__ import annotations

import dataclasses
import json

from eval.score import score_suite


def main() -> None:
    report = score_suite()
    print(json.dumps(dataclasses.asdict(report)))


if __name__ == "__main__":
    main()
