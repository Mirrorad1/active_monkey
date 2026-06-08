"""FROZEN: print the language score report as JSON (used by the M3b loop)."""
from __future__ import annotations

import dataclasses
import json

from eval.lang_score import score_language


def main() -> None:
    print(json.dumps(dataclasses.asdict(score_language())))


if __name__ == "__main__":
    main()
