"""Runnable example: score this artifact with the FROZEN scorer.

    uv run python examples/score_model.py <artifact_dir>
"""
import json, sys
from active_loop.artifacts import score_artifact

def main(artifact_dir):
    # Use a short config here for speed; drop the kwargs for the full frozen config.
    print(json.dumps(score_artifact(artifact_dir, seeds=(20, 21), turns=60), indent=2))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
