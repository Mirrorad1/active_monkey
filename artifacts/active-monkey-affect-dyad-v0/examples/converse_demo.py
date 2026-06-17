"""Runnable example: load this artifact and run a scripted-partner demo session.

    uv run python examples/converse_demo.py <artifact_dir>
"""
import sys
from active_loop.artifacts import load_agent_from_artifact
from active_loop.affect_spec import U, POS, NEU, ASK
import numpy as np

def main(artifact_dir):
    agent = load_agent_from_artifact(artifact_dir, which="init")
    correct = {c: c % 4 for c in range(U)}
    rng = np.random.default_rng(0)
    for t in range(30):
        code = int(rng.integers(0, U))
        agent.perceive(code)
        r = agent.act()
        val = POS if r == correct[code] else (NEU if r == ASK else 0)
        agent.observe_feedback(code, val)
        print(f"t={t:2d} code={code} response={r} valence={val}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
