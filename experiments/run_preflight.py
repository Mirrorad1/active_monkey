"""
run_preflight.py — thin CLI wrapper for the Evolvability Preflight framework.

Usage
-----
    uv run --python .venv python experiments/run_preflight.py \\
        --config experiments/configs/preflight/thermosense_smoke.json \\
        [--run-id my-run-1] \\
        [--output-dir results/preflight]

This script is a pass-through to `ecology.evolvability.__main__.main`.
It adds the repo root to sys.path so it can be run directly as a script
without installing the package, matching the pattern used by other
experiments/ scripts in this repo.

Exit codes
----------
0 : preflight completed (even if the scientific verdict is negative)
1 : infrastructure error (file not found, gate crash, etc.)
"""
import sys
from pathlib import Path

# Insert repo root so `ecology` is importable when run as a script
_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ecology.evolvability.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
