"""Exercise audit rejection, including definitions outside the package namespace.

Run with `uv run --python .venv python <path-to-this-file>` from the repo root,
with the pinned Lean/lake on PATH. Controls live only in ignored .lake storage.
"""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
source = (ROOT / "BoundaryTransmission/Audit.lean").read_text()
controls = {
    "local_axiom": ("axiom outsideNamespaceAxiom : False", "Local axiom is forbidden"),
    "proof_definition": ("def outsideNamespaceDefinition : False := by sorry", "Unexpected axiom sorryAx"),
    "proof_opaque": ("opaque outsideNamespaceOpaque : False := by sorry", "Unexpected axiom sorryAx"),
    "theorem": ("theorem outsideNamespaceTheorem : False := by sorry", "Unexpected axiom sorryAx"),
}
for label, (bad, diagnostic) in controls.items():
    with tempfile.TemporaryDirectory(prefix="audit-control-", dir=ROOT / ".lake") as directory:
        path = Path(directory) / "BoundaryTransmission/Control.lean"
        path.parent.mkdir()
        path.write_text(source.replace("open Lean Elab Command in", bad + "\n\nopen Lean Elab Command in"))
        result = subprocess.run(["lake", "env", "lean", "-R", directory, str(path)],
                                cwd=ROOT, capture_output=True, text=True)
        if result.returncode == 0 or diagnostic not in result.stdout + result.stderr:
            raise RuntimeError(f"Negative control {label} failed:\n{result.stdout}\n{result.stderr}")
        print(f"PASS: {label} rejected: {diagnostic}", flush=True)
subprocess.run(["lake", "env", "lean", "BoundaryTransmission/Audit.lean"], cwd=ROOT, check=True)
