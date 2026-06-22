#!/usr/bin/env bash
# sparse-llm/runpod_r3p.sh — run the R3' paradigm-routing test on a single-GPU RunPod pod.
# TORCH stack (not jax): on a CUDA pod `pip install torch` already gives the CUDA build.
#
# Usage (after `export GITHUB_TOKEN=github_pat_...`):
#   bash sparse-llm/runpod_r3p.sh
# Override the grid via env: SPARSE_MODEL / SPARSE_LENGTHS / SPARSE_DENSITY / SPARSE_NFACTS.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Mirrorad1/active_monkey.git}"
BRANCH="${BRANCH:-sparse-llm}"
WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="${REPO_DIR:-${WORKDIR}/active-loop}"
export HF_HOME="${HF_HOME:-${WORKDIR}/hf-cache}"
export SPARSE_MODEL="${SPARSE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export SPARSE_LENGTHS="${SPARSE_LENGTHS:-2048 4096}"
export SPARSE_DENSITY="${SPARSE_DENSITY:-0.04}"
export SPARSE_NFACTS="${SPARSE_NFACTS:-10}"

echo "==> [1/5] GPU"
export CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "  (no nvidia-smi?)"

echo "==> [2/5] clone ${REPO_URL} @ ${BRANCH}"
mkdir -p "${WORKDIR}"
if [ -d "${REPO_DIR}/.git" ]; then
  git -C "${REPO_DIR}" fetch --depth 1 origin "${BRANCH}" && git -C "${REPO_DIR}" checkout "${BRANCH}"
  git -C "${REPO_DIR}" reset --hard "origin/${BRANCH}"
else
  CLONE_URL="${REPO_URL}"
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    CLONE_URL="$(printf '%s' "${REPO_URL}" | sed -E "s#https://#https://${GITHUB_TOKEN}@#")"
  fi
  git clone --depth 1 --branch "${BRANCH}" "${CLONE_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"

echo "==> [3/5] venv + torch(CUDA) + transformers"
pip install -q --upgrade uv || true
uv venv --python 3.12 .venv-r3p
uv pip install --python .venv-r3p torch transformers accelerate

echo "==> [4/5] real GPU kernel check (not just device discovery)"
.venv-r3p/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "FATAL: torch not on CUDA"
y = (torch.ones(1024,1024,device="cuda") @ torch.ones(1024,1024,device="cuda")).sum().item()
assert y == 1024**3, "FATAL: GPU kernel wrong"
print("OK: real CUDA kernel on", torch.cuda.get_device_name(0))
PY

echo "==> [5/5] R3' routing (model=${SPARSE_MODEL} lengths=${SPARSE_LENGTHS} density=${SPARSE_DENSITY})"
PYTHONPATH=sparse-llm .venv-r3p/bin/python sparse-llm/r3p_routing.py | tee sparse-llm/r3p_result.txt
echo "==========================================================================="
echo " DONE. Result in sparse-llm/r3p_result.txt — download: runpodctl send ${REPO_DIR}/sparse-llm/r3p_result.txt"
echo "==========================================================================="
