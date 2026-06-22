#!/usr/bin/env bash
# runpod/setup_pcc_outer_loop.sh
# ---------------------------------------------------------------------------
# Run this ON a single-GPU RunPod pod to execute the PCC frozen outer-loop
# experiment with Qwen/GSM8K.
#
# Recommended template:
#   runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu
#
# Usage:
#   export GITHUB_TOKEN=github_pat_...   # only needed for the private repo
#   # BRANCH is optional when running from a clone of the desired branch.
#   # Set BRANCH only when you want the script to fetch/reset a different branch.
#   bash runpod/setup_pcc_outer_loop.sh
#
# Useful overrides:
#   LIMIT=48 MAX_NEW_TOKENS=192 OUTPUT=experiments/outputs/pcc_qwen_gsm8k.json
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct DATASET=openai/gsm8k SPLIT=test
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Mirrorad1/active_monkey.git}"
BRANCH="${BRANCH:-}"
WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="${REPO_DIR:-${WORKDIR}/active-loop}"
PY_VERSION="${PY_VERSION:-3.12}"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DATASET="${DATASET:-openai/gsm8k}"
DATASET_CONFIG="${DATASET_CONFIG:-main}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-48}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
ALPHA="${ALPHA:-0.1}"
OUTPUT="${OUTPUT:-experiments/outputs/pcc_qwen_gsm8k.json}"
HF_HOME="${HF_HOME:-${WORKDIR}/hf-cache}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

echo "==> [1/8] Single-GPU environment"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME
unset LD_LIBRARY_PATH || true
echo "    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "    HF_HOME=${HF_HOME}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || \
  echo "    (nvidia-smi unavailable; REQUIRE_CUDA=${REQUIRE_CUDA})"

echo "==> [2/8] Base packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends git ca-certificates curl

echo "==> [3/8] Install uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH"; exit 1; }
uv --version

echo "==> [4/8] Clone ${REPO_URL} @ ${BRANCH}"
mkdir -p "${WORKDIR}"
if [ -d "${REPO_DIR}/.git" ]; then
  if [ -n "${BRANCH}" ]; then
    echo "    Repo already present at ${REPO_DIR}; fetching ${BRANCH}"
    git -C "${REPO_DIR}" fetch --depth 1 origin "${BRANCH}"
    git -C "${REPO_DIR}" checkout "${BRANCH}"
    git -C "${REPO_DIR}" reset --hard "origin/${BRANCH}"
  else
    echo "    Using existing checkout at ${REPO_DIR}; set BRANCH to fetch/reset another branch."
  fi
else
  CLONE_BRANCH="${BRANCH:-codex/pcc-runpod}"
  CLONE_URL="${REPO_URL}"
  if [ -n "${GITHUB_TOKEN:-${GH_TOKEN:-}}" ]; then
    TOKEN="${GITHUB_TOKEN:-${GH_TOKEN}}"
    CLONE_URL="$(printf '%s' "${REPO_URL}" | sed -E "s#https://#https://${TOKEN}@#")"
    echo "    Using GitHub token for private-repo auth"
  fi
  git clone --depth 1 --branch "${CLONE_BRANCH}" "${CLONE_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"
echo "    HEAD: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"

if [ ! -f experiments/pcc_outer_loop.py ]; then
  echo "ERROR: experiments/pcc_outer_loop.py is missing."
  echo "       Push a branch containing the PCC harness, then rerun with BRANCH=<that-branch>."
  exit 1
fi

echo "==> [5/8] Create isolated PCC venv and install HF stack"
uv venv --python "${PY_VERSION}" .venv-pcc
uv pip install --python .venv-pcc --upgrade torch transformers datasets accelerate

echo "==> [6/8] Verify PyTorch sees a real CUDA GPU"
.venv-pcc/bin/python - <<'PY'
import os
import sys
import torch

print("torch.__version__:", torch.__version__)
print("cuda available  :", torch.cuda.is_available())
print("devices         :", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

require_cuda = os.environ.get("REQUIRE_CUDA", "1") == "1"
if require_cuda and not torch.cuda.is_available():
    raise SystemExit("FATAL: REQUIRE_CUDA=1 but torch.cuda.is_available() is false")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    x = torch.ones((1024, 1024), device=device)
    y = x @ x
    torch.cuda.synchronize()
    if float(y[0, 0].item()) != 1024.0:
        raise SystemExit("FATAL: CUDA matmul produced wrong output")
    print("OK: real CUDA matmul executed on", torch.cuda.get_device_name(0))
else:
    print("WARNING: running without CUDA because REQUIRE_CUDA=0", file=sys.stderr)
PY

echo "==> [7/8] Dependency-free smoke check"
PYTHONPATH=. .venv-pcc/bin/python experiments/pcc_outer_loop.py \
  --backend smoke \
  --iterations 1 \
  --output experiments/outputs/pcc_smoke_runpod.json

echo "==> [8/8] Real PCC run: ${MODEL} on ${DATASET}/${DATASET_CONFIG} ${SPLIT}, limit=${LIMIT}"
mkdir -p "$(dirname "${OUTPUT}")"
PYTHONPATH=. .venv-pcc/bin/python experiments/pcc_outer_loop.py \
  --backend hf \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --dataset-config "${DATASET_CONFIG}" \
  --split "${SPLIT}" \
  --limit "${LIMIT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --alpha "${ALPHA}" \
  --output "${OUTPUT}"

echo "==========================================================================="
echo " PCC RUN COMPLETE"
echo "   report: ${REPO_DIR}/${OUTPUT}"
echo
echo "To download from the pod:"
echo "   runpodctl send ${REPO_DIR}/${OUTPUT}"
echo
echo "Then terminate the pod when the report is received."
echo "==========================================================================="
