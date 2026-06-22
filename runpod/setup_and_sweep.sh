#!/usr/bin/env bash
# runpod/setup_and_sweep.sh
# ---------------------------------------------------------------------------
# Run this ON a single-GPU RunPod pod (CUDA 12.8+ image for any GPU; Blackwell
# REQUIRES cu128+, Hopper/Ada work on any cu12x). It clones the embodied-physics
# branch, installs deps + the CUDA jaxlib, runs a real-kernel GPU check, then runs
# the Phase-2.5 calibration SWEEP on the MJX-batched substrate.
#
# Unlike setup_and_train.sh this does NOT train — it reuses the COMMITTED trained
# checkpoint (the gait the batched substrate forages with) and only sweeps food-field
# calibrations to answer: is there a stable + competitive embodied population?
#
# Usage (after `export GITHUB_TOKEN=github_pat_...`):
#   bash runpod/setup_and_sweep.sh
# Override the grid via env: CAPACITIES / REGENS / SEEDS / HORIZON / MAX_POP.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Mirrorad1/active_monkey.git}"
BRANCH="${BRANCH:-embodied-physics-substrate}"
WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="${REPO_DIR:-${WORKDIR}/active-loop}"
PY_VERSION="${PY_VERSION:-3.12}"
CAPACITIES="${CAPACITIES:-8 12 16 22 30}"
REGENS="${REGENS:-0.3 0.5}"
SEEDS="${SEEDS:-0 1 2}"
HORIZON="${HORIZON:-300}"
MAX_POP="${MAX_POP:-256}"
FOUNDERS="${FOUNDERS:-30}"

echo "==> [1/7] Single-GPU env"
export CUDA_VISIBLE_DEVICES=0 JAX_PLATFORM_NAME=cuda
unset LD_LIBRARY_PATH || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || \
  echo "    (nvidia-smi unavailable — verify this is a GPU pod)"

echo "==> [2/7] Headless GL libs (MuJoCo EGL — only used if you render later)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends libegl1 libgles2 libosmesa6 libglfw3 git ca-certificates curl || true

echo "==> [3/7] Install uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi
uv --version

echo "==> [4/7] Clone ${REPO_URL} @ ${BRANCH}"
mkdir -p "${WORKDIR}"
if [ -d "${REPO_DIR}/.git" ]; then
  git -C "${REPO_DIR}" fetch --depth 1 origin "${BRANCH}"
  git -C "${REPO_DIR}" checkout "${BRANCH}"
  git -C "${REPO_DIR}" reset --hard "origin/${BRANCH}"
else
  CLONE_URL="${REPO_URL}"
  if [ -n "${GITHUB_TOKEN:-${GH_TOKEN:-}}" ]; then
    TOKEN="${GITHUB_TOKEN:-${GH_TOKEN}}"
    CLONE_URL="$(printf '%s' "${REPO_URL}" | sed -E "s#https://#https://${TOKEN}@#")"
  fi
  git clone --depth 1 --branch "${BRANCH}" "${CLONE_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"
echo "    HEAD: $(git rev-parse --short HEAD)"

echo "==> [5/7] venv + deps, then FORCE the CUDA jaxlib over the CPU one in uv.lock"
uv venv --python "${PY_VERSION}" .venv
uv sync --frozen || uv sync
uv pip install --python .venv --upgrade \
  "jax==0.10.1" "jaxlib==0.10.1" "jax-cuda12-plugin[with-cuda]==0.10.1" "jax-cuda12-pjrt==0.10.1"

echo "==> [6/7] Verify JAX runs a REAL GPU kernel (not just device discovery)"
PYTHONPATH=. uv run --python .venv python - <<'PY'
import jax, jax.numpy as jnp
print("devices:", jax.devices())
assert any("cuda" in str(d).lower() for d in jax.devices()), "FATAL: JAX not on GPU"
y = (jnp.ones((1024, 1024)) @ jnp.ones((1024, 1024))).block_until_ready()
assert float(y[0, 0]) == 1024.0, "FATAL: GPU kernel produced wrong output"
print(f"OK: real GPU kernel on {y.devices()}")
PY

echo "==> [7/7] Phase-2.5 calibration sweep (batched substrate)"
PYTHONPATH=. uv run --python .venv python -m embodied.sweep_phase2p5 \
  --capacities ${CAPACITIES} --regens ${REGENS} --seeds ${SEEDS} \
  --founders ${FOUNDERS} --horizon ${HORIZON} --max-pop ${MAX_POP}

echo "==========================================================================="
echo " SWEEP COMPLETE — verdict above + written to embodied/outputs/sweep_phase2p5.txt"
echo " Download it:  runpodctl send ${REPO_DIR}/embodied/outputs/sweep_phase2p5.txt"
echo "==========================================================================="
