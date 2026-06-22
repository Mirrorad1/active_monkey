#!/usr/bin/env bash
# runpod/setup_and_train.sh
# ---------------------------------------------------------------------------
# Run this ON a single-GPU RunPod pod (runpod/pytorch CUDA 12.x, Python 3.11).
# It provisions a headless GL stack, installs uv, clones the embodied-physics
# branch, installs the project deps, then FORCES the CUDA jaxlib on top so the
# GPU build wins over the CPU jaxlib pinned in uv.lock, and finally runs the
# FULL 30M-step PPO training. Prints the checkpoint path at the end.
#
# SINGLE GPU ONLY. The train.py device_put_replicated shim is correct for
# exactly one device; on multi-GPU it silently degenerates. We pin
# CUDA_VISIBLE_DEVICES=0 to be safe even on a larger box.
#
# Usage:
#   bash runpod/setup_and_train.sh
# (or, if you scp'd it standalone:  bash setup_and_train.sh )
# ---------------------------------------------------------------------------
set -euo pipefail

# --- Tunables (override via env) -------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/Mirrorad1/active_monkey.git}"
BRANCH="${BRANCH:-embodied-physics-substrate}"
WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="${REPO_DIR:-${WORKDIR}/active-loop}"
PY_VERSION="${PY_VERSION:-3.12}"          # 3.12 is the validated interpreter
SEED="${SEED:-0}"
OUT="${OUT:-embodied/checkpoints/quadruped_forage}"
# ---------------------------------------------------------------------------

echo "==> [1/8] Single-GPU guard + environment"
export CUDA_VISIBLE_DEVICES=0            # pin to one device (shim is 1-device-correct)
export JAX_PLATFORM_NAME=cuda            # make JAX target the GPU
unset LD_LIBRARY_PATH || true            # JAX bundles its own CUDA libs; a stale
                                         # LD_LIBRARY_PATH can shadow them and break import
echo "    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || \
  echo "    (nvidia-smi not available — continuing, but verify this is a GPU pod)"

echo "==> [2/8] Headless GL libs for MuJoCo EGL (only needed if you render on the pod)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
# libegl1 + libgles2 are the essential EGL/GPU pair; libosmesa6 is the CPU
# software fallback; libglfw3 + libgl1-mesa-glx round it out. On Ubuntu 24.04
# libgl1-mesa-glx is renamed libgl1, so we tolerate either being absent.
apt-get install -y --no-install-recommends \
  libegl1 libgles2 libosmesa6 libglfw3 git ca-certificates curl || true
apt-get install -y --no-install-recommends libegl1-mesa libgl1-mesa-glx || \
  apt-get install -y --no-install-recommends libgl1 || true

echo "==> [3/8] Install uv (Astral package/proj manager)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installs to ~/.local/bin (or ~/.cargo/bin on older installers)
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH after install"; exit 1; }
uv --version

echo "==> [4/8] Clone ${REPO_URL} @ ${BRANCH}"
mkdir -p "${WORKDIR}"
if [ -d "${REPO_DIR}/.git" ]; then
  echo "    Repo already present at ${REPO_DIR}; fetching ${BRANCH}"
  git -C "${REPO_DIR}" fetch --depth 1 origin "${BRANCH}"
  git -C "${REPO_DIR}" checkout "${BRANCH}"
  git -C "${REPO_DIR}" reset --hard "origin/${BRANCH}"
else
  # Private repo: provide auth. Easiest is a GitHub Personal Access Token (PAT).
  # Export GITHUB_TOKEN (or GH_TOKEN) before running this script and we inject
  # it into the clone URL. If the repo is public this is harmless / unused.
  CLONE_URL="${REPO_URL}"
  if [ -n "${GITHUB_TOKEN:-${GH_TOKEN:-}}" ]; then
    TOKEN="${GITHUB_TOKEN:-${GH_TOKEN}}"
    # https://<token>@github.com/owner/repo.git
    CLONE_URL="$(printf '%s' "${REPO_URL}" | sed -E "s#https://#https://${TOKEN}@#")"
    echo "    Using GITHUB_TOKEN for private-repo auth"
  fi
  git clone --depth 1 --branch "${BRANCH}" "${CLONE_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"
echo "    HEAD: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"

echo "==> [5/8] Create venv (.venv) on Python ${PY_VERSION} and install project deps"
# CLAUDE.md mandates running via .venv; create it explicitly so `uv run --python .venv`
# resolves to this interpreter. uv will fetch a managed CPython if needed.
uv venv --python "${PY_VERSION}" .venv
# Install the locked project deps. NOTE: uv.lock pins the CPU jaxlib 0.10.1 —
# the GPU build is forced in the NEXT step so it wins.
uv sync --frozen || uv sync   # --frozen first; fall back if lock drift

echo "==> [6/8] Force CUDA jaxlib 0.10.1 ON TOP (GPU build must win over CPU jaxlib)"
# This MUST run AFTER `uv sync` so the CUDA jaxlib/plugin/pjrt overwrite the
# CPU jaxlib that uv.lock resolved. cuda12 wheels for 0.10.1 exist (May 2026).
# --upgrade lets pip pick the matching post-release; the == pins keep it exact.
uv pip install --python .venv --upgrade \
  "jax==0.10.1" \
  "jaxlib==0.10.1" \
  "jax-cuda12-plugin[with-cuda]==0.10.1" \
  "jax-cuda12-pjrt==0.10.1"

echo "==> [7/8] Verify JAX sees the GPU AND runs a real kernel (not just device discovery)"
# Seeing a device != the arch's kernels are present. On a too-new GPU (e.g. a
# Blackwell sm_120 card on an old jaxlib) jax.devices() can list the GPU yet a
# real op fails with 'no kernel image is available'. So we force an actual
# matmul on-device and block on it — this fails LOUDLY here, before a 30-90 min
# training run, on a broken/incompatible CUDA stack. (On Hopper/H100 this is a
# formality; it's cheap insurance.)
PYTHONPATH=. uv run --python .venv python - <<'PY'
import jax, jax.numpy as jnp
devs = jax.devices()
print("jax.__version__ :", jax.__version__)
print("jax.devices()   :", devs)
plats = {d.platform for d in devs}
assert "gpu" in plats or "cuda" in plats or any("cuda" in str(d).lower() for d in devs), \
    f"FATAL: JAX is not on the GPU (platforms={plats}). Re-check the CUDA jaxlib install / driver."
# Real kernel: a 1024x1024 matmul executed and blocked-on (catches 'no kernel image').
x = jnp.ones((1024, 1024), dtype=jnp.float32)
y = (x @ x).block_until_ready()
assert bool(jnp.isfinite(y).all()) and float(y[0, 0]) == 1024.0, \
    "FATAL: GPU kernel ran but produced wrong/non-finite output — CUDA stack is broken."
print(f"OK: real GPU kernel executed on {y.devices()} (1024x1024 matmul).")
PY

echo "==> [8/8] FULL training: 30M timesteps, 2048 envs (no --timesteps / --smoke flags)"
echo "    Do NOT pass --timesteps for the full run — it would fall back to SMOKE's num_envs=8."
echo "    Expect a ~1-3 min XLA compile before the first step (not a hang)."
START_TS="$(date +%s)"
# Bare command => FULL config (2048 envs, batch 1024, episode 1000, 5 evals).
# Checkpoint is written to <out>/params via brax.io.model.save_params.
PYTHONPATH=. uv run --python .venv python -m embodied.train --seed "${SEED}" --out "${OUT}"
END_TS="$(date +%s)"

CKPT="${REPO_DIR}/${OUT}/params"
echo "==========================================================================="
echo " TRAINING COMPLETE"
echo "   wall time     : $(( END_TS - START_TS )) s"
echo "   checkpoint    : ${CKPT}"
echo "   (also printed above as 'checkpoint: ... (wall time: ...s)')"
echo
echo " Next: download the checkpoint to your Mac and render there (glfw):"
echo "   on pod : runpodctl send ${CKPT}"
echo "   on Mac : runpodctl receive <one-time-code>"
echo "   on Mac : uv run --python .venv python -m embodied.demo \\"
echo "              --checkpoint embodied/checkpoints/quadruped_forage/params --steps 400"
echo "==========================================================================="
ls -la "${CKPT}" || { echo "ERROR: checkpoint not found at ${CKPT}"; exit 1; }
