#!/usr/bin/env bash
# C3 LLM-probe -- one-shot setup + run on a fresh RunPod single-GPU pod.
# Use the official runpod/pytorch CUDA 12.x image (torch already present); this
# script only adds transformers and runs the probe. See README_RUNPOD.md.
set -euo pipefail

cd "$(dirname "$0")"

echo "== [1/4] python + GPU =="
python -c "import sys; print('python', sys.version.split()[0])"

echo "== [2/4] install deps (torch ships with the runpod/pytorch image) =="
pip install -q --upgrade "transformers>=4.44" "accelerate>=0.33" "huggingface_hub>=0.24"

echo "== [3/4] REAL CUDA kernel check (fail loud, not after a long run) =="
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA device visible -- wrong pod image?"
x = (torch.ones(1024, 1024, device="cuda") @ torch.ones(1024, 1024, device="cuda"))
torch.cuda.synchronize()
assert float(x[0, 0]) == 1024.0, "CUDA matmul wrong -- kernel/arch mismatch"
print("CUDA OK:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
PY

echo "== [4/4] run probe (override args via: bash setup_runpod.sh --n 100 ...) =="
python llm_probe.py \
  --model "${C3_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}" \
  --device cuda --dtype bfloat16 \
  --n "${C3_N:-60}" --seed 7 \
  "$@"

echo
echo "== done. results in: $(pwd)/results/llm_probe_summary.json =="
echo "   pull to your Mac:  runpodctl send results/llm_probe_summary.json"
