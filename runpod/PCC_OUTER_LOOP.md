# RunPod PCC outer-loop experiment

This runs the PCC frozen outer-loop experiment on a single RunPod GPU pod:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- benchmark: `openai/gsm8k`
- output: `experiments/outputs/pcc_qwen_gsm8k.json`

The script is `runpod/setup_pcc_outer_loop.sh`.

## Branch to clone

The current remote branch is:

```text
codex/pcc-runpod
```

The pod can clone this branch directly. You do not need to have the branch
locally on the pod first.

## Pod choice

Use a single-GPU official RunPod PyTorch image, for example:

```text
runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu
```

Recommended cheap GPUs:

- RTX 4090: plenty for Qwen 0.5B and a small GSM8K slice.
- L40S: more headroom.
- H100: fast but overkill for this first run.

Disk: 40 GB is enough. A network volume mounted at `/workspace` is convenient
because Hugging Face model/dataset caches then survive pod restarts.

## Run command on the pod

For a private repo, create a fine-grained GitHub token with read-only Contents
access to `Mirrorad1/active_monkey`, then paste this on the pod:

```bash
export GITHUB_TOKEN=github_pat_xxxxxxxxx
export REPO_URL=https://github.com/Mirrorad1/active_monkey.git
export WORKDIR=/workspace
export REPO_DIR=/workspace/active-loop
git clone --depth 1 --branch codex/pcc-runpod \
  "https://${GITHUB_TOKEN}@github.com/Mirrorad1/active_monkey.git" \
  /workspace/active-loop
cd /workspace/active-loop
export LIMIT=48
export MAX_NEW_TOKENS=192
bash runpod/setup_pcc_outer_loop.sh
```

If the repo is already cloned on the pod, use:

```bash
cd /workspace/active-loop
git fetch origin codex/pcc-runpod
git checkout codex/pcc-runpod
git pull --ff-only
bash runpod/setup_pcc_outer_loop.sh
```

The script will:

1. pin the job to one GPU,
2. clone the requested branch,
3. create `.venv-pcc`,
4. install PyTorch from the CUDA 12.8 wheel index plus `transformers`,
   `datasets`, and `accelerate`,
5. verify a real CUDA matmul,
6. run the dependency-free smoke check,
7. run the Qwen/GSM8K PCC experiment,
8. write the JSON report to `experiments/outputs/pcc_qwen_gsm8k.json`.

## Download the report

At the end, the script prints a `runpodctl send ...` command. On the pod:

```bash
runpodctl send /workspace/active-loop/experiments/outputs/pcc_qwen_gsm8k.json
```

On your Mac:

```bash
runpodctl receive <one-time-code>
```

Terminate the pod after the report is received.

## Useful overrides

```bash
export PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
export PYTORCH_BACKEND=cu128
export MODEL=Qwen/Qwen2.5-0.5B-Instruct
export DATASET=openai/gsm8k
export DATASET_CONFIG=main
export SPLIT=test
export LIMIT=128
export MAX_NEW_TOKENS=256
export ALPHA=0.1
export OUTPUT=experiments/outputs/pcc_qwen_gsm8k_limit128.json
```

For a quick remote smoke of the full path, start with `LIMIT=8`. For a more
meaningful first pilot, use at least `LIMIT=48`.

## Current caveat

The current harness reloads the model once for baseline and once per candidate
scaffold. That is acceptable for a first Qwen-0.5B pilot on a RunPod GPU, but the
next optimization should load the model once and evaluate all scaffolds in the
same process before scaling to larger slices.

## Driver mismatch repair

If the pod prints an error like:

```text
torch.__version__: 2.12.1+cu130
The NVIDIA driver on your system is too old (found version 12080)
```

the environment installed a CUDA 13 PyTorch wheel on a CUDA 12.8-class RunPod
host driver. Do not try to update the driver from inside the container. Reinstall
PyTorch from the CUDA 12.8 wheel index:

```bash
cd /workspace/active-loop
git pull --ff-only
uv pip uninstall --python .venv-pcc torch
uv pip install --python .venv-pcc \
  --index-url https://download.pytorch.org/whl/cu128 \
  --torch-backend cu128 \
  --reinstall-package torch \
  torch
bash runpod/setup_pcc_outer_loop.sh
```

The updated script uses `https://download.pytorch.org/whl/cu128` by default and
does not run `--upgrade` when installing `transformers`, `datasets`, and
`accelerate`, so those packages do not silently replace the CUDA 12.8 torch
wheel. It also uses uv's `--torch-backend cu128` and `--reinstall-package torch`
flags so an already-installed `+cu130` wheel is actually replaced instead of
merely audited.
