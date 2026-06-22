---
name: huggingface-sandbox-experiment-env
description: |
  Use when running optional Hugging Face / transformers / datasets experiments
  inside an active-loop Codex worktree, especially when the project .venv is
  absent or intentionally should not receive heavyweight LLM dependencies, or
  when HF Hub downloads/cache locks fail with sandbox network or PermissionError
  messages. Covers isolated local venv setup and escalation points for model and
  dataset downloads.
author: Codex
version: 1.0.0
date: 2026-06-22
---

# Hugging Face Sandbox Experiment Environment

## Problem
Active-loop worktrees do not necessarily have a local `.venv`, and the
established main-checkout venv intentionally does not include heavyweight
optional LLM experiment packages such as `torch`, `transformers`, and `datasets`.

In the managed Codex sandbox, Hugging Face runs can also fail even after packages
are installed:

- DNS/network errors while downloading model or dataset metadata.
- `PermissionError: [Errno 1] Operation not permitted` for lock files under
  `~/.cache/huggingface/datasets/...`.
- `torch.__version__` shows a CUDA wheel newer than the RunPod host driver
  supports, e.g. `2.12.1+cu130` with driver `12080`, followed by
  `The NVIDIA driver on your system is too old`.

## Context / Trigger Conditions
Use this skill when:

- A script with `--backend hf` or similar reports missing `torch`,
  `transformers`, or `datasets`.
- A Hugging Face command fails with a message like:
  `nodename nor servname provided, or not known`.
- A cached dataset load fails when creating a file lock under
  `~/.cache/huggingface/datasets`.
- You need to run a small open-model benchmark without mutating the repo's
  established main test environment.

## Solution
1. Keep heavyweight LLM packages out of the established project verification
   interpreter unless the user explicitly wants them there.
2. Create a local ignored experiment venv inside the worktree:

   ```sh
   uv venv --python 3.12 .venv-pcc
   ```

3. Install optional HF packages into that venv:

   ```sh
   uv pip install --python .venv-pcc torch transformers datasets accelerate
   ```

4. Run the experiment with `.venv-pcc/bin/python`.
5. If Hub downloads fail due sandbox network restrictions, rerun the exact
   command with escalation.
6. If a cached dataset fails on a `~/.cache/huggingface/...lock`
   `PermissionError`, rerun the exact command with escalation. This is normal
   HF dataset cache locking, not an experiment-code error.

## RunPod Variant
For a small open-model benchmark run on RunPod, use the same isolation principle
inside the pod instead of installing optional HF packages into the project's
normal `.venv`.

1. Clone a committed/pushed branch on the pod. Uncommitted local work will not be
   visible remotely.
2. Create `.venv-pcc` in the cloned repo and install:

   ```sh
   uv pip install --python .venv-pcc torch transformers datasets accelerate
   ```

3. Verify the GPU with a real PyTorch CUDA matmul, not just `torch.cuda.is_available()`.
4. Write the experiment JSON to a file under `experiments/outputs/` and download
   it with `runpodctl send`.

Branch trap: if the setup script also contains clone/fetch logic, do not default
`BRANCH` to `main` when it is run from an already-cloned feature branch. That
will reset the fresh pod checkout away from the branch the user intentionally
cloned. Use an empty default (`BRANCH="${BRANCH:-}"`) and, when `REPO_DIR/.git`
already exists, keep the existing checkout unless `BRANCH` is explicitly set.
Only use a branch default for the no-repo-present clone path.

In this repo, `runpod/setup_pcc_outer_loop.sh` implements that pattern for
`experiments/pcc_outer_loop.py`.

PyTorch driver trap: do not install bare `torch` from default PyPI on RunPod
when the host driver is CUDA 12.8-class. It may resolve a newer CUDA wheel
such as `+cu130`, which fails before device enumeration. Install from the
CUDA-12.8 wheel index instead:

```sh
uv pip uninstall --python .venv-pcc torch
uv pip install --python .venv-pcc \
  --index-url https://download.pytorch.org/whl/cu128 \
  --torch-backend cu128 \
  --reinstall-package torch \
  torch
```

Then install the Hugging Face packages without `--upgrade`; otherwise the solver
can upgrade transitive dependencies and replace the working `+cu128` torch wheel
with the latest default-PyPI CUDA wheel again:

```sh
uv pip install --python .venv-pcc transformers datasets accelerate
```

If the current pod already hit the failure, `git pull --ff-only` the fixed
branch, uninstall/reinstall torch with the command above, and rerun the setup
script.

`uv pip uninstall` does not take `-y`; adding it makes the uninstall command
error out. Also, if `torch +cu130` is already installed, a plain install can say
`Audited 1 package` and leave it in place. Use `--reinstall-package torch` to
force replacement.

## Verification
A successful small HF plumbing run should:

- load the dataset split,
- load model weights,
- print the experiment JSON report,
- exit with code 0.

For code changes in this repo, still run focused tests with the established
main-checkout interpreter unless the tests specifically require the HF venv.

## Example
From an active-loop worktree:

```sh
uv venv --python 3.12 .venv-pcc
uv pip install --python .venv-pcc torch transformers datasets accelerate
.venv-pcc/bin/python experiments/pcc_outer_loop.py --backend hf --limit 8
```

If the final command fails on HF network or cache locks, rerun that final command
with escalated sandbox permissions.

## Notes
This is for optional experiment dependencies. It does not replace the
project-standing rule to verify normal repo tests with the established
main-checkout interpreter when the worktree lacks `.venv`.
