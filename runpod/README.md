# RunPod single-GPU training — `embodied/` quadruped-forage (Brax/MJX PPO)

Train the embodied PPO gait on a **single-GPU** RunPod pod, then render the
result on your Mac. Training is GPU-bound MJX physics + a tiny `(64,64)` MLP;
rendering is cheap and is best done locally where `glfw` "just works".

- **Repo:** `https://github.com/Mirrorad1/active_monkey.git`
- **Branch:** `embodied-physics-substrate`
- **Train cmd (FULL config):** `python -m embodied.train --out embodied/checkpoints/quadruped_forage`
- **Checkpoint:** `embodied/checkpoints/quadruped_forage/params`
- **Pinned stack:** jax / jaxlib **0.10.1** (CUDA12 build forced on the pod), brax **0.14.2**, mujoco **3.9.0**, Python **3.12**.

---

## 0. Why single GPU (read this first)

`embodied/train.py` installs a `jax.device_put_replicated` compat shim (JAX
0.10.1 removed the API that brax 0.14.2 still calls). The shim puts everything
on `devices[0]` and adds a size-1 leading (pmap/device) axis — **correct for
exactly one device**. On a multi-GPU pod it would silently place all replicas
on device 0 and train on a broken/desynced state. So:

- **Rent a 1×GPU pod.** If you end up on a bigger box, the setup script pins
  `CUDA_VISIBLE_DEVICES=0` so only one GPU is visible.
- Do **not** "scale up" to multi-GPU expecting a speedup — it would not just
  fail to scale, it would silently train wrong.

Two more landmines the script already handles:
- **uv.lock pins the CPU jaxlib.** A plain `uv sync` gives you CPU JAX and the
  GPU sits idle. The script installs `jax[cuda12]==0.10.1` **after** the sync so
  the CUDA jaxlib wins, then verifies `jax.devices()` shows a GPU.
- **Never pass `--timesteps` for the full run.** `--timesteps` keeps SMOKE's
  `num_envs=8` (you'd think you ran "30M × 2048" but actually 30M × 8 envs).
  The bare `python -m embodied.train` is the only path that applies the FULL
  config (2048 envs, batch 1024, episode 1000, 5 evals).

---

## 1. Create the pod (one GPU)

**Chosen GPU: H100 SXM (80 GB, Hopper / sm_90).** Mature, rock-solid JAX support
(no new-architecture kernel risk), High availability, and 80 GB of headroom for
the later population-scale phases. Overkill for the ant-scale Phase-1 gait (which
fits in a few GB and is throughput-bound), but a sound "ready for multiple phases"
choice. Cheaper alternatives that also work: **RTX 4090 (24 GB)** or **L40S (48 GB)**
(both Ada / sm_89, equally mature). If you ever pick a **Blackwell** card
(RTX PRO 6000 / 5090, sm_120) you MUST use a **CUDA 12.8+** image — Hopper/Ada
work on any CUDA 12.x.

**Base image / template:** the official **`runpod/pytorch`** CUDA 12.x + Python 3.11
image, e.g.

```
runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2204
```

(Confirm the exact latest patch tag at deploy; the load-bearing properties are
`cu12x` / `py3.11` / `ubuntu2204|2404`. The explicit-cuDNN tag
`runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu` also works — and is
the one to use for a Blackwell card.) cu1290 covers H100/Hopper perfectly.
The JAX CUDA12 wheels bundle their own CUDA + cuDNN 9.8, so only the host
**driver** matters (needs `>= 525`, which every modern RunPod host clears).

When deploying:
- GPU count: **1**.
- Container/volume disk: ~30–40 GB is plenty.
- Optional but nice: attach a **network volume at `/workspace`** so the
  checkpoint survives pod termination. The script clones into `/workspace`.

---

## 2. Private-repo auth (GitHub PAT)

`active_monkey` is private, so the pod needs credentials to clone. Easiest is a
**fine-grained GitHub Personal Access Token** with **Contents: Read-only** on
the `Mirrorad1/active_monkey` repo.

1. GitHub → Settings → Developer settings → **Fine-grained tokens** → Generate.
   Repository access: only `Mirrorad1/active_monkey`. Permission: **Contents → Read-only**.
2. On the pod, export it before running the script (it is injected into the
   clone URL and never written to disk):

   ```bash
   export GITHUB_TOKEN=github_pat_xxxxxxxxxxxxxxxxx
   ```

The script does `https://<token>@github.com/Mirrorad1/active_monkey.git`. If you
prefer SSH, set up a deploy key on the pod and override `REPO_URL` to the SSH
form instead — but the PAT path is the least friction.

---

## 3. Run the setup + training script

Get `runpod/setup_and_train.sh` onto the pod (either it's already in the cloned
repo, or `scp`/paste it standalone), then:

```bash
export GITHUB_TOKEN=github_pat_xxxxxxxxxxxxxxxxx   # from step 2 (skip if public)
bash setup_and_train.sh
```

What it does, in order:
1. Pin `CUDA_VISIBLE_DEVICES=0`, set `JAX_PLATFORM_NAME=cuda`, `unset LD_LIBRARY_PATH`.
2. `apt-get install` the headless EGL/GL libs (only matters if you render on the pod).
3. Install **uv**.
4. Clone `embodied-physics-substrate` into `/workspace/active-loop`.
5. `uv venv --python 3.12 .venv` + `uv sync` (installs the locked deps — **CPU jaxlib**).
6. **Force `jax[cuda12]==0.10.1` on top** so the CUDA jaxlib wins.
7. Verify JAX sees the GPU **and runs a real 1024×1024 matmul kernel** (hard-fails
   if it's still on CPU or the CUDA stack is broken/arch-incompatible).
8. Run **FULL** training (bare command, no `--timesteps`), print the checkpoint path.

Tunable via env vars: `BRANCH`, `REPO_DIR`, `PY_VERSION`, `SEED`, `OUT`.

---

## 4. Expected wall time

- **First step waits on a one-time XLA compile of ~1–3 min** — this is not a hang.
- **FULL run (30M steps × 2048 envs):** order of magnitude **~30–90 min** on a
  modern datacenter GPU (4090 / A100 / L40S class). Budget ~1 hour; allow up to
  2 h on a smaller/slower card.
- **Rough cost:** H100 SXM ≈ **$3.29/hr** → a ~30–90 min run is roughly **$1.6–$5**
  (and the fastest wall-clock of the sane options). On an RTX 4090 (≈$0.69/hr) the
  same job is well under **$1** if you'd rather minimize spend.

**Want a quicker first checkpoint to eyeball?** Don't use `--timesteps` (it
drops to 8 envs). Instead edit `FULL["num_timesteps"]` in `embodied/train.py`
to e.g. `12_000_000` and run the bare command — a 12M-step FULL-config run is
~15–35 min and keeps all 2048 envs.

---

## 5. Download the checkpoint to your Mac

The `params` file is tiny (a few-KB pickle: a 3-tuple
`(normalizer_params, policy_params, value_params)` — `normalize_observations`
is off, so there's no live normalizer state).

**Easiest — `runpodctl` (no SSH keys / ports):**

```bash
# on the pod:
runpodctl send /workspace/active-loop/embodied/checkpoints/quadruped_forage/params
# prints a one-time code, e.g. 8338-galileo-collect-fidel

# on your Mac (install runpodctl first: brew install runpod/runpodctl/runpodctl):
cd /Users/mirro/Projects/active-loop
runpodctl receive 8338-galileo-collect-fidel
# move/rename the received file into place:
mkdir -p embodied/checkpoints/quadruped_forage
mv params embodied/checkpoints/quadruped_forage/params
```

**Alternative — `scp`** (use the pod's SSH host/port from the RunPod UI):

```bash
cd /Users/mirro/Projects/active-loop
scp -P <port> root@<pod-ip>:/workspace/active-loop/embodied/checkpoints/quadruped_forage/params \
    embodied/checkpoints/quadruped_forage/params
```

---

## 6. Render on the Mac (glfw) and commit

`embodied/render.py` defaults to `MUJOCO_GL=glfw` (via `setdefault`), which
works natively on your Mac with a display. From the repo root:

```bash
cd /Users/mirro/Projects/active-loop
uv run --python .venv python -m embodied.demo \
  --checkpoint embodied/checkpoints/quadruped_forage/params --steps 400
cat embodied/outputs/embodied_pipeline.txt
```

This writes two videos and a metrics file under `embodied/outputs/`:
`embodied_thirdperson.mp4`, `embodied_firstperson.mp4`, `embodied_pipeline.txt`.
(MJX rollout for 400 steps on Mac CPU is seconds-to-low-minutes.)

### Check the gait is non-trivial (acceptance)

From `embodied/outputs/embodied_pipeline.txt`:
- **`final_dist_to_food` should be well below the start distance.** A random/
  untrained policy ends ≈ start distance (it flails in place); a real gait shows
  a clear decrease.
- **`total_reached_steps > 0`** (ideally many) means it actually reached the
  food (`reached` latches within `REACH_RADIUS=0.6` each step). Zero with a large
  residual distance = walking FAILED even if the "alive" reward kept it upright.
- **Determinism / render sanity:** `traj_hash` is stable across two runs **on the
  same machine** (MJX is bit-exact per-hardware — a pod hash will NOT match a Mac
  hash; only compare within one machine), and the two mp4s are non-identical
  (different cameras).

### Commit

The `embodied/.gitignore` ignores `outputs/*.mp4`, so commit the **checkpoint**
(and the metrics text if you want it tracked), not the videos:

```bash
git add embodied/checkpoints/quadruped_forage/params
git commit -m "embodied: trained quadruped-forage gait checkpoint (RunPod 1xGPU, 30M steps)"
```

---

## Headless rendering on the pod (only if you can't render on the Mac)

Set `MUJOCO_GL=egl` **before** the process starts — `render.py` uses
`setdefault`, so an already-exported env var wins, but with nothing set it
forces `glfw` and crashes on a headless pod. The setup script already installs
the EGL/GL apt libs.

```bash
cd /workspace/active-loop
export MUJOCO_GL=egl
PYTHONPATH=. uv run --python .venv python -m embodied.demo \
  --checkpoint embodied/checkpoints/quadruped_forage/params --steps 400
# then download embodied/outputs/*.mp4 with runpodctl send / scp
```

EGL on RunPod images is occasionally driver-dependent; if it misbehaves, fall
back to `MUJOCO_GL=osmesa` (CPU software, slower) — `libosmesa6` is installed.
But the recommended path remains **train on pod, render on Mac**.
