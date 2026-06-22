"""embodied.render — replay a Trajectory through MuJoCo and write two camera videos."""
import os
from pathlib import Path

import imageio
import mujoco
import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")  # match the backend documented in README

from embodied.env import ARENA_PATH  # noqa: E402 (import after env-var set)


def _render_frames(model, data, qpos_seq, camera, height, width) -> list:
    """Render each frame in qpos_seq through the given camera.

    Returns a list of (height, width, 3) uint8 numpy arrays.
    Exposed at module level so tests can call it directly for the cross-camera guard.
    """
    frames = []
    with mujoco.Renderer(model, height=height, width=width) as r:
        for q in qpos_seq:
            data.qpos[:] = np.asarray(q)
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=camera)
            frames.append(r.render().copy())
    return frames


def render(traj, out_dir, fps: int = 30, height: int = 480, width: int = 640) -> dict:
    """Replay *traj* and write two mp4 videos (third-person + first-person).

    Args:
        traj:    Trajectory (from embodied.rollout) whose .qpos list is replayed.
        out_dir: Directory where videos are written (created if needed).
        fps:     Frames-per-second for the output mp4 files.
        height:  Render height in pixels.
        width:   Render width in pixels.

    Returns:
        {"thirdperson": Path, "firstperson": Path}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(ARENA_PATH))
    data = mujoco.MjData(model)

    camera_map = [
        ("thirdperson", "track"),
        ("firstperson", "firstperson"),
    ]

    paths: dict[str, Path] = {}
    for key, cam_name in camera_map:
        frames = _render_frames(model, data, traj.qpos, cam_name, height, width)
        p = out_dir / f"embodied_{key}.mp4"
        imageio.mimsave(str(p), frames, fps=fps)
        paths[key] = p

    return paths
