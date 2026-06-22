import os
from pathlib import Path
import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "glfw")  # offscreen backend; document in README

ARENA = Path(__file__).resolve().parents[1] / "embodied" / "bodies" / "arena.xml"


def test_arena_loads_with_cameras_and_food():
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(ARENA))
    cams = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)}
    assert {"firstperson", "track"} <= cams
    food_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "food")
    assert food_id >= 0
    assert model.nu > 0  # has actuators (it can be driven)
    # Guard: compiler angle="degree" must be in effect — hip joint ranges are written
    # as degrees in the XML (±30°); without the setting MuJoCo 3.x defaults to radians
    # and interprets them as ±30 rad (≈±1718°), i.e. effectively unlimited.
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_1")
    lo, hi = model.jnt_range[jid]
    assert np.isclose(hi, np.deg2rad(30), atol=1e-3) and np.isclose(lo, np.deg2rad(-30), atol=1e-3), \
        "joint ranges must be interpreted as DEGREES (compiler angle=degree)"


def test_food_site_constant():
    from embodied import FOOD_SITE
    assert FOOD_SITE == "food"


def test_mujoco_offscreen_render_nonblack():
    import mujoco
    xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="2 2 0.1" rgba="0.3 0.5 0.3 1"/>
        <body pos="0 0 0.3"><freejoint/><geom type="box" size="0.2 0.2 0.2" rgba="0.8 0.2 0.2 1"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=120, width=160) as r:
        r.update_scene(data)
        frame = r.render()
    assert frame.shape == (120, 160, 3)
    assert frame.std() > 1.0  # not a constant/all-black frame


def test_env_builds_and_steps():
    import jax
    from embodied.env import EmbodiedForageEnv
    env = EmbodiedForageEnv()
    assert env.observation_size > 0 and env.action_size > 0
    state = env.reset(jax.random.PRNGKey(0))
    assert state.obs.shape == (env.observation_size,)
    act = jax.numpy.zeros(env.action_size)
    nstate = env.step(state, act)
    assert nstate.obs.shape == (env.observation_size,)
    assert jax.numpy.isfinite(nstate.reward)


def _fake_traj(n=12):
    import numpy as np
    from embodied.rollout import Trajectory
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(ARENA))
    q0 = np.zeros(model.nq)
    qpos = [q0.copy() for _ in range(n)]
    return Trajectory(qpos, [np.zeros(model.nv)] * n, [np.zeros(model.nu)] * n,
                      [1.0] * n, [0.0] * n, "deadbeefdeadbeef")


def test_render_writes_two_videos(tmp_path):
    from embodied.render import render, _render_frames
    import mujoco
    import numpy as np

    traj = _fake_traj()
    out = render(traj, tmp_path, fps=10)

    # Both mp4 files exist and are non-empty
    for k in ("thirdperson", "firstperson"):
        assert out[k].exists() and out[k].stat().st_size > 0

    # Cross-camera guard: different cameras must produce different pixels.
    # This catches the bug where camera= arg is silently ignored and both videos are identical.
    model = mujoco.MjModel.from_xml_path(str(ARENA))
    data = mujoco.MjData(model)
    track_frames = _render_frames(model, data, traj.qpos, "track", 120, 160)
    fp_frames = _render_frames(model, data, traj.qpos, "firstperson", 120, 160)
    assert not np.array_equal(track_frames[0], fp_frames[0]), \
        "track and firstperson cameras must render different pixels"


@pytest.mark.slow
def test_train_smoke_writes_loadable_checkpoint(tmp_path):
    from embodied.train import train, load_params
    ckpt = train(num_timesteps=2048, seed=0, out_dir=tmp_path)
    params = load_params(ckpt)
    assert params is not None


@pytest.mark.slow
def test_rollout_is_deterministic(tmp_path):
    from embodied.train import train
    from embodied.rollout import rollout
    ckpt = train(num_timesteps=2048, seed=0, out_dir=tmp_path)
    a = rollout(ckpt, n_steps=50, seed=0)
    b = rollout(ckpt, n_steps=50, seed=0)
    assert a.traj_hash == b.traj_hash
    assert len(a.qpos) == 50


@pytest.mark.slow
def test_demo_end_to_end(tmp_path, monkeypatch):
    from embodied import demo
    from embodied.train import train
    ckpt = train(num_timesteps=2048, seed=0, out_dir=tmp_path)
    monkeypatch.setattr(demo, "OUT_DIR", tmp_path / "out")
    demo.run(checkpoint=ckpt, steps=20)
    assert (tmp_path / "out" / "embodied_pipeline.txt").exists()
    assert (tmp_path / "out" / "embodied_thirdperson.mp4").exists()
