import os
import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "glfw")  # offscreen backend; document in README


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
