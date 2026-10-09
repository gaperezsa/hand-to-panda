#!/usr/bin/env python3
"""Gate 3a: can MuJoCo render offscreen on this headless box?

Deliberately isolated from LIBERO, robosuite and LeRobot. If this fails, nothing
downstream can work, and the cause is GPU/EGL -- not the simulator stack.
"""
import os
import sys

import numpy as np


def main() -> int:
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')!r}  "
          f"MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID')!r}  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    import mujoco
    print("mujoco", mujoco.__version__)

    # A red box on a white floor, lit from above. Trivial, but it exercises the whole
    # offscreen path: EGL context, framebuffer, depth test, shading, pixel readback.
    xml = """
    <mujoco>
      <visual><global offwidth="640" offheight="480"/></visual>
      <asset>
        <texture type="skybox" builtin="flat" rgb1="1 1 1" width="8" height="8"/>
      </asset>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
        <geom type="plane" size="2 2 .1" rgba=".9 .9 .9 1"/>
        <body pos="0 0 .3"><geom type="box" size=".3 .3 .3" rgba="1 0 0 1"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth, cam.lookat[:] = 3.0, -25.0, 45.0, [0, 0, 0.3]
    renderer.update_scene(data, cam)
    img = renderer.render()

    nonzero = int((img > 0).sum())
    red = int(((img[..., 0] > 150) & (img[..., 1] < 100) & (img[..., 2] < 100)).sum())
    print(f"rendered {img.shape} {img.dtype}  nonzero={nonzero}  "
          f"unique colours={len(np.unique(img.reshape(-1, 3), axis=0))}  red px={red}")

    if nonzero == 0:
        print("FAIL: frame is entirely black -- offscreen context produced nothing")
        return 1
    if red < 1000:
        print("FAIL: the red box is not in the frame; rendering ran but is wrong")
        return 1

    out = os.path.join(os.environ.get("PROJ", "."), "outputs", "egl_check.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import imageio.v3 as iio
    iio.imwrite(out, img)
    print(f"PASS: wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
