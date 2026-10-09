"""Build a synthetic .r3d that exercises the real parser path.

Renders the marker board from a moving camera, derives per-pixel depth analytically from
the table plane, and writes the whole thing in Record3D's bundle layout -- column-major K,
ARKit-convention camera-to-world quaternions, LZFSE-compressed float32 depth. If the
parser's conventions are wrong, the round trip disagrees with ground truth and the test
fails here rather than on a real capture.
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ARKIT_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = .25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, .25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, .25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, .25 * s
    return np.array([x, y, z, w])


def plane_depth(K, rvec, tvec, w, h):
    """Depth (along camera +z) to the table plane z=0, for every pixel."""
    import cv2
    R = cv2.Rodrigues(rvec)[0]
    Rwc = R.T
    C = (-R.T @ tvec).ravel()                      # camera centre in world
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pix = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    dirs = pix @ Rwc.T                              # ray directions in world
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -C[2] / dirs[..., 2]                    # hit z=0
    s[~np.isfinite(s)] = np.nan
    s[s <= 0] = np.nan
    return (s * dirs[..., 2] / np.abs(dirs[..., 2]) * np.abs(dirs[..., 2])).astype(np.float32) * 0 + \
           (s * 1.0 * np.abs(dirs[..., 2])).astype(np.float32)


def build(path, layout, marker_length, n=24, w=640, h=480, dw=160, dh=120, fps=30.0,
          centre=(-0.02, 0.12, 0.0), seed=0, anchor_frames=None):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from humanoid_challenge.synthetic import camera_matrix, look_at, render_board

    rng = np.random.default_rng(seed)
    K = camera_matrix(w, h, 0.9 * w)
    centre = np.asarray(centre, float)
    poses, ts = [], []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i in range(n):
            frac = i / max(n - 1, 1)
            # pan across the board, drifting in height
            eye = centre + np.array([0.25 * np.cos(2 * np.pi * frac),
                                     0.30 * np.sin(2 * np.pi * frac),
                                     1.30 + 0.08 * rng.uniform(-1, 1)])
            vis = layout if (anchor_frames is None or i < anchor_frames) else {}
            rvec, tvec = look_at(eye, centre)
            img = render_board(vis, marker_length, K, rvec, tvec, w, h) if vis else \
                np.full((h, w), 200, np.uint8)
            b = io.BytesIO()
            Image.fromarray(img).convert("RGB").save(b, "JPEG", quality=92)
            z.writestr(f"rgbd/{i}.jpg", b.getvalue())

            Kd = K.copy(); Kd[0] *= dw / w; Kd[1] *= dh / h
            d = plane_depth(Kd, rvec, tvec, dw, dh)
            d = np.nan_to_num(d, nan=0.0).astype(np.float32)
            try:
                import liblzfse
                z.writestr(f"rgbd/{i}.depth", liblzfse.compress(d.tobytes()))
            except Exception:
                z.writestr(f"rgbd/{i}.depth", d.tobytes())

            import cv2
            R = cv2.Rodrigues(rvec)[0]
            Rwc, C = R.T, (-R.T @ tvec).ravel()
            q = mat_to_quat(Rwc @ ARKIT_TO_OPENCV)   # parser re-applies ARKIT_TO_OPENCV
            poses.append([*q, *C])
            ts.append(i / fps)

        z.writestr("metadata", json.dumps({
            "K": list(K.T.flatten()),                # column-major, as Record3D writes it
            "w": w, "h": h, "dw": dw, "dh": dh, "fps": fps,
            "poses": poses, "frameTimestamps": ts,
            "initPose": [0, 0, 0, 1, 0, 0, 0],
        }))
    Path(path).write_bytes(buf.getvalue())
    return path
